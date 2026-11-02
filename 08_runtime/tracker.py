"""Persistent temporal model-to-HandState runtime adapter.

The caller supplies a five-channel crop: RGB plus the current simulated hand's
rendered silhouette and projected skeleton.  No detector output is consumed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor


_network = importlib.import_module("04_network.architecture")
_state = importlib.import_module("00_spec.state_definition")
_state_engine = importlib.import_module("02_state.state")
_constraints = importlib.import_module("02_state.constraints")

HandState = _state.HandState
ControlInput = _state_engine.ControlInput
state_update = _state_engine.state_update
project_constraints = _constraints.project_constraints
pack_state_features = _network.pack_state_features
TemporalStateCorrector = _network.TemporalStateCorrector


@dataclass
class TrackerFrame:
    state: HandState
    authority: float
    uncertainty: np.ndarray  # std: [m for translation, rad for rotation/joints]
    corrected_from_image: bool


class PersistentHandTracker:
    """Carry the simulated state and GRU memory across camera frames.

    ``step`` runs at inference time, predicts state with the project's existing
    transition engine, then fuses a bounded learned residual. When
    ``measurement_valid`` is false, visual authority is zero, the state coasts
    with damped velocity, and uncertainty grows.
    """

    def __init__(
        self,
        model: TemporalStateCorrector,
        initial_state: HandState,
        *,
        velocity_damping: float = 0.90,
        process_noise_translation_m_sqrt_s: float = 0.005,
        process_noise_rotation_deg_sqrt_s: float = 5.0,
        process_noise_joint_deg_sqrt_s: float = 8.0,
    ) -> None:
        if not 0.0 <= velocity_damping <= 1.0:
            raise ValueError("velocity_damping must be in [0,1]")
        self.model = model
        self.model.eval()
        self.state = initial_state.copy()
        self.device = next(model.parameters()).device
        self.hidden: Optional[Tensor] = None
        self.velocity_damping = float(velocity_damping)
        self.process_noise = np.asarray(
            [
                process_noise_translation_m_sqrt_s,
                process_noise_translation_m_sqrt_s,
                process_noise_translation_m_sqrt_s,
                np.deg2rad(process_noise_rotation_deg_sqrt_s),
                np.deg2rad(process_noise_rotation_deg_sqrt_s),
                np.deg2rad(process_noise_rotation_deg_sqrt_s),
                *([np.deg2rad(process_noise_joint_deg_sqrt_s)] * 21),
            ],
            dtype=np.float32,
        )
        self.uncertainty = np.asarray(
            [0.01, 0.01, 0.01, *([np.deg2rad(10.0)] * 24)],
            dtype=np.float32,
        )
        self.max_gap_seconds = 1.0

    def reset(self, state: Optional[HandState] = None) -> None:
        if state is not None:
            self.state = state.copy()
        self.hidden = None
        self.uncertainty = np.asarray(
            [0.01, 0.01, 0.01, *([np.deg2rad(10.0)] * 24)], dtype=np.float32
        )

    def _state_features(self) -> Tensor:
        s = self.state
        return pack_state_features(
            torch.as_tensor(s.T, dtype=torch.float32, device=self.device),
            torch.as_tensor(s.R, dtype=torch.float32, device=self.device),
            torch.as_tensor(s.q, dtype=torch.float32, device=self.device),
            torch.as_tensor(s.V, dtype=torch.float32, device=self.device),
            torch.as_tensor(s.omega, dtype=torch.float32, device=self.device),
            torch.as_tensor(s.qdot, dtype=torch.float32, device=self.device),
            torch.as_tensor(s.beta, dtype=torch.float32, device=self.device),
        )

    def step(
        self,
        rendered_crop: Tensor,
        *,
        dt_seconds: float,
        measurement_valid: bool = True,
    ) -> TrackerFrame:
        """Consume one current-frame crop and advance the persistent hand.

        ``rendered_crop`` is ``(5,H,W)`` or ``(1,5,H,W)``.  Frame timing must
        come from the camera/video timestamps, not an assumed constant FPS.
        """
        dt = float(dt_seconds)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_seconds must be finite and positive")
        if rendered_crop.ndim == 3:
            rendered_crop = rendered_crop.unsqueeze(0)
        if rendered_crop.ndim != 4 or rendered_crop.shape[0] != 1 or rendered_crop.shape[1] != 5:
            raise ValueError("rendered_crop must have shape (5,H,W) or (1,5,H,W)")
        rendered_crop = rendered_crop.to(device=self.device, dtype=torch.float32)

        if dt > self.max_gap_seconds:
            # Do not carry stale recurrent context across a long tracking gap.
            self.hidden = None
            self.state.V[:] = 0.0
            self.state.omega[:] = 0.0
            self.state.qdot[:] = 0.0

        state_features = self._state_features()
        dt_tensor = torch.as_tensor([dt], dtype=torch.float32, device=self.device)
        valid_tensor = torch.as_tensor(
            [float(measurement_valid)], dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            correction = self.model.forward_step(
                rendered_crop,
                state_features,
                dt_tensor,
                hidden=self.hidden,
                frame_valid=valid_tensor,
            )
        self.hidden = correction.hidden.detach()
        learned_authority = float(correction.authority[0].clamp(0, 1).cpu())
        authority = learned_authority if measurement_valid else 0.0

        control = ControlInput(
            delta_T=correction.delta_T[0].cpu().numpy().astype(np.float64),
            delta_R=correction.delta_R[0].cpu().numpy().astype(np.float64),
            delta_q=correction.delta_q[0].cpu().numpy().astype(np.float64),
            authority=authority,
        )
        updated = state_update(self.state, control, dt)
        # The transition engine integrates the previous velocity. Damping here
        # bounds its coast during low confidence and brief occlusion.
        updated.V *= self.velocity_damping
        updated.omega *= self.velocity_damping
        updated.qdot *= self.velocity_damping
        self.state = project_constraints(updated)

        correction_scale = np.asarray(
            [0.05] * 3 + [0.3490658504] * 3 + [0.5235987756] * 21,
            dtype=np.float32,
        )
        model_std = np.exp(correction.log_sigma[0].clamp(-6.0, 3.0)).cpu().numpy()
        model_std = model_std * correction_scale
        if measurement_valid:
            prior_variance = self.uncertainty ** 2
            observation_variance = model_std ** 2
            self.uncertainty = np.sqrt(
                (1.0 - authority) * prior_variance + authority * observation_variance
            )
        else:
            self.uncertainty = np.sqrt(
                self.uncertainty ** 2 + self.process_noise ** 2 * max(dt, 1e-4)
            )
        self.uncertainty = np.minimum(
            self.uncertainty,
            np.asarray([0.50] * 3 + [np.pi] * 24, dtype=np.float32),
        )

        return TrackerFrame(
            state=self.state.copy(),
            authority=authority,
            uncertainty=self.uncertainty.copy(),
            corrected_from_image=bool(measurement_valid and authority > 0.0),
        )
