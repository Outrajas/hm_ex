"""Authoritative state update engine and external input interface."""

import importlib
from dataclasses import dataclass
from typing import Optional
import numpy as np

_state_def = importlib.import_module("00_spec.state_definition")
_coord = importlib.import_module("00_spec.coordinate_system")

HandState = _state_def.HandState
quaternion_normalize = _coord.quaternion_normalize
quaternion_multiply = _coord.quaternion_multiply
rotation_vector_to_quaternion = _coord.rotation_vector_to_quaternion

@dataclass
class ControlInput:
    """
    External control/correction packet applied to the persistent hand state.
    Serves as the unified input protocol for human UI controls, synthetic
    trajectories, and future neural residual predictions.
    """
    # Absolute targets (Human slider / teleoperation mode)
    target_T: Optional[np.ndarray] = None      # 3D translation (m)
    target_R: Optional[np.ndarray] = None      # Unit quaternion [w, x, y, z]
    target_q: Optional[np.ndarray] = None      # 21-DOF joint angles (rad)

    # Relative residual deltas (Neural correction mode)
    delta_T: Optional[np.ndarray] = None       # Proposed translation shift (m)
    delta_R: Optional[np.ndarray] = None       # Proposed rotation vector (rad)
    delta_q: Optional[np.ndarray] = None       # Proposed joint shifts (rad)

    # Dynamic blending authority A in [0.0, 1.0]
    authority: float = 1.0

def state_update(current_state: HandState, control: ControlInput, dt: float) -> HandState:
    """
    Transitions the persistent state forward by blending the kinematic prior
    with external control inputs, updating both configuration and spatial velocities.
    """
    safe_dt = max(dt, 1e-5)
    alpha = float(np.clip(control.authority, 0.0, 1.0))

    # --- 1. Global Translation (T, V) ---
    prior_T = current_state.T + current_state.V * safe_dt
    if control.target_T is not None:
        next_T = (1.0 - alpha) * prior_T + alpha * control.target_T
    elif control.delta_T is not None:
        next_T = prior_T + alpha * control.delta_T
    else:
        next_T = prior_T

    # Derived translational velocity
    next_V = (next_T - current_state.T) / safe_dt

    # --- 2. Global Orientation (R, omega) ---
    if control.target_R is not None:
        if alpha >= 0.999:
            next_R = quaternion_normalize(control.target_R)
        else:
            # Slerp / linear blend approximation
            dot = np.dot(current_state.R, control.target_R)
            tgt_R = control.target_R if dot >= 0 else -control.target_R
            next_R = quaternion_normalize((1.0 - alpha) * current_state.R + alpha * tgt_R)
    elif control.delta_R is not None:
        dq = rotation_vector_to_quaternion(alpha * control.delta_R)
        next_R = quaternion_normalize(quaternion_multiply(dq, current_state.R))
    else:
        dq = rotation_vector_to_quaternion(current_state.omega * safe_dt)
        next_R = quaternion_normalize(quaternion_multiply(dq, current_state.R))

    # Derived angular velocity omega from R_prev -> R_next
    # q_diff = next_R * current_state.R^*
    R_conj = np.array([current_state.R[0], -current_state.R[1], -current_state.R[2], -current_state.R[3]])
    q_diff = quaternion_multiply(next_R, R_conj)
    if q_diff[0] < 0:
        q_diff = -q_diff
    vec_part = q_diff[1:4]
    vec_norm = np.linalg.norm(vec_part)
    angle = 2.0 * np.arctan2(vec_norm, q_diff[0])
    if vec_norm > 1e-7 and angle > 1e-7:
        next_omega = (vec_part / vec_norm) * (angle / safe_dt)
    else:
        next_omega = np.zeros(3, dtype=np.float64)

    # --- 3. Joint Articulations (q, qdot) ---
    prior_q = current_state.q + current_state.qdot * safe_dt
    if control.target_q is not None:
        next_q = (1.0 - alpha) * prior_q + alpha * control.target_q
    elif control.delta_q is not None:
        next_q = prior_q + alpha * control.delta_q
    else:
        next_q = prior_q

    # Derived joint velocities
    next_qdot = (next_q - current_state.q) / safe_dt

    return HandState(
        T=next_T,
        R=next_R,
        q=next_q,
        V=next_V,
        omega=next_omega,
        qdot=next_qdot,
        # Shape is a calibrated per-user parameter, not a per-frame output.
        # Preserve it across every temporal state transition.
        beta=current_state.beta.copy(),
        delta_G=current_state.delta_G.copy(),
        delta_G_dot=current_state.delta_G_dot.copy(),
    )
