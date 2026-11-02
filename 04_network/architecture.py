"""Recurrent visual correction model for a persistent articulated hand state.

One call consumes the current hand crop, its rendered prior, the current state
features, and the elapsed time. The GRU hidden state carries temporal evidence.
The model predicts bounded state residuals and their uncertainty; it does not
replace the simulated hand with an unrelated per-frame joint prediction.

Image channel contract (5 channels, NCHW): RGB crop [0, 1], rendered MANO
silhouette [0, 1], and rendered projected skeleton [0, 1]. State feature
contract is provided by ``pack_state_features`` (67 values).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


STATE_FEATURE_DIM = 67
STATE_CORRECTION_DIM = 27  # translation 3 + rotation vector 3 + joint q 21
IMAGE_CHANNELS = 5


class StateCorrection(NamedTuple):
    """One frame's bounded correction and the updated recurrent memory."""

    delta_T: Tensor       # (B, 3), meters
    delta_R: Tensor       # (B, 3), axis-angle rotation vector, radians
    delta_q: Tensor       # (B, 21), radians
    log_sigma: Tensor     # (B, 27), log std in normalized correction units
    authority: Tensor     # (B,), confidence gate in [0, 1]
    hidden: Tensor        # (B, hidden_dim)


class SequenceCorrection(NamedTuple):
    """Stacked per-frame outputs from ``forward_sequence``."""

    delta_T: Tensor       # (B, T, 3)
    delta_R: Tensor       # (B, T, 3)
    delta_q: Tensor       # (B, T, 21)
    log_sigma: Tensor     # (B, T, 27)
    authority: Tensor     # (B, T)
    hidden: Tensor        # final hidden state, (B, hidden_dim)


def _conv_block(in_channels: int, out_channels: int, kernel_size: int, stride: int) -> nn.Sequential:
    padding = kernel_size // 2
    groups = min(8, out_channels)
    while out_channels % groups:
        groups -= 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(inplace=True),
    )


def pack_state_features(
    T: Tensor,
    R: Tensor,
    q: Tensor,
    V: Tensor,
    omega: Tensor,
    qdot: Tensor,
    beta: Tensor,
) -> Tensor:
    """Pack batched ``HandState`` fields into the model's normalized 67-D input.

    Inputs use the project conventions: ``R`` is quaternion [w,x,y,z], ``T``
    and ``V`` are meters and meters/second, and angles are radians. ``beta`` is
    held fixed per user during tracking. The rotation is encoded by its first
    two matrix columns to avoid the quaternion sign ambiguity.
    """
    if T.ndim == 1:
        T = T.unsqueeze(0)
    if R.ndim == 1:
        R = R.unsqueeze(0)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if V.ndim == 1:
        V = V.unsqueeze(0)
    if omega.ndim == 1:
        omega = omega.unsqueeze(0)
    if qdot.ndim == 1:
        qdot = qdot.unsqueeze(0)
    if beta.ndim == 1:
        beta = beta.unsqueeze(0)

    tensors = (T, R, q, V, omega, qdot, beta)
    if len({int(t.shape[0]) for t in tensors}) != 1:
        raise ValueError("All HandState fields must have the same batch dimension.")
    expected = ((3,), (4,), (21,), (3,), (3,), (21,), (10,))
    for name, value, tail in zip(("T", "R", "q", "V", "omega", "qdot", "beta"), tensors, expected):
        if tuple(value.shape[1:]) != tail:
            raise ValueError(f"{name} must have shape (B, {tail[0]}), got {tuple(value.shape)}")

    dtype, device = T.dtype, T.device
    R, q, V, omega, qdot, beta = [x.to(device=device, dtype=dtype) for x in (R, q, V, omega, qdot, beta)]
    quat = F.normalize(R, dim=-1, eps=1e-8)
    w, x, y, z = quat.unbind(dim=-1)
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    rotation_6d = rotation[:, :, :2].reshape(-1, 6)

    features = torch.cat(
        (
            T / 0.5,
            rotation_6d,
            q / torch.pi,
            V / 3.0,
            omega / 15.0,
            qdot / 20.0,
            beta / 3.0,
        ),
        dim=-1,
    )
    if features.shape[-1] != STATE_FEATURE_DIM:
        raise RuntimeError(f"Internal state feature size is {features.shape[-1]}, expected {STATE_FEATURE_DIM}.")
    return features


class TemporalStateCorrector(nn.Module):
    """CNN + GRU that corrects a persistent simulated hand state from images.

    The image is expected to contain RGB plus the prior rendered silhouette and
    skeleton. Corrections are bounded by default to 5 cm translation, 20 degrees
    global rotation, and 30 degrees per internal joint per frame. ``authority``
    is a learned scalar gate; downstream state fusion can also use ``log_sigma``
    to propagate uncertainty or apply per-component confidence.
    """

    def __init__(
        self,
        image_channels: int = IMAGE_CHANNELS,
        state_feature_dim: int = STATE_FEATURE_DIM,
        hidden_dim: int = 256,
        max_delta_translation_m: float = 0.05,
        max_delta_rotation_rad: float = 0.3490658504,
        max_delta_joint_rad: float = 0.5235987756,
    ) -> None:
        super().__init__()
        if image_channels != IMAGE_CHANNELS:
            raise ValueError(f"Expected {IMAGE_CHANNELS} image channels (RGB + mask + skeleton).")
        if state_feature_dim != STATE_FEATURE_DIM:
            raise ValueError(f"Expected {STATE_FEATURE_DIM} normalized state features.")
        self.max_delta_translation_m = float(max_delta_translation_m)
        self.max_delta_rotation_rad = float(max_delta_rotation_rad)
        self.max_delta_joint_rad = float(max_delta_joint_rad)
        self.hidden_dim = int(hidden_dim)

        self.image_encoder = nn.Sequential(
            _conv_block(image_channels, 32, 5, 2),
            _conv_block(32, 64, 3, 2),
            _conv_block(64, 128, 3, 2),
            _conv_block(128, 192, 3, 2),
            _conv_block(192, 256, 3, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_feature_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 96),
            nn.SiLU(inplace=True),
        )
        self.temporal_cell = nn.GRUCell(256 + 96 + 2, hidden_dim)
        self.delta_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 128), nn.SiLU(inplace=True), nn.Linear(128, STATE_CORRECTION_DIM))
        self.uncertainty_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 128), nn.SiLU(inplace=True), nn.Linear(128, STATE_CORRECTION_DIM))
        self.authority_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 64), nn.SiLU(inplace=True), nn.Linear(64, 1))

        # Start conservatively: small residuals and modest correction authority.
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.zeros_(self.uncertainty_head[-1].bias)
        nn.init.zeros_(self.authority_head[-1].weight)
        nn.init.constant_(self.authority_head[-1].bias, -1.0)

    def initial_hidden(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def forward_step(
        self,
        image: Tensor,
        state_features: Tensor,
        dt_seconds: Tensor,
        hidden: Optional[Tensor] = None,
        frame_valid: Optional[Tensor] = None,
    ) -> StateCorrection:
        """Advance the recurrent state by one frame and predict one correction."""
        if image.ndim != 4 or image.shape[1] != IMAGE_CHANNELS:
            raise ValueError(f"image must have shape (B, {IMAGE_CHANNELS}, H, W).")
        batch = image.shape[0]
        if state_features.shape != (batch, STATE_FEATURE_DIM):
            raise ValueError(f"state_features must have shape (B, {STATE_FEATURE_DIM}).")
        if dt_seconds.ndim == 0:
            dt_seconds = dt_seconds.expand(batch)
        dt_seconds = dt_seconds.reshape(batch, 1).to(device=image.device, dtype=image.dtype)
        if hidden is None:
            hidden = self.initial_hidden(batch, device=image.device, dtype=image.dtype)
        elif hidden.shape != (batch, self.hidden_dim):
            raise ValueError(f"hidden must have shape (B, {self.hidden_dim}).")

        if frame_valid is None:
            valid = torch.ones(batch, 1, device=image.device, dtype=image.dtype)
        else:
            valid = frame_valid.reshape(batch, 1).to(device=image.device, dtype=image.dtype).clamp(0, 1)

        image_embedding = self.image_encoder(image) * valid
        state_embedding = self.state_encoder(state_features.to(device=image.device, dtype=image.dtype))
        # Preserve actual frame spacing while keeping unusually long gaps bounded.
        dt_feature = torch.log1p(dt_seconds.clamp(0.0, 1.0) / (1.0 / 30.0))
        recurrent_input = torch.cat((image_embedding, state_embedding, dt_feature, valid), dim=-1)
        next_hidden = self.temporal_cell(recurrent_input, hidden)

        raw_delta = torch.tanh(self.delta_head(next_hidden))
        delta_T = raw_delta[:, 0:3] * self.max_delta_translation_m
        delta_R = raw_delta[:, 3:6] * self.max_delta_rotation_rad
        delta_q = raw_delta[:, 6:27] * self.max_delta_joint_rad
        log_sigma = self.uncertainty_head(next_hidden).clamp(-6.0, 3.0)
        authority = torch.sigmoid(self.authority_head(next_hidden)).squeeze(-1)
        return StateCorrection(delta_T, delta_R, delta_q, log_sigma, authority, next_hidden)

    def forward_sequence(
        self,
        images: Tensor,
        state_features: Tensor,
        dt_seconds: Tensor,
        hidden: Optional[Tensor] = None,
        frame_valid: Optional[Tensor] = None,
    ) -> SequenceCorrection:
        """Run a chronological batch of windows shaped ``(B,T,...)``."""
        if images.ndim != 5:
            raise ValueError("images must have shape (B, T, C, H, W).")
        batch, steps = images.shape[:2]
        if state_features.shape != (batch, steps, STATE_FEATURE_DIM):
            raise ValueError(f"state_features must have shape (B, T, {STATE_FEATURE_DIM}).")
        if dt_seconds.ndim == 1:
            if dt_seconds.shape[0] != steps:
                raise ValueError("1-D dt_seconds must contain one time delta per sequence step.")
            dt_seconds = dt_seconds.unsqueeze(0).expand(batch, -1)
        if dt_seconds.shape != (batch, steps):
            raise ValueError("dt_seconds must have shape (B,T) or (T,).")
        if frame_valid is not None and frame_valid.shape != (batch, steps):
            raise ValueError("frame_valid must have shape (B,T).")

        outputs = []
        for step in range(steps):
            step_valid = None if frame_valid is None else frame_valid[:, step]
            result = self.forward_step(
                images[:, step],
                state_features[:, step],
                dt_seconds[:, step],
                hidden=hidden,
                frame_valid=step_valid,
            )
            outputs.append(result)
            hidden = result.hidden

        return SequenceCorrection(
            delta_T=torch.stack([x.delta_T for x in outputs], dim=1),
            delta_R=torch.stack([x.delta_R for x in outputs], dim=1),
            delta_q=torch.stack([x.delta_q for x in outputs], dim=1),
            log_sigma=torch.stack([x.log_sigma for x in outputs], dim=1),
            authority=torch.stack([x.authority for x in outputs], dim=1),
            hidden=hidden,
        )

    def forward(self, images: Tensor, state_features: Tensor, dt_seconds: Tensor, hidden: Optional[Tensor] = None) -> SequenceCorrection:
        """Alias for ``forward_sequence`` for standard PyTorch training calls."""
        return self.forward_sequence(images, state_features, dt_seconds, hidden=hidden)
