"""Build the detector-free RGB + simulated-prior input for live inference."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch import Tensor


_topology = importlib.import_module("00_spec.hand_topology")
_fk = importlib.import_module("02_state.forward_kinematics")
_mano = importlib.import_module("02_state.mano_surface")
_coord = importlib.import_module("00_spec.coordinate_system")
_state_def = importlib.import_module("00_spec.state_definition")

HandState = _state_def.HandState
HandBiometrics = importlib.import_module("02_state.hand_model").HandBiometrics
CameraIntrinsics = _coord.CameraIntrinsics
forward_kinematics = _fk.forward_kinematics
ManoSurface = _mano.ManoSurface
PARENT_INDICES = _topology.PARENT_INDICES
BONE_PAIRS = [(parent, child) for child, parent in PARENT_INDICES.items() if parent >= 0]


class RenderedPriorInput:
    """Crop an RGB frame around the current simulated hand and render its prior.

    Input channels are RGB, a MANO-surface silhouette, and the projected FK
    skeleton.  The surface mask uses the project's MANO_RIGHT.pkl skinned by
    its current forward-kinematics state; no hand-pose detector is involved.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        *,
        biometrics: Optional[HandBiometrics] = None,
        mano_model_path: Optional[str | Path] = None,
        output_size: int = 224,
        crop_margin: float = 1.6,
    ) -> None:
        self.intrinsics = intrinsics
        self.biometrics = HandBiometrics() if biometrics is None else biometrics
        self.output_size = int(output_size)
        self.crop_margin = float(crop_margin)
        if self.output_size < 64 or self.crop_margin < 1.0:
            raise ValueError("output_size must be >=64 and crop_margin must be >=1")
        model_path = None if mano_model_path is None else str(mano_model_path)
        self.mano = ManoSurface(model_path)
        if not self.mano.is_available:
            raise FileNotFoundError(
                "MANO_RIGHT.pkl is required to render the tracker silhouette. "
                "Pass its path to RenderedPriorInput(mano_model_path=...)."
            )

    def _project(self, points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points_mm = np.asarray(points_m, dtype=np.float32) * 1000.0
        z = points_mm[:, 2]
        valid = z > 1.0
        safe_z = np.where(valid, z, 1.0)
        uv = np.column_stack((
            self.intrinsics.fx * points_mm[:, 0] / safe_z + self.intrinsics.cx,
            self.intrinsics.fy * points_mm[:, 1] / safe_z + self.intrinsics.cy,
        )).astype(np.float32)
        return uv, valid

    def __call__(self, frame_bgr: np.ndarray, state: HandState) -> tuple[Tensor, np.ndarray]:
        """Return ``(5,H,W)`` float input and its image-to-crop affine.

        The BGR camera frame and current persistent state are the only runtime
        inputs.  ``state.T`` is in camera coordinates in meters.
        """
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame_bgr must have shape (H,W,3)")
        if frame.shape[1] != self.intrinsics.width or frame.shape[0] != self.intrinsics.height:
            raise ValueError("frame dimensions do not match CameraIntrinsics")

        joints_m, _ = forward_kinematics(state, self.biometrics)
        uv_joints, joint_valid = self._project(joints_m)
        if not joint_valid.any():
            raise ValueError("Simulated hand is behind the camera")
        visible_uv = uv_joints[joint_valid]
        low, high = visible_uv.min(axis=0), visible_uv.max(axis=0)
        center = (low + high) * 0.5
        side = max(float((high - low).max()), 24.0) * self.crop_margin
        left, top = center - side * 0.5
        scale = self.output_size / side
        affine = np.asarray([
            [scale, 0.0, -left * scale],
            [0.0, scale, -top * scale],
        ], dtype=np.float32)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.warpAffine(
            rgb, affine, (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        silhouette = np.zeros((self.output_size, self.output_size), dtype=np.uint8)
        vertices_m, _, _ = self.mano.forward(state, joints_m, self.biometrics)
        uv_vertices, vertex_valid = self._project(vertices_m)
        if int(vertex_valid.sum()) >= 3:
            crop_vertices = uv_vertices @ affine[:, :2].T + affine[:, 2]
            polygon = np.rint(crop_vertices[vertex_valid]).astype(np.int32)
            polygon[:, 0] = np.clip(polygon[:, 0], 0, self.output_size - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, self.output_size - 1)
            hull = cv2.convexHull(polygon)
            cv2.fillConvexPoly(silhouette, hull, 255, lineType=cv2.LINE_AA)

        skeleton = np.zeros_like(silhouette)
        crop_joints = uv_joints @ affine[:, :2].T + affine[:, 2]
        pixel_joints = np.rint(crop_joints).astype(np.int32)
        for parent, child in BONE_PAIRS:
            if joint_valid[parent] and joint_valid[child]:
                cv2.line(
                    skeleton,
                    tuple(pixel_joints[parent]),
                    tuple(pixel_joints[child]),
                    255,
                    2,
                    cv2.LINE_AA,
                )
        for joint_i, point in enumerate(pixel_joints):
            if joint_valid[joint_i]:
                cv2.circle(skeleton, tuple(point), 2, 255, -1, cv2.LINE_AA)

        channels = np.concatenate((
            crop_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0,
            silhouette[None].astype(np.float32) / 255.0,
            skeleton[None].astype(np.float32) / 255.0,
        ), axis=0)
        return torch.from_numpy(np.ascontiguousarray(channels)), affine
