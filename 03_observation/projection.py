"""Pinhole camera projection model and 2D bounds."""

import importlib
from typing import Tuple
import numpy as np

_coord = importlib.import_module("00_spec.coordinate_system")
CameraIntrinsics = _coord.CameraIntrinsics

def project_points_to_camera(points_3d: np.ndarray, intrinsics: CameraIntrinsics) -> Tuple[np.ndarray, np.ndarray]:
    points = np.atleast_2d(points_3d)
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    valid = Z > 0.01
    safe_Z = np.where(valid, Z, 1.0)
    u = (intrinsics.fx * (X / safe_Z)) + intrinsics.cx
    v = (intrinsics.fy * (Y / safe_Z)) + intrinsics.cy
    return np.column_stack([u, v]), valid
