"""Camera models and external segmentation observation contracts."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import importlib

_coord = importlib.import_module("00_spec.coordinate_system")
CameraIntrinsics = _coord.CameraIntrinsics

@dataclass(frozen=True)
class HandObservation:
    """External sensor observation packet for learned state estimation."""
    timestamp: float
    frame_id: int
    mask: np.ndarray                      # (H, W) uint8 binary silhouette mask [0, 255]
    contour: Optional[np.ndarray]         # (P, 2) float32 ordered boundary pixels
    bounding_box: np.ndarray              # [u_min, v_min, u_max, v_max]
    confidence: float                     # Quality score in [0.0, 1.0]
    camera_intrinsics: CameraIntrinsics   # Geometric pinhole projection model
