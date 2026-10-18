"""Global simulation and sensor parameters."""

from .coordinate_system import CameraIntrinsics

DEFAULT_CAMERA = CameraIntrinsics(
    fx=615.0,
    fy=615.0,
    cx=320.0,
    cy=240.0,
    width=640,
    height=480
)

DEFAULT_DT: float = 1.0 / 60.0  # 60 FPS baseline