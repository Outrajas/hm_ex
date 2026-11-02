"""Authoritative coordinate transformations, SO(3) algebra, and Euler mappings."""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    q = quaternion_normalize(q)
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x**2 + y**2)]
    ], dtype=np.float64)


def rotation_vector_to_quaternion(rot_vec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rot_vec)
    if theta < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rot_vec / theta
    half_theta = 0.5 * theta
    w = np.cos(half_theta)
    xyz = axis * np.sin(half_theta)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (radians: X-roll, Y-pitch, Z-yaw) to Hamilton quaternion."""
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return quaternion_normalize(np.array([w, x, y, z], dtype=np.float64))


def quaternion_geodesic_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = quaternion_normalize(q1)
    q2 = quaternion_normalize(q2)
    inner = np.abs(np.clip(np.dot(q1, q2), -1.0, 1.0))
    return float(2.0 * np.arccos(inner))


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    num_pts = points.shape[0]
    homogeneous = np.hstack([points, np.ones((num_pts, 1), dtype=np.float64)])
    transformed = (T @ homogeneous.T).T
    return transformed[:, :3]
