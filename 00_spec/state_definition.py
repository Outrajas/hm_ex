"""Authoritative Hand State Definition and Geometric Representations."""

from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np

GLOBAL_POS_DIM = 3
GLOBAL_ROT_DIM = 4       # Unit quaternion [w, x, y, z]
JOINT_DOF_COUNT = 21
MANO_BETA_DIM = 10

# PHASE B ARCHITECTURAL DIRECTIVE:
# The numerical Jacobian dG/d(delta_G) is identically zero in the calibrated MANO model.
# G = MANO(q, beta, T, R). delta_G is strictly quarantined from active neural state vectors.
ACTIVE_STATE_DIM = 65    # T (3) + R (4) + q (21) + V (3) + omega (3) + qdot (21) + beta (10)

@dataclass
class HandState:
    """Authoritative persistent hand tracking state vector X_t."""
    T: np.ndarray      # (3,) Global translation in meters
    R: np.ndarray      # (4,) Unit quaternion [w, x, y, z]
    q: np.ndarray      # (21,) Articulated joint angles in radians
    V: np.ndarray      # (3,) Linear velocity (m/s)
    omega: np.ndarray  # (3,) Angular velocity (rad/s)
    qdot: np.ndarray   # (21,) Joint angular velocities (rad/s)
    beta: np.ndarray = field(default_factory=lambda: np.zeros(MANO_BETA_DIM, dtype=np.float64))
    
    # Quarantined state (maintained for backward compatibility, not used in active model)
    delta_G: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float64))
    delta_G_dot: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float64))

    def copy(self) -> 'HandState':
        return HandState(
            T=self.T.copy(),
            R=self.R.copy(),
            q=self.q.copy(),
            V=self.V.copy(),
            omega=self.omega.copy(),
            qdot=self.qdot.copy(),
            beta=self.beta.copy(),
            delta_G=self.delta_G.copy(),
            delta_G_dot=self.delta_G_dot.copy()
        )

    @classmethod
    def neutral(cls, z_depth: float = 0.52) -> 'HandState':
        q = np.zeros(JOINT_DOF_COUNT, dtype=np.float64)
        q[5]  = np.radians(8.0)    # Index rest splay
        q[13] = np.radians(-6.0)   # Ring rest splay
        q[17] = np.radians(-14.0)  # Pinky rest splay

        return cls(
            T=np.array([0.0, -0.03, z_depth], dtype=np.float64),
            R=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            q=q,
            V=np.zeros(3, dtype=np.float64),
            omega=np.zeros(3, dtype=np.float64),
            qdot=np.zeros(JOINT_DOF_COUNT, dtype=np.float64),
            beta=np.zeros(MANO_BETA_DIM, dtype=np.float64)
        )

    def to_vector(self, include_quarantined: bool = False) -> np.ndarray:
        """Serializes state. Canonical active vector is 65D; legacy is 97D."""
        if include_quarantined:
            return np.concatenate([
                self.T, self.R, self.q,
                self.V, self.omega, self.qdot,
                self.beta, self.delta_G, self.delta_G_dot
            ])
        return np.concatenate([
            self.T, self.R, self.q,
            self.V, self.omega, self.qdot,
            self.beta
        ])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'HandState':
        if len(vec) == 55:  # Without shape beta
            return cls(
                T=vec[0:3].copy(), R=vec[3:7].copy(), q=vec[7:28].copy(),
                V=vec[28:31].copy(), omega=vec[31:34].copy(), qdot=vec[34:55].copy(),
                beta=np.zeros(MANO_BETA_DIM, dtype=np.float64)
            )
        elif len(vec) == 65:  # Canonical active state
            return cls(
                T=vec[0:3].copy(), R=vec[3:7].copy(), q=vec[7:28].copy(),
                V=vec[28:31].copy(), omega=vec[31:34].copy(), qdot=vec[34:55].copy(),
                beta=vec[55:65].copy()
            )
        elif len(vec) in [83, 87, 97]:  # Legacy padded vectors
            return cls(
                T=vec[0:3].copy(), R=vec[3:7].copy(), q=vec[7:28].copy(),
                V=vec[28:31].copy(), omega=vec[31:34].copy(), qdot=vec[34:55].copy(),
                beta=vec[55:65].copy() if len(vec) >= 65 else np.zeros(MANO_BETA_DIM, dtype=np.float64),
                delta_G=vec[65:81].copy() if len(vec) >= 81 else np.zeros(16, dtype=np.float64),
                delta_G_dot=vec[81:97].copy() if len(vec) >= 97 else np.zeros(16, dtype=np.float64)
            )
        raise ValueError(f"Unsupported vector dimension: {len(vec)}")

@dataclass
class SkeletonGeometry:
    joint_positions: np.ndarray    # (21, 3) 3D joint centers
    joint_rotations: np.ndarray    # (21, 3, 3) Local SO(3) frames
    bone_vectors: np.ndarray       # (20, 3) Segment vectors
    bone_lengths: np.ndarray       # (20,) Bone lengths in meters
    bone_frames: np.ndarray        # (20, 3, 3) Orthonormal bone charts

@dataclass
class GloveGeometry:
    vertices: np.ndarray           # (778, 3) MANO mesh vertices
    faces: np.ndarray              # (1538, 3) Triangular faces
    vertex_normals: np.ndarray     # (778, 3) Smooth vertex normals
    face_normals: np.ndarray       # (1538, 3) Face normals
    landmarks_21: np.ndarray       # (21, 3) Surface-registered landmarks

@dataclass
class StateSnapshot:
    state: HandState
    skeleton: SkeletonGeometry
    glove: GloveGeometry
    relationship: Any
    timestamp: float
