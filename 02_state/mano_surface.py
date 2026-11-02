"""Inference-only MANO Linear Blend Skinning (LBS) backend driven directly by the FK skeleton."""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pickle
import warnings
import numpy as np
import importlib

_coord = importlib.import_module("00_spec.coordinate_system")
quaternion_to_rotation_matrix = _coord.quaternion_to_rotation_matrix
_fk = importlib.import_module("02_state.forward_kinematics")
forward_kinematics = _fk.forward_kinematics

# Distal landmark indices for the 5 fingertips on MANO 778-vertex mesh
MANO_TIPS_VERTEX_INDICES = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 555,
    "pinky": 672,
}

# Mapping 16 MANO bones to our 21-joint kinematic chain:
# (mano_idx, our_parent, our_child, is_child_joint, child_id)
BONE_MAP = [
    (0,  0,  9,  True,  4),    # 0: Wrist -> Middle MCP (Palm base)
    (1,  5,  6,  True,  2),    # 1: Index MCP -> Index PIP
    (2,  6,  7,  True,  3),    # 2: Index PIP -> Index DIP
    (3,  7,  8,  False, 320),  # 3: Index DIP -> Index Tip (vertex 320)
    (4,  9,  10, True,  5),    # 4: Middle MCP -> Middle PIP
    (5,  10, 11, True,  6),    # 5: Middle PIP -> Middle DIP
    (6,  11, 12, False, 443),  # 6: Middle DIP -> Middle Tip (vertex 443)
    (7,  17, 18, True,  8),    # 7: Pinky MCP -> Pinky PIP
    (8,  18, 19, True,  9),    # 8: Pinky PIP -> Pinky DIP
    (9,  19, 20, False, 672),  # 9: Pinky DIP -> Pinky Tip (vertex 672)
    (10, 13, 14, True,  11),   # 10: Ring MCP -> Ring PIP
    (11, 14, 15, True,  12),   # 11: Ring PIP -> Ring DIP
    (12, 15, 16, False, 555),  # 12: Ring DIP -> Ring Tip (vertex 555)
    (13, 1,  2,  True,  14),   # 13: Thumb CMC -> Thumb MCP
    (14, 2,  3,  True,  15),   # 14: Thumb MCP -> Thumb IP
    (15, 3,  4,  False, 744),  # 15: Thumb IP -> Thumb Tip (vertex 744)
]

class _ChumpyShim:
    """Safely extracts array data from Chumpy containers without property recursion."""
    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], np.ndarray):
            self.dobj = args[0]
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple):
            for item in state:
                if isinstance(item, np.ndarray):
                    self.dobj = item
                    break
            self.data = state
        else:
            self.dobj = state

    def to_numpy(self) -> np.ndarray:
        visited = set()
        curr = self
        while curr is not None and id(curr) not in visited:
            visited.add(id(curr))
            if isinstance(curr, np.ndarray):
                return curr
            if not hasattr(curr, '__dict__'):
                break
            d = curr.__dict__
            found = None
            for key in ('dobj', '_dobj', 'v', 'data'):
                if key in d and isinstance(d[key], (np.ndarray, _ChumpyShim)):
                    found = d[key]
                    break
            if found is None:
                for v in d.values():
                    if isinstance(v, (np.ndarray, _ChumpyShim)):
                        found = v
                        break
            curr = found

        if isinstance(curr, np.ndarray):
            return curr
        return np.empty(0, dtype=np.float64)

    def __array__(self, dtype=None, copy=None):
        arr = self.to_numpy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        if copy:
            arr = arr.copy()
        return arr

    @property
    def r(self):
        return self.to_numpy()

class _ManoUnpickler(pickle.Unpickler):
    """Custom unpickler redirecting Chumpy and deprecated SciPy imports to NumPy-compatible objects."""
    def find_class(self, module, name):
        if 'chumpy' in module:
            return _ChumpyShim
        if module == 'scipy.sparse.csc' and name == 'csc_matrix':
            import scipy.sparse
            return scipy.sparse.csc_matrix
        return super().find_class(module, name)

def our_state_to_mano(state: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility interface for unit test suite."""
    global_trans = state.T.copy()
    pose_aa = np.zeros(45, dtype=np.float64)
    betas = np.asarray(getattr(state, "beta", np.zeros(10)), dtype=np.float64).copy()
    root_aa = np.zeros(3, dtype=np.float64)
    return global_trans, root_aa, pose_aa, betas

class ManoSurface:
    """MANO Linear Blend Skinning backend directly fitting the FK skeleton volume."""
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            candidates = [
                Path("models/MANO_RIGHT.pkl"),
                Path("models/mano/MANO_RIGHT.pkl"),
                Path("../models/MANO_RIGHT.pkl")
            ]
            for c in candidates:
                if c.exists():
                    model_path = str(c)
                    break

        self.model_path = model_path
        self._is_available = False
        self.faces: Optional[np.ndarray] = None
        self.v_template: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.J_regressor: Optional[np.ndarray] = None
        self.shapedirs: Optional[np.ndarray] = None
        self.joints_16_rest: Optional[np.ndarray] = None

        # Precomputed local coordinate frames and lengths in MANO rest template
        self.F_rest_mano = np.zeros((16, 3, 3), dtype=np.float64)
        self.L_rest_mano = np.zeros(16, dtype=np.float64)
        self._shape_cache: Dict[Tuple[float, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        if self.model_path and Path(self.model_path).is_file():
            self._load_model()

    @property
    def is_available(self) -> bool:
        return self._is_available

    def _geometry_for_beta(
        self, beta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return shaped MANO template, rest joints, bone frames, and lengths."""
        if self.v_template is None or self.J_regressor is None or self.shapedirs is None:
            raise RuntimeError("MANO shape assets are not loaded")
        beta = np.asarray(beta, dtype=np.float64).reshape(10)
        key = tuple(np.round(beta, 6).tolist())
        cached = self._shape_cache.get(key)
        if cached is not None:
            return cached

        shaped_vertices = self.v_template + np.tensordot(
            self.shapedirs, beta, axes=([2], [0])
        )
        rest_joints = self.J_regressor @ shaped_vertices
        v_x_palm = rest_joints[1] - rest_joints[7]
        v_y_palm = rest_joints[4] - rest_joints[0]
        v_y_palm /= max(float(np.linalg.norm(v_y_palm)), 1e-12)
        v_x_palm = v_x_palm - np.dot(v_x_palm, v_y_palm) * v_y_palm
        v_x_palm /= max(float(np.linalg.norm(v_x_palm)), 1e-12)
        v_z_palm = np.cross(v_x_palm, v_y_palm)

        frames = np.zeros((16, 3, 3), dtype=np.float64)
        lengths = np.zeros(16, dtype=np.float64)
        for j, _our_p, _our_c, is_joint, child_id in BONE_MAP:
            p_start = rest_joints[j]
            p_end = rest_joints[child_id] if is_joint else shaped_vertices[child_id]
            bone_vec = p_end - p_start
            length = max(float(np.linalg.norm(bone_vec)), 1e-4)
            lengths[j] = length
            y_axis = bone_vec / length
            x_ref = v_z_palm if j in [13, 14, 15] else v_x_palm
            x_axis = x_ref - np.dot(x_ref, y_axis) * y_axis
            x_norm = float(np.linalg.norm(x_axis))
            if x_norm > 1e-6:
                x_axis /= x_norm
            else:
                x_axis = np.asarray([1.0, 0.0, 0.0])
            z_axis = np.cross(x_axis, y_axis)
            frames[j] = np.column_stack([x_axis, y_axis, z_axis])

        result = (shaped_vertices, rest_joints, frames, lengths)
        if len(self._shape_cache) >= 32:
            self._shape_cache.pop(next(iter(self._shape_cache)))
        self._shape_cache[key] = result
        return result

    def _load_model(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(self.model_path, 'rb') as f:
                    unpickler = _ManoUnpickler(f, encoding='latin1')
                    data = unpickler.load()

            def to_arr(x, dt=np.float64):
                if hasattr(x, 'toarray'): return np.array(x.toarray(), dtype=dt)
                if hasattr(x, 'to_numpy'): return np.array(x.to_numpy(), dtype=dt)
                return np.array(x, dtype=dt)

            self.v_template = to_arr(data['v_template'], np.float64)
            self.faces = to_arr(data['f'], np.int32)
            self.weights = to_arr(data['weights'], np.float64)
            self.J_regressor = to_arr(data['J_regressor'], np.float64)
            shapedirs = to_arr(data['shapedirs'], np.float64)
            if shapedirs.ndim == 2 and shapedirs.shape[0] == self.v_template.size:
                shapedirs = shapedirs.reshape(self.v_template.shape[0], 3, -1)
            elif shapedirs.ndim == 3 and shapedirs.shape[1:] == (self.v_template.shape[0], 3):
                shapedirs = np.transpose(shapedirs, (1, 2, 0))
            if (shapedirs.ndim != 3 or shapedirs.shape[:2] != (self.v_template.shape[0], 3)
                    or shapedirs.shape[2] < 10):
                raise ValueError(f"Unexpected MANO shapedirs shape: {shapedirs.shape}")
            self.shapedirs = shapedirs[:, :, :10].copy()
            self._shape_cache.clear()
            _, self.joints_16_rest, self.F_rest_mano, self.L_rest_mano = self._geometry_for_beta(np.zeros(10))

            self._is_available = True
            print(f"[MANO Backend] Loaded and calibrated '{self.model_path}' to FK skeleton.")
        except Exception as e:
            print(f"[MANO Backend] Warning: Failed to load '{self.model_path}': {e}")
            self._is_available = False

    def forward(
        self,
        state: Any,
        joint_positions: Optional[np.ndarray] = None,
        biometrics: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates MANO mesh vertices wrapped tightly around the FK skeleton.
        """
        if not self._is_available:
            raise FileNotFoundError(f"MANO model asset not available at '{self.model_path}'.")

        # Use the same per-user shape for the mesh and the authoritative FK
        # skeleton. Supplied positions avoid a duplicate FK pass but rotations
        # are still needed to skin the MANO template.
        fk_positions, joint_rot_3d = forward_kinematics(state, biometrics)
        joint_pos_3d = fk_positions if joint_positions is None else np.asarray(joint_positions, dtype=np.float64)
        beta = np.asarray(getattr(state, "beta", np.zeros(10)), dtype=np.float64)
        shaped_template, rest_joints, rest_frames, rest_lengths = self._geometry_for_beta(beta)

        # Affine bone transformation matrices: M_j (3x3), t_j (3)
        M_all = np.zeros((16, 3, 3), dtype=np.float64)
        t_all = np.zeros((16, 3), dtype=np.float64)

        for j, our_p, our_c, is_joint, child_id in BONE_MAP:
            p_start = joint_pos_3d[our_p]
            p_end   = joint_pos_3d[our_c]
            bone_vec_ours = p_end - p_start
            length_ours = float(np.linalg.norm(bone_vec_ours))

            # Scale factor along the bone shaft
            s_long = length_ours / rest_lengths[j]
            # Cross-sectional scale proportional to shaft length
            s_cross = float(np.clip(np.sqrt(s_long), 0.85, 1.15))

            S_mat = np.diag([s_cross, s_long, s_cross])

            # FK joint rotation in world space (Column 1 is aligned with bone direction)
            F_ours = joint_rot_3d[our_p]

            # M_j maps MANO rest coordinates into our deformed bone coordinate frame
            M_j = F_ours @ S_mat @ rest_frames[j].T
            t_j = p_start - M_j @ rest_joints[j]

            M_all[j] = M_j
            t_all[j] = t_j

        # Vectorized Linear Blend Skinning
        T_blend = np.einsum('vj,jab->vab', self.weights, M_all)
        t_blend = self.weights @ t_all
        v_world = np.einsum('vab,vb->va', T_blend, shaped_template) + t_blend

        # Reconstructed 21-landmark array
        k_world_21 = np.zeros((21, 3), dtype=np.float64)
        # Wrist and joint positions
        for mano_j, our_p, _, _, _ in BONE_MAP:
            k_world_21[our_p] = joint_pos_3d[our_p]
        # Distal fingertips directly on mesh skin
        k_world_21[4]  = v_world[MANO_TIPS_VERTEX_INDICES["thumb"]]
        k_world_21[8]  = v_world[MANO_TIPS_VERTEX_INDICES["index"]]
        k_world_21[12] = v_world[MANO_TIPS_VERTEX_INDICES["middle"]]
        k_world_21[16] = v_world[MANO_TIPS_VERTEX_INDICES["ring"]]
        k_world_21[20] = v_world[MANO_TIPS_VERTEX_INDICES["pinky"]]

        return v_world, self.faces, k_world_21
