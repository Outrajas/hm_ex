"""Map Re:InterHand 3D joint labels into the project's persistent hand state.

The source labels are camera-frame joints in millimeters.  This module provides
an analytic inverse-kinematics initializer for the project's current FK model;
it does not treat MANO's 45 axis-angle values as if they were the project's
21 differently-defined joint coordinates.  The reported residual should be
used to filter or weight examples when the two skeletons do not agree.
"""

from __future__ import annotations

import importlib
from typing import Optional

import numpy as np


_state_def = importlib.import_module("00_spec.state_definition")
_fk = importlib.import_module("02_state.forward_kinematics")
_model = importlib.import_module("02_state.hand_model")
_topology = importlib.import_module("00_spec.hand_topology")
_constraints = importlib.import_module("00_spec.constraints")

HandState = _state_def.HandState
HandBiometrics = _model.HandBiometrics
forward_kinematics = _fk.forward_kinematics

_PALM_JOINTS = np.asarray([0, 1, 5, 9, 13, 17], dtype=np.int64)
_PALM_PAIRS = ((0, 1), (0, 5), (0, 9), (0, 13), (0, 17))
_THUMB_PAIRS = ((1, 2), (2, 3), (3, 4))
_FINGER_PAIRS = (
    (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
)
_FINGER_ROOTS = (
    (5, 11.0),
    (9, 1.0),
    (13, -10.0),
    (17, -23.0),
)


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _splay(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _unit(vector: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    return vector / norm


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to [w,x,y,z]."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.asarray([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                        (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
            q = np.asarray([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                            (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
            q = np.asarray([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                            0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
            q = np.asarray([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                            (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q


def _proper_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Kabsch rotation mapping column-vector source coordinates to target."""
    covariance = source.T @ target
    u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    return vt.T @ correction @ u.T


def calibrate_biometrics_from_joints(
    joints_camera_mm: np.ndarray,
    *,
    valid: Optional[np.ndarray] = None,
) -> HandBiometrics:
    """Estimate fixed FK length/width ratios from training-partition labels.

    Fit this once per person using training frames only, then freeze it for
    validation and live tracking.  Overall scale is fixed to 1.0 to remove
    its multiplicative ambiguity with the length ratios.
    """
    joints = np.asarray(joints_camera_mm, dtype=np.float64)
    if joints.ndim == 2:
        joints = joints[None]
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError("joints_camera_mm must have shape (N,21,3) or (21,3)")
    masks = np.ones(joints.shape[:2], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if masks.ndim == 1 and joints.shape[0] == 1:
        masks = masks[None]
    if masks.shape != joints.shape[:2]:
        raise ValueError("valid must have shape (N,21)")

    def median_ratio(pairs):
        ratios = []
        for parent, child in pairs:
            use = masks[:, parent] & masks[:, child]
            if use.any():
                observed = np.linalg.norm(joints[use, child] - joints[use, parent], axis=-1)
                nominal = float(_topology.BONE_LENGTHS[(parent, child)]) * 1000.0
                ratios.extend((observed / nominal).tolist())
        if not ratios:
            raise ValueError("No valid bone measurements for anthropometric calibration")
        return float(np.clip(np.median(ratios), 0.70, 1.40))

    palm_length = median_ratio(_PALM_PAIRS)
    finger_length = median_ratio(_FINGER_PAIRS)
    thumb_length = median_ratio(_THUMB_PAIRS)
    bio = HandBiometrics(
        overall_scale=1.0,
        palm_length_ratio=palm_length,
        palm_width_ratio=1.0,
        finger_length_ratio=finger_length,
        thumb_ratio=thumb_length,
    )
    neutral = HandState.neutral(z_depth=0.0)
    neutral.T[:] = 0.0
    neutral.q[:] = 0.0

    # Estimate palm width from pairwise MCP distances, which are invariant to
    # camera orientation. The old x-axis ratio was biased when the nominal
    # palm shape and the observed hand differed; it also clipped at 1.5.
    mcp_ids = np.asarray([1, 5, 9, 13, 17], dtype=np.int64)
    pair_ids = [(int(mcp_ids[a]), int(mcp_ids[b]))
                for a in range(len(mcp_ids)) for b in range(a + 1, len(mcp_ids))]
    observed = []
    for parent, child in pair_ids:
        use = masks[:, parent] & masks[:, child]
        if use.any():
            observed.append(float(np.median(
                np.linalg.norm(joints[use, child] - joints[use, parent], axis=-1)
            )))
        else:
            observed.append(np.nan)
    observed = np.asarray(observed, dtype=np.float64)
    usable_pairs = np.isfinite(observed)
    if usable_pairs.any():
        best_width, best_error = bio.palm_width_ratio, float("inf")
        for width in np.linspace(0.50, 3.50, 301):
            bio.palm_width_ratio = float(width)
            template_m, _ = forward_kinematics(neutral, bio)
            predicted = np.asarray([
                np.linalg.norm(template_m[child] - template_m[parent]) * 1000.0
                for parent, child in pair_ids
            ])
            error = float(np.mean((predicted[usable_pairs] - observed[usable_pairs]) ** 2))
            if error < best_error:
                best_width, best_error = float(width), error
        bio.palm_width_ratio = best_width
    return bio


def _first_bone_angles(direction_local: np.ndarray) -> tuple[float, float]:
    """Invert splay(a) @ Rx(f) @ [0,1,0] for abduction and flexion."""
    dx, dy, dz = _unit(direction_local)
    # rot_splay maps +Y to [sin(a), cos(a), 0] in this project's coordinates.
    abduction = float(np.arctan2(dx, dy))
    flexion = float(np.arctan2(dz, np.hypot(dx, dy)))
    return abduction, flexion


def fit_state_from_camera_joints(
    joints_camera_mm: np.ndarray,
    *,
    biometrics: Optional[HandBiometrics] = None,
    beta: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
    clamp_limits: bool = True,
) -> tuple[HandState, np.ndarray, float]:
    """Fit one target skeleton into ``HandState`` using analytic IK.

    Args:
        joints_camera_mm: (21,3) project-order joints in camera coordinates,
            millimeters.  For Re:InterHand ``world_joints_right`` must first be
            converted to the camera frame.  For the local ReInterHand subset,
            apply ``KEEP_21_INDICES`` to ``orig_kp_right`` first.
        biometrics: fixed per-user FK shape calibration.  The default uses the
            project's nominal anthropometry; calibrate this once per subject
            before making final training targets.
        beta: optional 10D MANO shape metadata to preserve in HandState.  The
            current project's FK does not yet consume MANO beta.
        valid: optional per-joint validity mask.  Wrist and six palm anchors
            must be valid; invalid finger chains receive neutral angles.

    Returns:
        ``(state, reconstructed_camera_joints_mm, mean_joint_residual_mm)``.
        State translation is meters; q is radians; quaternion is [w,x,y,z].
    """
    joints = np.asarray(joints_camera_mm, dtype=np.float64)
    if joints.shape != (21, 3):
        raise ValueError(f"joints_camera_mm must have shape (21,3), got {joints.shape}")
    if not np.isfinite(joints).all():
        raise ValueError("joints_camera_mm contains non-finite values")
    mask = np.ones(21, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if mask.shape != (21,):
        raise ValueError("valid must have shape (21,)")
    if not mask[_PALM_JOINTS].all():
        raise ValueError("Wrist and all five palm anchors are required for root orientation")

    bio = HandBiometrics() if biometrics is None else biometrics
    neutral = HandState.neutral(z_depth=0.0)
    neutral.T[:] = 0.0
    neutral.q[:] = 0.0
    template_m, _ = forward_kinematics(neutral, bio)
    source = template_m[_PALM_JOINTS] - template_m[0]
    target = joints[_PALM_JOINTS] - joints[0]
    root_rotation = _proper_alignment(source, target)
    local = (joints - joints[0]) @ root_rotation

    q = np.zeros(21, dtype=np.float64)

    # Thumb CMC has a fixed rest frame followed by two learned rotations.
    if mask[[1, 2, 3, 4]].all():
        thumb_base = _splay(np.deg2rad(35.0)) @ _rx(np.deg2rad(15.0))
        d01 = _unit(local[2] - local[1])
        abduction, flexion = _first_bone_angles(thumb_base.T @ d01)
        q[0], q[1] = abduction, flexion
        R1 = thumb_base @ _splay(q[0]) @ _rx(q[1])
        d12 = _unit(local[3] - local[2])
        q[2], q[3] = _first_bone_angles(R1.T @ d12)
        R2 = R1 @ _splay(q[2]) @ _rx(q[3])
        d23 = _unit(local[4] - local[3])
        distal_local = R2.T @ d23
        q[4] = np.arctan2(distal_local[2], distal_local[1])

    for base_idx, rest_fan_deg in _FINGER_ROOTS:
        joints_chain = [base_idx, base_idx + 1, base_idx + 2, base_idx + 3]
        if not mask[joints_chain].all():
            continue
        q_start = base_idx
        first_direction = _unit(local[base_idx + 1] - local[base_idx])
        abduction, flexion = _first_bone_angles(first_direction)
        q[q_start] = abduction - np.deg2rad(rest_fan_deg)
        q[q_start + 1] = flexion

        R_mcp = _splay(np.deg2rad(rest_fan_deg) + q[q_start]) @ _rx(q[q_start + 1])
        d_pip = _unit(local[base_idx + 2] - local[base_idx + 1])
        pip_local = R_mcp.T @ d_pip
        q[q_start + 2] = np.arctan2(pip_local[2], pip_local[1])

        R_pip = R_mcp @ _rx(q[q_start + 2])
        d_dip = _unit(local[base_idx + 3] - local[base_idx + 2])
        dip_local = R_pip.T @ d_dip
        q[q_start + 3] = np.arctan2(dip_local[2], dip_local[1])

    if clamp_limits:
        q = np.clip(q, _constraints.JOINT_LIMITS_LOWER, _constraints.JOINT_LIMITS_UPPER)

    state = HandState(
        T=joints[0] / 1000.0,
        R=_matrix_to_quaternion(root_rotation),
        q=q,
        V=np.zeros(3, dtype=np.float64),
        omega=np.zeros(3, dtype=np.float64),
        qdot=np.zeros(21, dtype=np.float64),
        beta=np.zeros(10, dtype=np.float64) if beta is None else np.asarray(beta, dtype=np.float64).reshape(10).copy(),
    )
    reconstructed_m, _ = forward_kinematics(state, bio)
    reconstructed_mm = reconstructed_m * 1000.0
    residual = float(np.linalg.norm(reconstructed_mm - joints, axis=-1).mean())
    return state, reconstructed_mm, residual


def fit_sequence_state_targets(
    joints_camera_mm: np.ndarray,
    timestamps_seconds: np.ndarray,
    *,
    biometrics: Optional[HandBiometrics] = None,
    beta: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a chronological sequence and derive finite-difference velocities.

    Inputs are a single identity/segment sequence.  This deliberately has no
    sequence crossing or sorting logic; the streaming dataset is responsible
    for supplying correctly ordered frames and timestamps.

    Returns the (N,65) serialized state vectors and (N,) FK residuals in mm.
    Velocities use the actual timestamp differences and are clamped to the
    project's velocity limits to prevent sparse-frame gaps creating extremes.
    """
    joints = np.asarray(joints_camera_mm, dtype=np.float64)
    times = np.asarray(timestamps_seconds, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError("joints_camera_mm must have shape (N,21,3)")
    if times.shape != (len(joints),) or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError("timestamps_seconds must be strictly increasing and shape (N,)")

    states, residuals = [], []
    for frame in joints:
        state, _, residual = fit_state_from_camera_joints(
            frame, biometrics=biometrics, beta=beta
        )
        states.append(state)
        residuals.append(residual)

    if len(states) > 1:
        for idx in range(len(states)):
            left = max(0, idx - 1)
            right = min(len(states) - 1, idx + 1)
            span = max(float(times[right] - times[left]), 1e-4)
            a, b = states[left], states[right]
            velocity = np.clip((b.T - a.T) / span,
                               -_constraints.MAX_TRANSLATIONAL_VELOCITY,
                               _constraints.MAX_TRANSLATIONAL_VELOCITY)
            qdot = np.clip((b.q - a.q) / span,
                           -_constraints.MAX_JOINT_VELOCITY,
                           _constraints.MAX_JOINT_VELOCITY)
            states[idx].V = velocity
            states[idx].qdot = qdot

    vectors = np.stack([state.to_vector() for state in states]) if states else np.zeros((0, 65))
    return vectors, np.asarray(residuals, dtype=np.float32)
