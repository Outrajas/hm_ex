"""Hierarchical forward kinematics supporting parametric hand biometrics."""

import importlib
from typing import Tuple, Optional
import numpy as np

_topology = importlib.import_module("00_spec.hand_topology")
_coord = importlib.import_module("00_spec.coordinate_system")
_state = importlib.import_module("00_spec.state_definition")
_model = importlib.import_module("02_state.hand_model")

NUM_JOINTS = _topology.NUM_JOINTS
BONE_LENGTHS = _topology.BONE_LENGTHS
quaternion_to_rotation_matrix = _coord.quaternion_to_rotation_matrix
HandState = _state.HandState
HandBiometrics = _model.HandBiometrics

def rot_splay(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

def rotx(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

def forward_kinematics(state: HandState, biometrics: Optional[HandBiometrics] = None) -> Tuple[np.ndarray, np.ndarray]:
    if biometrics is None:
        biometrics = HandBiometrics()

    scale = biometrics.overall_scale
    p_len = biometrics.palm_length_ratio
    p_wid = biometrics.palm_width_ratio
    f_len = biometrics.finger_length_ratio
    t_len = biometrics.thumb_ratio

    positions = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
    rotations = np.zeros((NUM_JOINTS, 3, 3), dtype=np.float64)

    R_root = quaternion_to_rotation_matrix(state.R)
    positions[0] = state.T
    rotations[0] = R_root

    # Anthropometric knuckle directions with lateral width scaling
    metacarpal_dirs = {
        1: np.array([0.70 * p_wid, 0.65, 0.28], dtype=np.float64),
        5: np.array([0.33 * p_wid, 0.944, 0.0], dtype=np.float64),
        9: np.array([0.02 * p_wid, 0.9998, 0.0], dtype=np.float64),
        13: np.array([-0.31 * p_wid, 0.951, 0.0], dtype=np.float64),
        17: np.array([-0.64 * p_wid, 0.768, 0.0], dtype=np.float64)
    }
    for k in metacarpal_dirs:
        metacarpal_dirs[k] /= np.linalg.norm(metacarpal_dirs[k])

    # 1. Thumb Chain
    L_01 = BONE_LENGTHS[(0, 1)] * scale * p_len
    dir_01 = metacarpal_dirs[1]
    R_cmc_rest = rot_splay(np.radians(35.0)) @ rotx(np.radians(15.0))
    R_cmc_art = rot_splay(state.q[0]) @ rotx(state.q[1])
    R_1 = R_root @ (R_cmc_rest @ R_cmc_art)
    positions[1] = positions[0] + R_root @ (dir_01 * L_01)
    rotations[1] = R_1

    L_12 = BONE_LENGTHS[(1, 2)] * scale * t_len
    R_mcp_art = rot_splay(state.q[2]) @ rotx(state.q[3])
    R_2 = R_1 @ R_mcp_art
    positions[2] = positions[1] + R_1 @ np.array([0.0, L_12, 0.0])
    rotations[2] = R_2

    L_23 = BONE_LENGTHS[(2, 3)] * scale * t_len
    R_ip_art = rotx(state.q[4])
    R_3 = R_2 @ R_ip_art
    positions[3] = positions[2] + R_2 @ np.array([0.0, L_23, 0.0])
    rotations[3] = R_3

    L_34 = BONE_LENGTHS[(3, 4)] * scale * t_len
    positions[4] = positions[3] + R_3 @ np.array([0.0, L_34, 0.0])
    rotations[4] = R_3

    # 2. Four Finger Chains
    finger_bases = [
        (5, 6, 7, 8, 5, metacarpal_dirs[5], np.radians(11.0)),
        (9, 10, 11, 12, 9, metacarpal_dirs[9], np.radians(1.0)),
        (13, 14, 15, 16, 13, metacarpal_dirs[13], np.radians(-10.0)),
        (17, 18, 19, 20, 17, metacarpal_dirs[17], np.radians(-23.0))
    ]

    for mcp_idx, pip_idx, dip_idx, tip_idx, q_start, base_dir, rest_fan in finger_bases:
        L_0_mcp = BONE_LENGTHS[(0, mcp_idx)] * scale * p_len
        positions[mcp_idx] = positions[0] + R_root @ (base_dir * L_0_mcp)

        total_abd = rest_fan + state.q[q_start]
        q_flex = state.q[q_start + 1]
        R_mcp = R_root @ (rot_splay(total_abd) @ rotx(q_flex))
        rotations[mcp_idx] = R_mcp

        L_mcp_pip = BONE_LENGTHS[(mcp_idx, pip_idx)] * scale * f_len
        positions[pip_idx] = positions[mcp_idx] + R_mcp @ np.array([0.0, L_mcp_pip, 0.0])
        R_pip = R_mcp @ rotx(state.q[q_start + 2])
        rotations[pip_idx] = R_pip

        L_pip_dip = BONE_LENGTHS[(pip_idx, dip_idx)] * scale * f_len
        positions[dip_idx] = positions[pip_idx] + R_pip @ np.array([0.0, L_pip_dip, 0.0])
        R_dip = R_pip @ rotx(state.q[q_start + 3])
        rotations[dip_idx] = R_dip

        L_dip_tip = BONE_LENGTHS[(dip_idx, tip_idx)] * scale * f_len
        positions[tip_idx] = positions[dip_idx] + R_dip @ np.array([0.0, L_dip_tip, 0.0])
        rotations[tip_idx] = R_dip

    return positions, rotations
