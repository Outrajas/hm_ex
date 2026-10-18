"""Authoritative 21-landmark hand kinematic topology."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

JOINTS: List[str] = [
    "WRIST",        # 0
    "THUMB_CMC",    # 1
    "THUMB_MCP",    # 2
    "THUMB_IP",     # 3
    "THUMB_TIP",    # 4
    "INDEX_MCP",    # 5
    "INDEX_PIP",    # 6
    "INDEX_DIP",    # 7
    "INDEX_TIP",    # 8
    "MIDDLE_MCP",   # 9
    "MIDDLE_PIP",   # 10
    "MIDDLE_DIP",   # 11
    "MIDDLE_TIP",   # 12
    "RING_MCP",     # 13
    "RING_PIP",     # 14
    "RING_DIP",     # 15
    "RING_TIP",     # 16
    "PINKY_MCP",    # 17
    "PINKY_PIP",    # 18
    "PINKY_DIP",    # 19
    "PINKY_TIP"     # 20
]

NUM_JOINTS: int = len(JOINTS)

PARENT_INDICES: Dict[int, int] = {
    0: -1,
    1: 0, 2: 1, 3: 2, 4: 3,       # Thumb
    5: 0, 6: 5, 7: 6, 8: 7,       # Index
    9: 0, 10: 9, 11: 10, 12: 11,  # Middle
    13: 0, 14: 13, 15: 14, 16: 15,# Ring
    17: 0, 18: 17, 19: 18, 20: 19 # Pinky
}

# Standard adult hand segment lengths in meters
BONE_LENGTHS: Dict[Tuple[int, int], float] = {
    (0, 1): 0.038, (1, 2): 0.035, (2, 3): 0.028, (3, 4): 0.022, # Thumb
    (0, 5): 0.085, (5, 6): 0.040, (6, 7): 0.025, (7, 8): 0.020, # Index
    (0, 9): 0.082, (9, 10): 0.045, (10, 11): 0.028, (11, 12): 0.022, # Middle
    (0, 13): 0.078, (13, 14): 0.040, (14, 15): 0.024, (15, 16): 0.020, # Ring
    (0, 17): 0.072, (17, 18): 0.033, (18, 19): 0.018, (19, 20): 0.016  # Pinky
}

# Mapping internal joint index (0 to 20) to anatomical motion semantics
# 21 internal DOFs:
# Thumb: CMC(2), MCP(2), IP(1)
# Fingers: MCP(2), PIP(1), DIP(1) each = 16
JOINT_DOF_NAMES: List[str] = [
    "thumb_cmc_abduction", "thumb_cmc_flexion",
    "thumb_mcp_abduction", "thumb_mcp_flexion",
    "thumb_ip_flexion",
    "index_mcp_abduction", "index_mcp_flexion",
    "index_pip_flexion", "index_dip_flexion",
    "middle_mcp_abduction", "middle_mcp_flexion",
    "middle_pip_flexion", "middle_dip_flexion",
    "ring_mcp_abduction", "ring_mcp_flexion",
    "ring_pip_flexion", "ring_dip_flexion",
    "pinky_mcp_abduction", "pinky_mcp_flexion",
    "pinky_pip_flexion", "pinky_dip_flexion"
]
NUM_INTERNAL_DOFS: int = len(JOINT_DOF_NAMES)