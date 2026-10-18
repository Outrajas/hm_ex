"""Anthropometric specification and biometric configuration for parametric hands."""

from dataclasses import dataclass

@dataclass
class HandBiometrics:
    overall_scale: float = 1.0        # 0.70 to 1.40
    palm_length_ratio: float = 1.0    # 0.70 to 1.40
    palm_width_ratio: float = 1.0     # 0.70 to 1.50
    finger_length_ratio: float = 1.0  # 0.70 to 1.40
    thumb_ratio: float = 1.0          # 0.70 to 1.40
    thickness_ratio: float = 1.0      # 0.60 to 1.60
    glove_alpha: float = 0.55         # 0.15 to 0.95
