"""Physiological joint range of motion limits and maximum velocities."""

import numpy as np

# Anatomical limits in radians: [min_angle, max_angle]
JOINT_LIMITS_LOWER = np.array([
    # Thumb CMC: Abduction [-0.35, 0.70], Flexion [-0.20, 0.60]
    -0.35, -0.20,
    # Thumb MCP: Abduction [-0.25, 0.25], Flexion [-0.10, 0.90]
    -0.25, -0.10,
    # Thumb IP: Flexion [-0.15, 1.40]
    -0.15,
    # Index: MCP Abduction [-0.35, 0.35], Flexion [0.0, 1.57], PIP [0.0, 1.74], DIP [0.0, 1.40]
    -0.35, 0.00, 0.00, 0.00,
    # Middle: MCP Abduction [-0.25, 0.25], Flexion [0.0, 1.57], PIP [0.0, 1.74], DIP [0.0, 1.40]
    -0.25, 0.00, 0.00, 0.00,
    # Ring: MCP Abduction [-0.25, 0.25], Flexion [0.0, 1.57], PIP [0.0, 1.74], DIP [0.0, 1.40]
    -0.25, 0.00, 0.00, 0.00,
    # Pinky: MCP Abduction [-0.35, 0.35], Flexion [0.0, 1.57], PIP [0.0, 1.74], DIP [0.0, 1.40]
    -0.35, 0.00, 0.00, 0.00
], dtype=np.float64)

JOINT_LIMITS_UPPER = np.array([
    0.70, 0.60,
    0.25, 0.90,
    1.40,
    0.35, 1.57, 1.74, 1.40,
    0.25, 1.57, 1.74, 1.40,
    0.25, 1.57, 1.74, 1.40,
    0.35, 1.57, 1.74, 1.40
], dtype=np.float64)

# Velocity constraints
MAX_TRANSLATIONAL_VELOCITY: float = 3.0   # m/s
MAX_ANGULAR_VELOCITY: float = 15.0        # rad/s
MAX_JOINT_VELOCITY: float = 20.0          # rad/s