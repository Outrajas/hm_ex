"""Deterministic physiological constraint projection."""

import importlib
import numpy as np

_state = importlib.import_module("00_spec.state_definition")
_spec_const = importlib.import_module("00_spec.constraints")

HandState = _state.HandState
JOINT_LIMITS_LOWER = _spec_const.JOINT_LIMITS_LOWER
JOINT_LIMITS_UPPER = _spec_const.JOINT_LIMITS_UPPER
MAX_TRANSLATIONAL_VELOCITY = _spec_const.MAX_TRANSLATIONAL_VELOCITY
MAX_ANGULAR_VELOCITY = _spec_const.MAX_ANGULAR_VELOCITY
MAX_JOINT_VELOCITY = _spec_const.MAX_JOINT_VELOCITY

def project_constraints(state: HandState) -> HandState:
    clamped_state = state.copy()
    clamped_state.q = np.clip(clamped_state.q, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    v_norm = np.linalg.norm(clamped_state.V)
    if v_norm > MAX_TRANSLATIONAL_VELOCITY:
        clamped_state.V = (clamped_state.V / v_norm) * MAX_TRANSLATIONAL_VELOCITY

    omega_norm = np.linalg.norm(clamped_state.omega)
    if omega_norm > MAX_ANGULAR_VELOCITY:
        clamped_state.omega = (clamped_state.omega / omega_norm) * MAX_ANGULAR_VELOCITY

    clamped_state.qdot = np.clip(clamped_state.qdot, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY)
    return clamped_state
