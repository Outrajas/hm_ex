# 02_state/dynamics.py
"""Deterministic temporal prediction machinery for state and derived geometries."""

import importlib
from typing import Tuple
import numpy as np

_spec = importlib.import_module("00_spec.state_definition")
_coord = importlib.import_module("00_spec.coordinate_system")
_fk = importlib.import_module("02_state.forward_kinematics")
_rel = importlib.import_module("03_observation.relationship")

HandState = _spec.HandState
StateSnapshot = _spec.StateSnapshot
rotation_vector_to_quaternion = _coord.rotation_vector_to_quaternion
quaternion_multiply = _coord.quaternion_multiply
quaternion_normalize = _coord.quaternion_normalize
forward_kinematics = _fk.forward_kinematics
compute_skeleton_geometry = _rel.compute_skeleton_geometry
compute_glove_geometry = _rel.compute_glove_geometry
evaluate_relationship = _rel.evaluate_relationship

VELOCITY_DAMPING = 0.985  # Deterministic physical dissipation per frame

def predict_kinematics(state: HandState, dt: float) -> HandState:
    """Deterministic motion extrapolator X_t^- = F(X_{t-1}, dt)."""
    dt_safe = float(np.clip(dt, 1e-4, 0.05))

    # Extrapolate translation & orientation
    T_next = state.T + state.V * dt_safe
    delta_q_rot = rotation_vector_to_quaternion(state.omega * dt_safe)
    R_next = quaternion_normalize(quaternion_multiply(delta_q_rot, state.R))

    # Extrapolate articulated skeletal angles
    q_next = state.q + state.qdot * dt_safe

    # Propagate surface tissue compliance
    delta_G_next = state.delta_G + state.delta_G_dot * dt_safe

    # Damped velocity continuation
    V_next = state.V * (VELOCITY_DAMPING ** (dt_safe * 60.0))
    omega_next = state.omega * (VELOCITY_DAMPING ** (dt_safe * 60.0))
    qdot_next = state.qdot * (VELOCITY_DAMPING ** (dt_safe * 60.0))
    delta_G_dot_next = state.delta_G_dot * (VELOCITY_DAMPING ** (dt_safe * 60.0))

    return HandState(
        T=T_next,
        R=R_next,
        q=q_next,
        V=V_next,
        omega=omega_next,
        qdot=qdot_next,
        beta=state.beta.copy(),  # Shape identity remains constant
        delta_G=delta_G_next,
        delta_G_dot=delta_G_dot_next
    )

def predict_full_snapshot(snapshot_prev: StateSnapshot, mano_backend: Any, dt: float) -> StateSnapshot:
    """Produces the complete temporal prediction bundle (X_t^-, K_t^-, G_t^-, R_t^-).
    
    The relationship R_t^- is evaluated from (K_t^-, G_t^-) and never independently integrated.
    """
    # 1. Extrapolate state
    predicted_state = predict_kinematics(snapshot_prev.state, dt)

    # 2. Derive predicted skeleton
    joint_pos, joint_rot = forward_kinematics(predicted_state)
    predicted_skeleton = compute_skeleton_geometry(joint_pos, joint_rot)

    # 3. Derive predicted MANO surface
    verts, faces, k21 = mano_backend.forward(predicted_state, joint_pos)
    predicted_glove = compute_glove_geometry(verts, faces, k21)

    # 4. Recompute predicted relationship directly from predicted geometries
    predicted_relationship = evaluate_relationship(predicted_skeleton, predicted_glove)

    return StateSnapshot(
        state=predicted_state,
        skeleton=predicted_skeleton,
        glove=predicted_glove,
        relationship=predicted_relationship,
        timestamp=snapshot_prev.timestamp + dt
    )