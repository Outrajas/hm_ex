"""Numerical Jacobian estimation and sensitivity analysis for delta_G on MANO surface."""

import sys
from pathlib import Path
import importlib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.import_module("00_spec.state_definition")
_fk = importlib.import_module("02_state.forward_kinematics")
_mano = importlib.import_module("02_state.mano_surface")

HandState = _spec.HandState
forward_kinematics = _fk.forward_kinematics
ManoSurface = _mano.ManoSurface

def run_deltaG_sensitivity_analysis():
    print("=================================================================")
    print(" SENSITIVITY ANALYSIS: Numerical Jacobian J = dG / d(delta_G)")
    print("=================================================================")

    mano = ManoSurface()
    if not mano.is_available:
        print("[ERROR] MANO model asset not available.")
        return

    base_state = HandState.neutral(z_depth=0.52)
    j_pos, _ = forward_kinematics(base_state)
    verts_base, _, _ = mano.forward(base_state, j_pos)
    v0_flat = verts_base.flatten()

    n_deltas = len(base_state.delta_G)
    eps_list = [1e-4, 1e-2, 0.1, 1.0]

    J = np.zeros((len(v0_flat), n_deltas), dtype=np.float64)
    max_disp_per_var = np.zeros(n_deltas, dtype=np.float64)
    mean_disp_per_var = np.zeros(n_deltas, dtype=np.float64)

    for i in range(n_deltas):
        disp_for_eps = []
        for eps in eps_list:
            pert_state = base_state.copy()
            pert_state.delta_G[i] += eps
            v_pert, _, _ = mano.forward(pert_state, j_pos)
            diff = v_pert.flatten() - v0_flat
            max_d = float(np.max(np.abs(diff)))
            disp_for_eps.append(max_d)

        pert_state = base_state.copy()
        pert_state.delta_G[i] += 0.01
        v_pert, _, _ = mano.forward(pert_state, j_pos)
        col = (v_pert.flatten() - v0_flat) / 0.01
        J[:, i] = col

        max_disp_per_var[i] = np.max(disp_for_eps)
        mean_disp_per_var[i] = np.mean(disp_for_eps)

    _, s_J, _ = np.linalg.svd(J, full_matrices=False)
    rank_tol = 1e-7
    eff_rank = int(np.sum(s_J > rank_tol))

    print(f"Jacobian Dimensions        : {J.shape[0]} vertices x {J.shape[1]} delta_G variables")
    print(f"Singular Values of J       : max={np.max(s_J):.4e}, min={np.min(s_J):.4e}")
    print(f"Numerical Rank of J (tol)  : {eff_rank} / {n_deltas}")
    print(f"Max Vertex Displacement    : {np.max(max_disp_per_var)*1000.0:6.4f} mm")
    print(f"Mean Vertex Displacement   : {np.mean(mean_disp_per_var)*1000.0:6.4f} mm")

    print("\n--- Variable Sensitivity Inspection ---")
    for i in range(n_deltas):
        print(f"delta_G[{i:2d}]: max_disp={max_disp_per_var[i]*1000.0:6.4f} mm | J_col_norm={np.linalg.norm(J[:, i]):.4e}")

    print("\n[ARCHITECTURAL VERIFICATION]")
    if eff_rank == 0:
        print("EXPLICIT FINDING: delta_G currently has IDENTICALLY ZERO EFFECT on the MANO mesh.")
        print("Reason: MANO LBS deformation is driven strictly by articulated joint angles (q) and shape (beta).")
        print("delta_G exists in HandState as a decoupled temporal state for future parametric tissue modulation.")
    else:
        print(f"FINDING: delta_G actively perturbs the surface with effective rank {eff_rank}.")

if __name__ == "__main__":
    run_deltaG_sensitivity_analysis()
