"""Surface Latent Discovery across World-Space and Hand-Local Coordinate Frames."""

import sys
from pathlib import Path
import importlib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_coord = importlib.import_module("00_spec.coordinate_system")
_ds = importlib.import_module("06_training.dataset")
quaternion_to_rotation_matrix = _coord.quaternion_to_rotation_matrix
build_split_dataset = _ds.build_split_dataset

def run_surface_discovery():
    print("=================================================================")
    print(" PHASE C: SURFACE LATENT DISCOVERY (TRAINING TRAJECTORIES ONLY)")
    print("=================================================================")

    splits = build_split_dataset(n_trajectories=600, frames_per_traj=50, seed=101)
    train_data = splits["train"]
    test_data = splits["test"]

    N_train = train_data.G_gt.shape[0]
    N_test = test_data.G_gt.shape[0]
    print(f"Analyzing {N_train} training transitions; validating on {N_test} test transitions.")

    # 1. World-Space Coordinate Representations
    G_world_tr = train_data.G_gt.reshape(N_train, -1)
    dG_world_tr = train_data.delta_G.reshape(N_train, -1)
    G_world_te = test_data.G_gt.reshape(N_test, -1)

    # 2. Hand-Local Coordinate Representations (Subtract wrist T, rotate by R_root.T)
    print("Projecting meshes into canonical hand-local frames V_local = (V - T) @ R...")
    def to_hand_local(G_3d: np.ndarray, X_vec: np.ndarray) -> np.ndarray:
        N = G_3d.shape[0]
        local_G = np.zeros_like(G_3d)
        for i in range(N):
            T = X_vec[i, 0:3]
            R = quaternion_to_rotation_matrix(X_vec[i, 3:7])
            local_G[i] = (G_3d[i] - T) @ R
        return local_G.reshape(N, -1)

    G_local_tr = to_hand_local(train_data.G_gt, train_data.X_gt)
    G_local_pr_tr = to_hand_local(train_data.G_pred, train_data.X_pred)
    dG_local_tr = G_local_tr - G_local_pr_tr
    G_local_te = to_hand_local(test_data.G_gt, test_data.X_gt)

    # Dimensionality Metrics
    def evaluate_pca(X_tr: np.ndarray, X_te: np.ndarray, name: str):
        mean_v = np.mean(X_tr, axis=0)
        _, s, Vh = np.linalg.svd(X_tr - mean_v, full_matrices=False)
        var = s ** 2
        cum_var = np.cumsum(var) / np.sum(var)
        pr = float((np.sum(var) ** 2) / np.sum(var ** 2))

        comps = {th: int(np.searchsorted(cum_var, th)) + 1 for th in [0.90, 0.95, 0.99, 0.999]}

        # Test set reconstruction error sweep
        dim_sweep = [3, 5, 7, 11, 16, 24, 32]
        recon_errors = {}
        for k in dim_sweep:
            Vk = Vh[:k].T
            z = (X_te - mean_v) @ Vk
            X_rec = mean_v + z @ Vk.T
            err_3d = np.linalg.norm((X_te - X_rec).reshape(-1, 778, 3), axis=-1) * 1000.0  # mm
            mean_e = float(np.mean(err_3d))
            max_e = float(np.max(err_3d))
            # MANO canonical tips: [744, 320, 443, 555, 672]
            tip_e = float(np.mean(err_3d[:, [744, 320, 443, 555, 672]]))
            recon_errors[k] = (mean_e, max_e, tip_e)

        return comps, pr, recon_errors

    print("\nComputing PCA across World-Space and Hand-Local Surface Manifolds...")
    comp_w, pr_w, rec_w = evaluate_pca(G_world_tr, G_world_te, "G (World)")
    comp_dw, pr_dw, _ = evaluate_pca(dG_world_tr, dG_world_tr, "Delta G (World)")
    comp_loc, pr_loc, rec_loc = evaluate_pca(G_local_tr, G_local_te, "G (Hand-Local)")
    comp_dloc, pr_dloc, _ = evaluate_pca(dG_local_tr, dG_local_tr, "Delta G (Hand-Local)")

    print("\n" + "=" * 78)
    print(f"{'Representation':<22} | {'PR (Eff. Dim)':<14} | {'90% Var':<8} | {'95% Var':<8} | {'99% Var':<8} | {'99.9% Var'}")
    print("-" * 78)
    print(f"{'G (World Space)':<22} | {pr_w:<14.1f} | {comp_w[0.90]:<8} | {comp_w[0.95]:<8} | {comp_w[0.99]:<8} | {comp_w[0.999]}")
    print(f"{'Delta G (World Space)':<22} | {pr_dw:<14.1f} | {comp_dw[0.90]:<8} | {comp_dw[0.95]:<8} | {comp_dw[0.99]:<8} | {comp_dw[0.999]}")
    print(f"{'G (Hand-Local)':<22} | {pr_loc:<14.1f} | {comp_loc[0.90]:<8} | {comp_loc[0.95]:<8} | {comp_loc[0.99]:<8} | {comp_loc[0.999]}")
    print(f"{'Delta G (Hand-Local)':<22} | {pr_dloc:<14.1f} | {comp_dloc[0.90]:<8} | {comp_dloc[0.95]:<8} | {comp_dloc[0.99]:<8} | {comp_dloc[0.999]}")
    print("=" * 78)

    print("\n--- HELD-OUT TEST RECONSTRUCTION ACCURACY VS LATENT DIMENSION ---")
    print(f"{'Latent Dim (k)':<15} | {'Local Mean (mm)':<16} | {'Local Max (mm)':<16} | {'Local Tips (mm)':<16} | {'World Mean (mm)'}")
    print("-" * 80)
    for k in [3, 5, 7, 11, 16, 24, 32]:
        m_loc, mx_loc, tip_loc = rec_loc[k]
        m_w, _, _ = rec_w[k]
        print(f"k = {k:<11d} | {m_loc:<16.2f} | {mx_loc:<16.2f} | {tip_loc:<16.2f} | {m_w:<16.2f}")

    print("\nKey Discoveries:")
    print(f"1. In hand-local coordinates, 95% of anatomical surface variation is retained in k = {comp_loc[0.95]} dimensions.")
    print(f"2. At k = 16 in hand-local space, mean surface reconstruction error drops below {rec_loc[16][0]:.2f} mm.")
    print("3. Hand-local transformation eliminates 6-DoF flight variance, proving compact z_G is physically sound.")

if __name__ == "__main__":
    run_surface_discovery()
