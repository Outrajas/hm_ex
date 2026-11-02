"""Mines and visualizes top failure cases across 8 distinct failure modes (Phase K)."""

import sys
from pathlib import Path
import importlib
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.import_module("00_spec.state_definition")
_ds = importlib.import_module("06_training.dataset")
_mano = importlib.import_module("02_state.mano_surface")
_rel = importlib.import_module("03_observation.relationship")

build_split_dataset = _ds.build_split_dataset
ManoSurface = _mano.ManoSurface
compute_skeleton_geometry = _rel.compute_skeleton_geometry
compute_glove_geometry = _rel.compute_glove_geometry
evaluate_consistency = _rel.evaluate_consistency

OUT_DIR = ROOT / "experiments" / "failure_cases"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_failure_mining():
    print("=================================================================")
    print(" PHASE K: AUTOMATED FAILURE CASE MINING & SNAPSHOT EXTRACTION")
    print("=================================================================")

    mano = ManoSurface()
    splits = build_split_dataset(n_trajectories=600, frames_per_traj=50, seed=101)
    test_data = splits["test"]
    N_test = test_data.X_pred.shape[0]

    k_errors = np.mean(np.linalg.norm(test_data.K_pred - test_data.K_gt, axis=-1), axis=1) * 1000.0
    g_errors = np.mean(np.linalg.norm(test_data.G_pred - test_data.G_gt, axis=-1), axis=1) * 1000.0
    tip_errors = np.mean(np.linalg.norm(test_data.K_pred[:, [4, 8, 12, 16, 20]] - test_data.K_gt[:, [4, 8, 12, 16, 20]], axis=-1), axis=1) * 1000.0
    velocities = np.linalg.norm(test_data.X_pred[:, 28:31], axis=-1)

    # Categories to mine
    cat_indices = {
        "largest_skeleton_error": int(np.argmax(k_errors)),
        "largest_glove_error": int(np.argmax(g_errors)),
        "largest_fingertip_error": int(np.argmax(tip_errors)),
        "fastest_motion": int(np.argmax(velocities)),
        "extreme_curl": int(np.argmax(np.sum(test_data.X_gt[:, 7:28], axis=1))),
        "abrupt_deviation": int(np.argmax(np.linalg.norm(test_data.delta_X[:, 7:28], axis=1)))
    }

    # Evaluate consistency on the mined candidates
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    print(f"Mining and saving failure visualizations to {OUT_DIR.name}/...")

    for cat_name, idx in cat_indices.items():
        canvas[:] = (20, 20, 26)
        cv2.putText(canvas, f"FAILURE MODE: {cat_name.upper()}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
        cv2.putText(canvas, f"Index: {idx} | Traj ID: {test_data.trajectory_ids[idx]}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
        cv2.putText(canvas, f"Delta K: {k_errors[idx]:.2f} mm | Delta G: {g_errors[idx]:.2f} mm | Tips: {tip_errors[idx]:.2f} mm", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 255, 100), 1)

        # 2D projection center
        u_pts = (test_data.K_gt[idx, :, 0] * 1200.0 + 320.0).astype(int)
        v_pts = (test_data.K_gt[idx, :, 1] * 1200.0 + 260.0).astype(int)
        for i in range(21):
            cv2.circle(canvas, (u_pts[i], v_pts[i]), 4, (0, 255, 120), -1)

        u_pr = (test_data.K_pred[idx, :, 0] * 1200.0 + 320.0).astype(int)
        v_pr = (test_data.K_pred[idx, :, 1] * 1200.0 + 260.0).astype(int)
        for i in range(21):
            cv2.circle(canvas, (u_pr[i], v_pr[i]), 4, (0, 100, 255), -1)
            cv2.line(canvas, (u_pts[i], v_pts[i]), (u_pr[i], v_pr[i]), (0, 160, 255), 2)

        out_img = OUT_DIR / f"{cat_name}.png"
        cv2.imwrite(str(out_img), canvas)
        print(f"  -> Saved {out_img.name} (Delta_K: {k_errors[idx]:.2f} mm, Tips: {tip_errors[idx]:.2f} mm)")

    print("[Phase K] Failure case extraction complete.")

if __name__ == "__main__":
    run_failure_mining()
