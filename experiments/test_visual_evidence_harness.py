"""Stage 1 & 2: Visual Evidence Harness & Non-Neural Observability Sanity Checks."""

import sys
from pathlib import Path
import time
import importlib
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.import_module("00_spec.state_definition")
_coord = importlib.import_module("00_spec.coordinate_system")
_fk = importlib.import_module("02_state.forward_kinematics")
_mano = importlib.import_module("02_state.mano_surface")
_const = importlib.import_module("02_state.constraints")

HandState = _spec.HandState
forward_kinematics = _fk.forward_kinematics
ManoSurface = _mano.ManoSurface
project_constraints = _const.project_constraints
quaternion_to_rotation_matrix = _coord.quaternion_to_rotation_matrix
euler_to_quaternion = _coord.euler_to_quaternion
quaternion_multiply = _coord.quaternion_multiply
quaternion_normalize = _coord.quaternion_normalize

OUT_DIR = ROOT / "experiments" / "evidence_inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 160
FOCAL_PX = 320.0
LIGHT_DIR = np.array([0.35, -0.65, -0.67], dtype=np.float64)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)

def render_hand_patch(verts: np.ndarray, faces: np.ndarray, R_cam: np.ndarray, T_cam: np.ndarray) -> np.ndarray:
    img = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

    pts_cam = (verts - T_cam) @ R_cam.T
    Z = pts_cam[:, 2]
    safe_z = np.maximum(Z, 0.05)
    u = (FOCAL_PX * (pts_cam[:, 0] / safe_z)) + (PATCH_SIZE * 0.5)
    v = (FOCAL_PX * (pts_cam[:, 1] / safe_z)) + (PATCH_SIZE * 0.5)
    pts_2d = np.column_stack([u, v]).astype(np.int32)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fnorm = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(fnorm, axis=1, keepdims=True)
    fnorm /= np.maximum(norm_len, 1e-9)

    cam_fnorm = fnorm @ R_cam.T
    front = cam_fnorm[:, 2] < -0.02
    front_indices = np.where(front)[0]

    mean_z = (v0[front_indices, 2] + v1[front_indices, 2] + v2[front_indices, 2]) / 3.0
    order = front_indices[np.argsort(mean_z)[::-1]]

    for fi in order:
        tri = faces[fi]
        diff = float(np.clip(-np.dot(fnorm[fi], LIGHT_DIR), 0.0, 1.0))
        intensity = int(np.clip(45 + 205 * diff, 0, 255))
        cv2.fillConvexPoly(img, pts_2d[tri], intensity)

    return img

def compute_sobel_edge(img: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return np.clip(mag, 0, 255).astype(np.uint8)

def apply_controlled_perturbation(
    state_gt: HandState,
    delta_trans_mm: float,
    delta_rot_deg: float,
    delta_flex_rad: float,
    rng: np.random.RandomState
) -> HandState:
    s_pert = state_gt.copy()

    if delta_trans_mm > 0:
        dir_t = rng.randn(3)
        dir_t /= max(np.linalg.norm(dir_t), 1e-6)
        s_pert.T += dir_t * (delta_trans_mm / 1000.0)

    if delta_rot_deg > 0:
        axis_r = rng.randn(3)
        axis_r /= max(np.linalg.norm(axis_r), 1e-6)
        angle_r = np.radians(delta_rot_deg)
        q_err = np.array([np.cos(angle_r * 0.5), *(axis_r * np.sin(angle_r * 0.5))], dtype=np.float64)
        s_pert.R = quaternion_normalize(quaternion_multiply(q_err, s_pert.R))

    if delta_flex_rad > 0:
        err_q = rng.uniform(-delta_flex_rad, delta_flex_rad, size=21)
        s_pert.q += err_q

    return project_constraints(s_pert)

def run_evidence_harness():
    print("=================================================================")
    print(" STAGE 1 & 2: VISUAL EVIDENCE HARNESS & OBSERVABILITY SANITY CHECKS")
    print("=================================================================")

    mano = ManoSurface()
    if not mano.is_available:
        print("[ERROR] MANO asset not found in models/.")
        return

    rng = np.random.RandomState(42)

    state_gt = HandState.neutral(z_depth=0.52)
    state_gt.q[0], state_gt.q[1], state_gt.q[3], state_gt.q[4] = 0.20, 0.25, 0.40, 0.60
    for idx in [5, 9, 13, 17]:
        state_gt.q[idx+1] = 0.70
        state_gt.q[idx+2] = 0.85
        state_gt.q[idx+3] = 0.50
    state_gt = project_constraints(state_gt)

    j_pos_gt, _ = forward_kinematics(state_gt)
    verts_gt, faces, _ = mano.forward(state_gt, j_pos_gt)

    # Center camera on the anatomical centroid of the hand mesh
    hand_center = np.mean(verts_gt, axis=0)
    R_cam = np.eye(3, dtype=np.float64)
    T_cam = np.array([hand_center[0], hand_center[1], 0.0], dtype=np.float64)

    t0_render = time.perf_counter()
    img_gt = render_hand_patch(verts_gt, faces, R_cam, T_cam)
    edge_gt = compute_sobel_edge(img_gt)
    render_lat_ms = (time.perf_counter() - t0_render) * 1000.0

    print(f"Base Render Latency (Render + Sobel): {render_lat_ms:.2f} ms/frame ({1000.0/render_lat_ms:.0f} FPS)")
    print(f"Hand Centroid: ({hand_center[0]:.3f}, {hand_center[1]:.3f}, {hand_center[2]:.3f}) m")
    print(f"Patch Resolution: {PATCH_SIZE}x{PATCH_SIZE} px (Focal: {FOCAL_PX} px)")

    levels = [
        {"name": "Level 1 (Sub-pixel/Micro)", "dt_mm": 2.0,  "dr_deg": 2.0,  "dq_rad": 0.05},
        {"name": "Level 2 (Nominal)",          "dt_mm": 5.0,  "dr_deg": 5.0,  "dq_rad": 0.10},
        {"name": "Level 3 (Moderate)",         "dt_mm": 10.0, "dr_deg": 10.0, "dq_rad": 0.20},
        {"name": "Level 4 (High Dynamic)",     "dt_mm": 20.0, "dr_deg": 20.0, "dq_rad": 0.40},
        {"name": "Level 5 (Severe Perturb)",   "dt_mm": 40.0, "dr_deg": 35.0, "dq_rad": 0.80},
    ]

    print("\n" + "=" * 88)
    print(f"{'Perturbation Level':<25} | {'Epsilon (t, R, q)':<20} | {'Gray MAE':<10} | {'Edge Residual':<14} | {'Monotonic?':<10}")
    print("-" * 88)

    prev_gray_mae = 0.0
    prev_edge_energy = 0.0
    all_monotonic = True

    for i, lvl in enumerate(levels):
        s_pert = apply_controlled_perturbation(state_gt, lvl["dt_mm"], lvl["dr_deg"], lvl["dq_rad"], rng)
        j_pos_pert, _ = forward_kinematics(s_pert)
        verts_pert, _, _ = mano.forward(s_pert, j_pos_pert)

        img_pert = render_hand_patch(verts_pert, faces, R_cam, T_cam)
        edge_pert = compute_sobel_edge(img_pert)

        res_gray = cv2.absdiff(img_gt, img_pert)
        res_edge = cv2.absdiff(edge_gt, edge_pert)

        gray_mae = float(np.mean(res_gray))
        edge_energy = float(np.mean(res_edge))

        monotonic = (gray_mae >= prev_gray_mae) and (edge_energy >= prev_edge_energy)
        if not monotonic and i > 0:
            all_monotonic = False

        prev_gray_mae = gray_mae
        prev_edge_energy = edge_energy

        eps_str = f"{lvl['dt_mm']:0.0f}mm, {lvl['dr_deg']:0.0f}deg, {lvl['dq_rad']:0.2f}rad"
        mono_str = "YES (Valid)" if (monotonic or i == 0) else "WARN"
        print(f"{lvl['name']:<25} | {eps_str:<20} | {gray_mae:<10.2f} | {edge_energy:<14.2f} | {mono_str:<10}")

        heat_gray = cv2.applyColorMap(res_gray, cv2.COLORMAP_JET)
        heat_edge = cv2.applyColorMap(res_edge, cv2.COLORMAP_JET)

        col_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2BGR)
        col_pert = cv2.cvtColor(img_pert, cv2.COLOR_GRAY2BGR)
        col_edge_gt = cv2.cvtColor(edge_gt, cv2.COLOR_GRAY2BGR)
        col_edge_pert = cv2.cvtColor(edge_pert, cv2.COLOR_GRAY2BGR)

        top_row = np.hstack([col_gt, col_pert, heat_gray])
        bot_row = np.hstack([col_edge_gt, col_edge_pert, heat_edge])
        plate = np.vstack([top_row, bot_row])

        cv2.putText(plate, f"{lvl['name']}: dt={lvl['dt_mm']}mm, dr={lvl['dr_deg']}deg, dq={lvl['dq_rad']}rad",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(plate, f"Gray MAE: {gray_mae:.2f} | Edge Res Energy: {edge_energy:.2f}",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

        out_img_path = OUT_DIR / f"evidence_level_{i+1}_{int(lvl['dt_mm'])}mm.png"
        cv2.imwrite(str(out_img_path), plate)

    print("=" * 88)

    print("\n--- STAGE 2 SANITY CHECK: SPATIAL LOCALIZATION TEST ---")
    s_wrist = apply_controlled_perturbation(state_gt, 15.0, 0.0, 0.0, rng)
    s_index = state_gt.copy()
    s_index.q[6] += 0.45  # Index MCP flexion
    s_index = project_constraints(s_index)

    img_wrist = render_hand_patch(mano.forward(s_wrist, forward_kinematics(s_wrist)[0])[0], faces, R_cam, T_cam)
    img_index = render_hand_patch(mano.forward(s_index, forward_kinematics(s_index)[0])[0], faces, R_cam, T_cam)

    res_wrist = cv2.absdiff(img_gt, img_wrist)
    res_index = cv2.absdiff(img_gt, img_index)

    # In camera frame (+Y is downwards, distal fingers point down past center v > 75)
    finger_mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool)
    finger_mask[int(PATCH_SIZE * 0.48):, :] = True

    index_finger_ratio = float(np.sum(res_index[finger_mask]) / max(np.sum(res_index), 1e-6))
    wrist_finger_ratio = float(np.sum(res_wrist[finger_mask]) / max(np.sum(res_wrist), 1e-6))

    print(f"Isolated Index Flexion: {index_finger_ratio*100:.1f}% of residual energy concentrated on distal fingers.")
    print(f"Isolated Wrist Translation: {wrist_finger_ratio*100:.1f}% distributed across palm and boundary.")
    print(f"Observability Verification: {'PASS (Strong Localized Signal)' if (all_monotonic and index_finger_ratio > 0.70) else 'FAIL'}")
    print(f"Visual inspection plates exported to: {OUT_DIR.relative_to(ROOT)}/")

if __name__ == "__main__":
    run_evidence_harness()
