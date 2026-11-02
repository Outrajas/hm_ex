"""Comprehensive Visual, Geometric, and Gradient Representation Validation.

Tests segment-stratified interaction frames with:
1. Accurate 3D pinhole projection in millimeters.
2. Grayscale Sobel magnitude with fixed contrast scaling.
3. Continuous gradient vector field (direction + magnitude).
4. True multi-channel Di Zenzo structure tensor.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

ROOT = Path(".")
CAPTURE_ID = "m--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands"
CAPTURE_DIR = ROOT / "datasets" / "reinterhand" / "captures" / CAPTURE_ID
SUBSET_DIR = CAPTURE_DIR / "derived" / "mugsy_subset"
MANIFEST_PATH = SUBSET_DIR / "manifest.csv"
CAM_PARAMS_PATH = CAPTURE_DIR / "Mugsy_cameras" / "cam_params.json"
OUT_DIR = ROOT / "experiments" / "subset_validation"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_dizenzo(img_bgr: np.ndarray):
    """Computes true Di Zenzo structure tensor eigenvalues and orientation."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    # Per-channel derivatives
    rx = cv2.Sobel(img_rgb[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    ry = cv2.Sobel(img_rgb[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(img_rgb[:, :, 1], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_rgb[:, :, 1], cv2.CV_32F, 0, 1, ksize=3)
    bx = cv2.Sobel(img_rgb[:, :, 2], cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(img_rgb[:, :, 2], cv2.CV_32F, 0, 1, ksize=3)

    # Structure tensor components
    gxx = rx**2 + gx**2 + bx**2
    gyy = ry**2 + gy**2 + by**2
    gxy = rx * ry + gx * gy + bx * by

    # Primary eigenvalue and edge magnitude
    diff = gxx - gyy
    disc = np.sqrt(np.maximum(diff**2 + 4.0 * (gxy**2), 0.0))
    lambda_max = 0.5 * ((gxx + gyy) + disc)
    dizenzo_mag = np.sqrt(np.maximum(lambda_max, 0.0))

    return dizenzo_mag

def compute_gradient_vector_field(gx: np.ndarray, gy: np.ndarray, mag: np.ndarray):
    """Encodes gradient orientation and magnitude into an intuitive HSV representation."""
    angles = np.arctan2(gy, gx)  # [-pi, pi]
    hue = ((angles + np.pi) / (2.0 * np.pi) * 180.0).astype(np.uint8)
    
    # Value mapped by fixed global scale (not min-max)
    norm_val = np.clip((mag / 200.0) ** 0.8 * 255.0, 0, 255).astype(np.uint8)
    sat = np.full_like(hue, 220)

    hsv = np.stack([hue, sat, norm_val], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def project_points(pts_3d: np.ndarray, cam: dict) -> np.ndarray:
    """Projects 3D world points (in mm) into 2D camera coordinates."""
    R = np.array(cam["R"], dtype=np.float64)
    t = np.array(cam["t"], dtype=np.float64).reshape(3, 1)
    
    # Explicit unit audit: enforce millimeters
    if np.linalg.norm(t) < 20.0:  # t in meters
        t = t * 1000.0

    f = np.array(cam["focal"], dtype=np.float64)
    c = np.array(cam["princpt"], dtype=np.float64)

    # pts_3d in mm
    pts_cam = (R @ pts_3d.T) + t
    x = pts_cam[0, :] / pts_cam[2, :]
    y = pts_cam[1, :] / pts_cam[2, :]

    u = f[0] * x + c[0]
    v = f[1] * y + c[1]
    return np.stack([u, v], axis=-1)

def run_validation():
    print("=" * 75)
    print(" RE:INTERHAND REVISED GRADIENT & PROJECTION VALIDATION")
    print("=" * 75)

    with open(CAM_PARAMS_PATH, "r", encoding="utf-8") as f:
        cams = json.load(f)
    cam = cams["400485"]

    print("Camera 400485 Extrinsics & Intrinsics:")
    print(f"  Translation Vector t (norm): {np.linalg.norm(cam['t']):.2f}")
    print(f"  Focal Length (fx, fy)     : {cam['focal']}")
    print(f"  Principal Point (cx, cy)  : {cam['princpt']}")

    # 1. Segment-Stratified Sampling across the 56 interactions
    segment_map = defaultdict(list)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            segment_map[r["segment"]].append(r)

    print(f"Total Unique Segments in Manifest: {len(segment_map)}")
    selected_segments = list(segment_map.keys())[:5]

    for idx, seg in enumerate(selected_segments):
        s = segment_map[seg][len(segment_map[seg]) // 2]  # Pick middle frame of segment
        fid = s["frame_id"]
        img_path = CAPTURE_DIR / s["image_path"]
        kp_r_path = CAPTURE_DIR / s["orig_kp_right"]
        kp_l_path = CAPTURE_DIR / s["orig_kp_left"]

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # 1. Compute Representations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(gx**2 + gy**2)

        # True Di Zenzo
        dizenzo_mag = compute_dizenzo(img)

        # Gradient Vector Field (HSV)
        gvf_bgr = compute_gradient_vector_field(gx, gy, sobel_mag)

        # Fixed Contrast Normalization (Human Visualization)
        sobel_vis = np.clip((sobel_mag / 250.0) ** 0.8 * 255.0, 0, 255).astype(np.uint8)
        sobel_vis_bgr = cv2.cvtColor(sobel_vis, cv2.COLOR_GRAY2BGR)

        dizenzo_vis = np.clip((dizenzo_mag / 350.0) ** 0.8 * 255.0, 0, 255).astype(np.uint8)
        dizenzo_vis_bgr = cv2.cvtColor(dizenzo_vis, cv2.COLOR_GRAY2BGR)

        # 2. Skeleton Projection
        vis_img = img.copy()

        # Right Hand -> Green (0, 255, 0)
        if kp_r_path.exists():
            with open(kp_r_path, "r", encoding="utf-8") as f:
                kps_r = np.array(json.load(f), dtype=np.float64)
            uvs_r = project_points(kps_r, cam)
            for u, v in uvs_r:
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(vis_img, (int(u), int(v)), 6, (0, 255, 0), -1)

        # Left Hand -> Blue (255, 0, 0) in BGR
        if kp_l_path.exists():
            with open(kp_l_path, "r", encoding="utf-8") as f:
                kps_l = np.array(json.load(f), dtype=np.float64)
            uvs_l = project_points(kps_l, cam)
            for u, v in uvs_l:
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(vis_img, (int(u), int(v)), 6, (255, 0, 0), -1)

        # Annotate labels onto subpanels
        cv2.putText(vis_img, f"Frame {fid}: {seg} (R:Grn, L:Blu)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(sobel_vis_bgr, "Grayscale Sobel Mag (Fixed Gamma Scale)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(gvf_bgr, "Sobel Vector Field (Hue: Angle, Val: Mag)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(dizenzo_vis_bgr, "Multi-Spectral Di Zenzo Tensor Mag", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # Build 2x2 Diagnostic Canvas
        top_row = np.hstack([vis_img, sobel_vis_bgr])
        bot_row = np.hstack([gvf_bgr, dizenzo_vis_bgr])
        canvas = np.vstack([top_row, bot_row])

        canvas_preview = cv2.resize(canvas, (1920, 1440))
        out_file = OUT_DIR / f"diagnostic_{idx+1}_{seg[:25]}_f{fid}.png"
        cv2.imwrite(str(out_file), canvas_preview)
        print(f"Generated: {out_file.name}")

    print("=" * 75)
    print(f"Diagnostic images written to: {OUT_DIR.resolve()}")
    print("=" * 75)

if __name__ == "__main__":
    run_validation()
