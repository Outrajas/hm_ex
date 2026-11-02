#!/usr/bin/env python
"""Local visual smoke test for one saved Phase 7 milestone; no detector used."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HandState = importlib.import_module("00_spec.state_definition").HandState
CameraIntrinsics = importlib.import_module("00_spec.coordinate_system").CameraIntrinsics
TemporalStateCorrector = importlib.import_module("04_network.architecture").TemporalStateCorrector
HandBiometrics = importlib.import_module("02_state.hand_model").HandBiometrics
forward_kinematics = importlib.import_module("02_state.forward_kinematics").forward_kinematics
RenderedPriorInput = importlib.import_module("08_runtime.render_input").RenderedPriorInput
PersistentHandTracker = importlib.import_module("08_runtime.tracker").PersistentHandTracker
PARENT_INDICES = importlib.import_module("00_spec.hand_topology").PARENT_INDICES


def draw_projection(frame: np.ndarray, joints_m: np.ndarray, intrinsics: CameraIntrinsics) -> None:
    points = np.asarray(joints_m, np.float32) * 1000.0
    z = points[:, 2]
    valid = z > 1.0
    uv = np.zeros((len(points), 2), np.float32)
    uv[valid, 0] = intrinsics.fx * points[valid, 0] / z[valid] + intrinsics.cx
    uv[valid, 1] = intrinsics.fy * points[valid, 1] / z[valid] + intrinsics.cy
    pix = np.rint(uv).astype(np.int32)
    for child, parent in PARENT_INDICES.items():
        if parent >= 0 and valid[parent] and valid[child]:
            cv2.line(frame, tuple(pix[parent]), tuple(pix[child]), (40, 220, 255), 2, cv2.LINE_AA)
    for i in np.flatnonzero(valid):
        cv2.circle(frame, tuple(pix[i]), 3, (40, 80, 255), -1, cv2.LINE_AA)


def draw_3d_panel(joints_m: np.ndarray, authority: float, uncertainty: np.ndarray) -> np.ndarray:
    panel = np.zeros((480, 480, 3), np.uint8)
    panel[:] = (25, 27, 32)
    pts = np.asarray(joints_m, np.float32) * 1000.0
    root = pts[0].copy()
    rel = pts - root
    # Fixed oblique projection of camera-space XYZ for a readable 3D skeleton view.
    x = rel[:, 0] - 0.42 * rel[:, 2]
    y = -rel[:, 1] - 0.25 * rel[:, 2]
    scale = 1.35
    pix = np.column_stack((240 + x * scale, 250 + y * scale)).astype(np.int32)
    for child, parent in PARENT_INDICES.items():
        if parent >= 0:
            cv2.line(panel, tuple(pix[parent]), tuple(pix[child]), (40, 210, 150), 3, cv2.LINE_AA)
    for p in pix:
        cv2.circle(panel, tuple(p), 5, (80, 180, 255), -1, cv2.LINE_AA)
    cv2.putText(panel, "3D hand state (oblique projection)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, f"visual authority: {authority:.2f}", (12, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, f"mean state uncertainty: {float(np.mean(uncertainty)):.4f}", (12, 474), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="A saved milestone_*.pt or best.pt from Google Drive")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--focal-scale", type=float, default=0.90, help="Approximate fx,fy as this fraction of image width")
    parser.add_argument("--initial-depth", type=float, default=0.52, help="Initial MANO root depth in meters")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    model = TemporalStateCorrector(hidden_dim=int(config.get("hidden_dim", 256))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    biometrics = HandBiometrics(**checkpoint["biometrics"])
    initial = HandState.neutral(z_depth=float(args.initial_depth))
    initial.beta = np.asarray(checkpoint.get("mean_shape_beta_train", np.zeros(10)), np.float64).copy()

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("Webcam opened but no frame was received")
    h, w = frame.shape[:2]
    f = float(args.focal_scale) * w
    intrinsics = CameraIntrinsics(f, f, (w - 1) / 2, (h - 1) / 2, w, h)
    renderer = RenderedPriorInput(
        intrinsics, biometrics=biometrics, mano_model_path=ROOT / "models" / "MANO_RIGHT.pkl",
        output_size=int(config.get("image_size", 128)), crop_margin=1.6,
    )
    tracker = PersistentHandTracker(model, initial)
    print(f"Device={device}; webcam={w}x{h}; estimated focal={f:.1f}px; press q to quit, r to reset, [ / ] to adjust depth")
    print("Smoke test only: no detector or depth sensor is used; camera intrinsics and initial depth are approximate.")
    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            now = time.perf_counter()
            dt = float(np.clip(now - previous_time, 1 / 240, 0.1))
            previous_time = now
            try:
                rendered_input, _ = renderer(frame, tracker.state)
                result = tracker.step(rendered_input, dt_seconds=dt, measurement_valid=True)
                joints_m, _ = forward_kinematics(result.state, biometrics)
            except ValueError as exc:
                cv2.putText(frame, f"Adjust depth: {exc}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                joints_m = None
                result = None
            view = frame.copy()
            if joints_m is not None and result is not None:
                draw_projection(view, joints_m, intrinsics)
                cv2.putText(view, f"authority={result.authority:.2f}  depth={result.state.T[2]:.2f}m", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                panel = draw_3d_panel(joints_m, result.authority, result.uncertainty)
            else:
                panel = np.zeros((480, 480, 3), np.uint8)
            display = np.hstack((cv2.resize(view, (640, 480)), panel))
            cv2.imshow("Phase 7 webcam smoke test | image + simulated 3D skeleton", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                reset = HandState.neutral(z_depth=float(args.initial_depth))
                reset.beta = initial.beta.copy()
                tracker.reset(reset)
            if key in (ord("["), ord("-")):
                tracker.state.T[2] += 0.02
            if key in (ord("]"), ord("="), ord("+")):
                tracker.state.T[2] = max(0.15, tracker.state.T[2] - 0.02)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
