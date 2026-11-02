#!/usr/bin/env python
"""Interactive webcam smoke test for a Phase 7 temporal checkpoint.

This is a manual-bootstrap test, not an automatic hand detector. Before starting
the recurrent model, align the simulated open hand to the visible palm using the
keyboard controls. The model then receives the same five-channel RGB + rendered
MANO + projected-skeleton input used during training.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
import time
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

state_def = importlib.import_module("00_spec.state_definition")
coord = importlib.import_module("00_spec.coordinate_system")
topology = importlib.import_module("00_spec.hand_topology")
hand_model = importlib.import_module("02_state.hand_model")
fk_api = importlib.import_module("02_state.forward_kinematics")
network = importlib.import_module("04_network.architecture")
runtime = importlib.import_module("08_runtime.tracker")
render_api = importlib.import_module("08_runtime.render_input")

HandState = state_def.HandState
HandBiometrics = hand_model.HandBiometrics
CameraIntrinsics = coord.CameraIntrinsics
TemporalStateCorrector = network.TemporalStateCorrector
PersistentHandTracker = runtime.PersistentHandTracker
RenderedPriorInput = render_api.RenderedPriorInput
forward_kinematics = fk_api.forward_kinematics
BONE_PAIRS = [(p, c) for c, p in topology.PARENT_INDICES.items() if p >= 0]

FINGER_COLORS = [
    (0, 165, 255), (0, 255, 0), (255, 255, 0),
    (255, 140, 0), (255, 0, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "phase7" / "webcam_epoch_020.pt"),
    )
    parser.add_argument("--mano", default=str(ROOT / "models" / "MANO_RIGHT.pkl"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=60.0,
                        help="Approximate webcam horizontal field of view in degrees.")
    parser.add_argument("--focal-px", type=float, default=0.0,
                        help="Optional calibrated focal length in pixels; overrides --hfov.")
    parser.add_argument("--depth", type=float, default=0.48,
                        help="Initial wrist depth in meters; adjust with +/- in the preview.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return device


def load_tracker(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Phase 7 checkpoint not found: {checkpoint_path}\n"
            "Download webcam_epoch_020.pt from the Drive phase7_capture_holdout_cached folder."
        )

    # This is the user's own training checkpoint and contains config/biometrics
    # metadata in addition to tensors, so load the complete trusted payload.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model", "config", "biometrics"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint is not a Phase 7 training checkpoint; missing {sorted(missing)}")

    config = payload["config"]
    if isinstance(config, argparse.Namespace):
        config = vars(config)
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config is not a dictionary.")
    hidden_dim = int(config.get("hidden_dim", 256))
    image_size = int(config.get("image_size", 128))
    input_contract = payload.get("input_contract", "")
    if input_contract and "5 channels" not in input_contract:
        raise ValueError(f"Unexpected checkpoint input contract: {input_contract}")

    model = TemporalStateCorrector(hidden_dim=hidden_dim)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()

    valid_bio_fields = {item.name for item in fields(HandBiometrics)}
    bio_values = {
        key: float(value)
        for key, value in dict(payload["biometrics"]).items()
        if key in valid_bio_fields
    }
    biometrics = HandBiometrics(**bio_values)
    return payload, model, biometrics, image_size


def make_seed(depth: float) -> tuple[HandState, tuple[float, float, float]]:
    state = HandState.neutral(z_depth=depth)
    # Open hand, palm facing the camera, fingers pointing toward the top of
    # the image for the common webcam coordinate convention.
    yaw = math.pi
    state.R = coord.euler_to_quaternion(0.0, 0.0, yaw)
    state.T[:] = [0.0, 0.06, depth]
    return state, (0.0, 0.0, yaw)


def project_joints(joints_m: np.ndarray, intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    z = joints_m[:, 2]
    valid = z > 0.01
    safe_z = np.where(valid, z, 1.0)
    uv = np.column_stack((
        intr.fx * joints_m[:, 0] / safe_z + intr.cx,
        intr.fy * joints_m[:, 1] / safe_z + intr.cy,
    ))
    return uv, valid


def draw_camera_overlay(frame: np.ndarray, state: HandState, biometrics: HandBiometrics,
                        intr: CameraIntrinsics, tracking: bool, authority: float,
                        fps: float) -> np.ndarray:
    out = frame.copy()
    joints_m, _ = forward_kinematics(state, biometrics)
    uv, valid = project_joints(joints_m, intr)
    pts = np.rint(uv).astype(np.int32)
    for bone_i, (parent, child) in enumerate(BONE_PAIRS):
        if valid[parent] and valid[child]:
            cv2.line(out, tuple(pts[parent]), tuple(pts[child]),
                     FINGER_COLORS[min(bone_i // 4, 4)], 2, cv2.LINE_AA)
    for joint_i, point in enumerate(pts):
        if valid[joint_i]:
            cv2.circle(out, tuple(point), 4 if joint_i in (0, 4, 8, 12, 16, 20) else 3,
                       (255, 255, 255), -1, cv2.LINE_AA)

    mode = "TRACKING" if tracking else "ALIGN PRIOR, THEN SPACE"
    cv2.putText(out, f"Phase 7 webcam | {mode} | authority={authority:.2f} | {fps:.1f} FPS",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 255, 80), 2, cv2.LINE_AA)
    cv2.putText(out, "A/D x  W/S y  +/- depth  J/L yaw  I/K pitch  U/O roll  R reset  Q quit",
                (12, out.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_3d_skeleton(state: HandState, biometrics: HandBiometrics,
                     panel_size: tuple[int, int]) -> np.ndarray:
    width, height = panel_size
    panel = np.full((height, width, 3), (27, 29, 36), dtype=np.uint8)
    cv2.putText(panel, "SIMULATED 3D HAND (camera/world axes)", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 230, 240), 1, cv2.LINE_AA)
    joints, _ = forward_kinematics(state, biometrics)
    centered = joints - joints[0:1]

    yaw, pitch = 0.55, -0.22
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rot = np.asarray([
        [cy, sy * sp, sy * cp],
        [0.0, cp, -sp],
        [-sy, cy * sp, cy * cp],
    ], dtype=np.float64)
    view = centered @ rot.T
    span = max(float(np.ptp(view[:, 0])), float(np.ptp(view[:, 1])), 0.10)
    scale = min((width - 60) / span, (height - 72) / span)
    origin = np.asarray([width * 0.5, height * 0.57])
    px = np.rint(np.column_stack((
        origin[0] + view[:, 0] * scale,
        origin[1] - view[:, 1] * scale,
    ))).astype(np.int32)
    for bone_i, (parent, child) in enumerate(BONE_PAIRS):
        cv2.line(panel, tuple(px[parent]), tuple(px[child]),
                 FINGER_COLORS[min(bone_i // 4, 4)], 4, cv2.LINE_AA)
    for joint_i, point in enumerate(px):
        cv2.circle(panel, tuple(point), 5 if joint_i in (0, 4, 8, 12, 16, 20) else 3,
                   (245, 245, 245), -1, cv2.LINE_AA)
    return panel


def adjust_seed(state: HandState, angles: list[float], key: int) -> bool:
    """Apply a small manual bootstrap adjustment; return whether changed."""
    step = 0.01
    changed = True
    if key == ord("a"):
        state.T[0] -= step
    elif key == ord("d"):
        state.T[0] += step
    elif key == ord("w"):
        state.T[1] -= step
    elif key == ord("s"):
        state.T[1] += step
    elif key in (ord("+"), ord("=")):
        state.T[2] = max(0.18, state.T[2] - 0.02)
    elif key in (ord("-"), ord("_")):
        state.T[2] = min(1.20, state.T[2] + 0.02)
    elif key == ord("j"):
        angles[2] -= math.radians(5)
    elif key == ord("l"):
        angles[2] += math.radians(5)
    elif key == ord("i"):
        angles[1] -= math.radians(5)
    elif key == ord("k"):
        angles[1] += math.radians(5)
    elif key == ord("u"):
        angles[0] -= math.radians(5)
    elif key == ord("o"):
        angles[0] += math.radians(5)
    elif key in (ord("0"),):
        state.q[:] = 0.0
        state.q[5], state.q[13], state.q[17] = np.radians([8.0, -6.0, -14.0])
    elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
        finger = int(chr(key)) - 1
        base = [5, 9, 13, 17][finger]
        state.q[base + 1:base + 4] = np.clip(
            state.q[base + 1:base + 4] + math.radians(10), 0.0, math.radians(95)
        )
    elif key == ord("t"):
        state.q[1:5] = np.clip(state.q[1:5] + math.radians(10), 0.0, math.radians(95))
    else:
        changed = False
    if changed and key not in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4"), ord("t")):
        state.R = coord.euler_to_quaternion(*angles)
    return changed


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    mano_path = Path(args.mano).expanduser().resolve()
    if not mano_path.is_file():
        raise FileNotFoundError(f"MANO model file not found: {mano_path}")

    payload, model, biometrics, image_size = load_tracker(checkpoint_path, device)
    state, angles_tuple = make_seed(args.depth)
    angles = list(angles_tuple)
    tracker = PersistentHandTracker(model, state)

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError("Webcam opened but returned no frames.")
    height, width = first.shape[:2]
    focal = float(args.focal_px) if args.focal_px > 0 else 0.5 * width / math.tan(math.radians(args.hfov) * 0.5)
    intr = CameraIntrinsics(focal, focal, width * 0.5, height * 0.5, width, height)
    renderer = RenderedPriorInput(
        intr, biometrics=biometrics, mano_model_path=mano_path,
        output_size=image_size, crop_margin=1.6,
    )

    tracking = False
    authority = 0.0
    last_time = time.perf_counter()
    fps_ema = 0.0
    window_name = "Phase 7 trained model — webcam smoke test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}; input size: {image_size}; webcam: {width}x{height}; focal~{focal:.1f}px")
    print("Align the green skeleton over an open palm facing the camera, fingers up; press SPACE to start.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            now = time.perf_counter()
            dt = float(np.clip(now - last_time, 1.0 / 240.0, 0.25))
            last_time = now
            fps = 1.0 / dt
            fps_ema = fps if fps_ema == 0.0 else 0.90 * fps_ema + 0.10 * fps

            crop, _ = renderer(frame, state)
            if tracking:
                result = tracker.step(crop, dt_seconds=dt, measurement_valid=True)
                state = result.state
                authority = result.authority

            camera_view = draw_camera_overlay(frame, state, biometrics, intr,
                                              tracking, authority, fps_ema)
            panel = draw_3d_skeleton(state, biometrics, (420, height))
            crop_rgb = crop[:3].permute(1, 2, 0).cpu().numpy()
            crop_bgr = cv2.cvtColor(np.clip(crop_rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            skeleton_mask = crop[4].cpu().numpy() > 0.5
            crop_bgr[skeleton_mask] = (0, 255, 0)
            crop_view = cv2.resize(crop_bgr, (180, 180), interpolation=cv2.INTER_NEAREST)
            x0, y0 = 14, max(34, height - 196)
            panel[y0:y0 + 180, x0:x0 + 180] = crop_view
            cv2.putText(panel, "RGB crop + prior skeleton", (x0, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 225, 235), 1, cv2.LINE_AA)
            cv2.putText(panel, f"depth {state.T[2]:.2f} m | learned authority {authority:.2f}",
                        (12, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (220, 225, 235), 1, cv2.LINE_AA)

            canvas = np.zeros((height, width + 420, 3), dtype=np.uint8)
            canvas[:, :width] = camera_view
            canvas[:, width:] = panel
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                tracking = True
                tracker.reset(state)
                print("Temporal correction started; press SPACE again to reset its recurrent memory.")
                continue
            if key == ord("r"):
                state, angles_tuple = make_seed(args.depth)
                angles = list(angles_tuple)
                tracker.reset(state)
                tracking = False
                authority = 0.0
                continue
            if adjust_seed(state, angles, key):
                tracker.reset(state)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
