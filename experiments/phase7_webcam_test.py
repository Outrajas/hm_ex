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


class OrientedRenderedPriorInput(RenderedPriorInput):
    """Use raw-camera intrinsics while displaying a rotated/mirrored frame."""

    def __init__(self, intrinsics, *, raw_width, raw_height, rotation, mirror, **kwargs):
        super().__init__(intrinsics, **kwargs)
        self.raw_width = int(raw_width)
        self.raw_height = int(raw_height)
        self.rotation = int(rotation)
        self.mirror = bool(mirror)
        self.raw_cx = (self.raw_width - 1) * 0.5
        self.raw_cy = (self.raw_height - 1) * 0.5
        self.raw_fx = float(intrinsics.fx)
        self.raw_fy = float(intrinsics.fy)
        self.last_joints = None
        self.last_vertices = None
        self.last_faces = None

        # RenderedPriorInput already skins MANO while making its silhouette.
        # Keep that exact result for the display so the webcam loop does not
        # run MANO a second time per frame.
        original_forward = self.mano.forward

        def cached_forward(*args, **kwargs):
            vertices, faces, joints = original_forward(*args, **kwargs)
            self.last_vertices = vertices
            self.last_faces = faces
            self.last_joints = args[1] if len(args) > 1 and args[1] is not None else joints
            return vertices, faces, joints

        self.mano.forward = cached_forward

    def _project(self, points_m):
        points_mm = np.asarray(points_m, dtype=np.float32) * 1000.0
        z = points_mm[:, 2]
        valid = z > 1.0
        safe_z = np.where(valid, z, 1.0)
        u = self.raw_fx * points_mm[:, 0] / safe_z + self.raw_cx
        v = self.raw_fy * points_mm[:, 1] / safe_z + self.raw_cy

        if self.rotation == 90:
            u, v = self.raw_height - 1.0 - v, u
        elif self.rotation == 180:
            u, v = self.raw_width - 1.0 - u, self.raw_height - 1.0 - v
        elif self.rotation == 270:
            u, v = v, self.raw_width - 1.0 - u
        if self.mirror:
            u = self.intrinsics.width - 1.0 - u
        return np.column_stack((u, v)).astype(np.float32), valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "phase7" / "webcam_epoch_020.pt"),
    )
    parser.add_argument("--mano", default=str(ROOT / "models" / "MANO_RIGHT.pkl"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0,
                        help="Rotate image and MANO projection together if the raw camera feed needs it.")
    parser.add_argument("--mirror", action="store_true",
                        help="Mirror the camera image horizontally (selfie view).")
    parser.add_argument("--correction-strength", type=float, default=1.0,
                        help=("Scale the model's already confidence-gated correction. "
                              "Default 1.0 matches training evaluation; lower only to damp instability."))
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


def draw_camera_overlay(frame: np.ndarray, joints_m: np.ndarray, vertices_m: np.ndarray,
                        faces: np.ndarray, renderer: OrientedRenderedPriorInput,
                        tracking: bool, authority: float, fps: float) -> np.ndarray:
    out = frame.copy()
    uv, valid = renderer._project(joints_m)
    mesh_uv, mesh_valid = renderer._project(vertices_m)
    faces = np.asarray(faces, dtype=np.int32)
    if int(mesh_valid.sum()) >= 3:
        vertices_px = np.rint(mesh_uv).astype(np.int32)
        # Match main.py's MANO renderer: depth-sort all front-facing triangles
        # and shade each face. Blend the complete model over the camera frame
        # so the physical hand remains visible beneath the predicted surface.
        triangles = vertices_m[faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(normal_lengths, 1e-9)
        front = np.flatnonzero((normals[:, 2] < -0.05) & mesh_valid[faces].all(axis=1))
        mean_depth = triangles[:, :, 2].mean(axis=1)
        order = front[np.argsort(mean_depth[front])[::-1]]
        light_dir = np.asarray([0.4, -0.7, -0.6], dtype=np.float32)
        light_dir /= max(float(np.linalg.norm(light_dir)), 1e-9)
        mesh_layer = out.copy()
        for face_i in order:
            tri = faces[face_i]
            polygon = vertices_px[tri]
            # Avoid sending fully offscreen triangles through the rasterizer.
            if (polygon[:, 0].max() < 0 or polygon[:, 1].max() < 0 or
                    polygon[:, 0].min() >= out.shape[1] or
                    polygon[:, 1].min() >= out.shape[0]):
                continue
            diffuse = float(np.clip(-np.dot(normals[face_i], light_dir), 0.0, 1.0))
            shade = 0.35 + 0.65 * diffuse
            color = (int(120 * shade), int(145 * shade), int(175 * shade))
            cv2.fillConvexPoly(mesh_layer, polygon, color, lineType=cv2.LINE_8)
        cv2.addWeighted(mesh_layer, 0.48, out, 0.52, 0.0, out)
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


def draw_3d_mano(state: HandState, joints_m: np.ndarray, vertices_m: np.ndarray,
                 faces: np.ndarray, panel_size: tuple[int, int],
                 input_size: int) -> np.ndarray:
    width, height = panel_size
    panel = np.full((height, width, 3), (27, 29, 36), dtype=np.uint8)
    cv2.putText(panel, "LIVE MANO SURFACE + FK SKELETON", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 230, 240), 1, cv2.LINE_AA)
    cv2.putText(panel, "MANO shape is rendered from the current state; not a measured hand fit",
                (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (180, 190, 205), 1, cv2.LINE_AA)

    yaw, pitch = 0.55, -0.22
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rot = np.asarray([
        [cy, sy * sp, sy * cp],
        [0.0, cp, -sp],
        [-sy, cy * sp, cy * cp],
    ], dtype=np.float64)
    mesh_view = (vertices_m - joints_m[0:1]) @ rot.T
    joint_view = (joints_m - joints_m[0:1]) @ rot.T
    # Fixed metres-to-pixels scale: finger articulation must not zoom the view.
    scale = min((width - 48) / 0.30, (height - 96) / 0.40)
    origin = np.asarray([width * 0.5, height * 0.48])
    mesh_px = np.rint(np.column_stack((
        origin[0] + mesh_view[:, 0] * scale,
        origin[1] + mesh_view[:, 1] * scale,
    ))).astype(np.int32)
    joint_px = np.rint(np.column_stack((
        origin[0] + joint_view[:, 0] * scale,
        origin[1] + joint_view[:, 1] * scale,
    ))).astype(np.int32)

    # Painter's algorithm is enough for this small live preview mesh.
    faces = np.asarray(faces, dtype=np.int32)
    tri_view = mesh_view[faces]
    normals = np.cross(tri_view[:, 1] - tri_view[:, 0], tri_view[:, 2] - tri_view[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    tri_depth = tri_view[:, :, 2].mean(axis=1)
    # As in main.py, omit back-facing surfaces rather than arbitrary triangles.
    # All camera-facing triangles remain, preserving a complete visible skin.
    front = np.flatnonzero(normals[:, 2] < -1e-8)
    order = front[np.argsort(tri_depth[front])[::-1]]
    light = np.abs(normals @ np.asarray([0.25, 0.55, -0.80]))
    shades = 0.45 + 0.55 * light
    for face_i in order:
        shade = float(shades[face_i])
        color = (int(95 * shade), int(135 * shade), int(175 * shade))
        cv2.fillConvexPoly(panel, mesh_px[faces[face_i]], color, lineType=cv2.LINE_8)

    for bone_i, (parent, child) in enumerate(BONE_PAIRS):
        cv2.line(panel, tuple(joint_px[parent]), tuple(joint_px[child]),
                 FINGER_COLORS[min(bone_i // 4, 4)], 4, cv2.LINE_AA)
    for joint_i, point in enumerate(joint_px):
        cv2.circle(panel, tuple(point), 5 if joint_i in (0, 4, 8, 12, 16, 20) else 3,
                   (245, 245, 245), -1, cv2.LINE_AA)
    cv2.putText(panel, f"T=({state.T[0]:+.2f},{state.T[1]:+.2f},{state.T[2]:.2f}) m",
                (12, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (220, 225, 235), 1, cv2.LINE_AA)
    cv2.putText(panel, f"MODEL INPUT: {input_size}x{input_size} crop",
                (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 190, 205), 1, cv2.LINE_AA)
    return panel


def orient_frame(frame: np.ndarray, rotation: int, mirror: bool) -> np.ndarray:
    if rotation == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mirror:
        frame = cv2.flip(frame, 1)
    return frame


def damp_state(previous: HandState, updated: HandState, strength: float) -> HandState:
    """Blend the learned update conservatively to suppress unstable jumps."""
    strength = float(np.clip(strength, 0.0, 1.0))
    out = previous.copy()
    for name in ("T", "q", "V", "omega", "qdot"):
        a, b = getattr(previous, name), getattr(updated, name)
        setattr(out, name, a + strength * (b - a))
    q1, q2 = previous.R.copy(), updated.R.copy()
    if float(np.dot(q1, q2)) < 0:
        q2 = -q2
    q = q1 + strength * (q2 - q1)
    out.R = q / max(float(np.linalg.norm(q)), 1e-9)
    out.beta = updated.beta.copy()
    return out


def finger_curl_target(state: HandState, percentages) -> np.ndarray:
    """Manual articulation controls, independent of the model or camera image."""
    q = state.q.copy()
    thumb = percentages[0] / 100.0
    q[0:5] = thumb * np.asarray([0.5, 0.6, 0.3, 0.9, 1.3])
    for percent, base in zip(percentages[1:], (5, 9, 13, 17)):
        q[base + 1:base + 4] = (percent / 100.0) * np.asarray([1.45, 1.65, 1.30])
    limits = importlib.import_module('02_state.constraints')
    return np.clip(q, limits.JOINT_LIMITS_LOWER, limits.JOINT_LIMITS_UPPER)


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
    if changed and key in tuple(map(ord, "jlikuo")):
        state.R = coord.euler_to_quaternion(*angles)
    return changed


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.correction_strength <= 1.0:
        raise ValueError("--correction-strength must be between 0 and 1")
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
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError("Webcam opened but returned no frames.")
    raw_height, raw_width = first.shape[:2]
    first = orient_frame(first, args.rotate, args.mirror)
    height, width = first.shape[:2]
    focal = float(args.focal_px) if args.focal_px > 0 else 0.5 * raw_width / math.tan(math.radians(args.hfov) * 0.5)
    intr = CameraIntrinsics(
        focal, focal, (width - 1) * 0.5, (height - 1) * 0.5, width, height
    )
    renderer = OrientedRenderedPriorInput(
        intr, raw_width=raw_width, raw_height=raw_height,
        rotation=args.rotate, mirror=args.mirror,
        biometrics=biometrics, mano_model_path=mano_path,
        output_size=image_size, crop_margin=1.6,
    )

    tracking = False
    authority = 0.0
    last_time = time.perf_counter()
    fps_ema = 0.0
    last_report = last_time
    model_steps = 0
    window_name = "Phase 7 trained model — webcam smoke test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    controls_window = "MANUAL finger controls - moving a slider pauses the model"
    cv2.namedWindow(controls_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(controls_window, 520, 300)
    finger_names = ("Thumb", "Index", "Middle", "Ring", "Pinky")
    for name in finger_names:
        cv2.createTrackbar(name, controls_window, 0, 100, lambda value: None)
    last_curls = (0, 0, 0, 0, 0)
    manual_target = None
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}; model input: {image_size}x{image_size}; camera output: {width}x{height}; focal~{focal:.1f}px")
    if width < args.width or height < args.height:
        print(f"[WARN] Camera returned {width}x{height}, below requested {args.width}x{args.height}.")
    print(f"Camera transform: rotate={args.rotate} degrees; mirror={args.mirror}; correction strength={args.correction_strength:.2f}")
    print("Align the green skeleton over an open palm facing the camera, fingers up; press SPACE to start.")
    print("Finger sliders demonstrate MANUAL articulation. SPACE toggles learned corrections; R resets.")
    panel_width = 520

    try:
        while True:
            frame_started = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = orient_frame(frame, args.rotate, args.mirror)
            now = time.perf_counter()
            actual_dt = max(now - last_time, 1e-6)
            dt = float(np.clip(actual_dt, 1.0 / 240.0, 0.25))
            last_time = now
            fps = 1.0 / actual_dt
            fps_ema = fps if fps_ema == 0.0 else 0.90 * fps_ema + 0.10 * fps

            curls = tuple(cv2.getTrackbarPos(name, controls_window) for name in finger_names)
            if curls != last_curls:
                manual_target = finger_curl_target(state, curls)
                last_curls = curls
                tracking = False
                authority = 0.0
            if manual_target is not None and not tracking:
                state.q += (1.0 - math.exp(-actual_dt / 0.08)) * (manual_target - state.q)
                state.V[:] = 0.0
                state.omega[:] = 0.0
                state.qdot[:] = 0.0
                tracker.reset(state)

            crop, _ = renderer(frame, state)
            crop_finished = time.perf_counter()
            # The input is rendered from the current prior. If tracking is on,
            # display the corrected state on this same camera frame below.
            display_state = state.copy()
            joints_m = renderer.last_joints
            vertices_m = renderer.last_vertices
            faces = renderer.last_faces
            if joints_m is None or vertices_m is None or faces is None:
                raise RuntimeError("MANO prior render did not return its cached mesh.")
            if tracking:
                previous_state = state.copy()
                result = tracker.step(crop, dt_seconds=dt, measurement_valid=True)
                state = damp_state(previous_state, result.state, args.correction_strength)
                tracker.state = state.copy()
                authority = result.authority
                # main.py renders MANO from the current HandState each frame.
                # Do the same after inference so the skeleton and skin show
                # the just-predicted pose instead of lagging one frame behind.
                joints_m, _ = forward_kinematics(state, biometrics)
                vertices_m, faces, _ = renderer.mano.forward(state, joints_m, biometrics)
                display_state = state.copy()
                model_steps += 1

            inference_finished = time.perf_counter()

            camera_view = draw_camera_overlay(
                frame, joints_m, vertices_m, faces, renderer,
                tracking, authority, fps_ema,
            )
            cv2.putText(camera_view, "MODEL CORRECTION" if tracking else "MANUAL POSE / ALIGNMENT",
                        (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 210, 255), 2, cv2.LINE_AA)
            panel = draw_3d_mano(
                display_state, joints_m, vertices_m, faces,
                (panel_width, height), image_size,
            )
            crop_rgb = crop[:3].permute(1, 2, 0).cpu().numpy()
            crop_bgr = cv2.cvtColor(np.clip(crop_rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            skeleton_mask = crop[4].cpu().numpy() > 0.5
            crop_bgr[skeleton_mask] = (0, 255, 0)
            crop_view = cv2.resize(crop_bgr, (180, 180), interpolation=cv2.INTER_LINEAR)
            x0, y0 = 14, max(34, height - 196)
            panel[y0:y0 + 180, x0:x0 + 180] = crop_view
            cv2.putText(panel, "RGB crop + prior skeleton", (x0, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 225, 235), 1, cv2.LINE_AA)
            cv2.putText(panel, f"depth {display_state.T[2]:.2f} m | learned authority {authority:.2f}",
                        (12, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (220, 225, 235), 1, cv2.LINE_AA)

            canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
            canvas[:, :width] = camera_view
            canvas[:, width:] = panel
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            finished = time.perf_counter()
            if finished - last_report >= 2.0:
                print(
                    f"[live] camera={width}x{height} FPS={fps_ema:.1f} "
                    f"capture={(now-frame_started)*1000:.1f}ms "
                    f"input={(crop_finished-now)*1000:.1f}ms "
                    f"model={(inference_finished-crop_finished)*1000:.1f}ms "
                    f"display={(finished-inference_finished)*1000:.1f}ms "
                    f"model_steps={model_steps} depth={state.T[2]:.3f}m",
                    flush=True,
                )
                last_report = finished
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                tracking = not tracking
                manual_target = None
                tracker.reset(state)
                print("Model corrections active." if tracking else "Model paused; manual pose controls active.")
                continue
            if key == ord("r"):
                state, angles_tuple = make_seed(args.depth)
                angles = list(angles_tuple)
                tracker.reset(state)
                tracking = False
                authority = 0.0
                manual_target = None
                for name in finger_names:
                    cv2.setTrackbarPos(name, controls_window, 0)
                last_curls = (0, 0, 0, 0, 0)
                continue
            if adjust_seed(state, angles, key):
                tracking = False
                authority = 0.0
                manual_target = None
                tracker.reset(state)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
