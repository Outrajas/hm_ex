#!/usr/bin/env python
"""Train a resumable temporal visual corrector for the persistent hand state.

The reader indexes paths and loads frames lazily. Checkpoints are written after
every epoch and can live on Google Drive or another persistent mounted volume.
For a single 56-segment Mugsy subset, the locked 44/12 segment split is used.
With multiple captures, validation groups are held out by capture when possible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

data_api = importlib.import_module("06_training.reinterhand_sequences")
target_api = importlib.import_module("06_training.targets")
architecture = importlib.import_module("04_network.architecture")
render_api = importlib.import_module("08_runtime.render_input")
state_api = importlib.import_module("00_spec.state_definition")
coords = importlib.import_module("00_spec.coordinate_system")
fk_api = importlib.import_module("02_state.forward_kinematics")
constraints = importlib.import_module("00_spec.constraints")

HandState = state_api.HandState
HandBiometrics = target_api.HandBiometrics
CameraIntrinsics = coords.CameraIntrinsics
TemporalStateCorrector = architecture.TemporalStateCorrector
pack_state_features = architecture.pack_state_features
RenderedPriorInput = render_api.RenderedPriorInput
forward_kinematics = fk_api.forward_kinematics

SEED = 20251018
STATE_DIM = 65
STATE_SLICES = {
    "T": slice(0, 3), "R": slice(3, 7), "q": slice(7, 28),
    "V": slice(28, 31), "omega": slice(31, 34),
    "qdot": slice(34, 55), "beta": slice(55, 65),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=[], help="Subset manifest.csv; repeat for multiple captures.")
    parser.add_argument("--validation-manifest", action="append", default=[], help="Fixed, capture-held-out validation manifest; repeat as needed.")
    parser.add_argument("--capture", action="append", default=[], help="Official capture root with frame_list.txt; repeat as needed.")
    parser.add_argument("--data-root", default="", help="Search under this root for derived/mugsy_subset/manifest.csv files.")
    parser.add_argument("--camera-group", default="Mugsy_cameras")
    parser.add_argument("--envmap-mode", default="envmap_per_segment")
    parser.add_argument("--camera-id", default="", help="Optional camera filter, e.g. 400485.")
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints" / "phase5_temporal"))
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint-dir/latest.pt if present.")
    parser.add_argument("--init-checkpoint", default="", help="Initialize model weights from another dataset checkpoint; optimizer and sampler start fresh.")
    parser.add_argument("--checkpoint-every-steps", type=int, default=25, help="Persist model/optimizer state every N batches; 0 disables mid-epoch checkpoints.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--batch-windows", type=int, default=4)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--max-frame-gap", type=int, default=30, help="Break windows at larger frame-number gaps; 30 is 1s at 30fps.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--val-windows", type=int, default=32, help="Fixed validation windows each epoch; 0 means all.")
    parser.add_argument("--max-train-windows", type=int, default=0, help="Debug/smoke-test cap; 0 means all.")
    parser.add_argument("--max-val-windows", type=int, default=0, help="Debug cap; 0 uses --val-windows/all.")
    parser.add_argument("--workers", type=int, default=0, help="Reserved for future async decoding; decoding is currently lazy and in-process.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return device


def print_resource_status(device: torch.device, checkpoint_dir: Path) -> None:
    try:
        import psutil
        vm = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        print(
            f"RAM: available={vm.available / 1024**3:.1f} GiB; "
            f"trainer={process.memory_info().rss / 1024**3:.2f} GiB"
        )
    except Exception:
        available = None
        try:
            if os.name == "nt":
                import ctypes

                class _MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                status = _MemoryStatus()
                status.dwLength = ctypes.sizeof(status)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    available = int(status.ullAvailPhys)
            else:
                pages = os.sysconf("SC_AVPHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                available = int(pages * page_size)
        except Exception:
            pass
        if available is None:
            print("RAM: system availability could not be queried")
        else:
            print(f"RAM: available={available / 1024**3:.1f} GiB; process usage unavailable")
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f"VRAM: free={free_bytes / 1024**3:.1f} / "
            f"{total_bytes / 1024**3:.1f} GiB; device={torch.cuda.get_device_name(device)}"
        )
    try:
        disk = __import__("shutil").disk_usage(checkpoint_dir)
        print(f"Checkpoint volume: free={disk.free / 1024**3:.1f} GiB")
    except OSError:
        pass
    print("Memory policy: RGB frames and MANO renders are loaded per sequence batch; the image corpus is never copied into RAM.")


def discover_sources(args: argparse.Namespace, *, validation: bool = False):
    if validation:
        manifests = [Path(x).expanduser().resolve() for x in args.validation_manifest]
        captures = []
        if not manifests:
            raise ValueError("Validation source requested but --validation-manifest was not provided.")
    else:
        manifests = [Path(x).expanduser().resolve() for x in args.manifest]
        captures = [Path(x).expanduser().resolve() for x in args.capture]
    if not validation and args.data_root:
        root = Path(args.data_root).expanduser().resolve()
        manifests.extend(sorted(root.glob("**/derived/mugsy_subset/manifest.csv")))
    if not validation and not manifests and not captures:
        local = ROOT / "datasets" / "reinterhand" / "captures"
        manifests = sorted(local.glob("*/derived/mugsy_subset/manifest.csv"))
    if not validation and args.validation_manifest:
        held_out = {Path(x).expanduser().resolve() for x in args.validation_manifest}
        manifests = [Path(path).resolve() for path in manifests if Path(path).resolve() not in held_out]
    manifests = list(dict.fromkeys(manifests))
    captures = list(dict.fromkeys(captures))
    if not manifests and not captures:
        raise FileNotFoundError("Pass --manifest, --capture, or --data-root. No local Re:InterHand subset was found.")

    records = []
    camera_by_capture = {}
    camera_paths = {}

    def register_camera(capture_root: Path, camera_path: Path):
        capture_key = capture_root.name
        existing_path = camera_paths.get(capture_key)
        if existing_path is not None and existing_path.resolve() != camera_path.resolve():
            raise ValueError(
                f"Two source roots share capture key {capture_key!r}: "
                f"{existing_path} and {camera_path}. Rename or separate the roots."
            )
        with camera_path.open("r", encoding="utf-8") as stream:
            camera_by_capture[capture_key] = json.load(stream)
        camera_paths[capture_key] = camera_path

    for manifest in manifests:
        recs = data_api.records_from_subset_manifest(
            manifest, camera_id=args.camera_id or None
        )
        capture_root = data_api.capture_root_from_manifest(manifest)
        camera_candidates = (
            capture_root / args.camera_group / "cam_params.json",
            capture_root / "cam_params.json",
        )
        camera_path = next((path for path in camera_candidates if path.is_file()), camera_candidates[0])
        if not camera_path.is_file():
            raise FileNotFoundError(camera_path)
        register_camera(capture_root, camera_path)
        records.extend(recs)
    for capture_root in captures:
        recs = data_api.records_from_capture(
            capture_root,
            camera_group=args.camera_group,
            envmap_mode=args.envmap_mode,
            camera_ids=[args.camera_id] if args.camera_id else None,
        )
        camera_path = capture_root / args.camera_group / "cam_params.json"
        register_camera(capture_root, camera_path)
        records.extend(recs)

    unique = {}
    for record in records:
        key = (str(record.image_path.resolve()), record.camera_id)
        unique[key] = record
    records = sorted(unique.values(), key=lambda r: (r.capture_id, r.segment, r.camera_id, r.frame_idx))
    return records, camera_by_capture, camera_paths


def group_split(records, fraction: float, seed: int):
    captures = sorted({r.capture_id for r in records})
    if len(captures) >= 2:
        train_idx, val_idx = data_api.split_by_group(
            records, validation_fraction=fraction, seed=seed, group_by="capture"
        )
        train_groups = {records[i].capture_id for i in train_idx}
        val_groups = {records[i].capture_id for i in val_idx}
        return train_idx, val_idx, train_groups, val_groups, "capture"

    groups = sorted({(r.capture_id, r.segment) for r in records})
    if len(captures) == 1 and len(groups) == 56:
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(sorted({r.segment for r in records}), dtype=object)
        rng.shuffle(shuffled)
        train_segments = set(shuffled[:44].tolist())
        val_segments = set(shuffled[44:].tolist())
        train = np.asarray([i for i, r in enumerate(records) if r.segment in train_segments], np.int64)
        val = np.asarray([i for i, r in enumerate(records) if r.segment in val_segments], np.int64)
        return train, val, train_segments, val_segments, "segment-44/12"

    train, val = data_api.split_by_group(
        records, validation_fraction=fraction, seed=seed, group_by="segment"
    )
    train_groups = {(records[i].capture_id, records[i].segment) for i in train}
    val_groups = {(records[i].capture_id, records[i].segment) for i in val}
    return train, val, train_groups, val_groups, "segment"


def dataset_signature(records, camera_paths, split_seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(split_seed).encode())
    for record in records:
        digest.update(f"{record.capture_id}|{record.segment}|{record.camera_id}|{record.frame_idx}|{record.image_path}\n".encode())
        for label, asset_path in (
            ("image", record.image_path),
            ("right_keypoints", record.right_keypoints_path),
            ("right_mano", record.right_mano_path),
        ):
            if asset_path is None:
                digest.update(f"{label}|none\n".encode())
                continue
            stat = asset_path.stat()
            digest.update(
                f"{label}|{asset_path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
            )
    for key in sorted(camera_paths):
        path = camera_paths[key]
        stat = path.stat()
        digest.update(f"{path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:24]


def biometrics_from_checkpoint(payload: dict, checkpoint_path: Path) -> HandBiometrics | None:
    raw = payload.get("biometrics")
    if raw is None:
        print(
            f"Checkpoint has no stored hand calibration: {checkpoint_path}; "
            "the first chunk will calibrate from its training labels."
        )
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid biometrics metadata in checkpoint: {checkpoint_path}")
    biometrics = HandBiometrics(**raw)
    values = np.asarray(list(biometrics.__dict__.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite biometrics metadata in checkpoint: {checkpoint_path}")
    return biometrics


def same_biometrics(left: HandBiometrics, right: HandBiometrics) -> bool:
    left_values = left.__dict__
    right_values = right.__dict__
    return left_values.keys() == right_values.keys() and all(
        math.isclose(float(left_values[key]), float(right_values[key]), rel_tol=0.0, abs_tol=1e-8)
        for key in left_values
    )


def load_or_fit_targets(
    records, camera_by_capture, train_ids, cache_path: Path, signature: str,
    fixed_biometrics: HandBiometrics | None = None,
):
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as saved:
                if str(saved["signature"].item()) == signature and len(saved["states"]) == len(records):
                    biometrics = HandBiometrics(**json.loads(str(saved["biometrics_json"].item())))
                    if fixed_biometrics is None or same_biometrics(biometrics, fixed_biometrics):
                        print(f"Loaded cached HandState targets: {cache_path}")
                        camera_joints = saved["camera_joints"].copy()
                        states = saved["states"].copy()
                        residuals = saved["residuals"].copy()
                        # Always rebuild motion features when reading a cache. Older
                        # caches used centered differences and therefore contained
                        # future-frame information. The causal implementation below
                        # also makes cache reuse safe after upgrading this script.
                        derive_target_velocities(records, states)
                        temp_path = cache_path.with_suffix(".tmp.npz")
                        np.savez_compressed(
                            temp_path,
                            signature=np.asarray(signature),
                            camera_joints=camera_joints,
                            states=states,
                            residuals=residuals,
                            biometrics_json=np.asarray(json.dumps(biometrics.__dict__)),
                        )
                        os.replace(temp_path, cache_path)
                        print("Rebuilt cached motion features with past-only differences")
                        return camera_joints, states, residuals, biometrics
                    print("Target cache uses different hand proportions; rebuilding for the checkpoint's fixed calibration.")
        except Exception as exc:
            print(f"Target cache will be rebuilt ({exc})")

    camera_joints = np.empty((len(records), 21, 3), np.float32)
    betas = np.empty((len(records), 10), np.float32)
    for i, record in enumerate(records):
        with record.right_keypoints_path.open("r", encoding="utf-8") as stream:
            world = np.asarray(json.load(stream), np.float32)
        if world.shape != (24, 3):
            raise ValueError(f"Expected 24x3 right-hand labels: {record.right_keypoints_path}")
        cam = camera_by_capture[record.capture_id][record.camera_id]
        rotation = np.asarray(cam["R"], np.float32).reshape(3, 3)
        translation = np.asarray(cam["t"], np.float32).reshape(3)
        camera_joints[i] = world[data_api.KEEP_21_INDICES] @ rotation.T + translation
        if record.right_mano_path is not None and record.right_mano_path.is_file():
            with record.right_mano_path.open("r", encoding="utf-8") as stream:
                mano = json.load(stream)
            betas[i] = np.asarray(mano["shape"], np.float32).reshape(10)
        else:
            betas[i] = 0.0
        if (i + 1) % 1000 == 0 or i + 1 == len(records):
            print(f"Read 3D labels {i + 1}/{len(records)}")

    biometrics = fixed_biometrics or target_api.calibrate_biometrics_from_joints(camera_joints[train_ids])
    print(
        ("Frozen checkpoint shape calibration: " if fixed_biometrics is not None else "Train-only shape calibration: ")
        +
        f"palm={biometrics.palm_length_ratio:.3f}, "
        f"width={biometrics.palm_width_ratio:.3f}, "
        f"finger={biometrics.finger_length_ratio:.3f}, "
        f"thumb={biometrics.thumb_ratio:.3f}"
    )
    states = np.empty((len(records), STATE_DIM), np.float32)
    residuals = np.empty(len(records), np.float32)
    for i, joints in enumerate(camera_joints):
        state, _, residual = target_api.fit_state_from_camera_joints(
            joints, biometrics=biometrics, beta=betas[i], clamp_limits=True
        )
        states[i] = state.to_vector().astype(np.float32)
        residuals[i] = residual
        if (i + 1) % 1000 == 0 or i + 1 == len(records):
            print(f"Fit HandState targets {i + 1}/{len(records)}")
    derive_target_velocities(records, states)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp_path,
        signature=np.asarray(signature),
        camera_joints=camera_joints,
        states=states,
        residuals=residuals,
        biometrics_json=np.asarray(json.dumps(biometrics.__dict__)),
    )
    os.replace(temp_path, cache_path)
    print(f"Cached targets: {cache_path}")
    return camera_joints, states, residuals, biometrics


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray([
        aw*bw-ax*bx-ay*by-az*bz,
        aw*bx+ax*bw+ay*bz-az*by,
        aw*by-ax*bz+ay*bw+az*bx,
        aw*bz+ax*by-ay*bx+az*bw,
    ], np.float64)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], np.float64) / max(float(np.dot(q, q)), 1e-12)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    if q[0] < 0:
        q = -q
    sin_half = float(np.linalg.norm(q[1:]))
    if sin_half < 1e-9:
        return 2.0 * q[1:]
    return q[1:] / sin_half * (2.0 * math.atan2(sin_half, float(q[0])))


def derive_target_velocities(records, states: np.ndarray) -> None:
    """Fill motion features using only poses available before each frame.

    At frame t, V/qdot/omega describe the t-2 -> t-1 transition. The current
    frame's pose and all future poses are excluded, matching a causal tracker
    whose state is updated before it receives the current image.
    """
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault((record.capture_id, record.segment, record.camera_id), []).append(i)
    max_v = float(getattr(constraints, "MAX_TRANSLATIONAL_VELOCITY", 5.0))
    max_qdot = float(getattr(constraints, "MAX_JOINT_VELOCITY", 40.0))
    for indices in groups.values():
        indices.sort(key=lambda i: records[i].frame_idx)
        for pos, idx in enumerate(indices):
            # Zero velocity until two earlier observations are available.
            states[idx, STATE_SLICES["V"]] = 0.0
            states[idx, STATE_SLICES["omega"]] = 0.0
            states[idx, STATE_SLICES["qdot"]] = 0.0
            if pos >= 2:
                previous = indices[pos - 1]
                earlier = indices[pos - 2]
                dt = max(records[previous].timestamp_seconds - records[earlier].timestamp_seconds, 1e-4)
                continuous = (
                    records[idx].frame_idx - records[previous].frame_idx <= 30
                    and records[previous].frame_idx - records[earlier].frame_idx <= 30
                )
                if continuous:
                    # This is the velocity known at the end of frame t-1,
                    # which can be carried into frame t without peeking ahead.
                    states[idx, STATE_SLICES["V"]] = np.clip(
                        (states[previous, STATE_SLICES["T"]] - states[earlier, STATE_SLICES["T"]]) / dt,
                        -max_v, max_v,
                    )
                    states[idx, STATE_SLICES["qdot"]] = np.clip(
                        (states[previous, STATE_SLICES["q"]] - states[earlier, STATE_SLICES["q"]]) / dt,
                        -max_qdot, max_qdot,
                    )
                    q0 = states[earlier, STATE_SLICES["R"]]
                    q1 = states[previous, STATE_SLICES["R"]]
                    states[idx, STATE_SLICES["omega"]] = np.clip(
                        quat_to_rotvec(quat_mul(q1, quat_inverse(q0))) / dt, -20.0, 20.0
                    )


def build_windows(records, allowed_ids, args):
    raw = data_api.sequence_windows(
        records,
        window_size=args.window_size,
        stride=args.stride,
        max_frame_gap=args.max_frame_gap,
    )
    allowed = set(map(int, allowed_ids))
    return [window for window in raw if all(int(i) in allowed for i in window)]


def _perturb_sequence(target_vecs: np.ndarray, times: np.ndarray, rng: np.random.Generator):
    priors = target_vecs.copy()
    trans_noise = rng.normal(0, 0.018, size=3)
    joint_noise = rng.normal(0, np.deg2rad(9.0), size=21)
    rotation_noise = rng.normal(0, np.deg2rad(7.0), size=3)
    prev_t = times[0]
    lo = np.asarray(constraints.JOINT_LIMITS_LOWER, np.float32)
    hi = np.asarray(constraints.JOINT_LIMITS_UPPER, np.float32)
    for j, vec in enumerate(target_vecs):
        if j:
            dt = max(float(times[j] - prev_t), 1e-3)
            rho = math.exp(-dt / 0.45)
            trans_noise = rho * trans_noise + rng.normal(0, 0.004, 3)
            joint_noise = rho * joint_noise + rng.normal(0, np.deg2rad(2.2), 21)
            rotation_noise = rho * rotation_noise + rng.normal(0, np.deg2rad(1.5), 3)
        prev_t = times[j]
        priors[j, STATE_SLICES["T"]] += trans_noise
        priors[j, STATE_SLICES["q"]] = np.clip(vec[STATE_SLICES["q"]] + joint_noise, lo, hi)
        noise_q = coords.rotation_vector_to_quaternion(rotation_noise)
        priors[j, STATE_SLICES["R"]] = quat_mul(noise_q, vec[STATE_SLICES["R"]])
        priors[j, STATE_SLICES["R"]] /= max(float(np.linalg.norm(priors[j, STATE_SLICES["R"]])), 1e-9)
    return priors


def make_renderers(records, camera_by_capture, biometrics, output_size, mano_path):
    renderers = {}
    for record in records:
        key = (record.capture_id, record.camera_id)
        if key in renderers:
            continue
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record.image_path)
        h, w = image.shape[:2]
        params = camera_by_capture[record.capture_id][record.camera_id]
        focal = np.asarray(params["focal"], np.float64).reshape(2)
        principal = np.asarray(params["princpt"], np.float64).reshape(2)
        intrinsics = CameraIntrinsics(float(focal[0]), float(focal[1]), float(principal[0]), float(principal[1]), w, h)
        renderers[key] = RenderedPriorInput(
            intrinsics, biometrics=biometrics, mano_model_path=mano_path,
            output_size=output_size, crop_margin=1.6,
        )
        del image
    return renderers


def _augment_input(channels: np.ndarray, rng: np.random.Generator, *, enabled: bool, occlude: bool):
    result = channels.copy()
    if enabled:
        contrast = float(rng.uniform(0.8, 1.2))
        brightness = float(rng.uniform(-0.08, 0.08))
        result[:3] = np.clip((result[:3] - 0.5) * contrast + 0.5 + brightness, 0, 1)
    was_occluded = bool(occlude and rng.random() < 0.55)
    if was_occluded:
        h, w = result.shape[1:]
        rects = 1 if rng.random() < 0.75 else 2
        for _ in range(rects):
            rw = int(rng.uniform(0.12, 0.35) * w)
            rh = int(rng.uniform(0.12, 0.35) * h)
            x = int(rng.integers(0, max(1, w - rw + 1)))
            y = int(rng.integers(0, max(1, h - rh + 1)))
            result[:, y:y + rh, x:x + rw] = 0.0
    return result, was_occluded


def build_sample(window, records, states, renderers, rng, *, train: bool, image_size: int):
    ids = np.asarray(window, np.int64)
    target = states[ids].copy()
    times = np.asarray([records[int(i)].timestamp_seconds for i in ids], np.float64)
    prior = _perturb_sequence(target, times, rng) if train else _perturb_sequence(target, times, np.random.default_rng(SEED + int(ids[0])))
    # The recurrent model receives motion features from its perturbed prior,
    # computed causally within this window. Do not pass label-derived velocity
    # features through from the target states.
    derive_target_velocities([records[int(i)] for i in ids], prior)
    images, input_states, deltas = [], [], []
    valid = np.ones(len(ids), np.float32)
    gate_target = np.full(len(ids), 0.90, np.float32)
    if train and len(ids) >= 2 and rng.random() < 0.22:
        start = int(rng.integers(0, len(ids)))
        length = int(rng.integers(1, min(4, len(ids) - start) + 1))
        valid[start:start + length] = 0.0
    for pos, index in enumerate(ids):
        record = records[int(index)]
        prior_state = HandState.from_vector(prior[pos].astype(np.float64))
        image_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(record.image_path)
        if valid[pos] < 0.5:
            channels = np.zeros((5, image_size, image_size), np.float32)
            gate_target[pos] = 0.03
        else:
            renderer = renderers[(record.capture_id, record.camera_id)]
            tensor, _ = renderer(image_bgr, prior_state)
            channels = tensor.numpy().astype(np.float32, copy=False)
            channels, occluded = _augment_input(channels, rng, enabled=train, occlude=train)
            if occluded:
                gate_target[pos] = 0.42
        images.append(channels)
        input_states.append(prior[pos])
        desired = np.concatenate((
            target[pos, STATE_SLICES["T"]] - prior[pos, STATE_SLICES["T"]],
            quat_to_rotvec(quat_mul(
                target[pos, STATE_SLICES["R"]],
                quat_inverse(prior[pos, STATE_SLICES["R"]]),
            )),
            target[pos, STATE_SLICES["q"]] - prior[pos, STATE_SLICES["q"]],
        )).astype(np.float32)
        deltas.append(desired)
        del image_bgr
    dt = np.diff(times, prepend=times[0] - 1 / 30).astype(np.float32)
    dt = np.clip(dt, 1 / 240, 1.0)
    return (
        np.stack(images), np.stack(input_states).astype(np.float32),
        np.stack(deltas), valid, dt, gate_target,
    )


def batch_features(state_vectors: torch.Tensor, device: torch.device) -> torch.Tensor:
    b, t, _ = state_vectors.shape
    flat = state_vectors.reshape(b * t, STATE_DIM)
    feat = pack_state_features(
        flat[:, STATE_SLICES["T"]], flat[:, STATE_SLICES["R"]],
        flat[:, STATE_SLICES["q"]], flat[:, STATE_SLICES["V"]],
        flat[:, STATE_SLICES["omega"]], flat[:, STATE_SLICES["qdot"]],
        flat[:, STATE_SLICES["beta"]],
    )
    return feat.reshape(b, t, -1).to(device)


def _save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _rotation_apply_torch(delta: torch.Tensor, prior_q: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
    half = angle * 0.5
    q_delta = torch.cat((torch.cos(half), delta * (torch.sin(half) / angle)), dim=-1)
    w1, x1, y1, z1 = q_delta.unbind(-1)
    w2, x2, y2, z2 = prior_q.unbind(-1)
    return F.normalize(torch.stack((
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
    ), dim=-1), dim=-1)


def _quaternion_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dot = torch.abs(torch.sum(F.normalize(pred, dim=-1) * F.normalize(target, dim=-1), dim=-1)).clamp(0, 1)
    return (2.0 * torch.acos(dot.clamp(max=1 - 1e-7))).square()


def run_eval(model, windows, records, states, camera_joints, renderers, device, image_size):
    model.eval()
    rng = np.random.default_rng(SEED + 901)
    batch_samples = [build_sample(w, records, states, renderers, rng, train=False, image_size=image_size) for w in windows]
    if not batch_samples:
        return {"prior_mpjpe_mm": float("nan"), "corrected_mpjpe_mm": float("nan"), "authority": float("nan")}
    all_prior, all_corr, all_target = [], [], []
    all_auth = []
    with torch.inference_mode():
        for start in range(0, len(batch_samples), 4):
            batch = batch_samples[start:start + 4]
            bx = torch.from_numpy(np.stack([x[0] for x in batch])).to(device)
            sv_np = np.stack([x[1] for x in batch])
            sv = torch.from_numpy(sv_np).to(device)
            features = batch_features(sv, device)
            dt = torch.from_numpy(np.stack([x[4] for x in batch])).to(device)
            output = model.forward_sequence(bx, features, dt)
            delta = torch.cat((output.delta_T, output.delta_R, output.delta_q), dim=-1).cpu().numpy()
            auth = output.authority.cpu().numpy()
            for b, sample in enumerate(batch):
                length = sample[0].shape[0]
                for t in range(length):
                    prior_vec = sample[1][t]
                    prior_state = HandState.from_vector(prior_vec.astype(np.float64))
                    pred_delta = delta[b, t]
                    prior_pose = forward_kinematics(prior_state)[0] * 1000.0
                    corrected = prior_state.copy()
                    corrected.T += float(auth[b, t]) * pred_delta[:3]
                    corrected.R = quat_mul(
                        coords.rotation_vector_to_quaternion(float(auth[b, t]) * pred_delta[3:6]),
                        corrected.R,
                    )
                    corrected.q = np.clip(
                        corrected.q + float(auth[b, t]) * pred_delta[6:],
                        constraints.JOINT_LIMITS_LOWER,
                        constraints.JOINT_LIMITS_UPPER,
                    )
                    corrected_pose = forward_kinematics(corrected)[0] * 1000.0
                    frame_index = int(windows[start + b][t])
                    all_prior.append(prior_pose)
                    all_corr.append(corrected_pose)
                    all_target.append(camera_joints[frame_index])
                    all_auth.append(float(auth[b, t]))
    prior_error = np.linalg.norm(np.asarray(all_prior) - np.asarray(all_target), axis=-1).mean()
    corrected_error = np.linalg.norm(np.asarray(all_corr) - np.asarray(all_target), axis=-1).mean()
    return {
        "prior_mpjpe_mm": float(prior_error),
        "corrected_mpjpe_mm": float(corrected_error),
        "authority": float(np.mean(all_auth)),
        "frames": len(all_auth),
    }


def main():
    args = parse_args()
    if args.epochs < 1 or args.window_size < 2 or args.batch_windows < 1:
        raise ValueError("epochs and batch-windows must be positive; window-size must be >=2")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = choose_device(args.device)
    if args.device == "auto" and device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    records, camera_by_capture, camera_paths = discover_sources(args)
    if args.validation_manifest:
        validation_records, validation_cameras, validation_camera_paths = discover_sources(
            args, validation=True
        )
        train_images = {str(r.image_path.resolve()) for r in records}
        val_images = {str(r.image_path.resolve()) for r in validation_records}
        overlap = train_images.intersection(val_images)
        if overlap:
            raise ValueError(f"Training and validation manifests share {len(overlap)} image paths.")

        validation_id_map = {}
        for old_id in validation_cameras:
            new_id = f"validation::{old_id}"
            if new_id in camera_by_capture:
                raise ValueError(f"Validation capture key collision: {new_id}")
            validation_id_map[old_id] = new_id
        validation_records = [
            replace(record, capture_id=validation_id_map[record.capture_id])
            for record in validation_records
        ]
        camera_by_capture.update({validation_id_map[key]: value for key, value in validation_cameras.items()})
        camera_paths.update({validation_id_map[key]: value for key, value in validation_camera_paths.items()})
        train_count = len(records)
        records.extend(validation_records)
        train_idx = np.arange(train_count, dtype=np.int64)
        val_idx = np.arange(train_count, len(records), dtype=np.int64)
        train_groups = {records[i].capture_id for i in train_idx}
        val_groups = {records[i].capture_id for i in val_idx}
        split_kind = "explicit-validation-manifest"
    else:
        train_idx, val_idx, train_groups, val_groups, split_kind = group_split(
            records, args.validation_fraction, args.split_seed
        )
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    resume_payload = None
    init_payload = None
    fixed_biometrics = None
    if args.resume and latest_path.is_file():
        resume_payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        fixed_biometrics = biometrics_from_checkpoint(resume_payload, latest_path)
    elif args.init_checkpoint:
        init_path = Path(args.init_checkpoint).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"Initialization checkpoint not found: {init_path}")
        init_payload = torch.load(init_path, map_location="cpu", weights_only=False)
        if "model" not in init_payload:
            raise ValueError(f"Checkpoint has no model state_dict: {init_path}")
        fixed_biometrics = biometrics_from_checkpoint(init_payload, init_path)
    signature = dataset_signature(records, camera_paths, args.split_seed)
    print_resource_status(device, checkpoint_dir)
    target_cache = checkpoint_dir / "target_cache" / f"states_{signature}.npz"
    camera_joints, target_states, fit_residuals, biometrics = load_or_fit_targets(
        records, camera_by_capture, train_idx, target_cache, signature,
        fixed_biometrics=fixed_biometrics,
    )
    train_windows = build_windows(records, train_idx, args)
    val_windows = build_windows(records, val_idx, args)
    rng_windows = np.random.default_rng(args.split_seed + 44)
    if args.val_windows > 0 and len(val_windows) > args.val_windows:
        selected = np.sort(rng_windows.choice(len(val_windows), args.val_windows, replace=False))
        val_windows = [val_windows[int(i)] for i in selected]
    if args.max_train_windows > 0:
        train_windows = train_windows[:args.max_train_windows]
    if args.max_val_windows > 0:
        val_windows = val_windows[:args.max_val_windows]
    if not train_windows or not val_windows:
        raise RuntimeError(f"No sequence windows: train={len(train_windows)}, val={len(val_windows)}. Lower --window-size or adjust --max-frame-gap.")

    mano_path = ROOT / "models" / "MANO_RIGHT.pkl"
    if not mano_path.is_file():
        raise FileNotFoundError(f"MANO surface asset not found: {mano_path}")
    renderers = make_renderers(records, camera_by_capture, biometrics, args.image_size, mano_path)
    model = TemporalStateCorrector(hidden_dim=args.hidden_dim).to(device)
    if init_payload is not None:
        model.load_state_dict(init_payload["model"], strict=True)
        print(f"Initialized model weights from {init_path}; optimizer starts fresh for this chunk.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 1
    best_val = float("inf")
    resume_order = None
    resume_offset = 0
    resume_augment_state = None
    if resume_payload is not None:
        saved = resume_payload
        if saved.get("dataset_signature") != signature:
            raise RuntimeError("Checkpoint dataset signature differs; choose a fresh --checkpoint-dir or the matching data.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if saved.get("scaler"):
            scaler.load_state_dict(saved["scaler"])
        if saved.get("epoch_complete", True):
            start_epoch = int(saved["epoch"]) + 1
        else:
            start_epoch = int(saved["epoch"])
            resume_order = np.asarray(saved["epoch_order"], dtype=np.int64)
            resume_offset = int(saved["next_offset"])
            resume_augment_state = saved["augment_rng_state"]
        best_val = float(saved.get("best_val_mpjpe_mm", float("inf")))
        if "torch_rng" in saved:
            torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if "numpy_rng" in saved:
            np.random.set_state(saved["numpy_rng"])
        if resume_order is not None:
            print(f"Resuming epoch {start_epoch} at batch offset {resume_offset}/{len(resume_order)} from {latest_path}")
        else:
            print(f"Resumed after epoch {start_epoch - 1} from {latest_path}")

    avg_fit = float(np.mean(fit_residuals[val_idx]))
    print(
        f"Data: {len(records):,} frames; split={split_kind}; "
        f"train={len(train_idx):,} / val={len(val_idx):,}; "
        f"windows={len(train_windows):,}/{len(val_windows):,}; "
        f"validation IK/FK floor~{avg_fit:.2f}mm"
    )
    print(f"Checkpoints and target cache: {checkpoint_dir}")
    if start_epoch > args.epochs:
        print(f"Checkpoint is already at epoch {start_epoch - 1}, >= requested --epochs {args.epochs}.")
        return

    scales = torch.as_tensor(
        [0.05] * 3 + [0.34906585] * 3 + [0.52359878] * 21,
        dtype=torch.float32, device=device,
    ).view(1, 1, 27)

    def checkpoint_payload(epoch, *, epoch_complete, next_offset=0, order=None,
                           augment_rng_state=None, validation=None, best_score=None):
        return {
            "epoch": epoch,
            "epoch_complete": bool(epoch_complete),
            "next_offset": int(next_offset),
            "epoch_order": None if order is None else np.asarray(order, dtype=np.int64),
            "augment_rng_state": augment_rng_state,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_mpjpe_mm": best_val if best_score is None else best_score,
            "validation": validation,
            "dataset_signature": signature,
            "split": {
                "kind": split_kind, "seed": args.split_seed,
                "train_ids": train_idx, "val_ids": val_idx,
                "train_groups": sorted(map(str, train_groups)),
                "val_groups": sorted(map(str, val_groups)),
            },
            "biometrics": dict(biometrics.__dict__),
            "camera_by_capture": camera_by_capture,
            "config": vars(args),
            "input_contract": "RGB + rendered MANO silhouette + projected FK skeleton; NCHW 5 channels",
            "output_contract": "delta_T meters, delta_R axis-angle radians, delta_q radians, log_sigma, authority",
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "numpy_rng": np.random.get_state(),
        }

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        if epoch == start_epoch and resume_order is not None:
            if len(resume_order) != len(train_windows):
                raise RuntimeError("Mid-epoch checkpoint has a different window count; keep the original window configuration.")
            order = resume_order
            offset_start = resume_offset
            rng = np.random.default_rng()
            rng.bit_generator.state = resume_augment_state
        else:
            order = np.arange(len(train_windows))
            np.random.shuffle(order)
            offset_start = 0
            rng = np.random.default_rng(SEED + epoch * 1009)
        running_loss, steps = 0.0, 0
        for offset in range(offset_start, len(order), args.batch_windows):
            chosen = order[offset:offset + args.batch_windows]
            samples = [build_sample(train_windows[int(w)], records, target_states, renderers, rng, train=True, image_size=args.image_size) for w in chosen]
            bx = torch.from_numpy(np.stack([s[0] for s in samples])).to(device)
            state_np = np.stack([s[1] for s in samples])
            states_t = torch.from_numpy(state_np).to(device)
            target_delta = torch.from_numpy(np.stack([s[2] for s in samples])).to(device)
            frame_valid = torch.from_numpy(np.stack([s[3] for s in samples])).to(device)
            features = batch_features(states_t, device)
            dt = torch.from_numpy(np.stack([s[4] for s in samples])).to(device)
            authority_target = torch.from_numpy(np.stack([s[5] for s in samples])).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model.forward_sequence(bx, features, dt, frame_valid=frame_valid)
                predicted = torch.cat((output.delta_T, output.delta_R, output.delta_q), dim=-1)
                applied = predicted * output.authority[..., None]
                normalized_error = (applied - target_delta) / scales
                valid_weight = frame_valid[..., None]
                denom = valid_weight.sum().clamp_min(1.0) * predicted.shape[-1]
                pose_loss = (normalized_error.square() * valid_weight).sum() / denom
                log_sigma = output.log_sigma.float().clamp(-5.0, 2.0)
                nll_each = normalized_error.float().square() * torch.exp(-2 * log_sigma) + 2 * log_sigma
                uncertainty_loss = (nll_each * valid_weight).sum() / denom
                gate_loss = F.binary_cross_entropy(
                    output.authority.float().clamp(1e-5, 1 - 1e-5),
                    authority_target.float(),
                )
                if predicted.shape[1] > 1:
                    pair_valid = frame_valid[:, 1:] * frame_valid[:, :-1]
                    motion = (applied[:, 1:] - applied[:, :-1]) / scales
                    smooth_loss = (motion.square().mean(dim=-1) * pair_valid).sum() / pair_valid.sum().clamp_min(1)
                else:
                    smooth_loss = predicted.new_zeros(())
                loss = pose_loss + 0.025 * uncertainty_loss + 0.04 * gate_loss + 0.002 * smooth_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().float().cpu())
            steps += 1
            batch_number = offset // args.batch_windows + 1
            if batch_number % 25 == 0 or batch_number == 1:
                print(f"epoch {epoch:02d} step {batch_number:04d}/{math.ceil(len(order)/args.batch_windows):04d} loss={running_loss/steps:.4f}")
            next_offset = offset + len(chosen)
            if args.checkpoint_every_steps > 0 and batch_number % args.checkpoint_every_steps == 0:
                payload = checkpoint_payload(
                    epoch,
                    epoch_complete=False,
                    next_offset=next_offset,
                    order=order,
                    augment_rng_state=rng.bit_generator.state,
                    best_score=best_val,
                )
                _save_checkpoint(latest_path, payload)
                print(f"Saved mid-epoch recovery checkpoint at step {batch_number}")
        scheduler.step()
        resume_order = None
        resume_offset = 0
        resume_augment_state = None
        val = run_eval(model, val_windows, records, target_states, camera_joints, renderers, device, args.image_size)
        elapsed = time.perf_counter() - epoch_start
        print(
            f"epoch {epoch:02d}/{args.epochs}: loss={running_loss/max(steps,1):.4f}; "
            f"val prior={val['prior_mpjpe_mm']:.2f}mm corrected={val['corrected_mpjpe_mm']:.2f}mm "
            f"authority={val['authority']:.3f}; {val.get('frames', 0)} frames; {elapsed:.1f}s"
        )
        improved = val["corrected_mpjpe_mm"] < best_val
        best_val = min(best_val, val["corrected_mpjpe_mm"])
        payload = checkpoint_payload(
            epoch, epoch_complete=True, validation=val, best_score=best_val
        )
        _save_checkpoint(latest_path, payload)
        if improved:
            _save_checkpoint(best_path, payload)
        print(f"Saved resumable checkpoint: {latest_path}")

    print(f"Best validation checkpoint: {best_path} ({best_val:.2f} mm on sampled validation windows)")
    print("The saved score is a prototype metric against the current HandState/FK adapter; it is not yet a live-camera result.")


if __name__ == "__main__":
    main()
