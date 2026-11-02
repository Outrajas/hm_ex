#!/usr/bin/env python
"""Train a resumable temporal visual corrector for the persistent hand state.

The reader indexes paths and loads frames lazily. Checkpoints are written after
every epoch and can live on Google Drive or another persistent mounted volume.
For a single 56-segment Mugsy subset, the locked 44/12 segment split is used.
With multiple captures, validation groups are held out by capture when possible.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from threading import RLock

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
    parser.add_argument("--capture", action="append", default=[], help="Official capture root with frame_list.txt; repeat as needed.")
    parser.add_argument("--data-root", default="", help="Search under this root for derived/mugsy_subset/manifest.csv files.")
    parser.add_argument("--camera-group", default="Mugsy_cameras")
    parser.add_argument("--envmap-mode", default="envmap_per_segment")
    parser.add_argument("--camera-id", default="", help="Optional camera filter, e.g. 400485.")
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints" / "phase5_temporal"))
    parser.add_argument("--index-cache-dir", default="", help="Reuse an existing record index when checkpoint output moves.")
    parser.add_argument("--target-cache-path", default="", help="Reuse an existing fitted-state cache when checkpoint output moves.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint-dir/latest.pt if present.")
    parser.add_argument("--checkpoint-every-steps", type=int, default=25, help="Persist model/optimizer state every N batches; 0 disables mid-epoch checkpoints.")
    parser.add_argument("--milestone-gib", type=float, default=10.0, help="Run validation and save a downloadable milestone checkpoint for each GiB of training images read.")
    parser.add_argument("--pause-at-milestone", action="store_true", help="Pause after each 10-GiB milestone so the saved model can be tested locally before continuing.")
    parser.add_argument("--deployment-checkpoint", default="", help="Stable Drive path overwritten with the current full checkpoint at each milestone and after final validation.")
    parser.add_argument("--control-file", default="", help="Drive text file containing continue/pause/paused; checked after every training batch.")
    parser.add_argument("--split-mode", choices=("capture", "segment"), default="capture", help="Capture holdout for generalization audit, or segment holdout for final all-capture training.")
    parser.add_argument("--holdout-capture-token", default="HIR112", help="Capture name substring held out in capture mode; use empty string for seeded capture split.")
    parser.add_argument("--warmstart-checkpoint", default="", help="Load model weights only from a previous phase, with a fresh optimizer and split.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--batch-windows", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=8,
                        help="Parallel CPU image/MANO preparation workers; batches are prefetched while CUDA trains.")
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
    parser.add_argument("--precomputed-cache", default="", help="Optional completed CPU-rendered input bank produced by phase7_precompute_cache.py.")
    parser.add_argument("--ram-cache-shards", type=int, default=12,
                        help="Maximum decoded precomputed shards retained in CPU RAM.")
    parser.add_argument("--webcam-checkpoint-every-epochs", type=int, default=5,
                        help="Save a named full checkpoint and update the stable webcam model every N completed epochs; 0 disables.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    # Respect the explicit run configuration. The Colab runner should select
    # CUDA for training and a less frequent Drive checkpoint interval.
    return args


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


def _records_to_dicts(records):
    return [{
        "capture_id": r.capture_id, "segment": r.segment,
        "frame_idx": int(r.frame_idx), "camera_id": str(r.camera_id),
        "image_path": str(r.image_path),
        "right_keypoints_path": str(r.right_keypoints_path),
        "right_mano_path": None if r.right_mano_path is None else str(r.right_mano_path),
        "timestamp_seconds": float(r.timestamp_seconds),
    } for r in records]


def _records_from_dicts(items):
    return [data_api.ReInterHandRecord(
        capture_id=x["capture_id"], segment=x["segment"],
        frame_idx=int(x["frame_idx"]), camera_id=str(x["camera_id"]),
        image_path=Path(x["image_path"]),
        right_keypoints_path=Path(x["right_keypoints_path"]),
        right_mano_path=None if x["right_mano_path"] is None else Path(x["right_mano_path"]),
        timestamp_seconds=float(x["timestamp_seconds"]),
    ) for x in items]


def _source_signature(source: Path, camera_path: Path, args, kind: str) -> str:
    paths = [source / "frame_list.txt", camera_path] if kind == "capture" else [source, camera_path]
    stamps = []
    for path in paths:
        stat = path.stat()
        stamps.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    payload = {
        "kind": kind, "stamps": stamps, "camera_group": args.camera_group,
        "envmap_mode": args.envmap_mode, "camera_id": args.camera_id,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_record_index(path: Path, signature: str):
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("signature") == signature:
            return _records_from_dicts(saved["records"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _save_record_index(path: Path, signature: str, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps({
        "signature": signature, "records": _records_to_dicts(records),
    }, separators=(",", ":")), encoding="utf-8")
    os.replace(temp, path)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


def _report_stage_progress(name: str, done: int, total: int, started: float) -> None:
    elapsed = max(time.perf_counter() - started, 1e-6)
    rate = done / elapsed
    eta = (total - done) / max(rate, 1e-6)
    print(
        f"[{name}] {done:,}/{total:,} ({100.0 * done / max(total, 1):.1f}%) "
        f"elapsed {_format_duration(elapsed)}, ETA {_format_duration(eta)} "
        f"({rate:.1f} items/s)", flush=True,
    )


def discover_sources(args: argparse.Namespace, index_cache_dir: Path):
    manifests = [Path(x).expanduser().resolve() for x in args.manifest]
    captures = [Path(x).expanduser().resolve() for x in args.capture]
    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
        manifests.extend(sorted(root.glob("**/derived/mugsy_subset/manifest.csv")))
    if not manifests and not captures:
        local = ROOT / "datasets" / "reinterhand" / "captures"
        manifests = sorted(local.glob("*/derived/mugsy_subset/manifest.csv"))
    manifests = list(dict.fromkeys(manifests))
    captures = list(dict.fromkeys(captures))
    if not manifests and not captures:
        raise FileNotFoundError("Pass --manifest, --capture, or --data-root. No local Re:InterHand subset was found.")

    records = []
    camera_by_capture = {}
    camera_paths = {}
    stage_start = time.perf_counter()
    reused = 0
    for manifest in manifests:
        capture_root = data_api.capture_root_from_manifest(manifest)
        camera_candidates = (
            capture_root / args.camera_group / "cam_params.json",
            capture_root / "cam_params.json",
        )
        camera_path = next((path for path in camera_candidates if path.is_file()), camera_candidates[0])
        if not camera_path.is_file():
            raise FileNotFoundError(camera_path)
        with camera_path.open("r", encoding="utf-8") as stream:
            camera_by_capture[capture_root.name] = json.load(stream)
        camera_paths[capture_root.name] = camera_path
        cache_path = index_cache_dir / f"manifest_{hashlib.sha1(str(manifest.resolve()).encode()).hexdigest()[:16]}.json"
        signature = _source_signature(manifest, camera_path, args, "manifest")
        recs = _load_record_index(cache_path, signature)
        if recs is None:
            print(f"[index] Scanning manifest {manifest}; this capture index will be saved when complete.", flush=True)
            recs = data_api.records_from_subset_manifest(manifest, camera_id=args.camera_id or None)
            _save_record_index(cache_path, signature, recs)
            print(f"[index] Saved {len(recs):,} records for {capture_root.name}.", flush=True)
        else:
            reused += 1
            print(f"[index] Reused {len(recs):,} cached records for {capture_root.name}.", flush=True)
        records.extend(recs)
    for capture_root in captures:
        camera_path = capture_root / args.camera_group / "cam_params.json"
        with camera_path.open("r", encoding="utf-8") as stream:
            camera_by_capture[capture_root.name] = json.load(stream)
        camera_paths[capture_root.name] = camera_path
        cache_path = index_cache_dir / f"capture_{hashlib.sha1(str(capture_root.resolve()).encode()).hexdigest()[:16]}.json"
        signature = _source_signature(capture_root, camera_path, args, "capture")
        recs = _load_record_index(cache_path, signature)
        if recs is None:
            print(f"[index] Scanning capture {capture_root.name}; progress is reported every 2,048 candidates.", flush=True)
            scan_start = time.perf_counter()
            capture_index_key = hashlib.sha1(str(capture_root.resolve()).encode()).hexdigest()[:16]
            camera_index_dir = index_cache_dir / f"capture_{capture_index_key}_cameras"
            selected_cameras = sorted(camera_by_capture[capture_root.name])
            if args.camera_id:
                selected_cameras = [c for c in selected_cameras if str(c) == str(args.camera_id)]
            cached_camera_records = {}
            camera_signatures = {}
            for camera_id in selected_cameras:
                camera_cache = camera_index_dir / f"camera_{camera_id}.json"
                camera_signature = hashlib.sha256(f"{signature}:{camera_id}".encode()).hexdigest()
                camera_signatures[camera_id] = (camera_cache, camera_signature)
                partial_records = _load_record_index(camera_cache, camera_signature)
                if partial_records is not None:
                    cached_camera_records[camera_id] = partial_records
                    print(f"[index] Reusing cached camera {camera_id}: {len(partial_records):,} records.", flush=True)

            def report_scan(done, total, camera_id):
                elapsed = max(time.perf_counter() - scan_start, 1e-6)
                rate = done / elapsed
                eta = (total - done) / max(rate, 1e-6)
                print(
                    f"[index] {capture_root.name}: {done:,}/{total:,} candidates "
                    f"({100.0 * done / max(total, 1):.1f}%), camera {camera_id}; "
                    f"elapsed {_format_duration(elapsed)}, ETA {_format_duration(eta)}",
                    flush=True,
                )
            def save_camera_index(camera_id, camera_records):
                camera_cache, camera_signature = camera_signatures[camera_id]
                _save_record_index(camera_cache, camera_signature, camera_records)
                print(f"[index] Saved camera {camera_id} index ({len(camera_records):,} records).", flush=True)
            recs = data_api.records_from_capture(
                capture_root,
                camera_group=args.camera_group,
                envmap_mode=args.envmap_mode,
                camera_ids=[args.camera_id] if args.camera_id else None,
                progress_callback=report_scan,
                cached_camera_records=cached_camera_records,
                camera_complete_callback=save_camera_index,
            )
            _save_record_index(cache_path, signature, recs)
            print(f"[index] Saved {len(recs):,} records for {capture_root.name}.", flush=True)
        else:
            reused += 1
            print(f"[index] Reused {len(recs):,} cached records for {capture_root.name}.", flush=True)
        records.extend(recs)

    unique = {}
    for record in records:
        key = (str(record.image_path.resolve()), record.camera_id)
        unique[key] = record
    records = sorted(unique.values(), key=lambda r: (r.capture_id, r.segment, r.camera_id, r.frame_idx))
    print(
        f"[index] Complete: {len(records):,} unique records; cached sources reused={reused}; "
        f"elapsed {_format_duration(time.perf_counter() - stage_start)}.",
        flush=True,
    )
    return records, camera_by_capture, camera_paths


def group_split(records, fraction: float, seed: int, *, mode: str = "capture", holdout_token: str = "HIR112"):
    captures = sorted({r.capture_id for r in records})
    if mode == "capture" and len(captures) >= 2:
        matching = [c for c in captures if holdout_token and holdout_token.lower() in c.lower()]
        if matching:
            val_capture = set(matching)
            train_idx = np.asarray([i for i, r in enumerate(records) if r.capture_id not in val_capture], np.int64)
            val_idx = np.asarray([i for i, r in enumerate(records) if r.capture_id in val_capture], np.int64)
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise RuntimeError("Capture holdout produced an empty split")
            return train_idx, val_idx, set(captures) - val_capture, val_capture, "capture-holdout-" + "+".join(sorted(val_capture))
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
    for key in sorted(camera_paths):
        path = camera_paths[key]
        stat = path.stat()
        digest.update(f"{path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:24]


def load_or_fit_targets(records, camera_by_capture, train_ids, cache_path: Path, signature: str):
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as saved:
                if str(saved["signature"].item()) == signature and len(saved["states"]) == len(records):
                    print(f"Loaded cached HandState targets: {cache_path}")
                    return saved["camera_joints"].copy(), saved["states"].copy(), saved["residuals"].copy(), HandBiometrics(**json.loads(str(saved["biometrics_json"].item())))
        except Exception as exc:
            print(f"Target cache will be rebuilt ({exc})")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial_dir = cache_path.parent / f"partial_{signature}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 512
    chunk_total = math.ceil(len(records) / chunk_size)
    label_started = time.perf_counter()
    camera_joints = np.empty((len(records), 21, 3), np.float32)
    betas = np.empty((len(records), 10), np.float32)
    # Persist label chunks to Drive so interruption during preprocessing loses at most 512 frames.
    for start in range(0, len(records), chunk_size):
        stop = min(start + chunk_size, len(records))
        label_path = partial_dir / f"labels_{start:07d}_{stop:07d}.npz"
        if label_path.is_file():
            with np.load(label_path, allow_pickle=False) as saved:
                if str(saved["signature"].item()) == signature:
                    camera_joints[start:stop] = saved["camera_joints"]
                    betas[start:stop] = saved["betas"]
                    if (start // chunk_size + 1) % 4 == 0 or stop == len(records):
                        _report_stage_progress("labels/cache", stop, len(records), label_started)
                    continue
        chunk_joints = np.empty((stop - start, 21, 3), np.float32)
        chunk_betas = np.empty((stop - start, 10), np.float32)
        for local, i in enumerate(range(start, stop)):
            record = records[i]
            with record.right_keypoints_path.open("r", encoding="utf-8") as stream:
                world = np.asarray(json.load(stream), np.float32)
            if world.shape != (24, 3):
                raise ValueError(f"Expected 24x3 right-hand labels: {record.right_keypoints_path}")
            cam = camera_by_capture[record.capture_id][record.camera_id]
            rotation = np.asarray(cam["R"], np.float32).reshape(3, 3)
            translation = np.asarray(cam["t"], np.float32).reshape(3)
            chunk_joints[local] = world[data_api.KEEP_21_INDICES] @ rotation.T + translation
            if record.right_mano_path is not None and record.right_mano_path.is_file():
                with record.right_mano_path.open("r", encoding="utf-8") as stream:
                    mano = json.load(stream)
                chunk_betas[local] = np.asarray(mano["shape"], np.float32).reshape(10)
            else:
                chunk_betas[local] = 0.0
        tmp = label_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, signature=np.asarray(signature), camera_joints=chunk_joints, betas=chunk_betas)
        os.replace(tmp, label_path)
        camera_joints[start:stop] = chunk_joints
        betas[start:stop] = chunk_betas
        if (start // chunk_size + 1) % 4 == 0 or stop == len(records):
            print(f"Saved label cache chunk {start // chunk_size + 1}/{chunk_total} to Drive", flush=True)
            _report_stage_progress("labels", stop, len(records), label_started)

    biometrics = target_api.calibrate_biometrics_from_joints(camera_joints[train_ids])
    print(
        "Train-only shape calibration: "
        f"palm={biometrics.palm_length_ratio:.3f}, "
        f"width={biometrics.palm_width_ratio:.3f}, "
        f"finger={biometrics.finger_length_ratio:.3f}, "
        f"thumb={biometrics.thumb_ratio:.3f}"
    )
    states = np.empty((len(records), STATE_DIM), np.float32)
    residuals = np.empty(len(records), np.float32)
    state_started = time.perf_counter()
    # State-fit chunks are resumable independently; velocity features are derived after all poses exist.
    for start in range(0, len(records), chunk_size):
        stop = min(start + chunk_size, len(records))
        state_path = partial_dir / f"states_{start:07d}_{stop:07d}.npz"
        if state_path.is_file():
            with np.load(state_path, allow_pickle=False) as saved:
                if str(saved["signature"].item()) == signature:
                    states[start:stop] = saved["states"]
                    residuals[start:stop] = saved["residuals"]
                    if (start // chunk_size + 1) % 4 == 0 or stop == len(records):
                        _report_stage_progress("state-fit/cache", stop, len(records), state_started)
                    continue
        chunk_states = np.empty((stop - start, STATE_DIM), np.float32)
        chunk_residuals = np.empty(stop - start, np.float32)
        for local, i in enumerate(range(start, stop)):
            state, _, residual = target_api.fit_state_from_camera_joints(
                camera_joints[i], biometrics=biometrics, beta=betas[i], clamp_limits=True
            )
            chunk_states[local] = state.to_vector().astype(np.float32)
            chunk_residuals[local] = residual
        tmp = state_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, signature=np.asarray(signature), states=chunk_states, residuals=chunk_residuals)
        os.replace(tmp, state_path)
        states[start:stop] = chunk_states
        residuals[start:stop] = chunk_residuals
        if (start // chunk_size + 1) % 4 == 0 or stop == len(records):
            print(f"Saved fitted-state cache chunk {start // chunk_size + 1}/{chunk_total} to Drive", flush=True)
            _report_stage_progress("state-fit", stop, len(records), state_started)
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
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault((record.capture_id, record.segment, record.camera_id), []).append(i)
    max_v = float(getattr(constraints, "MAX_TRANSLATIONAL_VELOCITY", 5.0))
    max_qdot = float(getattr(constraints, "MAX_JOINT_VELOCITY", 40.0))
    for indices in groups.values():
        indices.sort(key=lambda i: records[i].frame_idx)
        for pos, idx in enumerate(indices):
            left = indices[max(0, pos - 1)]
            right = indices[min(len(indices) - 1, pos + 1)]
            dt = max(records[right].timestamp_seconds - records[left].timestamp_seconds, 1e-4)
            if right != left and records[right].frame_idx - records[left].frame_idx <= 30:
                states[idx, STATE_SLICES["V"]] = np.clip(
                    (states[right, STATE_SLICES["T"]] - states[left, STATE_SLICES["T"]]) / dt,
                    -max_v, max_v,
                )
                states[idx, STATE_SLICES["qdot"]] = np.clip(
                    (states[right, STATE_SLICES["q"]] - states[left, STATE_SLICES["q"]]) / dt,
                    -max_qdot, max_qdot,
                )
                q0 = states[left, STATE_SLICES["R"]]
                q1 = states[right, STATE_SLICES["R"]]
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


class PrecomputedSampleStore:
    """Small LRU over CPU-precomputed, compressed window shards.

    The bank contains a finite set of randomized prior/render variants. It
    keeps the expensive image decode, crop, MANO silhouette and augmentation
    out of the CUDA training loop. Shards are consumed in local window order.
    """

    def __init__(self, root: Path, signature: str, window_count: int, max_ram_shards: int = 12):
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "cache_manifest.json"
        if not manifest_path.is_file():
            candidates = []
            if self.root.is_dir():
                for child in self.root.iterdir():
                    candidate = child / "cache_manifest.json"
                    try:
                        info = json.loads(candidate.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if info.get("complete") and info.get("dataset_signature") == signature:
                        candidates.append((candidate, info))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Expected one complete matching cache under {self.root}; found {len(candidates)}"
                )
            manifest_path, self.manifest = candidates[0]
            self.root = manifest_path.parent
        else:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not self.manifest.get("complete"):
            raise RuntimeError(f"Precomputed cache is incomplete: {manifest_path}")
        if self.manifest.get("dataset_signature") != signature:
            raise RuntimeError("Precomputed cache dataset signature does not match this run.")
        if int(self.manifest.get("train_windows", -1)) != int(window_count):
            raise RuntimeError("Precomputed cache window count differs from this run.")
        self.shard_windows = int(self.manifest["shard_windows"])
        self.variants = int(self.manifest["variants"])
        self.image_size = int(self.manifest["image_size"])
        self.max_ram_shards = max(1, int(max_ram_shards))
        self._cache = OrderedDict()
        self._cache_lock = RLock()

    def locality_order(self, rng: np.random.Generator, count: int) -> np.ndarray:
        shard_count = math.ceil(count / self.shard_windows)
        shard_order = rng.permutation(shard_count)
        pieces = []
        for shard_id in shard_order:
            start = int(shard_id) * self.shard_windows
            stop = min(start + self.shard_windows, count)
            pieces.append(rng.permutation(np.arange(start, stop, dtype=np.int64)))
        return np.concatenate(pieces) if pieces else np.empty(0, np.int64)

    def _load_shard(self, shard_id: int):
        # Multiple CPU workers can request items from the same compressed NPZ
        # while the GPU trains the preceding batch. Protect the LRU and avoid
        # duplicate decompression when a shard is first touched.
        with self._cache_lock:
            if shard_id in self._cache:
                self._cache.move_to_end(shard_id)
                return self._cache[shard_id]
            path = self.root / f"train_shard_{shard_id:05d}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Precomputed input shard missing: {path}")
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["dataset_signature"].item()) != self.manifest["dataset_signature"]:
                    raise RuntimeError(f"Stale/corrupt precomputed shard: {path}")
                data = {key: saved[key].copy() for key in ("x", "states", "deltas", "valid", "dt", "gate")}
            self._cache[shard_id] = data
            while len(self._cache) > self.max_ram_shards:
                self._cache.popitem(last=False)
            return data

    def get(self, window_id: int, variant: int):
        shard_id = int(window_id) // self.shard_windows
        local_id = int(window_id) - shard_id * self.shard_windows
        data = self._load_shard(shard_id)
        if local_id >= len(data["x"]):
            raise IndexError(f"Window {window_id} is outside precomputed shard {shard_id}")
        variant = int(variant) % self.variants
        return (
            data["x"][local_id, variant].astype(np.float32) / 255.0,
            data["states"][local_id, variant].astype(np.float32, copy=False),
            data["deltas"][local_id, variant].astype(np.float32, copy=False),
            data["valid"][local_id, variant].astype(np.float32, copy=False),
            data["dt"][local_id, variant].astype(np.float32, copy=False),
            data["gate"][local_id, variant].astype(np.float32, copy=False),
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
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temp)
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    try:
        os.replace(temp, path)
    except OSError:
        # Google Drive's FUSE mount can reject atomic rename with ENOENT even
        # while ordinary copies work. Retain the last checkpoint if rename fails,
        # then copy the fully written temporary checkpoint into place.
        if temp.is_file():
            shutil.copy2(temp, path)
            temp.unlink(missing_ok=True)
        elif not path.is_file():
            raise


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
    if not windows:
        return {"prior_mpjpe_mm": float("nan"), "corrected_mpjpe_mm": float("nan"), "authority": float("nan")}
    all_prior, all_corr, all_target = [], [], []
    all_auth = []
    with torch.inference_mode():
        for start in range(0, len(windows), 4):
            # Build only one small validation batch at a time; full-capture evaluation
            # must not retain every crop and MANO render in RAM.
            rng = np.random.default_rng(SEED + 901 + start)
            window_batch = windows[start:start + 4]
            batch = [build_sample(w, records, states, renderers, rng, train=False, image_size=image_size) for w in window_batch]
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
                    frame_index = int(window_batch[b][t])
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
    if args.epochs < 1 or args.window_size < 2 or args.batch_windows < 1 or args.milestone_gib <= 0:
        raise ValueError("epochs, batch-windows, and milestone-gib must be positive; window-size must be >=2")
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

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    index_cache_dir = Path(args.index_cache_dir).expanduser().resolve() if args.index_cache_dir else checkpoint_dir / "record_index"
    records, camera_by_capture, camera_paths = discover_sources(args, index_cache_dir)
    print(f"[dataset] records={len(records):,}; captures={len(camera_by_capture)}; split={args.split_mode}", flush=True)
    train_idx, val_idx, train_groups, val_groups, split_kind = group_split(
        records, args.validation_fraction, args.split_seed,
        mode=args.split_mode, holdout_token=args.holdout_capture_token,
    )
    base_signature = dataset_signature(records, camera_paths, args.split_seed)
    split_hash = hashlib.sha256(
        base_signature.encode() + split_kind.encode()
        + np.asarray(train_idx, np.int64).tobytes() + np.asarray(val_idx, np.int64).tobytes()
    ).hexdigest()[:24]
    signature = split_hash
    print_resource_status(device, checkpoint_dir)
    target_cache = Path(args.target_cache_path).expanduser().resolve() if args.target_cache_path else checkpoint_dir / "target_cache" / f"states_{signature}.npz"
    print(f"[targets] Loading/building camera labels and fitted HandState targets at {target_cache}", flush=True)
    camera_joints, target_states, fit_residuals, biometrics = load_or_fit_targets(
        records, camera_by_capture, train_idx, target_cache, signature
    )
    train_windows = build_windows(records, train_idx, args)
    val_windows_full = build_windows(records, val_idx, args)
    val_windows = val_windows_full
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
    counts_by_capture = {}
    for record in records:
        counts_by_capture[record.capture_id] = counts_by_capture.get(record.capture_id, 0) + 1
    print(
        f"[ready] split={split_kind}; train_frames={len(train_idx):,}; val_frames={len(val_idx):,}; "
        f"train_windows={len(train_windows):,}; val_windows={len(val_windows):,}; "
        f"epochs={args.epochs}; records_by_capture={counts_by_capture}",
        flush=True,
    )

    sample_cache = None
    if args.precomputed_cache:
        sample_cache = PrecomputedSampleStore(
            Path(args.precomputed_cache), signature, len(train_windows),
            max_ram_shards=args.ram_cache_shards,
        )
        if int(sample_cache.manifest.get("window_size", -1)) != args.window_size:
            raise RuntimeError("Precomputed cache window size differs from this training run.")
        print(
            f"[inputs] Using CPU-precomputed bank: {sample_cache.variants} variants, "
            f"{sample_cache.shard_windows}-window shards at {sample_cache.root}; "
            f"RAM LRU cap={sample_cache.max_ram_shards} decoded shards", flush=True,
        )

    mano_path = ROOT / "models" / "MANO_RIGHT.pkl"
    if not mano_path.is_file():
        raise FileNotFoundError(f"MANO surface asset not found: {mano_path}")
    renderers = make_renderers(records, camera_by_capture, biometrics, args.image_size, mano_path)
    model = TemporalStateCorrector(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if device.type == "cuda":
        print(
            f"[ready] model on {torch.cuda.get_device_name(device)}; "
            f"GPU memory allocated={torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB; "
            f"AMP={'on' if amp_enabled else 'off'}",
            flush=True,
        )
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    control_path = Path(args.control_file).expanduser().resolve() if args.control_file else None
    if control_path is not None:
        control_path.parent.mkdir(parents=True, exist_ok=True)
        if not control_path.exists():
            control_path.write_text("continue\n", encoding="utf-8")

    def read_training_command():
        if control_path is None or not control_path.exists():
            return "continue"
        command = control_path.read_text(encoding="utf-8").strip().lower()
        if command in ("", "continue"):
            return "continue"
        if command in ("pause", "paused"):
            return command
        raise ValueError(f"Unknown training command in {control_path}: {command!r}; use continue or pause")

    def write_training_command(command):
        if control_path is None:
            return
        temp_command = control_path.with_suffix(control_path.suffix + ".tmp")
        with temp_command.open("w", encoding="utf-8") as stream:
            stream.write(command + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_command, control_path)

    initial_command = read_training_command()
    if initial_command == "pause":
        raise RuntimeError(f"Pause command is set in {control_path}. Change it to 'continue' before resuming.")
    if initial_command == "paused":
        raise RuntimeError(f"Training is paused. Change {control_path} to 'continue', then rerun this same phase to resume latest.pt.")
    start_epoch = 1
    best_val = float("inf")
    bytes_processed = 0
    milestone_index = 1
    milestone_bytes = max(1, int(args.milestone_gib * 1024**3))
    next_milestone_bytes = milestone_bytes
    resume_order = None
    resume_offset = 0
    resume_augment_state = None
    if args.resume and latest_path.is_file():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
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
        bytes_processed = int(saved.get("train_image_bytes_read", 0))
        milestone_index = int(saved.get("next_milestone_index", 1))
        next_milestone_bytes = int(saved.get("next_milestone_bytes", milestone_bytes))
        if "torch_rng" in saved:
            cpu_rng = saved["torch_rng"]
            if isinstance(cpu_rng, torch.Tensor):
                cpu_rng = cpu_rng.detach().to(device="cpu", dtype=torch.uint8)
            torch.set_rng_state(cpu_rng)
        if device.type == "cuda" and saved.get("cuda_rng"):
            cuda_rng = [
                state.detach().to(device="cpu", dtype=torch.uint8)
                if isinstance(state, torch.Tensor) else state
                for state in saved["cuda_rng"]
            ]
            torch.cuda.set_rng_state_all(cuda_rng)
        if "numpy_rng" in saved:
            np.random.set_state(saved["numpy_rng"])
        if resume_order is not None:
            print(f"Resuming epoch {start_epoch} at batch offset {resume_offset}/{len(resume_order)} from {latest_path}")
        else:
            print(f"Resumed after epoch {start_epoch - 1} from {latest_path}")
    elif args.warmstart_checkpoint:
        warm = torch.load(args.warmstart_checkpoint, map_location=device, weights_only=False)
        if "model" not in warm:
            raise RuntimeError("Warmstart file has no model weights")
        model.load_state_dict(warm["model"])
        print(f"Warm-started model weights from {args.warmstart_checkpoint}; optimizer and scheduler start fresh")

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
            "mean_shape_beta_train": target_states[train_idx, STATE_SLICES["beta"]].mean(axis=0).astype(np.float32),
            "camera_by_capture": camera_by_capture,
            "config": vars(args),
            "train_image_bytes_read": int(bytes_processed),
            "next_milestone_index": int(milestone_index),
            "next_milestone_bytes": int(next_milestone_bytes),
            "input_contract": "RGB + rendered MANO silhouette + projected FK skeleton; NCHW 5 channels",
            "output_contract": "delta_T meters, delta_R axis-angle radians, delta_q radians, log_sigma, authority",
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "numpy_rng": np.random.get_state(),
        }

    training_started = time.perf_counter()
    print(f"[train] Starting epoch {start_epoch}/{args.epochs}.", flush=True)
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
            rng = np.random.default_rng(SEED + epoch * 1009)
            if sample_cache is not None:
                order = sample_cache.locality_order(rng, len(train_windows))
            else:
                order = np.arange(len(train_windows))
                rng.shuffle(order)
            offset_start = 0
        running_loss, steps = 0.0, 0
        def submit_samples(pool, chosen):
            # Give each window a private RNG so threads never race on a shared
            # NumPy generator. Saving the state from before the next batch is
            # queued makes an interrupted run reproducible on resume.
            rng_before = copy.deepcopy(rng.bit_generator.state)
            if sample_cache is not None:
                variants = rng.integers(0, sample_cache.variants, size=len(chosen))
                # Decode/decompress the next batch in CPU workers while the
                # current batch is running on CUDA. This makes the RAM buffer
                # a real prefetch queue instead of loading synchronously.
                futures = [
                    pool.submit(sample_cache.get, int(window_id), int(variant))
                    for window_id, variant in zip(chosen, variants)
                ]
                return futures, rng_before
            seeds = rng.integers(0, np.iinfo(np.uint64).max,
                                 size=len(chosen), dtype=np.uint64)
            futures = [
                pool.submit(
                    build_sample, train_windows[int(window_id)], records,
                    target_states, renderers,
                    np.random.default_rng(int(seed)),
                    train=True, image_size=args.image_size,
                )
                for window_id, seed in zip(chosen, seeds)
            ]
            return futures, rng_before

        # CPU workers overlap image reads, MANO input rendering and augmentation
        # for batch N+1 with the CUDA forward/backward/optimizer step for batch N.
        # This keeps the T4 fed without loading the image corpus into RAM.
        with ThreadPoolExecutor(max_workers=max(1, args.loader_workers)) as sample_pool:
            pending = None
            for offset in range(offset_start, len(order), args.batch_windows):
                chosen = order[offset:offset + args.batch_windows]
                if pending is None:
                    pending = submit_samples(sample_pool, chosen)
                samples = [future.result() for future in pending[0]]
                next_offset = offset + len(chosen)
                next_pending = None
                checkpoint_rng_state = copy.deepcopy(rng.bit_generator.state)
                if next_offset < len(order):
                    next_chosen = order[next_offset:next_offset + args.batch_windows]
                    next_pending = submit_samples(sample_pool, next_chosen)
                    checkpoint_rng_state = next_pending[1]
                if sample_cache is not None:
                    batch_image_bytes = sum(int(sample[0].nbytes) for sample in samples)
                else:
                    batch_image_bytes = sum(
                        records[int(frame_id)].image_path.stat().st_size
                        for window_id in chosen
                        for frame_id in train_windows[int(window_id)]
                    )
                bx = torch.from_numpy(np.stack([s[0] for s in samples]))
                state_np = np.stack([s[1] for s in samples])
                states_t = torch.from_numpy(state_np)
                target_delta = torch.from_numpy(np.stack([s[2] for s in samples]))
                frame_valid = torch.from_numpy(np.stack([s[3] for s in samples]))
                features = batch_features(states_t, device)
                dt = torch.from_numpy(np.stack([s[4] for s in samples]))
                authority_target = torch.from_numpy(np.stack([s[5] for s in samples]))
                if device.type == "cuda":
                    bx = bx.pin_memory().to(device, non_blocking=True)
                    states_t = states_t.pin_memory().to(device, non_blocking=True)
                    target_delta = target_delta.pin_memory().to(device, non_blocking=True)
                    frame_valid = frame_valid.pin_memory().to(device, non_blocking=True)
                    dt = dt.pin_memory().to(device, non_blocking=True)
                    authority_target = authority_target.pin_memory().to(device, non_blocking=True)
                else:
                    bx = bx.to(device)
                    states_t = states_t.to(device)
                    target_delta = target_delta.to(device)
                    frame_valid = frame_valid.to(device)
                    dt = dt.to(device)
                    authority_target = authority_target.to(device)
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
                    # BCE on probabilities is not allowed inside autocast. Keep
                    # the model forward and main losses on CUDA/AMP, while doing
                    # this small gate loss in FP32 for stable AMP training.
                    with torch.autocast(device_type=device.type, enabled=False):
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
                total_batches = math.ceil(len(order) / args.batch_windows)
                if batch_number % 10 == 0 or batch_number == 1 or batch_number == total_batches:
                    elapsed_batch_stage = time.perf_counter() - epoch_start
                    first_batch = offset_start // args.batch_windows + 1
                    observed_batches = max(batch_number - first_batch + 1, 1)
                    avg_batch_seconds = elapsed_batch_stage / observed_batches
                    remaining_current = max(total_batches - batch_number, 0)
                    approximate_epoch = avg_batch_seconds * total_batches
                    remaining_seconds = (
                        avg_batch_seconds * remaining_current
                        + approximate_epoch * max(args.epochs - epoch, 0)
                    )
                    gpu_mem = (
                        f" GPU={torch.cuda.memory_allocated(device) / 1024**3:.2f}GiB"
                        if device.type == "cuda" else ""
                    )
                    print(
                        f"[train] epoch {epoch:02d}/{args.epochs} step {batch_number:04d}/{total_batches:04d} "
                        f"({100.0 * batch_number / max(total_batches, 1):.1f}%) "
                        f"loss={running_loss / max(steps, 1):.4f}; elapsed={_format_duration(elapsed_batch_stage)}; "
                        f"rough ETA={_format_duration(remaining_seconds)}{gpu_mem}",
                        flush=True,
                    )
                next_offset = offset + len(chosen)
                bytes_processed += int(batch_image_bytes)
                if args.checkpoint_every_steps > 0 and batch_number % args.checkpoint_every_steps == 0:
                    payload = checkpoint_payload(
                        epoch,
                        epoch_complete=False,
                        next_offset=next_offset,
                        order=order,
                        augment_rng_state=checkpoint_rng_state,
                        best_score=best_val,
                    )
                    _save_checkpoint(latest_path, payload)
                    print(f"Saved mid-epoch recovery checkpoint at step {batch_number}")
                command = read_training_command()
                if command == "pause":
                    payload = checkpoint_payload(
                        epoch,
                        epoch_complete=False,
                        next_offset=next_offset,
                        order=order,
                        augment_rng_state=checkpoint_rng_state,
                        best_score=best_val,
                    )
                    _save_checkpoint(latest_path, payload)
                    pause_path = checkpoint_dir / f"paused_epoch_{epoch:03d}_step_{batch_number:05d}.pt"
                    _save_checkpoint(pause_path, payload)
                    if args.deployment_checkpoint:
                        _save_checkpoint(Path(args.deployment_checkpoint), payload)
                    write_training_command("paused")
                    event = {
                        "event": "manual_pause",
                        "epoch": int(epoch),
                        "step": int(batch_number),
                        "next_offset": int(next_offset),
                        "checkpoint": str(pause_path),
                        "train_image_bytes_read": int(bytes_processed),
                    }
                    with (checkpoint_dir / "commands.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(event) + "\n")
                        log.flush()
                        os.fsync(log.fileno())
                    print(f"PAUSE SAVED after epoch {epoch}, step {batch_number}. Recovery: {latest_path}")
                    print(f"Stable local-test model updated: {args.deployment_checkpoint or '(not configured)'}")
                    print(f"Set {control_path} to 'continue' and rerun this same phase to resume.")
                    return
                while bytes_processed >= next_milestone_bytes:
                    milestone_eval = run_eval(
                        model, val_windows, records, target_states, camera_joints,
                        renderers, device, args.image_size,
                    )
                    improved = milestone_eval["corrected_mpjpe_mm"] < best_val
                    best_val = min(best_val, milestone_eval["corrected_mpjpe_mm"])
                    milestone_path = checkpoint_dir / f"milestone_{milestone_index:03d}_{bytes_processed / 1024**3:.1f}GiB.pt"
                    milestone_index += 1
                    next_milestone_bytes += milestone_bytes
                    payload = checkpoint_payload(
                        epoch, epoch_complete=False, next_offset=next_offset, order=order,
                        augment_rng_state=checkpoint_rng_state,
                        validation=milestone_eval, best_score=best_val,
                    )
                    _save_checkpoint(latest_path, payload)
                    _save_checkpoint(milestone_path, payload)
                    if args.deployment_checkpoint:
                        _save_checkpoint(Path(args.deployment_checkpoint), payload)
                    if improved:
                        _save_checkpoint(best_path, payload)
                    row = {
                        "milestone_gib": round(bytes_processed / 1024**3, 3),
                        "epoch": epoch, "step": batch_number,
                        "train_image_bytes_read": int(bytes_processed),
                        "validation": milestone_eval,
                        "checkpoint": str(milestone_path),
                    }
                    with (checkpoint_dir / "milestones.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        os.fsync(log.fileno())
                    print(
                        f"10-GiB milestone {milestone_index - 1}: "
                        f"read={bytes_processed / 1024**3:.2f} GiB; "
                        f"val prior={milestone_eval['prior_mpjpe_mm']:.2f} mm; "
                        f"corrected={milestone_eval['corrected_mpjpe_mm']:.2f} mm; "
                        f"saved={milestone_path}"
                    )
                    if args.pause_at_milestone:
                        print(f"Milestone saved. To pause now, set {control_path} to 'pause'; the trainer checks after each batch.")
                pending = next_pending
        scheduler.step()
        resume_order = None
        resume_offset = 0
        resume_augment_state = None
        val = run_eval(model, val_windows, records, target_states, camera_joints, renderers, device, args.image_size)
        elapsed = time.perf_counter() - epoch_start
        average_epoch = (time.perf_counter() - training_started) / max(epoch - start_epoch + 1, 1)
        remaining_epochs_eta = average_epoch * max(args.epochs - epoch, 0)
        print(
            f"epoch {epoch:02d}/{args.epochs}: loss={running_loss/max(steps,1):.4f}; "
            f"val prior={val['prior_mpjpe_mm']:.2f}mm corrected={val['corrected_mpjpe_mm']:.2f}mm "
            f"authority={val['authority']:.3f}; {val.get('frames', 0)} frames; "
            f"elapsed={_format_duration(elapsed)}; epochs left={args.epochs - epoch}; "
            f"rough remaining={_format_duration(remaining_epochs_eta)}",
            flush=True,
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
        if args.webcam_checkpoint_every_epochs > 0 and epoch % args.webcam_checkpoint_every_epochs == 0:
            webcam_path = checkpoint_dir / f"webcam_epoch_{epoch:03d}.pt"
            _save_checkpoint(webcam_path, payload)
            if args.deployment_checkpoint:
                _save_checkpoint(Path(args.deployment_checkpoint), payload)
            print(
                f"Saved webcam-test checkpoint at epoch {epoch}: {webcam_path}; "
                f"stable endpoint={args.deployment_checkpoint or '(not configured)'}",
                flush=True,
            )

    if best_path.is_file():
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model"])
        full_validation = run_eval(
            model, val_windows_full, records, target_states, camera_joints,
            renderers, device, args.image_size,
        )
        full_validation.update({
            "checkpoint": str(best_path),
            "dataset_signature": signature,
            "split_kind": split_kind,
            "full_validation_windows": len(val_windows_full),
            "full_validation_frames": int(full_validation.get("frames", 0)),
        })
        report_path = checkpoint_dir / "full_validation.json"
        temp_report = report_path.with_suffix(".tmp")
        temp_report.write_text(json.dumps(full_validation, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_report, report_path)
        if args.deployment_checkpoint:
            deploy_payload = dict(best_payload)
            deploy_payload["full_validation"] = full_validation
            _save_checkpoint(Path(args.deployment_checkpoint), deploy_payload)
        print(f"Full validation saved: {report_path}; corrected={full_validation['corrected_mpjpe_mm']:.2f} mm")
    print(f"Best validation checkpoint: {best_path} ({best_val:.2f} mm on milestone/epoch validation windows)")
    print("The saved score is a prototype metric against the current HandState/FK adapter; it is not yet a live-camera result.")


if __name__ == "__main__":
    main()
