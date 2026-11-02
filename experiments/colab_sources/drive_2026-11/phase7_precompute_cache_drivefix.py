#!/usr/bin/env python3
"""Build a resumable CPU-rendered input bank for Phase 7 training.

This deliberately precomputes a finite bank of prior/noise/occlusion variants.
The original trainer creates a new random prior and MANO render every time a
window is visited; a finite cache removes that repeated CPU work at the cost of
reusing one of N pre-rendered variants across epochs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch


def fmt(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    try:
        os.replace(temp, path)
    except OSError:
        shutil.copy2(temp, path)
        temp.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays) -> None:
    temp = path.with_name(path.stem + ".tmp.npz")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temp, path)
    except OSError:
        shutil.copy2(temp, path)
        temp.unlink(missing_ok=True)


def import_trainer(path: Path):
    root = path.resolve().parents[1]
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("phase7_trainer_for_precompute", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load trainer source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args_for_trainer(args):
    return SimpleNamespace(
        manifest=args.manifest,
        capture=args.capture,
        data_root=args.data_root,
        camera_group=args.camera_group,
        envmap_mode=args.envmap_mode,
        camera_id=args.camera_id,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
        validation_fraction=args.validation_fraction,
        holdout_capture_token=args.holdout_capture_token,
        window_size=args.window_size,
        stride=args.stride,
        max_frame_gap=args.max_frame_gap,
        val_windows=args.val_windows,
        max_train_windows=0,
        max_val_windows=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", required=True, type=Path)
    parser.add_argument("--capture", action="append", default=[])
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--data-root", default="")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--camera-group", default="Mugsy_cameras")
    parser.add_argument("--envmap-mode", default="envmap_per_segment")
    parser.add_argument("--camera-id", default="")
    parser.add_argument("--split-mode", choices=("capture", "segment"), default="capture")
    parser.add_argument("--holdout-capture-token", default="HIR112")
    parser.add_argument("--split-seed", type=int, default=20251018)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--max-frame-gap", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--val-windows", type=int, default=32)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--shard-windows", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-shards", type=int, default=0, help="Optional resumable smoke cap; 0 means all shards.")
    args = parser.parse_args()

    if not args.capture and not args.manifest and not args.data_root:
        raise ValueError("Supply --capture, --manifest, or --data-root")
    if args.variants < 1 or args.shard_windows < 1 or args.workers < 1:
        raise ValueError("variants, shard-windows, and workers must be positive")
    torch.set_num_threads(max(1, min(4, args.workers)))
    print(f"CPU precompute only; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)

    trainer_path = args.trainer.resolve()
    trainer = import_trainer(trainer_path)
    ta = args_for_trainer(args)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    records, camera_by_capture, camera_paths = trainer.discover_sources(
        ta, checkpoint_dir / "record_index"
    )
    train_idx, val_idx, _, _, split_kind = trainer.group_split(
        records, ta.validation_fraction, ta.split_seed,
        mode=ta.split_mode, holdout_token=ta.holdout_capture_token,
    )
    base_signature = trainer.dataset_signature(records, camera_paths, ta.split_seed)
    split_hash = hashlib.sha256(
        base_signature.encode() + split_kind.encode()
        + np.asarray(train_idx, np.int64).tobytes()
        + np.asarray(val_idx, np.int64).tobytes()
    ).hexdigest()[:24]

    target_path = checkpoint_dir / "target_cache" / f"states_{split_hash}.npz"
    print(f"[targets] Reusing/building the same fitted-state target cache: {target_path}", flush=True)
    camera_joints, states, residuals, biometrics = trainer.load_or_fit_targets(
        records, camera_by_capture, train_idx, target_path, split_hash
    )
    del camera_joints, residuals

    train_windows = trainer.build_windows(records, train_idx, ta)
    if not train_windows:
        raise RuntimeError("No training windows were generated")
    renderers = trainer.make_renderers(
        records, camera_by_capture, biometrics, args.image_size,
        trainer.ROOT / "models" / "MANO_RIGHT.pkl",
    )

    signature_payload = {
        "dataset_signature": split_hash,
        "window_size": args.window_size,
        "stride": args.stride,
        "max_frame_gap": args.max_frame_gap,
        "image_size": args.image_size,
        "variants": args.variants,
        "shard_windows": args.shard_windows,
        "format": "phase7-render-bank-v1",
    }
    cache_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()[:24]
    final_dir = cache_dir / cache_signature
    final_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = final_dir / "cache_manifest.json"
    shard_count = math.ceil(len(train_windows) / args.shard_windows)
    manifest = {
        **signature_payload,
        "cache_signature": cache_signature,
        "train_windows": len(train_windows),
        "shard_count": shard_count,
        "complete": False,
        "completed_shards": [],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("cache_signature") != cache_signature:
                raise RuntimeError(f"Cache directory signature collision: {manifest_path}")
            manifest.update(old)
            manifest["complete"] = False
        except (OSError, ValueError):
            pass
    atomic_json(manifest_path, manifest)

    started = time.perf_counter()
    # The manifest is atomically updated only after atomic_npz has completed.
    # Reopening every completed NPZ through Drive FUSE creates hundreds of
    # blocking metadata/read requests before any progress is printed. Trust the
    # durable completed_shards list here; the dataset signature above prevents
    # reusing shards from a different preparation run.
    completed = {
        int(shard_id)
        for shard_id in manifest.get("completed_shards", [])
        if 0 <= int(shard_id) < shard_count
    }
    print(
        f"[cache] Resume manifest accepted {len(completed):,}/{shard_count:,} "
        "completed shards; no completed shard payloads will be reread.",
        flush=True,
    )
    todo = [i for i in range(shard_count) if i not in completed]
    if args.max_shards:
        todo = todo[:args.max_shards]
    print(
        f"[cache] records={len(records):,}; train windows={len(train_windows):,}; "
        f"variants/window={args.variants}; shards={shard_count:,}; "
        f"already complete={len(completed):,}; this pass={len(todo):,}; output={final_dir}",
        flush=True,
    )

    def generate(window_id: int, variant_id: int):
        seed = (args.split_seed + 0x1F123BB5 * (variant_id + 1) + window_id * 0x45D9F3B) & ((1 << 63) - 1)
        rng = np.random.default_rng(seed)
        return trainer.build_sample(
            train_windows[window_id], records, states, renderers, rng,
            train=True, image_size=args.image_size,
        )

    image_stage_root = Path("/content/phase7_image_stage")
    image_stage_root.mkdir(parents=True, exist_ok=True)

    for done_pos, shard_id in enumerate(todo, 1):
        start = shard_id * args.shard_windows
        stop = min(start + args.shard_windows, len(train_windows))
        shard_path = final_dir / f"train_shard_{shard_id:05d}.npz"
        n = stop - start
        x = np.empty((n, args.variants, args.window_size, 5, args.image_size, args.image_size), np.uint8)
        state = np.empty((n, args.variants, args.window_size, trainer.STATE_DIM), np.float32)
        delta = np.empty((n, args.variants, args.window_size, 27), np.float32)
        valid = np.empty((n, args.variants, args.window_size), np.float32)
        dt = np.empty_like(valid)
        gate = np.empty_like(valid)
        jobs = {}
        # Google Drive's FUSE mount can leave concurrent cv2.imread calls in
        # uninterruptible folio_wait_bit_common. Stage this shard's unique RGB
        # frames sequentially to Colab's local disk, then let workers touch
        # only local files while rendering the 4 variants.
        shard_stage = image_stage_root / f"shard_{shard_id:05d}"
        shutil.rmtree(shard_stage, ignore_errors=True)
        shard_stage.mkdir(parents=True, exist_ok=True)
        frame_ids = sorted({int(frame_id) for window_id in range(start, stop)
                            for frame_id in train_windows[window_id]})
        original_records = {}
        print(f"[stage] shard {shard_id + 1}/{shard_count}: copying {len(frame_ids)} unique frames sequentially from Drive", flush=True)
        stage_started = time.perf_counter()
        try:
            for pos, frame_id in enumerate(frame_ids, 1):
                record = records[frame_id]
                source = Path(record.image_path)
                suffix = source.suffix or ".jpg"
                staged_path = shard_stage / f"frame_{frame_id:06d}{suffix}"
                shutil.copyfile(source, staged_path)
                original_records[frame_id] = record
                records[frame_id] = SimpleNamespace(
                    capture_id=record.capture_id, segment=record.segment,
                    frame_idx=record.frame_idx, camera_id=record.camera_id,
                    image_path=staged_path,
                    right_keypoints_path=record.right_keypoints_path,
                    right_mano_path=record.right_mano_path,
                    timestamp_seconds=record.timestamp_seconds,
                )
                if pos % 32 == 0 or pos == len(frame_ids):
                    print(f"[stage] shard {shard_id + 1}: {pos}/{len(frame_ids)} frames copied; elapsed={fmt(time.perf_counter() - stage_started)}", flush=True)

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for local_id, window_id in enumerate(range(start, stop)):
                    for variant_id in range(args.variants):
                        future = pool.submit(generate, window_id, variant_id)
                        jobs[future] = (local_id, variant_id)
                for future in as_completed(jobs):
                    local_id, variant_id = jobs[future]
                    sample = future.result()
                    x[local_id, variant_id] = np.rint(np.clip(sample[0], 0, 1) * 255).astype(np.uint8)
                    state[local_id, variant_id] = sample[1]
                    delta[local_id, variant_id] = sample[2]
                    valid[local_id, variant_id] = sample[3]
                    dt[local_id, variant_id] = sample[4]
                    gate[local_id, variant_id] = sample[5]
        finally:
            for frame_id, record in original_records.items():
                records[frame_id] = record
            shutil.rmtree(shard_stage, ignore_errors=True)

        atomic_npz(
            shard_path,
            dataset_signature=np.asarray(split_hash),
            cache_signature=np.asarray(cache_signature),
            window_start=np.asarray(start, np.int64),
            window_stop=np.asarray(stop, np.int64),
            x=x, states=state, deltas=delta, valid=valid, dt=dt, gate=gate,
        )
        completed.add(shard_id)
        manifest["completed_shards"] = sorted(completed)
        manifest["complete"] = len(completed) == shard_count
        atomic_json(manifest_path, manifest)
        total_done = sum(min(args.shard_windows, len(train_windows) - i * args.shard_windows) for i in completed)
        elapsed = time.perf_counter() - started
        rate = total_done / max(elapsed, 1e-6)
        eta = (len(train_windows) - total_done) / max(rate, 1e-6)
        print(
            f"[cache] shard {shard_id + 1}/{shard_count} saved; "
            f"windows={total_done:,}/{len(train_windows):,} ({100*total_done/len(train_windows):.1f}%); "
            f"elapsed={fmt(elapsed)}; ETA={fmt(eta)}; file={shard_path.name}",
            flush=True,
        )

    if args.max_shards and len(completed) < shard_count:
        print(f"[cache] Partial run saved and resumable. Completed {len(completed)}/{shard_count} shards.", flush=True)
    elif len(completed) == shard_count:
        manifest["complete"] = True
        atomic_json(manifest_path, manifest)
        print(f"[cache] COMPLETE: {len(train_windows):,} windows x {args.variants} variants. Training may now use --precomputed-cache {final_dir}", flush=True)


if __name__ == "__main__":
    main()
