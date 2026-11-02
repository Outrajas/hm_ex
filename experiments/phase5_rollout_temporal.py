#!/usr/bin/env python
"""Causal free-running validation of the persistent Phase 5 hand tracker.

The first validation pose is used only to simulate a manual/bootstrap seed;
all later poses are produced by the tracker from RGB frames and its own state.
No validation labels are read by the tracker after that seed.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
trainer = importlib.import_module("experiments.phase5_train_temporal")
runtime = importlib.import_module("08_runtime.tracker")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--capture", action="append", default=[])
    parser.add_argument("--data-root", default="")
    parser.add_argument("--camera-group", default="Mugsy_cameras")
    parser.add_argument("--envmap-mode", default="envmap_per_segment")
    parser.add_argument("--camera-id", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def validation_runs(records, val_ids, max_gap):
    groups = defaultdict(list)
    for index in val_ids:
        record = records[int(index)]
        groups[(record.capture_id, record.segment, record.camera_id)].append(int(index))
    runs = []
    for key, indices in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        indices.sort(key=lambda i: records[i].frame_idx)
        run = []
        previous = None
        for index in indices:
            frame = records[index].frame_idx
            if previous is not None and frame - previous > max_gap:
                if len(run) >= 2:
                    runs.append((key, run))
                run = []
            run.append(index)
            previous = frame
        if len(run) >= 2:
            runs.append((key, run))
    return runs


def perturbed_seed(target_state):
    state = target_state.copy()
    state.T += np.asarray([0.015, -0.010, 0.020], np.float64)
    rotvec = np.deg2rad(np.asarray([5.0, -7.0, 4.0], np.float64))
    state.R = trainer.quat_mul(
        trainer.coords.rotation_vector_to_quaternion(rotvec), state.R
    )
    q_noise = np.deg2rad(8.0) * np.sin(np.arange(21, dtype=np.float64) * 1.7)
    state.q = np.clip(
        state.q + q_noise,
        trainer.constraints.JOINT_LIMITS_LOWER,
        trainer.constraints.JOINT_LIMITS_UPPER,
    )
    # The seed is a one-time manual/bootstrap estimate, so no target-derived
    # velocity is carried into the rollout.
    state.V[:] = 0.0
    state.omega[:] = 0.0
    state.qdot[:] = 0.0
    return state


def metrics(state, target_joints_mm, biometrics):
    joints = trainer.forward_kinematics(state, biometrics)[0] * 1000.0
    pred_rel = joints - joints[0:1]
    target_rel = target_joints_mm - target_joints_mm[0:1]
    d_rel = np.linalg.norm(pred_rel - target_rel, axis=-1)
    d_cam = np.linalg.norm(joints - target_joints_mm, axis=-1)
    tips = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
    return np.asarray([
        d_rel.mean(), d_rel[tips].mean(), d_cam.mean(), d_cam[0],
    ], np.float64)


def update_metrics(bucket, row):
    bucket.append(row)


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = trainer.choose_device(args.device)
    discovery_args = argparse.Namespace(
        manifest=args.manifest, capture=args.capture, data_root=args.data_root,
        camera_group=args.camera_group, envmap_mode=args.envmap_mode,
        camera_id=args.camera_id,
    )
    records, camera_by_capture, camera_paths = trainer.discover_sources(discovery_args)
    train_ids, val_ids, *_ = trainer.group_split(
        records, float(config["validation_fraction"]), int(config["split_seed"])
    )
    signature = trainer.dataset_signature(records, camera_paths, int(config["split_seed"]))
    if signature != checkpoint.get("dataset_signature"):
        raise RuntimeError("Checkpoint dataset signature does not match these records/cameras.")
    cache = checkpoint_path.parent / "target_cache" / f"states_{signature}.npz"
    camera_joints, target_states, residuals, biometrics = trainer.load_or_fit_targets(
        records, camera_by_capture, train_ids, cache, signature
    )

    model = trainer.TemporalStateCorrector(hidden_dim=int(config["hidden_dim"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    trackers_api = runtime.PersistentHandTracker
    renderers = trainer.make_renderers(
        records, camera_by_capture, biometrics,
        int(config["image_size"]), ROOT / "models" / "MANO_RIGHT.pkl",
    )
    runs = validation_runs(records, val_ids, int(config["max_frame_gap"]))
    full_by_segment, blind_by_segment, dropout_by_segment = defaultdict(list), defaultdict(list), defaultdict(list)
    frame_count = 0

    for run_no, (group_key, run) in enumerate(runs, 1):
        segment = str(group_key[1])
        seed = perturbed_seed(trainer.HandState.from_vector(target_states[run[0]].astype(np.float64)))
        visual = trackers_api(model, seed)
        blind = trackers_api(model, seed)
        dropout = trackers_api(model, seed)
        first_gt = camera_joints[run[0]]
        initial = metrics(seed, first_gt, biometrics)
        update_metrics(full_by_segment[segment], initial)
        update_metrics(blind_by_segment[segment], initial)
        update_metrics(dropout_by_segment[segment], initial)
        run_len = len(run)
        dropout_start = max(1, int(run_len * 0.40))
        dropout_end = min(run_len, dropout_start + 15)

        for pos in range(1, run_len):
            index, previous = run[pos], run[pos - 1]
            record = records[index]
            dt = float(record.timestamp_seconds - records[previous].timestamp_seconds)
            if not np.isfinite(dt) or dt <= 0:
                continue
            frame = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(record.image_path)
            renderer = renderers[(record.capture_id, record.camera_id)]

            visual_crop, _ = renderer(frame, visual.state)
            visual_frame = visual.step(visual_crop, dt_seconds=dt, measurement_valid=True)

            blank = torch.zeros((5, int(config["image_size"]), int(config["image_size"])), dtype=torch.float32)
            blind.step(blank, dt_seconds=dt, measurement_valid=False)

            is_dropout = dropout_start <= pos < dropout_end
            if is_dropout:
                dropout.step(blank, dt_seconds=dt, measurement_valid=False)
            else:
                dropout_crop, _ = renderer(frame, dropout.state)
                dropout.step(dropout_crop, dt_seconds=dt, measurement_valid=True)

            gt = camera_joints[index]
            update_metrics(full_by_segment[segment], metrics(visual_frame.state, gt, biometrics))
            update_metrics(blind_by_segment[segment], metrics(blind.state, gt, biometrics))
            update_metrics(dropout_by_segment[segment], metrics(dropout.state, gt, biometrics))
            frame_count += 1
            del frame
        print(f"Rollout {run_no}/{len(runs)}: {segment}, {run_len} frames")

    def summarize(groups):
        values = np.concatenate([np.asarray(v, np.float64) for v in groups.values() if v], axis=0)
        return values.mean(axis=0)

    full, blind, dropout = map(summarize, (full_by_segment, blind_by_segment, dropout_by_segment))
    print("\nCAUSAL HELD-OUT ROLLOUT (mm; one noisy bootstrap seed per contiguous validation run)")
    print(f"Scored frames: {sum(len(v) for v in full_by_segment.values())}; recurrent updates: {frame_count}; mean target IK/FK fit residual: {float(np.mean(residuals[val_ids])):.2f}")
    print(f"{'Stream':<24} {'Root-rel':>10} {'Tips':>10} {'Camera':>10} {'Wrist':>10}")
    print("-" * 68)
    for name, row in (("Visual tracker", full), ("No-vision coast", blind), ("Visual + 15f dropout", dropout)):
        print(f"{name:<24} {row[0]:10.2f} {row[1]:10.2f} {row[2]:10.2f} {row[3]:10.2f}")
    print("\nPer-segment root-relative MPJPE (mm)")
    print(f"{'Segment':<40} {'Frames':>7} {'Visual':>10} {'No vision':>11} {'Dropout':>10}")
    for segment in sorted(full_by_segment):
        v, b, d = full_by_segment[segment], blind_by_segment[segment], dropout_by_segment[segment]
        print(f"{segment:<40} {len(v):7d} {np.mean(v, axis=0)[0]:10.2f} {np.mean(b, axis=0)[0]:11.2f} {np.mean(d, axis=0)[0]:10.2f}")
    print("\nBootstrap uses the first frame's fitted state plus fixed translation/rotation/joint perturbations. No labels are used after initialization; this does not test automatic hand detection or initialization.")


if __name__ == "__main__":
    main()
