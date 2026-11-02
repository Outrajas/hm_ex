#!/usr/bin/env python
"""Evaluate a saved chunk checkpoint using only its persistent validation manifest.

This remains usable after training images have been cleaned from the session
disk. It reports per-frame correction metrics and a causal rollout that starts
from one deliberately perturbed bootstrap pose, then consumes only RGB frames.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
trainer = importlib.import_module("experiments.phase5_train_temporal")
runtime = importlib.import_module("08_runtime.tracker")

TIP_IDS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="A chunk's best.pt or latest.pt")
    parser.add_argument("--validation-manifest", action="append", required=True)
    parser.add_argument("--camera-group", default="Mugsy_cameras")
    parser.add_argument("--camera-id", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dropout-frames", type=int, default=15)
    parser.add_argument("--report", default="", help="JSON output; defaults beside the checkpoint")
    return parser.parse_args()


def load_validation(args):
    discovery = SimpleNamespace(
        validation_manifest=args.validation_manifest,
        manifest=[],
        capture=[],
        data_root="",
        camera_group=args.camera_group,
        camera_id=args.camera_id,
        envmap_mode="envmap_per_segment",
    )
    return trainer.discover_sources(discovery, validation=True)


def fit_targets(records, camera_by_capture, biometrics):
    camera_joints = np.empty((len(records), 21, 3), dtype=np.float32)
    states = np.empty((len(records), trainer.STATE_DIM), dtype=np.float32)
    residuals = np.empty(len(records), dtype=np.float32)
    for i, record in enumerate(records):
        with record.right_keypoints_path.open("r", encoding="utf-8") as stream:
            world = np.asarray(json.load(stream), dtype=np.float32)
        if world.shape != (24, 3) or not np.isfinite(world).all():
            raise ValueError(f"Invalid 24x3 right-hand labels: {record.right_keypoints_path}")
        camera = camera_by_capture[record.capture_id][record.camera_id]
        rotation = np.asarray(camera["R"], dtype=np.float32).reshape(3, 3)
        translation = np.asarray(camera["t"], dtype=np.float32).reshape(3)
        joints = world[trainer.data_api.KEEP_21_INDICES] @ rotation.T + translation
        camera_joints[i] = joints
        beta = np.zeros(10, dtype=np.float32)
        if record.right_mano_path is not None and record.right_mano_path.is_file():
            with record.right_mano_path.open("r", encoding="utf-8") as stream:
                beta = np.asarray(json.load(stream).get("shape", beta), dtype=np.float32).reshape(10)
        state, _, residual = trainer.target_api.fit_state_from_camera_joints(
            joints, biometrics=biometrics, beta=beta, clamp_limits=True
        )
        states[i] = state.to_vector().astype(np.float32)
        residuals[i] = residual
        if (i + 1) % 1000 == 0 or i + 1 == len(records):
            print(f"Prepared validation targets {i + 1}/{len(records)}", flush=True)
    trainer.derive_target_velocities(records, states)
    return camera_joints, states, residuals


def contiguous_runs(records, max_gap):
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[(record.capture_id, record.segment, record.camera_id)].append(i)
    runs = []
    for key, indices in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        indices.sort(key=lambda i: records[i].frame_idx)
        current = []
        previous_frame = None
        for index in indices:
            frame = records[index].frame_idx
            if previous_frame is not None and frame - previous_frame > max_gap:
                if current:
                    runs.append((key, current))
                current = []
            current.append(index)
            previous_frame = frame
        if current:
            runs.append((key, current))
    return runs


def score_state(state, target_joints, biometrics):
    predicted = trainer.forward_kinematics(state, biometrics)[0] * 1000.0
    pred_relative = predicted - predicted[0:1]
    target_relative = target_joints - target_joints[0:1]
    distance = np.linalg.norm(pred_relative - target_relative, axis=-1)
    camera_distance = np.linalg.norm(predicted - target_joints, axis=-1)
    return np.asarray(
        [distance.mean(), distance[TIP_IDS].mean(), camera_distance.mean(), camera_distance[0]],
        dtype=np.float64,
    )


def apply_delta(state, delta, authority):
    corrected = state.copy()
    corrected.T += authority * delta[:3]
    corrected.R = trainer.quat_mul(
        trainer.coords.rotation_vector_to_quaternion(authority * delta[3:6]),
        corrected.R,
    )
    corrected.q = np.clip(
        corrected.q + authority * delta[6:],
        trainer.constraints.JOINT_LIMITS_LOWER,
        trainer.constraints.JOINT_LIMITS_UPPER,
    )
    return corrected


def perturbed_seed(target):
    state = target.copy()
    state.T += np.asarray([0.015, -0.010, 0.020], dtype=np.float64)
    rotvec = np.deg2rad(np.asarray([5.0, -7.0, 4.0], dtype=np.float64))
    state.R = trainer.quat_mul(
        trainer.coords.rotation_vector_to_quaternion(rotvec), state.R
    )
    q_noise = np.deg2rad(8.0) * np.sin(np.arange(21, dtype=np.float64) * 1.7)
    state.q = np.clip(
        state.q + q_noise,
        trainer.constraints.JOINT_LIMITS_LOWER,
        trainer.constraints.JOINT_LIMITS_UPPER,
    )
    state.V[:] = 0.0
    state.omega[:] = 0.0
    state.qdot[:] = 0.0
    return state


def summarize(rows):
    if not rows:
        return np.full(4, np.nan, dtype=np.float64)
    return np.asarray(rows, dtype=np.float64).mean(axis=0)


def main():
    args = parse_args()
    if args.dropout_frames < 0:
        raise ValueError("--dropout-frames cannot be negative")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    biometrics = trainer.biometrics_from_checkpoint(checkpoint, checkpoint_path)
    if biometrics is None:
        raise ValueError("Checkpoint has no frozen biometrics; cannot reproduce its state/FK geometry.")
    device = trainer.choose_device(args.device)
    records, camera_by_capture, _ = load_validation(args)
    if not records:
        raise ValueError("Validation manifest contains no frames.")
    camera_joints, target_states, residuals = fit_targets(records, camera_by_capture, biometrics)

    model = trainer.TemporalStateCorrector(hidden_dim=int(config["hidden_dim"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    mano_path = ROOT / "models" / "MANO_RIGHT.pkl"
    if not mano_path.is_file():
        raise FileNotFoundError(mano_path)
    renderers = trainer.make_renderers(
        records, camera_by_capture, biometrics, int(config["image_size"]), mano_path
    )

    # Open-loop validation uses a noisy state prior on each window; unlike the
    # rollout below, it consults per-frame GT to form those synthetic priors.
    windows = []
    for _, run in contiguous_runs(records, int(config["max_frame_gap"])):
        size = int(config["window_size"])
        windows.extend(run[start:start + size] for start in range(0, len(run), size))
    window_ids = [int(index) for window in windows for index in window]
    if len(window_ids) != len(records) or set(window_ids) != set(range(len(records))):
        raise RuntimeError("Open-loop windows do not cover each validation frame exactly once")
    open_prior, open_corrected = [], []
    open_by_segment = defaultdict(lambda: [[], []])
    for window_no, window in enumerate(windows, 1):
        sample = trainer.build_sample(
            window, records, target_states, renderers,
            np.random.default_rng(trainer.SEED + int(window[0])),
            train=False, image_size=int(config["image_size"]),
        )
        images = torch.from_numpy(sample[0][None]).to(device)
        vectors = torch.from_numpy(sample[1][None]).to(device)
        features = trainer.batch_features(vectors, device)
        dt = torch.from_numpy(sample[4][None]).to(device)
        with torch.inference_mode():
            output = model.forward_sequence(images, features, dt)
        deltas = torch.cat((output.delta_T, output.delta_R, output.delta_q), dim=-1)[0].cpu().numpy()
        authority = output.authority[0].cpu().numpy()
        for pos, index in enumerate(window):
            prior = trainer.HandState.from_vector(sample[1][pos].astype(np.float64))
            corrected = apply_delta(prior, deltas[pos], float(authority[pos]))
            before = score_state(prior, camera_joints[int(index)], biometrics)
            after = score_state(corrected, camera_joints[int(index)], biometrics)
            open_prior.append(before)
            open_corrected.append(after)
            bucket = open_by_segment[records[int(index)].segment]
            bucket[0].append(before)
            bucket[1].append(after)
        if window_no % 20 == 0 or window_no == len(windows):
            print(f"Open-loop windows {window_no}/{len(windows)}", flush=True)

    # Causal rollout initializes from a single noisy bootstrap pose, then gets
    # no labels. It includes a synthetic 15-frame visual blackout per run.
    visual_rows, coast_rows, dropout_rows = [], [], []
    causal_by_segment = defaultdict(lambda: [[], [], []])
    run_list = contiguous_runs(records, int(config["max_frame_gap"]))
    for run_no, (group, run) in enumerate(run_list, 1):
        if len(run) < 2:
            continue
        segment = str(group[1])
        seed = perturbed_seed(trainer.HandState.from_vector(target_states[run[0]].astype(np.float64)))
        visual = runtime.PersistentHandTracker(model, seed)
        coast = runtime.PersistentHandTracker(model, seed)
        dropout = runtime.PersistentHandTracker(model, seed)
        first_target = camera_joints[run[0]]
        initial_score = score_state(seed, first_target, biometrics)
        for aggregate, per_segment in (
            (visual_rows, causal_by_segment[segment][0]),
            (coast_rows, causal_by_segment[segment][1]),
            (dropout_rows, causal_by_segment[segment][2]),
        ):
            aggregate.append(initial_score)
            per_segment.append(initial_score)

        drop_start = max(1, int(len(run) * 0.40))
        drop_end = min(len(run), drop_start + max(0, args.dropout_frames))
        for pos in range(1, len(run)):
            index, previous = run[pos], run[pos - 1]
            record = records[index]
            dt = float(record.timestamp_seconds - records[previous].timestamp_seconds)
            if not np.isfinite(dt) or dt <= 0:
                continue
            frame = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(record.image_path)
            renderer = renderers[(record.capture_id, record.camera_id)]
            visual_input, _ = renderer(frame, visual.state)
            visual_output = visual.step(visual_input, dt_seconds=dt, measurement_valid=True)
            blank = torch.zeros((5, int(config["image_size"]), int(config["image_size"])))
            coast_output = coast.step(blank, dt_seconds=dt, measurement_valid=False)
            if drop_start <= pos < drop_end:
                dropout_output = dropout.step(blank, dt_seconds=dt, measurement_valid=False)
            else:
                dropout_input, _ = renderer(frame, dropout.state)
                dropout_output = dropout.step(dropout_input, dt_seconds=dt, measurement_valid=True)
            target = camera_joints[index]
            values = (
                score_state(visual_output.state, target, biometrics),
                score_state(coast_output.state, target, biometrics),
                score_state(dropout_output.state, target, biometrics),
            )
            for value, aggregate, per_segment in zip(
                values,
                (visual_rows, coast_rows, dropout_rows),
                causal_by_segment[segment],
            ):
                aggregate.append(value)
                per_segment.append(value)
            del frame
        if run_no % 10 == 0 or run_no == len(run_list):
            print(f"Causal runs {run_no}/{len(run_list)}", flush=True)

    names = ("Root-rel mm", "Tips mm", "Camera mm", "Wrist mm")
    print(f"\nCHECKPOINT EVALUATION — {checkpoint_path}")
    print(f"Device: {device}; validation frames: {len(records)}; target fit residual: {float(np.mean(residuals)):.2f} mm")
    print("\nOPEN-LOOP CORRECTION (GT-derived noisy prior on each window)")
    print(f"{'Pose':<18}" + "".join(f"{name:>13}" for name in names))
    print("-" * 70)
    for label, rows in (("Prior", open_prior), ("Corrected", open_corrected)):
        values = summarize(rows)
        print(f"{label:<18}" + "".join(f"{value:13.2f}" for value in values))

    print(f"\nCAUSAL ROLLOUT (one noisy bootstrap pose; synthetic blackout={args.dropout_frames} frames)")
    print(f"{'Stream':<18}" + "".join(f"{name:>13}" for name in names))
    print("-" * 70)
    causal_summaries = {}
    for label, rows in (("Visual", visual_rows), ("No-vision coast", coast_rows), ("Visual + dropout", dropout_rows)):
        values = summarize(rows)
        causal_summaries[label] = values.tolist()
        print(f"{label:<18}" + "".join(f"{value:13.2f}" for value in values))
    print("\nPer-segment causal root-relative MPJPE (mm)")
    print(f"{'Segment':<32} {'Frames':>7} {'Visual':>10} {'Coast':>10} {'Dropout':>10}")
    for segment in sorted(causal_by_segment):
        vals = causal_by_segment[segment]
        print(
            f"{segment:<32} {len(vals[0]):7d} "
            f"{np.mean(vals[0], axis=0)[0]:10.2f} "
            f"{np.mean(vals[1], axis=0)[0]:10.2f} "
            f"{np.mean(vals[2], axis=0)[0]:10.2f}"
        )

    report_path = Path(args.report).expanduser().resolve() if args.report else checkpoint_path.parent / "phase6_validation_report.json"
    report = {
        "checkpoint": str(checkpoint_path),
        "validation_manifests": [str(Path(value).expanduser().resolve()) for value in args.validation_manifest],
        "validation_frames": len(records),
        "target_fit_residual_mm": float(np.mean(residuals)),
        "open_loop_prior": summarize(open_prior).tolist(),
        "open_loop_corrected": summarize(open_corrected).tolist(),
        "causal_rollout": causal_summaries,
        "bootstrap_assumption": "First GT-fitted validation pose plus fixed translation, rotation, and joint perturbations; later labels are used only for scoring.",
        "dropout_frames": args.dropout_frames,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
