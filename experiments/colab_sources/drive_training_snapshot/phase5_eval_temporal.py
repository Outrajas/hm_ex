#!/usr/bin/env python
"""Evaluate a saved Phase 5 temporal state model on every held-out frame."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
trainer = importlib.import_module("experiments.phase5_train_temporal")


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


def chunks_for_all_validation(records, val_ids, window_size, max_frame_gap):
    valid = set(map(int, val_ids))
    groups = defaultdict(list)
    for i in val_ids:
        record = records[int(i)]
        groups[(record.capture_id, record.segment, record.camera_id)].append(int(i))
    windows = []
    for ids in groups.values():
        ids.sort(key=lambda i: records[i].frame_idx)
        runs, run, previous = [], [], None
        for index in ids:
            frame = records[index].frame_idx
            if previous is not None and frame - previous > max_frame_gap:
                if run:
                    runs.append(run)
                run = []
            run.append(index)
            previous = frame
        if run:
            runs.append(run)
        for seq in runs:
            for start in range(0, len(seq), window_size):
                part = seq[start:start + window_size]
                if part and all(index in valid for index in part):
                    windows.append(np.asarray(part, dtype=np.int64))
    return windows


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = trainer.choose_device(args.device)
    discovery_args = argparse.Namespace(
        manifest=args.manifest,
        capture=args.capture,
        data_root=args.data_root,
        camera_group=args.camera_group,
        envmap_mode=args.envmap_mode,
        camera_id=args.camera_id,
    )
    records, camera_by_capture, camera_paths = trainer.discover_sources(discovery_args)
    train_ids, val_ids, *_ = trainer.group_split(
        records, float(config["validation_fraction"]), int(config["split_seed"])
    )
    signature = trainer.dataset_signature(records, camera_paths, int(config["split_seed"]))
    if signature != checkpoint.get("dataset_signature"):
        raise RuntimeError("The checkpoint does not match the discovered records/camera files.")
    target_cache = checkpoint_path.parent / "target_cache" / f"states_{signature}.npz"
    camera_joints, target_states, residuals, biometrics = trainer.load_or_fit_targets(
        records, camera_by_capture, train_ids, target_cache, signature
    )
    model = trainer.TemporalStateCorrector(
        hidden_dim=int(config["hidden_dim"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mano_path = ROOT / "models" / "MANO_RIGHT.pkl"
    renderers = trainer.make_renderers(
        records, camera_by_capture, biometrics, int(config["image_size"]), mano_path
    )
    windows = chunks_for_all_validation(
        records,
        val_ids,
        int(config["window_size"]),
        int(config["max_frame_gap"]),
    )
    errors = defaultdict(list)
    authorities = []
    prior_relative_all, corrected_relative_all = [], []
    prior_camera_all, corrected_camera_all, wrist_all, target_all = [], [], [], []
    for w_i, window in enumerate(windows):
        sample = trainer.build_sample(
            window,
            records,
            target_states,
            renderers,
            np.random.default_rng(0),
            train=False,
            image_size=int(config["image_size"]),
        )
        images = torch.from_numpy(sample[0][None]).to(device)
        state_vectors = torch.from_numpy(sample[1][None]).to(device)
        features = trainer.batch_features(state_vectors, device)
        dt = torch.from_numpy(sample[4][None]).to(device)
        with torch.inference_mode():
            output = model.forward_sequence(images, features, dt)
        deltas = torch.cat((output.delta_T, output.delta_R, output.delta_q), dim=-1)[0].cpu().numpy()
        auth = output.authority[0].cpu().numpy()
        for t_i, index in enumerate(window):
            prior_state = trainer.HandState.from_vector(sample[1][t_i].astype(np.float64))
            prior_pose = trainer.forward_kinematics(prior_state)[0] * 1000.0
            corrected = prior_state.copy()
            corrected.T += float(auth[t_i]) * deltas[t_i, :3]
            corrected.R = trainer.quat_mul(
                trainer.coords.rotation_vector_to_quaternion(float(auth[t_i]) * deltas[t_i, 3:6]),
                corrected.R,
            )
            corrected.q = np.clip(
                corrected.q + float(auth[t_i]) * deltas[t_i, 6:],
                trainer.constraints.JOINT_LIMITS_LOWER,
                trainer.constraints.JOINT_LIMITS_UPPER,
            )
            corrected_pose = trainer.forward_kinematics(corrected)[0] * 1000.0
            target_pose = camera_joints[int(index)]
            target_relative = target_pose - target_pose[0:1]
            prior_relative_error = float(np.linalg.norm(
                (prior_pose - prior_pose[0:1]) - target_relative, axis=-1
            ).mean())
            corrected_relative_error = float(np.linalg.norm(
                (corrected_pose - corrected_pose[0:1]) - target_relative, axis=-1
            ).mean())
            prior_camera_error = float(np.linalg.norm(prior_pose - target_pose, axis=-1).mean())
            corrected_camera_error = float(np.linalg.norm(corrected_pose - target_pose, axis=-1).mean())
            wrist_error = float(np.linalg.norm(corrected_pose[0] - target_pose[0]))
            segment = records[int(index)].segment
            errors[segment].append((prior_relative_error, corrected_relative_error,
                                    prior_camera_error, corrected_camera_error, wrist_error))
            authorities.append(float(auth[t_i]))
            prior_relative_all.append(prior_relative_error)
            corrected_relative_all.append(corrected_relative_error)
            prior_camera_all.append(prior_camera_error)
            corrected_camera_all.append(corrected_camera_error)
            wrist_all.append(wrist_error)
            target_all.append(int(index))
        if (w_i + 1) % 20 == 0 or w_i + 1 == len(windows):
            print(f"Evaluated {w_i + 1}/{len(windows)} temporal chunks")

    if len(set(target_all)) != len(target_all) or set(target_all) != set(map(int, val_ids)):
        missing = set(map(int, val_ids)) - set(target_all)
        raise RuntimeError(f"Full validation coverage failed: evaluated {len(set(target_all))}/{len(val_ids)}; missing {len(missing)}")
    print("\nFULL HELD-OUT VALIDATION (each frame once)")
    print(f"Frames: {len(target_all)}/{len(val_ids)}; mean target IK/FK residual: {float(np.mean(residuals[val_ids])):.2f} mm")
    print(f"Root-relative pose MPJPE: prior={float(np.mean(prior_relative_all)):.2f} mm; corrected={float(np.mean(corrected_relative_all)):.2f} mm; gain={float(np.mean(prior_relative_all)-np.mean(corrected_relative_all)):+.2f} mm")
    print(f"Camera-space joint MPJPE: prior={float(np.mean(prior_camera_all)):.2f} mm; corrected={float(np.mean(corrected_camera_all)):.2f} mm; gain={float(np.mean(prior_camera_all)-np.mean(corrected_camera_all)):+.2f} mm")
    print(f"Corrected wrist-position error: {float(np.mean(wrist_all)):.2f} mm")
    print(f"Mean learned authority:            {float(np.mean(authorities)):.3f}")
    print("\nPer-segment root-relative MPJPE (mm)")
    print(f"{'Segment':<40} {'Frames':>6} {'Prior':>9} {'Corrected':>11} {'Gain':>8}")
    for segment in sorted(errors):
        values = np.asarray(errors[segment], np.float64)
        prior_mean, corrected_mean = values.mean(axis=0)[:2]
        print(f"{segment:<40} {len(values):6d} {prior_mean:9.2f} {corrected_mean:11.2f} {prior_mean-corrected_mean:8.2f}")
    report = {
        "checkpoint": str(checkpoint_path),
        "frames": len(target_all),
        "prior_root_relative_mpjpe_mm": float(np.mean(prior_relative_all)),
        "corrected_root_relative_mpjpe_mm": float(np.mean(corrected_relative_all)),
        "root_relative_gain_mm": float(np.mean(prior_relative_all) - np.mean(corrected_relative_all)),
        "prior_camera_space_mpjpe_mm": float(np.mean(prior_camera_all)),
        "corrected_camera_space_mpjpe_mm": float(np.mean(corrected_camera_all)),
        "camera_space_gain_mm": float(np.mean(prior_camera_all) - np.mean(corrected_camera_all)),
        "corrected_wrist_position_error_mm": float(np.mean(wrist_all)),
        "mean_authority": float(np.mean(authorities)),
        "mean_target_fit_residual_mm": float(np.mean(residuals[val_ids])),
        "segments": {
            segment: {
                "frames": len(values),
                "prior_mpjpe_mm": float(np.asarray(values).mean(axis=0)[0]),
                "corrected_mpjpe_mm": float(np.asarray(values).mean(axis=0)[1]),
                "prior_camera_space_mpjpe_mm": float(np.asarray(values).mean(axis=0)[2]),
                "corrected_camera_space_mpjpe_mm": float(np.asarray(values).mean(axis=0)[3]),
                "corrected_wrist_position_error_mm": float(np.asarray(values).mean(axis=0)[4]),
            }
            for segment, values in errors.items()
        },
    }
    report_path = checkpoint_path.parent / "full_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved full validation report: {report_path}")


if __name__ == "__main__":
    main()
