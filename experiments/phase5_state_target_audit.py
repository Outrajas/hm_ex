"""Audit Re:InterHand-to-HandState target fitting on the local Mugsy subset.

Calibration uses training segments only.  The held-out segment table reports
the error incurred by representing annotated joints with the current project
FK, before any neural network is trained.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sequence_data = importlib.import_module("06_training.reinterhand_sequences")
target_fitter = importlib.import_module("06_training.targets")

CAPTURE_ROOT = ROOT / "datasets" / "reinterhand" / "captures"
SUBSET_SUFFIX = Path("derived") / "mugsy_subset" / "manifest.csv"
MANIFEST_PATHS = sorted(CAPTURE_ROOT.glob(f"*/{SUBSET_SUFFIX.as_posix()}"))
if not MANIFEST_PATHS:
    raise FileNotFoundError(f"No local Mugsy subset manifest under {CAPTURE_ROOT}")
MANIFEST_PATH = MANIFEST_PATHS[0]
CAPTURE_DIR = MANIFEST_PATH.parents[2]
CAMERA_PATH = CAPTURE_DIR / "Mugsy_cameras" / "cam_params.json"
CAMERA_ID = "400485"
SEED = 20251018
TRAIN_SEGMENTS = 44
VAL_SEGMENTS = 12


def main() -> None:
    if not CAMERA_PATH.is_file():
        raise FileNotFoundError(CAMERA_PATH)
    records = sequence_data.records_from_subset_manifest(
        MANIFEST_PATH, camera_id=CAMERA_ID
    )
    with CAMERA_PATH.open("r", encoding="utf-8") as stream:
        camera = json.load(stream)[CAMERA_ID]
    rotation = np.asarray(camera["R"], dtype=np.float32)
    translation = np.asarray(camera["t"], dtype=np.float32)

    segments = np.asarray(sorted({r.segment for r in records}), dtype=object)
    if len(segments) != TRAIN_SEGMENTS + VAL_SEGMENTS:
        raise RuntimeError(
            f"Expected 56 segments for the locked split; found {len(segments)}"
        )
    split_rng = np.random.default_rng(SEED)
    split_rng.shuffle(segments)
    train_segments = set(segments[:TRAIN_SEGMENTS].tolist())
    val_segments = set(segments[TRAIN_SEGMENTS:].tolist())
    if train_segments & val_segments:
        raise RuntimeError("Segment leakage in the locked split")

    camera_joints = np.empty((len(records), 21, 3), dtype=np.float32)
    for index, record in enumerate(records):
        with record.right_keypoints_path.open("r", encoding="utf-8") as stream:
            world = np.asarray(json.load(stream), dtype=np.float32)
        if world.shape != (24, 3):
            raise ValueError(f"Expected 24 x 3 keypoints: {record.right_keypoints_path}")
        selected = world[sequence_data.KEEP_21_INDICES]
        camera_joints[index] = selected @ rotation.T + translation
        if (index + 1) % 1000 == 0 or index + 1 == len(records):
            print(f"Loaded keypoint labels {index + 1}/{len(records)}")

    # Verify the inverse equations against this project's own FK before
    # interpreting a large real-data residual as a dataset/model mismatch.
    state_definition = importlib.import_module("00_spec.state_definition")
    forward_kinematics = importlib.import_module("02_state.forward_kinematics").forward_kinematics
    synthetic = state_definition.HandState.neutral(z_depth=0.55)
    synthetic.q[:] = np.deg2rad([
        12, 18, -8, 24, 32, 6, 45, 58, 20, -2, 50, 60, 18,
        -8, 42, 55, 12, -15, 48, 62, 16,
    ])
    synthetic_joints_m, _ = forward_kinematics(synthetic)
    synthetic_fit_state, synthetic_fit, synthetic_residual = target_fitter.fit_state_from_camera_joints(
        synthetic_joints_m * 1000.0,
        biometrics=target_fitter.HandBiometrics(),
        clamp_limits=False,
    )
    print(f"Synthetic FK->IK self-check residual: {synthetic_residual:.6f} mm")

    train_idx = np.asarray(
        [i for i, record in enumerate(records) if record.segment in train_segments],
        dtype=np.int64,
    )
    val_idx = np.asarray(
        [i for i, record in enumerate(records) if record.segment in val_segments],
        dtype=np.int64,
    )
    biometrics = target_fitter.calibrate_biometrics_from_joints(camera_joints[train_idx])
    print(
        "Calibration from train segments only: "
        f"palm_length={biometrics.palm_length_ratio:.3f}, "
        f"palm_width={biometrics.palm_width_ratio:.3f}, "
        f"finger_length={biometrics.finger_length_ratio:.3f}, "
        f"thumb_length={biometrics.thumb_ratio:.3f}"
    )
    print(
        f"Split: {len(train_segments)} train / {len(val_segments)} val segments; "
        f"{len(train_idx)} train / {len(val_idx)} val frames"
    )

    reconstructed = np.empty_like(camera_joints)
    reconstructed_free = np.empty_like(camera_joints)
    residuals = np.empty(len(records), dtype=np.float32)
    q_targets = np.empty((len(records), 21), dtype=np.float32)
    q_free = np.empty_like(q_targets)
    for index in range(len(records)):
        state, fitted, residual = target_fitter.fit_state_from_camera_joints(
            camera_joints[index], biometrics=biometrics
        )
        free_state, fitted_free, _ = target_fitter.fit_state_from_camera_joints(
            camera_joints[index], biometrics=biometrics, clamp_limits=False
        )
        reconstructed[index] = fitted
        reconstructed_free[index] = fitted_free
        residuals[index] = residual
        q_targets[index] = state.q
        q_free[index] = free_state.q
        if (index + 1) % 1000 == 0 or index + 1 == len(records):
            print(f"Fitted state targets {index + 1}/{len(records)}")

    joint_error = np.linalg.norm(reconstructed - camera_joints, axis=-1)
    joint_error_free = np.linalg.norm(reconstructed_free - camera_joints, axis=-1)
    tip_joints = [4, 8, 12, 16, 20]
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    def summarize(label: str, indices: np.ndarray) -> None:
        e = joint_error[indices]
        tips = [float(e[:, joint].mean()) for joint in tip_joints]
        print(
            f"{label:<6} MPJPE={e.mean():7.2f} mm | median={np.median(e):7.2f} "
            f"p90={np.percentile(e, 90):7.2f} | "
            + " ".join(f"{name}={value:.2f}" for name, value in zip(finger_names, tips))
        )

    print("\nCURRENT FK FIT RESIDUAL (before neural training)")
    summarize("train", train_idx)
    summarize("val", val_idx)
    print(
        f"val unclamped MPJPE={joint_error_free[val_idx].mean():.2f} mm "
        f"(clamped={joint_error[val_idx].mean():.2f} mm)"
    )
    print("\nVAL PER-JOINT RESIDUAL")
    for joint in range(21):
        print(
            f"joint {joint:02d}: clamped={joint_error[val_idx, joint].mean():6.2f} mm; "
            f"unclamped={joint_error_free[val_idx, joint].mean():6.2f} mm; "
            f"q={np.rad2deg(q_free[val_idx, joint]).mean():7.1f} deg"
        )
    print("\nVAL PER-SEGMENT")
    print(f"{'Segment':<48} {'Frames':>6} {'MPJPE mm':>10} {'Tip mean':>10}")
    for segment in sorted(val_segments):
        ids = np.asarray(
            [i for i in val_idx if records[int(i)].segment == segment],
            dtype=np.int64,
        )
        if not len(ids):
            continue
        print(
            f"{segment:<48} {len(ids):6d} {joint_error[ids].mean():10.2f} "
            f"{joint_error[ids][:, tip_joints].mean():10.2f}"
        )

    lower = np.asarray(target_fitter._constraints.JOINT_LIMITS_LOWER)
    upper = np.asarray(target_fitter._constraints.JOINT_LIMITS_UPPER)
    at_limit = np.isclose(q_targets, lower[None], atol=1e-5) | np.isclose(
        q_targets, upper[None], atol=1e-5
    )
    print(f"\nq targets at a joint limit: {at_limit.mean():.2%}")
    names = target_fitter._topology.JOINT_DOF_NAMES
    outside = (q_free[val_idx] < lower[None]) | (q_free[val_idx] > upper[None])
    print("VAL unclamped q distribution (deg; target outside configured range)")
    for dof, name in enumerate(names):
        values = np.rad2deg(q_free[val_idx, dof])
        print(
            f"{name:<24} outside={outside[:, dof].mean():6.1%} "
            f"median={np.median(values):7.1f} p05={np.percentile(values, 5):7.1f} "
            f"p95={np.percentile(values, 95):7.1f}"
        )
    print("Camera translation convention: camera_R @ world_mm + camera_t_mm.")


if __name__ == "__main__":
    main()
