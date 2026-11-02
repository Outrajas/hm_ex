"""Compare raw MediaPipe hand landmarks with temporal and state-layer streams.

Usage (CPU only):
    python experiments/phase4_state_on_mediapipe.py path/to/webcam.mp4

The script uses MediaPipe Tasks' HandLandmarker VIDEO mode. If the task model
is not supplied with --model, it is downloaded once to the user's cache from
Google's public MediaPipe model storage. Outputs are written beside the input
video unless --output-dir is given.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
import urllib.request
from collections import deque
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mediapipe as mp
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Install the existing environment dependencies first: mediapipe, "
        "opencv-python, numpy, matplotlib. This script does not install packages."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

state_module = importlib.import_module("02_state.state")
constraint_module = importlib.import_module("02_state.constraints")
fk_module = importlib.import_module("02_state.forward_kinematics")
topology = importlib.import_module("00_spec.hand_topology")
HandState = state_module.HandState
ControlInput = state_module.ControlInput
state_update = state_module.state_update
project_constraints = constraint_module.project_constraints
forward_kinematics = fk_module.forward_kinematics
HandBiometrics = fk_module.HandBiometrics

JOINTS = 21
TIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
PARENTS = np.asarray([topology.PARENT_INDICES[i] for i in range(1, JOINTS)], dtype=np.int64)
CHILDREN = np.arange(1, JOINTS, dtype=np.int64)
DEFAULT_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input webcam video path")
    parser.add_argument("--model", type=Path, help="HandLandmarker .task file; cached download used by default")
    parser.add_argument("--output-dir", type=Path, help="Directory for NPZ metrics and trajectory PNG")
    parser.add_argument("--calibration-frames", type=int, default=30,
                        help="First N detections used for per-user bone-length medians")
    parser.add_argument("--max-prediction-seconds", type=float, default=0.75,
                        help="After this long without a detection, hold the last predicted pose")
    return parser.parse_args()


def resolve_model(model_path: Path | None) -> Path:
    if model_path is not None:
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe task model not found: {model_path}")
        return model_path
    cache = Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_suffix(".download")
        print(f"Downloading MediaPipe hand model to {cache}")
        try:
            urllib.request.urlretrieve(DEFAULT_TASK_URL, temp)
            temp.replace(cache)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
    return cache


def make_landmarker(model_path: Path):
    tasks = mp.tasks
    options = tasks.vision.HandLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=str(model_path), delegate=tasks.BaseOptions.Delegate.CPU),
        running_mode=tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return tasks.vision.HandLandmarker.create_from_options(options)


def read_video(video_path: Path, model_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    frames_world, frames_image, detections, confidences, times = [], [], [], [], []
    frame_index = 0
    last_ms = -1
    with make_landmarker(model_path) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            prop_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_s = prop_ms / 1000.0 if math.isfinite(prop_ms) and prop_ms > 0 else frame_index / fps
            timestamp_ms = max(int(round(timestamp_s * 1000)), last_ms + 1)
            last_ms = timestamp_ms
            timestamp_s = timestamp_ms / 1000.0
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(image, timestamp_ms)
            has_hand = bool(result.hand_world_landmarks and result.hand_landmarks)
            if has_hand:
                lm_world = result.hand_world_landmarks[0]
                lm_image = result.hand_landmarks[0]
                world = np.asarray([[p.x, p.y, p.z] for p in lm_world], dtype=np.float32)
                image_xy = np.asarray([[p.x, p.y] for p in lm_image], dtype=np.float32)
                # Tasks currently returns category score for the detected hand.
                try:
                    confidence = float(result.handedness[0][0].score)
                except (AttributeError, IndexError, TypeError):
                    confidence = 1.0
                if world.shape != (JOINTS, 3) or image_xy.shape != (JOINTS, 2):
                    raise RuntimeError("MediaPipe returned a hand with an unexpected landmark count")
            else:
                world = np.full((JOINTS, 3), np.nan, dtype=np.float32)
                image_xy = np.full((JOINTS, 2), np.nan, dtype=np.float32)
                confidence = 0.0
            frames_world.append(world)
            frames_image.append(image_xy)
            detections.append(has_hand)
            confidences.append(confidence)
            times.append(timestamp_s)
            frame_index += 1
            if frame_index % 300 == 0:
                print(f"Processed {frame_index} frames ({sum(detections)} detections)")
    cap.release()
    if not frame_index:
        raise RuntimeError("The input video contained no readable frames")
    return (
        np.stack(frames_world), np.stack(frames_image), np.asarray(detections, bool),
        np.asarray(confidences, np.float64), np.asarray(times, np.float64), fps,
    )


def bone_lengths(poses: np.ndarray) -> np.ndarray:
    vec = poses[:, CHILDREN, :] - poses[:, PARENTS, :]
    return np.linalg.norm(vec, axis=-1)


def kinematic_project(pose: np.ndarray, lengths_m: np.ndarray, fallback_dirs: np.ndarray) -> np.ndarray:
    """Tree projection with calibrated segment lengths, anchored at the wrist."""
    out = np.empty_like(pose, dtype=np.float64)
    out[0] = pose[0]
    for child in range(1, JOINTS):
        parent = int(PARENTS[child - 1])
        direction = pose[child] - pose[parent]
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1e-8:
            direction = fallback_dirs[child - 1]
            norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            direction = np.array([0.0, 1.0, 0.0])
            norm = 1.0
        out[child] = out[parent] + direction * (lengths_m[child - 1] / norm)
    return out


def lowpass_alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
    return 1.0 / (1.0 + tau / max(dt, 1e-5))


def one_euro(data: np.ndarray, valid: np.ndarray, timestamps: np.ndarray,
             min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> np.ndarray:
    result = np.full_like(data, np.nan, dtype=np.float64)
    x_prev = dx_prev = None
    t_prev = None
    for i, (sample, good) in enumerate(zip(data, valid)):
        if not good:
            continue
        x = sample.astype(np.float64)
        if x_prev is None:
            x_prev, dx_prev, t_prev = x.copy(), np.zeros_like(x), timestamps[i]
            result[i] = x
            continue
        dt = max(float(timestamps[i] - t_prev), 1e-5)
        derivative = (x - x_prev) / dt
        dx = lowpass_alpha(d_cutoff, dt) * derivative + (1 - lowpass_alpha(d_cutoff, dt)) * dx_prev
        cutoff = min_cutoff + beta * np.abs(dx)
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-4))
        alpha = 1.0 / (1.0 + tau / dt)
        filtered = alpha * x + (1 - alpha) * x_prev
        result[i] = filtered
        x_prev, dx_prev, t_prev = filtered, dx, timestamps[i]
    return result


def lag_ms(reference: np.ndarray, candidate: np.ndarray, timestamps: np.ndarray, valid: np.ndarray) -> float:
    """Positive means candidate lags raw; aggregates normalized 3D xcorr by fingertip."""
    ids = np.flatnonzero(valid & np.isfinite(reference[:, TIPS, :]).all(axis=(1, 2))
                         & np.isfinite(candidate[:, TIPS, :]).all(axis=(1, 2)))
    if len(ids) < 8:
        return float("nan")
    # Only use uninterrupted runs so missing detections do not create false shifts.
    runs = np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1)
    best_lags, dts = [], []
    for run in runs:
        if len(run) < 8:
            continue
        dt = float(np.median(np.diff(timestamps[run])))
        if dt <= 0:
            continue
        limit = max(1, min(len(run) // 4, int(round(0.25 / dt))))
        for tip in TIPS:
            a = reference[run, tip, :].astype(float)
            b = candidate[run, tip, :].astype(float)
            a -= a.mean(axis=0, keepdims=True)
            b -= b.mean(axis=0, keepdims=True)
            corr = []
            shifts = range(-limit, limit + 1)
            for shift in shifts:
                if shift < 0:
                    aa, bb = a[-shift:], b[:shift]
                elif shift > 0:
                    aa, bb = a[:-shift], b[shift:]
                else:
                    aa, bb = a, b
                denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
                corr.append(float(np.sum(aa * bb) / denom) if denom > 1e-12 else -1.0)
            # At positive shift, the candidate is later than the raw reference.
            best_lags.append(float(list(shifts)[int(np.argmax(corr))]))
            dts.append(dt)
    if not best_lags:
        return float("nan")
    return float(np.median(np.asarray(best_lags) * np.asarray(dts)) * 1000.0)


def jitter_mm(stream: np.ndarray, valid: np.ndarray) -> float:
    ids = np.flatnonzero(valid & np.isfinite(stream[:, TIPS, :]).all(axis=(1, 2)))
    runs = np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1)
    second = [np.diff(stream[r][:, TIPS, :], n=2, axis=0) for r in runs if len(r) >= 3]
    if not second:
        return float("nan")
    values = np.concatenate(second, axis=0)
    return float(np.sqrt(np.mean(np.sum(values * values, axis=-1))) * 1000.0)


def bone_cv_mm(stream: np.ndarray, valid: np.ndarray) -> float:
    good = stream[valid & np.isfinite(stream).all(axis=(1, 2))]
    if len(good) < 2:
        return float("nan")
    lengths = bone_lengths(good)
    means = lengths.mean(axis=0)
    cvs = lengths.std(axis=0) / np.maximum(means, 1e-9)
    return float(np.mean(cvs))


def make_state_stream(raw: np.ndarray, detected: np.ndarray, confidence: np.ndarray,
                      timestamps: np.ndarray, calibration_n: int,
                      max_prediction_seconds: float):
    detected_ids = np.flatnonzero(detected)
    if not len(detected_ids):
        raise RuntimeError("MediaPipe did not detect a hand in any frame")
    calib_ids = detected_ids[:max(1, calibration_n)]
    calibrated_lengths = np.median(bone_lengths(raw[calib_ids]), axis=0).astype(np.float64)

    # Existing FK provides the neutral articulated direction fallback if an input bone collapses.
    fk_neutral = HandState.neutral(z_depth=0.0)
    fk_neutral.T[:] = 0.0
    neutral_pose, _ = forward_kinematics(fk_neutral, HandBiometrics())
    neutral_dirs = neutral_pose[CHILDREN] - neutral_pose[PARENTS]
    neutral_dirs /= np.maximum(np.linalg.norm(neutral_dirs, axis=-1, keepdims=True), 1e-9)

    out = np.full_like(raw, np.nan, dtype=np.float64)
    state = HandState.neutral(z_depth=0.0)
    state.T[:] = raw[detected_ids[0], 0]
    last_pose = None
    last_detection_time = None
    previous_detected_pose = None
    previous_detected_time = None
    previous_frame_time = None
    recent = deque(maxlen=5)
    fallback_directions = neutral_dirs.copy()
    for i, t in enumerate(timestamps):
        dt = (float(t - previous_frame_time) if previous_frame_time is not None else 1.0 / 30.0)
        dt = max(dt, 1e-5)
        if detected[i]:
            measurement = raw[i].astype(np.float64)
            authority = float(np.clip(confidence[i], 0.0, 1.0))
            # state_update advances the existing persistent state and confidence-blends root translation.
            state = state_update(state, ControlInput(target_T=measurement[0], authority=authority), dt)
            state = project_constraints(state)
            anchored = measurement + (state.T - measurement[0])[None, :]
            projected = kinematic_project(anchored, calibrated_lengths, fallback_directions)
            # Confidence controls the correction from measured geometry to the fixed-length projection.
            blended = measurement * (1.0 - authority) + projected * authority
            # Restore fixed lengths after blending, so this stream remains kinematically consistent.
            blended = kinematic_project(blended, calibrated_lengths, fallback_directions)
            # Keep the state-layer output aligned with the confidence-blended root state.
            blended += (state.T - blended[0])[None, :]
            out[i] = blended
            directions = blended[CHILDREN] - blended[PARENTS]
            fallback_directions = directions / np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-9)
            previous_detected_pose, previous_detected_time = last_pose, last_detection_time
            last_pose, last_detection_time = blended.copy(), float(t)
            recent.append(measurement.copy())
        elif last_pose is not None:
            gap = float(t - last_detection_time)
            if gap > max_prediction_seconds:
                out[i] = last_pose
            elif previous_detected_pose is not None and previous_detected_time is not None:
                span = max(last_detection_time - previous_detected_time, 1e-5)
                velocity = (last_pose - previous_detected_pose) / span
                out[i] = last_pose + velocity * gap
            else:
                out[i] = last_pose
            # Track the predicted wrist through the existing constrained state transition.
            state = state_update(state, ControlInput(authority=0.0), dt)
            state = project_constraints(state)
        else:
            # Before the first detection there is no valid subject-specific prediction.
            out[i] = np.nan
        # Every frame advances the state integrator, including missed detections.
        previous_frame_time = float(t)
    return out, calibrated_lengths, neutral_pose, recent


def tune_one_euro(raw: np.ndarray, detected: np.ndarray, timestamps: np.ndarray):
    valid_ids = np.flatnonzero(detected)
    prefix_end = max(8, int(math.ceil(len(detected) * 0.20)))
    tune_valid = np.zeros_like(detected)
    tune_valid[:prefix_end] = detected[:prefix_end]
    if int(tune_valid.sum()) < 8:
        tune_valid[:] = False
        tune_valid[valid_ids[:min(len(valid_ids), 30)]] = True
    best = None
    # Tune once on the opening fifth: minimize jitter while penalizing temporal lag.
    for min_cutoff in (0.5, 1.0, 2.0, 4.0):
        for beta in (0.0, 0.01, 0.03, 0.08, 0.15):
            candidate = one_euro(raw, detected, timestamps, min_cutoff, beta)
            jitter = jitter_mm(candidate, tune_valid)
            lag = abs(lag_ms(raw, candidate, timestamps, tune_valid))
            # Lag penalty is expressed in mm using the opening-section tip speed.
            ids = np.flatnonzero(tune_valid)
            speed = 0.0
            if len(ids) > 1:
                displacement = np.linalg.norm(np.diff(raw[ids][:, TIPS, :], axis=0), axis=-1)
                elapsed = np.maximum(np.diff(timestamps[ids]), 1e-5)
                speed = float(np.nanmedian(displacement / elapsed[:, None])) * 1000.0
            objective = jitter + (speed * lag / 1000.0)
            if math.isfinite(objective) and (best is None or objective < best[0]):
                best = (objective, min_cutoff, beta)
    return (best[1], best[2]) if best else (1.0, 0.03)


def dropout_max_drift(stream: np.ndarray, raw: np.ndarray, detected: np.ndarray) -> float:
    drops = np.flatnonzero(~detected)
    if not len(drops):
        return float("nan")
    max_drift = 0.0
    valid_ids = np.flatnonzero(detected)
    for i in drops:
        prior_ids = valid_ids[valid_ids < i]
        if not len(prior_ids) or not np.isfinite(stream[i]).all():
            continue
        # The hold reference is the mean of up to five most recent detected poses.
        hold = np.mean(raw[prior_ids[-5:]], axis=0)
        drift = np.linalg.norm(stream[i] - hold, axis=-1)
        max_drift = max(max_drift, float(np.max(drift)) * 1000.0)
    return max_drift


def find_fast_window(raw: np.ndarray, detected: np.ndarray, timestamps: np.ndarray):
    n = len(raw)
    window = max(10, int(round(n * 0.12)))
    vel = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dt = max(float(timestamps[i] - timestamps[i - 1]), 1e-5)
        if detected[i] and detected[i - 1]:
            vel[i] = np.nanmean(np.linalg.norm(raw[i, TIPS, :] - raw[i - 1, TIPS, :], axis=-1)) / dt
    smooth = np.convolve(vel, np.ones(max(1, min(5, window))) / max(1, min(5, window)), mode="same")
    start = int(np.argmax(np.convolve(smooth, np.ones(window), mode="same")))
    start = max(0, min(start - window // 2, n - window))
    return start, start + window


def save_trajectory_plot(path: Path, streams: dict[str, np.ndarray], timestamps: np.ndarray,
                         raw: np.ndarray, detected: np.ndarray):
    start, stop = find_fast_window(raw, detected, timestamps)
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    colors = {"raw": "#777777", "1-Euro": "#1976d2", "state": "#d32f2f"}
    for ax, tip, name in zip(axes, TIPS, TIP_NAMES):
        for stream_name, values in streams.items():
            # MediaPipe camera/world y is plotted in millimeters; sign is retained as provided.
            ax.plot(timestamps[start:stop], values[start:stop, tip, 1] * 1000.0,
                    label=stream_name, color=colors[stream_name], linewidth=1.2)
        ax.set_ylabel(f"{name}\ny (mm)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Fingertip y trajectories, fast-motion window {timestamps[start]:.2f}–{timestamps[stop-1]:.2f} s")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else video.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model(args.model.expanduser().resolve() if args.model else None)
    print("CPU only: CUDA is disabled")
    print(f"Video: {video}")
    print(f"Model: {model_path}")
    raw, image_landmarks, detected, confidence, timestamps, fps = read_video(video, model_path)
    n = len(raw)
    if not detected.any():
        raise RuntimeError("No hand detected; no stream metrics can be computed")
    print(f"Frames: {n}; detections: {int(detected.sum())}/{n} ({detected.mean():.1%}); nominal FPS: {fps:.3f}")

    min_cutoff, beta = tune_one_euro(raw, detected, timestamps)
    euro = one_euro(raw, detected, timestamps, min_cutoff, beta)
    state_stream, subject_lengths, fk_neutral, _ = make_state_stream(
        raw, detected, confidence, timestamps, args.calibration_frames,
        args.max_prediction_seconds,
    )
    streams = {"raw": raw.astype(np.float64), "1-Euro": euro, "state": state_stream}
    # Save API landmarks untouched, along with all derived stream arrays and calibration metadata.
    archive_path = output_dir / f"{video.stem}_phase4_landmarks.npz"
    np.savez_compressed(
        archive_path,
        world_landmarks_m=raw.astype(np.float32),
        image_landmarks_normalized=image_landmarks.astype(np.float32),
        detection_flag=detected,
        detection_confidence=confidence.astype(np.float32),
        timestamp_seconds=timestamps.astype(np.float64),
        raw_stream_m=raw.astype(np.float32),
        one_euro_stream_m=euro.astype(np.float32),
        state_stream_m=state_stream.astype(np.float32),
        calibrated_bone_lengths_m=subject_lengths.astype(np.float32),
        one_euro_min_cutoff=np.float64(min_cutoff),
        one_euro_beta=np.float64(beta),
    )
    plot_path = output_dir / f"{video.stem}_phase4_fingertip_y.png"
    save_trajectory_plot(plot_path, streams, timestamps, raw, detected)

    print("\nSTREAM METRICS")
    print(f"1-Euro opening-prefix tuning: min_cutoff={min_cutoff:g}, beta={beta:g}")
    print(f"{'Stream':<10} {'Jitter RMS mm':>15} {'Lag vs raw ms':>15} {'Bone CV %':>12} {'Drop max drift mm':>19}")
    print("-" * 75)
    for name, values in streams.items():
        jitter = jitter_mm(values, detected)
        lag = lag_ms(raw, values, timestamps, detected) if name != "raw" else 0.0
        cv = bone_cv_mm(values, detected) * 100.0
        drift = dropout_max_drift(values, raw, detected)
        print(f"{name:<10} {jitter:15.3f} {lag:15.3f} {cv:12.3f} {drift:19.3f}")
    print(f"\nSaved frame landmarks and stream arrays: {archive_path}")
    print(f"Saved fast-motion fingertip trajectory: {plot_path}")
    print(
        "Assumptions: MediaPipe world coordinates are meters and use its hand-local "
        "coordinate frame; image landmarks are stored normalized to [0,1]. The wrist "
        "is the root (joint 0), and the repo's 21-joint parent topology matches "
        "MediaPipe's wrist/thumb/index/middle/ring/pinky ordering. Bone lengths are "
        "the per-edge medians from the first N detections (default N=30). Confidence "
        "is MediaPipe's handedness category score; when unavailable it defaults to 1. "
        "The FK model supplies fallback directions for degenerate bones; calibrated "
        "lengths are enforced by tree projection. Dropout drift compares each "
        "prediction with the mean of up to five preceding detected poses."
    )


if __name__ == "__main__":
    main()
