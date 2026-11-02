"""Offline diagnosis and temporal-filter comparison for a Phase 4 NPZ.

Usage:
    python experiments/phase4b_state_fix.py datasets/hand_test_phase4_landmarks.npz

No video decoding or MediaPipe inference is performed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PARENTS = np.asarray([0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
CHILDREN = np.arange(1, 21)
TIPS = np.asarray([4, 8, 12, 16, 20])
TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
MIN_CUTOFFS = (0.5, 1.0, 2.0, 4.0)
BETAS = (0.0, 0.01, 0.05, 0.2)
DAMPING = 0.9
MAX_SPEED_M_S = 2.5
MAX_DROPPED_FRAMES = 8


def args_parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "npz", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "hand_test_phase4_landmarks.npz",
        help="Phase 4 NPZ (default: datasets/hand_test_phase4_landmarks.npz)",
    )
    p.add_argument("--output-dir", type=Path, help="Output folder; defaults beside the NPZ")
    return p.parse_args()


def lengths_of(poses: np.ndarray) -> np.ndarray:
    return np.linalg.norm(poses[..., CHILDREN, :] - poses[..., PARENTS, :], axis=-1)


def project_lengths(pose: np.ndarray, lengths: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    out = np.empty_like(pose, dtype=np.float64)
    out[0] = pose[0]
    for j, parent in enumerate(PARENTS, start=1):
        direction = pose[j] - out[parent]
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm < 1e-9:
            direction = fallback[j - 1]
            norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction, norm = np.array([0.0, 1.0, 0.0]), 1.0
        out[j] = out[parent] + direction * (lengths[j - 1] / norm)
    return out


def alpha(cutoff: np.ndarray | float, dt: float):
    tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-5))
    return 1.0 / (1.0 + tau / max(float(dt), 1e-5))


def one_euro(raw: np.ndarray, detected: np.ndarray, times: np.ndarray,
             min_cutoff: float, beta: float) -> np.ndarray:
    out = np.full_like(raw, np.nan, dtype=np.float64)
    x_prev = dx_prev = t_prev = None
    for i in range(len(raw)):
        if not detected[i]:
            continue
        x = raw[i].astype(np.float64)
        if x_prev is None:
            out[i] = x
            x_prev, dx_prev, t_prev = x.copy(), np.zeros_like(x), float(times[i])
            continue
        dt = max(float(times[i] - t_prev), 1e-5)
        deriv = (x - x_prev) / dt
        ad = float(alpha(1.0, dt))
        dx = ad * deriv + (1.0 - ad) * dx_prev
        filt = alpha(min_cutoff + beta * np.abs(dx), dt) * x
        filt += (1.0 - alpha(min_cutoff + beta * np.abs(dx), dt)) * x_prev
        out[i] = filt
        x_prev, dx_prev, t_prev = filt, dx, float(times[i])
    return out


def contiguous_runs(mask: np.ndarray):
    ix = np.flatnonzero(mask)
    return [r for r in np.split(ix, np.flatnonzero(np.diff(ix) != 1) + 1) if len(r)]


def jitter_mm(stream: np.ndarray, detected: np.ndarray) -> float:
    parts = [np.diff(stream[r][:, TIPS, :], n=2, axis=0)
             for r in contiguous_runs(detected) if len(r) >= 3]
    if not parts:
        return float("nan")
    d2 = np.concatenate(parts, axis=0)
    return float(np.sqrt(np.mean(np.sum(d2 * d2, axis=-1))) * 1000.0)


def lag_ms(raw: np.ndarray, filt: np.ndarray, times: np.ndarray, detected: np.ndarray) -> float:
    shifts_ms = []
    for run in contiguous_runs(detected):
        if len(run) < 8:
            continue
        dt = float(np.median(np.diff(times[run])))
        if dt <= 0:
            continue
        max_shift = max(1, min(len(run) // 4, int(round(0.25 / dt))))
        for tip in TIPS:
            a = raw[run, tip].astype(float)
            b = filt[run, tip].astype(float)
            a -= a.mean(axis=0, keepdims=True)
            b -= b.mean(axis=0, keepdims=True)
            corr = []
            shifts = list(range(-max_shift, max_shift + 1))
            for shift in shifts:
                if shift < 0:
                    aa, bb = a[-shift:], b[:shift]
                elif shift > 0:
                    aa, bb = a[:-shift], b[shift:]
                else:
                    aa, bb = a, b
                den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
                corr.append(float(np.sum(aa * bb) / den) if den > 1e-12 else -1.0)
            shifts_ms.append(shifts[int(np.argmax(corr))] * dt * 1000.0)
    return float(np.median(shifts_ms)) if shifts_ms else float("nan")


def bone_cv_pct(stream: np.ndarray, detected: np.ndarray) -> float:
    valid = detected & np.isfinite(stream).all(axis=(1, 2))
    if valid.sum() < 2:
        return float("nan")
    b = lengths_of(stream[valid])
    return float(np.mean(b.std(axis=0) / np.maximum(b.mean(axis=0), 1e-9)) * 100.0)


def dropout_drift(prediction: np.ndarray, raw: np.ndarray, detected: np.ndarray) -> float:
    max_drift = 0.0
    valid_ids = np.flatnonzero(detected)
    for i in np.flatnonzero(~detected):
        previous = valid_ids[valid_ids < i]
        if len(previous) == 0 or not np.isfinite(prediction[i]).all():
            continue
        held = np.mean(raw[previous[-5:]], axis=0)
        max_drift = max(max_drift, float(np.max(np.linalg.norm(prediction[i] - held, axis=-1))) * 1000.0)
    return max_drift


def longest_dropout(detected: np.ndarray):
    runs = [r for r in contiguous_runs(~detected) if np.any(detected[:r[0]])]
    if not runs:
        return np.asarray([], dtype=int)
    return max(runs, key=len)


def debug_old_drift(old_state: np.ndarray | None, raw: np.ndarray, detected: np.ndarray,
                    times: np.ndarray, confidence: np.ndarray):
    run = longest_dropout(detected)
    if not len(run):
        print("\nDRIFT DEBUG: no detection-drop frames in this clip")
        return
    if old_state is None:
        print("\nDRIFT DEBUG: state_stream_m absent; reconstructing the old unbounded linear extrapolation")
        old_state = np.full_like(raw, np.nan, dtype=np.float64)
        valid = np.flatnonzero(detected)
        for i in valid:
            old_state[i] = raw[i]
        for i in run:
            before = valid[valid < i]
            if not len(before):
                continue
            last = int(before[-1])
            earlier = before[-2] if len(before) > 1 else None
            if earlier is not None:
                vel = (raw[last] - raw[earlier]) / max(times[last] - times[earlier], 1e-5)
                old_state[i] = raw[last] + vel * (times[i] - times[last])
            else:
                old_state[i] = raw[last]
    print(f"\nLONGEST DROPOUT: frames {run[0]}–{run[-1]} ({len(run)} frames; {times[run[-1]] - times[run[0]] + np.median(np.diff(times)):.3f} s)")
    print("Predicted fingertip positions are xyz mm in MediaPipe world coordinates; velocity is mm/s.")
    print(f"{'frame':>6} {'time s':>9} {'authority':>10} {'speed mm/s':>12}  fingertip xyz (mm): thumb, index, middle, ring, pinky")
    print("-" * 125)
    for i in run:
        if not np.isfinite(old_state[i]).all():
            continue
        dt = max(float(times[i] - times[i - 1]), 1e-5) if i else 1.0 / 30.0
        if i and np.isfinite(old_state[i - 1]).all():
            velocity = (old_state[i] - old_state[i - 1]) / dt
        else:
            velocity = np.zeros((21, 3))
        speed = float(np.max(np.linalg.norm(velocity[TIPS], axis=-1))) * 1000.0
        authority = float(confidence[i]) if detected[i] else 0.0
        tips_mm = old_state[i, TIPS] * 1000.0
        positions = np.array2string(tips_mm, precision=1, suppress_small=True, max_line_width=200)
        print(f"{i:6d} {times[i]:9.3f} {authority:10.3f} {speed:12.1f}  {positions}")
    print(
        "Cause: the original state code extrapolated the last two detected 21-joint poses "
        "at constant velocity, without damping or a per-joint speed cap, and allowed prediction "
        "for 0.75 seconds (about 22 frames at 30 FPS). During a dropout its measurement authority "
        "is zero. Small landmark noise in the two detections therefore becomes persistent "
        "velocity and can create visible drift. The new filter caps and damps velocity and holds "
        "after 8 missing frames."
    )


def tune_noise_prefix(raw: np.ndarray, detected: np.ndarray, times: np.ndarray):
    """Robustly estimate measurement and process noise from only opening 20%."""
    end = max(3, int(math.ceil(len(raw) * 0.20)))
    prefix = detected[:end]
    ids = np.flatnonzero(prefix)
    if len(ids) < 4:
        ids = np.flatnonzero(detected)[:max(4, min(int(detected.sum()), 20))]
    samples = raw[ids].astype(np.float64)
    # For white measurement noise, variance of the second difference is about 6 R.
    d2 = np.diff(samples, n=2, axis=0)
    measurement_var = float(np.nanmedian(d2 * d2) / 6.0) if d2.size else (0.004 ** 2)
    measurement_var = float(np.clip(measurement_var, 0.0005 ** 2, 0.03 ** 2))
    if len(samples) >= 3:
        dt = max(float(np.median(np.diff(times[ids]))), 1e-4)
        vel = np.diff(samples, axis=0) / dt
        accel = np.diff(vel, axis=0)
        process_var = float(np.nanmedian(accel * accel) * dt**4)
    else:
        process_var = measurement_var * 0.05
    process_var = float(np.clip(process_var, measurement_var * 0.005, measurement_var * 1.0))
    return process_var, measurement_var, end


def kalman_state(raw: np.ndarray, detected: np.ndarray, confidence: np.ndarray,
                 times: np.ndarray, lengths: np.ndarray, q_var: float, r_var: float):
    n = len(raw)
    out = np.full_like(raw, np.nan, dtype=np.float64)
    valid_ids = np.flatnonzero(detected)
    fallback = np.zeros((20, 3), dtype=np.float64)
    if len(valid_ids):
        first = raw[valid_ids[0]]
        vec = first[CHILDREN] - first[PARENTS]
        fallback = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-9)
    x = v = None
    p_pos = p_vel = None
    last_meas = None
    missing = 0
    max_v = MAX_SPEED_M_S
    for i in range(n):
        dt = max(float(times[i] - times[i - 1]), 1e-5) if i else (float(np.median(np.diff(times))) if n > 1 else 1/30)
        if x is None:
            if not detected[i]:
                continue
            x = raw[i].astype(np.float64).copy()
            v = np.zeros_like(x)
            p_pos = np.full((21, 1), max(r_var, 1e-8), dtype=np.float64)
            p_vel = np.full((21, 1), q_var / max(dt * dt, 1e-6), dtype=np.float64)
            last_meas = x.copy()
            out[i] = project_lengths(x, lengths, fallback)
            missing = 0
            continue

        x_pred = x + v * dt
        v_pred = v * DAMPING
        speed = np.linalg.norm(v_pred, axis=-1, keepdims=True)
        v_pred *= np.minimum(1.0, max_v / np.maximum(speed, 1e-9))
        p_pos = p_pos + (dt * dt) * p_vel + q_var
        p_vel = p_vel + q_var / max(dt * dt, 1e-6)

        if detected[i]:
            z = raw[i].astype(np.float64)
            meas_r = r_var / max(float(confidence[i]), 0.05)
            gain = p_pos / (p_pos + meas_r)
            residual = z - x_pred
            x = x_pred + gain * residual
            # A modest residual-derived velocity update; confidence enters through gain.
            v = v_pred + (0.25 * gain / dt) * residual
            speed = np.linalg.norm(v, axis=-1, keepdims=True)
            v *= np.minimum(1.0, max_v / np.maximum(speed, 1e-9))
            p_pos = (1.0 - gain) * p_pos
            p_vel = np.maximum((1.0 - 0.25 * gain) * p_vel, 1e-9)
            last_meas = z.copy()
            missing = 0
        else:
            missing += 1
            if missing <= MAX_DROPPED_FRAMES:
                x, v = x_pred, v_pred
            else:
                # Hold the last detected pose; velocity decays instead of integrating indefinitely.
                x = last_meas.copy()
                v *= 0.0
        out[i] = project_lengths(x, lengths, fallback)
    return out


def simple_dropout_baselines(raw: np.ndarray, detected: np.ndarray, times: np.ndarray):
    hold = np.full_like(raw, np.nan, dtype=np.float64)
    linear = np.full_like(raw, np.nan, dtype=np.float64)
    last = None
    prior = None
    last_t = prior_t = None
    for i, t in enumerate(times):
        if detected[i]:
            hold[i] = linear[i] = raw[i]
            prior, prior_t, last, last_t = last, last_t, raw[i].copy(), float(t)
        elif last is not None:
            hold[i] = last
            if prior is not None and prior_t is not None:
                vel = (last - prior) / max(last_t - prior_t, 1e-5)
                linear[i] = last + vel * (float(t) - last_t)
            else:
                linear[i] = last
    return hold, linear


def main():
    args = args_parse()
    npz_path = args.npz.expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"Phase 4 archive not found: {npz_path}")
    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path)
    required = ("world_landmarks_m", "detection_flag", "detection_confidence", "timestamp_seconds")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError("NPZ missing fields: " + ", ".join(missing))
    raw = data["world_landmarks_m"].astype(np.float64)
    detected = data["detection_flag"].astype(bool)
    confidence = data["detection_confidence"].astype(np.float64)
    times = data["timestamp_seconds"].astype(np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (21, 3) or len(times) != len(raw):
        raise ValueError("Expected (frames, 21, 3) world landmarks and one timestamp per frame")
    if not detected.any():
        raise RuntimeError("No detected frames in the archive")

    old_state = data["state_stream_m"].astype(np.float64) if "state_stream_m" in data else None
    debug_old_drift(old_state, raw, detected, times, confidence)

    # Calibrate per-edge lengths from up to 30 opening detections.
    ids = np.flatnonzero(detected)
    calibration = ids[:min(30, len(ids))]
    subject_lengths = np.median(lengths_of(raw[calibration]), axis=0)
    fallback_vec = raw[calibration[0], CHILDREN] - raw[calibration[0], PARENTS]
    fallback = fallback_vec / np.maximum(np.linalg.norm(fallback_vec, axis=-1, keepdims=True), 1e-9)

    process_var, measurement_var, prefix_end = tune_noise_prefix(raw, detected, times)
    print(f"\nPREFIX NOISE TUNING: frames 0–{prefix_end - 1} only")
    print(f"Estimated process variance: {process_var:.8g} m^2; measurement variance: {measurement_var:.8g} m^2")
    state = kalman_state(raw, detected, confidence, times, subject_lengths, process_var, measurement_var)

    table = []
    euro_streams = {}
    for cutoff in MIN_CUTOFFS:
        for beta in BETAS:
            stream = one_euro(raw, detected, times, cutoff, beta)
            name = f"1-Euro c={cutoff:g}, b={beta:g}"
            metrics = {
                "name": name,
                "stream": stream,
                "jitter": jitter_mm(stream, detected),
                "lag": lag_ms(raw, stream, times, detected),
                "bone_cv": bone_cv_pct(stream, detected),
                "drop_drift": dropout_drift(stream, raw, detected),
            }
            table.append(metrics)
            euro_streams[(cutoff, beta)] = metrics
    state_metrics = {
        "name": "Kalman state",
        "stream": state,
        "jitter": jitter_mm(state, detected),
        "lag": lag_ms(raw, state, times, detected),
        "bone_cv": bone_cv_pct(state, detected),
        "drop_drift": dropout_drift(state, raw, detected),
    }
    table.append(state_metrics)

    print("\nPARETO TABLE — jitter vs absolute lag; bone CV and dropout drift are also shown")
    print(f"{'Setting':<22} {'Jitter mm':>11} {'Lag ms':>10} {'Bone CV %':>11} {'Drop drift mm':>15}")
    print("-" * 73)
    for row in table:
        print(f"{row['name']:<22} {row['jitter']:11.3f} {row['lag']:10.3f} {row['bone_cv']:11.3f} {row['drop_drift']:15.3f}")

    feasible_euro = [r for r in table[:-1] if abs(r["lag"]) <= 20.0]
    feasible_state = [state_metrics] if abs(state_metrics["lag"]) <= 20.0 else []
    print("\nBEST SETTINGS AT |lag| <= 20 ms")
    print(f"{'Family':<16} {'Setting':<22} {'Jitter mm':>11} {'Lag ms':>10} {'Bone CV %':>11} {'Drop drift mm':>15}")
    print("-" * 89)
    for family, candidates in (("1-Euro", feasible_euro), ("state", feasible_state)):
        if candidates:
            best = min(candidates, key=lambda r: r["jitter"])
            print(f"{family:<16} {best['name']:<22} {best['jitter']:11.3f} {best['lag']:10.3f} {best['bone_cv']:11.3f} {best['drop_drift']:15.3f}")
        else:
            print(f"{family:<16} {'none meets lag limit':<22}")

    hold, linear = simple_dropout_baselines(raw, detected, times)
    print("\nDROPOUT DRIFT BASELINES (max joint displacement from last-5-detection hold)")
    print(f"Hold last pose:          {dropout_drift(hold, raw, detected):.3f} mm")
    print(f"Linear extrapolation:    {dropout_drift(linear, raw, detected):.3f} mm")

    fig, ax = plt.subplots(figsize=(9, 6))
    for row in table[:-1]:
        ax.scatter(row["lag"], row["jitter"], s=35, alpha=0.75, color="#2468a2")
        ax.annotate(row["name"].replace("1-Euro ", ""), (row["lag"], row["jitter"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.scatter(state_metrics["lag"], state_metrics["jitter"], marker="*", s=220,
               color="#c62828", label="Kalman state")
    ax.axvline(20.0, color="#777777", linestyle="--", linewidth=1, label="20 ms limit")
    ax.axvline(-20.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Lag vs raw (ms; absolute 20 ms eligibility limit)")
    ax.set_ylabel("Fingertip second-difference RMS jitter (mm)")
    ax.set_title("Temporal filter trade-off")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / f"{npz_path.stem}_phase4b_jitter_vs_lag.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    print(f"\nSaved jitter-vs-lag plot: {plot_path}")
    print(
        "Assumptions: NPZ world coordinates are meters, MediaPipe wrist is joint 0, "
        "and the standard 21-joint parent tree matches the stored joint order. Per-user "
        "bone lengths use medians from the first 30 detected frames. Process and "
        "measurement noise are estimated from second differences and acceleration in "
        "the first 20% of frames. Authority equals the MediaPipe handedness confidence "
        "on detections and zero on misses. The requested 20 ms criterion is applied to "
        "absolute lag magnitude."
    )


if __name__ == "__main__":
    main()
