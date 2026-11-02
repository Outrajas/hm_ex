"""Fair offline comparison of 1-Euro and Kalman-style hand filters.

Reads only the Phase 4 NPZ; it never opens a video or invokes MediaPipe.
Run from the repository root, for example:
    python experiments/phase4c_fair_compare.py datasets/hand_test_phase4_landmarks.npz
"""

from __future__ import annotations

import argparse
import math
import time
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
BETAS = (0.0, 1.0, 10.0, 100.0, 1000.0)  # Hz per (m/s); world coordinates are meters.
DAMPING = 0.9
MAX_SPEED_M_S = 2.5
MAX_DROPPED_FRAMES = 8
HIGH_Q_MULTIPLIER = 10.0
SEED = 20261020


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "npz", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "hand_test_phase4_landmarks.npz",
        help="Saved Phase 4 NPZ (default: datasets/hand_test_phase4_landmarks.npz)",
    )
    p.add_argument("--output-dir", type=Path, help="Output folder; defaults beside the NPZ")
    return p.parse_args()


def bone_lengths(poses):
    return np.linalg.norm(poses[..., CHILDREN, :] - poses[..., PARENTS, :], axis=-1)


def project_bones(pose, lengths, fallback):
    out = np.empty_like(pose, dtype=np.float64)
    out[0] = pose[0]
    for child, parent in enumerate(PARENTS, start=1):
        d = pose[child] - out[parent]
        norm = float(np.linalg.norm(d))
        if not math.isfinite(norm) or norm < 1e-9:
            d = fallback[child - 1]
            norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            d, norm = np.array([0., 1., 0.]), 1.0
        out[child] = out[parent] + d * (lengths[child - 1] / norm)
    return out


def contiguous_runs(mask):
    ix = np.flatnonzero(mask)
    return [r for r in np.split(ix, np.flatnonzero(np.diff(ix) != 1) + 1) if len(r)]


def jitter_mm(stream, valid):
    chunks = [np.diff(stream[r][:, TIPS], n=2, axis=0)
              for r in contiguous_runs(valid) if len(r) >= 3]
    if not chunks:
        return float("nan")
    d2 = np.concatenate(chunks, axis=0)
    return float(np.sqrt(np.mean(np.sum(d2 * d2, axis=-1))) * 1000.0)


def lag_subframe_ms(raw, candidate, times, valid):
    """Median lag from trajectory cross-correlation with parabolic peak refinement."""
    estimates = []
    for run in contiguous_runs(valid):
        if len(run) < 10:
            continue
        dt = float(np.median(np.diff(times[run])))
        if dt <= 0:
            continue
        maxlag = min(len(run) // 4, max(2, int(round(0.30 / dt))))
        for tip in TIPS:
            a = raw[run, tip].astype(np.float64)
            b = candidate[run, tip].astype(np.float64)
            a -= a.mean(axis=0, keepdims=True)
            b -= b.mean(axis=0, keepdims=True)
            shifts = np.arange(-maxlag, maxlag + 1)
            corr = np.empty(len(shifts), dtype=np.float64)
            for k, shift in enumerate(shifts):
                if shift < 0:
                    aa, bb = a[-shift:], b[:shift]
                elif shift > 0:
                    aa, bb = a[:-shift], b[shift:]
                else:
                    aa, bb = a, b
                # Normalize each coordinate independently before aggregating trajectories.
                numerator = np.sum(aa * bb)
                denominator = np.linalg.norm(aa) * np.linalg.norm(bb)
                corr[k] = numerator / denominator if denominator > 1e-12 else -1.0
            peak = int(np.argmax(corr))
            offset = 0.0
            if 0 < peak < len(corr) - 1:
                left, center, right = corr[peak - 1:peak + 2]
                curvature = left - 2.0 * center + right
                if abs(curvature) > 1e-12:
                    offset = float(np.clip(0.5 * (left - right) / curvature, -1.0, 1.0))
            estimates.append((shifts[peak] + offset) * dt * 1000.0)
    return float(np.median(estimates)) if estimates else float("nan")


def alpha(cutoff, dt):
    tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(float(dt), 1e-5))


def one_euro(raw, detected, times, min_cutoff, beta, return_adapt=False, timed=False):
    n = len(raw)
    out = np.full_like(raw, np.nan, dtype=np.float64)
    adaptive_frame = np.zeros(n, dtype=bool)
    latencies = np.zeros(n, dtype=np.float64) if timed else None
    xprev = dxprev = tprev = None
    for i in range(n):
        t0 = time.perf_counter_ns() if timed else 0
        if detected[i]:
            x = raw[i].astype(np.float64)
            if xprev is None:
                out[i] = x
                xprev, dxprev, tprev = x.copy(), np.zeros_like(x), float(times[i])
            else:
                dt = max(float(times[i] - tprev), 1e-5)
                deriv = (x - xprev) / dt
                ad = float(alpha(1.0, dt))
                dx = ad * deriv + (1.0 - ad) * dxprev
                cutoff = min_cutoff + beta * np.abs(dx)
                adaptive_frame[i] = bool(np.median(cutoff) > 2.0 * min_cutoff)
                af = alpha(cutoff, dt)
                y = af * x + (1.0 - af) * xprev
                out[i] = y
                xprev, dxprev, tprev = y, dx, float(times[i])
        if timed:
            latencies[i] = (time.perf_counter_ns() - t0) / 1e6
    if timed:
        return out, float(adaptive_frame[detected].mean()) if detected.any() else 0.0, latencies
    return (out, float(adaptive_frame[detected].mean()) if detected.any() else 0.0) if return_adapt else out


def estimate_noise(raw, detected, times):
    end = max(3, int(math.ceil(len(raw) * 0.20)))
    ids = np.flatnonzero(detected[:end])
    if len(ids) < 4:
        ids = np.flatnonzero(detected)[:min(20, int(detected.sum()))]
    s = raw[ids].astype(np.float64)
    d2 = np.diff(s, n=2, axis=0)
    r = float(np.nanmedian(d2 * d2) / 6.0) if d2.size else 1.6e-5
    r = float(np.clip(r, 0.0005**2, 0.03**2))
    if len(s) >= 3:
        dt = max(float(np.median(np.diff(times[ids]))), 1e-4)
        acceleration = np.diff(np.diff(s, axis=0) / dt, axis=0)
        q = float(np.nanmedian(acceleration * acceleration) * dt**4)
    else:
        q = r * 0.05
    q = float(np.clip(q, r * 0.005, r))
    return q, r, end


def kalman_filter(raw, detected, confidence, times, lengths, q_var, r_var,
                  project=False, timed=False):
    n = len(raw)
    out = np.full_like(raw, np.nan, dtype=np.float64)
    latency = np.zeros(n, dtype=np.float64) if timed else None
    ids = np.flatnonzero(detected)
    first_vec = raw[ids[0], CHILDREN] - raw[ids[0], PARENTS]
    fallback = first_vec / np.maximum(np.linalg.norm(first_vec, axis=-1, keepdims=True), 1e-9)
    x = v = p = pv = last_measurement = None
    missing = 0
    for i in range(n):
        t0 = time.perf_counter_ns() if timed else 0
        dt = max(float(times[i] - times[i - 1]), 1e-5) if i else (float(np.median(np.diff(times))) if n > 1 else 1/30)
        if x is None:
            if detected[i]:
                x = raw[i].astype(np.float64).copy()
                v = np.zeros_like(x)
                p = np.full((21, 1), max(r_var, 1e-8))
                pv = np.full((21, 1), q_var / max(dt * dt, 1e-6))
                last_measurement = x.copy()
                out[i] = project_bones(x, lengths, fallback) if project else x
            if timed:
                latency[i] = (time.perf_counter_ns() - t0) / 1e6
            continue

        xp = x + v * dt
        vp = v * 0.9
        speed = np.linalg.norm(vp, axis=-1, keepdims=True)
        vp *= np.minimum(1.0, 2.5 / np.maximum(speed, 1e-9))
        p = p + dt * dt * pv + q_var
        pv = pv + q_var / max(dt * dt, 1e-6)
        if detected[i]:
            z = raw[i].astype(np.float64)
            meas_r = r_var / max(float(confidence[i]), 0.05)
            gain = p / (p + meas_r)
            residual = z - xp
            x = xp + gain * residual
            v = vp + (0.25 * gain / dt) * residual
            speed = np.linalg.norm(v, axis=-1, keepdims=True)
            v *= np.minimum(1.0, 2.5 / np.maximum(speed, 1e-9))
            p = (1.0 - gain) * p
            pv = np.maximum((1.0 - 0.25 * gain) * pv, 1e-9)
            last_measurement = z.copy()
            missing = 0
        else:
            missing += 1
            if missing <= MAX_DROPPED_FRAMES:
                x, v = xp, vp
            else:
                x = last_measurement.copy()
                v *= 0.0
        out[i] = project_bones(x, lengths, fallback) if project else x
        if timed:
            latency[i] = (time.perf_counter_ns() - t0) / 1e6
    return (out, latency) if timed else out


def drift_metric(pred, truth, frames):
    errors = np.linalg.norm(pred[frames] - truth[frames], axis=-1) * 1000.0
    return float(errors.mean()), float(errors[-1].mean())


def synthetic_runs(detected, times, lengths=(3, 8, 15, 30), count=50):
    rng = np.random.default_rng(SEED)
    eligible = np.arange(int(0.20 * len(detected)), len(detected))
    allowed = np.zeros(len(detected), bool)
    allowed[eligible] = detected[eligible]
    spans = contiguous_runs(allowed)
    result = {}
    for run_len in lengths:
        starts = [int(span[j]) for span in spans for j in range(len(span) - run_len + 1)
                  if np.all(np.diff(span[j:j + run_len]) == 1)]
        if not starts:
            result[run_len] = []
            continue
        result[run_len] = [np.arange(s, s + run_len, dtype=int)
                           for s in rng.choice(starts, size=count, replace=len(starts) < count)]
    return result


def select_tuned_euro(raw, detected, times):
    end = max(3, int(math.ceil(0.20 * len(raw))))
    prefix_valid = detected.copy()
    prefix_valid[end:] = False
    rows = []
    for cutoff in MIN_CUTOFFS:
        for beta in BETAS:
            filt, adapt_frac = one_euro(raw, detected, times, cutoff, beta, return_adapt=True)
            jitter = jitter_mm(filt, prefix_valid)
            lag = lag_subframe_ms(raw, filt, times, prefix_valid)
            rows.append((cutoff, beta, jitter, lag, adapt_frac))
    feasible = [r for r in rows if np.isfinite(r[3]) and abs(r[3]) <= 50]
    chosen = min(feasible or rows, key=lambda r: r[2])
    return chosen, rows


def evaluate_synthetic(raw, detected, confidence, times, lengths, q, r, euro_params):
    trials = synthetic_runs(detected, times)
    print("\nSYNTHETIC DROPOUTS — errors against the actual hidden detections; mm")
    print(f"{'Run':>5} {'Runs':>5} {'Method':<22} {'Mean error':>12} {'Final error':>12}")
    print("-" * 62)
    for run_len, frame_runs in trials.items():
        aggregate = {name: [[], []] for name in (
            "hold-last", "1-Euro hold", "Kalman damped", "Kalman + bone projection")}
        for frames in frame_runs:
            masked = detected.copy()
            masked[frames] = False
            euro = one_euro(raw, masked, times, euro_params[0], euro_params[1])
            k_plain = kalman_filter(raw, masked, confidence, times, lengths, q, r, project=False)
            k_projected = kalman_filter(raw, masked, confidence, times, lengths, q, r, project=True)
            hold_last = np.full_like(raw, np.nan, dtype=np.float64)
            euro_hold = np.full_like(raw, np.nan, dtype=np.float64)
            last_raw = last_euro = None
            for i in range(len(raw)):
                if masked[i]:
                    last_raw = raw[i]
                    last_euro = euro[i]
                elif last_raw is not None:
                    hold_last[i] = last_raw
                    euro_hold[i] = last_euro
            for name, pred in (("hold-last", hold_last), ("1-Euro hold", euro_hold),
                               ("Kalman damped", k_plain), ("Kalman + bone projection", k_projected)):
                mean_err, final_err = drift_metric(pred, raw, frames)
                aggregate[name][0].append(mean_err)
                aggregate[name][1].append(final_err)
        if not frame_runs:
            print(f"{run_len:5d} {0:5d} {'no eligible stretch':<22} {'n/a':>12} {'n/a':>12}")
            continue
        for name, (means, finals) in aggregate.items():
            print(f"{run_len:5d} {len(frame_runs):5d} {name:<22} {np.mean(means):12.2f} {np.mean(finals):12.2f}")


def latency_stats(fn, n):
    # The filter returns one timing per input frame; discard the first frame as initialization.
    values = fn()
    values = values[1:] if len(values) > 1 else values
    return float(np.median(values)), float(np.percentile(values, 95))


def main():
    args = parse_args()
    path = args.npz.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NPZ not found: {path}")
    outdir = args.output_dir.expanduser().resolve() if args.output_dir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    archive = np.load(path)
    required = ("world_landmarks_m", "detection_flag", "detection_confidence", "timestamp_seconds")
    missing = [k for k in required if k not in archive]
    if missing:
        raise KeyError("NPZ missing required arrays: " + ", ".join(missing))
    raw = archive["world_landmarks_m"].astype(np.float64)
    detected = archive["detection_flag"].astype(bool)
    confidence = archive["detection_confidence"].astype(np.float64)
    times = archive["timestamp_seconds"].astype(np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (21, 3) or not detected.any():
        raise ValueError("Expected detected MediaPipe world landmarks shaped (frames, 21, 3)")

    calibration_ids = np.flatnonzero(detected)[:min(30, int(detected.sum()))]
    subject_lengths = np.median(bone_lengths(raw[calibration_ids]), axis=0)
    q, r, prefix_end = estimate_noise(raw, detected, times)
    print(f"Loaded {len(raw)} frames from {path}; detections={int(detected.sum())}/{len(raw)}")
    print(f"Noise estimated on opening prefix frames 0–{prefix_end - 1}: process={q:.7g} m^2, measurement={r:.7g} m^2")

    tuned, prefix_rows = select_tuned_euro(raw, detected, times)
    print("\n1-EURO PREFIX TUNING (first 20% only; selected setting minimizes prefix jitter among |lag| <= 50 ms)")
    print(f"Selected min_cutoff={tuned[0]:g} Hz, beta={tuned[1]:g} Hz/(m/s); prefix jitter={tuned[2]:.3f} mm, lag={tuned[3]:.1f} ms")

    rows = []
    euro_result = {}
    for cutoff in MIN_CUTOFFS:
        for beta in BETAS:
            stream, adapt_frac = one_euro(raw, detected, times, cutoff, beta, return_adapt=True)
            row = {"family": "1-Euro", "name": f"1E c{cutoff:g} b{beta:g}", "jitter": jitter_mm(stream, detected),
                   "lag": lag_subframe_ms(raw, stream, times, detected), "adapt": adapt_frac,
                   "bone_cv": np.mean(np.std(bone_lengths(stream[detected]), axis=0) /
                                      np.maximum(np.mean(bone_lengths(stream[detected]), axis=0), 1e-9)) * 100.0}
            rows.append(row)
            euro_result[(cutoff, beta)] = stream
    state = kalman_filter(raw, detected, confidence, times, subject_lengths, q, r, project=True)
    high_state = kalman_filter(raw, detected, confidence, times, subject_lengths, q * HIGH_Q_MULTIPLIER, r, project=True)
    for name, stream, q_used in (("Kalman q1x", state, q), (f"Kalman q{HIGH_Q_MULTIPLIER:g}x", high_state, q * HIGH_Q_MULTIPLIER)):
        bonevals = bone_lengths(stream[detected])
        rows.append({"family": "Kalman", "name": name, "jitter": jitter_mm(stream, detected),
                     "lag": lag_subframe_ms(raw, stream, times, detected), "adapt": float("nan"),
                     "bone_cv": np.mean(np.std(bonevals, axis=0) / np.maximum(np.mean(bonevals, axis=0), 1e-9)) * 100.0,
                     "q": q_used})

    print("\nPARETO TABLE — full clip; lag is sub-frame cross-correlation (ms)")
    print(f"{'Setting':<17} {'Jitter mm':>10} {'Lag ms':>9} {'Adaptive >2x':>14} {'Bone CV %':>11}")
    print("-" * 65)
    for row in rows:
        adapt = f"{row['adapt'] * 100:12.1f}%" if np.isfinite(row["adapt"]) else f"{'—':>13}"
        print(f"{row['name']:<17} {row['jitter']:10.3f} {row['lag']:9.1f} {adapt:>14} {row['bone_cv']:11.3f}")

    print("\nMATCHED-LAG JITTER (linear interpolation within each family’s measured lag range)")
    print(f"{'Target lag ms':>14} {'1-Euro jitter mm':>18} {'Kalman jitter mm':>18}")
    print("-" * 54)
    for target in (15.0, 30.0, 50.0):
        vals = []
        for family in ("1-Euro", "Kalman"):
            group = [r for r in rows if r["family"] == family and np.isfinite(r["lag"]) and np.isfinite(r["jitter"])]
            # Collapse duplicate lag values before interpolation.
            by_lag = {}
            for rrow in group:
                by_lag.setdefault(round(float(rrow["lag"]), 3), []).append(float(rrow["jitter"]))
            xs = np.asarray(sorted(by_lag), dtype=float)
            ys = np.asarray([np.mean(by_lag[x]) for x in xs], dtype=float)
            if len(xs) >= 2 and xs[0] <= target <= xs[-1]:
                vals.append(f"{np.interp(target, xs, ys):.3f}")
            else:
                vals.append("n/a")
        print(f"{target:14.1f} {vals[0]:>18} {vals[1]:>18}")

    evaluate_synthetic(raw, detected, confidence, times, subject_lengths, q, r, (tuned[0], tuned[1]))

    # Per-frame Python CPU cost for the selected 1-Euro setting and both Kalman variants.
    _, _, euro_latency = one_euro(raw, detected, times, tuned[0], tuned[1], timed=True)
    _, state_latency = kalman_filter(raw, detected, confidence, times, subject_lengths, q, r, project=True, timed=True)
    _, high_latency = kalman_filter(raw, detected, confidence, times, subject_lengths, q * HIGH_Q_MULTIPLIER, r, project=True, timed=True)
    print("\nFILTER LATENCY — Python CPU per frame")
    print(f"{'Filter':<22} {'Median ms':>11} {'p95 ms':>10}")
    print("-" * 45)
    for name, values in (("1-Euro selected", euro_latency), ("Kalman q1x", state_latency), (f"Kalman q{HIGH_Q_MULTIPLIER:g}x", high_latency)):
        print(f"{name:<22} {np.median(values):11.4f} {np.percentile(values, 95):10.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    euro_rows = [rrow for rrow in rows if rrow["family"] == "1-Euro"]
    ax.scatter([rrow["lag"] for rrow in euro_rows], [rrow["jitter"] for rrow in euro_rows],
               c=[BETAS.index(float(rrow["name"].split("b")[-1])) for rrow in euro_rows],
               cmap="viridis", s=45, label="1-Euro settings")
    for rrow in rows:
        if rrow["family"] == "Kalman":
            ax.scatter(rrow["lag"], rrow["jitter"], marker="*", s=220, label=rrow["name"])
    ax.set_xlabel("Sub-frame lag vs raw (ms)")
    ax.set_ylabel("Fingertip second-difference RMS jitter (mm)")
    ax.set_title("Phase 4c filter Pareto comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plot_path = outdir / f"{path.stem}_phase4c_pareto.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    print(f"\nSaved Pareto plot: {plot_path}")
    print(
        "Assumptions: NPZ world landmarks are meters; MediaPipe's standard 21-joint "
        "wrist/thumb/index/middle/ring/pinky order is used. Beta units are Hz/(m/s), "
        "because coordinates are meters and derivatives are meters/second. Lag is the "
        "median across five fingertip xyz trajectories and contiguous detected runs; "
        "positive lag means the filter trails raw. Adaptive-cutoff fraction counts "
        "detected frames whose median coordinate cutoff exceeds 2x min_cutoff. Synthetic "
        "runs are sampled with replacement when fewer than 50 eligible starts exist, "
        "and the final-frame error is averaged over joint mean distances at each run's "
        "last masked frame."
    )


if __name__ == "__main__":
    main()
