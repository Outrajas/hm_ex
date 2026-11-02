# Hand Pose Experiments: Results So Far

This report consolidates the experiment outputs and run diagnostics shared in the Codex conversation. It preserves the reported numerical results. Numbers that were not present in the supplied outputs are identified as unreported rather than estimated.

## 1. Dataset and evaluation setup

- Dataset: Mugsy subset, camera `400485`, `5,114` frames.
- Image dimensions reported by the successful Colab run: `667 × 1024` pixels.
- Segment split: `44` training segments and `12` validation segments, fixed seed `20251018`.
- Successful OOF run frame split: `4,246` training frames and `868` validation frames.
- Keypoints: right-hand 3D annotations, selected to `21` joints using `KEEP_21_INDICES`; output is root-relative camera-frame coordinates in mm.
- Image input for the direct model: grayscale and Sobel magnitude, `2` channels, crop size `128 × 96` (reported in the runner as `IMAGE_W=128`, `IMAGE_H=96`).
- Prior-conditioned refiner input: `4` channels: grayscale, Sobel magnitude, rendered prior skeleton lines, and rendered prior joint dots.
- Compute device for the successful later Colab run: CUDA, Tesla T4.
- Reported camera calibration file in that run: `/content/dataset/cam_params.json`.

## 2. Phase 0: baseline and direct-regressor results

The first reported Phase 0 run staged `5,114` frames in `115.0` seconds. The table below reproduces all reported validation metrics. MPJPE and fingertip MPJPE are in mm.

| Method / checkpoint | Split or epoch | MPJPE | Thumb | Index | Middle | Ring | Pinky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean pose | val | 44.71 | 67.28 | 63.48 | 61.68 | 64.56 | 67.58 |
| Nearest neighbor | val | 40.78 | 61.24 | 58.07 | 55.69 | 58.72 | 61.51 |
| Overfit-64 | train | 2.96 | 3.98 | 4.09 | 3.81 | 3.64 | 3.67 |
| WebcamAlignedLCF | epoch 02 | 31.97 | 48.98 | 40.12 | 42.15 | 45.29 | 50.02 |
| WebcamAlignedLCF | epoch 04 | 32.83 | 52.28 | 45.11 | 44.96 | 43.60 | 48.07 |
| WebcamAlignedLCF | epoch 06 | 29.65 | 47.95 | 40.59 | 35.47 | 35.38 | 39.68 |
| WebcamAlignedLCF | epoch 08 | 26.63 | 46.11 | 38.19 | 32.63 | 32.61 | 38.16 |
| WebcamAlignedLCF | epoch 10 | 24.39 | 43.35 | 36.45 | 29.73 | 28.69 | 33.97 |
| WebcamAlignedLCF | epoch 12 | 23.94 | 39.21 | 37.21 | 31.90 | 28.25 | 29.49 |
| WebcamAlignedLCF | epoch 14 | 23.39 | 39.63 | 35.98 | 30.62 | 29.16 | 30.81 |
| WebcamAlignedLCF | epoch 16 | 21.85 | 37.01 | 34.02 | 29.19 | 26.70 | 28.65 |
| WebcamAlignedLCF | epoch 18 | 21.94 | 36.02 | 35.54 | 30.59 | 27.54 | 28.12 |
| WebcamAlignedLCF | epoch 20 | **21.52** | 35.71 | 33.76 | 29.14 | 26.45 | 29.06 |

The Overfit-64 diagnostic was explicitly marked `PASS` at `2.96 mm` train MPJPE, below its `3 mm` threshold. A separate prompt described `24.38 mm` as the direct-regressor comparison value; the supplied Phase 0 epoch-20 output instead reports `21.52 mm`. These are distinct reported values and should not be conflated.

## 3. Phase 1: prior-conditioned refinement

The initial Phase 1 run used the Phase 0 direct model and evaluated one through three repeated refinement steps on held-out segments. Synthetic priors used severities `0.3`, `0.6`, and `1.0`. All errors below are in mm.

### Synthetic priors

| Prior severity | Before MPJPE | After one step MPJPE | Thumb | Index | Middle | Ring | Pinky |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 16.00 | 15.93 | 16.45 | 18.31 | 18.90 | 18.66 | 17.33 |
| 0.6 | 19.82 | 19.49 | 21.22 | 24.21 | 24.81 | 23.81 | 21.36 |
| 1.0 | 25.35 | 24.80 | 27.80 | 32.16 | 34.73 | 32.18 | 28.31 |

### Realistic priors from the direct model

| Refinement iteration | MPJPE | Thumb | Index | Middle | Ring | Pinky |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21.52 | 35.71 | 33.76 | 29.14 | 26.45 | 29.06 |
| 1 | 21.76 | 35.73 | 34.41 | 29.31 | 26.78 | 29.39 |
| 2 | 22.20 | 35.78 | 35.50 | 29.87 | 27.32 | 29.99 |
| 3 | 22.73 | 35.84 | 36.81 | 30.69 | 27.97 | 30.64 |

The synthetic-prior improvements were small (`0.07`, `0.33`, and `0.55 mm` respectively). On the realistic direct-model prior, refinement worsened MPJPE at every iteration, from `21.52 mm` to `22.73 mm` after iteration `3`.

The runner reported saving `/content/phase1_combined_tracker.pt`. In the later successful Colab session, checkpoint discovery printed that no old checkpoint was found under `/content`, the current folder, or Drive; the old model was therefore not evaluated in that session.

## 4. OOF direct-regressor and gated-refiner experiment

The next successful Colab run used a Tesla T4 and loaded `5,114` camera-400485 frames at `667 × 1024`. It split the data into `4,246` train and `868` validation frames, grouped by `44` train and `12` validation segments.

### Out-of-fold direct models

Four fold models were trained for `20` epochs each. Reported fold frame counts:

| Fold | Fold training frames | Held-out train-fold frames |
|---:|---:|---:|
| 1 | 3,229 | 1,017 |
| 2 | 3,349 | 897 |
| 3 | 3,264 | 982 |
| 4 | 2,896 | 1,350 |

The validation prior was the mean of the four fold-model predictions:

| Method | Validation MPJPE | Thumb | Index | Middle | Ring | Pinky |
|---|---:|---:|---:|---:|---:|---:|
| OOF direct ensemble | 24.79 | 43.67 | 36.62 | 31.81 | 30.74 | 33.78 |

This is `3.27 mm` higher than the earlier reported Phase 0 epoch-20 direct result of `21.52 mm`. The OOF ensemble and the earlier all-training-segment direct model use different training constructions, so this is not a controlled same-model comparison.

### Gated-refiner validation

The gated refiner trained for `25` epochs. The result table reports the unmodified OOF prior and then three recropped refinement steps:

| Iteration | MPJPE | Thumb | Index | Middle | Ring | Pinky |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24.79 | 43.67 | 36.62 | 31.81 | 30.74 | 33.78 |
| 1 | 27.76 | 46.47 | 39.93 | 34.09 | 35.29 | 37.77 |
| 2 | 33.42 | 51.15 | 45.61 | 40.54 | 43.48 | 48.08 |
| 3 | 39.80 | 56.06 | 51.65 | 47.77 | 53.69 | 60.63 |

The first correction worsened MPJPE by `2.97 mm`; the third worsened it by `15.01 mm` relative to iteration `0`. Every reported fingertip error rose across repeated iterations.

### Latency and saved checkpoint

- Median batch-1 latency on CUDA: one refiner step, `1.923 ms`.
- Median batch-1 latency on CUDA: direct ensemble plus two refinements, `13.843 ms`.
- Saved checkpoint: `/content/phase2_oof_gated_refiner.pt`.
- The run printed all `25` epoch completions.

## 5. Damped-correction diagnostic

After the gated-refiner run, a diagnostic tested correction scales `alpha = 0.00`, `0.05`, `0.10`, `0.20`, `0.35`, `0.50`, and `1.00`, for steps `0` through `3`. Step `0` is the unmodified OOF prior and is identical for all alpha values. All errors are in mm; each fingertip entry is its per-tip MPJPE.

| Alpha | Step | MPJPE | Thumb | Index | Middle | Ring | Pinky |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 24.79 | 43.67 | 36.62 | 31.81 | 30.74 | 33.78 |
| 0.00 | 1 | 24.79 | 43.67 | 36.62 | 31.81 | 30.74 | 33.78 |
| 0.00 | 2 | 24.79 | 43.67 | 36.62 | 31.81 | 30.74 | 33.78 |
| 0.00 | 3 | 24.79 | 43.67 | 36.62 | 31.81 | 30.74 | 33.78 |
| 0.05 | 0 | 24.79 | 43.67 | 36.62 | 31.81 | 30.74 | 33.78 |
| 0.05 | 1 | 24.77 | 43.61 | 36.56 | 31.76 | 30.72 | 33.71 |
| 0.05 | 2 | 24.78 | 43.57 | 36.53 | 31.72 | 30.72 | 33.68 |
| 0.05 | 3 | 24.80 | 43.55 | 36.52 | 31.70 | 30.76 | 33.68 |
| 0.10 | 0 | 24.79 | 43.67 | 36.62 | 31.81 | 30.74 | 33.78 |
| 0.10 | 1 | 24.78 | 43.57 | 36.53 | 31.72 | 30.72 | 33.67 |
| 0.10 | 2 | 24.83 | 43.54 | 36.53 | 31.70 | 30.82 | 33.71 |
| 0.10 | 3 | 24.95 | 43.59 | 36.61 | 31.74 | 31.04 | 33.86 |
| 0.20 | 0 | 24.79 | 43.67 | 36.62 | 31.81 | 30.74 | 33.78 |
| 0.20 | 1 | 24.83 | 43.54 | 36.53 | 31.69 | 30.81 | 33.68 |
| 0.20 | 2 | 25.13 | 43.71 | 36.79 | 31.85 | 31.35 | 34.11 |
| 0.20 | 3 | 25.66 | 44.15 | 37.32 | 32.22 | 32.24 | 34.96 |
| 0.35 | 0 | 24.79 | 43.67 | 36.62 | 31.81 | 30.74 | 33.78 |
| 0.35 | 1 | 25.04 | 43.64 | 36.71 | 31.78 | 31.14 | 33.92 |
| 0.35 | 2 | 26.04 | 44.52 | 37.76 | 32.50 | 32.82 | 35.52 |
| 0.35 | 3 | 27.54 | 45.93 | 39.34 | 33.82 | 35.05 | 37.98 |
| 0.50 | 0 | 24.79 | 43.67 | 36.62 | 31.81 | 30.74 | 33.78 |
| 0.50 | 1 | 25.42 | 43.94 | 37.12 | 32.03 | 31.72 | 34.41 |
| 0.50 | 2 | 27.38 | 45.88 | 39.23 | 33.61 | 34.80 | 37.65 |
| 0.50 | 3 | 29.99 | 47.99 | 42.03 | 36.52 | 38.53 | 42.20 |
| 1.00 | 0 | 24.79 | 43.67 | 36.62 | 31.81 | 30.74 | 33.78 |
| 1.00 | 1 | 27.76 | 46.47 | 39.93 | 34.09 | 35.29 | 37.77 |
| 1.00 | 2 | 33.42 | 51.15 | 45.61 | 40.54 | 43.48 | 48.08 |
| 1.00 | 3 | 39.80 | 56.06 | 51.65 | 47.77 | 53.69 | 60.63 |

The lowest displayed overall error is `24.77 mm` at `alpha=0.05`, step `1`, a `0.02 mm` reduction from `24.79 mm`. At `alpha=0.05`, step `3` is `24.80 mm`; at `alpha=0.10`, step `3` is `24.95 mm`. The small one-step differences do not establish a meaningful improvement; stronger and repeated corrections worsen the result.

## 6. Execution and debugging history

These were run/debugging events, not additional model-quality measurements:

1. An earlier local Phase 1 run was stopped after approximately `18 minutes 30 seconds`; no completed result was reported from that run.
2. A first Colab asset lookup failed on `orig_fits/right/Keypoints/keypoint-287312.json` because the resolver searched from the subset directory rather than resolving the capture-root-relative path.
3. The next resolver found the flattened `keypoints_left` and `keypoints_right` exports but treated same-basename left/right keypoint files as ambiguous. The successful run then loaded all `5,114` records, so this path issue was resolved for that run.
4. The first damped-diagnostic cell raised a broadcasting `ValueError`: `current` was shaped `(868, 21, 3)`, but indexing `apply_refiner(...)[0]` reduced each proposed pose to `(21, 3)` and the stacked array became incompatible. Removing `[0]` fixed the shape mismatch; the corrected diagnostic produced the `28` rows in Section 5.

## 7. Overall readout

- The direct regressor is the strongest demonstrated model result: `21.52 mm` in the first Phase 0 report.
- The OOF ensemble scored `24.79 mm` in the later setup.
- The learned realistic-prior refiner worsened the OOF result, and repeated corrections diverged.
- Damping to `alpha=0.05` produced a best displayed change of only `0.02 mm`, which is too small to claim a reliable gain from these results.
- No old checkpoint was available in the later Colab runtime, so its checkpoint-loading/evaluation branch did not produce a model score in that run.

## 8. Values not reported in the supplied outputs

The conversation did not include a numeric camera in-frame projection pass percentage, per-run total wall time for the successful Colab experiment, uncertainty calibration statistics, or a new trained checkpoint comparison using exactly the same `44`-segment training data as the OOF models. Those values are intentionally not inferred here.
