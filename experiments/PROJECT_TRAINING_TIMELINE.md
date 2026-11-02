# Project training and experiment timeline

Status: 2026-11-02. This is a chronological reconstruction from the chat logs, local source tree, Google Drive run artifacts, and the saved result report. Many early events have no trustworthy timestamp, so phases are ordered by experiment rather than assigned invented dates.

## Executive state

The project has a useful end-to-end **training-data and model-training prototype**, not a working live tracker yet.

- The design goal is a small recurrent model that reads camera evidence plus a rendered MANO/FK prior and predicts bounded changes to the persistent hand state: translation, global orientation, and 21 joint angles. A separate 3D view should show the resulting hand state and uncertainty.
- The Phase 7 model was trained from scratch on the acquired ReInterHand capture data, not initialized from the earlier 5,114-frame direct regressor.
- Drive’s final quality report says **172,496 acquired images** have finite MANO training targets and no unusable images. They represent three captures, with HIR112 present only through the acquired prefix. The report checks joins, integrity, and sampled projections; it does not certify that every publisher-provided MANO fit is geometrically correct.
- The saved cache/index used by the Phase 7 run contains **171,680 records**, which is **816 fewer** than the quality report. The discrepancy is exactly the difference between HIR112’s 8,581 quality-report frames and the index’s 7,765 HIR112 frames. The current precomputed cache therefore cannot be described as covering all 172,496 quality-checked images.
- The 20-epoch Phase 7 run finished and saved checkpoints. Its fixed 192-frame sampled validation was noisy: the best milestone was 25.59 mm versus 26.59 mm for the prior; the last Drive milestone was 26.46 mm versus the same 26.59 mm prior. This is a small held-out improvement, not proof of live tracking.
- The local webcam smoke test moved the simulated hand without visibly fitting it to the user’s hand. That outcome is consistent with the test’s neutral-pose startup and missing camera-to-hand initialization. It is not a valid live-tracker demonstration.
- Phase 7’s runtime contract is incomplete: 08_runtime/inference.py is empty. The working webcam smoke script imports PersistentHandTracker directly and never connects the Phase 7 model to hm_ex/main.py.
- The trained checkpoint is small as a file: Drive’s update_model.pt is 17,735,303 bytes (about 16.9 MiB). That is a checkpoint payload size, not a runtime-speed result or a pure model-only export.

The full numeric tables for Phases 0–2 are in [experiment_results_report.md](experiment_results_report.md). This timeline records the major results and adds the later MediaPipe, temporal-filter, data-preparation, and Phase 7 evidence.

## 1. Initial formulation and the 5,114-frame baseline

The early work used the Mugsy subset from ReInterHand: 5,114 frames, official camera 400485, 667 × 1,024 images, right-hand 3D keypoints selected to 21 joints, and a fixed split of 44 training segments / 12 validation segments. That split produced 4,246 train frames and 868 validation frames. Labels were root-relative camera-frame millimeters. The direct CNN used grayscale and Sobel channels with a 128 × 96 crop.

Phase 0 established that the loader and labels could support learning:

| Method | Split | MPJPE | Thumb | Index | Middle | Ring | Pinky |
|---|---|---:|---:|---:|---:|---:|---:|
| Mean pose | val | 44.71 | 67.28 | 63.48 | 61.68 | 64.56 | 67.58 |
| Nearest image-neighbor | val | 40.78 | 61.24 | 58.07 | 55.69 | 58.72 | 61.51 |
| Overfit-64 | train | 2.96 | 3.98 | 4.09 | 3.81 | 3.64 | 3.67 |
| WebcamAlignedLCF, epoch 20 | val | 21.52 | 35.71 | 33.76 | 29.14 | 26.45 | 29.06 |

The overfit test passed its under-3-mm threshold. The direct model’s complete epoch curve is in the results report; it reached 24.39 mm at epoch 10, 21.85 mm at epoch 16, and 21.52 mm at epoch 20. A separate prompt mentioned 24.38 mm as a comparison value; that is distinct from the supplied Phase 0 epoch-20 result.

This is a useful baseline, but it is a different model, data split, coordinate metric, and task from the later capture-held-out MANO-state model. Its 21.52-mm root-relative score cannot be directly ranked against Phase 7’s camera-space state/FK result.

## 2. Prior-conditioned refiner experiments

### First refiner

The Phase 1 network received a prior pose and rendered prior cues. On synthetic priors, one correction reduced the reported errors only slightly:

| Prior severity | Before | After one step |
|---:|---:|---:|
| 0.3 | 16.00 mm | 15.93 mm |
| 0.6 | 19.82 mm | 19.49 mm |
| 1.0 | 25.35 mm | 24.80 mm |

On realistic priors from the direct model, refinement made the result worse: 21.52 mm at iteration 0, then 21.76, 22.20, and 22.73 mm after iterations 1, 2, and 3. The synthetic noise was not representative of the real regressor’s errors.

### Out-of-fold and gated refiner

A later run trained four 20-epoch direct models to make segment-wise out-of-fold priors. Their held-out ensemble scored 24.79 mm. A 25-epoch gated refiner then produced 27.76, 33.42, and 39.80 mm after one, two, and three iterations, so repeated correction diverged.

A damped-update diagnostic tried alpha values 0.00, 0.05, 0.10, 0.20, 0.35, 0.50, and 1.00 over four steps. The best displayed result was only 24.77 mm at alpha 0.05, step 1, a 0.02-mm change from 24.79 mm. At alpha 1.00 the errors rose to 39.80 mm by step 3. The detailed 28-row table is in the results report. This experiment showed that an unconstrained residual refiner could not reliably correct the direct regressor; it did not establish a working iterative tracker.

## 3. Image-axis, channel, and MediaPipe diagnostics

The requested Phase 2 direct-regressor diagnosis trained the grayscale input for 20 epochs, then stopped with a NameError before the evaluation code ran. No valid grayscale MPJPE report, Sobel-only result, grayscale-plus-Sobel comparison, oracle-depth score, or per-segment table was produced from that attempt.

MediaPipe was used as a diagnostic baseline, not as the intended deployed pose estimator. Several cell versions failed because MediaPipe 1.1.1 exposed Tasks but not the legacy mp.solutions API; a bone-adjacency assertion also rejected the assumed ordering. The successful candidate audit reported 806 detected validation frames out of 868 (92.86%). The original crop alone supplied candidates on 764 frames.

| Diagnostic | Reported result |
|---|---:|
| MediaPipe 2D error, original crop, 764 frames | 26.16 px |
| 2D error, best original or horizontally mirrored crop, 806 frames | 23.18 px |
| 2D-nearest candidate MPJPE, 806 frames | 38.08 mm |
| 3D-oracle candidate MPJPE, 806 frames | 34.15 mm |
| 2D-nearest and 3D-oracle selected different candidate | 374/806 (46.40%) |
| PA-MPJPE, MediaPipe raw | 15.84 mm |
| PA-MPJPE, bone-scaled MediaPipe | 13.45 mm |
| PA-MPJPE, Sobel direct model | 7.88 mm |

For the 2D-nearest candidate, the eight MediaPipe world-coordinate sign tests gave: +++ 38.08 mm; ++- 64.36; +-+ 129.49; +-- 143.00; -++ 125.34; -+- 140.54; --+ 184.45; --- 194.92. These sign tests show that the positive XYZ convention was best among the tested choices, not that MediaPipe solved camera-space pose.

The requested overlay export did not complete: the script stopped on segment T0011 because it found zero detected frames there. Later standalone-cell versions also hit a missing Sobel model. No comparable CNN-vs-MediaPipe all-frame benchmark was completed in that cell.

## 4. Local video state-filter experiments

The local test video was datasets/hand_test.mp4, 1,821 frames at nominal 30.005 FPS. MediaPipe detected 1,587 frames (87.1%). The first run failed in 1-Euro tuning because displacement had shape (358, 5) and elapsed time had shape (358,); after that was fixed, state-stream creation failed because last_time was undefined. No final jitter/lag/Pareto report from Phase 4, 4b, or 4c was shared. Those scripts are diagnostics and are not used in Phase 7 training.

## 5. Shift from static Mugsy tests to sequence-level ReInterHand data

The project then expanded from the 5,114-frame subset to three acquired ReInterHand captures in Google Drive:

| Capture | Final QC images | Image bytes | QC status |
|---|---:|---:|---|
| 202107 participant0 two-hands | 100,815 | 80,648,325,484 | 100,815 verified; complete archive |
| BKS383 two-hands | 63,100 | 46,929,532,756 | 63,100 verified; 25,401 images repaired or written |
| HIR112 two-hands | 8,581 | 6,537,332,005 | Acquired prefix; one boundary image completed; full capture not acquired |

The final QC report sums to **172,496 images and 134,115,190,245 image bytes, about 124.9 GiB**. It says all acquired images have MANO training targets, no unusable images, and no frames with partially missing keypoint targets. Projection spot checks were positive-depth and in-frame for the listed cameras, with one BKS383 left-hand sample rate of 91.67% in-frame. The report’s stated scope was data integrity and joins; it explicitly did not prove every annotation fit is geometrically perfect, and it says the full HIR112 capture is outside the acquired scope.

The Drive runbook’s earlier archive audit had reported 132,273,892,122 bytes (123.19 GiB), before the final repair additions. The final QC image-byte total is larger. These values describe different acquisition snapshots and should not be mixed.

The lineage matters:
- The complete 5,114-frame Mugsy subset was the early model baseline.
- The full InterHand2.6M corpus was planned but was not used in the Phase 7 run.
- The current Phase 7 training used ReInterHand capture data and started with fresh weights, explicitly excluding the old 5,114-frame checkpoint.

## 6. Indexing, labels, and CPU-precomputed inputs

The first full index reported 171,680 unique records: 100,815 from 202107, 63,100 from BKS383, and 7,765 from the HIR112 prefix. The user-provided later preflight reported 172,496 audited image/label joins. The final QC report independently gives 8,581 HIR112 images. This leaves the 816-record index/cache discrepancy to resolve before any retraining. The saved all-indexed capture roots file also reports 171,680 unique records.

The target-building process initially spent hours on Drive file I/O and was interrupted by an Errno 5 Drive read error while checking a HIR112 MANO JSON. Later it reused indexed records and loaded a cached HandState target archive. The logs show that the input pipeline, rather than the model’s CUDA compute, was the main bottleneck. GPU memory was reported at about 0.11 GiB during one training trace; that is compatible with a compact network plus small active batches and does not mean the GPU was absent.

To move CPU decoding, rendering, and augmentation out of the CUDA loop, the Phase 7 cache builder staged the window inputs into Drive. It completed **425 shards**, **27,140 six-frame windows × 4 variants**, and reported completion at 10:12:04 total elapsed. The resulting path was under checkpoints/phase7_capture_holdout/precomputed_inputs/1e72186b7149de228ddd9cba. The precompute required roughly 10 hours 12 minutes because it read/cropped/rendered frames and saved shards through Drive; it was not model training.

## 7. Phase 7 temporal MANO-state training

The active model used a CNN image encoder plus GRU state update. Each time step accepted a 128 × 128 five-channel crop (RGB, rendered MANO silhouette, projected FK skeleton), 67 normalized state features, elapsed time, and recurrent memory. It predicted a bounded 27-D correction: 3 translation, 3 global rotation, and 21 joint-angle deltas, plus uncertainty and a learned authority gate.

Training generated priors close to the fitted target: translation noise standard deviation 18 mm, global rotation 7°, and each joint 9°, with correlated changes over the six-frame sequence. The supervised objective was pose-state residual plus uncertainty, gate, and a small smoothness term. The silhouette and skeleton were model inputs; the training objective did **not** include direct pixel, silhouette-overlap, or image-reprojection fitting loss. The design therefore learns local pose corrections from visual evidence when the starting MANO state is already close. It was not trained to discover an arbitrary hand from a neutral initialization.

The run used:
- Fresh model weights; no Phase 0/Phase 1 checkpoint warm start.
- Capture-level holdout with HIR112 as validation in the reported run.
- Six-frame windows, stride six.
- A completed CPU-rendered input cache, then a CUDA Tesla T4 trainer.
- 20 epochs and milestone checkpoints every approximately 10 GiB of cached training input read.
- A fixed sampled validation subset of 192 frames per milestone, not the full 7,765-frame validation index.
- Checkpoints saved to Drive, including latest.pt, best.pt, webcam_epoch_005/010/015/020.pt, milestone checkpoints, and the root update_model.pt deployment endpoint.

Drive’s milestones.jsonl contains 99 checkpoint-evaluation records spanning epoch 1 through 20 and 10.078 through 990.022 GiB of cumulative input reads. The 990 GiB is repeated training/cache traffic across epochs and variants, not 990 GiB of unique source data.

| Epoch | Best sampled val MPJPE | Last sampled val MPJPE |
|---:|---:|---:|
| 1 | 26.34 | 26.47 |
| 2 | 26.18 | 26.63 |
| 3 | 26.00 | 26.25 |
| 4 | 26.16 | 26.57 |
| 5 | 26.06 | 26.72 |
| 6 | 26.79 | 26.79 |
| 7 | 25.59 | 25.59 |
| 8 | 26.23 | 26.97 |
| 9 | 26.51 | 27.03 |
| 10 | 26.39 | 26.80 |
| 11 | 26.57 | 27.09 |
| 12 | 26.33 | 27.81 |
| 13 | 26.57 | 26.78 |
| 14 | 26.23 | 26.23 |
| 15 | 26.22 | 26.80 |
| 16 | 26.76 | 27.22 |
| 17 | 26.45 | 26.45 |
| 18 | 26.72 | 26.72 |
| 19 | 26.44 | 26.44 |
| 20 | 26.46 | 26.46 |

The best logged checkpoint was the 340.005-GiB milestone at epoch 7, step 357: prior 26.5884 mm, corrected 25.5885 mm, authority 0.8913, 192 sampled frames. The final 990.022-GiB milestone at epoch 20, step 391 was prior 26.5884 mm, corrected 26.4570 mm, authority 0.8839, also 192 frames. That is approximately 1.00 mm gain at the best point and 0.13 mm at the last point. A separate chat message reported 26.79 mm corrected; that conflicts with the Drive milestone log’s 26.4570 mm final value and may refer to a different checkpoint/evaluation point. Both values are retained as separate evidence until the exact checkpoint and evaluation output are matched.

No full_validation.json was present in the completed run folder. The 192-frame sampled result is therefore the completed evidence available here; it is not a complete score on all held-out frames. The validation/fitted-state floor was reported around 12.63 mm. Earlier training messages also noted loss/validation oscillation and checkpointing best milestones, so selecting the epoch-20 endpoint blindly is not justified.

## 8. Checkpoint locations and live inference status

Google Drive locations:
- Project root: Persistent Visual-to-MANO Hand Tracker.
- Dataset: 04_DATASET_STAGING/ReInterHand/01_extracted, with archives, manifests, checksums, and repair reports in sibling folders.
- Training scripts and bundles: 04_DATASET_STAGING/ReInterHand/03_training_ready.
- Checkpoints: 04_DATASET_STAGING/ReInterHand/03_training_ready/checkpoints.
- Phase 7 milestone logs/checkpoints: checkpoints/phase7_capture_holdout_cached.
- Earlier incomplete/cache-attempt folder: checkpoints/phase7_capture_holdout.
- Final image QC: 03_training_ready/FINAL_QUALITY_REPORT.json.
- Saved training source and CPU precompute files are mirrored into this repository under experiments/colab_sources.

Local copy:
- The reported downloaded update_model.pt exists at phase7_vram_bundle/update_model.pt and is 17,735,303 bytes.
- The actual Phase 7 bundle is under phase7_vram_bundle.
- The intended visualization shell is hm_ex/main.py, but that file renders synthetic motion and does not capture webcam frames or load the Phase 7 checkpoint.

The Phase 7 smoke test is experiments/colab_sources/phase7_bundle_snapshot/phase7_webcam_smoke.py. It initializes HandState.neutral at a user-provided default depth (0.52 m), assumes focal length is 0.90 × image width, centers the principal point, crops around the projected current simulated hand, and feeds RGB plus the simulated MANO silhouette and FK skeleton to PersistentHandTracker. It has no hand detector, no user-selected initialization, and no fitting/optimization step that can bring a distant initial pose onto the observed hand. Pressing q exits; r resets; [ and ] adjust the assumed depth.

The smoke test’s input resembles the training crop only when the simulated hand is already near the real hand and the approximate camera calibration is reasonable. A random-looking overlay is therefore expected when startup alignment fails. It does not demonstrate that weights are random. However, the weak/noisy sampled validation improvement means the checkpoint is not yet demonstrated to be a strong corrector either.

There is also no direct image-fitting objective in the training loss. A real bootstrapped tracker still needs:
1. calibrated camera intrinsics and a coarse initial root location/depth/orientation;
2. an image-based alignment step (for example, optimize rendered MANO mask/edges against the selected hand region, without using MediaPipe);
3. the learned temporal model to make bounded state updates after that alignment;
4. a live app that shows the camera view and a separate 3D hand/uncertainty view;
5. offline replay evaluation and live webcam testing with the same checkpoint and explicit tracking metrics.

## 9. What is currently working, incomplete, or broken

### Working evidence
- The Phase 0 keypoint loader, camera projection, segment split, and compact direct network learned a lower error than mean pose and nearest neighbor on its own split.
- The 64-sample overfit diagnostic passed.
- MediaPipe image/world landmarks and the Sobel baseline produced useful diagnostic comparisons.
- The acquired ReInterHand image subset has a Drive QC report with target joins and integrity results.
- The CPU cache completed all 425 shards and the GPU trainer completed 20 epochs with recoverable checkpoints.
- The Phase 7 model’s validation evaluator ran repeatedly and generated 99 milestone records.

### Incomplete or broken
- The Phase 7 artifact is not an app-level live tracker. The smoke test starts with a neutral pose and no initialization; the visualization shell is a synthetic simulator.
- 08_runtime/inference.py is empty, so there is no supported inference entrypoint.
- The current training loss has no explicit image-space silhouette or reprojection objective.
- The validation metric is based on 192 sampled frames per checkpoint. No full-validation report was found.
- The data count is inconsistent by 816 frames between quality report/preflight and record index/cache.
- The baseline and Phase 7 MPJPE values use different splits and definitions; the numbers should not be presented as a controlled comparison.
- The downloaded Phase 7 checkpoint’s local webcam output reportedly moved randomly and did not fit the hand. It has not passed a valid initialization-aware webcam test.
- The local environment’s PyTorch import previously failed with WinError 5 on c10.dll in the Phase 0 virtual environment, so local offline evaluation is not established by this report.
- The full HIR112 capture and the full InterHand2.6M corpus were not part of this training run.

## 10. Recommended next milestones

1. Freeze this run and preserve best.pt, latest.pt, update_model.pt, milestones.jsonl, and the source bundle.
2. Reconcile the 816 missing HIR112 index/cache records. Regenerate the record index, label cache, and window manifest together so counts and manifests match the final QC report.
3. Re-evaluate best.pt and update_model.pt on the same held-out windows, reporting prior versus corrected state and saving side-by-side image/GT/prior/corrected overlays. Do not use the epoch-20 endpoint by default.
4. Add an initialization-aware inference entrypoint that uses calibrated intrinsics and a hand ROI or manual coarse placement. Do not call it a full tracker until alignment works on a known test clip.
5. Add an image-fitting objective or alignment step, then measure capture-held-out validation and a recorded webcam clip.
6. Only if those gates show consistent gains should the project spend more compute or scale to additional captures / InterHand2.6M.

## Source files and artifacts

- Colab source snapshots: [colab_sources/README.md](colab_sources/README.md)
- Consolidated Phase 0–2 numerical report: [experiment_results_report.md](experiment_results_report.md)
- Phase 7 model and state trainer: experiments/colab_sources/phase7_bundle_snapshot/phase5_train_temporal.py
- Phase 7 CPU cache scripts: experiments/colab_sources/drive_2026-11/
- Phase 7 initial local test: experiments/colab_sources/phase7_bundle_snapshot/phase7_webcam_smoke.py
- Existing local phase scripts: phase0_sanity.py, phase1_refine.py, phase2_diagnose.py, phase2_oof_gated_refiner_colab.py, phase4_state_on_mediapipe.py, phase4b_state_fix.py, and phase4c_fair_compare.py
