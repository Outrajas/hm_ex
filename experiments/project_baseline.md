# Persistent visual-to-MANO tracker: baseline and target

Updated 2026-10-21. This records the current starting point and the intended
system so later experiments use the same reference.

## Intended inference loop

The hand in the 3D view is the persistent state, not a one-frame pose guess:

1. Keep a state `X_t = (T, R, q, V, omega, qdot, beta)`. `T` is translation,
   `R` is global orientation, `q` is the articulated hand pose, and `beta` is
   the user's calibrated hand shape.
2. Render the state as a kinematic skeleton and a skinned hand surface, and
   project it into the camera view.
3. A visual model compares current RGB evidence with the predicted state and
   emits bounded `delta_T`, `delta_R`, and `delta_q`, with confidence or
   uncertainty. A recurrent state carries information across frames.
4. Fuse that correction into the persistent state. During occlusion, propagate
   the state and grow uncertainty; when visual evidence returns, use it to
   correct the estimate. Physical limits are priors and safety bounds, not a
   reason to force uncertain image evidence into an implausible pose.
5. Show the camera image with the fitted hand and a separate 3D view of the
   same state.

At runtime the model should use the camera image and its own previous state;
it should not depend on a hand-pose detector. Ground-truth skeletons and MANO
annotations are training/evaluation supervision, not runtime inputs.

## Locked V0 measurement

The strongest reported single-frame visual baseline is the Phase 0
`WebcamAlignedLCFTracker`: **21.52 mm validation MPJPE** on the Mugsy subset,
camera `400485`, using the fixed 44-train / 12-validation segment split
(4,246 / 868 frames). The input was a grayscale-plus-Sobel crop at 128 x 96;
the output was 21 root-relative 3D joints. Mean-pose baseline was 44.71 mm,
nearest-neighbor baseline 40.78 mm, and the 64-frame overfit diagnostic was
2.96 mm train MPJPE.

This is a joint-coordinate baseline, **not** yet a MANO-state or temporal
`delta_q` baseline. The reported realistic-prior refinement changed 21.52 mm
to 21.76, 22.20, and 22.73 mm after one, two, and three steps. A separate
out-of-fold run started at 24.79 mm and worsened with repeated corrections.
Those runs do not demonstrate a useful learned state update.

The Gemini discussion reports different 3-seed experiments: static / two-frame
/ frame-difference inputs at 31.23 / 31.87 / 32.38 mm; hard kinematic
projection at 31.54 mm before projection and 32.43 mm after; and soft anatomy
regularization at 31.32 mm baseline and 32.90 mm at the stronger penalty.
These are useful negative results for those particular setups, but their
protocol differs from the Phase 0 21.52 mm run. Keep the results separate until
the data, split, target, and evaluation are reproduced identically.

## Current implementation boundary

- `00_spec` and `02_state` define a 21-DOF hand state, state updates,
  constraints, and forward kinematics. The state update accepts `delta_q`.
- `02_state/mano_surface.py` skins a MANO template to that skeleton and now
  applies the 10D `HandState.beta` through the first ten MANO shape directions.
  It still does not convert project q values into the original 45D MANO pose;
  the mesh pose follows the project's FK skeleton, and `HandBiometrics` remains
  a separate calibrated skeleton-shape parameter.
- `live_webcam_test.py` has a camera UI and 3D view, but derives pose mostly
  from color/silhouette curl heuristics. The optional LCF model only adjusts
  those curl estimates; it does not predict a persistent image-conditioned
  `delta_q` sequence.
- The reported 5,114-frame Mugsy subset is appropriate for the locked V0 joint
  benchmark. Its frames are temporally subsampled, so it is not by itself a
  sufficient test of dense-video jitter reduction or occlusion recovery.

## Local data audit

- The available Mugsy manifest has 5,114 frames across 56 segments, all from
  one participant/capture. The derived image tree contains 5,115 image files.
- Within-segment numeric frame-ID gaps have median 9, 90th percentile 24, and
  maximum 441. These samples cannot stand in for contiguous 30 FPS video when
  teaching short-term dynamics.
- MANO pose/shape annotation files exist locally for this capture, but the
  selected RGB crop set is still only 5,114 frames. The InterHand2.6M inventory
  currently reports no downloaded annotations or images.
- Therefore this local copy is sufficient for an end-to-end plumbing check,
  but it is not a broad, data-rich training source. The training plan must
  stream curated multi-subject clips from the larger ReInterHand source and
  cache resumable shards/checkpoints outside the temporary Colab VM.

## First temporal model contract

`04_network/architecture.py` now defines `TemporalStateCorrector`: a CNN reads
the RGB crop plus the prior rendered silhouette and skeleton, a GRU carries
state across frames, and heads return bounded `delta_T`, `delta_R`, `delta_q`,
log uncertainty, and an authority gate. Its 67-D state input contains the
normalized current state, velocities, and fixed per-user shape. The resumable
trainer now exercises this contract on synthetic perturbed-state priors; a
realistic visual initializer and live integration remain to be built.

`06_training/reinterhand_sequences.py` now indexes frames without decoding a
capture into RAM, loads one RGB/MANO/keypoint record on demand, makes windows
within a segment only, and splits by whole captures or segments. It uses the
official per-segment frame list for dense render indexing. `06_training/targets.py`
adds an analytic inverse-kinematics mapping from the project's 21 ordered joint
labels into root pose and the project's 21 q values, with an FK residual per
frame. This is an initialization/target adapter, not a claim that Re:InterHand's
MANO shape is already represented by the project's current `HandBiometrics`.

The adapter has now been audited on the fixed local 44/12 split. Its synthetic
FK-to-IK round trip is **0.000000 mm**. With anthropometry calibrated only from
the 4,246 training frames, the project FK residual on 868 held-out frames is
**7.97 mm MPJPE with current joint limits** and **4.86 mm without clamping**.
Calibration was palm length 0.970, palm width 0.700, finger length 0.905, and
thumb length 0.866. About **32.15%** of unclamped validation angles fall outside
the current constraints, so training/runtime must account for the state-limit
gap rather than assume the labels are already physically projectable. These
are geometry-fit measurements, not neural tracking scores.

`08_runtime/tracker.py` now carries the GRU hidden state, applies bounded
corrections through the existing state transition, preserves shape, damps
velocity, and grows uncertainty on invalid frames. `08_runtime/render_input.py`
builds the detector-free five-channel camera crop from RGB plus the simulated
MANO silhouette and FK skeleton. A runtime audit passed: state update retained
beta, invalid-frame authority was 0, uncertainty increased, MANO loaded, and
the rendered input had shape 5 x 128 x 128. The audit used untrained weights;
it validates plumbing only, not pose accuracy or latency.
The bundled model stores 20 shape directions; the runtime uses its first ten to
match the available 10D Re:InterHand beta annotation. A beta value of 2.0 on the
first component changed mean mesh vertex position by 1.443 mm in the audit.

## First trained temporal-state diagnostic

`experiments/phase5_train_temporal.py` performs a resumable training run. It
reads image frames lazily, caches fitted targets, creates temporally smooth
synthetic state priors, renders each prior into the RGB + MANO-silhouette +
FK-skeleton input, augments occlusions, and trains the CNN-GRU to predict
bounded `delta_T`, `delta_R`, `delta_q`, uncertainty, and authority. It saves
`latest.pt` every 25 optimizer steps and after every epoch, including optimizer,
scheduler, RNG, and batch offset for mid-epoch recovery; it also maintains
`best.pt`. The reader supports both the original capture tree and the flattened
`/content/dataset` layout, with optional MANO parameter sidecars. The separate
`experiments/phase5_eval_temporal.py` evaluates each validation frame once and
writes `full_validation.json`.

On the local CPU-only environment, 5,114 frames from one capture completed 20
epochs in about 30 minutes. The fixed 192-frame epoch sample selected epoch 16
at 21.34 mm camera-space MPJPE versus 22.16 mm for its synthetic prior. The
full 868-frame pass gave:

- Root-relative FK MPJPE: 16.91 mm prior, 16.45 mm corrected (0.46 mm gain).
- Camera-space FK MPJPE: 23.57 mm prior, 22.50 mm corrected (1.07 mm gain).
- Mean corrected wrist-position error: 14.61 mm.
- Mean target IK/FK residual: 7.97 mm; mean learned authority: 0.888.

This is only a synthetic-prior correction diagnostic: every evaluation prior
starts from the fitted ground-truth state plus generated noise. It does not
show that the network can initialize from an unconditioned webcam image and it
is not a fair comparison with the Phase 0 direct regressor's 21.52 mm score.
The validation set is from the same participant/capture, with whole segments
held out. The 0.46 mm root-relative gain is weak evidence; the next experiment
must train/evaluate against out-of-fold visual priors and unseen captures.

## Schedule estimate and data threshold

These are engineering estimates, not a promise about free-tier GPU availability:

- **About 1–2 weeks** for a usable local-camera MVP: sequence-safe data windows,
  a calibrated subject shape, a first image-conditioned state initializer,
  recurrent `delta_q` training, and the camera/3D demo. It should work on the
  current participant and selected capture, but will not be broad-generalizing.
- **About 4–8 weeks** for a credible data-rich prototype if we can stage a
  curated multi-capture subset and accumulate cloud training time: target
  roughly 50k–200k aligned frames from at least 20 captures/participants, with
  dense sequences reserved for temporal training and captures held out for
  evaluation. Train in resumable shards and save checkpoints outside each
  ephemeral VM.
- **Longer than that** if every 1.2 TB asset must be downloaded, unpacked, or
  mapped before any training. That is not necessary for the first useful model.

The official Re:InterHand release separates image modes and annotation assets;
its site describes MANO fits and original fits as separate downloads, while the
official InterWild reader notes that Mugsy `envmap_per_frame` is 5 FPS and the
other setting uses the dense frame list. We should train the temporal model on
dense `envmap_per_segment` clips, and use sparse relighting/per-frame imagery
for appearance diversity where it fits the sequence protocol.

The IK adapter residual gate and a first recurrent train/eval loop now pass.
The remaining high-priority gate is a realistic prior: generate direct-regressor
predictions out of fold on training segments, fit those into `HandState`, train
the recurrent model on those priors, and test on the held-out segments. Then
repeat with held-out captures. Do not interpret this synthetic-prior result or
the small-capture geometry fit as evidence that the model generalizes across
people.

## First end-to-end milestones

1. Preserve one reproducible V0 score and its exact segment split.
2. Verify the sequence timestamps and label mapping; form windows only within
   each chronological segment and split whole segments between train and val.
3. Establish a reliable mapping between ReInterHand MANO annotations and the
   project's state parameters. Keep identity/shape parameters fixed per subject
   while pose and global transform change frame by frame.
4. Train a static RGB-to-state initializer, then a recurrent visual residual
   model that predicts `delta_q` and global-pose updates from the current image
   plus the previous simulated state.
5. Compare raw image evidence, temporal state updates, and occlusion recovery
   on held-out dense sequences. Report 3D joint error, reprojection error,
   jitter, recovery after occlusion, uncertainty calibration, and per-frame
   latency.
6. Connect the validated state to the existing camera and 3D views. Preserve a
   saved, reproducible checkpoint outside ephemeral Colab storage.

## Interpretation guardrail

The target is a persistent, uncertainty-aware visual state estimator whose
output drives the simulated skeleton and glove. A detector skeleton may be a
debugging reference, but it is not the runtime architecture. Temporal memory
and image evidence must be tested on continuous sequences; filtering sparse
keyframes is not an adequate substitute.
