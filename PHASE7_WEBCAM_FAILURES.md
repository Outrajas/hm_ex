# Phase 7 webcam failure record — 2026-11-02

## Outcome

The webcam experiment has NOT demonstrated reliable visual hand fitting or
continuous tracking. The user observed rigid-looking articulation, random drift,
shrinking, an inverted rendered hand, incomplete mesh surfaces, and lower FPS
than main.py. Rendering improvements must not be presented as model accuracy.

## Relevant scripts and checkpoint

- experiments/phase7_webcam_test.py: local webcam viewer and checkpoint loader.
- 08_runtime/tracker.py: trained residual application and recurrent state.
- 08_runtime/render_input.py: RGB plus rendered MANO silhouette and FK skeleton.
- experiments/phase5_train_temporal.py: training samples, loss and validation.
- Local asset: checkpoints/phase7/webcam_epoch_020.pt (ignored model binary).
- main.py: interactive simulator reference; its reported 60 FPS does not include
  the webcam inference pipeline.

## Confirmed defects and changes

1. Live updates integrated V*dt and qdot*dt before adding residuals, whereas
   training supervises target minus the supplied prior. Measured observations
   now apply residuals directly; missing observations retain velocity prediction.
2. The right pane negated camera-space Y again. Its projection now uses Y down.
   Camera rotation defaults to zero; mirror/rotation transform image and prior
   projections consistently. These options do not change hand chirality.
3. Skipping alternating mesh triangles caused holes. Visible front-facing
   triangles are now all drawn, with back-face culling as in main.py.
4. Dynamic viewport rescaling made articulation change apparent size. The side
   pane now uses fixed metres-to-pixels scaling. This does not prevent predicted
   camera depth from drifting.
5. Duplicate MANO generation and Python per-triangle normals added runtime cost.
   The viewer reuses the input renderer's mesh and vectorizes normals. At this
   revision the camera view uses the full depth-sorted mesh overlay; it costs
   more CPU time than the prior sparse wire overlay. The corrected mesh is
   regenerated once after inference to avoid display lag.
6. FPS was calculated from clamped timestep. It now uses actual wall-clock frame
   intervals. Logs split capture, input, model and display costs.
7. Finger sliders now provide smooth, reversible manual articulation of all five
   fingers. Moving a slider pauses model corrections. SPACE toggles model mode;
   R resets. Manual motion is explicitly labelled and is NOT camera tracking.
   Translation keys no longer overwrite orientation with stale Euler values.

## Training limitation established by source inspection

_perturb_sequence starts each frame from its labelled state plus noise (initial
translation standard deviation 18 mm, joints 9 degrees, rotation 7 degrees).
build_sample supplies those priors independently of the model's previous output.
Validation also corrects those labelled, perturbed priors. It does not establish
automatic initialization, recovery, or long-running closed-loop webcam accuracy.
The current model has no separate hand locator. Its crop follows the simulated
prior; a misplaced prior can crop the wrong image region.

## Verification and performance evidence

The actual checkpoint loaded strictly and executed a finite CPU forward pass.
This proves weights are used, not that predictions fit a real hand.

Synthetic 1280x720 images, CPU with four Torch threads, real checkpoint and MANO,
20 measured iterations after two warmups:

| Revision | Input ms | Model ms | Drawing ms | Total ms |
|---|---:|---:|---:|---:|
| Full triangle drawing before back-face culling | 4.14 | 6.29 | 23.30 | 34.75 |
| Back-face culling and simpler triangle rasterization | 4.45 | 6.62 | 15.68 | 26.98 |

These are medians excluding camera capture, GUI display and slider processing.
They do not demonstrate 60 FPS. The 128x128 model input remains the checkpoint's
training resolution; requested webcam resolution is negotiated with the driver.

Regression tests in tests/test_phase7_inference.py cover measured residuals,
zero-residual stability, missing-observation coasting, and independent opening
and closing of all five fingers in FK and the actual MANO mesh.
Both tests passed on 2026-11-02 using the local .phase0_env Python environment.
The physical webcam and interactive slider GUI were not exercised by these tests.

## Remaining work

Validate visual initialization on unseen images, then train and evaluate using
the model's own preceding predictions. Evaluate recorded webcam sequences before
claiming live fitting. Profile actual camera/GUI timing before promising 60 FPS.
No new training is started by the viewer fixes or regression checks.

## Follow-up inference correction — 2026-11-03

The live viewer's extra `--correction-strength` multiplier defaulted to `0.25`
after `PersistentHandTracker` had already applied the network's learned
authority. That attenuated each learned update a second time and differed from
the Phase 7 validation path. The default is now `1.0`; lowering it is an
explicit user-selected damping option.

While tracking, the preview now re-skins MANO once after inference and displays
the corrected state on the same camera frame. Before this change, the mesh and
skeleton were rendered from the pre-correction prior and only reflected the
update on the next loop iteration. The camera overlay now follows `main.py`'s
depth-sorted, front-facing MANO triangle rendering instead of drawing a hull
and a sparse set of edges, so the whole hand surface is visible. The extra
skinning and full camera overlay cost CPU time, so live FPS may be lower.
`main.py`'s 60 FPS simulator omits camera capture, input rendering, neural
inference, and this corrected-pose display pass; it is not a like-for-like
throughput target.

This is an inference/rendering correction, not a new automatic initializer.
The model still starts from a seeded MANO pose and has no independent full-frame
hand locator. Its input crop follows the current simulated pose. Thus the
checkpoint is still not established as a reliable hands-off webcam tracker;
this change only makes its learned corrections visible at full trained
strength and removes the display's one-frame pose lag.
