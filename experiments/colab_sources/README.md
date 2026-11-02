# Colab source archive

This folder preserves the runnable source files that were saved in the project’s Google Drive and the trainer files packaged in the Phase 7 bundle. It contains source code only. Dataset archives, extracted images, model weights, checkpoints, and Colab authentication state are intentionally excluded.

## Files downloaded from Drive

The files in drive_2026-11 were fetched from the ReInterHand 03_training_ready folder on Drive and preserved byte-for-byte from their stored versions:

| File | Drive modified (UTC) | Bytes | Role |
|---|---:|---:|---|
| phase7_vram_training_cell.py | 2026-11-01 14:42 | 14,077 | Colab training entry cell. Extracts the Phase 7 bundle, preflights the data/cache, and launches the trainer with checkpoint and pause handling. |
| phase7_cpu_precompute_cell.py | 2026-10-30 11:50 | 3,378 | CPU precompute launcher. |
| phase7_precompute_cache_drivefix.py | 2026-10-30 11:56 | 14,755 | CPU-side cached-input builder and Drive filesystem workaround. |
| repair_existing_data.py | 2026-10-25 09:03 | 23,920 | Image repair and quality-join audit. This prepares data; it is not a model-training script. |

## Trainer files from the Phase 7 bundle

The matching Drive bundle, phase7_vram_bundle.zip, was listed at 3,606,854 bytes and modified 2026-11-01 14:22 UTC. The following script files are copied from its locally extracted source snapshot:

- phase7_bundle_snapshot/phase5_train_temporal.py — active temporal model trainer used by the Colab entry cell.
- phase7_bundle_snapshot/phase7_precompute_cache.py — cache generator called by the CPU preprocessing workflow.
- phase7_bundle_snapshot/phase7_webcam_smoke.py — local smoke test. It is not a complete webcam initialization or fitting application.

Earlier Drive source snapshots are also preserved in drive_training_snapshot:

- phase5_train_temporal.py and phase5_eval_temporal.py — earlier temporal trainer and evaluator.
- phase6_chunk_runner.py and phase6_evaluate_checkpoint.py — chunk-training workflow and evaluator.

The experiments folder also retains the phase0 through phase6 scripts and related preparation/evaluation utilities. These are prior source snapshots, not replacements for the active trainer copied from the Phase 7 bundle.

## Discovery limits and provenance

The connected Drive folder contained the script files above and the checkpoint/log folders. A saved .ipynb notebook for the full Colab session was not found in the accessible project folder, so this archive cannot promise a byte-for-byte export of every inline cell ever run. The phase0 through phase4 experiment scripts that are present in this repository are included separately under experiments.

The exact entry-cell settings and limitations are documented in PROJECT_TRAINING_TIMELINE.md. The Phase 7 training outputs are in Drive under Persistent Visual-to-MANO Hand Tracker / 04_DATASET_STAGING / ReInterHand / 03_training_ready / checkpoints. Only source files and text reports belong in this Git repository.
