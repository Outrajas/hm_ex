# Phase 6: train the temporal hand-state corrector on InterHand chunks

This is a controlled scale-up of the current MANO/state correction prototype. It does **not** create a general-purpose frontier model. The useful target is a hand-specific live tracker: image plus rendered current hand state in, bounded translation/rotation/joint updates and uncertainty out.

## What is ready

- `06_training/interhand26m_index.py` is the scalable path when training over many captures. It streams the split metadata twice (annotation IDs, then image rows) and streams the large joint and optional MANO label files once each for all selected captures. It writes one all-camera manifest per capture, even before image payloads are staged; the trainer preflight requires the images before training. `06_training/interhand26m_chunk.py` remains the smaller path for converting one capture at a time.
- `experiments/phase6_chunk_runner.py` stores a persistent shuffled chunk-order ledger, resumes a chunk from its saved checkpoint, initializes the next chunk from the prior chunk's best weights, and carries forward the checkpoint's frozen hand-proportion calibration so each capture uses the same state/FK geometry. It prints process/resource/progress/ETA information and asks for `TRAIN` before **each** training process. Before that prompt, it read-only checks every training and validation image/keypoint path and confirms complete temporal windows exist. On restart it verifies the manifest/checkpoint hashes and skips a completed chunk rather than retraining it. Mid-epoch checkpoints default to every 25 batches; lower `--checkpoint-every-steps` for disconnect-prone runs, or set it to 0 for epoch-only saves.
- Pre-register all intended capture chunks in the plan before starting if you want a global seeded shuffle. The runner stops cleanly at the first shuffled capture whose manifest-listed images/keypoints are not staged or valid; it does not request training approval for that chunk. Stage or repair that capture, then rerun with the same plan and checkpoint root.
- `06_training/phase6_replay.py` copies up to 64 complete six-frame windows from each finished capture into persistent storage. Later chunks include those older windows automatically. It verifies copied file sizes and records SHA-256 hashes; it never deletes source images.
- `experiments/phase5_train_temporal.py` accepts explicit `--validation-manifest` inputs, so a held-out capture stays outside the training/replay pool and provides consistent validation across chunks.
- `experiments/phase6_train_chunks.ps1` runs that monitor in the current PowerShell, or opens a separate visible PowerShell with `-NewWindow`.
- `experiments/phase6_inspect_interhand_tar.py` inventories one downloaded tar part without extracting it, grouping readable image entries by capture and reporting whether the part ended at an unreadable tar boundary.
- Source images are removed only when a plan explicitly sets `cleanup_root`, that path is inside `--staging-root`, it contains only listed chunk images (no extra files or other camera views), the path does not overlap checkpoints or replay storage, held-out validation images/labels are outside it, and the chunk has both `latest.pt` and `best.pt`. Cleanup requires `--replay-root`; the runner verifies the replay manifest, metadata, asset sizes, and file hashes before removing source images. Cleanup checks again immediately before removal and resumes safely if a prior removal was interrupted.

No data was downloaded and no training was started while preparing this workflow. The Phase 5 causal checkpoint at `phase5_causal_model/latest.pt` was mid-epoch (`epoch 16`, `next_offset 200`, `epoch_complete false`), so it is a resumable partial run, not a completed model. The earlier noncausal checkpoint is unsuitable for live evaluation because it used future-frame motion information.

## Data choice and scale

Start with **30 fps, Train (H)**, the human-annotated subset, and only a few captures for the first measured chunk. InterHand2.6M lists 528,510 H training frames at 30 fps, versus 5,716,488 H+M frames; M denotes machine annotations. Its 30 fps image collection is listed as 365 GB unpacked, so do not try to put the whole collection on this laptop or a free notebook disk. The 5 fps variant is 80 GB and was downsampled to reduce temporal redundancy. See the [official InterHand2.6M page](https://mks0601.github.io/InterHand2.6M/) and [official repository/license](https://github.com/facebookresearch/InterHand2.6M). The dataset is CC-BY-NC 4.0, so this training path is for noncommercial use.

Current local inventory (2026-10-22): 8 logical CPU threads, 15.71 GiB RAM total with 3.12 GiB available, 64.68 GiB free on C:, and only 3 files totaling 7,253 bytes under `datasets/interhand2.6m`—no usable InterHand image/annotation corpus is staged locally. The venv's PyTorch import fails loading `c10.dll` with WinError 5, and `nvidia-smi` is unavailable, so local GPU capability is unconfirmed. The previous CPU run took about 80 seconds per epoch over 4,246 frames. Linear extrapolation to 528,510 frames is about 2.8 hours per epoch or 55 hours for 20 epochs, before possible image-decoding, MANO-rendering, and target-fitting overhead. The chunk runner now fails before its approval prompt if PyTorch cannot import, so this environment must be repaired or the run moved to Colab/Kaggle.

For cloud runs, Colab's free limits and GPU availability fluctuate; Google says free sessions can run for at most 12 hours depending on availability and usage. Kaggle currently documents 12-hour notebook sessions, 20 GB of saved `/kaggle/working` output, and variable GPU availability. Use a GPU notebook, persistent checkpoint storage, and chunks small enough to finish well inside one session. Do not assume either service will provide a GPU next time. See [Colab's official FAQ](https://research.google.com/colaboratory/faq.html) and [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks).

Re:InterHand is still useful later: its official page describes 738,725 third-person 30 fps images and more diverse rendered appearances with stable 3D ground truth. Use it for appearance/lighting coverage and later mixed training, while InterHand2.6M H frames add human-captured sequence evidence. See the [official Re:InterHand page](https://mks0601.github.io/ReInterHand/).

## Minimal first chunk in Colab

1. Upload [the prepared Phase 6 source bundle](phase6_colab_source_20261022.zip) to Drive. It contains the tracker source, Phase 5/6 scripts, and `models/MANO_RIGHT.pkl`, but no dataset or checkpoint. Upload any optional seed checkpoint separately. Then run this setup cell:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !mkdir -p /content/stage /content/drive/MyDrive/hand_tracker
   !unzip -q -o /content/drive/MyDrive/hand_tracker/phase6_colab_source_20261022.zip -d /content
   !pip install -q ijson
   ```

   Upload the compact [Re:InterHand temporal anchor](../phase6_reinterhand_anchor_128w.zip) to `/content/drive/MyDrive/hand_tracker/`. It contains 128 six-frame windows (768 labeled frames, 56 source segments; about 645 MB) and is used in every InterHand2.6M chunk to reduce forgetting. Keep the ZIP on Drive; extract it to the session's local disk after each Colab reconnect:

   ```python
   !unzip -q -o /content/drive/MyDrive/hand_tracker/phase6_reinterhand_anchor_128w.zip -d /content/stage/reinterhand_anchor
   ```

   The runner validates every anchor image/label before requesting approval. If the session restarts, rerun setup and extraction; checkpoints remain under Drive.

   The public GitHub branch may not contain this uncommitted Phase 5/6 work. The existing seed is at `C:\Users\ojasa\Downloads\state_mach\phase5_causal_model\best.pt`; copy it to `/content/drive/MyDrive/hand_tracker/phase5_causal_model/best.pt` if you want to initialize from it. Otherwise leave out `--initial-checkpoint`.

2. Download the official [annotation archive](https://drive.google.com/drive/folders/12RNG9slv9i_TsXSoZ6pQAq-Fa98eGLoy) once and extract its Train JSON files under `/content/stage/interhand/annotations/train`. Save the official [human Train annotation-ID list](https://drive.google.com/file/d/1Tz6P2pyc55L5ZcGx1W85v5hAnB-MZ8ok/view) at `/content/drive/MyDrive/hand_tracker/InterHand2.6M_train_H_ids.txt`. These annotation files are separate from the multipart image archive.

3. Download only the first 30 fps image part and inspect its tar index. The official image release uses 1.9 GB parts; the probe prints progress while scanning and groups complete image payloads by capture:

   ```python
   !mkdir -p /content/stage/archive
   !wget -c -O /content/stage/archive/partaa \
     https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/InterHand2.6M/InterHand2.6M.images.30.fps.v1.0/InterHand2.6M.images.30.fps.v1.0.tar.partaa
   !python /content/hm_ex/experiments/phase6_inspect_interhand_tar.py /content/stage/archive/partaa
   ```

   Choose the held-out capture from the captures reported by this probe if possible. If its image payloads span parts, download only consecutive parts as needed. Keep archive parts outside the extracted image directory so interrupted staging can resume.

4. Index H labels for the desired capture set once. The indexer streams metadata and labels with byte progress and writes all-camera manifests to Drive. It shards per-frame keypoint/MANO files into subfolders to avoid extremely large Drive directories; it does not scan the unstaged image tree:

   ```python
   !python /content/hm_ex/06_training/interhand26m_index.py \
     --root /content/stage/interhand \
     --split train \
     --human-annotation-ids /content/drive/MyDrive/hand_tracker/InterHand2.6M_train_H_ids.txt \
     --out-root /content/drive/MyDrive/hand_tracker/manifests \
     --validation-capture-id VALIDATION_ID \
     --validation-root /content/drive/MyDrive/hand_tracker/manifests/validation
   ```

   Omit `--capture-ids` to index all captures from the Train camera JSON. For a small pilot, add `--capture-ids TRAIN_ID VALIDATION_ID`. The indexer routes the selected held-out capture to `--validation-root` during the same streaming pass; training manifests go under `--out-root`. The generated roots are named `CaptureN_ALLCAM` and match `phase6_make_plan.py --all-cameras`.

5. Pre-register all training captures and reserve one Train (H) capture for fixed validation. Substitute the held-out ID chosen from the tar inventory:

   ```python
   !python /content/hm_ex/experiments/phase6_make_plan.py \
     --camera-json /content/stage/interhand/annotations/train/InterHand2.6M_train_camera.json \
     --manifest-root /content/drive/MyDrive/hand_tracker/manifests \
     --validation-root /content/drive/MyDrive/hand_tracker/manifests/validation \
     --out-plan /content/drive/MyDrive/hand_tracker/phase6_chunks.json \
     --holdout-capture VALIDATION_ID --all-cameras \
     --cleanup-images-root /content/stage/interhand/images/train
   ```

   The official validation split is machine-annotated, so keep this Train (H) capture held out from training and replay.

6. Stage all images listed in the held-out manifest, then create a persistent validation replay set. Replace the capture ID and part list with the values identified by the probe:

   ```python
   !python /content/hm_ex/experiments/phase6_extract_interhand_capture.py \
     --capture-id VALIDATION_ID \
     --output-root /content/stage/interhand/images/train \
     --manifest /content/drive/MyDrive/hand_tracker/manifests/validation/CaptureVALIDATION_ID_ALLCAM/derived/mugsy_subset/manifest.csv \
     /content/stage/archive/partaa /content/stage/archive/partab
   !python /content/hm_ex/06_training/phase6_replay.py \
      --manifest /content/drive/MyDrive/hand_tracker/manifests/validation/CaptureVALIDATION_ID_ALLCAM/derived/mugsy_subset/manifest.csv \
     --out-root /content/drive/MyDrive/hand_tracker/replay/validation/CaptureVALIDATION_ID_ALLCAM \
     --max-windows 12 --window-size 60 --stride 60 --seed 77
   ```

   If extraction reports missing manifest images, add the next consecutive archive part and rerun. Completed files are reused. The replay builder copies 12 fixed 60-frame temporal windows to a separate persistent Drive directory and verifies each copy. These longer runs support the causal rollout and 15-frame dropout check. Use that copied manifest for validation so later Colab sessions do not need to restage the held-out capture.

   7. Start the chunk runner with the complete plan, Drive checkpoint/replay roots, and fixed validation manifest. `%run` keeps the `TRAIN` prompt interactive in Colab. Use `--chunk-order plan` when downloading consecutive tar parts and the probe confirms the plan's capture order follows the archive; use `--chunk-order shuffle` when captures are already staged or archive access is not sequential. The model trainer still shuffles whole temporal windows within each capture in both modes. Keep the same order mode and checkpoint root across reconnects; the runner stores order in `chunk_order.json`. It stops before approval at the first unstaged chunk and prints device/memory/process/elapsed/ETA information. Stage the reported capture with the extractor and rerun this same cell; after each chunk it asks again before training the next one. At the next approval prompt, decline if you want to inspect the finished chunk before authorizing another. With `--cleanup-images-root` in the plan, it removes only that capture's staged image folder after both checkpoint files and the Drive replay copy pass verification. Omit that plan option to retain source images.

   ```python
   %run /content/hm_ex/experiments/phase6_chunk_runner.py --plan /content/drive/MyDrive/hand_tracker/phase6_chunks.json --checkpoint-root /content/drive/MyDrive/hand_tracker/checkpoints --epochs 1 --checkpoint-every-steps 25 --device auto --chunk-order plan --staging-root /content/stage --validation-manifest /content/drive/MyDrive/hand_tracker/replay/validation/CaptureVALIDATION_ID_ALLCAM/derived/mugsy_subset/manifest.csv --anchor-manifest /content/stage/reinterhand_anchor/derived/mugsy_subset/manifest.csv --replay-root /content/drive/MyDrive/hand_tracker/replay --replay-windows-per-chunk 64
   ```

   Add `--initial-checkpoint /content/drive/MyDrive/hand_tracker/phase5_causal_model/best.pt` only if you uploaded it. Start with one epoch per capture and review the measured throughput, validation metrics, and memory before increasing the workload. The plan shown above opts into deleting the completed training capture's staged image directory only after checkpoint and replay verification. Omit `--cleanup-images-root` during plan creation to retain those images.

8. After a chunk finishes, inspect both its saved metrics and a label-free causal rollout on the persistent validation windows:

   ```python
   !python /content/hm_ex/experiments/phase6_evaluate_checkpoint.py \
     --checkpoint /content/drive/MyDrive/hand_tracker/checkpoints/CaptureTRAIN_ID_ALLCAM/best.pt \
     --validation-manifest /content/drive/MyDrive/hand_tracker/replay/validation/CaptureVALIDATION_ID_ALLCAM/derived/mugsy_subset/manifest.csv \
     --device auto
   ```

   This evaluator needs only the checkpoint, MANO model, and held-out replay manifest; it does not need the cleaned training capture. The causal rollout uses the first validation pose as a deliberately perturbed bootstrap seed, then uses RGB only and reports a synthetic 15-frame blackout.
## Local visible PowerShell

The laptop's GPU capability is unconfirmed: `nvidia-smi` is unavailable and the current venv cannot load PyTorch's `c10.dll`. Its 64.68 GiB free disk also cannot hold the official 365 GB 30 fps image set. To run a **small** measured chunk after repairing the environment and staging it locally, call this from PowerShell. `-NewWindow` opens a visible window; the runner waits for `TRAIN` before each chunk.

```powershell
& "C:\Users\ojasa\Downloads\state_mach\hm_ex\experiments\phase6_train_chunks.ps1" `
  -Plan "C:\path\to\phase6_chunks.json" `
  -CheckpointRoot "C:\path\to\persistent_checkpoints" `
  -ValidationManifest "C:\path\to\held_out_capture\derived\mugsy_subset\manifest.csv" `
  -AnchorManifest "C:\path\to\reinterhand_anchor\derived\mugsy_subset\manifest.csv" `
  -ReplayRoot "C:\path\to\persistent_replay" `
  -InitialCheckpoint "C:\Users\ojasa\Downloads\state_mach\phase5_causal_model\best.pt" `
  -Epochs 1 -CheckpointEverySteps 25 -ReplayWindowsPerChunk 64 -Device auto -ChunkOrder plan -NewWindow
```

## How to scale without pretending one pass is enough

The runner shuffles chunk order and the trainer shuffles complete temporal windows within each chunk; it never shuffles individual frames out of their temporal windows. With `--replay-root`, every new chunk also trains against the sampled windows from all completed chunks, while the fixed validation manifest remains separate. This is a bounded rehearsal set, not a mathematically equivalent full-corpus epoch. Replay images and checkpoints are preserved on Drive. The optional `--cleanup-images-root` reclaims only the completed capture's staged image directory after checkpoint and replay verification; it leaves manifests, labels, the anchor, validation data, and persistent replay untouched. Cleanup checks also refuse to remove unlisted files or the fixed validation images and labels.

For the tracker goal, evaluate in this order: (1) small measured capture chunk, (2) multiple human-labeled captures with capture-level holdout, (3) free-running temporal rollout on unseen clips with no later labels, (4) mixed Re:InterHand appearance data, (5) live webcam latency, orientation uncertainty, and occlusion recovery. Report these as prototype milestones, not frontier-scale training.
