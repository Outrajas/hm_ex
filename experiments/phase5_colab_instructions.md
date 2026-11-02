# Phase 5 training on Colab

The code bundle contains the current project modules and `MANO_RIGHT.pkl`; it
does not contain the Re:InterHand images or your trained checkpoint.

1. Upload `hm_ex_phase5_bundle_current.zip` to Google Drive.
2. In Colab, make the already-prepared flat subset available at
   `/content/dataset/manifest.csv`, `/content/dataset/cam_params.json`, and the
   matching image and `keypoints_right/` directories. The reader also accepts
   the original capture layout when given its manifest path.
3. Run this cell. Drive stores both the target cache and resumable checkpoints:

```python
from google.colab import drive
drive.mount('/content/drive')

!unzip -q -o /content/drive/MyDrive/hm_ex_phase5_bundle_current.zip -d /content/hm_ex
!python -u /content/hm_ex/experiments/phase5_train_temporal.py \
  --manifest /content/dataset/manifest.csv \
  --checkpoint-dir /content/drive/MyDrive/hand_tracker_checkpoints/phase5 \
  --device auto --epochs 20 --resume
```

`--device auto` uses CUDA when Colab assigns a GPU and otherwise uses CPU. On a
disconnect, rerun the same cell: `latest.pt` is saved every 25 batches and at
each epoch boundary, and `--resume` continues from its saved optimizer, random
state, and batch offset. `best.pt` preserves the best sampled validation score.
The report script can evaluate every validation frame after training:

```python
!python -u /content/hm_ex/experiments/phase5_eval_temporal.py \
  --checkpoint /content/drive/MyDrive/hand_tracker_checkpoints/phase5/best.pt \
  --manifest /content/dataset/manifest.csv --device auto
```

For the official capture layout, use `--capture /path/to/capture_root` instead
of `--manifest`, and keep `--camera-group Mugsy_cameras`. Use dense
`envmap_per_segment` frames for temporal training. A flat manifest may omit
MANO sidecar files; in that case the trainer uses zero shape coefficients.

The current 5,114-frame subset is a pipeline baseline, not a data-rich model.
For cross-person testing, pass multiple manifests/captures and split by capture
where possible. Do not merge adjacent frames from one segment across train and
validation.
