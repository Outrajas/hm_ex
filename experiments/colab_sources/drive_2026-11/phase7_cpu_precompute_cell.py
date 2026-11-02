# Phase 7 CPU-only resumable precompute cell. Never starts training.
from pathlib import Path
import json
import os
import runpy
import shutil
import sys
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from google.colab import drive
drive.mount("/content/drive")

import torch
if torch.cuda.is_available():
    raise RuntimeError("Select a CPU runtime before running this preprocessing cell.")
torch.set_num_threads(2)

DRIVE_PROJECT = Path("/content/drive/MyDrive/Persistent Visual-to-MANO Hand Tracker/04_DATASET_STAGING/ReInterHand")
TRAINING_READY = DRIVE_PROJECT / "03_training_ready"
EXTRACTED = DRIVE_PROJECT / "01_extracted"
CHECKPOINT_ROOT = TRAINING_READY / "checkpoints"
CHECKPOINT_DIR = CHECKPOINT_ROOT / "phase7_capture_holdout"
BUNDLE_PATH = TRAINING_READY / "phase7_colab_bundle.zip"
BUNDLE_ROOT = Path("/content/phase7_hm_ex_progress_v2")
TRAINER = BUNDLE_ROOT / "experiments" / "phase5_train_temporal.py"
PRECOMPUTER = BUNDLE_ROOT / "experiments" / "phase7_precompute_cache.py"
PATCHED_PRECOMPUTER = TRAINING_READY / "phase7_precompute_cache_drivefix.py"

if not BUNDLE_PATH.is_file():
    raise FileNotFoundError(f"Prepared project bundle not found: {BUNDLE_PATH}")
BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(BUNDLE_PATH) as archive:
    archive.extractall(BUNDLE_ROOT)
if not TRAINER.is_file() or not PRECOMPUTER.is_file():
    raise FileNotFoundError("Bundle is missing the Phase 7 trainer or precompute script.")

# Use the Drive-persisted fix, so a fresh Colab runtime always restores it.
if PATCHED_PRECOMPUTER.is_file():
    shutil.copy2(PATCHED_PRECOMPUTER, PRECOMPUTER)
    print("Installed Drive-safe shard precomputer (sequential local image staging).", flush=True)

index_path = CHECKPOINT_ROOT / "all_indexed_capture_roots.json"
if index_path.is_file():
    roots = json.loads(index_path.read_text(encoding="utf-8"))["capture_roots"]
else:
    report = json.loads((TRAINING_READY / "FINAL_QUALITY_REPORT.json").read_text(encoding="utf-8"))
    labels = {"BKS383", "202107", "HIR112"}
    captures = {item["label"]: item["capture"] for item in report.get("captures", []) if item.get("label") in labels}
    roots = [str(EXTRACTED / captures[label]) for label in ("BKS383", "202107", "HIR112")]
for root in roots:
    root_path = Path(root)
    if not (root_path / "frame_list.txt").is_file():
        raise FileNotFoundError(f"Indexed capture root is incomplete: {root}")

cache_root = CHECKPOINT_DIR / "precomputed_inputs"
argv = [
    str(PRECOMPUTER), "--trainer", str(TRAINER),
    "--checkpoint-dir", str(CHECKPOINT_DIR), "--cache-dir", str(cache_root),
    "--split-mode", "capture", "--holdout-capture-token", "HIR112",
    "--window-size", "6", "--stride", "6", "--max-frame-gap", "30",
    "--image-size", "128", "--variants", "4", "--shard-windows", "64",
    "--workers", "2",
]
for root in roots:
    argv.extend(["--capture", str(root)])

print("CPU preparation only; model training will not start.", flush=True)
print("Capture roots:", roots, flush=True)
print("Target checkpoint remains intact:", CHECKPOINT_DIR / "latest.pt", flush=True)
print("Cache output:", cache_root, flush=True)
print("Variants: 4; workers: 2; completed shards will be reused.", flush=True)
old_argv = sys.argv
try:
    sys.argv = argv
    runpy.run_path(str(PRECOMPUTER), run_name="__main__")
finally:
    sys.argv = old_argv
