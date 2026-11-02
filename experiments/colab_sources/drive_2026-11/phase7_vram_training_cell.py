# Google Colab entry cell for the ReInterHand temporal MANO-state trainer.
# Upload this file and run it as one Colab cell, or paste its contents into a cell.
# Full fresh-start Colab run using the completed CPU-precomputed input bank.

from pathlib import Path
import json
import hashlib
import os
import runpy
import asyncio
import threading
import traceback
import shutil
import sys
import zipfile

from google.colab import drive, files

# --------------------------- User controls ----------------------------------
START_TRAINING = True  # This run has been explicitly approved by the user.
PHASE = "capture_holdout"  # "capture_holdout" first; "all_data" after reviewing it.
EPOCHS = 20
PAUSE_AT_MILESTONE = True  # Print a pause reminder at milestones; pause is requested through the Drive command file.
MILESTONE_GIB = 10.0

DRIVE_PROJECT = Path(
    "/content/drive/MyDrive/Persistent Visual-to-MANO Hand Tracker/04_DATASET_STAGING/ReInterHand"
)
TRAINING_READY = DRIVE_PROJECT / "03_training_ready"
EXTRACTED = DRIVE_PROJECT / "01_extracted"
BUNDLE_PATH = TRAINING_READY / "phase7_vram_bundle.zip"
CHECKPOINT_ROOT = TRAINING_READY / "checkpoints"
DEPLOYMENT_CHECKPOINT = CHECKPOINT_ROOT / "update_model.pt"
CONTROL_FILE = CHECKPOINT_ROOT / "phase7_command.txt"
PRECOMPUTED_CACHE = (
    CHECKPOINT_ROOT / "phase7_capture_holdout" / "precomputed_inputs"
    / "1e72186b7149de228ddd9cba"
)
# The completed CPU-precompute run already built the capture record indexes
# beside its target cache. Reuse those exact indexes instead of scanning all
# extracted frames again.
INDEX_CACHE_DIR = CHECKPOINT_ROOT / "phase7_capture_holdout" / "record_index"

EXPECTED = {"BKS383": 63100, "HIR112": 8581, "202107": 100815}
EXPECTED_TOTAL = sum(EXPECTED.values())

drive.mount("/content/drive")
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
if not CONTROL_FILE.is_file():
    CONTROL_FILE.write_text("continue\n", encoding="utf-8")
print(f"Live training control file: {CONTROL_FILE} (the running cell's Command box writes pause/continue)")
report_path = TRAINING_READY / "FINAL_QUALITY_REPORT.json"
if not report_path.is_file():
    raise FileNotFoundError(f"Missing completed audit report: {report_path}")
report = json.loads(report_path.read_text(encoding="utf-8"))
if int(report.get("MANO_trainable_images", -1)) != EXPECTED_TOTAL:
    raise RuntimeError("Audit total differs from the prepared run; inspect FINAL_QUALITY_REPORT.json")
if int(report.get("verified_images", -1)) != EXPECTED_TOTAL:
    raise RuntimeError("Not all expected images are verified; refusing to train")

report_captures = {item["label"]: item for item in report.get("captures", [])}
capture_roots = {}
for label, expected_count in EXPECTED.items():
    item = report_captures.get(label)
    if not item or int(item.get("images_trainable_MANO", -1)) != expected_count:
        raise RuntimeError(f"Unexpected audit count for {label}")
    capture_name = item.get("capture")
    if not capture_name:
        raise RuntimeError(f"Audit report has no capture folder name for {label}")
    # Preparation artifacts are stored under the full capture name, not the
    # short label (BKS383/HIR112/202107).
    prepared_capture = TRAINING_READY / capture_name
    manifest = prepared_capture / "trainable_manifest.jsonl"
    overlay = prepared_capture / "projection_overlay.png"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing audited image manifest for {label}: {manifest}")
    if not overlay.is_file() or overlay.stat().st_size == 0:
        raise FileNotFoundError(f"Missing projection spot-check overlay for {label}: {overlay}")
    valid_rows = 0
    with manifest.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("usable") or not row.get("mano_right") or not row.get("kp_right"):
                raise RuntimeError(f"Invalid right-hand target at {manifest}:{line_no}")
            valid_rows += 1
    if valid_rows != expected_count:
        raise RuntimeError(f"{label}: manifest has {valid_rows:,} rows; expected {expected_count:,}")

    candidates = [
        p for p in EXTRACTED.iterdir()
        if p.is_dir() and (p.name == capture_name or label.lower() in p.name.lower())
    ]
    candidates = [
        p for p in candidates
        if (p / "frame_list.txt").is_file()
        and (p / "Mugsy_cameras" / "cam_params.json").is_file()
        and (p / "orig_fits" / "right" / "Keypoints").is_dir()
        and (p / "mano_fits" / "params").is_dir()
        and (p / "Mugsy_cameras" / "envmap_per_segment" / "images").is_dir()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one extracted capture root for {label}; found {candidates}")
    capture_roots[label] = candidates[0]

print(f"Audit preflight passed: {EXPECTED_TOTAL:,} verified image/label joins")
print("Images by capture:", EXPECTED)
print("Data roots:", {k: str(v) for k, v in capture_roots.items()})
if not INDEX_CACHE_DIR.is_dir():
    raise FileNotFoundError(f"Completed capture record-index cache is missing: {INDEX_CACHE_DIR}")
required_indexes = [
    INDEX_CACHE_DIR / f"capture_{hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:16]}.json"
    for root in capture_roots.values()
]
missing_indexes = [p for p in required_indexes if not p.is_file()]
if missing_indexes:
    raise FileNotFoundError(
        "Refusing to rescan the full dataset; expected existing record indexes are missing: "
        + ", ".join(str(p) for p in missing_indexes)
    )
print(f"Verified {len(required_indexes)} reusable record indexes in {INDEX_CACHE_DIR}")
print(f"Checkpoint and resumable target cache root: {CHECKPOINT_ROOT}")
print(f"Stable latest-model file (overwritten every {MILESTONE_GIB:g} GiB): {DEPLOYMENT_CHECKPOINT}")
print("Training initialization: fresh random weights; the historical 5,114-frame run is not loaded or included.")

# Fail before record indexing if the completed GPU-input bank is missing or
# does not match the precompute run described in the notebook log.
cache_manifest_path = PRECOMPUTED_CACHE / "cache_manifest.json"
if not cache_manifest_path.is_file():
    raise FileNotFoundError(f"Completed precomputed input manifest not found: {cache_manifest_path}")
cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
if not cache_manifest.get("complete"):
    raise RuntimeError("Precomputed input bank is not marked complete; refusing to train.")
if int(cache_manifest.get("train_windows", -1)) != 27140:
    raise RuntimeError(f"Unexpected cached training-window count: {cache_manifest.get('train_windows')}")
if int(cache_manifest.get("variants", -1)) != 4:
    raise RuntimeError(f"Expected four precomputed variants per window; found {cache_manifest.get('variants')}")
print(
    f"Cached-input preflight passed: {PRECOMPUTED_CACHE}; "
    f"{cache_manifest['train_windows']:,} windows x {cache_manifest['variants']} variants"
)

if not BUNDLE_PATH.is_file():
    print("Upload the prepared phase7_colab_bundle.zip from your computer.")
    uploaded = files.upload()
    zip_candidates = [(name, content) for name, content in uploaded.items() if name.lower().endswith(".zip")]
    if len(zip_candidates) != 1:
        raise RuntimeError("Upload exactly one ZIP: phase7_colab_bundle.zip")
    upload_name, upload_data = zip_candidates[0]
    if "phase7_colab_bundle" not in upload_name.lower():
        raise RuntimeError(f"Unexpected ZIP name: {upload_name}")
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUNDLE_PATH.write_bytes(upload_data)

# Extract into a versioned folder so a stale trainer from an earlier run can
# never shadow the cached-input implementation in phase7_vram_bundle.zip.
bundle_root = Path("/content/phase7_vram_bundle")
trainer_path = bundle_root / "experiments" / "phase5_train_temporal.py"
if not trainer_path.is_file():
    bundle_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE_PATH) as archive:
        for member in archive.infolist():
            target = (bundle_root / member.filename).resolve()
            if not target.is_relative_to(bundle_root.resolve()):
                raise RuntimeError(f"Unsafe bundle path: {member.filename}")
        archive.extractall(bundle_root)
if not trainer_path.is_file():
    raise FileNotFoundError(f"Bundle is missing the trainer: {trainer_path}")

if PHASE not in ("capture_holdout", "all_data"):
    raise ValueError("PHASE must be 'capture_holdout' or 'all_data'")
if EPOCHS < 1:
    raise ValueError("EPOCHS must be positive")

if PHASE == "capture_holdout":
    # This dedicated output folder is empty for the new run. Old runs and their
    # checkpoints are deliberately not used as initialization.
    checkpoint_dir = CHECKPOINT_ROOT / "phase7_capture_holdout_cached"
    split_args = ["--split-mode", "capture", "--holdout-capture-token", "HIR112"]
    warmstart_args = []
else:
    first_best = CHECKPOINT_ROOT / "phase7_capture_holdout" / "best.pt"
    if not first_best.is_file():
        raise FileNotFoundError("Run and review the capture-holdout phase before all-data fine-tuning")
    checkpoint_dir = CHECKPOINT_ROOT / "phase7_all_data"
    split_args = ["--split-mode", "segment", "--holdout-capture-token", ""]
    warmstart_args = ["--warmstart-checkpoint", str(first_best)]

checkpoint_dir.mkdir(parents=True, exist_ok=True)
latest_checkpoint = checkpoint_dir / "latest.pt"
resume_existing_run = latest_checkpoint.is_file()
if not resume_existing_run:
    # The first launch starts from random weights. Remove the prior stable
    # webcam endpoint so it cannot be mistaken for this run's model; epoch-5
    # exports will recreate it from the new weights.
    DEPLOYMENT_CHECKPOINT.unlink(missing_ok=True)
    CONTROL_FILE.write_text("continue\n", encoding="utf-8")
    print("Fresh initialization confirmed: no checkpoint in this run folder; old runs will not be loaded.")
else:
    print(f"Checkpoint for this new run found; recovery will resume from {latest_checkpoint}.")

argv = [
    str(trainer_path),
    "--capture", str(capture_roots["BKS383"]),
    "--capture", str(capture_roots["202107"]),
    "--capture", str(capture_roots["HIR112"]),
    "--checkpoint-dir", str(checkpoint_dir),
    "--index-cache-dir", str(INDEX_CACHE_DIR),
    "--target-cache-path", str(next((CHECKPOINT_ROOT / "phase7_capture_holdout" / "target_cache").glob("states_*.npz"))),
    "--precomputed-cache", str(PRECOMPUTED_CACHE),
    "--deployment-checkpoint", str(DEPLOYMENT_CHECKPOINT),
    "--control-file", str(CONTROL_FILE),
    "--device", "cuda",
    "--epochs", str(EPOCHS),
    "--window-size", "6",
    "--batch-windows", "64",
    "--loader-workers", "8",
    "--ram-cache-shards", "12",
    "--stride", "6",
    "--max-frame-gap", "30",
    "--image-size", "128",
    "--hidden-dim", "256",
    "--learning-rate", "0.0002",
    "--weight-decay", "0.0001",
    "--checkpoint-every-steps", "10",
    "--webcam-checkpoint-every-epochs", "5",
    "--milestone-gib", str(MILESTONE_GIB),
    "--val-windows", "32",
    *split_args,
    *warmstart_args,
]
if resume_existing_run:
    argv.append("--resume")
if PAUSE_AT_MILESTONE:
    argv.append("--pause-at-milestone")

print(f"Selected phase: {PHASE}; checkpoint directory: {checkpoint_dir}")
print("Training command arguments:", argv[1:])
if not START_TRAINING:
    print("Preflight only. Set START_TRAINING=True after you approve the run.")
else:
    saved_command = CONTROL_FILE.read_text(encoding="utf-8").strip().lower()
    if resume_existing_run and saved_command in ("pause", "paused"):
        response = input(
            "A checkpoint for this fresh run exists. Type continue to resume from latest.pt, "
            "or anything else to leave it stopped: "
        ).strip().lower()
        if response != "continue":
            raise RuntimeError("Training left paused; no work was started.")
        CONTROL_FILE.write_text("continue\n", encoding="utf-8")

    os.environ["PYTHONPATH"] = str(bundle_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.argv = argv
    import ipywidgets as widgets
    from IPython.display import display

    command_box = widgets.Text(
        value="", placeholder="type pause", description="Command:",
        layout=widgets.Layout(width="360px"),
    )
    send_button = widgets.Button(description="Send", button_style="warning")
    command_status = widgets.HTML(value="Type <b>pause</b> and click Send. Wait for the trainer to print PAUSE SAVED before disconnecting.")

    def send_training_command(_):
        command = command_box.value.strip().lower()
        if command != "pause":
            command_status.value = "Enter <b>pause</b> while training is running. To resume after a saved pause, rerun this cell and type <b>continue</b> at its prompt."
            return
        command_tmp = CONTROL_FILE.with_suffix(CONTROL_FILE.suffix + ".tmp")
        command_tmp.write_text("pause\n", encoding="utf-8")
        os.replace(command_tmp, CONTROL_FILE)
        command_box.value = ""
        command_status.value = "Pause requested. Waiting for the next batch boundary to save checkpoints..."

    send_button.on_click(send_training_command)
    display(widgets.HBox([command_box, send_button]), command_status)

    thread_errors = []
    def train_in_background():
        try:
            runpy.run_path(str(trainer_path), run_name="__main__")
        except BaseException:
            thread_errors.append(traceback.format_exc())

    worker = threading.Thread(target=train_in_background, name="phase7-training", daemon=True)
    worker.start()
    print("Training started. Use the Command box above: type pause, click Send, and wait for PAUSE SAVED.")
    while worker.is_alive():
        await asyncio.sleep(0.5)
    if thread_errors:
        raise RuntimeError(thread_errors[0])
    print("Training process ended. If it paused, set START_TRAINING=True and rerun this cell; type continue at the resume prompt.")
