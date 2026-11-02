#!/usr/bin/env python
"""Permission-gated, resumable trainer for capture-sized dataset chunks.

The plan JSON contains already converted chunk manifests. Capture order can
follow the plan or a persistent seeded shuffle; temporal windows remain
shuffled within each training epoch. Every training subprocess is preceded by
explicit interactive approval. Staged data is removed only when requested by
the plan and after checkpoints and replay have been verified.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "experiments" / "phase5_train_temporal.py"
REPLAY_BUILDER = ROOT / "06_training" / "phase6_replay.py"
STEP_RE = re.compile(r"epoch\s+(\d+)\s+step\s+(\d+)/(\d+)")
EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+):.*?;\s*([0-9.]+)s")
_PROCESS_CPU_BASELINES = {}


class ChunkPending(Exception):
    """The next planned chunk has not been staged yet."""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, help="JSON list of {name, manifest, staged_root?} chunks")
    p.add_argument("--checkpoint-root", required=True, help="Persistent location, e.g. Google Drive")
    p.add_argument("--initial-checkpoint", default="", help="Model weights to use for the first chunk")
    p.add_argument("--validation-manifest", action="append", default=[], help="Fixed validation capture manifest; repeat as needed")
    p.add_argument("--anchor-manifest", action="append", default=[], help="Persistent Re:InterHand rehearsal manifest included in every chunk; repeat as needed")
    p.add_argument("--replay-root", default="", help="Persistent directory for sampled older temporal windows")
    p.add_argument("--replay-windows-per-chunk", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1, help="Epochs per chunk; approval is requested for every chunk")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--shuffle-seed", type=int, default=20251022)
    p.add_argument(
        "--chunk-order", choices=("shuffle", "plan"), default="shuffle",
        help="Shuffle capture chunks (default) or follow their plan order; temporal windows are shuffled in either mode.",
    )
    p.add_argument("--batch-windows", type=int, default=4)
    p.add_argument("--window-size", type=int, default=6)
    p.add_argument("--val-windows", type=int, default=32)
    p.add_argument(
        "--checkpoint-every-steps", type=int, default=25,
        help="Save resumable state every N batches; 0 saves only at epoch boundaries",
    )
    p.add_argument("--staging-root", default="", help="Only cleanup paths strictly below this directory")
    return p.parse_args()


def resource_snapshot(path: Path) -> list[str]:
    lines = []
    lines.append(f"CPU: logical cores={os.cpu_count() or 'unknown'}")
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}; VRAM free/total={free/2**30:.1f}/{total/2**30:.1f} GiB")
        else:
            lines.append("GPU: unavailable; training will use CPU")
    except Exception as exc:
        lines.append(f"GPU query unavailable: {exc}")
    ram = system_ram_available_gib()
    if ram is not None:
        lines.append(f"RAM: system available={ram:.1f} GiB")
    try:
        disk = shutil.disk_usage(path)
        lines.append(f"Checkpoint disk: free={disk.free/2**30:.1f} GiB")
    except OSError:
        pass
    return lines


def probe_training_runtime(device_preference: str) -> str:
    """Fail before approval if this Python cannot actually run the trainer."""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch cannot be imported by the runner's Python; no training "
            "approval will be requested. Repair this environment or use a "
            "working Colab/Kaggle runtime. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if device_preference == "cpu":
        return f"PyTorch runtime ready: CPU; logical CPUs={os.cpu_count() or 'unknown'}"
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        raise RuntimeError(
            "CUDA could not be queried in this Python; no training approval "
            f"will be requested. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    if device_preference == "cuda" and not cuda_available:
        raise RuntimeError(
            "--device cuda was requested, but CUDA is unavailable in this Python. "
            "No training approval will be requested."
        )
    if device_preference == "auto" and cuda_available:
        device_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        return (
            f"PyTorch runtime ready: cuda ({device_name}); "
            f"VRAM free/total={free/2**30:.1f}/{total/2**30:.1f} GiB"
        )
    if device_preference == "cuda":
        device_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        return (
            f"PyTorch runtime ready: cuda ({device_name}); "
            f"VRAM free/total={free/2**30:.1f}/{total/2**30:.1f} GiB"
        )
    return f"PyTorch runtime ready: CPU; logical CPUs={os.cpu_count() or 'unknown'}"


def system_ram_available_gib():
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except Exception:
        if os.name != "nt":
            return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return status.ullAvailPhys / 2**30
    return None


def windows_child_metrics(pid: int):
    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(FileTime), ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime), ctypes.POINTER(FileTime),
    ]
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong,
    ]
    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        return None
    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        creation, exit_time, kernel, user = FileTime(), FileTime(), FileTime(), FileTime()
        if not kernel32.GetProcessTimes(
            handle, ctypes.byref(creation), ctypes.byref(exit_time),
            ctypes.byref(kernel), ctypes.byref(user),
        ):
            return counters.WorkingSetSize / 2**30, None
        cpu_ticks = ((kernel.high << 32) | kernel.low) + ((user.high << 32) | user.low)
        now = time.monotonic()
        prior = _PROCESS_CPU_BASELINES.get(pid)
        _PROCESS_CPU_BASELINES[pid] = (cpu_ticks, now)
        cpu_percent = None
        if prior is not None and now > prior[1]:
            cpu_percent = max(0.0, (cpu_ticks - prior[0]) / 10_000_000 / (now - prior[1]) * 100.0)
        return counters.WorkingSetSize / 2**30, cpu_percent
    finally:
        kernel32.CloseHandle(handle)


def child_resource_snapshot(pid: int) -> list[str]:
    lines = []
    try:
        import psutil
        child = psutil.Process(pid)
        lines.append(
            f"[monitor] process RSS={child.memory_info().rss/2**30:.2f} GiB; "
            f"CPU={child.cpu_percent(interval=None):.0f}%"
        )
        lines.append(f"[monitor] system RAM available={psutil.virtual_memory().available/2**30:.1f} GiB")
    except Exception:
        if os.name == "nt":
            try:
                metrics = windows_child_metrics(pid)
                if metrics is not None:
                    child_gib, cpu_percent = metrics
                    cpu_text = "n/a" if cpu_percent is None else f"{cpu_percent:.0f}%"
                    lines.append(f"[monitor] process RSS={child_gib:.2f} GiB; CPU={cpu_text}")
            except Exception:
                pass
            ram = system_ram_available_gib()
            if ram is not None:
                lines.append(f"[monitor] system RAM available={ram:.1f} GiB")
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            lines.append(f"[monitor] GPU free/total={free/2**30:.1f}/{total/2**30:.1f} GiB")
    except Exception:
        pass
    return lines


def manifest_count(manifest: Path) -> int:
    with manifest.open("r", newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def verify_checkpoint(path: Path) -> dict:
    """Load the project's own checkpoint and validate its saved model state."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload is not a dictionary: {path}")
    required = ("model", "optimizer", "scheduler", "dataset_signature", "epoch", "epoch_complete", "next_offset")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Checkpoint is missing resumable fields {missing}: {path}")
    if not isinstance(payload["model"], dict) or not payload["model"]:
        raise ValueError(f"Checkpoint lacks a non-empty model state dictionary: {path}")
    if not isinstance(payload["optimizer"], dict) or not {"state", "param_groups"}.issubset(payload["optimizer"]):
        raise ValueError(f"Checkpoint lacks optimizer state needed for recovery: {path}")
    if not isinstance(payload["scheduler"], dict) or not payload["scheduler"]:
        raise ValueError(f"Checkpoint lacks scheduler state needed for recovery: {path}")
    if not payload.get("dataset_signature"):
        raise ValueError(f"Checkpoint lacks a dataset signature: {path}")
    if int(payload["epoch"]) < 1 or int(payload["next_offset"]) < 0:
        raise ValueError(f"Checkpoint epoch/offset metadata is invalid: {path}")
    if not bool(payload["epoch_complete"]):
        raise ValueError(f"Checkpoint is mid-epoch and cannot certify chunk completion: {path}")
    for key, value in payload["model"].items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite weights in checkpoint tensor {key!r}: {path}")
    return payload


def verify_checkpoint_pair(latest_path: Path, best_path: Path):
    latest_payload = verify_checkpoint(latest_path)
    best_payload = verify_checkpoint(best_path)
    if latest_payload["dataset_signature"] != best_payload["dataset_signature"]:
        raise ValueError("latest.pt and best.pt refer to different dataset signatures")
    if set(latest_payload["model"]) != set(best_payload["model"]):
        raise ValueError("latest.pt and best.pt have different model state keys")
    if int(best_payload["epoch"]) > int(latest_payload["epoch"]):
        raise ValueError("best.pt epoch is later than latest.pt")
    return latest_payload, best_payload


def safe_chunk_component(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "chunk"


def checked_cleanup_path(
    chunk: dict,
    staging_root: Path,
    checkpoint_root: Path,
    manifests,
    protected_roots=(),
    allow_missing_images=False,
):
    raw = chunk.get("cleanup_root")
    if not raw:
        return None
    candidate = Path(raw).expanduser().resolve()
    stage = staging_root.resolve()
    checkpoint = checkpoint_root.resolve()
    if candidate == stage or stage not in candidate.parents:
        raise ValueError(f"Refusing cleanup outside the staging root: {candidate}")
    if candidate == checkpoint or candidate in checkpoint.parents or checkpoint in candidate.parents:
        raise ValueError("Refusing cleanup that overlaps persistent checkpoints")
    for protected_root in protected_roots:
        protected = Path(protected_root).resolve()
        if candidate == protected or candidate in protected.parents or protected in candidate.parents:
            raise ValueError(f"Refusing cleanup that overlaps persistent data: {protected}")
    manifest_paths = manifests if isinstance(manifests, (list, tuple)) else [manifests]
    expected_images = set()
    row_count = 0
    for manifest in manifest_paths:
        manifest_path = Path(manifest).expanduser().resolve()
        if candidate == manifest_path or candidate in manifest_path.parents:
            raise ValueError("Refusing cleanup that would remove a manifest")
        # Stream each CSV: a capture may contain hundreds of thousands of frames.
        with manifest_path.open("r", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                raw_image = row.get("image_path")
                if not raw_image:
                    raise ValueError("Manifest row has an empty image_path; refusing cleanup")
                image_path = Path(raw_image).expanduser().resolve()
                if candidate not in image_path.parents:
                    raise ValueError(
                        f"Cleanup path does not contain every image; outside path: {image_path}"
                    )
                expected_images.add(image_path)
                for label_column in ("orig_kp_right", "mano_right"):
                    raw_labels = row.get(label_column)
                    if raw_labels:
                        label_path = Path(raw_labels).expanduser().resolve()
                        if candidate == label_path or candidate in label_path.parents:
                            raise ValueError(
                                f"Refusing cleanup that would remove {label_column} labels: {label_path}"
                            )
                row_count += 1
    if not row_count or not expected_images:
        raise ValueError("Manifest has no image paths; refusing cleanup")

    if not candidate.is_dir():
        raise ValueError(f"Cleanup target is not an existing directory: {candidate}")
    actual_files = set()
    for path in candidate.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Refusing cleanup directory containing a symbolic link: {path}")
        if path.is_file():
            actual_files.add(path.resolve())
    unexpected = actual_files - expected_images
    missing = expected_images - actual_files
    if unexpected or (missing and not allow_missing_images):
        examples = []
        if unexpected:
            examples.append("unlisted file(s): " + ", ".join(str(p) for p in sorted(unexpected)[:3]))
        if missing and not allow_missing_images:
            examples.append("missing manifest image(s): " + ", ".join(str(p) for p in sorted(missing)[:3]))
        raise ValueError(
            "Refusing cleanup unless the directory contains exactly this chunk's listed images; "
            + "; ".join(examples)
        )
    return candidate


def assert_cleanup_excludes_manifest(cleanup_root: Path, manifest: Path) -> None:
    candidate = cleanup_root.resolve()
    with manifest.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for column in ("image_path", "orig_kp_right", "mano_right"):
                raw_asset = row.get(column)
                if not raw_asset:
                    continue
                asset_path = Path(raw_asset).expanduser().resolve()
                if candidate == asset_path or candidate in asset_path.parents:
                    raise ValueError(
                        f"Refusing cleanup that would remove held-out validation {column} {asset_path}"
                    )


def _replay_builder_command(
    name, source_manifests, index, args, replay_root, *,
    check_only=False, check_all_assets=False,
):
    replay_dir = replay_root / safe_chunk_component(name)
    command = [
        sys.executable, "-u", str(REPLAY_BUILDER),
        "--out-root", str(replay_dir),
        "--max-windows", str(args.replay_windows_per_chunk),
        "--window-size", str(args.window_size),
        "--stride", str(args.window_size),
        "--max-frame-gap", "30",
        "--seed", str(args.shuffle_seed + index),
    ]
    if check_only:
        command.append("--check-only")
    if check_all_assets:
        command.append("--check-all-assets")
    for manifest in source_manifests:
        command.extend(("--manifest", str(manifest)))
    return replay_dir, command


def _write_replay_entry(name, source_manifests, index, args, replay_root, catalog_path):
    replay_dir, command = _replay_builder_command(
        name, source_manifests, index, args, replay_root
    )
    replay_manifest = replay_dir / "derived" / "mugsy_subset" / "manifest.csv"
    metadata_path = replay_dir / "replay_metadata.json"
    print("Creating persistent replay windows from this chunk; source data remains untouched.")
    print("Replay command:", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=ROOT, check=True)
    if not replay_manifest.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Replay builder did not produce a verified manifest under {replay_dir}")
    verify_replay_assets(metadata_path)
    entry = {
        "manifest_path": str(replay_manifest),
        "manifest_sha256": sha256_file(replay_manifest),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    catalog = {"schema_version": 1, "entries": {}}
    if catalog_path.is_file():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog.setdefault("entries", {})[name] = entry
    catalog["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json_atomic(catalog_path, catalog)
    return entry


def verify_replay_assets(metadata_path: Path, *, verify_hashes=False) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assets = metadata.get("copied_assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"Replay metadata has no copied asset inventory: {metadata_path}")
    for asset in assets:
        path = Path(asset["path"]).expanduser().resolve()
        if not path.is_file() or path.stat().st_size != int(asset["size"]):
            raise ValueError(f"Replay asset missing or size changed: {path}")
        if verify_hashes and sha256_file(path) != asset.get("sha256"):
            raise ValueError(f"Replay asset hash mismatch: {path}")


def verify_replay_entry(entry: dict, name: str, *, verify_asset_hashes=False) -> None:
    replay_manifest = Path(entry["manifest_path"]).expanduser().resolve()
    metadata_path = Path(entry["metadata_path"]).expanduser().resolve()
    if not replay_manifest.is_file() or sha256_file(replay_manifest) != entry.get("manifest_sha256"):
        raise RuntimeError(f"Replay manifest missing or changed for {name!r}: {replay_manifest}")
    if not metadata_path.is_file() or sha256_file(metadata_path) != entry.get("metadata_sha256"):
        raise RuntimeError(f"Replay metadata missing or changed for {name!r}: {metadata_path}")
    verify_replay_assets(metadata_path, verify_hashes=verify_asset_hashes)


def load_replay_catalog(path: Path, chunks, checkpoint_root: Path) -> dict:
    catalog = None
    corrupt_existing = False
    if path.is_file():
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict) and isinstance(candidate.get("entries"), dict):
                catalog = candidate
        except (OSError, json.JSONDecodeError):
            pass
        corrupt_existing = catalog is None
    if catalog is None:
        catalog = {"schema_version": 1, "entries": {}}

    recovered = False
    for chunk in chunks:
        name = str(chunk["name"])
        if name in catalog["entries"]:
            continue
        done_path = checkpoint_root / safe_chunk_component(name) / "chunk_done.json"
        if not done_path.is_file():
            continue
        done = json.loads(done_path.read_text(encoding="utf-8"))
        entry = done.get("replay_manifest")
        if entry is None:
            continue
        verify_replay_entry(entry, name)
        catalog["entries"][name] = entry
        recovered = True

    if recovered or corrupt_existing or not path.is_file():
        catalog["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json_atomic(path, catalog)
    return catalog


def run_chunk(
    chunk,
    index,
    args,
    checkpoint_root,
    initial_checkpoint,
    staging_root,
    replay_root,
    replay_catalog_path,
    previous_replay_manifests,
):
    name = str(chunk["name"])
    safe_name = safe_chunk_component(name)
    out_dir = checkpoint_root / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "latest.pt"
    best = out_dir / "best.pt"
    done_path = out_dir / "chunk_done.json"

    source_values = chunk.get("manifests", chunk.get("manifest"))
    if isinstance(source_values, str):
        source_values = [source_values]
    if not isinstance(source_values, list) or not source_values:
        raise ValueError(f"Chunk {name!r} must define manifest or manifests")
    plan_dir = Path(args.plan).expanduser().resolve().parent
    source_manifests = [
        (plan_dir / Path(value).expanduser()).resolve()
        if not Path(value).expanduser().is_absolute()
        else Path(value).expanduser().resolve()
        for value in source_values
    ]
    manifests = list(dict.fromkeys([*source_manifests, *previous_replay_manifests]))
    anchor_manifests = [Path(value).expanduser().resolve() for value in args.anchor_manifest]
    manifests = list(dict.fromkeys([*manifests, *anchor_manifests]))
    validation_manifests = [Path(value).expanduser().resolve() for value in args.validation_manifest]
    all_manifests = list(dict.fromkeys([*manifests, *validation_manifests]))
    if done_path.is_file():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("chunk_name") != name:
            raise ValueError(f"Checkpoint folder collision: {done_path} belongs to {done.get('chunk_name')!r}")
        recorded_manifests = done.get("manifest_files", [])
        current_manifests = [
            {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}
            for path in all_manifests
        ]
        if recorded_manifests != current_manifests:
            raise ValueError(
                f"Chunk {name!r} has a completion marker but its manifest changed or is missing. "
                "Restore the original manifest or stage it under a new chunk name."
            )
        if int(done.get("requested_epochs", -1)) == args.epochs:
            latest_payload, best_payload = verify_checkpoint_pair(latest, best)
            if latest_payload.get("dataset_signature") != done.get("dataset_signature"):
                raise ValueError(f"Latest checkpoint signature disagrees with {done_path}")
            if best_payload.get("dataset_signature") != done.get("dataset_signature"):
                raise ValueError(f"Best checkpoint signature disagrees with {done_path}")
            if sha256_file(latest) != done.get("latest_sha256") or sha256_file(best) != done.get("best_sha256"):
                raise ValueError(f"Checkpoint file hash mismatch; refusing to skip {name!r}")
            print(f"\nChunk {name}: verified complete checkpoint; no training process will be started.")
            if replay_root is not None:
                catalog = load_replay_catalog(replay_catalog_path, [chunk], checkpoint_root)
                replay_entry = catalog.get("entries", {}).get(name)
                if replay_entry is None:
                    replay_entry = done.get("replay_manifest")
                    if replay_entry is not None:
                        verify_replay_entry(replay_entry, name, verify_asset_hashes=True)
                        catalog.setdefault("entries", {})[name] = replay_entry
                        catalog["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                        write_json_atomic(replay_catalog_path, catalog)
                    else:
                        _write_replay_entry(name, source_manifests, index, args, replay_root, replay_catalog_path)
                        catalog = json.loads(replay_catalog_path.read_text(encoding="utf-8"))
                        replay_entry = catalog["entries"][name]
                verify_replay_entry(replay_entry, name, verify_asset_hashes=True)
            cleanup_raw = chunk.get("cleanup_root")
            if cleanup_raw:
                if staging_root is None:
                    raise ValueError("A cleanup_root was set in the plan but --staging-root was not provided")
                cleanup_path = Path(cleanup_raw).expanduser()
                cleanup_path = (plan_dir / cleanup_path).resolve() if not cleanup_path.is_absolute() else cleanup_path.resolve()
                if cleanup_path.exists():
                    checked_cleanup_path(
                        {"cleanup_root": str(cleanup_path)}, staging_root, checkpoint_root,
                        source_manifests,
                        protected_roots=([replay_root] if replay_root is not None else ()),
                        allow_missing_images=True,
                    )
                    for manifest in validation_manifests:
                        assert_cleanup_excludes_manifest(cleanup_path, manifest)
                    print(f"Removing the explicitly requested staged image directory: {cleanup_path}")
                    shutil.rmtree(cleanup_path)
                else:
                    print(f"Staged chunk is already absent after its verified completion: {cleanup_path}")
            return best, True

    missing_sources = [manifest for manifest in source_manifests if not manifest.is_file()]
    if missing_sources:
        raise ChunkPending(
            f"Next shuffled chunk {name!r} is not staged yet. Expected manifest(s): "
            + ", ".join(str(path) for path in missing_sources)
        )
    for manifest in [*previous_replay_manifests, *validation_manifests]:
        if not manifest.is_file():
            raise FileNotFoundError(f"Persistent replay or validation manifest missing: {manifest}")
    count = sum(manifest_count(manifest) for manifest in manifests)
    resume = latest.is_file()
    init_path = None if resume else initial_checkpoint
    if not resume and init_path is not None and not init_path.is_file():
        raise FileNotFoundError(f"Initial weights missing: {init_path}")

    plan_dir = Path(args.plan).expanduser().resolve().parent
    cleanup = None
    if chunk.get("cleanup_root"):
        if staging_root is None:
            raise ValueError("A cleanup_root was set in the plan but --staging-root was not provided")
        cleanup_raw = Path(chunk["cleanup_root"]).expanduser()
        cleanup = (plan_dir / cleanup_raw).resolve() if not cleanup_raw.is_absolute() else cleanup_raw.resolve()
        checked_cleanup_path(
            {"cleanup_root": str(cleanup)}, staging_root, checkpoint_root,
            source_manifests,
            protected_roots=([replay_root] if replay_root is not None else ()),
        )
        for manifest in validation_manifests:
            assert_cleanup_excludes_manifest(cleanup, manifest)

    print("\n" + "=" * 78)
    print(f"Chunk {index}: {name}")
    print(f"Manifest rows: {count:,}; manifests: {len(manifests)}")
    for manifest in manifests:
        print(f"  {manifest}")
    print(f"Checkpoint output: {out_dir}")
    print(f"Mode: {'resume current chunk' if resume else 'initialize weights and train new chunk'}")
    for line in resource_snapshot(checkpoint_root):
        print(line)
    if replay_root is not None:
        _, preflight_command = _replay_builder_command(
            name, source_manifests, index, args, replay_root,
            check_only=True, check_all_assets=True,
        )
        print("Checking training assets and replay-window coverage before approval...")
        preflight = subprocess.run(preflight_command, cwd=ROOT, check=False)
        if preflight.returncode != 0:
            raise ChunkPending(
                f"Training preflight did not pass for {name!r} (exit {preflight.returncode}). "
                "No training approval was requested and no training process was started. "
                "Stage or repair this capture's manifest-listed images/keypoints, then rerun "
                "with the same plan and checkpoint root. See the preflight diagnostic above."
            )
    else:
        preflight_root = checkpoint_root / ".preflight"
        _, preflight_command = _replay_builder_command(
            name, source_manifests, index, args, preflight_root,
            check_only=True, check_all_assets=True,
        )
        print("Checking training assets and temporal-window coverage before approval...")
        subprocess.run(preflight_command, cwd=ROOT, check=True)
    for anchor_index, anchor_manifest in enumerate(anchor_manifests, 1):
        _, anchor_command = _replay_builder_command(
            f"{name}_anchor_{anchor_index}", [anchor_manifest], index, args,
            replay_root or checkpoint_root / ".preflight_anchor",
            check_only=True, check_all_assets=True,
        )
        print(f"Checking persistent anchor assets/windows before approval: {anchor_manifest}")
        try:
            subprocess.run(anchor_command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Anchor manifest preflight failed (exit {exc.returncode}): {anchor_manifest}. "
                "No training approval was requested. Restore the persistent Re:InterHand anchor assets first."
            ) from exc
    for validation_index, validation_manifest in enumerate(validation_manifests, 1):
        validation_name = f"{name}_validation_{validation_index}"
        _, validation_command = _replay_builder_command(
            validation_name, [validation_manifest], index, args,
            checkpoint_root / ".preflight_validation",
            check_only=True, check_all_assets=True,
        )
        print(f"Checking validation assets/windows before approval: {validation_manifest}")
        subprocess.run(validation_command, cwd=ROOT, check=True)
    answer = input(f"Approve {args.epochs} epoch(s) of training for '{name}'? Type TRAIN to proceed: ").strip()
    if answer != "TRAIN":
        print(f"Skipped {name}; no training process was started.")
        return initial_checkpoint, False

    command = [
        sys.executable, "-u", str(TRAINER),
        "--checkpoint-dir", str(out_dir), "--device", args.device,
        "--epochs", str(args.epochs), "--learning-rate", str(args.learning_rate),
        "--batch-windows", str(args.batch_windows), "--window-size", str(args.window_size),
        "--val-windows", str(args.val_windows),
        "--checkpoint-every-steps", str(args.checkpoint_every_steps),
    ]
    for manifest in manifests:
        command.extend(("--manifest", str(manifest)))
    for manifest in validation_manifests:
        command.extend(("--validation-manifest", str(manifest)))
    if resume:
        command.append("--resume")
    elif init_path is not None:
        command.extend(("--init-checkpoint", str(init_path)))

    log_path = out_dir / "train.log"
    start = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    epoch_durations = []
    current_epoch_start = start
    print(f"Training PID will be shown after launch; started {start_utc}")
    print("Command:", subprocess.list2cmdline(command))
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        print(f"Training process PID: {process.pid}")
        output_queue = queue.Queue()

        def pump_output():
            assert process.stdout is not None
            for output_line in process.stdout:
                output_queue.put(output_line)
            output_queue.put(None)

        threading.Thread(target=pump_output, daemon=True).start()
        output_closed = False
        while not output_closed:
            try:
                line = output_queue.get(timeout=30.0)
            except queue.Empty:
                elapsed = time.monotonic() - start
                print(f"[monitor] heartbeat; elapsed={elapsed/60:.1f} min; trainer is still running")
                for status in child_resource_snapshot(process.pid):
                    print(status)
                if process.poll() is not None:
                    # Give the reader thread time to forward any buffered tail.
                    continue
                continue
            if line is None:
                output_closed = True
                continue
            print(line, end="")
            log.write(line)
            elapsed = time.monotonic() - start
            match = STEP_RE.search(line)
            if match:
                epoch, step, total_steps = map(int, match.groups())
                epoch_elapsed = time.monotonic() - current_epoch_start
                per_step = epoch_elapsed / max(step, 1)
                remaining_steps = max(total_steps - step, 0)
                future_epochs = max(args.epochs - epoch, 0)
                known_epoch = sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0.0
                epoch_eta = per_step * remaining_steps
                if not known_epoch:
                    known_epoch = per_step * total_steps + 20.0
                eta = epoch_eta + future_epochs * known_epoch
                print(f"[monitor] elapsed={elapsed/60:.1f} min; ETA~{eta/60:.1f} min; epoch {epoch}/{args.epochs}, step {step}/{total_steps}")
                for status in child_resource_snapshot(process.pid):
                    print(status)
            match = EPOCH_RE.search(line)
            if match:
                epoch_durations.append(float(match.group(3)))
                current_epoch_start = time.monotonic()
                remain = max(args.epochs - int(match.group(1)), 0)
                avg_epoch = sum(epoch_durations) / len(epoch_durations)
                print(f"[monitor] completed epoch {match.group(1)}/{args.epochs}; total elapsed={elapsed/60:.1f} min; estimated remaining={remain*avg_epoch/60:.1f} min")
        return_code = process.wait()
    elapsed = time.monotonic() - start
    print(f"Chunk finished with exit={return_code}; elapsed={elapsed/60:.1f} min; log={log_path}")
    if return_code != 0:
        raise RuntimeError(f"Trainer failed for chunk {name}; checkpoint is preserved for resume.")
    if not latest.is_file() or not best.is_file():
        raise RuntimeError("Trainer exited successfully but latest.pt or best.pt is missing; refusing cleanup.")
    latest_payload, best_payload = verify_checkpoint_pair(latest, best)
    if int(latest_payload.get("epoch", 0)) < args.epochs:
        raise ValueError(
            f"Trainer checkpoint reached epoch {latest_payload.get('epoch')}, below requested {args.epochs}; refusing cleanup"
        )
    replay_entry = None
    if replay_root is not None:
        replay_entry = _write_replay_entry(
            name, source_manifests, index, args, replay_root, replay_catalog_path
        )
    done_record = {
        "chunk_name": name,
        "requested_epochs": args.epochs,
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_files": [
            {"path": str(manifest), "sha256": sha256_file(manifest)}
            for manifest in all_manifests
        ],
        "dataset_signature": latest_payload["dataset_signature"],
        "latest_epoch": int(latest_payload["epoch"]),
        "best_epoch": int(best_payload["epoch"]),
        "latest_sha256": sha256_file(latest),
        "best_sha256": sha256_file(best),
        "replay_manifest": replay_entry,
    }
    write_json_atomic(done_path, done_record)

    if cleanup is not None:
        if cleanup.exists():
            if replay_entry is None:
                raise RuntimeError("Refusing source cleanup without a persistent replay recovery entry.")
            verify_replay_entry(replay_entry, name, verify_asset_hashes=True)
            cleanup = checked_cleanup_path(
                {"cleanup_root": str(cleanup)}, staging_root, checkpoint_root,
                source_manifests,
                protected_roots=([replay_root] if replay_root is not None else ()),
                allow_missing_images=True,
            )
            for manifest in validation_manifests:
                assert_cleanup_excludes_manifest(cleanup, manifest)
            print(f"Verified checkpoint. Removing staged chunk: {cleanup}")
            shutil.rmtree(cleanup)
        else:
            print(f"Verified checkpoint; staged chunk is already absent: {cleanup}")
    else:
        print("No staged-data directory was marked for cleanup; source remains in place.")
    return best, True


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.checkpoint_every_steps < 0:
        raise ValueError("--checkpoint-every-steps cannot be negative")
    if args.replay_windows_per_chunk < 1:
        raise ValueError("--replay-windows-per-chunk must be positive")
    if args.replay_root and not args.validation_manifest:
        raise ValueError("--replay-root requires a fixed --validation-manifest to avoid validating on replayed training data")
    try:
        print(probe_training_runtime(args.device), flush=True)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    plan_path = Path(args.plan).expanduser().resolve()
    chunks = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Plan must be a non-empty JSON list")
    names = [str(chunk.get("name", "")) for chunk in chunks]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("Every chunk needs a unique, non-empty name")
    if any(chunk.get("cleanup_root") for chunk in chunks) and not args.replay_root:
        raise ValueError(
            "A plan with cleanup_root requires --replay-root so a persistent verified "
            "temporal recovery set exists before source images can be removed."
        )
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    replay_root = Path(args.replay_root).expanduser().resolve() if args.replay_root else None
    replay_catalog_path = checkpoint_root / "replay_catalog.json"
    if replay_root is not None:
        if replay_root == checkpoint_root or replay_root in checkpoint_root.parents or checkpoint_root in replay_root.parents:
            raise ValueError("--replay-root and --checkpoint-root must be separate, non-overlapping directories")
        if staging_root := (Path(args.staging_root).expanduser().resolve() if args.staging_root else None):
            if replay_root == staging_root or replay_root in staging_root.parents or staging_root in replay_root.parents:
                raise ValueError("--replay-root and --staging-root must be separate, non-overlapping directories")
        replay_root.mkdir(parents=True, exist_ok=True)
    for validation_manifest in args.validation_manifest:
        if not Path(validation_manifest).expanduser().resolve().is_file():
            raise FileNotFoundError(f"Validation manifest missing: {validation_manifest}")
    for anchor_manifest in args.anchor_manifest:
        if not Path(anchor_manifest).expanduser().resolve().is_file():
            raise FileNotFoundError(f"Anchor manifest missing: {anchor_manifest}")
    safe_names = [safe_chunk_component(name) for name in names]
    if len(safe_names) != len(set(safe_names)):
        raise ValueError("Chunk names collide after conversion to safe checkpoint folder names")
    chunks_by_name = {str(chunk["name"]): chunk for chunk in chunks}
    order_path = checkpoint_root / "chunk_order.json"
    if order_path.is_file():
        saved_order = json.loads(order_path.read_text(encoding="utf-8"))
        saved_mode = str(saved_order.get("mode", "shuffle"))
        if saved_mode != args.chunk_order:
            raise ValueError(
                f"This checkpoint root has a {saved_mode!r} chunk-order ledger; "
                f"continue with --chunk-order {saved_mode} or choose a new checkpoint root."
            )
        order = list(saved_order.get("order", []))
        removed = sorted(set(order) - set(names))
        if removed:
            raise ValueError(
                "The chunk plan removed previously registered chunks; keep those entries so "
                f"the training chain remains recoverable: {removed}"
            )
        saved_names = set(order)
        additions = [name for name in names if name not in saved_names]
        if additions:
            if args.chunk_order == "shuffle":
                rng = random.Random(int(saved_order.get("seed", args.shuffle_seed)) + len(order))
                rng.shuffle(additions)
            order.extend(additions)
            write_json_atomic(order_path, {
                "seed": int(saved_order.get("seed", args.shuffle_seed)),
                "mode": args.chunk_order,
                "order": order,
            })
            print(f"Appended {len(additions)} new chunks in {args.chunk_order} order.")
    else:
        order = names.copy()
        if args.chunk_order == "shuffle":
            random.Random(args.shuffle_seed).shuffle(order)
        write_json_atomic(order_path, {
            "seed": args.shuffle_seed,
            "mode": args.chunk_order,
            "order": order,
        })
    chunks = [chunks_by_name[name] for name in order]
    initial = Path(args.initial_checkpoint).expanduser().resolve() if args.initial_checkpoint else None
    staging_root = Path(args.staging_root).expanduser().resolve() if args.staging_root else None
    print(f"Persistent {args.chunk_order} chunk order: {len(chunks)} chunks; ledger={order_path}")
    print("The script requests fresh approval before each chunk; decline stops before that training process starts.")
    print(f"Persistent checkpoint directory: {checkpoint_root}")
    if staging_root:
        print(f"Cleanup is restricted to children of: {staging_root}")
    if replay_root:
        print(f"Replay windows are persisted under: {replay_root}; fixed validation manifests: {len(args.validation_manifest)}")
    replay_catalog = load_replay_catalog(replay_catalog_path, chunks, checkpoint_root)
    for index, chunk in enumerate(chunks, 1):
        previous_replay_manifests = []
        if replay_root is not None:
            for previous_name in order[:index - 1]:
                entry = replay_catalog.get("entries", {}).get(previous_name)
                if entry is None:
                    raise RuntimeError(
                        f"No persistent replay set is registered for completed chunk {previous_name!r}; "
                        "restore its staged source before continuing."
                    )
                replay_manifest = Path(entry["manifest_path"]).expanduser().resolve()
                verify_replay_entry(entry, previous_name)
                previous_replay_manifests.append(replay_manifest)
        try:
            initial, completed = run_chunk(
                chunk, index, args, checkpoint_root, initial, staging_root,
                replay_root, replay_catalog_path, previous_replay_manifests,
            )
        except ChunkPending as exc:
            print(f"\n{exc}")
            print("No training process was started. Stage this chunk, then rerun the same plan and checkpoint root.")
            return
        if replay_root is not None and completed:
            replay_catalog = json.loads(replay_catalog_path.read_text(encoding="utf-8"))
        if not completed:
            print("Stopped at user decision; earlier checkpoints remain saved.")
            return
    print(f"All {len(chunks)} chunks processed. Final model checkpoint: {initial}")


if __name__ == "__main__":
    main()
