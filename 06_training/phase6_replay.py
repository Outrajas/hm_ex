#!/usr/bin/env python
"""Create a small persistent replay set of complete temporal windows.

The selected images and labels are copied out of a staged capture so later
training chunks can rehearse earlier motion without retaining every source
image. This tool never deletes or edits its source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
data_api = importlib.import_module("06_training.reinterhand_sequences")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="Source capture manifest; repeat only for manifests from one capture root")
    parser.add_argument("--out-root", required=True, help="New persistent replay capture directory")
    parser.add_argument("--max-windows", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--max-frame-gap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20251022)
    parser.add_argument("--check-only", action="store_true", help="Validate temporal-window availability without copying files")
    parser.add_argument("--check-all-assets", action="store_true", help="In check-only mode, verify every manifest image and right-hand label exists")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned.strip("._") or "capture"


def copy_verified(source: Path, destination: Path, final_destination: Path) -> dict:
    shutil.copy2(source, destination)
    source_size = source.stat().st_size
    if destination.stat().st_size != source_size:
        raise IOError(f"Replay copy size mismatch: {source} -> {destination}")
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if destination_hash != source_hash:
        raise IOError(f"Replay copy hash mismatch: {source} -> {destination}")
    return {
        "path": str(final_destination.resolve()),
        "size": source_size,
        "sha256": destination_hash,
    }


def camera_file(capture_root: Path) -> Path:
    candidates = (
        capture_root / "Mugsy_cameras" / "cam_params.json",
        capture_root / "cam_params.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No cam_params.json under {capture_root}")


def _existing_matches(out_root: Path, identity: dict) -> bool:
    meta_path = out_root / "replay_metadata.json"
    manifest_path = out_root / "derived" / "mugsy_subset" / "manifest.csv"
    if not meta_path.is_file() or not manifest_path.is_file():
        return False
    try:
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if saved.get("identity") != identity:
        return False
    for asset in saved.get("copied_assets", []):
        path = Path(asset["path"]).expanduser().resolve()
        if not path.is_file() or path.stat().st_size != int(asset["size"]):
            return False
        if sha256_file(path) != asset["sha256"]:
            return False
    return bool(saved.get("copied_assets"))


def main():
    args = parse_args()
    if args.max_windows < 1 or args.window_size < 2 or args.stride < args.window_size:
        raise ValueError("Require max-windows >= 1, window-size >= 2, and stride >= window-size to avoid duplicate replay frames.")
    if args.max_frame_gap < 1:
        raise ValueError("--max-frame-gap must be positive")

    manifests = [Path(value).expanduser().resolve() for value in args.manifest]
    for path in manifests:
        if not path.is_file():
            raise FileNotFoundError(path)
    roots = {data_api.capture_root_from_manifest(path) for path in manifests}
    if len(roots) != 1:
        raise ValueError("One replay output must come from exactly one source capture root.")
    source_root = next(iter(roots))
    cam_source = camera_file(source_root)

    records = []
    for manifest in manifests:
        records.extend(data_api.records_from_subset_manifest(manifest))
    unique = {}
    for record in records:
        unique[(str(record.image_path.resolve()), record.camera_id)] = record
    records = sorted(
        unique.values(),
        key=lambda item: (item.capture_id, item.segment, item.camera_id, item.frame_idx),
    )
    windows = data_api.sequence_windows(
        records,
        window_size=args.window_size,
        stride=args.stride,
        max_frame_gap=args.max_frame_gap,
    )
    if not windows:
        raise ValueError("The source manifests contain no complete temporal windows.")

    rng = np.random.default_rng(args.seed)
    selected_ids = np.sort(
        rng.choice(len(windows), size=min(args.max_windows, len(windows)), replace=False)
    )
    selected = [windows[int(i)] for i in selected_ids]
    selected_records = {int(record_id) for window in selected for record_id in window}
    for record_id in selected_records:
        record = records[record_id]
        if not record.image_path.is_file() or not record.right_keypoints_path.is_file():
            raise FileNotFoundError(f"Selected replay image/label missing for frame {record.frame_idx}: {record.image_path}")
    source_hashes = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in manifests
    ]
    identity = {
        "source_manifests": source_hashes,
        "max_windows": args.max_windows,
        "window_size": args.window_size,
        "stride": args.stride,
        "max_frame_gap": args.max_frame_gap,
        "seed": args.seed,
    }

    if args.check_only:
        checked_assets = 0
        if args.check_all_assets:
            for record in records:
                if not record.image_path.is_file() or not record.right_keypoints_path.is_file():
                    raise FileNotFoundError(
                        f"Manifest asset missing at frame {record.frame_idx}: {record.image_path} / {record.right_keypoints_path}"
                    )
                checked_assets += 1
                if checked_assets % 10000 == 0:
                    print(f"Preflight checked {checked_assets:,}/{len(records):,} manifest rows", flush=True)
        print(
            f"Replay preflight OK: capture={source_root.name}; available_windows={len(windows):,}; "
            f"selected_windows={len(selected):,}; frames={len(selected_records):,}; "
            f"checked_assets={checked_assets:,}; "
            f"camera={cam_source}"
        )
        return

    out_root = Path(args.out_root).expanduser().resolve()
    if out_root.exists():
        if _existing_matches(out_root, identity):
            manifest_path = out_root / "derived" / "mugsy_subset" / "manifest.csv"
            print(f"Reusing matching replay set: {manifest_path}")
            return
        raise FileExistsError(
            f"Replay output already exists and does not match this request: {out_root}. "
            "It was left untouched; choose a new output directory or inspect it manually."
        )
    out_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = out_root.with_name(out_root.name + ".building")
    if temp_root.exists():
        state_path = temp_root / "replay_build_state.json"
        if not state_path.is_file():
            raise FileExistsError(f"Incomplete replay build lacks identity metadata at {temp_root}; inspect it before continuing.")
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        if saved_state.get("identity") != identity:
            raise FileExistsError(f"A different incomplete replay build exists at {temp_root}; inspect it before continuing.")
        print(f"Resuming interrupted replay copy at {temp_root}")
    else:
        temp_root.mkdir(parents=True)
        (temp_root / "replay_build_state.json").write_text(
            json.dumps({"identity": identity}, indent=2), encoding="utf-8"
        )
    final_manifest = out_root / "derived" / "mugsy_subset" / "manifest.csv"
    temp_manifest = temp_root / "derived" / "mugsy_subset" / "manifest.csv"
    temp_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cam_source, temp_root / "cam_params.json")

    rows = []
    selected_summary = []
    copied_assets = []
    copied_bytes = 0
    try:
        for window_number, window in enumerate(selected, start=1):
            segment_name = f"replay_window_{window_number:06d}"
            first = records[int(window[0])]
            selected_summary.append({
                "source_segment": first.segment,
                "camera_id": first.camera_id,
                "first_frame": int(first.frame_idx),
                "last_frame": int(records[int(window[-1])].frame_idx),
            })
            for frame_number, record_index in enumerate(window, start=1):
                record = records[int(record_index)]
                source_paths = [record.image_path, record.right_keypoints_path]
                if record.right_mano_path is not None and record.right_mano_path.is_file():
                    source_paths.append(record.right_mano_path)
                for source in source_paths:
                    if not source.is_file():
                        raise FileNotFoundError(f"Replay source disappeared: {source}")

                stem = f"w{window_number:06d}_f{frame_number:02d}_{record.frame_idx:08d}_{safe_name(record.camera_id)}"
                image_relative = Path("images") / safe_name(record.camera_id) / f"{stem}{record.image_path.suffix.lower() or '.jpg'}"
                keypoint_relative = Path("keypoints_right") / f"{stem}.json"
                image_dest = temp_root / image_relative
                keypoint_dest = temp_root / keypoint_relative
                image_dest.parent.mkdir(parents=True, exist_ok=True)
                keypoint_dest.parent.mkdir(parents=True, exist_ok=True)
                copied_assets.append(copy_verified(
                    record.image_path, image_dest, out_root / image_relative
                ))
                decoded = cv2.imread(str(image_dest), cv2.IMREAD_COLOR)
                if decoded is None or decoded.size == 0:
                    raise IOError(f"Copied replay image cannot be decoded: {image_dest}")
                del decoded
                copied_assets.append(copy_verified(
                    record.right_keypoints_path, keypoint_dest, out_root / keypoint_relative
                ))
                copied_bytes += record.image_path.stat().st_size + record.right_keypoints_path.stat().st_size

                mano_path = ""
                if record.right_mano_path is not None and record.right_mano_path.is_file():
                    mano_relative = Path("mano_right") / f"{stem}.json"
                    mano_dest = temp_root / mano_relative
                    mano_dest.parent.mkdir(parents=True, exist_ok=True)
                    copied_assets.append(copy_verified(
                        record.right_mano_path, mano_dest, out_root / mano_relative
                    ))
                    copied_bytes += record.right_mano_path.stat().st_size
                    mano_path = str((out_root / mano_relative).resolve())

                rows.append({
                    "frame_id": str(record.frame_idx),
                    "segment": segment_name,
                    "camera_id": str(record.camera_id),
                    "image_path": str((out_root / image_relative).resolve()),
                    "orig_kp_right": str((out_root / keypoint_relative).resolve()),
                    "mano_right": mano_path,
                })
            if window_number % 8 == 0 or window_number == len(selected):
                print(
                    f"Copied replay windows {window_number}/{len(selected)}; "
                    f"frames={len(rows):,}; staged={copied_bytes / 2**20:.1f} MiB",
                    flush=True,
                )

        with temp_manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("frame_id", "segment", "camera_id", "image_path", "orig_kp_right", "mano_right"),
            )
            writer.writeheader()
            writer.writerows(rows)

        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "identity": identity,
            "source_capture_root": str(source_root),
            "source_camera_file": str(cam_source),
            "selected_window_count": len(selected),
            "frame_count": len(rows),
            "copied_bytes": copied_bytes,
            "copied_assets": copied_assets,
            "selected_windows": selected_summary,
            "manifest_path": str(final_manifest),
        }
        (temp_root / "replay_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        os.replace(temp_root, out_root)
    except Exception:
        # The original capture is untouched. Keep the partial staging directory
        # for inspection instead of silently deleting files on an error.
        raise

    print(f"Created persistent replay set: {final_manifest}")
    print(
        f"Windows={len(selected):,}; frames={len(rows):,}; "
        f"copied={copied_bytes / 2**20:.1f} MiB; source capture={source_root.name}"
    )


if __name__ == "__main__":
    main()
