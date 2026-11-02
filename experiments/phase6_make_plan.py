#!/usr/bin/env python
"""Pre-register capture chunks for reproducible multi-session training.

The plan may refer to manifests that have not been staged yet. The chunk
runner can follow this plan order or persist one seeded shuffle, and stops at
the first missing manifest until that capture has been staged and converted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-json", required=True, help="InterHand split camera JSON")
    parser.add_argument("--manifest-root", required=True, help="Persistent directory where per-capture converter outputs will be written")
    parser.add_argument("--validation-root", default="", help="Persistent root for the sampled held-out validation capture; default is MANIFEST_ROOT/validation")
    parser.add_argument("--out-plan", required=True)
    parser.add_argument("--holdout-capture", required=True, type=int, help="Capture ID reserved only for the fixed validation manifest")
    parser.add_argument("--capture-ids", nargs="*", type=int, default=None, help="Training capture IDs; default is every capture in the camera JSON")
    parser.add_argument("--camera-id", default="", help="Use one camera ID; otherwise choose a shared camera ID present in every selected capture")
    parser.add_argument("--all-cameras", action="store_true", help="Include every calibrated camera in each capture; keeps one training chunk per capture")
    parser.add_argument(
        "--cleanup-images-root", default="",
        help="Opt in to reclaim each staged CaptureN image folder after its checkpoint and replay are verified; must be under the runner's --staging-root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    camera_path = Path(args.camera_json).expanduser().resolve()
    manifest_root = Path(args.manifest_root).expanduser().resolve()
    validation_root = Path(args.validation_root).expanduser().resolve() if args.validation_root else manifest_root / "validation"
    out_path = Path(args.out_plan).expanduser().resolve()
    cleanup_images_root = (
        Path(args.cleanup_images_root).expanduser().resolve()
        if args.cleanup_images_root else None
    )
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Plan already exists: {out_path}; pass --overwrite to replace it.")

    camera_data = json.loads(camera_path.read_text(encoding="utf-8"))
    available_ids = sorted(map(int, camera_data))
    capture_ids = sorted(set(args.capture_ids if args.capture_ids is not None else available_ids))
    if args.holdout_capture not in capture_ids:
        raise ValueError("--holdout-capture must be included in the selected capture IDs")
    train_ids = [capture_id for capture_id in capture_ids if capture_id != args.holdout_capture]
    if not train_ids:
        raise ValueError("At least one non-validation training capture is required")
    missing = [capture_id for capture_id in capture_ids if str(capture_id) not in camera_data]
    if missing:
        raise ValueError(f"Capture IDs missing from camera JSON: {missing}")

    camera_sets = {
        str(capture_id): set(map(str, camera_data[str(capture_id)].get("campos", {})))
        for capture_id in capture_ids
    }
    common_cameras = set.intersection(*camera_sets.values()) if camera_sets else set()
    if args.all_cameras and args.camera_id:
        raise ValueError("Use --all-cameras or --camera-id, not both.")
    if args.all_cameras:
        camera_id = ""
        camera_tag = "ALLCAM"
    else:
        camera_id = str(args.camera_id) if args.camera_id else (sorted(common_cameras)[0] if common_cameras else "")
        if not camera_id:
            raise ValueError("No camera ID is shared across the selected captures; pass a camera ID available in all of them or use --all-cameras.")
        unavailable = [capture_id for capture_id in capture_ids if camera_id not in camera_sets[str(capture_id)]]
        if unavailable:
            raise ValueError(f"Camera {camera_id!r} is missing from capture IDs: {unavailable}")
        camera_tag = camera_id

    chunks = []
    for capture_id in train_ids:
        name = f"Capture{capture_id}_{camera_tag}"
        manifest = manifest_root / name / "derived" / "mugsy_subset" / "manifest.csv"
        chunk = {"name": name, "manifest": str(manifest)}
        if cleanup_images_root is not None:
            chunk["cleanup_root"] = str(cleanup_images_root / f"Capture{capture_id}")
        chunks.append(chunk)

    validation_name = f"Capture{args.holdout_capture}_{camera_tag}"
    validation_manifest = validation_root / validation_name / "derived" / "mugsy_subset" / "manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(out_path.name + ".tmp")
    temp_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    os.replace(temp_path, out_path)

    print(f"Wrote {len(chunks)} pre-registered training chunks: {out_path}")
    if args.all_cameras:
        print("Camera selection: every calibrated camera for each capture")
    else:
        print(f"Shared camera ID: {camera_id}")
    print(f"Reserved validation capture: {args.holdout_capture}")
    print(f"Persistent validation output root: {validation_root / validation_name}")
    print(f"Pass this separately to the runner: --validation-manifest {validation_manifest}")
    if cleanup_images_root is None:
        print("Staged source images will be retained after training.")
    else:
        print(
            "Opt-in cleanup enabled: each training CaptureN image folder is removed only "
            "after verified latest/best checkpoints and persistent replay."
        )
    print("The chunk runner will persist the requested capture order and wait at any unstaged manifest.")


if __name__ == "__main__":
    main()
