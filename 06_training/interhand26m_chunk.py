"""Convert one InterHand2.6M capture/camera into a lazy project manifest.

The source images stay in place; this writes only a compact manifest, camera
calibration, and one small right-hand 3D-keypoint JSON per selected frame.
Requires ``ijson`` for streaming the large annotation files.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import re
from pathlib import Path

import numpy as np


# InterHand order is thumb4..thumb1, index4..index1, ..., pinky4..pinky1,
# wrist. The tracker order is wrist, then each finger from proximal to tip.
INTERHAND_RIGHT_TO_TRACKER = np.asarray(
    [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16],
    dtype=np.int64,
)
PROJECT_KEEP_INDICES = np.asarray(
    [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22],
    dtype=np.int64,
)
RIGHT_INDEX = np.arange(21, dtype=np.int64)


def _write_text_if_changed(path: Path, content: str) -> bool:
    """Avoid changing persistent sidecar mtimes when regenerated content matches.

    The temporal trainer includes camera-file mtime in its dataset signature;
    rewriting identical calibration after a Colab reconnect would otherwise
    make an in-progress checkpoint appear to belong to different data.
    """
    if path.is_file():
        try:
            if path.read_text(encoding="utf-8") == content:
                return False
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)
    return True


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Extracted official InterHand2.6M root")
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--capture-id", required=True, type=int, help="Numeric InterHand capture, e.g. 0")
    p.add_argument(
        "--camera-id", action="append", default=[],
        help="Camera key from the official camera JSON; repeat to select multiple. Omit to include every camera in the capture.",
    )
    p.add_argument("--out", required=True, help="Temporary per-capture/camera manifest root")
    p.add_argument("--human-annotation-ids", default="", help="Official H annotation-ID list; required unless --all-annotations")
    p.add_argument("--all-annotations", action="store_true", help="Include H+M rows. M labels are machine annotations.")
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()


def _read_ids(path: Path) -> set[int]:
    raw = path.read_text(encoding="utf-8")
    values = re.findall(r"\d+", raw)
    if not values:
        raise ValueError(f"No numeric annotation IDs found in {path}")
    return {int(value) for value in values}


def _annotation_has_right(ann: dict) -> bool:
    hand_type = ann.get("hand_type", "")
    if isinstance(hand_type, str) and hand_type.lower() not in ("right", "interacting"):
        return False
    if not isinstance(hand_type, str):
        values = np.asarray(hand_type).reshape(-1)
        if not len(values) or float(values[0]) <= 0:
            return False
    validity = np.asarray(ann.get("joint_valid", []), dtype=np.float32).reshape(-1)
    return validity.size >= 42 and bool(validity[20] > 0)


def _camera_params(camera_json: dict, capture_id: int) -> dict:
    source = camera_json[str(capture_id)]
    result = {}
    for camera_id, campos in source["campos"].items():
        rotation = np.asarray(source["camrot"][camera_id], dtype=np.float32).reshape(3, 3)
        center = np.asarray(campos, dtype=np.float32).reshape(3)
        # Official world2cam is R @ (world - camera_position), in millimeters.
        translation = -(rotation @ center)
        result[str(camera_id)] = {
            "R": rotation.tolist(),
            "t": translation.tolist(),
            "focal": np.asarray(source["focal"][camera_id], dtype=np.float32).reshape(2).tolist(),
            "princpt": np.asarray(source["princpt"][camera_id], dtype=np.float32).reshape(2).tolist(),
        }
    return result


def _raw_project_order(canonical_world: np.ndarray) -> np.ndarray:
    if canonical_world.shape != (21, 3):
        raise ValueError(f"Expected canonical 21x3 joints, got {canonical_world.shape}")
    raw = np.zeros((24, 3), dtype=np.float32)
    raw[PROJECT_KEEP_INDICES] = canonical_world
    # Unselected entries are deliberately filled with the wrist and are never
    # consumed by KEEP_21_INDICES.
    raw[np.setdiff1d(np.arange(24), PROJECT_KEEP_INDICES)] = canonical_world[0]
    return raw


def main():
    args = _args()
    if abs(args.fps - 30.0) > 1e-6:
        raise SystemExit(
            "This adapter currently supports native 30 fps frame indexing only. "
            "The temporal reader interprets frame_idx / 30 as time."
        )
    try:
        import ijson
    except ImportError as exc:
        raise SystemExit("Install the streaming JSON reader first: pip install ijson") from exc

    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    split_root = root / "annotations" / args.split
    data_path = split_root / f"InterHand2.6M_{args.split}_data.json"
    camera_path = split_root / f"InterHand2.6M_{args.split}_camera.json"
    joints_path = split_root / f"InterHand2.6M_{args.split}_joint_3d.json"
    mano_path = split_root / f"InterHand2.6M_{args.split}_MANO_NeuralAnnot.json"
    image_root = root / "images" / args.split
    for path in (data_path, camera_path, joints_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    h_ids = None
    if args.human_annotation_ids:
        h_ids = _read_ids(Path(args.human_annotation_ids).expanduser().resolve())
    elif not args.all_annotations:
        raise ValueError("Pass --human-annotation-ids for H-only data, or explicitly choose --all-annotations.")

    capture_key = str(args.capture_id)
    with camera_path.open("r", encoding="utf-8") as stream:
        all_cameras = json.load(stream)
    camera_params = _camera_params(all_cameras, args.capture_id)
    requested_camera_ids = set(map(str, args.camera_id)) if args.camera_id else set(camera_params)
    missing_cameras = sorted(requested_camera_ids - set(camera_params))
    if missing_cameras:
        raise KeyError(f"Camera IDs missing from {camera_path}: {missing_cameras}")
    if not requested_camera_ids:
        raise ValueError(f"No cameras are calibrated for capture {capture_key}.")

    selected_images = {}
    print(
        f"Indexing image metadata for capture {capture_key}, "
        f"{len(requested_camera_ids)} camera(s): {', '.join(sorted(requested_camera_ids))}..."
    )
    with data_path.open("rb") as stream:
        for image in ijson.items(stream, "images.item"):
            if (
                str(image.get("capture")) != capture_key
                or str(image.get("camera")) not in requested_camera_ids
            ):
                continue
            selected_images[int(image["id"])] = image
    if not selected_images:
        raise ValueError("No images found for this capture/camera selection in the selected split.")
    found_cameras = {str(image["camera"]) for image in selected_images.values()}
    absent_cameras = requested_camera_ids - found_cameras
    if absent_cameras:
        raise ValueError(f"No image metadata found for requested cameras: {sorted(absent_cameras)}")
    print(
        f"Indexed {len(selected_images):,} image rows across {len(found_cameras)} camera(s); "
        "streaming annotations..."
    )

    annotations_by_image = {}
    with data_path.open("rb") as stream:
        for ann in ijson.items(stream, "annotations.item"):
            image_id = int(ann["image_id"])
            if image_id not in selected_images:
                continue
            if h_ids is not None and int(ann["id"]) not in h_ids:
                continue
            if not _annotation_has_right(ann):
                continue
            previous = annotations_by_image.get(image_id)
            score = int(np.asarray(ann["joint_valid"], dtype=np.float32)[:21].sum())
            if previous is None or score > previous[0]:
                annotations_by_image[image_id] = (score, int(ann["id"]))
    selected_images = {k: v for k, v in selected_images.items() if k in annotations_by_image}
    if not selected_images:
        raise ValueError("No selected images have a valid right hand under the annotation filter.")
    used_frames = {int(image["frame_idx"]) for image in selected_images.values()}

    label_dir = out_root / "keypoints_right"
    mano_dir = out_root / "mano_right"
    manifest_dir = out_root / "derived" / "mugsy_subset"
    label_dir.mkdir(parents=True, exist_ok=True)
    if mano_path.is_file():
        mano_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    camera_text = json.dumps(camera_params)
    _write_text_if_changed(out_root / "cam_params.json", camera_text)

    labels_written = set()
    with joints_path.open("rb") as stream:
        frame_iter = ijson.kvitems(stream, capture_key)
        for frame_key, entry in frame_iter:
            frame_idx = int(frame_key)
            if frame_idx not in used_frames:
                continue
            world = np.asarray(entry["world_coord"], dtype=np.float32)
            valid = np.asarray(entry.get("joint_valid", np.ones((42, 3))), dtype=np.float32)
            valid = valid.reshape(42, -1).max(axis=1)
            if world.shape != (42, 3) or not np.isfinite(world).all():
                continue
            if not np.all(valid[RIGHT_INDEX] > 0):
                continue
            canonical = world[INTERHAND_RIGHT_TO_TRACKER]
            label_path = label_dir / f"keypoint-{frame_idx:06d}.json"
            _write_text_if_changed(
                label_path,
                json.dumps(_raw_project_order(canonical).tolist()),
            )
            labels_written.add(frame_idx)

    if not labels_written:
        raise ValueError("No valid right-hand 3D labels were found for selected image frames.")

    mano_written = set()
    if mano_path.is_file():
        print("Extracting optional right-hand MANO shape sidecars...")
        with mano_path.open("rb") as stream:
            for frame_key, entry in ijson.kvitems(stream, capture_key):
                frame_idx = int(frame_key)
                if frame_idx not in labels_written:
                    continue
                right = entry.get("right") or {}
                shape = np.asarray(right.get("shape", []), dtype=np.float32).reshape(-1)
                if shape.size != 10 or not np.isfinite(shape).all():
                    continue
                pose = np.asarray(right.get("pose", []), dtype=np.float32).reshape(-1)
                trans = np.asarray(right.get("trans", []), dtype=np.float32).reshape(-1)
                mano_record = {"shape": shape.tolist()}
                if pose.size == 48 and np.isfinite(pose).all():
                    mano_record["pose"] = pose.tolist()
                if trans.size == 3 and np.isfinite(trans).all():
                    mano_record["trans"] = trans.tolist()
                mano_file = mano_dir / f"mano-{frame_idx:06d}.json"
                _write_text_if_changed(mano_file, json.dumps(mano_record))
                mano_written.add(frame_idx)

    manifest_path = manifest_dir / "manifest.csv"
    image_dir = image_root
    written = 0
    temp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    with temp_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("frame_id", "segment", "camera_id", "image_path", "orig_kp_right", "mano_right"),
        )
        writer.writeheader()
        for image_id, image in sorted(selected_images.items(), key=lambda item: (str(item[1]["seq_name"]), int(item[1]["frame_idx"]))):
            frame_idx = int(image["frame_idx"])
            if frame_idx not in labels_written:
                continue
            image_path = (image_dir / str(image["file_name"])).resolve()
            if not image_path.is_file():
                continue
            label_path = label_dir / f"keypoint-{frame_idx:06d}.json"
            writer.writerow({
                "frame_id": frame_idx,
                "segment": str(image["seq_name"]),
                "camera_id": str(image["camera"]),
                "image_path": str(image_path),
                "orig_kp_right": str(label_path.resolve()),
                "mano_right": str((mano_dir / f"mano-{frame_idx:06d}.json").resolve())
                if frame_idx in mano_written else "",
            })
            written += 1

    if written == 0:
        temp_manifest.unlink(missing_ok=True)
        raise ValueError(f"No manifest rows written; check image paths under {image_dir}")
    if manifest_path.is_file() and filecmp.cmp(temp_manifest, manifest_path, shallow=False):
        temp_manifest.unlink()
    else:
        os.replace(temp_manifest, manifest_path)
    print(f"Wrote {written:,} rows across {len(found_cameras)} camera(s): {manifest_path}")
    print(f"Camera calibration: {out_root / 'cam_params.json'}")
    print(f"World keypoints are mm; camera translation was converted from camera center to t=-R@center.")
    print(f"Keypoint mapping: right wrist plus each finger proximal-to-tip; timestamps use {args.fps:g} fps.")
    print(f"MANO shape sidecars: {len(mano_written):,} frames" if mano_path.is_file() else "MANO fits unavailable; state fitting will use the 3D joints only.")
    if h_ids is None:
        print("WARNING: H+M was requested. Machine annotations are included; keep validation on H labels.")


if __name__ == "__main__":
    main()
