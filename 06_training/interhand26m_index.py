#!/usr/bin/env python
"""Index InterHand2.6M H labels across captures with bounded-memory passes.

The split metadata file is streamed once for annotations and once for image
metadata; the large 3D-joint and optional MANO files are each streamed once for
all selected captures. This avoids reparsing those label files per capture.
Image paths are written even when their payloads have not been staged; the
chunk runner blocks training until every required image exists.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import time
from pathlib import Path, PurePosixPath

try:
    import ijson
except ImportError as exc:
    raise SystemExit("Install the streaming JSON reader first: pip install ijson") from exc

import numpy as np

# Running this file directly puts 06_training on sys.path.
from interhand26m_chunk import (
    INTERHAND_RIGHT_TO_TRACKER,
    _annotation_has_right,
    _camera_params,
    _raw_project_order,
    _read_ids,
    _write_text_if_changed,
    RIGHT_INDEX,
)


class ProgressReader:
    """Binary reader that reports bounded-memory parser progress periodically."""

    def __init__(self, path: Path, label: str):
        self.path = path
        self.label = label
        self.total = path.stat().st_size
        self.bytes_read = 0
        self.last_report = time.monotonic()
        self.next_report = 256 * 1024 * 1024
        self.stream = path.open("rb")

    def read(self, size=-1):
        data = self.stream.read(size)
        self.bytes_read += len(data)
        now = time.monotonic()
        if self.bytes_read >= self.next_report or now - self.last_report >= 30:
            pct = 100.0 * self.bytes_read / max(self.total, 1)
            print(
                f"{self.label}: {self.bytes_read / 2**30:.2f}/"
                f"{self.total / 2**30:.2f} GiB ({pct:.1f}%)",
                flush=True,
            )
            self.last_report = now
            while self.next_report <= self.bytes_read:
                self.next_report += 256 * 1024 * 1024
        return data

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Official extracted InterHand2.6M root")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--human-annotation-ids", required=True, help="Official H annotation-ID list")
    parser.add_argument("--out-root", required=True, help="Persistent root for CaptureN_ALLCAM manifests and labels")
    parser.add_argument("--capture-ids", nargs="*", type=int, default=None, help="Captures to index; default is every capture in the camera JSON")
    parser.add_argument(
        "--validation-capture-id", type=int, default=None,
        help="Route this selected capture to --validation-root instead of --out-root.",
    )
    parser.add_argument(
        "--validation-root", default="",
        help="Persistent output root for the held-out capture; defaults to OUT_ROOT/validation.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing different manifest")
    return parser.parse_args()


def _capture_name(capture_id: str) -> str:
    return f"Capture{int(capture_id)}_ALLCAM"


def _stream_selected_joint_labels(joints_path, selected_frames, out_roots):
    written = {capture: set() for capture in selected_frames}
    current_capture = None
    current_frame = None
    current_frame_key = None
    selected = False
    world_values = []
    valid_values = []
    total_selected = sum(len(frame_ids) for frame_ids in selected_frames.values())
    processed = 0

    with ProgressReader(joints_path, "3D joint labels") as stream:
        for prefix, event, value in ijson.parse(stream):
            if prefix == "" and event == "map_key":
                current_capture = str(value)
                current_frame = None
                current_frame_key = None
                selected = False
                continue

            if current_capture not in selected_frames:
                continue

            if prefix == current_capture and event == "map_key":
                try:
                    current_frame = int(value)
                    current_frame_key = str(value)
                except (TypeError, ValueError):
                    current_frame = None
                    current_frame_key = None
                selected = current_frame in selected_frames[current_capture] if current_frame is not None else False
                if selected:
                    world_values = []
                    valid_values = []
                continue

            if not selected or current_frame is None:
                continue

            base = f"{current_capture}.{current_frame_key}"
            if prefix.startswith(base + ".world_coord") and event == "number":
                world_values.append(value)
            elif prefix.startswith(base + ".joint_valid") and event in ("number", "boolean"):
                valid_values.append(float(value))
            elif prefix == base and event == "end_map":
                world = np.asarray(world_values, dtype=np.float32)
                if world.size == 126:
                    world = world.reshape(42, 3)
                    valid = (
                        np.asarray(valid_values, dtype=np.float32).reshape(42, -1).max(axis=1)
                        if valid_values else np.ones(42, dtype=np.float32)
                    )
                    if np.isfinite(world).all() and np.all(valid[RIGHT_INDEX] > 0):
                        canonical = world[INTERHAND_RIGHT_TO_TRACKER]
                        label_dir = (
                            out_roots[current_capture]
                            / "keypoints_right"
                            / f"{current_frame // 1000:04d}"
                        )
                        label_path = label_dir / f"keypoint-{current_frame:06d}.json"
                        _write_text_if_changed(
                            label_path,
                            json.dumps(_raw_project_order(canonical).tolist()),
                        )
                        written[current_capture].add(current_frame)

                processed += 1
                selected = False
                current_frame = None
                current_frame_key = None
                if processed % 10000 == 0 or processed == total_selected:
                    print(f"Indexed 3D labels {processed:,}/{total_selected:,}", flush=True)

    return written


def _stream_selected_mano(mano_path, selected_frames, valid_label_frames, out_roots):
    if not mano_path.is_file():
        print("MANO NeuralAnnot file not present; 3D-joint labels will be used without per-frame MANO fits.")
        return {capture: set() for capture in selected_frames}

    written = {capture: set() for capture in selected_frames}
    current_capture = None
    current_frame = None
    current_frame_key = None
    selected = False
    values = {"shape": [], "pose": [], "trans": []}
    total_selected = sum(len(frame_ids) for frame_ids in valid_label_frames.values())
    processed = 0

    with ProgressReader(mano_path, "MANO labels") as stream:
        for prefix, event, value in ijson.parse(stream):
            if prefix == "" and event == "map_key":
                current_capture = str(value)
                current_frame = None
                current_frame_key = None
                selected = False
                continue

            if current_capture not in selected_frames:
                continue

            if prefix == current_capture and event == "map_key":
                try:
                    current_frame = int(value)
                    current_frame_key = str(value)
                except (TypeError, ValueError):
                    current_frame = None
                    current_frame_key = None
                selected = (
                    current_frame in valid_label_frames[current_capture]
                    if current_frame is not None else False
                )
                if selected:
                    values = {"shape": [], "pose": [], "trans": []}
                continue

            if not selected or current_frame is None:
                continue

            frame_base = f"{current_capture}.{current_frame_key}"
            base = f"{frame_base}.right"
            if event == "number":
                for key in values:
                    if prefix.startswith(f"{base}.{key}"):
                        values[key].append(value)
                        break
            elif prefix == frame_base and event == "end_map":
                shape = np.asarray(values["shape"], dtype=np.float32)
                if shape.size == 10 and np.isfinite(shape).all():
                    record = {"shape": shape.tolist()}
                    pose = np.asarray(values["pose"], dtype=np.float32)
                    trans = np.asarray(values["trans"], dtype=np.float32)
                    if pose.size == 48 and np.isfinite(pose).all():
                        record["pose"] = pose.tolist()
                    if trans.size == 3 and np.isfinite(trans).all():
                        record["trans"] = trans.tolist()
                    mano_dir = (
                        out_roots[current_capture]
                        / "mano_right"
                        / f"{current_frame // 1000:04d}"
                    )
                    mano_path_out = mano_dir / f"mano-{current_frame:06d}.json"
                    _write_text_if_changed(mano_path_out, json.dumps(record))
                    written[current_capture].add(current_frame)

                processed += 1
                selected = False
                current_frame = None
                current_frame_key = None
                if processed % 10000 == 0 or processed == total_selected:
                    print(f"Indexed MANO labels {processed:,}/{total_selected:,}", flush=True)

    return written


def main():
    args = parse_args()
    if abs(args.fps - 30.0) > 1e-6:
        raise ValueError("The temporal reader currently assumes native 30 fps frame indices.")

    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    annotation_root = root / "annotations" / args.split
    data_path = annotation_root / f"InterHand2.6M_{args.split}_data.json"
    camera_path = annotation_root / f"InterHand2.6M_{args.split}_camera.json"
    joints_path = annotation_root / f"InterHand2.6M_{args.split}_joint_3d.json"
    mano_path = annotation_root / f"InterHand2.6M_{args.split}_MANO_NeuralAnnot.json"
    image_root = root / "images" / args.split
    for required in (data_path, camera_path, joints_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    human_ids = _read_ids(Path(args.human_annotation_ids).expanduser().resolve())
    camera_data = json.loads(camera_path.read_text(encoding="utf-8"))
    available_captures = set(camera_data)
    selected_captures = (
        {str(value) for value in args.capture_ids}
        if args.capture_ids is not None else available_captures
    )
    missing_captures = sorted(selected_captures - available_captures)
    if missing_captures:
        raise ValueError(f"Capture IDs missing from camera JSON: {missing_captures}")
    if not selected_captures:
        raise ValueError("No capture IDs selected.")

    validation_capture = (
        str(args.validation_capture_id)
        if args.validation_capture_id is not None else None
    )
    if validation_capture is not None and validation_capture not in selected_captures:
        raise ValueError(
            "--validation-capture-id must be included in --capture-ids "
            "(or in the camera JSON when indexing all captures)."
        )
    if args.validation_root and validation_capture is None:
        raise ValueError("--validation-root requires --validation-capture-id.")
    validation_root = (
        Path(args.validation_root).expanduser().resolve()
        if args.validation_root else out_root / "validation"
    )

    out_roots = {
        capture: (validation_root if capture == validation_capture else out_root)
        / _capture_name(capture)
        for capture in sorted(selected_captures, key=int)
    }
    camera_params_by_capture = {
        capture: _camera_params(camera_data, int(capture))
        for capture in out_roots
    }

    print(f"Indexing H annotations once from {data_path}...")
    annotations_by_image = {}
    with ProgressReader(data_path, "H annotation index") as stream:
        for annotation in ijson.items(stream, "annotations.item"):
            if int(annotation.get("id", -1)) not in human_ids:
                continue
            if not _annotation_has_right(annotation):
                continue
            image_id = int(annotation["image_id"])
            score = int(np.asarray(annotation["joint_valid"], dtype=np.float32)[:21].sum())
            previous = annotations_by_image.get(image_id)
            if previous is None or score > previous:
                annotations_by_image[image_id] = score
    if not annotations_by_image:
        raise ValueError("No right-hand Train (H) annotations matched the supplied ID list.")
    print(f"Selected {len(annotations_by_image):,} human-labeled image annotations.")

    index_root = out_root / ".interhand_index"
    index_root.mkdir(parents=True, exist_ok=True)
    temp_paths = {}
    streams = {}
    selected_frames = {capture: set() for capture in selected_captures}
    image_counts = {capture: 0 for capture in selected_captures}
    found_image_ids = set()
    try:
        for capture, capture_root in out_roots.items():
            path = index_root / f"{_capture_name(capture)}.jsonl.tmp"
            temp_paths[capture] = path
            streams[capture] = path.open("w", encoding="utf-8")

        print("Indexing selected image metadata once...")
        with ProgressReader(data_path, "Selected image metadata") as stream:
            for image in ijson.items(stream, "images.item"):
                image_id = int(image["id"])
                if image_id not in annotations_by_image:
                    continue
                capture = str(image.get("capture"))
                if capture not in selected_captures:
                    continue
                camera_id = str(image.get("camera", ""))
                if camera_id not in camera_params_by_capture[capture]:
                    raise KeyError(f"Camera {camera_id!r} absent from capture {capture} calibration.")
                row = {
                    "image_id": image_id,
                    "frame_idx": int(image["frame_idx"]),
                    "segment": str(image["seq_name"]),
                    "camera_id": camera_id,
                    "file_name": str(image["file_name"]),
                }
                streams[capture].write(json.dumps(row, separators=(",", ":")) + "\n")
                selected_frames[capture].add(row["frame_idx"])
                image_counts[capture] += 1
                found_image_ids.add(image_id)
    finally:
        for handle in streams.values():
            handle.close()

    for capture, temp_path in temp_paths.items():
        os.replace(temp_path, index_root / f"{_capture_name(capture)}.jsonl")
    if selected_captures == available_captures:
        missing_image_rows = len(annotations_by_image) - len(found_image_ids)
        if missing_image_rows:
            print(f"Warning: {missing_image_rows:,} selected annotations had no matching image metadata.")
    del annotations_by_image, found_image_ids
    for capture in selected_frames:
        if not selected_frames[capture]:
            raise ValueError(f"No human-labeled images found for selected capture {capture}.")

    print(
        f"Selected {sum(image_counts.values()):,} images across "
        f"{sum(len(frames) for frames in selected_frames.values()):,} unique capture/frame pairs."
    )
    labelled_frames = _stream_selected_joint_labels(joints_path, selected_frames, out_roots)
    mano_frames = _stream_selected_mano(mano_path, selected_frames, labelled_frames, out_roots)

    for capture, capture_root in out_roots.items():
        _write_text_if_changed(capture_root / "cam_params.json", json.dumps(camera_params_by_capture[capture]))
        manifest_dir = capture_root / "derived" / "mugsy_subset"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest.csv"
        temp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        rows_written = 0
        with (index_root / f"{_capture_name(capture)}.jsonl").open("r", encoding="utf-8") as source, temp_manifest.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=("frame_id", "segment", "camera_id", "image_path", "orig_kp_right", "mano_right"),
            )
            writer.writeheader()
            for line in source:
                image = json.loads(line)
                frame_idx = int(image["frame_idx"])
                if frame_idx not in labelled_frames[capture]:
                    continue
                relative_image = PurePosixPath(str(image["file_name"]).replace("\\", "/"))
                if not relative_image.parts or relative_image.is_absolute() or ".." in relative_image.parts:
                    raise ValueError(f"Image path escapes the InterHand split root: {relative_image}")
                image_path = image_root.joinpath(*relative_image.parts)
                kp_path = capture_root / "keypoints_right" / f"keypoint-{frame_idx:06d}.json"
                mano_output = capture_root / "mano_right" / f"mano-{frame_idx:06d}.json"
                writer.writerow({
                    "frame_id": frame_idx,
                    "segment": image["segment"],
                    "camera_id": image["camera_id"],
                    "image_path": str(image_path),
                    "orig_kp_right": str(kp_path.resolve()),
                    "mano_right": str(mano_output.resolve()) if frame_idx in mano_frames[capture] else "",
                })
                rows_written += 1

        if manifest_path.is_file() and filecmp.cmp(temp_manifest, manifest_path, shallow=False):
            temp_manifest.unlink()
        elif manifest_path.exists() and not args.overwrite:
            temp_manifest.unlink(missing_ok=True)
            raise FileExistsError(f"A different manifest already exists: {manifest_path}; pass --overwrite to replace it.")
        else:
            os.replace(temp_manifest, manifest_path)

        if not rows_written:
            raise ValueError(f"No manifest rows written for capture {capture}.")
        print(
            f"Capture {capture}: rows={rows_written:,}; cameras={len(camera_params_by_capture[capture])}; "
            f"image checks deferred; labels={len(labelled_frames[capture]):,}; "
            f"MANO={len(mano_frames[capture]):,}; manifest={manifest_path}",
            flush=True,
        )

    print("Index complete. Image paths may be absent until the corresponding archive images are staged; chunk preflight will enforce their presence before training.")


if __name__ == "__main__":
    main()
