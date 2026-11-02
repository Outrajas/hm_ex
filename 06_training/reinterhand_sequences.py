"""Streaming, sequence-safe reader for local Re:InterHand captures.

This module stores paths and reads one frame at a time; it never loads a whole
capture, image tree, or 1.2 TB corpus into RAM.  It supports the project's
derived Mugsy subset manifest and the official capture layout used for
``envmap_per_segment`` / ``envmap_per_frame`` images.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


KEEP_21_INDICES = np.asarray(
    [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22],
    dtype=np.int64,
)


@dataclass(frozen=True)
class ReInterHandRecord:
    capture_id: str
    segment: str
    frame_idx: int
    camera_id: str
    image_path: Path
    right_keypoints_path: Path
    right_mano_path: Optional[Path]
    timestamp_seconds: float


@dataclass
class ReInterHandFrame:
    record: ReInterHandRecord
    rgb: np.ndarray                 # H x W x 3, RGB uint8
    joints_right_mm: np.ndarray     # 21 x 3, project order, world mm
    joints_camera_mm: np.ndarray    # 21 x 3, project order, camera mm
    projected_uv: np.ndarray        # 21 x 2, image pixels
    mano_pose: np.ndarray           # 48 axis-angle values
    mano_shape: np.ndarray          # 10 values
    mano_trans: np.ndarray          # 3 values, source units preserved
    camera: dict


def _json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _resolve_capture_path(capture_root: Path, raw_path: str) -> Path:
    raw = Path(str(raw_path).replace("\\", "/"))
    return raw if raw.is_absolute() else capture_root / raw


def capture_root_from_manifest(manifest_path: Path) -> Path:
    """Return the official capture root or the root of a flattened subset."""
    manifest_path = Path(manifest_path).resolve()
    parent = manifest_path.parent
    if parent.name == "mugsy_subset" and parent.parent.name == "derived":
        return manifest_path.parents[2]
    return parent


def _resolve_flat_asset(root: Path, raw_path: str, column: str, camera_id: str) -> Path:
    raw = str(raw_path).replace("\\", "/").strip()
    direct = Path(raw)
    if direct.is_absolute() and direct.is_file():
        return direct
    for candidate in (root / raw, root / "dataset" / raw):
        if candidate.is_file():
            return candidate

    name = Path(raw).name
    if column == "orig_kp_right":
        folders = (root / "keypoints_right", root / "orig_fits" / "right" / "Keypoints")
    elif column == "image_path":
        folders = (
            root / "images" / str(camera_id),
            root / "images",
            root / "derived" / "mugsy_subset" / "images" / str(camera_id),
        )
    else:
        folders = (
            root / "mano_right",
            root / "mano_fits" / "params",
            root / "mano_params" / "right",
        )
    for folder in folders:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve {column} asset {raw!r} under {root}")


def records_from_subset_manifest(
    manifest_path: Path,
    *,
    camera_id: Optional[str] = None,
    frame_rate: float = 30.0,
) -> list[ReInterHandRecord]:
    """Index an existing ``derived/mugsy_subset/manifest.csv`` lazily."""
    manifest_path = Path(manifest_path).resolve()
    capture_root = capture_root_from_manifest(manifest_path)
    flattened = not (manifest_path.parent.name == "mugsy_subset" and manifest_path.parent.parent.name == "derived")
    result = []
    with manifest_path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            cam = str(row.get("camera_id", ""))
            if camera_id is not None and cam != str(camera_id):
                continue
            frame_idx = int(row["frame_id"])
            image_path = (
                _resolve_flat_asset(capture_root, row["image_path"], "image_path", cam)
                if flattened else _resolve_capture_path(capture_root, row["image_path"])
            )
            kp_path = (
                _resolve_flat_asset(capture_root, row["orig_kp_right"], "orig_kp_right", cam)
                if flattened else _resolve_capture_path(capture_root, row["orig_kp_right"])
            )
            mano_path = None
            raw_mano = row.get("mano_right", "")
            if raw_mano:
                try:
                    mano_path = (
                        _resolve_flat_asset(capture_root, raw_mano, "mano_right", cam)
                        if flattened else _resolve_capture_path(capture_root, raw_mano)
                    )
                except FileNotFoundError:
                    # MANO shape is optional for the 3D-keypoint state target.
                    mano_path = None
            result.append(ReInterHandRecord(
                capture_id=capture_root.name,
                segment=str(row["segment"]),
                frame_idx=frame_idx,
                camera_id=cam,
                image_path=image_path,
                right_keypoints_path=kp_path,
                right_mano_path=mano_path,
                timestamp_seconds=frame_idx / float(frame_rate),
            ))
    if not result:
        raise ValueError(f"No matching frames in {manifest_path}")
    return result


def records_from_capture(
    capture_root: Path,
    *,
    camera_group: str = "Mugsy_cameras",
    envmap_mode: str = "envmap_per_segment",
    camera_ids: Optional[Iterable[str]] = None,
    source_fps: float = 30.0,
) -> list[ReInterHandRecord]:
    """Index one complete official capture without opening its image files.

    ``frame_list.txt`` is the authoritative list for per-segment renders.  The
    per-frame Mugsy render is sparse; its image IDs are discovered from the
    chosen camera directory and joined to the capture's segment map.
    """
    capture_root = Path(capture_root).resolve()
    frame_list = capture_root / "frame_list.txt"
    camera_path = capture_root / camera_group / "cam_params.json"
    if not frame_list.is_file() or not camera_path.is_file():
        raise FileNotFoundError(f"Expected frame_list.txt and {camera_group}/cam_params.json under {capture_root}")
    cameras = _json(camera_path)
    allowed = set(map(str, camera_ids)) if camera_ids is not None else set(cameras)

    segment_by_frame: dict[int, str] = {}
    with frame_list.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.strip().split()
            if len(fields) >= 2:
                segment_by_frame[int(fields[1])] = fields[0]

    result = []
    for cam in sorted(allowed.intersection(cameras)):
        image_dir = capture_root / camera_group / envmap_mode / "images" / cam
        if not image_dir.is_dir():
            continue

        # The dense per-segment render uses frame_list.txt as its index. This
        # avoids walking millions of image entries just to construct a catalog.
        # The sparse per-frame mode has its own sampled filenames, so list only
        # that camera's smaller directory.
        if envmap_mode == "envmap_per_segment":
            frame_entries = [(frame, segment) for frame, segment in segment_by_frame.items()]
        else:
            frame_entries = []
            for image_path in image_dir.iterdir():
                if not image_path.is_file() or image_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                try:
                    frame_idx = int(image_path.stem)
                except ValueError:
                    continue
                segment = segment_by_frame.get(frame_idx)
                if segment is not None:
                    frame_entries.append((frame_idx, segment))

        for frame_idx, segment in frame_entries:
            image_path = image_dir / f"{frame_idx:06d}.png"
            if not image_path.is_file() and envmap_mode == "envmap_per_segment":
                alternate = image_dir / f"{frame_idx}.png"
                image_path = alternate if alternate.is_file() else image_path
            if not image_path.is_file() and envmap_mode != "envmap_per_segment":
                # Preserve the discovered extension for sampled per-frame files.
                image_path = next(
                    (p for p in image_dir.glob(f"{frame_idx}.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")),
                    image_path,
                )
            if not image_path.is_file():
                continue
            kp_path = capture_root / "orig_fits" / "right" / "Keypoints" / f"keypoint-{frame_idx:06d}.json"
            mano_path = capture_root / "mano_fits" / "params" / f"{frame_idx}_right.json"
            if not kp_path.is_file() or not mano_path.is_file():
                continue
            result.append(ReInterHandRecord(
                capture_id=capture_root.name,
                segment=segment,
                frame_idx=frame_idx,
                camera_id=str(cam),
                image_path=image_path,
                right_keypoints_path=kp_path,
                right_mano_path=mano_path,
                timestamp_seconds=frame_idx / float(source_fps),
            ))
    result.sort(key=lambda item: (item.capture_id, item.segment, item.camera_id, item.frame_idx))
    if not result:
        raise ValueError(f"No usable frames found under {capture_root / camera_group / envmap_mode}")
    return result


def split_by_group(
    records: list[ReInterHandRecord],
    *,
    validation_fraction: float = 0.15,
    seed: int = 20251018,
    group_by: str = "capture",
) -> tuple[np.ndarray, np.ndarray]:
    """Split by capture or segment, never by individual frame.

    For a one-capture subset, callers should use ``group_by='segment'``.  For
    the full data, use captures as groups to measure subject/capture transfer.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if group_by == "capture":
        keys = [(r.capture_id,) for r in records]
    elif group_by == "segment":
        keys = [(r.capture_id, r.segment) for r in records]
    else:
        raise ValueError("group_by must be 'capture' or 'segment'")
    groups = sorted(set(keys))
    if len(groups) < 2:
        raise ValueError(f"Need at least two {group_by} groups to make a leakage-free split")
    rng = np.random.default_rng(seed)
    order = np.arange(len(groups))
    rng.shuffle(order)
    n_val = min(len(groups) - 1, max(1, int(round(len(groups) * validation_fraction))))
    val_groups = {groups[int(i)] for i in order[:n_val]}
    train_groups = set(groups) - val_groups
    train = np.asarray([i for i, key in enumerate(keys) if key in train_groups], dtype=np.int64)
    val = np.asarray([i for i, key in enumerate(keys) if key in val_groups], dtype=np.int64)
    if set(keys[i] for i in train).intersection(keys[i] for i in val):
        raise RuntimeError("Internal split error: group leakage detected")
    return train, val


def sequence_windows(
    records: list[ReInterHandRecord],
    *,
    window_size: int,
    stride: int = 1,
    max_frame_gap: Optional[int] = 3,
) -> list[np.ndarray]:
    """Generate chronological windows; no window crosses segment/camera/gap."""
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive")
    groups: dict[tuple[str, str, str], list[tuple[int, ReInterHandRecord]]] = {}
    for idx, record in enumerate(records):
        groups.setdefault((record.capture_id, record.segment, record.camera_id), []).append((idx, record))

    windows: list[np.ndarray] = []
    for members in groups.values():
        members.sort(key=lambda pair: pair[1].frame_idx)
        runs: list[list[int]] = [[]]
        previous = None
        for index, record in members:
            if previous is not None and max_frame_gap is not None and record.frame_idx - previous > max_frame_gap:
                if runs[-1]:
                    runs.append([])
            runs[-1].append(index)
            previous = record.frame_idx
        for run in runs:
            if len(run) < window_size:
                continue
            for start in range(0, len(run) - window_size + 1, stride):
                windows.append(np.asarray(run[start:start + window_size], dtype=np.int64))
    return windows


def load_frame(record: ReInterHandRecord, camera_params: dict) -> ReInterHandFrame:
    """Read one RGB frame and labels. World keypoints remain in source mm."""
    image_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {record.image_path}")
    keypoints = np.asarray(_json(record.right_keypoints_path), dtype=np.float32)
    if keypoints.shape != (24, 3):
        raise ValueError(f"Expected 24 right-hand keypoints in {record.right_keypoints_path}; got {keypoints.shape}")
    if record.right_mano_path is not None and record.right_mano_path.is_file():
        params = _json(record.right_mano_path)
        pose = np.asarray(params["pose"], dtype=np.float32).reshape(-1)
        shape = np.asarray(params["shape"], dtype=np.float32).reshape(-1)
        trans = np.asarray(params["trans"], dtype=np.float32).reshape(-1)
    else:
        pose = np.zeros(48, np.float32)
        shape = np.zeros(10, np.float32)
        trans = np.zeros(3, np.float32)
    if pose.size != 48 or shape.size != 10 or trans.size != 3:
        raise ValueError(f"Unexpected MANO parameter dimensions at {record.right_mano_path}")
    camera = camera_params[record.camera_id]
    rotation = np.asarray(camera["R"], dtype=np.float32).reshape(3, 3)
    translation = np.asarray(camera["t"], dtype=np.float32).reshape(3)
    focal = np.asarray(camera["focal"], dtype=np.float32).reshape(2)
    principal = np.asarray(camera["princpt"], dtype=np.float32).reshape(2)
    joints_camera = keypoints[KEEP_21_INDICES] @ rotation.T + translation
    depth = np.maximum(joints_camera[:, 2], 1.0)
    projected = np.column_stack((
        focal[0] * joints_camera[:, 0] / depth + principal[0],
        focal[1] * joints_camera[:, 1] / depth + principal[1],
    )).astype(np.float32)
    return ReInterHandFrame(
        record=record,
        rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        joints_right_mm=keypoints[KEEP_21_INDICES].copy(),
        joints_camera_mm=joints_camera,
        projected_uv=projected,
        mano_pose=pose,
        mano_shape=shape,
        mano_trans=trans,
        camera=camera,
    )


def crop_frame(
    frame: ReInterHandFrame,
    *,
    output_size: int = 224,
    margin: float = 1.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Square RGB crop and matching projected labels for one right hand.

    Returns RGB uint8 ``(S,S,3)``, crop-space 2D points ``(21,2)``, and the
    2x3 image-to-crop affine matrix.  Callers can use the affine to transform
    augmentation labels consistently.
    """
    uv = np.asarray(frame.projected_uv, dtype=np.float32)
    low, high = uv.min(axis=0), uv.max(axis=0)
    center = (low + high) * 0.5
    side = max(float((high - low).max()), 16.0) * float(margin)
    left, top = center - side * 0.5
    scale = float(output_size) / side
    affine = np.asarray([
        [scale, 0.0, -left * scale],
        [0.0, scale, -top * scale],
    ], dtype=np.float32)
    crop = cv2.warpAffine(
        frame.rgb, affine, (output_size, output_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    crop_uv = uv @ affine[:, :2].T + affine[:, 2]
    return crop, crop_uv.astype(np.float32), affine
