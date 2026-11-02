#!/usr/bin/env python
"""Build a compact temporal Re:InterHand rehearsal set from the existing ZIP.

This samples complete, non-overlapping windows across source segments and
copies only the selected images and labels. The source ZIP is read-only and is
never unpacked in full or modified.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import random
import zipfile
from pathlib import Path, PurePosixPath


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Existing Re:InterHand subset ZIP")
    parser.add_argument("--out", required=True, type=Path, help="New compact anchor ZIP; existing output is never overwritten")
    parser.add_argument("--max-windows", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--max-frame-gap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261022)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_member(names: set[str], row: dict[str, str], column: str) -> str:
    raw = str(row.get(column, "")).replace("\\", "/").strip()
    if not raw:
        raise ValueError(f"Manifest row {row.get('frame_id')} lacks {column}")
    basename = PurePosixPath(raw).name
    camera = str(row.get("camera_id", ""))
    candidates = [raw]
    if column == "image_path":
        candidates.extend((f"images/{camera}/{basename}", f"images/{basename}"))
    elif column == "orig_kp_right":
        candidates.extend((f"keypoints_right/{basename}", f"orig_fits/right/Keypoints/{basename}"))
    elif column == "mano_right":
        candidates.extend((f"mano_right/{basename}", f"mano_fits/params/{basename}"))
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise FileNotFoundError(
        f"Could not map manifest {column}={raw!r} to a member in the source ZIP"
    )


def build_windows(rows, window_size: int, stride: int, max_gap: int):
    groups = {}
    for index, row in enumerate(rows):
        key = (str(row["segment"]), str(row["camera_id"]))
        groups.setdefault(key, []).append((index, int(row["frame_id"])))
    windows_by_group = {}
    for key, members in groups.items():
        members.sort(key=lambda item: item[1])
        runs = []
        current = []
        previous = None
        for index, frame in members:
            if previous is not None and frame - previous > max_gap:
                if current:
                    runs.append(current)
                current = []
            current.append(index)
            previous = frame
        if current:
            runs.append(current)
        windows = []
        for run in runs:
            for start in range(0, len(run) - window_size + 1, stride):
                windows.append(run[start:start + window_size])
        if windows:
            windows_by_group[key] = windows
    return windows_by_group


def select_stratified_windows(windows_by_group, limit: int, seed: int):
    rng = random.Random(seed)
    groups = list(sorted(windows_by_group))
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(windows_by_group[group])
    selected = []
    while len(selected) < limit:
        advanced = False
        for group in groups:
            if windows_by_group[group] and len(selected) < limit:
                selected.append((group, windows_by_group[group].pop()))
                advanced = True
        if not advanced:
            break
    return selected


def write_member_from_source(source_zip, output_zip, source_name, output_name):
    source_info = source_zip.getinfo(source_name)
    out_info = zipfile.ZipInfo(output_name)
    out_info.compress_type = (
        zipfile.ZIP_STORED if PurePosixPath(output_name).suffix.lower() == ".png"
        else zipfile.ZIP_DEFLATED
    )
    digest = hashlib.sha256()
    count = 0
    with source_zip.open(source_name, "r") as source, output_zip.open(out_info, "w") as target:
        while True:
            block = source.read(4 * 1024 * 1024)
            if not block:
                break
            target.write(block)
            digest.update(block)
            count += len(block)
    if count != source_info.file_size:
        raise IOError(f"Short read from {source_name}: {count}/{source_info.file_size} bytes")
    return {"path": output_name, "size": count, "sha256": digest.hexdigest()}


def main():
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing anchor ZIP: {output}")
    if args.max_windows < 1 or args.window_size < 2 or args.stride < args.window_size:
        raise ValueError("Require max-windows >= 1, window-size >= 2, and stride >= window-size")
    if args.max_frame_gap < 1:
        raise ValueError("--max-frame-gap must be positive")

    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"An interrupted build exists at {temporary}; inspect it before retrying")
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive, "r") as source:
        names = set(source.namelist())
        if "manifest.csv" not in names or "cam_params.json" not in names:
            raise ValueError("Source ZIP must contain manifest.csv and cam_params.json at its root")
        manifest_bytes = source.read("manifest.csv")
        camera_bytes = source.read("cam_params.json")
        rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8-sig"))))
        if not rows:
            raise ValueError("Source manifest is empty")
        cameras = json.loads(camera_bytes)

        records = []
        for row in rows:
            camera_id = str(row.get("camera_id", ""))
            if camera_id not in cameras:
                raise KeyError(f"Camera {camera_id!r} absent from cam_params.json")
            records.append({
                "frame_id": int(row["frame_id"]),
                "segment": str(row["segment"]),
                "camera_id": camera_id,
                "image_member": resolve_member(names, row, "image_path"),
                "keypoint_member": resolve_member(names, row, "orig_kp_right"),
                "mano_member": resolve_member(names, row, "mano_right") if row.get("mano_right") else "",
            })

        windows_by_group = build_windows(
            records, args.window_size, args.stride, args.max_frame_gap
        )
        windows = select_stratified_windows(
            windows_by_group, args.max_windows, args.seed
        )
        if not windows:
            raise ValueError("No complete temporal windows found in the source manifest")
        selected_records = {index for _, window in windows for index in window}
        selected_rows = []
        asset_inventory = []
        copied_bytes = 0

        try:
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
            ) as target:
                target.writestr("cam_params.json", camera_bytes)
                for window_number, (group, window) in enumerate(windows, start=1):
                    segment_name = f"replay_window_{window_number:06d}"
                    for frame_number, record_index in enumerate(window, start=1):
                        record = records[record_index]
                        stem = (
                            f"w{window_number:06d}_f{frame_number:02d}_"
                            f"{record['frame_id']:08d}_{record['camera_id']}"
                        )
                        image_suffix = PurePosixPath(record["image_member"]).suffix.lower() or ".png"
                        image_path = f"images/{record['camera_id']}/{stem}{image_suffix}"
                        keypoint_path = f"keypoints_right/{stem}.json"
                        mano_path = f"mano_right/{stem}_right.json" if record["mano_member"] else ""
                        for source_name, output_name in (
                            (record["image_member"], image_path),
                            (record["keypoint_member"], keypoint_path),
                        ):
                            asset = write_member_from_source(
                                source, target, source_name, output_name
                            )
                            asset["source_member"] = source_name
                            asset_inventory.append(asset)
                            copied_bytes += asset["size"]
                        if record["mano_member"]:
                            asset = write_member_from_source(
                                source, target, record["mano_member"], mano_path
                            )
                            asset["source_member"] = record["mano_member"]
                            asset_inventory.append(asset)
                            copied_bytes += asset["size"]
                        selected_rows.append({
                            "frame_id": record["frame_id"],
                            "segment": segment_name,
                            "camera_id": record["camera_id"],
                            "image_path": image_path,
                            "orig_kp_right": keypoint_path,
                            "mano_right": mano_path,
                        })
                    if window_number % 16 == 0 or window_number == len(windows):
                        print(
                            f"Prepared Re:InterHand windows {window_number}/{len(windows)}; "
                            f"frames={len(selected_rows):,}; assets={copied_bytes / 2**20:.1f} MiB",
                            flush=True,
                        )

                manifest_stream = io.StringIO(newline="")
                fieldnames = (
                    "frame_id", "segment", "camera_id", "image_path",
                    "orig_kp_right", "mano_right",
                )
                writer = csv.DictWriter(manifest_stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(selected_rows)
                manifest_out = manifest_stream.getvalue().encode("utf-8")
                target.writestr("derived/mugsy_subset/manifest.csv", manifest_out)

                source_segments = sorted({group[0] for group, _ in windows})
                metadata = {
                    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "source_archive": str(archive),
                    "source_archive_size": archive.stat().st_size,
                    "source_manifest_sha256": sha256_bytes(manifest_bytes),
                    "source_rows": len(rows),
                    "window_size": args.window_size,
                    "stride": args.stride,
                    "max_frame_gap": args.max_frame_gap,
                    "seed": args.seed,
                    "selected_window_count": len(windows),
                    "selected_frame_count": len(selected_rows),
                    "source_segments_covered": source_segments,
                    "copied_asset_bytes": copied_bytes,
                    "copied_assets": asset_inventory,
                    "manifest_sha256": sha256_bytes(manifest_out),
                }
                target.writestr(
                    "replay_metadata.json",
                    json.dumps(metadata, indent=2).encode("utf-8"),
                )
        except Exception:
            print(f"Incomplete anchor build retained for inspection: {temporary}")
            raise

    temporary.replace(output)
    print(
        f"Created {output}: {len(windows):,} windows, {len(selected_rows):,} frames, "
        f"{len(source_segments):,} source segments, {copied_bytes / 2**20:.1f} MiB of assets."
    )
    print("The source archive was not modified. Extract this ZIP into a persistent Drive folder before training.")


if __name__ == "__main__":
    main()
