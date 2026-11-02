#!/usr/bin/env python
"""Read a raw InterHand multipart-tar prefix without extracting image payloads.

This is a read-only probe. Ordered parts are exposed as one seekable virtual
archive, so headers and payloads split across boundaries are counted correctly.
It reports visible image paths grouped by capture and the final path reached.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import re
import tarfile
import time
from collections import defaultdict
from pathlib import Path


CAPTURE_RE = re.compile(r"(?:^|/)(Capture\d+)(?:/|$)", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class ConcatenatedParts:
    """Seekable virtual concatenation of split archive files; never copies data."""

    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.files = [path.open("rb") for path in paths]
        self.sizes = [path.stat().st_size for path in paths]
        self.ends = []
        total = 0
        for size in self.sizes:
            total += size
            self.ends.append(total)
        self.size = total
        self.position = 0
        self.name = ",".join(str(path) for path in paths)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 1:
            offset += self.position
        elif whence == 2:
            offset += self.size
        elif whence != 0:
            raise ValueError(f"Invalid seek mode: {whence}")
        if offset < 0:
            raise ValueError("Negative seek position")
        self.position = int(offset)
        return self.position

    def read(self, size=-1):
        if size is None or size < 0:
            size = max(0, self.size - self.position)
        remaining = min(int(size), max(0, self.size - self.position))
        chunks = []
        while remaining:
            file_index = bisect_right(self.ends, self.position)
            if file_index >= len(self.files):
                break
            file_start = 0 if file_index == 0 else self.ends[file_index - 1]
            local_position = self.position - file_start
            self.files[file_index].seek(local_position)
            requested = min(remaining, self.sizes[file_index] - local_position)
            block = self.files[file_index].read(requested)
            if not block:
                break
            chunks.append(block)
            self.position += len(block)
            remaining -= len(block)
        return b"".join(chunks)

    def close(self):
        for stream in self.files:
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive_parts", type=Path, nargs="+",
        help="One or more .tar.partXX objects in their original order; a prefix is scanned as one archive",
    )
    parser.add_argument("--max-members", type=int, default=0, help="Stop after this many tar members; 0 scans the whole part")
    return parser.parse_args()


def main():
    args = parse_args()
    parts = [path.expanduser().resolve() for path in args.archive_parts]
    missing_parts = [path for path in parts if not path.is_file()]
    if missing_parts:
        raise FileNotFoundError("Missing archive part(s): " + ", ".join(map(str, missing_parts)))
    if args.max_members < 0:
        raise ValueError("--max-members cannot be negative")

    images_by_capture = defaultdict(lambda: {"count": 0, "declared_bytes": 0, "first": None, "last": None})
    first_member = None
    last_member = None
    member_count = 0
    image_count = 0
    complete_image_count = 0
    incomplete_image_count = 0
    complete_image_bytes = 0
    first_incomplete_image = None
    scan_issue = None
    archive_bytes = sum(path.stat().st_size for path in parts)
    last_progress = time.monotonic()

    try:
        # Random-access mode seeks over image payloads. The virtual reader makes
        # headers/payloads that straddle physical parts appear contiguous.
        with ConcatenatedParts(parts) as stream, tarfile.open(fileobj=stream, mode="r:") as archive:
            for member in archive:
                member_count += 1
                first_member = first_member or member.name
                last_member = member.name
                if member.isfile() and Path(member.name).suffix.lower() in IMAGE_SUFFIXES:
                    match = CAPTURE_RE.search(member.name.replace("\\", "/"))
                    capture = match.group(1) if match else "capture_not_in_path"
                    row = images_by_capture[capture]
                    row["count"] += 1
                    row["declared_bytes"] += int(member.size)
                    row["first"] = row["first"] or member.name
                    row["last"] = member.name
                    image_count += 1
                    if member.offset_data + member.size <= archive_bytes:
                        row["complete_count"] = row.get("complete_count", 0) + 1
                        row["complete_bytes"] = row.get("complete_bytes", 0) + int(member.size)
                        complete_image_count += 1
                        complete_image_bytes += int(member.size)
                    else:
                        row["incomplete_count"] = row.get("incomplete_count", 0) + 1
                        incomplete_image_count += 1
                        first_incomplete_image = first_incomplete_image or member.name
                if args.max_members and member_count >= args.max_members:
                    break
                now = time.monotonic()
                if member_count % 50000 == 0 or now - last_progress >= 30:
                    scanned = stream.tell()
                    print(
                        f"Tar scan: members={member_count:,}; images={image_count:,}; "
                        f"input={scanned / 2**30:.2f}/{archive_bytes / 2**30:.2f} GiB",
                        flush=True,
                    )
                    last_progress = now
    except (tarfile.TarError, EOFError, OSError) as exc:
        # A part boundary can split a tar record. Preserve and report entries
        # successfully indexed before that boundary instead of calling it a
        # complete archive or losing the useful prefix inventory.
        scan_issue = f"{type(exc).__name__}: {exc}"

    report = {
        "archive_parts_in_order": [str(path) for path in parts],
        "prefix_bytes": archive_bytes,
        "members_indexed": member_count,
        "image_files_indexed": image_count,
        "complete_image_files": complete_image_count,
        "complete_image_payload_bytes": complete_image_bytes,
        "incomplete_image_files": incomplete_image_count,
        "first_incomplete_image": first_incomplete_image,
        "first_member": first_member,
        "last_member": last_member,
        "capture_groups": dict(sorted(images_by_capture.items())),
        "scan_issue": scan_issue,
        "warning": "Only complete_image_files have payloads fully inside this part. Even complete files do not prove a capture or temporal sequence is complete.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
