#!/usr/bin/env python
"""Stage one InterHand capture from ordered split-tar parts.

Only image members under CaptureN are written. The split parts are read as one
virtual tar stream, so a tar header or image payload may cross a part boundary
without requiring an end-of-archive marker in the last downloaded part.
When given the capture manifest, the command reports whether every expected
image is now present; an incomplete capture exits nonzero and is not ready for
the chunk trainer.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath

from phase6_inspect_interhand_tar import ConcatenatedParts


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
COPY_BUFFER = 4 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-id", type=int, required=True)
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Image split root, e.g. /content/stage/interhand/images/train",
    )
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Persistent CaptureN_ALLCAM manifest used to verify completeness",
    )
    parser.add_argument(
        "archive_parts", type=Path, nargs="+",
        help="Consecutive InterHand .tar.partXX files in their original order",
    )
    return parser.parse_args()


def member_capture_path(name: str, capture_name: str) -> PurePosixPath | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    try:
        capture_pos = next(i for i, part in enumerate(parts) if part.lower() == capture_name.lower())
    except StopIteration:
        return None
    relative = PurePosixPath(*parts[capture_pos:])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe capture-relative archive path: {name!r}")
    return relative


def manifest_images(manifest: Path, output_root: Path) -> set[Path]:
    expected = set()
    with manifest.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw_path = row.get("image_path", "").strip()
            if not raw_path:
                raise ValueError("Manifest has an empty image_path")
            image_path = Path(raw_path).expanduser().resolve()
            try:
                image_path.relative_to(output_root)
            except ValueError as exc:
                raise ValueError(
                    f"Manifest image is outside --output-root: {image_path}"
                ) from exc
            expected.add(image_path)
    if not expected:
        raise ValueError(f"Manifest has no image rows: {manifest}")
    return expected


def main():
    args = parse_args()
    capture_name = f"Capture{args.capture_id}"
    output_root = args.output_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    parts = [path.expanduser().resolve() for path in args.archive_parts]
    missing_parts = [path for path in parts if not path.is_file()]
    if missing_parts:
        raise FileNotFoundError("Missing archive part(s): " + ", ".join(map(str, missing_parts)))
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    expected = manifest_images(manifest, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root = output_root / ".phase6_extract_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    reused = 0
    copied_bytes = 0
    member_count = 0
    issue = None
    fatal_issue = None
    archive_bytes = sum(path.stat().st_size for path in parts)
    last_progress = time.monotonic()

    print(f"Capture: {capture_name}")
    print(f"Archive parts: {len(parts)}; manifest images expected: {len(expected):,}")
    print(f"Staging under: {output_root}")

    try:
        with ConcatenatedParts(parts) as virtual_archive:
            with tarfile.open(fileobj=virtual_archive, mode="r|") as archive:
                for member in archive:
                    member_count += 1
                    now = time.monotonic()
                    if member_count % 50000 == 0 or now - last_progress >= 30:
                        scanned = virtual_archive.tell()
                        print(
                            f"Tar stage scan: members={member_count:,}; "
                            f"input={scanned / 2**30:.2f}/{archive_bytes / 2**30:.2f} GiB; "
                            f"copied={copied:,}; reused={reused:,}",
                            flush=True,
                        )
                        last_progress = now
                    if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    relative = member_capture_path(member.name, capture_name)
                    if relative is None:
                        continue

                    destination = (output_root / Path(*relative.parts)).resolve()
                    try:
                        destination.relative_to(output_root)
                    except ValueError as exc:
                        raise ValueError(f"Archive member escapes output root: {member.name!r}") from exc
                    if destination not in expected:
                        # Do not stage extra frames/cameras that are absent from
                        # the selected human-annotation manifest.
                        continue

                    if destination.is_file():
                        if destination.stat().st_size != member.size:
                            raise FileExistsError(
                                f"Existing image size differs from archive; inspect before replacing: {destination}"
                            )
                        reused += 1
                        continue

                    source = archive.extractfile(member)
                    if source is None:
                        issue = f"Could not read tar payload for {member.name}"
                        break
                    temp_path = temp_root / f"{member_count:012d}.partial"
                    written = 0
                    try:
                        with source, temp_path.open("wb") as output:
                            while True:
                                block = source.read(COPY_BUFFER)
                                if not block:
                                    break
                                output.write(block)
                                written += len(block)
                        if written != member.size:
                            issue = (
                                f"Truncated image payload for {member.name}: "
                                f"read {written:,}/{member.size:,} bytes"
                            )
                            temp_path.unlink(missing_ok=True)
                            break
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temp_path.replace(destination)
                    except Exception:
                        temp_path.unlink(missing_ok=True)
                        raise
                    copied += 1
                    copied_bytes += written
                    if copied % 1000 == 0:
                        print(
                            f"Staged {copied:,} images ({copied_bytes / 2**30:.2f} GiB); "
                            f"tar members scanned={member_count:,}",
                            flush=True,
                        )
    except (tarfile.TarError, EOFError) as exc:
        issue = f"{type(exc).__name__}: {exc}"
    except OSError as exc:
        fatal_issue = f"{type(exc).__name__}: {exc}"

    present = {path for path in expected if path.is_file()}
    missing = sorted(expected - present)
    print(
        f"Staging summary: copied={copied:,}; reused={reused:,}; "
        f"present={len(present):,}/{len(expected):,}; "
        f"new_bytes={copied_bytes:,}; tar_members={member_count:,}"
    )
    if fatal_issue:
        print(f"Staging failed: {fatal_issue}")
        raise SystemExit(2)
    if issue:
        print(f"Archive ended before a clean scan: {issue}")
    if missing:
        print(f"Capture is incomplete: {len(missing):,} manifest images are missing.")
        for path in missing[:10]:
            print(f"  missing: {path}")
        print("Pass the next consecutive archive part(s) and rerun; existing complete files are retained.")
        raise SystemExit(2)

    print("Capture staging verified against every image path in the manifest.")
    try:
        temp_root.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
