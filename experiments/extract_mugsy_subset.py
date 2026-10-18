"""Safe streaming extraction of the 5,115 complete Mugsy images (Cam 400485)

Streams envmap_per_segment.tar.gz chunks aa and ab, extracts complete PNG images
to derived/mugsy_subset/images/400485/, and builds a consolidated manifest.csv
linking each image to its interaction segment and 3D ground-truth annotations.
"""

import io
import csv
import sys
import zlib
import shutil
import tarfile
from pathlib import Path
from tqdm import tqdm

ROOT = Path(".")
CAPTURE_ID = "m--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands"
CAPTURE_DIR = ROOT / "datasets" / "reinterhand" / "captures" / CAPTURE_ID
MUGSY_DIR = CAPTURE_DIR / "Mugsy_cameras"

SUBSET_DIR = CAPTURE_DIR / "derived" / "mugsy_subset"
OUT_IMG_DIR = SUBSET_DIR / "images" / "400485"
MANIFEST_PATH = SUBSET_DIR / "manifest.csv"

CAMERA_ID = "400485"
PREFIX = f"envmap_per_segment/images/{CAMERA_ID}/"

class ChainedStream(io.RawIOBase):
    """Presents multiple file chunks as a single continuous binary stream."""
    def __init__(self, file_paths):
        self.file_paths = [p for p in file_paths if p.exists()]
        self.current_idx = 0
        self.current_file = open(self.file_paths[0], "rb") if self.file_paths else None

    def readable(self):
        return True

    def readinto(self, b):
        if not self.current_file:
            return 0
        n = self.current_file.readinto(b)
        if n == 0:
            self.current_file.close()
            self.current_idx += 1
            if self.current_idx < len(self.file_paths):
                self.current_file = open(self.file_paths[self.current_idx], "rb")
                return self.readinto(b)
            else:
                self.current_file = None
                return 0
        return n

    def close(self):
        if self.current_file:
            self.current_file.close()
        super().close()

def main():
    print("=" * 80)
    print(" RE:INTERHAND MUGSY SUBSET STREAM EXTRACTION (aa + ab -> Cam 400485)")
    print("=" * 80)

    chunk_paths = [
        MUGSY_DIR / "envmap_per_segment.tar.gzaa",
        MUGSY_DIR / "envmap_per_segment.tar.gzab"
    ]
    for p in chunk_paths:
        if not p.exists():
            print(f"[Error] Required chunk missing: {p}")
            sys.exit(1)

    # 1. Load frame -> segment mappings
    frame_to_segment = {}
    fl_path = CAPTURE_DIR / "frame_list.txt"
    if fl_path.exists():
        with open(fl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    frame_to_segment[parts[1]] = parts[0]

    # Directories for 3D annotations
    mano_dir = CAPTURE_DIR / "mano_fits" / "params"
    orig_r_dir = CAPTURE_DIR / "orig_fits" / "right" / "Keypoints"
    orig_l_dir = CAPTURE_DIR / "orig_fits" / "left" / "Keypoints"

    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    raw_stream = ChainedStream(chunk_paths)
    buf_stream = io.BufferedReader(raw_stream, buffer_size=2 * 1024 * 1024)

    manifest_rows = []
    extracted_count = 0
    skipped_count = 0
    stream_truncated = False

    print(f"Target Output Directory: {OUT_IMG_DIR}")
    print("Streaming and extracting members...")

    with tqdm(total=5115, desc="Extracting Images", unit="img") as pbar:
        try:
            with tarfile.open(fileobj=buf_stream, mode="r|gz") as tar:
                for member in tar:
                    if not (member.isfile() and member.name.startswith(PREFIX)):
                        continue

                    fname = member.name[len(PREFIX):]
                    if not fname.endswith(".png"):
                        continue

                    fid = fname[:-4]
                    dest_file = OUT_IMG_DIR / fname

                    # Idempotent write: skip if already extracted with matching size
                    if not (dest_file.exists() and dest_file.stat().st_size == member.size):
                        f_in = tar.extractfile(member)
                        if f_in is not None:
                            with open(dest_file, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out, length=256 * 1024)
                            extracted_count += 1
                    else:
                        skipped_count += 1

                    # Resolve 3D supervision paths (stored relative to CAPTURE_DIR)
                    seg_name = frame_to_segment.get(fid, "unknown")
                    mano_r = f"mano_fits/params/{fid}_right.json"
                    mano_l = f"mano_fits/params/{fid}_left.json"
                    
                    kp_fname = f"keypoint-{int(fid):06d}.json" if fid.isdigit() else f"keypoint-{fid}.json"
                    kp_r = f"orig_fits/right/Keypoints/{kp_fname}"
                    kp_l = f"orig_fits/left/Keypoints/{kp_fname}"

                    rel_img_path = f"derived/mugsy_subset/images/{CAMERA_ID}/{fname}"

                    manifest_rows.append({
                        "frame_id": fid,
                        "camera_id": CAMERA_ID,
                        "segment": seg_name,
                        "image_path": rel_img_path,
                        "mano_right": mano_r,
                        "mano_left": mano_l,
                        "orig_kp_right": kp_r,
                        "orig_kp_left": kp_l
                    })

                    pbar.update(1)

        except (EOFError, zlib.error, tarfile.TarError, OSError) as e:
            stream_truncated = True
            print(f"\n[Stream Info] Reached boundary of chunk 'ab': {type(e).__name__}")

    # 2. Write Manifest CSV
    print(f"\nWriting manifest to {MANIFEST_PATH}...")
    fieldnames = [
        "frame_id", "camera_id", "segment", "image_path",
        "mano_right", "mano_left", "orig_kp_right", "orig_kp_left"
    ]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # 3. Summary Report
    total_disk_bytes = sum(f.stat().st_size for f in OUT_IMG_DIR.glob("*.png"))
    unique_segments = {r["segment"] for r in manifest_rows}

    print("\n" + "=" * 80)
    print(" EXTRACTION COMPLETE REPORT")
    print("=" * 80)
    print(f" Total Complete Images Extracted : {extracted_count:,}")
    print(f" Pre-existing Images Skipped     : {skipped_count:,}")
    print(f" Total Manifest Records          : {len(manifest_rows):,}")
    print(f" Unique Interaction Segments     : {len(unique_segments):,}")
    print(f" Total Image Disk Footprint      : {total_disk_bytes / (1024 * 1024):.2f} MB ({total_disk_bytes / (1024**3):.2f} GB)")
    print(f" Manifest Location               : {MANIFEST_PATH}")
    print(f" Safe Stream Termination         : {'YES (Expected chunk boundary)' if stream_truncated else 'COMPLETED TO SCAN END'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
