"""Read-only audit of CHECKSUM to determine Mugsy image scope and structure."""

from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
CAPTURE_ID = "m--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands"
CHECKSUM_FILE = ROOT / "datasets" / "reinterhand" / "captures" / CAPTURE_ID / "CHECKSUM"

def audit_manifest():
    print("=" * 70)
    print(" RE:INTERHAND CAPTURE 0: CHECKSUM MANIFEST AUDIT")
    print("=" * 70)

    if not CHECKSUM_FILE.exists():
        print(f"[Error] CHECKSUM file not found at: {CHECKSUM_FILE}")
        return

    category_counts = defaultdict(int)
    mugsy_cam_images = defaultdict(int)
    mugsy_cam_masks = defaultdict(int)
    other_modalities = set()

    with open(CHECKSUM_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            path_str = parts[0]

            # Track top-level asset categories
            top = path_str.split("/")[0]
            category_counts[top] += 1

            # Track Mugsy envmap_per_segment specifics
            if path_str.startswith("Mugsy_cameras/envmap_per_segment/"):
                subparts = path_str.split("/")
                # Pattern: Mugsy_cameras/envmap_per_segment/<images|masks>/<CAM_ID>/<FRAME>.png
                if len(subparts) >= 5:
                    modality_type = subparts[2]  # images or masks
                    cam_id = subparts[3]
                    if modality_type == "images":
                        mugsy_cam_images[cam_id] += 1
                    elif modality_type == "masks":
                        mugsy_cam_masks[cam_id] += 1
            elif not (path_str.startswith("mano_fits") or path_str.startswith("orig_fits")):
                other_modalities.add(top)

    print("\n--- 1. OVERALL ASSETS LISTED IN CHECKSUM ---")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:<25}: {count:,} files")

    print("\n--- 2. MUGSY ENVMAP_PER_SEGMENT CAMERAS ---")
    total_imgs = sum(mugsy_cam_images.values())
    total_masks = sum(mugsy_cam_masks.values())
    print(f"  Total Images Listed: {total_imgs:,}")
    print(f"  Total Masks Listed : {total_masks:,}")
    print(f"  Cameras rendered ({len(mugsy_cam_images)} active cameras out of 20 calibrated):")
    for cam in sorted(mugsy_cam_images.keys()):
        print(f"    - Camera {cam}: {mugsy_cam_images[cam]:,} images | {mugsy_cam_masks[cam]:,} masks")

    # Sample file path check
    with open(CHECKSUM_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if "envmap_per_segment/images" in line:
                print(f"\n  Sample Image Path : {line.split()[0]}")
                break

    print("\n" + "=" * 70)

if __name__ == "__main__":
    audit_manifest()
