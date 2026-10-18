"""Read-only Verification Engine for Re:InterHand Dataset."""

import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "datasets" / "reinterhand"
CAPTURES_DIR = DATASET_ROOT / "captures"

TARGET_CAPTURE = "m-20210701-1058-0000000-pilot-relightablehandsy-participant0-two-hands"

# Official Re:InterHand 24-keypoint ordering
EXPECTED_KEYPOINTS = [
    "Wrist", "Thumb_0", "Thumb_1", "Thumb_2", "Thumb_3", "Thumb_4",
    "Index_1", "Index_2", "Index_3", "Index_4",
    "Middle_1", "Middle_2", "Middle_3", "Middle_4",
    "Ring_1", "Ring_2", "Ring_3", "Ring_4",
    "Pinky_0", "Pinky_1", "Pinky_2", "Pinky_3", "Pinky_4",
    "Forearm_Stub"
]

def verify_reinterhand():
    print("=================================================================")
    print(" RE:INTERHAND CAPTURE 0 VERIFICATION")
    print("=================================================================")

    cap_dir = CAPTURES_DIR / TARGET_CAPTURE
    if not cap_dir.exists():
        # Check if unbundled under generic name
        candidates = list(CAPTURES_DIR.glob("*"))
        if len(candidates) == 0:
            print(f"[STATUS] Target capture not yet staged under: {cap_dir.relative_to(ROOT)}")
            print("Run download scripts in datasets/reinterhand/raw/scripts/ to populate assets.")
            return
        cap_dir = candidates[0]

    print(f"Inspecting Capture Directory: {cap_dir.name}")

    checks = {
        "CHECKSUM": (cap_dir / "CHECKSUM").exists(),
        "frame_list.txt": (cap_dir / "frame_list.txt").exists(),
        "frame_list_orig.txt": (cap_dir / "frame_list_orig.txt").exists(),
        "mano_fits/meshes": (cap_dir / "mano_fits" / "meshes").exists(),
        "mano_fits/params": (cap_dir / "mano_fits" / "params").exists(),
        "orig_fits/right/Keypoints": (cap_dir / "orig_fits" / "right" / "Keypoints").exists(),
        "orig_fits/right/Meshes": (cap_dir / "orig_fits" / "right" / "Meshes").exists(),
        "Mugsy_cameras/envmap_per_segment": (cap_dir / "Mugsy_cameras" / "envmap_per_segment").exists()
    }

    print("\nAsset Inventory Checklist:")
    for k, passed in checks.items():
        print(f"  [{'OK' if passed else 'MISSING'}] {k}")

    # Inspect Keypoint ordering if orig_fits is present
    kp_dir = cap_dir / "orig_fits" / "right" / "Keypoints"
    if kp_dir.exists():
        kp_files = list(kp_dir.glob("*.json"))
        if kp_files:
            with open(kp_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"\nKeypoint Verification ({kp_files[0].name}):")
            print(f"  Keypoints in file: {len(data)} (Expected 24)")
            print(f"  Root: {EXPECTED_KEYPOINTS[0]} | Forearm stub: {EXPECTED_KEYPOINTS[-1]}")

    print("=================================================================")

if __name__ == "__main__":
    verify_reinterhand()
