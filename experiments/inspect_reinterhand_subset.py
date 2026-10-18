"""Inspects the local aa-ac Mugsy subset and maps them to 3D annotations."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAPTURE_ID = "m--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands"
CAPTURE_DIR = ROOT / "datasets" / "reinterhand" / "captures" / CAPTURE_ID

def inspect_subset():
    print("================================================================")
    print(" RE:INTERHAND LOCAL SUBSET INSPECTION (aa-ac)")
    print("================================================================")

    if not CAPTURE_DIR.exists():
        print(f"[Error] Capture directory not found at: {CAPTURE_DIR}")
        return

    # 1. Check frame lists
    frame_list_path = CAPTURE_DIR / "frame_list.txt"
    if frame_list_path.exists():
        with open(frame_list_path, "r") as f:
            segments = [line.strip() for line in f if line.strip()]
        print(f"Total segments in frame_list.txt: {len(segments):,}")
    else:
        print("[Warning] frame_list.txt missing.")

    # 2. Inspect Mugsy extracted images & cameras
    mugsy_img_dir = CAPTURE_DIR / "Mugsy_cameras" / "envmap_per_segment" / "images"
    if mugsy_img_dir.exists():
        cameras = [d.name for d in mugsy_img_dir.iterdir() if d.is_dir()]
        print(f"Available Mugsy Camera Viewports: {cameras}")
        
        total_images = 0
        cam_counts = {}
        for cam in cameras:
            cam_dir = mugsy_img_dir / cam
            imgs = list(cam_dir.glob("*.png"))
            cam_counts[cam] = len(imgs)
            total_images += len(imgs)
        print(f"Total extracted Mugsy images in local subset: {total_images:,}")
        for cam, count in cam_counts.items():
            print(f"  - {cam}: {count:,} frames")
    else:
        print("[Status] Mugsy envmap_per_segment images directory not yet extracted.")

    # 3. Inspect 3D supervision pairing (MANO & Original Fits)
    mano_params_dir = CAPTURE_DIR / "mano_fits" / "params"
    orig_kp_right = CAPTURE_DIR / "orig_fits" / "right" / "Keypoints"
    
    if mano_params_dir.exists():
        mano_files = list(mano_params_dir.glob("*.json"))
        print(f"Total MANO parameter files available: {len(mano_files):,}")
    else:
        print("[Status] mano_fits/params not found.")

    if orig_kp_right.exists():
        orig_files = list(orig_kp_right.glob("*.json"))
        print(f"Total Original Right Keypoint files available: {len(orig_files):,}")
    else:
        print("[Status] orig_fits/right/Keypoints not found.")

    print("================================================================")

if __name__ == "__main__":
    inspect_subset()