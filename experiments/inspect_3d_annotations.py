"""Comprehensive inspection and schema alignment for Re:InterHand 3D annotations."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAPTURE_ID = "m--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands"
CAPTURE_DIR = ROOT / "datasets" / "reinterhand" / "captures" / CAPTURE_ID

def inspect():
    print("=" * 70)
    print(" RE:INTERHAND 3D ANNOTATIONS & METADATA DEEP INSPECTION")
    print("=" * 70)

    # 1. Inspect frame_list.txt vs frame_list_orig.txt
    fl_path = CAPTURE_DIR / "frame_list.txt"
    fl_orig_path = CAPTURE_DIR / "frame_list_orig.txt"

    print("\n--- 1. FRAME LIST HEADS & FORMAT ---")
    fl_lines = []
    if fl_path.exists():
        with open(fl_path, "r", encoding="utf-8") as f:
            fl_lines = [line.strip() for line in f if line.strip()]
        print(f"frame_list.txt (Valid Segments): {len(fl_lines):,} entries")
        print("First 5 lines of frame_list.txt:")
        for line in fl_lines[:5]:
            print(f"  {line}")

    if fl_orig_path.exists():
        with open(fl_orig_path, "r", encoding="utf-8") as f:
            fl_orig_lines = [line.strip() for line in f if line.strip()]
        print(f"\nframe_list_orig.txt (Raw Capture Frames): {len(fl_orig_lines):,} entries")
        print("First 5 lines of frame_list_orig.txt:")
        for line in fl_orig_lines[:5]:
            print(f"  {line}")

    # 2. Inspect MANO Parameters
    mano_params_dir = CAPTURE_DIR / "mano_fits" / "params"
    print("\n--- 2. MANO FITS (mano_fits/params) ---")
    if mano_params_dir.exists():
        r_files = list(mano_params_dir.glob("*_right.json"))
        l_files = list(mano_params_dir.glob("*_left.json"))
        print(f"Right-hand MANO files : {len(r_files):,}")
        print(f"Left-hand MANO files  : {len(l_files):,}")

        if r_files:
            sample_r = r_files[0]
            print(f"\nSample filename: {sample_r.name}")
            with open(sample_r, "r", encoding="utf-8") as f:
                r_data = json.load(f)
            print("Keys & Array Shapes in MANO param JSON:")
            for k, v in r_data.items():
                if isinstance(v, list):
                    arr = np.array(v)
                    print(f"  - {k}: shape {arr.shape}, dtype {arr.dtype} (sample: {v[:3]}...)")
                else:
                    print(f"  - {k}: {type(v).__name__} = {v}")

    # 3. Inspect Original 3D Keypoints
    orig_r_dir = CAPTURE_DIR / "orig_fits" / "right" / "Keypoints"
    orig_l_dir = CAPTURE_DIR / "orig_fits" / "left" / "Keypoints"
    print("\n--- 3. ORIGINAL KEYPOINTS (orig_fits/*/Keypoints) ---")
    if orig_r_dir.exists():
        orig_r_files = list(orig_r_dir.glob("*.json"))
        orig_l_files = list(orig_l_dir.glob("*.json")) if orig_l_dir.exists() else []
        print(f"Right-hand keypoint files: {len(orig_r_files):,}")
        print(f"Left-hand keypoint files : {len(orig_l_files):,}")

        if orig_r_files:
            sample_kp = orig_r_files[0]
            print(f"\nSample filename: {sample_kp.name}")
            with open(sample_kp, "r", encoding="utf-8") as f:
                kp_data = json.load(f)
            print("Structure of Keypoint JSON:")
            if isinstance(kp_data, dict):
                print(f"  Top-level keys: {list(kp_data.keys())}")
                for k, v in list(kp_data.items())[:5]:
                    arr = np.array(v)
                    print(f"  - '{k}': shape {arr.shape} -> {v}")
            elif isinstance(kp_data, list):
                arr = np.array(kp_data)
                print(f"  List of {len(kp_data)} elements. Shape: {arr.shape}")
                print(f"  Sample element 0: {kp_data[0]}")

    # 4. Inspect Camera Calibration
    cam_json = CAPTURE_DIR / "Mugsy_cameras" / "cam_params.json"
    print("\n--- 4. MUGSY CAMERA PARAMETERS (cam_params.json) ---")
    if cam_json.exists():
        with open(cam_json, "r", encoding="utf-8") as f:
            cam_data = json.load(f)
        print(f"Root keys / Camera IDs: {list(cam_data.keys())}")
        first_cam = list(cam_data.keys())[0]
        print(f"Fields for Camera '{first_cam}':")
        for k, v in cam_data[first_cam].items():
            arr = np.array(v)
            print(f"  - {k}: shape {arr.shape}")
    else:
        print("[Notice] cam_params.json not found in CAPTURE_DIR/Mugsy_cameras.")

    # 5. Test Pairing for First Valid Segment
    print("\n--- 5. MODALITY PAIRING CHECK ---")
    if fl_lines:
        first_entry = fl_lines[0]
        # Frame entry format check
        tokens = first_entry.split()
        print(f"Testing pairing for line: '{first_entry}' (tokens: {tokens})")

        # Try extract frame index
        frame_cand = None
        for tok in tokens:
            cleaned = "".join(c for c in tok if c.isdigit())
            if cleaned:
                frame_cand = int(cleaned)
                break

        if frame_cand is not None:
            print(f"Extracted candidate integer frame index: {frame_cand}")
            mano_r = mano_params_dir / f"{frame_cand}_right.json"
            mano_l = mano_params_dir / f"{frame_cand}_left.json"
            kp_r1 = orig_r_dir / f"keypoint-{frame_cand:06d}.json"
            kp_r2 = orig_r_dir / f"{frame_cand}.json"

            print(f"  MANO Right ({mano_r.name}): {'EXISTS' if mano_r.exists() else 'NOT FOUND'}")
            print(f"  MANO Left  ({mano_l.name}): {'EXISTS' if mano_l.exists() else 'NOT FOUND'}")
            print(f"  Keypoint Right (keypoint-%06d): {'EXISTS' if kp_r1.exists() else 'NOT FOUND'}")
            if not kp_r1.exists():
                print(f"  Keypoint Right ({kp_r2.name}): {'EXISTS' if kp_r2.exists() else 'NOT FOUND'}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    inspect()
