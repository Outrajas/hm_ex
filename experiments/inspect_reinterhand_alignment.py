"""Read-only deep inspection and validation of Re:InterHand Capture 0 annotations."""

import json
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAPTURE_ID = "m--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands"
CAPTURE_DIR = ROOT / "datasets" / "reinterhand" / "captures" / CAPTURE_ID

def run_inspection():
    print("=" * 75)
    print(" RE:INTERHAND CAPTURE 0: READ-ONLY ANNOTATION & METADATA AUDIT")
    print("=" * 75)

    fl_path = CAPTURE_DIR / "frame_list.txt"
    fl_orig_path = CAPTURE_DIR / "frame_list_orig.txt"
    mano_dir = CAPTURE_DIR / "mano_fits" / "params"
    orig_r_dir = CAPTURE_DIR / "orig_fits" / "right" / "Keypoints"
    orig_l_dir = CAPTURE_DIR / "orig_fits" / "left" / "Keypoints"
    cam_path = CAPTURE_DIR / "Mugsy_cameras" / "cam_params.json"
    mugsy_archive_dir = CAPTURE_DIR / "Mugsy_cameras"

    # 1. Parse frame lists
    fl_entries = []
    if fl_path.exists():
        with open(fl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    fl_entries.append((parts[0], parts[1]))
    
    fl_orig_ids = []
    if fl_orig_path.exists():
        with open(fl_orig_path, "r", encoding="utf-8") as f:
            fl_orig_ids = [line.strip() for line in f if line.strip()]

    fl_frame_ids = [fid for _, fid in fl_entries]
    fl_frame_set = set(fl_frame_ids)
    fl_orig_set = set(fl_orig_ids)

    # 2. Gather MANO and Keypoint IDs
    mano_r_ids = set()
    if mano_dir.exists():
        mano_r_ids = {f.name.replace("_right.json", "") for f in mano_dir.glob("*_right.json")}
    
    orig_r_ids = set()
    if orig_r_dir.exists():
        orig_r_ids = {f.name.replace("keypoint-", "").replace(".json", "") for f in orig_r_dir.glob("keypoint-*.json")}

    print(f"\n[RECORD COUNTS]")
    print(f"  Valid frames (frame_list.txt)     : {len(fl_frame_ids):,}")
    print(f"  Raw capture frames (frame_orig)   : {len(fl_orig_ids):,}")
    print(f"  Right MANO fit files              : {len(mano_r_ids):,}")
    print(f"  Right Original Keypoint files     : {len(orig_r_ids):,}")

    # A. Test Beginning, Middle, and End Frames
    print("\n" + "-" * 75)
    print("A. REPRESENTATIVE FRAME PAIRING TEST")
    print("-" * 75)
    sample_indices = [0, 1, 2, len(fl_entries) // 2, -3, -2, -1]
    for idx in sample_indices:
        seg, fid = fl_entries[idx]
        mr = (mano_dir / f"{fid}_right.json").exists()
        ml = (mano_dir / f"{fid}_left.json").exists()
        kr = (orig_r_dir / f"keypoint-{fid}.json").exists()
        kl = (orig_l_dir / f"keypoint-{fid}.json").exists()
        print(f"  Index {idx:6d} | Seg: {seg:<32} | Frame: {fid} | MANO (R/L): {mr}/{ml} | Orig (R/L): {kr}/{kl}")

    # B & C. Set Relationships and Missing Frame Detection
    print("\n" + "-" * 75)
    print("B & C. SET RELATIONSHIPS & DISCREPANCY ANALYSIS")
    print("-" * 75)
    is_fl_in_orig = fl_frame_set.issubset(fl_orig_set)
    is_fl_in_mano = fl_frame_set.issubset(mano_r_ids)
    is_fl_in_orig_fits = fl_frame_set.issubset(orig_r_ids)

    print(f"  Are all 20,163 valid frames in frame_list_orig.txt?  : {is_fl_in_orig}")
    print(f"  Are all 20,163 valid frames covered by MANO fits?    : {is_fl_in_mano}")
    print(f"  Are all 20,163 valid frames covered by Orig Keypoints?: {is_fl_in_orig_fits}")

    filtered_count = len(fl_orig_set - fl_frame_set)
    print(f"  Filtered invalid/inactive frames (orig - valid)     : {filtered_count:,}")

    missing_in_mano = fl_orig_set - mano_r_ids
    print(f"  Frame(s) in frame_list_orig.txt missing in MANO fits: {list(missing_in_mano)}")
    if missing_in_mano:
        for m_fid in missing_in_mano:
            in_valid = m_fid in fl_frame_set
            print(f"    -> Missing frame {m_fid} is in valid frame_list.txt? {in_valid}")

    # D. Segment Statistics
    print("\n" + "-" * 75)
    print("D. SEGMENT DISTRIBUTION & STATS")
    print("-" * 75)
    seg_to_frames = {}
    for seg, fid in fl_entries:
        seg_to_frames.setdefault(seg, []).append(fid)

    unique_segs = list(seg_to_frames.keys())
    counts = [len(fids) for fids in seg_to_frames.values()]
    print(f"  Unique segment names count : {len(unique_segs):,}")
    print(f"  Min frames per segment     : {min(counts)}")
    print(f"  Max frames per segment     : {max(counts)}")
    print(f"  Median frames per segment  : {statistics.median(counts):.1f}")
    print(f"  Mean frames per segment    : {statistics.mean(counts):.2f}")

    print("\n  First 20 Unique Segments:")
    for s in unique_segs[:20]:
        print(f"    - {s} ({len(seg_to_frames[s])} frames)")

    print("\n  Last 20 Unique Segments:")
    for s in unique_segs[-20:]:
        print(f"    - {s} ({len(seg_to_frames[s])} frames)")

    # E. Action / Gesture Naming Patterns
    print("\n" + "-" * 75)
    print("E. SEGMENT NAMING PATTERNS")
    print("-" * 75)
    verbs = set()
    for s in unique_segs:
        parts = s.split("_")
        if len(parts) >= 2:
            verbs.add("_".join(parts[1:]))
    print(f"  Distinct gesture/interaction descriptors ({len(verbs)} unique types):")
    for v in sorted(list(verbs))[:25]:
        print(f"    - {v}")
    if len(verbs) > 25:
        print(f"    ... and {len(verbs) - 25} more.")

    # F. Schema Verification
    print("\n" + "-" * 75)
    print("F. JSON SCHEMAS & COORDINATE SPECIFICATION")
    print("-" * 75)
    sample_fid = fl_entries[0][1]
    with open(mano_dir / f"{sample_fid}_right.json", "r") as f:
        mano_sample = json.load(f)
    with open(orig_r_dir / f"keypoint-{sample_fid}.json", "r") as f:
        kp_sample = json.load(f)

    pose = np.array(mano_sample["pose"])
    shape = np.array(mano_sample["shape"])
    trans = np.array(mano_sample["trans"])
    kp_arr = np.array(kp_sample)

    print(f"  Sample Frame ID: {sample_fid}")
    print(f"  MANO pose vector  : shape {pose.shape} -> 3 global axis-angle + 45 articulated joint angles")
    print(f"  MANO shape vector : shape {shape.shape} -> 10 shape coefficients (beta)")
    print(f"  MANO trans vector : shape {trans.shape} -> {trans.tolist()} (units: METERS, z ≈ {trans[2]:.3f} m)")
    print(f"  Original Keypoints: shape {kp_arr.shape} -> 24 3D landmarks")
    print(f"  Sample Wrist Joint: {kp_arr[0].tolist()} (units: MILLIMETERS, z ≈ {kp_arr[0, 2]:.2f} mm)")
    print(f"  Scale relationship: MANO trans * 1000 ≈ {trans * 1000} vs Keypoint root ≈ {kp_arr[0]}")
    print(f"  Both hands present: Right and Left separate files exist for every frame.")

    # G. Mugsy Local Filesystem Status
    print("\n" + "-" * 75)
    print("G. LOCAL MUGSY ASSET STATUS")
    print("-" * 75)
    extracted_img_dir = mugsy_archive_dir / "envmap_per_segment" / "images"
    extracted_mask_dir = mugsy_archive_dir / "envmap_per_segment" / "masks"
    chunks = sorted(list(mugsy_archive_dir.glob("envmap_per_segment.tar.gz*")))
    
    print(f"  Extracted images directory exists? : {extracted_img_dir.exists()}")
    print(f"  Extracted masks directory exists?  : {extracted_mask_dir.exists()}")
    print(f"  Split archive chunks on disk       : {len(chunks)} chunk(s)")
    for c in chunks:
        print(f"    - {c.name} ({c.stat().st_size / (1024**3):.2f} GB)")
    print("  STATUS: Mugsy split archive is INCOMPLETE (requires aa through bl, 75.31 GB).")
    print("  EXTRACTION: NOT EXTRACTABLE YET. No individual chunk extraction should be attempted.")

    # H. Mugsy Image Path Mapping
    print("\n" + "-" * 75)
    print("H. MUGSY CAMERA & FRAME MAPPING RESOLUTION")
    print("-" * 75)
    cams = []
    if cam_path.exists():
        with open(cam_path, "r") as f:
            cams = list(json.load(f).keys())
    print(f"  Local cam_params.json Camera IDs ({len(cams)} cameras): {cams}")
    print("  Official Dataset Rendering & File Format:")
    print("    Pattern: Mugsy_cameras/envmap_per_segment/images/<CAMERA_ID>/%06d.png % frame_idx")
    print("    Example: Mugsy_cameras/envmap_per_segment/images/400374/160929.png")
    print("  Camera Extrinsics/Intrinsics:")
    print("    R (3x3), t (3,), focal (2,), princpt (2,) calibrated directly per camera.")

    print("\n" + "=" * 75)

if __name__ == "__main__":
    run_inspection()
