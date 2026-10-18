"""InterHand2.6M Verification, Integrity Inspection & Inventory Engine."""

import sys
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_ROOT = ROOT / "datasets" / "interhand2.6m"
MANIFEST_DIR = DATASET_ROOT / "manifests"

def inspect_dataset():
    print("=================================================================")
    print(" INTERHAND2.6M DATASET INSPECTION & VERIFICATION")
    print("=================================================================")

    manifest_file = MANIFEST_DIR / "download_manifest.json"
    if not manifest_file.exists():
        print("[ERROR] download_manifest.json not found.")
        return

    with open(manifest_file, "r", encoding="utf-8-sig") as f:
        manifest = json.load(f)

    annot_dir = DATASET_ROOT / "annotations"
    images_dir = DATASET_ROOT / "images"
    raw_dir = DATASET_ROOT / "raw"

    found_annotations = list(annot_dir.glob("*.json"))
    raw_archives = list(raw_dir.glob("*.tar.gz")) + list(raw_dir.glob("*.zip"))
    image_files = list(images_dir.rglob("*.jpg"))

    def get_size_mb(p: Path) -> float:
        if not p.exists(): return 0.0
        if p.is_file(): return p.stat().st_size / (1024 * 1024)
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)

    inventory = {
        "dataset_name": manifest["dataset_name"],
        "status": "Ready for download" if len(found_annotations) == 0 else "Partially / Fully Available",
        "raw_archives_count": len(raw_archives),
        "raw_archives_size_mb": round(get_size_mb(raw_dir), 2),
        "annotations_present": [f.name for f in found_annotations],
        "annotations_size_mb": round(get_size_mb(annot_dir), 2),
        "extracted_images_count": len(image_files),
        "extracted_images_size_mb": round(get_size_mb(images_dir), 2),
        "splits_available": {},
        "missing_required_files": []
    }

    print(f"Raw Archives Located : {len(raw_archives)} files ({inventory['raw_archives_size_mb']:.1f} MB)")
    print(f"Annotations Located  : {len(found_annotations)} JSON files ({inventory['annotations_size_mb']:.1f} MB)")
    print(f"Images Located       : {len(image_files)} extracted images ({inventory['extracted_images_size_mb']:.1f} MB)")

    for req_file in manifest["required_archives"]["annotations"].keys():
        fpath = annot_dir / req_file
        if not fpath.exists():
            inventory["missing_required_files"].append(req_file)

    for split in ["train", "val", "test"]:
        data_json = annot_dir / f"InterHand2.6M_{split}_data.json"
        joint_json = annot_dir / f"InterHand2.6M_{split}_joint_3d.json"
        mano_json = annot_dir / f"InterHand2.6M_{split}_MANO_NeuralAnnot.json"

        if data_json.exists():
            print(f"\n--- Inspecting {split.upper()} split ---")
            with open(data_json, "r", encoding="utf-8-sig") as f:
                d = json.load(f)
            n_imgs = len(d.get("images", []))
            n_annots = len(d.get("annotations", []))
            captures = list(set(img["capture"] for img in d.get("images", [])))
            seqs = list(set(img["seq_name"] for img in d.get("images", [])))
            cams = list(set(img["camera"] for img in d.get("images", [])))

            split_info = {
                "images_indexed": n_imgs,
                "annotations_indexed": n_annots,
                "distinct_captures": len(captures),
                "distinct_sequences": len(seqs),
                "distinct_cameras": len(cams),
                "has_joint_3d": joint_json.exists(),
                "has_mano": mano_json.exists()
            }
            inventory["splits_available"][split] = split_info
            print(f"  Indexed Images   : {n_imgs:,}")
            print(f"  Indexed Sequences: {len(seqs):,} across {len(captures)} captures")
            print(f"  Camera Views     : {len(cams)} synchronized viewports")
            print(f"  3D Joint Annot   : {'Available' if joint_json.exists() else 'Missing'}")
            print(f"  MANO Fits        : {'Available' if mano_json.exists() else 'Missing'}")

    inv_file = MANIFEST_DIR / "dataset_inventory.json"
    with open(inv_file, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    print("\n" + "=" * 65)
    print(f"Machine-readable inventory saved to: {inv_file.relative_to(ROOT)}")
    if len(inventory["missing_required_files"]) > 0:
        print(f"[STATUS] Awaiting manual download of {len(inventory['missing_required_files'])} annotation files.")
    else:
        print("[STATUS] All required annotation bundles present on disk.")

if __name__ == "__main__":
    inspect_dataset()
