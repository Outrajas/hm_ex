"""Native InterHand2.6M Data Reader.

Exposes native InterHand annotations, camera calibrations, and MANO fits
without transforming coordinates or mapping into HandState.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np

@dataclass
class NativeInterHandFrame:
    """Raw InterHand2.6M sample with native coordinates and conventions."""
    image_id: int
    image_path: Path
    capture_id: int
    seq_name: str
    frame_idx: int
    camera_id: str
    hand_type: str                  # "right", "left", or "interacting"
    hand_type_valid: bool
    img_width: int
    img_height: int
    # Camera calibration (Native OpenCV convention, mm)
    focal: np.ndarray               # (2,) [fx, fy]
    princpt: np.ndarray             # (2,) [cx, cy]
    campos: np.ndarray              # (3,) mm in world coordinates
    camrot: np.ndarray              # (3, 3) rotation matrix
    # Native 3D joints (21 per hand, mm)
    world_joints_right: Optional[np.ndarray] = None   # (21, 3)
    world_joints_left: Optional[np.ndarray] = None    # (21, 3)
    joint_valid_right: Optional[np.ndarray] = None    # (21,)
    joint_valid_left: Optional[np.ndarray] = None     # (21,)
    # Native MANO parameters
    mano_pose_right: Optional[np.ndarray] = None      # (48,) Axis-angle
    mano_shape_right: Optional[np.ndarray] = None     # (10,) Beta
    mano_trans_right: Optional[np.ndarray] = None     # (3,) Translation (mm)
    mano_pose_left: Optional[np.ndarray] = None       # (48,) Axis-angle
    mano_shape_left: Optional[np.ndarray] = None      # (10,) Beta
    mano_trans_left: Optional[np.ndarray] = None      # (3,) Translation (mm)

class InterHandReader:
    """Read-only access engine for InterHand2.6M dataset."""
    def __init__(self, dataset_root: Path):
        self.root = Path(dataset_root)
        self.annot_dir = self.root / "annotations"
        self.image_dir = self.root / "images"

        self._data_cache: Dict[str, Any] = {}
        self._camera_cache: Dict[str, Any] = {}
        self._joint_cache: Dict[str, Any] = {}
        self._mano_cache: Dict[str, Any] = {}

    def load_split_annotations(self, split: str = "train", load_mano: bool = True):
        """Loads official JSON annotation files into memory for a given split."""
        data_path = self.annot_dir / f"InterHand2.6M_{split}_data.json"
        cam_path = self.annot_dir / f"InterHand2.6M_{split}_camera.json"
        joint_path = self.annot_dir / f"InterHand2.6M_{split}_joint_3d.json"
        mano_path = self.annot_dir / f"InterHand2.6M_{split}_MANO_NeuralAnnot.json"

        if not data_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {data_path}")

        print(f"[InterHandReader] Loading {split} metadata...")
        with open(data_path, "r", encoding="utf-8") as f:
            self._data_cache[split] = json.load(f)

        print(f"[InterHandReader] Loading {split} camera calibrations...")
        with open(cam_path, "r", encoding="utf-8") as f:
            self._camera_cache[split] = json.load(f)

        print(f"[InterHandReader] Loading {split} 3D joint annotations...")
        with open(joint_path, "r", encoding="utf-8") as f:
            self._joint_cache[split] = json.load(f)

        if load_mano and mano_path.exists():
            print(f"[InterHandReader] Loading {split} MANO parameters...")
            with open(mano_path, "r", encoding="utf-8") as f:
                self._mano_cache[split] = json.load(f)

    def get_sequence_catalog(self, split: str) -> List[Tuple[int, str]]:
        """Returns unique (capture_id, sequence_name) records present in the split."""
        if split not in self._data_cache:
            self.load_split_annotations(split, load_mano=False)
        images = self._data_cache[split].get("images", [])
        seqs = sorted(list(set((int(img["capture"]), str(img["seq_name"])) for img in images)))
        return seqs

    def get_frame(self, split: str, image_id: int) -> NativeInterHandFrame:
        """Constructs a NativeInterHandFrame with exact raw InterHand fields."""
        if split not in self._data_cache:
            self.load_split_annotations(split)

        img_meta = self._data_cache[split]["images"][image_id]
        annot_meta = self._data_cache[split]["annotations"][image_id]
        cam_meta = self._camera_cache[split][str(img_meta["capture"])]

        cam_id = str(img_meta["camera"])
        focal = np.array(cam_meta["focal"][cam_id], dtype=np.float64)
        princpt = np.array(cam_meta["princpt"][cam_id], dtype=np.float64)
        campos = np.array(cam_meta["campos"][cam_id], dtype=np.float64)
        camrot = np.array(cam_meta["camrot"][cam_id], dtype=np.float64)

        frame = NativeInterHandFrame(
            image_id=image_id,
            image_path=self.image_dir / img_meta["file_name"],
            capture_id=int(img_meta["capture"]),
            seq_name=str(img_meta["seq_name"]),
            frame_idx=int(img_meta["frame_idx"]),
            camera_id=cam_id,
            hand_type=str(annot_meta["hand_type"]),
            hand_type_valid=bool(annot_meta["hand_type_valid"]),
            img_width=int(img_meta["width"]),
            img_height=int(img_meta["height"]),
            focal=focal,
            princpt=princpt,
            campos=campos,
            camrot=camrot
        )

        # 3D Joints
        str_id = str(image_id)
        if split in self._joint_cache and str_id in self._joint_cache[split]:
            j_entry = self._joint_cache[split][str_id]
            w_coord = np.array(j_entry["world_coord"], dtype=np.float64).reshape(-1, 3)
            val = np.array(j_entry["joint_valid"], dtype=bool)
            # InterHand stores 21 right-hand joints followed by 21 left-hand joints
            frame.world_joints_right = w_coord[:21]
            frame.joint_valid_right = val[:21]
            if len(w_coord) >= 42:
                frame.world_joints_left = w_coord[21:42]
                frame.joint_valid_left = val[21:42]

        # MANO parameters
        if split in self._mano_cache and str_id in self._mano_cache[split]:
            m_entry = self._mano_cache[split][str_id]
            if "right" in m_entry and m_entry["right"] is not None:
                frame.mano_pose_right = np.array(m_entry["right"]["pose"], dtype=np.float64)
                frame.mano_shape_right = np.array(m_entry["right"]["shape"], dtype=np.float64)
                frame.mano_trans_right = np.array(m_entry["right"]["trans"], dtype=np.float64)
            if "left" in m_entry and m_entry["left"] is not None:
                frame.mano_pose_left = np.array(m_entry["left"]["pose"], dtype=np.float64)
                frame.mano_shape_left = np.array(m_entry["left"]["shape"], dtype=np.float64)
                frame.mano_trans_left = np.array(m_entry["left"]["trans"], dtype=np.float64)

        return frame
