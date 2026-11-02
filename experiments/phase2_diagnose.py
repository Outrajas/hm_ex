"""Phase 2 diagnostics: axis errors, oracle depth, reprojection, and input ablation.

Single-file runner for Google Colab or the project checkout. It follows the
Phase 0 data/crop/split/training recipe and trains three fresh 20-epoch direct
models: grayscale only, Sobel only, and grayscale plus Sobel. It selects CUDA
automatically when available and otherwise runs on CPU. It does not download
data or write checkpoints; it prints diagnostic tables and stops.

The Phase 0 module forces CPU with CUDA_VISIBLE_DEVICES at import time, so this
script carries forward its loader/model/loss recipe locally while parameterizing
the first convolution for the requested one- and two-channel comparisons.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit(
        "phase2_diagnose.py requires PyTorch, NumPy, and OpenCV (cv2). "
        "Install them in this runtime, then rerun."
    ) from exc


# ------------------------------- Configuration ------------------------------

SEED = 20251018
TRAIN_SEGMENTS = 44
VAL_SEGMENTS = 12
TRAIN_EPOCHS = 20
BATCH_SIZE = 64
LR = 1.2e-3
WEIGHT_DECAY = 1e-4
IMAGE_W, IMAGE_H = 128, 96  # phase0 uses W=128, H=96 (96x128 pixels).
CAMERA_ID = "400485"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

KEEP_21_INDICES = [
    0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 19, 20, 21, 22,
]
BONE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGERTIP_JOINTS = [4, 8, 12, 16, 20]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
MODES = ("grayscale", "sobel", "both")


# ----------------------------- Dataset discovery ----------------------------

def _find_manifest() -> Path:
    # In a pasted Colab cell, __file__ is not defined. In that case search
    # Colab's usual dataset location and the current working directory first.
    script_file = globals().get("__file__")
    if script_file:
        here = Path(script_file).resolve()
        project_root = here.parents[1] if len(here.parents) > 1 else here.parent
    else:
        project_root = Path.cwd()
    capture_id = (
        "m--20210701--1058--0000000--pilot--relightablehandsy--"
        "participant0--two-hands"
    )
    likely = [
        project_root / "datasets" / "reinterhand" / "captures" / capture_id
        / "derived" / "mugsy_subset" / "manifest.csv",
        Path("/content/dataset/manifest.csv"),
        Path.cwd() / "manifest.csv",
    ]
    for path in likely:
        if path.is_file():
            return path.resolve()

    roots = [Path("/content/dataset"), Path("/content"), Path.cwd()]
    for root in roots:
        if not root.is_dir():
            continue
        found = list(root.rglob("manifest.csv"))
        if found:
            return next(
                (p.resolve() for p in found if "mugsy_subset" in str(p).lower()),
                found[0].resolve(),
            )

    # If a Drive-mounted dataset is in use, mount/search only after local paths.
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        drive_root = Path("/content/drive/MyDrive")
        found = list(drive_root.rglob("manifest.csv"))
        if found:
            return next(
                (p.resolve() for p in found if "mugsy_subset" in str(p).lower()),
                found[0].resolve(),
            )
    except Exception as exc:
        print(f"Drive search unavailable: {exc}")

    raise FileNotFoundError(
        "manifest.csv not found. Put the dataset under /content/dataset or "
        "mount it in Google Drive."
    )


MANIFEST_PATH = _find_manifest()
SUBSET_DIR = MANIFEST_PATH.parent


def _find_camera_and_capture_root() -> tuple[Path, Path]:
    ancestors = (SUBSET_DIR, *SUBSET_DIR.parents)
    for parent in ancestors:
        candidate = parent / "Mugsy_cameras" / "cam_params.json"
        if candidate.is_file():
            return candidate.resolve(), parent.resolve()
        candidate = parent / "cam_params.json"
        if candidate.is_file():
            return candidate.resolve(), parent.resolve()

    for root in (SUBSET_DIR.parent, SUBSET_DIR.parent.parent, Path("/content")):
        if root.is_dir():
            found = list(root.rglob("cam_params.json"))
            if found:
                cam = next(
                    (p for p in found if p.parent.name.lower() == "mugsy_cameras"),
                    found[0],
                )
                cap = cam.parent.parent if cam.parent.name.lower() == "mugsy_cameras" else cam.parent
                return cam.resolve(), cap.resolve()

    raise FileNotFoundError("cam_params.json not found near the manifest.")


CAMERA_PATH, CAPTURE_DIR = _find_camera_and_capture_root()


def _build_asset_index() -> dict[str, set[Path]]:
    index: dict[str, set[Path]] = {}
    roots = dict.fromkeys((CAPTURE_DIR, SUBSET_DIR, SUBSET_DIR.parent))
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".json", ".png", ".jpg", ".jpeg",
            }:
                index.setdefault(path.name.lower(), set()).add(path.resolve())
    return index


ASSET_INDEX = _build_asset_index()


def resolve_asset(rel_path: str) -> Path:
    rel = str(rel_path).replace("\\", "/").strip()
    rel_path_obj = Path(rel)
    parts = rel_path_obj.parts
    candidates = [
        Path(rel), CAPTURE_DIR / rel, SUBSET_DIR / rel,
        SUBSET_DIR.parent / rel, SUBSET_DIR.parent.parent / rel,
    ]
    if parts and parts[0] == CAPTURE_DIR.name:
        candidates.append(CAPTURE_DIR.joinpath(*parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    # Try the manifest path under each ancestor; original paths can be rooted
    # above the subset directory.
    for ancestor in (SUBSET_DIR, *SUBSET_DIR.parents):
        candidate = ancestor.joinpath(*parts)
        if candidate.is_file():
            return candidate.resolve()

    # The Colab export may flatten orig_fits/right/Keypoints to
    # keypoints_right/, and similarly for left.
    lower_parts = [part.lower() for part in parts]
    side = "right" if "right" in lower_parts else (
        "left" if "left" in lower_parts else None
    )
    hits = ASSET_INDEX.get(rel_path_obj.name.lower(), set())
    if side:
        side_dir = f"keypoints_{side}"
        side_hits = {
            path for path in hits
            if side_dir in [part.lower() for part in path.parts]
        }
        if len(side_hits) == 1:
            return next(iter(side_hits))
        if len(side_hits) > 1:
            exact = {p for p in side_hits if p.parent.name.lower() == side_dir}
            if len(exact) == 1:
                return next(iter(exact))

    if len(hits) == 1:
        return next(iter(hits))
    if len(hits) > 1:
        # For images or differently named exports, prefer a unique suffix match.
        wanted_suffix = "/".join(part.lower() for part in parts[-3:])
        suffix_hits = {
            p for p in hits
            if "/".join(p.as_posix().lower().split("/")[-3:]) == wanted_suffix
        }
        if len(suffix_hits) == 1:
            return next(iter(suffix_hits))
        raise FileNotFoundError(
            f"Asset basename is ambiguous: {rel}; matches: "
            f"{[str(p) for p in list(hits)[:6]]}"
        )
    raise FileNotFoundError(f"Could not resolve manifest asset: {rel}")


# --------------------------------- Geometry ----------------------------------

def crop_affine(uv: np.ndarray, margin: float = 1.4) -> np.ndarray:
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    center = (lo + hi) * 0.5
    side = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 16.0) * margin
    left, top = center[0] - side * 0.5, center[1] - side * 0.5
    return np.asarray(
        [
            [IMAGE_W / side, 0.0, -left * IMAGE_W / side],
            [0.0, IMAGE_H / side, -top * IMAGE_H / side],
        ],
        dtype=np.float32,
    )


def camera_project(points: np.ndarray, focal: np.ndarray, principal: np.ndarray) -> np.ndarray:
    z = points[..., 2]
    if np.any(z <= 1e-6):
        raise ValueError("Predicted point has nonpositive camera-space depth.")
    return np.stack(
        (focal[0] * points[..., 0] / z + principal[0],
         focal[1] * points[..., 1] / z + principal[1]),
        axis=-1,
    ).astype(np.float32)


def channels_from_gray(gray_u8: np.ndarray) -> dict[str, np.ndarray]:
    gray = gray_u8.astype(np.float32) / 255.0
    raw = gray_u8.astype(np.float32)
    gx = cv2.Sobel(raw, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 1020.0)
    gy = cv2.Sobel(raw, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 1020.0)
    sobel = np.clip(np.sqrt(gx * gx + gy * gy), 0.0, 1.0)
    return {
        "grayscale": gray[None].astype(np.float32),
        "sobel": sobel[None].astype(np.float32),
        "both": np.stack((gray, sobel)).astype(np.float32),
    }


def normalize_uv(uv: np.ndarray) -> np.ndarray:
    normalized = np.empty_like(uv, dtype=np.float32)
    normalized[:, 0] = 2.0 * uv[:, 0] / IMAGE_W - 1.0
    normalized[:, 1] = 2.0 * uv[:, 1] / IMAGE_H - 1.0
    return normalized


def rotate_camera_xy(y3d: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    out = y3d.copy()
    x, y = y3d[:, 0].copy(), y3d[:, 1].copy()
    out[:, 0] = c * x + s * y
    out[:, 1] = -s * x + c * y
    return out


def augment_one(gray_u8: np.ndarray, uv: np.ndarray, y3d: np.ndarray, rng):
    angle_deg = float(rng.uniform(-15.0, 15.0))
    angle_rad = np.deg2rad(angle_deg)
    scale = float(rng.uniform(0.90, 1.10))
    matrix = cv2.getRotationMatrix2D(
        (IMAGE_W * 0.5, IMAGE_H * 0.5), angle_deg, scale
    ).astype(np.float32)
    matrix[0, 2] += float(rng.uniform(-0.10, 0.10) * IMAGE_W)
    matrix[1, 2] += float(rng.uniform(-0.10, 0.10) * IMAGE_H)
    warped = cv2.warpAffine(
        gray_u8, matrix, (IMAGE_W, IMAGE_H), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    brightness = float(rng.uniform(0.80, 1.20))
    offset = float(rng.uniform(-16.0, 16.0))
    warped = np.clip(warped.astype(np.float32) * brightness + offset, 0, 255).astype(np.uint8)
    uv_aug = uv @ matrix[:, :2].T + matrix[:, 2]
    y3d_aug = rotate_camera_xy(y3d, angle_rad)
    return channels_from_gray(warped), y3d_aug, normalize_uv(uv_aug)


# -------------------------- Model copied from Phase 0 ------------------------

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[:, :3], d6[:, 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-7)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1, eps=1e-7)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def reconstruct_joints_from_bones(bone_vectors: torch.Tensor) -> torch.Tensor:
    batch = bone_vectors.shape[0]
    joints = torch.zeros((batch, 21, 3), dtype=bone_vectors.dtype, device=bone_vectors.device)
    for idx, (parent, child) in enumerate(BONE_PAIRS):
        joints[:, child] = joints[:, parent] + bone_vectors[:, idx]
    return joints


class WebcamAlignedLCFTracker(nn.Module):
    """Phase 0 LCF tracker; only its first convolution is channel-configurable."""
    def __init__(self, canonical_lens: torch.Tensor, in_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 8)),
        )
        self.shared_fc = nn.Sequential(
            nn.Linear(256 * 6 * 8, 512), nn.ReLU(inplace=True), nn.Dropout(0.1)
        )
        self.meta_head = nn.Linear(512, 5 * 3)
        self.latent_frame_head = nn.Linear(512, 6)
        self.phalanx_head = nn.Linear(512, 15 * 3)
        self.proj2d_head = nn.Linear(512, 21 * 2)
        self.register_buffer("bone_lens", canonical_lens.clone())

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        feat = self.shared_fc(torch.flatten(self.encoder(x), 1))
        meta_cam_u = F.normalize(
            self.meta_head(feat).reshape(batch, 5, 3), dim=-1, eps=1e-7
        )
        latent_frame = rotation_6d_to_matrix(self.latent_frame_head(feat))
        phalanx_local_u = F.normalize(
            self.phalanx_head(feat).reshape(batch, 15, 3), dim=-1, eps=1e-7
        )
        phalanx_cam_u = torch.bmm(
            latent_frame, phalanx_local_u.transpose(1, 2)
        ).transpose(1, 2)
        bone_dirs = torch.zeros((batch, 20, 3), dtype=x.dtype, device=x.device)
        bone_dirs[:, 0] = meta_cam_u[:, 0]
        bone_dirs[:, 1:4] = phalanx_cam_u[:, 0:3]
        for finger_idx, bone_offset in enumerate((4, 8, 12, 16)):
            phalanx_offset = 3 + finger_idx * 3
            bone_dirs[:, bone_offset] = meta_cam_u[:, finger_idx + 1]
            bone_dirs[:, bone_offset + 1:bone_offset + 4] = (
                phalanx_cam_u[:, phalanx_offset:phalanx_offset + 3]
            )
        bone_vectors = bone_dirs * self.bone_lens.view(1, 20, 1)
        joints = reconstruct_joints_from_bones(bone_vectors)
        proj2d = torch.tanh(self.proj2d_head(feat).reshape(batch, 21, 2))
        return joints, bone_dirs, proj2d


class MetricLossEvaluator:
    """Same Phase 0 position, bone-length, and bone-direction objective."""
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_pos = torch.norm(pred - target, dim=-1).mean()
        parent_ids = [p for p, _ in BONE_PAIRS]
        child_ids = [c for _, c in BONE_PAIRS]
        pred_bones = pred[:, child_ids] - pred[:, parent_ids]
        target_bones = target[:, child_ids] - target[:, parent_ids]
        pred_len = torch.norm(pred_bones, dim=-1)
        target_len = torch.norm(target_bones, dim=-1)
        l_len = F.l1_loss(pred_len, target_len)
        pred_u = pred_bones / (pred_len.unsqueeze(-1) + 1e-7)
        target_u = target_bones / (target_len.unsqueeze(-1) + 1e-7)
        cos_sim = torch.clamp((pred_u * target_u).sum(dim=-1), -1.0, 1.0)
        l_dir = (1.0 - cos_sim).mean()
        return l_pos + 0.5 * l_len + 20.0 * l_dir


# -------------------------------- Data loader --------------------------------

def load_data():
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("camera_id", CAMERA_ID) == CAMERA_ID]
    if not rows:
        raise RuntimeError(f"Manifest has no rows for camera {CAMERA_ID}.")

    with CAMERA_PATH.open("r", encoding="utf-8") as f:
        camera_root = json.load(f)
    camera = camera_root[CAMERA_ID]
    rotation = np.asarray(camera["R"], dtype=np.float32)
    translation = np.asarray(camera["t"], dtype=np.float32).reshape(3)
    focal = np.asarray(camera["focal"], dtype=np.float32).reshape(2)
    principal = np.asarray(camera["princpt"], dtype=np.float32).reshape(2)

    records = []
    image_size = None
    for index, row in enumerate(rows):
        kp_path = resolve_asset(row["orig_kp_right"])
        image_path = resolve_asset(row["image_path"])
        with kp_path.open("r", encoding="utf-8") as f:
            kp_world = np.asarray(json.load(f), dtype=np.float32)
        if kp_world.shape != (24, 3):
            raise ValueError(
                f"{row.get('frame_id', index)}: expected 24x3 right keypoints, "
                f"got {kp_world.shape}."
            )

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        h, w = image.shape
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            raise ValueError("Image dimensions vary within the manifest.")

        camera_points = kp_world @ rotation.T + translation
        selected = camera_points[KEEP_21_INDICES]
        z = selected[:, 2]
        if np.any(z <= 1e-6):
            raise ValueError(f"Invalid camera depth in {row.get('frame_id', index)}")
        uv = np.stack(
            (focal[0] * selected[:, 0] / z + principal[0],
             focal[1] * selected[:, 1] / z + principal[1]),
            axis=1,
        ).astype(np.float32)
        affine = crop_affine(uv, margin=1.4)
        crop = cv2.warpAffine(
            image, affine, (IMAGE_W, IMAGE_H), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        uv_crop = uv @ affine[:, :2].T + affine[:, 2]
        root_camera = selected[0].copy()
        root_relative = selected - root_camera[None, :]

        records.append({
            "frame_id": row.get("frame_id", str(index)),
            "segment": row["segment"],
            "crop_gray": crop,
            "uv_crop": uv_crop.astype(np.float32),
            "uv_image": uv,
            "root_camera": root_camera.astype(np.float32),
            "y3d": root_relative.astype(np.float32),
        })

    return records, image_size, focal, principal


# ------------------------------- Split/training ------------------------------

def build_split(records):
    segments = np.asarray(sorted({r["segment"] for r in records}), dtype=object)
    if len(segments) != TRAIN_SEGMENTS + VAL_SEGMENTS:
        raise ValueError(
            f"Expected 56 segments for the 44/12 split; found {len(segments)}."
        )
    split_rng = np.random.default_rng(SEED)
    split_rng.shuffle(segments)
    train_segments = set(segments[:TRAIN_SEGMENTS].tolist())
    val_segments = set(segments[TRAIN_SEGMENTS:].tolist())
    if train_segments & val_segments:
        raise AssertionError("Segment leakage in train/validation split.")
    train_idx = np.asarray(
        [i for i, r in enumerate(records) if r["segment"] in train_segments],
        dtype=np.int64,
    )
    val_idx = np.asarray(
        [i for i, r in enumerate(records) if r["segment"] in val_segments],
        dtype=np.int64,
    )
    return train_segments, val_segments, train_idx, val_idx


def make_base_features(records, mode: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack([
        channels_from_gray(record["crop_gray"])[mode] for record in records
    ])).to(device)


def augment_mode(gray_u8, uv, y3d, rng, mode):
    channels, y3d_aug, y2d_aug = augment_one(gray_u8, uv, y3d, rng)
    return channels[mode], y3d_aug, y2d_aug


def train_one_mode(records, train_idx, val_idx, y_all, y2d_all, mode, device):
    # Reset RNGs before each ablation so initialization, sample order, and
    # augmentation randomness are paired across the three input modes.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    x_all = make_base_features(records, mode, device)
    train_y = y_all[train_idx]
    canonical_lens = torch.from_numpy(np.stack([
        np.linalg.norm(
            train_y[:, child] - train_y[:, parent], axis=1
        ).mean()
        for parent, child in BONE_PAIRS
    ]).astype(np.float32))
    model = WebcamAlignedLCFTracker(
        canonical_lens, in_channels=x_all.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_EPOCHS, eta_min=1e-4
    )
    criterion = MetricLossEvaluator()
    aug_rng = np.random.default_rng(SEED + 1)
    started = time.perf_counter()

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        order = np.random.permutation(train_idx)
        for start in range(0, len(order), BATCH_SIZE):
            batch_ids = order[start:start + BATCH_SIZE]
            samples = [
                augment_mode(
                    records[int(i)]["crop_gray"],
                    records[int(i)]["uv_crop"],
                    y_all[int(i)],
                    aug_rng,
                    mode,
                )
                for i in batch_ids
            ]
            bx = torch.from_numpy(np.stack([s[0] for s in samples])).to(device)
            by3 = torch.from_numpy(np.stack([s[1] for s in samples])).to(device)
            by2 = torch.from_numpy(np.stack([s[2] for s in samples])).to(device)

            optimizer.zero_grad(set_to_none=True)
            pred3, _, pred2 = model(bx)
            loss = criterion(pred3, by3) + 15.0 * F.smooth_l1_loss(pred2, by2)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"  {mode}: epoch {epoch:02d}/{TRAIN_EPOCHS} complete")

    elapsed = time.perf_counter() - started
    model.eval()
    val_pred_batches = []
    with torch.inference_mode():
        for start in range(0, len(val_idx), 128):
            ids = val_idx[start:start + 128]
            pred, _, _ = model(x_all[ids])
            val_pred_batches.append(pred.cpu().numpy())
    predictions = np.concatenate(val_pred_batches, axis=0).astype(np.float32)
    return predictions, elapsed


# --------------------------------- Metrics ------------------------------------

def mean_joint_distance(pred, target):
    return np.linalg.norm(pred - target, axis=-1).mean(axis=0)


def component_mae(pred, target):
    return np.abs(pred - target).mean(axis=0)


def reprojection_metrics(pred_root_relative, target_root_relative, records,
                         val_idx, focal, principal):
    # Add the GT camera-space root to both prediction and label. This keeps the
    # root translation/depth known and evaluates the real calibrated projection.
    pred_camera = np.stack([
        pred + records[int(i)]["root_camera"][None, :]
        for i, pred in zip(val_idx, pred_root_relative)
    ])
    target_uv = np.stack([records[int(i)]["uv_image"] for i in val_idx])
    pred_uv = camera_project(pred_camera, focal, principal)
    pixel_error = np.linalg.norm(pred_uv - target_uv, axis=-1)
    return pixel_error.mean(axis=0)


def report_mode(mode, pred, records, val_idx, y_all, focal, principal):
    target = y_all[val_idx]
    xyz_mae = component_mae(pred, target)
    joint_mpjpe = mean_joint_distance(pred, target)
    reproj = reprojection_metrics(pred, target, records, val_idx, focal, principal)

    oracle_pred = pred.copy()
    oracle_pred[..., 2] = target[..., 2]
    oracle_mpjpe = mean_joint_distance(oracle_pred, target)

    # Component table: overall averages over all joints, plus each tip.
    print(f"\n[{mode.upper()}] AXIS COMPONENT MAE AND JOINT MPJPE (mm)")
    print(f"{'Point':<10}{'MPJPE':>10}{'|dx|':>10}{'|dy|':>10}{'|dz|':>10}")
    print("-" * 50)
    print(
        f"{'Overall':<10}{joint_mpjpe.mean():10.2f}"
        f"{xyz_mae[:, 0].mean():10.2f}{xyz_mae[:, 1].mean():10.2f}"
        f"{xyz_mae[:, 2].mean():10.2f}"
    )
    for name, j in zip(FINGER_NAMES, FINGERTIP_JOINTS):
        print(
            f"{name:<10}{joint_mpjpe[j]:10.2f}"
            f"{xyz_mae[j, 0]:10.2f}{xyz_mae[j, 1]:10.2f}"
            f"{xyz_mae[j, 2]:10.2f}"
        )

    print(f"\n[{mode.upper()}] ORACLE-DEPTH MPJPE (mm; replace predicted relative z with GT)")
    print(f"{'Point':<10}{'MPJPE':>10}")
    print("-" * 20)
    print(f"{'Overall':<10}{oracle_mpjpe.mean():10.2f}")
    for name, j in zip(FINGER_NAMES, FINGERTIP_JOINTS):
        print(f"{name:<10}{oracle_mpjpe[j]:10.2f}")

    print(f"\n[{mode.upper()}] REAL-CAMERA 2D REPROJECTION ERROR (pixels; GT root translation/depth)")
    print(f"{'Point':<10}{'Mean px':>10}")
    print("-" * 20)
    print(f"{'Overall':<10}{reproj.mean():10.2f}")
    for name, j in zip(FINGER_NAMES, FINGERTIP_JOINTS):
        print(f"{name:<10}{reproj[j]:10.2f}")

    segment_rows = []
    for segment in sorted({records[int(i)]["segment"] for i in val_idx}):
        local_positions = [
            pos for pos, i in enumerate(val_idx)
            if records[int(i)]["segment"] == segment
        ]
        segment_pred = pred[local_positions]
        segment_target = target[local_positions]
        segment_mpjpe = np.linalg.norm(
            segment_pred - segment_target, axis=-1
        ).mean()
        segment_rows.append((segment, len(local_positions), float(segment_mpjpe)))

    print(f"\n[{mode.upper()}] PER-SEGMENT MPJPE (mm; sorted by segment)")
    print(f"{'Segment':<58}{'Frames':>8}{'MPJPE':>10}")
    print("-" * 76)
    for segment, frames, value in segment_rows:
        print(f"{segment:<58}{frames:8d}{value:10.2f}")

    return float(joint_mpjpe.mean()), float(oracle_mpjpe.mean()), float(reproj.mean())


def main():
    wall_start = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    print(f"Active compute device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Camera:   {CAMERA_PATH} (camera {CAMERA_ID})")

    records, image_size, focal, principal = load_data()
    train_segments, val_segments, train_idx, val_idx = build_split(records)
    y_all = np.stack([r["y3d"] for r in records]).astype(np.float32)

    print(
        f"Loaded {len(records):,} frames ({image_size[0]}x{image_size[1]}), "
        f"split {len(train_segments)}/{len(val_segments)} segments, "
        f"{len(train_idx):,}/{len(val_idx):,} frames."
    )
    print("Training three paired 20-epoch Phase 0 recipe runs.")

    summary = []
    for mode in MODES:
        print(f"\nTraining input mode: {mode}")
        pred, elapsed = train_one_mode(
            records, train_idx, val_idx, y_all,
            np.stack([normalize_uv(r["uv_crop"]) for r in records]).astype(np.float32),
            mode, DEVICE,
        )
        metrics = report_mode(
            mode, pred, records, val_idx, y_all, focal, principal
        )
        summary.append((mode, *metrics, elapsed))

    print("\nINPUT ABLATION SUMMARY (validation)")
    print(f"{'Input':<12}{'MPJPE mm':>12}{'OracleZ mm':>12}{'2D px':>10}{'Train sec':>12}")
    print("-" * 58)
    for mode, mpjpe, oracle, reproj, elapsed in summary:
        print(f"{mode:<12}{mpjpe:12.2f}{oracle:12.2f}{reproj:10.2f}{elapsed:12.1f}")

    print("\nAssumptions:")
    print("- The phase0 KEEP_21_INDICES mapping selects the 21 right-hand joints.")
    print("- Camera-space labels, root-relative targets, and errors use millimeters.")
    print("- Axis components are mean absolute coordinate errors (|dx|, |dy|, |dz|); MPJPE is Euclidean 3D distance.")
    print("- Oracle depth replaces each predicted root-relative joint z with its GT root-relative z.")
    print("- Reprojection adds the GT camera-space root to predicted relative joints and uses the official camera intrinsics.")
    print("- No mirroring is applied; labels remain right-hand/chirality-consistent.")
    print(f"Total runtime: {time.perf_counter() - wall_start:.1f}s")


if __name__ == "__main__":
    main()
