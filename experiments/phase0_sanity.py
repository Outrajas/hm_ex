"""Phase 0 sanity run: camera labels, crops, split, baselines, and LCF fit.

CPU-only, fixed-seed, and intentionally self-contained. The tracker follows the
WebcamAlignedLCFTracker architecture in experiments_and__tests.ipynb; its input
stem accepts the requested grayscale + Sobel channels.
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

# Do not allow CUDA initialization or use, even on machines with a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit(
        "phase0_sanity.py requires an existing Python environment with "
        "PyTorch, NumPy, and OpenCV (cv2). No dependencies were installed."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_ID = (
    "m--20210701--1058--0000000--pilot--relightablehandsy--"
    "participant0--two-hands"
)
CAPTURE_DIR = ROOT / "datasets" / "reinterhand" / "captures" / CAPTURE_ID
SUBSET_DIR = CAPTURE_DIR / "derived" / "mugsy_subset"
MANIFEST_PATH = SUBSET_DIR / "manifest.csv"
CAMERA_PATH = CAPTURE_DIR / "Mugsy_cameras" / "cam_params.json"
OVERLAY_PATH = Path(__file__).resolve().parent / "phase0_sanity_overlay.png"

# Exact mapping from the phase-training notebook in the workspace parent.
KEEP_21_INDICES = [
    0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22
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

IMAGE_W, IMAGE_H = 128, 96  # User's 96x128 means HxW, matching phase training.
SEED = 20251018
TRAIN_SEGMENTS = 44
VAL_SEGMENTS = 12
OVERFIT_SAMPLES = 64
OVERFIT_MAX_EPOCHS = 250
TRAIN_EPOCHS = 20
BATCH_SIZE = 64
LR = 1.2e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cpu")


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[:, :3], d6[:, 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-7)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1, eps=1e-7)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def reconstruct_joints_from_bones(bone_vectors: torch.Tensor) -> torch.Tensor:
    batch = bone_vectors.shape[0]
    joints = torch.zeros(
        (batch, 21, 3), dtype=bone_vectors.dtype, device=bone_vectors.device
    )
    for idx, (parent, child) in enumerate(BONE_PAIRS):
        joints[:, child] = joints[:, parent] + bone_vectors[:, idx]
    return joints


class WebcamAlignedLCFTracker(nn.Module):
    """Training-notebook LCF model, with its input stem widened to two channels."""

    def __init__(self, canonical_lens: torch.Tensor):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 8)),
        )
        self.shared_fc = nn.Sequential(
            nn.Linear(256 * 6 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
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

        bone_dirs = torch.zeros(
            (batch, 20, 3), dtype=x.dtype, device=x.device
        )
        bone_dirs[:, 0] = meta_cam_u[:, 0]
        bone_dirs[:, 1:4] = phalanx_cam_u[:, 0:3]
        finger_offsets = [4, 8, 12, 16]
        for finger_idx in range(4):
            bone_offset = finger_offsets[finger_idx]
            phalanx_offset = 3 + finger_idx * 3
            bone_dirs[:, bone_offset] = meta_cam_u[:, finger_idx + 1]
            bone_dirs[:, bone_offset + 1:bone_offset + 4] = (
                phalanx_cam_u[:, phalanx_offset:phalanx_offset + 3]
            )

        bone_vecs = bone_dirs * self.bone_lens.view(1, 20, 1)
        joints = reconstruct_joints_from_bones(bone_vecs)
        kp2d = torch.tanh(self.proj2d_head(feat).reshape(batch, 21, 2))
        return joints, bone_dirs, kp2d


class MetricLossEvaluator:
    """LCF metric loss copied from the phase-training notebook."""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_pos = torch.norm(pred - target, dim=-1).mean()
        pred_bones = pred[:, [c for _, c in BONE_PAIRS]] - pred[:, [p for p, _ in BONE_PAIRS]]
        target_bones = target[:, [c for _, c in BONE_PAIRS]] - target[:, [p for p, _ in BONE_PAIRS]]
        pred_len = torch.norm(pred_bones, dim=-1)
        target_len = torch.norm(target_bones, dim=-1)
        l_len = F.l1_loss(pred_len, target_len)
        eps = 1e-7
        pred_u = pred_bones / (pred_len.unsqueeze(-1) + eps)
        target_u = target_bones / (target_len.unsqueeze(-1) + eps)
        cos_sim = torch.clamp((pred_u * target_u).sum(dim=-1), -1.0, 1.0)
        l_dir = (1.0 - cos_sim).mean()
        return l_pos + 0.5 * l_len + 20.0 * l_dir


def read_inputs():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")
    if not CAMERA_PATH.is_file():
        raise FileNotFoundError(f"Missing camera parameters: {CAMERA_PATH}")

    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["camera_id"] == "400485"]
    if not rows:
        raise RuntimeError("Manifest has no rows for camera 400485.")

    with CAMERA_PATH.open("r", encoding="utf-8") as f:
        camera = json.load(f)["400485"]
    rotation = np.asarray(camera["R"], dtype=np.float64)
    translation = np.asarray(camera["t"], dtype=np.float64)
    fx, fy = map(float, camera["focal"])
    cx, cy = map(float, camera["princpt"])

    records = []
    overlay_indices = {
        0, len(rows) // 3, (2 * len(rows)) // 3, len(rows) - 1
    }
    counts = {
        "mm_inside": 0, "m_inside": 0, "total": 0,
        "mm_positive": 0, "m_positive": 0,
    }
    image_size = None

    for row_idx, row in enumerate(rows):
        with (CAPTURE_DIR / row["orig_kp_right"]).open("r", encoding="utf-8") as f:
            kp_world = np.asarray(json.load(f), dtype=np.float32)
        if kp_world.shape != (24, 3):
            raise ValueError(
                f"{row['frame_id']} right keypoints: expected (24,3), got {kp_world.shape}"
            )

        image_path = CAPTURE_DIR / row["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        h, w = image.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            raise ValueError(f"Image dimensions changed: {image_size} vs {(w, h)}")

        kp_cam_mm = kp_world @ rotation.T + translation.astype(np.float32)
        kp_cam_m_test = kp_world @ rotation.T + (translation * 1000.0).astype(np.float32)

        def project(points: np.ndarray) -> np.ndarray:
            z = points[:, 2]
            uv = np.full((len(points), 2), np.nan, dtype=np.float32)
            positive = z > 1e-6
            uv[positive, 0] = fx * points[positive, 0] / z[positive] + cx
            uv[positive, 1] = fy * points[positive, 1] / z[positive] + cy
            return uv

        uv_all = project(kp_cam_mm)
        uv_m_test = project(kp_cam_m_test)
        counts["total"] += 24
        counts["mm_positive"] += int(np.isfinite(uv_all).all(axis=1).sum())
        counts["m_positive"] += int(np.isfinite(uv_m_test).all(axis=1).sum())
        counts["mm_inside"] += int(
            ((uv_all[:, 0] >= 0) & (uv_all[:, 0] < w) &
             (uv_all[:, 1] >= 0) & (uv_all[:, 1] < h)).sum()
        )
        counts["m_inside"] += int(
            ((uv_m_test[:, 0] >= 0) & (uv_m_test[:, 0] < w) &
             (uv_m_test[:, 1] >= 0) & (uv_m_test[:, 1] < h)).sum()
        )

        selected_cam = kp_cam_mm[KEEP_21_INDICES]
        selected_uv = uv_all[KEEP_21_INDICES]
        if not np.isfinite(selected_uv).all():
            raise ValueError(f"Nonpositive depth in selected joints: {row['frame_id']}")
        root_relative = selected_cam - selected_cam[0:1]

        x_min, y_min = selected_uv.min(axis=0)
        x_max, y_max = selected_uv.max(axis=0)
        center_x, center_y = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
        side = max(float(x_max - x_min), float(y_max - y_min), 16.0) * 1.4
        left, top = center_x - side * 0.5, center_y - side * 0.5
        affine = np.asarray(
            [[IMAGE_W / side, 0.0, -left * IMAGE_W / side],
             [0.0, IMAGE_H / side, -top * IMAGE_H / side]],
            dtype=np.float32,
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        crop = cv2.warpAffine(
            gray, affine, (IMAGE_W, IMAGE_H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        uv_crop = selected_uv @ affine[:, :2].T + affine[:, 2]
        records.append({
            "frame_id": row["frame_id"],
            "segment": row["segment"],
            "image": image.copy() if row_idx in overlay_indices else None,
            "uv_image": selected_uv.astype(np.float32),
            "crop_gray": crop,
            "uv_crop": uv_crop.astype(np.float32),
            "y3d": root_relative.astype(np.float32),
        })

    coverage_mm = counts["mm_inside"] / counts["total"]
    coverage_m = counts["m_inside"] / counts["total"]
    assert coverage_mm > 0.95, (
        f"Camera t in mm projects only {coverage_mm:.2%} of points inside the image."
    )
    assert coverage_mm > coverage_m, (
        "The t-unit audit did not favor millimeters; inspect calibration units."
    )
    print(
        f"Projection audit: t=mm {counts['mm_inside']}/{counts['total']} "
        f"({coverage_mm:.2%}) in-frame; t interpreted as m "
        f"{counts['m_inside']}/{counts['total']} ({coverage_m:.2%})."
    )
    return records, image_size


def make_channels(gray_u8: np.ndarray) -> np.ndarray:
    gray = gray_u8.astype(np.float32) / 255.0
    gray_255 = gray_u8.astype(np.float32)
    gx = cv2.Sobel(gray_255, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 1020.0)
    gy = cv2.Sobel(gray_255, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 1020.0)
    mag = np.sqrt(gx * gx + gy * gy)
    return np.stack([gray, np.clip(mag, 0.0, 1.0)], axis=0).astype(np.float32)


def save_overlay(records: list[dict]) -> None:
    chosen = [0, len(records) // 3, (2 * len(records)) // 3, len(records) - 1]
    tiles = []
    for idx in chosen:
        rec = records[idx]
        canvas = rec["image"].copy()
        points = rec["uv_image"]
        for parent, child in BONE_PAIRS:
            p = tuple(np.rint(points[parent]).astype(int))
            c = tuple(np.rint(points[child]).astype(int))
            cv2.line(canvas, p, c, (70, 220, 40), 2, cv2.LINE_AA)
        for j, (u, v) in enumerate(points):
            cv2.circle(canvas, (int(round(u)), int(round(v))), 4, (0, 80, 255), -1)
            cv2.putText(
                canvas, str(j), (int(round(u)) + 4, int(round(v)) - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
            )
        cv2.putText(
            canvas, f"{rec['frame_id']} | {rec['segment'][:40]}",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        tiles.append(canvas)

    h, w = tiles[0].shape[:2]
    overlay = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        y0, x0 = (i // 2) * h, (i % 2) * w
        overlay[y0:y0 + h, x0:x0 + w] = tile
    if not cv2.imwrite(str(OVERLAY_PATH), overlay):
        raise RuntimeError(f"Could not write overlay image: {OVERLAY_PATH}")
    print(f"Four-frame projection overlay: {OVERLAY_PATH}")


def rotate_camera_xy(y3d: np.ndarray, angle_rad: float) -> np.ndarray:
    # OpenCV's positive angle convention in image pixels: [cos,sin;-sin,cos].
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    out = y3d.copy()
    x, y = y3d[:, 0].copy(), y3d[:, 1].copy()
    out[:, 0] = c * x + s * y
    out[:, 1] = -s * x + c * y
    return out


def augment_one(gray_u8: np.ndarray, uv: np.ndarray, y3d: np.ndarray, rng):
    angle_deg = float(rng.uniform(-15.0, 15.0))
    angle = np.deg2rad(angle_deg)
    scale = float(rng.uniform(0.90, 1.10))
    shift_x = float(rng.uniform(-0.10, 0.10) * IMAGE_W)
    shift_y = float(rng.uniform(-0.10, 0.10) * IMAGE_H)
    center = (IMAGE_W * 0.5, IMAGE_H * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, scale).astype(np.float32)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y

    warped = cv2.warpAffine(
        gray_u8, matrix, (IMAGE_W, IMAGE_H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    brightness = float(rng.uniform(0.80, 1.20))
    offset = float(rng.uniform(-16.0, 16.0))
    warped = np.clip(warped.astype(np.float32) * brightness + offset, 0, 255).astype(np.uint8)
    uv_aug = uv @ matrix[:, :2].T + matrix[:, 2]
    y3d_aug = rotate_camera_xy(y3d, angle)
    y2d_aug = np.empty_like(uv_aug, dtype=np.float32)
    y2d_aug[:, 0] = 2.0 * uv_aug[:, 0] / IMAGE_W - 1.0
    y2d_aug[:, 1] = 2.0 * uv_aug[:, 1] / IMAGE_H - 1.0
    return make_channels(warped), y3d_aug, y2d_aug


def normalize_crop_points(uv: np.ndarray) -> np.ndarray:
    out = np.empty_like(uv, dtype=np.float32)
    out[:, 0] = 2.0 * uv[:, 0] / IMAGE_W - 1.0
    out[:, 1] = 2.0 * uv[:, 1] / IMAGE_H - 1.0
    return out


def score(pred: np.ndarray, target: np.ndarray) -> tuple[float, list[float]]:
    distance = np.linalg.norm(pred - target, axis=-1)
    return float(distance.mean()), [float(x) for x in distance[:, FINGERTIP_JOINTS].mean(axis=0)]


def metric_line(
    method: str, split: str, mpjpe: float, tips: list[float], latency_ms: float | None = None
) -> str:
    tip_text = " ".join(f"{name}={value:7.2f}" for name, value in zip(FINGER_NAMES, tips))
    latency = "" if latency_ms is None else f"{latency_ms:9.3f}"
    return f"{method:<20} {split:<12} {mpjpe:9.2f}  {tip_text}  {latency}"


def predict_metrics(model: nn.Module, x: torch.Tensor, target: np.ndarray):
    preds = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            pred, _, _ = model(x[start:start + 128])
            preds.append(pred.cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    return score(pred_np, target)


def train_tracker(
    model: WebcamAlignedLCFTracker,
    train_records: list[dict],
    train_indices: np.ndarray,
    train_y3d: np.ndarray,
    train_y2d: np.ndarray,
    x_base: torch.Tensor,
    epochs: int,
    augment: bool,
    val_x: torch.Tensor | None = None,
    val_y3d: np.ndarray | None = None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-4
    )
    criterion = MetricLossEvaluator()
    rng = np.random.default_rng(SEED + (1 if augment else 2))
    val_history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(train_indices))
        for offset in range(0, len(order), BATCH_SIZE):
            local_batch = order[offset:offset + BATCH_SIZE]
            indices = train_indices[local_batch]
            if augment:
                batch = [
                    augment_one(
                        train_records[int(i)]["crop_gray"],
                        train_records[int(i)]["uv_crop"],
                        train_y3d[local_pos],
                        rng,
                    )
                    for local_pos, i in zip(local_batch, indices)
                ]
                bx = torch.from_numpy(np.stack([item[0] for item in batch])).to(DEVICE)
                by3d = torch.from_numpy(np.stack([item[1] for item in batch])).to(DEVICE)
                by2d = torch.from_numpy(np.stack([item[2] for item in batch])).to(DEVICE)
            else:
                bx = x_base[indices]
                by3d = torch.from_numpy(train_y3d[local_batch]).to(DEVICE)
                by2d = torch.from_numpy(train_y2d[local_batch]).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pred3d, _, pred2d = model(bx)
            loss = criterion(pred3d, by3d) + 15.0 * F.smooth_l1_loss(pred2d, by2d)
            loss.backward()
            optimizer.step()

        scheduler.step()
        if val_x is not None and val_y3d is not None and epoch % 2 == 0:
            mpjpe, tips = predict_metrics(model, val_x, val_y3d)
            val_history.append((epoch, mpjpe, tips))
        if not augment:
            train_mpjpe, _ = predict_metrics(
                model, x_base[train_indices], train_y3d
            )
            if train_mpjpe < 3.0:
                break

    return val_history, time.perf_counter() - started


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)

    records, image_size = read_inputs()
    save_overlay(records)

    segments = sorted({r["segment"] for r in records})
    if len(segments) != TRAIN_SEGMENTS + VAL_SEGMENTS:
        raise ValueError(
            f"Expected 56 segments (44/12 split), found {len(segments)}."
        )
    split_rng = np.random.default_rng(SEED)
    shuffled = np.asarray(segments, dtype=object)
    split_rng.shuffle(shuffled)
    train_segments = set(shuffled[:TRAIN_SEGMENTS].tolist())
    val_segments = set(shuffled[TRAIN_SEGMENTS:].tolist())
    assert len(train_segments) == 44 and len(val_segments) == 12
    assert not (train_segments & val_segments), "Segment leakage in split."

    train_indices = np.asarray(
        [i for i, r in enumerate(records) if r["segment"] in train_segments],
        dtype=np.int64,
    )
    val_indices = np.asarray(
        [i for i, r in enumerate(records) if r["segment"] in val_segments],
        dtype=np.int64,
    )
    if set(r["segment"] for r in (records[i] for i in train_indices)) & val_segments:
        raise AssertionError("Frame-level split leaked a held-out segment.")

    x_all_np = np.stack([make_channels(r["crop_gray"]) for r in records])
    x_all = torch.from_numpy(x_all_np).to(DEVICE)
    y_all = np.stack([r["y3d"] for r in records]).astype(np.float32)
    y2d_all = np.stack([normalize_crop_points(r["uv_crop"]) for r in records]).astype(np.float32)
    train_y3d = y_all[train_indices]
    val_y3d = y_all[val_indices]
    train_y2d = y2d_all[train_indices]
    val_x = x_all[val_indices]
    train_x = x_all[train_indices]

    # a) Mean-pose baseline.
    mean_pose = train_y3d.mean(axis=0, keepdims=True)
    mean_pred = np.repeat(mean_pose, len(val_indices), axis=0)
    mean_mpjpe, mean_tips = score(mean_pred, val_y3d)

    # b) Nearest-neighbor baseline using 12x16 two-channel crop descriptors.
    with torch.inference_mode():
        low_res = F.adaptive_avg_pool2d(x_all, (12, 16)).flatten(1)
        low_train = low_res[train_indices]
        low_val = low_res[val_indices]
        norm_train = (low_train * low_train).sum(dim=1)
        nn_indices = []
        for start in range(0, len(low_val), 64):
            chunk = low_val[start:start + 64]
            distances = (
                (chunk * chunk).sum(dim=1, keepdim=True)
                + norm_train.unsqueeze(0)
                - 2.0 * (chunk @ low_train.T)
            )
            nn_indices.extend(torch.argmin(distances, dim=1).tolist())
    nn_pred = train_y3d[np.asarray(nn_indices, dtype=np.int64)]
    nn_mpjpe, nn_tips = score(nn_pred, val_y3d)

    # c) Overfit diagnostic: same architecture, exactly 64 train frames, no augmentation.
    overfit_rng = np.random.default_rng(SEED + 64)
    overfit_local = overfit_rng.choice(
        len(train_indices), size=min(OVERFIT_SAMPLES, len(train_indices)), replace=False
    )
    overfit_indices = train_indices[overfit_local]
    canonical_lens = torch.from_numpy(np.stack([
        np.linalg.norm(
            train_y3d[:, child] - train_y3d[:, parent], axis=1
        ).mean()
        for parent, child in BONE_PAIRS
    ]).astype(np.float32))
    overfit_model = WebcamAlignedLCFTracker(canonical_lens).to(DEVICE)
    overfit_model.train()
    overfit_y = y_all[overfit_indices]
    overfit_x = x_all[overfit_indices]
    overfit_y2d = y2d_all[overfit_indices]
    # Use local 0..N positions so the shared trainer indexes corresponding targets.
    local_records = [records[i] for i in overfit_indices]
    _, overfit_seconds = train_tracker(
        overfit_model,
        local_records,
        np.arange(len(overfit_indices), dtype=np.int64),
        overfit_y,
        overfit_y2d,
        overfit_x,
        epochs=OVERFIT_MAX_EPOCHS,
        augment=False,
    )
    overfit_mpjpe, overfit_tips = predict_metrics(overfit_model, overfit_x, overfit_y)
    overfit_status = "PASS" if overfit_mpjpe < 3.0 else "PIPELINE BUG: >=3 mm"

    # d) Main 20-epoch run. Build canonical lengths from all 44 training segments.
    model = WebcamAlignedLCFTracker(canonical_lens).to(DEVICE)
    val_history, train_seconds = train_tracker(
        model,
        records,
        train_indices,
        train_y3d,
        train_y2d,
        x_all,
        epochs=TRAIN_EPOCHS,
        augment=True,
        val_x=val_x,
        val_y3d=val_y3d,
    )

    # e) CPU batch-1 forward latency, measured under inference_mode.
    model.eval()
    one = val_x[:1]
    with torch.inference_mode():
        for _ in range(20):
            model(one)
        timings = []
        for _ in range(200):
            t0 = time.perf_counter()
            model(one)
            timings.append((time.perf_counter() - t0) * 1000.0)
    latency_ms = float(statistics.median(timings))

    # One report table, including train-only overfit and all requested val checkpoints.
    header = (
        f"{'Method':<20} {'Split/Epoch':<12} {'MPJPE mm':>9}  "
        f"{'Thumb':>7} {'Index':>7} {'Middle':>7} {'Ring':>7} {'Pinky':>7}  "
        f"{'Latency ms':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    print(metric_line("Mean pose", "val", mean_mpjpe, mean_tips))
    print(metric_line("Nearest neighbor", "val", nn_mpjpe, nn_tips))
    print(metric_line(f"Overfit64 {overfit_status}", "train", overfit_mpjpe, overfit_tips))
    for epoch, mpjpe, tips in val_history:
        print(metric_line(
            "WebcamAlignedLCF", f"epoch {epoch:02d}", mpjpe, tips,
            latency_ms if epoch == TRAIN_EPOCHS else None,
        ))
    print(
        f"\nAssumptions: camera t is millimeters (audited); exact KEEP_21_INDICES "
        f"from the adjacent phase-training notebook; 96x128 is HxW (128x96 input); "
        f"right hand is used without mirroring, so right-hand chirality is preserved."
    )
    print(
        f"Split: {len(train_segments)} train / {len(val_segments)} val segments, "
        f"{len(train_indices)} / {len(val_indices)} frames, no segment overlap. "
        f"Input image size: {image_size[0]}x{image_size[1]}. "
        f"Overfit {overfit_seconds:.1f}s; main training {train_seconds:.1f}s."
    )
    print(f"Total wall time: {time.perf_counter() - WALL_START:.1f}s")


WALL_START = time.perf_counter()


if __name__ == "__main__":
    main()
