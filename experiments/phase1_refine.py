"""Phase 1: prior-conditioned right-hand pose refinement, CPU-only."""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import phase0_sanity as p0


SEED = p0.SEED
EPOCHS = 25
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
FINGER_CHAINS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
]


class PoseRefiner(nn.Module):
    """Phase 0 encoder with a four-channel stem and a plain 63-value head."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2),
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
        self.delta_head = nn.Linear(512, 21 * 3)

    def forward(self, x):
        feat = self.shared_fc(torch.flatten(self.encoder(x), 1))
        return self.delta_head(feat).reshape(-1, 21, 3)


def camera_data(records):
    """Load only source gray frames and camera-space root positions."""
    with p0.MANIFEST_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["camera_id"] == "400485"]
    if len(rows) != len(records):
        raise ValueError("Manifest rows no longer match phase0 records.")
    with p0.CAMERA_PATH.open("r", encoding="utf-8") as f:
        cam = json.load(f)["400485"]
    rot = np.asarray(cam["R"], np.float32)
    trans = np.asarray(cam["t"], np.float32)
    focal = np.asarray(cam["focal"], np.float32)
    principal = np.asarray(cam["princpt"], np.float32)

    # Keep a high-resolution, 3.5x-margin image buffer around each ground-truth
    # box. It preserves room for prior-centered crops without retaining 3.5 GB
    # of full-resolution grayscale frames in RAM.
    gray_frames, buffer_affines = [], []
    roots = np.empty((len(rows), 3), np.float32)
    for i, row in enumerate(rows):
        path = p0.CAPTURE_DIR / row["image_path"]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(path)
        gt_uv = records[i]["uv_image"]
        height, width = gray.shape[:2]
        buffer_affine = crop_affine(gt_uv, width, height, margin=3.5,
                                    out_w=384, out_h=288)
        buffer = cv2.warpAffine(
            gray, buffer_affine, (384, 288), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        gray_frames.append(buffer)
        buffer_affines.append(buffer_affine)
        with (p0.CAPTURE_DIR / row["orig_kp_right"]).open("r", encoding="utf-8") as f:
            world = np.asarray(json.load(f), np.float32)
        cam_pts = world @ rot.T + trans
        roots[i] = cam_pts[p0.KEEP_21_INDICES[0]]
    return gray_frames, np.stack(buffer_affines), roots, focal, principal


def project(points_cam, focal, principal):
    z = points_cam[:, 2]
    if np.any(z <= 1.0):
        raise ValueError("Prior projected to a nonpositive/near-zero camera depth.")
    return np.stack(
        (focal[0] * points_cam[:, 0] / z + principal[0],
         focal[1] * points_cam[:, 1] / z + principal[1]), axis=1
    ).astype(np.float32)


def rotation_matrix(axis, angle):
    axis = np.asarray(axis, np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    c, s, d = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.asarray([
        [c + x*x*d, x*y*d - z*s, x*z*d + y*s],
        [y*x*d + z*s, c + y*y*d, y*z*d - x*s],
        [z*x*d - y*s, z*y*d + x*s, c + z*z*d],
    ], np.float32)


def synthesize_prior(gt, rng, severity=None):
    s = float(rng.uniform(0.2, 1.0) if severity is None else severity)
    prior = gt.copy()
    for chain in FINGER_CHAINS:
        for hinge_pos in range(len(chain) - 1):
            hinge = chain[hinge_pos]
            subtree = chain[hinge_pos + 1:]
            axis = rng.normal(size=3)
            angle = np.deg2rad(rng.uniform(-25.0, 25.0) * s)
            mat = rotation_matrix(axis, angle)
            prior[subtree] = (prior[subtree] - prior[hinge]) @ mat.T + prior[hinge]
    global_rot = rotation_matrix(rng.normal(size=3), np.deg2rad(rng.uniform(-10, 10)))
    prior = prior @ global_rot.T
    prior += rng.normal(0.0, 8.0, size=(1, 3)).astype(np.float32)
    return prior.astype(np.float32)


def crop_affine(uv, width, height, margin=1.4, out_w=None, out_h=None):
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    center = (lo + hi) * 0.5
    side = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 16.0) * margin
    left, top = center[0] - side * 0.5, center[1] - side * 0.5
    out_w = p0.IMAGE_W if out_w is None else out_w
    out_h = p0.IMAGE_H if out_h is None else out_h
    return np.asarray([
        [out_w / side, 0.0, -left * out_w / side],
        [0.0, out_h / side, -top * out_h / side],
    ], np.float32)


def channels_from_prior(gray, uv, affine):
    crop = cv2.warpAffine(
        gray, affine, (p0.IMAGE_W, p0.IMAGE_H), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    xy = uv @ affine[:, :2].T + affine[:, 2]
    lines = np.zeros((p0.IMAGE_H, p0.IMAGE_W), np.uint8)
    dots = np.zeros_like(lines)
    points = np.rint(xy).astype(int)
    for parent, child in p0.BONE_PAIRS:
        cv2.line(lines, tuple(points[parent]), tuple(points[child]), 255, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(dots, tuple(point), 3, 255, -1, cv2.LINE_AA)
    base = p0.make_channels(crop)
    return np.concatenate((base, lines[None].astype(np.float32) / 255.0,
                           dots[None].astype(np.float32) / 255.0), axis=0), xy


def aug_refine(gray, crop_uv, gt, prior, rng):
    # Call the phase0 augmentation implementation for the exact crop geometry,
    # brightness jitter and camera-XY label rotation. Recover that affine from
    # its transformed keypoints to apply the same geometry to rendered channels.
    base, gt_aug, uv_norm = p0.augment_one(gray, crop_uv, gt, rng)
    uv_aug = uv_norm.copy()
    uv_aug[:, 0] = (uv_aug[:, 0] + 1) * p0.IMAGE_W * 0.5
    uv_aug[:, 1] = (uv_aug[:, 1] + 1) * p0.IMAGE_H * 0.5
    design = np.column_stack((crop_uv, np.ones(len(crop_uv))))
    affine_t, *_ = np.linalg.lstsq(design, uv_aug, rcond=None)
    mat = affine_t.T.astype(np.float32)
    linear = mat[:, :2]
    angle = float(np.arctan2(linear[0, 1], linear[0, 0]))
    prior_aug = p0.rotate_camera_xy(prior, angle)
    return base, prior_aug.astype(np.float32), gt_aug, uv_norm, mat


def pose_input(index, prior, records, gray_frames, buffer_affines, roots, focal, principal):
    uv_img = project(roots[index:index+1] + prior, focal, principal)
    affine = crop_affine(uv_img, 667, 1024)
    src = np.eye(3, dtype=np.float32)
    src[:2] = buffer_affines[index]
    dst = np.eye(3, dtype=np.float32)
    dst[:2] = affine
    mapping = dst @ np.linalg.inv(src)
    prior_gray = cv2.warpAffine(
        gray_frames[index], mapping[:2], (p0.IMAGE_W, p0.IMAGE_H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    inp, uv_crop = channels_from_prior(prior_gray, uv_img, affine)
    return inp, uv_crop, affine


def make_sample(index, prior, records, gray_frames, buffer_affines, roots, focal, principal, aug_rng=None):
    uv_img = project(roots[index:index+1] + prior, focal, principal)
    inp, uv_crop, affine = pose_input(index, prior, records, gray_frames,
                                      buffer_affines, roots, focal, principal)
    gt = records[index]["y3d"]
    if aug_rng is None:
        target2d = p0.normalize_crop_points(uv_crop)
        mat = np.asarray([[1, 0, 0], [0, 1, 0]], np.float32)
        return inp, prior.astype(np.float32), gt, target2d, roots[index], affine, mat
    # p0.augment_one takes a gray crop; preserve exact augmentation settings by
    # applying it to channel zero, then warp the prior render masks identically.
    gray_crop = np.clip(inp[0] * 255.0, 0, 255).astype(np.uint8)
    gt_uv_crop = records[index]["uv_image"] @ affine[:, :2].T + affine[:, 2]
    base, prior_aug, gt_aug, uv_norm, mat = aug_refine(
        gray_crop, gt_uv_crop, gt, prior, aug_rng
    )
    lines = cv2.warpAffine(inp[2], mat, (p0.IMAGE_W, p0.IMAGE_H),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    dots = cv2.warpAffine(inp[3], mat, (p0.IMAGE_W, p0.IMAGE_H),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return (np.concatenate((base, lines[None], dots[None]), axis=0), prior_aug,
            gt_aug, uv_norm, roots[index], affine, mat)


def affine_gt(records, index):
    uv, crop = records[index]["uv_image"], records[index]["uv_crop"]
    design = np.column_stack((uv, np.ones(len(uv))))
    coeff, *_ = np.linalg.lstsq(design, crop, rcond=None)
    return coeff.T.astype(np.float32)


def run_refiner(model, inp, prior):
    with torch.inference_mode():
        delta = model(torch.from_numpy(inp[None]).to(DEVICE))[0].cpu().numpy()
    return (prior + delta).astype(np.float32)


def score_pose(pred, target):
    return p0.score(np.asarray(pred), np.asarray(target))


def metric_row(method, split, mpjpe, tips, before=None, latency=None):
    detail = " ".join(f"{n}={v:.2f}" for n, v in zip(p0.FINGER_NAMES, tips))
    bef = "" if before is None else f" before={before:.2f}"
    lat = "" if latency is None else f" latency={latency:.3f}ms"
    return f"{method:<24} {split:<15} {mpjpe:8.2f}{bef}  {detail}{lat}"


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
    started = time.perf_counter()

    records, _ = p0.read_inputs()
    segments = sorted({r["segment"] for r in records})
    rng_split = np.random.default_rng(SEED)
    perm = np.asarray(segments, dtype=object)
    rng_split.shuffle(perm)
    train_segments = set(perm[:p0.TRAIN_SEGMENTS].tolist())
    val_segments = set(perm[p0.TRAIN_SEGMENTS:].tolist())
    train_idx = np.asarray([i for i, r in enumerate(records) if r["segment"] in train_segments])
    val_idx = np.asarray([i for i, r in enumerate(records) if r["segment"] in val_segments])
    assert not train_segments.intersection(val_segments)
    gray_frames, buffer_affines, roots, focal, principal = camera_data(records)
    y_all = np.stack([r["y3d"] for r in records]).astype(np.float32)
    y2_all = np.stack([p0.normalize_crop_points(r["uv_crop"]) for r in records]).astype(np.float32)

    # Retrain phase0 direct regressor with its 20-epoch recipe.
    x2 = torch.from_numpy(np.stack([p0.make_channels(r["crop_gray"]) for r in records]))
    direct = p0.WebcamAlignedLCFTracker(torch.from_numpy(np.stack([
        np.linalg.norm(y_all[train_idx, c] - y_all[train_idx, a], axis=1).mean()
        for a, c in p0.BONE_PAIRS
    ]).astype(np.float32))).to(DEVICE)
    p0.train_tracker(direct, records, train_idx, y_all[train_idx], y2_all[train_idx],
                     x2, epochs=p0.TRAIN_EPOCHS, augment=True)

    refiner = PoseRefiner().to(DEVICE)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=1.2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)
    rng = np.random.default_rng(SEED + 100)
    for epoch in range(1, EPOCHS + 1):
        refiner.train()
        order = np.random.permutation(train_idx)
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start+BATCH_SIZE]
            samples = []
            for idx in ids:
                gt = y_all[idx]
                prior = synthesize_prior(gt, rng)
                samples.append(make_sample(int(idx), prior, records, gray_frames,
                                           buffer_affines, roots, focal, principal,
                                           aug_rng=rng))
            bx = torch.from_numpy(np.stack([s[0] for s in samples])).to(DEVICE)
            pri = torch.from_numpy(np.stack([s[1] for s in samples])).to(DEVICE)
            tgt = torch.from_numpy(np.stack([s[2] for s in samples])).to(DEVICE)
            uv_tgt = torch.from_numpy(np.stack([s[3] for s in samples])).to(DEVICE)
            roots_b = torch.from_numpy(np.stack([s[4] for s in samples])).to(DEVICE)
            crop_aff = torch.from_numpy(np.stack([s[5] for s in samples])).to(DEVICE)
            aug_aff = torch.from_numpy(np.stack([s[6] for s in samples])).to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            corrected = pri + refiner(bx)
            loss3d = F.l1_loss(corrected, tgt)
            # Invert phase0's camera-XY label rotation before pinhole projection;
            # then apply the identical prior crop and image augmentation affine.
            angles = torch.atan2(aug_aff[:, 0, 1], aug_aff[:, 0, 0])
            c, s = torch.cos(angles)[:, None], torch.sin(angles)[:, None]
            unrot = corrected.clone()
            x, y = corrected[..., 0], corrected[..., 1]
            unrot[..., 0] = c * x - s * y
            unrot[..., 1] = s * x + c * y
            camera_points = unrot + roots_b[:, None, :]
            z = camera_points[..., 2].clamp_min(1.0)
            uv = torch.stack((
                focal[0] * camera_points[..., 0] / z + principal[0],
                focal[1] * camera_points[..., 1] / z + principal[1],
            ), dim=-1)
            uv_h = torch.cat((uv, torch.ones_like(uv[..., :1])), dim=-1)
            crop_h = torch.bmm(uv_h, crop_aff.transpose(1, 2))
            aug_h = torch.cat((crop_h, torch.ones_like(crop_h[..., :1])), dim=-1)
            pred_crop = torch.bmm(aug_h, aug_aff.transpose(1, 2))
            pred_norm = pred_crop.clone()
            pred_norm[..., 0] = 2.0 * pred_crop[..., 0] / p0.IMAGE_W - 1.0
            pred_norm[..., 1] = 2.0 * pred_crop[..., 1] / p0.IMAGE_H - 1.0
            loss2d = F.smooth_l1_loss(pred_norm, uv_tgt)
            loss = loss3d + 15.0 * loss2d
            loss.backward()
            optimizer.step()
        scheduler.step()

    refiner.eval()
    direct.eval()
    # Synthetic-prior severity sweep, one refinement step.
    rows = []
    for severity in (0.3, 0.6, 1.0):
        rng_eval = np.random.default_rng(SEED + int(severity * 1000))
        before, after = [], []
        for idx in val_idx:
            idx = int(idx)
            gt = y_all[idx]
            prior = synthesize_prior(gt, rng_eval, severity)
            inp, _, _ = pose_input(idx, prior, records, gray_frames, buffer_affines,
                                   roots, focal, principal)
            refined = run_refiner(refiner, inp, prior)
            before.append(prior)
            after.append(refined)
        bscore, _ = score_pose(np.stack(before), y_all[val_idx])
        ascore, tips = score_pose(np.stack(after), y_all[val_idx])
        rows.append((f"Synthetic s={severity:.1f}", ascore, tips, bscore))

    # Realistic prior and iterative recropping. Direct predictions from phase0.
    direct_preds = []
    with torch.inference_mode():
        for idx in val_idx:
            direct_preds.append(direct(x2[int(idx):int(idx)+1])[0][0].cpu().numpy())
    priors = np.stack(direct_preds)
    for iteration in range(4):
        if iteration == 0:
            current = priors.copy()
        else:
            current = refined_stack
        scores, tips = score_pose(current, y_all[val_idx])
        rows.append((f"Realistic iter {iteration}", scores, tips, None))
        if iteration < 3:
            refined_stack = []
            for idx, prior in zip(val_idx, current):
                inp, _, _ = pose_input(int(idx), prior, records, gray_frames,
                                       buffer_affines, roots, focal, principal)
                refined_stack.append(run_refiner(refiner, inp, prior))
            refined_stack = np.stack(refined_stack)

    # End-to-end CPU latency includes recropping/rendering the prior input.
    idx = int(val_idx[0])
    sample_prior = priors[0].copy()
    one_times = []
    pair_times = []
    with torch.inference_mode():
        for _ in range(20):
            inp, _, _ = pose_input(idx, sample_prior, records, gray_frames,
                                   buffer_affines, roots, focal, principal)
            run_refiner(refiner, inp, sample_prior)
            current = sample_prior
            for _ in range(2):
                inp, _, _ = pose_input(idx, current, records, gray_frames,
                                       buffer_affines, roots, focal, principal)
                current = run_refiner(refiner, inp, current)
        for _ in range(200):
            t0 = time.perf_counter()
            inp, _, _ = pose_input(idx, sample_prior, records, gray_frames,
                                   buffer_affines, roots, focal, principal)
            run_refiner(refiner, inp, sample_prior)
            one_times.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            with torch.inference_mode():
                d = direct(x2[idx:idx+1])[0][0].cpu().numpy()
                for _ in range(2):
                    inp, _, _ = pose_input(idx, d, records, gray_frames,
                                           buffer_affines, roots, focal, principal)
                    d = run_refiner(refiner, inp, d)
            pair_times.append((time.perf_counter() - t0) * 1000)
    latency_one = statistics.median(one_times)
    latency_combo = statistics.median(pair_times)

    header = f"{'Method':<24} {'Val subset':<15} {'After mm':>9} {'Before mm':>11}  Thumb Index Middle Ring Pinky"
    print("\n" + header)
    print("-" * len(header))
    for method, after, tips, before in rows:
        print(metric_row(method, "held-out segments", after, tips, before))
    print(f"\nLatency: one refiner step {latency_one:.3f} ms; direct + 2 refinements {latency_combo:.3f} ms (median of 200, batch 1, CPU).")
    print(f"Split: {len(train_segments)}/{len(val_segments)} segments, {len(train_idx)}/{len(val_idx)} frames; seed={SEED}; total wall time {time.perf_counter()-started:.1f}s.")
    print("Assumptions: phase0 root-relative camera-frame joints define targets; camera root translation is recovered from the same right-hand annotations; global root shift is represented in the prior and corrected by the 63 deltas; severity uses the supplied value directly for the sweep.")


if __name__ == "__main__":
    main()
