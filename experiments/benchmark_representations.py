"""Representation Benchmark: Evaluating Edge Representations on 3D Pose Recovery.

Compares:
1. Scalar Sobel (|G|)
2. Directional Sobel (Gx, Gy, |G|)
3. Di Zenzo (lambda_max^0.5)
4. Hybrid Vector Field (Gx, Gy, |G|, DiZenzo)
5. RGB Reference

Target: Root-relative 3D landmarks (24 keypoints, mm).
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(".")
CAPTURE_ID = "m--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands"
CAPTURE_DIR = ROOT / "datasets" / "reinterhand" / "captures" / CAPTURE_ID
MANIFEST_PATH = CAPTURE_DIR / "derived" / "mugsy_subset" / "manifest.csv"

IN_W, IN_H = 256, 192  # Preserves 4:3 native aspect ratio

def compute_representations(img_bgr: np.ndarray):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1. Sobel Components
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 1020.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 1020.0
    g_mag = np.sqrt(gx**2 + gy**2)

    # 2. Di Zenzo Structure Tensor
    rx = cv2.Sobel(img_rgb[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    ry = cv2.Sobel(img_rgb[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
    gx_c = cv2.Sobel(img_rgb[:, :, 1], cv2.CV_32F, 1, 0, ksize=3)
    gy_c = cv2.Sobel(img_rgb[:, :, 1], cv2.CV_32F, 0, 1, ksize=3)
    bx = cv2.Sobel(img_rgb[:, :, 2], cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(img_rgb[:, :, 2], cv2.CV_32F, 0, 1, ksize=3)

    gxx = rx**2 + gx_c**2 + bx**2
    gyy = ry**2 + gy_c**2 + by**2
    gxy = rx * ry + gx_c * gy_c + bx * by

    diff = gxx - gyy
    disc = np.sqrt(np.maximum(diff**2 + 4.0 * (gxy**2), 0.0))
    lambda_max = 0.5 * ((gxx + gyy) + disc)
    dizenzo_mag = (np.sqrt(np.maximum(lambda_max, 0.0)) / 1766.7).astype(np.float32)

    rgb_norm = (img_rgb / 255.0).astype(np.float32)

    return {
        "sobel_mag": g_mag[..., None],
        "sobel_dir": np.stack([gx, gy, g_mag], axis=-1),
        "dizenzo": dizenzo_mag[..., None],
        "hybrid": np.stack([gx, gy, g_mag, dizenzo_mag], axis=-1),
        "rgb": rgb_norm
    }

class HandPoseDataset(Dataset):
    def __init__(self, samples, rep_key):
        self.samples = samples
        self.rep_key = rep_key

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(str(CAPTURE_DIR / s["image_path"]))
        img_resized = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)

        reps = compute_representations(img_resized)
        x = torch.from_numpy(reps[self.rep_key]).permute(2, 0, 1).float()

        # Load right hand 24 keypoints (in mm)
        with open(CAPTURE_DIR / s["orig_kp_right"], "r") as f:
            kps = np.array(json.load(f), dtype=np.float32)

        # Root-relative target in mm (wrist is joint 0)
        kps_rel = (kps - kps[0:1]).flatten()
        y = torch.from_numpy(kps_rel).float()

        return x, y

class LightweightPoseProbe(nn.Module):
    def __init__(self, in_channels, out_dim=72):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 8))
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 6 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        feat = self.features(x)
        return self.head(torch.flatten(feat, 1))

def run_benchmark():
    print("=" * 75)
    print(" RE:INTERHAND REPRESENTATION BENCHMARK (560 STRATIFIED SAMPLES)")
    print("=" * 75)

    # 1. Stratify 10 frames per segment across all 56 segments
    segment_map = defaultdict(list)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            segment_map[r["segment"]].append(r)

    train_samples, test_samples = [], []
    for seg, items in segment_map.items():
        indices = np.linspace(0, len(items) - 1, 10, dtype=int)
        stratified = [items[i] for i in indices]
        # 8 train, 2 test per segment
        train_samples.extend(stratified[:8])
        test_samples.extend(stratified[8:])

    print(f"Dataset Split: {len(train_samples)} Train | {len(test_samples)} Test across 56 segments.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}\n")

    variants = [
        ("Scalar Sobel", "sobel_mag", 1),
        ("Directional Sobel", "sobel_dir", 3),
        ("Di Zenzo", "dizenzo", 1),
        ("Hybrid Vector Field", "hybrid", 4),
        ("RGB Ceiling", "rgb", 3)
    ]

    results = {}

    for name, rep_key, channels in variants:
        train_loader = DataLoader(HandPoseDataset(train_samples, rep_key), batch_size=16, shuffle=True)
        test_loader = DataLoader(HandPoseDataset(test_samples, rep_key), batch_size=16, shuffle=False)

        model = LightweightPoseProbe(in_channels=channels, out_dim=72).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()

        for epoch in range(15):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

        # Evaluate MPJPE (mm) on held-out test split
        model.eval()
        total_error = 0.0
        total_joints = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx).view(-1, 24, 3)
                gt = by.view(-1, 24, 3)
                # Euclidean distance per joint in mm
                dist = torch.norm(pred - gt, dim=-1)
                total_error += dist.sum().item()
                total_joints += dist.numel()

        mpjpe = total_error / total_joints
        results[name] = mpjpe
        print(f"  [{name:<20}] Test MPJPE: {mpjpe:6.2f} mm")

    print("\n" + "=" * 75)
    print(" BENCHMARK SUMMARY (Mean Per-Joint Position Error in mm)")
    print("=" * 75)
    for name, error in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:<25} : {error:6.2f} mm")
    print("=" * 75)

if __name__ == "__main__":
    run_benchmark()
