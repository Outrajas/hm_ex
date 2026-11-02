"""Colab follow-up: OOF-trained, trust-gated pose refinement.

Expected in /content/experiments:
  phase0_sanity.py, phase1_refine.py
Expected in /content/dataset (or in the zip in Drive): manifest.csv,
  cam_params.json, and the image/keypoint files named by manifest.csv.

The phase0/phase1 modules are loaded with CUDA enabled; their source files are
left unchanged. This script trains direct-regressor fold models for OOF priors,
trains one refiner on those priors, and evaluates fixed correction gates on the
same segment-held-out validation split.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------- Colab paths and setup ---------------------------

DATA_ROOT = Path("/content/dataset")
DRIVE_ZIP = Path("/content/drive/MyDrive/reinterhand_5114_dataset.zip")
MODULE_DIR = Path("/content/experiments")
SEED = 20251018
N_OOF_FOLDS = 4
DIRECT_EPOCHS = 20
REFINER_EPOCHS = 25
BATCH_SIZE = 64


def ensure_dataset():
    if not list(DATA_ROOT.rglob("manifest.csv")) and DRIVE_ZIP.is_file():
        print(f"Extracting {DRIVE_ZIP} to {DATA_ROOT} ...")
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(DRIVE_ZIP) as zf:
            zf.extractall(DATA_ROOT)
    manifests = list(DATA_ROOT.rglob("manifest.csv"))
    cameras = list(DATA_ROOT.rglob("cam_params.json"))
    if not manifests or not cameras:
        raise FileNotFoundError(
            "Expected manifest.csv and cam_params.json under /content/dataset "
            "or /content/drive/MyDrive/reinterhand_5114_dataset.zip."
        )
    return manifests[0], cameras[0]


def load_module_with_cuda(name: str, path: Path, *, phase1: bool = False):
    """Load the supplied phase module after replacing its CPU-only guards in memory."""
    source = path.read_text(encoding="utf-8")
    source = source.replace('os.environ["CUDA_VISIBLE_DEVICES"] = ""\n', "")
    source = source.replace(
        'DEVICE = torch.device("cpu")',
        'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
        + ("\np0.DEVICE = DEVICE" if phase1 else ""),
    )
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Select Runtime > Change runtime type > T4 GPU in Colab.")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng_py_start = time.perf_counter()

    manifest_path, camera_path = ensure_dataset()
    if not (MODULE_DIR / "phase0_sanity.py").is_file() or not (MODULE_DIR / "phase1_refine.py").is_file():
        raise FileNotFoundError(
            "Upload phase0_sanity.py and phase1_refine.py to /content/experiments first."
        )

    p0 = load_module_with_cuda("phase0_sanity", MODULE_DIR / "phase0_sanity.py")
    p1 = load_module_with_cuda("phase1_refine", MODULE_DIR / "phase1_refine.py", phase1=True)
    device = torch.device("cuda")
    p0.DEVICE = device
    p1.DEVICE = device

    # The Colab archive used in the pasted notebook has these files at its root.
    # For a nested archive, keep paths relative to the discovered capture folder.
    data_base = manifest_path.parent
    p0.ROOT = data_base
    p0.CAPTURE_DIR = data_base
    p0.SUBSET_DIR = data_base
    p0.MANIFEST_PATH = manifest_path
    p0.CAMERA_PATH = camera_path
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Manifest: {manifest_path}")

    records, image_size = p0.read_inputs()
    segments = sorted({r["segment"] for r in records})
    if len(segments) != 56:
        raise ValueError(f"Expected 56 segments, found {len(segments)}.")
    split_rng = np.random.default_rng(SEED)
    shuffled = np.asarray(segments, dtype=object)
    split_rng.shuffle(shuffled)
    train_segments = set(shuffled[:44].tolist())
    val_segments = set(shuffled[44:].tolist())
    train_idx = np.asarray([i for i, r in enumerate(records) if r["segment"] in train_segments], dtype=np.int64)
    val_idx = np.asarray([i for i, r in enumerate(records) if r["segment"] in val_segments], dtype=np.int64)
    assert not train_segments.intersection(val_segments)

    print("Staging prior crops and camera roots...")
    gray_buffers, buffer_affines, roots, focal, principal = p1.camera_data(records)
    y_all = np.stack([r["y3d"] for r in records]).astype(np.float32)
    y2_all = np.stack([p0.normalize_crop_points(r["uv_crop"]) for r in records]).astype(np.float32)
    x_np = np.stack([p0.make_channels(r["crop_gray"]) for r in records])
    x_all = torch.from_numpy(x_np)  # CPU source; minibatches are moved to T4.

    # ---------------------- Segment-wise OOF direct priors ----------------------

    def new_direct(fit_indices):
        lengths = np.stack([
            np.linalg.norm(y_all[fit_indices, child] - y_all[fit_indices, parent], axis=1).mean()
            for parent, child in p0.BONE_PAIRS
        ]).astype(np.float32)
        return p0.WebcamAlignedLCFTracker(torch.from_numpy(lengths)).to(device)

    def predict_direct(model, indices):
        model.eval()
        out = []
        with torch.inference_mode():
            for start in range(0, len(indices), 128):
                ids = indices[start:start + 128]
                pred, _, _ = model(x_all[ids].to(device, non_blocking=True))
                out.append(pred.cpu().numpy())
        return np.concatenate(out, axis=0)

    train_seg_list = sorted(train_segments)
    fold_rng = np.random.default_rng(SEED + 1)
    fold_order = np.asarray(train_seg_list, dtype=object)
    fold_rng.shuffle(fold_order)
    folds = [set(x.tolist()) for x in np.array_split(fold_order, N_OOF_FOLDS)]
    oof_priors = np.full_like(y_all, np.nan)
    print(f"\nTraining {N_OOF_FOLDS} segment-held-out direct models for OOF priors...")
    for fold_no, held_segments in enumerate(folds, start=1):
        held_idx = np.asarray([i for i in train_idx if records[i]["segment"] in held_segments], dtype=np.int64)
        fit_idx = np.asarray([i for i in train_idx if records[i]["segment"] not in held_segments], dtype=np.int64)
        print(f"  Fold {fold_no}/{N_OOF_FOLDS}: fit {len(fit_idx)}, held out {len(held_idx)} frames")
        torch.manual_seed(SEED + fold_no)
        model = new_direct(fit_idx)
        p0.train_tracker(model, records, fit_idx, y_all[fit_idx], y2_all[fit_idx],
                         x_all, epochs=DIRECT_EPOCHS, augment=True)
        oof_priors[held_idx] = predict_direct(model, held_idx)
        del model
        torch.cuda.empty_cache()
    if not np.isfinite(oof_priors[train_idx]).all():
        raise RuntimeError("Some training samples did not receive an OOF prior.")

    # Final phase0-style model provides realistic validation priors and baseline.
    print("Training final direct model on all 44 training segments...")
    torch.manual_seed(SEED)
    direct = new_direct(train_idx)
    p0.train_tracker(direct, records, train_idx, y_all[train_idx], y2_all[train_idx],
                     x_all, epochs=DIRECT_EPOCHS, augment=True)
    val_priors = predict_direct(direct, val_idx)
    direct_mpjpe, direct_tips = p0.score(val_priors, y_all[val_idx])

    # --------------------- Train refiner on realistic OOF priors ---------------------

    print("\nTraining one-step refiner from OOF priors...")
    refiner = p1.PoseRefiner().to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=1.2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=REFINER_EPOCHS, eta_min=1e-4
    )
    aug_rng = np.random.default_rng(SEED + 100)
    for epoch in range(1, REFINER_EPOCHS + 1):
        refiner.train()
        order = np.random.permutation(train_idx)
        running = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            batch = [p1.make_sample(int(i), oof_priors[i], records, gray_buffers,
                                    buffer_affines, roots, focal, principal,
                                    aug_rng=aug_rng) for i in ids]
            bx = torch.from_numpy(np.stack([s[0] for s in batch])).to(device)
            pri = torch.from_numpy(np.stack([s[1] for s in batch])).to(device)
            target3d = torch.from_numpy(np.stack([s[2] for s in batch])).to(device)
            target2d = torch.from_numpy(np.stack([s[3] for s in batch])).to(device)
            root_cam = torch.from_numpy(np.stack([s[4] for s in batch])).to(device)
            crop_aff = torch.from_numpy(np.stack([s[5] for s in batch])).to(device)
            aug_aff = torch.from_numpy(np.stack([s[6] for s in batch])).to(device)

            optimizer.zero_grad(set_to_none=True)
            corrected = pri + refiner(bx)
            loss3d = F.l1_loss(corrected, target3d)
            # Undo the camera-XY augmentation, pinhole-project, then apply the
            # prior crop and augmentation transforms to match phase0 2D labels.
            angle = torch.atan2(aug_aff[:, 0, 1], aug_aff[:, 0, 0])
            c, s = torch.cos(angle)[:, None], torch.sin(angle)[:, None]
            unrot = corrected.clone()
            xx, yy = corrected[..., 0], corrected[..., 1]
            unrot[..., 0] = c * xx - s * yy
            unrot[..., 1] = s * xx + c * yy
            cam = unrot + root_cam[:, None, :]
            z = cam[..., 2].clamp_min(1.0)
            uv = torch.stack((focal[0] * cam[..., 0] / z + principal[0],
                              focal[1] * cam[..., 1] / z + principal[1]), dim=-1)
            uv_h = torch.cat((uv, torch.ones_like(uv[..., :1])), dim=-1)
            crop_xy = torch.bmm(uv_h, crop_aff.transpose(1, 2))
            crop_h = torch.cat((crop_xy, torch.ones_like(crop_xy[..., :1])), dim=-1)
            aug_xy = torch.bmm(crop_h, aug_aff.transpose(1, 2))
            pred2d = aug_xy.clone()
            pred2d[..., 0] = 2 * aug_xy[..., 0] / p0.IMAGE_W - 1
            pred2d[..., 1] = 2 * aug_xy[..., 1] / p0.IMAGE_H - 1
            loss = loss3d + 15.0 * F.smooth_l1_loss(pred2d, target2d)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        scheduler.step()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:02d}/{REFINER_EPOCHS}: loss={running / max(1, (len(order)+BATCH_SIZE-1)//BATCH_SIZE):.4f}")

    # ---------------------- Damped correction / drift report ----------------------

    refiner.eval()

    def refine(index, prior):
        inp, _, _ = p1.pose_input(int(index), prior, records, gray_buffers,
                                  buffer_affines, roots, focal, principal)
        return p1.run_refiner(refiner, inp, prior)

    def apply_gate(prior_stack, alpha):
        return np.stack([
            prior + alpha * (refine(idx, prior) - prior)
            for idx, prior in zip(val_idx, prior_stack)
        ])

    rows = [("Direct baseline", 0.0, direct_mpjpe, direct_tips)]
    one_step = {}
    for alpha in (0.25, 0.50, 0.75, 1.00):
        pred = apply_gate(val_priors, alpha)
        score, tips = p0.score(pred, y_all[val_idx])
        one_step[alpha] = pred
        rows.append(("One step", alpha, score, tips))

    # Re-crop on the newly gated pose on every iteration to expose drift.
    for alpha in (0.25, 0.50, 0.75):
        current = val_priors.copy()
        for iteration in range(1, 4):
            current = apply_gate(current, alpha)
            score, tips = p0.score(current, y_all[val_idx])
            rows.append((f"Iter {iteration}", alpha, score, tips))

    print("\n" + "=" * 112)
    print("PHASE 2: OOF-TRAINED REFINER WITH FIXED CORRECTION GATES (HELD-OUT SEGMENTS)")
    print("=" * 112)
    print(f"{'Method':<20} {'Gate':>6} {'MPJPE mm':>10}  Thumb  Index Middle  Ring Pinky")
    print("-" * 76)
    for method, alpha, score, tips in rows:
        print(f"{method:<20} {alpha:6.2f} {score:10.2f}  " +
              " ".join(f"{v:6.2f}" for v in tips))

    checkpoint = "/content/phase2_oof_gated_refiner.pt"
    torch.save({
        "direct_state_dict": {k: v.detach().cpu() for k, v in direct.state_dict().items()},
        "refiner_state_dict": {k: v.detach().cpu() for k, v in refiner.state_dict().items()},
        "seed": SEED,
        "train_segments": sorted(train_segments),
        "val_segments": sorted(val_segments),
        "oof_folds": N_OOF_FOLDS,
        "gates": [0.25, 0.50, 0.75, 1.00],
    }, checkpoint)
    print(f"\nSaved checkpoint: {checkpoint}")
    print(f"Runtime: {time.perf_counter() - rng_py_start:.1f}s | image size: {image_size}")


if __name__ == "__main__":
    main()
