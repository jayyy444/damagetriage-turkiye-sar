"""
dataset.py — loads QuickQuakeBuildings SAR+Optical building patches.

Expected layout after extracting the dataset (see README "Get the dataset"):
    <DATA_ROOT>/
        <OSMID>_SAR.mat        # SAR intensity patch        (h5py key 'x1')
        <OSMID>_SARftp.mat     # SAR building footprint     (h5py key 'x2')
        <OSMID>_opt.mat        # optical patch               (h5py key 'x3')
        <OSMID>_optftp.mat     # optical building footprint (h5py key 'x4')
        fold-1.txt ... fold-5.txt   # "<OSMID>,<label>" lines, label: 0=intact 1=damaged

This is the exact layout QQB ships (github.com/ya0-sun/PostEQ-SARopt-BuildingDamage).
Preprocessing (percentile clip + normalize for SAR/optical, nearest resize
for footprint masks) matches the paper's reference implementation so
SAR-HUB / ImageNet pretrained weights transfer correctly.
"""
import os
import json

import cv2
import h5py
import numpy as np
import torch
import kornia.augmentation as K

import config


def _resolve_path(root, osmid, suffix):
    """The dataset zip is shipped as root/damaged/ + root/intact/
    subfolders (per QQB's own README diagram), but their reference code
    reads paths flat (root/<osmid>_SAR.mat directly) — a mismatch in the
    original repo. Check both layouts so you don't have to manually merge
    the two subfolders after extracting."""
    flat = os.path.join(root, f"{osmid}{suffix}")
    if os.path.exists(flat):
        return flat
    for sub in ("damaged", "intact"):
        candidate = os.path.join(root, sub, f"{osmid}{suffix}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find {osmid}{suffix} under {root} (checked flat, damaged/, intact/ — "
        f"did you extract the dataset zip without renaming folders?)"
    )


def _load_sar(path):
    with h5py.File(path, "r") as f:
        x = np.float32(f["x1"])
    x[x < -100] = x[x > -100].min()  # fill missing pixels with min of valid ones
    x = cv2.resize(x, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_LINEAR_EXACT)
    p1, p99 = np.percentile(x, (1, 99))
    x = np.clip(x, p1, p99)
    x = ((x - p1) / (p99 - p1 + 1e-8)).astype(np.float32)  # np.percentile upcasts to float64 — cast back
    return np.stack([x, x, x], 0)  # 3,H,W — replicate to 3ch so ResNet-style backbones work


def _load_footprint(path, key):
    with h5py.File(path, "r") as f:
        x = np.float32(f[key])
    x = cv2.resize(x, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return np.stack([x, x, x], 0)


def _load_opt(path):
    with h5py.File(path, "r") as f:
        x = np.float32(f["x3"])
    x = cv2.resize(np.transpose(x, (1, 2, 0)), (config.IMG_SIZE, config.IMG_SIZE),
                    interpolation=cv2.INTER_LINEAR_EXACT)
    x = np.transpose(x, (2, 0, 1))
    p1, p99 = np.percentile(x, (1, 99))
    x = np.clip(x, p1, p99)
    x = ((x - p1) / (p99 - p1 + 1e-8)).astype(np.float32)  # np.percentile upcasts to float64 — cast back
    return x


class QQBDataset(torch.utils.data.Dataset):
    def __init__(self, root, folds):
        self.root = root
        self.ids, self.labels = [], []
        for fold in folds:
            fold_path = os.path.join(root, fold)
            with open(fold_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    osmid, label = line.split(",")
                    self.ids.append(osmid)
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        bid = self.ids[idx]
        label = np.float32(self.labels[idx])
        sar = _load_sar(_resolve_path(self.root, bid, "_SAR.mat"))
        sarftp = _load_footprint(_resolve_path(self.root, bid, "_SARftp.mat"), "x2")
        opt = _load_opt(_resolve_path(self.root, bid, "_opt.mat"))
        optftp = _load_footprint(_resolve_path(self.root, bid, "_optftp.mat"), "x4")
        images = {
            "sar": torch.from_numpy(sar.copy()),
            "sarftp": torch.from_numpy(sarftp.copy()),
            "opt": torch.from_numpy(opt.copy()),
            "optftp": torch.from_numpy(optftp.copy()),
        }
        return images, label, bid


class Augment(torch.nn.Module):
    """Applies the same random crop/flip params across all given modality
    tensors so they stay spatially aligned (kornia's `params` reuse) —
    e.g. the SAR and optical crop must show the same physical patch."""

    def __init__(self):
        super().__init__()
        self.crop = K.RandomResizedCrop((config.IMG_SIZE, config.IMG_SIZE), scale=(0.8, 1.0))
        self.hflip = K.RandomHorizontalFlip(p=0.5)
        self.vflip = K.RandomVerticalFlip(p=0.5)

    def forward(self, *tensors):
        first, *rest = tensors
        first = self.crop(first)
        rest = [self.crop(t, self.crop._params) for t in rest]
        first = self.hflip(first)
        rest = [self.hflip(t, self.hflip._params) for t in rest]
        first = self.vflip(first)
        rest = [self.vflip(t, self.vflip._params) for t in rest]
        return (first, *rest)


def build_manifest(root, folds=None, out_path=None):
    """Write a labels.json manifest (building_id, fold, label) — used by
    run_pipeline.py to know prep is done, and handy for quick inspection
    without touching the raw fold files directly."""
    folds = folds or config.ALL_FOLDS
    records = []
    for fold in folds:
        fold_path = os.path.join(root, fold)
        if not os.path.exists(fold_path):
            continue
        with open(fold_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                osmid, label = line.split(",")
                records.append({"building_id": osmid, "fold": fold, "label": int(label)})
    out_path = out_path or os.path.join(root, "labels.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"{len(records)} buildings across {len(folds)} folds -> {out_path}")
    return records


if __name__ == "__main__":
    build_manifest(config.DATA_ROOT)
