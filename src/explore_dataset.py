"""
explore_dataset.py — quick EDA on your extracted QQB dataset. Run this
after extracting the dataset (see README "Get the dataset & weights") to
get a plain-English summary printed to the terminal + a sample image grid
saved as a PNG — screenshot/print that PNG to show your professor.

Usage:
    python src/explore_dataset.py --data_root data/qqb --samples 3
"""
import os
import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import _resolve_path


def _raw_sar(root, bid):
    with h5py.File(_resolve_path(root, bid, "_SAR.mat"), "r") as f:
        return np.float32(f["x1"])


def _raw_sarftp(root, bid):
    with h5py.File(_resolve_path(root, bid, "_SARftp.mat"), "r") as f:
        return np.float32(f["x2"])


def _raw_opt(root, bid):
    with h5py.File(_resolve_path(root, bid, "_opt.mat"), "r") as f:
        x = np.float32(f["x3"])
    return np.transpose(x, (1, 2, 0))  # H,W,3 for imshow


def _raw_optftp(root, bid):
    with h5py.File(_resolve_path(root, bid, "_optftp.mat"), "r") as f:
        return np.float32(f["x4"])


def summarize(root):
    all_ids, all_labels = [], []
    per_fold = {}
    for i in range(1, 6):
        ff = f"fold-{i}.txt"
        path = os.path.join(root, ff)
        if not os.path.exists(path):
            print(f"  ! {ff} not found, skipping")
            continue
        ids, labels = [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                bid, lbl = line.split(",")
                ids.append(bid)
                labels.append(int(lbl))
        per_fold[ff] = (ids, labels)
        all_ids += ids
        all_labels += labels

    total = len(all_ids)
    damaged = sum(all_labels)
    intact = total - damaged

    print(f"\n=== QQB dataset summary: {root} ===")
    print(f"Total labeled buildings : {total}")
    print(f"  damaged  : {damaged}  ({100 * damaged / max(1, total):.1f}%)")
    print(f"  intact   : {intact}  ({100 * intact / max(1, total):.1f}%)")
    print(f"Folds found : {len(per_fold)}/5")
    for ff, (ids, labels) in per_fold.items():
        d = sum(labels)
        print(f"  {ff:<12} {len(ids):>5} buildings  ({d} damaged, {len(ids) - d} intact)")
    print("\nEach building = 4 files: <id>_SAR.mat, <id>_SARftp.mat, <id>_opt.mat, <id>_optftp.mat")
    print("SAR = radar intensity patch. SARftp/optftp = building footprint masks (which pixels are the building).")
    print("opt = 3-channel optical patch. Labels: 0 = intact, 1 = damaged.")
    print("Files can sit flat in data_root, or split into damaged/ + intact/ subfolders — either works.")
    return all_ids, all_labels, per_fold


def visualize_samples(root, ids, labels, n_samples, out_path):
    damaged_ids = [bid for bid, l in zip(ids, labels) if l == 1][:n_samples]
    intact_ids = [bid for bid, l in zip(ids, labels) if l == 0][:n_samples]
    rows = damaged_ids + intact_ids
    if not rows:
        print("No buildings found to visualize.")
        return

    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]

    for r, bid in enumerate(rows):
        label_name = "DAMAGED" if bid in damaged_ids else "intact"
        sar = _raw_sar(root, bid)
        sarftp = _raw_sarftp(root, bid)
        opt = _raw_opt(root, bid)
        optftp = _raw_optftp(root, bid)
        opt_disp = np.clip((opt - opt.min()) / (opt.max() - opt.min() + 1e-8), 0, 1)

        axes[r, 0].imshow(sar, cmap="gray")
        axes[r, 0].set_title(f"{bid}\nSAR ({label_name})", fontsize=8)
        axes[r, 1].imshow(sarftp, cmap="gray")
        axes[r, 1].set_title("SAR footprint", fontsize=8)
        axes[r, 2].imshow(opt_disp)
        axes[r, 2].set_title("Optical", fontsize=8)
        axes[r, 3].imshow(optftp, cmap="gray")
        axes[r, 3].set_title("Optical footprint", fontsize=8)
        for c in range(4):
            axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSample visualization saved -> {out_path}  (open this to show your professor)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/qqb")
    ap.add_argument("--samples", type=int, default=3, help="damaged + intact examples each")
    ap.add_argument("--out", default="outputs/dataset_sample.png")
    args = ap.parse_args()

    ids, labels, _ = summarize(args.data_root)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    visualize_samples(args.data_root, ids, labels, args.samples, args.out)
