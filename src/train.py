"""
train.py — trains MultiModalDamageClassifier using 5-fold cross-validation
splits (fold-1..5.txt, as shipped by QQB). One fold held out for
validation, the rest train. Class imbalance (damaged is the rare class)
is handled with a weighted random sampler.
"""
import os
import time
import datetime
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve

import config
from dataset import QQBDataset, Augment
from model import MultiModalDamageClassifier

MODALITIES = {
    "all": ["sar", "sarftp", "opt", "optftp"],
    "sar": ["sar", "sarftp"],
    "opt": ["opt", "optftp"],
}


def best_f1(probs, labels):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precision * recall / (precision + recall + 1e-8)
    idx = int(np.argmax(f1s))
    return float(f1s[idx]), float(precision[idx]), float(recall[idx])


def main(epochs=30, batch_size=16, lr=1e-4, val_fold=None, mode=None):
    val_fold = val_fold or config.VAL_FOLD
    mode = mode or config.MODE
    keys = MODALITIES[mode]
    train_folds = [f for f in config.ALL_FOLDS if f != val_fold]

    train_ds = QQBDataset(config.DATA_ROOT, train_folds)
    val_ds = QQBDataset(config.DATA_ROOT, [val_fold])
    now_start = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now_start}] train: {len(train_ds)}  val: {len(val_ds)}  mode: {mode}  val_fold: {val_fold}", flush=True)

    labels = np.array(train_ds.labels)
    class_counts = np.array([max(1, (labels == c).sum()) for c in (0, 1)])
    weight_per_class = 1.0 / class_counts
    sample_weights = weight_per_class[labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sar_pretrain = config.SAR_PRETRAIN if os.path.exists(config.SAR_PRETRAIN) else None
    if mode in ("all", "sar") and sar_pretrain is None:
        print("note: no SAR-HUB weights found at", config.SAR_PRETRAIN,
              "-> SAR branch trains from random init (see README to download them)")
    model = MultiModalDamageClassifier(mode=mode, sar_pretrain=sar_pretrain).to(device)
    aug = Augment().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = torch.nn.BCEWithLogitsLoss()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"model_{mode}_{val_fold.replace('.txt','')}.pt")

    best_auroc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for images, label, _ in train_loader:
            raw = {k: images[k].to(device) for k in keys}
            augmented = aug(*[raw[k] for k in keys])
            batch = dict(zip(keys, augmented))
            label = label.to(device)

            opt.zero_grad()
            logits = model(**batch)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * label.size(0)
        sched.step()
        train_loss = total_loss / len(train_ds)

        model.eval()
        probs, gts = [], []
        with torch.no_grad():
            for images, label, _ in val_loader:
                batch = {k: images[k].to(device) for k in keys}
                logits = model(**batch)
                probs.append(torch.sigmoid(logits).cpu().item())
                gts.append(label.item())
        probs, gts = np.array(probs), np.array(gts)
        auroc = roc_auc_score(gts, probs) if len(set(gts.tolist())) > 1 else float("nan")
        f1, prec, rec = best_f1(probs, gts)

        elapsed = time.time() - t0
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now_str}] epoch {epoch:02d}  train_loss {train_loss:.4f}  val_auroc {auroc:.4f}  "
              f"best_f1 {f1:.4f}  precision {prec:.4f}  recall {rec:.4f}  ({elapsed:.1f}s)", flush=True)

        if not np.isnan(auroc) and auroc >= best_auroc:
            best_auroc = auroc
            torch.save({"state_dict": model.state_dict(), "mode": mode}, ckpt_path)

    print(f"Best val AUROC {best_auroc:.4f} -> checkpoint saved to {ckpt_path}")
    return ckpt_path, best_auroc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_fold", default=None, help="e.g. fold-1.txt (default: config.VAL_FOLD)")
    ap.add_argument("--mode", default=None, choices=["all", "sar", "opt"])
    args = ap.parse_args()
    main(args.epochs, args.batch_size, args.lr, args.val_fold, args.mode)
