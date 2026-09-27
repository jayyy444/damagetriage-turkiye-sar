"""
infer.py — runs the trained model over a fold (default: config.VAL_FOLD)
and writes outputs/predictions.json + outputs/review_queue.json, mirroring
the interface of the original mexico-earthquake build's infer.py so the
rest of the pipeline (gradcam/aggregate/report) doesn't need to change.
"""
import os
import json
import argparse

import torch
from torch.utils.data import DataLoader

import config
from dataset import QQBDataset
from model import MultiModalDamageClassifier

MODALITIES = {
    "all": ["sar", "sarftp", "opt", "optftp"],
    "sar": ["sar", "sarftp"],
    "opt": ["opt", "optftp"],
}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MultiModalDamageClassifier(mode=ckpt["mode"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mode"]


def _default_ckpt(fold):
    return os.path.join(config.CHECKPOINT_DIR, f"model_{config.MODE}_{fold.replace('.txt','')}.pt")


def run_inference(ckpt_path=None, fold=None):
    fold = fold or config.VAL_FOLD
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path or _default_ckpt(fold)
    model, mode = load_model(ckpt_path, device)
    keys = MODALITIES[mode]

    ds = QQBDataset(config.DATA_ROOT, [fold])
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    predictions, review_queue = [], []
    with torch.no_grad():
        for images, label, bid in loader:
            batch = {k: images[k].to(device) for k in keys}
            prob = torch.sigmoid(model(**batch)).item()
            pred_label = config.CLASSES[int(prob >= 0.5)]
            entry = {
                "building_id": bid[0],
                "true_label": config.CLASSES[int(label.item())],
                "predicted_label": pred_label,
                "probability_damaged": round(prob, 4),
                "confidence": round(max(prob, 1 - prob), 4),
            }
            predictions.append(entry)
            if abs(prob - 0.5) < config.REVIEW_MARGIN:
                review_queue.append(entry)

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUTS_DIR, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)
    with open(os.path.join(config.OUTPUTS_DIR, "review_queue.json"), "w") as f:
        json.dump(review_queue, f, indent=2)

    print(f"{len(predictions)} predictions -> outputs/predictions.json")
    print(f"{len(review_queue)} flagged (probability within {config.REVIEW_MARGIN} of 0.5) "
          f"-> outputs/review_queue.json")
    return predictions, review_queue


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--fold", default=None)
    args = ap.parse_args()
    run_inference(args.checkpoint, args.fold)
