"""
explain.py — generates paired SAR + optical Grad-CAM heatmaps for the
highest-confidence "damaged" predictions, so a reviewer can see *why* the
model called it damage in both modalities side by side.
"""
import os
import json

import torch

import config
from dataset import QQBDataset
from model import MultiModalDamageClassifier
from gradcam import generate_gradcam, overlay_heatmap

MODALITIES = {
    "all": ["sar", "sarftp", "opt", "optftp"],
    "sar": ["sar", "sarftp"],
    "opt": ["opt", "optftp"],
}


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MultiModalDamageClassifier(mode=ckpt["mode"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mode"]


def generate_heatmaps_for(building_ids, ckpt_path, fold_root=None, max_items=20):
    fold_root = fold_root or config.DATA_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mode = _load_model(ckpt_path, device)
    keys = MODALITIES[mode]

    out_dir = os.path.join(config.OUTPUTS_DIR, "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    ds = QQBDataset(fold_root, config.ALL_FOLDS)
    id_to_idx = {bid: i for i, bid in enumerate(ds.ids)}

    saved = []
    for bid in building_ids[:max_items]:
        if bid not in id_to_idx:
            continue
        images, _, _ = ds[id_to_idx[bid]]
        inputs = {k: images[k].unsqueeze(0) for k in keys}

        for modality in [m for m in ("sar", "opt") if m in keys]:
            cam, prob = generate_gradcam(model, inputs, modality, device)
            base = images[modality].numpy()
            overlay = overlay_heatmap(base, cam)
            safe_bid = bid.replace("/", "_").replace("\\", "_")
            out_path = os.path.join(out_dir, f"{safe_bid}_{modality}_heatmap.png")
            overlay.save(out_path)
            saved.append({
                "building_id": bid, "modality": modality,
                "probability_damaged": round(prob, 4), "heatmap_path": out_path,
            })

    print(f"Saved {len(saved)} heatmaps -> {out_dir}")
    return saved


if __name__ == "__main__":
    with open(os.path.join(config.OUTPUTS_DIR, "predictions.json")) as f:
        preds = json.load(f)
    damaged = [p for p in preds if p["predicted_label"] == "damaged"]
    damaged.sort(key=lambda p: p["probability_damaged"], reverse=True)
    top_ids = [p["building_id"] for p in damaged]
    ckpt = os.path.join(config.CHECKPOINT_DIR, f"model_{config.MODE}_{config.VAL_FOLD.replace('.txt','')}.pt")
    generate_heatmaps_for(top_ids, ckpt)
