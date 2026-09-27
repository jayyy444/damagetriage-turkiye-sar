# DamageTriage — SAR + Optical build (2023 Türkiye-Syria earthquake)

This is a v2 of the DamageTriage building-damage triage pipeline, rebuilt
on the **QuickQuakeBuildings (QQB)** dataset instead of xView2/xBD's
Mexico City earthquake subset. Same overall idea (classify -> flag
low-confidence predictions for human review -> Grad-CAM explain -> PDF
report), new dataset, new fusion architecture.

## What changed vs the mexico-earthquake build

| | mexico-earthquake (xBD) | this build (QQB) |
|---|---|---|
| Disaster | 2017 Mexico City earthquake | 2023 Türkiye-Syria earthquake |
| Input | single optical pre/post pair | **SAR + optical, 4 patches per building** (SAR, SAR footprint, optical, optical footprint) |
| Classes | 4 (no/minor/major/destroyed) | 2 (intact/damaged) — QQB frames it as binary, arguably anomaly detection given the imbalance |
| Model | one Siamese ResNet18 | **4-branch ResNet18 late-fusion** (SAR branch optionally SAR-HUB pretrained, not ImageNet) |
| Buildings | small — Mexico City has few actual image tiles despite dense polygons | ~4,000 buildings, single city (Islahiye) |
| Geo-aggregation | block-level, using lat/lon | **not available** — QQB ships cropped patches keyed by OSM ID only, no coordinates. `aggregate.py` does overall/top-priority stats instead. |
| Explainability | one Grad-CAM heatmap per prediction | **paired SAR + optical Grad-CAM heatmaps** — shows which modality (and where) drove each call |
| License | xView2 challenge terms | CC BY-NC-SA 4.0 — **academic/non-commercial use only** |

Trade-off going in: you lose the 4-level severity granularity and gained
a second modality, a much more recent/novel disaster, and a genuinely
harder, more publishable "dual-modality explainability" story for your
viva.

## Get the dataset & weights

1. **Dataset**: download from the QQB repo's TeraBox link —
   https://terabox.com/s/1LFynV38hkF2-xEAJwPGM8w
   Extract it into `data/qqb/`. The zip ships as `damaged/` + `intact/`
   subfolders plus `fold-1.txt`...`fold-5.txt` at the top level — **you
   don't need to merge the subfolders manually**, `dataset.py` checks
   both the flat and split layout automatically.
2. **SAR-HUB weights** (optional but recommended — see `weights/README.txt`):
   https://drive.google.com/file/d/1JgCQIXMFYbTBhGbXCb1nXlLC62Ahv9qW/view?usp=drive_link
   Save as `weights/ResNet18_TSX.pth`.
3. Source repo (for reference / benchmark numbers to compare your model
   against — see `images/table1.png` there): https://github.com/ya0-sun/PostEQ-SARopt-BuildingDamage
   Paper: Sun, Wang & Eineder, *"QuickQuakeBuildings: Post-Earthquake
   SAR-Optical Dataset for Quick Damaged-Building Detection,"* IEEE GRSL
   2024. https://ieeexplore.ieee.org/document/10542156

**License note**: QQB is CC BY-NC-SA 4.0. Fine for a college major
project; don't use it or a model trained on it commercially. Cite the
paper above if you write this up anywhere.

## Setup

```bash
pip install -r requirements.txt
```

## Run

One command, does everything (manifest -> train -> infer -> Grad-CAM
heatmaps -> aggregate -> PDF report):

```bash
python src/run_pipeline.py --data_root data/qqb --epochs 30 --mode all
```

`--mode` controls which modalities feed the model:
- `all` — SAR + optical + both footprints (the real pitch, needs SAR-HUB weights to be worth it)
- `sar` — SAR + SAR footprint only
- `opt` — optical + optical footprint only (closest to your old build, useful as an ablation baseline in your report — "look, fusion beats either modality alone")

Re-running skips training if a checkpoint already exists for that
mode+val_fold; pass `--force_train` to redo it.

Individual steps also run standalone if you want to iterate on one piece
(`python src/train.py --help`, `python src/infer.py --help`, etc.) — same
as the old build.

Training needs a GPU. See `notebooks/README.txt` for the Colab flow (SAR-HUB weights want reloading each session unless you keep the dataset on Drive).

## Outputs

- `outputs/predictions.json` — every building's probability + predicted label
- `outputs/review_queue.json` — predictions within `config.REVIEW_MARGIN` of the 0.5 boundary, flagged for manual review
- `outputs/triage_summary.json` — overall stats (damaged %, avg confidence, top-priority building IDs)
- `outputs/heatmaps/` — paired SAR + optical Grad-CAM overlays for the highest-confidence damaged buildings
- `outputs/reports/report_<timestamp>.pdf` — the disaster-response summary, embeds a sample of the heatmaps

## Architecture (for your viva)

Four independent ResNet18 encoders (SAR, SAR-footprint, optical,
optical-footprint), each reduced to a 128-d embedding, concatenated, and
passed through a small MLP head to one logit (`BCEWithLogitsLoss`, binary
damaged/intact). SAR gets SAR-HUB (TerraSAR-X-pretrained) weights since
ImageNet stats don't transfer well to radar; the other three branches use
plain ImageNet init. Class imbalance handled with a `WeightedRandomSampler`
on top of the paper's own 5-fold split.

**Grad-CAM on a 4-branch fusion model**: instead of one heatmap, we hook
one branch's `layer4` at a time and backprop the fused logit through just
that branch. That gives a separate SAR heatmap and optical heatmap per
building — genuinely more informative than a single merged map, since you
can point at both and say "here's what the radar picked up on, here's
what the photo picked up on."

## Known limitations (be upfront about these in your report/viva)

- Binary labels only, no severity levels — QQB itself frames this as
  closer to anomaly detection than fine-grained classification, given
  the damaged class is a small minority.
- No geo-aggregation — the dataset doesn't expose building coordinates,
  only OSM IDs. `aggregate.py` gives you overall/top-priority stats, not
  a map.
- Single city (Islahiye), ~4,000 buildings — good for a project, not
  enough to claim general disaster-damage detection out of the box.
- SAR-HUB weights are optional-but-important; without them the SAR
  branch starts from random init and will underperform noticeably.
