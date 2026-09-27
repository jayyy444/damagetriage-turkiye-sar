"""
Central config for the SAR+Optical build (2023 Turkiye-Syria earthquake,
QuickQuakeBuildings dataset). Change paths/thresholds here — nothing else
in the codebase hardcodes these values.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data", "qqb")   # extracted QQB .mat files + fold-*.txt live here
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")

CLASSES = ["intact", "damaged"]          # binary — matches QQB's OSM damage labels
IMG_SIZE = 224

# predictions whose P(damaged) falls within this margin of 0.5 get routed
# to the review queue instead of being trusted outright
REVIEW_MARGIN = 0.15                     # i.e. prob in [0.35, 0.65] -> flagged

# fold used for validation/inference reporting; the other 4 folds train
VAL_FOLD = "fold-1.txt"
ALL_FOLDS = ["fold-1.txt", "fold-2.txt", "fold-3.txt", "fold-4.txt", "fold-5.txt"]

MODE = "all"   # "all" (sar+sarftp+opt+optftp fusion) | "sar" | "opt"

# optional pretrained weights — see README "Get the dataset & weights"
SAR_PRETRAIN = os.path.join(ROOT, "weights", "ResNet18_TSX.pth")   # SAR-HUB TerraSAR-X weights
OPT_PRETRAIN = "imagenet"
