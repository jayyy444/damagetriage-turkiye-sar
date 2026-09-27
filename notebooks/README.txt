Training needs a GPU (four ResNet18 branches per forward pass is heavier
than the old single-branch build — Colab's free T4 handles it fine at
batch_size=16, just expect ~2x the per-epoch time of the mexico-earthquake
build).

Steps:
1. Open a new Colab notebook, set Runtime -> T4 GPU.
2. Clone this project + install requirements.txt.
3. Download the dataset + SAR-HUB weights (see README.md "Get the dataset
   & weights") straight into Colab's disk — do this every session, Colab's
   disk doesn't persist, or mount Drive and keep the extracted dataset there.
4. Run:
     python src/run_pipeline.py --data_root /content/qqb --epochs 30
5. Download outputs/reports/*.pdf and outputs/heatmaps/ from the Colab
   file browser (or copy to Drive) before the runtime disconnects.

No changes needed to run_pipeline.py itself — just point --data_root at
wherever you extracted the dataset.
