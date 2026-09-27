"""
streamlit_app.py — DamageTriage Command Center (frontend only).

Reads only files already produced by run_pipeline.py (outputs/*.json,
outputs/heatmaps/*.png) — doesn't touch the model or run inference itself.
Run with:
    streamlit run streamlit_app.py
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
HEATMAP_DIR = os.path.join(OUTPUTS_DIR, "heatmaps")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
THEME_CSS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "theme.css")

ISLAHIYE_LAT, ISLAHIYE_LON = 36.893, 36.321

COLORS = {
    "bg": "#0a0e14",
    "surface": "#111820",
    "grid": "#1e2a3a",
    "accent": "#00d4aa",
    "alert": "#ff4d3d",
    "warn": "#f5a623",
    "ok": "#3dd68c",
    "text": "#e8edf4",
    "muted": "#7a8ba0",
}


def safe_id(building_id):
    """Building IDs from the real dataset can contain '/' (e.g.
    'damaged/1139190969') — heatmap filenames have that sanitized, so any
    lookup against outputs/heatmaps/ must apply the same replacement."""
    return building_id.replace("/", "_").replace("\\", "_")


def building_coords(building_id):
    """Deterministic scatter layout around Islahiye — QQB ships no lat/lon."""
    digest = hashlib.md5(building_id.encode()).hexdigest()
    h1, h2 = int(digest[:8], 16), int(digest[8:16], 16)
    lat = ISLAHIYE_LAT + ((h1 % 1000) / 1000 - 0.5) * 0.08
    lon = ISLAHIYE_LON + ((h2 % 1000) / 1000 - 0.5) * 0.10
    return lat, lon


def short_id(building_id):
    return building_id.split("/")[-1] if "/" in building_id else building_id


def badge_html(label):
    cls = label if label in ("damaged", "intact", "review") else "neutral"
    return f'<span class="dt-badge {cls}">{label}</span>'


@st.cache_data
def load_json(name):
    path = os.path.join(OUTPUTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_theme_css():
    if os.path.exists(THEME_CSS):
        with open(THEME_CSS, encoding="utf-8") as f:
            return f.read()
    return ""


def apply_mpl_theme():
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["muted"],
        "axes.titlecolor": COLORS["text"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "text.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.35,
        "font.family": "sans-serif",
        "font.size": 10,
    })


def render_header():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(f"""
<div class="dt-header">
  <div class="dt-header-main">
    <p class="dt-eyebrow">Satellite Intelligence · Emergency Operations</p>
    <h1 class="dt-title">DamageTriage Command Center</h1>
    <p class="dt-subtitle">
      2023 Türkiye–Syria Earthquake · Islahiye sector · SAR + Optical fusion ·
      QuickQuakeBuildings dataset
    </p>
  </div>
  <div class="dt-status-panel">
    <div class="dt-status-row"><span class="dt-pulse"></span> Pipeline data linked</div>
    <div class="dt-status-row">SYNC {now}</div>
    <div class="dt-status-row">MODE SAR+OPT FUSION</div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_kpi_grid(summary):
    st.markdown(f"""
<div class="dt-kpi-grid">
  <div class="dt-kpi">
    <p class="dt-kpi-label">Buildings scanned</p>
    <p class="dt-kpi-value">{summary["total_buildings"]:,}</p>
    <p class="dt-kpi-delta neutral">Full QQB fold coverage</p>
  </div>
  <div class="dt-kpi">
    <p class="dt-kpi-label">Predicted damaged</p>
    <p class="dt-kpi-value">{summary["predicted_damaged"]}</p>
    <p class="dt-kpi-delta alert">{summary["damaged_pct"]}% of sector</p>
  </div>
  <div class="dt-kpi">
    <p class="dt-kpi-label">Avg confidence</p>
    <p class="dt-kpi-value">{summary["avg_confidence"] * 100:.1f}%</p>
    <p class="dt-kpi-delta ok">Model certainty</p>
  </div>
  <div class="dt-kpi">
    <p class="dt-kpi-label">Flagged for review</p>
    <p class="dt-kpi-value">{summary["flagged_for_review"]}</p>
    <p class="dt-kpi-delta review">Near decision boundary</p>
  </div>
</div>
""", unsafe_allow_html=True)


def render_empty(title, text, icon="◌"):
    st.markdown(f"""
<div class="dt-empty">
  <div class="dt-empty-icon">{icon}</div>
  <p class="dt-empty-title">{title}</p>
  <p class="dt-empty-text">{text}</p>
</div>
""", unsafe_allow_html=True)


def probability_color(prob):
    if prob >= 0.7:
        return COLORS["alert"]
    if prob >= 0.4:
        return COLORS["warn"]
    return COLORS["ok"]


def plot_intelligence_map(df, highlight_id=None):
    apply_mpl_theme()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#060910")

    lats, lons = zip(*[building_coords(bid) for bid in df["building_id"]])
    colors = [probability_color(p) for p in df["probability_damaged"]]
    sizes = 12 + df["probability_damaged"].values * 80

    ax.scatter(lons, lats, c=colors, s=sizes, alpha=0.75, edgecolors="#1e2a3a", linewidths=0.4)

    if highlight_id:
        hlat, hlon = building_coords(highlight_id)
        ax.scatter([hlon], [hlat], s=220, facecolors="none", edgecolors=COLORS["accent"],
                   linewidths=2, zorder=5)

    ax.axhline(ISLAHIYE_LAT, color=COLORS["accent"], alpha=0.15, linewidth=0.8, linestyle="--")
    ax.axvline(ISLAHIYE_LON, color=COLORS["accent"], alpha=0.15, linewidth=0.8, linestyle="--")
    ax.scatter([ISLAHIYE_LON], [ISLAHIYE_LAT], marker="+", s=120, c=COLORS["accent"],
               alpha=0.6, linewidths=1.5, zorder=4)

    for i in range(5):
        r = 0.012 + i * 0.012
        circle = plt.Circle((ISLAHIYE_LON, ISLAHIYE_LAT), r, fill=False,
                              color=COLORS["accent"], alpha=0.08 - i * 0.012, linewidth=0.6)
        ax.add_patch(circle)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Disaster Intelligence Map — Islahiye Sector", fontsize=11, pad=12)
    ax.grid(True, alpha=0.2)

    legend_patches = [
        mpatches.Patch(color=COLORS["alert"], label="High damage probability"),
        mpatches.Patch(color=COLORS["warn"], label="Uncertain / review"),
        mpatches.Patch(color=COLORS["ok"], label="Low damage probability"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.85,
              facecolor=COLORS["surface"], edgecolor=COLORS["grid"], fontsize=8)

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm):
    apply_mpl_theme()
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="YlOrRd", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["intact", "damaged"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["intact", "damaged"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > cm.max() / 2 else COLORS["text"], fontsize=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.ax.yaxis.set_tick_params(color=COLORS["muted"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["muted"])
    fig.tight_layout()
    return fig


def plot_roc(fpr, tpr, roc_auc):
    apply_mpl_theme()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=COLORS["accent"], linewidth=2, label=f"AUROC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=COLORS["muted"], linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", facecolor=COLORS["surface"], edgecolor=COLORS["grid"])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_pr(rec, prec):
    apply_mpl_theme()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, color=COLORS["alert"], linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_distribution(counts, title, color):
    apply_mpl_theme()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(counts.index.astype(str), counts.values, color=color, edgecolor=COLORS["grid"], linewidth=0.6)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts.values) * 0.02,
                str(val), ha="center", va="bottom", fontsize=8, color=COLORS["muted"])
    fig.tight_layout()
    return fig


def render_satellite_image(path, label):
    st.markdown(f'<div class="dt-sat-frame"><span class="dt-sat-label">{label}</span>', unsafe_allow_html=True)
    st.image(path, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def enrich_predictions(pred_df):
    df = pred_df.copy()
    lats, lons = zip(*[building_coords(bid) for bid in df["building_id"]])
    df["lat"] = lats
    df["lon"] = lons
    df["osm_id"] = df["building_id"].apply(short_id)
    df["needs_review"] = (df["probability_damaged"] - 0.5).abs() <= 0.15
    return df


# ── Page config & theme ──
st.set_page_config(
    page_title="DamageTriage Command Center",
    layout="wide",
    page_icon="🛰",
    initial_sidebar_state="expanded",
)

st.markdown(f"<style>{load_theme_css()}</style>", unsafe_allow_html=True)

render_header()

summary = load_json("triage_summary.json")
predictions = load_json("predictions.json")
review_queue = load_json("review_queue.json")

if summary is None or predictions is None:
    st.error(
        "**Data link failed** — `outputs/triage_summary.json` or `outputs/predictions.json` "
        "not found. Run `python src/run_pipeline.py --data_root data/qqb --mode all` first."
    )
    render_empty(
        "Awaiting pipeline output",
        "The command center reads pre-computed inference results. "
        "Start the ML pipeline to populate the dashboard.",
        "⬡",
    )
    st.stop()

pred_df = enrich_predictions(pd.DataFrame(predictions))
y_true = (pred_df["true_label"] == "damaged").astype(int)
y_prob = pred_df["probability_damaged"]
y_pred = (pred_df["predicted_label"] == "damaged").astype(int)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### Mission brief")
    st.markdown(
        "**Objective:** Binary building-damage classification (damaged / intact) "
        "from paired SAR + optical satellite patches.\n\n"
        "**Architecture:** 4-branch ResNet18 late-fusion "
        "(SAR, SAR-footprint, optical, optical-footprint).\n\n"
        "**Dataset:** QuickQuakeBuildings — Islahiye, Türkiye · CC BY-NC-SA 4.0.\n\n"
        "**Explainability:** Paired Grad-CAM heatmaps per modality."
    )
    st.divider()
    st.markdown("**Sector filters**")
    global_label = st.selectbox("Predicted status", ["all", "damaged", "intact"], key="global_label")
    global_min_prob = st.slider("Min damage probability", 0.0, 1.0, 0.0, 0.05, key="global_min_prob")
    show_review_only = st.checkbox("Review queue only", value=False, key="show_review_only")
    st.divider()
    st.caption(
        "~96% of buildings are intact — a heavily imbalanced disaster dataset. "
        "AUROC and PR curves in AI Results are more meaningful than raw accuracy."
    )

filtered_df = pred_df.copy()
if global_label != "all":
    filtered_df = filtered_df[filtered_df["predicted_label"] == global_label]
filtered_df = filtered_df[filtered_df["probability_damaged"] >= global_min_prob]
if show_review_only and review_queue:
    review_ids = {r["building_id"] for r in review_queue}
    filtered_df = filtered_df[filtered_df["building_id"].isin(review_ids)]

# ── Navigation ──
tabs = st.tabs([
    "Command Center",
    "Intelligence Map",
    "Damage Analysis",
    "Satellite Compare",
    "Priority Response",
    "Satellite Imagery",
    "AI Results",
    "Reports",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Command Center
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    render_kpi_grid(summary)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Severity distribution</p></div>', unsafe_allow_html=True)
        counts = pred_df["predicted_label"].value_counts()
        st.pyplot(plot_distribution(counts, "Predicted label breakdown", COLORS["accent"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Ground truth vs model</p></div>', unsafe_allow_html=True)
        compare = pd.DataFrame({
            "true": pred_df["true_label"].value_counts(),
            "predicted": pred_df["predicted_label"].value_counts(),
        }).fillna(0)
        apply_mpl_theme()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        x = np.arange(len(compare.index))
        w = 0.35
        ax.bar(x - w / 2, compare["true"], w, label="Ground truth", color=COLORS["muted"], edgecolor=COLORS["grid"])
        ax.bar(x + w / 2, compare["predicted"], w, label="Predicted", color=COLORS["accent"], edgecolor=COLORS["grid"])
        ax.set_xticks(x)
        ax.set_xticklabels(compare.index.astype(str))
        ax.set_ylabel("Count")
        ax.legend(facecolor=COLORS["surface"], edgecolor=COLORS["grid"], fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Sector overview map</p><span class="dt-panel-meta">Synthetic layout — QQB has no coordinates</span></div>', unsafe_allow_html=True)
    st.pyplot(plot_intelligence_map(pred_df[pred_df["predicted_label"] == "damaged"].head(120)), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "High average confidence reflects the intact majority class. "
        "Use Intelligence Map and AI Results for operational metrics."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Intelligence Map
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Geospatial damage overlay</p><span class="dt-panel-meta">Islahiye · OSM-indexed buildings</span></div>', unsafe_allow_html=True)

    map_col1, map_col2 = st.columns([3, 1])
    with map_col2:
        map_filter = st.radio("Map layer", ["All buildings", "Damaged only", "Review queue"], key="map_layer")
        selected_on_map = st.selectbox(
            "Highlight building",
            [""] + filtered_df.sort_values("probability_damaged", ascending=False)["building_id"].tolist(),
            key="map_highlight",
        )

    map_df = filtered_df
    if map_filter == "Damaged only":
        map_df = map_df[map_df["predicted_label"] == "damaged"]
    elif map_filter == "Review queue" and review_queue:
        review_ids = {r["building_id"] for r in review_queue}
        map_df = map_df[map_df["building_id"].isin(review_ids)]

    highlight = selected_on_map if selected_on_map else None
    with map_col1:
        if map_df.empty:
            render_empty("No buildings match filters", "Adjust sidebar filters or map layer to display assets.", "◎")
        else:
            st.pyplot(plot_intelligence_map(map_df, highlight_id=highlight), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Interactive sector map</p></div>', unsafe_allow_html=True)
    if not map_df.empty:
        map_points = map_df[["lat", "lon"]].copy()
        st.map(map_points, zoom=11)
    else:
        render_empty("Map unavailable", "No coordinate points to render.", "◎")
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "Building positions are deterministically scattered around Islahiye for triage visualization — "
        "the QQB dataset does not ship lat/lon coordinates."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Damage Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Building damage inspector</p></div>', unsafe_allow_html=True)

    inspect_col1, inspect_col2 = st.columns([1, 2])
    with inspect_col1:
        label_filter = st.selectbox("Filter predictions", ["all", "damaged", "intact"], key="inspect_filter")
        view_df = pred_df if label_filter == "all" else pred_df[pred_df["predicted_label"] == label_filter]
        view_df = view_df.sort_values("probability_damaged", ascending=False)
        chosen = st.selectbox("Select building", view_df["building_id"].tolist(), key="inspect_building")
        st.dataframe(
            view_df[["osm_id", "predicted_label", "probability_damaged", "confidence", "true_label"]].head(50),
            use_container_width=True,
            hide_index=True,
        )

    row = pred_df[pred_df["building_id"] == chosen].iloc[0]
    prob_pct = row["probability_damaged"] * 100

    with inspect_col2:
        st.markdown('<div class="dt-inspector">', unsafe_allow_html=True)
        st.markdown(f"""
<div>
  <p style="font-family: var(--dt-mono); font-size: 0.72rem; color: var(--dt-muted); margin: 0 0 0.5rem 0;">BUILDING {short_id(chosen)}</p>
  <div class="dt-stat-row"><span class="dt-stat-key">Predicted</span><span>{badge_html(row["predicted_label"])}</span></div>
  <div class="dt-stat-row"><span class="dt-stat-key">Ground truth</span><span>{badge_html(row["true_label"])}</span></div>
  <div class="dt-stat-row"><span class="dt-stat-key">P(damaged)</span><span class="dt-stat-val">{row["probability_damaged"]:.4f}</span></div>
  <div class="dt-stat-row"><span class="dt-stat-key">Confidence</span><span class="dt-stat-val">{row["confidence"]:.4f}</span></div>
  <div class="dt-stat-row"><span class="dt-stat-key">OSM path</span><span class="dt-stat-val">{row["building_id"]}</span></div>
</div>
""", unsafe_allow_html=True)

        st.progress(min(max(row["probability_damaged"], 0.0), 1.0))
        st.caption(f"Damage probability: {prob_pct:.1f}%")

        sid = safe_id(chosen)
        sar_path = os.path.join(HEATMAP_DIR, f"{sid}_sar_heatmap.png")
        opt_path = os.path.join(HEATMAP_DIR, f"{sid}_opt_heatmap.png")
        has_sar, has_opt = os.path.exists(sar_path), os.path.exists(opt_path)

        img_col1, img_col2 = st.columns(2)
        if has_sar:
            with img_col1:
                render_satellite_image(sar_path, "SAR · Grad-CAM")
        if has_opt:
            with img_col2:
                render_satellite_image(opt_path, "Optical · Grad-CAM")
        if not has_sar and not has_opt:
            render_empty(
                "No AI heatmaps",
                "Grad-CAM overlays are pre-generated for top-priority damaged buildings only.",
                "▣",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Satellite Compare (Before/After modality)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Dual-modality satellite comparison</p><span class="dt-panel-meta">SAR vs Optical · Grad-CAM explainability</span></div>', unsafe_allow_html=True)

    heatmap_buildings = summary.get("top_priority_building_ids", [])
    if not heatmap_buildings:
        damaged_ids = pred_df[pred_df["predicted_label"] == "damaged"]["building_id"].tolist()
        heatmap_buildings = damaged_ids[:20]

    compare_id = st.selectbox("Building", heatmap_buildings, key="compare_building")
    sid = safe_id(compare_id)
    sar_path = os.path.join(HEATMAP_DIR, f"{sid}_sar_heatmap.png")
    opt_path = os.path.join(HEATMAP_DIR, f"{sid}_opt_heatmap.png")

    row = pred_df[pred_df["building_id"] == compare_id].iloc[0]
    st.markdown(
        f"**{short_id(compare_id)}** · Predicted **{row['predicted_label']}** · "
        f"P(damaged) **{row['probability_damaged']:.2f}** · "
        f"Ground truth **{row['true_label']}**"
    )

    cmp1, cmp2 = st.columns(2)
    if os.path.exists(sar_path) and os.path.exists(opt_path):
        with cmp1:
            render_satellite_image(sar_path, "SAR radar channel")
        with cmp2:
            render_satellite_image(opt_path, "Optical imagery channel")
        st.caption(
            "Side-by-side Grad-CAM shows which modality and region drove the fusion model's decision — "
            "radar penetrates cloud/debris; optical captures visible structural damage."
        )
    else:
        render_empty(
            "Comparison unavailable",
            f"No paired heatmaps for {short_id(compare_id)}. "
            "Run the pipeline or select a top-priority building.",
            "⬒",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Modality fusion schematic</p></div>', unsafe_allow_html=True)
    st.code(
        "  [SAR patch] ──► ResNet18 ──► 128-d ─┐\n"
        "  [SAR footprint] ──► ResNet18 ──► 128-d ─┤\n"
        "  [Optical patch] ──► ResNet18 ──► 128-d ─┼──► concat ──► MLP ──► P(damaged)\n"
        "  [Optical footprint] ──► ResNet18 ──► 128-d ─┘",
        language=None,
    )
    st.caption("Grad-CAM hooks layer4 per branch independently — two heatmaps, one fused decision.")
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Priority Response
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    pri_col1, pri_col2 = st.columns([1, 1])

    with pri_col1:
        st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Top priority assets</p></div>', unsafe_allow_html=True)
        top_ids = summary.get("top_priority_building_ids", [])
        if top_ids:
            for i, bid in enumerate(top_ids[:15], 1):
                row = pred_df[pred_df["building_id"] == bid]
                if row.empty:
                    continue
                r = row.iloc[0]
                st.markdown(f"""
<div class="dt-priority-item">
  <span class="dt-priority-rank">{i:02d}</span>
  <span class="dt-priority-id">{short_id(bid)}</span>
  {badge_html(r["predicted_label"])}
  <span style="font-family: var(--dt-mono); font-size: 0.72rem; color: var(--dt-alert);">{r["probability_damaged"]:.2f}</span>
</div>
""", unsafe_allow_html=True)
        else:
            render_empty("No priority list", "Pipeline did not emit top_priority_building_ids.", "▲")
        st.markdown("</div>", unsafe_allow_html=True)

    with pri_col2:
        st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Human review queue</p><span class="dt-panel-meta">Near 0.5 decision boundary</span></div>', unsafe_allow_html=True)
        if review_queue:
            rq_df = pd.DataFrame(review_queue)
            rq_df["distance_from_0.5"] = (rq_df["probability_damaged"] - 0.5).abs()
            rq_df = rq_df.sort_values("distance_from_0.5")
            st.dataframe(
                rq_df.drop(columns=["distance_from_0.5"]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"{len(review_queue)} buildings require analyst confirmation before dispatch.")
        else:
            render_empty("Review queue clear", "No uncertain predictions flagged by the pipeline.", "✓")
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Satellite Imagery Gallery
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Grad-CAM satellite gallery</p><span class="dt-panel-meta">Top-priority damaged buildings</span></div>', unsafe_allow_html=True)

    if os.path.isdir(HEATMAP_DIR):
        sar_files = sorted(f for f in os.listdir(HEATMAP_DIR) if f.endswith("_sar_heatmap.png"))
        gallery_search = st.text_input("Filter by OSM ID", "", key="gallery_search")

        if gallery_search:
            sar_files = [f for f in sar_files if gallery_search in f]

        if not sar_files:
            render_empty("No imagery found", "No heatmaps in outputs/heatmaps/ match your filter.", "▣")
        else:
            st.caption(f"{len(sar_files)} SAR/optical pairs available")
            for i in range(0, len(sar_files), 2):
                cols = st.columns(2)
                for col, fname in zip(cols, sar_files[i:i + 2]):
                    bid = fname.replace("_sar_heatmap.png", "")
                    opt_fname = fname.replace("_sar_", "_opt_")
                    opt_path = os.path.join(HEATMAP_DIR, opt_fname)
                    with col:
                        st.markdown(f"**{bid}**")
                        sub1, sub2 = st.columns(2)
                        with sub1:
                            render_satellite_image(os.path.join(HEATMAP_DIR, fname), "SAR")
                        if os.path.exists(opt_path):
                            with sub2:
                                render_satellite_image(opt_path, "OPT")
    else:
        render_empty(
            "Heatmap directory missing",
            "Run the pipeline to generate outputs/heatmaps/.",
            "▣",
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — AI Results
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Model performance · binary classifier</p></div>', unsafe_allow_html=True)

    cm = confusion_matrix(y_true, y_pred)
    ai_col1, ai_col2 = st.columns([1, 1])
    with ai_col1:
        st.pyplot(plot_confusion_matrix(cm), use_container_width=False)
    with ai_col2:
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{precision:.3f}")
        m2.metric("Recall", f"{recall:.3f}")
        m3.metric("F1", f"{f1:.3f}")
        st.caption(f"TP {tp} · FP {fp} · FN {fn} · TN {tn}")

    st.markdown("</div>", unsafe_allow_html=True)

    curve_col1, curve_col2 = st.columns(2)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    with curve_col1:
        st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">ROC curve</p></div>', unsafe_allow_html=True)
        st.pyplot(plot_roc(fpr, tpr, roc_auc), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with curve_col2:
        st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Precision–recall curve</p></div>', unsafe_allow_html=True)
        st.pyplot(plot_pr(rec, prec), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "PR curve is more informative than ROC on this imbalanced dataset — "
        "only ~4% of buildings are damaged."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Reports
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="dt-panel"><div class="dt-panel-head"><p class="dt-panel-title">Operational reports</p></div>', unsafe_allow_html=True)

    if os.path.isdir(REPORTS_DIR):
        pdf_files = sorted(
            [f for f in os.listdir(REPORTS_DIR) if f.lower().endswith(".pdf")],
            reverse=True,
        )
        if pdf_files:
            selected_pdf = st.selectbox("PDF report", pdf_files, key="report_select")
            pdf_path = os.path.join(REPORTS_DIR, selected_pdf)
            st.download_button(
                "Download report",
                data=open(pdf_path, "rb").read(),
                file_name=selected_pdf,
                mime="application/pdf",
            )
            st.caption(f"Generated report: {selected_pdf}")
        else:
            render_empty(
                "No PDF reports yet",
                "Reports are created by the pipeline at outputs/reports/report_<timestamp>.pdf",
                "📄",
            )
    else:
        render_empty(
            "Reports directory missing",
            "Run the full pipeline to generate PDF disaster-response summaries.",
            "📄",
        )

    st.divider()
    st.subheader("Export prediction data")
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions CSV",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions_export.csv",
        mime="text/csv",
    )

    st.markdown("</div>", unsafe_allow_html=True)
