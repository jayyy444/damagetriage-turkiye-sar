"""
PDF disaster response report — summary stats, review-queue count,
top-priority building table, and a sample of paired SAR+optical Grad-CAM
heatmaps. Uses fpdf2 (pip install fpdf2), no other deps — same approach
as the original mexico-earthquake build's report.py.
"""
import os
import json
import datetime
from fpdf import FPDF

import config


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "DamageTriage - Disaster Response Report", ln=True, align="C")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, "Event: 2023 Turkiye-Syria Earthquake - SAR + Optical", ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_report(predictions, review_queue, summary, out_path=None, heatmap_dir=None, top_n=15):
    out_path = out_path or os.path.join(
        config.OUTPUTS_DIR, "reports",
        f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pdf = Report()
    pdf.add_page()

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 6, f"Total buildings assessed: {summary['total_buildings']}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Severity Breakdown", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"  damaged        {summary['predicted_damaged']:>5}   ({summary['damaged_pct']}%)", ln=True)
    pdf.cell(0, 6, f"  intact         {summary['predicted_intact']:>5}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Needs Human Review (uncertain prediction)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0, 6,
        f"{summary['flagged_for_review']} of {summary['total_buildings']} predictions fell within "
        f"+/-{config.REVIEW_MARGIN} of the 0.5 decision boundary and are flagged for manual "
        f"review before dispatch. Full list in review_queue.json."
    )
    pdf.ln(4)

    priority = sorted([p for p in predictions if p["predicted_label"] == "damaged"],
                       key=lambda p: p["probability_damaged"], reverse=True)[:top_n]
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"Top {len(priority)} Priority Buildings (highest damage probability)", ln=True)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(90, 6, "Building ID (OSM)", border=1)
    pdf.cell(45, 6, "P(damaged)", border=1)
    pdf.cell(45, 6, "Confidence", border=1, ln=True)
    pdf.set_font("Helvetica", "", 8)
    for p in priority:
        bid = p["building_id"]
        bid_short = bid if len(bid) <= 45 else bid[:42] + "..."
        pdf.cell(90, 6, bid_short, border=1)
        pdf.cell(45, 6, f"{p['probability_damaged']:.2f}", border=1)
        pdf.cell(45, 6, f"{p['confidence']:.2f}", border=1, ln=True)

    if heatmap_dir and os.path.isdir(heatmap_dir):
        imgs = sorted(os.listdir(heatmap_dir))[:8]
        if imgs:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "Sample Damage Heatmaps (Grad-CAM, SAR + Optical)", ln=True)
            pdf.set_font("Helvetica", "", 8)
            pdf.multi_cell(0, 5, "Red = region the model's decision relied on most, per modality.")
            x0, y0 = 10, pdf.get_y() + 4
            w = 55
            for i, img_name in enumerate(imgs):
                col, row = i % 3, i // 3
                pdf.image(os.path.join(heatmap_dir, img_name),
                          x=x0 + col * (w + 5), y=y0 + row * (w + 12), w=w)

    pdf.set_y(-30)
    pdf.set_font("Helvetica", "I", 7)
    pdf.multi_cell(
        0, 4,
        "Source imagery: Capella Space (SAR) and Maxar Open Data (optical), CC BY 4.0. "
        "Building footprints/labels: OpenStreetMap + HOT, ODbL. Dataset: QuickQuakeBuildings "
        "(Sun et al., 2024), CC BY-NC-SA 4.0 - academic/non-commercial use only."
    )

    pdf.output(out_path)
    print(f"Report saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    with open(os.path.join(config.OUTPUTS_DIR, "predictions.json")) as f:
        predictions = json.load(f)
    with open(os.path.join(config.OUTPUTS_DIR, "review_queue.json")) as f:
        review_queue = json.load(f)
    with open(os.path.join(config.OUTPUTS_DIR, "triage_summary.json")) as f:
        summary = json.load(f)
    generate_report(predictions, review_queue, summary,
                     heatmap_dir=os.path.join(config.OUTPUTS_DIR, "heatmaps"))
