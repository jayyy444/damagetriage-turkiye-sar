"""
aggregate.py — summary statistics across all predictions.

Note: unlike the mexico-earthquake build, this dataset doesn't ship
lat/lon per building patch (QQB gives cropped .mat patches keyed by OSM
building ID only, not coordinates), so there's no geo-block aggregation
here. This produces overall + top-priority severity/confidence summary
instead — still useful for a "how bad is this area overall" report.
"""
import os
import json

import config


def aggregate(predictions):
    total = len(predictions)
    damaged = [p for p in predictions if p["predicted_label"] == "damaged"]
    review = [p for p in predictions if abs(p["probability_damaged"] - 0.5) < config.REVIEW_MARGIN]

    summary = {
        "total_buildings": total,
        "predicted_damaged": len(damaged),
        "predicted_intact": total - len(damaged),
        "damaged_pct": round(100 * len(damaged) / max(1, total), 1),
        "flagged_for_review": len(review),
        "avg_confidence": round(sum(p["confidence"] for p in predictions) / max(1, total), 4),
        "top_priority_building_ids": [
            p["building_id"] for p in
            sorted(damaged, key=lambda p: p["probability_damaged"], reverse=True)[:20]
        ],
    }
    return summary


if __name__ == "__main__":
    with open(os.path.join(config.OUTPUTS_DIR, "predictions.json")) as f:
        predictions = json.load(f)
    summary = aggregate(predictions)
    out_path = os.path.join(config.OUTPUTS_DIR, "triage_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {out_path}")
