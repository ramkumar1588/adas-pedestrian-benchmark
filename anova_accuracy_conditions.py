"""
stats_rq1.py  —  Two-way ANOVA for RQ1
────────────────────────────────────────────────────────────────────────────────
DV  : per-image recall@0.5 (did the model detect every GT pedestrian?)
IV1 : architecture  —  yolo | rtdetr
IV2 : condition     —  run separately for weather / time_of_day / road_type

Tests H0-1: no interaction effect between architecture and condition on recall.
Rejects H0-1 if interaction p < 0.05.

Install:  pip install pandas statsmodels scipy
Run:      python stats_rq1.py
────────────────────────────────────────────────────────────────────────────────
"""

import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

from utils import load_config
_cfg        = load_config()
SUBSET_DIR  = _cfg["paths"]["subset_dir"]
PRED_DIR    = _cfg["paths"]["output_dir"]
GT_PATH     = SUBSET_DIR / "coco" / "annotations" / "instances.json"
META_CSV    = SUBSET_DIR / "metadata.csv"
OUT_DIR     = PRED_DIR / "stats"
OUT_DIR.mkdir(exist_ok=True)

YOLO_MODELS   = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
RTDETR_MODELS = ["rtdetr-r50", "rtdetr-r101"]
ALL_MODELS    = YOLO_MODELS + RTDETR_MODELS
IOU_THRESH    = 0.5

# Condition columns and their levels
CONDITIONS = {
    "weather_bin": {"good": "Clear/Cloudy", "bad": "Rain/Snow/Fog"},
    "time_bin":    {"day":  "Day",           "night": "Night/Twilight"},
    "road_bin":    {"highway": "Highway",    "other": "City/Rural"},
}


# ── HELPERS ────────────────────────────────────────────────────────────────────
def iou_xywh(b1, b2):
    """IoU between two [x,y,w,h] boxes."""
    ax1, ay1 = b1[0], b1[1]
    ax2, ay2 = b1[0]+b1[2], b1[1]+b1[3]
    bx1, by1 = b2[0], b2[1]
    bx2, by2 = b2[0]+b2[2], b2[1]+b2[3]
    ix = max(0, min(ax2,bx2) - max(ax1,bx1))
    iy = max(0, min(ay2,by2) - max(ay1,by1))
    inter = ix * iy
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter/union if union > 0 else 0.0


def per_image_recall(gt_by_img, preds_by_img, iou_thresh=IOU_THRESH):
    """Return {image_id: recall} for images that have ≥1 GT pedestrian."""
    out = {}
    for img_id, gts in gt_by_img.items():
        if not gts:
            continue
        preds = sorted(preds_by_img.get(img_id, []), key=lambda p: -p["score"])
        matched = set()
        for pred in preds:
            for j, gt in enumerate(gts):
                if j in matched:
                    continue
                if iou_xywh(pred["bbox"], gt) >= iou_thresh:
                    matched.add(j)
                    break
        out[img_id] = len(matched) / len(gts)
    return out


# ── LOAD GROUND TRUTH ─────────────────────────────────────────────────────────
print("Loading ground truth …")
with open(GT_PATH) as f:
    gt_json = json.load(f)

gt_by_img = {img["id"]: [] for img in gt_json["images"]}
for ann in gt_json["annotations"]:
    gt_by_img[ann["image_id"]].append(ann["bbox"])

# ── LOAD METADATA ─────────────────────────────────────────────────────────────
meta = {}
with open(META_CSV) as f:
    for row in csv.DictReader(f):
        meta[int(row["frame_id"])] = row

# ── BUILD OBSERVATIONS DATAFRAME ──────────────────────────────────────────────
print("Computing per-image recall for all models …")
records = []

for model_name in ALL_MODELS:
    pred_path = PRED_DIR / model_name / "predictions_coco.json"
    if not pred_path.exists():
        print(f"  [SKIP] {model_name} — no predictions found")
        continue

    with open(pred_path) as f:
        preds = json.load(f)

    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    recall_map = per_image_recall(gt_by_img, preds_by_img)
    arch       = "yolo" if model_name in YOLO_MODELS else "rtdetr"

    for img_id, recall in recall_map.items():
        m = meta.get(img_id, {})
        records.append({
            "model":       model_name,
            "arch":        arch,
            "image_id":    img_id,
            "recall":      recall,
            "weather_bin": m.get("weather_bin", ""),
            "time_bin":    m.get("time_bin",    ""),
            "road_bin":    m.get("road_bin",    ""),
        })

df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "per_image_recall.csv", index=False)
print(f"  → {len(df):,} observations  "
      f"({df['model'].nunique()} models × {df['image_id'].nunique()} images)")


# ── TWO-WAY ANOVA per condition ────────────────────────────────────────────────
print("\n" + "="*65)
print("  TWO-WAY ANOVA RESULTS  (DV = per-image recall@0.5)")
print("="*65)

anova_results = []

for cond_col, levels in CONDITIONS.items():
    cond_label = cond_col.replace("_bin", "").replace("_", " ").title()
    sub = df[df[cond_col].isin(levels.keys())].copy()
    sub["arch"]  = sub["arch"].astype("category")
    sub["cond"]  = sub[cond_col].astype("category")

    formula = "recall ~ C(arch) + C(cond) + C(arch):C(cond)"
    lm      = ols(formula, data=sub).fit()
    anova   = sm.stats.anova_lm(lm, typ=2)

    p_main_arch  = anova.loc["C(arch)",       "PR(>F)"]
    p_main_cond  = anova.loc["C(cond)",       "PR(>F)"]
    p_interaction= anova.loc["C(arch):C(cond)","PR(>F)"]

    f_arch  = anova.loc["C(arch)",        "F"]
    f_cond  = anova.loc["C(cond)",        "F"]
    f_inter = anova.loc["C(arch):C(cond)","F"]

    # Eta-squared (partial): SS_effect / (SS_effect + SS_residual)
    ss_inter = anova.loc["C(arch):C(cond)", "sum_sq"]
    ss_res   = anova.loc["Residual",        "sum_sq"]
    eta2     = ss_inter / (ss_inter + ss_res)

    decision = "REJECT H0-1" if p_interaction < 0.05 else "FAIL TO REJECT H0-1"

    print(f"\n  Condition: {cond_label}")
    print(f"  {'Effect':<25} {'F':>8} {'p':>10} {'Sig':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*6}")
    print(f"  {'Architecture (arch)':<25} {f_arch:>8.3f} {p_main_arch:>10.4f} "
          f"{'*' if p_main_arch < 0.05 else ' ':>6}")
    print(f"  {cond_label:<25} {f_cond:>8.3f} {p_main_cond:>10.4f} "
          f"{'*' if p_main_cond < 0.05 else ' ':>6}")
    print(f"  {'Interaction':<25} {f_inter:>8.3f} {p_interaction:>10.4f} "
          f"{'*' if p_interaction < 0.05 else ' ':>6}")
    print(f"  Partial η² (interaction) = {eta2:.4f}")
    print(f"  Decision: {decision}  (α = 0.05)")

    anova_results.append({
        "condition":        cond_label,
        "F_arch":           round(f_arch,  4),
        "p_arch":           round(p_main_arch,   4),
        "F_cond":           round(f_cond,  4),
        "p_cond":           round(p_main_cond,   4),
        "F_interaction":    round(f_inter, 4),
        "p_interaction":    round(p_interaction, 4),
        "partial_eta2":     round(eta2,    4),
        "reject_H0":        p_interaction < 0.05,
    })

    # Marginal means
    print(f"\n  Marginal recall means (arch × {cond_label}):")
    pivot = sub.groupby(["arch", cond_col])["recall"].mean().unstack()
    print(pivot.round(4).to_string())

# ── SAVE ──────────────────────────────────────────────────────────────────────
pd.DataFrame(anova_results).to_csv(OUT_DIR / "anova_rq1.csv", index=False)
print(f"\n  Saved → {OUT_DIR / 'anova_rq1.csv'}")
print(f"  Saved → {OUT_DIR / 'per_image_recall.csv'}")
print("\n  * p < 0.05")
