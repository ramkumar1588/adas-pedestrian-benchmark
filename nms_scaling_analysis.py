"""
stats_rq4.py  —  NMS latency vs pedestrian density  (RQ4)
────────────────────────────────────────────────────────────────────────────────
Tests H0-4: NMS latency does not increase with pedestrian count.
Rejects H0-4 if regression slope p < 0.05.

Also finds the density threshold where NMS overhead exceeds 20% of total
inference time and compares YOLOv8 total latency vs RT-DETR at each bin.

Install:  pip install pandas scipy numpy
Run:      python stats_rq4.py
────────────────────────────────────────────────────────────────────────────────
"""

import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from utils import load_config
PRED_DIR = load_config()["paths"]["output_dir"]
OUT_DIR  = PRED_DIR / "stats"
OUT_DIR.mkdir(exist_ok=True)

YOLO_MODELS   = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
RTDETR_MODELS = ["rtdetr-r50", "rtdetr-r101"]

# Density bins from synopsis
BINS        = [0, 3, 6, 11, 9999]
BIN_LABELS  = ["0-2", "3-5", "6-10", "10+"]
NMS_OVERHEAD_THRESH = 20.0   # percent


# ── LOAD LATENCY CSVs ─────────────────────────────────────────────────────────
def load_latency(model_name: str) -> pd.DataFrame | None:
    path = PRED_DIR / model_name / "latency.csv"
    if not path.exists():
        print(f"  [SKIP] {model_name}")
        return None
    df = pd.read_csv(path)
    df["model"] = model_name
    df["arch"]  = "yolo" if model_name in YOLO_MODELS else "rtdetr"
    return df

print("Loading latency data …")
frames = []
for m in YOLO_MODELS + RTDETR_MODELS:
    df = load_latency(m)
    if df is not None:
        frames.append(df)
        print(f"  {m}: {len(df):,} rows")

all_df = pd.concat(frames, ignore_index=True)

# Add density bin
all_df["density_bin"] = pd.cut(
    all_df["num_pedestrians"],
    bins=BINS, labels=BIN_LABELS, right=False
)

yolo_df   = all_df[all_df["arch"] == "yolo"].copy()
rtdetr_df = all_df[all_df["arch"] == "rtdetr"].copy()


# ── LINEAR REGRESSION: NMS latency vs pedestrian count (per YOLO model) ──────
print("\n" + "="*65)
print("  LINEAR REGRESSION: NMS (postprocess) latency vs pedestrian count")
print("  H0-4: slope = 0  (NMS latency independent of density)")
print("="*65)

reg_results = []
for model_name in YOLO_MODELS:
    sub = yolo_df[yolo_df["model"] == model_name].copy()
    if sub.empty:
        continue
    x = sub["num_pedestrians"].values
    y = sub["postprocess_ms"].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    decision = "REJECT H0-4" if p < 0.05 else "FAIL TO REJECT H0-4"
    print(f"\n  {model_name}")
    print(f"    slope     = {slope:.4f} ms/pedestrian")
    print(f"    intercept = {intercept:.4f} ms")
    print(f"    R²        = {r**2:.4f}")
    print(f"    p-value   = {p:.4f}   → {decision}")
    reg_results.append({
        "model": model_name, "slope": round(slope,4),
        "intercept": round(intercept,4), "r_squared": round(r**2,4),
        "p_value": round(p,4), "reject_H0": p < 0.05,
    })

pd.DataFrame(reg_results).to_csv(OUT_DIR/"regression_rq4.csv", index=False)


# ── NMS OVERHEAD % PER DENSITY BIN ───────────────────────────────────────────
print("\n" + "="*65)
print("  NMS OVERHEAD % BY DENSITY BIN (YOLO models)")
print("="*65)

overhead_rows = []
for model_name in YOLO_MODELS:
    sub = yolo_df[yolo_df["model"] == model_name].copy()
    if sub.empty:
        continue
    print(f"\n  {model_name}")
    print(f"  {'Bin':<8} {'N':>5} {'NMS mean':>10} {'Total mean':>12} {'NMS%':>7} {'Flag'}")
    print(f"  {'-'*8} {'-'*5} {'-'*10} {'-'*12} {'-'*7} {'-'*4}")
    for bin_label in BIN_LABELS:
        g = sub[sub["density_bin"] == bin_label]
        if g.empty:
            continue
        nms_mean  = g["postprocess_ms"].mean()
        tot_mean  = g["total_ms"].mean()
        overhead  = nms_mean / tot_mean * 100 if tot_mean > 0 else 0
        flag      = "<-- THRESHOLD" if overhead >= NMS_OVERHEAD_THRESH else ""
        print(f"  {bin_label:<8} {len(g):>5} {nms_mean:>10.3f} {tot_mean:>12.3f} "
              f"{overhead:>6.1f}% {flag}")
        overhead_rows.append({
            "model": model_name, "density_bin": bin_label,
            "n": len(g), "nms_mean_ms": round(nms_mean,3),
            "total_mean_ms": round(tot_mean,3), "nms_overhead_pct": round(overhead,2),
            "exceeds_threshold": overhead >= NMS_OVERHEAD_THRESH,
        })

pd.DataFrame(overhead_rows).to_csv(OUT_DIR/"nms_overhead_rq4.csv", index=False)


# ── YOLO vs RT-DETR TOTAL LATENCY CROSSOVER ──────────────────────────────────
print("\n" + "="*65)
print("  YOLO vs RT-DETR TOTAL LATENCY BY DENSITY BIN")
print("="*65)

crossover_rows = []
print(f"\n  {'Bin':<8}", end="")
for m in YOLO_MODELS:
    print(f" {m:>12}", end="")
for m in RTDETR_MODELS:
    print(f" {m:>14}", end="")
print()
print("  " + "-"*8 + ("-"*13)*len(YOLO_MODELS) + ("-"*15)*len(RTDETR_MODELS))

for bin_label in BIN_LABELS:
    print(f"  {bin_label:<8}", end="")
    row = {"density_bin": bin_label}
    for m in YOLO_MODELS + RTDETR_MODELS:
        sub = all_df[(all_df["model"] == m) & (all_df["density_bin"] == bin_label)]
        if sub.empty:
            print(f" {'N/A':>12}", end="")
        else:
            v = sub["total_ms"].mean()
            print(f" {v:>11.2f}ms", end="")
            row[m] = round(v,3)
    print()
    crossover_rows.append(row)

pd.DataFrame(crossover_rows).to_csv(OUT_DIR/"latency_crossover_rq4.csv", index=False)

print(f"\n  Saved → {OUT_DIR/'regression_rq4.csv'}")
print(f"  Saved → {OUT_DIR/'nms_overhead_rq4.csv'}")
print(f"  Saved → {OUT_DIR/'latency_crossover_rq4.csv'}")
