"""
evaluate_map.py
────────────────────────────────────────────────────────────────────────────────
Compute mAP@0.5 and mAP@0.5:0.95 for all models using pycocotools.
Stratifies results by time_of_day, weather, and road_type (RQ1).

Install:
    pip install pycocotools

Usage:
    python evaluate_map.py
    python evaluate_map.py --model yolov8n
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import json
from pathlib import Path

from pycocotools.coco     import COCO
from pycocotools.cocoeval import COCOeval

from utils import load_config
_cfg        = load_config()
SUBSET_DIR  = _cfg["paths"]["subset_dir"]
COCO_GT     = SUBSET_DIR / "coco" / "annotations" / "instances.json"
PRED_DIR    = _cfg["paths"]["output_dir"]
METADATA_CSV= SUBSET_DIR / "metadata.csv"

ALL_MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "rtdetr-r50", "rtdetr-r101"]

# Stratification groups (must match values in metadata.csv)
STRATA = {
    "time_bin":    {"day": "Day", "night": "Night/Twilight"},
    "weather_bin": {"good": "Clear/Cloudy", "bad": "Rain/Snow/Fog"},
    "road_bin":    {"highway": "Highway",  "other": "City/Rural"},
}


def load_metadata() -> dict[int, dict]:
    """Returns {image_id (int): row} mapping."""
    meta = {}
    with open(METADATA_CSV) as f:
        for row in csv.DictReader(f):
            image_id = int(row["frame_id"])
            meta[image_id] = row
    return meta


def evaluate_subset(coco_gt: COCO, pred_path: Path,
                    image_ids: list[int] | None = None) -> dict:
    """
    Run COCOeval on a (optionally filtered) set of image_ids.
    Returns dict with map50 and map50_95.
    """
    with open(pred_path) as f:
        raw_preds = json.load(f)

    if not raw_preds:
        return {"map50": 0.0, "map50_95": 0.0, "n_images": 0}

    # Filter predictions to the requested image ids
    if image_ids is not None:
        id_set    = set(image_ids)
        raw_preds = [p for p in raw_preds if p["image_id"] in id_set]

    if not raw_preds:
        return {"map50": 0.0, "map50_95": 0.0, "n_images": 0}

    coco_dt  = coco_gt.loadRes(raw_preds)
    eval_ids = image_ids if image_ids is not None else None

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    if eval_ids:
        coco_eval.params.imgIds = eval_ids
    coco_eval.params.catIds = [1]   # category 1 = pedestrian

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # stats[0] = mAP@0.5:0.95,  stats[1] = mAP@0.5
    return {
        "map50_95": round(float(coco_eval.stats[0]), 4),
        "map50":    round(float(coco_eval.stats[1]), 4),
        "n_images": len(set(p["image_id"] for p in raw_preds)),
    }


def run_evaluation(model_name: str, meta: dict[int, dict]):
    pred_path = PRED_DIR / model_name / "predictions_coco.json"
    if not pred_path.exists():
        print(f"  [SKIP] {model_name} — no predictions_coco.json found")
        return None

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")

    # Load ground truth (suppress pycocotools stdout)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_gt = COCO(str(COCO_GT))

    # All images — overall mAP
    print("\n  [Overall]")
    overall = evaluate_subset(coco_gt, pred_path)

    results = {"model": model_name, "overall": overall, "stratified": {}}

    # Stratified evaluation (RQ1)
    for col, groups in STRATA.items():
        results["stratified"][col] = {}
        for bin_val, label in groups.items():
            ids = [iid for iid, row in meta.items() if row.get(col) == bin_val]
            print(f"\n  [{label}]  ({len(ids)} images)")
            with contextlib.redirect_stdout(buf):
                r = evaluate_subset(coco_gt, pred_path, image_ids=ids)
            results["stratified"][col][bin_val] = {**r, "label": label}
            print(f"    mAP@0.5      = {r['map50']:.4f}")
            print(f"    mAP@0.5:0.95 = {r['map50_95']:.4f}")

    # Save per-model results
    out = PRED_DIR / model_name / "map_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [OK] Saved -> {out}")
    return results


def print_summary_table(all_results: list[dict]):
    if not all_results:
        return
    print(f"\n{'='*80}")
    print("  SUMMARY: mAP@0.5 by Model and Condition (RQ1)")
    print(f"{'='*80}")

    # Header
    cols = ["Overall", "Day", "Night", "Clear", "Rain", "Highway", "City"]
    print(f"  {'Model':<14} " + " ".join(f"{c:>9}" for c in cols))
    print(f"  {'-'*14} " + " ".join(f"{'-'*9}" for _ in cols))

    for r in all_results:
        s  = r["stratified"]
        row_vals = [
            r["overall"]["map50"],
            s.get("time_bin",    {}).get("day",     {}).get("map50", 0),
            s.get("time_bin",    {}).get("night",   {}).get("map50", 0),
            s.get("weather_bin", {}).get("good",    {}).get("map50", 0),
            s.get("weather_bin", {}).get("bad",     {}).get("map50", 0),
            s.get("road_bin",    {}).get("highway", {}).get("map50", 0),
            s.get("road_bin",    {}).get("other",   {}).get("map50", 0),
        ]
        print(f"  {r['model']:<14} " +
              " ".join(f"{v:>9.4f}" for v in row_vals))

    print(f"{'='*80}\n")

    # Save master CSV
    master_csv = PRED_DIR / "map_summary.csv"
    with open(master_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + cols)
        for r in all_results:
            s = r["stratified"]
            w.writerow([
                r["model"],
                r["overall"]["map50"],
                s.get("time_bin",    {}).get("day",     {}).get("map50", 0),
                s.get("time_bin",    {}).get("night",   {}).get("map50", 0),
                s.get("weather_bin", {}).get("good",    {}).get("map50", 0),
                s.get("weather_bin", {}).get("bad",     {}).get("map50", 0),
                s.get("road_bin",    {}).get("highway", {}).get("map50", 0),
                s.get("road_bin",    {}).get("other",   {}).get("map50", 0),
            ])
    print(f"  Saved summary CSV -> {master_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=ALL_MODELS)
    return p.parse_args()


def main():
    args = parse_args()
    meta = load_metadata()

    all_results = []
    for name in args.models:
        r = run_evaluation(name, meta)
        if r:
            all_results.append(r)

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
