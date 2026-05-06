"""
pipeline.py
────────────────────────────────────────────────────────────────────────────────
ZOD inference + evaluation pipeline driven by config.yaml.

Install:
    pip install ultralytics tqdm pyyaml pycocotools

Usage:
    python pipeline.py                          # uses config.yaml
    python pipeline.py --config my_config.yaml  # custom config
    python pipeline.py --dry-run                # show what would run, no inference
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import io
import json
import contextlib
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

# ── ARG PARSING ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="ZOD inference pipeline")
    p.add_argument("--config",  default="config.yaml", help="Path to config YAML")
    p.add_argument("--dry-run", action="store_true",   help="Print plan without running")
    return p.parse_args()

# ── CONFIG LOADING ────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Resolve paths
    cfg["paths"]["subset_dir"] = Path(cfg["paths"]["subset_dir"])
    cfg["paths"]["output_dir"] = Path(cfg["paths"]["output_dir"])
    cfg["paths"]["model_dir"]  = Path(cfg["paths"]["model_dir"])
    cfg["paths"]["model_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def get_enabled_models(cfg: dict) -> list[dict]:
    """Return list of enabled model dicts, with overrides merged in."""
    overrides = cfg.get("overrides", {}) or {}
    inf       = cfg["inference"]
    enabled   = []
    for name, model_cfg in cfg["models"].items():
        if not model_cfg.get("enabled", False):
            continue
        # Merge global inference settings + per-model overrides
        settings = {**inf, **(overrides.get(name, {}) or {})}
        enabled.append({
            "name":     name,
            "weight":   model_cfg["weight"],
            "type":     model_cfg.get("type", "yolo"),
            **settings,
        })
    return enabled


def print_plan(models: list[dict], cfg: dict):
    stages = cfg["stages"]
    print("\n" + "═" * 62)
    print("  PIPELINE PLAN")
    print("═" * 62)
    print(f"  Subset dir : {cfg['paths']['subset_dir']}")
    print(f"  Output dir : {cfg['paths']['output_dir']}")
    print(f"  Model dir  : {cfg['paths']['model_dir']}  (weights cached here)")
    print(f"  Stages     : " +
          ", ".join(k for k, v in stages.items() if v))
    print(f"\n  {'Model':<14} {'Type':<8} {'Weight':<16} "
          f"{'Device':<6} {'ImgSz':<7} {'Conf':<6} {'IoU'}")
    print(f"  {'-'*14} {'-'*8} {'-'*16} {'-'*6} {'-'*7} {'-'*6} {'-'*6}")
    for m in models:
        print(f"  {m['name']:<14} {m['type']:<8} {m['weight']:<16} "
              f"{m['device']:<6} {m['imgsz']:<7} {m['conf']:<6} {m['iou']}")
    print(f"\n  Total enabled models : {len(models)}")
    print("═" * 62 + "\n")


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_metadata(subset_dir: Path) -> dict[str, dict]:
    meta = {}
    with open(subset_dir / "metadata.csv") as f:
        for row in csv.DictReader(f):
            meta[row["frame_id"]] = row
    return meta


def load_images(subset_dir: Path) -> list[Path]:
    return sorted((subset_dir / "images").glob("*.jpg"))


def resolve_weight(weight: str, model_dir: Path) -> str:
    """
    Returns the path to the model weight file.

    Logic:
      1. If model_dir/weight already exists  → use it (no download).
      2. If not → tell Ultralytics to download into model_dir, then load.

    After the first run, every subsequent run loads from model_dir instantly.
    model_dir layout:
      D:/dataset/zod/models/
        yolov8n.pt
        yolov8s.pt
        yolov8m.pt
        yolov8l.pt
        rtdetr-l.pt
        rtdetr-x.pt
    """
    cached = model_dir / weight
    if cached.exists():
        size_mb = cached.stat().st_size / (1024 ** 2)
        print(f"  [CACHED]  {weight}  ({size_mb:.1f} MB)  ->  {cached}")
        return str(cached)

    # Ultralytics downloads to CWD when given just a filename — check there first
    import shutil as _shutil
    cwd_path = Path(weight)
    if cwd_path.exists():
        _shutil.copy2(str(cwd_path), str(cached))
        size_mb = cached.stat().st_size / (1024 ** 2)
        print(f"  [CACHED]  {weight}  ({size_mb:.1f} MB)  ->  {cached}")
        return str(cached)

    print(f"  [DOWNLOAD] {weight} not found in model_dir — downloading …")
    print(f"             Destination: {model_dir}")
    # Return just the name; YOLO will download to CWD.
    # run_inference copies it to model_dir after loading.
    return weight


# ── STAGE 1: INFERENCE ────────────────────────────────────────────────────────
def run_inference(model_cfg: dict, images: list[Path],
                  meta: dict, output_dir: Path,
                  model_dir: Path, warmup_runs: int) -> dict:

    from ultralytics import YOLO
    import torch

    name   = model_cfg["name"]
    weight = model_cfg["weight"]
    device = model_cfg["device"]
    imgsz  = model_cfg["imgsz"]
    conf   = model_cfg["conf"]
    iou    = model_cfg["iou"]
    person = model_cfg["person_cls"]

    out = output_dir / name
    out.mkdir(parents=True, exist_ok=True)

    weight_path = resolve_weight(weight, model_dir)
    print(f"  Loading {weight} on {device} …")
    model = YOLO(weight_path)
    model.to(device)

    # Copy weight to model_dir if YOLO just downloaded it to CWD
    dst = model_dir / weight
    if not dst.exists():
        import shutil as _shutil
        cwd_file = Path(weight)
        if cwd_file.exists():
            _shutil.copy2(str(cwd_file), str(dst))

    # Warmup
    if warmup_runs > 0 and images:
        print(f"  Warming up ({warmup_runs} pass{'es' if warmup_runs > 1 else ''}) …")
        for _ in range(warmup_runs):
            model(str(images[0]), verbose=False, device=device, imgsz=imgsz)
        if device == "cuda":
            torch.cuda.synchronize()

    coco_preds   = []
    latency_rows = []

    for img_path in tqdm(images, desc=f"  {name}", unit="img", ncols=72):
        fid      = img_path.stem
        image_id = int(fid)
        row      = meta.get(fid, {})

        results = model(
            str(img_path),
            conf=conf, iou=iou, imgsz=imgsz,
            verbose=False, device=device,
        )
        result = results[0]
        speed  = result.speed

        pre_ms  = speed.get("preprocess",  0.0)
        inf_ms  = speed.get("inference",   0.0)
        post_ms = speed.get("postprocess", 0.0)
        tot_ms  = pre_ms + inf_ms + post_ms

        n_preds = 0
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls.item()) != person:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                coco_preds.append({
                    "image_id":    image_id,
                    "category_id": 1,
                    "bbox":        [round(x1,2), round(y1,2),
                                    round(x2-x1,2), round(y2-y1,2)],
                    "score":       round(float(box.conf.item()), 4),
                })
                n_preds += 1

        latency_rows.append({
            "frame_id":        fid,
            "image_id":        image_id,
            "preprocess_ms":   round(pre_ms,  3),
            "inference_ms":    round(inf_ms,  3),
            "postprocess_ms":  round(post_ms, 3),
            "total_ms":        round(tot_ms,  3),
            "num_preds":       n_preds,
            "num_pedestrians": int(row.get("num_pedestrians", 0)),
            "time_of_day":     row.get("time_of_day", ""),
            "scraped_weather": row.get("scraped_weather", ""),
            "road_type":       row.get("road_type", ""),
            "time_bin":        row.get("time_bin", ""),
            "weather_bin":     row.get("weather_bin", ""),
            "road_bin":        row.get("road_bin", ""),
            "cell":            row.get("cell", ""),
        })

    # Save predictions_coco.json
    pred_json = out / "predictions_coco.json"
    with open(pred_json, "w") as f:
        json.dump(coco_preds, f)

    # Save latency.csv
    lat_fields = [
        "frame_id","image_id",
        "preprocess_ms","inference_ms","postprocess_ms","total_ms",
        "num_preds","num_pedestrians",
        "time_of_day","scraped_weather","road_type",
        "time_bin","weather_bin","road_bin","cell",
    ]
    lat_csv = out / "latency.csv"
    with open(lat_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lat_fields)
        w.writeheader()
        w.writerows(latency_rows)

    # Build summary
    n          = len(latency_rows)
    tot_times  = [r["total_ms"]       for r in latency_rows]
    inf_times  = [r["inference_ms"]   for r in latency_rows]
    post_times = [r["postprocess_ms"] for r in latency_rows]

    def _stats(vals):
        s = sorted(vals)
        return {"mean":   round(sum(s)/len(s), 3),
                "median": round(s[len(s)//2], 3),
                "p95":    round(s[int(len(s)*0.95)], 3),
                "min":    round(s[0], 3),
                "max":    round(s[-1], 3)}

    total_dets = len(coco_preds)
    summary = {
        "model":              name,
        "weight":             weight,
        "type":               model_cfg["type"],
        "num_images":         n,
        "conf":               conf,
        "iou":                iou,
        "imgsz":              imgsz,
        "device":             device,
        "total_detections":   total_dets,
        "avg_dets_per_image": round(total_dets / max(n, 1), 2),
        "latency_ms": {
            "inference":   _stats(inf_times),
            "postprocess": _stats(post_times),
            "total":       _stats(tot_times),
        },
        "nms_overhead_pct": round(
            sum(post_times) / max(sum(tot_times), 1e-9) * 100, 2),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"    Detections   : {total_dets:,}  ({summary['avg_dets_per_image']} / image)")
    print(f"    Latency mean : {summary['latency_ms']['total']['mean']} ms/image  "
          f"(inf={summary['latency_ms']['inference']['mean']} ms  "
          f"post={summary['latency_ms']['postprocess']['mean']} ms)")
    print(f"    NMS overhead : {summary['nms_overhead_pct']}%")
    print(f"    Saved to     : {out}")

    return summary


# ── STAGE 2: EVALUATE ─────────────────────────────────────────────────────────
STRATA = {
    "time_bin":    {"day": "Day", "night": "Night/Twilight"},
    "weather_bin": {"good": "Clear/Cloudy", "bad": "Rain/Snow/Fog"},
    "road_bin":    {"highway": "Highway",   "other": "City/Rural"},
}

def _coco_eval(coco_gt, pred_path: Path, image_ids=None) -> dict:
    from pycocotools.coco     import COCO
    from pycocotools.cocoeval import COCOeval

    with open(pred_path) as f:
        preds = json.load(f)

    if not preds:
        return {"map50": 0.0, "map50_95": 0.0}

    if image_ids is not None:
        id_set = set(image_ids)
        preds  = [p for p in preds if p["image_id"] in id_set]

    if not preds:
        return {"map50": 0.0, "map50_95": 0.0}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dt   = coco_gt.loadRes(preds)
        ev   = COCOeval(coco_gt, dt, "bbox")
        if image_ids:
            ev.params.imgIds = image_ids
        ev.params.catIds = [1]
        ev.evaluate(); ev.accumulate(); ev.summarize()

    return {"map50_95": round(float(ev.stats[0]), 4),
            "map50":    round(float(ev.stats[1]), 4)}


def run_evaluation(model_name: str, output_dir: Path,
                   subset_dir: Path, meta: dict) -> dict | None:
    from pycocotools.coco import COCO

    pred_path = output_dir / model_name / "predictions_coco.json"
    if not pred_path.exists():
        print(f"  [SKIP] {model_name}: no predictions file found")
        return None

    gt_path = subset_dir / "coco" / "annotations" / "instances.json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_gt = COCO(str(gt_path))

    print(f"  Evaluating {model_name} …", end="", flush=True)
    overall = _coco_eval(coco_gt, pred_path)
    print(f"  mAP@0.5={overall['map50']:.4f}  mAP@0.5:0.95={overall['map50_95']:.4f}")

    result = {"model": model_name, "overall": overall, "stratified": {}}

    for col, groups in STRATA.items():
        result["stratified"][col] = {}
        for bin_val, label in groups.items():
            ids = [int(fid) for fid, row in meta.items()
                   if row.get(col) == bin_val]
            r   = _coco_eval(coco_gt, pred_path, image_ids=ids)
            result["stratified"][col][bin_val] = {**r, "label": label,
                                                  "n_images": len(ids)}

    with open(output_dir / model_name / "map_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── STAGE 3: REPORT ───────────────────────────────────────────────────────────
def print_report(inf_summaries: list[dict], eval_results: list[dict],
                 output_dir: Path):

    # ── Latency table
    print(f"\n{'═'*72}")
    print("  LATENCY SUMMARY (ms/image, mean)")
    print(f"{'═'*72}")
    print(f"  {'Model':<14} {'Type':<8} {'Inference':>10} "
          f"{'Postprocess':>12} {'Total':>8} {'NMS%':>7} {'Dets/img':>9}")
    print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*12} {'-'*8} {'-'*7} {'-'*9}")
    for s in inf_summaries:
        print(f"  {s['model']:<14} {s['type']:<8} "
              f"{s['latency_ms']['inference']['mean']:>10.2f} "
              f"{s['latency_ms']['postprocess']['mean']:>12.2f} "
              f"{s['latency_ms']['total']['mean']:>8.2f} "
              f"{s['nms_overhead_pct']:>6.1f}% "
              f"{s['avg_dets_per_image']:>9.2f}")

    # ── mAP table
    if eval_results:
        cols = ["Overall","Day","Night","Clear","Rain","Highway","City"]
        print(f"\n{'═'*72}")
        print("  mAP@0.5 — Stratified by Condition (RQ1)")
        print(f"{'═'*72}")
        print(f"  {'Model':<14} " + " ".join(f"{c:>9}" for c in cols))
        print(f"  {'-'*14} " + " ".join(f"{'-'*9}" for _ in cols))
        csv_rows = []
        for r in eval_results:
            s = r["stratified"]
            vals = [
                r["overall"]["map50"],
                s.get("time_bin",    {}).get("day",     {}).get("map50", 0),
                s.get("time_bin",    {}).get("night",   {}).get("map50", 0),
                s.get("weather_bin", {}).get("good",    {}).get("map50", 0),
                s.get("weather_bin", {}).get("bad",     {}).get("map50", 0),
                s.get("road_bin",    {}).get("highway", {}).get("map50", 0),
                s.get("road_bin",    {}).get("other",   {}).get("map50", 0),
            ]
            print(f"  {r['model']:<14} " +
                  " ".join(f"{v:>9.4f}" for v in vals))
            csv_rows.append([r["model"]] + vals)

        # Save CSV
        csv_path = output_dir / "map_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model"] + cols)
            w.writerows(csv_rows)
        print(f"\n  Saved -> {csv_path}")

    # Save master summary JSON
    master = output_dir / "pipeline_results.json"
    with open(master, "w") as f:
        json.dump({"inference": inf_summaries, "evaluation": eval_results}, f, indent=2)
    print(f"  Saved -> {master}")
    print(f"{'═'*72}\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if not Path(args.config).exists():
        print(f"[ERR] Config not found: {args.config}")
        sys.exit(1)

    cfg     = load_config(args.config)
    models  = get_enabled_models(cfg)
    stages  = cfg["stages"]
    paths   = cfg["paths"]
    warmup  = cfg.get("warmup", {})

    if not models:
        print("[ERR] No models enabled in config.yaml. Set enabled: true for at least one model.")
        sys.exit(1)

    print_plan(models, cfg)

    if args.dry_run:
        print("  [DRY RUN] Exiting without running inference.\n")
        return

    images = load_images(paths["subset_dir"])
    meta   = load_metadata(paths["subset_dir"])
    print(f"  {len(images):,} images  |  {len(meta):,} metadata rows\n")

    inf_summaries = []
    eval_results  = []

    for model_cfg in models:
        name = model_cfg["name"]

        # ── Inference
        if stages.get("inference", True):
            print(f"\n{'─'*62}")
            print(f"  INFERENCE: {name}")
            print(f"{'─'*62}")
            t0      = time.perf_counter()
            summary = run_inference(
                model_cfg, images, meta,
                paths["output_dir"],
                model_dir=paths["model_dir"],
                warmup_runs=warmup.get("runs", 3) if warmup.get("enabled", True) else 0,
            )
            elapsed = (time.perf_counter() - t0) / 60
            print(f"    Done in {elapsed:.1f} min")
            inf_summaries.append(summary)

        # ── Evaluate
        if stages.get("evaluate", True):
            print(f"\n  EVALUATION: {name}")
            r = run_evaluation(name, paths["output_dir"],
                               paths["subset_dir"], meta)
            if r:
                eval_results.append(r)

    # ── Report
    if stages.get("report", True) and (inf_summaries or eval_results):
        print_report(inf_summaries, eval_results, paths["output_dir"])


if __name__ == "__main__":
    main()
