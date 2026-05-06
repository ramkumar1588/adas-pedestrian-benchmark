"""
benchmark_tensorrt.py  —  RQ2: Benchmark TensorRT engines + paired t-test
────────────────────────────────────────────────────────────────────────────────
For each model and each precision (FP32 / FP16 / INT8):
  1. Run inference on the 1250 subset images using the TRT engine
  2. Save predictions_coco.json and latency.csv
  3. Compute mAP via pycocotools
  4. Compute accuracy retention ratio = mAP_prec / mAP_FP32

Then runs paired t-tests on retention ratio (FP16 and INT8) between
YOLOv8 and RT-DETR to test H0-2.

Install:  pip install ultralytics pycocotools pandas scipy
Run:      python benchmark_tensorrt.py
────────────────────────────────────────────────────────────────────────────────
"""

import csv
import io
import json
import contextlib
import time
from pathlib import Path

import pandas as pd
from scipy import stats
from tqdm import tqdm
from ultralytics import YOLO, RTDETR

from utils import load_config
_cfg         = load_config()
SUBSET_DIR   = _cfg["paths"]["subset_dir"]
PRED_DIR     = _cfg["paths"]["output_dir"]
MODEL_DIR    = _cfg["paths"]["model_dir"]
GT_PATH      = SUBSET_DIR / "coco" / "annotations" / "instances.json"
META_CSV     = SUBSET_DIR / "metadata.csv"
ENGINE_MANIFEST = MODEL_DIR / "trt_engines.json"
OUT_DIR      = PRED_DIR / "stats"
OUT_DIR.mkdir(exist_ok=True)

PERSON_CLASS = 0
PRECISIONS   = ["fp32", "fp16", "int8"]
YOLO_MODELS  = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
RTDETR_MODELS= ["rtdetr-r50", "rtdetr-r101"]


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_images():
    return sorted((SUBSET_DIR / "images").glob("*.jpg"))

def load_metadata():
    meta = {}
    with open(META_CSV) as f:
        for row in csv.DictReader(f):
            meta[row["frame_id"]] = row
    return meta

def coco_map(gt_path, pred_path, image_ids=None):
    from pycocotools.coco     import COCO
    from pycocotools.cocoeval import COCOeval
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_gt = COCO(str(gt_path))
    with open(pred_path) as f:
        preds = json.load(f)
    if not preds:
        return 0.0, 0.0
    if image_ids:
        preds = [p for p in preds if p["image_id"] in set(image_ids)]
    if not preds:
        return 0.0, 0.0
    with contextlib.redirect_stdout(buf):
        dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, dt, "bbox")
        if image_ids:
            ev.params.imgIds = image_ids
        ev.params.catIds = [1]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return round(float(ev.stats[1]), 4), round(float(ev.stats[0]), 4)


# ── INFERENCE WITH TRT ENGINE ─────────────────────────────────────────────────
def run_engine(model_name, prec_name, engine_path, images, meta, out_dir, arch="yolo"):
    """Run inference with one TRT engine. Returns {map50, map50_95, latencies}."""
    engine_out = out_dir / model_name / f"trt_{prec_name}"
    engine_out.mkdir(parents=True, exist_ok=True)

    pred_json = engine_out / "predictions_coco.json"
    lat_csv   = engine_out / "latency.csv"

    if pred_json.exists() and lat_csv.exists():
        print(f"  [CACHED] {model_name} {prec_name.upper()}")
    else:
        print(f"  Running {model_name} {prec_name.upper()} …")
        model = RTDETR(str(engine_path)) if arch == "rtdetr" else YOLO(str(engine_path))

        # Warmup: 5 passes to let CUDA/TRT reach steady-state before timing
        print(f"  Warming up {model_name} {prec_name.upper()} (5 passes) …")
        for _ in range(5):
            model(str(images[0]), verbose=False)

        coco_preds = []
        lat_rows   = []

        for img_path in tqdm(images, desc=f"  {model_name}/{prec_name}",
                             unit="img", ncols=70):
            fid      = img_path.stem
            image_id = int(fid)
            row_meta = meta.get(fid, {})

            results = model(str(img_path), verbose=False)
            result  = results[0]
            speed   = result.speed

            n_preds = 0
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls.item()) != PERSON_CLASS:
                        continue
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    coco_preds.append({
                        "image_id":    image_id,
                        "category_id": 1,
                        "bbox":        [round(x1,2),round(y1,2),
                                        round(x2-x1,2),round(y2-y1,2)],
                        "score":       round(float(box.conf.item()),4),
                    })
                    n_preds += 1

            lat_rows.append({
                "frame_id":       fid,
                "image_id":       image_id,
                "precision":      prec_name,
                "preprocess_ms":  round(speed.get("preprocess",0),3),
                "inference_ms":   round(speed.get("inference",0),3),
                "postprocess_ms": round(speed.get("postprocess",0),3),
                "total_ms":       round(sum(speed.values()),3),
                "num_preds":      n_preds,
                "num_pedestrians":int(row_meta.get("num_pedestrians",0)),
            })

        with open(pred_json,"w") as f:
            json.dump(coco_preds, f)
        fields = ["frame_id","image_id","precision",
                  "preprocess_ms","inference_ms","postprocess_ms","total_ms",
                  "num_preds","num_pedestrians"]
        with open(lat_csv,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(lat_rows)

    # Compute mAP
    map50, map50_95 = coco_map(GT_PATH, pred_json)

    # Latency stats
    lat_df = pd.read_csv(lat_csv)
    mean_total = lat_df["total_ms"].mean()

    return {"map50": map50, "map50_95": map50_95,
            "mean_latency_ms": round(mean_total,3)}


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if not ENGINE_MANIFEST.exists():
        print(f"[ERR] Engine manifest not found: {ENGINE_MANIFEST}")
        print("      Run tensorrt_export.py first.")
        return

    with open(ENGINE_MANIFEST) as f:
        engines = json.load(f)

    images = load_images()
    meta   = load_metadata()

    print(f"\nBenchmarking TRT engines on {len(images):,} images …\n")

    results = []   # {model, arch, precision, map50, map50_95, mean_latency_ms}

    for model_name, prec_map in engines.items():
        arch = "yolo" if model_name in YOLO_MODELS else "rtdetr"
        for prec_name in PRECISIONS:
            raw_path = prec_map.get(prec_name, "")
            if not raw_path:
                print(f"  [SKIP] {model_name} {prec_name.upper()} — engine not found")
                continue
            engine_path = Path(raw_path)
            if not engine_path.exists():
                print(f"  [SKIP] {model_name} {prec_name.upper()} — engine not found")
                continue
            r = run_engine(model_name, prec_name, engine_path,
                           images, meta, PRED_DIR, arch=arch)
            results.append({"model": model_name, "arch": arch,
                             "precision": prec_name, **r})

    if not results:
        print("[ERR] No benchmark results. Check that engines exist.")
        return

    df = pd.DataFrame(results)

    # ── Accuracy retention ratio ──────────────────────────────────────────────
    fp32 = df[df["precision"]=="fp32"][["model","map50"]].rename(columns={"map50":"map50_fp32"})
    df   = df.merge(fp32, on="model", how="left")
    df["retention_ratio"] = (df["map50"] / df["map50_fp32"]).round(4)

    df.to_csv(OUT_DIR/"trt_benchmark_rq2.csv", index=False)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TRT BENCHMARK SUMMARY (RQ2)")
    print(f"{'='*70}")
    print(f"  {'Model':<14} {'Precision':<8} {'mAP@0.5':>9} "
          f"{'Retention':>10} {'Latency ms':>11}")
    print(f"  {'-'*14} {'-'*8} {'-'*9} {'-'*10} {'-'*11}")
    for _, row in df.iterrows():
        ret = f"{row['retention_ratio']:.4f}" if pd.notna(row.get("retention_ratio")) else "—"
        print(f"  {row['model']:<14} {row['precision'].upper():<8} "
              f"{row['map50']:>9.4f} {ret:>10} "
              f"{row['mean_latency_ms']:>10.2f}ms")

    # ── Paired t-test: YOLOv8 vs RT-DETR retention ratio ─────────────────────
    print(f"\n{'='*70}")
    print("  PAIRED T-TEST: H0-2 (no difference in retention at FP16/INT8)")
    print(f"{'='*70}")

    test_results = []
    for prec in ["fp16","int8"]:
        sub = df[df["precision"]==prec].copy()
        yolo_ret  = sub[sub["arch"]=="yolo"  ]["retention_ratio"].dropna().values
        rtdet_ret = sub[sub["arch"]=="rtdetr"]["retention_ratio"].dropna().values

        if len(yolo_ret)==0 or len(rtdet_ret)==0:
            print(f"\n  {prec.upper()}: insufficient data for t-test")
            continue

        # Paired t-test requires equal length — use min length
        n    = min(len(yolo_ret), len(rtdet_ret))
        t, p = stats.ttest_rel(yolo_ret[:n], rtdet_ret[:n])
        decision = "REJECT H0-2" if p < 0.05 else "FAIL TO REJECT H0-2"

        print(f"\n  Precision: {prec.upper()}")
        print(f"    YOLOv8  retention (mean) = {yolo_ret.mean():.4f}")
        print(f"    RT-DETR retention (mean) = {rtdet_ret.mean():.4f}")
        print(f"    t = {t:.4f},  p = {p:.4f}   → {decision}  (α=0.05)")
        if p < 0.05:
            winner = "YOLOv8" if yolo_ret.mean() > rtdet_ret.mean() else "RT-DETR"
            print(f"    Ha-2 supported: {winner} retains higher accuracy at {prec.upper()}")
        test_results.append({"precision":prec,"t_stat":round(t,4),
                              "p_value":round(p,4),"reject_H0":p<0.05,
                              "yolo_retention":round(yolo_ret.mean(),4),
                              "rtdetr_retention":round(rtdet_ret.mean(),4)})

    pd.DataFrame(test_results).to_csv(OUT_DIR/"ttest_rq2.csv", index=False)
    print(f"\n  Saved → {OUT_DIR/'trt_benchmark_rq2.csv'}")
    print(f"  Saved → {OUT_DIR/'ttest_rq2.csv'}")


if __name__ == "__main__":
    main()
