"""
generate_report.py
────────────────────────────────────────────────────────────────────────────────
Consolidates ALL evaluation outputs into a single structured JSON report.
Covers every research question, hypothesis, and model configuration.

Run after the full pipeline:
    python generate_report.py

Outputs:
    predictions/report/full_report.json      ← complete structured report
    predictions/report/rq{1-4}_summary.json  ← per-RQ summaries
    predictions/report/model_comparison.json ← model × metric matrix
────────────────────────────────────────────────────────────────────────────────
"""

import json
import csv
from pathlib import Path
from datetime import datetime, timezone

from utils import load_config
_cfg       = load_config()
SUBSET_DIR = _cfg["paths"]["subset_dir"]
PRED_DIR   = _cfg["paths"]["output_dir"]
STATS_DIR  = PRED_DIR / "stats"
INTERP_DIR = PRED_DIR / "interpretability"
REPORT_DIR = PRED_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS    = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "rtdetr-r50", "rtdetr-r101"]
YOLO_MODELS   = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
RTDETR_MODELS = ["rtdetr-r50", "rtdetr-r101"]
PRECISIONS    = ["fp32", "fp16", "int8"]


# ── LOADERS ───────────────────────────────────────────────────────────────────
def load_json(path: Path, default=None):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    print(f"  [MISSING] {path.name}")
    return default

def load_csv_as_records(path: Path):
    if not path.exists():
        print(f"  [MISSING] {path.name}")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def try_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def try_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None

def try_bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("true","1","yes"):
        return True
    if str(v).lower() in ("false","0","no"):
        return False
    return None


# ── SECTION: STUDY METADATA ───────────────────────────────────────────────────
def build_study_meta() -> dict:
    return {
        "title":        "Comparative Study of CNN (YOLOv8) and Transformer (RT-DETR) "
                        "for Robust Real-Time Pedestrian Detection in ADAS for Edge Devices",
        "author":       "Ramkumar Rajachandrasekaran",
        "institution":  "Walsh College",
        "course":       "QM640: Data Analytics Capstone",
        "advisor":      "Dr. Arun Sharma",
        "github":       "https://github.com/ramkumar1588/adas-pedestrian-benchmark",
        "dataset":      "Zenseact Open Dataset (ZOD) Frames",
        "dataset_url":  "https://zod.zenseact.com/",
        "license":      "CC BY-SA 4.0",
        "subset_size":  1250,
        "eval_split":   "val",
        "zero_shot":    True,
        "gpu":          "NVIDIA RTX 4060",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "yolo": {
                "variants":    YOLO_MODELS,
                "pretrained":  "COCO",
                "architecture":"CNN (CSPDarknet + C2f)",
                "nms":         True,
            },
            "rtdetr": {
                "variants":    RTDETR_MODELS,
                "pretrained":  "COCO",
                "architecture":"Transformer (ResNet + AIFI decoder)",
                "nms":         False,
            },
        },
        "stratification": {
            "time_of_day":  {"day": 975, "night_twilight": 275},
            "weather":      {"clear_cloudy": 1062, "rain_snow_fog": 188},
            "road_type":    {"highway": 500, "city_rural": 750},
        },
    }


# ── SECTION: RQ1 ──────────────────────────────────────────────────────────────
def build_rq1() -> dict:
    print("\n  Building RQ1 …")

    # Per-model stratified mAP from map_results.json
    stratified_map = {}
    for model in ALL_MODELS:
        path = PRED_DIR / model / "map_results.json"
        r    = load_json(path)
        if r is None:
            continue
        stratified_map[model] = {
            "overall": r.get("overall", {}),
            "by_time_of_day":  r.get("stratified", {}).get("time_bin",    {}),
            "by_weather":      r.get("stratified", {}).get("weather_bin", {}),
            "by_road_type":    r.get("stratified", {}).get("road_bin",    {}),
        }

    # ANOVA results
    anova_rows = load_csv_as_records(STATS_DIR / "anova_rq1.csv")
    anova = {}
    for row in anova_rows:
        cond = row.get("condition", "unknown")
        anova[cond] = {
            "F_architecture":    try_float(row.get("F_arch")),
            "p_architecture":    try_float(row.get("p_arch")),
            "F_condition":       try_float(row.get("F_cond")),
            "p_condition":       try_float(row.get("p_cond")),
            "F_interaction":     try_float(row.get("F_interaction")),
            "p_interaction":     try_float(row.get("p_interaction")),
            "partial_eta2":      try_float(row.get("partial_eta2")),
            "reject_H0":         try_bool(row.get("reject_H0")),
            "alpha":             0.05,
        }

    # Overall decision: H0-1 rejected if ANY condition shows significant interaction
    any_reject = any(v.get("reject_H0") for v in anova.values()) if anova else None

    return {
        "research_question": (
            "How does zero-shot pedestrian detection accuracy (mAP@0.5, mAP@0.5:0.95) "
            "of COCO pre-trained YOLOv8 and RT-DETR differ when stratified across "
            "ZOD time-of-day, weather, and road-type attributes?"
        ),
        "hypothesis": {
            "H0": "No significant interaction effect between model architecture and "
                  "environmental condition on pedestrian detection mAP.",
            "Ha": "Significant interaction effect exists, with RT-DETR showing higher "
                  "robustness under adverse conditions.",
        },
        "statistical_test": "Two-way ANOVA (typ=2), α = 0.05",
        "dependent_variable": "Per-image recall@IoU=0.5",
        "independent_variables": ["architecture (yolo|rtdetr)", "condition (weather|time|road)"],
        "stratified_map":  stratified_map,
        "anova_by_condition": anova,
        "decision": {
            "reject_H0": any_reject,
            "conclusion": (
                "H0-1 rejected: significant interaction between architecture and "
                "environmental condition detected."
                if any_reject
                else "H0-1 not rejected: no significant interaction found."
                if any_reject is not None
                else "Pending — run anova_accuracy_conditions.py"
            ),
        },
    }


# ── SECTION: RQ2 ──────────────────────────────────────────────────────────────
def build_rq2() -> dict:
    print("  Building RQ2 …")

    # TRT benchmark per model × precision
    benchmark_rows = load_csv_as_records(STATS_DIR / "trt_benchmark_rq2.csv")
    benchmark = {}
    for row in benchmark_rows:
        model = row.get("model","")
        prec  = row.get("precision","")
        if model not in benchmark:
            benchmark[model] = {}
        benchmark[model][prec] = {
            "map50":               try_float(row.get("map50")),
            "map50_95":            try_float(row.get("map50_95")),
            "mean_latency_ms":     try_float(row.get("mean_latency_ms")),
            "retention_ratio":     try_float(row.get("retention_ratio")),
            "arch": "yolo" if model in YOLO_MODELS else "rtdetr",
        }

    # Paired t-test results
    ttest_rows = load_csv_as_records(STATS_DIR / "ttest_rq2.csv")
    ttest = {}
    for row in ttest_rows:
        prec = row.get("precision","")
        ttest[prec] = {
            "t_statistic":        try_float(row.get("t_stat")),
            "p_value":            try_float(row.get("p_value")),
            "reject_H0":          try_bool(row.get("reject_H0")),
            "yolo_retention":     try_float(row.get("yolo_retention")),
            "rtdetr_retention":   try_float(row.get("rtdetr_retention")),
            "alpha":              0.05,
        }

    any_reject = any(v.get("reject_H0") for v in ttest.values()) if ttest else None

    return {
        "research_question": (
            "What is the accuracy-latency trade-off for each architecture when "
            "COCO pre-trained models are exported to TensorRT at FP32, FP16, "
            "and INT8 precision on an NVIDIA RTX 4060 GPU?"
        ),
        "hypothesis": {
            "H0": "No significant difference in accuracy retention ratio between "
                  "YOLOv8 and RT-DETR under FP16 and INT8 quantization.",
            "Ha": "CNN-based YOLOv8 retains significantly higher accuracy under "
                  "INT8 quantization due to differences in numerical sensitivity "
                  "of attention mechanisms.",
        },
        "statistical_test": "Paired t-test, α = 0.05",
        "metric": "Accuracy retention ratio = mAP_quantized / mAP_FP32",
        "precisions": PRECISIONS,
        "benchmark_by_model": benchmark,
        "ttest_by_precision": ttest,
        "decision": {
            "reject_H0": any_reject,
            "conclusion": (
                "H0-2 rejected: significant difference in quantization sensitivity "
                "between YOLOv8 and RT-DETR."
                if any_reject
                else "H0-2 not rejected: no significant difference in retention."
                if any_reject is not None
                else "Pending — run export_tensorrt.py and benchmark_tensorrt.py"
            ),
        },
    }


# ── SECTION: RQ3 ──────────────────────────────────────────────────────────────
def build_rq3() -> dict:
    print("  Building RQ3 …")

    mw = load_json(INTERP_DIR / "mannwhitney_rq3.json")
    spatial_rows = load_csv_as_records(INTERP_DIR / "spatial_iou_rq3.csv")

    # Aggregate spatial IoU per model
    spatial_by_model = {}
    for row in spatial_rows:
        model = row.get("model","")
        iou   = try_float(row.get("spatial_iou"))
        if model not in spatial_by_model:
            spatial_by_model[model] = []
        if iou is not None:
            spatial_by_model[model].append(iou)

    spatial_stats = {}
    for model, ious in spatial_by_model.items():
        if ious:
            sorted_v = sorted(ious)
            n = len(sorted_v)
            spatial_stats[model] = {
                "n":      n,
                "mean":   round(sum(ious)/n, 4),
                "median": round(sorted_v[n//2], 4),
                "min":    round(sorted_v[0], 4),
                "max":    round(sorted_v[-1], 4),
                "arch":   "yolo" if model in YOLO_MODELS else "rtdetr",
                "method": "GradCAM" if model in YOLO_MODELS else "Cross-attention",
            }

    reject = mw.get("reject_H0") if mw else None

    return {
        "research_question": (
            "What systematic differences exist between GradCAM saliency maps (YOLOv8) "
            "and decoder cross-attention maps (RT-DETR) in correctly detected vs missed "
            "pedestrian samples under zero-shot evaluation?"
        ),
        "hypothesis": {
            "H0": "No significant difference in spatial overlap (IoU) between "
                  "saliency/attention activation regions and ground-truth bounding boxes.",
            "Ha": "RT-DETR decoder cross-attention maps exhibit significantly higher "
                  "spatial overlap with ground-truth pedestrian bounding boxes.",
        },
        "statistical_test":   "Mann-Whitney U test (one-tailed), α = 0.05",
        "metric":             "Spatial IoU = area(heatmap_mask ∩ GT_box) / area(heatmap_mask ∪ GT_box)",
        "heatmap_threshold":  0.4,
        "n_samples":          500,
        "yolo_method":        "GradCAM from SPPF backbone layer (layer 9)",
        "rtdetr_method":      "Decoder cross-attention from last TransformerDecoderLayer",
        "yolo_model":         mw.get("yolo_model")   if mw else None,
        "rtdetr_model":       mw.get("rtdetr_model") if mw else None,
        "spatial_iou_by_model": spatial_stats,
        "mann_whitney": {
            "U_statistic":       mw.get("U_statistic")  if mw else None,
            "p_value":           mw.get("p_value")       if mw else None,
            "effect_r":          mw.get("effect_r")      if mw else None,
            "yolo_median_iou":   mw.get("yolo_median_iou") if mw else None,
            "rtdetr_median_iou": mw.get("rtdetr_median_iou") if mw else None,
            "reject_H0":         reject,
            "alpha":             0.05,
        },
        "decision": {
            "reject_H0": reject,
            "conclusion": (
                "H0-3 rejected: RT-DETR cross-attention shows significantly higher "
                "spatial overlap with GT pedestrian boxes."
                if reject
                else "H0-3 not rejected: no significant difference in spatial overlap."
                if reject is not None
                else "Pending — run interpretability.py"
            ),
        },
    }


# ── SECTION: RQ4 ──────────────────────────────────────────────────────────────
def build_rq4() -> dict:
    print("  Building RQ4 …")

    reg_rows      = load_csv_as_records(STATS_DIR / "regression_rq4.csv")
    overhead_rows = load_csv_as_records(STATS_DIR / "nms_overhead_rq4.csv")
    crossover_rows= load_csv_as_records(STATS_DIR / "latency_crossover_rq4.csv")

    # Regression results per YOLO model
    regression = {}
    for row in reg_rows:
        regression[row.get("model","")] = {
            "slope_ms_per_pedestrian": try_float(row.get("slope")),
            "intercept_ms":            try_float(row.get("intercept")),
            "r_squared":               try_float(row.get("r_squared")),
            "p_value":                 try_float(row.get("p_value")),
            "reject_H0":               try_bool(row.get("reject_H0")),
        }

    # NMS overhead by density bin
    overhead = {}
    for row in overhead_rows:
        model = row.get("model","")
        binn  = row.get("density_bin","")
        if model not in overhead:
            overhead[model] = {}
        overhead[model][binn] = {
            "n":                 try_int(row.get("n")),
            "nms_mean_ms":       try_float(row.get("nms_mean_ms")),
            "total_mean_ms":     try_float(row.get("total_mean_ms")),
            "nms_overhead_pct":  try_float(row.get("nms_overhead_pct")),
            "exceeds_threshold": try_bool(row.get("exceeds_threshold")),
        }

    # Find threshold bin per model
    threshold = {}
    for model, bins in overhead.items():
        for bin_label, stats in bins.items():
            if stats.get("exceeds_threshold"):
                threshold[model] = bin_label
                break

    # Latency crossover table
    crossover = {}
    for row in crossover_rows:
        binn = row.get("density_bin","")
        crossover[binn] = {
            k: try_float(v) for k, v in row.items() if k != "density_bin"
        }

    any_reject = any(v.get("reject_H0") for v in regression.values()) if regression else None

    return {
        "research_question": (
            "How does YOLOv8 NMS post-processing latency scale with pedestrian "
            "density compared to RT-DETR end-to-end inference time?"
        ),
        "hypothesis": {
            "H0": "YOLOv8 NMS latency does not increase significantly as a function "
                  "of pedestrian count per frame.",
            "Ha": "YOLOv8 NMS latency increases linearly with pedestrian density, "
                  "and there exists a density threshold beyond which YOLOv8 total "
                  "inference time exceeds RT-DETR end-to-end latency.",
        },
        "statistical_test":       "Linear regression, α = 0.05",
        "nms_overhead_threshold": 20.0,
        "density_bins":           ["0-2", "3-5", "6-10", "10+"],
        "regression_by_model":    regression,
        "nms_overhead_by_model":  overhead,
        "nms_threshold_bin":      threshold,
        "latency_crossover":      crossover,
        "decision": {
            "reject_H0": any_reject,
            "conclusion": (
                "H0-4 rejected: NMS latency increases significantly with pedestrian density."
                if any_reject
                else "H0-4 not rejected: NMS latency does not significantly increase."
                if any_reject is not None
                else "Pending — run nms_scaling_analysis.py"
            ),
        },
    }


# ── SECTION: MODEL COMPARISON MATRIX ─────────────────────────────────────────
def build_model_matrix() -> dict:
    print("  Building model comparison matrix …")
    matrix = {}

    for model in ALL_MODELS:
        arch = "yolo" if model in YOLO_MODELS else "rtdetr"
        entry = {"arch": arch, "metrics": {}}

        # mAP from pipeline results
        map_path = PRED_DIR / model / "map_results.json"
        r = load_json(map_path)
        if r:
            entry["metrics"]["map50"]    = r.get("overall",{}).get("map50")
            entry["metrics"]["map50_95"] = r.get("overall",{}).get("map50_95")

        # Latency from summary.json
        sum_path = PRED_DIR / model / "summary.json"
        s = load_json(sum_path)
        if s:
            lat = s.get("latency_ms", {})
            entry["metrics"]["inference_mean_ms"]   = lat.get("inference",   {}).get("mean")
            entry["metrics"]["postprocess_mean_ms"] = lat.get("postprocess", {}).get("mean")
            entry["metrics"]["total_mean_ms"]       = lat.get("total",       {}).get("mean")
            entry["metrics"]["nms_overhead_pct"]    = s.get("nms_overhead_pct")

        # TRT retention ratios (FP16 and INT8)
        for prec in ["fp16", "int8"]:
            trt_pred = PRED_DIR / model / f"trt_{prec}" / "predictions_coco.json"
            trt_lat  = PRED_DIR / model / f"trt_{prec}" / "latency.csv"
            if trt_pred.exists():
                # Retention is stored in the benchmark CSV
                pass   # populated from trt_benchmark_rq2.csv below

        matrix[model] = entry

    # Fill TRT retention from benchmark CSV
    trt_rows = load_csv_as_records(STATS_DIR / "trt_benchmark_rq2.csv")
    for row in trt_rows:
        model = row.get("model","")
        prec  = row.get("precision","")
        if model in matrix and prec != "fp32":
            key = f"trt_{prec}_retention"
            matrix[model]["metrics"][key] = try_float(row.get("retention_ratio"))
            matrix[model]["metrics"][f"trt_{prec}_latency_ms"] = \
                try_float(row.get("mean_latency_ms"))

    return matrix


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("Generating report …")
    print(f"  Reading from: {PRED_DIR}")

    study  = build_study_meta()
    rq1    = build_rq1()
    rq2    = build_rq2()
    rq3    = build_rq3()
    rq4    = build_rq4()
    matrix = build_model_matrix()

    # ── Full report ───────────────────────────────────────────────────────────
    full_report = {
        "study":                  study,
        "rq1_accuracy_stratified":rq1,
        "rq2_tensorrt_quantization": rq2,
        "rq3_interpretability":   rq3,
        "rq4_nms_density_scaling":rq4,
        "model_comparison_matrix":matrix,
    }

    full_path = REPORT_DIR / "full_report.json"
    with open(full_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\n  [OK] full_report.json        ({full_path.stat().st_size//1024:,} KB)")

    # ── Per-RQ summaries ──────────────────────────────────────────────────────
    for rq_key, rq_data, label in [
        ("rq1", rq1, "rq1_summary.json"),
        ("rq2", rq2, "rq2_summary.json"),
        ("rq3", rq3, "rq3_summary.json"),
        ("rq4", rq4, "rq4_summary.json"),
    ]:
        # Compact version: just decision + key stats
        summary = {
            "research_question": rq_data["research_question"],
            "hypothesis":        rq_data["hypothesis"],
            "statistical_test":  rq_data["statistical_test"],
            "decision":          rq_data["decision"],
        }
        # Add key numeric results
        if rq_key == "rq1" and rq_data.get("anova_by_condition"):
            summary["anova_p_values"] = {
                cond: {
                    "p_interaction": v.get("p_interaction"),
                    "partial_eta2":  v.get("partial_eta2"),
                    "reject_H0":     v.get("reject_H0"),
                }
                for cond, v in rq_data["anova_by_condition"].items()
            }
        if rq_key == "rq2" and rq_data.get("ttest_by_precision"):
            summary["ttest"] = rq_data["ttest_by_precision"]
        if rq_key == "rq3" and rq_data.get("mann_whitney"):
            summary["mann_whitney"] = rq_data["mann_whitney"]
        if rq_key == "rq4" and rq_data.get("nms_threshold_bin"):
            summary["nms_threshold_bin"] = rq_data["nms_threshold_bin"]

        out = REPORT_DIR / label
        with open(out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  [OK] {label}")

    # ── Model comparison JSON ─────────────────────────────────────────────────
    comp_path = REPORT_DIR / "model_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(matrix, f, indent=2, default=str)
    print(f"  [OK] model_comparison.json")

    # ── Console: decisions summary ────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  HYPOTHESIS DECISIONS")
    print(f"{'='*62}")
    for label, rq_data in [("RQ1",rq1),("RQ2",rq2),("RQ3",rq3),("RQ4",rq4)]:
        dec = rq_data["decision"]
        reject = dec.get("reject_H0")
        icon   = "[REJECT H0]" if reject else \
                 "[FAIL TO REJECT H0]" if reject is False else "[PENDING]"
        print(f"  {label}  {icon}")
        print(f"       {dec['conclusion']}")
    print(f"{'='*62}")
    print(f"\n  All reports saved to: {REPORT_DIR}\n")


if __name__ == "__main__":
    main()
