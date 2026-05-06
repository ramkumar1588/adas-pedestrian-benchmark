"""
run_all.py  —  Master runner for the full ZOD evaluation pipeline
────────────────────────────────────────────────────────────────────────────────
Runs every script in the correct order.  Edit RUN_* flags to skip stages.

    python run_all.py                 # full pipeline
    python run_all.py --skip-trt      # skip TensorRT (no TRT installed)
    python run_all.py --only-stats    # only run statistical tests
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

STEPS = [
    # (script, label, flag_to_skip)
    ("pipeline.py",                  "Inference + stratified mAP   (RQ1 data)",    "skip_inference"),
    ("anova_accuracy_conditions.py", "Two-way ANOVA                (RQ1 test)",    "skip_stats"),
    ("nms_scaling_analysis.py",      "NMS density regression       (RQ4 test)",    "skip_stats"),
    ("export_tensorrt.py",           "TensorRT export FP32/FP16/INT8 (RQ2 prep)", "skip_trt"),
    ("benchmark_tensorrt.py",        "TensorRT benchmark + t-test  (RQ2 test)",    "skip_trt"),
    ("interpretability.py",          "GradCAM + cross-attention    (RQ3 test)",    "skip_interp"),
    ("generate_report.py",           "Consolidated JSON report     (all RQs)",     "skip_report"),
]


def run(script: str, label: str):
    print(f"\n{'═'*65}")
    print(f"  ▶  {label}")
    print(f"     {script}")
    print(f"{'═'*65}")
    t0  = time.perf_counter()
    ret = subprocess.run([sys.executable, str(HERE / script)])
    elapsed = (time.perf_counter() - t0) / 60
    if ret.returncode != 0:
        print(f"\n  [ERR] {script} exited with code {ret.returncode}")
        return False
    print(f"\n  Done in {elapsed:.1f} min")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-stats",     action="store_true")
    parser.add_argument("--skip-trt",       action="store_true",
                        help="Skip TensorRT export/benchmark (no TRT installed)")
    parser.add_argument("--skip-interp",    action="store_true",
                        help="Skip GradCAM/attention interpretability")
    parser.add_argument("--skip-report",    action="store_true",
                        help="Skip final report generation")
    parser.add_argument("--only-stats",     action="store_true",
                        help="Run only statistical tests + report (requires prior inference)")
    parser.add_argument("--yes", "-y",      action="store_true",
                        help="Skip interactive confirmation prompts (for automated runs)")
    args = parser.parse_args()

    skip_flags = {
        "skip_inference": args.skip_inference or args.only_stats,
        "skip_stats":     args.skip_stats,
        "skip_trt":       args.skip_trt,
        "skip_interp":    args.skip_interp,
        "skip_report":    args.skip_report,
    }

    print("ZOD Full Evaluation Pipeline")
    print("="*65)
    for script, label, flag in STEPS:
        if skip_flags.get(flag, False):
            print(f"  [SKIP] {label}")
        else:
            print(f"  [RUN ] {label}")
    print("="*65)
    if not args.yes:
        input("\nPress Enter to start, Ctrl+C to cancel …\n")

    passed = failed = 0
    for script, label, flag in STEPS:
        if skip_flags.get(flag, False):
            continue
        ok = run(script, label)
        if ok:
            passed += 1
        else:
            failed += 1
            if not args.yes:
                cont = input(f"\n  Script failed. Continue anyway? [y/N]: ")
                if cont.lower() != "y":
                    break

    print(f"\n{'='*65}")
    print(f"  Pipeline complete:  {passed} passed  /  {failed} failed")
    print(f"{'='*65}")
    print(f"\n  All outputs in:")
    print(f"    D:/dataset/zod/subset_1250/predictions/")
    print(f"    └── {{model}}/predictions_coco.json    ← COCO predictions")
    print(f"    └── {{model}}/latency.csv              ← per-image latency")
    print(f"    └── {{model}}/map_results.json         ← stratified mAP")
    print(f"    └── {{model}}/trt_{{prec}}/            ← TRT benchmarks")
    print(f"    └── stats/anova_rq1.csv               ← RQ1 ANOVA results")
    print(f"    └── stats/regression_rq4.csv          ← RQ4 regression")
    print(f"    └── stats/nms_overhead_rq4.csv        ← RQ4 NMS threshold")
    print(f"    └── stats/trt_benchmark_rq2.csv       ← RQ2 benchmark")
    print(f"    └── stats/ttest_rq2.csv               ← RQ2 paired t-test")
    print(f"    └── interpretability/spatial_iou_rq3.csv  ← RQ3 IoU")
    print(f"    └── interpretability/mannwhitney_rq3.json ← RQ3 test")
    print(f"    └── interpretability/visualizations/  ← overlay images")


if __name__ == "__main__":
    main()
