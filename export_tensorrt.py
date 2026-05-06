"""
tensorrt_export.py  —  RQ2: Export models to TensorRT at FP32 / FP16 / INT8
────────────────────────────────────────────────────────────────────────────────
Exports each enabled model to TensorRT at three precisions.
Engines are saved to model_dir alongside the original .pt weights.

Requirements:
  - TensorRT installed  (comes with CUDA toolkit or install separately)
  - pip install ultralytics
  - RTX 4060 (compute capability 8.9 — Ada Lovelace)

Run:  python tensorrt_export.py
      python tensorrt_export.py --models yolov8n rtdetr-r50
────────────────────────────────────────────────────────────────────────────────
"""

import sys
# onnx/onnxslim installed to a short path to work around Windows MAX_PATH limit
# (Windows Store Python's site-packages path is too long for onnx's test data)
sys.path.insert(0, r"C:\onnxlib")

import argparse
import json
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

CONFIG_PATH = Path("config.yaml")

# INT8 calibration: number of images to use (subset of eval set)
INT8_CALIB_IMAGES = 100
INT8_CALIB_IMGSZ  = 1280


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["model_dir"]  = Path(cfg["paths"]["model_dir"])
    cfg["paths"]["subset_dir"] = Path(cfg["paths"]["subset_dir"])
    return cfg


def get_calib_images(subset_dir: Path, n: int) -> list[str]:
    """Return a list of calibration image paths for INT8."""
    images = sorted((subset_dir / "images").glob("*.jpg"))[:n]
    return [str(p) for p in images]


def export_model(name: str, weight: str, model_dir: Path,
                 calib_images: list[str], imgsz: int = INT8_CALIB_IMGSZ):
    """Export one model at FP32, FP16, INT8. Returns dict of engine paths."""
    pt_path = model_dir / weight
    if not pt_path.exists():
        print(f"  [SKIP] {name}: weight not found at {pt_path}")
        print(f"         Run pipeline.py --models {name} first to download.")
        return {}

    engines = {}
    precisions = [
        ("fp32", dict(half=False, int8=False)),
        ("fp16", dict(half=True,  int8=False)),
        ("int8", dict(half=False, int8=True)),
    ]

    for prec_name, kwargs in precisions:
        engine_path = model_dir / f"{pt_path.stem}_{prec_name}.engine"

        if engine_path.exists():
            print(f"  [CACHED] {name} {prec_name.upper()} → {engine_path.name}")
            engines[prec_name] = engine_path
            continue

        print(f"\n  Exporting {name} → {prec_name.upper()} …")
        t0 = time.perf_counter()

        try:
            model = YOLO(str(pt_path))

            extra = {}
            if kwargs["int8"] and calib_images:
                # Ultralytics INT8 calibration requires a dataset YAML
                calib_dir  = Path(calib_images[0]).parent
                calib_yaml = model_dir / f"{name}_calib.yaml"
                with open(calib_yaml, "w") as yf:
                    yaml.dump({
                        "path":  str(calib_dir.parent),
                        "train": calib_dir.name,
                        "val":   calib_dir.name,
                        "nc":    80,
                        "names": {i: f"cls{i}" for i in range(80)},
                    }, yf)
                extra["data"] = str(calib_yaml)

            exported = model.export(
                format="engine",
                imgsz=imgsz,
                device=0,            # RTX 4060
                workspace=4,         # GB for TRT optimiser
                simplify=True,
                **kwargs,
                **extra,
            )
            elapsed = time.perf_counter() - t0

            # Ultralytics saves the engine next to the .pt file with suffix .engine
            # Rename to include precision tag
            default_engine = Path(str(pt_path).replace(".pt", ".engine"))
            if default_engine.exists():
                default_engine.rename(engine_path)
                engines[prec_name] = engine_path
                size_mb = engine_path.stat().st_size / (1024**2)
                print(f"    Done in {elapsed:.0f}s  |  {size_mb:.0f} MB  → {engine_path.name}")
            else:
                print(f"    [WARN] Expected engine at {default_engine} — not found.")

        except Exception as e:
            print(f"    [ERR] {prec_name.upper()} export failed: {e}")

    return engines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model names to export (default: all enabled)")
    args = parser.parse_args()

    cfg       = load_config()
    model_dir = cfg["paths"]["model_dir"]
    sub_dir   = cfg["paths"]["subset_dir"]
    imgsz     = cfg["inference"]["imgsz"]
    calib_img = get_calib_images(sub_dir, INT8_CALIB_IMAGES)

    print(f"TensorRT export — RTX 4060")
    print(f"Model dir   : {model_dir}")
    print(f"Calibration : {len(calib_img)} images for INT8")
    print(f"imgsz       : {imgsz}")

    # Determine which models to export
    enabled = {
        name: mcfg["weight"]
        for name, mcfg in cfg["models"].items()
        if mcfg.get("enabled", False)
    }
    if args.models:
        enabled = {k: v for k, v in enabled.items() if k in args.models}

    if not enabled:
        print("[ERR] No models selected.")
        return

    print(f"\nModels to export: {list(enabled.keys())}\n")

    all_engines = {}
    for name, weight in enabled.items():
        print(f"\n{'─'*55}")
        print(f"  {name}  ({weight})")
        print(f"{'─'*55}")
        engines = export_model(name, weight, model_dir, calib_img, imgsz)
        all_engines[name] = {k: str(v) for k, v in engines.items()}

    # Save engine manifest
    manifest = model_dir / "trt_engines.json"
    with open(manifest, "w") as f:
        json.dump(all_engines, f, indent=2)

    print(f"\n{'='*55}")
    print("  EXPORT SUMMARY")
    print(f"{'='*55}")
    for name, eng in all_engines.items():
        for prec, path in eng.items():
            size_mb = Path(path).stat().st_size/(1024**2) if Path(path).exists() else 0
            print(f"  {name:<14} {prec.upper():<6} {size_mb:>6.0f} MB  {Path(path).name}")

    print(f"\n  Manifest saved → {manifest}")
    print(f"  Next: python benchmark_tensorrt.py")


if __name__ == "__main__":
    main()
