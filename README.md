# ADAS Pedestrian Detection Benchmark

**Comparative Study of CNN (YOLOv8) and Transformer (RT-DETR) for Robust Real-Time Pedestrian Detection in ADAS for Edge Devices**

Walsh College · QM640 Data Analytics Capstone · Ramkumar Rajachandrasekaran

---

## Status

**Pipeline complete — full ZOD validation set (10,023 frames)**

| Step | Status |
|------|--------|
| Inference — all 6 models × 10,023 frames | ✅ Complete |
| Stratified mAP + Two-way ANOVA (RQ1) | ✅ Complete |
| NMS regression (RQ4) | ✅ Complete — H0-4 **rejected** |
| TRT export — 18 engines (FP32/FP16/INT8) | ✅ Complete |
| TRT benchmark + paired t-test (RQ2) | ✅ Complete |
| GradCAM + cross-attention spatial IoU (RQ3) | ✅ Complete (methodology refinement pending) |
| Interpretability visualizations | 🔄 In Progress |
| Final report | 🔄 In Progress |

## Key Results (ZOD Full Validation Set, n = 10,023)

| Model | mAP@0.5 | INT8 Retention | FP16 Latency |
|-------|---------|----------------|-------------|
| YOLOv8n | 0.124 | 0.946 | 21.9 ms |
| YOLOv8s | 0.162 | 0.830 | 40.0 ms |
| YOLOv8m | 0.203 | 0.831 | 22.3 ms |
| YOLOv8l | 0.222 | 0.863 | 25.8 ms |
| RT-DETR-R50 | **0.234** | **0.973** | 37.7 ms |
| RT-DETR-R101 | 0.212 | **1.036** | 38.3 ms |

> **Key finding:** FP16 is lossless for all models. INT8 causes 13–17% accuracy loss in YOLOv8 but near-zero loss in RT-DETR — a fundamental architectural difference in quantisation robustness.

## Overview

This repository contains the full evaluation pipeline for a zero-shot pedestrian detection benchmark comparing YOLOv8 (CNN) and RT-DETR (Transformer) models on the [Zenseact Open Dataset (ZOD)](https://zod.zenseact.com/). No fine-tuning is performed — COCO pre-trained weights are evaluated directly on ZOD driving data.

**Four research questions are addressed:**

| RQ | Topic | Statistical Test | Decision |
|----|-------|-----------------|---------|
| RQ1 | Stratified mAP across weather, time-of-day, road type | Two-way ANOVA | H0 not rejected |
| RQ2 | TensorRT FP32 / FP16 / INT8 accuracy-latency trade-off | Paired t-test | H0 not rejected (underpowered) |
| RQ3 | GradCAM vs RT-DETR cross-attention spatial alignment | Mann-Whitney U | H0 not rejected (methodology limited) |
| RQ4 | NMS latency scaling with pedestrian density | Linear regression | **H0 rejected** |

---

## Repository Structure

```
adas-pedestrian-benchmark/
│
├── config.yaml                        ← Edit this first: set your data paths
├── utils.py                           ← Shared config loader (do not edit)
├── requirements.txt
├── run_all.py                         ← Master runner
├── .gitignore
├── README.md
│
├── setup/                             ← Run ONCE, in order, before anything else
│   ├── check_gpu.py                   │  Step 1 — verify CUDA + RTX 4060
│   ├── preflight_check.py             │  Step 2 — verify ZOD dataset structure
│   └── create_subset.py              ←  Step 3 — build stratified 1250-frame subset
│
├── helpers/                           ← Optional: exploration and diagnostics
│   └── explore_trainval.py            │  Inspect trainval-frames-full.json structure
│
│   (core pipeline — run via run_all.py or individually)
├── pipeline.py                        ← Inference + stratified mAP  (RQ1 data)
├── evaluate_map.py                    ← pycocotools mAP by condition
├── anova_accuracy_conditions.py       ← Two-way ANOVA               (RQ1 test)
├── nms_scaling_analysis.py            ← NMS density regression       (RQ4 test)
├── export_tensorrt.py                 ← TensorRT FP32/FP16/INT8 export (RQ2 prep)
├── benchmark_tensorrt.py              ← TRT latency + paired t-test  (RQ2 test)
├── interpretability.py                ← GradCAM + cross-attention    (RQ3 test)
└── generate_report.py                 ← Consolidated JSON report     (all RQs)
```

---

## Prerequisites

- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA RTX 4060 (8 GB VRAM) or equivalent Ada/Ampere GPU
- **CUDA**: 12.1 or later
- **Python**: 3.10 or later

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ramkumar1588/adas-pedestrian-benchmark.git
cd adas-pedestrian-benchmark
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA support

> **Important:** Install the CUDA-enabled PyTorch build first. The default `pip install torch` installs a CPU-only build.

```bash
# CUDA 12.1 (recommended for RTX 4060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 4060
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** If `pycocotools` fails to install, use:
> ```bash
> pip install pycocotools-windows
> ```

---

## ZOD Dataset Setup

### Requesting access

ZOD requires access approval. Email **opendataset@zenseact.com** with:
- Your name and institution
- Intended use of the dataset

Once approved, you will receive a Dropbox download link.

### Downloading manually via browser

If you downloaded the dataset through the ZOD web portal rather than the CLI, the files arrive as separate zip archives. Extract them to a common parent folder.

> **Windows quirk:** Colons (`:`) in filenames are replaced with underscores (`_`) when downloaded through a browser.
> The SDK handles this automatically — no manual renaming needed.

### Required folder structure

After extracting all archives, your dataset folder must look exactly like this:

```
<your_data_root>/zod/frames/
│
├── infos/                                    ← extract zod_frames_infos.zip here
│   ├── trainval-frames-full.json             ← 451 MB index of all 100k frames
│   └── single_frames/
│       ├── 000000/
│       │   ├── metadata.json                 ← time_of_day, weather, road_type …
│       │   ├── calibration.json
│       │   ├── ego_motion.json
│       │   └── oxts.hdf5
│       ├── 000001/
│       └── …  (100,000 frame folders)
│
├── images/                                   ← extract zod_frames_images_blur.zip here
│   └── single_frames/
│       ├── 000000/
│       │   └── camera_front_blur/
│       │       └── 000000_india_2021-04-19T10_23_10.444124Z.jpg
│       ├── 000001/
│       └── …
│
└── annotations/                              ← extract zod_frames_annotations.zip here
    └── single_frames/
        ├── 000000/
        │   └── annotations/
        │       ├── object_detection.json
        │       ├── lane_markings.json
        │       ├── traffic_signs.json
        │       └── road_condition.json
        ├── 000001/
        └── …
```

### Setting your paths in config.yaml

Open `config.yaml` and update the three dataset paths to match where you extracted the archives. Use forward slashes on all platforms:

```yaml
paths:
  dataset_root:     "data/zod/frames/infos"        # relative to repo root
  images_root:      "data/zod/frames/images"
  annotations_root: "data/zod/frames/annotations"
```

**Or use absolute paths if the data is on an external drive:**

```yaml
paths:
  dataset_root:     "D:/dataset/zod/frames/infos"
  images_root:      "D:/dataset/zod/frames/images"
  annotations_root: "D:/dataset/zod/frames/annotations"
  subset_dir:       "D:/dataset/zod/subset_1250"
  output_dir:       "D:/dataset/zod/subset_1250/predictions"
  model_dir:        "D:/dataset/zod/models"
```

---

## Step-by-Step Execution

### Step 1 — Verify GPU

```bash
python setup/check_gpu.py
```

Expected output: all `[OK]` checks, confirmed RTX 4060, smoke-test latency < 200 ms.

### Step 2 — Verify dataset structure

```bash
python setup/preflight_check.py
```

Expected output: 100,000 frame folders detected in all three roots, 10/10 sampled frames fully accessible.

### Step 3 — Build the evaluation dataset

**Option A — Stratified 1,250-frame subset** (faster, for development):
```bash
python setup/create_subset.py
```

**Option B — Full 10,023-frame validation split** (used for all reported results):
```bash
python build_full_val.py
# Then update config.yaml: subset_dir → D:/data/full_val
```

Expected output:
```
subset_1250/
├── metadata.csv
├── frame_ids.txt
├── images/          ← 1,250 .jpg files
├── yolo/
│   ├── dataset.yaml
│   └── labels/      ← 1,250 YOLO .txt label files
└── coco/
    └── annotations/
        └── instances.json
```

### Step 4 — Run the full pipeline

```bash
# Full pipeline (3–4 hours on RTX 4060)
python run_all.py

# Skip TensorRT if not yet installed
python run_all.py --skip-trt

# Only statistical tests (after inference is done)
python run_all.py --only-stats

# Skip interpretability (fastest complete run)
python run_all.py --skip-interp
```

### Step 5 — Run scripts individually

Each script can also be run standalone in this order:

```bash
# RQ1 — inference + mAP
python pipeline.py                        # ~3–4 hrs (all 6 models)
python evaluate_map.py                    # ~2 min
python anova_accuracy_conditions.py       # ~5 min

# RQ4 — NMS analysis (requires pipeline.py output)
python nms_scaling_analysis.py            # ~1 min

# RQ2 — TensorRT (requires tensorrt installation)
python export_tensorrt.py                 # ~30–60 min
python benchmark_tensorrt.py             # ~2–3 hrs

# RQ3 — Interpretability
python interpretability.py               # ~45–90 min

# Final report (run last, or any time to see partial results)
python generate_report.py
```

### Run a single model (recommended for first test)

```bash
python pipeline.py --models yolov8n
```

---

## Controlling Which Models Run

Edit `config.yaml` to enable or disable individual models:

```yaml
models:
  yolov8n:
    enabled: true      # ← change to false to skip
    weight:  yolov8n.pt
    type:    yolo

  rtdetr-r50:
    enabled: false     # ← skip this model
    weight:  rtdetr-l.pt
    type:    rtdetr
```

Model weights are downloaded automatically on first run and cached in `model_dir`.

---

## Output Structure

After a full pipeline run:

```
data/subset_1250/predictions/
│
├── yolov8n/
│   ├── predictions_coco.json     ← COCO-format detections
│   ├── latency.csv               ← per-image timing + metadata
│   ├── summary.json              ← aggregate latency stats
│   └── map_results.json          ← stratified mAP by condition
│
├── yolov8s/ … yolov8l/ … rtdetr-r50/ … rtdetr-r101/
│   └── (same structure as above)
│
├── stats/
│   ├── per_image_recall.csv      ← RQ1 DV
│   ├── anova_rq1.csv             ← RQ1 ANOVA results
│   ├── regression_rq4.csv        ← RQ4 regression slopes + p-values
│   ├── nms_overhead_rq4.csv      ← RQ4 NMS overhead by density bin
│   ├── latency_crossover_rq4.csv ← RQ4 YOLO vs RT-DETR by bin
│   ├── trt_benchmark_rq2.csv     ← RQ2 accuracy retention ratios
│   └── ttest_rq2.csv             ← RQ2 paired t-test results
│
├── interpretability/
│   ├── spatial_iou_rq3.csv       ← per-sample spatial IoU
│   ├── mannwhitney_rq3.json      ← RQ3 Mann-Whitney U results
│   └── visualizations/           ← GradCAM + attention overlay images
│
└── report/
    ├── full_report.json          ← complete structured report (all RQs)
    ├── rq1_summary.json
    ├── rq2_summary.json
    ├── rq3_summary.json
    ├── rq4_summary.json
    └── model_comparison.json     ← model × metric matrix
```

---

## Models

| Model | Architecture | COCO mAP@0.5:0.95 | NMS |
|-------|-------------|-------------------|-----|
| YOLOv8n | CNN (CSPDarknet) | 37.3 | Yes |
| YOLOv8s | CNN (CSPDarknet) | 44.9 | Yes |
| YOLOv8m | CNN (CSPDarknet) | 50.2 | Yes |
| YOLOv8l | CNN (CSPDarknet) | 52.9 | Yes |
| RT-DETR-R50 | Transformer (ResNet-50) | 53.1 | No |
| RT-DETR-R101 | Transformer (ResNet-101) | 54.3 | No |

All models use COCO pre-trained weights. No fine-tuning on ZOD is performed (zero-shot evaluation).

---

## Evaluation Dataset

All reported results use the **full ZOD validation split (10,023 frames)**. A stratified 1,250-frame subset is also available for faster development runs.

### Full validation split distribution (n = 10,023)

| Condition | Group | Count | % |
|-----------|-------|-------|---|
| Time of day | Day | 7,943 | 79.2% |
| | Night/Twilight | 2,080 | 20.8% |
| Weather | Clear/Cloudy | 8,372 | 83.5% |
| | Rain/Snow/Fog | 1,651 | 16.5% |
| Road type | Highway | 1,256 | 12.5% |
| | City/Rural | 8,767 | 87.5% |
| Pedestrian frames | ≥1 pedestrian | 6,035 | 60.2% |
| | Background | 3,988 | 39.8% |

---

## Helper Scripts

The `helpers/` directory contains optional scripts for dataset exploration:

```bash
# Inspect the structure and contents of trainval-frames-full.json
python helpers/explore_trainval.py
```

This prints the top-level split counts, a full example frame entry, all file paths, and verifies which root directory (`dataset_root`) resolves your paths correctly.

---

## Citation

If you use this code or the ZOD dataset in your work, please cite:

```bibtex
@inproceedings{zod2023,
  author    = {Alibeigi, Mina and Ljungbergh, William and Tonderski, Adam and
               Hess, Georg and Lilja, Adam and Lindstr{\"o}m, Carl and
               Motorniuk, Daria and Fu, Junsheng and Widahl, Jenny and
               Petersson, Christoffer},
  title     = {Zenseact Open Dataset: A large-scale and diverse multimodal
               dataset for autonomous driving},
  booktitle = {Proceedings of the IEEE/CVF ICCV},
  year      = {2023},
  pages     = {20178--20188}
}

@software{jocher2023ultralytics,
  author  = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  title   = {Ultralytics YOLOv8},
  year    = {2023},
  url     = {https://github.com/ultralytics/ultralytics}
}
```

---

## License

**Code**: MIT License

**Dataset**: CC BY-SA 4.0 — Zenseact AB. Any public use must include the full attribution notice from [zod.zenseact.com](https://zod.zenseact.com/).
