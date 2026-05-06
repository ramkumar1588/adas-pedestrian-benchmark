"""
create_subset.py  —  Build the stratified 1,250-frame pedestrian evaluation subset.
Run ONCE from the repo root after preflight_check.py passes.

    python setup/create_subset.py

Target distribution (N = 1250, Hamilton rounding):
  Day   x Clear   x Highway : 332   Night x Clear   x Highway :  93
  Day   x Clear   x Other   : 497   Night x Clear   x Other   : 140
  Day   x Bad     x Highway :  59   Night x Bad     x Highway :  16
  Day   x Bad     x Other   :  88   Night x Bad     x Other   :  25
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from utils import load_config

# ── CONFIG ─────────────────────────────────────────────────────────────────────
_cfg             = load_config()
DATASET_ROOT     = _cfg["paths"]["dataset_root"]      # infos/ — trainval JSON + metadata
IMAGES_ROOT      = _cfg["paths"]["images_root"]        # images/ — camera_front_blur
ANNOTATIONS_ROOT = _cfg["paths"]["annotations_root"]   # annotations/ — object_detection.json
OUTPUT_DIR       = _cfg["paths"]["subset_dir"]         # where subset is written

N_TARGET    = 1250
SEED        = 42
MAX_WORKERS = 8
IMG_W       = 3848
IMG_H       = 2168

# ── CELL TARGETS (Hamilton rounding, sum = 1250) ──────────────────────────────
CELL_TARGETS = {
    ("day",   "good", "highway"):  332,
    ("day",   "good", "other"):    497,
    ("day",   "bad",  "highway"):   59,
    ("day",   "bad",  "other"):     88,
    ("night", "good", "highway"):   93,
    ("night", "good", "other"):    140,
    ("night", "bad",  "highway"):   16,
    ("night", "bad",  "other"):     25,
}
assert sum(CELL_TARGETS.values()) == N_TARGET

# ── CLASSIFIERS ───────────────────────────────────────────────────────────────
_GOOD_WEATHER = ("clear", "cloud", "overcast", "sunny")

def time_bin(v):  return "day"     if v.lower().strip() == "day" else "night"
def weather_bin(v): return "good"  if any(s in v.lower() for s in _GOOD_WEATHER) else "bad"
def road_bin(v):  return "highway" if v.lower().strip() in {"highway","motorway","expressway","freeway"} else "other"

# ── PATH HELPERS ──────────────────────────────────────────────────────────────
def resolve_image_path(frame):
    """JSON path → actual image path (Windows: colons become underscores)."""
    blur = frame.get("camera_frames", {}).get("front_blur", [])
    if not blur:
        return None
    rel   = blur[0]["filepath"]
    p     = Path(rel)
    fixed = p.parent / p.name.replace(":", "_")
    return IMAGES_ROOT / fixed

def resolve_annotation_path(frame):
    """JSON path → actual annotation path (inner annotations/ folder kept)."""
    ann = frame.get("annotations", {}).get("object_detection", {})
    rel = ann.get("filepath", "")
    return (ANNOTATIONS_ROOT / rel) if rel else None

# ── STEP 1: LOAD VAL FRAMES ───────────────────────────────────────────────────
def load_val_frames():
    path = DATASET_ROOT / "trainval-frames-full.json"
    print(f"Loading trainval JSON  ({path.stat().st_size / 1e6:.0f} MB) …")
    with open(path) as f:
        data = json.load(f)
    frames = data.get("val", [])
    print(f"  -> {len(frames):,} val frames")
    return frames

# ── STEP 2: SCAN METADATA ─────────────────────────────────────────────────────
def _read_meta(frame):
    meta_path = DATASET_ROOT / frame.get("metadata_path", "")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return None
    tod     = meta.get("time_of_day", "")
    weather = meta.get("scraped_weather", "")
    road    = meta.get("road_type", "")
    n_ped   = meta.get("num_pedestrians", 0)
    return {
        **frame,
        "time_of_day":     tod,
        "scraped_weather": weather,
        "road_type":       road,
        "num_pedestrians": n_ped,
        "time_bin":        time_bin(tod),
        "weather_bin":     weather_bin(weather),
        "road_bin":        road_bin(road),
        "cell":            (time_bin(tod), weather_bin(weather), road_bin(road)),
    }

def scan_val_frames(val_frames):
    print(f"\nScanning metadata for {len(val_frames):,} val frames …")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_read_meta, f): f for f in val_frames}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Reading metadata", unit="frame"):
            r = fut.result()
            if r is not None and r["num_pedestrians"] > 0:
                results.append(r)
    print(f"  -> {len(results):,} frames with >= 1 pedestrian")
    return results

# ── STEP 3: STRATIFIED SAMPLE ─────────────────────────────────────────────────
def stratified_sample(enriched):
    random.seed(SEED)
    buckets = defaultdict(list)
    for f in enriched:
        buckets[f["cell"]].append(f)
    print("\nCell availability vs target:")
    print(f"  {'Cell':<30} {'Available':>10} {'Target':>8}  Status")
    print("  " + "-" * 58)
    selected = []
    for cell, target in CELL_TARGETS.items():
        pool = buckets.get(cell, [])
        random.shuffle(pool)
        n_take    = min(target, len(pool))
        shortfall = target - n_take
        status    = "OK" if shortfall == 0 else f"SHORT by {shortfall}"
        print(f"  {' x '.join(cell):<30} {len(pool):>10,} {target:>8,}  {status}")
        selected.extend(pool[:n_take])
    print(f"\n  Total selected: {len(selected):,} / {N_TARGET:,}")
    return selected

# ── STEP 4: PARSE PEDESTRIAN BOXES ───────────────────────────────────────────
def _parse_boxes(ann_path):
    try:
        with open(ann_path) as f:
            anns = json.load(f)
    except Exception:
        return []
    boxes = []
    for obj in anns:
        props = obj.get("properties", {})
        if props.get("class") != "Pedestrian" or props.get("unclear", False):
            continue
        coords = obj.get("geometry", {}).get("coordinates", [])
        if not coords:
            continue
        xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
        xmin = max(0.0, min(xs)); ymin = max(0.0, min(ys))
        xmax = min(IMG_W, max(xs)); ymax = min(IMG_H, max(ys))
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def _to_yolo(xmin, ymin, xmax, ymax):
    return (f"0 {((xmin+xmax)/2)/IMG_W:.6f} {((ymin+ymax)/2)/IMG_H:.6f} "
            f"{(xmax-xmin)/IMG_W:.6f} {(ymax-ymin)/IMG_H:.6f}")

# ── STEP 5: BUILD OUTPUT ─────────────────────────────────────────────────────
def build_output(selected):
    img_dir  = OUTPUT_DIR / "images"
    lbl_dir  = OUTPUT_DIR / "yolo" / "labels"
    coco_dir = OUTPUT_DIR / "coco" / "annotations"
    for d in (img_dir, lbl_dir, coco_dir):
        d.mkdir(parents=True, exist_ok=True)

    coco_images = []; coco_annotations = []; meta_rows = []; ann_id = 1

    def _process(frame):
        fid      = frame["id"]
        image_id = int(fid)
        src = resolve_image_path(frame)
        dst = img_dir / f"{fid}.jpg"
        if src and src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        ann_path = resolve_annotation_path(frame)
        boxes    = _parse_boxes(ann_path) if ann_path else []
        (lbl_dir / f"{fid}.txt").write_text("\n".join(_to_yolo(*b) for b in boxes))
        return {
            "fid": fid, "image_id": image_id, "boxes": boxes,
            "meta": {
                "frame_id": fid, "time_of_day": frame["time_of_day"],
                "scraped_weather": frame["scraped_weather"],
                "road_type": frame["road_type"],
                "num_pedestrians": frame["num_pedestrians"],
                "time_bin": frame["time_bin"], "weather_bin": frame["weather_bin"],
                "road_bin": frame["road_bin"], "cell": " x ".join(frame["cell"]),
                "image_filename": f"{fid}.jpg",
            }
        }

    print(f"\nExporting {len(selected):,} frames …")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_process, f): f for f in selected}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Copying + annotating", unit="frame"):
            r = fut.result()
            if r:
                results.append(r)

    for r in results:
        coco_images.append({"id": r["image_id"], "file_name": f"{r['fid']}.jpg",
                             "width": IMG_W, "height": IMG_H})
        for xmin, ymin, xmax, ymax in r["boxes"]:
            bw, bh = xmax - xmin, ymax - ymin
            coco_annotations.append({
                "id": ann_id, "image_id": r["image_id"], "category_id": 1,
                "bbox": [xmin, ymin, bw, bh], "area": bw * bh, "iscrowd": 0,
            })
            ann_id += 1
        meta_rows.append(r["meta"])

    # COCO JSON
    coco_out = coco_dir / "instances.json"
    with open(coco_out, "w") as f:
        json.dump({
            "info": {"description": "ZOD Frames Pedestrian Subset 1250",
                     "version": "1.0", "year": 2026,
                     "contributor": "Ramkumar Rajachandrasekaran"},
            "licenses": [],
            "categories": [{"id": 1, "name": "pedestrian", "supercategory": "person"}],
            "images": coco_images, "annotations": coco_annotations,
        }, f, indent=2)
    print(f"  [OK] COCO JSON   -> {coco_out}  ({len(coco_images)} images, {len(coco_annotations)} annotations)")

    # metadata.csv
    csv_out = OUTPUT_DIR / "metadata.csv"
    fields  = ["frame_id","time_of_day","scraped_weather","road_type",
               "num_pedestrians","time_bin","weather_bin","road_bin","cell","image_filename"]
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(meta_rows)
    print(f"  [OK] metadata.csv -> {csv_out}")

    # frame_ids.txt
    (OUTPUT_DIR / "frame_ids.txt").write_text("\n".join(r["meta"]["frame_id"] for r in results))

    # YOLO dataset.yaml
    yaml_out = OUTPUT_DIR / "yolo" / "dataset.yaml"
    yaml_out.write_text(f"path: {OUTPUT_DIR.as_posix()}\nval: images\nnc: 1\nnames:\n  0: pedestrian\n")
    print(f"  [OK] dataset.yaml -> {yaml_out}")

# ── DISTRIBUTION REPORT ───────────────────────────────────────────────────────
def print_report(selected):
    from collections import Counter
    total = len(selected)
    def pct(n): return f"{n:>5,}  ({n/total*100:.1f}%)"
    print("\n" + "=" * 55)
    tod  = Counter(f["time_bin"]    for f in selected)
    wth  = Counter(f["weather_bin"] for f in selected)
    road = Counter(f["road_bin"]    for f in selected)
    print(f"  Day             : {pct(tod['day'])}")
    print(f"  Night/Twilight  : {pct(tod['night'])}")
    print(f"  Clear/Cloudy    : {pct(wth['good'])}")
    print(f"  Rain/Snow/Fog   : {pct(wth['bad'])}")
    print(f"  Highway         : {pct(road['highway'])}")
    print(f"  City/Rural      : {pct(road['other'])}")
    print(f"  Total           : {total:,}")
    print("=" * 55)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  ZOD Stratified Subset Creator  (N = 1250)")
    print("=" * 55)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    val_frames = load_val_frames()
    enriched   = scan_val_frames(val_frames)
    selected   = stratified_sample(enriched)
    build_output(selected)
    print_report(selected)
    print(f"\n[OK] Subset ready at: {OUTPUT_DIR}")
    print("     Next: python run_all.py")

if __name__ == "__main__":
    main()
