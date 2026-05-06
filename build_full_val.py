"""
build_full_val.py
─────────────────────────────────────────────────────────────────────────────
Builds a full ZOD validation split dataset from the raw ZOD Frames data.
Processes all 10,023 validation frames into the same flat structure used by
the benchmark pipeline (same format as subset_1250).

Output structure:
  D:/data/full_val/
  ├── images/           <- {frame_id}.jpg (flat)
  ├── yolo/
  │   ├── dataset.yaml
  │   └── labels/       <- {frame_id}.txt  (YOLO format, class 0 = pedestrian)
  ├── coco/
  │   └── annotations/
  │       └── instances.json
  ├── metadata.csv
  └── frame_ids.txt
"""

import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ── PATHS ─────────────────────────────────────────────────────────────────────
FRAMES_ROOT      = Path(r"D:\dataset\zod\frames")
INFOS_ROOT       = FRAMES_ROOT / "infos"
IMAGES_ROOT      = FRAMES_ROOT / "images"
ANNOTATIONS_ROOT = FRAMES_ROOT / "annotations"
OUTPUT_DIR       = Path(r"D:\data\full_val")

IMG_W = 3848
IMG_H = 2168
MAX_WORKERS = 8


# ── CLASSIFIERS (same as create_subset.py) ────────────────────────────────────
_GOOD_WEATHER_SUBSTRINGS = ("clear", "cloud", "overcast", "sunny")
_HIGHWAY_VALUES = {"highway", "motorway", "expressway", "freeway"}

def time_bin(v: str) -> str:
    return "day" if v.lower().strip() == "day" else "night"

def weather_bin(v: str) -> str:
    v_lower = v.lower().strip()
    return "good" if any(s in v_lower for s in _GOOD_WEATHER_SUBSTRINGS) else "bad"

def road_bin(v: str) -> str:
    return "highway" if v.lower().strip() in _HIGHWAY_VALUES else "other"


# ── PATH RESOLVERS ────────────────────────────────────────────────────────────
def resolve_image_path(frame: dict) -> Path | None:
    blur = frame.get("camera_frames", {}).get("front_blur", [])
    if not blur:
        return None
    rel   = blur[0]["filepath"]
    p     = Path(rel)
    fixed = p.parent / p.name.replace(":", "_")
    return IMAGES_ROOT / fixed


def resolve_annotation_path(frame: dict) -> Path | None:
    ann = frame.get("annotations", {}).get("object_detection", {})
    rel = ann.get("filepath", "")
    if not rel:
        return None
    return ANNOTATIONS_ROOT / rel


def resolve_metadata_path(frame: dict) -> Path | None:
    rel = frame.get("metadata_path", "")
    if not rel:
        return None
    return INFOS_ROOT / rel


# ── ANNOTATION PARSING ────────────────────────────────────────────────────────
def _parse_boxes(ann_path: Path) -> list[tuple]:
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
        xs   = [p[0] for p in coords]
        ys   = [p[1] for p in coords]
        xmin = max(0.0, min(xs))
        ymin = max(0.0, min(ys))
        xmax = min(IMG_W, max(xs))
        ymax = min(IMG_H, max(ys))
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def _to_yolo_line(xmin, ymin, xmax, ymax) -> str:
    cx = ((xmin + xmax) / 2) / IMG_W
    cy = ((ymin + ymax) / 2) / IMG_H
    w  = (xmax - xmin) / IMG_W
    h  = (ymax - ymin) / IMG_H
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ── PER-FRAME WORKER ─────────────────────────────────────────────────────────
def _process_frame(frame: dict, img_dir: Path, lbl_dir: Path) -> dict | None:
    fid = frame["id"]

    # Read metadata
    meta_path = resolve_metadata_path(frame)
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return None

    tod     = meta.get("time_of_day", "")
    weather = meta.get("scraped_weather", "")
    road    = meta.get("road_type", "")
    n_ped   = meta.get("num_pedestrians", 0)

    # Copy image
    src = resolve_image_path(frame)
    dst = img_dir / f"{fid}.jpg"
    if src and src.exists() and not dst.exists():
        shutil.copy2(src, dst)
    elif not (src and src.exists()):
        return None  # skip frames with missing images

    # Parse pedestrian boxes
    ann_path = resolve_annotation_path(frame)
    boxes    = _parse_boxes(ann_path) if ann_path else []

    # Write YOLO label (empty file = no pedestrians = background image)
    (lbl_dir / f"{fid}.txt").write_text(
        "\n".join(_to_yolo_line(*b) for b in boxes)
    )

    return {
        "fid":    fid,
        "boxes":  boxes,
        "meta": {
            "frame_id":        fid,
            "time_of_day":     tod,
            "scraped_weather": weather,
            "road_type":       road,
            "num_pedestrians": n_ped,
            "time_bin":        time_bin(tod),
            "weather_bin":     weather_bin(weather),
            "road_bin":        road_bin(road),
            "cell":            f"{time_bin(tod)} x {weather_bin(weather)} x {road_bin(road)}",
            "image_filename":  f"{fid}.jpg",
        },
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ZOD Full Validation Set Builder")
    print("=" * 60)

    for p, name in [(INFOS_ROOT, "infos"), (IMAGES_ROOT, "images"),
                    (ANNOTATIONS_ROOT, "annotations")]:
        assert p.exists(), f"Not found: {p}  ({name})"

    # Load val frame list
    trainval = INFOS_ROOT / "trainval-frames-full.json"
    print(f"\nLoading {trainval.stat().st_size / 1e6:.0f} MB trainval JSON …")
    with open(trainval) as f:
        data = json.load(f)
    val_frames = data["val"]
    print(f"  {len(val_frames):,} validation frames")

    # Create output directories
    img_dir  = OUTPUT_DIR / "images"
    lbl_dir  = OUTPUT_DIR / "yolo" / "labels"
    coco_dir = OUTPUT_DIR / "coco" / "annotations"
    for d in (img_dir, lbl_dir, coco_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Process all frames in parallel
    print(f"\nProcessing {len(val_frames):,} frames (copying images + writing labels) …")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_process_frame, f, img_dir, lbl_dir): f
                   for f in val_frames}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Frames", unit="frame"):
            r = fut.result()
            if r:
                results.append(r)

    results.sort(key=lambda r: r["fid"])
    print(f"  {len(results):,} frames processed successfully")

    n_with_peds = sum(1 for r in results if r["boxes"])
    n_bg        = len(results) - n_with_peds
    print(f"  {n_with_peds:,} frames with pedestrians")
    print(f"  {n_bg:,} background frames (no pedestrians)")

    # COCO JSON
    print("\nWriting COCO JSON …")
    coco_images, coco_annotations = [], []
    ann_id = 1
    for r in results:
        image_id = int(r["fid"])
        coco_images.append({
            "id": image_id, "file_name": f"{r['fid']}.jpg",
            "width": IMG_W, "height": IMG_H,
        })
        for xmin, ymin, xmax, ymax in r["boxes"]:
            bw, bh = xmax - xmin, ymax - ymin
            coco_annotations.append({
                "id": ann_id, "image_id": image_id,
                "category_id": 1,
                "bbox": [xmin, ymin, bw, bh],
                "area": bw * bh, "iscrowd": 0,
            })
            ann_id += 1

    coco_out = coco_dir / "instances.json"
    with open(coco_out, "w") as f:
        json.dump({
            "info": {"description": "ZOD Frames — Full Validation Split",
                     "version": "1.0", "year": 2026},
            "licenses": [],
            "categories": [{"id": 1, "name": "pedestrian", "supercategory": "person"}],
            "images": coco_images,
            "annotations": coco_annotations,
        }, f)
    print(f"  [OK] {coco_out}  ({len(coco_images):,} images, {len(coco_annotations):,} annotations)")

    # metadata.csv
    csv_out = OUTPUT_DIR / "metadata.csv"
    fields  = ["frame_id","time_of_day","scraped_weather","road_type",
               "num_pedestrians","time_bin","weather_bin","road_bin",
               "cell","image_filename"]
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(r["meta"] for r in results)
    print(f"  [OK] {csv_out}")

    # frame_ids.txt
    ids_out = OUTPUT_DIR / "frame_ids.txt"
    ids_out.write_text("\n".join(r["fid"] for r in results))
    print(f"  [OK] {ids_out}")

    # dataset.yaml
    yaml_out = OUTPUT_DIR / "yolo" / "dataset.yaml"
    yaml_out.write_text(
        f"path: {OUTPUT_DIR.as_posix()}\n"
        f"val:  images\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: pedestrian\n"
    )
    print(f"  [OK] {yaml_out}")

    # Summary
    from collections import Counter
    metas = [r["meta"] for r in results]
    tod   = Counter(m["time_bin"]    for m in metas)
    wth   = Counter(m["weather_bin"] for m in metas)
    road  = Counter(m["road_bin"]    for m in metas)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  DISTRIBUTION SUMMARY  (N = {total:,})")
    print(f"{'='*60}")
    print(f"  Day              : {tod['day']:>6,}  ({tod['day']/total*100:.1f}%)")
    print(f"  Night/Twilight   : {tod['night']:>6,}  ({tod['night']/total*100:.1f}%)")
    print(f"  Clear/Cloudy     : {wth['good']:>6,}  ({wth['good']/total*100:.1f}%)")
    print(f"  Rain/Snow/Fog    : {wth['bad']:>6,}  ({wth['bad']/total*100:.1f}%)")
    print(f"  Highway          : {road['highway']:>6,}  ({road['highway']/total*100:.1f}%)")
    print(f"  City/Rural       : {road['other']:>6,}  ({road['other']/total*100:.1f}%)")
    print(f"{'='*60}")
    print(f"\n[DONE]  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
