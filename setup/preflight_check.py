import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
"""
preflight_check.py  (v5 — final)
───────────────────────────────────────────────────────────────────────────────
Two confirmed quirks in your browser-downloaded ZOD dataset:

  1. Filenames: colons replaced with underscores (Windows can't use ':')
       JSON  : 000000_india_2021-04-19T10:23:10.444124Z.jpg
       Disk  : 000000_india_2021-04-19T10_23_10.444124Z.jpg

  2. Annotations: extra 'annotations/' subfolder is kept (not stripped)
       Correct: annotations\single_frames\{fid}\annotations\object_detection.json

Run with:  python preflight_check.py
"""

import json
import random
from pathlib import Path

from utils import load_config
_cfg             = load_config()
DATASET_ROOT     = _cfg["paths"]["dataset_root"]
IMAGES_ROOT      = _cfg["paths"]["images_root"]
ANNOTATIONS_ROOT = _cfg["paths"]["annotations_root"]
TRAINVAL_JSON    = DATASET_ROOT / "trainval-frames-full.json"

N_SAMPLE = 10
SEED     = 42

REQUIRED_META_FIELDS = [
    "time_of_day", "scraped_weather", "road_type",
    "road_condition", "num_pedestrians",
]

errors = 0
def check(cond, label, detail=""):
    global errors
    icon = "[OK ]" if cond else "[ERR]"
    suffix = f"\n         -> {detail}" if detail else ""
    print(f"  {icon}  {label}{suffix}")
    if not cond: errors += 1
    return cond

def resolve_image_path(frame: dict) -> Path | None:
    """
    JSON  : single_frames/{fid}/camera_front_blur/{fid}_xxx_2021-04-19T10:23:10.444124Z.jpg
    Disk  : single_frames/{fid}/camera_front_blur/{fid}_xxx_2021-04-19T10_23_10.444124Z.jpg
    Fix   : replace ':' with '_' in the filename only.
    """
    blur = frame.get("camera_frames", {}).get("front_blur", [])
    if not blur:
        return None
    rel      = blur[0]["filepath"]           # e.g. single_frames/000000/camera_front_blur/xxx.jpg
    p        = Path(rel)
    fixed    = p.parent / p.name.replace(":", "_")   # fix Windows filename
    return IMAGES_ROOT / fixed

def resolve_annotation_path(frame: dict) -> Path | None:
    """
    JSON  : single_frames/{fid}/annotations/object_detection.json
    Disk  : ANNOTATIONS_ROOT / single_frames/{fid}/annotations/object_detection.json
    Fix   : use the JSON path as-is (inner 'annotations/' subfolder IS present on disk).
    """
    ann = frame.get("annotations", {}).get("object_detection", {})
    rel = ann.get("filepath", "")
    if not rel:
        return None
    return ANNOTATIONS_ROOT / rel   # no stripping needed

# ── 1. ROOT DIRECTORIES ───────────────────────────────────────────────────────
print("\n-- 1. Root directories --------------------------------------------------")
check(DATASET_ROOT.exists(),     "infos\\ exists",        str(DATASET_ROOT))
check(IMAGES_ROOT.exists(),      "images\\ exists",       str(IMAGES_ROOT))
check(ANNOTATIONS_ROOT.exists(), "annotations\\ exists",  str(ANNOTATIONS_ROOT))

# ── 2. FILE COUNTS ────────────────────────────────────────────────────────────
print("\n-- 2. Actual file counts ------------------------------------------------")
n_meta  = sum(1 for _ in DATASET_ROOT.glob("single_frames/*/metadata.json"))
n_jpgs  = sum(1 for _ in IMAGES_ROOT.glob("single_frames/*/camera_front_blur/*.jpg"))
n_annos = sum(1 for _ in ANNOTATIONS_ROOT.glob("single_frames/*/annotations/object_detection.json"))
print(f"  metadata.json files              : {n_meta:>8,}")
print(f"  camera_front_blur/*.jpg files    : {n_jpgs:>8,}")
print(f"  annotations/object_detection.json: {n_annos:>8,}")
check(n_meta  > 0, "metadata files found")
check(n_jpgs  > 0, "image files found")
check(n_annos > 0, "annotation files found")

# ── 3. TRAINVAL JSON ──────────────────────────────────────────────────────────
print("\n-- 3. trainval-frames-full.json -----------------------------------------")
check(TRAINVAL_JSON.exists(), "File exists")
with open(TRAINVAL_JSON) as f:
    data = json.load(f)
print(f"  train={len(data['train']):,}  val={len(data['val']):,}  "
      f"blacklisted={len(data.get('blacklisted', [])):,}")

# ── 4. PER-FRAME CHECKS ───────────────────────────────────────────────────────
print(f"\n-- 4. Sampling {N_SAMPLE} random val frames ------------------------------")
random.seed(SEED)
sample = random.sample(data["val"], N_SAMPLE)

meta_ok = img_ok = anno_ok = ped_ok = 0
tod_values = set(); weather_values = set(); road_values = set()

for frame in sample:
    fid = frame["id"]
    print(f"\n  -- Frame {fid} -------------------------------------------------")

    # metadata
    meta_path = DATASET_ROOT / frame.get("metadata_path", "")
    if check(meta_path.exists(), "metadata.json"):
        with open(meta_path) as f:
            meta = json.load(f)
        missing = [k for k in REQUIRED_META_FIELDS if k not in meta]
        if check(not missing, "All required fields present",
                 f"missing: {missing}" if missing else ""):
            meta_ok += 1
            tod  = meta.get("time_of_day", "")
            wthr = meta.get("scraped_weather", "")
            road = meta.get("road_type", "")
            nped = meta.get("num_pedestrians", 0)
            tod_values.add(tod); weather_values.add(wthr); road_values.add(road)
            if nped > 0: ped_ok += 1
            print(f"       time_of_day     = '{tod}'")
            print(f"       scraped_weather = '{wthr}'")
            print(f"       road_type       = '{road}'")
            print(f"       num_pedestrians = {nped}")

    # image
    img_path = resolve_image_path(frame)
    print(f"     image : {img_path}")
    if img_path and check(img_path.exists(), "image file exists"):
        img_ok += 1
        print(f"       size = {img_path.stat().st_size // 1024:,} KB")

    # annotation
    ann_path = resolve_annotation_path(frame)
    print(f"     anno  : {ann_path}")
    if ann_path and check(ann_path.exists(), "object_detection.json exists"):
        with open(ann_path) as f:
            ann = json.load(f)
        if check(isinstance(ann, list), "Annotation is a list"):
            ped_objs = [o for o in ann
                        if o.get("properties", {}).get("class") == "Pedestrian"
                        and not o.get("properties", {}).get("unclear", False)]
            if ped_objs:
                coords = ped_objs[0].get("geometry", {}).get("coordinates", [])
                if check(len(coords) >= 2, f"{len(ped_objs)} pedestrian box(es) with coordinates"):
                    anno_ok += 1
            else:
                print(f"       [WRN] No Pedestrian objects (normal if 0 peds)")
                anno_ok += 1

# ── 5. UNIQUE VALUES ──────────────────────────────────────────────────────────
print("\n-- 5. Unique metadata values seen ---------------------------------------")
print(f"  time_of_day     : {sorted(tod_values)}")
print(f"  scraped_weather : {sorted(weather_values)}")
print(f"  road_type       : {sorted(road_values)}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREFLIGHT SUMMARY")
print("=" * 60)
print(f"  Frames sampled          : {N_SAMPLE}")
print(f"  Metadata readable       : {meta_ok} / {N_SAMPLE}")
print(f"  Images accessible       : {img_ok} / {N_SAMPLE}")
print(f"  Annotations parseable   : {anno_ok} / {N_SAMPLE}")
print(f"  Frames with pedestrians : {ped_ok} / {N_SAMPLE}")
print(f"  Total errors            : {errors}")
if errors == 0:
    print(f"\n  [OK] All checks passed — safe to run create_subset.py")
else:
    print(f"\n  [ERR] {errors} check(s) failed")
print("=" * 60 + "\n")
