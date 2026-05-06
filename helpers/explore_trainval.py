"""
Explore and understand the structure of trainval-frames-full.json
Usage: python explore_trainval.py
"""

import json
import os
from pathlib import Path
from pprint import pprint

# ── CONFIG ───────────────────────────────────────────────────────────────────
JSON_PATH = r"D:\dataset\zod\frames\infos\trainval-frames-full.json"
# ─────────────────────────────────────────────────────────────────────────────


def hr(title=""):
    width = 70
    if title:
        print(f"\n{'─' * 3} {title} {'─' * (width - len(title) - 5)}")
    else:
        print("─" * width)


def explore(json_path: str):
    path = Path(json_path)
    assert path.exists(), f"File not found: {json_path}"

    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\n📄 File : {path.name}")
    print(f"📦 Size : {file_size_mb:.1f} MB")

    with open(path, "r") as f:
        data = json.load(f)

    # ── 1. TOP-LEVEL STRUCTURE ───────────────────────────────────────────────
    hr("1. Top-level keys")
    print(f"Type     : {type(data).__name__}")
    if isinstance(data, dict):
        print(f"Keys     : {list(data.keys())}")
        for key, value in data.items():
            if isinstance(value, list):
                print(f"  '{key}' → list of {len(value)} items")
            elif isinstance(value, dict):
                print(f"  '{key}' → dict with {len(value)} keys")
            else:
                print(f"  '{key}' → {type(value).__name__}: {value}")
    elif isinstance(data, list):
        print(f"Length   : {len(data)} items")

    # ── 2. SPLIT COUNTS ─────────────────────────────────────────────────────
    hr("2. Split counts (train / val / blacklisted)")
    if isinstance(data, dict):
        for split, items in data.items():
            if isinstance(items, list):
                print(f"  {split:15s}: {len(items):6,} frames")

    # ── 3. SINGLE FRAME ENTRY ────────────────────────────────────────────────
    hr("3. Full structure of first frame entry")
    # grab first available frame from whichever split comes first
    first_frame = None
    if isinstance(data, dict):
        for split, items in data.items():
            if isinstance(items, list) and items:
                first_frame = items[0]
                print(f"  (taken from split: '{split}')\n")
                break
    elif isinstance(data, list):
        first_frame = data[0]

    if first_frame:
        pprint(first_frame, width=100, sort_dicts=False)

    # ── 4. ALL TOP-LEVEL KEYS IN A FRAME ────────────────────────────────────
    hr("4. Keys inside a frame entry")
    if first_frame and isinstance(first_frame, dict):
        for key, val in first_frame.items():
            val_type = type(val).__name__
            if isinstance(val, dict):
                print(f"  {key:30s} → dict  {list(val.keys())}")
            elif isinstance(val, list):
                print(f"  {key:30s} → list  ({len(val)} items)")
            elif isinstance(val, str) and len(val) > 60:
                print(f"  {key:30s} → {val_type}  '{val[:60]}...'")
            else:
                print(f"  {key:30s} → {val_type}  {repr(val)}")

    # ── 5. ANNOTATIONS BREAKDOWN ─────────────────────────────────────────────
    hr("5. Annotations block")
    if first_frame and "annotations" in first_frame:
        ann = first_frame["annotations"]
        print(f"  Type   : {type(ann).__name__}")
        if isinstance(ann, dict):
            for project, ann_info in ann.items():
                print(f"\n  AnnotationProject: '{project}'")
                pprint(ann_info, indent=4, width=100)
        elif isinstance(ann, list):
            print(f"  Length : {len(ann)}")
            pprint(ann[0] if ann else {}, indent=4, width=100)

    # ── 6. CAMERA FRAMES BREAKDOWN ───────────────────────────────────────────
    hr("6. Camera frames block")
    cam_key = next(
        (k for k in (first_frame or {}) if "camera" in k.lower()), None
    )
    if cam_key:
        cam_data = first_frame[cam_key]
        print(f"  Key    : '{cam_key}'")
        print(f"  Type   : {type(cam_data).__name__}")
        if isinstance(cam_data, dict):
            for cam_name, frames in cam_data.items():
                print(f"\n  Camera : '{cam_name}'  ({len(frames)} frame(s))")
                pprint(frames[0] if frames else {}, indent=4, width=100)
    elif first_frame and "camera_frames" not in first_frame:
        print("  (no 'camera_frames' key found — check key name above)")

    # ── 7. LIDAR FRAMES ──────────────────────────────────────────────────────
    hr("7. Lidar frames block")
    lidar_key = next(
        (k for k in (first_frame or {}) if "lidar" in k.lower()), None
    )
    if lidar_key:
        lidar_data = first_frame[lidar_key]
        print(f"  Key    : '{lidar_key}'")
        if isinstance(lidar_data, dict):
            for lidar_name, frames in lidar_data.items():
                n = len(frames) if isinstance(frames, list) else "?"
                sample = frames[0] if isinstance(frames, list) and frames else frames
                print(f"\n  Lidar  : '{lidar_name}'  ({n} frame(s))")
                pprint(sample, indent=4, width=100)

    # ── 8. PATH SAMPLES ──────────────────────────────────────────────────────
    hr("8. All file paths found in first frame")
    def extract_paths(obj, results=None):
        if results is None:
            results = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("filepath", "path") or (isinstance(v, str) and ("/" in v or "\\" in v)):
                    results.append((k, v))
                else:
                    extract_paths(v, results)
        elif isinstance(obj, list):
            for item in obj:
                extract_paths(item, results)
        return results

    if first_frame:
        paths = extract_paths(first_frame)
        seen = set()
        for k, v in paths:
            if v not in seen and isinstance(v, str):
                seen.add(v)
                print(f"  [{k}]  {v}")

    # ── 9. VERIFY PATHS EXIST (against a root) ───────────────────────────────
    hr("9. Path resolution check")
    # Guess possible roots to test against
    candidate_roots = [
        path.parent,           # .../infos/
        path.parent.parent,    # .../frames/
    ]
    sample_paths = [v for k, v in extract_paths(first_frame or {})][:3]
    for root in candidate_roots:
        resolved = [root / p for p in sample_paths]
        exists = [p.exists() for p in resolved]
        pct = sum(exists) / len(exists) * 100 if exists else 0
        print(f"\n  Root: {root}")
        print(f"  → {sum(exists)}/{len(exists)} sample paths exist ({pct:.0f}%)")
        for p, e in zip(resolved, exists):
            icon = "✅" if e else "❌"
            print(f"    {icon}  {p}")

    hr()
    print("\n✅ Done. Use the path resolution results above to set dataset_root.\n")


if __name__ == "__main__":
    explore(JSON_PATH)
