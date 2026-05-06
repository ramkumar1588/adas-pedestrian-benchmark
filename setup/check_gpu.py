import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
"""
check_gpu.py
────────────────────────────────────────────────────────────────────────────────
Verifies your RTX 4060 is ready for inference before running run_inference.py.
Checks CUDA, PyTorch, Ultralytics, and runs a real 1-image smoke test.

Run with:  python check_gpu.py
────────────────────────────────────────────────────────────────────────────────
"""

import sys
import subprocess
from pathlib import Path

OK  = "[OK ]"
ERR = "[ERR]"
WRN = "[WRN]"
errors = 0

def check(cond, label, detail=""):
    global errors
    icon = OK if cond else ERR
    suffix = f"\n         -> {detail}" if detail else ""
    print(f"  {icon}  {label}{suffix}")
    if not cond:
        errors += 1
    return cond

# ── 1. PYTHON ─────────────────────────────────────────────────────────────────
print("\n-- 1. Python ------------------------------------------------------------")
ver = sys.version_info
check(ver >= (3, 9), f"Python >= 3.9", f"Found {ver.major}.{ver.minor}.{ver.micro}")

# ── 2. PYTORCH + CUDA ─────────────────────────────────────────────────────────
print("\n-- 2. PyTorch + CUDA ----------------------------------------------------")
try:
    import torch
    check(True, f"torch {torch.__version__} installed")

    cuda_available = torch.cuda.is_available()
    check(cuda_available, "CUDA available")

    if cuda_available:
        n_gpus = torch.cuda.device_count()
        check(n_gpus > 0, f"{n_gpus} GPU(s) detected")

        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / (1024 ** 3)
            name  = props.name
            check(True, f"GPU {i}: {name}  ({vram:.1f} GB VRAM)")

            # RTX 4060 has 8GB VRAM
            is_4060 = "4060" in name
            if is_4060:
                print(f"         -> RTX 4060 confirmed")
            else:
                print(f"  {WRN}  Expected RTX 4060 but found: {name}")

        # CUDA version
        cuda_ver = torch.version.cuda
        check(cuda_ver is not None, f"CUDA version: {cuda_ver}")

        # cuDNN
        cudnn_ver = torch.backends.cudnn.version()
        check(cudnn_ver is not None, f"cuDNN version: {cudnn_ver}")

        # Quick tensor on GPU
        try:
            t = torch.zeros(1).cuda()
            check(True, "Tensor allocation on GPU OK")
            del t
        except Exception as e:
            check(False, "Tensor allocation on GPU", str(e))

        # TF32 (RTX 30xx/40xx performance feature)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        print(f"         TF32 enabled (Ampere/Ada perf feature)")

    else:
        print(f"\n  {ERR}  CUDA not available. Most likely causes:")
        print(f"       a) PyTorch was installed WITHOUT CUDA support (CPU-only build)")
        print(f"          Fix: pip uninstall torch torchvision torchaudio")
        print(f"               pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print(f"       b) NVIDIA drivers not installed or outdated")
        print(f"          Fix: update drivers from https://www.nvidia.com/drivers")
        print(f"       c) CUDA toolkit not installed")
        print(f"          Fix: install from https://developer.nvidia.com/cuda-downloads")

except ImportError:
    check(False, "PyTorch installed",
          "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

# ── 3. ULTRALYTICS ────────────────────────────────────────────────────────────
print("\n-- 3. Ultralytics -------------------------------------------------------")
try:
    import ultralytics
    check(True, f"ultralytics {ultralytics.__version__} installed")

    # Check version is recent enough for RT-DETR
    parts = ultralytics.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    check((major, minor) >= (8, 0),
          "Version >= 8.0 (required for RT-DETR support)",
          f"Found {ultralytics.__version__}")

except ImportError:
    check(False, "Ultralytics installed", "pip install ultralytics")

# ── 4. OTHER DEPENDENCIES ─────────────────────────────────────────────────────
print("\n-- 4. Other dependencies ------------------------------------------------")
deps = {
    "tqdm":          "pip install tqdm",
    "pycocotools":   "pip install pycocotools",
    "numpy":         "pip install numpy",
    "cv2":           "pip install opencv-python",
}
for pkg, install_cmd in deps.items():
    try:
        mod = __import__(pkg if pkg != "cv2" else "cv2")
        ver = getattr(mod, "__version__", "?")
        check(True, f"{pkg} {ver}")
    except ImportError:
        check(False, f"{pkg} installed", install_cmd)

# ── 5. SMOKE TEST: real inference on one image ────────────────────────────────
print("\n-- 5. Smoke test: YOLOv8n inference on GPU ------------------------------")
from utils import load_config
SUBSET_DIR = load_config()["paths"]["subset_dir"] / "images"
sample_img = next(SUBSET_DIR.glob("*.jpg"), None) if SUBSET_DIR.exists() else None

if sample_img is None:
    print(f"  {WRN}  No images found in {SUBSET_DIR}")
    print(f"       Skipping smoke test (run create_subset.py first)")
else:
    try:
        import time
        import torch
        from ultralytics import YOLO

        print(f"       Image: {sample_img.name}")
        model = YOLO("yolov8n.pt")
        model.to("cuda")

        # Warmup (first inference is slower due to CUDA init)
        print(f"       Warming up …")
        for _ in range(3):
            _ = model(str(sample_img), verbose=False, device="cuda")

        # Timed run
        print(f"       Timing 5 runs …")
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            results = model(str(sample_img), imgsz=1280,
                            verbose=False, device="cuda")
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        result = results[0]
        speed  = result.speed

        check(avg_ms < 200,
              f"YOLOv8n inference  avg={avg_ms:.1f} ms/image (wall clock)",
              f"Expected < 200 ms on RTX 4060 at imgsz=1280")

        print(f"\n       Ultralytics speed breakdown (ms):")
        print(f"         preprocess  : {speed.get('preprocess',  0):.2f}")
        print(f"         inference   : {speed.get('inference',   0):.2f}")
        print(f"         postprocess : {speed.get('postprocess', 0):.2f}")

        # Memory usage
        mem_alloc = torch.cuda.memory_allocated()  / (1024**2)
        mem_reserv= torch.cuda.memory_reserved()   / (1024**2)
        print(f"\n       GPU memory:")
        print(f"         Allocated : {mem_alloc:.0f} MB")
        print(f"         Reserved  : {mem_reserv:.0f} MB")

        person_dets = sum(1 for b in result.boxes if int(b.cls) == 0)
        print(f"\n       Person detections in sample: {person_dets}")

    except Exception as e:
        check(False, "Smoke test", str(e))

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  GPU CHECK SUMMARY")
print("=" * 60)
print(f"  Total errors : {errors}")
if errors == 0:
    print(f"\n  {OK} RTX 4060 ready — run:  python run_inference.py")
    print(f"\n  Expected runtimes on RTX 4060 (imgsz=1280, 1250 images):")
    print(f"    YOLOv8n   :  ~10-15 min")
    print(f"    YOLOv8s   :  ~15-20 min")
    print(f"    YOLOv8m   :  ~25-35 min")
    print(f"    YOLOv8l   :  ~35-50 min")
    print(f"    RT-DETR-R50:  ~40-55 min")
    print(f"    RT-DETR-R101: ~55-75 min")
    print(f"    Total (all 6): ~3-4 hours")
    print(f"\n  Tip: run one model at a time to monitor GPU memory:")
    print(f"    python run_inference.py --models yolov8n")
else:
    print(f"\n  {ERR} Fix {errors} issue(s) above before running inference")
    print(f"\n  Quick fix for CUDA-enabled PyTorch (CUDA 12.1):")
    print(f"    pip uninstall torch torchvision torchaudio -y")
    print(f"    pip install torch torchvision torchaudio \\")
    print(f"        --index-url https://download.pytorch.org/whl/cu121")
print("=" * 60 + "\n")
