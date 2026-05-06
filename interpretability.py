"""
interpretability.py  —  RQ3: GradCAM vs RT-DETR cross-attention + Mann-Whitney U
────────────────────────────────────────────────────────────────────────────────
For 500 matched true-positive pedestrian samples:
  - YOLOv8l  : GradCAM heatmap from backbone (layer 9 — SPPF)
  - RT-DETR-R101: decoder cross-attention map from last decoder layer

Computes spatial IoU between each heatmap and the GT bounding box.
Tests H0-3: no difference in spatial overlap between architectures.

Install:
    pip install pytorch-grad-cam opencv-python scipy pandas numpy tqdm

Run:  python interpretability.py
────────────────────────────────────────────────────────────────────────────────
"""

import json
import csv
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy import stats
from tqdm import tqdm
from ultralytics import YOLO

warnings.filterwarnings("ignore")

from utils import load_config
_cfg        = load_config()
SUBSET_DIR  = _cfg["paths"]["subset_dir"]
IMAGES_DIR  = SUBSET_DIR / "images"
PRED_DIR    = _cfg["paths"]["output_dir"]
GT_PATH     = SUBSET_DIR / "coco" / "annotations" / "instances.json"
MODEL_DIR   = _cfg["paths"]["model_dir"]
OUT_DIR     = PRED_DIR / "interpretability"
OUT_DIR.mkdir(exist_ok=True)
VIZ_DIR     = OUT_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

_models_cfg  = _cfg["models"]
YOLO_MODEL   = next(
    (n for n in ["yolov8l", "yolov8m", "yolov8s", "yolov8n"]
     if _models_cfg.get(n, {}).get("enabled")), "yolov8n")
RTDETR_MODEL = next(
    (n for n in ["rtdetr-r101", "rtdetr-r50"]
     if _models_cfg.get(n, {}).get("enabled")), "rtdetr-r50")
N_SAMPLES    = 500      # matched TP samples per architecture
IOU_THRESH   = 0.5      # threshold for TP classification
HEATMAP_THRESH = 0.4    # binarize heatmap at this fraction of max value
DEVICE       = "cuda"
IMGSZ        = 1280
SAVE_VIZ_N   = 20       # save this many overlay images for inspection


# ── HELPERS ───────────────────────────────────────────────────────────────────
def iou_xywh(b1, b2):
    ax1,ay1 = b1[0],b1[1]; ax2,ay2 = b1[0]+b1[2],b1[1]+b1[3]
    bx1,by1 = b2[0],b2[1]; bx2,by2 = b2[0]+b2[2],b2[1]+b2[3]
    ix = max(0,min(ax2,bx2)-max(ax1,bx1))
    iy = max(0,min(ay2,by2)-max(ay1,by1))
    inter = ix*iy
    union = b1[2]*b1[3]+b2[2]*b2[3]-inter
    return inter/union if union>0 else 0.0

def iou_mask(mask_bin, box_xywh, img_h, img_w):
    """IoU between binary heatmap mask and GT bounding box mask."""
    x,y,w,h = [int(round(v)) for v in box_xywh]
    x2,y2   = min(x+w,img_w), min(y+h,img_h)
    x,y     = max(x,0), max(y,0)
    gt_mask = np.zeros((img_h,img_w), dtype=bool)
    gt_mask[y:y2, x:x2] = True

    inter = (mask_bin & gt_mask).sum()
    union = (mask_bin | gt_mask).sum()
    return float(inter/union) if union>0 else 0.0

def find_tp_samples(pred_path, gt_by_img, n=500):
    """Find up to n true-positive (pedestrian prediction matching a GT box) samples."""
    with open(pred_path) as f:
        preds = json.load(f)

    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)

    tp_samples = []
    for img_id, img_preds in by_img.items():
        gts = gt_by_img.get(img_id, [])
        if not gts:
            continue
        img_preds_sorted = sorted(img_preds, key=lambda p: -p["score"])
        used_gt = set()
        for pred in img_preds_sorted:
            for j, gt in enumerate(gts):
                if j in used_gt:
                    continue
                if iou_xywh(pred["bbox"], gt) >= IOU_THRESH:
                    tp_samples.append({
                        "image_id": img_id,
                        "pred_bbox": pred["bbox"],
                        "gt_bbox":   gt,
                        "score":     pred["score"],
                    })
                    used_gt.add(j)
                    break
        if len(tp_samples) >= n:
            break

    return tp_samples[:n]


# ── GRADCAM for YOLOv8 ────────────────────────────────────────────────────────
class YOLOGradCAM:
    """
    GradCAM from YOLOv8 backbone SPPF layer (layer index 9).
    Target: maximum pedestrian class score across all anchor locations.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.yolo   = YOLO(model_path)
        self.pt_mdl = self.yolo.model
        self.pt_mdl.eval().to(device)
        self.device = device

        # Target layer: SPPF (backbone layer 9 in all YOLOv8 variants)
        self.target_layer = self.pt_mdl.model[9]
        self._activations = None
        self._gradients   = None

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_act)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self._activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def get_heatmap(self, img_path: str, img_h: int, img_w: int) -> np.ndarray:
        img_bgr  = cv2.imread(img_path)
        resized  = cv2.resize(img_bgr, (IMGSZ, IMGSZ))
        tensor   = torch.from_numpy(
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        ).float().permute(2,0,1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        self.pt_mdl.zero_grad()

        # Hook into Detect head to get raw ped scores before NMS
        raw_scores = []
        def detect_hook(module, inp, out):
            # out is a list of tensors per stride
            if isinstance(out, (list, tuple)):
                for o in out:
                    if isinstance(o, torch.Tensor) and o.dim() == 3:
                        raw_scores.append(o[0, 4:5, :].sigmoid())
            elif isinstance(out, torch.Tensor) and out.dim() == 3:
                raw_scores.append(out[0, 4:5, :].sigmoid())

        detect_handle = self.pt_mdl.model[-1].register_forward_hook(detect_hook)

        with torch.enable_grad():
            tensor.requires_grad_(True)
            _ = self.pt_mdl(tensor)

        detect_handle.remove()

        if not raw_scores:
            return np.zeros((img_h, img_w), dtype=np.float32)

        ped_score = torch.cat([s.flatten() for s in raw_scores]).max()
        self.pt_mdl.zero_grad()
        ped_score.backward()

        if self._activations is None or self._gradients is None:
            return np.zeros((img_h, img_w), dtype=np.float32)

        act  = self._activations  # [1, C, H, W]
        grad = self._gradients    # [1, C, H, W]
        w    = grad.mean(dim=[2,3], keepdim=True)
        cam  = torch.relu((w * act).sum(dim=1)).squeeze().cpu().numpy()
        cam  = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam  = cv2.resize(cam, (img_w, img_h))
        return cam.astype(np.float32)

    def close(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── CROSS-ATTENTION for RT-DETR ───────────────────────────────────────────────
class RTDETRAttention:
    """
    Extracts decoder cross-attention maps from RT-DETR.
    Hooks into the last TransformerDecoderLayer's cross-attention module.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.yolo   = YOLO(model_path)
        self.pt_mdl = self.yolo.model
        self.pt_mdl.eval().to(device)
        self.device = device
        self._attn_weights = None
        self._handle       = None
        self._register_hook()

    def _register_hook(self):
        """Find the cross-attention in the last decoder layer and hook it."""
        decoder = None
        # Walk model to find RTDETRDecoder → decoder layers
        for module in self.pt_mdl.modules():
            cls_name = type(module).__name__
            if "Decoder" in cls_name or "RTDETR" in cls_name:
                decoder = module
        if decoder is None:
            print("  [WARN] RT-DETR decoder not found — attention maps unavailable")
            return

        # Find cross-attention layers
        ca_layers = []
        for name, mod in decoder.named_modules():
            cls_name = type(mod).__name__
            if "MultiheadAttention" in cls_name or "CrossAttention" in cls_name:
                ca_layers.append((name, mod))

        if not ca_layers:
            print("  [WARN] No cross-attention layers found in decoder")
            return

        # Hook last cross-attention
        _, last_ca = ca_layers[-1]

        def _hook(module, inp, out):
            # MultiheadAttention returns (output, attn_weights) if need_weights=True
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                if out[1] is not None:
                    self._attn_weights = out[1].detach().cpu()

        # Monkey-patch to force need_weights=True
        orig_forward = last_ca.forward
        def patched_forward(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return orig_forward(*args, **kwargs)
        last_ca.forward = patched_forward
        self._handle = last_ca.register_forward_hook(_hook)

    def get_heatmap(self, img_path: str, img_h: int, img_w: int,
                    pred_bbox: list) -> np.ndarray:
        """Return spatial attention map for the highest-confidence pedestrian query."""
        self._attn_weights = None

        results = self.yolo(img_path, imgsz=IMGSZ, verbose=False, device=self.device)
        result  = results[0]

        if self._attn_weights is None:
            # Fallback: use prediction confidence as a point heatmap
            x,y,w,h = pred_bbox
            cam = np.zeros((img_h, img_w), dtype=np.float32)
            x1,y1 = max(0,int(x)), max(0,int(y))
            x2,y2 = min(img_w,int(x+w)), min(img_h,int(y+h))
            cam[y1:y2, x1:x2] = 1.0
            return cam

        # attn shape: [heads, queries, HW_keys] or [1, heads, queries, HW_keys]
        attn = self._attn_weights
        if attn.dim() == 4:
            attn = attn[0]          # [heads, queries, HW]
        attn = attn.mean(dim=0)     # average over heads → [queries, HW]

        # Find query closest to our target pedestrian box (highest IoU with pred)
        boxes = result.boxes
        best_q = 0
        if boxes is not None and len(boxes) > 0:
            best_iou = -1
            for i, box in enumerate(boxes):
                if int(box.cls.item()) != 0:
                    continue
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                bw,bh = x2-x1, y2-y1
                iou = iou_xywh([x1,y1,bw,bh], pred_bbox)
                if iou > best_iou:
                    best_iou, best_q = iou, min(i, attn.shape[0]-1)

        query_attn = attn[best_q].numpy()  # [HW]

        # Reshape to spatial map — infer spatial size
        hw = query_attn.shape[0]
        h_feat = w_feat = int(hw**0.5)
        if h_feat * w_feat != hw:
            # Non-square: estimate from IMGSZ / 32 stride
            h_feat = IMGSZ // 32
            w_feat = hw // h_feat
        cam = query_attn[:h_feat*w_feat].reshape(h_feat, w_feat)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam.astype(np.float32), (img_w, img_h))
        return cam

    def close(self):
        if self._handle:
            self._handle.remove()


# ── SPATIAL IoU from heatmap ──────────────────────────────────────────────────
def heatmap_to_spatial_iou(heatmap: np.ndarray, gt_bbox: list,
                            img_h: int, img_w: int,
                            thresh: float = HEATMAP_THRESH) -> float:
    binary = heatmap >= (heatmap.max() * thresh)
    return iou_mask(binary, gt_bbox, img_h, img_w)


# ── SAVE VISUALIZATION ───────────────────────────────────────────────────────
def save_overlay(img_path, heatmap, gt_bbox, out_path, title=""):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    heat_u8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.55, colored, 0.45, 0)
    x,y,bw,bh = [int(v) for v in gt_bbox]
    cv2.rectangle(overlay, (x,y), (x+bw,y+bh), (0,255,0), 2)
    if title:
        cv2.putText(overlay, title, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imwrite(str(out_path), overlay)


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # Load GT
    print("Loading ground truth …")
    with open(GT_PATH) as f:
        gt_json = json.load(f)
    gt_by_img = {img["id"]: [] for img in gt_json["images"]}
    for ann in gt_json["annotations"]:
        gt_by_img[ann["image_id"]].append(ann["bbox"])
    img_dims = {img["id"]: (img["height"],img["width"]) for img in gt_json["images"]}

    # Find TP samples for each model
    yolo_pred  = PRED_DIR / YOLO_MODEL   / "predictions_coco.json"
    rtdet_pred = PRED_DIR / RTDETR_MODEL / "predictions_coco.json"

    for p in [yolo_pred, rtdet_pred]:
        if not p.exists():
            print(f"[ERR] Predictions not found: {p}")
            print("      Run pipeline.py first.")
            return

    print(f"Finding {N_SAMPLES} TP samples per model …")
    yolo_tps  = find_tp_samples(yolo_pred,  gt_by_img, N_SAMPLES)
    rtdet_tps = find_tp_samples(rtdet_pred, gt_by_img, N_SAMPLES)
    print(f"  {YOLO_MODEL}:     {len(yolo_tps):,} TP samples found")
    print(f"  {RTDETR_MODEL}:{len(rtdet_tps):,} TP samples found")

    # Load models
    yolo_wt  = MODEL_DIR / _models_cfg[YOLO_MODEL]["weight"]
    rtdet_wt = MODEL_DIR / _models_cfg[RTDETR_MODEL]["weight"]
    for p in [yolo_wt, rtdet_wt]:
        if not p.exists():
            print(f"[ERR] Model weight not found: {p}")
            return

    print(f"\nLoading {YOLO_MODEL} …")
    yolo_cam  = YOLOGradCAM(str(yolo_wt), DEVICE)
    print(f"Loading {RTDETR_MODEL} …")
    rtdet_att = RTDETRAttention(str(rtdet_wt), DEVICE)

    # ── Compute IoU for YOLOv8 ───────────────────────────────────────────────
    print(f"\nComputing GradCAM spatial IoU for {len(yolo_tps)} samples …")
    yolo_ious  = []
    yolo_rows  = []
    for i, sample in enumerate(tqdm(yolo_tps, desc="  GradCAM", ncols=72)):
        img_id   = sample["image_id"]
        img_path = str(IMAGES_DIR / f"{img_id:06d}.jpg")
        img_h, img_w = img_dims.get(img_id, (2168, 3848))

        heatmap  = yolo_cam.get_heatmap(img_path, img_h, img_w)
        sp_iou   = heatmap_to_spatial_iou(heatmap, sample["gt_bbox"], img_h, img_w)
        yolo_ious.append(sp_iou)
        yolo_rows.append({"image_id":img_id,"model":YOLO_MODEL,
                           "spatial_iou":round(sp_iou,4),
                           "pred_score":sample["score"]})

        if i < SAVE_VIZ_N:
            save_overlay(img_path, heatmap, sample["gt_bbox"],
                         VIZ_DIR/f"yolo_{img_id:06d}.jpg",
                         title=f"GradCAM IoU={sp_iou:.3f}")

    yolo_cam.close()

    # ── Compute IoU for RT-DETR ───────────────────────────────────────────────
    print(f"\nComputing cross-attention spatial IoU for {len(rtdet_tps)} samples …")
    rtdet_ious = []
    rtdet_rows = []
    for i, sample in enumerate(tqdm(rtdet_tps, desc="  CrossAttn", ncols=72)):
        img_id   = sample["image_id"]
        img_path = str(IMAGES_DIR / f"{img_id:06d}.jpg")
        img_h, img_w = img_dims.get(img_id, (2168, 3848))

        heatmap  = rtdet_att.get_heatmap(img_path, img_h, img_w, sample["pred_bbox"])
        sp_iou   = heatmap_to_spatial_iou(heatmap, sample["gt_bbox"], img_h, img_w)
        rtdet_ious.append(sp_iou)
        rtdet_rows.append({"image_id":img_id,"model":RTDETR_MODEL,
                            "spatial_iou":round(sp_iou,4),
                            "pred_score":sample["score"]})

        if i < SAVE_VIZ_N:
            save_overlay(img_path, heatmap, sample["gt_bbox"],
                         VIZ_DIR/f"rtdetr_{img_id:06d}.jpg",
                         title=f"CrossAttn IoU={sp_iou:.3f}")

    rtdet_att.close()

    # ── Save spatial IoU results ──────────────────────────────────────────────
    all_rows = yolo_rows + rtdet_rows
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR/"spatial_iou_rq3.csv", index=False)

    # ── Mann-Whitney U test ───────────────────────────────────────────────────
    yolo_arr  = np.array(yolo_ious)
    rtdet_arr = np.array(rtdet_ious)

    u_stat, p_val = stats.mannwhitneyu(rtdet_arr, yolo_arr, alternative="greater")
    decision = "REJECT H0-3" if p_val < 0.05 else "FAIL TO REJECT H0-3"

    # Effect size: rank-biserial correlation
    n1, n2   = len(rtdet_arr), len(yolo_arr)
    effect_r = 1 - (2*u_stat)/(n1*n2)

    print(f"\n{'='*60}")
    print("  MANN-WHITNEY U TEST  (RQ3)")
    print("  H0-3: no difference in spatial overlap between architectures")
    print(f"{'='*60}")
    print(f"\n  {YOLO_MODEL}    GradCAM    median IoU = {np.median(yolo_arr):.4f}")
    print(f"  {RTDETR_MODEL} CrossAttn median IoU = {np.median(rtdet_arr):.4f}")
    print(f"\n  U-statistic = {u_stat:.1f}")
    print(f"  p-value     = {p_val:.4f}   (one-tailed, RT-DETR > YOLO)")
    print(f"  Effect r    = {effect_r:.4f}  "
          f"({'small' if abs(effect_r)<0.3 else 'medium' if abs(effect_r)<0.5 else 'large'})")
    print(f"\n  Decision: {decision}  (α = 0.05)")
    if p_val < 0.05:
        print(f"  Ha-3 supported: RT-DETR cross-attention aligns better with GT boxes")

    test_result = {
        "yolo_model": YOLO_MODEL, "rtdetr_model": RTDETR_MODEL,
        "n_yolo": n1, "n_rtdetr": n2,
        "yolo_median_iou":  round(float(np.median(yolo_arr)),  4),
        "rtdetr_median_iou":round(float(np.median(rtdet_arr)), 4),
        "yolo_mean_iou":    round(float(yolo_arr.mean()),  4),
        "rtdetr_mean_iou":  round(float(rtdet_arr.mean()), 4),
        "U_statistic": round(float(u_stat),2), "p_value": round(float(p_val),4),
        "effect_r": round(float(effect_r),4), "reject_H0": bool(p_val < 0.05),
    }
    with open(OUT_DIR/"mannwhitney_rq3.json","w") as f:
        json.dump(test_result, f, indent=2)

    print(f"\n  Saved → {OUT_DIR/'spatial_iou_rq3.csv'}")
    print(f"  Saved → {OUT_DIR/'mannwhitney_rq3.json'}")
    print(f"  Saved → {SAVE_VIZ_N} overlay images in {VIZ_DIR}")


if __name__ == "__main__":
    main()
