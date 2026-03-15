"""
depth_server.py  –  ParcelVision Stage 5-12 backend
====================================================
Exposes two endpoints:
  POST /depth   →  { "points": [[u,v], ...] }
                ←  { "depths": [z, ...] }   (meters)

  POST /measure →  { "frame_b64": "...", "objects": [...] }
                ←  { "objects": [ { object_id, label,
                                    length, width, height,
                                    volume_m3, confidence } ] }

Run:  python depth_server.py
"""

from __future__ import annotations

import base64
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from flask import Flask, jsonify, request
from PIL import Image
import sys
sys.path.append("./Depth-Anything-V2")
# ── DepthAnything v2 (small, CPU-safe) ──────────────────────────────────────
try:
    from depth_anything_v2.dpt import DepthAnythingV2   # pip install depth-anything-v2
    _USE_DA2 = True
except ImportError:
    _USE_DA2 = False
    print("[depth_server] DepthAnything v2 not found – falling back to MiDaS.")

# ── MiDaS fallback ───────────────────────────────────────────────────────────
try:
    import timm   # noqa: F401
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Camera intrinsics  (edit to match your camera / calibration)
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_FX   = 525.0   # focal length  x  (pixels)
CAMERA_FY   = 525.0   # focal length  y  (pixels)
CAMERA_CX   = 320.0   # principal point x
CAMERA_CY   = 240.0   # principal point y
DEPTH_SCALE = 4.0     # metric scale for relative-depth models (tune per scene)

# ─────────────────────────────────────────────────────────────────────────────
# Depth model loader
# ─────────────────────────────────────────────────────────────────────────────
_depth_model  = None
_depth_transform = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model_type: str = "none"
_last_depth_map: np.ndarray | None = None   # cache for legacy /depth calls


def _load_depth_model():
    global _depth_model, _depth_transform, _model_type

    if _depth_model is not None:
        return

    # ── Try DepthAnything v2 ───────────────────────────────────────────────
    if _USE_DA2:
        try:
            cfg = {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            }

            model = DepthAnythingV2(**cfg)

            # Correct way to load checkpoint
            state_dict = torch.load(
                "depth_anything_v2_vits.pth",
                map_location=_device
            )

            model.load_state_dict(state_dict)

            model.to(_device)
            model.eval()

            _depth_model = model
            _model_type = "da2"

            print(f"[depth_server] DepthAnything v2 (vits) loaded on {_device}")
            return

        except Exception as e:
            print(f"[depth_server] DA2 load failed: {e} – trying MiDaS")

    # ── Try MiDaS fallback ─────────────────────────────────────────────────
    if _HAS_TIMM:
        try:
            midas = torch.hub.load(
                "intel-isl/MiDaS",
                "MiDaS_small",
                trust_repo=True
            )

            midas.to(_device).eval()

            transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True
            )

            _depth_model = midas
            _depth_transform = transforms.small_transform
            _model_type = "midas"

            print(f"[depth_server] MiDaS_small loaded on {_device}")
            return

        except Exception as e:
            print(f"[depth_server] MiDaS load failed: {e} – using dummy depth")

    # ── Dummy fallback ─────────────────────────────────────────────────────
    _model_type = "dummy"
    print("[depth_server] WARNING: using dummy depth model (gradient plane).")


def _infer_depth(bgr_frame: np.ndarray) -> np.ndarray:
    """Return a float32 depth map in *meters*, same H×W as input."""
    _load_depth_model()
    h, w = bgr_frame.shape[:2]

    if _model_type == "da2":
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with torch.no_grad():
            depth = _depth_model.infer_image(pil)          # relative depth H×W
        depth = np.array(depth, dtype=np.float32)
        # Normalise to [0,1] then scale to metres
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = depth * DEPTH_SCALE
        return depth

    if _model_type == "midas":
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        inp = _depth_transform(rgb).to(_device)
        with torch.no_grad():
            pred = _depth_model(inp)
            pred = F.interpolate(pred.unsqueeze(1), size=(h, w),
                                 mode="bicubic", align_corners=False).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)
        # MiDaS is *inverse* depth – invert and scale
        depth = 1.0 / (depth + 1e-8)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = depth * DEPTH_SCALE
        return depth

    # dummy: linear gradient 0.5 m … 4.5 m
    depth = np.linspace(0.5, 4.5, w, dtype=np.float32)
    return np.tile(depth, (h, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (Stages 7-10)
# ─────────────────────────────────────────────────────────────────────────────

def _pixels_to_3d(us: np.ndarray, vs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Back-project pixel coordinates to 3-D camera space."""
    X = (us - CAMERA_CX) * zs / CAMERA_FX
    Y = (vs - CAMERA_CY) * zs / CAMERA_FY
    return np.stack([X, Y, zs], axis=-1)   # (N, 3)


def _filter_point_cloud(pts: np.ndarray) -> np.ndarray:
    """Statistical outlier removal + floor plane filtering.

    Coordinate conventions (camera space):
      X = rightward,  Y = downward,  Z = depth (away from camera).
    Floor removal trims the lowest 5 % of Y values.
    This is correct when the camera is roughly horizontal — i.e. Y axis
    points down in the scene.  If the camera is tilted significantly,
    reconsider which axis to trim.
    """
    if len(pts) < 10:
        return pts

    # ── Statistical outlier removal ──────────────────────────────────────────
    centroid = pts.mean(axis=0)                       # (3,) centre of mass
    dists    = np.linalg.norm(pts - centroid, axis=1) # distance of each pt from centre
    sigma    = dists.std()
    threshold = dists.mean() + 2.5 * sigma            # FIX: was mean.mean() (wrong axis)
    pts = pts[dists < threshold]

    # ── Remove floor (lowest 5 % of Y values) ───────────────────────────────
    if len(pts) > 20:
        y_thresh = np.percentile(pts[:, 1], 5)
        pts = pts[pts[:, 1] > y_thresh]

    # ── Edge trim: remove extreme 3 % in depth ───────────────────────────────
    if len(pts) > 20:
        z_lo = np.percentile(pts[:, 2], 3)
        z_hi = np.percentile(pts[:, 2], 97)
        pts = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)]

    return pts


def _fit_bounding_box(pts: np.ndarray) -> dict[str, float]:
    """Axis-aligned bounding box → length / width / height in metres."""
    if len(pts) < 4:
        return {"length": 0.0, "width": 0.0, "height": 0.0, "volume_m3": 0.0}

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()

    length = float(x_max - x_min)   # horizontal span
    height = float(y_max - y_min)   # vertical span
    width  = float(z_max - z_min)   # depth span

    # Clamp implausible values (sensor noise)
    length = np.clip(length, 0.01, 10.0)
    height = np.clip(height, 0.01, 10.0)
    width  = np.clip(width,  0.01, 10.0)

    volume = round(float(length * width * height), 4)
    return {
        "length": round(length, 3),
        "width":  round(width,  3),
        "height": round(height, 3),
        "volume_m3": volume,
    }


def _depth_confidence(depth_vals: np.ndarray) -> float:
    """Reliability score based on depth variance (low variance = high confidence)."""
    if len(depth_vals) < 2:
        return 0.5
    cv = depth_vals.std() / (depth_vals.mean() + 1e-8)
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def _final_confidence(
    detection_conf: float,
    segmentation_score: float,
    depth_rel: float,
    tracking_stability: float,
) -> float:
    score = (
        0.30 * detection_conf
        + 0.30 * segmentation_score
        + 0.20 * depth_rel
        + 0.20 * tracking_stability
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: mask + depth fusion  →  full measurement pipeline for one object
# ─────────────────────────────────────────────────────────────────────────────

def _measure_object(
    depth_map: np.ndarray,
    mask_np: np.ndarray,        # binary H×W  (0 or 1)
    obj_meta: dict[str, Any],
) -> dict[str, Any]:
    """Fuse mask with depth, build point cloud, extract 3-D dimensions."""
    h, w = depth_map.shape

    # Resize mask to match depth map
    if mask_np.shape != (h, w):
        mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

    # Stage 6 – mask-depth fusion
    binary = (mask_np > 0.5).astype(bool)
    object_depth = depth_map * binary             # zero out background

    # Pixel coordinates of object
    vs, us = np.where(binary)
    if len(us) < 10:
        return None
    zs = object_depth[vs, us]

    # Remove zero-depth pixels (background bleed)
    valid = zs > 0.05
    us, vs, zs = us[valid], vs[valid], zs[valid]
    if len(us) < 10:
        return None

    # Stage 7 – point cloud generation (sample up to 2000 pts for speed)
    MAX_PTS = 2000
    if len(us) > MAX_PTS:
        idx = np.random.choice(len(us), MAX_PTS, replace=False)
        us, vs, zs = us[idx], vs[idx], zs[idx]
    pts = _pixels_to_3d(us.astype(float), vs.astype(float), zs)

    # Stage 8 – point cloud filtering
    pts = _filter_point_cloud(pts)
    if len(pts) < 4:
        return None

    # Stage 9 & 10 – bounding box + dimensions
    dims = _fit_bounding_box(pts)

    # Stage 11 – confidence
    depth_rel   = _depth_confidence(zs)
    tracking_st = obj_meta.get("mask_stability", 0.5)
    conf = _final_confidence(
        detection_conf     = obj_meta.get("confidence", 0.5),
        segmentation_score = obj_meta.get("segmentation_score", 0.5),
        depth_rel          = depth_rel,
        tracking_stability = tracking_st,
    )

    return {
        "object_id":  obj_meta["object_id"],
        "label":      obj_meta.get("class", "unknown"),
        "length":     dims["length"],
        "width":      dims["width"],
        "height":     dims["height"],
        "volume_m3":  dims["volume_m3"],
        "confidence": conf,
        # extras (can be dropped if not needed)
        "point_count": len(pts),
        "mean_depth_m": round(float(zs.mean()), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flask endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    _load_depth_model()
    return jsonify({"status": "ok", "model": _model_type, "device": _device})


@app.route("/depth", methods=["POST"])
def endpoint_depth():
    """
    Stage 5 – per-point depth query.

    Full contract (recommended):
      Input:  { "frame_b64": "<base64 JPEG/PNG>",
                "points":    [[u, v], ...] }
      Output: { "depths": [z_meters, ...] }

    Legacy contract (frame omitted – uses last cached depth map):
      Input:  { "points": [[u, v], ...] }
      Output: { "depths": [z_meters, ...] }

    Note: always sending frame_b64 gives more accurate per-frame depth.
    If Person A cannot send the frame, it should call /measure instead.
    """
    global _last_depth_map
    try:
        data = request.get_json(force=True)

        if "frame_b64" in data:
            frame_bytes = base64.b64decode(data["frame_b64"])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                return jsonify({"error": "invalid image"}), 400
            depth_map = _infer_depth(bgr)
            _last_depth_map = depth_map          # cache for legacy callers
        elif _last_depth_map is not None:
            depth_map = _last_depth_map          # legacy: reuse last frame
        else:
            return jsonify({"error": "no frame_b64 provided and no cached depth map"}), 400

        h, w = depth_map.shape
        points = data.get("points", [])
        depths = []
        for u, v in points:
            u_c = int(np.clip(u, 0, w - 1))
            v_c = int(np.clip(v, 0, h - 1))
            depths.append(round(float(depth_map[v_c, u_c]), 3))

        return jsonify({"depths": depths})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "internal error"}), 500


@app.route("/measure", methods=["POST"])
def endpoint_measure():
    """
    Stages 6-12 – full 3-D measurement for tracked objects
    Input:
      {
        "frame_b64": "<base64 image>",
        "objects": [
          {
            "object_id": 1,
            "class": "Chair",
            "confidence": 0.87,
            "segmentation_score": 0.74,
            "mask_stability": 0.91,
            "mask_rle": {          // Run-length encoding of binary mask
              "counts": [0,120,...],
              "shape": [480, 640]
            }
          }, ...
        ]
      }
    Output (Stage 12 format):
      {
        "objects": [
          {
            "object_id": 1,
            "label": "Chair",
            "length": 0.52,
            "width": 0.55,
            "height": 0.88,
            "volume_m3": 0.251,
            "confidence": 0.79
          }, ...
        ],
        "inference_ms": 47.3
      }
    """
    try:
        t0 = time.perf_counter()
        data = request.get_json(force=True)

        # Decode frame
        frame_bytes = base64.b64decode(data["frame_b64"])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            return jsonify({"error": "invalid image"}), 400

        frame_h, frame_w = bgr.shape[:2]

        # Run depth estimation once for the whole frame
        global _last_depth_map
        depth_map = _infer_depth(bgr)   # H×W float32 metres
        _last_depth_map = depth_map     # keep for legacy /depth callers

        # Measure each tracked object
        results = []
        for obj in data.get("objects", []):
            mask_np = _decode_mask(obj.get("mask_rle"), frame_h, frame_w)
            if mask_np is None:
                continue
            measured = _measure_object(depth_map, mask_np, obj)
            if measured is not None:
                results.append(measured)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return jsonify({"objects": results, "inference_ms": elapsed_ms})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "internal error"}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Mask encoding/decoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode_mask(
    mask_rle: dict | None,
    h: int,
    w: int,
) -> np.ndarray | None:
    """Decode RLE mask dict → binary H×W numpy array."""
    if mask_rle is None:
        return None
    try:
        counts = mask_rle["counts"]
        shape  = mask_rle.get("shape", [h, w])
        flat   = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        idx    = 0
        val    = 0
        for run in counts:
            flat[idx: idx + run] = val
            idx += run
            val = 1 - val
        mask = flat.reshape(shape[0], shape[1])
        if (shape[0], shape[1]) != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask
    except Exception:
        traceback.print_exc()
        return None


def encode_mask_rle(binary_mask: np.ndarray) -> dict:
    """Encode a binary H×W numpy array as RLE (for the client side)."""
    flat = binary_mask.flatten().tolist()
    counts = []
    run    = 0
    cur    = 0
    for v in flat:
        if v == cur:
            run += 1
        else:
            counts.append(run)
            run = 1
            cur = v
    counts.append(run)
    return {"counts": counts, "shape": list(binary_mask.shape)}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ParcelVision Depth Server  (Stages 5-12)")
    print("  GET  /health")
    print("  POST /depth    – per-point depth query")
    print("  POST /measure  – full 3-D measurement pipeline")
    print("=" * 60)
    _load_depth_model()   # eager load
    app.run(host="0.0.0.0", port=5000, threaded=False)