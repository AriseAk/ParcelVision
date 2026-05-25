"""
depth_server.py  –  ParcelVision Stage 5-12 backend
====================================================
Exposes:
  GET  /health
  POST /depth    { "frame_b64": "...", "points": [[u,v],...] }
              →  { "depths": [z,...] }
  POST /measure  { "frame_b64": "...", "objects": [...] }
              →  { "objects": [...], "inference_ms": float }

Run:  python depth_server.py
"""

from __future__ import annotations

import base64
import sys
import time
import traceback
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request

# ── DepthAnything V2 ──────────────────────────────────────────────────────────
sys.path.insert(0, "./Depth-Anything-V2")   # local clone takes priority
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    _USE_DA2 = True
    print("[depth_server] DepthAnything V2 import OK")
except ImportError as _e:
    _USE_DA2 = False
    print(f"[depth_server] DepthAnything V2 not found ({_e}) – will try MiDaS")

# ── MiDaS fallback ────────────────────────────────────────────────────────────
try:
    import timm  # noqa: F401
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Camera intrinsics — tune these to your camera
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_FX   = 525.0   # focal length x (pixels)
CAMERA_FY   = 525.0   # focal length y (pixels)
CAMERA_CX   = 320.0   # principal point x
CAMERA_CY   = 240.0   # principal point y
DEPTH_SCALE = 4.0     # relative-to-metric scale factor (tune per scene)

# ─────────────────────────────────────────────────────────────────────────────
# Model state
# ─────────────────────────────────────────────────────────────────────────────
_depth_model     = None
_depth_transform = None
_device          = "cuda" if torch.cuda.is_available() else "cpu"
_model_type: str = "none"
_last_depth_map: np.ndarray | None = None


def _load_depth_model() -> None:
    global _depth_model, _depth_transform, _model_type

    if _depth_model is not None:
        return

    # ── DepthAnything V2 ─────────────────────────────────────────────────────
    if _USE_DA2:
        try:
            cfg = {
                "encoder":      "vits",
                "features":     64,
                "out_channels": [48, 96, 192, 384],
            }
            model = DepthAnythingV2(**cfg)
            state = torch.load(
                "depth_anything_v2_vits.pth",
                map_location=_device,
                weights_only=True,
            )
            model.load_state_dict(state)
            model.to(_device).eval()
            _depth_model = model
            _model_type  = "da2"
            print(f"[depth_server] DA2 vits loaded on {_device}")
            return
        except Exception:
            print("[depth_server] DA2 load failed:")
            traceback.print_exc()
            print("[depth_server] Falling back to MiDaS")

    # ── MiDaS ────────────────────────────────────────────────────────────────
    if _HAS_TIMM:
        try:
            midas = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            midas.to(_device).eval()
            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            _depth_model     = midas
            _depth_transform = transforms.small_transform
            _model_type      = "midas"
            print(f"[depth_server] MiDaS_small loaded on {_device}")
            return
        except Exception:
            print("[depth_server] MiDaS load failed:")
            traceback.print_exc()

    # ── Dummy (linear gradient, offline testing only) ─────────────────────────
    _model_type = "dummy"
    print("[depth_server] WARNING: using dummy depth model")


# ─────────────────────────────────────────────────────────────────────────────
# Depth inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer_depth(bgr: np.ndarray) -> np.ndarray:
    """Return float32 depth map in metres, H x W, same spatial size as input."""
    _load_depth_model()
    h, w = bgr.shape[:2]

    if _model_type == "da2":
        # DA2's infer_image() expects a BGR numpy array (H, W, 3).
        # Do NOT convert to PIL or RGB — the method handles that internally.
        with torch.no_grad():
            depth = _depth_model.infer_image(bgr)   # returns H x W float32 relative depth
        depth = np.asarray(depth, dtype=np.float32)
        d_min, d_max = float(depth.min()), float(depth.max())
        depth = (depth - d_min) / (d_max - d_min + 1e-8)
        return (depth * DEPTH_SCALE).astype(np.float32)

    if _model_type == "midas":
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = _depth_transform(rgb).to(_device)
        with torch.no_grad():
            pred = _depth_model(inp)
        pred  = F.interpolate(
            pred.unsqueeze(1), size=(h, w),
            mode="bicubic", align_corners=False
        ).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)
        # MiDaS outputs inverse-depth — flip before normalising
        depth = 1.0 / (depth + 1e-8)
        d_min, d_max = float(depth.min()), float(depth.max())
        depth = (depth - d_min) / (d_max - d_min + 1e-8)
        return (depth * DEPTH_SCALE).astype(np.float32)

    # Dummy: horizontal gradient 0.5 … 4.5 m
    row = np.linspace(0.5, 4.5, w, dtype=np.float32)
    return np.tile(row, (h, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Geometry — Stages 7-10
# ─────────────────────────────────────────────────────────────────────────────

def _pixels_to_3d(us: np.ndarray, vs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    X = (us - CAMERA_CX) * zs / CAMERA_FX
    Y = (vs - CAMERA_CY) * zs / CAMERA_FY
    return np.stack([X, Y, zs], axis=-1)   # (N, 3)


def _filter_point_cloud(pts: np.ndarray) -> np.ndarray:
    """Statistical outlier removal + floor / depth-edge trim.

    Camera-space: X right, Y down, Z into scene.
    'Floor' = large Y values (bottom of image).
    """
    if len(pts) < 10:
        return pts

    # Outlier removal — threshold lives in distance space
    centroid  = pts.mean(axis=0)
    dists     = np.linalg.norm(pts - centroid, axis=1)
    threshold = dists.mean() + 2.5 * dists.std()
    pts = pts[dists < threshold]

    # Floor: remove highest 5 % of Y (largest Y = lowest in scene)
    if len(pts) > 20:
        pts = pts[pts[:, 1] < np.percentile(pts[:, 1], 95)]

    # Depth edge trim: remove 3 % tails
    if len(pts) > 20:
        z_lo = np.percentile(pts[:, 2], 3)
        z_hi = np.percentile(pts[:, 2], 97)
        pts  = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)]

    return pts


def _fit_bounding_box(pts: np.ndarray) -> dict[str, float]:
    if len(pts) < 4:
        return {"length": 0.0, "width": 0.0, "height": 0.0, "volume_m3": 0.0}

    length = float(np.clip(pts[:, 0].max() - pts[:, 0].min(), 0.01, 10.0))
    height = float(np.clip(pts[:, 1].max() - pts[:, 1].min(), 0.01, 10.0))
    width  = float(np.clip(pts[:, 2].max() - pts[:, 2].min(), 0.01, 10.0))
    return {
        "length":    round(length, 3),
        "width":     round(width,  3),
        "height":    round(height, 3),
        "volume_m3": round(length * width * height, 4),
    }


def _depth_confidence(zs: np.ndarray) -> float:
    if len(zs) < 2:
        return 0.5
    return float(np.clip(1.0 - zs.std() / (zs.mean() + 1e-8), 0.0, 1.0))


def _final_confidence(det: float, seg: float, depth_rel: float, track: float) -> float:
    return round(float(np.clip(
        0.30 * det + 0.30 * seg + 0.20 * depth_rel + 0.20 * track, 0.0, 1.0
    )), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Per-object pipeline — Stages 6-12
# ─────────────────────────────────────────────────────────────────────────────

def _measure_object(
    depth_map: np.ndarray,
    mask_np:   np.ndarray,
    meta:      dict[str, Any],
) -> dict[str, Any] | None:

    oid = meta.get("object_id", "?")
    h, w = depth_map.shape

    if mask_np.shape != (h, w):
        mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

    vs, us = np.where(mask_np > 0.5)
    if len(us) < 10:
        print(f"[measure_object] id={oid} sparse mask ({len(us)} px) — skip")
        return None

    zs = depth_map[vs, us]

    # Remove pixels where depth is effectively zero (mask bleed onto background)
    valid = zs > 0.05
    us, vs, zs = us[valid], vs[valid], zs[valid]
    if len(us) < 10:
        print(f"[measure_object] id={oid} only {len(us)} px after depth filter — skip")
        return None

    # Random subsample for speed
    if len(us) > 2000:
        idx = np.random.choice(len(us), 2000, replace=False)
        us, vs, zs = us[idx], vs[idx], zs[idx]

    pts = _pixels_to_3d(us.astype(float), vs.astype(float), zs)
    pts = _filter_point_cloud(pts)
    if len(pts) < 4:
        print(f"[measure_object] id={oid} only {len(pts)} pts after filter — skip")
        return None

    dims = _fit_bounding_box(pts)
    conf = _final_confidence(
        det       = meta.get("confidence",         0.5),
        seg       = meta.get("segmentation_score", 0.5),
        depth_rel = _depth_confidence(zs),
        track     = meta.get("mask_stability",     0.5),
    )

    print(f"[measure_object] id={oid} "
          f"L={dims['length']:.2f} W={dims['width']:.2f} H={dims['height']:.2f} "
          f"pts={len(pts)} conf={conf:.2f} depth_mean={zs.mean():.2f}m")

    return {
        "object_id":    oid,
        "label":        meta.get("class", "unknown"),
        "length":       dims["length"],
        "width":        dims["width"],
        "height":       dims["height"],
        "volume_m3":    dims["volume_m3"],
        "confidence":   conf,
        "point_count":  len(pts),
        "mean_depth_m": round(float(zs.mean()), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RLE decoder
# ─────────────────────────────────────────────────────────────────────────────

def _decode_rle(rle: dict | None, h: int, w: int) -> np.ndarray | None:
    if rle is None:
        return None
    try:
        counts = rle["counts"]
        sh     = rle.get("shape", [h, w])
        flat   = np.zeros(sh[0] * sh[1], dtype=np.uint8)
        idx, val = 0, 0
        for run in counts:
            flat[idx: idx + run] = val
            idx += run
            val  = 1 - val
        mask = flat.reshape(sh[0], sh[1])
        if (sh[0], sh[1]) != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask
    except Exception:
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Shared frame decoder
# ─────────────────────────────────────────────────────────────────────────────

def _decode_frame(data: dict) -> np.ndarray | None:
    raw = data.get("frame_b64")
    if not raw:
        return None
    arr = np.frombuffer(base64.b64decode(raw), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)   # None on failure


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    _load_depth_model()
    return jsonify({"status": "ok", "model": _model_type, "device": _device})


@app.route("/depth", methods=["POST"])
def endpoint_depth():
    global _last_depth_map
    try:
        data      = request.get_json(force=True)
        bgr       = _decode_frame(data)

        if bgr is not None:
            depth_map       = _infer_depth(bgr)
            _last_depth_map = depth_map
        elif _last_depth_map is not None:
            depth_map = _last_depth_map
        else:
            return jsonify({"error": "no frame_b64 and no cached depth map"}), 400

        h, w   = depth_map.shape
        depths = []
        for u, v in data.get("points", []):
            depths.append(round(float(
                depth_map[int(np.clip(v, 0, h-1)), int(np.clip(u, 0, w-1))]
            ), 3))

        return jsonify({"depths": depths})

    except Exception:
        err = traceback.format_exc()
        print("[/depth] EXCEPTION:\n" + err)
        return jsonify({"error": "internal error", "traceback": err}), 500


@app.route("/measure", methods=["POST"])
def endpoint_measure():
    global _last_depth_map
    try:
        t0   = time.perf_counter()
        data = request.get_json(force=True)

        bgr = _decode_frame(data)
        if bgr is None:
            return jsonify({"error": "missing or unreadable frame_b64"}), 400

        n_objs = len(data.get("objects", []))
        print(f"[/measure] frame {bgr.shape[1]}x{bgr.shape[0]}, {n_objs} objects")

        depth_map       = _infer_depth(bgr)
        _last_depth_map = depth_map
        h, w            = depth_map.shape

        print(f"[/measure] depth min={depth_map.min():.3f}m "
              f"max={depth_map.max():.3f}m mean={depth_map.mean():.3f}m")

        results = []
        for obj in data.get("objects", []):
            mask = _decode_rle(obj.get("mask_rle"), h, w)
            if mask is None:
                print(f"[/measure] obj {obj.get('object_id')} RLE decode failed")
                continue
            m = _measure_object(depth_map, mask, obj)
            if m is not None:
                results.append(m)

        ms = round((time.perf_counter() - t0) * 1000, 1)
        print(f"[/measure] done {ms}ms → {len(results)}/{n_objs} measured")
        return jsonify({"objects": results, "inference_ms": ms})

    except Exception:
        err = traceback.format_exc()
        print("[/measure] EXCEPTION:\n" + err)
        # Forward traceback to client so you can read it without switching terminals
        return jsonify({"error": "internal error", "traceback": err}), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ParcelVision Depth Server")
    print(f"  device  : {_device}")
    print("  GET  /health")
    print("  POST /depth")
    print("  POST /measure")
    print("=" * 60)
    _load_depth_model()
    app.run(host="0.0.0.0", port=5000, threaded=False)