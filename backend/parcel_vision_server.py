"""
parcel_vision_server.py  –  ParcelVision fully integrated backend
==================================================================
Combines server.py (YOLO-World detection + tracking + scene state)
with depth_server.py (DepthAnythingV2 / MiDaS + point cloud measurement)
into a single process.

Endpoints:
  GET  /           health check
  GET  /health     depth model status
  POST /detect     { multipart image } → { scene: [...] }
  GET  /scene      current scene snapshot

Run:
  python parcel_vision_server.py
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
from flask_cors import CORS
from ultralytics import YOLO

from tracker import ObjectTracker
from scene_state import SceneStateManager


# ─────────────────────────────────────────────────────────────────────────────
# DepthAnything V2 import
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, "./Depth-Anything-V2")
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    _USE_DA2 = True
    print("[server] DepthAnything V2 import OK")
except ImportError as _e:
    _USE_DA2 = False
    print(f"[server] DepthAnything V2 not found ({_e}) – will try MiDaS")

try:
    import timm  # noqa: F401
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CONF_THRESHOLD   = 0.3
JPEG_QUALITY     = 70

# Camera intrinsics — tune to your camera
CAMERA_FX   = 525.0
CAMERA_FY   = 525.0
CAMERA_CX   = 320.0
CAMERA_CY   = 240.0
DEPTH_SCALE = 4.0     # relative-to-metric scale factor


# ─────────────────────────────────────────────────────────────────────────────
# Flask setup
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


# ─────────────────────────────────────────────────────────────────────────────
# Depth model state
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
            print(f"[server] DA2 vits loaded on {_device}")
            return
        except Exception:
            print("[server] DA2 load failed:")
            traceback.print_exc()
            print("[server] Falling back to MiDaS")

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
            print(f"[server] MiDaS_small loaded on {_device}")
            return
        except Exception:
            print("[server] MiDaS load failed:")
            traceback.print_exc()

    # ── Dummy ─────────────────────────────────────────────────────────────────
    _model_type = "dummy"
    print("[server] WARNING: using dummy depth model (horizontal gradient)")


# ─────────────────────────────────────────────────────────────────────────────
# Depth inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer_depth(bgr: np.ndarray) -> np.ndarray:
    """Return float32 depth map in metres, H x W."""
    _load_depth_model()
    h, w = bgr.shape[:2]

    if _model_type == "da2":
        with torch.no_grad():
            depth = _depth_model.infer_image(bgr)
        depth = np.asarray(depth, dtype=np.float32)
        d_min, d_max = float(depth.min()), float(depth.max())
        depth = (depth - d_min) / (d_max - d_min + 1e-8)
        return (depth * DEPTH_SCALE).astype(np.float32)

    if _model_type == "midas":
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = _depth_transform(rgb).to(_device)
        with torch.no_grad():
            pred = _depth_model(inp)
        pred = F.interpolate(
            pred.unsqueeze(1), size=(h, w),
            mode="bicubic", align_corners=False
        ).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)
        depth = 1.0 / (depth + 1e-8)
        d_min, d_max = float(depth.min()), float(depth.max())
        depth = (depth - d_min) / (d_max - d_min + 1e-8)
        return (depth * DEPTH_SCALE).astype(np.float32)

    # Dummy: horizontal gradient 0.5 … 4.5 m
    row = np.linspace(0.5, 4.5, w, dtype=np.float32)
    return np.tile(row, (h, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pixels_to_3d(us: np.ndarray, vs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    X = (us - CAMERA_CX) * zs / CAMERA_FX
    Y = (vs - CAMERA_CY) * zs / CAMERA_FY
    return np.stack([X, Y, zs], axis=-1)   # (N, 3)


def _filter_point_cloud(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 10:
        return pts

    centroid  = pts.mean(axis=0)
    dists     = np.linalg.norm(pts - centroid, axis=1)
    threshold = dists.mean() + 2.5 * dists.std()
    pts = pts[dists < threshold]

    if len(pts) > 20:
        pts = pts[pts[:, 1] < np.percentile(pts[:, 1], 95)]

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


def _final_confidence(det: float, seg: float,
                      depth_rel: float, track: float) -> float:
    return round(float(np.clip(
        0.30 * det + 0.30 * seg + 0.20 * depth_rel + 0.20 * track,
        0.0, 1.0
    )), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Per-object measurement pipeline
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
        print(f"[measure] id={oid} sparse mask ({len(us)} px) — skip")
        return None

    zs    = depth_map[vs, us]
    valid = zs > 0.05
    us, vs, zs = us[valid], vs[valid], zs[valid]
    if len(us) < 10:
        print(f"[measure] id={oid} only {len(us)} px after depth filter — skip")
        return None

    if len(us) > 2000:
        idx = np.random.choice(len(us), 2000, replace=False)
        us, vs, zs = us[idx], vs[idx], zs[idx]

    pts = _pixels_to_3d(us.astype(float), vs.astype(float), zs)
    pts = _filter_point_cloud(pts)
    if len(pts) < 4:
        print(f"[measure] id={oid} only {len(pts)} pts after filter — skip")
        return None

    dims = _fit_bounding_box(pts)
    conf = _final_confidence(
        det       = meta.get("confidence",         0.5),
        seg       = meta.get("segmentation_score", 0.5),
        depth_rel = _depth_confidence(zs),
        track     = meta.get("mask_stability",     0.5),
    )

    print(f"[measure] id={oid} "
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
# Mask helpers
# ─────────────────────────────────────────────────────────────────────────────

def bbox_to_mask(x1: int, y1: int, x2: int, y2: int,
                 frame_h: int, frame_w: int) -> np.ndarray:
    """
    Filled bounding-box proxy mask.
    yolov8s-world.pt produces no pixel masks, so we fill the bbox region.
    Swap in a real segmentation mask here if you switch to yolo11n-seg.pt.
    """
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    x1c  = int(np.clip(x1, 0, frame_w - 1))
    y1c  = int(np.clip(y1, 0, frame_h - 1))
    x2c  = int(np.clip(x2, 0, frame_w - 1))
    y2c  = int(np.clip(y2, 0, frame_h - 1))
    mask[y1c:y2c, x1c:x2c] = 1
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# YOLO-World + tracker setup
# ─────────────────────────────────────────────────────────────────────────────

print("Loading YOLO model...")

yolo_model = YOLO("yolov8s-world.pt")
yolo_model.to(_device)

yolo_model.set_classes([
    "sofa",
    "chair",
    "table",
    "cardboard box",
    "carton box",
    "shipping box",
    "package",
])

tracker       = ObjectTracker()
scene_manager = SceneStateManager()

frame_index = 0

torch.set_grad_enabled(False)
print("YOLO model loaded successfully")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({"message": "ParcelVision integrated server running"})


@app.route("/health", methods=["GET"])
def health():
    _load_depth_model()
    return jsonify({
        "status": "ok",
        "depth_model": _model_type,
        "device": _device,
    })


@app.route("/detect", methods=["POST"])
def detect():
    global _last_depth_map, frame_index

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    t0    = time.perf_counter()
    file  = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    orig_h, orig_w = frame.shape[:2]
    frame_resized  = cv2.resize(frame, (416, 320))

    # ── YOLO-World detection ─────────────────────────────────────────────────
    results = yolo_model(frame_resized, verbose=False)[0]

    boxes   = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confs   = results.boxes.conf.cpu().numpy()

    scale_x = orig_w / 416
    scale_y = orig_h / 320

    detections = []
    for box, cls, conf in zip(boxes, classes, confs):
        if conf < CONF_THRESHOLD:
            continue

        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        label  = yolo_model.names[int(cls)]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        area   = (x2 - x1) * (y2 - y1)

        detections.append({
            "bbox":           (x1, y1, x2, y2),
            "confidence":     float(conf),
            "center":         (cx, cy),
            "area":           area,
            "business_class": label,
        })

    # ── Tracker + scene state ─────────────────────────────────────────────────
    detections   = tracker.update(detections)
    scene_inv    = [
        {
            "object_id":      det["id"],
            "label":          det["business_class"],
            "confidence":     float(det["confidence"]),
            "bbox":           det["bbox"],
            "center":         det["center"],
            "area":           det["area"],
            "stability_score": det.get("stability_score", 0),
        }
        for det in detections
    ]
    scene_state  = scene_manager.update(scene_inv, frame_index)
    frame_index += 1

    # ── Depth inference ───────────────────────────────────────────────────────
    measurements = {}

    if scene_state:
        try:
            depth_map       = _infer_depth(frame)
            _last_depth_map = depth_map
            h_d, w_d        = depth_map.shape

            print(f"[detect] depth min={depth_map.min():.3f}m "
                  f"max={depth_map.max():.3f}m mean={depth_map.mean():.3f}m")

            # ── Per-object measurement ────────────────────────────────────────
            for obj in scene_state.values():
                x1, y1, x2, y2 = obj["bbox"]
                mask_np = bbox_to_mask(x1, y1, x2, y2, orig_h, orig_w)

                # Resize mask to depth map resolution if needed
                if mask_np.shape != (h_d, w_d):
                    mask_np = cv2.resize(
                        mask_np, (w_d, h_d), interpolation=cv2.INTER_NEAREST
                    )

                meta = {
                    "object_id":          obj["id"],
                    "class":              obj["label"],
                    "confidence":         obj["confidence"],
                    "segmentation_score": obj.get("stability", obj["confidence"]),
                    "mask_stability":     obj.get("stability", 0.5),
                }

                m = _measure_object(depth_map, mask_np, meta)
                if m is not None:
                    measurements[obj["id"]] = m

        except Exception:
            print("[detect] depth/measurement error:")
            traceback.print_exc()

    # ── Merge measurements into scene state ───────────────────────────────────
    for obj in scene_state.values():
        oid = obj["id"]
        if oid in measurements:
            m = measurements[oid]
            obj["dimensions"] = {
                "length":    m["length"],
                "width":     m["width"],
                "height":    m["height"],
                "depth":     m["width"],
                "volume_m3": m["volume_m3"],
            }
            obj["measurement_confidence"] = m["confidence"]
            obj["mean_depth_m"]           = m.get("mean_depth_m")
        else:
            obj.setdefault("dimensions",             None)
            obj.setdefault("measurement_confidence", None)
            obj.setdefault("mean_depth_m",           None)

    ms = round((time.perf_counter() - t0) * 1000, 1)
    print(f"[detect] done {ms}ms | "
          f"objects={len(scene_state)} measured={len(measurements)}")

    return jsonify({"scene": list(scene_state.values())})


@app.route("/scene", methods=["GET"])
def get_scene():
    return jsonify({"scene": list(scene_manager.objects.values())})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ParcelVision Integrated Server")
    print(f"  device      : {_device}")
    print("  GET  /health")
    print("  POST /detect")
    print("  GET  /scene")
    print("=" * 60)
    _load_depth_model()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
