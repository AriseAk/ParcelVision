from __future__ import annotations

import sys
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

from tracker import ObjectTracker
from scene_state import SceneStateManager

from segment_anything import sam_model_registry, SamPredictor

prev_gray      = None
prev_pts       = None
motion_smooth  = 0.0
calibrating    = False
motion_accum   = 0.0
prev_dimensions = {}

# ─────────────────────────────────────────────────────────────────────────────
# DepthAnything
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, "./Depth-Anything-V2")

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    _USE_DA2 = True
except:
    _USE_DA2 = False

try:
    import timm
    _HAS_TIMM = True
except:
    _HAS_TIMM = False


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.01

# Defaults — overridden per-request by EXIF-derived values from frontend
CAMERA_FX = 554.0
CAMERA_FY = 554.0
CAMERA_CX = 320.0
CAMERA_CY = 240.0
DEPTH_SCALE = 4.0

last_detections = []

# ─────────────────────────────────────────────────────────────────────────────
# Flask
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

_device = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Depth model
# ─────────────────────────────────────────────────────────────────────────────

_depth_model     = None
_depth_transform = None
_model_type      = "none"


def _load_depth_model():
    global _depth_model, _depth_transform, _model_type
    if _depth_model is not None:
        return

    if _USE_DA2:
        try:
            model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
            state = torch.load("depth_anything_v2_vits.pth", map_location=_device)
            model.load_state_dict(state)
            model.to(_device).eval()
            _depth_model = model
            _model_type  = "da2"
            print("✅ DepthAnything V2 loaded")
            return
        except Exception as e:
            print(f"DA2 failed: {e}")

    if _HAS_TIMM:
        try:
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas.to(_device).eval()
            transforms     = torch.hub.load("intel-isl/MiDaS", "transforms")
            _depth_model   = midas
            _depth_transform = transforms.small_transform
            _model_type    = "midas"
            print("✅ MiDaS loaded")
            return
        except Exception as e:
            print(f"MiDaS failed: {e}")

    _model_type = "dummy"
    print("⚠️  Using dummy depth — measurements will be wrong!")


def _infer_depth(frame):
    _load_depth_model()
    h, w = frame.shape[:2]

    if _model_type == "da2":
        depth = _depth_model.infer_image(frame)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth * DEPTH_SCALE

    if _model_type == "midas":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = _depth_transform(rgb).to(_device)
        with torch.no_grad():
            pred = _depth_model(inp)
        pred  = F.interpolate(pred.unsqueeze(1), size=(h, w)).squeeze()
        depth = pred.cpu().numpy()
        depth = 1.0 / (depth + 1e-8)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth * DEPTH_SCALE

    return np.ones((h, w), dtype=np.float32)


def estimate_ground_depth(depth_map):
    h, w = depth_map.shape
    ground_region = depth_map[int(h * 0.75):h, :]
    return float(np.median(ground_region))


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def _pixels_to_3d(us, vs, zs, fx=None, fy=None, cx=None, cy=None):
    fx = fx or CAMERA_FX
    fy = fy or CAMERA_FY
    cx = cx or CAMERA_CX
    cy = cy or CAMERA_CY
    X  = (us - cx) * zs / fx
    Y  = (vs - cy) * zs / fy
    return np.stack([X, Y, zs], axis=-1)


def _filter_point_cloud(pts):
    if len(pts) < 10:
        return pts
    centroid = pts.mean(axis=0)
    dists    = np.linalg.norm(pts - centroid, axis=1)
    return pts[dists < dists.mean() + 2.5 * dists.std()]


def _fit_bounding_box(pts):
    if len(pts) < 4:
        return None
    l = float(pts[:, 0].max() - pts[:, 0].min())
    w = float(pts[:, 2].max() - pts[:, 2].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    return {"length": l, "width": w, "height": h, "volume_m3": l * w * h}


def _measure_object(depth_map, mask_np, meta, fx=None, fy=None):
    vs, us = np.where(mask_np > 0.5)
    if len(us) < 20:
        print("❌ Mask too small")
        return None

    zs  = depth_map[vs, us]
    pts = _pixels_to_3d(us, vs, zs, fx=fx, fy=fy)
    pts = _filter_point_cloud(pts)
    box = _fit_bounding_box(pts)
    if box is None:
        return None

    print(f"✅ Measurement: L={box['length']:.2f} W={box['width']:.2f} H={box['height']:.2f}")

    obj_id = meta["object_id"]
    alpha  = 0.7
    if obj_id in prev_dimensions:
        prev = prev_dimensions[obj_id]
        box["length"] = alpha * prev["length"] + (1 - alpha) * box["length"]
        box["width"]  = alpha * prev["width"]  + (1 - alpha) * box["width"]
        box["height"] = alpha * prev["height"] + (1 - alpha) * box["height"]

    prev_dimensions[obj_id] = box
    return {"object_id": meta["object_id"], "label": meta["class"], **box}


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

print("Loading YOLO World...")
yolo_model = YOLO("yolov8s-world.pt")
yolo_model.to(_device)
yolo_model.set_classes([
    "box", "cardboard box", "carton", "parcel",
    "package", "container", "brown box",
    "shipping box", "crate", "rectangular box",
])

print("Loading SAM...")
sam           = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(_device)
sam_predictor = SamPredictor(sam)

tracker       = ObjectTracker()
scene_manager = SceneStateManager()


# ─────────────────────────────────────────────────────────────────────────────
# Motion
# ─────────────────────────────────────────────────────────────────────────────

def compute_camera_motion(frame):
    global prev_gray, prev_pts, motion_smooth

    small = cv2.resize(frame, (320, 240))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        prev_pts  = cv2.goodFeaturesToTrack(gray, 300, 0.01, 5)
        return 0.0

    if prev_pts is None or len(prev_pts) < 10:
        prev_pts  = cv2.goodFeaturesToTrack(gray, 300, 0.01, 5)
        prev_gray = gray
        return 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
    if next_pts is None:
        prev_gray = gray
        prev_pts  = None
        return 0.0

    good_old = prev_pts[status == 1]
    good_new = next_pts[status == 1]

    if len(good_old) < 5 or len(good_new) < 5:
        prev_gray = gray
        prev_pts  = None
        return 0.0

    movement      = np.linalg.norm(good_new - good_old, axis=1)
    raw_motion    = float(np.mean(movement) * 5)
    motion_smooth = 0.8 * motion_smooth + 0.2 * raw_motion

    if np.random.rand() < 0.1:
        prev_pts = cv2.goodFeaturesToTrack(gray, 300, 0.01, 5)
    else:
        prev_pts = good_new.reshape(-1, 1, 2)

    prev_gray = gray
    return motion_smooth


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        run_detection = request.form.get("detect", "1") == "1"
        print("RUN DETECTION:", run_detection)

        file  = request.files["image"]
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid image"}), 400

        motion = compute_camera_motion(frame)
        print(f"📍 Motion: {motion:.2f}")

        global calibrating, motion_accum
        if calibrating and motion > 0.5:
            motion_accum += motion

        h, w = frame.shape[:2]

        # ── Read camera intrinsics from request ──
        camera_height = float(request.form.get("camera_height", 1.5))
        req_fx        = float(request.form.get("fx", CAMERA_FX))
        req_fy        = float(request.form.get("fy", CAMERA_FY))
        print(f"📷 fx={req_fx:.1f} fy={req_fy:.1f} cam_h={camera_height}m")

        # ── Log raw EXIF tags on first frame ──
        exif_debug = request.form.get("exif_debug")
        exif_method = request.form.get("exif_method")
        if exif_debug:
            print(f"📷 EXIF RAW: {exif_debug}")
            print(f"📷 EXIF METHOD: {exif_method}")

        # ── Detection ──
        if run_detection:
            results = yolo_model(frame, verbose=False)[0]
            boxes   = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs   = results.boxes.conf.cpu().numpy()
        else:
            boxes, classes, confs = [], [], []

        print("RAW DETECTIONS:", len(boxes))

        if run_detection:
            sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = []

            for box, cls, conf in zip(boxes, classes, confs):
                print(f"  Detected: {yolo_model.names[int(cls)]} conf={conf:.3f}")
                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx_    = (x1 + x2) / 2
                cy_    = (y1 + y2) / 2
                label  = yolo_model.names[int(cls)]

                masks, _, _ = sam_predictor.predict(
                    box=np.array([x1, y1, x2, y2]), multimask_output=False
                )
                mask_np = (masks[0] > 0.5).astype(np.uint8)

                if mask_np.sum() < 20:
                    print("⚠️  SAM weak mask → fallback bbox")
                    mask_np = np.zeros((h, w), dtype=np.uint8)
                    mask_np[y1:y2, x1:x2] = 1

                detections.append({
                    "bbox":           [x1, y1, x2, y2],
                    "confidence":     float(conf),
                    "mask":           mask_np,
                    "business_class": label,
                    "area":           float((x2 - x1) * (y2 - y1)),
                    "center":         [float(cx_), float(cy_)],
                })

        # ── Depth + scale ──
        depth_map = _infer_depth(frame)
        ground    = estimate_ground_depth(depth_map)
        print(f"GROUND DEPTH: {ground:.3f}  DEPTH MODEL: {_model_type}")
        scale = camera_height / (ground + 1e-6)

        global last_detections

        # Non-detection frames: return cached scene (safe, no numpy)
        if not run_detection:
            return jsonify({"scene": last_detections, "motion": motion})

        detections = tracker.update(detections)
        scene      = []

        for det in detections:
            m = _measure_object(depth_map, det["mask"], {
                "object_id": det["id"],
                "class":     det["business_class"],
            }, fx=req_fx, fy=req_fy)

            if m is not None:
                m["length"]   *= scale
                m["width"]    *= scale
                m["height"]   *= scale
                m["volume_m3"] = m["length"] * m["width"] * m["height"]

            scene.append({
                "object_id":  int(det["id"]),
                "label":      det["business_class"],
                "confidence": float(det["confidence"]),
                "bbox":       det["bbox"],
                "center":     det["center"],
                "dimensions": m,
            })

        # Only update cache when we got detections
        if len(scene) > 0:
            last_detections = scene

        return jsonify({"scene": last_detections, "motion": motion})

    except Exception as e:
        print("ERROR:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/start_calibration", methods=["POST"])
def start_calibration():
    global calibrating, motion_accum
    calibrating  = True
    motion_accum = 0.0
    return jsonify({"status": "started"})


@app.route("/end_calibration", methods=["POST"])
def end_calibration():
    global calibrating, motion_accum
    calibrating = False
    real_height = float(request.json.get("height", 1.2))
    if motion_accum < 1e-5:
        return jsonify({"error": "not enough motion"}), 400
    scale = real_height / motion_accum
    return jsonify({"scale": scale, "motion": motion_accum})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
