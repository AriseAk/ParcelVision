from __future__ import annotations

import json
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

prev_gray       = None
prev_pts        = None
motion_smooth   = 0.0
calibrating     = False
motion_accum    = 0.0
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
CAMERA_FX      = 554.0
CAMERA_FY      = 554.0
last_detections = []

# ─────────────────────────────────────────────────────────────────────────────
# Scan session state
# ─────────────────────────────────────────────────────────────────────────────

scan_session = {
    'frames':            [],
    'active':            False,
    'prev_gray':         None,
    'prev_kp':           None,
    'prev_des':          None,
    'pose_R':            np.eye(3,     dtype=np.float64),
    'pose_t':            np.zeros((3, 1), dtype=np.float64),
    'last_scale':        1.0,
    'scale_initialized': False,
    'global_scale':      1.0,
}


def reset_scan_session():
    scan_session['frames']            = []
    scan_session['active']            = False
    scan_session['prev_gray']         = None
    scan_session['prev_kp']           = None
    scan_session['prev_des']          = None
    scan_session['pose_R']            = np.eye(3, dtype=np.float64)
    scan_session['pose_t']            = np.zeros((3, 1), dtype=np.float64)
    scan_session['last_scale']        = 1.0
    scan_session['scale_initialized'] = False
    scan_session['global_scale']      = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Flask
# ─────────────────────────────────────────────────────────────────────────────

app    = Flask(__name__)
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
            midas            = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas.to(_device).eval()
            transforms       = torch.hub.load("intel-isl/MiDaS", "transforms")
            _depth_model     = midas
            _depth_transform = transforms.small_transform
            _model_type      = "midas"
            print("✅ MiDaS loaded")
            return
        except Exception as e:
            print(f"MiDaS failed: {e}")

    _model_type = "dummy"
    print("⚠️  Using dummy depth")


def _infer_depth(frame):
    _load_depth_model()
    h, w = frame.shape[:2]

    if _model_type == "da2":
        depth = _depth_model.infer_image(frame)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype(np.float32)

    if _model_type == "midas":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = _depth_transform(rgb).to(_device)
        with torch.no_grad():
            pred = _depth_model(inp)
        pred  = F.interpolate(pred.unsqueeze(1), size=(h, w)).squeeze()
        depth = pred.cpu().numpy()
        depth = 1.0 / (depth + 1e-8)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype(np.float32)

    return np.ones((h, w), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Ground depth
#
# FIXED: depth models output DISPARITY-like values where closer = higher value.
# The floor (close to camera) has HIGH disparity.  Using the 75th percentile
# of the bottom strip gives us a stable, close-to-floor value.
# The old 20th/25th percentile was picking FAR objects in the bottom strip.
# ─────────────────────────────────────────────────────────────────────────────

def estimate_ground_depth(depth_map, detected_bboxes=None):
    h, w   = depth_map.shape
    region = depth_map[int(h * 0.75):h, :].copy().astype(np.float32)

    if detected_bboxes:
        for bbox in detected_bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            ry1 = max(0, y1 - int(h * 0.75))
            ry2 = max(0, y2 - int(h * 0.75))
            rx1 = max(0, x1)
            rx2 = min(w, x2)
            if ry2 > ry1 and rx2 > rx1:
                region[ry1:ry2, rx1:rx2] = np.nan

    valid = region[~np.isnan(region)]
    if len(valid) < 100:
        valid = depth_map[h - 10:h, :].flatten()

    # 75th percentile = high disparity = the CLOSE floor plane
    val = float(np.percentile(valid, 75))
    return max(val, 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Global scale lock
# ─────────────────────────────────────────────────────────────────────────────

def get_or_init_global_scale(depth_map, camera_height, bboxes):
    if scan_session['scale_initialized']:
        return scan_session['global_scale']
    ground = estimate_ground_depth(depth_map, bboxes)
    scale  = camera_height / ground
    scan_session['global_scale']      = scale
    scan_session['scale_initialized'] = True
    print(f"🔒 Global scale LOCKED: {scale:.4f}  ground_disp={ground:.4f}")
    return scale


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def _pixels_to_3d(us, vs, zs, fx, fy, cx, cy):
    X = (us - cx) * zs / fx
    Y = (vs - cy) * zs / fy
    return np.stack([X, Y, zs], axis=-1)


def _filter_point_cloud(pts, sigma=2.5):
    if len(pts) < 10:
        return pts
    centroid = pts.mean(axis=0)
    dists    = np.linalg.norm(pts - centroid, axis=1)
    return pts[dists < dists.mean() + sigma * dists.std()]


def _fit_bounding_box(pts):
    if len(pts) < 4:
        return None
    l = float(pts[:, 0].max() - pts[:, 0].min())
    w = float(pts[:, 2].max() - pts[:, 2].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    return {"length": l, "width": w, "height": h, "volume_m3": l * w * h}


# ─────────────────────────────────────────────────────────────────────────────
# Central mask sampling
# ─────────────────────────────────────────────────────────────────────────────

def _get_central_mask_pixels(mask_np):
    vs, us = np.where(mask_np > 0.5)
    if len(us) < 20:
        return us, vs

    u_min, u_max = us.min(), us.max()
    v_min, v_max = vs.min(), vs.max()
    u_range = u_max - u_min
    v_range = v_max - v_min

    shrink = 0.25
    u_lo = u_min + shrink * u_range
    u_hi = u_max - shrink * u_range
    v_lo = v_min + shrink * v_range
    v_hi = v_max - shrink * v_range

    if u_hi <= u_lo or v_hi <= v_lo:
        return us, vs

    inner    = (us >= u_lo) & (us <= u_hi) & (vs >= v_lo) & (vs <= v_hi)
    us_inner = us[inner]
    vs_inner = vs[inner]

    return (us_inner, vs_inner) if len(us_inner) >= 20 else (us, vs)


# ─────────────────────────────────────────────────────────────────────────────
# Dimension smoothing
# ─────────────────────────────────────────────────────────────────────────────

def smooth_dimensions(prev, new):
    if prev is None:
        return new
    alpha   = 0.4
    ratio_l = abs(new['length'] - prev['length']) / (prev['length'] + 1e-6)
    ratio_w = abs(new['width']  - prev['width'])  / (prev['width']  + 1e-6)
    ratio_h = abs(new['height'] - prev['height']) / (prev['height'] + 1e-6)
    if max(ratio_l, ratio_w, ratio_h) > 0.4:
        return new
    smoothed = {
        'length': alpha * prev['length'] + (1 - alpha) * new['length'],
        'width':  alpha * prev['width']  + (1 - alpha) * new['width'],
        'height': alpha * prev['height'] + (1 - alpha) * new['height'],
    }
    smoothed['volume_m3'] = smoothed['length'] * smoothed['width'] * smoothed['height']
    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# Object measurement  (live /detect mode)
#
# FIXED: removed z_cv gate.  Relative depth maps output disparity-like values.
# A perfectly flat surface still has high z_cv in disparity space because
# disparity is non-linearly related to depth.  The gate rejected everything.
# We now only do percentile outlier removal which is safe and model-agnostic.
# ─────────────────────────────────────────────────────────────────────────────

def _measure_object(depth_map, mask_np, meta, fx, fy, cx, cy):
    us, vs = _get_central_mask_pixels(mask_np)
    if len(us) < 20:
        print("❌ Mask too small")
        return None

    zs = depth_map[vs, us].astype(np.float32)

    # Remove top/bottom 10% outliers — no z_cv gate
    p10, p90 = np.percentile(zs, 10), np.percentile(zs, 90)
    valid    = (zs >= p10) & (zs <= p90)
    us_f     = us[valid]
    vs_f     = vs[valid]
    zs_f     = zs[valid]

    if len(us_f) < 10:
        print("❌ Too few pixels after percentile filter")
        return None

    pts = _pixels_to_3d(us_f.astype(float), vs_f.astype(float), zs_f, fx, fy, cx, cy)
    pts = _filter_point_cloud(pts)
    box = _fit_bounding_box(pts)
    if box is None:
        return None

    print(f"✅ Measurement (pre-scale): L={box['length']:.4f} W={box['width']:.4f} H={box['height']:.4f}")

    obj_id = meta["object_id"]
    box    = smooth_dimensions(prev_dimensions.get(obj_id), box)
    prev_dimensions[obj_id] = box
    return {"object_id": meta["object_id"], "label": meta["class"], **box}


# ─────────────────────────────────────────────────────────────────────────────
# Visual odometry
# ─────────────────────────────────────────────────────────────────────────────

def estimate_pose_from_frame(bgr, fx, fy, cx, cy):
    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    orb     = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(gray, None)

    if des is None or len(kp) < 20:
        scan_session['prev_gray'] = gray
        scan_session['prev_kp']   = kp
        scan_session['prev_des']  = des
        return None, None

    if scan_session['prev_gray'] is None or scan_session['prev_des'] is None:
        scan_session['prev_gray'] = gray
        scan_session['prev_kp']   = kp
        scan_session['prev_des']  = des
        # First frame — identity pose
        return np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(scan_session['prev_des'], des, k=2)

    good = [m for pair in matches if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]

    if len(good) < 15:
        scan_session['prev_gray'] = gray
        scan_session['prev_kp']   = kp
        scan_session['prev_des']  = des
        return None, None

    pts1 = np.float32([scan_session['prev_kp'][m.queryIdx].pt for m in good])
    pts2 = np.float32([kp[m.trainIdx].pt                      for m in good])

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        scan_session['prev_gray'] = gray
        scan_session['prev_kp']   = kp
        scan_session['prev_des']  = des
        return None, None

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    scan_session['prev_gray'] = gray
    scan_session['prev_kp']   = kp
    scan_session['prev_des']  = des
    return R, t


# ─────────────────────────────────────────────────────────────────────────────
# IMU
# ─────────────────────────────────────────────────────────────────────────────

def imu_delta_rotation(imu_readings):
    if not imu_readings or len(imu_readings) < 2:
        return None
    dt_total = (imu_readings[-1]['ts'] - imu_readings[0]['ts']) / 1000.0
    if dt_total <= 0:
        return None
    deg_to_rad = np.pi / 180.0
    n          = len(imu_readings)
    ax = sum(r.get('gx', 0) for r in imu_readings) * deg_to_rad * (dt_total / n)
    ay = sum(r.get('gy', 0) for r in imu_readings) * deg_to_rad * (dt_total / n)
    az = sum(r.get('gz', 0) for r in imu_readings) * deg_to_rad * (dt_total / n)
    Rx, _ = cv2.Rodrigues(np.array([ax, 0,  0]))
    Ry, _ = cv2.Rodrigues(np.array([0,  ay, 0]))
    Rz, _ = cv2.Rodrigues(np.array([0,  0,  az]))
    return Rz @ Ry @ Rx


def rotation_angle_diff(R1, R2):
    trace     = np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(trace))


def fuse_rotation(R_orb, R_imu):
    if R_imu is None:
        return R_orb
    if rotation_angle_diff(R_orb, R_imu) > 15.0:
        return R_imu
    return R_orb


# ─────────────────────────────────────────────────────────────────────────────
# Backend motion (used only by /detect)
# ─────────────────────────────────────────────────────────────────────────────

def compute_camera_motion(frame):
    global prev_gray, prev_pts, motion_smooth
    small = cv2.resize(frame, (320, 240))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if prev_gray is None or prev_pts is None or len(prev_pts) < 10:
        prev_gray = gray
        prev_pts  = cv2.goodFeaturesToTrack(gray, 300, 0.01, 5)
        return 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
    if next_pts is None:
        prev_gray = gray; prev_pts = None
        return 0.0

    good_old = prev_pts[status == 1]
    good_new = next_pts[status == 1]
    if len(good_old) < 5:
        prev_gray = gray; prev_pts = None
        return 0.0

    movement      = np.linalg.norm(good_new - good_old, axis=1)
    raw_motion    = float(np.mean(movement) * 5)
    motion_smooth = 0.8 * motion_smooth + 0.2 * raw_motion
    prev_pts      = (cv2.goodFeaturesToTrack(gray, 300, 0.01, 5)
                     if np.random.rand() < 0.1
                     else good_new.reshape(-1, 1, 2))
    prev_gray     = gray
    return motion_smooth


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
# /detect  — live single-frame estimation
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        run_detection = request.form.get("detect", "1") == "1"

        file  = request.files["image"]
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid image"}), 400

        motion = compute_camera_motion(frame)

        global calibrating, motion_accum
        if calibrating and motion > 0.5:
            motion_accum += motion

        camera_height = float(request.form.get("camera_height", 1.5))
        req_fx        = float(request.form.get("fx", CAMERA_FX))
        req_fy        = float(request.form.get("fy", CAMERA_FY))
        img_w         = int(request.form.get("img_w", frame.shape[1]))
        img_h         = int(request.form.get("img_h", frame.shape[0]))
        req_cx        = img_w / 2.0
        req_cy        = img_h / 2.0

        print(f"📷 fx={req_fx:.1f} fy={req_fy:.1f} cam_h={camera_height}m motion={motion:.2f}")

        exif_debug = request.form.get("exif_debug")
        if exif_debug:
            print(f"📷 EXIF: {exif_debug}  method={request.form.get('exif_method')}")

        h, w = frame.shape[:2]

        if run_detection:
            results = yolo_model(frame, verbose=False)[0]
            boxes   = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs   = results.boxes.conf.cpu().numpy()
        else:
            boxes, classes, confs = [], [], []

        detections = []
        if run_detection:
            sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for box, cls, conf in zip(boxes, classes, confs):
                if conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box)
                label = yolo_model.names[int(cls)]
                masks, _, _ = sam_predictor.predict(
                    box=np.array([x1, y1, x2, y2]), multimask_output=False)
                mask_np = (masks[0] > 0.5).astype(np.uint8)
                if mask_np.sum() < 20:
                    mask_np = np.zeros((h, w), dtype=np.uint8)
                    mask_np[y1:y2, x1:x2] = 1
                detections.append({
                    "bbox":           [x1, y1, x2, y2],
                    "confidence":     float(conf),
                    "mask":           mask_np,
                    "business_class": label,
                    "area":           float((x2 - x1) * (y2 - y1)),
                    "center":         [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                })

        depth_map = _infer_depth(frame)
        bboxes_for_ground = [d['bbox'] for d in detections]
        ground = estimate_ground_depth(depth_map, bboxes_for_ground)
        scale  = camera_height / ground
        print(f"GROUND DISP: {ground:.4f}  SCALE: {scale:.4f}  MODEL: {_model_type}")

        global last_detections
        if not run_detection:
            return jsonify({"scene": last_detections, "motion": motion})

        detections = tracker.update(detections)
        scene      = []

        for det in detections:
            m = _measure_object(
                depth_map, det["mask"],
                {"object_id": det["id"], "class": det["business_class"]},
                fx=req_fx, fy=req_fy, cx=req_cx, cy=req_cy,
            )
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

        if scene:
            last_detections = scene

        return jsonify({"scene": last_detections, "motion": motion})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# /start_scan
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/start_scan", methods=["POST"])
def start_scan():
    reset_scan_session()
    scan_session['active'] = True
    print("Scan session started")
    return jsonify({'status': 'scan started'})


# ─────────────────────────────────────────────────────────────────────────────
# /scan_frame
#
# KEY BUG FIXES vs previous version:
#
# BUG 1 (FRAME REJECTION) — z_cv threshold 0.35 rejected ALL frames.
#   Relative depth (DepthAnything / MiDaS) outputs disparity-like values.
#   Even a clean flat surface returns cv >> 0.35 in disparity space because
#   disparity is 1/depth — small depth changes → large disparity variance.
#   REMOVED the z_cv gate entirely.  Using only percentile outlier removal.
#
# BUG 2 (MOTION STALE) — frontend motionRef was stale during scanning.
#   The /detect loop STOPS when scanning starts.  motionRef was last set by
#   /detect response, so it stays at whatever it was at scan start (often 0).
#   Frontend was gating on stale 0 → sending NO frames.
#   FIXED in frontend: independent optical flow computed directly in scan loop,
#   not relying on /detect responses.
#   Backend: NO motion gating in scan_frame — frontend already gates.
#
# BUG 3 (WRONG PERCENTILE FOR GROUND) — 20th/25th percentile of bottom strip
#   returned FAR objects, not the floor.  Depth models output disparity where
#   CLOSER = HIGHER value.  Floor (close) = high disparity.
#   FIXED: use 75th percentile → picks high-disparity (close = floor) values.
#
# BUG 4 (BROKEN POSE ACCUMULATION) — metric_t = t * scale is dimensionally
#   wrong.  recoverPose t is unit-norm in camera coords.  scale is
#   disparity_unit / meter.  Multiplying gives nonsense translation magnitude.
#   FIXED: rotation-only accumulation for pose_R. Translation stored but not
#   used for metric positioning — the depth point clouds are the source of
#   truth for geometry, not integrated t.
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/scan_frame", methods=["POST"])
def scan_frame():
    try:
        file  = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'bad frame'}), 400

        imu_readings  = json.loads(request.form.get('imu', '[]'))
        camera_height = float(request.form.get('camera_height', 1.5))
        fx            = float(request.form.get('fx', 554))
        fy            = float(request.form.get('fy', 554))
        img_w         = int(request.form.get('img_w', frame.shape[1]))
        img_h         = int(request.form.get('img_h', frame.shape[0]))
        cx            = img_w / 2.0
        cy            = img_h / 2.0

        # BUG 2 FIX: no backend motion gate here — frontend gates independently
        # (backend motion_smooth is stale during scanning)

        # ── Detection ────────────────────────────────────────────────────
        results = yolo_model(frame, verbose=False)[0]
        boxes   = results.boxes.xyxy.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()
        bboxes  = [list(map(int, b)) for b, c in zip(boxes, confs) if c >= CONF_THRESHOLD]

        if not bboxes:
            print("⏭ Skipped: no detection")
            return jsonify({'status': 'skipped', 'reason': 'no_detection',
                            'frame_count': len(scan_session['frames'])})

        # ── Depth + global scale lock ────────────────────────────────────
        depth_map = _infer_depth(frame)
        scale     = get_or_init_global_scale(depth_map, camera_height, bboxes)
        scan_session['last_scale'] = scale

        # ── Pose estimation ──────────────────────────────────────────────
        R_orb, t = estimate_pose_from_frame(frame, fx, fy, cx, cy)

        is_first = (R_orb is not None
                    and np.allclose(R_orb, np.eye(3))
                    and np.allclose(t, 0))

        if R_orb is None or t is None:
            # Pose failed — reuse last known rotation for this frame
            current_R = scan_session['pose_R'].copy()
            current_t = scan_session['pose_t'].copy()
            print("⚠️  Pose failed — using last known pose")
        else:
            R_imu = imu_delta_rotation(imu_readings)
            R     = fuse_rotation(R_orb, R_imu)

            # BUG 4 FIX: accumulate rotation only.
            # recoverPose t is unit-norm — we can't convert it to metric
            # without a known baseline.  Point clouds carry the metric info.
            scan_session['pose_R'] = R @ scan_session['pose_R']
            # Minimal t accumulation for rough view diversity (not metric)
            if not is_first:
                scan_session['pose_t'] = (scan_session['pose_t'] +
                                          scan_session['pose_R'] @ (t * 0.03))

            current_R = scan_session['pose_R'].copy()
            current_t = scan_session['pose_t'].copy()

        # ── SAM segmentation ────────────────────────────────────────────
        x1, y1, x2, y2 = bboxes[0]
        sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        masks, _, _ = sam_predictor.predict(
            box=np.array([x1, y1, x2, y2]), multimask_output=False)
        mask_np = (masks[0] > 0.5).astype(np.uint8)

        if mask_np.sum() < 20:
            mask_np = np.zeros((img_h, img_w), dtype=np.uint8)
            mask_np[y1:y2, x1:x2] = 1

        # ── Central mask pixels ─────────────────────────────────────────
        us, vs = _get_central_mask_pixels(mask_np)
        if len(us) < 20:
            print("⏭ Skipped: mask too small after central crop")
            return jsonify({'status': 'skipped', 'reason': 'mask_too_small',
                            'frame_count': len(scan_session['frames'])})

        # ── Depth sampling — percentile only, no z_cv gate (BUG 1 FIX) ─
        zs_raw = depth_map[vs, us].astype(np.float32)
        p10, p90 = np.percentile(zs_raw, 10), np.percentile(zs_raw, 90)
        valid    = (zs_raw >= p10) & (zs_raw <= p90)
        us_c, vs_c, zs_c = us[valid], vs[valid], zs_raw[valid]

        if len(us_c) < 10:
            print("⏭ Skipped: too few depth pixels")
            return jsonify({'status': 'skipped', 'reason': 'too_few_depth_pixels',
                            'frame_count': len(scan_session['frames'])})

        # ── 3D projection → world frame ─────────────────────────────────
        zs_metric = zs_c * scale
        pts_cam   = _pixels_to_3d(us_c.astype(float), vs_c.astype(float),
                                   zs_metric, fx, fy, cx, cy)
        pts_cam   = _filter_point_cloud(pts_cam)
        pts_world = (current_R @ pts_cam.T).T + current_t.T

        scan_session['frames'].append({'pts_world': pts_world})
        n = len(scan_session['frames'])
        print(f"✅ Frame {n} accepted — scale={scale:.4f} pts={len(pts_cam)}")

        return jsonify({'status': 'ok', 'frame_count': n})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# /compute_dimensions
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/compute_dimensions", methods=["POST"])
def compute_dimensions():
    try:
        frames = scan_session['frames']
        if len(frames) < 3:
            reset_scan_session()
            return jsonify({'error': f'only {len(frames)} frames — scan more'}), 400

        all_pts = np.vstack([f['pts_world'] for f in frames])
        print(f"Fusing {len(all_pts)} pts from {len(frames)} frames")

        # Two-pass outlier removal
        for sigma in [2.5, 2.0]:
            if len(all_pts) < 20:
                break
            centroid = all_pts.mean(axis=0)
            dists    = np.linalg.norm(all_pts - centroid, axis=1)
            all_pts  = all_pts[dists < dists.mean() + sigma * dists.std()]

        print(f"After filtering: {len(all_pts)} pts")
        if len(all_pts) < 20:
            reset_scan_session()
            return jsonify({'error': 'not enough clean points'}), 400

        # PCA bounding box — sort dims so length >= width >= height
        mean      = all_pts.mean(axis=0)
        centered  = all_pts - mean
        _, eigvecs = np.linalg.eigh(np.cov(centered.T))
        projected  = centered @ eigvecs

        raw_dims = sorted([
            float(projected[:, i].max() - projected[:, i].min())
            for i in range(3)
        ], reverse=True)

        length = float(np.clip(raw_dims[0], 0.01, 5.0))
        width  = float(np.clip(raw_dims[1], 0.01, 5.0))
        height = float(np.clip(raw_dims[2], 0.01, 5.0))
        volume = length * width * height

        print(f"✅ L={length:.3f} W={width:.3f} H={height:.3f} Vol={volume:.4f}")
        reset_scan_session()

        return jsonify({'dimensions': {
            'length':    round(length, 3),
            'width':     round(width,  3),
            'height':    round(height, 3),
            'volume_m3': round(volume, 4),
        }})

    except Exception as e:
        traceback.print_exc()
        reset_scan_session()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Legacy calibration
# ─────────────────────────────────────────────────────────────────────────────

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
    return jsonify({"scale": real_height / motion_accum, "motion": motion_accum})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
