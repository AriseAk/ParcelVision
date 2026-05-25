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
    'ground_disparity':  None,   # locked ground disparity for this scan
    'camera_height':     1.5,    # locked camera height for this scan
}


def reset_scan_session():
    scan_session['frames']           = []
    scan_session['active']           = False
    scan_session['prev_gray']        = None
    scan_session['prev_kp']          = None
    scan_session['prev_des']         = None
    scan_session['pose_R']           = np.eye(3, dtype=np.float64)
    scan_session['pose_t']           = np.zeros((3, 1), dtype=np.float64)
    scan_session['last_scale']       = 1.0
    scan_session['ground_disparity'] = None
    scan_session['camera_height']    = 1.5


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
    """
    Returns a disparity map normalized to [0, 1].
    Convention: HIGHER value = CLOSER to camera (disparity convention).
    Both DepthAnythingV2 and MiDaS outputs are converted to this convention.
    """
    _load_depth_model()
    h, w = frame.shape[:2]

    if _model_type == "da2":
        # DA2 raw output: higher = closer (disparity-like). Normalize to [0,1].
        depth = _depth_model.infer_image(frame)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-8:
            return np.ones((h, w), dtype=np.float32) * 0.5
        depth = (depth - d_min) / (d_max - d_min)
        return depth.astype(np.float32)

    if _model_type == "midas":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = _depth_transform(rgb).to(_device)
        with torch.no_grad():
            pred = _depth_model(inp)
        pred  = F.interpolate(pred.unsqueeze(1), size=(h, w)).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)
        # MiDaS raw output: higher = closer (inverse depth). Normalize to [0,1].
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-8:
            return np.ones((h, w), dtype=np.float32) * 0.5
        depth = (depth - d_min) / (d_max - d_min)
        return depth

    return np.ones((h, w), dtype=np.float32) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Ground disparity estimation
#
# Disparity convention: higher = closer.
# Floor is the closest large flat surface → use HIGH percentile of bottom strip.
# We exclude detected object bboxes to avoid picking the box itself.
# ─────────────────────────────────────────────────────────────────────────────

def estimate_ground_disparity(depth_map, detected_bboxes=None):
    h, w = depth_map.shape

    # Use a narrow center-bottom strip (bottom 20%, center 60% width)
    # to avoid edges/reflections
    strip_y_start = int(h * 0.80)
    strip_x_start = int(w * 0.20)
    strip_x_end   = int(w * 0.80)

    region = depth_map[strip_y_start:h, strip_x_start:strip_x_end].copy().astype(np.float32)

    # Mask out detected object pixels within the strip
    if detected_bboxes:
        for bbox in detected_bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            ry1 = max(0, y1 - strip_y_start)
            ry2 = max(0, y2 - strip_y_start)
            rx1 = max(0, x1 - strip_x_start)
            rx2 = min(strip_x_end - strip_x_start, x2 - strip_x_start)
            if ry2 > ry1 and rx2 > rx1:
                region[ry1:ry2, rx1:rx2] = np.nan

    valid = region[~np.isnan(region)]

    if len(valid) < 50:
        # Fallback: use full bottom row
        valid = depth_map[h - 5:h, :].flatten()

    if len(valid) < 10:
        return 0.8  # Safe fallback

    # Reject high-variance strips (reflections, clutter)
    std_val = float(np.std(valid))
    mean_val = float(np.mean(valid))
    if std_val / (mean_val + 1e-6) > 0.4:
        # High variance — use median instead of percentile
        val = float(np.median(valid))
    else:
        # Floor = highest disparity in the strip = 80th percentile
        val = float(np.percentile(valid, 80))

    return max(val, 0.05)  # Prevent division by zero


# ─────────────────────────────────────────────────────────────────────────────
# Core metric conversion
#
# Physical model:
#   disparity d ∝ 1/Z  (monocular depth models output disparity-like values)
#   We know: Z_ground = camera_height, d_ground = measured ground disparity
#   Therefore: Z(d) = camera_height * (d_ground / d)
#
# This gives METRIC depth for any pixel given its disparity value.
# No magic scale factor needed — physically derived from known camera height.
# ─────────────────────────────────────────────────────────────────────────────

def disparity_to_metric_depth(disparity_values, ground_disparity, camera_height):
    """
    Convert normalized disparity values to metric depth in meters.
    
    disparity_values: np.array of disparity values (higher = closer)
    ground_disparity: float, disparity of the ground plane
    camera_height:    float, camera height above ground in meters
    
    Returns: metric depth in meters (higher = farther)
    """
    # Avoid division by zero
    d = np.clip(disparity_values.astype(np.float32), 1e-6, None)
    # Z = H * (d_ground / d)
    # When d = d_ground → Z = H (floor is at camera height, correct)
    # When d > d_ground → Z < H (closer than floor, shouldn't happen for a box)
    # When d < d_ground → Z > H (farther than floor, reasonable for background)
    Z = camera_height * (ground_disparity / d)
    return Z


# ─────────────────────────────────────────────────────────────────────────────
# Locked ground disparity for scan session
# ─────────────────────────────────────────────────────────────────────────────

def get_or_init_ground_disparity(depth_map, camera_height, bboxes):
    if scan_session['ground_disparity'] is not None:
        return scan_session['ground_disparity']

    gd = estimate_ground_disparity(depth_map, bboxes)
    scan_session['ground_disparity'] = gd
    scan_session['camera_height']    = camera_height
    print(f"🔒 Ground disparity LOCKED: {gd:.4f}  camera_height={camera_height:.2f}m")
    return gd


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def _pixels_to_3d(us, vs, Z_metric, fx, fy, cx, cy):
    """
    Project pixels to 3D using metric depth Z.
    Z_metric must be in meters (real-world depth, not disparity).
    """
    X = (us - cx) * Z_metric / fx
    Y = (vs - cy) * Z_metric / fy
    return np.stack([X, Y, Z_metric], axis=-1)


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

    shrink = 0.20  # Slightly less aggressive crop
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
# Object measurement (live /detect mode)
#
# Uses physically correct disparity → metric depth conversion.
# No arbitrary scale factor — derived from camera_height + ground_disparity.
# ─────────────────────────────────────────────────────────────────────────────

def _measure_object(depth_map, mask_np, meta, fx, fy, cx, cy,
                    ground_disparity, camera_height):
    us, vs = _get_central_mask_pixels(mask_np)
    if len(us) < 20:
        print("❌ Mask too small")
        return None

    disp_vals = depth_map[vs, us].astype(np.float32)

    # Remove top/bottom 10% disparity outliers
    p10, p90 = np.percentile(disp_vals, 10), np.percentile(disp_vals, 90)
    valid    = (disp_vals >= p10) & (disp_vals <= p90)
    us_f     = us[valid]
    vs_f     = vs[valid]
    disp_f   = disp_vals[valid]

    if len(us_f) < 10:
        print("❌ Too few pixels after percentile filter")
        return None

    # Convert disparity → metric depth
    Z_metric = disparity_to_metric_depth(disp_f, ground_disparity, camera_height)

    # Sanity check: object should be between 0.05m and 10m from camera
    z_valid = (Z_metric > 0.05) & (Z_metric < 10.0)
    if z_valid.sum() < 10:
        print(f"❌ Metric depth out of range: min={Z_metric.min():.3f} max={Z_metric.max():.3f}")
        return None

    us_f   = us_f[z_valid]
    vs_f   = vs_f[z_valid]
    Z_metric = Z_metric[z_valid]

    pts = _pixels_to_3d(us_f.astype(float), vs_f.astype(float),
                        Z_metric, fx, fy, cx, cy)
    pts = _filter_point_cloud(pts)
    box = _fit_bounding_box(pts)
    if box is None:
        return None

    print(f"✅ Measurement: L={box['length']:.3f} W={box['width']:.3f} H={box['height']:.3f}  "
          f"ground_disp={ground_disparity:.3f} cam_h={camera_height:.2f}  "
          f"Z_median={float(np.median(Z_metric)):.3f}m")

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
    trace = np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1)
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

        # Sanity-check fx: mobile phones at 1080p have fx ~1000-1500
        # At 720p: ~900-1200. Flag suspicious values.
        if req_fx < 300 or req_fx > 3000:
            print(f"⚠️  Suspicious fx={req_fx:.1f} — using 720p estimate")
            req_fx = (img_w / 2.0) / np.tan(np.radians(35))  # ~65° VFOV typical mobile
            req_fy = req_fx

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

        # Infer depth (disparity map, higher = closer)
        depth_map = _infer_depth(frame)
        bboxes_for_ground = [d['bbox'] for d in detections]

        # Estimate ground disparity (per-frame for live mode)
        ground_disparity = estimate_ground_disparity(depth_map, bboxes_for_ground)
        print(f"GROUND DISP: {ground_disparity:.4f}  MODEL: {_model_type}  "
              f"cam_h={camera_height:.2f}m  "
              f"→ floor Z = {disparity_to_metric_depth(np.array([ground_disparity]), ground_disparity, camera_height)[0]:.3f}m (should ≈ {camera_height:.2f})")

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
                ground_disparity=ground_disparity,
                camera_height=camera_height,
            )

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
# Uses locked ground_disparity (set on first accepted frame) for consistent
# metric conversion across all frames in the scan session.
# Physically correct: Z = camera_height * (ground_disp / obj_disp)
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

        # Sanity-check fx
        if fx < 300 or fx > 3000:
            fx = (img_w / 2.0) / np.tan(np.radians(35))
            fy = fx

        # ── Detection ────────────────────────────────────────────────────
        results = yolo_model(frame, verbose=False)[0]
        boxes   = results.boxes.xyxy.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()
        bboxes  = [list(map(int, b)) for b, c in zip(boxes, confs) if c >= CONF_THRESHOLD]

        if not bboxes:
            print("⏭ Skipped: no detection")
            return jsonify({'status': 'skipped', 'reason': 'no_detection',
                            'frame_count': len(scan_session['frames'])})

        # ── Depth inference ──────────────────────────────────────────────
        depth_map = _infer_depth(frame)

        # ── Lock ground disparity on first frame ─────────────────────────
        ground_disparity = get_or_init_ground_disparity(depth_map, camera_height, bboxes)

        # ── Pose estimation ──────────────────────────────────────────────
        R_orb, t = estimate_pose_from_frame(frame, fx, fy, cx, cy)

        is_first = (R_orb is not None
                    and np.allclose(R_orb, np.eye(3))
                    and np.allclose(t, 0))

        if R_orb is None or t is None:
            current_R = scan_session['pose_R'].copy()
            current_t = scan_session['pose_t'].copy()
            print("⚠️  Pose failed — using last known pose")
        else:
            R_imu = imu_delta_rotation(imu_readings)
            R     = fuse_rotation(R_orb, R_imu)
            scan_session['pose_R'] = R @ scan_session['pose_R']
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
            print("⏭ Skipped: mask too small")
            return jsonify({'status': 'skipped', 'reason': 'mask_too_small',
                            'frame_count': len(scan_session['frames'])})

        # ── Disparity sampling with percentile filter ────────────────────
        disp_raw = depth_map[vs, us].astype(np.float32)
        p10, p90 = np.percentile(disp_raw, 10), np.percentile(disp_raw, 90)
        valid    = (disp_raw >= p10) & (disp_raw <= p90)
        us_c, vs_c, disp_c = us[valid], vs[valid], disp_raw[valid]

        if len(us_c) < 10:
            print("⏭ Skipped: too few depth pixels")
            return jsonify({'status': 'skipped', 'reason': 'too_few_depth_pixels',
                            'frame_count': len(scan_session['frames'])})

        # ── Convert disparity → metric depth ────────────────────────────
        Z_metric = disparity_to_metric_depth(disp_c, ground_disparity, camera_height)

        # Sanity check
        z_valid = (Z_metric > 0.05) & (Z_metric < 10.0)
        if z_valid.sum() < 10:
            print(f"⏭ Skipped: bad metric depth range min={Z_metric.min():.3f} max={Z_metric.max():.3f}")
            return jsonify({'status': 'skipped', 'reason': 'bad_metric_depth',
                            'frame_count': len(scan_session['frames'])})

        us_c     = us_c[z_valid]
        vs_c     = vs_c[z_valid]
        Z_metric = Z_metric[z_valid]

        # ── 3D projection → camera frame → world frame ───────────────────
        pts_cam   = _pixels_to_3d(us_c.astype(float), vs_c.astype(float),
                                   Z_metric, fx, fy, cx, cy)
        pts_cam   = _filter_point_cloud(pts_cam)
        pts_world = (current_R @ pts_cam.T).T + current_t.T

        scan_session['frames'].append({'pts_world': pts_world})
        n = len(scan_session['frames'])
        print(f"✅ Frame {n} accepted — gnd_disp={ground_disparity:.4f} "
              f"Z_median={float(np.median(Z_metric)):.3f}m pts={len(pts_cam)}")

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

        # PCA bounding box
        mean      = all_pts.mean(axis=0)
        centered  = all_pts - mean
        _, eigvecs = np.linalg.eigh(np.cov(centered.T))
        projected  = centered @ eigvecs

        # Consistent dim assignment: largest = length, smallest = height
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
