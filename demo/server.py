from __future__ import annotations

import json
import sys
import traceback
from collections import deque

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

# ─────────────────────────────────────────────────────────────────────────────
# Global motion state
# ─────────────────────────────────────────────────────────────────────────────

prev_gray       = None
prev_pts        = None
motion_smooth   = 0.0
prev_dimensions = {}

# ─────────────────────────────────────────────────────────────────────────────
# DepthAnything / MiDaS
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, "./Depth-Anything-V2")

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    _USE_DA2 = True
except Exception:
    _USE_DA2 = False

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CONF_THRESHOLD  = 0.01
CAMERA_FX       = 554.0
CAMERA_FY       = 554.0

GAMMA = 0.65

SCALE_MIN = 1e-5
SCALE_MAX = 1e-2

CARD_WIDTH_M  = 0.08560
CARD_HEIGHT_M = 0.05398
CARD_ASPECT   = CARD_WIDTH_M / CARD_HEIGHT_M

last_detections: list = []

# ─────────────────────────────────────────────────────────────────────────────
# Multi-frame calibration accumulator
# ─────────────────────────────────────────────────────────────────────────────

calibration_scales: list[float] = []
_confirmed_scale: float | None  = None
_prev_scale:      float | None  = None


def is_valid_scale(scale: float) -> bool:
    return SCALE_MIN < scale < SCALE_MAX


def add_calibration_frame(scale: float) -> bool:
    if is_valid_scale(scale):
        calibration_scales.append(scale)
    return len(calibration_scales) >= 10


def finalize_calibration() -> float | None:
    global _confirmed_scale, _prev_scale
    if len(calibration_scales) < 3:
        return None
    median_scale = float(np.median(calibration_scales))
    if not is_valid_scale(median_scale):
        return None
    if _prev_scale is not None:
        median_scale = 0.8 * _prev_scale + 0.2 * median_scale
    _confirmed_scale = median_scale
    _prev_scale      = median_scale
    calibration_scales.clear()
    print(f"✅ Calibration finalised: scale={_confirmed_scale:.8f}  "
          f"Z_cal@fx=920 → {_confirmed_scale * 920:.3f}m")
    return _confirmed_scale


def reset_calibration():
    global _confirmed_scale
    calibration_scales.clear()
    _confirmed_scale = None


# ─────────────────────────────────────────────────────────────────────────────
# Scan session
# ─────────────────────────────────────────────────────────────────────────────

scan_session: dict = {
    'frames':    [],
    'active':    False,
    'prev_gray': None,
    'prev_kp':   None,
    'prev_des':  None,
    'pose_R':    np.eye(3,     dtype=np.float64),
    'pose_t':    np.zeros((3, 1), dtype=np.float64),
}


def reset_scan_session():
    global prev_dimensions
    scan_session['frames']    = []
    scan_session['active']    = False
    scan_session['prev_gray'] = None
    scan_session['prev_kp']   = None
    scan_session['prev_des']  = None
    scan_session['pose_R']    = np.eye(3, dtype=np.float64)
    scan_session['pose_t']    = np.zeros((3, 1), dtype=np.float64)
    prev_dimensions.clear()
    print("🔄 Scan session reset — dimension memory cleared")


# ─────────────────────────────────────────────────────────────────────────────
# Flask
# ─────────────────────────────────────────────────────────────────────────────

app     = Flask(__name__)
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
    print("⚠️  Using dummy depth — measurements will be inaccurate")


def _infer_depth(frame: np.ndarray) -> np.ndarray:
    """
    Returns a disparity-like map normalised to [0, 1].
    Convention: HIGHER value = CLOSER to camera (disparity).
    """
    _load_depth_model()
    h, w = frame.shape[:2]

    if _model_type == "da2":
        depth = _depth_model.infer_image(frame).astype(np.float32)
    elif _model_type == "midas":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = _depth_transform(rgb).to(_device)
        with torch.no_grad():
            pred = _depth_model(inp)
        pred  = F.interpolate(pred.unsqueeze(1), size=(h, w)).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)
    else:
        return np.ones((h, w), dtype=np.float32) * 0.5

    p2, p98 = np.percentile(depth, 2), np.percentile(depth, 98)
    if p98 - p2 < 1e-8:
        return np.ones((h, w), dtype=np.float32) * 0.5
    depth = np.clip((depth - p2) / (p98 - p2), 0.0, 1.0)
    return depth.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Scale-based metric conversion
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_metric(depth_values: np.ndarray,
                    scale: float,
                    fx: float,
                    gamma: float = GAMMA) -> np.ndarray:
    d      = np.clip(depth_values.astype(np.float32), 1e-4, 1.0)
    d_corr = np.power(d, gamma)
    Z_cal  = scale * fx
    Z = Z_cal / (d_corr ** 0.7)
    return np.clip(Z, 0.02, 10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pixels_to_3d(us: np.ndarray, vs: np.ndarray, Z: np.ndarray,
                  fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    X = (us - cx) * Z / fx
    Y = (vs - cy) * Z / fy
    return np.stack([X, Y, Z], axis=-1)


def _filter_point_cloud(pts: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    if len(pts) < 10:
        return pts
    centroid = pts.mean(axis=0)
    dists    = np.linalg.norm(pts - centroid, axis=1)
    return pts[dists < dists.mean() + sigma * dists.std()]


def _fit_bounding_box(pts: np.ndarray) -> dict | None:
    if len(pts) < 50:
        print(f"⚠️  Too few points for bbox: {len(pts)} < 50")
        return None
    l = float(np.percentile(pts[:, 0], 95) - np.percentile(pts[:, 0], 5))
    w = float(np.percentile(pts[:, 2], 95) - np.percentile(pts[:, 2], 5))
    h = float(np.percentile(pts[:, 1], 95) - np.percentile(pts[:, 1], 5))
    if l <= 0 or w <= 0 or h <= 0:
        return None
    return {"length": l, "width": w, "height": h, "volume_m3": l * w * h}


# ─────────────────────────────────────────────────────────────────────────────
# Card orientation / aspect ratio check
# ─────────────────────────────────────────────────────────────────────────────

def check_card_orientation(pixel_w: float, pixel_h: float,
                            img_w: int, img_h: int) -> tuple[bool, str]:
    if pixel_w <= 0 or pixel_h <= 0:
        return False, "zero dimensions"
    min_px = img_w * 0.03
    if pixel_w < min_px or pixel_h < min_px:
        return False, f"bbox too small ({pixel_w:.0f}×{pixel_h:.0f})"

    aspect = pixel_w / pixel_h
    tol    = 0.15
    lo, hi = CARD_ASPECT * (1 - tol), CARD_ASPECT * (1 + tol)

    if lo <= aspect <= hi:
        return True, "ok"
    portrait = pixel_h / pixel_w
    if lo <= portrait <= hi:
        return True, "ok (portrait)"
    return False, f"bad aspect {aspect:.2f} (expected {CARD_ASPECT:.2f}±{tol*100:.0f}%)"


# ─────────────────────────────────────────────────────────────────────────────
# Central mask sampling
# ─────────────────────────────────────────────────────────────────────────────

def _get_central_mask_pixels(mask_np: np.ndarray):
    """
    Returns the INNER pixels of the mask (shrunk 30% on each side).
    Used for DEPTH SAMPLING only — avoids noisy segmentation edges
    where the depth model is unreliable.
    Do NOT use this for geometry extent — it discards the top/bottom/sides
    of the mask, which destroys height information.
    """
    vs, us = np.where(mask_np > 0.5)
    if len(us) < 20:
        return us, vs
    u_min, u_max = us.min(), us.max()
    v_min, v_max = vs.min(), vs.max()
    shrink = 0.30
    u_lo = u_min + shrink * (u_max - u_min)
    u_hi = u_max - shrink * (u_max - u_min)
    v_lo = v_min + shrink * (v_max - v_min)
    v_hi = v_max - shrink * (v_max - v_min)
    if u_hi <= u_lo or v_hi <= v_lo:
        return us, vs
    inner = (us >= u_lo) & (us <= u_hi) & (vs >= v_lo) & (vs <= v_hi)
    us_in, vs_in = us[inner], vs[inner]
    return (us_in, vs_in) if len(us_in) >= 20 else (us, vs)


def _get_geometry_mask_pixels(mask_np: np.ndarray,
                               max_pts: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns pixels spread across the FULL spatial extent of the mask.
    Used for GEOMETRY only (3D point cloud extent, NOT depth sampling).

    Strategy: sample uniformly from the full mask but bias toward the
    perimeter (top/bottom rows, left/right cols) so PCA sees the true
    spatial extent of all faces — especially height (vertical extent).

    We take:
      - All pixels on the top 10% and bottom 10% of mask rows (height extent)
      - All pixels on the left 10% and right 10% of mask cols (width extent)
      - A random subsample of the interior for depth/volume fill

    This guarantees the extreme Y values (top and bottom of box face) are
    always represented, so height is never collapsed by central-only sampling.
    """
    vs, us = np.where(mask_np > 0.5)
    if len(us) < 20:
        return us, vs

    v_min, v_max = int(vs.min()), int(vs.max())
    u_min, u_max = int(us.min()), int(us.max())
    v_range = max(1, v_max - v_min)
    u_range = max(1, u_max - u_min)

    edge_v = 0.10   # top/bottom 10% of rows → captures full height
    edge_u = 0.10   # left/right 10% of cols

    # Perimeter band masks
    top_band    = vs <= v_min + edge_v * v_range
    bot_band    = vs >= v_max - edge_v * v_range
    left_band   = us <= u_min + edge_u * u_range
    right_band  = us >= u_max - edge_u * u_range
    perimeter   = top_band | bot_band | left_band | right_band

    peri_us, peri_vs = us[perimeter], vs[perimeter]
    rest_us, rest_vs = us[~perimeter], vs[~perimeter]

    # Subsample interior so we don't blow up memory
    n_interior = min(len(rest_us), max_pts - len(peri_us))
    if n_interior > 0 and len(rest_us) > 0:
        idx = np.random.choice(len(rest_us), n_interior, replace=False)
        rest_us, rest_vs = rest_us[idx], rest_vs[idx]
        out_us = np.concatenate([peri_us, rest_us])
        out_vs = np.concatenate([peri_vs, rest_vs])
    else:
        out_us, out_vs = peri_us, peri_vs

    return out_us.astype(np.int32), out_vs.astype(np.int32)


def _mask_to_contour(mask_np: np.ndarray, max_points: int = 60) -> list | None:
    """
    Extract the largest contour from a binary SAM mask and simplify it.
    Returns [[x,y], ...] in frame pixel space, or None.
    """
    contours, _ = cv2.findContours(
        (mask_np * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:
        return None

    perimeter = cv2.arcLength(largest, True)
    epsilon   = 0.01 * perimeter
    approx    = cv2.approxPolyDP(largest, epsilon, True)
    while len(approx) > max_points and epsilon < perimeter:
        epsilon *= 1.5
        approx = cv2.approxPolyDP(largest, epsilon, True)

    return approx.reshape(-1, 2).astype(int).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Dimension smoothing (live mode)
# ─────────────────────────────────────────────────────────────────────────────

def smooth_dimensions(prev: dict | None, new: dict) -> dict:
    if prev is None:
        return new
    alpha = 0.4
    max_jump = max(
        abs(new['length'] - prev['length']) / (prev['length'] + 1e-6),
        abs(new['width']  - prev['width'])  / (prev['width']  + 1e-6),
        abs(new['height'] - prev['height']) / (prev['height'] + 1e-6),
    )
    if max_jump > 0.4:
        return new
    s = {
        'length': alpha * prev['length'] + (1 - alpha) * new['length'],
        'width':  alpha * prev['width']  + (1 - alpha) * new['width'],
        'height': alpha * prev['height'] + (1 - alpha) * new['height'],
    }
    s['volume_m3'] = s['length'] * s['width'] * s['height']
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Depth consistency filter
# ─────────────────────────────────────────────────────────────────────────────

def _filter_depth_consistency(us: np.ndarray, vs: np.ndarray,
                               disp_vals: np.ndarray,
                               tolerance: float = 0.20
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median_depth = np.median(disp_vals)
    valid = np.abs(disp_vals - median_depth) < tolerance
    return us[valid], vs[valid], disp_vals[valid]


# ─────────────────────────────────────────────────────────────────────────────
# Object measurement (live /detect mode)
# ─────────────────────────────────────────────────────────────────────────────

def _measure_object(depth_map: np.ndarray, mask_np: np.ndarray,
                    meta: dict,
                    fx: float, fy: float, cx: float, cy: float,
                    scale: float) -> dict | None:

    us, vs = _get_central_mask_pixels(mask_np)
    if len(us) < 20:
        return None

    disp_vals = depth_map[vs, us].astype(np.float32)
    p10, p90  = np.percentile(disp_vals, 10), np.percentile(disp_vals, 90)
    valid     = (disp_vals >= p10) & (disp_vals <= p90)
    us_f, vs_f, disp_f = us[valid], vs[valid], disp_vals[valid]
    if len(us_f) < 10:
        return None

    us_f, vs_f, disp_f = _filter_depth_consistency(us_f, vs_f, disp_f)
    if len(us_f) < 10:
        return None

    Z_metric = depth_to_metric(disp_f, scale, fx)

    z_ok = (Z_metric > 0.02) & (Z_metric < 10.0)
    if z_ok.sum() < 10:
        print(f"❌ Metric depth out of range: {Z_metric.min():.3f}–{Z_metric.max():.3f} m  "
              f"(scale={scale:.6f} fx={fx:.1f} Z_cal={scale*fx:.3f}m)")
        return None

    us_f, vs_f, Z_metric = us_f[z_ok], vs_f[z_ok], Z_metric[z_ok]
    pts = _pixels_to_3d(us_f.astype(float), vs_f.astype(float), Z_metric, fx, fy, cx, cy)
    pts = _filter_point_cloud(pts)
    box = _fit_bounding_box(pts)
    if box is None:
        return None

    print(f"✅ Live: L={box['length']:.3f} W={box['width']:.3f} H={box['height']:.3f}  "
          f"Z_cal={scale*fx:.3f}m  Z_med={float(np.median(Z_metric)):.3f}m")

    obj_id = meta["object_id"]
    box    = smooth_dimensions(prev_dimensions.get(obj_id), box)
    prev_dimensions[obj_id] = box
    return {"object_id": meta["object_id"], "label": meta["class"], **box}


# ─────────────────────────────────────────────────────────────────────────────
# Backend motion estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_camera_motion(frame: np.ndarray) -> float:
    global prev_gray, prev_pts, motion_smooth
    small = cv2.resize(frame, (320, 240))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if prev_gray is None or prev_pts is None or len(prev_pts) < 10:
        prev_gray = gray
        prev_pts  = cv2.goodFeaturesToTrack(gray, 300, 0.01, 5)
        return 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
    if next_pts is None:
        prev_gray = gray; prev_pts = None; return 0.0

    good_old = prev_pts[status == 1]
    good_new = next_pts[status == 1]
    if len(good_old) < 5:
        prev_gray = gray; prev_pts = None; return 0.0

    movement      = np.linalg.norm(good_new - good_old, axis=1)
    raw_motion    = float(np.mean(movement) * 5)
    motion_smooth = 0.8 * motion_smooth + 0.2 * raw_motion
    prev_pts      = (cv2.goodFeaturesToTrack(gray, 300, 0.01, 5)
                     if np.random.rand() < 0.1
                     else good_new.reshape(-1, 1, 2))
    prev_gray = gray
    return motion_smooth


# ─────────────────────────────────────────────────────────────────────────────
# Visual odometry — rotation only
# ─────────────────────────────────────────────────────────────────────────────

def estimate_rotation_from_frame(bgr: np.ndarray,
                                  fx: float, fy: float,
                                  cx: float, cy: float) -> np.ndarray | None:
    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    orb     = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(gray, None)

    if des is None or len(kp) < 20:
        scan_session['prev_gray'] = gray
        scan_session['prev_kp']   = kp
        scan_session['prev_des']  = des
        return None

    if scan_session['prev_gray'] is None or scan_session['prev_des'] is None:
        scan_session['prev_gray'] = gray
        scan_session['prev_kp']   = kp
        scan_session['prev_des']  = des
        return np.eye(3, dtype=np.float64)

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(scan_session['prev_des'], des, k=2)
    good    = [m for pair in matches if len(pair) == 2
               for m, n in [pair] if m.distance < 0.75 * n.distance]

    if len(good) < 15:
        scan_session['prev_gray'] = gray
        scan_session['prev_kp']   = kp
        scan_session['prev_des']  = des
        return None

    pts1 = np.float32([scan_session['prev_kp'][m.queryIdx].pt for m in good])
    pts2 = np.float32([kp[m.trainIdx].pt                      for m in good])

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        scan_session['prev_gray'] = gray
        scan_session['prev_kp']   = kp
        scan_session['prev_des']  = des
        return None

    _, R, _t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    scan_session['prev_gray'] = gray
    scan_session['prev_kp']   = kp
    scan_session['prev_des']  = des
    return R


# ─────────────────────────────────────────────────────────────────────────────
# IMU
# ─────────────────────────────────────────────────────────────────────────────

def imu_delta_rotation(imu_readings: list) -> np.ndarray | None:
    if not imu_readings or len(imu_readings) < 2:
        return None
    dt_total = (imu_readings[-1]['ts'] - imu_readings[0]['ts']) / 1000.0
    if dt_total <= 0:
        return None
    deg_to_rad = np.pi / 180.0
    n  = len(imu_readings)
    ax = sum(r.get('gx', 0) for r in imu_readings) * deg_to_rad * (dt_total / n)
    ay = sum(r.get('gy', 0) for r in imu_readings) * deg_to_rad * (dt_total / n)
    az = sum(r.get('gz', 0) for r in imu_readings) * deg_to_rad * (dt_total / n)
    Rx, _ = cv2.Rodrigues(np.array([ax, 0,  0]))
    Ry, _ = cv2.Rodrigues(np.array([0,  ay, 0]))
    Rz, _ = cv2.Rodrigues(np.array([0,  0,  az]))
    return Rz @ Ry @ Rx


def fuse_rotation(R_orb: np.ndarray | None, R_imu: np.ndarray | None) -> np.ndarray:
    if R_orb is None and R_imu is None:
        return np.eye(3, dtype=np.float64)
    if R_orb is None:
        return R_imu
    if R_imu is None:
        return R_orb
    trace = np.clip((np.trace(R_orb @ R_imu.T) - 1) / 2, -1, 1)
    angle_diff = float(np.degrees(np.arccos(trace)))
    return R_imu if angle_diff > 15.0 else R_orb


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_scale(form) -> float | None:
    raw = form.get("scale")
    if raw is None:
        return None
    try:
        s = float(raw)
        return s if is_valid_scale(s) else None
    except (ValueError, TypeError):
        return None


def _sanitise_fx(req_fx: float, img_w: int) -> float:
    if req_fx < 300 or req_fx > 3000:
        print(f"⚠️  Suspicious fx={req_fx:.1f} — using 65° HFOV fallback")
        return (img_w / 2.0) / np.tan(np.radians(32.5))
    return req_fx


# ─────────────────────────────────────────────────────────────────────────────
# Load models
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
        # fast=1: skip SAM segmentation, return YOLO bbox + cached contour only.
        # Frontend sends this when motion is high — YOLO alone is ~10ms vs
        # SAM's ~150ms, so the overlay snaps to the box almost immediately.
        fast_mode = request.form.get("fast", "0") == "1"

        file  = request.files["image"]
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid image"}), 400

        motion = compute_camera_motion(frame)

        req_fx = float(request.form.get("fx", CAMERA_FX))
        req_fy = float(request.form.get("fy", CAMERA_FY))
        img_w  = int(request.form.get("img_w", frame.shape[1]))
        img_h  = int(request.form.get("img_h", frame.shape[0]))
        cx     = img_w / 2.0
        cy     = img_h / 2.0
        req_fx = _sanitise_fx(req_fx, img_w)
        req_fy = req_fx

        scale = _parse_scale(request.form)
        if scale:
            print(f"📷 fx={req_fx:.1f}  scale={scale:.8f}  Z_cal={scale*req_fx:.3f}m  motion={motion:.2f}  fast={fast_mode}")
        else:
            print(f"📷 fx={req_fx:.1f}  scale=None  motion={motion:.2f}  fast={fast_mode}")

        h, w = frame.shape[:2]

        global last_detections

        # ── FAST MODE: YOLO only, reuse cached SAM contours ──────────────────
        # Runs in ~10-20ms vs ~200ms for full SAM pass.
        # Bboxes are updated to follow the object; contours stay from last full pass.
        if fast_mode and run_detection:
            results = yolo_model(frame, verbose=False)[0]
            boxes   = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs   = results.boxes.conf.cpu().numpy()

            fast_scene = []
            # Build a lookup of last known contours keyed by approximate position
            last_lookup = {tuple(d['bbox']): d for d in last_detections}

            for box, cls, conf in zip(boxes, classes, confs):
                if conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box)
                label = yolo_model.names[int(cls)]
                # Reuse the contour from whichever previous detection overlaps most
                cached_contour = None
                cached_dims    = None
                best_iou = 0.0
                for prev in last_detections:
                    pb = prev['bbox']
                    ix1 = max(x1, pb[0]); iy1 = max(y1, pb[1])
                    ix2 = min(x2, pb[2]); iy2 = min(y2, pb[3])
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2-ix1)*(iy2-iy1)
                        union = (x2-x1)*(y2-y1) + (pb[2]-pb[0])*(pb[3]-pb[1]) - inter
                        iou   = inter / (union + 1e-6)
                        if iou > best_iou:
                            best_iou       = iou
                            cached_contour = prev.get('contour')
                            cached_dims    = prev.get('dimensions')

                fast_scene.append({
                    "object_id":  last_detections[0]['object_id'] if last_detections else 0,
                    "label":      label,
                    "confidence": float(conf),
                    "bbox":       [x1, y1, x2, y2],
                    "contour":    cached_contour,   # reuse last SAM contour
                    "center":     [float((x1+x2)/2), float((y1+y2)/2)],
                    "dimensions": cached_dims,       # keep last measured dims
                })

            if fast_scene:
                last_detections = fast_scene
            return jsonify({"scene": last_detections, "motion": motion, "fast": True})

        # ── NORMAL MODE: full YOLO + SAM pass ────────────────────────────────
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
                contour = _mask_to_contour(mask_np)
                detections.append({
                    "bbox":           [x1, y1, x2, y2],
                    "confidence":     float(conf),
                    "mask":           mask_np,
                    "contour":        contour,
                    "business_class": label,
                    "area":           float((x2 - x1) * (y2 - y1)),
                    "center":         [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                })

        if not run_detection:
            return jsonify({"scene": last_detections, "motion": motion})

        detections = tracker.update(detections)

        depth_map = None
        if scale is not None and detections:
            depth_map = _infer_depth(frame)

        scene = []
        for det in detections:
            dims = None
            if scale is not None and depth_map is not None:
                dims = _measure_object(
                    depth_map, det["mask"],
                    {"object_id": det["id"], "class": det["business_class"]},
                    fx=req_fx, fy=req_fy, cx=cx, cy=cy,
                    scale=scale,
                )
            scene.append({
                "object_id":  int(det["id"]),
                "label":      det["business_class"],
                "confidence": float(det["confidence"]),
                "bbox":       det["bbox"],
                "contour":    det.get("contour"),
                "center":     det["center"],
                "dimensions": dims,
            })

        if scene:
            last_detections = scene

        return jsonify({"scene": last_detections, "motion": motion})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# /calibrate_frame  — accumulate multi-frame calibration data
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/calibrate_frame", methods=["POST"])
def calibrate_frame():
    try:
        pixel_w = float(request.form.get("pixel_w", 0))
        pixel_h = float(request.form.get("pixel_h", 0))
        img_w   = int(request.form.get("img_w", 1280))
        img_h   = int(request.form.get("img_h", 720))
        fx_raw  = float(request.form.get("fx", 920))
        fx      = _sanitise_fx(fx_raw, img_w)

        valid, reason = check_card_orientation(pixel_w, pixel_h, img_w, img_h)
        if not valid:
            return jsonify({
                "status":  "rejected",
                "reason":  reason,
                "count":   len(calibration_scales),
                "ready":   False,
            })

        scale_w = CARD_WIDTH_M  / pixel_w
        scale_h = CARD_HEIGHT_M / pixel_h
        scale   = (scale_w + scale_h) / 2.0

        ready = add_calibration_frame(scale)
        count = len(calibration_scales)

        print(f"📐 Cal frame: px={pixel_w:.0f}×{pixel_h:.0f}  "
              f"scale={scale:.8f}  frames={count}  ready={ready}")

        response = {
            "status": "accepted",
            "count":  count,
            "ready":  ready,
            "scale":  scale,
        }

        if ready:
            final = finalize_calibration()
            response["final_scale"] = final
            response["z_cal_m"]     = round(final * fx, 3) if final else None
            print(f"📐 Calibration ready: scale={final:.8f}  Z_cal={final*fx:.3f}m")

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# /confirm_calibration — lock in the accumulated scale
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/confirm_calibration", methods=["POST"])
def confirm_calibration():
    try:
        fx_raw = float(request.form.get("fx", 920))
        img_w  = int(request.form.get("img_w", 1280))
        fx     = _sanitise_fx(fx_raw, img_w)

        pixel_w = float(request.form.get("pixel_w", 0))
        pixel_h = float(request.form.get("pixel_h", 0))
        if pixel_w > 0 and pixel_h > 0:
            scale_w = CARD_WIDTH_M  / pixel_w
            scale_h = CARD_HEIGHT_M / pixel_h
            add_calibration_frame((scale_w + scale_h) / 2.0)

        final = finalize_calibration()
        if final is None:
            return jsonify({"error": "not enough valid calibration frames"}), 400

        return jsonify({
            "scale":   final,
            "z_cal_m": round(final * fx, 3),
            "mm_per_px": round(final * 1000, 4),
        })

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
    print("🔵 Scan session started")
    return jsonify({'status': 'scan started'})


# ─────────────────────────────────────────────────────────────────────────────
# /scan_frame  — now returns contour + bbox for live overlay during scan
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/scan_frame", methods=["POST"])
def scan_frame():
    try:
        file  = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'bad frame'}), 400

        imu_readings = json.loads(request.form.get('imu', '[]'))
        fx           = float(request.form.get('fx', 554))
        fy           = float(request.form.get('fy', 554))
        img_w        = int(request.form.get('img_w', frame.shape[1]))
        img_h        = int(request.form.get('img_h', frame.shape[0]))
        cx           = img_w / 2.0
        cy           = img_h / 2.0
        fx           = _sanitise_fx(fx, img_w)
        fy           = fx

        scale = _parse_scale(request.form)
        if scale is None:
            print("⏭ Skipped: no_scale")
            return jsonify({'status': 'skipped', 'reason': 'no_scale',
                            'frame_count': len(scan_session['frames']),
                            'bbox': None, 'contour': None})

        Z_cal = scale * fx
        print(f"🔍 scan_frame: scale={scale:.8f}  fx={fx:.1f}  Z_cal={Z_cal:.3f}m  frames={len(scan_session['frames'])}")

        # ── Detection ────────────────────────────────────────────────────
        results = yolo_model(frame, verbose=False)[0]
        boxes   = results.boxes.xyxy.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()
        bboxes  = [list(map(int, b)) for b, c in zip(boxes, confs) if c >= CONF_THRESHOLD]

        if not bboxes:
            if last_detections:
                bboxes = [last_detections[0]['bbox']]
                print("⚠️  Using last known bbox")
            else:
                print("⏭ Skipped: no_detection")
                return jsonify({'status': 'skipped', 'reason': 'no_detection',
                                'frame_count': len(scan_session['frames']),
                                'bbox': None, 'contour': None})

        # ── Depth inference ──────────────────────────────────────────────
        depth_map = _infer_depth(frame)

        # ── Rotation estimation ──────────────────────────────────────────
        R_orb = estimate_rotation_from_frame(frame, fx, fy, cx, cy)
        R_imu = imu_delta_rotation(imu_readings)
        R     = fuse_rotation(R_orb, R_imu)
        scan_session['pose_R'] = R @ scan_session['pose_R']
        current_R = scan_session['pose_R'].copy()

        # ── SAM segmentation ────────────────────────────────────────────
        x1, y1, x2, y2 = bboxes[0]
        sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        masks, _, _ = sam_predictor.predict(
            box=np.array([x1, y1, x2, y2]), multimask_output=False)
        mask_np = (masks[0] > 0.5).astype(np.uint8)
        if mask_np.sum() < 20:
            mask_np = np.zeros((img_h, img_w), dtype=np.uint8)
            mask_np[y1:y2, x1:x2] = 1

        # ── Extract contour for frontend overlay ─────────────────────────
        contour = _mask_to_contour(mask_np)

        # ── Central mask pixels — for DEPTH SAMPLING only ──────────────────
        # _get_central_mask_pixels shrinks 30% on each side to avoid noisy
        # depth edges.  We use this ONLY to get a reliable median Z.
        us_depth, vs_depth = _get_central_mask_pixels(mask_np)
        if len(us_depth) < 20:
            print("⏭ Skipped: mask_too_small")
            return jsonify({'status': 'skipped', 'reason': 'mask_too_small',
                            'frame_count': len(scan_session['frames']),
                            'bbox': bboxes[0], 'contour': contour})

        # ── Depth sampling on central pixels ────────────────────────────────
        disp_raw  = depth_map[vs_depth, us_depth].astype(np.float32)
        p10, p90  = np.percentile(disp_raw, 10), np.percentile(disp_raw, 90)
        valid_idx = (disp_raw >= p10) & (disp_raw <= p90)
        us_dc, vs_dc, disp_dc = us_depth[valid_idx], vs_depth[valid_idx], disp_raw[valid_idx]

        if len(us_dc) < 10:
            print("⏭ Skipped: too_few_depth_pixels")
            return jsonify({'status': 'skipped', 'reason': 'too_few_depth_pixels',
                            'frame_count': len(scan_session['frames']),
                            'bbox': bboxes[0], 'contour': contour})

        us_dc, vs_dc, disp_dc = _filter_depth_consistency(us_dc, vs_dc, disp_dc)
        if len(us_dc) < 10:
            print("⏭ Skipped: depth_inconsistent")
            return jsonify({'status': 'skipped', 'reason': 'depth_inconsistent',
                            'frame_count': len(scan_session['frames']),
                            'bbox': bboxes[0], 'contour': contour})

        # Median Z from central pixels — stable, edge-noise-free
        Z_center_vals = depth_to_metric(disp_dc, scale, fx)
        z_ok_c = (Z_center_vals > 0.02) & (Z_center_vals < 10.0)
        if z_ok_c.sum() < 10:
            print("⏭ Skipped: bad_metric_depth")
            return jsonify({'status': 'skipped', 'reason': 'bad_metric_depth',
                            'frame_count': len(scan_session['frames']),
                            'bbox': bboxes[0], 'contour': contour})

        Z_median = float(np.median(Z_center_vals[z_ok_c]))
        print(f"   Z_med={Z_median:.3f}m  valid_depth={z_ok_c.sum()}")

        # ── Geometry pixels — full spatial extent of the mask ───────────────
        # _get_geometry_mask_pixels keeps ALL perimeter rows/cols so the
        # top and bottom of the box face are always represented.
        # This is what fixes the height collapse bug — the central-only
        # sampling was discarding 60% of the vertical extent every frame.
        us_geom, vs_geom = _get_geometry_mask_pixels(mask_np, max_pts=2000)

        # Assign each geometry pixel the MEDIAN depth of the face
        # (not per-pixel depth — monocular depth is unreliable at edges).
        # This is sound because the box face is approximately planar at Z_median.
        # The spread in X/Y (from pixel positions) gives width/height;
        # the Z variation across frames (from camera motion) gives depth.
        Z_geom = np.full(len(us_geom), Z_median, dtype=np.float32)

        # ── 3D projection + rotation ─────────────────────────────────────
        pts_cam   = _pixels_to_3d(us_geom.astype(float), vs_geom.astype(float),
                                   Z_geom, fx, fy, cx, cy)
        pts_cam   = _filter_point_cloud(pts_cam)
        pts_world = (current_R @ pts_cam.T).T

        scan_session['frames'].append({
            'pts_world':     pts_world,
            # Store the metric height implied directly by pixel extent + depth.
            # metric_h = pixel_rows_of_mask * Z_median / fy
            # This is a reliable single-frame height estimate for the front/side faces.
            # compute_dimensions uses the max across frames as a height floor.
            'pixel_height_m': float((vs_geom.max() - vs_geom.min()) * Z_median / fy),
        })
        n = len(scan_session['frames'])
        ph_m = scan_session['frames'][-1]['pixel_height_m']
        print(f"✅ Frame {n} accepted — Z_med={Z_median:.3f}m  pts={len(pts_cam)}  pixel_h={ph_m:.3f}m")

        return jsonify({
            'status':      'ok',
            'frame_count': n,
            'bbox':        bboxes[0],        # [x1,y1,x2,y2] in frame pixels
            'contour':     contour,           # [[x,y],...] simplified SAM contour
        })

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

        # Single-pass outlier removal with generous sigma.
        # The old two-pass (2.5→2.0) was too aggressive on shell point clouds
        # where valid extremal points (top/bottom of box) look like outliers
        # relative to the dense face interior.
        if len(all_pts) >= 20:
            centroid = all_pts.mean(axis=0)
            dists    = np.linalg.norm(all_pts - centroid, axis=1)
            all_pts  = all_pts[dists < dists.mean() + 3.0 * dists.std()]

        print(f"After filtering: {len(all_pts)} pts")
        if len(all_pts) < 20:
            reset_scan_session()
            return jsonify({'error': 'not enough clean points'}), 400

        # PCA to find the natural axes of the point cloud
        mean      = all_pts.mean(axis=0)
        centered  = all_pts - mean
        _, eigvecs = np.linalg.eigh(np.cov(centered.T))
        projected  = centered @ eigvecs

        # Use 2nd–98th percentile for extent — wider than before (was 5–95th).
        # The 5th–95th percentile was discarding the outermost 10% of points
        # which are exactly the extremal face points that define box size.
        raw_dims = sorted(
            [float(np.percentile(projected[:, i], 2) - np.percentile(projected[:, i], 98))
             for i in range(3)],
            reverse=True,
            key=abs
        )
        raw_dims = [abs(d) for d in raw_dims]

        length = float(np.clip(raw_dims[0], 0.005, 5.0))
        width  = float(np.clip(raw_dims[1], 0.005, 5.0))
        height = float(np.clip(raw_dims[2], 0.005, 5.0))

        # ── Pixel-space height floor ─────────────────────────────────────────
        # For each accepted frame we stored the mask bbox (v_min, v_max) and
        # Z_median.  Convert pixel height → metric height using pinhole model:
        #   metric_height = (v_max - v_min) * Z_median / fy
        # Take the max across frames — the frame where the box is most upright
        # (front/back view) gives the most accurate height.
        # This acts as a floor: PCA never returns less than what the pixel
        # geometry directly implies.
        pixel_heights = []
        for f in frames:
            ph = f.get('pixel_height_m')
            if ph is not None and ph > 0.01:
                pixel_heights.append(ph)

        if pixel_heights:
            pixel_height_max = float(np.percentile(pixel_heights, 75))
            print(f"   Pixel-space height floor: {pixel_height_max:.3f}m  "
                  f"(from {len(pixel_heights)} frames)")
            if pixel_height_max > height:
                print(f"   ↑ Height lifted from {height:.3f}m → {pixel_height_max:.3f}m by pixel floor")
                height = pixel_height_max

        volume = length * width * height
        print(f"✅ Final: L={length:.3f} W={width:.3f} H={height:.3f} Vol={volume:.4f} m³")
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
# Legacy stubs
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/start_calibration", methods=["POST"])
def start_calibration():
    return jsonify({"status": "deprecated — use /calibrate_frame"})

@app.route("/end_calibration", methods=["POST"])
def end_calibration():
    return jsonify({"status": "deprecated — use /confirm_calibration"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)