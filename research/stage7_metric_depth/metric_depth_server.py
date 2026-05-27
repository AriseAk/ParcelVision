"""
metric_depth_server.py — ParcelVision Stage 7
=============================================
Drop-in replacement for stage6_imu_fusion/fixed_scale_server.py.

Pipeline change
───────────────
Old:  DepthAnything V2  →  relative disparity  →  scale = camera_height / ground  →  metres  (FRAGILE)
New:  ZoeDepth          →  absolute metric depth in metres  (no scale factor anywhere)

Everything else is unchanged from fixed_scale_server.py:
    • YOLO World detection
    • SAM segmentation
    • IoU tracker (tracker.py)
    • SceneStateManager (scene_state.py)
    • Visual odometry + IMU fusion
    • Multi-frame scan session (/start_scan → /scan_frame × N → /compute_dimensions)
    • /detect, /start_scan, /scan_frame, /compute_dimensions
    • Identical JSON response shapes — frontend is unchanged

Removed vs fixed_scale_server.py:
    ✗ estimate_ground_depth()
    ✗ get_or_init_global_scale()
    ✗ scan_session['scale_initialized'] / 'global_scale' / 'last_scale'
    ✗ scale = camera_height / ground  in /detect and /scan_frame
    ✗ m["length"] *= scale  etc. in /detect
    ✗ zs_metric = zs_c * scale  in /scan_frame
    ✗ DepthAnything V2 / MiDaS import chain

Fixes applied in this production version
─────────────────────────────────────────
    [F1] /detect uses measure_object_dimensions() from pointcloud_builder —
         two-pass sampling (central crop for depth, full mask for height)
         replaces the old single-pass _measure_object() that under-estimated
         height due to shrink=0.25 cropping away the top/bottom edges.

    [F2] /scan_frame drops the erroneous pose_t accumulation with magic
         constant (* 0.03).  Translation from recoverPose is unit-norm (not
         metric), so we accumulate rotation only.  The point clouds themselves
         carry the metric info via ZoeDepth — pose_t is held at zero.

    [F3] /compute_dimensions uses fit_pca_bbox() from pointcloud_builder,
         which now resolves PCA axes to semantic length/width/height by
         identifying the vertical axis — consistent L/W/H across scan angles.

    [F4] scan_frame uses shrink=0.10 (was 0.25) and the full-mask height
         refinement pass, consistent with /detect.

Run
───
    cd research/stage7_metric_depth
    python metric_depth_server.py
"""

from __future__ import annotations

import json
import sys
import traceback

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

# ── Stage 3 helpers (unchanged) ───────────────────────────────────────────────
sys.path.insert(0, "../stage3_segmentation_tracking")
from tracker import ObjectTracker
from scene_state import SceneStateManager

# ── Stage 7: ZoeDepth + geometry ─────────────────────────────────────────────
from zoedepth_runner import infer_metric_depth, get_runner
from pointcloud_builder import (
    measure_object_dimensions,   # top-level single-frame helper  [F1]
    mask_to_point_cloud,
    filter_point_cloud,
    get_central_pixels,
    get_full_mask_pixels,
    pixels_to_3d,
    fit_pca_bbox,
    depth_confidence,
    final_confidence,
)
from temporal_smoothing import SceneSmoother

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.01

# Default focal lengths — overridden per-request via fx / fy form fields
DEFAULT_FX = 554.0
DEFAULT_FY = 554.0

_device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Flask
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# Models  (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

print("[server] Loading YOLO World …")
yolo_model = YOLO("yolov8s-world.pt")
yolo_model.to(_device)
yolo_model.set_classes([
    "box", "cardboard box", "carton", "parcel",
    "package", "container", "brown box",
    "shipping box", "crate", "rectangular box",
])

print("[server] Loading SAM …")
from segment_anything import sam_model_registry, SamPredictor
_sam          = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
_sam.to(_device)
sam_predictor = SamPredictor(_sam)

# ZoeDepth — pre-loaded at startup (see __main__ block)
zoe_runner = get_runner(variant="ZoeD_NK")

tracker         = ObjectTracker()
scene_manager   = SceneStateManager()
_scene_smoother = SceneSmoother(alpha=0.35, jump_threshold=0.40)

# ─────────────────────────────────────────────────────────────────────────────
# Scan session state
# ─────────────────────────────────────────────────────────────────────────────
# scale_initialized / global_scale / last_scale REMOVED — ZoeDepth is metric.
# pose_t removed from accumulation [F2] — recoverPose translation is unit-norm.

scan_session: dict = {
    "frames":    [],         # list of {"pts_world": np.ndarray (N,3)}
    "active":    False,
    "prev_gray": None,
    "prev_kp":   None,
    "prev_des":  None,
    "pose_R":    np.eye(3, dtype=np.float64),   # accumulated rotation only
}


def reset_scan_session() -> None:
    scan_session["frames"]    = []
    scan_session["active"]    = False
    scan_session["prev_gray"] = None
    scan_session["prev_kp"]   = None
    scan_session["prev_des"]  = None
    scan_session["pose_R"]    = np.eye(3, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Last known scene (returned when detect=0)
# ─────────────────────────────────────────────────────────────────────────────

last_detections: list = []

# ─────────────────────────────────────────────────────────────────────────────
# Optical flow motion estimate  (UX feedback in /detect — not used in scan)
# ─────────────────────────────────────────────────────────────────────────────

_prev_gray_of:  np.ndarray | None = None
_prev_pts_of:   np.ndarray | None = None
_motion_smooth: float             = 0.0


def compute_camera_motion(frame: np.ndarray) -> float:
    global _prev_gray_of, _prev_pts_of, _motion_smooth

    small = cv2.resize(frame, (320, 240))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if _prev_gray_of is None or _prev_pts_of is None or len(_prev_pts_of) < 10:
        _prev_gray_of = gray
        _prev_pts_of  = cv2.goodFeaturesToTrack(gray, 300, 0.01, 5)
        return 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        _prev_gray_of, gray, _prev_pts_of, None
    )
    if next_pts is None:
        _prev_gray_of = gray
        _prev_pts_of  = None
        return 0.0

    good_old = _prev_pts_of[status == 1]
    good_new = next_pts[status == 1]
    if len(good_old) < 5:
        _prev_gray_of = gray
        _prev_pts_of  = None
        return 0.0

    raw_motion     = float(np.mean(np.linalg.norm(good_new - good_old, axis=1)) * 5)
    _motion_smooth = 0.8 * _motion_smooth + 0.2 * raw_motion

    # Refresh feature points occasionally to avoid drift
    _prev_pts_of = (
        cv2.goodFeaturesToTrack(gray, 300, 0.01, 5)
        if np.random.rand() < 0.1
        else good_new.reshape(-1, 1, 2)
    )
    _prev_gray_of = gray
    return _motion_smooth


# ─────────────────────────────────────────────────────────────────────────────
# Visual odometry  (rotation only — translation is unit-norm from recoverPose)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_rotation_from_frame(
    bgr: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray | None:
    """
    Estimate inter-frame rotation using ORB + Essential matrix.

    Returns R (3×3) or None if not enough features / matches.

    NOTE: translation from recoverPose is NOT metric (unit-norm vector).
    We return only R and discard t.  The point clouds carry metric geometry
    via ZoeDepth, so rotation-only pose accumulation is correct [F2].
    """
    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    orb     = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(gray, None)

    if des is None or len(kp) < 20:
        scan_session.update(prev_gray=gray, prev_kp=kp, prev_des=des)
        return None

    if scan_session["prev_gray"] is None or scan_session["prev_des"] is None:
        scan_session.update(prev_gray=gray, prev_kp=kp, prev_des=des)
        return np.eye(3, dtype=np.float64)   # first frame — identity

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(scan_session["prev_des"], des, k=2)
    good    = [
        m for pair in matches
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    ]

    if len(good) < 15:
        scan_session.update(prev_gray=gray, prev_kp=kp, prev_des=des)
        return None

    pts1 = np.float32([scan_session["prev_kp"][m.queryIdx].pt for m in good])
    pts2 = np.float32([kp[m.trainIdx].pt                      for m in good])

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        scan_session.update(prev_gray=gray, prev_kp=kp, prev_des=des)
        return None

    _, R, _t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    scan_session.update(prev_gray=gray, prev_kp=kp, prev_des=des)
    return R


# ─────────────────────────────────────────────────────────────────────────────
# IMU helpers
# ─────────────────────────────────────────────────────────────────────────────

def imu_delta_rotation(imu_readings: list) -> np.ndarray | None:
    """Integrate gyroscope readings into a delta rotation matrix."""
    if not imu_readings or len(imu_readings) < 2:
        return None
    dt_total = (imu_readings[-1]["ts"] - imu_readings[0]["ts"]) / 1000.0
    if dt_total <= 0:
        return None
    deg2rad = np.pi / 180.0
    n       = len(imu_readings)
    ax = sum(r.get("gx", 0) for r in imu_readings) * deg2rad * (dt_total / n)
    ay = sum(r.get("gy", 0) for r in imu_readings) * deg2rad * (dt_total / n)
    az = sum(r.get("gz", 0) for r in imu_readings) * deg2rad * (dt_total / n)
    Rx, _ = cv2.Rodrigues(np.array([ax, 0,  0]))
    Ry, _ = cv2.Rodrigues(np.array([0,  ay, 0]))
    Rz, _ = cv2.Rodrigues(np.array([0,  0,  az]))
    return Rz @ Ry @ Rx


def rotation_angle_diff(R1: np.ndarray, R2: np.ndarray) -> float:
    trace = np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def fuse_rotation(R_orb: np.ndarray, R_imu: np.ndarray | None) -> np.ndarray:
    """
    Fuse ORB visual odometry rotation with IMU delta rotation.
    Falls back to IMU if the two disagree by more than 15° (likely ORB error).
    Falls back to ORB if IMU is unavailable.
    """
    if R_imu is None:
        return R_orb
    if rotation_angle_diff(R_orb, R_imu) > 15.0:
        print("[pose] ORB/IMU disagree > 15° — using IMU")
        return R_imu
    return R_orb


# ─────────────────────────────────────────────────────────────────────────────
# SAM segmentation helper
# ─────────────────────────────────────────────────────────────────────────────

def segment_bbox(
    bbox:  list[int],
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Run SAM on one bounding box.  Falls back to filled rectangle if mask is
    empty (e.g. SAM checkpoint mismatch or textureless surface).
    """
    x1, y1, x2, y2 = bbox
    masks, _, _ = sam_predictor.predict(
        box=np.array([x1, y1, x2, y2]),
        multimask_output=False,
    )
    mask = (masks[0] > 0.5).astype(np.uint8)
    if mask.sum() < 20:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "depth_model":  zoe_runner.model_type,
        "depth_loaded": zoe_runner.is_loaded,
        "device":       _device,
        "scale_mode":   "metric_absolute",
    })


# ─────────────────────────────────────────────────────────────────────────────
# /detect — live single-frame estimation
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    """
    Single-frame detection + measurement.

    Form fields
    -----------
    image   : JPEG/PNG frame
    detect  : "1" run full detection (default), "0" return last_detections
    fx, fy  : focal lengths in pixels (defaults: 554)
    img_w   : frame width  in pixels
    img_h   : frame height in pixels

    Response
    --------
    {
      "scene": [ {object_id, label, confidence, bbox, center, dimensions}, … ],
      "motion": float
    }

    dimensions dict:
    {
      "object_id", "label",
      "length", "width", "height",   ← metres
      "volume_m3",
      "confidence", "point_count", "mean_depth_m"
    }
    """
    global last_detections
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        run_detection = request.form.get("detect", "1") == "1"

        npimg = np.frombuffer(request.files["image"].read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid image"}), 400

        motion = compute_camera_motion(frame)

        fx    = float(request.form.get("fx",    DEFAULT_FX))
        fy    = float(request.form.get("fy",    DEFAULT_FY))
        img_w = int(request.form.get("img_w",   frame.shape[1]))
        img_h = int(request.form.get("img_h",   frame.shape[0]))
        cx    = img_w / 2.0
        cy    = img_h / 2.0

        if not run_detection:
            return jsonify({"scene": last_detections, "motion": motion})

        # ── Detection ─────────────────────────────────────────────────────────
        results = yolo_model(frame, verbose=False)[0]
        boxes   = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()

        detections = []
        sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for box, cls, conf in zip(boxes, classes, confs):
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box)
            label   = yolo_model.names[int(cls)]
            mask_np = segment_bbox([x1, y1, x2, y2], img_h, img_w)
            detections.append({
                "bbox":           [x1, y1, x2, y2],
                "confidence":     float(conf),
                "mask":           mask_np,
                "business_class": label,
                "area":           float((x2 - x1) * (y2 - y1)),
                "center":         [float((x1 + x2) / 2), float((y1 + y2) / 2)],
            })

        # ── Metric depth ──────────────────────────────────────────────────────
        # ZoeDepth returns absolute depth in metres — no scale factor [F1]
        depth_map = infer_metric_depth(frame)
        print(f"[/detect] depth stats: {zoe_runner.depth_stats(depth_map)}")

        # ── Tracking ──────────────────────────────────────────────────────────
        detections = tracker.update(detections)

        # ── Measure each object  [F1] ─────────────────────────────────────────
        # measure_object_dimensions uses two-pass sampling:
        #   pass 1 (central crop 80%) → depth-based L/W/H
        #   pass 2 (full mask)        → height refinement (never crops top/bottom)
        scene = []
        for det in detections:
            oid  = int(det["id"])
            raw  = measure_object_dimensions(
                depth_map, det["mask"],
                fx=fx, fy=fy, cx=cx, cy=cy,
            )
            dims = None
            if raw is not None:
                raw["object_id"] = oid
                raw["label"]     = det["business_class"]

                # Temporal smoothing per object
                smoothed = _scene_smoother.update(oid, raw)

                # Depth confidence
                zs_mask   = depth_map[det["mask"] > 0.5]
                zs_valid  = zs_mask[zs_mask > 0.05]
                d_conf    = depth_confidence(zs_valid) if len(zs_valid) > 0 else 0.5
                conf_val  = final_confidence(
                    det       = det["confidence"],
                    seg       = 0.5,
                    depth_rel = d_conf,
                    track     = 0.5,
                )
                smoothed["confidence"] = conf_val
                dims = smoothed

                print(
                    f"[/detect] id={oid}  "
                    f"L={dims['length']:.3f}m  W={dims['width']:.3f}m  "
                    f"H={dims['height']:.3f}m  conf={conf_val:.2f}"
                )

            scene.append({
                "object_id":  oid,
                "label":      det["business_class"],
                "confidence": float(det["confidence"]),
                "bbox":       det["bbox"],
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
# /start_scan
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/start_scan", methods=["POST"])
def start_scan():
    """Reset scan session and begin collecting frames."""
    reset_scan_session()
    scan_session["active"] = True
    _scene_smoother.reset()
    print("[server] Scan session started")
    return jsonify({"status": "scan started"})


# ─────────────────────────────────────────────────────────────────────────────
# /scan_frame — multi-frame accumulation
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/scan_frame", methods=["POST"])
def scan_frame():
    """
    Accept one frame into the active scan session.

    Each frame:
      1. Detects the primary parcel with YOLO
      2. Runs ZoeDepth for metric depth
      3. Estimates inter-frame rotation (ORB + IMU fusion)  [F2: rotation only]
      4. Segments with SAM
      5. Back-projects masked pixels to world-space 3D using ZoeDepth depth
      6. Appends world-space point cloud (rotated by accumulated pose_R)

    call /compute_dimensions after ≥ 3 frames.

    Form fields
    -----------
    image   : JPEG/PNG frame
    imu     : JSON list of IMU readings [{ts, gx, gy, gz}, …]
    fx, fy  : focal lengths in pixels
    img_w, img_h : frame dimensions

    Response
    --------
    {"status": "ok", "frame_count": N}
    or {"status": "skipped", "reason": "…", "frame_count": N}
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        npimg = np.frombuffer(request.files["image"].read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "bad frame"}), 400

        imu_readings = json.loads(request.form.get("imu", "[]"))
        fx    = float(request.form.get("fx",    DEFAULT_FX))
        fy    = float(request.form.get("fy",    DEFAULT_FY))
        img_w = int(request.form.get("img_w",   frame.shape[1]))
        img_h = int(request.form.get("img_h",   frame.shape[0]))
        cx    = img_w / 2.0
        cy    = img_h / 2.0

        # ── Detection ─────────────────────────────────────────────────────────
        results = yolo_model(frame, verbose=False)[0]
        boxes   = results.boxes.xyxy.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()
        bboxes  = [list(map(int, b)) for b, c in zip(boxes, confs) if c >= CONF_THRESHOLD]

        if not bboxes:
            print("[scan_frame] Skipped: no detection")
            return jsonify({
                "status":      "skipped",
                "reason":      "no_detection",
                "frame_count": len(scan_session["frames"]),
            })

        # ── Metric depth ──────────────────────────────────────────────────────
        depth_map = infer_metric_depth(frame)
        print(f"[scan_frame] depth: {zoe_runner.depth_stats(depth_map)}")

        # ── Rotation estimation  [F2] ─────────────────────────────────────────
        # Accumulate rotation ONLY.  Translation from recoverPose is unit-norm
        # (not metric) so we do NOT accumulate it.  The ZoeDepth depth values
        # are metric, so the point clouds carry the correct 3D scale without
        # needing metric translation.
        R_orb = estimate_rotation_from_frame(frame, fx, fy, cx, cy)

        if R_orb is None:
            current_R = scan_session["pose_R"].copy()
            print("[scan_frame] Pose failed — using last known rotation")
        else:
            R_imu = imu_delta_rotation(imu_readings)
            R     = fuse_rotation(R_orb, R_imu)
            scan_session["pose_R"] = R @ scan_session["pose_R"]
            current_R = scan_session["pose_R"].copy()

        # ── SAM segmentation ──────────────────────────────────────────────────
        x1, y1, x2, y2 = bboxes[0]
        sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask_np = segment_bbox([x1, y1, x2, y2], img_h, img_w)

        # ── Pixel sampling — two passes  [F4] ────────────────────────────────
        # Pass 1: central crop (shrink=0.10) for clean depth pixels
        us_c, vs_c = get_central_pixels(mask_np, shrink=0.10)
        if len(us_c) < 20:
            print("[scan_frame] Skipped: mask too small after crop")
            return jsonify({
                "status":      "skipped",
                "reason":      "mask_too_small",
                "frame_count": len(scan_session["frames"]),
            })

        zs_raw = depth_map[vs_c, us_c].astype(np.float32)
        valid  = zs_raw > 0.05
        us_v, vs_v, zs_v = us_c[valid], vs_c[valid], zs_raw[valid]
        if len(us_v) < 10:
            return jsonify({
                "status":      "skipped",
                "reason":      "depth_invalid",
                "frame_count": len(scan_session["frames"]),
            })

        p5,  p95 = np.percentile(zs_v, 5), np.percentile(zs_v, 95)
        keep     = (zs_v >= p5) & (zs_v <= p95)
        us_c2, vs_c2, zs_c2 = us_v[keep], vs_v[keep], zs_v[keep]
        if len(us_c2) < 10:
            return jsonify({
                "status":      "skipped",
                "reason":      "too_few_depth_pixels",
                "frame_count": len(scan_session["frames"]),
            })

        # Pass 2: full mask for height completeness
        us_f, vs_f = get_full_mask_pixels(mask_np)
        zs_f       = depth_map[vs_f, us_f].astype(np.float32)
        valid_f    = zs_f > 0.05
        us_f, vs_f, zs_f = us_f[valid_f], vs_f[valid_f], zs_f[valid_f]
        if len(us_f) >= 20:
            lo_f, hi_f   = np.percentile(zs_f, 5), np.percentile(zs_f, 95)
            keep_f        = (zs_f >= lo_f) & (zs_f <= hi_f)
            us_f, vs_f, zs_f = us_f[keep_f], vs_f[keep_f], zs_f[keep_f]

        # Merge both passes
        us_all = np.concatenate([us_c2, us_f]) if len(us_f) >= 20 else us_c2
        vs_all = np.concatenate([vs_c2, vs_f]) if len(vs_f) >= 20 else vs_c2
        zs_all = np.concatenate([zs_c2, zs_f]) if len(zs_f) >= 20 else zs_c2

        # Subsample if huge
        if len(us_all) > 3000:
            idx    = np.random.choice(len(us_all), 3000, replace=False)
            us_all, vs_all, zs_all = us_all[idx], vs_all[idx], zs_all[idx]

        # ── 3D projection → world frame ───────────────────────────────────────
        # zs_all is in metres from ZoeDepth — no * scale
        pts_cam   = pixels_to_3d(
            us_all.astype(float), vs_all.astype(float),
            zs_all, fx, fy, cx, cy,
        )
        pts_cam   = filter_point_cloud(pts_cam)
        if len(pts_cam) < 10:
            return jsonify({
                "status":      "skipped",
                "reason":      "point_cloud_too_sparse",
                "frame_count": len(scan_session["frames"]),
            })

        # Rotate into world frame using accumulated rotation  [F2]
        pts_world = (current_R @ pts_cam.T).T

        scan_session["frames"].append({"pts_world": pts_world})
        n = len(scan_session["frames"])
        print(
            f"[scan_frame] Frame {n} accepted — "
            f"pts={len(pts_world)}  depth_mean={zs_all.mean():.3f}m"
        )

        return jsonify({"status": "ok", "frame_count": n})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# /compute_dimensions — fuse all scan frames into final L×W×H
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/compute_dimensions", methods=["POST"])
def compute_dimensions():
    """
    Fuse all accumulated scan frames and return final dimensions.

    Requires ≥ 3 frames.  Resets the scan session on return.

    Response
    --------
    {
      "dimensions": {
        "length", "width", "height",   ← metres
        "volume_m3",
        "pca_reliable"                 ← bool: True if vertical axis was identified
      }
    }

    Notes on pca_reliable
    ─────────────────────
    fit_pca_bbox identifies the vertical axis by finding which PCA eigenvector
    is most aligned with world-up [0,1,0].  If pca_reliable=False the world
    point cloud was too planar or degenerate to identify the vertical axis
    confidently; dims are sorted by variance (largest=length) instead.
    Improve by scanning more angles or ensuring the camera is not level with
    the top of the box.
    """
    try:
        frames = scan_session["frames"]
        if len(frames) < 3:
            reset_scan_session()
            return jsonify({
                "error": f"only {len(frames)} frames — scan at least 3"
            }), 400

        all_pts = np.vstack([f["pts_world"] for f in frames])
        print(
            f"[compute_dimensions] Fusing {len(all_pts)} pts "
            f"from {len(frames)} frames"
        )

        # Two-pass outlier removal
        for sigma in (2.5, 2.0):
            if len(all_pts) < 20:
                break
            centroid = all_pts.mean(axis=0)
            dists    = np.linalg.norm(all_pts - centroid, axis=1)
            all_pts  = all_pts[dists < dists.mean() + sigma * dists.std()]

        print(f"[compute_dimensions] After filtering: {len(all_pts)} pts")
        if len(all_pts) < 20:
            reset_scan_session()
            return jsonify({"error": "not enough clean points after filtering"}), 400

        # PCA bounding box with semantic axis resolution  [F3]
        box = fit_pca_bbox(all_pts)
        if box is None:
            reset_scan_session()
            return jsonify({"error": "PCA fit failed"}), 400

        print(
            f"[compute_dimensions] Result: "
            f"L={box['length']:.3f}m  W={box['width']:.3f}m  "
            f"H={box['height']:.3f}m  V={box['volume_m3']:.4f}m³  "
            f"pca_reliable={box.get('pca_reliable')}"
        )
        reset_scan_session()
        return jsonify({"dimensions": box})

    except Exception as e:
        traceback.print_exc()
        reset_scan_session()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Legacy calibration stubs — kept for frontend compatibility
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/start_calibration", methods=["POST"])
def start_calibration():
    """No-op — ZoeDepth is internally calibrated, no user calibration needed."""
    return jsonify({"status": "not_required", "reason": "ZoeDepth is metric"})


@app.route("/end_calibration", methods=["POST"])
def end_calibration():
    """No-op — see /start_calibration."""
    return jsonify({"status": "not_required", "reason": "ZoeDepth is metric"})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ParcelVision Metric Depth Server  (Stage 7 — production)")
    print(f"  device      : {_device}")
    print("  depth model : ZoeDepth ZoeD_NK (metric, no scale factor)")
    print("  endpoints   :")
    print("    GET  /health")
    print("    POST /detect")
    print("    POST /start_scan")
    print("    POST /scan_frame")
    print("    POST /compute_dimensions")
    print("    POST /start_calibration  (no-op stub)")
    print("    POST /end_calibration    (no-op stub)")
    print("=" * 60)

    # Pre-load ZoeDepth at startup to avoid first-request latency
    print("[server] Pre-loading ZoeDepth …")
    zoe_runner.load()
    print("[server] Ready.\n")

    app.run(host="0.0.0.0", port=5000, threaded=False)