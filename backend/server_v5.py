from __future__ import annotations

import json
import traceback

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

from tracker import ObjectTracker
from scene_state import SceneStateManager

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.01

# ISO/IEC 7810 ID-1 credit card (metres)
CARD_W = 0.08560
CARD_H = 0.05398
CARD_ASPECT = CARD_W / CARD_H  # ≈ 1.586

# Card corners in world coords: TL, TR, BR, BL (Z=0 plane)
CARD_OBJ_PTS = np.array([
    [0.0,   0.0,   0.0],
    [CARD_W, 0.0,  0.0],
    [CARD_W, CARD_H, 0.0],
    [0.0,   CARD_H, 0.0],
], dtype=np.float64)

YOLO_CLASSES = [
    "box", "cardboard box", "carton", "parcel",
    "package", "container", "brown box",
    "shipping box", "crate", "rectangular box",
    "credit card", "card",
]

# ─────────────────────────────────────────────────────────────────────────────
# App + models
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

_device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading YOLO World...")
yolo = YOLO("yolov8s-world.pt")
yolo.to(_device)
yolo.set_classes(YOLO_CLASSES)

print("Loading SAM...")
_sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
_sam.to(_device)
sam = SamPredictor(_sam)

tracker = ObjectTracker()
scene_manager = SceneStateManager()

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

# Calibration: accumulates PnP frames, locks R/t/K
class CalibState:
    def __init__(self):
        self.rvecs: list[np.ndarray] = []
        self.tvecs: list[np.ndarray] = []
        self.R: np.ndarray | None = None
        self.t: np.ndarray | None = None
        self.K: np.ndarray | None = None
        self._prev_t: np.ndarray | None = None

    def reset(self):
        self.__init__()

    @property
    def count(self) -> int:
        return len(self.rvecs)

    @property
    def ready(self) -> bool:
        return self.count >= 10

    @property
    def locked(self) -> bool:
        return self.R is not None

    def add(self, rvec: np.ndarray, tvec: np.ndarray):
        self.rvecs.append(rvec.copy())
        self.tvecs.append(tvec.copy())

    def finalize(self, K: np.ndarray) -> bool:
        if self.count < 3:
            return False
        tvecs = np.array([t.ravel() for t in self.tvecs])
        med_t = np.median(tvecs, axis=0).reshape(3, 1)
        # Pick frame whose tvec is closest to median
        best = int(np.argmin(np.linalg.norm(tvecs - med_t.ravel(), axis=1)))
        R, _ = cv2.Rodrigues(self.rvecs[best])
        # EMA smoothing vs previous calibration
        if self._prev_t is not None:
            med_t = 0.8 * self._prev_t + 0.2 * med_t
        self.R = R
        self.t = med_t
        self.K = K.copy()
        self._prev_t = med_t
        self.rvecs.clear()
        self.tvecs.clear()
        print(f"✅ Calibration locked: t={med_t.ravel().round(3)}  cam_pos={self.cam_pos().round(3)}")
        return True

    def cam_pos(self) -> np.ndarray:
        return (-self.R.T @ self.t).ravel()


calib = CalibState()

# Scan session
scan: dict = {
    "active": False,
    "frames": [],
    "pose_R": np.eye(3, dtype=np.float64),
    "prev_gray": None,
    "prev_kp": None,
    "prev_des": None,
}

# Per-object smoothed dimensions (live mode)
_prev_dims: dict[int, dict] = {}

# Last detections (returned when detection is skipped)
_last_scene: list = []

# Motion estimation state
_prev_gray_motion: np.ndarray | None = None
_motion_smooth: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_K(fx: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)


def sanitize_fx(fx: float, img_w: int) -> float:
    if not (300 <= fx <= 3000):
        print(f"⚠️  fx={fx:.1f} out of range — using 65° HFOV fallback")
        return (img_w / 2.0) / np.tan(np.radians(32.5))
    return fx


def parse_K(form, img_w: int, img_h: int) -> np.ndarray:
    fx = sanitize_fx(float(form.get("fx", 554.0)), img_w)
    return build_K(fx, img_w / 2.0, img_h / 2.0)

# ─────────────────────────────────────────────────────────────────────────────
# PnP helpers
# ─────────────────────────────────────────────────────────────────────────────

def solve_card_pnp(img_pts: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Solve PnP for credit card corners. img_pts: (4,2) float64, clockwise TL→TR→BR→BL."""
    if img_pts.shape != (4, 2):
        return None
    ok, rvec, tvec = cv2.solvePnP(CARD_OBJ_PTS, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE)
    return (rvec, tvec) if ok else None


def reproj_error(rvec, tvec, K, img_pts: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(CARD_OBJ_PTS, rvec, tvec, K, None)
    return float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1)))


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into TL, TR, BR, BL order."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float64)


def extract_card_corners(mask: np.ndarray) -> np.ndarray | None:
    """Extract 4 ordered corners from card mask. Returns (4,2) or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 500:
        return None
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    pts = approx.reshape(-1, 2).astype(np.float64) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float64)
    return order_corners(pts)

# ─────────────────────────────────────────────────────────────────────────────
# Ray → floor intersection
# ─────────────────────────────────────────────────────────────────────────────

def ray_floor(u: float, v: float, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """Cast ray through pixel (u,v), intersect with Z=0 world plane. Returns [X,Y,0] or None."""
    ray_world = R.T @ (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    C = (-R.T @ t).ravel()
    if abs(ray_world[2]) < 1e-8:
        return None
    s = -C[2] / ray_world[2]
    return C + s * ray_world if s > 0 else None

# ─────────────────────────────────────────────────────────────────────────────
# Box measurement via floor projection
# ─────────────────────────────────────────────────────────────────────────────

def measure_box(bbox: list[int], mask: np.ndarray,
                R: np.ndarray, t: np.ndarray, K: np.ndarray) -> dict | None:
    vs, us = np.where(mask > 0)
    if len(us) < 20:
        return None

    # --- Base footprint: bottom 30% of mask pixels ---
    v_thresh = np.percentile(vs, 70)
    bot = vs >= v_thresh
    b_us, b_vs = us[bot], vs[bot]

    step = max(1, len(b_us) // 80)
    floor_pts = []
    for u, v in zip(b_us[::step], b_vs[::step]):
        P = ray_floor(float(u), float(v), R, t, K)
        if P is not None:
            floor_pts.append(P[:2])

    if len(floor_pts) < 8:
        return None

    fp = np.array(floor_pts)
    mean_fp = fp.mean(axis=0)
    _, eigvecs = np.linalg.eigh(np.cov((fp - mean_fp).T))
    proj = (fp - mean_fp) @ eigvecs
    length = float(np.percentile(proj[:, 1], 95) - np.percentile(proj[:, 1], 5))
    width  = float(np.percentile(proj[:, 0], 95) - np.percentile(proj[:, 0], 5))

    # --- Height: top 10% of mask pixels → vertical world distance ---
    top = vs <= np.percentile(vs, 10)
    t_us, t_vs = us[top], vs[top]
    height = _estimate_height(t_us, t_vs, R, t, K, mean_fp)

    if not (0.01 <= length <= 3.0 and 0.01 <= width <= 3.0 and 0.005 <= height <= 3.0):
        return None
    if width > length:
        length, width = width, length

    return {"length": round(length, 3), "width": round(width, 3),
            "height": round(height, 3), "volume_m3": round(length * width * height, 4)}


def _estimate_height(t_us, t_vs, R, t, K, base_xy) -> float:
    K_inv = np.linalg.inv(K)
    C = (-R.T @ t).ravel()
    heights = []
    step = max(1, len(t_us) // 60)
    for u, v in zip(t_us[::step], t_vs[::step]):
        ray = R.T @ (K_inv @ np.array([float(u), float(v), 1.0]))
        A = ray[:2].reshape(2, 1)
        b = (base_xy - C[:2]).reshape(2, 1)
        if np.linalg.norm(A) < 1e-8:
            continue
        s = float((A.T @ b) / (A.T @ A))
        if s < 0:
            continue
        h = float((C + s * ray)[2])
        if 0.001 < h < 3.0:
            heights.append(h)
    return float(np.median(heights)) if heights else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Temporal smoothing
# ─────────────────────────────────────────────────────────────────────────────

def smooth_dims(prev: dict | None, new: dict) -> dict:
    if prev is None:
        return new
    jump = max(
        abs(new["length"] - prev["length"]) / (prev["length"] + 1e-6),
        abs(new["width"]  - prev["width"])  / (prev["width"]  + 1e-6),
        abs(new["height"] - prev["height"]) / (prev["height"] + 1e-6),
    )
    if jump > 0.4:
        return new  # large jump → accept raw
    a = 0.4
    s = {k: a * prev[k] + (1 - a) * new[k] for k in ("length", "width", "height")}
    s["volume_m3"] = s["length"] * s["width"] * s["height"]
    return s

# ─────────────────────────────────────────────────────────────────────────────
# Card aspect check
# ─────────────────────────────────────────────────────────────────────────────

def check_card_aspect(pixel_w: float, pixel_h: float, img_w: int) -> tuple[bool, str]:
    if pixel_w <= 0 or pixel_h <= 0:
        return False, "zero dimensions"
    if pixel_w < img_w * 0.03 or pixel_h < img_w * 0.03:
        return False, f"card too small ({pixel_w:.0f}×{pixel_h:.0f}px)"
    tol = 0.18
    lo, hi = CARD_ASPECT * (1 - tol), CARD_ASPECT * (1 + tol)
    for ar in (pixel_w / pixel_h, pixel_h / pixel_w):
        if lo <= ar <= hi:
            return True, "ok"
    return False, f"bad aspect {pixel_w/pixel_h:.2f} (expected {CARD_ASPECT:.2f}±{tol*100:.0f}%)"

# ─────────────────────────────────────────────────────────────────────────────
# Motion estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_motion(frame: np.ndarray) -> float:
    global _prev_gray_motion, _motion_smooth
    gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
    if _prev_gray_motion is None:
        _prev_gray_motion = gray
        return 0.0
    pts = cv2.goodFeaturesToTrack(_prev_gray_motion, 300, 0.01, 5)
    if pts is None:
        _prev_gray_motion = gray
        return 0.0
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(_prev_gray_motion, gray, pts, None)
    if next_pts is None or status is None:
        _prev_gray_motion = gray
        return 0.0
    good = status.ravel() == 1
    if good.sum() < 5:
        _prev_gray_motion = gray
        return 0.0
    raw = float(np.mean(np.linalg.norm(next_pts[good] - pts[good], axis=1)) * 5)
    _motion_smooth = 0.8 * _motion_smooth + 0.2 * raw
    _prev_gray_motion = gray
    return _motion_smooth

# ─────────────────────────────────────────────────────────────────────────────
# Visual odometry (scan mode)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_rotation(bgr: np.ndarray, K: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(gray, None)

    prev_gray = scan["prev_gray"]
    prev_des  = scan["prev_des"]
    prev_kp   = scan["prev_kp"]
    scan.update(prev_gray=gray, prev_kp=kp, prev_des=des)

    if des is None or len(kp) < 20 or prev_gray is None or prev_des is None:
        return np.eye(3, dtype=np.float64)

    matches = [m for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(prev_des, des, k=2)
               if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(matches) < 15:
        return np.eye(3, dtype=np.float64)

    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp[m.trainIdx].pt      for m in matches])
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return np.eye(3, dtype=np.float64)
    _, R, _, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R

# ─────────────────────────────────────────────────────────────────────────────
# SAM segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment(frame_rgb: np.ndarray, bbox: list[int], img_h: int, img_w: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    masks, _, _ = sam.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
    mask = (masks[0] > 0.5).astype(np.uint8)
    if mask.sum() < 20:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
    return mask

# ─────────────────────────────────────────────────────────────────────────────
# Scan session helpers
# ─────────────────────────────────────────────────────────────────────────────

def reset_scan():
    global _prev_dims
    scan.update(active=False, frames=[], pose_R=np.eye(3, dtype=np.float64),
                prev_gray=None, prev_kp=None, prev_des=None)
    _prev_dims.clear()
    print("🔄 Scan reset")

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    global _last_scene, _prev_dims
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        run_det = request.form.get("detect", "1") == "1"
        frame = cv2.imdecode(np.frombuffer(request.files["image"].read(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid image"}), 400

        motion = compute_motion(frame)
        img_h, img_w = frame.shape[:2]
        K = parse_K(request.form, img_w, img_h)

        if not run_det:
            return jsonify({"scene": _last_scene, "motion": motion, "calibrated": calib.locked})

        results = yolo(frame, verbose=False)[0]
        detections = []
        sam.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                   results.boxes.cls.cpu().numpy(),
                                   results.boxes.conf.cpu().numpy()):
            if conf < CONF_THRESHOLD:
                continue
            bbox = list(map(int, box))
            label = yolo.names.get(int(cls), f"class_{int(cls)}")
            mask = segment(None, bbox, img_h, img_w)  # SAM already has image set
            detections.append({
                "bbox": bbox, "confidence": float(conf), "mask": mask,
                "business_class": label, "area": float((bbox[2]-bbox[0])*(bbox[3]-bbox[1])),
                "center": [float((bbox[0]+bbox[2])/2), float((bbox[1]+bbox[3])/2)],
            })

        detections = tracker.update(detections)
        scene = []

        for det in detections:
            label = det["business_class"].lower()
            dims = None

            if "card" in label and not calib.locked and calib.count > 0:
                # Contribute card frame to calibration
                corners = extract_card_corners(det["mask"])
                if corners is not None:
                    result = solve_card_pnp(corners, K)
                    if result and reproj_error(*result, K, corners) < 8.0:
                        calib.add(*result)

            elif "card" not in label and calib.locked:
                raw = measure_box(det["bbox"], det["mask"], calib.R, calib.t, calib.K)
                if raw:
                    oid = det["id"]
                    dims = smooth_dims(_prev_dims.get(oid), raw)
                    _prev_dims[oid] = dims

            scene.append({
                "object_id":  int(det["id"]),
                "label":      det["business_class"],
                "confidence": float(det["confidence"]),
                "bbox":       det["bbox"],
                "center":     det["center"],
                "dimensions": dims,
            })

        if scene:
            _last_scene = scene

        return jsonify({"scene": _last_scene, "motion": motion,
                        "calibrated": calib.locked, "cal_frames": calib.count})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/calibrate_frame", methods=["POST"])
def calibrate_frame():
    """
    Accumulate one calibration frame from card corners or bbox fallback.
    Auto-finalizes after 10 valid frames.
    """
    try:
        img_w = int(request.form.get("img_w", 1280))
        img_h = int(request.form.get("img_h", 720))
        K = parse_K(request.form, img_w, img_h)

        # Prefer explicit corners JSON
        image_points = None
        corners_json = request.form.get("corners")
        if corners_json:
            pts = np.array(json.loads(corners_json), dtype=np.float64)
            if pts.shape == (4, 2):
                image_points = pts

        # Fallback: synthesize corners from bbox
        if image_points is None:
            pixel_w = float(request.form.get("pixel_w", 0))
            pixel_h = float(request.form.get("pixel_h", 0))
            ok, reason = check_card_aspect(pixel_w, pixel_h, img_w)
            if not ok:
                return jsonify({"status": "rejected", "reason": reason,
                                "count": calib.count, "ready": calib.ready})
            cx, cy = img_w / 2.0, img_h / 2.0
            bx = float(request.form.get("bbox_x", cx - pixel_w / 2))
            by = float(request.form.get("bbox_y", cy - pixel_h / 2))
            image_points = np.array([[bx, by], [bx+pixel_w, by],
                                      [bx+pixel_w, by+pixel_h], [bx, by+pixel_h]], dtype=np.float64)

        result = solve_card_pnp(image_points, K)
        if result is None:
            return jsonify({"status": "rejected", "reason": "PnP failed",
                            "count": calib.count, "ready": calib.ready})

        err = reproj_error(*result, K, image_points)
        if err > 10.0:
            return jsonify({"status": "rejected",
                            "reason": f"reprojection error {err:.1f}px > 10px",
                            "count": calib.count, "ready": calib.ready})

        calib.add(*result)
        print(f"📐 Cal frame #{calib.count}  reproj={err:.2f}px")

        resp = {"status": "accepted", "count": calib.count,
                "ready": calib.ready, "reproj_error": round(err, 2)}

        if calib.ready:
            calib.finalize(K)
            resp["cam_pos"] = calib.cam_pos().round(4).tolist()

        return jsonify(resp)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/confirm_calibration", methods=["POST"])
def confirm_calibration():
    """Force-finalize calibration with whatever frames we have (min 3)."""
    try:
        img_w = int(request.form.get("img_w", 1280))
        img_h = int(request.form.get("img_h", 720))
        K = parse_K(request.form, img_w, img_h)

        # Accept one more frame if provided
        corners_json = request.form.get("corners")
        if corners_json:
            pts = np.array(json.loads(corners_json), dtype=np.float64)
            if pts.shape == (4, 2):
                result = solve_card_pnp(pts, K)
                if result:
                    calib.add(*result)

        pixel_w = float(request.form.get("pixel_w", 0))
        pixel_h = float(request.form.get("pixel_h", 0))
        if pixel_w > 0 and pixel_h > 0:
            cx, cy = img_w / 2.0, img_h / 2.0
            bx = float(request.form.get("bbox_x", cx - pixel_w / 2))
            by = float(request.form.get("bbox_y", cy - pixel_h / 2))
            pts = np.array([[bx, by], [bx+pixel_w, by],
                             [bx+pixel_w, by+pixel_h], [bx, by+pixel_h]], dtype=np.float64)
            result = solve_card_pnp(pts, K)
            if result:
                calib.add(*result)

        if calib.count < 3:
            return jsonify({"error": f"need ≥3 frames, have {calib.count}"}), 400

        if not calib.finalize(K):
            return jsonify({"error": "finalization failed"}), 500

        C = calib.cam_pos()
        return jsonify({"cam_pos": C.round(4).tolist(), "z_cam_m": round(float(C[2]), 3)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/reset_calibration", methods=["POST"])
def reset_calibration():
    calib.reset()
    return jsonify({"status": "reset"})


@app.route("/start_scan", methods=["POST"])
def start_scan():
    reset_scan()
    scan["active"] = True
    return jsonify({"status": "scan started"})


@app.route("/scan_frame", methods=["POST"])
def scan_frame():
    try:
        if not calib.locked:
            return jsonify({"status": "skipped", "reason": "not_calibrated",
                            "frame_count": len(scan["frames"])}), 200

        frame = cv2.imdecode(np.frombuffer(request.files["image"].read(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "bad frame"}), 400

        img_h, img_w = frame.shape[:2]
        K = parse_K(request.form, img_w, img_h)

        # Detect box
        results = yolo(frame, verbose=False)[0]
        bboxes = [list(map(int, b)) for b, c in zip(results.boxes.xyxy.cpu().numpy(),
                                                      results.boxes.conf.cpu().numpy())
                  if c >= CONF_THRESHOLD]
        if not bboxes:
            if _last_scene:
                bboxes = [_last_scene[0]["bbox"]]
            else:
                return jsonify({"status": "skipped", "reason": "no_detection",
                                "frame_count": len(scan["frames"])})

        # SAM mask
        sam.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = segment(None, bboxes[0], img_h, img_w)

        vs, us = np.where(mask > 0)
        if len(us) < 20:
            return jsonify({"status": "skipped", "reason": "mask_too_small",
                            "frame_count": len(scan["frames"])})

        # Accumulate visual rotation
        R_delta = estimate_rotation(frame, K)
        scan["pose_R"] = R_delta @ scan["pose_R"]
        cur_R = scan["pose_R"].copy()

        # Project mask pixels → world floor points
        world_pts = []
        step = max(1, len(us) // 150)
        for u, v in zip(us[::step], vs[::step]):
            P = ray_floor(float(u), float(v), calib.R, calib.t, calib.K)
            if P is not None:
                P_rot = cur_R[:2, :2] @ P[:2]
                world_pts.append([P_rot[0], P_rot[1], P[2]])

        if len(world_pts) < 10:
            return jsonify({"status": "skipped", "reason": "no_floor_projection",
                            "frame_count": len(scan["frames"])})

        scan["frames"].append({"pts": np.array(world_pts)})
        n = len(scan["frames"])
        print(f"✅ Scan frame {n} — {len(world_pts)} pts")
        return jsonify({"status": "ok", "frame_count": n})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/compute_dimensions", methods=["POST"])
def compute_dimensions():
    try:
        frames = scan["frames"]
        if len(frames) < 3:
            reset_scan()
            return jsonify({"error": f"only {len(frames)} frames — scan more"}), 400

        all_pts = np.vstack([f["pts"] for f in frames])
        print(f"Fusing {len(all_pts)} pts from {len(frames)} frames")

        # Two-pass outlier removal
        for sigma in (2.5, 2.0):
            if len(all_pts) < 20:
                break
            d = np.linalg.norm(all_pts - all_pts.mean(axis=0), axis=1)
            all_pts = all_pts[d < d.mean() + sigma * d.std()]

        if len(all_pts) < 20:
            reset_scan()
            return jsonify({"error": "not enough clean points after filtering"}), 400

        # PCA bounding box
        centered = all_pts[:, :2] - all_pts[:, :2].mean(axis=0)
        _, eigvecs = np.linalg.eigh(np.cov(centered.T))
        proj = centered @ eigvecs

        length = float(np.clip(np.percentile(proj[:, 1], 95) - np.percentile(proj[:, 1], 5), 0.005, 5.0))
        width  = float(np.clip(np.percentile(proj[:, 0], 95) - np.percentile(proj[:, 0], 5), 0.005, 5.0))
        height = float(np.clip(np.percentile(all_pts[:, 2], 95) - np.percentile(all_pts[:, 2], 5), 0.005, 3.0))
        if width > length:
            length, width = width, length

        print(f"✅ Final: L={length:.3f} W={width:.3f} H={height:.3f}")
        reset_scan()

        return jsonify({"dimensions": {
            "length": round(length, 3), "width": round(width, 3),
            "height": round(height, 3), "volume_m3": round(length * width * height, 4),
        }})

    except Exception as e:
        traceback.print_exc()
        reset_scan()
        return jsonify({"error": str(e)}), 500


# Legacy stubs
@app.route("/start_calibration", methods=["POST"])
def start_calibration():
    calib.reset()
    return jsonify({"status": "calibration started — use /calibrate_frame"})

@app.route("/end_calibration", methods=["POST"])
def end_calibration():
    return jsonify({"status": "deprecated — use /confirm_calibration"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)