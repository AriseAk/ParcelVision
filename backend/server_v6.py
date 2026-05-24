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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.25

# ISO/IEC 7810 ID-1 credit card (metres)
CARD_W = 0.08560
CARD_H = 0.05398
CARD_ASPECT = CARD_W / CARD_H  # ≈ 1.586

CARD_OBJ_PTS = np.array([
    [0.0,    0.0,    0.0],
    [CARD_W, 0.0,    0.0],
    [CARD_W, CARD_H, 0.0],
    [0.0,    CARD_H, 0.0],
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
# Calibration State
# ─────────────────────────────────────────────────────────────────────────────

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
        best = int(np.argmin(np.linalg.norm(tvecs - med_t.ravel(), axis=1)))
        R, _ = cv2.Rodrigues(self.rvecs[best])
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

    def floor_normal_world(self) -> np.ndarray:
        """World Z-axis [0,0,1] — floor normal always points up in our convention."""
        return np.array([0.0, 0.0, 1.0])


calib = CalibState()

# ─────────────────────────────────────────────────────────────────────────────
# PnP helpers
# ─────────────────────────────────────────────────────────────────────────────

def solve_card_pnp(img_pts: np.ndarray, K: np.ndarray):
    if img_pts.shape != (4, 2):
        return None
    ok, rvec, tvec = cv2.solvePnP(
        CARD_OBJ_PTS, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE
    )
    return (rvec, tvec) if ok else None


def reproj_error(rvec, tvec, K, img_pts: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(CARD_OBJ_PTS, rvec, tvec, K, None)
    return float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1)))


def order_corners(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)], pts[np.argmin(d)],
        pts[np.argmax(s)], pts[np.argmax(d)]
    ], dtype=np.float64)


def extract_card_corners(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 500:
        return None
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    pts = (approx.reshape(-1, 2).astype(np.float64)
           if len(approx) == 4
           else cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float64))
    return order_corners(pts)

# ─────────────────────────────────────────────────────────────────────────────
# CORE GEOMETRY — Pure Linear Algebra, No Heuristics
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_ray_world(u: float, v: float, R: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert a pixel (u,v) to a unit direction vector in world coordinates.
    Ray origin is the camera centre C = -R.T @ t
    """
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_world = R.T @ ray_cam
    return ray_world / np.linalg.norm(ray_world)


def ray_plane_intersect(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    plane_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
    plane_d: float = 0.0,
) -> np.ndarray | None:
    """
    Intersect ray with plane defined by dot(normal, X) = plane_d.
    Returns 3D world point or None if ray is parallel to plane.
    plane_normal=[0,0,1], plane_d=0  →  Z=0 floor plane.
    """
    denom = float(np.dot(plane_normal, ray_dir))
    if abs(denom) < 1e-8:
        return None
    t = (plane_d - float(np.dot(plane_normal, ray_origin))) / denom
    if t < 0:
        return None
    return ray_origin + t * ray_dir


def ray_line_closest_point(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    line_origin: np.ndarray,
    line_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Find the closest points between two skew lines (ray and vertical line).
    Uses least-squares on the parametric form:
        P1(s) = ray_origin + s * ray_dir
        P2(t) = line_origin + t * line_dir
    Returns (point_on_ray, point_on_line, residual_mm) or None.
    """
    d1 = ray_dir / np.linalg.norm(ray_dir)
    d2 = line_dir / np.linalg.norm(line_dir)

    # Build 2×2 system
    b = line_origin - ray_origin
    a11 = float(np.dot(d1, d1))
    a12 = -float(np.dot(d1, d2))
    a21 = float(np.dot(d1, d2))
    a22 = -float(np.dot(d2, d2))
    b1  = float(np.dot(b, d1))
    b2  = float(np.dot(b, d2))

    A = np.array([[a11, a12], [a21, a22]])
    rhs = np.array([b1, b2])

    try:
        params, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None

    s, t = params
    P1 = ray_origin + s * d1
    P2 = line_origin + t * d2
    residual = float(np.linalg.norm(P1 - P2)) * 1000  # mm
    return P1, P2, residual


def measure_face(
    corners_2d: dict,          # {"bl":[u,v], "br":[u,v], "tl":[u,v]}  pixels
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> dict | None:
    """
    Pure geometric measurement of one visible box face.

    corners_2d keys:
      bl  = bottom-left  pixel (where left vertical edge meets floor)
      br  = bottom-right pixel (where right vertical edge meets floor)
      tl  = top-left     pixel (top of the left vertical edge)

    Returns dict with width_m, height_m and debug info.
    """
    cam_pos = (-R.T @ t).ravel()          # camera centre in world coords
    floor_normal = np.array([0.0, 0.0, 1.0])

    bl_px = np.array(corners_2d["bl"], dtype=np.float64)
    br_px = np.array(corners_2d["br"], dtype=np.float64)
    tl_px = np.array(corners_2d["tl"], dtype=np.float64)

    # ── Step 1: Bottom corners → floor intersection ──────────────────────────
    ray_bl = pixel_to_ray_world(bl_px[0], bl_px[1], R, K)
    ray_br = pixel_to_ray_world(br_px[0], br_px[1], R, K)

    P_bl = ray_plane_intersect(cam_pos, ray_bl, floor_normal, 0.0)
    P_br = ray_plane_intersect(cam_pos, ray_br, floor_normal, 0.0)

    if P_bl is None or P_br is None:
        return {"error": "floor intersection failed — check calibration"}

    width = float(np.linalg.norm(P_br - P_bl))

    # ── Step 2: Top-left pixel → vertical line intersection ──────────────────
    ray_tl = pixel_to_ray_world(tl_px[0], tl_px[1], R, K)
    vertical_dir = np.array([0.0, 0.0, 1.0])  # straight up

    result = ray_line_closest_point(cam_pos, ray_tl, P_bl, vertical_dir)
    if result is None:
        return {"error": "ray-line intersection failed"}

    _, P_top, residual_mm = result
    height = float(P_top[2])  # Z coordinate = height above floor

    if height < 0.005 or height > 3.0:
        return {"error": f"height {height:.3f}m out of plausible range"}
    if width < 0.005 or width > 3.0:
        return {"error": f"width {width:.3f}m out of plausible range"}

    return {
        "width_m":      round(width, 4),
        "height_m":     round(height, 4),
        "P_bl":         P_bl.tolist(),
        "P_br":         P_br.tolist(),
        "P_top":        P_top.tolist(),
        "residual_mm":  round(residual_mm, 2),
    }


def fuse_two_faces(face1: dict, face2: dict) -> dict:
    """
    face1 = front face  → gives width1, height1
    face2 = side face   → gives width2 (= depth), height2

    We trust the larger width as the "length".
    Heights are median-filtered.
    """
    w1 = face1["width_m"]
    w2 = face2["width_m"]
    h1 = face1["height_m"]
    h2 = face2["height_m"]

    length = max(w1, w2)
    width  = min(w1, w2)
    height = float(np.median([h1, h2]))
    volume = round(length * width * height, 6)

    return {
        "length_m":  round(length, 4),
        "width_m":   round(width,  4),
        "height_m":  round(height, 4),
        "volume_m3": volume,
        "height_consistency_mm": round(abs(h1 - h2) * 1000, 1),
        "face1_residual_mm": face1.get("residual_mm"),
        "face2_residual_mm": face2.get("residual_mm"),
    }

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
# Motion estimation (kept for UX feedback only, not used in measurement)
# ─────────────────────────────────────────────────────────────────────────────

_prev_gray_motion: np.ndarray | None = None
_motion_smooth: float = 0.0


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
# SAM segmentation helper (for card detection only)
# ─────────────────────────────────────────────────────────────────────────────

def segment_bbox(bbox: list[int], img_h: int, img_w: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    masks, _, _ = sam.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
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
        "status": "ok",
        "calibrated": calib.locked,
        "cal_frames": calib.count,
        "device": _device,
    })


@app.route("/reset_calibration", methods=["POST"])
def reset_calibration():
    calib.reset()
    global _prev_gray_motion, _motion_smooth
    _prev_gray_motion = None
    _motion_smooth = 0.0
    return jsonify({"status": "reset"})


@app.route("/calibrate_frame", methods=["POST"])
def calibrate_frame():
    """
    Accept one calibration frame.
    Expects: image file + fx + img_w + img_h
    Optionally: corners JSON [[x,y],[x,y],[x,y],[x,y]]
                OR pixel_w, pixel_h, bbox_x, bbox_y (fallback)
    Auto-finalizes after 10 valid frames.
    """
    try:
        img_w = int(request.form.get("img_w", 1280))
        img_h = int(request.form.get("img_h", 720))
        K = parse_K(request.form, img_w, img_h)

        image_points = None

        # Prefer explicit corners
        corners_json = request.form.get("corners")
        if corners_json:
            pts = np.array(json.loads(corners_json), dtype=np.float64)
            if pts.shape == (4, 2):
                image_points = pts

        # Try to detect card from image if uploaded
        if image_points is None and "image" in request.files:
            frame = cv2.imdecode(
                np.frombuffer(request.files["image"].read(), np.uint8),
                cv2.IMREAD_COLOR
            )
            if frame is not None:
                results = yolo(frame, verbose=False)[0]
                sam.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for box, cls, conf in zip(
                    results.boxes.xyxy.cpu().numpy(),
                    results.boxes.cls.cpu().numpy(),
                    results.boxes.conf.cpu().numpy()
                ):
                    if conf < 0.15:
                        continue
                    label = yolo.names.get(int(cls), "")
                    if "card" not in label.lower():
                        continue
                    bbox = list(map(int, box))
                    mask = segment_bbox(bbox, img_h, img_w)
                    corners = extract_card_corners(mask)
                    if corners is not None:
                        image_points = corners
                        break

        # Fallback: synthetic corners from bbox dimensions
        if image_points is None:
            pixel_w = float(request.form.get("pixel_w", 0))
            pixel_h = float(request.form.get("pixel_h", 0))
            ok, reason = check_card_aspect(pixel_w, pixel_h, img_w)
            if not ok:
                return jsonify({
                    "status": "rejected", "reason": reason,
                    "count": calib.count, "ready": calib.ready
                })
            cx, cy = img_w / 2.0, img_h / 2.0
            bx = float(request.form.get("bbox_x", cx - pixel_w / 2))
            by = float(request.form.get("bbox_y", cy - pixel_h / 2))
            image_points = np.array([
                [bx, by], [bx + pixel_w, by],
                [bx + pixel_w, by + pixel_h], [bx, by + pixel_h]
            ], dtype=np.float64)

        result = solve_card_pnp(image_points, K)
        if result is None:
            return jsonify({
                "status": "rejected", "reason": "PnP failed",
                "count": calib.count, "ready": calib.ready
            })

        err = reproj_error(*result, K, image_points)
        if err > 12.0:
            return jsonify({
                "status": "rejected",
                "reason": f"reprojection error {err:.1f}px > 12px",
                "count": calib.count, "ready": calib.ready
            })

        calib.add(*result)
        print(f"📐 Cal frame #{calib.count}  reproj={err:.2f}px")

        resp = {
            "status": "accepted",
            "count": calib.count,
            "ready": calib.ready,
            "reproj_error": round(err, 2),
        }

        if calib.ready:
            calib.finalize(K)
            resp["locked"] = True
            resp["cam_pos"] = calib.cam_pos().round(4).tolist()

        return jsonify(resp)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/confirm_calibration", methods=["POST"])
def confirm_calibration():
    """Force-finalize with whatever frames we have (min 3)."""
    try:
        img_w = int(request.form.get("img_w", 1280))
        img_h = int(request.form.get("img_h", 720))
        K = parse_K(request.form, img_w, img_h)

        corners_json = request.form.get("corners")
        if corners_json:
            pts = np.array(json.loads(corners_json), dtype=np.float64)
            if pts.shape == (4, 2):
                result = solve_card_pnp(pts, K)
                if result:
                    calib.add(*result)

        if calib.count < 3:
            return jsonify({"error": f"need ≥3 frames, have {calib.count}"}), 400

        if not calib.finalize(K):
            return jsonify({"error": "finalization failed"}), 500

        C = calib.cam_pos()
        return jsonify({
            "cam_pos": C.round(4).tolist(),
            "z_cam_m": round(float(C[2]), 3),
            "locked": True,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/detect_card", methods=["POST"])
def detect_card():
    """
    Detect credit card in frame for calibration assistance.
    Returns bbox and whether YOLO found a card.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        frame = cv2.imdecode(
            np.frombuffer(request.files["image"].read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        if frame is None:
            return jsonify({"error": "invalid image"}), 400

        img_h, img_w = frame.shape[:2]
        motion = compute_motion(frame)

        results = yolo(frame, verbose=False)[0]
        cards = []
        for box, cls, conf in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
            results.boxes.conf.cpu().numpy()
        ):
            if conf < 0.15:
                continue
            label = yolo.names.get(int(cls), "")
            if "card" in label.lower():
                bbox = list(map(int, box))
                pw = bbox[2] - bbox[0]
                ph = bbox[3] - bbox[1]
                ok, reason = check_card_aspect(pw, ph, img_w)
                cards.append({
                    "bbox": bbox,
                    "confidence": float(conf),
                    "aspect_ok": ok,
                    "aspect_reason": reason,
                    "pixel_w": pw,
                    "pixel_h": ph,
                })

        return jsonify({
            "cards": cards,
            "motion": motion,
            "calibrated": calib.locked,
            "cal_frames": calib.count,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/measure_face", methods=["POST"])
def measure_face_route():
    """
    Measure one face of the box from 3 user-tapped corner pixels.

    Body (JSON or form):
      corners: { "bl": [u,v], "br": [u,v], "tl": [u,v] }
      fx, img_w, img_h
      face: "front" | "side"  (label only, for bookkeeping)

    Returns width_m, height_m for this face.
    """
    try:
        if not calib.locked:
            return jsonify({"error": "not calibrated"}), 400

        data = request.get_json(silent=True) or {}
        form = request.form

        def get(k, default=None):
            return data.get(k) or form.get(k) or default

        corners = data.get("corners") or json.loads(form.get("corners", "{}"))
        img_w = int(get("img_w", 1280))
        img_h = int(get("img_h", 720))
        K = parse_K(form if form else {}, img_w, img_h)

        # Override K from JSON body if provided
        if "fx" in data:
            fx = sanitize_fx(float(data["fx"]), img_w)
            K = build_K(fx, img_w / 2.0, img_h / 2.0)

        if not all(k in corners for k in ("bl", "br", "tl")):
            return jsonify({"error": "corners must have bl, br, tl"}), 400

        result = measure_face(corners, calib.R, calib.t, K)
        if "error" in result:
            return jsonify(result), 422

        result["face"] = get("face", "unknown")
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/compute_volume", methods=["POST"])
def compute_volume():
    """
    Fuse two face measurements into final L×W×H.

    Body (JSON):
      face1: { width_m, height_m, residual_mm, ... }   (front face)
      face2: { width_m, height_m, residual_mm, ... }   (side face)
    """
    try:
        data = request.get_json(silent=True) or {}
        face1 = data.get("face1")
        face2 = data.get("face2")

        if not face1 or not face2:
            return jsonify({"error": "need face1 and face2"}), 400

        result = fuse_two_faces(face1, face2)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/calibration_status", methods=["GET"])
def calibration_status():
    resp = {
        "locked": calib.locked,
        "count": calib.count,
        "ready": calib.ready,
    }
    if calib.locked:
        resp["cam_pos"] = calib.cam_pos().round(4).tolist()
        resp["z_cam_m"] = round(float(calib.cam_pos()[2]), 3)
    return jsonify(resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
