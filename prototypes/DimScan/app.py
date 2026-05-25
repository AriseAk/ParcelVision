from __future__ import annotations

import json
import os
import traceback

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CARD_W = 0.08560   # ISO/IEC 7810 ID-1, metres
CARD_H = 0.05398
CARD_ASPECT = CARD_W / CARD_H   # ≈ 1.586

# Object points: TL, TR, BR, BL  (matches order_corners output)
CARD_OBJ_PTS = np.array([
    [0.0,    0.0,    0.0],
    [CARD_W, 0.0,    0.0],
    [CARD_W, CARD_H, 0.0],
    [0.0,    CARD_H, 0.0],
], dtype=np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# App + models
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

_device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading SAM...")
_sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
_sam.to(_device)
sam = SamPredictor(_sam)

print("Loading YOLO World (optional, for box detection)...")
try:
    yolo = YOLO("yolov8s-world.pt")
    yolo.to(_device)
    yolo.set_classes(["box", "cardboard box", "carton", "parcel", "package", "shipping box"])
    YOLO_AVAILABLE = True
except Exception as e:
    print(f"YOLO unavailable: {e}")
    YOLO_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Camera intrinsics
# ─────────────────────────────────────────────────────────────────────────────

def build_K(fx: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)


def sanitize_fx(fx: float, img_w: int) -> float:
    lo, hi = img_w * 0.4, img_w * 1.5
    if not (lo <= fx <= hi):
        fallback = (img_w / 2.0) / np.tan(np.radians(36.5))  # 73 deg HFOV
        print(f"fx={fx:.1f} out of range, using fallback {fallback:.1f}")
        return fallback
    return fx


def parse_K(data: dict, img_w: int, img_h: int) -> np.ndarray:
    fx = sanitize_fx(float(data.get("fx") or 0), img_w)
    cx = float(data.get("cx") or img_w / 2.0)
    cy = float(data.get("cy") or img_h / 2.0)
    return build_K(fx, cx, cy)

# ─────────────────────────────────────────────────────────────────────────────
# Calibration State
# ─────────────────────────────────────────────────────────────────────────────

class CalibState:
    def __init__(self):
        self.rvecs:   list[np.ndarray] = []
        self.tvecs:   list[np.ndarray] = []
        self.corners: list[np.ndarray] = []   # raw image corners per frame
        self.R: np.ndarray | None = None
        self.t: np.ndarray | None = None
        self.K: np.ndarray | None = None

    def reset(self): self.__init__()

    @property
    def count(self)  -> int:  return len(self.rvecs)
    @property
    def ready(self)  -> bool: return self.count >= 5
    @property
    def locked(self) -> bool: return self.R is not None

    def add(self, rvec, tvec, img_pts: np.ndarray):
        self.rvecs.append(rvec.copy())
        self.tvecs.append(tvec.copy())
        self.corners.append(img_pts.copy())

    # ── Self-calibrate fx from the card's known geometry ─────────────────────
    # The card is a rigid rectangle of known size. For a given fx, PnP returns
    # a camera height. The CORRECT fx is the one that gives the most consistent
    # (lowest variance) height estimate across all frames.
    # This removes any dependence on the browser's HFOV guess.
    def _self_calibrate_fx(self, cx: float, cy: float) -> float:
        best_fx   = None
        best_var  = float("inf")

        # Search from 20° to 110° HFOV — covers ultra-wide through telephoto
        img_w_approx = cx * 2
        hfovs = np.linspace(20, 110, 180)   # 0.5° steps
        for hfov in hfovs:
            fx_cand = (img_w_approx / 2.0) / np.tan(np.radians(hfov / 2.0))
            K_cand  = build_K(fx_cand, cx, cy)
            heights = []
            for img_pts in self.corners:
                result = solve_card_pnp(img_pts, K_cand)
                if result is None:
                    continue
                rvec, tvec = result
                R_c, _ = cv2.Rodrigues(rvec)
                cam_z  = abs(float((-R_c.T @ tvec).ravel()[2]))
                if 0.05 < cam_z < 5.0:
                    heights.append(cam_z)
            if len(heights) < max(2, self.count // 2):
                continue
            var = float(np.var(heights))
            if var < best_var:
                best_var  = var
                best_fx   = fx_cand
                best_hfov = hfov
                best_h    = float(np.mean(heights))

        if best_fx is not None:
            print(f"  Self-cal fx={best_fx:.0f} (HFOV={best_hfov:.1f}°) "
                  f"cam_h={best_h*100:.1f}cm  var={best_var*1e4:.2f}×10⁻⁴")
            return best_fx

        # Fallback: use what the browser reported
        print("  Self-cal failed — keeping browser fx")
        return cx * 2.0 / np.tan(np.radians(65 / 2.0))

    def finalize(self, K: np.ndarray) -> bool:
        if self.count < 1:
            return False

        # ── Step 1: self-calibrate fx from card geometry ──────────────────────
        cx, cy    = float(K[0, 2]), float(K[1, 2])
        true_fx   = self._self_calibrate_fx(cx, cy)
        K_cal     = build_K(true_fx, cx, cy)

        # ── Step 2: re-solve PnP for all frames with calibrated K ─────────────
        rvecs_cal, tvecs_cal = [], []
        for img_pts in self.corners:
            result = solve_card_pnp(img_pts, K_cal)
            if result is not None:
                rvecs_cal.append(result[0])
                tvecs_cal.append(result[1])

        if not rvecs_cal:
            return False

        tvecs_arr = np.array([t.ravel() for t in tvecs_cal])
        med_t     = np.median(tvecs_arr, axis=0)
        dists     = np.linalg.norm(tvecs_arr - med_t, axis=1)
        n_keep    = max(1, int(len(rvecs_cal) * 0.75))
        keep_idx  = np.argsort(dists)[:n_keep]

        R_sum = np.zeros((3, 3))
        t_sum = np.zeros(3)
        for i in keep_idx:
            Ri, _ = cv2.Rodrigues(rvecs_cal[i])
            R_sum += Ri
            t_sum += tvecs_arr[i]

        U, _, Vt = np.linalg.svd(R_sum)
        R_avg = U @ Vt
        if np.linalg.det(R_avg) < 0:
            U[:, -1] *= -1
            R_avg = U @ Vt

        self.R = R_avg
        self.t = (t_sum / n_keep).reshape(3, 1)
        self.K = K_cal          # store calibrated K (with corrected fx)
        self.rvecs.clear()
        self.tvecs.clear()
        self.corners.clear()

        C = self.cam_pos()
        print(f"Calib locked: cam_height={self.cam_height()*100:.1f}cm  pos={C.round(3)}")
        return True

    def cam_pos(self)    -> np.ndarray: return (-self.R.T @ self.t).ravel()
    def cam_height(self) -> float:      return abs(float(self.cam_pos()[2]))


calib = CalibState()

# ─────────────────────────────────────────────────────────────────────────────
# Corner ordering + SAM refine
# ─────────────────────────────────────────────────────────────────────────────

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)],   # BL
    ], dtype=np.float64)


def _aspect_ok(ordered: np.ndarray) -> bool:
    """
    Check that the ordered quad has a width/height ratio plausible for a
    credit card (1.586 ± ~40% tolerance to handle perspective distortion).
    """
    w = float(np.linalg.norm(ordered[1] - ordered[0]))   # TR - TL  (top edge)
    h = float(np.linalg.norm(ordered[3] - ordered[0]))   # BL - TL  (left edge)
    if h < 1e-3:
        return False
    ratio = w / h
    # Allow 0.9 – 2.8 to cover heavy perspective angles
    ok = 0.9 < ratio < 2.8
    if not ok:
        print(f"  SAM aspect ratio {ratio:.2f} implausible (expected ~{CARD_ASPECT:.2f}), falling back")
    return ok


def sam_refine_corners(frame: np.ndarray, tapped_pts: np.ndarray) -> np.ndarray:
    """
    Given 4 roughly tapped card corners, use SAM to find precise edges.
    Returns ordered (4,2) TL,TR,BR,BL. Falls back to ordered tapped_pts if SAM fails.
    """
    fallback = order_corners(tapped_pts)
    try:
        x1 = int(max(0, tapped_pts[:, 0].min() - 30))
        y1 = int(max(0, tapped_pts[:, 1].min() - 30))
        x2 = int(min(frame.shape[1], tapped_pts[:, 0].max() + 30))
        y2 = int(min(frame.shape[0], tapped_pts[:, 1].max() + 30))

        sam.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Use both bbox and point prompts for better accuracy
        masks, scores, _ = sam.predict(
            point_coords=tapped_pts.astype(np.float32),
            point_labels=np.ones(len(tapped_pts), dtype=np.int32),
            box=np.array([x1, y1, x2, y2]),
            multimask_output=True
        )
        best_mask = masks[np.argmax(scores)]
        mask = (best_mask > 0.5).astype(np.uint8)

        if mask.sum() < 200:
            print("  SAM mask too small, using tapped corners directly")
            return fallback

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return fallback

        cnt    = max(contours, key=cv2.contourArea)
        eps    = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)

        if len(approx) == 4:
            refined = approx.reshape(-1, 2).astype(np.float64)
        else:
            refined = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float64)

        ordered = order_corners(refined)
        print(f"  SAM refined: area={mask.sum()} pts={ordered.round(1).tolist()}")

        # Reject SAM result if the quad looks geometrically wrong
        if not _aspect_ok(ordered):
            return fallback

        return ordered

    except Exception as e:
        print(f"  SAM refine error: {e}")
        return fallback

# ─────────────────────────────────────────────────────────────────────────────
# PnP
# ─────────────────────────────────────────────────────────────────────────────

def solve_card_pnp(img_pts: np.ndarray, K: np.ndarray):
    """
    Solve PnP for the credit card and return (rvec, tvec) or None.

    FIX 1: Use the default iterative solver instead of SOLVEPNP_IPPE_SQUARE.
            IPPE_SQUARE assumes a *square* object; our card (85.6×54mm) is
            rectangular, so that flag was semantically wrong and produced
            spurious negative-tz solutions.

    FIX 2: Extract tz with .ravel()[2] to avoid the NumPy DeprecationWarning
            (and future error) that occurs when converting a 1-element array
            to a scalar via float(tvec[2]).
    """
    if img_pts.shape != (4, 2):
        return None

    # Iterative PnP – correct for non-square rectangles
    ok, rvec, tvec = cv2.solvePnP(
        CARD_OBJ_PTS,
        img_pts.reshape(4, 1, 2),   # explicit shape avoids any ambiguity
        K,
        None,                        # no distortion coefficients
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None

    # FIX 2: safe scalar extraction – no DeprecationWarning
    tz = float(tvec.ravel()[2])

    if tz <= 0:
        print(f"  PnP rejected: tz={tz:.3f}m (negative — corner order likely flipped)")
        return None
    if not (0.05 < tz < 3.0):
        print(f"  PnP rejected: tz={tz:.3f}m out of range [0.05, 3.0]")
        return None

    # Card must be roughly horizontal (lying on floor).
    # R_card[:,2] is the card normal in camera frame.
    # For a flat-floor card viewed from above the dot with camera Z should be
    # substantial (≥ 0.15 is lenient enough for ~80° tilt).
    R_card, _ = cv2.Rodrigues(rvec)
    if abs(R_card[2, 2]) < 0.15:
        print(f"  PnP rejected: card not horizontal R[2,2]={R_card[2,2]:.2f}")
        return None

    return rvec, tvec


def reproj_error(rvec, tvec, K, img_pts) -> float:
    proj, _ = cv2.projectPoints(CARD_OBJ_PTS, rvec, tvec, K, None)
    return float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1)))

# ─────────────────────────────────────────────────────────────────────────────
# Core geometry
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_ray_world(u, v, R, K):
    ray_cam   = np.linalg.inv(K) @ np.array([u, v, 1.0])
    ray_world = R.T @ ray_cam
    return ray_world / np.linalg.norm(ray_world)


def ray_plane_intersect(origin, direction, normal=None, d=0.0):
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    denom = float(np.dot(normal, direction))
    if abs(denom) < 1e-8:
        return None
    t = (d - float(np.dot(normal, origin))) / denom
    return None if t < 0 else origin + t * direction


def ray_line_closest(ray_o, ray_d, line_o, line_d):
    d1   = ray_d  / np.linalg.norm(ray_d)
    d2   = line_d / np.linalg.norm(line_d)
    b    = line_o - ray_o
    d1d2 = float(np.dot(d1, d2))
    denom = 1.0 - d1d2 * d1d2
    if abs(denom) < 1e-8:
        s = float(np.dot(b, d1))
        t = float(np.dot(b, d2))
    else:
        bd1 = float(np.dot(b, d1))
        bd2 = float(np.dot(b, d2))
        s   = (bd1 - d1d2 * bd2) / denom
        t   = (d1d2 * bd1 - bd2) / denom
    t  = max(0.0, t)
    P1 = ray_o  + s * d1
    P2 = line_o + t * d2
    return P1, P2, float(np.linalg.norm(P1 - P2)) * 1000


def measure_face(corners_2d, R, t, K):
    """
    Measure one face of the box.

    corners_2d: dict with keys "bl", "br", "tl" — each a [x, y] pixel coord.
      bl = bottom-left  (floor, left vertical edge)
      br = bottom-right (floor, right vertical edge)
      tl = top-left     (top of left vertical edge)

    Width  = horizontal distance on the floor between bl and br.
    Height = vertical distance (world Z) from the floor up to the top-left corner.
    """
    cam_pos = (-R.T @ t).ravel()
    bl = np.array(corners_2d["bl"], dtype=np.float64)
    br = np.array(corners_2d["br"], dtype=np.float64)
    tl = np.array(corners_2d["tl"], dtype=np.float64)

    # Project bl and br onto the floor plane (Z=0) via ray casting
    P_bl = ray_plane_intersect(cam_pos, pixel_to_ray_world(bl[0], bl[1], R, K))
    P_br = ray_plane_intersect(cam_pos, pixel_to_ray_world(br[0], br[1], R, K))
    if P_bl is None or P_br is None:
        return {"error": "floor intersection failed — recalibrate"}

    # Width: straight-line distance on the floor
    width = float(np.linalg.norm(P_br - P_bl))

    # Height: find the world point on the vertical line through P_bl that is
    # closest to the ray cast through the top-left pixel.
    _, P_top, resid = ray_line_closest(
        cam_pos, pixel_to_ray_world(tl[0], tl[1], R, K),
        P_bl, np.array([0., 0., 1.])   # vertical line upward from bl
    )
    height  = abs(float(P_top[2]))
    cam_h   = abs(float(cam_pos[2]))

    # Sanity checks
    if height < 0.005 or height > cam_h * 1.5:
        return {"error": f"height {height*100:.1f}cm implausible (cam={cam_h*100:.1f}cm)"}
    if width < 0.005 or width > 3.0:
        return {"error": f"width {width*100:.1f}cm implausible"}

    print(f"  Face: W={width*100:.1f}cm H={height*100:.1f}cm res={resid:.1f}mm cam_h={cam_h*100:.1f}cm")
    return {
        "width_m":      round(width,  4),
        "height_m":     round(height, 4),
        "residual_mm":  round(resid,  2),
        "cam_height_m": round(cam_h,  4),
    }


def fuse_two_faces(face1: dict, face2: dict) -> dict:
    """
    Combine measurements from two perpendicular faces to get L × W × H.

    face1 and face2 each provide:
      width_m    — the horizontal span of that face
      height_m   — the box height as seen from that face

    Since the two faces share the same height (it's the same box), we
    take an inverse-residual-weighted average of the two height readings
    for a more accurate estimate.

    The larger of the two widths becomes the box *length*, the smaller
    becomes the box *width*.

    Volume = length × width × height  (all in metres → converted to cm³
    in the frontend via × 1 000 000).
    """
    w1 = float(face1["width_m"])
    h1 = float(face1["height_m"])
    r1 = float(face1.get("residual_mm") or 10)

    w2 = float(face2["width_m"])
    h2 = float(face2["height_m"])
    r2 = float(face2.get("residual_mm") or 10)

    # Inverse-residual weights (lower residual → higher weight)
    wt1 = 1.0 / max(r1, 0.1)
    wt2 = 1.0 / max(r2, 0.1)
    h_fused = (wt1 * h1 + wt2 * h2) / (wt1 + wt2)

    length = max(w1, w2)
    width  = min(w1, w2)
    volume = length * width * h_fused   # m³

    result = {
        "length_m":  round(length,   4),
        "width_m":   round(width,    4),
        "height_m":  round(h_fused,  4),
        "volume_m3": round(volume,   6),
        "height_consistency_mm": round(abs(h1 - h2) * 1000, 1),
        "face1_residual_mm": round(r1, 2),
        "face2_residual_mm": round(r2, 2),
    }

    print(
        f"  Volume: L={length*100:.1f}cm W={width*100:.1f}cm "
        f"H={h_fused*100:.1f}cm  Vol={volume*1e6:.0f}cm³  "
        f"consistency={result['height_consistency_mm']}mm"
    )
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", "calibrated": calib.locked,
        "cal_frames": calib.count, "device": _device,
        "cam_height_cm": round(calib.cam_height() * 100, 1) if calib.locked else None,
    })


@app.route("/reset_calibration", methods=["POST"])
def reset_calibration():
    calib.reset()
    return jsonify({"status": "reset"})


@app.route("/calibrate_card", methods=["POST"])
def calibrate_card():
    """
    User manually taps 4 corners of the credit card.
    SAM refines the taps to precise corners, then PnP solves camera pose.

    Form fields:
      image    : JPEG frame (required for SAM refine)
      corners  : JSON [[x,y],[x,y],[x,y],[x,y]]  in video pixel coordinates
      fx,cx,cy,img_w,img_h
    """
    try:
        img_w = int(request.form.get("img_w", 1280))
        img_h = int(request.form.get("img_h", 720))
        K     = parse_K(request.form, img_w, img_h)

        corners_json = request.form.get("corners")
        if not corners_json:
            return jsonify({"error": "corners required"}), 400

        raw_pts = np.array(json.loads(corners_json), dtype=np.float64)
        if raw_pts.shape != (4, 2):
            return jsonify({"error": "need exactly 4 corners [[x,y]x4]"}), 400

        # SAM refine if image provided, otherwise fall back to tapped corners
        if "image" in request.files:
            frame = cv2.imdecode(
                np.frombuffer(request.files["image"].read(), np.uint8),
                cv2.IMREAD_COLOR
            )
            refined = sam_refine_corners(frame, raw_pts) if frame is not None else order_corners(raw_pts)
        else:
            refined = order_corners(raw_pts)

        result = solve_card_pnp(refined, K)
        if result is None:
            return jsonify({
                "status": "rejected",
                "reason": "PnP failed — ensure card is flat on floor and tap corners precisely",
                "count": calib.count,
            })

        err = reproj_error(*result, K, refined)
        if err > 12.0:
            return jsonify({
                "status": "rejected",
                "reason": f"Reprojection error {err:.1f}px (limit 12px) — tap card corners more precisely",
                "count": calib.count,
                "reproj_error": round(err, 2),
            })

        calib.add(*result, refined)
        # Safe scalar extraction (FIX 2)
        tz_log = float(result[1].ravel()[2])
        print(f"Cal frame #{calib.count}  reproj={err:.2f}px  tz={tz_log:.3f}m")

        resp = {
            "status": "accepted",
            "count":  calib.count,
            "ready":  calib.ready,
            "reproj_error": round(err, 2),
            "refined_corners": refined.round(1).tolist(),
        }

        if calib.ready:
            calib.finalize(K)
            resp["locked"]        = True
            resp["cam_height_cm"] = round(calib.cam_height() * 100, 1)

        return jsonify(resp)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/confirm_calibration", methods=["POST"])
def confirm_calibration():
    """Force finalize with frames collected so far (need >= 1)."""
    try:
        img_w = int(request.form.get("img_w", 1280))
        img_h = int(request.form.get("img_h", 720))
        K     = parse_K(request.form, img_w, img_h)

        if calib.count < 1:
            return jsonify({"error": "no calibration frames — tap card corners first"}), 400
        if not calib.finalize(K):
            return jsonify({"error": "finalization failed"}), 500

        return jsonify({
            "locked":        True,
            "cam_pos":       calib.cam_pos().round(4).tolist(),
            "cam_height_cm": round(calib.cam_height() * 100, 1),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/calibration_status", methods=["GET"])
def calibration_status():
    resp = {"locked": calib.locked, "count": calib.count, "ready": calib.ready}
    if calib.locked:
        resp["cam_height_cm"] = round(calib.cam_height() * 100, 1)
    return jsonify(resp)


@app.route("/measure_face", methods=["POST"])
def measure_face_route():
    try:
        if not calib.locked:
            return jsonify({"error": "not calibrated"}), 400
        data  = request.get_json(silent=True) or {}
        img_w = int(data.get("img_w", 1280))
        img_h = int(data.get("img_h", 720))
        K     = parse_K(data, img_w, img_h)
        corners = data.get("corners", {})
        if not all(k in corners for k in ("bl", "br", "tl")):
            return jsonify({"error": "need corners: bl, br, tl"}), 400
        result = measure_face(corners, calib.R, calib.t, K)
        if "error" in result:
            return jsonify(result), 422
        result["face"] = data.get("face", "unknown")
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/compute_volume", methods=["POST"])
def compute_volume():
    """
    Accepts JSON: { "face1": {...}, "face2": {...} }
    where each face object is the direct response from /measure_face.
    Returns fused L/W/H/Volume.
    """
    try:
        data = request.get_json(silent=True) or {}
        f1   = data.get("face1")
        f2   = data.get("face2")
        if not f1 or not f2:
            return jsonify({"error": "need face1 and face2"}), 400
        # Validate required keys
        for label, face in (("face1", f1), ("face2", f2)):
            for key in ("width_m", "height_m"):
                if key not in face:
                    return jsonify({"error": f"{label} missing key: {key}"}), 400
        return jsonify(fuse_two_faces(f1, f2))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def serve_frontend():
    return send_from_directory(BASE_DIR, "dimscan.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)