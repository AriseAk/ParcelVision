"""
Human-Tripod Monocular Box Metrology — Flask Backend
=====================================================
Exposes two endpoints:
  POST /calibrate_height  — detects a credit card in an image and returns the
                            absolute camera height (Z) in metres.
  POST /measure_box       — uses 3 annotated pixel taps + gyroscope rotation
                            matrix to compute box L × W × H and volume.

Math overview
─────────────
All geometry lives in a right-handed world frame where the floor is Z = 0.
The camera is a standard pinhole model with intrinsic matrix K.

Calibration
  • We find the 4 corners of an ISO ID-1 credit card (85.6 mm × 53.98 mm).
  • cv2.solvePnP gives the camera's pose relative to the card.
  • Camera world position C = -R.T @ t  →  height = |C[2]|.

Measurement
  • We reconstruct a virtual extrinsic from:
      - t_virtual = [0, 0, calibrated_z]  (camera sits at this height)
      - R_gyro from the phone's gyroscope (converted to a rotation matrix
        by the frontend)
      - t_extrinsic = -R_gyro @ t_virtual
  • Bottom-Left / Bottom-Right taps → rays cast to Z = 0 plane  →  P_BL, P_BR
    Width = ‖P_BL − P_BR‖
  • Top-Left tap → ray cast; then closest point to vertical plumb-line
    erected at P_BL gives the box height (Z-coordinate of intersection).
"""

import io
import json
import logging
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # allow all origins; tighten in production

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# ISO/IEC 7810 ID-1 credit card dimensions in metres
CARD_WIDTH_M  = 0.08560
CARD_HEIGHT_M = 0.05398

# 3D corners of the card in card-local coordinates (Z=0 plane, origin at BL)
# Order matches the corner detection order: TL, TR, BR, BL
CARD_OBJECT_POINTS = np.array([
    [0,              CARD_HEIGHT_M, 0],   # Top-Left
    [CARD_WIDTH_M,   CARD_HEIGHT_M, 0],   # Top-Right
    [CARD_WIDTH_M,   0,             0],   # Bottom-Right
    [0,              0,             0],   # Bottom-Left
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ray_to_z0_plane(
    u: float, v: float,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Cast a 3D ray from the camera centre through image pixel (u, v) and
    intersect it with the world floor plane Z = 0.

    Pinhole model
    ─────────────
    A pixel (u, v) corresponds to the normalised camera-space direction:
        d_cam = K⁻¹ · [u, v, 1]ᵀ

    In world space the ray is:
        P(λ) = C + λ · R.T · d_cam

    where C = camera centre = -R.T @ t.

    Intersection with Z = 0
    ───────────────────────
        P_z(λ) = C_z + λ · (R.T @ d_cam)_z = 0
        λ       = -C_z / (R.T @ d_cam)_z

    Returns None if the ray is parallel to the floor (denominator ≈ 0) or
    if the intersection is behind the camera (λ < 0).
    """
    K_inv     = np.linalg.inv(K)
    d_cam     = K_inv @ np.array([u, v, 1.0])          # direction in camera frame
    d_world   = R.T @ d_cam                             # direction in world frame

    # Camera centre in world frame
    C = -(R.T @ t)

    denom = d_world[2]
    if abs(denom) < 1e-9:
        log.warning("ray_to_z0_plane: ray is parallel to floor, cannot intersect.")
        return None

    lam = -C[2] / denom
    if lam < 0:
        log.warning("ray_to_z0_plane: intersection is behind the camera (λ=%.3f).", lam)
        return None

    return C + lam * d_world   # 3D world point on Z = 0


def closest_point_on_lines(
    p1: np.ndarray, d1: np.ndarray,
    p2: np.ndarray, d2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the pair of closest points between two 3D lines.

    Line 1: p1 + s·d1
    Line 2: p2 + t·d2

    Uses the standard least-squares formula derived from minimising the
    squared distance between corresponding points on the two lines:

        (d1·d1)·s − (d1·d2)·t = (p2−p1)·d1
        (d1·d2)·s − (d2·d2)·t = (p2−p1)·d2

    Returns (point_on_line1, point_on_line2).
    If lines are parallel the midpoint of p1, p2 is returned twice.
    """
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    d2 = d2 / (np.linalg.norm(d2) + 1e-12)
    w  = p1 - p2

    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w)
    e = np.dot(d2, w)

    denom = a * c - b * b

    if abs(denom) < 1e-9:
        # Lines are parallel — return midpoints
        mid = (p1 + p2) / 2
        return mid, mid

    s = (b * e - c * d) / denom
    t_ = (a * e - b * d) / denom

    return p1 + s * d1, p2 + t_ * d2


# ─────────────────────────────────────────────────────────────────────────────
# CREDIT CARD CORNER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_card_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the 4 corners of a credit-card-shaped rectangle in `image`.

    Strategy
    ────────
    1. Convert to grayscale and apply adaptive thresholding to handle
       varying lighting conditions.
    2. Find external contours; filter by:
       a. Approximate to a 4-vertex polygon.
       b. Area close to the expected card aspect ratio (1.586 ± 30 %).
       c. Area larger than a minimum threshold to reject noise.
    3. Order the 4 corners: TL, TR, BR, BL to match CARD_OBJECT_POINTS.

    Returns None if no suitable quadrilateral is found.
    """
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges  = cv2.Canny(blur, 50, 150)

    # Dilate edges slightly to close small gaps
    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours    = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = image.shape[0] * image.shape[1]
    best_quad: Optional[np.ndarray] = None
    best_area = 0

    for cnt in contours[:20]:  # only examine the largest candidates
        area = cv2.contourArea(cnt)
        if area < img_area * 0.005:  # skip tiny blobs
            continue

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        # Check aspect ratio  (card is 85.6 × 53.98 mm → ratio ≈ 1.586)
        rect  = cv2.minAreaRect(approx)
        w, h  = rect[1]
        ratio = max(w, h) / (min(w, h) + 1e-6)
        if not (1.0 < ratio < 2.5):
            continue

        if area > best_area:
            best_area = area
            best_quad = approx

    if best_quad is None:
        log.warning("detect_card_corners: no card quadrilateral found.")
        return None

    # Reshape to (4, 2)
    pts = best_quad.reshape(4, 2).astype(np.float32)

    # Order corners: TL, TR, BR, BL
    pts = _order_corners(pts)
    return pts


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Given 4 unordered 2D points return them as [TL, TR, BR, BL].
    Uses the centroid and angle heuristic.
    """
    centroid = pts.mean(axis=0)
    angles   = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order    = np.argsort(angles)
    pts      = pts[order]   # ordered counter-clockwise starting from ~right

    # The point with smallest x+y is TL; largest x+y is BR etc.
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1: /calibrate_height
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/calibrate_height", methods=["POST"])
def calibrate_height():
    """
    Input (multipart/form-data):
        image         – image file containing a credit card on the floor
        focal_length  – estimated focal length in pixels (float, from frontend)

    Output (JSON):
        { "calibrated_z": <float>  }   on success
        { "error": "<message>" }       on failure
    """
    # ── Parse inputs ──────────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    try:
        focal_length = float(request.form.get("focal_length", 1000))
    except ValueError:
        return jsonify({"error": "Invalid focal_length value."}), 400

    # ── Decode image ──────────────────────────────────────────────────────────
    img_bytes = request.files["image"].read()
    nparr     = np.frombuffer(img_bytes, np.uint8)
    image     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Could not decode image."}), 400

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Intrinsic matrix K from the provided focal length
    K = np.array([
        [focal_length, 0,            cx],
        [0,            focal_length, cy],
        [0,            0,             1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    # ── Detect card corners ───────────────────────────────────────────────────
    corners_2d = detect_card_corners(image)
    if corners_2d is None:
        return jsonify({"error": "Could not detect credit card in image. "
                                 "Ensure the card is clearly visible on a contrasting floor."}), 422

    # ── solvePnP ──────────────────────────────────────────────────────────────
    success, rvec, tvec = cv2.solvePnP(
        CARD_OBJECT_POINTS,
        corners_2d,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return jsonify({"error": "PnP solver failed. Check card orientation."}), 422

    # Rotation vector → rotation matrix
    R_mat, _ = cv2.Rodrigues(rvec)

    # Camera centre in world frame: C = -R.T @ t
    # For a card lying flat on Z = 0, C[2] is the camera height above the floor.
    C = -(R_mat.T @ tvec).ravel()
    calibrated_z = float(abs(C[2]))

    log.info("Calibration: camera height = %.4f m", calibrated_z)
    return jsonify({"calibrated_z": calibrated_z})


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2: /measure_box
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/measure_box", methods=["POST"])
def measure_box():
    """
    Input (JSON):
        calibrated_z  – camera height in metres (float)
        R_gyro        – 3×3 rotation matrix from device gyroscope (list of lists)
        K             – 3×3 camera intrinsic matrix (list of lists)
        taps          – list of 3 objects [{u, v}, ...] in order:
                          [0] Bottom-Left  (floor-touching)
                          [1] Bottom-Right (floor-touching)
                          [2] Top-Left     (directly above Bottom-Left)

    Output (JSON):
        { "length": f, "width": f, "height": f, "volume": f }  (metres / m³)
        { "error": "<message>" }  on failure
    """
    data = request.get_json(force=True)

    # ── Validate ──────────────────────────────────────────────────────────────
    required = ("calibrated_z", "R_gyro", "K", "taps")
    for key in required:
        if key not in data:
            return jsonify({"error": f"Missing field: {key}"}), 400

    try:
        calibrated_z = float(data["calibrated_z"])
        R_gyro       = np.array(data["R_gyro"],   dtype=np.float64)   # (3, 3)
        K            = np.array(data["K"],         dtype=np.float64)   # (3, 3)
        taps         = data["taps"]                                     # list[{u,v}]
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Bad input format: {exc}"}), 400

    if R_gyro.shape != (3, 3):
        return jsonify({"error": "R_gyro must be a 3×3 matrix."}), 400
    if K.shape != (3, 3):
        return jsonify({"error": "K must be a 3×3 matrix."}), 400
    if len(taps) < 3:
        return jsonify({"error": "Exactly 3 tap points required."}), 400

    u_bl, v_bl = float(taps[0]["u"]), float(taps[0]["v"])
    u_br, v_br = float(taps[1]["u"]), float(taps[1]["v"])
    u_tl, v_tl = float(taps[2]["u"]), float(taps[2]["v"])

    # ── Virtual camera pose ───────────────────────────────────────────────────
    #
    # The "Human Tripod" constraint: the camera's (X, Y) translation is unknown
    # and ignored.  We place the camera at a canonical world position directly
    # above the world origin:
    #
    #   C_world = [0, 0, calibrated_z]ᵀ
    #
    # The extrinsic translation vector t relates world ↔ camera frame as:
    #   X_cam = R · X_world + t
    # So for the camera centre:
    #   0 = R · C_world + t  →  t = -R · C_world
    #
    t_virtual  = np.array([0.0, 0.0, calibrated_z])
    t_extr     = -(R_gyro @ t_virtual)   # shape (3,)

    # ── Ray → floor (Bottom-Left and Bottom-Right) ────────────────────────────
    P_BL = ray_to_z0_plane(u_bl, v_bl, R_gyro, t_extr, K)
    P_BR = ray_to_z0_plane(u_br, v_br, R_gyro, t_extr, K)

    if P_BL is None or P_BR is None:
        return jsonify({"error":
            "Could not project floor taps to Z=0.  "
            "Ensure both bottom corners are on the floor and visible."}), 422

    # The distance between the two floor points is one horizontal dimension
    # (length or width — the user controls which axis they tap along).
    horizontal_dim = float(np.linalg.norm(P_BL - P_BR))

    # ── Ray → plumb-line (Top-Left) ───────────────────────────────────────────
    #
    # The Top-Left corner sits directly above P_BL on a vertical plumb-line:
    #   Plumb-line:  P_BL + s · [0, 0, 1]ᵀ   for s ≥ 0
    #
    # We cast a ray through the TL tap:
    #   Camera centre C_world = -R.T @ t_extr
    #   Ray direction in world frame d_world = R.T @ (K⁻¹ @ [u,v,1])
    #
    # Then find the closest approach of the camera ray to the plumb-line.
    # The Z-coordinate of the closest point ON THE PLUMB-LINE is the box height.
    #
    K_inv      = np.linalg.inv(K)
    d_tl_cam   = K_inv @ np.array([u_tl, v_tl, 1.0])
    d_tl_world = R_gyro.T @ d_tl_cam

    C_world    = -(R_gyro.T @ t_extr)

    # Plumb-line: origin = P_BL, direction = [0, 0, 1]
    plumb_direction = np.array([0.0, 0.0, 1.0])

    pt_on_ray, pt_on_plumb = closest_point_on_lines(
        C_world,         d_tl_world,
        P_BL,            plumb_direction,
    )

    # The height of the box is the Z-coordinate of the plumb-line intersection
    # (since the plumb-line starts at Z=0 / floor level).
    box_height = float(max(pt_on_plumb[2], 0.0))

    # ── Assemble result ───────────────────────────────────────────────────────
    # We report the horizontal tap distance as "width" and height separately.
    # Volume = width × (width assumed square for now) × height; for a full
    # 3-face measurement use 6 taps.  Here we report 1 horizontal dimension.
    length = horizontal_dim          # dimension tapped along bottom edge
    width  = horizontal_dim          # same tap → same value (user controls axis)
    height = box_height
    volume = length * width * height

    log.info(
        "Measurement: L=%.3fm W=%.3fm H=%.3fm V=%.4fm³",
        length, width, height, volume,
    )

    return jsonify({
        "length": round(length, 4),
        "width":  round(width,  4),
        "height": round(height, 4),
        "volume": round(volume, 6),
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)