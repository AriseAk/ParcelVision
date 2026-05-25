"""
Human-Tripod Monocular Box Metrology — Flask Backend  v8
=========================================================

Key improvements over v7:
  - SAM card detection: multi-point prompting (centre + 4 quadrant points)
    so small/distant cards are segmented more reliably
  - Aspect ratio gate: softened to 1.2–2.1 with a scored "best fit" picker
    instead of hard reject, so cards at shallow angles still pass
  - VP focal length: explicit minimum convergence check — if the card is too
    flat (top/bottom edges nearly parallel) we report WHY VP failed so the
    frontend can show the right coaching message
  - Card size sanity check: if the card occupies < 0.3 % of the image we
    return a specific "card too small / too far" error
  - calculate_focal_length_from_card: now also tries the OTHER pair of edges
    (left/right) independently and picks the VP pair that gives the strongest
    convergence signal

Endpoints (unchanged):
  GET  /status
  POST /calibrate_height   (image + tap_x + tap_y + focal_length)
  POST /measure_box        (image + calibrated_z + R_gyro + K + taps)
  POST /debug_sam          (image + tap_x + tap_y)
"""

import base64
import json
import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CARD_WIDTH_M  = 0.08560
CARD_HEIGHT_M = 0.05398
CARD_RATIO    = CARD_WIDTH_M / CARD_HEIGHT_M   # ≈ 1.5860

# Wider tolerance — at shallow angles the apparent ratio shifts
CARD_RATIO_MIN = 1.20
CARD_RATIO_MAX = 2.10

CARD_OBJECT_POINTS = np.array([
    [0,             CARD_HEIGHT_M, 0],
    [CARD_WIDTH_M,  CARD_HEIGHT_M, 0],
    [CARD_WIDTH_M,  0,             0],
    [0,             0,             0],
], dtype=np.float64)

CAMERA_HEIGHT_MIN = 0.20
CAMERA_HEIGHT_MAX = 3.00

# Card must cover at least this fraction of image area to be usable
CARD_MIN_AREA_FRAC = 0.003   # 0.3 %


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _order_corners(pts: np.ndarray) -> np.ndarray:
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype=np.float32)


def _get_line_intersection(p1, p2, p3, p4) -> Optional[np.ndarray]:
    l1 = np.cross([p1[0], p1[1], 1.], [p2[0], p2[1], 1.])
    l2 = np.cross([p3[0], p3[1], 1.], [p4[0], p4[1], 1.])
    v  = np.cross(l1, l2)
    if abs(v[2]) < 1e-7:
        return None
    return np.array([v[0] / v[2], v[1] / v[2]])


def _vp_convergence_angle_deg(p1, p2, p3, p4) -> float:
    """
    Return the angle (degrees) between the two lines p1-p2 and p3-p4.
    0° = parallel (VP at infinity, no usable geometry).
    Larger = more perspective convergence = better VP signal.
    """
    d1 = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
    d2 = np.array([p4[0] - p3[0], p4[1] - p3[1]], dtype=float)
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = abs(np.dot(d1 / n1, d2 / n2))
    cos_a = min(1.0, cos_a)
    return float(np.degrees(np.arccos(cos_a)))


def calculate_focal_length_from_card(
    corners: np.ndarray, w: int, h: int
) -> Tuple[Optional[float], str]:
    """
    Compute focal length from card vanishing points.

    Tries both edge pairs independently:
      Pair A: top edge (TL→TR) + bottom edge (BL→BR) → VP_horizontal
      Pair B: left edge (TL→BL) + right edge (TR→BR) → VP_vertical
    Picks the pair with the strongest convergence signal.

    Returns (focal_px, reason_string).
    focal_px is None if VP geometry is not reliable.
    reason_string explains why if None.
    """
    TL, TR, BR, BL = corners.astype(float)
    cx, cy = w / 2., h / 2.

    MIN_CONVERGENCE_DEG = 2.0   # card must show at least 2° of perspective

    results = []

    # ── Pair A: horizontal VP (top & bottom edges) ──────────────────────────
    ang_a = _vp_convergence_angle_deg(TL, TR, BL, BR)
    log.info("VP pair A (horiz) convergence: %.2f°", ang_a)
    if ang_a >= MIN_CONVERGENCE_DEG:
        vp1 = _get_line_intersection(TL, TR, BL, BR)
        vp2 = _get_line_intersection(TL, BL, TR, BR)
        if vp1 is not None and vp2 is not None:
            v1 = np.array([vp1[0] - cx, vp1[1] - cy])
            v2 = np.array([vp2[0] - cx, vp2[1] - cy])
            dot = float(np.dot(v1, v2))
            if dot < 0:
                f = float(np.sqrt(-dot))
                results.append((f, ang_a, "pair_A"))

    # ── Pair B: vertical VP (left & right edges) ────────────────────────────
    ang_b = _vp_convergence_angle_deg(TL, BL, TR, BR)
    log.info("VP pair B (vert) convergence: %.2f°", ang_b)
    if ang_b >= MIN_CONVERGENCE_DEG:
        vp1b = _get_line_intersection(TL, BL, TR, BR)
        vp2b = _get_line_intersection(TL, TR, BL, BR)
        if vp1b is not None and vp2b is not None:
            v1b = np.array([vp1b[0] - cx, vp1b[1] - cy])
            v2b = np.array([vp2b[0] - cx, vp2b[1] - cy])
            dot_b = float(np.dot(v1b, v2b))
            if dot_b < 0:
                f_b = float(np.sqrt(-dot_b))
                results.append((f_b, ang_b, "pair_B"))

    if not results:
        max_ang = max(ang_a, ang_b)
        if max_ang < MIN_CONVERGENCE_DEG:
            return None, (
                f"Card edges are nearly parallel (max convergence {max_ang:.1f}° < "
                f"{MIN_CONVERGENCE_DEG}°). Move closer to the card or tilt phone "
                f"more steeply so the card looks like a clear trapezoid."
            )
        return None, "VP dot product ≥ 0 for both edge pairs — geometry degenerate."

    # Pick result with strongest convergence and valid focal range
    valid = [(f, ang, src) for f, ang, src in results if 100.0 < f < 4000.0]
    if not valid:
        fs = [f for f, _, _ in results]
        return None, f"VP focal lengths out of range: {fs} — try moving closer."

    # Among valid, prefer the one with more convergence
    best_f, best_ang, best_src = max(valid, key=lambda x: x[1])
    log.info("VP focal: f=%.1f px (convergence=%.2f°, %s)", best_f, best_ang, best_src)
    return best_f, f"ok ({best_src}, convergence={best_ang:.1f}°)"


def ray_to_plane_z(u, v, R, t, K, z_world=0.0):
    d_cam   = np.linalg.inv(K) @ np.array([u, v, 1.])
    d_world = R.T @ d_cam
    C       = -(R.T @ t)
    denom   = d_world[2]
    if abs(denom) < 1e-9:
        return None
    lam = (z_world - C[2]) / denom
    if lam < -0.5:
        return None
    return C + lam * d_world


def ray_to_z0_plane(u, v, R, t, K):
    return ray_to_plane_z(u, v, R, t, K, 0.0)


def closest_point_on_lines(p1, d1, p2, d2):
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    d2 = d2 / (np.linalg.norm(d2) + 1e-12)
    w  = p1 - p2
    a, b, c = np.dot(d1, d1), np.dot(d1, d2), np.dot(d2, d2)
    d, e    = np.dot(d1, w), np.dot(d2, w)
    denom   = a * c - b * b
    if abs(denom) < 1e-9:
        mid = (p1 + p2) / 2
        return mid, mid
    s  = (b * e - c * d) / denom
    t_ = (a * e - b * d) / denom
    return p1 + s * d1, p2 + t_ * d2


# ─────────────────────────────────────────────────────────────────────────────
# MASK / BBOX HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _corners_from_mask(mask_bool: np.ndarray) -> Optional[np.ndarray]:
    uint8 = mask_bool.astype(np.uint8) * 255
    conts, _ = cv2.findContours(uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not conts:
        return None
    cnt    = max(conts, key=cv2.contourArea)
    peri   = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        return _order_corners(approx.reshape(4, 2).astype(np.float32))
    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect).astype(np.float32)
    return _order_corners(box)


def _mask_aspect_ratio(corners: np.ndarray) -> float:
    rect = cv2.minAreaRect(corners.astype(np.float32))
    w, h = rect[1]
    if min(w, h) < 1:
        return 0.0
    return float(max(w, h) / min(w, h))


def _mask_area_px(corners: np.ndarray) -> float:
    rect = cv2.minAreaRect(corners.astype(np.float32))
    w, h = rect[1]
    return float(w * h)


def _expand_bbox(bbox, img_shape, pad_frac=0.15):
    x1, y1, x2, y2 = bbox
    pw = int((x2 - x1) * pad_frac)
    ph = int((y2 - y1) * pad_frac)
    H, W = img_shape[:2]
    return (max(0, x1 - pw), max(0, y1 - ph),
            min(W - 1, x2 + pw), min(H - 1, y2 + ph))


# ─────────────────────────────────────────────────────────────────────────────
# SAM LOADER
# ─────────────────────────────────────────────────────────────────────────────
_sam_predictor = None
_sam_auto_gen  = None
_sam_available = False
_sam_tried     = False


def _load_sam() -> bool:
    global _sam_predictor, _sam_auto_gen, _sam_available, _sam_tried
    if _sam_tried:
        return _sam_available
    _sam_tried = True

    checkpoint = os.environ.get("SAM_CHECKPOINT", "sam_vit_b_01ec64.pth")
    model_type = os.environ.get("SAM_MODEL_TYPE", "vit_b")
    if not os.path.exists(checkpoint):
        log.warning("SAM checkpoint '%s' not found.", checkpoint)
        return False
    try:
        import torch
        from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam    = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        _sam_predictor = SamPredictor(sam)
        _sam_auto_gen  = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=500,
        )
        _sam_available = True
        log.info("SAM ready on %s.", device)
    except Exception as exc:
        log.warning("SAM load failed: %s", exc)
        _sam_available = False
    return _sam_available


def _run_sam_multi_point(
    image_bgr: np.ndarray,
    cx: float, cy: float,
    extra_radius_frac: float = 0.30,
) -> Optional[np.ndarray]:
    """
    Run SAM with the tap centre + 4 surrounding points as foreground prompts.
    This dramatically improves small/distant card detection because the
    centre alone can land on a floor tile gap rather than the card.

    extra_radius_frac: fraction of the shorter image dimension to use as
    the radius for the surrounding points.
    """
    if not _load_sam() or _sam_predictor is None:
        return None

    h, w = image_bgr.shape[:2]
    r = min(h, w) * extra_radius_frac * 0.5   # half the card expected half-size

    # 5-point prompt: centre + cardinal offsets scaled to likely card size
    # We use a smaller radius (card is ~80px wide at 1.5m) to keep points on card
    card_r = min(h, w) * 0.04   # ~4% of shorter dim ≈ half card width at typical distance
    pts = np.array([
        [cx,           cy],
        [cx - card_r,  cy],
        [cx + card_r,  cy],
        [cx,           cy - card_r * 0.6],
        [cx,           cy + card_r * 0.6],
    ], dtype=np.float32)
    # Clamp to image
    pts[:, 0] = np.clip(pts[:, 0], 1, w - 2)
    pts[:, 1] = np.clip(pts[:, 1], 1, h - 2)
    labels = np.ones(len(pts), dtype=np.int32)

    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        _sam_predictor.set_image(rgb)
        masks, scores, _ = _sam_predictor.predict(
            point_coords=pts,
            point_labels=labels,
            multimask_output=True,
        )

        # Score each mask by how close its aspect ratio is to CARD_RATIO
        best_mask  = None
        best_score = -1.0

        for i, mask in enumerate(masks):
            corners = _corners_from_mask(mask.astype(bool))
            if corners is None:
                continue
            ar = _mask_aspect_ratio(corners)
            # Ratio score: 1.0 at perfect card ratio, falls off linearly
            ratio_score = max(0.0, 1.0 - abs(ar - CARD_RATIO) / CARD_RATIO)
            combined = float(scores[i]) * 0.4 + ratio_score * 0.6
            log.info("  SAM candidate %d: ar=%.3f sam_score=%.3f ratio_score=%.3f combined=%.3f",
                     i, ar, float(scores[i]), ratio_score, combined)
            if combined > best_score:
                best_score = combined
                best_mask  = mask.astype(bool)

        if best_mask is not None:
            log.info("SAM multi-point best mask score=%.3f", best_score)
            return best_mask

    except Exception as exc:
        log.warning("SAM multi-point error: %s", exc)

    return None


def _run_sam_auto(image_bgr: np.ndarray) -> list:
    if not _load_sam() or _sam_auto_gen is None:
        return []
    try:
        rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = _sam_auto_gen.generate(rgb)
        log.info("SAM auto: %d masks.", len(masks))
        return masks
    except Exception as exc:
        log.warning("SAM auto error: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# CARD DETECTION — tap-prompted SAM with multi-point + scored selection
# ─────────────────────────────────────────────────────────────────────────────

def detect_card_from_tap(
    image_bgr: np.ndarray,
    tap_x: float,
    tap_y: float,
) -> Tuple[Optional[np.ndarray], float, bool, str]:
    """
    Segment the card the user tapped using SAM multi-point prompt.

    Returns (corners_2d, aspect_ratio, ok: bool, message: str)
    """
    img_h, img_w = image_bgr.shape[:2]
    img_area = img_h * img_w

    if not _load_sam():
        return None, 0.0, False, "SAM not available — check server setup."

    mask = _run_sam_multi_point(image_bgr, tap_x, tap_y)
    if mask is None:
        return None, 0.0, False, "SAM segmentation failed."

    corners = _corners_from_mask(mask)
    if corners is None:
        return None, 0.0, False, "Could not extract corners from SAM mask."

    ar        = _mask_aspect_ratio(corners)
    mask_area = _mask_area_px(corners)

    log.info("Card tap-SAM: aspect ratio=%.3f  mask_area=%.0f px² (%.4f of image)",
             ar, mask_area, mask_area / img_area)

    # ── Size check ─────────────────────────────────────────────────────────
    if mask_area / img_area < CARD_MIN_AREA_FRAC:
        return corners, ar, False, (
            f"Card is too small in the image ({mask_area / img_area * 100:.2f}% of frame). "
            f"Move closer — the card should fill at least 3–5% of the screen."
        )

    # ── Aspect ratio check ─────────────────────────────────────────────────
    if not (CARD_RATIO_MIN <= ar <= CARD_RATIO_MAX):
        hint = ""
        if ar < CARD_RATIO_MIN:
            hint = " The detected region looks too square — try tapping closer to the centre of the card."
        elif ar > CARD_RATIO_MAX:
            hint = " The detected region looks too elongated — SAM may have grabbed floor tiles. Move card to a plainer surface."
        return corners, ar, False, (
            f"Detected region ratio={ar:.2f} (expected {CARD_RATIO_MIN:.2f}–{CARD_RATIO_MAX:.2f}).{hint}"
        )

    return corners, ar, True, f"Card detected (ratio={ar:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# YOLO-WORLD — box detection only (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
_yolo_model     = None
_yolo_available = False
_yolo_tried     = False


def _load_yolo() -> bool:
    global _yolo_model, _yolo_available, _yolo_tried
    if _yolo_tried:
        return _yolo_available
    _yolo_tried = True

    weights_path = os.environ.get("YOLO_WEIGHTS", "yolov8s-world.pt")
    try:
        from ultralytics import YOLO
        if not os.path.exists(weights_path):
            log.warning("YOLO-World weights '%s' not found.", weights_path)
            return False
        log.info("Loading YOLO-World for box detection …")
        _yolo_model = YOLO(weights_path)
        _yolo_model.set_classes(["cardboard box", "shipping box", "package", "box"])
        _yolo_available = True
        log.info("YOLO-World ready.")
    except Exception as exc:
        log.warning("YOLO-World load failed: %s", exc)
        _yolo_available = False
    return _yolo_available


def _yolo_detect_box_bbox(image_bgr: np.ndarray):
    if not _load_yolo():
        return None
    try:
        results = _yolo_model.predict(image_bgr, conf=0.20, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        best  = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes[best].astype(int)
        log.info("YOLO-World: box bbox (%d,%d,%d,%d) conf=%.2f",
                 x1, y1, x2, y2, float(confs[best]))
        return (x1, y1, x2, y2)
    except Exception as exc:
        log.warning("YOLO-World inference error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE DECODE / PnP
# ─────────────────────────────────────────────────────────────────────────────

def _decode_image() -> Optional[np.ndarray]:
    if "image" not in request.files:
        return None
    raw = request.files["image"].read()
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def _run_pnp(corners_2d, K):
    ok, rvec, tvec = cv2.solvePnP(
        CARD_OBJECT_POINTS, corners_2d, K, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None, None
    R_mat, _ = cv2.Rodrigues(rvec)
    C = -(R_mat.T @ tvec).ravel()
    return float(abs(C[2])), R_mat, C


# ─────────────────────────────────────────────────────────────────────────────
# BOX MEASUREMENT HELPERS (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

def _measure_top_face_at_height(corners_px, K, R, t, z_height):
    world_pts = []
    for px in corners_px:
        pt = ray_to_plane_z(float(px[0]), float(px[1]), R, t, K, z_world=z_height)
        if pt is not None:
            world_pts.append(pt[:2])
    if len(world_pts) < 3:
        return None
    wc = np.array(world_pts)
    all_dists = sorted(
        [np.linalg.norm(wc[i] - wc[j])
         for i in range(len(wc)) for j in range(i + 1, len(wc))],
        reverse=True,
    )
    unique = [all_dists[0]]
    for d in all_dists[1:]:
        if abs(d - unique[-1]) > 0.01:
            unique.append(d)
    return float(unique[0]), float(unique[1] if len(unique) > 1 else unique[0])


def _estimate_height_from_top_edge(top_mask_bool, K, R, t):
    rows, cols = np.where(top_mask_bool)
    if len(rows) == 0:
        return 0.0
    max_row   = int(rows.max())
    edge_cols = cols[rows == max_row]
    sample_cols = edge_cols[
        np.linspace(0, len(edge_cols) - 1, min(7, len(edge_cols)), dtype=int)
    ]
    K_inv   = np.linalg.inv(K)
    C_world = -(R.T @ t)
    floor_anchors = []
    for col in sample_cols:
        pt = ray_to_plane_z(float(col), float(max_row), R, t, K, 0.0)
        if pt is not None:
            floor_anchors.append(pt[:2])
    if not floor_anchors:
        return 0.0
    anchor_xy = np.mean(floor_anchors, axis=0)
    anchor_3d = np.array([anchor_xy[0], anchor_xy[1], 0.0])
    vert_dir  = np.array([0., 0., 1.])
    h_vals = []
    for col in sample_cols:
        d_cam   = K_inv @ np.array([float(col), float(max_row), 1.0])
        d_world = R.T @ d_cam
        _, pt_p = closest_point_on_lines(C_world, d_world, anchor_3d, vert_dir)
        h = float(pt_p[2])
        if 0.01 < h < 2.5:
            h_vals.append(h)
    if not h_vals:
        return 0.0
    result = float(np.median(h_vals))
    log.info("Height strategy B: %.4f m (%d samples)", result, len(h_vals))
    return result


def _estimate_height_hough(top_mask_bool, image_bgr, K, R, t):
    K_inv   = np.linalg.inv(K)
    C_world = -(R.T @ t)
    rows, cols = np.where(top_mask_bool)
    if len(rows) == 0:
        return 0.0
    max_row_top = int(rows.max())
    min_row_top = int(rows.min())
    strip_height = max(20, int((max_row_top - min_row_top) * 0.9))
    strip_y1 = max_row_top
    strip_y2 = min(image_bgr.shape[0] - 1, max_row_top + strip_height)
    if strip_y2 <= strip_y1 + 5:
        return 0.0
    gray_strip  = cv2.cvtColor(image_bgr[strip_y1:strip_y2, :], cv2.COLOR_BGR2GRAY)
    edges_strip = cv2.Canny(gray_strip, 30, 100)
    lines = cv2.HoughLinesP(edges_strip, 1, np.pi / 180,
                            threshold=40, minLineLength=30, maxLineGap=15)
    if lines is None:
        return 0.0
    best_y = None
    for ln in lines:
        x1, y1_l, x2, y2_l = ln[0]
        angle = abs(np.degrees(np.arctan2(y2_l - y1_l, x2 - x1)))
        if angle < 20 or angle > 160:
            abs_y = (y1_l + y2_l) / 2 + strip_y1
            if best_y is None or abs_y < best_y:
                best_y = abs_y
    if best_y is None:
        return 0.0
    cx_top   = float(np.mean(cols))
    pt_floor = ray_to_plane_z(cx_top, best_y, R, t, K, z_world=0.0)
    if pt_floor is None:
        return 0.0
    anchor_3d = np.array([pt_floor[0], pt_floor[1], 0.0])
    vert_dir  = np.array([0., 0., 1.])
    d_cam     = K_inv @ np.array([cx_top, float(max_row_top), 1.0])
    d_world   = R.T @ d_cam
    _, pt_p   = closest_point_on_lines(C_world, d_world, anchor_3d, vert_dir)
    h = float(pt_p[2])
    if 0.01 < h < 2.5:
        log.info("Height strategy C: %.4f m", h)
        return h
    return 0.0


def _sam_measure_box(masks, image_bgr, K, R, t):
    img_h, img_w = image_bgr.shape[:2]
    img_area     = img_h * img_w
    candidates   = []

    for m in masks:
        if m["area"] < img_area * 0.01 or m["area"] > img_area * 0.65:
            continue
        if m.get("predicted_iou", 1.0) < 0.84:
            continue
        seg = m["segmentation"].astype(np.uint8) * 255
        conts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not conts:
            continue
        cnt    = max(conts, key=cv2.contourArea)
        hull   = cv2.convexHull(cnt)
        rect   = cv2.minAreaRect(hull)
        w_r, h_r = rect[1]
        if w_r < 5 or h_r < 5:
            continue
        ratio  = max(w_r, h_r) / (min(w_r, h_r) + 1e-6)
        fill   = m["area"] / (w_r * h_r + 1e-6)
        box_px = cv2.boxPoints(rect).astype(np.float32)
        M_mom  = cv2.moments(cnt)
        cy_px  = M_mom["m01"] / (M_mom["m00"] + 1e-6)
        candidates.append(dict(
            mask=m["segmentation"], corners_px=box_px,
            area=m["area"], ratio=ratio, fill=fill,
            centroid_y=cy_px, iou=m.get("predicted_iou", 1.0),
        ))

    if not candidates:
        return None

    top_cands = [c for c in candidates if c["ratio"] < 2.8 and c["fill"] > 0.62]
    if not top_cands:
        top_cands = [c for c in candidates if c["ratio"] < 3.5]
    if not top_cands:
        return None

    top_cands.sort(key=lambda c: c["area"] * c["fill"], reverse=True)
    top = top_cands[0]

    result_z0 = _measure_top_face_at_height(top["corners_px"], K, R, t, 0.0)
    if result_z0 is None:
        return None
    length_approx, width_approx = result_z0
    if not (0.02 < length_approx < 5.0 and 0.02 < width_approx < 5.0):
        return None

    top_mask_bool = top["mask"]
    top_cy        = top["centroid_y"]
    box_height    = 0.0
    K_inv         = np.linalg.inv(K)
    C_world       = -(R.T @ t)

    def _iou(a, b):
        return np.logical_and(a, b).sum() / (np.logical_or(a, b).sum() + 1e-6)

    side_cands = [
        c for c in candidates
        if c["centroid_y"] > top_cy
        and c["ratio"] > 1.1
        and c is not top
        and _iou(c["mask"], top_mask_bool) < 0.35
    ]
    if side_cands:
        side_cands.sort(key=lambda c: c["area"] * c["fill"], reverse=True)
        side = side_cands[0]
        px_sorted = sorted(side["corners_px"], key=lambda p: p[1])
        bottom_px, top_px = px_sorted[2:], px_sorted[:2]
        anchor_xys = []
        for px in bottom_px:
            pt = ray_to_plane_z(float(px[0]), float(px[1]), R, t, K, 0.0)
            if pt is not None:
                anchor_xys.append(pt[:2])
        if anchor_xys:
            anchor_xy = np.mean(anchor_xys, axis=0)
            anchor_3d = np.array([anchor_xy[0], anchor_xy[1], 0.0])
            vert_dir  = np.array([0., 0., 1.])
            h_vals = []
            for px in top_px:
                d_cam   = K_inv @ np.array([float(px[0]), float(px[1]), 1.0])
                d_world = R.T @ d_cam
                _, pt_p = closest_point_on_lines(C_world, d_world, anchor_3d, vert_dir)
                h = float(pt_p[2])
                if 0.01 < h < 2.5:
                    h_vals.append(h)
            if h_vals:
                box_height = float(np.median(h_vals))

    if box_height < 0.005:
        box_height = _estimate_height_from_top_edge(top_mask_bool, K, R, t)
    if box_height < 0.005:
        box_height = _estimate_height_hough(top_mask_bool, image_bgr, K, R, t)

    if not (0.0 <= box_height < 3.0):
        box_height = 0.0

    if box_height > 0.005:
        result_zh = _measure_top_face_at_height(
            top["corners_px"], K, R, t, z_height=box_height)
        if result_zh is not None:
            l2, w2 = result_zh
            if 0.02 < l2 < 5.0 and 0.02 < w2 < 5.0:
                length_approx, width_approx = l2, w2

    return {"length": length_approx, "width": width_approx, "height": box_height}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: /status
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "sam_available":  _sam_available  or _load_sam(),
        "yolo_available": _yolo_available or _load_yolo(),
        "mode": "tap-to-sam-v8",
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: /debug_sam
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/debug_sam", methods=["POST"])
def debug_sam():
    image = _decode_image()
    if image is None:
        return jsonify({"error": "No image or decode failed."}), 400

    try:
        tap_x = float(request.form["tap_x"])
        tap_y = float(request.form["tap_y"])
    except (KeyError, ValueError):
        return jsonify({"error": "tap_x and tap_y required."}), 400

    corners, ar, ok, msg = detect_card_from_tap(image, tap_x, tap_y)

    debug_img = image.copy()
    cv2.circle(debug_img, (int(tap_x), int(tap_y)), 12, (0, 255, 255), -1)
    cv2.circle(debug_img, (int(tap_x), int(tap_y)), 14, (0, 0, 0), 2)

    if corners is not None:
        pts = corners.astype(int)
        color = (0, 255, 0) if ok else (0, 100, 255)
        cv2.polylines(debug_img, [pts], True, color, 3)
        for i, (lbl, col) in enumerate(zip(
                ["TL", "TR", "BR", "BL"],
                [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255)])):
            x, y = int(pts[i][0]), int(pts[i][1])
            cv2.circle(debug_img, (x, y), 8, col, -1)
            cv2.putText(debug_img, lbl, (x + 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    status_text = f"ar={ar:.2f}  {'OK' if ok else 'FAIL'}  {msg[:60]}"
    cv2.putText(debug_img, status_text, (10, debug_img.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    _, buf = cv2.imencode(".png", debug_img)
    b64    = base64.b64encode(buf.tobytes()).decode()
    return jsonify({
        "ok":                     ok,
        "aspect_ratio":           ar,
        "message":                msg,
        "corners":                corners.tolist() if corners is not None else None,
        "sam_available":          _sam_available,
        "debug_image_png_base64": b64,
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: /calibrate_height
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/calibrate_height", methods=["POST"])
def calibrate_height():
    image = _decode_image()
    if image is None:
        return jsonify({"error": "No image provided or decode failed."}), 400

    try:
        tap_x = float(request.form["tap_x"])
        tap_y = float(request.form["tap_y"])
        focal_length_frontend = float(request.form.get("focal_length", 0)) or None
    except (KeyError, ValueError):
        return jsonify({"error": "tap_x, tap_y required; focal_length optional."}), 400

    h, w = image.shape[:2]
    log.info("Calibrate: %dx%d  tap=(%.0f,%.0f)  f_frontend=%s px",
             w, h, tap_x, tap_y,
             f"{focal_length_frontend:.1f}" if focal_length_frontend else "none")

    if w < 64 or h < 64:
        return jsonify({"error": f"Image too small ({w}×{h})."}), 400

    corners_2d, ar, ok, msg = detect_card_from_tap(image, tap_x, tap_y)

    if not ok:
        return jsonify({
            "error": msg,
            "aspect_ratio": ar,
            "sam_available": _sam_available,
            "retry": True,
        }), 422

    log.info("Card corners (tap-SAM ar=%.3f): %s", ar, corners_2d.tolist())

    # ── VP focal length ───────────────────────────────────────────────────────
    vp_f, vp_reason = calculate_focal_length_from_card(corners_2d, w, h)
    vp_f_valid = vp_f is not None and 100.0 < vp_f < 4000.0

    if vp_f_valid:
        focal_length_used   = vp_f
        focal_length_source = "vanishing_point"
        log.info("VP focal: %.1f px  (%s)", focal_length_used, vp_reason)
    else:
        # No heuristic fallback — return a clear error with coaching message
        log.warning("VP failed: %s", vp_reason)
        return jsonify({
            "error": (
                f"Could not compute focal length from card geometry. "
                f"{vp_reason} "
                f"Tip: move the card to ~40–60 cm from the camera and tilt your "
                f"phone so the card edges clearly converge (trapezoid shape)."
            ),
            "aspect_ratio": ar,
            "vp_reason": vp_reason,
            "sam_available": _sam_available,
            "retry": True,
        }), 422

    K = np.array([
        [focal_length_used, 0,                w / 2.0],
        [0,                 focal_length_used, h / 2.0],
        [0,                 0,                1       ],
    ], dtype=np.float64)

    calibrated_z, R_mat, C = _run_pnp(corners_2d, K)
    if calibrated_z is None:
        return jsonify({"error": "PnP solve failed. Ensure card is fully visible and tilted."}), 422

    log.info("Camera height: %.4f m", calibrated_z)

    if not (CAMERA_HEIGHT_MIN <= calibrated_z <= CAMERA_HEIGHT_MAX):
        log.warning("Height %.3f m outside sane range.", calibrated_z)
        return jsonify({
            "error": (
                f"Camera height {calibrated_z:.2f} m is outside the expected range "
                f"({CAMERA_HEIGHT_MIN}–{CAMERA_HEIGHT_MAX} m). "
                f"Hold phone at chest height (~0.8–1.4 m) and ensure card is flat on the floor."
            ),
            "retry": True,
        }), 422

    return jsonify({
        "calibrated_z":        calibrated_z,
        "focal_length_used":   focal_length_used,
        "focal_length_source": focal_length_source,
        "detection_method":    "tap_sam_multipoint",
        "aspect_ratio":        ar,
        "vp_reason":           vp_reason,
        "sam_available":       _sam_available,
        "vp_focal_length":     vp_f,
        "debug_corners":       corners_2d.tolist(),
        "debug_cam_center":    C.tolist(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: /measure_box (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/measure_box", methods=["POST"])
def measure_box():
    use_image = "image" in request.files

    if use_image:
        try:
            calibrated_z = float(request.form["calibrated_z"])
            R_gyro  = np.array(json.loads(request.form["R_gyro"]), dtype=np.float64)
            K       = np.array(json.loads(request.form["K"]),      dtype=np.float64)
            taps_raw = request.form.get("taps")
            taps    = json.loads(taps_raw) if taps_raw else None
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"Bad form fields: {exc}"}), 400
        image = _decode_image()
        if image is None:
            return jsonify({"error": "Could not decode image."}), 400
    else:
        data = request.get_json(force=True)
        for k in ("calibrated_z", "R_gyro", "K", "taps"):
            if k not in data:
                return jsonify({"error": f"Missing: {k}"}), 400
        try:
            calibrated_z = float(data["calibrated_z"])
            R_gyro  = np.array(data["R_gyro"], dtype=np.float64)
            K       = np.array(data["K"],      dtype=np.float64)
            taps    = data["taps"]
        except (ValueError, TypeError) as exc:
            return jsonify({"error": f"Bad format: {exc}"}), 400
        image = None

    if R_gyro.shape != (3, 3) or K.shape != (3, 3):
        return jsonify({"error": "R_gyro and K must be 3×3."}), 400

    t_extr = -(R_gyro @ np.array([0., 0., calibrated_z]))
    log.info("Measure: z=%.4f, image=%s", calibrated_z, use_image)

    sam_result = None
    if image is not None:
        yolo_box_bbox = _yolo_detect_box_bbox(image)

        if _sam_available or _load_sam():
            if yolo_box_bbox is not None:
                expanded = _expand_bbox(yolo_box_bbox, image.shape, pad_frac=0.20)
                x1e, y1e, x2e, y2e = expanded
                crop = image[y1e:y2e, x1e:x2e]
                masks_crop = _run_sam_auto(crop)
                masks = []
                for m in masks_crop:
                    full_mask = np.zeros(image.shape[:2], dtype=bool)
                    full_mask[y1e:y2e, x1e:x2e] = m["segmentation"]
                    m2 = dict(m)
                    m2["segmentation"] = full_mask
                    masks.append(m2)
            else:
                masks = _run_sam_auto(image)

            if masks:
                sam_result = _sam_measure_box(masks, image, K, R_gyro, t_extr)

    if sam_result is None:
        if not taps or len(taps) < 3:
            return jsonify({
                "error": "SAM measurement failed and no tap coordinates supplied.",
                "sam_available": _sam_available,
            }), 422

        u_bl, v_bl = float(taps[0]["u"]), float(taps[0]["v"])
        u_br, v_br = float(taps[1]["u"]), float(taps[1]["v"])
        u_tl, v_tl = float(taps[2]["u"]), float(taps[2]["v"])

        P_BL = ray_to_z0_plane(u_bl, v_bl, R_gyro, t_extr, K)
        P_BR = ray_to_z0_plane(u_br, v_br, R_gyro, t_extr, K)
        if P_BL is None or P_BR is None:
            return jsonify({"error": "Floor projection failed. Tilt phone more steeply."}), 422

        horiz   = float(np.linalg.norm(P_BL - P_BR))
        K_inv   = np.linalg.inv(K)
        d_tl    = R_gyro.T @ (K_inv @ np.array([u_tl, v_tl, 1.]))
        C_world = -(R_gyro.T @ t_extr)
        _, pt_p = closest_point_on_lines(C_world, d_tl, P_BL, np.array([0., 0., 1.]))
        h_box   = float(max(pt_p[2], 0.))
        sam_result = {"length": horiz, "width": horiz, "height": h_box, "tap_fallback": True}
        log.info("Tap fallback: L=W=%.4f H=%.4f", horiz, h_box)

    L = sam_result["length"]
    W = sam_result["width"]
    H = sam_result["height"]

    return jsonify({
        "length": round(L, 4),
        "width":  round(W, 4),
        "height": round(H, 4),
        "volume": round(L * W * H, 6),
        "sam_used":     not sam_result.get("tap_fallback", False) and use_image,
        "tap_fallback": sam_result.get("tap_fallback", False),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)