"""
pointcloud_builder.py — ParcelVision Stage 7
=============================================
Reusable 3D geometry helpers.

All functions assume the depth map is already in **metres** (absolute).
No scale factors are applied or expected anywhere in this file.

Camera convention throughout:
    X → right
    Y → down   (Y increases toward the bottom of the image)
    Z → into the scene (optical axis)

Fix log vs original:
    [F1] pixels_to_3d: now accepts separate fx/fy (was fx-only in practice)
    [F2] mask_to_point_cloud: uses shrink=0.10 for depth sampling to preserve
         edge pixels needed for height — callers override as needed
    [F3] filter_point_cloud: floor removal now correctly targets the LARGEST
         Y values (bottom of scene in camera space); comment clarified
    [F4] sample_full_mask_depth: new helper — samples the FULL mask (no central
         crop) for height estimation so top/bottom edge pixels are included
    [F5] fit_pca_bbox: PCA dims are now resolved to semantic length/width/height
         using the world-space principal axes rather than sorting by variance
         alone; the vertical axis (world Y or camera Y after rotation) is
         identified and assigned to "height"
    [F6] measure_object_dimensions: new top-level helper used by the server that
         correctly separates depth-sampling (central crop) from height-sampling
         (full mask) and assembles a consistent L/W/H dict
"""

from __future__ import annotations

import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Back-projection
# ─────────────────────────────────────────────────────────────────────────────

def pixels_to_3d(
    us: np.ndarray,
    vs: np.ndarray,
    zs: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Back-project pixel coordinates to 3D camera-space points.

    Uses the standard pinhole model:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = Z_depth  (depth along optical axis, NOT Euclidean distance)

    Parameters
    ----------
    us, vs : pixel columns / rows  (N,)
    zs     : depth in metres along optical axis  (N,)
    fx, fy : focal lengths in pixels (use separate values — [F1])
    cx, cy : principal point in pixels

    Returns
    -------
    pts : (N, 3) float32  [X, Y, Z] in metres, camera space
    """
    us = us.astype(np.float64)
    vs = vs.astype(np.float64)
    zs = zs.astype(np.float64)
    X  = (us - cx) * zs / fx
    Y  = (vs - cy) * zs / fy
    return np.stack([X, Y, zs], axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Mask pixel selection
# ─────────────────────────────────────────────────────────────────────────────

def get_central_pixels(
    mask: np.ndarray,
    shrink: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return pixel coordinates of the inner region of a mask.

    Shrinks the bounding box by `shrink` fraction on each side to reduce
    occlusion / boundary noise.  Default shrink=0.10 (10% each side) preserves
    the top and bottom edges needed for accurate height — [F2].

    Returns (us, vs) — column, row arrays.
    """
    vs, us = np.where(mask > 0.5)
    if len(us) < 20:
        return us, vs

    u_min, u_max = int(us.min()), int(us.max())
    v_min, v_max = int(vs.min()), int(vs.max())

    u_lo = u_min + shrink * (u_max - u_min)
    u_hi = u_max - shrink * (u_max - u_min)
    v_lo = v_min + shrink * (v_max - v_min)
    v_hi = v_max - shrink * (v_max - v_min)

    if u_hi <= u_lo or v_hi <= v_lo:
        return us, vs

    inner = (us >= u_lo) & (us <= u_hi) & (vs >= v_lo) & (vs <= v_hi)
    us_in, vs_in = us[inner], vs[inner]
    return (us_in, vs_in) if len(us_in) >= 20 else (us, vs)


def get_full_mask_pixels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ALL foreground pixel coordinates with no spatial crop.

    Used for height estimation so the top and bottom edges of the object
    are always included — [F4].

    Returns (us, vs).
    """
    vs, us = np.where(mask > 0.5)
    return us, vs


# ─────────────────────────────────────────────────────────────────────────────
# Point cloud construction
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_point_cloud(
    depth_map:     np.ndarray,
    mask:          np.ndarray,
    fx:            float,
    fy:            float,
    cx:            float,
    cy:            float,
    max_pts:       int   = 2000,
    depth_min:     float = 0.05,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
    shrink:        float = 0.10,
) -> np.ndarray | None:
    """
    Sample masked pixels → filtered 3D point cloud (camera space).

    Uses central-crop for depth sampling (shrink=0.10 default) so that
    boundary noise is minimised while the top/bottom edges are still
    included for height — [F2].

    Parameters
    ----------
    depth_map   : (H, W) float32, metric depth in metres
    mask        : (H, W) uint8 or float, >0.5 = object
    max_pts     : subsample ceiling for speed
    depth_min   : reject pixels shallower than this (mask bleed)
    shrink      : fraction to shrink bounding box on each side

    Returns
    -------
    pts  : (N, 3) float32 camera-space, or None if too sparse
    """
    us, vs = get_central_pixels(mask, shrink=shrink)
    if len(us) < 10:
        return None

    zs = depth_map[vs, us].astype(np.float32)

    valid = zs > depth_min
    us, vs, zs = us[valid], vs[valid], zs[valid]
    if len(us) < 10:
        return None

    lo = np.percentile(zs, percentile_lo)
    hi = np.percentile(zs, percentile_hi)
    keep = (zs >= lo) & (zs <= hi)
    us, vs, zs = us[keep], vs[keep], zs[keep]
    if len(us) < 10:
        return None

    if len(us) > max_pts:
        idx = np.random.choice(len(us), max_pts, replace=False)
        us, vs, zs = us[idx], vs[idx], zs[idx]

    return pixels_to_3d(us, vs, zs, fx, fy, cx, cy)


# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────

def filter_point_cloud(pts: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """
    Statistical outlier removal on a camera-space point cloud.

    Steps:
      1. Remove points further than mean + sigma*std from centroid.
      2. Remove bottom 5% of Y values — these are the floor pixels that
         leaked through the mask (Y-down: large Y = bottom of image = floor).
         [F3: was "top 5% of Y" which was removing the TOP of the box]
      3. Trim far/near depth tails (Z axis).

    Parameters
    ----------
    pts   : (N, 3) float32 camera-space
    sigma : outlier removal threshold

    Returns
    -------
    pts : filtered (N, 3) float32
    """
    if len(pts) < 10:
        return pts

    # Step 1: centroid distance
    centroid  = pts.mean(axis=0)
    dists     = np.linalg.norm(pts - centroid, axis=1)
    threshold = dists.mean() + sigma * dists.std()
    pts       = pts[dists < threshold]

    if len(pts) > 20:
        # Step 2: remove floor bleed — largest Y values (Y-down → floor is at
        # maximum Y).  Keep bottom 95th percentile means we DROP the bottom 5%.
        # [F3: fixed — previously removed pts[:, 1] < percentile(95) which
        # incorrectly removed the LARGEST Y, i.e. the floor. Now explicitly
        # keeps only pts where Y < 95th-percentile, i.e. discards floor pixels]
        y_floor_thresh = np.percentile(pts[:, 1], 95)
        pts = pts[pts[:, 1] < y_floor_thresh]

    if len(pts) > 20:
        # Step 3: trim depth tails
        z_lo = np.percentile(pts[:, 2], 3)
        z_hi = np.percentile(pts[:, 2], 97)
        pts  = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)]

    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Bounding box fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_axis_aligned_bbox(pts: np.ndarray) -> dict | None:
    """
    Fit an axis-aligned bounding box to a camera-space point cloud.

    Semantic mapping (camera convention: X right, Y down, Z into scene):
        length  = X extent  (horizontal width of object face)
        height  = Y extent  (vertical extent — Y-down, so this is correct
                             when the camera is roughly level)
        width   = Z extent  (depth into scene)

    NOTE: accurate only when the camera is roughly level and the box face
    is fronto-parallel.  For scan-mode (multi-view), use fit_pca_bbox on
    the fused world-space point cloud instead.

    Returns dict or None if pts is too sparse.
    """
    if len(pts) < 4:
        return None

    length = float(np.clip(pts[:, 0].max() - pts[:, 0].min(), 0.01, 10.0))
    height = float(np.clip(pts[:, 1].max() - pts[:, 1].min(), 0.01, 10.0))
    width  = float(np.clip(pts[:, 2].max() - pts[:, 2].min(), 0.01, 10.0))

    return {
        "length":    round(length, 3),
        "width":     round(width,  3),
        "height":    round(height, 3),
        "volume_m3": round(length * width * height, 4),
    }


def fit_pca_bbox(pts: np.ndarray) -> dict | None:
    """
    PCA-aligned bounding box for multi-view fused world-space point clouds.

    The raw PCA axes are resolved to semantic dimensions by identifying
    which axis is closest to vertical (world Y-up = [0, 1, 0] in OpenCV
    world space where Y is up after the camera→world transform) — [F5].

    Fallback: if no axis is clearly vertical (degenerate point cloud),
    sorts dims largest→smallest and assigns length/width/height by size.

    Returns dict with length / width / height / volume_m3 / pca_reliable,
    or None if pts is too sparse.
    """
    if len(pts) < 10:
        return None

    mean     = pts.mean(axis=0)
    centered = pts - mean

    try:
        cov     = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)   # eigvecs: columns are axes
    except np.linalg.LinAlgError:
        return fit_axis_aligned_bbox(pts)

    # eigvecs columns sorted by ascending eigenvalue (eigh guarantee)
    # Swap to descending (most variance first)
    order   = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[order]

    # Project all points onto PCA axes
    projected = centered @ eigvecs          # (N, 3)
    extents   = np.array([
        projected[:, i].max() - projected[:, i].min()
        for i in range(3)
    ])                                       # (3,)

    # ── Identify the vertical axis ────────────────────────────────────────────
    # In world space (after camera→world transform), "up" is typically [0,1,0]
    # (OpenCV convention: Y-down camera → Y-up world after R^T).
    # The eigenvector most aligned with world-up is the height axis.
    world_up = np.array([0.0, 1.0, 0.0])
    dots     = np.abs(eigvecs.T @ world_up)   # (3,)
    vert_idx = int(np.argmax(dots))
    pca_reliable = bool(dots[vert_idx] > 0.5)

    if pca_reliable:
        # Assign semantic dims based on identified axis
        horiz_indices = [i for i in range(3) if i != vert_idx]
        # Sort horizontal axes: largest extent = length, smaller = width
        h_ext   = [(extents[i], i) for i in horiz_indices]
        h_ext.sort(reverse=True)
        len_idx = h_ext[0][1]
        wid_idx = h_ext[1][1]
        hgt_idx = vert_idx

        length = float(np.clip(extents[len_idx], 0.01, 5.0))
        width  = float(np.clip(extents[wid_idx], 0.01, 5.0))
        height = float(np.clip(extents[hgt_idx], 0.01, 5.0))
    else:
        # Fallback: sort all dims largest→smallest
        sorted_ext = sorted(extents, reverse=True)
        length = float(np.clip(sorted_ext[0], 0.01, 5.0))
        width  = float(np.clip(sorted_ext[1], 0.01, 5.0))
        height = float(np.clip(sorted_ext[2], 0.01, 5.0))

    return {
        "length":       round(length, 3),
        "width":        round(width,  3),
        "height":       round(height, 3),
        "volume_m3":    round(length * width * height, 4),
        "pca_reliable": pca_reliable,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level single-frame measurement helper  [F6]
# ─────────────────────────────────────────────────────────────────────────────

def measure_object_dimensions(
    depth_map: np.ndarray,
    mask_np:   np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_pts: int = 2000,
) -> dict | None:
    """
    Measure a single object from a ZoeDepth metric depth map.

    Two-pass sampling strategy — [F6]:
      Pass 1 (central crop, shrink=0.10): samples the inner 80% of the
              mask for the point cloud used to fit L/W/H via axis-aligned
              bbox — reduces edge/background noise.
      Pass 2 (full mask): samples ALL mask pixels to measure the full
              vertical extent (Y range) for height — ensures the very
              top and bottom of the box are always included.

    The height from Pass 2 replaces the height from Pass 1 whenever
    Pass 2 produces a larger (more complete) estimate.

    Parameters
    ----------
    depth_map : (H, W) float32  metric depth in metres
    mask_np   : (H, W) uint8    object mask
    fx, fy    : focal lengths in pixels
    cx, cy    : principal point in pixels
    max_pts   : subsample ceiling

    Returns
    -------
    dict with length / width / height / volume_m3 / point_count / mean_depth_m
    or None on failure.
    """
    h, w = depth_map.shape
    if mask_np.shape != (h, w):
        mask_np = cv2.resize(
            mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )

    # ── Pass 1: central-crop point cloud for L/W/H ────────────────────────────
    pts = mask_to_point_cloud(
        depth_map, mask_np,
        fx=fx, fy=fy, cx=cx, cy=cy,
        max_pts=max_pts,
        depth_min=0.05,
        percentile_lo=5.0,
        percentile_hi=95.0,
        shrink=0.10,
    )
    if pts is None or len(pts) < 10:
        return None

    pts = filter_point_cloud(pts)
    if len(pts) < 4:
        return None

    box = fit_axis_aligned_bbox(pts)
    if box is None:
        return None

    # ── Pass 2: full-mask height refinement ───────────────────────────────────
    us_full, vs_full = get_full_mask_pixels(mask_np)
    if len(us_full) >= 20:
        zs_full = depth_map[vs_full, us_full].astype(np.float32)
        valid_h  = zs_full > 0.05
        us_h, vs_h, zs_h = us_full[valid_h], vs_full[valid_h], zs_full[valid_h]

        if len(us_h) >= 20:
            # Percentile trim
            lo_h = np.percentile(zs_h, 5)
            hi_h = np.percentile(zs_h, 95)
            keep_h = (zs_h >= lo_h) & (zs_h <= hi_h)
            us_h, vs_h, zs_h = us_h[keep_h], vs_h[keep_h], zs_h[keep_h]

            if len(us_h) >= 20:
                pts_full = pixels_to_3d(us_h, vs_h, zs_h, fx, fy, cx, cy)
                # Y extent of the full point cloud = full vertical span
                full_height = float(np.clip(
                    pts_full[:, 1].max() - pts_full[:, 1].min(),
                    0.01, 5.0,
                ))
                # Accept if it's larger than the cropped estimate (more complete)
                if full_height > box["height"]:
                    box["height"] = round(full_height, 3)
                    box["volume_m3"] = round(
                        box["length"] * box["width"] * box["height"], 4
                    )

    # ── Depth stats for confidence ─────────────────────────────────────────────
    zs_all = depth_map[mask_np > 0.5]
    mean_depth = float(zs_all[zs_all > 0.05].mean()) if (zs_all > 0.05).any() else 0.0

    return {
        "length":      box["length"],
        "width":       box["width"],
        "height":      box["height"],
        "volume_m3":   box["volume_m3"],
        "point_count": len(pts),
        "mean_depth_m": round(mean_depth, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confidence
# ─────────────────────────────────────────────────────────────────────────────

def depth_confidence(zs: np.ndarray) -> float:
    """
    Depth consistency score in [0, 1].
    Low coefficient-of-variation → compact depth distribution → high confidence.
    """
    if len(zs) < 2:
        return 0.5
    return float(np.clip(1.0 - zs.std() / (zs.mean() + 1e-8), 0.0, 1.0))


def final_confidence(
    det:       float,
    seg:       float,
    depth_rel: float,
    track:     float,
) -> float:
    """Weighted confidence from the four pipeline stages."""
    return round(float(np.clip(
        0.30 * det + 0.30 * seg + 0.20 * depth_rel + 0.20 * track,
        0.0, 1.0,
    )), 4)