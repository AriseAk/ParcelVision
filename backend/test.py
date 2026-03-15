"""
test_online_images.py  –  ParcelVision offline tester
======================================================
Downloads furniture images from the web, runs the full
YOLO segmentation + depth_server measurement pipeline,
and displays results — no camera needed.

Usage:
    # make sure depth_server.py is running first:
    #   python depth_server.py

    python test_online_images.py
    python test_online_images.py --url "https://example.com/chair.jpg"
    python test_online_images.py --file my_image.jpg
"""

import argparse
import base64
import time
import urllib.request

import cv2 as cv
import numpy as np
import requests
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEPTH_SERVER_URL = "http://127.0.0.1:5000"
JPEG_QUALITY     = 90
CONF_THRESHOLD   = 0.25          # slightly lower than live — static images are easier
ALLOWED_CLASSES  = {"chair", "couch", "dining table"}
LABEL_MAP        = {"couch": "Sofa", "dining table": "Table", "chair": "Chair"}

# A small set of freely-licensed furniture images (Wikimedia Commons / Unsplash)
DEFAULT_TEST_URLS = [
    # wooden chair
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Poltrona_Frau-Archibald.jpg/800px-Poltrona_Frau-Archibald.jpg",
    # dining table with chairs
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Dining_room_at_Ascott_House.jpg/1024px-Dining_room_at_Ascott_House.jpg",
    # sofa
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Sofa_wikipedia.jpg/1024px-Sofa_wikipedia.jpg",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image_from_url(url: str) -> np.ndarray | None:
    """Download an image URL → BGR numpy array."""
    print(f"  Downloading {url[:80]}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        arr = np.frombuffer(data, np.uint8)
        bgr = cv.imdecode(arr, cv.IMREAD_COLOR)
        if bgr is None:
            print("  ERROR: could not decode image")
        return bgr
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return None


def load_image_from_file(path: str) -> np.ndarray | None:
    bgr = cv.imread(path)
    if bgr is None:
        print(f"  ERROR: could not read file {path}")
    return bgr


def encode_frame(bgr: np.ndarray) -> str:
    """BGR numpy → base64 JPEG string."""
    _, jpg = cv.imencode(".jpg", bgr, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(jpg.tobytes()).decode()


def encode_mask_rle(binary_mask: np.ndarray) -> dict:
    flat = binary_mask.flatten().tolist()
    counts, run, cur = [], 0, 0
    for v in flat:
        if v == cur:
            run += 1
        else:
            counts.append(run)
            run, cur = 1, v
    counts.append(run)
    return {"counts": counts, "shape": list(binary_mask.shape)}


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation(model: YOLO, bgr: np.ndarray) -> list[dict]:
    """Run YOLO on a single frame, return list of detection dicts."""
    results = model(bgr, verbose=False)
    detections = []

    if results[0].masks is None:
        return detections

    for box, raw_mask in zip(results[0].boxes, results[0].masks.data):
        class_id   = int(box.cls[0])
        class_name = model.names[class_id]
        conf       = float(box.conf[0])

        if class_name not in ALLOWED_CLASSES or conf < CONF_THRESHOLD:
            continue

        detections.append({
            "object_id":    len(detections) + 1,   # simple 1-based ID for test
            "class":        LABEL_MAP.get(class_name, class_name),
            "confidence":   round(conf, 4),
            "segmentation_score": round(conf, 4),   # proxy for offline test
            "mask_stability":     1.0,
            "bounding_box": box.xyxy[0].cpu().numpy(),
            "mask_tensor":  raw_mask,
        })

    return detections


def build_measure_payload(bgr: np.ndarray, detections: list[dict]) -> dict:
    """Build the JSON payload for POST /measure."""
    h, w = bgr.shape[:2]
    objects = []

    for det in detections:
        # Resize mask tensor to frame resolution
        mask_np = F.interpolate(
            det["mask_tensor"].unsqueeze(0).unsqueeze(0).float(),
            size=(h, w), mode="nearest"
        ).squeeze().cpu().numpy()
        binary = (mask_np > 0.5).astype(np.uint8)

        objects.append({
            "object_id":          det["object_id"],
            "class":              det["class"],
            "confidence":         det["confidence"],
            "segmentation_score": det["segmentation_score"],
            "mask_stability":     det["mask_stability"],
            "mask_rle":           encode_mask_rle(binary),
        })

    return {
        "frame_b64": encode_frame(bgr),
        "objects":   objects,
    }


def call_measure(payload: dict) -> dict | None:
    """POST /measure and return parsed JSON, or None on failure."""
    try:
        resp = requests.post(
            f"{DEPTH_SERVER_URL}/measure",
            json=payload,
            timeout=30.0,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            result = resp.json()
            print(f"  Server error {resp.status_code}")
            # If the server forwarded a traceback, print it
            if "traceback" in result:
                print("  Server traceback:\n" + result["traceback"])
            else:
                print(f"  Response: {resp.text[:400]}")
            return None
    except requests.exceptions.Timeout:
        print("  Request timed out — is depth_server.py running?")
        return None
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

COLORS = [
    (0, 200, 100),
    (0, 120, 255),
    (255, 180, 0),
    (180, 0, 255),
    (0, 220, 220),
]


def draw_results(
    bgr: np.ndarray,
    detections: list[dict],
    measurements: dict[int, dict],
) -> np.ndarray:
    """Overlay masks, boxes, and dimension labels onto the image."""
    out     = bgr.copy()
    overlay = bgr.copy()
    h, w    = bgr.shape[:2]

    for det in detections:
        oid   = det["object_id"]
        color = COLORS[(oid - 1) % len(COLORS)]

        # ── Mask overlay ─────────────────────────────────────────────────────
        mask_np = F.interpolate(
            det["mask_tensor"].unsqueeze(0).unsqueeze(0).float(),
            size=(h, w), mode="nearest"
        ).squeeze().cpu().numpy()
        binary = (mask_np > 0.5).astype(np.uint8)

        colored = np.zeros_like(out)
        colored[binary == 1] = color
        cv.addWeighted(colored, 0.45, overlay, 0.55, 0, overlay)

        # ── Bounding box ─────────────────────────────────────────────────────
        x1, y1, x2, y2 = map(int, det["bounding_box"])
        cv.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # ── Label block ──────────────────────────────────────────────────────
        m = measurements.get(oid)
        lines = [
            f"ID:{oid} {det['class']}  det={det['confidence']:.2f}",
        ]
        if m:
            lines += [
                f"L:{m['length']:.2f}m  W:{m['width']:.2f}m  H:{m['height']:.2f}m",
                f"Vol:{m['volume_m3']:.3f}m³  conf:{m['confidence']:.2f}",
                f"depth:{m['mean_depth_m']:.2f}m  pts:{m['point_count']}",
            ]
        else:
            lines.append("(no measurement)")

        font      = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1
        line_h    = 18

        # Background box for readability
        max_w   = max(cv.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
        box_top = max(y1 - line_h * len(lines) - 6, 0)
        cv.rectangle(out, (x1, box_top), (x1 + max_w + 6, y1), (20, 20, 20), -1)

        for i, line in enumerate(lines):
            y_pos = box_top + (i + 1) * line_h
            cv.putText(out, line, (x1 + 3, y_pos), font, font_scale, color, thickness, cv.LINE_AA)

    cv.addWeighted(overlay, 0.35, out, 0.65, 0, out)
    return out


def print_results(image_label: str, detections: list[dict], measurements: dict[int, dict]):
    """Pretty-print measurement results to the terminal."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {image_label}")
    print(sep)
    if not detections:
        print("  No furniture detected.")
        return

    for det in detections:
        oid = det["object_id"]
        m   = measurements.get(oid)
        print(f"\n  [{oid}] {det['class']:12s}  detection_conf={det['confidence']:.2f}")
        if m:
            print(f"       Length : {m['length']:.3f} m")
            print(f"       Width  : {m['width']:.3f} m")
            print(f"       Height : {m['height']:.3f} m")
            print(f"       Volume : {m['volume_m3']:.4f} m³")
            print(f"       Conf   : {m['confidence']:.3f}")
            print(f"       Depth  : {m['mean_depth_m']:.2f} m (mean)")
            print(f"       Points : {m['point_count']}")
        else:
            print("       Measurement: failed (check depth_server logs)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    bgr: np.ndarray,
    label: str,
    model: YOLO,
    show: bool = True,
    save: str | None = None,
):
    print(f"\n{'='*60}")
    print(f"Processing: {label}")
    print(f"  Resolution: {bgr.shape[1]}x{bgr.shape[0]}")

    # Step 1 — segmentation
    t0 = time.perf_counter()
    detections = run_segmentation(model, bgr)
    seg_ms = (time.perf_counter() - t0) * 1000
    print(f"  Segmentation: {len(detections)} object(s) in {seg_ms:.1f}ms")

    if not detections:
        print("  Nothing to measure — skipping.")
        if show:
            cv.imshow(label, bgr)
        return

    # Step 2 — measurement via depth_server
    payload = build_measure_payload(bgr, detections)
    print(f"  Calling /measure ...")
    result = call_measure(payload)

    measurements: dict[int, dict] = {}
    if result:
        for m in result.get("objects", []):
            measurements[m["object_id"]] = m
        print(f"  Inference: {result.get('inference_ms')}ms  "
              f"({len(measurements)}/{len(detections)} measured)")

    # Step 3 — print + visualise
    print_results(label, detections, measurements)

    vis = draw_results(bgr, detections, measurements)

    if save:
        cv.imwrite(save, vis)
        print(f"  Saved → {save}")

    if show:
        cv.imshow(label[:60], vis)
        print("  Press any key to continue...")
        cv.waitKey(0)
        cv.destroyAllWindows()


def main():
    # Health check
    print("Checking depth server...")
    try:
        r = requests.get(f"{DEPTH_SERVER_URL}/health", timeout=5.0)
        info = r.json()
        print(f"OK model={info['model']} device={info['device']}")
    except Exception:
        print("Depth server not reachable. Run depth_server.py first.")
        return

    # Load YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n-seg.pt")
    model.to(device)
    print("YOLO running on", device)

    # Load test image
    bgr = cv.imread("chair1.jpg")
    if bgr is None:
        print("test.jpg not found in this folder")
        return

    # Run pipeline
    process_image(
        bgr,
        "test.jpg",
        model,
        show=True,
        save="result.jpg"
    )

    print("Saved result to result.jpg")


if __name__ == "__main__":
    main()