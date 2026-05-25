"""
main.py – ParcelVision (full pipeline: Stages 1-12)
====================================================
Runs YOLO segmentation + tracking (Stages 1-4) locally,
calls depth_server.py (Stages 5-12) for 3-D measurement.

Key fixes vs previous version:
  - last_measured_at stamped on DISPATCH not on response
    (prevents infinite retry when server returns empty list)
  - Full server response logging for debugging
  - stamp_measured() method added to FurnitureTracker
  - Traceback printed on worker exceptions
"""

import base64
import threading
import time
import traceback as tb

import cv2 as cv
import numpy as np
import requests
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CONF_THRESHOLD    = 0.3
IOU_THRESHOLD     = 0.45
MAX_LOST_FRAMES   = 60
SMOOTHING_ALPHA   = 0.6

ALLOWED_CLASSES   = {"chair", "couch", "dining table"}
LABEL_MAP         = {"couch": "Sofa", "dining table": "Table", "chair": "Chair"}

SCORE_W_CONF      = 0.40
SCORE_W_TEMPORAL  = 0.30
SCORE_W_STABILITY = 0.30

DEPTH_SERVER_URL  = "http://127.0.0.1:5000"
MEASURE_INTERVAL  = 1.0     # seconds between measurements per object
JPEG_QUALITY      = 70


# ─────────────────────────────────────────────────────────────────────────────
# RLE helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode_mask_rle(binary_mask: np.ndarray) -> dict:
    flat = binary_mask.flatten().tolist()
    counts = []
    run, cur = 0, 0
    for v in flat:
        if v == cur:
            run += 1
        else:
            counts.append(run)
            run, cur = 1, v
    counts.append(run)
    return {"counts": counts, "shape": list(binary_mask.shape)}


# ─────────────────────────────────────────────────────────────────────────────
# Track dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    track_id: int
    class_name: str
    bbox: np.ndarray
    mask_tensor: torch.Tensor
    confidence: float

    frames_seen: int = 1
    frames_lost: int = 0

    conf_history: list = field(default_factory=list)
    area_history: list = field(default_factory=list)

    temporal_consistency: float = 0.0
    mask_stability: float = 0.0
    segmentation_score: float = 0.0

    measurement: dict | None = None
    # ── Set to now() the moment a request is dispatched, NOT when it returns.
    # This is the key fix: prevents infinite retries when the server returns
    # an empty list (e.g. depth model still loading, mask too small, etc.)
    last_measured_at: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────

def bbox_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# FurnitureTracker
# ─────────────────────────────────────────────────────────────────────────────

class FurnitureTracker:

    def __init__(self):
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    # ── Main thread only ─────────────────────────────────────────────────────

    def update(self, detections: list[dict]) -> list[dict]:
        matched_tids, matched_dets = self._match(detections)

        for tid, di in zip(matched_tids, matched_dets):
            self._update_track(self._tracks[tid], detections[di])

        unmatched = [i for i in range(len(detections)) if i not in matched_dets]
        for di in unmatched:
            self._create_track(detections[di])

        matched_set = set(matched_tids)
        for tid in list(self._tracks):
            if tid not in matched_set:
                self._tracks[tid].frames_lost += 1

        self._tracks = {tid: t for tid, t in self._tracks.items()
                        if t.frames_lost <= MAX_LOST_FRAMES}

        return self._build_inventory()

    # ── Cross-thread (called by MeasurementWorker) ───────────────────────────

    def get_track(self, track_id: int) -> Track | None:
        with self._lock:
            return self._tracks.get(track_id)

    def stamp_measured(self, track_id: int, ts: float):
        """Mark a dispatch timestamp BEFORE the request is sent.
        Prevents infinite retry loops when the server response is empty."""
        with self._lock:
            if track_id in self._tracks:
                self._tracks[track_id].last_measured_at = ts

    def set_measurement(self, track_id: int, m: dict):
        """Store a successful measurement result from the server."""
        with self._lock:
            if track_id in self._tracks:
                self._tracks[track_id].measurement = m
                # Also refresh the timestamp so the cooldown restarts from now
                self._tracks[track_id].last_measured_at = time.time()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _match(self, detections):
        if not self._tracks or not detections:
            return [], []

        tids   = list(self._tracks.keys())
        tboxes = [self._tracks[t].bbox for t in tids]
        dboxes = [d["bounding_box"] for d in detections]

        iou_m = np.zeros((len(tids), len(detections)))
        for ti, tb_ in enumerate(tboxes):
            for di, db in enumerate(dboxes):
                iou_m[ti, di] = bbox_iou(tb_, db)

        m_tids, m_dets, used = [], [], set()
        order = np.dstack(np.unravel_index(
            np.argsort(iou_m, axis=None)[::-1], iou_m.shape))[0]

        for ti, di in order:
            if iou_m[ti, di] < IOU_THRESHOLD:
                break
            tid = tids[ti]
            if tid in m_tids or di in used:
                continue
            m_tids.append(tid)
            m_dets.append(di)
            used.add(di)

        return m_tids, m_dets

    def _create_track(self, det: dict):
        t = Track(
            track_id=self._next_id,
            class_name=det["class_name"],
            bbox=det["bounding_box"].copy(),
            mask_tensor=det["mask_tensor"].clone(),
            confidence=det["confidence"],
        )
        t.conf_history.append(det["confidence"])
        t.area_history.append(float(det["mask_tensor"].sum()))
        self._compute_scores(t)
        self._tracks[self._next_id] = t
        self._next_id += 1

    def _update_track(self, t: Track, det: dict):
        t.frames_lost = 0
        t.frames_seen += 1
        t.bbox = det["bounding_box"].copy()
        t.confidence = det["confidence"]

        nm = det["mask_tensor"].float()
        om = t.mask_tensor.float()
        if nm.shape != om.shape:
            nm = F.interpolate(nm.unsqueeze(0).unsqueeze(0),
                               size=om.shape, mode="nearest").squeeze()
        t.mask_tensor = (SMOOTHING_ALPHA * nm + (1 - SMOOTHING_ALPHA) * om).clamp(0, 1)

        t.conf_history.append(det["confidence"])
        t.area_history.append(float(nm.sum()))
        if len(t.conf_history) > 30:
            t.conf_history = t.conf_history[-30:]
            t.area_history = t.area_history[-30:]

        self._compute_scores(t)

    def _compute_scores(self, t: Track):
        t.temporal_consistency = t.frames_seen / max(t.frames_seen + t.frames_lost, 1)

        if len(t.area_history) >= 2:
            arr = np.array(t.area_history)
            cv_ = arr.std() / (arr.mean() + 1e-6)
            t.mask_stability = float(np.clip(1 - cv_, 0, 1))
        else:
            t.mask_stability = 1.0

        avg_conf = float(np.mean(t.conf_history)) if t.conf_history else t.confidence
        t.segmentation_score = (
            SCORE_W_CONF      * avg_conf +
            SCORE_W_TEMPORAL  * t.temporal_consistency +
            SCORE_W_STABILITY * t.mask_stability
        )

    def _build_inventory(self) -> list[dict]:
        return [{
            "object_id":            t.track_id,
            "class":                t.class_name,
            "confidence":           round(t.confidence, 4),
            "segmentation_score":   round(t.segmentation_score, 4),
            "bounding_box":         t.bbox,
            "mask":                 t.mask_tensor,
            "frames_seen":          t.frames_seen,
            "temporal_consistency": round(t.temporal_consistency, 4),
            "mask_stability":       round(t.mask_stability, 4),
            "measurement":          t.measurement,
        } for t in self._tracks.values()]


# ─────────────────────────────────────────────────────────────────────────────
# Background measurement worker
# ─────────────────────────────────────────────────────────────────────────────

class MeasurementWorker(threading.Thread):

    def __init__(self, tracker: FurnitureTracker):
        super().__init__(daemon=True)
        self._tracker  = tracker
        self._queue: list[tuple] = []
        self._lock     = threading.Lock()
        self._running  = True

    def submit(self, frame_bgr: np.ndarray, objects: list[dict]):
        with self._lock:
            self._queue.append((frame_bgr.copy(), objects))
            # Keep only the two most recent frames; discard older ones
            if len(self._queue) > 2:
                self._queue.pop(0)

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            job = None
            with self._lock:
                if self._queue:
                    job = self._queue.pop()     # process newest first
            if job is None:
                time.sleep(0.02)
                continue
            frame_bgr, objects = job
            self._process(frame_bgr, objects)

    def _process(self, frame_bgr: np.ndarray, objects: list[dict]):
        try:
            now = time.time()
            to_measure = []

            for obj in objects:
                track = self._tracker.get_track(obj["object_id"])
                if track is None:
                    continue
                if (now - track.last_measured_at) < MEASURE_INTERVAL:
                    continue

                # ── CRITICAL FIX: stamp the timestamp NOW, before the HTTP
                # request. If the server returns an empty list (depth model
                # loading, mask too sparse, etc.) last_measured_at would
                # otherwise stay at 0 and the worker would retry every single
                # frame, flooding the server and the log.
                self._tracker.stamp_measured(obj["object_id"], now)

                h, w = frame_bgr.shape[:2]
                mask_np = F.interpolate(
                    obj["mask"].unsqueeze(0).unsqueeze(0).float(),
                    size=(h, w), mode="nearest"
                ).squeeze().cpu().numpy()
                binary = (mask_np > 0.5).astype(np.uint8)

                pixel_count = int(binary.sum())
                print(f"[Worker] queuing obj {obj['object_id']} ({obj['class']}) "
                      f"mask_px={pixel_count}")

                to_measure.append({
                    "object_id":          obj["object_id"],
                    "class":              obj["class"],
                    "confidence":         obj["confidence"],
                    "segmentation_score": obj["segmentation_score"],
                    "mask_stability":     obj["mask_stability"],
                    "mask_rle":           encode_mask_rle(binary),
                })

            if not to_measure:
                return

            _, jpg = cv.imencode(".jpg", frame_bgr,
                                 [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            frame_b64 = base64.b64encode(jpg.tobytes()).decode()

            payload  = {"frame_b64": frame_b64, "objects": to_measure}
            response = requests.post(f"{DEPTH_SERVER_URL}/measure",
                                     json=payload, timeout=20.0)

            print(f"[Worker] server status={response.status_code}")

            if response.status_code == 200:
                result  = response.json()
                returned = result.get("objects", [])
                print(f"[Worker] received {len(returned)} measurements "
                      f"in {result.get('inference_ms', '?')} ms")

                if not returned:
                    # Server OK but returned nothing — print full body for debug
                    print(f"[Worker] WARNING empty response body: {response.text[:600]}")

                for m in returned:
                    self._tracker.set_measurement(m["object_id"], m)
                    print(f"[Worker] obj {m['object_id']} → "
                          f"L={m['length']:.2f} W={m['width']:.2f} "
                          f"H={m['height']:.2f} conf={m['confidence']:.2f}")
            else:
                print(f"[Worker] server error {response.status_code}: "
                      f"{response.text[:400]}")

        except requests.exceptions.Timeout:
            print("[Worker] request timed out — is the depth server still "
                  "loading the model?")
        except Exception as e:
            print(f"[Worker] unhandled exception: {e}")
            tb.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def score_color(score: float):
    return (0, int(score * 255), int((1 - score) * 255))


def draw_inventory(frame: np.ndarray, inventory: list[dict],
                   frame_h: int, frame_w: int):
    overlay = frame.copy()

    for obj in inventory:
        mask_np = F.interpolate(
            obj["mask"].unsqueeze(0).unsqueeze(0).float(),
            size=(frame_h, frame_w), mode="nearest"
        ).squeeze().cpu().numpy()
        binary = (mask_np > 0.5).astype(np.uint8)
        color  = score_color(obj["segmentation_score"])

        colored = np.zeros_like(frame)
        colored[binary == 1] = color
        cv.addWeighted(colored, 0.45, overlay, 0.55, 0, overlay)

        x1, y1, x2, y2 = map(int, obj["bounding_box"])
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        m = obj.get("measurement")

        lines = [
            f"ID:{obj['object_id']} {obj['class']}",
            f"seg:{obj['segmentation_score']:.2f}  tc:{obj['temporal_consistency']:.2f}",
        ]
        if m:
            lines += [
                f"L:{m['length']:.2f}m  W:{m['width']:.2f}m  H:{m['height']:.2f}m",
                f"Vol:{m['volume_m3']:.3f}m3  conf:{m['confidence']:.2f}",
            ]
        else:
            lines.append("measuring...")

        for i, line in enumerate(lines):
            y_pos = max(y1 - 8 - (len(lines) - 1 - i) * 17, 14)
            cv.putText(frame, line, (x1 + 1, y_pos + 1),
                       cv.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv.LINE_AA)
            cv.putText(frame, line, (x1, y_pos),
                       cv.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv.LINE_AA)

    cv.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    try:
        r    = requests.get(f"{DEPTH_SERVER_URL}/health", timeout=3.0)
        info = r.json()
        print(f"[main] depth server OK  model={info['model']}  device={info['device']}")
    except Exception:
        print("[main] WARNING: depth server not reachable — dimensions will be missing.")

    model  = YOLO("yolo11n-seg.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tracker = FurnitureTracker()
    worker  = MeasurementWorker(tracker)
    worker.start()

    fps_counter = 0
    fps_start   = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        fps_counter += 1

        results    = model(frame, verbose=False)
        detections = []

        if results[0].masks is not None:
            for box, raw_mask in zip(results[0].boxes, results[0].masks.data):
                class_id   = int(box.cls[0])
                class_name = model.names[class_id]
                conf       = float(box.conf[0])
                if class_name in ALLOWED_CLASSES and conf >= CONF_THRESHOLD:
                    detections.append({
                        "class_name":   LABEL_MAP.get(class_name, class_name),
                        "confidence":   conf,
                        "bounding_box": box.xyxy[0].cpu().numpy(),
                        "mask_tensor":  raw_mask,
                    })

        inventory = tracker.update(detections)

        if inventory:
            worker.submit(frame, inventory)

        draw_inventory(frame, inventory, frame_h, frame_w)

        obj_count = len(inventory)
        cv.putText(frame, f"Objects: {obj_count}", (10, 24),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv.imshow("ParcelVision", frame)

        if time.perf_counter() - fps_start >= 1.0:
            print(f"FPS: {fps_counter:2d} | Tracked: {obj_count}")
            for obj in inventory:
                m = obj.get("measurement")
                dim = (f"L={m['length']:.2f} W={m['width']:.2f} "
                       f"H={m['height']:.2f} conf={m['confidence']:.2f}"
                       if m else "no measurement yet")
                print(f"  [{obj['object_id']}] {obj['class']:6s} "
                      f"seg={obj['segmentation_score']:.2f}  {dim}")
            fps_counter = 0
            fps_start   = time.perf_counter()

        if cv.waitKey(1) == ord("q"):
            break

    worker.stop()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()