import cv2 as cv
import time
import numpy as np
import torch
from ultralytics import YOLO
from dataclasses import dataclass, field

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
MAX_LOST_FRAMES = 10
SMOOTHING_ALPHA = 0.6
ALLOWED_CLASSES = {"chair", "couch", "dining table"}
LABEL_MAP = {"couch": "Sofa", "dining table": "Table", "chair": "Chair"}
SCORE_W_CONF = 0.40
SCORE_W_TEMPORAL = 0.30
SCORE_W_STABILITY = 0.30

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

def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)

class FurnitureTracker:
    def __init__(self):
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        matched_track_ids, matched_det_idxs = self._match(detections)

        for tid, di in zip(matched_track_ids, matched_det_idxs):
            self._update_track(self._tracks[tid], detections[di])

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_idxs]
        for di in unmatched_dets:
            self._create_track(detections[di])

        matched_tids = set(matched_track_ids)
        stale = [tid for tid in self._tracks if tid not in matched_tids]
        for tid in stale:
            self._tracks[tid].frames_lost += 1
            
        self._tracks = {tid: t for tid, t in self._tracks.items() if t.frames_lost <= MAX_LOST_FRAMES}
        return self._build_inventory()

    def _match(self, detections):
        if not self._tracks or not detections:
            return [], []

        track_ids = list(self._tracks.keys())
        track_boxes = [self._tracks[tid].bbox for tid in track_ids]
        det_boxes = [d["bounding_box"] for d in detections]

        iou_matrix = np.zeros((len(track_ids), len(detections)))
        for ti, tb in enumerate(track_boxes):
            for di, db in enumerate(det_boxes):
                iou_matrix[ti, di] = bbox_iou(tb, db)

        matched_tids, matched_dets = [], []
        used_dets = set()
        order = np.dstack(np.unravel_index(np.argsort(iou_matrix, axis=None)[::-1], iou_matrix.shape))[0]

        for ti, di in order:
            if iou_matrix[ti, di] < IOU_THRESHOLD:
                break
            tid = track_ids[ti]
            if tid in matched_tids or di in used_dets:
                continue
            matched_tids.append(tid)
            matched_dets.append(di)
            used_dets.add(di)

        return matched_tids, matched_dets

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

        new_mask = det["mask_tensor"].float()
        old_mask = t.mask_tensor.float()
        if new_mask.shape != old_mask.shape:
            new_mask = torch.nn.functional.interpolate(
                new_mask.unsqueeze(0).unsqueeze(0), size=old_mask.shape, mode="nearest"
            ).squeeze()
        t.mask_tensor = (SMOOTHING_ALPHA * new_mask + (1 - SMOOTHING_ALPHA) * old_mask).clamp(0, 1)

        t.conf_history.append(det["confidence"])
        t.area_history.append(float(new_mask.sum()))

        window = 30
        if len(t.conf_history) > window:
            t.conf_history = t.conf_history[-window:]
            t.area_history = t.area_history[-window:]

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
            SCORE_W_CONF * avg_conf +
            SCORE_W_TEMPORAL * t.temporal_consistency +
            SCORE_W_STABILITY * t.mask_stability
        )

    def _build_inventory(self) -> list[dict]:
        return [{
            "object_id": t.track_id,
            "class": t.class_name,
            "confidence": round(t.confidence, 4),
            "segmentation_score": round(t.segmentation_score, 4),
            "bounding_box": t.bbox,
            "mask": t.mask_tensor,
            "frames_seen": t.frames_seen,
            "temporal_consistency": round(t.temporal_consistency, 4),
            "mask_stability": round(t.mask_stability, 4),
        } for t in self._tracks.values()]


def score_color(score: float):
    return (0, int(score * 255), int((1 - score) * 255))

def draw_inventory(frame: np.ndarray, inventory: list[dict], model_names, frame_h, frame_w):
    overlay = frame.copy()
    for obj in inventory:
        mask_np = torch.nn.functional.interpolate(
            obj["mask"].unsqueeze(0).unsqueeze(0).float(), size=(frame_h, frame_w), mode="nearest"
        ).squeeze().cpu().numpy()
        binary = (mask_np > 0.5).astype(np.uint8)

        color = score_color(obj["segmentation_score"])
        colored = np.zeros_like(frame)
        colored[binary == 1] = color
        cv.addWeighted(colored, 0.45, overlay, 0.55, 0, overlay)

        x1, y1, x2, y2 = map(int, obj["bounding_box"])
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        lines = [
            f"ID:{obj['object_id']} {obj['class']}",
            f"conf:{obj['confidence']:.2f} seg:{obj['segmentation_score']:.2f}",
            f"frames:{obj['frames_seen']} tc:{obj['temporal_consistency']:.2f}",
        ]
        for i, line in enumerate(lines):
            y_pos = max(y1 - 10 - (len(lines) - 1 - i) * 16, 12)
            cv.putText(frame, line, (x1, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv.LINE_AA)

    cv.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    model = YOLO("yolo11n-seg.pt")
    model.to("cuda")
    tracker = FurnitureTracker()

    warmup_frames, measure_frames = 10, 30
    frame_count, total_inf_time, fps_counter = 0, 0.0, 0
    fps_start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        fps_counter += 1
        frame_count += 1

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model(frame, verbose=False)
        torch.cuda.synchronize()
        inf_ms = (time.perf_counter() - t0) * 1000

        if warmup_frames < frame_count <= (warmup_frames + measure_frames):
            total_inf_time += inf_ms
        if frame_count == (warmup_frames + measure_frames):
            avg = total_inf_time / measure_frames
            print(f"[Benchmark] Avg inference: {avg:.2f} ms | FPS cap: {1000/avg:.1f}")

        detections = []
        if results[0].masks is not None:
            for box, raw_mask in zip(results[0].boxes, results[0].masks.data):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf = float(box.conf[0])

                if class_name in ALLOWED_CLASSES and conf >= CONF_THRESHOLD:
                    detections.append({
                        "class_name": LABEL_MAP.get(class_name, class_name),
                        "confidence": conf,
                        "bounding_box": box.xyxy[0].cpu().numpy(),
                        "mask_tensor": raw_mask,
                    })

        object_inventory = tracker.update(detections)
        draw_inventory(frame, object_inventory, model.names, frame_h, frame_w)

        cv.putText(frame, f"Objects: {len(object_inventory)}", (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv.imshow("ParcelVision - SegMode", frame)

        if time.perf_counter() - fps_start_time >= 1.0:
            print(f"FPS: {fps_counter:2d} | Tracked objects: {len(object_inventory)}")
            for obj in object_inventory:
                print(f"  [{obj['object_id']}] {obj['class']:6s} seg={obj['segmentation_score']:.2f} tc={obj['temporal_consistency']:.2f} stab={obj['mask_stability']:.2f} frames={obj['frames_seen']}")
            fps_counter = 0
            fps_start_time = time.perf_counter()

        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()