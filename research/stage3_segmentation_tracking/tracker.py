import numpy as np

IOU_THRESHOLD = 0.5
MAX_LOST_FRAMES = 10
HISTORY_LENGTH = 30
MIN_STABLE_FRAMES = 5


def compute_iou(boxA, boxB):

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)

    interArea = interWidth * interHeight
    union = areaA + areaB - interArea

    if union == 0:
        return 0

    return interArea / union


class ObjectTracker:

    def __init__(self):

        self.tracks = {}
        self.next_id = 0

    def update(self, detections):

        matched_tracks = set()

        for detection in detections:

            best_iou = 0
            best_track_id = None

            for track_id, track in self.tracks.items():

                # prevent one track being matched multiple times
                if track_id in matched_tracks:
                    continue

                iou = compute_iou(detection["bbox"], track["bbox"])

                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_iou > IOU_THRESHOLD and best_track_id is not None:

                track = self.tracks[best_track_id]

                track["bbox"] = detection["bbox"]
                track["confidence"] = detection["confidence"]
                track["frames_seen"] += 1
                track["frames_lost"] = 0

                # Update histories
                track["confidence_history"].append(detection["confidence"])
                track["area_history"].append(detection["area"])
                track["center_history"].append(detection["center"])

                # Keep history window small
                track["confidence_history"] = track["confidence_history"][-HISTORY_LENGTH:]
                track["area_history"] = track["area_history"][-HISTORY_LENGTH:]
                track["center_history"] = track["center_history"][-HISTORY_LENGTH:]

                # Compute stability score
                track["stability_score"] = self.compute_stability(track)

                detection["id"] = best_track_id
                detection["stability_score"] = track["stability_score"]

                matched_tracks.add(best_track_id)

            else:

                new_id = self.next_id
                self.next_id += 1

                self.tracks[new_id] = {

                    "bbox": detection["bbox"],
                    "confidence": detection["confidence"],
                    "frames_seen": 1,
                    "frames_lost": 0,

                    "confidence_history": [detection["confidence"]],
                    "area_history": [detection["area"]],
                    "center_history": [detection["center"]],

                    "stability_score": 0
                }

                detection["id"] = new_id
                detection["stability_score"] = 0

                matched_tracks.add(new_id)

        to_delete = []

        for track_id, track in self.tracks.items():

            if track_id not in matched_tracks:
                track["frames_lost"] += 1

            if track["frames_lost"] > MAX_LOST_FRAMES:
                to_delete.append(track_id)

        for tid in to_delete:
            del self.tracks[tid]

        # Filter stable detections
        stable_detections = []

        for det in detections:

            track = self.tracks.get(det["id"], None)

            if track is None:
                continue

            if track["frames_seen"] >= MIN_STABLE_FRAMES:
                stable_detections.append(det)

        return stable_detections

    def compute_stability(self, track):

        conf_hist = np.array(track["confidence_history"])
        area_hist = np.array(track["area_history"])
        centers = np.array(track["center_history"])

        conf_mean = conf_hist.mean()

        area_std = area_hist.std() / (area_hist.mean() + 1e-6)

        if len(centers) > 1:
            center_std = np.linalg.norm(centers.std(axis=0))
        else:
            center_std = 0

        temporal_consistency = track["frames_seen"] / (
            track["frames_seen"] + track["frames_lost"] + 1e-6
        )

        stability_score = (
            0.4 * conf_mean
            + 0.3 * temporal_consistency
            + 0.3 * (1 - min(area_std + center_std, 1))
        )

        return float(stability_score)