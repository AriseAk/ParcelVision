import cv2 as cv
import time
from ultralytics import YOLO
import torch
import numpy as np

from tracker import ObjectTracker
from scene_state import SceneStateManager


def build_scene_inventory(detections):

    scene_inventory = []

    for det in detections:

        record = {
            "object_id": det["id"],
            "label": det["business_class"],
            "confidence": float(det["confidence"]),
            "bbox": det["bbox"],
            "center": det["center"],
            "area": det["area"],
            "stability_score": float(det.get("stability_score", 0))
        }

        scene_inventory.append(record)

    return scene_inventory


def main():

    cap = cv.VideoCapture(0)

    model = YOLO("yolov8s-world.pt")
    model.to("cuda")

    # Open-vocabulary prompts
    model.set_classes([
        "sofa",
        "chair",
        "table",
        "cardboard box",
        "carton box",
        "shipping box",
        "package"
    ])

    tracker = ObjectTracker()
    scene_manager = SceneStateManager()

    frame_index = 0

    warmup_frames = 10
    measure_frames = 30

    frame_count = 0
    total_inference_time = 0

    fps_counter = 0
    fps_start_time = time.perf_counter()
    fps_display = "FPS: 0"

    CONF_THRESHOLD = 0.3

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        fps_counter += 1

        torch.cuda.synchronize()
        start = time.perf_counter()

        results = model(frame, verbose=False)

        torch.cuda.synchronize()
        end = time.perf_counter()

        inference_ms = (end - start) * 1000
        frame_count += 1

        if warmup_frames < frame_count <= (warmup_frames + measure_frames):
            total_inference_time += inference_ms

        if frame_count == (warmup_frames + measure_frames):
            average = total_inference_time / measure_frames
            print(f"Average inference time: {average:.2f} ms")
            print(f"Estimated FPS: {1000 / average:.2f}")

        result = results[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        current_frame_detections = []

        for box, cls, conf in zip(boxes, classes, confs):

            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)

            class_id = int(cls)
            class_name = model.names[class_id]

            width = x2 - x1
            height = y2 - y1
            area = width * height

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detection = {
                "id": None,
                "model_class": class_name,
                "business_class": class_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "area": area
            }

            current_frame_detections.append(detection)

        # Tracker update
        current_frame_detections = tracker.update(current_frame_detections)

        # Scene Inventory
        scene_inventory = build_scene_inventory(current_frame_detections)

        # Scene State update
        scene_state = scene_manager.update(scene_inventory, frame_index)
        frame_index += 1

        # Visualization
        for obj in scene_inventory:

            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label"]
            conf = obj["confidence"]
            stability = obj["stability_score"]

            text = f"ID {obj['object_id']} | {label} {conf:.2f} | S:{stability:.2f}"

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv.putText(
                frame,
                text,
                (x1, max(y1 - 10, 0)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

        # FPS display
        if time.perf_counter() - fps_start_time >= 1.0:

            fps_display = f"FPS: {fps_counter}"
            fps_counter = 0
            fps_start_time = time.perf_counter()

            # Debug output
            print("Scene Inventory:", scene_inventory)

        cv.putText(frame, fps_display,
                   (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1,
                   (0, 0, 0),
                   2)

        cv.imshow("ParcelVision", frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()