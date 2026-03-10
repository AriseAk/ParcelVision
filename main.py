import cv2 as cv
import time
from ultralytics import YOLO
import torch
import numpy as np

CLASS_MAPPING = {

    "couch": {
        "business_name": "sofa",
        "density_group": "upholstered",
        "fragile": False
    },

    "chair": {
        "business_name": "chair",
        "density_group": "wood_light",
        "fragile": False
    },

    "dining table": {
        "business_name": "table",
        "density_group": "wood_heavy",
        "fragile": False
    }

}

def compute_iou(boxA, boxB):
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    areaI = interWidth * interHeight
    areaU=float(areaA + areaB - areaI)

    if areaU==0:
        return 0
    
    iou = areaI / areaU
    return iou

def main():
    cap = cv.VideoCapture(0)
    model = YOLO("yolov8n.pt")
    model.to("cuda")
    warmup_frames = 10
    measure_frames = 30
    
    frame_count = 0
    total_inference_time = 0

    fps_counter = 0
    fps_start_time = time.perf_counter()
    fps_display = "FPS: 0"

    CONF_THRESHOLD = 0.3

    stability_log = []
    object_tracks={}
    object_metrics={}
    frame_index = 0
    previous_detections = []
    next_object_id = 0
    MAX_LOG_FRAMES = 200

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

        result=results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        current_frame_detections = []

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int,box)
            class_id = int(cls)
            class_name = model.names[int(class_id)]
            if conf < CONF_THRESHOLD:
                continue
            if class_name not in CLASS_MAPPING:
                continue
            metadata = CLASS_MAPPING[class_name]
            business_label = metadata["business_name"]

            width=x2-x1
            length=y2-y1
            area=width*length

            cx=(x1+x2)/2
            cy=(y1+y2)/2

            detection = {
                "id": None,
                "model_class": class_name,
                "business_class": business_label,
                "density_group": metadata["density_group"],
                "fragile": metadata["fragile"],
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "area": area
            }

            current_frame_detections.append(detection)
        
        for current_detection in current_frame_detections:
            best_iou = 0
            best_match_id = None
            for prev_detection in previous_detections:
                iou = compute_iou(current_detection["bbox"], prev_detection["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = prev_detection["id"]

            if best_iou > 0.5 and best_match_id != None:
                current_detection["id"]=best_match_id
            else:
                current_detection["id"]=next_object_id
                next_object_id+=1

        previous_detections = current_frame_detections.copy()

        for detection in current_frame_detections:
            x1,y1,x2,y2 = detection["bbox"]
            business_label=detection["business_class"]
            conf=detection["confidence"]
            label = f"ID {detection['id']}|{business_label.capitalize()} {conf*100:.2f}%"
            text_position = (x1, max(y1 - 10, 0))
            cv.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0), 
                2)
            cv.putText(
                frame,
                label,
                text_position,
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2)

        if frame_index < MAX_LOG_FRAMES:

            for detection_index, detection in enumerate(current_frame_detections):
                record = {
                    "frame": frame_index,
                    "temp_id": detection_index,
                    "confidence": detection["confidence"],
                    "area": detection["area"],
                    "center_x": detection["center"][0],
                    "center_y": detection["center"][1]
                }

                stability_log.append(record)

        frame_index += 1

        if frame_index == MAX_LOG_FRAMES:
            for record in stability_log:
                t_id = record["temp_id"]
                if t_id not in object_tracks:
                    object_tracks[t_id] = []
                object_tracks[t_id].append(record)

            for key,history in object_tracks.items():
                confidence_series = [r["confidence"] for r in history]
                center_x_series = [r["center_x"] for r in history]
                center_y_series = [r["center_y"] for r in history]
                area_series = [r["area"] for r in history]

                std_confidence=np.std(confidence_series)
                std_center_x=np.std(center_x_series)
                std_center_y=np.std(center_y_series)
                std_area=np.std(area_series)

                rec={
                    "temp_id":key,
                    "std_confidence":std_confidence,
                    "std_center_x":std_center_x,
                    "std_center_y":std_center_y,
                    "std_area":std_area
                }

                object_metrics[key]=rec

            print(object_metrics)

        if time.perf_counter() - fps_start_time >= 1.0:
            fps_display = f"FPS: {fps_counter}"
            fps_counter = 0
            fps_start_time = time.perf_counter()
        
        cv.putText(frame, fps_display, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv.imshow("ParcelVision", frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()