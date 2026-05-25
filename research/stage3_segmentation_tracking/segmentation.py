import cv2 as cv
import time
import numpy as np
import torch
from ultralytics import YOLO

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        return

    model = YOLO("yolo11n-seg.pt")
    model.to("cuda")

    warmup_frames, measure_frames, frame_count = 10, 30, 0
    total_inference_time = 0
    fps_counter, fps_start_time = 0, time.perf_counter()

    CONF_THRESHOLD = 0.3
    ALLOWED_CLASSES = {"chair", "couch", "dining table"}
    label_map = {"couch": "Sofa", "dining table": "Table", "chair": "Chair"}

    while True:
        ret, frame = cap.read()
        if not ret: break

        fps_counter += 1
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        results = model(frame, verbose=False)
        
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - start) * 1000
        frame_count += 1

        if warmup_frames < frame_count <= (warmup_frames + measure_frames):
            total_inference_time += inference_ms
        if frame_count == (warmup_frames + measure_frames):
            avg = total_inference_time / measure_frames
            print(f"Avg Inference: {avg:.2f} ms | FPS: {1000/avg:.2f}")

        result = results[0]
        detected_objects = []

        if result.masks is not None:
            overlay = frame.copy()
            
            for mask_polygon, box, raw_mask in zip(result.masks.xy, result.boxes, result.masks.data):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf = float(box.conf[0])

                if class_name not in ALLOWED_CLASSES or conf < CONF_THRESHOLD:
                    continue
                
                obj_record = {
                    "class_name": label_map.get(class_name, class_name),
                    "confidence": conf,
                    "bounding_box": box.xyxy[0].cpu().numpy(), 
                    "mask_tensor": raw_mask 
                }
                detected_objects.append(obj_record)

                points = np.array(mask_polygon, dtype=np.int32)
                cv.fillPoly(overlay, [points], (0, 255, 0))

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                display_label = f"{obj_record['class_name']} {conf*100:.1f}%"
                
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, display_label, (x1, max(y1 - 10, 0)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv.imshow("ParcelVision - SegMode", frame)

        if time.perf_counter() - fps_start_time >= 1.0:
            print(f"Full pipeline FPS: {fps_counter} | Objects in frame: {len(detected_objects)}")
            fps_counter, fps_start_time = 0, time.perf_counter()

        if cv.waitKey(1) == ord('q'): break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()