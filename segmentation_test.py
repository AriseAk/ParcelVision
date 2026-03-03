import cv2 as cv
import time
import numpy as np
import torch
from ultralytics import YOLO

def main():
    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    cv.namedWindow("ParcelVision - SegMode", cv.WINDOW_NORMAL)
    cv.startWindowThread()

    model = YOLO("yolo11n-seg.pt") 
    model.to("cuda")
    
    warmup_frames = 10
    measure_frames = 30
    frame_count = 0
    total_inference_time = 0
    fps_counter = 0
    fps_start_time = time.perf_counter()

    CONF_THRESHOLD = 0.3
    ALLOWED_CLASSES = {"chair", "couch", "dining table"}
    label_map = {"couch": "Sofa", "dining table": "Table", "chair": "Chair"}

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
            print(f"--- Benchmark Complete ---")
            print(f"Avg Inference: {average:.2f} ms | Est. FPS: {1000/average:.2f}")

        result = results[0]
        if result.masks is not None:
            overlay = frame.copy()
            for mask, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf = float(box.conf[0])

                if class_name not in ALLOWED_CLASSES or conf < CONF_THRESHOLD:
                    continue

                mask_points = np.array(mask, dtype=np.int32)
                cv.fillPoly(overlay, [mask_points], (0, 255, 0))

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                business_label = label_map.get(class_name, class_name)
                display_text = f"{business_label} {conf*100:.1f}%"
                
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, display_text, (x1, max(y1 - 10, 0)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv.imshow("ParcelVision - SegMode", frame)

        if time.perf_counter() - fps_start_time >= 1.0:
            print(f"Full pipeline FPS: {fps_counter}")
            fps_counter = 0
            fps_start_time = time.perf_counter()
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    for i in range(5): cv.waitKey(1)

if __name__ == "__main__":
    main()