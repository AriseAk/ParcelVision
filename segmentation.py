import cv2 as cv
import time
import numpy as np
import torch
from ultralytics import YOLO

def main():
    # Try index 0, 1, or 2 if 0 fails
    cap = cv.VideoCapture(2)
    
    if not cap.isOpened():
        print("Error: Could not open video device. Try changing the index in cv.VideoCapture(0)")
        return

    # Using YOLOv8 or v11 Segmentation
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_counter += 1
        torch.cuda.synchronize() 
        start = time.perf_counter()
        
        # Run inference
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
        
        # 1. Handle Segmentation Masks
        if result.masks is not None:
            for mask, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf = float(box.conf[0])

                if class_name not in ALLOWED_CLASSES or conf < CONF_THRESHOLD:
                    continue

                # Create a semi-transparent overlay for the mask
                mask_points = np.array(mask, dtype=np.int32)
                overlay = frame.copy()
                
                # Pick a color (BGR): Green for furniture
                cv.fillPoly(overlay, [mask_points], (0, 255, 0))
                
                # Blend the overlay with the original frame (alpha=0.4)
                cv.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

                # 2. Handle Bounding Boxes & Labels
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Business Logic Labeling
                label_map = {"couch": "Sofa", "dining table": "Table", "chair": "Chair"}
                business_label = label_map.get(class_name, class_name)
                
                display_text = f"{business_label} {conf*100:.1f}%"
                
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, display_text, (x1, max(y1 - 10, 0)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv.imshow("ParcelVision - SegMode", frame)

        # FPS calculation for the full loop
        if time.perf_counter() - fps_start_time >= 1.0:
            print(f"Full pipeline FPS: {fps_counter}")
            fps_counter = 0
            fps_start_time = time.perf_counter()
        
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()