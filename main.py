import cv2 as cv
import time
from ultralytics import YOLO
import torch

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

        result=results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        detections=[]

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
            label = f"{business_label.capitalize()} {conf*100:.2f}%"
            text_position = (x1, max(y1 - 10, 0))

            item={
                "class_name":class_name,
                "business_label":business_label.capitalize(),
                "confidence":conf,
                "density_group":metadata["density_group"],
                "fragile":metadata["fragile"],
                "bounding_box":[x1,y1,x2,y2]
            }

            detections.append(item)

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

        cv.imshow("ParcelVision", frame)

        if time.perf_counter() - fps_start_time >= 1.0:
            print("Full pipeline FPS:", fps_counter)
            fps_counter = 0
            fps_start_time = time.perf_counter()
        
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()