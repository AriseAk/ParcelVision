import cv2 as cv
from ultralytics import YOLO

all_outputs = {}

def main():
    cap=cv.VideoCapture(0)
    model = YOLO("yolov8n.pt")
    CONF_THRESHOLD = 0.5
    ALLOWED_CLASSES = {"chair", "couch", "dining table"}
    while True:
        ret,frame=cap.read()
        if not ret:
            print("Failed")
            break
        
        results = model(frame, device="cuda")

        for i, result in enumerate(results):
            detections_for_image = []
            if len(result.boxes) == 0:
                all_outputs[frame.item] = []
                continue
            for j in range(len(result.boxes)):
                class_id = int(result.boxes.cls[j].item())
                confidence = result.boxes.conf[j].item()
                bbox = result.boxes.xyxy[j].tolist()
                class_name = model.names[class_id]
                if class_name not in ALLOWED_CLASSES:
                    continue
                if confidence < CONF_THRESHOLD:
                    continue
                if class_name == "couch":
                    business_label = "sofa"
                elif class_name == "dining table":
                    business_label = "table"
                else:
                    business_label = class_name
                detection_dict = {
                    "class": business_label,
                    "bbox": bbox,
                    "confidence": round(confidence, 2)
                }
                detections_for_image.append(detection_dict)
            all_outputs[frame.item] = detections_for_image

        cv.imshow("Object Detection",frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    print(all_outputs)