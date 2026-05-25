from ultralytics import YOLO
from pathlib import Path

IMAGE_DIR = Path("data/images")
CONF_THRESHOLD = 0.5
ALLOWED_CLASSES = {"chair", "couch", "dining table"}

model = YOLO("yolov8n.pt")

image_list = list(IMAGE_DIR.glob("*.jpg"))
results = model(image_list, device="cuda")
all_outputs = {}

for i, result in enumerate(results):
    image_name = image_list[i].name
    detections_for_image = []
    if len(result.boxes) == 0:
        all_outputs[image_name] = []
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
    all_outputs[image_name] = detections_for_image

for image, detections in all_outputs.items():
    print(f"\nImage: {image}")
    print(detections)
