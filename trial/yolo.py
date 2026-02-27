# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

from ultralytics import YOLO
from pathlib import Path

# image_path="data/images/chair1.jpg"
image_dir = Path("data/images")
image_list=list(image_dir.glob("*.jpg"))
# print(image_list)
model = YOLO("yolov8n.pt")
results = model(image_list, device="cuda")

# for result in results:
# result=results[0]

#     print("Type of results:", type(results))
#     print("Number of results:", len(results))

#     print("Type of first result:", type(result))

#     print("Boxes object:", result.boxes)
#     print("Number of detections:", len(result.boxes))

#     print("Shape of boxes.xyxy:", result.boxes.xyxy.shape)
#     print("Shape of boxes.conf:", result.boxes.conf.shape)
#     print("Shape of boxes.cls:", result.boxes.cls.shape)

#     print("\nRaw xyxy tensor:\n", result.boxes.xyxy)
#     print("\nRaw confidence tensor:\n", result.boxes.conf)
#     print("\nRaw class tensor:\n", result.boxes.cls)

# print("\nClass names mapping:", model.names)

for i in range(len(results)):
    result = results[i]
    print(f"\nImage: {image_list[i].name}")
    if len(result.boxes) == 0:
        print("No detections")
        continue
    for j in range(len(result.boxes)):
        class_id = int(result.boxes.cls[j].item())
        if class_id in [56, 57, 60]:
            confidence = result.boxes.conf[j].item()
            bbox = result.boxes.xyxy[j].tolist()       
            class_name = model.names[class_id]
            print(f"Detected: {class_name}")
            print(f"Confidence: {confidence:.2f}")
            print(f"BBox: {bbox}")
            print("-" * 30)