import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        return
    model = YOLO("yolo11n-seg.pt") 
    model.to("cuda")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        result = results[0]
        if result.masks is not None:
            mask_data = result.masks.data.cpu()
            print(f"type: {type(result.masks)}")
            print(f"shape: {mask_data.shape}")
            print(f"min: {torch.min(mask_data)}")
            print(f"max: {torch.max(mask_data)}")
            print(f"count: {len(mask_data)}")
            break
    cap.release()

if __name__ == "__main__":
    main()