from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
import torch
import requests
from ultralytics import YOLO

from tracker import ObjectTracker
from scene_state import SceneStateManager
from depth_fusion import DepthFusion
from volume_estimator import VolumeEstimator


# ------------------------------------------------
# Flask Setup
# ------------------------------------------------

app = Flask(__name__)
CORS(app)


# ------------------------------------------------
# Load Models / Modules
# ------------------------------------------------

print("Loading YOLO model...")

model = YOLO("yolov8s-world.pt")
model.to("cuda")

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

depth_fusion = DepthFusion()
volume_estimator = VolumeEstimator()

frame_index = 0
CONF_THRESHOLD = 0.3

torch.set_grad_enabled(False)

print("Model loaded successfully")


# ------------------------------------------------
# Routes
# ------------------------------------------------

@app.route("/")
def home():
    return {"message": "ParcelVision backend running"}


# ------------------------------------------------
# Detection Route
# ------------------------------------------------

@app.route("/detect", methods=["POST"])
def detect():

    global frame_index

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    # Decode image
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    orig_h, orig_w = frame.shape[:2]

    # Resize for faster inference
    frame_resized = cv2.resize(frame, (416, 320))

    # Run YOLO
    results = model(frame_resized, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    detections = []

    scale_x = orig_w / 416
    scale_y = orig_h / 320

    for box, cls, conf in zip(boxes, classes, confs):

        if conf < CONF_THRESHOLD:
            continue

        # Rescale bounding boxes
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        class_id = int(cls)
        label = model.names[class_id]

        width = x2 - x1
        height = y2 - y1
        area = width * height

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": float(conf),
            "center": (cx, cy),
            "area": area,
            "business_class": label
        })


    # ------------------------------------------------
    # Tracker
    # ------------------------------------------------

    detections = tracker.update(detections)


    # ------------------------------------------------
    # Frame Inventory
    # ------------------------------------------------

    scene_inventory = []

    for det in detections:

        scene_inventory.append({
            "object_id": det["id"],
            "label": det["business_class"],
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "center": det["center"],
            "area": det["area"],
            "stability_score": det.get("stability_score", 0)
        })


    # ------------------------------------------------
    # Scene State Update
    # ------------------------------------------------

    scene_state = scene_manager.update(scene_inventory, frame_index)
    frame_index += 1


    # ------------------------------------------------
    # Depth Fusion
    # ------------------------------------------------

    points = []

    for obj in scene_state.values():
        cx, cy = obj["center"]
        points.append([cx, cy])


    depths = []

    try:

        depth_response = requests.post(
            "http://localhost:6000/depth",
            json={"points": points}
        )

        depths = depth_response.json()["depths"]

    except:
        depths = [None] * len(points)


    # ------------------------------------------------
    # Compute 3D position + volume
    # ------------------------------------------------

    for obj, depth in zip(scene_state.values(), depths):

        if depth is None:
            continue

        world_pos = depth_fusion.pixel_to_world(obj["center"], depth)

        dims = depth_fusion.bbox_to_dimensions(obj["bbox"], depth)

        volume = volume_estimator.compute_volume(
            obj["label"],
            dims["width"],
            dims["height"]
        )

        obj["world_position"] = world_pos
        obj["dimensions"] = volume


    return jsonify({
        "scene": list(scene_state.values())
    })


# ------------------------------------------------
# Scene Route
# ------------------------------------------------

@app.route("/scene", methods=["GET"])
def get_scene():

    return jsonify({
        "scene": list(scene_manager.objects.values())
    })


# ------------------------------------------------
# Run Server
# ------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)