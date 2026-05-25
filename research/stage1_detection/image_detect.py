from __future__ import annotations

import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# SAM
from segment_anything import sam_model_registry, SamPredictor

# ─────────────────────────────────────────────────────────────
# DepthAnything
# ─────────────────────────────────────────────────────────────

sys.path.insert(0, "./Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

_device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.15

CAMERA_FX = 525.0
CAMERA_FY = 525.0
CAMERA_CX = 320.0
CAMERA_CY = 240.0
DEPTH_SCALE = 4.0

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────

print("Loading YOLO World...")
yolo_model = YOLO("yolov8s-world.pt")
yolo_model.to(_device)

yolo_model.set_classes([
    "box",
    "parcel",
    "package",
])

print("Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(_device)
sam_predictor = SamPredictor(sam)

print("Loading DepthAnything...")
depth_model = DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384],
)

depth_model.load_state_dict(
    torch.load("depth_anything_v2_vits.pth", map_location=_device)
)
depth_model.to(_device).eval()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def infer_depth(frame):
    depth = depth_model.infer_image(frame)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth * DEPTH_SCALE


def pixels_to_3d(us, vs, zs):
    X = (us - CAMERA_CX) * zs / CAMERA_FX
    Y = (vs - CAMERA_CY) * zs / CAMERA_FY
    return np.stack([X, Y, zs], axis=-1)


def measure(depth_map, mask):
    vs, us = np.where(mask > 0.5)

    if len(us) < 20:
        return None

    zs = depth_map[vs, us]
    pts = pixels_to_3d(us, vs, zs)

    l = pts[:, 0].max() - pts[:, 0].min()
    w = pts[:, 2].max() - pts[:, 2].min()
    h = pts[:, 1].max() - pts[:, 1].min()

    return {
        "length": float(l),
        "width": float(w),
        "height": float(h),
        "volume_m3": float(l * w * h),
    }


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    h, w = frame.shape[:2]

    # YOLO
    results = yolo_model(frame, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    print("Detections:", len(boxes))

    sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    depth_map = infer_depth(frame)

    output = []

    for box, cls, conf in zip(boxes, classes, confs):
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)

        print("Detected:", yolo_model.names[int(cls)], conf)

        # SAM mask
        masks, _, _ = sam_predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=False
        )

        mask = masks[0].astype(np.uint8)

        print("Mask pixels:", mask.sum())

        # fallback
        if mask.sum() < 10:
            print("⚠️ fallback bbox mask")
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1

        dims = measure(depth_map, mask)

        output.append({
            "label": yolo_model.names[int(cls)],
            "confidence": float(conf),
            "dimensions": dims
        })

    return jsonify({"objects": output})


if __name__ == "__main__":
    app.run(port=5001, debug=False)