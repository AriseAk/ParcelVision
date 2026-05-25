# 📦 ParcelVision

> Mobile computer vision pipeline for 3D logistics volume estimation (m³)

---

## 🔍 Project Summary

ParcelVision set out to build a mobile CV pipeline capable of estimating the real-world 3D volume of logistics items (boxes, parcels, bags) using only a standard smartphone camera.

**The core finding:** True metric 3D volume from a monocular camera is mathematically infeasible without a physical reference or depth hardware. ML models can predict relative shape, but not absolute metric size — a fundamental consequence of scale ambiguity in single-camera geometry.

We successfully built a **robust 2D detection, segmentation, and tracking pipeline**, and exhaustively tested every monocular scale-recovery approach. The engineering foundation is production-ready and fully reusable for the recommended next steps.

---

## ✅ Pipeline — What Was Built

| Stage | Technology | Notes |
|---|---|---|
| **Detection** | YOLOv8-World | Open-vocabulary, covers full logistics item vocabulary |
| **Segmentation** | Segment Anything Model (SAM) | Per-object pixel masks; bounding box fallback included |
| **Tracking** | IoU-based temporal smoothing | Stable object IDs and dimension estimates across frames |
| **Depth Estimation** | DepthAnything V2 / MiDaS | Relative per-frame 3D point cloud generation |
| **IMU Sensor Fusion** | Gyroscope + Accelerometer + Optical Flow + Kalman Filter | Fully implemented; failed due to sensor drift (see below) |

---

## 🚧 The Technical Roadblock

Image pixels carry no absolute scale. A pixel could represent 1 cm or 1 m — the camera equation alone cannot resolve this.

### Approaches Tested & Rejected

| Approach | Rejection Reason |
|---|---|
| Manual calibration | Poor UX; error-prone in real conditions |
| Reference objects (known-size marker) | Requires user placement; rejected on UX grounds |
| Single-frame 3D reconstruction | Insufficient accuracy (outside ±10–20% target) |
| IMU sensor fusion | Unrecoverable gyroscope + accelerometer drift on consumer hardware |

None of the above achieved the required **±10–20% volumetric accuracy** without user intervention that would make the app impractical.

---

## 🗺️ Recommended Next Steps

The IMU path has been exhausted. Future development must use hardware-assisted scale recovery:

### Path A — LiDAR (iPhone Pro / iPad Pro)
Direct metric depth output from the LiDAR sensor eliminates scale ambiguity entirely. No estimation required.

### Path B — WebXR + ARCore / ARKit
Visual Inertial Odometry via ARCore (Android) or ARKit (iOS) provides accurate metric pose estimation on supported devices without additional hardware.

**Both paths are architecturally compatible with the existing pipeline.** The detection, segmentation, and tracking stages are fully reusable — only the scale-recovery layer needs to be swapped in.

---

## 📁 Repository

Full technical report, architecture benchmarks, and code:
**[Internship_Parcelvision](https://drive.google.com/drive/folders/1ZtrVU9mV3CgniIXBlsuVc8K5P2Jmg6iy)** *(link to internal repo)*

---

## 🛠️ Tech Stack

- Python
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2) / [MiDaS](https://github.com/isl-org/MiDaS)
- OpenCV (optical flow, IoU tracking)
- NumPy / SciPy (Kalman filtering, IMU integration)
