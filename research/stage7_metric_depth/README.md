# Stage 7 — Metric Depth Pipeline

Replaces `stage6_imu_fusion/fixed_scale_server.py` with a clean metric depth pipeline.

## The One Core Change

| | Old (fixed_scale_server.py) | New (metric_depth_server.py) |
|---|---|---|
| Depth model | DepthAnything V2 | **ZoeDepth (ZoeD_NK)** |
| Depth output | Relative disparity (0–1 after normalisation) | **Absolute metres** |
| Scale recovery | `scale = camera_height / ground_disparity` | **Not needed** |
| Fragility | Breaks if camera height changes, floor is occluded, or lighting changes | Stable — model-internal calibration |

Everything else is **identical** to `fixed_scale_server.py`:
- YOLO World detection
- SAM segmentation
- IoU tracker (`tracker.py`)
- `SceneStateManager` (`scene_state.py`)
- Visual odometry + IMU fusion
- Multi-frame scan session
- `/detect`, `/start_scan`, `/scan_frame`, `/compute_dimensions`
- Identical JSON response shapes (frontend unchanged)

---

## Files

```
stage7_metric_depth/
├── zoedepth_runner.py        ← ZoeDepth wrapper, returns metric metres
├── pointcloud_builder.py     ← 3D geometry helpers (extracted from server)
├── temporal_smoothing.py     ← Per-object dimension EMA smoother
├── metric_depth_server.py    ← Main Flask server (replaces fixed_scale_server.py)
└── requirements.txt
```

---

## Setup

```bash
cd research/stage7_metric_depth
pip install -r requirements.txt

# SAM checkpoint (if not already present in backend/)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# ZoeDepth weights are downloaded automatically on first run via torch.hub
```

---

## Running

```bash
python metric_depth_server.py
```

ZoeDepth weights (~350 MB) are downloaded on first startup via `torch.hub`.
Subsequent starts are instant.

---

## Recommended Build Order (from architecture doc)

Follow this — do NOT try to do everything at once:

1. **Single image test** — run `python zoedepth_runner.py your_image.jpg`  
   Verify depth colourmaps look correct (near objects bright, far dark).

2. **Single image + point cloud** — run the server, POST one frame to `/detect`,  
   check `mean_depth_m` in the response is plausible (~0.5–3.0 m for typical scenes).

3. **Single image + dimensions** — verify L/W/H outputs are in the right ballpark  
   before touching multi-frame.

4. **Multi-frame scan** — only once step 3 is validated.

---

## What Was Removed

These were all symptoms of relative depth not having metric scale:

- `estimate_ground_depth()` — estimated floor disparity to derive scale
- `get_or_init_global_scale()` — locked a per-session scale factor
- `scan_session['scale_initialized']` / `'global_scale'` / `'last_scale']`
- `scale = camera_height / ground` in `/detect`
- `m["length"] *= scale` / `m["width"] *= scale` / `m["height"] *= scale`
- `zs_metric = zs_c * scale` in `/scan_frame`
- All of `stage2_depth_estimation/` (DepthAnything V2, MiDaS fallback)

These are **not deleted** — `stage2_depth_estimation/` and `stage6_imu_fusion/`  
remain as research artifacts and report evidence.

---

## Calibration Endpoints

`/start_calibration` and `/end_calibration` return `status: not_required` for  
frontend compatibility. ZoeDepth is internally calibrated — no user calibration step.

---

## Troubleshooting

**ZoeDepth loads but depths look wrong (everything ~0.1 m or ~10 m)**  
→ Check lighting. ZoeDepth degrades badly in very low light or pure white walls.  
→ Try `ZoeD_N` (NYU indoor-only model) instead of `ZoeD_NK`.

**CUDA out of memory**  
→ ZoeDepth ViT backbone is heavier than DepthAnything ViT-S.  
→ Reduce input resolution: add `frame = cv2.resize(frame, (640, 480))` before `infer_metric_depth()`.  
→ Or run on CPU (slower but works).

**`ModuleNotFoundError: No module named 'tracker'`**  
→ Make sure you're running from `stage7_metric_depth/` and `stage3_segmentation_tracking/` exists at `../stage3_segmentation_tracking/`.
