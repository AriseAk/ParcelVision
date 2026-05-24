# ◈ TRIPOD — Monocular Box Metrology

Measure the 3D dimensions of cardboard boxes using a **single smartphone camera**
and the **Human Tripod constraint** (no depth networks, no SLAM, no ARCore).

---

## How It Works

1. **Calibration** — Place a credit card flat on the floor. The backend finds its
   corners with OpenCV, runs `solvePnP`, and derives the absolute camera height Z.
2. **Measurement** — Freeze the frame and tap 3 corners:
   - **BL** — Bottom-Left (on the floor)
   - **BR** — Bottom-Right (on the floor)
   - **TL** — Top-Left (directly above BL)
   
   The backend casts rays through each pixel using the gyroscope rotation matrix
   and intersects them with the floor plane (Z = 0) or a vertical plumb-line.

---

## Project Structure

```
├── backend/
│   ├── app.py              ← Flask server (2 endpoints)
│   └── requirements.txt
└── frontend/
    ├── app/
    │   ├── layout.tsx
    │   └── page.tsx        ← React/Next.js app
    ├── package.json
    ├── next.config.js
    └── tsconfig.json
```

---

## Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
# → Listening on http://0.0.0.0:5000
```

---

## Frontend Setup

```bash
cd frontend
npm install
# Set backend URL (defaults to http://localhost:5000)
echo "NEXT_PUBLIC_BACKEND_URL=http://<your-machine-ip>:5000" > .env.local
npm run dev
# → http://localhost:3000
```

> **Mobile testing**: your phone must be on the same LAN as the dev server.
> Use `npm run dev -- --hostname 0.0.0.0` and navigate to `http://<PC-IP>:3000`.

---

## API Reference

### `POST /calibrate_height`
**Content-Type:** `multipart/form-data`

| Field          | Type   | Description                          |
|----------------|--------|--------------------------------------|
| `image`        | file   | Photo containing a credit card       |
| `focal_length` | float  | Estimated focal length in pixels     |

**Response:**
```json
{ "calibrated_z": 1.142 }
```

---

### `POST /measure_box`
**Content-Type:** `application/json`

```json
{
  "calibrated_z": 1.142,
  "R_gyro":  [[...], [...], [...]],
  "K":       [[...], [...], [...]],
  "taps":    [
    { "u": 512, "v": 640 },
    { "u": 720, "v": 640 },
    { "u": 512, "v": 480 }
  ]
}
```

**Response:**
```json
{
  "length": 0.3012,
  "width":  0.3012,
  "height": 0.2408,
  "volume": 0.021865
}
```

---

## Math Summary

### Calibration
```
K_inv · [u, v, 1]ᵀ            → direction in camera frame
C = -R.T @ t                   → camera centre in world frame
calibrated_z = |C[2]|          → height above Z=0 floor
```

### Floor ray intersection (Bottom corners)
```
P(λ) = C + λ · R.T · d_cam
λ    = -C_z / (R.T · d_cam)_z
P_floor = C + λ · d_world
```

### Height via plumb-line (Top-Left corner)
```
Plumb-line: P_BL + s · [0,0,1]
Camera ray: C + t · d_tl_world
→ closest_point_on_lines(...)[1].z  = box height
```

### Euler → Rotation Matrix (ZXY intrinsic, browser convention)
```
R = Rz(α) · Rx(β) · Ry(γ) · Rx(−90°)
```
