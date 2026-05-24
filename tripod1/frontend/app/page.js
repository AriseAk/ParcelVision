"use client";

import { useRef, useState, useEffect, useCallback } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// MATH HELPERS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Converts device orientation Euler angles (alpha, beta, gamma) to a 3×3
 * rotation matrix aligned with the camera's optical axis convention:
 *   Z-forward (into scene), X-right, Y-down.
 *
 * DeviceOrientation uses the ZXY intrinsic Euler convention:
 *   R = Rz(alpha) · Rx(beta) · Ry(gamma)
 *
 * We then apply a fixed 90° rotation around X to flip from the browser's
 * "screen-up" frame to the camera's "Z-forward / Y-down" frame.
 */
function eulerToRotationMatrix(alpha, beta, gamma) {
  const toRad = Math.PI / 180;
  const a = alpha * toRad; // rotation around Z (compass heading)
  const b = beta  * toRad; // rotation around X (front-back tilt)
  const g = gamma * toRad; // rotation around Y (left-right tilt)

  // Rz(alpha)
  const Rz = [
    [ Math.cos(a), -Math.sin(a), 0],
    [ Math.sin(a),  Math.cos(a), 0],
    [           0,            0, 1],
  ];

  // Rx(beta)
  const Rx = [
    [1,           0,            0],
    [0, Math.cos(b), -Math.sin(b)],
    [0, Math.sin(b),  Math.cos(b)],
  ];

  // Ry(gamma)
  const Ry = [
    [ Math.cos(g), 0, Math.sin(g)],
    [           0, 1,           0],
    [-Math.sin(g), 0, Math.cos(g)],
  ];

  // R_device = Rz · Rx · Ry  (ZXY intrinsic)
  const R_device = mat3Mul(mat3Mul(Rz, Rx), Ry);

  // Flip-frame: rotate -90° around X so that camera Z points forward/down
  // instead of out of the screen.
  const Rx90 = [
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
  ];

  return mat3Mul(R_device, Rx90);
}

/** Multiply two 3×3 matrices. */
function mat3Mul(A, B) {
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++)
        C[i][j] += A[i][k] * B[k][j];
  return C;
}

/**
 * Estimate focal length in pixels from the camera's field-of-view.
 * Uses the relation: f = (width / 2) / tan(hFOV / 2).
 * Falls back to a standard 70° horizontal FOV if the track is unavailable.
 */
function estimateFocalLength(videoTrack, videoEl) {
  const hFovDeg = 70; // typical smartphone wide camera
  const width = videoEl.videoWidth || 1280;
  const f = (width / 2) / Math.tan((hFovDeg * Math.PI) / 360);
  return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5000";

const TAP_SEQUENCE = [
  { label: "BL", instruction: "Tap the BOTTOM-LEFT corner of the box (touching floor)" },
  { label: "BR", instruction: "Tap the BOTTOM-RIGHT corner of the box (touching floor)" },
  { label: "TL", instruction: "Tap the TOP-LEFT corner directly above the first tap" },
];

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
export default function Home() {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [phase, setPhase]             = useState("intro");
  const [status, setStatus]           = useState("");
  const [calibratedZ, setCalibratedZ] = useState(null);
  const [taps, setTaps]               = useState([]);
  const [result, setResult]           = useState(null);
  const [gyro, setGyro]               = useState({ alpha: 0, beta: 0, gamma: 0 });
  const [gyroMatrix, setGyroMatrix]   = useState(eulerToRotationMatrix(0, 0, 0));
  const [frozenFrame, setFrozenFrame] = useState(null); // base64 PNG
  const [permissionError, setPermissionError] = useState("");
  const [loading, setLoading]         = useState(false);

  // ── Camera Setup ─────────────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (e) {
      setPermissionError("Camera access denied. Please allow camera permissions and reload.");
      console.error(e);
    }
  }, []);

  // ── Gyroscope Listener ───────────────────────────────────────────────────
  useEffect(() => {
    function handleOrientation(e) {
      const a = e.alpha ?? 0;
      const b = e.beta  ?? 0;
      const g = e.gamma ?? 0;
      setGyro({ alpha: a, beta: b, gamma: g });
      setGyroMatrix(eulerToRotationMatrix(a, b, g));
    }

    // iOS 13+ requires explicit permission
    if (typeof DeviceOrientationEvent.requestPermission === "function") {
      DeviceOrientationEvent.requestPermission().then((state) => {
        if (state === "granted") window.addEventListener("deviceorientation", handleOrientation, true);
      });
    } else {
      window.addEventListener("deviceorientation", handleOrientation, true);
    }

    return () => window.removeEventListener("deviceorientation", handleOrientation, true);
  }, []);

  // ── Capture a frame to canvas and return base64 PNG ──────────────────────
  const captureFrame = useCallback(() => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;

    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL("image/png");
  }, []);

  // ── Build intrinsic matrix K ─────────────────────────────────────────────
  const buildK = useCallback(() => {
    const video = videoRef.current;
    if (!video) return [[1000,0,640],[0,1000,360],[0,0,1]];
    const track = streamRef.current?.getVideoTracks()[0] ?? null;
    const f  = estimateFocalLength(track, video);
    const cx = video.videoWidth  / 2;
    const cy = video.videoHeight / 2;
    return [
      [f,  0, cx],
      [0,  f, cy],
      [0,  0,  1],
    ];
  }, []);

  // ── Phase 1: Calibrate ────────────────────────────────────────────────────
  const handleCalibrate = useCallback(async () => {
    setLoading(true);
    setStatus("Capturing frame…");

    const frameB64 = captureFrame();
    if (!frameB64) { setStatus("Could not capture frame."); setLoading(false); return; }

    const video = videoRef.current;
    const track = streamRef.current?.getVideoTracks()[0] ?? null;
    const focalLength = estimateFocalLength(track, video);

    // Convert base64 data URL to Blob
    const res  = await fetch(frameB64);
    const blob = await res.blob();

    const form = new FormData();
    form.append("image", blob, "calibration.png");
    form.append("focal_length", String(focalLength));

    try {
      setStatus("Sending to backend…");
      const response = await fetch(`${BACKEND_URL}/calibrate_height`, {
        method: "POST",
        body: form,
      });
      const data = await response.json();
      if (data.error) { setStatus(`Error: ${data.error}`); setLoading(false); return; }
      setCalibratedZ(data.calibrated_z);
      setPhase("calibrated");
      setStatus(`✓ Camera height: ${data.calibrated_z.toFixed(3)} m`);
    } catch (e) {
      setStatus(`Network error: ${e}`);
    }
    setLoading(false);
  }, [captureFrame]);

  // ── Freeze frame for tapping ──────────────────────────────────────────────
  const handleFreezeForMeasure = useCallback(() => {
    const frame = captureFrame();
    setFrozenFrame(frame);
    setTaps([]);
    setPhase("measuring");
    setStatus(TAP_SEQUENCE[0].instruction);
  }, [captureFrame]);

  // ── Handle taps on the frozen image ──────────────────────────────────────
  const handleImageTap = useCallback((e) => {
    if (phase !== "measuring") return;
    if (taps.length >= 3) return;

    // Map click position → original video pixel coordinates
    const img    = e.currentTarget;
    const rect   = img.getBoundingClientRect();
    const scaleX = (videoRef.current?.videoWidth  ?? img.naturalWidth)  / rect.width;
    const scaleY = (videoRef.current?.videoHeight ?? img.naturalHeight) / rect.height;
    const u = (e.clientX - rect.left) * scaleX;
    const v = (e.clientY - rect.top)  * scaleY;

    const newTap  = { u, v, label: TAP_SEQUENCE[taps.length].label };
    const newTaps = [...taps, newTap];
    setTaps(newTaps);

    if (newTaps.length < 3) {
      setStatus(TAP_SEQUENCE[newTaps.length].instruction);
    } else {
      setStatus("All 3 points selected. Tap 'Measure' to calculate.");
    }
  }, [phase, taps]);

  // ── Phase 2: Measure ──────────────────────────────────────────────────────
  const handleMeasure = useCallback(async () => {
    if (taps.length < 3 || calibratedZ === null) return;
    setLoading(true);
    setStatus("Calculating…");

    const payload = {
      calibrated_z: calibratedZ,
      R_gyro: gyroMatrix,
      K: buildK(),
      taps: taps.map(t => ({ u: t.u, v: t.v })),
    };

    try {
      const response = await fetch(`${BACKEND_URL}/measure_box`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (data.error) { setStatus(`Error: ${data.error}`); setLoading(false); return; }
      setResult(data);
      setPhase("result");
    } catch (e) {
      setStatus(`Network error: ${e}`);
    }
    setLoading(false);
  }, [taps, calibratedZ, gyroMatrix, buildK]);

  // ── Reset ─────────────────────────────────────────────────────────────────
  const handleReset = useCallback(() => {
    setPhase("calibrated");
    setTaps([]);
    setFrozenFrame(null);
    setResult(null);
    setStatus(`Camera height: ${calibratedZ?.toFixed(3)} m — ready to measure`);
  }, [calibratedZ]);

  // ── Start camera on mount ─────────────────────────────────────────────────
  useEffect(() => {
    startCamera();
    return () => {
      streamRef.current?.getTracks().forEach(t => t.stop());
    };
  }, [startCamera]);

  // ─────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <main className="root">
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} style={{ display: "none" }} />

      {/* Live camera feed (always running in background) */}
      <video
        ref={videoRef}
        className="video-bg"
        playsInline
        muted
        style={{ display: phase === "measuring" || phase === "result" ? "none" : "block" }}
      />

      {/* ── FROZEN FRAME (measuring phase) ── */}
      {(phase === "measuring" || phase === "result") && frozenFrame && (
        <div className="frozen-container">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={frozenFrame}
            alt="Frozen frame for measurement"
            className="frozen-img"
            onClick={handleImageTap}
            draggable={false}
          />
          {/* Render tap markers */}
          {taps.map((tap, i) => (
            <TapMarker
              key={i}
              tap={tap}
              index={i}
              frozenFrame={frozenFrame}
              videoWidth={videoRef.current?.videoWidth ?? 1280}
              videoHeight={videoRef.current?.videoHeight ?? 720}
            />
          ))}
        </div>
      )}

      {/* ── HUD OVERLAY ── */}
      <div className="hud">
        {/* Header */}
        <div className="hud-header">
          <span className="logo">◈ TRIPOD</span>
          <span className="tagline">Monocular Box Metrology</span>
        </div>

        {/* Gyro readout */}
        <div className="gyro-strip">
          <span>α {gyro.alpha.toFixed(1)}°</span>
          <span>β {gyro.beta.toFixed(1)}°</span>
          <span>γ {gyro.gamma.toFixed(1)}°</span>
        </div>

        {/* Permission error */}
        {permissionError && <div className="error-banner">{permissionError}</div>}

        {/* ── Phase: INTRO ── */}
        {phase === "intro" && (
          <div className="panel">
            <h2 className="panel-title">Setup</h2>
            <ol className="steps">
              <li>Place a <strong>credit card</strong> flat on the floor.</li>
              <li>Hold phone against your <strong>chest</strong>, lens pointing down toward the card.</li>
              <li>Tap <em>Calibrate</em> to measure camera height.</li>
            </ol>
            <button className="btn primary" onClick={() => setPhase("calibrating")}>
              Begin Calibration →
            </button>
          </div>
        )}

        {/* ── Phase: CALIBRATING ── */}
        {phase === "calibrating" && (
          <div className="panel">
            <h2 className="panel-title">Calibration</h2>
            <p className="instruction">
              Aim at the credit card on the floor. Keep the phone steady against your chest.
            </p>
            {status && <p className="status">{status}</p>}
            <button className="btn primary" onClick={handleCalibrate} disabled={loading}>
              {loading ? "Processing…" : "📷 Capture & Calibrate"}
            </button>
          </div>
        )}

        {/* ── Phase: CALIBRATED ── */}
        {phase === "calibrated" && (
          <div className="panel">
            <h2 className="panel-title">Ready to Measure</h2>
            {status && <p className="status success">{status}</p>}
            <p className="instruction">
              Now place the box in view. Keep phone against your chest and tilt to see the box.
              Tap below to freeze the frame and mark corners.
            </p>
            <button className="btn primary" onClick={handleFreezeForMeasure}>
              🔒 Freeze Frame & Tap Corners →
            </button>
          </div>
        )}

        {/* ── Phase: MEASURING ── */}
        {phase === "measuring" && (
          <div className="panel semi">
            <h2 className="panel-title">Mark Corners</h2>
            <p className="instruction tap-instruction">{status}</p>
            <div className="tap-progress">
              {TAP_SEQUENCE.map((s, i) => (
                <div key={i} className={`tap-step ${i < taps.length ? "done" : i === taps.length ? "active" : ""}`}>
                  <span className="tap-dot">{i < taps.length ? "✓" : i + 1}</span>
                  <span>{s.label}</span>
                </div>
              ))}
            </div>
            <div className="btn-row">
              <button className="btn secondary" onClick={() => { setTaps([]); setStatus(TAP_SEQUENCE[0].instruction); }}>
                Reset Taps
              </button>
              <button
                className="btn primary"
                onClick={handleMeasure}
                disabled={taps.length < 3 || loading}
              >
                {loading ? "Calculating…" : "Measure →"}
              </button>
            </div>
          </div>
        )}

        {/* ── Phase: RESULT ── */}
        {phase === "result" && result && (
          <div className="panel result-panel">
            <h2 className="panel-title">📦 Measurement Result</h2>
            <div className="metric-grid">
              <Metric label="Length" value={result.length} unit="m" />
              <Metric label="Width"  value={result.width}  unit="m" />
              <Metric label="Height" value={result.height} unit="m" />
              <Metric label="Volume" value={result.volume} unit="m³" accent />
            </div>
            <div className="btn-row">
              <button className="btn secondary" onClick={handleReset}>Measure Again</button>
              <button className="btn secondary" onClick={() => {
                setPhase("calibrating");
                setCalibratedZ(null);
                setFrozenFrame(null);
                setTaps([]);
                setResult(null);
                setStatus("");
              }}>
                Re-Calibrate
              </button>
            </div>
          </div>
        )}
      </div>

      <style jsx global>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          background: #000;
          font-family: 'Courier New', 'Lucida Console', monospace;
          color: #e2f0e2;
          overflow: hidden;
          height: 100dvh;
        }
        .root { position: relative; width: 100vw; height: 100dvh; overflow: hidden; }
        .video-bg { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; z-index: 0; }
        .frozen-container { position: absolute; inset: 0; z-index: 0; }
        .frozen-img { width: 100%; height: 100%; object-fit: contain; background: #000; cursor: crosshair; display: block; }
        .hud { position: absolute; inset: 0; z-index: 10; display: flex; flex-direction: column; pointer-events: none; }
        .hud-header { display: flex; align-items: baseline; gap: 12px; padding: 16px 20px 8px; background: linear-gradient(to bottom, rgba(0,0,0,0.85), transparent); pointer-events: none; }
        .logo { font-size: 1.4rem; font-weight: 700; letter-spacing: 0.15em; color: #6fffb0; text-shadow: 0 0 12px #6fffb0aa; }
        .tagline { font-size: 0.65rem; letter-spacing: 0.2em; color: #6fffb066; text-transform: uppercase; }
        .gyro-strip { display: flex; gap: 16px; padding: 4px 20px; font-size: 0.65rem; letter-spacing: 0.1em; color: #6fffb099; pointer-events: none; }
        .panel { margin-top: auto; background: rgba(0, 12, 6, 0.92); border-top: 1px solid #6fffb033; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); padding: 24px 20px 32px; pointer-events: all; }
        .panel.semi { background: rgba(0, 12, 6, 0.80); }
        .panel.result-panel { background: rgba(0, 20, 10, 0.95); border-top: 1px solid #6fffb066; }
        .panel-title { font-size: 0.75rem; letter-spacing: 0.25em; text-transform: uppercase; color: #6fffb0; margin-bottom: 14px; }
        .instruction { font-size: 0.85rem; line-height: 1.6; color: #b8d4b8; margin-bottom: 16px; }
        .tap-instruction { background: rgba(111,255,176,0.08); border-left: 3px solid #6fffb0; padding: 10px 14px; border-radius: 0 4px 4px 0; font-size: 0.8rem; }
        .status { font-size: 0.8rem; color: #b8d4b8; margin-bottom: 14px; font-style: italic; }
        .status.success { color: #6fffb0; }
        .steps { padding-left: 20px; margin-bottom: 20px; font-size: 0.82rem; line-height: 1.9; color: #b8d4b8; }
        .steps strong { color: #6fffb0; }
        .steps em { color: #ffe166; }
        .btn { width: 100%; padding: 14px; border: none; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 0.85rem; letter-spacing: 0.1em; cursor: pointer; transition: all 0.15s ease; }
        .btn.primary { background: #6fffb0; color: #001a0a; font-weight: 700; }
        .btn.primary:hover:not(:disabled) { background: #9fffcc; box-shadow: 0 0 20px #6fffb055; }
        .btn.primary:disabled { background: #2a4a36; color: #4a8a6a; cursor: not-allowed; }
        .btn.secondary { background: transparent; color: #6fffb0; border: 1px solid #6fffb044; font-weight: 400; }
        .btn.secondary:hover { background: rgba(111,255,176,0.08); }
        .btn-row { display: flex; gap: 10px; }
        .btn-row .btn { flex: 1; }
        .tap-progress { display: flex; gap: 8px; margin-bottom: 16px; }
        .tap-step { display: flex; align-items: center; gap: 6px; flex: 1; padding: 8px 10px; border-radius: 4px; font-size: 0.7rem; letter-spacing: 0.1em; border: 1px solid #2a4a36; color: #4a8a6a; transition: all 0.2s; }
        .tap-step.active { border-color: #ffe166; color: #ffe166; background: rgba(255,225,102,0.08); }
        .tap-step.done { border-color: #6fffb044; color: #6fffb0; background: rgba(111,255,176,0.06); }
        .tap-dot { width: 20px; height: 20px; border-radius: 50%; border: 1px solid currentColor; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; flex-shrink: 0; }
        .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }
        .metric { background: rgba(111,255,176,0.05); border: 1px solid #6fffb022; border-radius: 4px; padding: 14px 12px; }
        .metric.accent { border-color: #6fffb066; background: rgba(111,255,176,0.1); grid-column: span 2; }
        .metric-label { font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase; color: #6fffb077; margin-bottom: 6px; }
        .metric-value { font-size: 1.4rem; font-weight: 700; color: #e2f0e2; }
        .metric.accent .metric-value { font-size: 1.8rem; color: #6fffb0; text-shadow: 0 0 16px #6fffb066; }
        .metric-unit { font-size: 0.7rem; color: #6fffb066; margin-left: 4px; }
        .tap-marker { position: absolute; transform: translate(-50%, -50%); pointer-events: none; z-index: 20; }
        .tap-marker-ring { width: 28px; height: 28px; border-radius: 50%; border: 2px solid; display: flex; align-items: center; justify-content: center; font-size: 0.6rem; font-weight: 700; }
        .error-banner { background: rgba(255,80,80,0.15); border: 1px solid #ff505066; color: #ff8080; padding: 10px 16px; font-size: 0.78rem; margin: 0 20px; border-radius: 4px; pointer-events: all; }
      `}</style>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────

function Metric({ label, value, unit, accent }) {
  return (
    <div className={`metric${accent ? " accent" : ""}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        {value.toFixed(3)}
        <span className="metric-unit">{unit}</span>
      </div>
    </div>
  );
}

/**
 * Renders a tap-point marker on the frozen image.
 * Converts original video pixel coords → displayed image CSS coords.
 */
function TapMarker({ tap, index, frozenFrame, videoWidth, videoHeight }) {
  const colors = ["#ffe166", "#ff8c42", "#6fffb0"];
  const color  = colors[index] ?? "#fff";
  const [pos, setPos] = useState(null);

  useEffect(() => {
    function computePos() {
      const img = document.querySelector(".frozen-img");
      if (!img) return;
      const rect = img.getBoundingClientRect();
      const scaleX = rect.width  / videoWidth;
      const scaleY = rect.height / videoHeight;
      setPos({
        x: rect.left + tap.u * scaleX,
        y: rect.top  + tap.v * scaleY,
      });
    }
    computePos();
    window.addEventListener("resize", computePos);
    return () => window.removeEventListener("resize", computePos);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tap, videoWidth, videoHeight, frozenFrame]);

  if (!pos) return null;

  return (
    <div className="tap-marker" style={{ left: pos.x, top: pos.y, position: "fixed" }}>
      <div
        className="tap-marker-ring"
        style={{ borderColor: color, color, boxShadow: `0 0 10px ${color}88` }}
      >
        {tap.label}
      </div>
    </div>
  );
}
