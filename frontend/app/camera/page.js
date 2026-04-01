"use client";

import { useRef, useEffect, useState, useCallback } from "react";

// ─── Config ────────────────────────────────────────────────────────────────
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
const DETECT_INTERVAL_MS = 600;
const SCAN_FRAME_INTERVAL_MS = 800;

// ─── Helpers ────────────────────────────────────────────────────────────────
function getFocalLength(videoEl) {
  const w = videoEl.videoWidth || 1280;
  // Approximate fx from a 65° HFOV assumption
  return (w / 2) / Math.tan((65 * Math.PI) / 180 / 2);
}

async function captureFrame(videoEl, canvasEl) {
  const w = videoEl.videoWidth;
  const h = videoEl.videoHeight;
  canvasEl.width = w;
  canvasEl.height = h;
  canvasEl.getContext("2d").drawImage(videoEl, 0, 0, w, h);
  return new Promise((res) =>
    canvasEl.toBlob((b) => res(b), "image/jpeg", 0.88)
  );
}

function fmtM(v) {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)} cm`;
}
function fmtVol(v) {
  if (v == null) return "—";
  if (v < 0.001) return `${(v * 1e6).toFixed(0)} cm³`;
  return `${(v * 1e6).toFixed(0)} cm³`;
}

// ─── Phases ─────────────────────────────────────────────────────────────────
const PHASE = {
  IDLE: "IDLE",
  CALIBRATING: "CALIBRATING",
  DETECTING: "DETECTING",
  SCANNING: "SCANNING",
  RESULT: "RESULT",
};

// ─── Main Component ──────────────────────────────────────────────────────────
export default function CameraPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const detectTimerRef = useRef(null);
  const scanTimerRef = useRef(null);

  const [phase, setPhase] = useState(PHASE.IDLE);
  const [calFrames, setCalFrames] = useState(0);
  const [calReady, setCalReady] = useState(false);
  const [calibrated, setCalibrated] = useState(false);
  const [scene, setScene] = useState([]);
  const [scanFrames, setScanFrames] = useState(0);
  const [result, setResult] = useState(null);
  const [motion, setMotion] = useState(0);
  const [error, setError] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [reproj, setReproj] = useState(null);
  const [scanStatus, setScanStatus] = useState("");
  const [loading, setLoading] = useState(false);

  // ── Camera init ────────────────────────────────────────────────────────────
  useEffect(() => {
    let stream;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setCameraReady(true);
        };
      } catch (e) {
        setError("Camera access denied: " + e.message);
      }
    })();
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
      clearInterval(detectTimerRef.current);
      clearInterval(scanTimerRef.current);
    };
  }, []);

  // ── Draw overlays ──────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const video = videoRef.current;
    if (!video) return;

    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    scene.forEach((obj) => {
      const [x1, y1, x2, y2] = obj.bbox;
      const isCard = obj.label.toLowerCase().includes("card");

      ctx.strokeStyle = isCard ? "#00ffcc" : "#ff6b35";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Corner ticks
      const tw = 12;
      ctx.lineWidth = 3;
      [[x1, y1], [x2, y1], [x2, y2], [x1, y2]].forEach(([cx, cy], i) => {
        const sx = i < 2 ? 1 : -1;
        const sy = i === 0 || i === 3 ? 1 : -1;
        ctx.beginPath();
        ctx.moveTo(cx, cy + sy * tw);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx + sx * tw, cy);
        ctx.stroke();
      });

      // Label
      const label = obj.dimensions
        ? `${(obj.dimensions.length * 100).toFixed(1)}×${(obj.dimensions.width * 100).toFixed(1)}×${(obj.dimensions.height * 100).toFixed(1)} cm`
        : obj.label;

      ctx.font = "bold 13px 'DM Mono', monospace";
      const tw2 = ctx.measureText(label).width + 10;
      ctx.fillStyle = isCard ? "rgba(0,255,204,0.15)" : "rgba(255,107,53,0.15)";
      ctx.fillRect(x1, y1 - 22, tw2, 20);
      ctx.fillStyle = isCard ? "#00ffcc" : "#ff6b35";
      ctx.fillText(label, x1 + 5, y1 - 6);
    });
  }, [scene]);

  // ── Calibration loop ───────────────────────────────────────────────────────
  const startCalibration = useCallback(async () => {
    setError(null);
    setCalFrames(0);
    setCalReady(false);
    setReproj(null);
    setPhase(PHASE.CALIBRATING);

    // Reset server-side calibration
    await fetch(`${API_BASE}/reset_calibration`, { method: "POST" }).catch(() => {});

    detectTimerRef.current = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current) return;
      try {
        const blob = await captureFrame(videoRef.current, canvasRef.current);
        const fx = getFocalLength(videoRef.current);
        const imgW = videoRef.current.videoWidth;
        const imgH = videoRef.current.videoHeight;

        const fd = new FormData();
        fd.append("image", blob, "frame.jpg");
        fd.append("img_w", imgW);
        fd.append("img_h", imgH);
        fd.append("fx", fx);

        // Also run YOLO to get card bbox/corners for PnP
        const detFd = new FormData();
        detFd.append("image", blob, "frame.jpg");
        detFd.append("fx", fx);
        const detRes = await fetch(`${API_BASE}/detect`, { method: "POST", body: detFd });
        const detData = await detRes.json();
        if (detData.scene) setScene(detData.scene);

        // Find card in scene
        const card = detData.scene?.find((o) =>
          o.label.toLowerCase().includes("card")
        );

        if (card) {
          const [x1, y1, x2, y2] = card.bbox;
          fd.append("pixel_w", x2 - x1);
          fd.append("pixel_h", y2 - y1);
          fd.append("bbox_x", x1);
          fd.append("bbox_y", y1);
        } else {
          return; // No card detected this frame
        }

        const res = await fetch(`${API_BASE}/calibrate_frame`, { method: "POST", body: fd });
        const data = await res.json();

        if (data.count !== undefined) setCalFrames(data.count);
        if (data.reproj_error !== undefined) setReproj(data.reproj_error);
        if (data.ready) {
          setCalReady(true);
          clearInterval(detectTimerRef.current);
          // Auto-confirm when ready
          await confirmCalibration(fx, imgW, imgH);
        }
      } catch (e) {
        console.error("Cal frame error:", e);
      }
    }, DETECT_INTERVAL_MS);
  }, []);

  const confirmCalibration = useCallback(async (fxVal, imgW, imgH) => {
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("fx", fxVal || getFocalLength(videoRef.current));
      fd.append("img_w", imgW || videoRef.current.videoWidth);
      fd.append("img_h", imgH || videoRef.current.videoHeight);
      const res = await fetch(`${API_BASE}/confirm_calibration`, { method: "POST", body: fd });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setCalibrated(true);
      setPhase(PHASE.DETECTING);
      startDetection();
    } catch (e) {
      setError("Calibration failed: " + e.message);
      setPhase(PHASE.IDLE);
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Detection loop ─────────────────────────────────────────────────────────
  const startDetection = useCallback(() => {
    clearInterval(detectTimerRef.current);
    detectTimerRef.current = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current) return;
      try {
        const blob = await captureFrame(videoRef.current, canvasRef.current);
        const fx = getFocalLength(videoRef.current);
        const fd = new FormData();
        fd.append("image", blob, "frame.jpg");
        fd.append("fx", fx);
        fd.append("detect", "1");
        const res = await fetch(`${API_BASE}/detect`, { method: "POST", body: fd });
        const data = await res.json();
        if (data.scene) setScene(data.scene);
        if (data.motion !== undefined) setMotion(data.motion);
        if (data.calibrated !== undefined && !data.calibrated) {
          setCalibrated(false);
        }
      } catch (e) {
        console.error("Detect error:", e);
      }
    }, DETECT_INTERVAL_MS);
  }, []);

  // ── Scan mode ──────────────────────────────────────────────────────────────
  const startScan = useCallback(async () => {
    clearInterval(detectTimerRef.current);
    setScanFrames(0);
    setScanStatus("Starting scan…");
    setResult(null);
    setError(null);

    await fetch(`${API_BASE}/start_scan`, { method: "POST" }).catch(() => {});
    setPhase(PHASE.SCANNING);

    scanTimerRef.current = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current) return;
      try {
        const blob = await captureFrame(videoRef.current, canvasRef.current);
        const fx = getFocalLength(videoRef.current);
        const fd = new FormData();
        fd.append("image", blob, "frame.jpg");
        fd.append("fx", fx);
        fd.append("imu", "[]");
        const res = await fetch(`${API_BASE}/scan_frame`, { method: "POST", body: fd });
        const data = await res.json();
        if (data.frame_count !== undefined) setScanFrames(data.frame_count);
        if (data.status === "ok") setScanStatus(`Captured ${data.frame_count} frames — keep moving around the box`);
        else if (data.reason) setScanStatus(`Waiting: ${data.reason.replace(/_/g, " ")}`);
      } catch (e) {
        console.error("Scan frame error:", e);
      }
    }, SCAN_FRAME_INTERVAL_MS);
  }, []);

  const finishScan = useCallback(async () => {
    clearInterval(scanTimerRef.current);
    setLoading(true);
    setScanStatus("Computing dimensions…");
    try {
      const res = await fetch(`${API_BASE}/compute_dimensions`, { method: "POST" });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data.dimensions);
      setPhase(PHASE.RESULT);
    } catch (e) {
      setError("Scan failed: " + e.message);
      setPhase(PHASE.DETECTING);
      startDetection();
    } finally {
      setLoading(false);
      setScanStatus("");
    }
  }, [startDetection]);

  const resetAll = useCallback(async () => {
    clearInterval(detectTimerRef.current);
    clearInterval(scanTimerRef.current);
    await fetch(`${API_BASE}/reset_calibration`, { method: "POST" }).catch(() => {});
    setPhase(PHASE.IDLE);
    setCalibrated(false);
    setCalFrames(0);
    setCalReady(false);
    setScene([]);
    setResult(null);
    setError(null);
    setMotion(0);
    setScanFrames(0);
    setScanStatus("");
  }, []);

  // ── Motion indicator bar ───────────────────────────────────────────────────
  const motionPct = Math.min(100, (motion / 20) * 100);
  const motionColor =
    motionPct < 30 ? "#00ffcc" : motionPct < 65 ? "#f5c518" : "#ff4444";

  // ── First box with dimensions in scene ─────────────────────────────────────
  const liveBox = scene.find(
    (o) => !o.label.toLowerCase().includes("card") && o.dimensions
  );

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg: #0a0a0c;
          --panel: #111115;
          --border: #1e1e26;
          --accent: #ff6b35;
          --accent2: #00ffcc;
          --text: #e8e8f0;
          --muted: #555568;
          --warn: #f5c518;
          --danger: #ff4444;
        }

        body {
          background: var(--bg);
          color: var(--text);
          font-family: 'DM Mono', monospace;
          min-height: 100dvh;
          overflow: hidden;
        }

        .root {
          display: grid;
          grid-template-rows: 48px 1fr 220px;
          grid-template-columns: 1fr 300px;
          height: 100dvh;
          gap: 0;
        }

        /* ── Header ── */
        .header {
          grid-column: 1 / -1;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 20px;
          border-bottom: 1px solid var(--border);
          background: var(--panel);
        }
        .header-logo {
          font-family: 'Syne', sans-serif;
          font-size: 18px;
          font-weight: 800;
          letter-spacing: -0.5px;
          color: var(--text);
        }
        .header-logo span { color: var(--accent); }
        .phase-badge {
          font-size: 10px;
          font-weight: 500;
          letter-spacing: 2px;
          text-transform: uppercase;
          padding: 3px 10px;
          border-radius: 2px;
          border: 1px solid;
        }
        .phase-badge.idle    { color: var(--muted); border-color: var(--muted); }
        .phase-badge.cal     { color: var(--warn);  border-color: var(--warn);  }
        .phase-badge.detect  { color: var(--accent2); border-color: var(--accent2); }
        .phase-badge.scan    { color: var(--accent); border-color: var(--accent); animation: pulseBorder 1s infinite; }
        .phase-badge.result  { color: var(--accent2); border-color: var(--accent2); }

        @keyframes pulseBorder {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }

        /* ── Viewport ── */
        .viewport {
          position: relative;
          background: #000;
          overflow: hidden;
        }
        .viewport video,
        .viewport canvas.overlay {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
        canvas.overlay { pointer-events: none; }

        .viewport-ui {
          position: absolute;
          inset: 0;
          pointer-events: none;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          padding: 16px;
        }
        .corner-frame {
          position: absolute;
          inset: 0;
          pointer-events: none;
        }
        .corner-frame::before,
        .corner-frame::after,
        .corner-frame > span::before,
        .corner-frame > span::after {
          content: '';
          position: absolute;
          width: 24px;
          height: 24px;
          border-color: rgba(255,107,53,0.4);
          border-style: solid;
        }
        .corner-frame::before { top: 16px; left: 16px; border-width: 2px 0 0 2px; }
        .corner-frame::after  { top: 16px; right: 16px; border-width: 2px 2px 0 0; }
        .corner-frame > span::before { bottom: 16px; left: 16px; border-width: 0 0 2px 2px; position: absolute; }
        .corner-frame > span::after  { bottom: 16px; right: 16px; border-width: 0 2px 2px 0; position: absolute; }

        .motion-bar-wrap {
          display: flex;
          align-items: center;
          gap: 8px;
          pointer-events: none;
          align-self: flex-end;
        }
        .motion-label { font-size: 9px; letter-spacing: 2px; color: var(--muted); text-transform: uppercase; }
        .motion-track {
          width: 80px;
          height: 3px;
          background: var(--border);
          border-radius: 2px;
          overflow: hidden;
        }
        .motion-fill {
          height: 100%;
          border-radius: 2px;
          transition: width 0.3s ease, background 0.3s ease;
        }

        /* Scan progress arc */
        .scan-overlay {
          position: absolute;
          inset: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-direction: column;
          gap: 12px;
          pointer-events: none;
        }
        .scan-ring {
          width: 100px;
          height: 100px;
        }
        .scan-ring-track { fill: none; stroke: var(--border); stroke-width: 3; }
        .scan-ring-fill  {
          fill: none;
          stroke: var(--accent);
          stroke-width: 3;
          stroke-linecap: round;
          stroke-dasharray: 283;
          transition: stroke-dashoffset 0.4s ease;
        }
        .scan-count {
          font-family: 'Syne', sans-serif;
          font-size: 26px;
          font-weight: 800;
          fill: var(--text);
        }
        .scan-sub { font-size: 9px; fill: var(--muted); letter-spacing: 1px; }
        .scan-hint {
          font-size: 11px;
          color: var(--accent);
          letter-spacing: 1px;
          background: rgba(255,107,53,0.08);
          padding: 6px 14px;
          border: 1px solid rgba(255,107,53,0.2);
          border-radius: 2px;
          max-width: 260px;
          text-align: center;
        }

        /* Cal guide */
        .cal-guide {
          position: absolute;
          inset: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          pointer-events: none;
        }
        .cal-card-outline {
          width: 220px;
          height: 139px;
          border: 2px dashed rgba(0,255,204,0.35);
          border-radius: 8px;
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .cal-card-label {
          font-size: 10px;
          color: rgba(0,255,204,0.5);
          letter-spacing: 2px;
          text-transform: uppercase;
        }

        /* ── Right panel ── */
        .panel {
          grid-row: 2 / 4;
          background: var(--panel);
          border-left: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          overflow-y: auto;
        }
        .panel-section {
          padding: 16px;
          border-bottom: 1px solid var(--border);
        }
        .panel-label {
          font-size: 9px;
          letter-spacing: 2.5px;
          text-transform: uppercase;
          color: var(--muted);
          margin-bottom: 12px;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .panel-label::after {
          content: '';
          flex: 1;
          height: 1px;
          background: var(--border);
        }

        /* Status tile */
        .status-tile {
          background: var(--bg);
          border: 1px solid var(--border);
          border-radius: 4px;
          padding: 12px;
          font-size: 11px;
          line-height: 1.7;
        }
        .status-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .status-key { color: var(--muted); }
        .status-val { color: var(--text); }
        .status-val.ok   { color: var(--accent2); }
        .status-val.warn { color: var(--warn); }
        .status-val.bad  { color: var(--danger); }

        /* Cal progress */
        .cal-progress {
          margin-top: 10px;
        }
        .cal-bar-track {
          height: 4px;
          background: var(--border);
          border-radius: 2px;
          overflow: hidden;
          margin-top: 4px;
        }
        .cal-bar-fill {
          height: 100%;
          background: var(--warn);
          border-radius: 2px;
          transition: width 0.4s ease;
        }
        .cal-bar-fill.ready { background: var(--accent2); }
        .cal-bar-label {
          font-size: 10px;
          color: var(--muted);
          margin-top: 4px;
          display: flex;
          justify-content: space-between;
        }

        /* Dim cards */
        .dim-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 6px;
          margin-top: 2px;
        }
        .dim-card {
          background: var(--bg);
          border: 1px solid var(--border);
          border-radius: 4px;
          padding: 10px 12px;
        }
        .dim-card.full { grid-column: 1 / -1; border-color: var(--accent); }
        .dim-card-label { font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }
        .dim-card-value { font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 800; color: var(--text); line-height: 1; }
        .dim-card-unit  { font-size: 10px; color: var(--muted); margin-top: 2px; font-family: 'DM Mono', monospace; }
        .dim-card.full .dim-card-value { font-size: 24px; color: var(--accent2); }

        /* Buttons */
        .btn {
          display: block;
          width: 100%;
          padding: 11px 16px;
          border: none;
          border-radius: 3px;
          cursor: pointer;
          font-family: 'DM Mono', monospace;
          font-size: 11px;
          font-weight: 500;
          letter-spacing: 1.5px;
          text-transform: uppercase;
          transition: opacity 0.15s, transform 0.1s;
        }
        .btn:active { transform: scale(0.98); }
        .btn:disabled { opacity: 0.35; cursor: not-allowed; }
        .btn-primary { background: var(--accent); color: #fff; }
        .btn-primary:hover:not(:disabled) { opacity: 0.88; }
        .btn-secondary { background: transparent; color: var(--accent2); border: 1px solid var(--accent2); }
        .btn-secondary:hover:not(:disabled) { background: rgba(0,255,204,0.06); }
        .btn-ghost { background: transparent; color: var(--muted); border: 1px solid var(--border); }
        .btn-ghost:hover:not(:disabled) { color: var(--text); border-color: var(--muted); }
        .btn-danger { background: transparent; color: var(--danger); border: 1px solid rgba(255,68,68,0.3); }
        .btn-danger:hover:not(:disabled) { background: rgba(255,68,68,0.06); }
        .btn + .btn { margin-top: 6px; }

        /* Error */
        .error-banner {
          background: rgba(255,68,68,0.08);
          border: 1px solid rgba(255,68,68,0.25);
          border-radius: 3px;
          padding: 10px 12px;
          font-size: 11px;
          color: var(--danger);
          line-height: 1.5;
          margin-bottom: 8px;
        }

        /* Result overlay on viewport */
        .result-viewport {
          position: absolute;
          bottom: 16px;
          left: 16px;
          background: rgba(10,10,12,0.85);
          border: 1px solid var(--accent2);
          border-radius: 4px;
          padding: 12px 16px;
          backdrop-filter: blur(8px);
          pointer-events: none;
        }
        .result-viewport-title {
          font-size: 9px;
          letter-spacing: 2.5px;
          text-transform: uppercase;
          color: var(--accent2);
          margin-bottom: 8px;
        }
        .result-dims {
          display: flex;
          gap: 16px;
          align-items: baseline;
        }
        .result-dim {
          font-family: 'Syne', sans-serif;
          font-size: 22px;
          font-weight: 800;
          color: var(--text);
        }
        .result-sep {
          font-size: 18px;
          color: var(--muted);
        }

        /* Bottom controls */
        .controls {
          grid-column: 1;
          background: var(--panel);
          border-top: 1px solid var(--border);
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 0 20px;
        }
        .ctrl-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 16px;
          border-radius: 3px;
          font-family: 'DM Mono', monospace;
          font-size: 11px;
          font-weight: 500;
          letter-spacing: 1.5px;
          text-transform: uppercase;
          cursor: pointer;
          border: none;
          transition: opacity 0.15s, transform 0.1s;
          white-space: nowrap;
        }
        .ctrl-btn:active { transform: scale(0.97); }
        .ctrl-btn:disabled { opacity: 0.3; cursor: not-allowed; }
        .ctrl-btn-primary { background: var(--accent); color: #fff; }
        .ctrl-btn-primary:hover:not(:disabled) { opacity: 0.88; }
        .ctrl-btn-secondary { background: transparent; color: var(--accent2); border: 1px solid var(--accent2); }
        .ctrl-btn-secondary:hover:not(:disabled) { background: rgba(0,255,204,0.06); }
        .ctrl-btn-warn { background: transparent; color: var(--warn); border: 1px solid var(--warn); }
        .ctrl-btn-warn:hover:not(:disabled) { background: rgba(245,197,24,0.06); }
        .ctrl-btn-ghost { background: transparent; color: var(--muted); border: 1px solid var(--border); }
        .ctrl-btn-ghost:hover:not(:disabled) { color: var(--text); }

        .ctrl-sep { flex: 1; }

        .indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--muted);
        }
        .indicator.on { background: var(--accent2); box-shadow: 0 0 6px var(--accent2); animation: blink 1.5s infinite; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }

        /* Loading spinner */
        .spinner {
          width: 14px; height: 14px;
          border: 2px solid transparent;
          border-top-color: currentColor;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
          display: inline-block;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Object list */
        .obj-list { display: flex; flex-direction: column; gap: 6px; margin-top: 2px; }
        .obj-item {
          background: var(--bg);
          border: 1px solid var(--border);
          border-radius: 3px;
          padding: 8px 10px;
          font-size: 10px;
        }
        .obj-item-header { display: flex; justify-content: space-between; margin-bottom: 4px; }
        .obj-item-label { color: var(--text); font-weight: 500; }
        .obj-item-conf { color: var(--muted); }
        .obj-item-dims { color: var(--accent); font-size: 11px; }

        @media (max-width: 600px) {
          .root {
            grid-template-rows: 44px 1fr auto auto;
            grid-template-columns: 1fr;
          }
          .panel {
            grid-row: 3;
            border-left: none;
            border-top: 1px solid var(--border);
            max-height: 40vh;
          }
          .controls { grid-column: 1; grid-row: 4; flex-wrap: wrap; padding: 12px; gap: 8px; }
        }
      `}</style>

      <div className="root">
        {/* ── Header ── */}
        <header className="header">
          <div className="header-logo">DIM<span>SCAN</span></div>
          <div
            className={`phase-badge ${
              phase === PHASE.IDLE ? "idle" :
              phase === PHASE.CALIBRATING ? "cal" :
              phase === PHASE.DETECTING ? "detect" :
              phase === PHASE.SCANNING ? "scan" : "result"
            }`}
          >
            {phase === PHASE.IDLE ? "Ready" :
             phase === PHASE.CALIBRATING ? "Calibrating" :
             phase === PHASE.DETECTING ? "Live Detect" :
             phase === PHASE.SCANNING ? "Scanning" : "Result"}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "10px", color: "var(--muted)" }}>
            <div className={`indicator ${phase === PHASE.DETECTING || phase === PHASE.SCANNING ? "on" : ""}`} />
            {calibrated ? "Calibrated" : "Uncalibrated"}
          </div>
        </header>

        {/* ── Viewport ── */}
        <div className="viewport">
          <video ref={videoRef} playsInline muted />
          <canvas className="overlay" ref={overlayRef} />

          {/* Corner frame decoration */}
          <div className="corner-frame"><span /></div>

          {/* Calibration card guide */}
          {phase === PHASE.CALIBRATING && (
            <div className="cal-guide">
              <div className="cal-card-outline">
                <span className="cal-card-label">Place Card Here</span>
              </div>
            </div>
          )}

          {/* Scan overlay */}
          {phase === PHASE.SCANNING && (
            <div className="scan-overlay">
              <svg className="scan-ring" viewBox="0 0 100 100">
                <circle className="scan-ring-track" cx="50" cy="50" r="45" />
                <circle
                  className="scan-ring-fill"
                  cx="50" cy="50" r="45"
                  style={{
                    strokeDashoffset: 283 - (Math.min(scanFrames, 20) / 20) * 283,
                    transform: "rotate(-90deg)",
                    transformOrigin: "50% 50%",
                  }}
                />
                <text className="scan-count" x="50" y="54" textAnchor="middle">{scanFrames}</text>
                <text className="scan-sub" x="50" y="66" textAnchor="middle">FRAMES</text>
              </svg>
              {scanStatus && <div className="scan-hint">{scanStatus}</div>}
            </div>
          )}

          {/* Live result overlay in detect mode */}
          {phase === PHASE.DETECTING && liveBox && (
            <div className="result-viewport">
              <div className="result-viewport-title">Live Dimensions</div>
              <div className="result-dims">
                <span className="result-dim">{fmtM(liveBox.dimensions.length)}</span>
                <span className="result-sep">×</span>
                <span className="result-dim">{fmtM(liveBox.dimensions.width)}</span>
                <span className="result-sep">×</span>
                <span className="result-dim">{fmtM(liveBox.dimensions.height)}</span>
              </div>
            </div>
          )}

          {/* Final result overlay */}
          {phase === PHASE.RESULT && result && (
            <div className="result-viewport" style={{ borderColor: "var(--accent2)" }}>
              <div className="result-viewport-title" style={{ color: "var(--accent)" }}>Scan Result</div>
              <div className="result-dims">
                <span className="result-dim">{fmtM(result.length)}</span>
                <span className="result-sep">×</span>
                <span className="result-dim">{fmtM(result.width)}</span>
                <span className="result-sep">×</span>
                <span className="result-dim">{fmtM(result.height)}</span>
              </div>
            </div>
          )}

          {/* Motion bar */}
          <div className="viewport-ui">
            <div />
            <div className="motion-bar-wrap">
              <span className="motion-label">Motion</span>
              <div className="motion-track">
                <div className="motion-fill" style={{ width: `${motionPct}%`, background: motionColor }} />
              </div>
            </div>
          </div>
        </div>

        {/* ── Right panel ── */}
        <aside className="panel">
          {/* Status */}
          <div className="panel-section">
            <div className="panel-label">System Status</div>
            {error && <div className="error-banner">{error}</div>}
            <div className="status-tile">
              <div className="status-row">
                <span className="status-key">Camera</span>
                <span className={`status-val ${cameraReady ? "ok" : "warn"}`}>{cameraReady ? "Ready" : "Init…"}</span>
              </div>
              <div className="status-row">
                <span className="status-key">Calibration</span>
                <span className={`status-val ${calibrated ? "ok" : "warn"}`}>{calibrated ? "Locked" : "Pending"}</span>
              </div>
              {reproj !== null && (
                <div className="status-row">
                  <span className="status-key">Reproj Error</span>
                  <span className={`status-val ${reproj < 4 ? "ok" : reproj < 8 ? "warn" : "bad"}`}>{reproj.toFixed(1)} px</span>
                </div>
              )}
              <div className="status-row">
                <span className="status-key">Objects</span>
                <span className="status-val">{scene.length}</span>
              </div>
            </div>

            {/* Cal progress */}
            {phase === PHASE.CALIBRATING && (
              <div className="cal-progress">
                <div className="cal-bar-label">
                  <span>Calibration Frames</span>
                  <span>{calFrames} / 10</span>
                </div>
                <div className="cal-bar-track">
                  <div
                    className={`cal-bar-fill${calReady ? " ready" : ""}`}
                    style={{ width: `${Math.min(100, (calFrames / 10) * 100)}%` }}
                  />
                </div>
                {calReady && (
                  <div style={{ fontSize: "10px", color: "var(--accent2)", marginTop: "6px" }}>
                    ✓ Calibration complete — finalizing…
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Dimensions */}
          {(liveBox || result) && (
            <div className="panel-section">
              <div className="panel-label">Dimensions</div>
              <div className="dim-grid">
                <div className="dim-card">
                  <div className="dim-card-label">Length</div>
                  <div className="dim-card-value">{((result || liveBox.dimensions).length * 100).toFixed(1)}</div>
                  <div className="dim-card-unit">cm</div>
                </div>
                <div className="dim-card">
                  <div className="dim-card-label">Width</div>
                  <div className="dim-card-value">{((result || liveBox.dimensions).width * 100).toFixed(1)}</div>
                  <div className="dim-card-unit">cm</div>
                </div>
                <div className="dim-card">
                  <div className="dim-card-label">Height</div>
                  <div className="dim-card-value">{((result || liveBox.dimensions).height * 100).toFixed(1)}</div>
                  <div className="dim-card-unit">cm</div>
                </div>
                <div className="dim-card full">
                  <div className="dim-card-label">Volume</div>
                  <div className="dim-card-value">{fmtVol((result || liveBox.dimensions).volume_m3)}</div>
                  <div className="dim-card-unit">cubic centimetres</div>
                </div>
              </div>
            </div>
          )}

          {/* Scene objects */}
          {scene.length > 0 && phase === PHASE.DETECTING && (
            <div className="panel-section">
              <div className="panel-label">Detections</div>
              <div className="obj-list">
                {scene.map((obj) => (
                  <div className="obj-item" key={obj.object_id}>
                    <div className="obj-item-header">
                      <span className="obj-item-label">#{obj.object_id} {obj.label}</span>
                      <span className="obj-item-conf">{(obj.confidence * 100).toFixed(0)}%</span>
                    </div>
                    {obj.dimensions && (
                      <div className="obj-item-dims">
                        {fmtM(obj.dimensions.length)} × {fmtM(obj.dimensions.width)} × {fmtM(obj.dimensions.height)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Panel controls (right side) */}
          <div className="panel-section" style={{ marginTop: "auto" }}>
            {phase === PHASE.IDLE && (
              <button
                className="btn btn-primary"
                disabled={!cameraReady || loading}
                onClick={startCalibration}
              >
                {loading ? <><span className="spinner" /> Working…</> : "▶ Start Calibration"}
              </button>
            )}

            {phase === PHASE.CALIBRATING && (
              <>
                <button
                  className="btn btn-secondary"
                  disabled={calFrames < 3 || loading}
                  onClick={() => {
                    clearInterval(detectTimerRef.current);
                    confirmCalibration(
                      getFocalLength(videoRef.current),
                      videoRef.current.videoWidth,
                      videoRef.current.videoHeight,
                    );
                  }}
                >
                  {loading ? <><span className="spinner" /> Finalizing…</> : "✓ Confirm Calibration"}
                </button>
                <button className="btn btn-ghost" onClick={resetAll}>Cancel</button>
              </>
            )}

            {phase === PHASE.DETECTING && (
              <>
                <button
                  className="btn btn-primary"
                  onClick={startScan}
                >
                  ⟳ Start Multi-Frame Scan
                </button>
                <button className="btn btn-ghost" onClick={resetAll}>Reset All</button>
              </>
            )}

            {phase === PHASE.SCANNING && (
              <>
                <button
                  className="btn btn-secondary"
                  disabled={scanFrames < 3 || loading}
                  onClick={finishScan}
                >
                  {loading ? <><span className="spinner" /> Computing…</> : `✓ Compute (${scanFrames} frames)`}
                </button>
                <button
                  className="btn btn-ghost"
                  onClick={() => {
                    clearInterval(scanTimerRef.current);
                    setPhase(PHASE.DETECTING);
                    startDetection();
                  }}
                >
                  Cancel Scan
                </button>
              </>
            )}

            {phase === PHASE.RESULT && (
              <>
                <button className="btn btn-primary" onClick={startScan}>
                  ⟳ Rescan
                </button>
                <button className="btn btn-ghost" onClick={() => { setPhase(PHASE.DETECTING); startDetection(); }}>
                  ← Back to Live
                </button>
                <button className="btn btn-danger" onClick={resetAll}>Reset All</button>
              </>
            )}
          </div>
        </aside>

        {/* ── Bottom controls ── */}
        <div className="controls">
          {phase === PHASE.IDLE && (
            <button
              className="ctrl-btn ctrl-btn-primary"
              disabled={!cameraReady || loading}
              onClick={startCalibration}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>
              Calibrate with Card
            </button>
          )}

          {phase === PHASE.DETECTING && (
            <>
              <button className="ctrl-btn ctrl-btn-secondary" onClick={startScan}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
                Multi-Frame Scan
              </button>
              <div className="ctrl-sep" />
              <button className="ctrl-btn ctrl-btn-ghost" onClick={resetAll}>Reset</button>
            </>
          )}

          {phase === PHASE.SCANNING && (
            <>
              <div style={{ fontSize: "10px", color: "var(--muted)", letterSpacing: "1px" }}>
                Move slowly around the box
              </div>
              <div className="ctrl-sep" />
              <button
                className="ctrl-btn ctrl-btn-warn"
                disabled={scanFrames < 3 || loading}
                onClick={finishScan}
              >
                {loading ? <><span className="spinner" /> Processing</> : `Finish (${scanFrames} frames)`}
              </button>
            </>
          )}

          {phase === PHASE.RESULT && (
            <>
              <div style={{ fontSize: "10px", color: "var(--accent2)", letterSpacing: "1px" }}>
                L {fmtM(result?.length)} × W {fmtM(result?.width)} × H {fmtM(result?.height)} — Vol {fmtVol(result?.volume_m3)}
              </div>
              <div className="ctrl-sep" />
              <button className="ctrl-btn ctrl-btn-primary" onClick={startScan}>Rescan</button>
              <button className="ctrl-btn ctrl-btn-ghost" onClick={resetAll}>Reset</button>
            </>
          )}

          {phase === PHASE.CALIBRATING && (
            <>
              <div style={{ fontSize: "10px", color: "var(--warn)", letterSpacing: "1px" }}>
                Hold card flat in frame — {calFrames}/10 frames
              </div>
              <div className="ctrl-sep" />
              {calFrames >= 3 && (
                <button
                  className="ctrl-btn ctrl-btn-secondary"
                  disabled={loading}
                  onClick={() => {
                    clearInterval(detectTimerRef.current);
                    confirmCalibration(
                      getFocalLength(videoRef.current),
                      videoRef.current.videoWidth,
                      videoRef.current.videoHeight,
                    );
                  }}
                >
                  Confirm
                </button>
              )}
              <button className="ctrl-btn ctrl-btn-ghost" onClick={resetAll}>Cancel</button>
            </>
          )}
        </div>
      </div>

      <canvas ref={canvasRef} style={{ display: "none" }} />
    </>
  );
}