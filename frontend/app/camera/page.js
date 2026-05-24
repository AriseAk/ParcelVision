"use client";

import { useRef, useEffect, useState, useCallback } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
const CAL_INTERVAL_MS = 700;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function getFx(videoEl) {
  const w = videoEl.videoWidth || 1280;
  return (w / 2) / Math.tan((65 * Math.PI) / 180 / 2);
}

async function captureBlob(videoEl, canvasEl, quality = 0.88) {
  const w = videoEl.videoWidth;
  const h = videoEl.videoHeight;
  canvasEl.width = w;
  canvasEl.height = h;
  canvasEl.getContext("2d").drawImage(videoEl, 0, 0, w, h);
  return new Promise((res) => canvasEl.toBlob((b) => res(b), "image/jpeg", quality));
}

function cm(m) {
  if (m == null) return "—";
  return `${(m * 100).toFixed(1)}`;
}

function vol(v) {
  if (v == null) return "—";
  return `${(v * 1e6).toFixed(0)}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Phases
// ─────────────────────────────────────────────────────────────────────────────
const PHASE = {
  CALIBRATE: "CALIBRATE",   // full-screen calibration page
  FRONT:     "FRONT",       // tap 3 corners of front face
  TURN:      "TURN",        // instruction: rotate box 90°
  SIDE:      "SIDE",        // tap 3 corners of side face
  RESULT:    "RESULT",      // show dimensions
};

const CORNER_LABELS = {
  bl: { label: "Bottom Left", color: "#ff6b35", hint: "Where the left vertical edge meets the floor" },
  br: { label: "Bottom Right", color: "#f5c518", hint: "Where the right vertical edge meets the floor" },
  tl: { label: "Top Left", color: "#00ffcc", hint: "The top of the left vertical edge" },
};

const TAP_ORDER = ["bl", "br", "tl"];

// ─────────────────────────────────────────────────────────────────────────────
// Calibration Page (full screen)
// ─────────────────────────────────────────────────────────────────────────────
function CalibrationPage({ videoRef, canvasRef, onCalibrated, onError }) {
  const [calFrames, setCalFrames] = useState(0);
  const [reproj, setReproj] = useState(null);
  const [cardVisible, setCardVisible] = useState(false);
  const [status, setStatus] = useState("Hold your credit card flat on the floor");
  const [locked, setLocked] = useState(false);
  const [loading, setLoading] = useState(false);
  const timerRef = useRef(null);
  const overlayRef = useRef(null);

  useEffect(() => {
    timerRef.current = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current || locked) return;
      try {
        const blob = await captureBlob(videoRef.current, canvasRef.current);
        const fx = getFx(videoRef.current);
        const imgW = videoRef.current.videoWidth;
        const imgH = videoRef.current.videoHeight;

        const fd = new FormData();
        fd.append("image", blob, "frame.jpg");
        fd.append("fx", fx);
        fd.append("img_w", imgW);
        fd.append("img_h", imgH);

        // First detect card position for overlay feedback
        const detRes = await fetch(`${API_BASE}/detect_card`, { method: "POST", body: fd });
        const detData = await detRes.json();

        if (detData.cards && detData.cards.length > 0) {
          const card = detData.cards[0];
          setCardVisible(card.aspect_ok);
          drawCardOverlay(overlayRef.current, card.bbox, card.aspect_ok, imgW, imgH);

          if (card.aspect_ok) {
            setStatus("Card detected! Hold still…");
            const fd2 = new FormData();
            const blob2 = await captureBlob(videoRef.current, canvasRef.current);
            fd2.append("image", blob2, "frame.jpg");
            fd2.append("fx", fx);
            fd2.append("img_w", imgW);
            fd2.append("img_h", imgH);
            fd2.append("pixel_w", card.pixel_w);
            fd2.append("pixel_h", card.pixel_h);
            fd2.append("bbox_x", card.bbox[0]);
            fd2.append("bbox_y", card.bbox[1]);

            const calRes = await fetch(`${API_BASE}/calibrate_frame`, { method: "POST", body: fd2 });
            const calData = await calRes.json();

            if (calData.count !== undefined) setCalFrames(calData.count);
            if (calData.reproj_error !== undefined) setReproj(calData.reproj_error);
            if (calData.locked || calData.ready) {
              clearInterval(timerRef.current);
              setLocked(true);
              setStatus("Calibration complete!");
              setTimeout(() => onCalibrated(), 800);
            } else {
              setStatus(`Capturing… ${calData.count}/10 frames`);
            }
          } else {
            setStatus(`Card found but ${card.aspect_reason}. Reposition.`);
          }
        } else {
          setCardVisible(false);
          clearCardOverlay(overlayRef.current);
          setStatus("Point camera at your credit card on the floor");
        }
      } catch (e) {
        console.error("Cal error:", e);
      }
    }, CAL_INTERVAL_MS);

    return () => clearInterval(timerRef.current);
  }, [locked]);

  const forceConfirm = useCallback(async () => {
    if (calFrames < 3) return;
    clearInterval(timerRef.current);
    setLoading(true);
    try {
      const fx = getFx(videoRef.current);
      const fd = new FormData();
      fd.append("fx", fx);
      fd.append("img_w", videoRef.current.videoWidth);
      fd.append("img_h", videoRef.current.videoHeight);
      const res = await fetch(`${API_BASE}/confirm_calibration`, { method: "POST", body: fd });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      onCalibrated();
    } catch (e) {
      onError("Calibration failed: " + e.message);
    } finally {
      setLoading(false);
    }
  }, [calFrames]);

  const progress = Math.min(100, (calFrames / 10) * 100);

  return (
    <div style={styles.calibPage}>
      {/* Live camera */}
      <div style={styles.calibViewport}>
        <video ref={videoRef} playsInline muted style={styles.calibVideo} />
        <canvas ref={overlayRef} style={styles.calibOverlay} />

        {/* Card placement guide */}
        <div style={styles.cardGuideWrap}>
          <div style={{
            ...styles.cardGuide,
            borderColor: cardVisible ? "rgba(0,255,204,0.7)" : "rgba(255,255,255,0.25)",
            boxShadow: cardVisible ? "0 0 20px rgba(0,255,204,0.3)" : "none",
            transition: "all 0.3s",
          }}>
            <span style={styles.cardGuideLabel}>
              {cardVisible ? "✓ CARD DETECTED" : "PLACE CARD HERE"}
            </span>
          </div>
        </div>

        {/* Lock indicator */}
        {locked && (
          <div style={styles.lockBanner}>
            <span style={{ fontSize: 28 }}>✓</span>
            <span style={{ fontSize: 14, fontWeight: 700, letterSpacing: 2 }}>CALIBRATED</span>
          </div>
        )}
      </div>

      {/* Bottom sheet */}
      <div style={styles.calibSheet}>
        <div style={styles.calibTitle}>Floor Calibration</div>
        <div style={styles.calibSub}>{status}</div>

        {/* Progress bar */}
        <div style={styles.progressWrap}>
          <div style={styles.progressTrack}>
            <div style={{
              ...styles.progressFill,
              width: `${progress}%`,
              background: calFrames >= 10 ? "#00ffcc" : "#f5c518",
            }} />
          </div>
          <div style={styles.progressLabel}>
            <span>{calFrames} / 10 frames</span>
            {reproj !== null && (
              <span style={{ color: reproj < 5 ? "#00ffcc" : reproj < 10 ? "#f5c518" : "#ff4444" }}>
                reproj {reproj.toFixed(1)}px
              </span>
            )}
          </div>
        </div>

        {/* Steps */}
        <div style={styles.calibSteps}>
          {[
            { n: 1, text: "Place credit card flat on the same floor as the box" },
            { n: 2, text: "Point camera so card fills ~1/4 of the frame" },
            { n: 3, text: "Hold still — auto-captures 10 frames" },
          ].map(({ n, text }) => (
            <div key={n} style={styles.calibStep}>
              <div style={styles.calibStepNum}>{n}</div>
              <div style={styles.calibStepText}>{text}</div>
            </div>
          ))}
        </div>

        <button
          style={{
            ...styles.btn,
            background: calFrames >= 3 ? "#ff6b35" : "#1e1e26",
            color: calFrames >= 3 ? "#fff" : "#555568",
            cursor: calFrames >= 3 ? "pointer" : "not-allowed",
          }}
          disabled={calFrames < 3 || loading}
          onClick={forceConfirm}
        >
          {loading ? "Finalising…" : calFrames >= 3 ? `Confirm (${calFrames} frames)` : `Need ${3 - calFrames} more frames`}
        </button>
      </div>
    </div>
  );
}

function drawCardOverlay(canvas, bbox, ok, imgW, imgH) {
  if (!canvas) return;
  canvas.width = imgW;
  canvas.height = imgH;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, imgW, imgH);
  const [x1, y1, x2, y2] = bbox;
  ctx.strokeStyle = ok ? "#00ffcc" : "#f5c518";
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
}

function clearCardOverlay(canvas) {
  if (!canvas) return;
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}

// ─────────────────────────────────────────────────────────────────────────────
// Corner Tap UI
// ─────────────────────────────────────────────────────────────────────────────
function TapView({ videoRef, canvasRef, faceLabel, onComplete, onBack }) {
  const overlayRef = useRef(null);
  const [tapped, setTapped] = useState({});   // { bl:[x,y], br:[x,y], tl:[x,y] }
  const [measuring, setMeasuring] = useState(false);
  const [error, setError] = useState(null);
  const videoContainerRef = useRef(null);

  // Draw tapped corners on overlay
  useEffect(() => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw crosshairs for each tapped corner
    Object.entries(tapped).forEach(([key, [x, y]]) => {
      const col = CORNER_LABELS[key].color;
      ctx.strokeStyle = col;
      ctx.fillStyle = col;
      ctx.lineWidth = 2;

      // Circle
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, Math.PI * 2);
      ctx.stroke();

      // Cross
      ctx.beginPath();
      ctx.moveTo(x - 18, y); ctx.lineTo(x + 18, y);
      ctx.moveTo(x, y - 18); ctx.lineTo(x, y + 18);
      ctx.stroke();

      // Label
      ctx.font = "bold 13px monospace";
      ctx.fillText(CORNER_LABELS[key].label, x + 16, y - 8);
    });

    // Draw lines between tapped points if we have bl+br
    if (tapped.bl && tapped.br) {
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(tapped.bl[0], tapped.bl[1]);
      ctx.lineTo(tapped.br[0], tapped.br[1]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (tapped.bl && tapped.tl) {
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(tapped.bl[0], tapped.bl[1]);
      ctx.lineTo(tapped.tl[0], tapped.tl[1]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [tapped]);

  const nextCorner = TAP_ORDER.find((k) => !tapped[k]);
  const allTapped = TAP_ORDER.every((k) => tapped[k]);

  const handleTap = useCallback((e) => {
    if (allTapped || measuring) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const video = videoRef.current;
    const scaleX = (video.videoWidth || 1280) / rect.width;
    const scaleY = (video.videoHeight || 720) / rect.height;

    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;

    setTapped((prev) => ({ ...prev, [nextCorner]: [x, y] }));
    setError(null);
  }, [nextCorner, allTapped, measuring]);

  const resetTaps = () => { setTapped({}); setError(null); };

  const submitFace = useCallback(async () => {
    if (!allTapped) return;
    setMeasuring(true);
    setError(null);
    try {
      const fx = getFx(videoRef.current);
      const imgW = videoRef.current.videoWidth;
      const imgH = videoRef.current.videoHeight;

      const body = JSON.stringify({
        corners: {
          bl: tapped.bl,
          br: tapped.br,
          tl: tapped.tl,
        },
        fx,
        img_w: imgW,
        img_h: imgH,
        face: faceLabel,
      });

      const res = await fetch(`${API_BASE}/measure_face`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      onComplete(data);
    } catch (e) {
      setError(e.message);
      setMeasuring(false);
    }
  }, [allTapped, tapped, faceLabel]);

  const cornerInfo = nextCorner ? CORNER_LABELS[nextCorner] : null;

  return (
    <div style={styles.tapPage}>
      {/* Viewport */}
      <div
        ref={videoContainerRef}
        style={styles.tapViewport}
        onClick={handleTap}
        onTouchEnd={handleTap}
      >
        <video ref={videoRef} playsInline muted style={styles.fillVideo} />
        <canvas ref={overlayRef} style={styles.fillOverlay} />

        {/* Corner prompt */}
        {!allTapped && cornerInfo && (
          <div style={{
            ...styles.tapPrompt,
            borderColor: cornerInfo.color,
            background: `${cornerInfo.color}18`,
          }}>
            <div style={{ color: cornerInfo.color, fontWeight: 700, fontSize: 13, letterSpacing: 1 }}>
              TAP → {cornerInfo.label.toUpperCase()}
            </div>
            <div style={{ color: "rgba(255,255,255,0.6)", fontSize: 11, marginTop: 3 }}>
              {cornerInfo.hint}
            </div>
          </div>
        )}

        {allTapped && !measuring && (
          <div style={styles.allTappedBanner}>
            All 3 corners tapped ✓
          </div>
        )}
      </div>

      {/* Bottom sheet */}
      <div style={styles.tapSheet}>
        {/* Face label */}
        <div style={styles.faceLabel}>
          {faceLabel === "front" ? "① Front Face" : "② Side Face"}
          <span style={{ fontSize: 11, color: "#555568", marginLeft: 8, fontWeight: 400 }}>
            {faceLabel === "front" ? "facing you directly" : "after rotating box 90°"}
          </span>
        </div>

        {/* Corner checklist */}
        <div style={styles.checklist}>
          {TAP_ORDER.map((key) => {
            const done = !!tapped[key];
            const active = nextCorner === key;
            const info = CORNER_LABELS[key];
            return (
              <div key={key} style={{
                ...styles.checkItem,
                borderColor: done ? info.color : active ? `${info.color}60` : "#1e1e26",
                background: done ? `${info.color}12` : "transparent",
              }}>
                <div style={{
                  width: 20, height: 20, borderRadius: "50%",
                  background: done ? info.color : "transparent",
                  border: `2px solid ${done ? info.color : active ? info.color : "#333"}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 12, color: "#000", flexShrink: 0,
                }}>
                  {done ? "✓" : ""}
                </div>
                <div>
                  <div style={{ fontSize: 12, color: done ? info.color : active ? "#fff" : "#555568", fontWeight: 600 }}>
                    {info.label}
                  </div>
                  <div style={{ fontSize: 10, color: "#555568" }}>{info.hint}</div>
                </div>
              </div>
            );
          })}
        </div>

        {error && <div style={styles.errorBox}>{error}</div>}

        <div style={styles.btnRow}>
          <button style={styles.btnGhost} onClick={onBack}>← Back</button>
          {Object.keys(tapped).length > 0 && (
            <button style={styles.btnGhost} onClick={resetTaps}>Reset Taps</button>
          )}
          <button
            style={{
              ...styles.btn,
              flex: 1,
              background: allTapped ? "#ff6b35" : "#1e1e26",
              color: allTapped ? "#fff" : "#555568",
              cursor: allTapped ? "pointer" : "not-allowed",
            }}
            disabled={!allTapped || measuring}
            onClick={submitFace}
          >
            {measuring ? "Measuring…" : allTapped ? "Measure →" : `${Object.keys(tapped).length}/3 tapped`}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Turn Instruction Page
// ─────────────────────────────────────────────────────────────────────────────
function TurnPage({ face1, onContinue }) {
  return (
    <div style={styles.turnPage}>
      <div style={styles.turnContent}>
        <div style={styles.turnIcon}>↻</div>
        <div style={styles.turnTitle}>Now Rotate the Box</div>
        <div style={styles.turnSub}>Turn the box 90° so you can see a different face</div>

        {/* Face 1 result preview */}
        <div style={styles.facePreview}>
          <div style={styles.facePreviewLabel}>Front face measured ✓</div>
          <div style={styles.faceDims}>
            <span>{cm(face1.width_m)} cm</span>
            <span style={{ color: "#555568", margin: "0 8px" }}>×</span>
            <span>{cm(face1.height_m)} cm</span>
          </div>
          <div style={{ fontSize: 10, color: "#555568", marginTop: 4 }}>
            width × height — residual {face1.residual_mm?.toFixed(1)}mm
          </div>
        </div>

        <div style={styles.turnInstructions}>
          {[
            "Keep the box on the same floor",
            "Rotate it 90° toward you",
            "You should now see the narrower or wider side",
            "Keep your credit card visible if possible",
          ].map((t, i) => (
            <div key={i} style={styles.turnStep}>
              <div style={styles.turnStepDot} />
              <span>{t}</span>
            </div>
          ))}
        </div>

        <button style={{ ...styles.btn, background: "#ff6b35", color: "#fff" }} onClick={onContinue}>
          Done — Tap Side Face →
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Result Page
// ─────────────────────────────────────────────────────────────────────────────
function ResultPage({ result, face1, face2, onRescan, onReset }) {
  const consistency = result.height_consistency_mm;
  const qualityOk = consistency < 15;

  return (
    <div style={styles.resultPage}>
      <div style={styles.resultHeader}>
        <div style={styles.resultTitle}>Measurement</div>
        <div style={{
          fontSize: 11, letterSpacing: 2, padding: "4px 10px",
          border: `1px solid ${qualityOk ? "#00ffcc" : "#f5c518"}`,
          color: qualityOk ? "#00ffcc" : "#f5c518",
          borderRadius: 2,
        }}>
          {qualityOk ? "HIGH ACCURACY" : "REVIEW"}
        </div>
      </div>

      {/* Big dims */}
      <div style={styles.bigDims}>
        <DimCard label="Length" value={cm(result.length_m)} unit="cm" accent="#ff6b35" />
        <DimCard label="Width"  value={cm(result.width_m)}  unit="cm" accent="#f5c518" />
        <DimCard label="Height" value={cm(result.height_m)} unit="cm" accent="#00ffcc" />
      </div>

      {/* Volume */}
      <div style={styles.volumeCard}>
        <div style={{ fontSize: 10, letterSpacing: 2, color: "#555568" }}>VOLUME</div>
        <div style={styles.volumeNum}>{vol(result.volume_m3)}</div>
        <div style={{ fontSize: 11, color: "#555568" }}>cm³</div>
      </div>

      {/* Quality indicators */}
      <div style={styles.qualityRow}>
        <QualBadge label="Height consistency" value={`${consistency?.toFixed(1)}mm`} good={qualityOk} />
        <QualBadge label="Face 1 residual" value={`${result.face1_residual_mm?.toFixed(1)}mm`} good={result.face1_residual_mm < 10} />
        <QualBadge label="Face 2 residual" value={`${result.face2_residual_mm?.toFixed(1)}mm`} good={result.face2_residual_mm < 10} />
      </div>

      {/* Per-face breakdown */}
      <div style={styles.faceBreakdown}>
        <div style={styles.faceRow}>
          <span style={{ color: "#555568", fontSize: 11 }}>Front face</span>
          <span style={{ fontSize: 12 }}>{cm(face1.width_m)} × {cm(face1.height_m)} cm</span>
        </div>
        <div style={styles.faceRow}>
          <span style={{ color: "#555568", fontSize: 11 }}>Side face</span>
          <span style={{ fontSize: 12 }}>{cm(face2.width_m)} × {cm(face2.height_m)} cm</span>
        </div>
      </div>

      {!qualityOk && (
        <div style={styles.warningBox}>
          ⚠ Height inconsistency {consistency?.toFixed(0)}mm — retap corners more precisely or rescan
        </div>
      )}

      <div style={styles.btnRow}>
        <button style={{ ...styles.btn, flex: 1, background: "#ff6b35", color: "#fff" }} onClick={onRescan}>
          ⟳ Rescan
        </button>
        <button style={{ ...styles.btnGhost }} onClick={onReset}>
          Reset All
        </button>
      </div>
    </div>
  );
}

function DimCard({ label, value, unit, accent }) {
  return (
    <div style={{ ...styles.dimCard, borderColor: `${accent}40` }}>
      <div style={{ fontSize: 9, letterSpacing: 2, color: "#555568", textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 32, fontWeight: 800, color: accent, lineHeight: 1 }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: "#555568" }}>{unit}</div>
    </div>
  );
}

function QualBadge({ label, value, good }) {
  return (
    <div style={{
      background: "#0a0a0c",
      border: `1px solid ${good ? "#1e3a2a" : "#3a2a1a"}`,
      borderRadius: 4,
      padding: "6px 10px",
      flex: 1,
    }}>
      <div style={{ fontSize: 9, color: "#555568", letterSpacing: 1 }}>{label}</div>
      <div style={{ fontSize: 12, color: good ? "#00ffcc" : "#f5c518", marginTop: 2 }}>{value}</div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────
export default function DimScan() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [phase, setPhase] = useState(PHASE.CALIBRATE);
  const [face1, setFace1] = useState(null);
  const [face2, setFace2] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);

  // Camera init
  useEffect(() => {
    let stream;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "environment",
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            setCameraReady(true);
          };
        }
      } catch (e) {
        setError("Camera access denied: " + e.message);
      }
    })();
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, []);

  const handleCalibrated = useCallback(() => {
    setPhase(PHASE.FRONT);
  }, []);

  const handleFace1 = useCallback((data) => {
    setFace1(data);
    setPhase(PHASE.TURN);
  }, []);

  const handleFace2 = useCallback(async (data) => {
    setFace2(data);
    // Compute volume
    try {
      const res = await fetch(`${API_BASE}/compute_volume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ face1, face2: data }),
      });
      const vol = await res.json();
      if (vol.error) throw new Error(vol.error);
      setResult(vol);
      setPhase(PHASE.RESULT);
    } catch (e) {
      setError("Volume compute failed: " + e.message);
      setPhase(PHASE.SIDE);
    }
  }, [face1]);

  const resetAll = useCallback(async () => {
    await fetch(`${API_BASE}/reset_calibration`, { method: "POST" }).catch(() => {});
    setFace1(null);
    setFace2(null);
    setResult(null);
    setError(null);
    setPhase(PHASE.CALIBRATE);
  }, []);

  const rescan = useCallback(() => {
    setFace1(null);
    setFace2(null);
    setResult(null);
    setPhase(PHASE.FRONT);
  }, []);

  if (!cameraReady && !error) {
    return (
      <div style={styles.splash}>
        <div style={styles.splashLogo}>DIM<span style={{ color: "#ff6b35" }}>SCAN</span></div>
        <div style={styles.splashSub}>Initialising camera…</div>
        <div style={styles.splashSpinner} />
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.splash}>
        <div style={styles.splashLogo}>DIM<span style={{ color: "#ff6b35" }}>SCAN</span></div>
        <div style={{ color: "#ff4444", fontSize: 14, padding: "0 24px", textAlign: "center" }}>{error}</div>
        <button style={{ ...styles.btn, marginTop: 24, background: "#ff6b35", color: "#fff", width: 200 }} onClick={() => window.location.reload()}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Space+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
        html, body { background: #0a0a0c; color: #e8e8f0; height: 100%; overflow: hidden; }
        :root { color-scheme: dark; }
      `}</style>

      {/* Hidden canvas for capture */}
      <canvas ref={canvasRef} style={{ display: "none" }} />

      {/* Phase router */}
      {phase === PHASE.CALIBRATE && (
        <CalibrationPage
          videoRef={videoRef}
          canvasRef={canvasRef}
          onCalibrated={handleCalibrated}
          onError={setError}
        />
      )}

      {phase === PHASE.FRONT && (
        <TapView
          videoRef={videoRef}
          canvasRef={canvasRef}
          faceLabel="front"
          onComplete={handleFace1}
          onBack={() => setPhase(PHASE.CALIBRATE)}
        />
      )}

      {phase === PHASE.TURN && (
        <TurnPage
          face1={face1}
          onContinue={() => setPhase(PHASE.SIDE)}
        />
      )}

      {phase === PHASE.SIDE && (
        <TapView
          videoRef={videoRef}
          canvasRef={canvasRef}
          faceLabel="side"
          onComplete={handleFace2}
          onBack={() => setPhase(PHASE.TURN)}
        />
      )}

      {phase === PHASE.RESULT && result && (
        <ResultPage
          result={result}
          face1={face1}
          face2={face2}
          onRescan={rescan}
          onReset={resetAll}
        />
      )}
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Styles (inline, mobile-first)
// ─────────────────────────────────────────────────────────────────────────────
const styles = {
  // Splash
  splash: {
    display: "flex", flexDirection: "column", alignItems: "center",
    justifyContent: "center", height: "100dvh", gap: 16,
    background: "#0a0a0c", fontFamily: "'Space Mono', monospace",
  },
  splashLogo: {
    fontFamily: "'Syne', sans-serif", fontSize: 36, fontWeight: 800,
    letterSpacing: -1, color: "#e8e8f0",
  },
  splashSub: { fontSize: 12, color: "#555568", letterSpacing: 2 },
  splashSpinner: {
    width: 28, height: 28, borderRadius: "50%",
    border: "2px solid #1e1e26", borderTopColor: "#ff6b35",
    animation: "spin 0.8s linear infinite",
  },

  // Calibration page
  calibPage: {
    display: "flex", flexDirection: "column", height: "100dvh",
    background: "#0a0a0c", fontFamily: "'Space Mono', monospace",
  },
  calibViewport: {
    position: "relative", flex: "0 0 55dvh", background: "#000", overflow: "hidden",
  },
  calibVideo: {
    position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover",
  },
  calibOverlay: {
    position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover",
    pointerEvents: "none",
  },
  cardGuideWrap: {
    position: "absolute", inset: 0, display: "flex",
    alignItems: "center", justifyContent: "center", pointerEvents: "none",
  },
  cardGuide: {
    width: "55%", paddingBottom: "34.7%", position: "relative",
    border: "2px dashed rgba(255,255,255,0.25)", borderRadius: 8,
  },
  cardGuideLabel: {
    position: "absolute", inset: 0, display: "flex", alignItems: "center",
    justifyContent: "center", fontSize: 10, letterSpacing: 2,
    color: "rgba(255,255,255,0.4)", fontFamily: "'Space Mono', monospace",
  },
  lockBanner: {
    position: "absolute", inset: 0, display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center", gap: 8,
    background: "rgba(0,255,204,0.08)", color: "#00ffcc",
  },
  calibSheet: {
    flex: 1, background: "#111115", borderTop: "1px solid #1e1e26",
    padding: "20px 20px 32px", overflowY: "auto",
    display: "flex", flexDirection: "column", gap: 14,
  },
  calibTitle: {
    fontFamily: "'Syne', sans-serif", fontSize: 22, fontWeight: 800,
    letterSpacing: -0.5, color: "#e8e8f0",
  },
  calibSub: { fontSize: 12, color: "#00ffcc", letterSpacing: 0.5, minHeight: 18 },
  progressWrap: { display: "flex", flexDirection: "column", gap: 4 },
  progressTrack: {
    height: 4, background: "#1e1e26", borderRadius: 2, overflow: "hidden",
  },
  progressFill: {
    height: "100%", borderRadius: 2, transition: "width 0.4s ease, background 0.3s",
  },
  progressLabel: {
    display: "flex", justifyContent: "space-between",
    fontSize: 10, color: "#555568", letterSpacing: 1,
  },
  calibSteps: { display: "flex", flexDirection: "column", gap: 10 },
  calibStep: {
    display: "flex", gap: 12, alignItems: "flex-start",
    background: "#0a0a0c", border: "1px solid #1e1e26",
    borderRadius: 4, padding: "10px 12px",
  },
  calibStepNum: {
    width: 22, height: 22, borderRadius: "50%", background: "#ff6b35",
    color: "#fff", fontSize: 11, fontWeight: 700,
    display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
  },
  calibStepText: { fontSize: 12, color: "#aaa", lineHeight: 1.5 },

  // Tap view
  tapPage: {
    display: "flex", flexDirection: "column", height: "100dvh",
    background: "#0a0a0c", fontFamily: "'Space Mono', monospace",
  },
  tapViewport: {
    position: "relative", flex: "0 0 52dvh", background: "#000",
    overflow: "hidden", cursor: "crosshair", touchAction: "none",
  },
  fillVideo: {
    position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover",
  },
  fillOverlay: {
    position: "absolute", inset: 0, width: "100%", height: "100%",
    objectFit: "cover", pointerEvents: "none",
  },
  tapPrompt: {
    position: "absolute", bottom: 12, left: 12, right: 12,
    background: "rgba(10,10,12,0.8)", border: "1px solid",
    borderRadius: 4, padding: "10px 12px", backdropFilter: "blur(8px)",
    pointerEvents: "none",
  },
  allTappedBanner: {
    position: "absolute", bottom: 12, left: "50%", transform: "translateX(-50%)",
    background: "rgba(0,255,204,0.12)", border: "1px solid #00ffcc",
    color: "#00ffcc", borderRadius: 4, padding: "8px 16px",
    fontSize: 12, letterSpacing: 1, whiteSpace: "nowrap",
  },
  tapSheet: {
    flex: 1, background: "#111115", borderTop: "1px solid #1e1e26",
    padding: "16px 16px 24px", overflowY: "auto",
    display: "flex", flexDirection: "column", gap: 12,
  },
  faceLabel: {
    fontFamily: "'Syne', sans-serif", fontSize: 18, fontWeight: 800,
    color: "#e8e8f0", letterSpacing: -0.3,
  },
  checklist: { display: "flex", flexDirection: "column", gap: 8 },
  checkItem: {
    display: "flex", gap: 12, alignItems: "flex-start",
    border: "1px solid", borderRadius: 4, padding: "10px 12px",
    transition: "all 0.2s",
  },
  errorBox: {
    background: "rgba(255,68,68,0.08)", border: "1px solid rgba(255,68,68,0.3)",
    borderRadius: 4, padding: "10px 12px", fontSize: 12, color: "#ff4444",
  },
  btnRow: { display: "flex", gap: 8, marginTop: "auto" },

  // Turn page
  turnPage: {
    display: "flex", alignItems: "center", justifyContent: "center",
    height: "100dvh", background: "#0a0a0c",
    fontFamily: "'Space Mono', monospace", padding: 24,
  },
  turnContent: {
    display: "flex", flexDirection: "column", gap: 20,
    alignItems: "center", maxWidth: 380, width: "100%",
  },
  turnIcon: {
    fontSize: 64, color: "#ff6b35", fontFamily: "'Syne', sans-serif",
    fontWeight: 800, lineHeight: 1,
  },
  turnTitle: {
    fontFamily: "'Syne', sans-serif", fontSize: 26, fontWeight: 800,
    color: "#e8e8f0", textAlign: "center", letterSpacing: -0.5,
  },
  turnSub: { fontSize: 13, color: "#555568", textAlign: "center", lineHeight: 1.6 },
  facePreview: {
    background: "#111115", border: "1px solid #00ffcc40",
    borderRadius: 6, padding: "14px 18px", width: "100%", textAlign: "center",
  },
  facePreviewLabel: { fontSize: 10, color: "#00ffcc", letterSpacing: 2, marginBottom: 6 },
  faceDims: {
    fontFamily: "'Syne', sans-serif", fontSize: 22, fontWeight: 800, color: "#e8e8f0",
  },
  turnInstructions: { display: "flex", flexDirection: "column", gap: 10, width: "100%" },
  turnStep: {
    display: "flex", gap: 12, alignItems: "center",
    fontSize: 13, color: "#aaa",
  },
  turnStepDot: {
    width: 6, height: 6, borderRadius: "50%",
    background: "#ff6b35", flexShrink: 0,
  },

  // Result page
  resultPage: {
    height: "100dvh", background: "#0a0a0c",
    fontFamily: "'Space Mono', monospace", padding: 20,
    display: "flex", flexDirection: "column", gap: 14, overflowY: "auto",
  },
  resultHeader: {
    display: "flex", alignItems: "center",
    justifyContent: "space-between", paddingTop: 8,
  },
  resultTitle: {
    fontFamily: "'Syne', sans-serif", fontSize: 24, fontWeight: 800,
    color: "#e8e8f0", letterSpacing: -0.5,
  },
  bigDims: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 },
  dimCard: {
    background: "#111115", border: "1px solid",
    borderRadius: 6, padding: "12px 10px",
    display: "flex", flexDirection: "column", gap: 4, alignItems: "center",
  },
  volumeCard: {
    background: "#111115", border: "1px solid #ff6b3530",
    borderRadius: 6, padding: "14px 18px",
    display: "flex", alignItems: "center", gap: 10,
  },
  volumeNum: {
    fontFamily: "'Syne', sans-serif", fontSize: 32, fontWeight: 800,
    color: "#ff6b35", flex: 1,
  },
  qualityRow: { display: "flex", gap: 8 },
  faceBreakdown: {
    background: "#111115", border: "1px solid #1e1e26",
    borderRadius: 6, padding: "12px 14px",
    display: "flex", flexDirection: "column", gap: 8,
  },
  faceRow: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
  },
  warningBox: {
    background: "rgba(245,197,24,0.08)", border: "1px solid rgba(245,197,24,0.3)",
    borderRadius: 4, padding: "10px 12px", fontSize: 11, color: "#f5c518",
  },

  // Shared buttons
  btn: {
    padding: "13px 16px", border: "none", borderRadius: 4,
    fontFamily: "'Space Mono', monospace", fontSize: 12, fontWeight: 700,
    letterSpacing: 1, textTransform: "uppercase", cursor: "pointer",
    transition: "opacity 0.15s",
  },
  btnGhost: {
    padding: "13px 14px", border: "1px solid #1e1e26", borderRadius: 4,
    background: "transparent", color: "#555568",
    fontFamily: "'Space Mono', monospace", fontSize: 11,
    cursor: "pointer", letterSpacing: 1,
  },
};
