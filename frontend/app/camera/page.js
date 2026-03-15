"use client"

import { useEffect, useRef, useState } from "react"

// Each detected class gets a consistent colour
const CLASS_COLORS = {
  sofa:           { r: 99,  g: 102, b: 241 },  // indigo
  chair:          { r: 16,  g: 185, b: 129 },  // emerald
  table:          { r: 245, g: 158, b: 11  },  // amber
  "cardboard box":{ r: 239, g: 68,  b: 68  },  // red
  "carton box":   { r: 239, g: 68,  b: 68  },
  "shipping box": { r: 239, g: 68,  b: 68  },
  package:        { r: 236, g: 72,  b: 153 },  // pink
}

function colorFor(label) {
  const key = label?.toLowerCase()
  return CLASS_COLORS[key] ?? { r: 100, g: 200, b: 255 }
}

export default function CameraPage() {
  const videoRef    = useRef(null)
  const canvasRef   = useRef(null)
  const detectRef   = useRef(false)
  const objectsRef  = useRef([])

  const [objects,   setObjects]   = useState([])
  const [fps,       setFps]       = useState(0)
  const [connected, setConnected] = useState(true)

  // FPS counter
  const fpsFrames = useRef(0)
  const fpsTimer  = useRef(null)

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: 640, height: 480 },
          audio: false,
        })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
        startDetectionLoop()
        startRenderLoop()
        fpsTimer.current = setInterval(() => {
          setFps(fpsFrames.current)
          fpsFrames.current = 0
        }, 1000)
      } catch (err) {
        console.error("Camera error:", err)
      }
    }

    startCamera()
    return () => clearInterval(fpsTimer.current)
  }, [])

  // ── Detection loop: POST frame every 400 ms ──────────────────────────────
  function startDetectionLoop() {
    setInterval(() => {
      if (!detectRef.current) captureAndDetect()
    }, 400)
  }

  async function captureAndDetect() {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    detectRef.current = true

    // Draw the current video frame onto the canvas so we can read it as JPEG
    const tmpCanvas = document.createElement("canvas")
    tmpCanvas.width  = 640
    tmpCanvas.height = 480
    tmpCanvas.getContext("2d").drawImage(video, 0, 0, 640, 480)

    tmpCanvas.toBlob(async (blob) => {
      const formData = new FormData()
      formData.append("image", blob, "frame.jpg")

      try {
        const res  = await fetch("http://127.0.0.1:5000/detect", {
          method: "POST",
          body:   formData,
        })
        const data = await res.json()
        const scene = data.scene ?? []
        objectsRef.current = scene
        setObjects(scene)
        setConnected(true)
      } catch {
        setConnected(false)
      }

      detectRef.current = false
    }, "image/jpeg", 0.8)
  }

  // ── Render loop: draw overlays every animation frame ────────────────────
  function startRenderLoop() {
    function render() {
      drawOverlays(objectsRef.current)
      fpsFrames.current++
      requestAnimationFrame(render)
    }
    requestAnimationFrame(render)
  }

  function drawOverlays(scene) {
    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext("2d")
    canvas.width  = 640
    canvas.height = 480

    // Mirror video frame as background
    ctx.drawImage(video, 0, 0, 640, 480)

    scene.forEach((obj) => {
      if (!obj.bbox) return
      const [x1, y1, x2, y2] = obj.bbox
      const { r, g, b }       = colorFor(obj.label)
      const w = x2 - x1
      const h = y2 - y1

      // ── Mask overlay (filled bbox with transparency) ───────────────────
      ctx.save()
      ctx.globalAlpha = 0.28
      ctx.fillStyle   = `rgb(${r},${g},${b})`
      ctx.fillRect(x1, y1, w, h)
      ctx.restore()

      // ── Bounding box border ────────────────────────────────────────────
      ctx.strokeStyle = `rgb(${r},${g},${b})`
      ctx.lineWidth   = 2
      ctx.strokeRect(x1, y1, w, h)

      // ── Corner accents ─────────────────────────────────────────────────
      const cs = Math.min(16, w * 0.2, h * 0.2)   // corner size
      ctx.lineWidth   = 3
      ctx.strokeStyle = `rgb(${r},${g},${b})`

      // top-left
      ctx.beginPath(); ctx.moveTo(x1, y1 + cs); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cs, y1); ctx.stroke()
      // top-right
      ctx.beginPath(); ctx.moveTo(x2 - cs, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cs); ctx.stroke()
      // bottom-left
      ctx.beginPath(); ctx.moveTo(x1, y2 - cs); ctx.lineTo(x1, y2); ctx.lineTo(x1 + cs, y2); ctx.stroke()
      // bottom-right
      ctx.beginPath(); ctx.moveTo(x2 - cs, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2 - cs, y2); ctx.stroke()

      // ── Label pill ────────────────────────────────────────────────────
      const label     = `${obj.label}  ${(obj.confidence * 100).toFixed(0)}%`
      const fontSize  = 12
      ctx.font        = `600 ${fontSize}px monospace`
      const textW     = ctx.measureText(label).width
      const pillH     = 20
      const pillX     = x1
      const pillY     = Math.max(y1 - pillH - 2, 0)

      ctx.fillStyle   = `rgb(${r},${g},${b})`
      ctx.beginPath()
      ctx.roundRect(pillX, pillY, textW + 12, pillH, 4)
      ctx.fill()

      ctx.fillStyle   = "#ffffff"
      ctx.fillText(label, pillX + 6, pillY + pillH - 5)

      // ── Dimensions + volume (if available) ────────────────────────────
      if (obj.dimensions) {
        const d    = obj.dimensions
        const conf = obj.measurement_confidence
          ? ` conf:${(obj.measurement_confidence * 100).toFixed(0)}%`
          : ""

        const lines = [
          `L ${d.length.toFixed(2)}m  W ${d.width.toFixed(2)}m  H ${d.height.toFixed(2)}m`,
          `Vol: ${d.volume_m3.toFixed(3)} m³${conf}`,
        ]

        ctx.font      = `500 11px monospace`
        const lineH   = 16
        const boxW    = Math.max(...lines.map(l => ctx.measureText(l).width)) + 12
        const boxH    = lines.length * lineH + 6
        const bx      = x1
        const by      = y2 + 4

        // background
        ctx.fillStyle   = `rgba(0,0,0,0.65)`
        ctx.beginPath()
        ctx.roundRect(bx, by, boxW, boxH, 4)
        ctx.fill()

        // border accent
        ctx.strokeStyle = `rgba(${r},${g},${b},0.7)`
        ctx.lineWidth   = 1
        ctx.stroke()

        // text
        ctx.fillStyle = "#ffffff"
        lines.forEach((line, i) => {
          ctx.fillText(line, bx + 6, by + lineH * (i + 1) - 2)
        })
      } else {
        // Still measuring indicator
        ctx.font      = "500 10px monospace"
        ctx.fillStyle = "rgba(0,0,0,0.55)"
        ctx.beginPath()
        ctx.roundRect(x1, y2 + 4, 90, 18, 4)
        ctx.fill()
        ctx.fillStyle = `rgba(${r},${g},${b},0.9)`
        ctx.fillText("measuring...", x1 + 6, y2 + 16)
      }
    })
  }

  // ── Sidebar object cards ─────────────────────────────────────────────────
  return (
    <div style={{
      display:         "flex",
      height:          "100vh",
      background:      "#0a0a0f",
      color:           "#e8e8f0",
      fontFamily:      "monospace",
      overflow:        "hidden",
    }}>

      {/* Camera panel */}
      <div style={{
        flex:           "0 0 640px",
        position:       "relative",
        borderRight:    "1px solid #1e1e2e",
      }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ position: "absolute", opacity: 0, pointerEvents: "none" }}
        />
        <canvas
          ref={canvasRef}
          style={{ display: "block", width: "640px", height: "480px" }}
        />

        {/* Status bar */}
        <div style={{
          position:   "absolute",
          bottom:     0,
          left:       0,
          right:      0,
          padding:    "6px 12px",
          background: "rgba(0,0,0,0.7)",
          display:    "flex",
          gap:        "16px",
          fontSize:   "11px",
          color:      "#888",
        }}>
          <span style={{ color: connected ? "#10b981" : "#ef4444" }}>
            ● {connected ? "LIVE" : "DISCONNECTED"}
          </span>
          <span>{fps} fps</span>
          <span>{objects.length} object{objects.length !== 1 ? "s" : ""}</span>
        </div>
      </div>

      {/* Sidebar */}
      <div style={{
        flex:        1,
        overflowY:   "auto",
        padding:     "16px",
        display:     "flex",
        flexDirection: "column",
        gap:         "8px",
      }}>

        <div style={{
          fontSize:     "11px",
          letterSpacing:"0.12em",
          color:        "#555",
          marginBottom: "4px",
          textTransform:"uppercase",
        }}>
          ParcelVision · Scene
        </div>

        {objects.length === 0 && (
          <div style={{ color: "#444", fontSize: "12px", marginTop: "24px", textAlign: "center" }}>
            No objects detected
          </div>
        )}

        {objects.map((obj) => {
          const { r, g, b } = colorFor(obj.label)
          const d = obj.dimensions
          return (
            <div key={obj.object_id} style={{
              background:   "#10101a",
              border:       `1px solid rgba(${r},${g},${b},0.35)`,
              borderRadius: "8px",
              padding:      "12px",
            }}>

              {/* Header row */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <div style={{
                    width:        "8px",
                    height:       "8px",
                    borderRadius: "50%",
                    background:   `rgb(${r},${g},${b})`,
                    flexShrink:   0,
                  }}/>
                  <span style={{ fontSize: "13px", fontWeight: 600, color: "#e8e8f0" }}>
                    {obj.label}
                  </span>
                  <span style={{ fontSize: "10px", color: "#555" }}>
                    #{obj.object_id}
                  </span>
                </div>
                <span style={{
                  fontSize:     "11px",
                  color:        `rgb(${r},${g},${b})`,
                  background:   `rgba(${r},${g},${b},0.1)`,
                  padding:      "2px 7px",
                  borderRadius: "4px",
                }}>
                  {(obj.confidence * 100).toFixed(0)}%
                </span>
              </div>

              {/* Dimensions */}
              {d ? (
                <div style={{ marginTop: "10px" }}>
                  <div style={{
                    display:             "grid",
                    gridTemplateColumns: "1fr 1fr 1fr",
                    gap:                 "6px",
                    marginBottom:        "8px",
                  }}>
                    {[
                      { label: "LENGTH", value: `${d.length.toFixed(2)} m` },
                      { label: "WIDTH",  value: `${d.width.toFixed(2)} m`  },
                      { label: "HEIGHT", value: `${d.height.toFixed(2)} m` },
                    ].map(({ label, value }) => (
                      <div key={label} style={{
                        background:   "#0a0a10",
                        borderRadius: "6px",
                        padding:      "6px 8px",
                        textAlign:    "center",
                      }}>
                        <div style={{ fontSize: "9px", color: "#555", letterSpacing: "0.1em" }}>
                          {label}
                        </div>
                        <div style={{ fontSize: "13px", color: "#e8e8f0", marginTop: "2px" }}>
                          {value}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Volume bar */}
                  <div style={{
                    background:   "#0a0a10",
                    borderRadius: "6px",
                    padding:      "8px 10px",
                    display:      "flex",
                    justifyContent: "space-between",
                    alignItems:   "center",
                  }}>
                    <span style={{ fontSize: "10px", color: "#555", letterSpacing: "0.1em" }}>
                      VOLUME
                    </span>
                    <span style={{ fontSize: "15px", color: `rgb(${r},${g},${b})`, fontWeight: 600 }}>
                      {d.volume_m3.toFixed(3)} m³
                    </span>
                  </div>

                  {/* Confidence bar */}
                  {obj.measurement_confidence != null && (
                    <div style={{ marginTop: "6px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "9px", color: "#555", marginBottom: "3px" }}>
                        <span>MEAS. CONFIDENCE</span>
                        <span>{(obj.measurement_confidence * 100).toFixed(0)}%</span>
                      </div>
                      <div style={{ height: "3px", background: "#1a1a2e", borderRadius: "2px" }}>
                        <div style={{
                          height:       "100%",
                          width:        `${obj.measurement_confidence * 100}%`,
                          background:   `rgb(${r},${g},${b})`,
                          borderRadius: "2px",
                          transition:   "width 0.3s ease",
                        }}/>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ marginTop: "8px", fontSize: "11px", color: "#444" }}>
                  measuring...
                </div>
              )}

            </div>
          )
        })}
      </div>
    </div>
  )
}
