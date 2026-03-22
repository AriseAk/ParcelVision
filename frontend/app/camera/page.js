"use client"

import { useEffect, useRef, useState } from "react"
import ExifReader from "exifreader"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://192.168.1.104:5000"

const CLASS_COLORS = {
  box:             { r: 226, g: 183, b: 20  },
  "cardboard box": { r: 226, g: 183, b: 20  },
  carton:          { r: 226, g: 183, b: 20  },
  parcel:          { r: 100, g: 200, b: 255 },
  package:         { r: 100, g: 200, b: 255 },
  container:       { r: 150, g: 220, b: 160 },
  "brown box":     { r: 200, g: 150, b: 80  },
}

function colorFor(label) {
  return CLASS_COLORS[label?.toLowerCase()] ?? { r: 226, g: 183, b: 20 }
}

// ── Compute fx from EXIF tags ─────────────────────────────────────────────
// Priority: FocalLength + FocalPlaneXResolution (most accurate)
//        → FocalLengthIn35mmFilm with phone crop factor correction
//        → hardcoded fallback
function focalLengthFromExif(tags, imgW = 640) {
  try {
    const focalMm  = tags["FocalLength"]?.value
    const fpxRes   = tags["FocalPlaneXResolution"]?.value
    const fpxUnit  = tags["FocalPlaneResolutionUnit"]?.value

    if (focalMm && fpxRes && fpxUnit) {
      // convert resolution to pixels per mm
      const pxPerMm = fpxUnit === 2 ? fpxRes / 25.4 : fpxRes / 10
      const fx = focalMm * pxPerMm
      return { fx, fy: fx, method: `FocalLength+FPlane fmm=${focalMm} pxPerMm=${pxPerMm.toFixed(1)}` }
    }

    const f35 = tags["FocalLengthIn35mmFilm"]?.value
    if (f35 && f35 > 0) {
      // phone sensor ~5.6mm wide vs 35mm full frame 36mm wide
      // actual_focal_mm = f35 * (5.6 / 36)
      // fx_px = actual_focal_mm * (imgW / 5.6)
      // simplified: fx = f35 * imgW / 36
      // BUT we need to scale by crop factor: fx = (f35 / 36) * imgW * (36/5.6)
      // = f35 * imgW / 5.6
      const fx = (f35 * imgW) / 5.6
      return { fx, fy: fx, method: `f35=${f35}mm crop-corrected` }
    }

    if (focalMm) {
      // raw focal mm, assume phone sensor width 5.6mm
      const fx = (focalMm / 5.6) * imgW
      return { fx, fy: fx, method: `FocalLength=${focalMm}mm sensorW=5.6mm assumed` }
    }
  } catch (e) {}

  // final fallback: 70° FOV
  const fx = 640 / (2 * Math.tan((70 * Math.PI) / 360))
  return { fx, fy: fx, method: "default 70deg FOV" }
}

export default function CameraPage() {
  const videoRef   = useRef(null)
  const canvasRef  = useRef(null)
  const detectRef  = useRef(false)
  const objectsRef = useRef([])
  const loopsRef   = useRef(false)
  const focalRef   = useRef(null)

  const [objects,      setObjects]      = useState([])
  const [fps,          setFps]          = useState(0)
  const [connected,    setConnected]    = useState(false)
  const [isMoving,     setIsMoving]     = useState(false)
  const [motion,       setMotion]       = useState(0)
  const [cameraHeight, setCameraHeight] = useState(null)
  const [focalInfo,    setFocalInfo]    = useState(null)

  const fpsFrames = useRef(0)
  const fpsTimer  = useRef(null)

  useEffect(() => {
    const h = prompt("Enter camera height from ground (meters):", "1.5")
    setCameraHeight(h && !isNaN(h) && parseFloat(h) > 0 ? parseFloat(h) : 1.5)
  }, [])

  useEffect(() => {
    if (cameraHeight === null) return

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: 640, height: 480 },
          audio: false,
        })
        if (!videoRef.current) return
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().catch(() => {})
          if (!loopsRef.current) {
            loopsRef.current = true
            startLoops()
            startRenderLoop()
            fpsTimer.current = setInterval(() => {
              setFps(fpsFrames.current)
              fpsFrames.current = 0
            }, 1000)
          }
        }
      } catch (err) {
        console.error("Camera error:", err)
      }
    }

    startCamera()
    return () => { if (fpsTimer.current) clearInterval(fpsTimer.current) }
  }, [cameraHeight])

  // ── Read EXIF once from first blob, send raw tags to backend for logging ──
  async function tryReadExif(blob) {
    if (focalRef.current) return

    const exifDebug = {}
    let focal = { fx: 554, fy: 554, method: "fallback" }

    try {
      const buf  = await blob.arrayBuffer()
      const tags = ExifReader.load(buf, { expanded: true })
      const exif = tags?.exif ?? tags

      // collect all relevant tags for backend logging
      ;[
        "FocalLength", "FocalLengthIn35mmFilm",
        "FocalPlaneXResolution", "FocalPlaneYResolution",
        "FocalPlaneResolutionUnit", "Make", "Model",
      ].forEach(k => {
        if (exif?.[k] != null) exifDebug[k] = exif[k].value ?? exif[k].description
      })

      focal = focalLengthFromExif(exif)
    } catch (e) {
      exifDebug["error"] = String(e)
    }

    focalRef.current = { fx: focal.fx, fy: focal.fy, exifDebug, method: focal.method }
    setFocalInfo(`fx=${focal.fx.toFixed(0)} (${focal.method})`)
  }

  async function captureAndDetect(runDetection) {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    if (detectRef.current) return
    detectRef.current = true

    try {
      const tmp = document.createElement("canvas")
      tmp.width  = 640
      tmp.height = 480
      tmp.getContext("2d").drawImage(video, 0, 0, 640, 480)

      tmp.toBlob(async (blob) => {
        await tryReadExif(blob)

        const focal = focalRef.current ?? { fx: 554, fy: 554 }

        const form = new FormData()
        form.append("image",         blob, "frame.jpg")
        form.append("camera_height", cameraHeight)
        form.append("detect",        runDetection ? "1" : "0")
        form.append("fx",            focal.fx.toFixed(2))
        form.append("fy",            focal.fy.toFixed(2))
        // send raw exif tags once (only on first frame when exifDebug exists)
        if (focal.exifDebug) {
          form.append("exif_debug", JSON.stringify(focal.exifDebug))
          form.append("exif_method", focal.method ?? "")
          delete focalRef.current.exifDebug  // only send once
        }

        try {
          const res  = await fetch(`${BACKEND_URL}/detect`, { method: "POST", body: form })
          const data = await res.json()
          const scene     = data.scene  ?? []
          const motionVal = data.motion ?? 0
          objectsRef.current = scene
          setObjects(scene)
          setMotion(motionVal)
          setIsMoving(motionVal > 2)
          setConnected(true)
        } catch (err) {
          console.error("❌ fetch error", err)
          setConnected(false)
        }

        detectRef.current = false
      }, "image/jpeg", 0.8)
    } catch (err) {
      console.error("❌ capture error", err)
      detectRef.current = false
    }
  }

  function startLoops() {
    let counter = 0
    setInterval(() => {
      if (detectRef.current) return
      counter++
      captureAndDetect(counter % 4 === 0)
    }, 100)
  }

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
    ctx.drawImage(video, 0, 0, 640, 480)

    scene.forEach((obj) => {
      if (!obj.bbox) return
      const [x1, y1, x2, y2] = obj.bbox
      const { r, g, b } = colorFor(obj.label)
      const w = x2 - x1
      const h = y2 - y1

      ctx.save()
      ctx.globalAlpha = 0.22
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillRect(x1, y1, w, h)
      ctx.restore()

      ctx.strokeStyle = `rgb(${r},${g},${b})`
      ctx.lineWidth = 2
      ctx.strokeRect(x1, y1, w, h)

      const cs = Math.min(16, w * 0.2, h * 0.2)
      ctx.lineWidth = 3
      ctx.beginPath(); ctx.moveTo(x1, y1 + cs);  ctx.lineTo(x1, y1); ctx.lineTo(x1 + cs, y1); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(x2 - cs, y1);  ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cs); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(x1, y2 - cs);  ctx.lineTo(x1, y2); ctx.lineTo(x1 + cs, y2); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(x2 - cs, y2);  ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - cs); ctx.stroke()

      const labelText = `${obj.label}  ${(obj.confidence * 100).toFixed(0)}%`
      ctx.font = `600 12px 'Roboto Mono', monospace`
      const textW = ctx.measureText(labelText).width
      const pillH = 20
      const pillX = x1
      const pillY = Math.max(y1 - pillH - 2, 0)
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.beginPath(); ctx.roundRect(pillX, pillY, textW + 12, pillH, 4); ctx.fill()
      ctx.fillStyle = "#2b2d3e"
      ctx.fillText(labelText, pillX + 6, pillY + pillH - 5)

      if (obj.dimensions) {
        const d = obj.dimensions
        const lines = [
          `L ${d.length.toFixed(2)}m  W ${d.width.toFixed(2)}m  H ${d.height.toFixed(2)}m`,
          `Vol: ${d.volume_m3.toFixed(3)} m³`,
        ]
        ctx.font = `500 11px 'Roboto Mono', monospace`
        const lineH = 16
        const boxW  = Math.max(...lines.map(l => ctx.measureText(l).width)) + 12
        const boxH  = lines.length * lineH + 6
        ctx.fillStyle = `rgba(43,45,62,0.85)`
        ctx.beginPath(); ctx.roundRect(x1, y2 + 4, boxW, boxH, 4); ctx.fill()
        ctx.strokeStyle = `rgba(${r},${g},${b},0.7)`
        ctx.lineWidth = 1; ctx.stroke()
        ctx.fillStyle = "#d1d0c5"
        lines.forEach((line, i) => ctx.fillText(line, x1 + 6, y2 + 4 + lineH * (i + 1) - 2))
      } else {
        ctx.font = "500 10px monospace"
        ctx.fillStyle = "rgba(43,45,62,0.75)"
        ctx.beginPath(); ctx.roundRect(x1, y2 + 4, 90, 18, 4); ctx.fill()
        ctx.fillStyle = `rgba(${r},${g},${b},0.9)`
        ctx.fillText("measuring...", x1 + 6, y2 + 16)
      }
    })
  }

  return (
    <div style={{
      display: "flex", height: "100vh",
      background: "#2b2d3e", color: "#d1d0c5",
      fontFamily: "'Roboto Mono', 'Courier New', monospace",
      overflow: "hidden",
    }}>
      <div style={{ flex: "0 0 640px", position: "relative", borderRight: "1px solid #323347" }}>
        <video ref={videoRef} autoPlay playsInline muted
          style={{ position: "absolute", opacity: 0, pointerEvents: "none" }} />
        <canvas ref={canvasRef}
          style={{ display: "block", width: "640px", height: "480px", background: "#1e1f2e" }} />

        <div style={{
          position: "absolute", top: 10, right: 10, fontSize: "11px",
          background: "rgba(43,45,62,0.85)",
          border: `1px solid ${isMoving ? "#e2b714" : "#646579"}`,
          padding: "5px 10px", borderRadius: "4px",
          color: isMoving ? "#e2b714" : "#646579", letterSpacing: "0.08em",
        }}>
          {isMoving ? "● MOVING" : "● STABLE"} &nbsp;|&nbsp; {motion.toFixed(2)}
        </div>

        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0,
          padding: "6px 14px", background: "rgba(30,31,46,0.9)",
          borderTop: "1px solid #323347", display: "flex",
          gap: "18px", fontSize: "11px", color: "#646579", letterSpacing: "0.07em",
          flexWrap: "wrap",
        }}>
          <span style={{ color: connected ? "#e2b714" : "#ef4444" }}>
            {connected ? "● live" : "● disconnected"}
          </span>
          <span>{fps} fps</span>
          <span>{objects.length} obj</span>
          <span>h: {cameraHeight?.toFixed(2)}m</span>
          {focalInfo && <span style={{ color: "#e2b714" }}>📷 {focalInfo}</span>}
        </div>
      </div>

      <div style={{
        flex: 1, overflowY: "auto", padding: "16px",
        display: "flex", flexDirection: "column", gap: "8px", background: "#272935",
      }}>
        <div style={{
          fontSize: "11px", letterSpacing: "0.15em", color: "#646579",
          marginBottom: "8px", textTransform: "uppercase",
          display: "flex", justifyContent: "space-between", alignItems: "center",
          borderBottom: "1px solid #323347", paddingBottom: "10px",
        }}>
          <span>ParcelVision · scene</span>
          <span style={{ color: "#e2b714" }}>{objects.length} detected</span>
        </div>

        {objects.length === 0 && (
          <div style={{ color: "#646579", fontSize: "12px", marginTop: "32px", textAlign: "center", letterSpacing: "0.1em" }}>
            no objects detected
          </div>
        )}

        {objects.map((obj) => {
          const { r, g, b } = colorFor(obj.label)
          const d = obj.dimensions
          return (
            <div key={obj.object_id} style={{
              background: "#2b2d3e", border: `1px solid rgba(${r},${g},${b},0.3)`,
              borderRadius: "6px", padding: "12px",
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: `rgb(${r},${g},${b})` }} />
                  <span style={{ fontSize: "13px", fontWeight: 600, color: "#d1d0c5" }}>{obj.label}</span>
                  <span style={{ fontSize: "10px", color: "#646579" }}>#{obj.object_id}</span>
                </div>
                <span style={{
                  fontSize: "11px", color: `rgb(${r},${g},${b})`,
                  background: `rgba(${r},${g},${b},0.1)`, padding: "2px 8px", borderRadius: "3px",
                }}>
                  {(obj.confidence * 100).toFixed(0)}%
                </span>
              </div>

              {d ? (
                <div style={{ marginTop: "10px" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "5px", marginBottom: "6px" }}>
                    {[
                      { label: "L", value: `${d.length.toFixed(2)}m` },
                      { label: "W", value: `${d.width.toFixed(2)}m`  },
                      { label: "H", value: `${d.height.toFixed(2)}m` },
                    ].map(({ label, value }) => (
                      <div key={label} style={{ background: "#323347", borderRadius: "4px", padding: "6px 8px", textAlign: "center" }}>
                        <div style={{ fontSize: "9px", color: "#646579", letterSpacing: "0.12em" }}>{label}</div>
                        <div style={{ fontSize: "13px", color: "#d1d0c5", marginTop: "2px" }}>{value}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{
                    background: "#323347", borderRadius: "4px", padding: "8px 10px",
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                  }}>
                    <span style={{ fontSize: "9px", color: "#646579", letterSpacing: "0.12em" }}>VOLUME</span>
                    <span style={{ fontSize: "15px", color: `rgb(${r},${g},${b})`, fontWeight: 700 }}>
                      {d.volume_m3.toFixed(3)} m³
                    </span>
                  </div>
                </div>
              ) : (
                <div style={{ marginTop: "8px", fontSize: "11px", color: "#646579" }}>measuring...</div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
