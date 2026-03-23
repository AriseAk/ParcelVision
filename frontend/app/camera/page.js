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

// ── Derive fx from whatever the browser exposes ───────────────────────────
// Priority:
//   1. getCapabilities() zoom + stream dimensions (Android Chrome)
//   2. EXIF FocalLength + FocalPlaneXResolution
//   3. EXIF FocalLengthIn35mmFilm with crop correction
//   4. FOV fallback using stream aspect ratio
function deriveFocalLength(track, exifTags, streamW, streamH) {
  // ── Method 1: use zoom from capabilities ──────────────────────────────
  try {
    const capabilities = track.getCapabilities?.() ?? {}
    const settings     = track.getSettings?.()     ?? {}
    const zoom         = settings.zoom ?? 1.0

    // Base FOV for main rear camera at zoom=1 is ~70° horizontal for most phones
    // fx = (streamW / 2) / tan(hfov/2)
    // At zoom=1, hfov ≈ 70°. At zoom=2, hfov ≈ 35° etc.
    // effective_hfov = 2 * arctan(tan(base_hfov/2) / zoom)
    const baseHfovRad  = (70 * Math.PI) / 180
    const effHfovRad   = 2 * Math.atan(Math.tan(baseHfovRad / 2) / zoom)
    const fx           = (streamW / 2) / Math.tan(effHfovRad / 2)

    // If capabilities exposes a zoom range, we can trust this more
    const hasZoomCap = capabilities.zoom?.min != null
    if (hasZoomCap || zoom !== 1.0) {
      return {
        fx, fy: fx,
        method: `zoom=${zoom.toFixed(2)} hfov=${(effHfovRad * 180 / Math.PI).toFixed(1)}°`,
      }
    }
  } catch (e) {}

  // ── Method 2: EXIF FocalLength + FocalPlaneXResolution ────────────────
  try {
    const focalMm = exifTags["FocalLength"]?.value
    const fpxRes  = exifTags["FocalPlaneXResolution"]?.value
    const fpxUnit = exifTags["FocalPlaneResolutionUnit"]?.value

    if (focalMm && fpxRes && fpxUnit) {
      const pxPerMm = fpxUnit === 2 ? fpxRes / 25.4 : fpxRes / 10
      const fx      = focalMm * pxPerMm * (streamW / 640)
      return { fx, fy: fx, method: `EXIF fmm=${focalMm} pxPerMm=${pxPerMm.toFixed(1)}` }
    }
  } catch (e) {}

  // ── Method 3: EXIF FocalLengthIn35mmFilm ──────────────────────────────
  try {
    const f35 = exifTags["FocalLengthIn35mmFilm"]?.value
    if (f35 && f35 > 0) {
      // 35mm frame = 36mm wide. fx_35mm = f35 * (36mm_frame_px / 36mm)
      // scaled to our stream: fx = (f35 / 36) * streamW * crop_factor
      // phone sensor ~5.6mm → crop = 36/5.6 ≈ 6.43
      const fx = (f35 * streamW) / 5.6
      return { fx, fy: fx, method: `EXIF f35=${f35}mm` }
    }
  } catch (e) {}

  // ── Method 4: FOV from stream aspect ratio ────────────────────────────
  // Most phone rear cameras: 4:3 native, hfov ~70°, vfov ~53°
  // Wide lens: hfov ~120°. We can't distinguish without extra info.
  // Use stream aspect to pick reasonable default.
  const aspect   = streamW / streamH
  // 4:3 aspect → main lens → 70° hfov
  // 16:9 aspect → may be cropped from wider lens → use 75°
  const hfovDeg  = aspect > 1.6 ? 75 : 70
  const hfovRad  = (hfovDeg * Math.PI) / 180
  const fx       = (streamW / 2) / Math.tan(hfovRad / 2)
  return { fx, fy: fx, method: `fallback ${hfovDeg}° hfov aspect=${aspect.toFixed(2)}` }
}

export default function CameraPage() {
  const videoRef   = useRef(null)
  const canvasRef  = useRef(null)
  const detectRef  = useRef(false)
  const objectsRef = useRef([])
  const loopsRef   = useRef(false)
  const focalRef   = useRef(null)
  const streamDims = useRef({ w: 640, h: 480 })

  const [objects,      setObjects]      = useState([])
  const [fps,          setFps]          = useState(0)
  const [connected,    setConnected]    = useState(false)
  const [isMoving,     setIsMoving]     = useState(false)
  const [motion,       setMotion]       = useState(0)
  const [cameraHeight, setCameraHeight] = useState(null)
  const [focalInfo,    setFocalInfo]    = useState(null)
  const [showSidebar,  setShowSidebar]  = useState(false)

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
          video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        })
        if (!videoRef.current) return
        videoRef.current.srcObject = stream

        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().catch(() => {})

          // ── Get actual stream dimensions ──
          const track    = stream.getVideoTracks()[0]
          const settings = track.getSettings?.() ?? {}
          const sw       = settings.width  || videoRef.current.videoWidth  || 640
          const sh       = settings.height || videoRef.current.videoHeight || 480
          streamDims.current = { w: sw, h: sh }

          // ── Derive focal length from track + capabilities ──
          const focal = deriveFocalLength(track, {}, sw, sh)
          focalRef.current = { fx: focal.fx, fy: focal.fy, method: focal.method }
          setFocalInfo(`fx=${focal.fx.toFixed(0)} (${focal.method})`)
          console.log("📷 Focal:", focal)

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

  // ── Try EXIF once to refine focal length ─────────────────────────────
  async function tryRefineWithExif(blob) {
    if (focalRef.current?.exifDone) return
    try {
      const buf  = await blob.arrayBuffer()
      const tags = ExifReader.load(buf, { expanded: true })
      const exif = tags?.exif ?? tags

      const track = videoRef.current?.srcObject?.getVideoTracks()[0]
      const sw    = streamDims.current.w
      const sh    = streamDims.current.h
      const focal = deriveFocalLength(track, exif, sw, sh)

      focalRef.current = { fx: focal.fx, fy: focal.fy, method: focal.method, exifDone: true }
      setFocalInfo(`fx=${focal.fx.toFixed(0)} (${focal.method})`)

      // send debug to backend once
      const exifDebug = {}
      ;["FocalLength","FocalLengthIn35mmFilm","FocalPlaneXResolution",
        "FocalPlaneResolutionUnit","Make","Model"].forEach(k => {
        if (exif?.[k] != null) exifDebug[k] = exif[k].value ?? exif[k].description
      })
      focalRef.current.exifDebug  = JSON.stringify(exifDebug)
      focalRef.current.exifMethod = focal.method
    } catch (e) {
      if (focalRef.current) focalRef.current.exifDone = true
    }
  }

  async function captureAndDetect(runDetection) {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    if (detectRef.current) return
    detectRef.current = true

    try {
      const sw = streamDims.current.w
      const sh = streamDims.current.h

      const tmp = document.createElement("canvas")
      tmp.width  = sw
      tmp.height = sh
      tmp.getContext("2d").drawImage(video, 0, 0, sw, sh)

      tmp.toBlob(async (blob) => {
        await tryRefineWithExif(blob)

        const focal = focalRef.current ?? { fx: 554, fy: 554 }

        const form = new FormData()
        form.append("image",         blob, "frame.jpg")
        form.append("camera_height", cameraHeight)
        form.append("detect",        runDetection ? "1" : "0")
        form.append("fx",            focal.fx.toFixed(2))
        form.append("fy",            focal.fy.toFixed(2))
        form.append("img_w",         sw)
        form.append("img_h",         sh)

        if (focal.exifDebug) {
          form.append("exif_debug",  focal.exifDebug)
          form.append("exif_method", focal.exifMethod ?? "")
          delete focalRef.current.exifDebug
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
    const sw  = streamDims.current.w
    const sh  = streamDims.current.h
    canvas.width  = sw
    canvas.height = sh
    ctx.drawImage(video, 0, 0, sw, sh)

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

      const cs = Math.min(20, w * 0.2, h * 0.2)
      ctx.lineWidth = 3
      ctx.beginPath(); ctx.moveTo(x1, y1 + cs);  ctx.lineTo(x1, y1); ctx.lineTo(x1 + cs, y1); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(x2 - cs, y1);  ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cs); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(x1, y2 - cs);  ctx.lineTo(x1, y2); ctx.lineTo(x1 + cs, y2); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(x2 - cs, y2);  ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - cs); ctx.stroke()

      const labelText = `${obj.label}  ${(obj.confidence * 100).toFixed(0)}%`
      ctx.font = `600 14px 'Roboto Mono', monospace`
      const textW = ctx.measureText(labelText).width
      const pillH = 24
      const pillX = x1
      const pillY = Math.max(y1 - pillH - 2, 0)
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.beginPath(); ctx.roundRect(pillX, pillY, textW + 14, pillH, 4); ctx.fill()
      ctx.fillStyle = "#2b2d3e"
      ctx.fillText(labelText, pillX + 7, pillY + pillH - 6)

      if (obj.dimensions) {
        const d = obj.dimensions
        const lines = [
          `L ${d.length.toFixed(2)}m  W ${d.width.toFixed(2)}m  H ${d.height.toFixed(2)}m`,
          `Vol: ${d.volume_m3.toFixed(3)} m³`,
        ]
        ctx.font = `500 13px 'Roboto Mono', monospace`
        const lineH = 18
        const boxW  = Math.max(...lines.map(l => ctx.measureText(l).width)) + 14
        const boxH  = lines.length * lineH + 8
        ctx.fillStyle = `rgba(43,45,62,0.88)`
        ctx.beginPath(); ctx.roundRect(x1, y2 + 4, boxW, boxH, 4); ctx.fill()
        ctx.strokeStyle = `rgba(${r},${g},${b},0.7)`
        ctx.lineWidth = 1; ctx.stroke()
        ctx.fillStyle = "#d1d0c5"
        lines.forEach((line, i) => ctx.fillText(line, x1 + 7, y2 + 4 + lineH * (i + 1) - 2))
      } else {
        ctx.font = "500 12px monospace"
        ctx.fillStyle = "rgba(43,45,62,0.8)"
        ctx.beginPath(); ctx.roundRect(x1, y2 + 4, 110, 20, 4); ctx.fill()
        ctx.fillStyle = `rgba(${r},${g},${b},0.9)`
        ctx.fillText("measuring...", x1 + 7, y2 + 18)
      }
    })
  }

  // ── Responsive layout: phone = stacked, desktop = side by side ──────────
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      height: "100dvh",
      background: "#2b2d3e",
      color: "#d1d0c5",
      fontFamily: "'Roboto Mono', 'Courier New', monospace",
      overflow: "hidden",
    }}>

      {/* ── Camera area ── */}
      <div style={{ position: "relative", flex: "0 0 auto" }}>
        <video ref={videoRef} autoPlay playsInline muted
          style={{ position: "absolute", opacity: 0, pointerEvents: "none" }} />

        <canvas ref={canvasRef} style={{
          display: "block",
          width: "100vw",
          height: "56.25vw",   /* 16:9 */
          maxHeight: "60dvh",
          objectFit: "cover",
          background: "#1e1f2e",
        }} />

        {/* motion badge */}
        <div style={{
          position: "absolute", top: 10, right: 10,
          fontSize: "11px", background: "rgba(43,45,62,0.9)",
          border: `1px solid ${isMoving ? "#e2b714" : "#646579"}`,
          padding: "4px 10px", borderRadius: "4px",
          color: isMoving ? "#e2b714" : "#646579", letterSpacing: "0.08em",
        }}>
          {isMoving ? "● MOVING" : "● STABLE"} | {motion.toFixed(1)}
        </div>

        {/* status bar */}
        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0,
          padding: "5px 12px", background: "rgba(30,31,46,0.92)",
          borderTop: "1px solid #323347",
          display: "flex", gap: "14px", fontSize: "11px",
          color: "#646579", letterSpacing: "0.06em", flexWrap: "wrap",
          alignItems: "center",
        }}>
          <span style={{ color: connected ? "#e2b714" : "#ef4444" }}>
            {connected ? "● live" : "● disconnected"}
          </span>
          <span>{fps} fps</span>
          <span>{objects.length} obj</span>
          <span>h:{cameraHeight?.toFixed(1)}m</span>
          {focalInfo && <span style={{ color: "#e2b714", fontSize: "10px" }}>📷 {focalInfo}</span>}

          {/* toggle sidebar button — mobile only */}
          <button
            onClick={() => setShowSidebar(s => !s)}
            style={{
              marginLeft: "auto",
              background: objects.length > 0 ? "#e2b714" : "#323347",
              color: objects.length > 0 ? "#2b2d3e" : "#646579",
              border: "none", borderRadius: "4px",
              padding: "3px 10px", fontSize: "11px",
              fontFamily: "inherit", cursor: "pointer",
              letterSpacing: "0.06em",
            }}>
            {showSidebar ? "hide" : `results (${objects.length})`}
          </button>
        </div>
      </div>

      {/* ── Sidebar / results panel ── */}
      <div style={{
        flex: 1,
        overflowY: "auto",
        padding: "12px",
        display: showSidebar || objects.length === 0 ? "flex" : "none",
        flexDirection: "column",
        gap: "8px",
        background: "#272935",
        // on wider screens always show
        ...(typeof window !== "undefined" && window.innerWidth > 640
          ? { display: "flex" }
          : {}),
      }}>
        <div style={{
          fontSize: "10px", letterSpacing: "0.15em", color: "#646579",
          marginBottom: "4px", textTransform: "uppercase",
          display: "flex", justifyContent: "space-between",
          borderBottom: "1px solid #323347", paddingBottom: "8px",
        }}>
          <span>ParcelVision · scene</span>
          <span style={{ color: "#e2b714" }}>{objects.length} detected</span>
        </div>

        {objects.length === 0 && (
          <div style={{ color: "#646579", fontSize: "12px", marginTop: "16px", textAlign: "center" }}>
            point camera at a box
          </div>
        )}

        {objects.map((obj) => {
          const { r, g, b } = colorFor(obj.label)
          const d = obj.dimensions
          return (
            <div key={obj.object_id} style={{
              background: "#2b2d3e",
              border: `1px solid rgba(${r},${g},${b},0.3)`,
              borderRadius: "6px", padding: "10px",
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "7px" }}>
                  <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: `rgb(${r},${g},${b})`, flexShrink: 0 }} />
                  <span style={{ fontSize: "13px", fontWeight: 600, color: "#d1d0c5" }}>{obj.label}</span>
                  <span style={{ fontSize: "10px", color: "#646579" }}>#{obj.object_id}</span>
                </div>
                <span style={{
                  fontSize: "11px", color: `rgb(${r},${g},${b})`,
                  background: `rgba(${r},${g},${b},0.1)`, padding: "2px 7px", borderRadius: "3px",
                }}>
                  {(obj.confidence * 100).toFixed(0)}%
                </span>
              </div>

              {d ? (
                <div style={{ marginTop: "8px" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "5px", marginBottom: "5px" }}>
                    {[
                      { label: "L", value: `${d.length.toFixed(2)}m` },
                      { label: "W", value: `${d.width.toFixed(2)}m`  },
                      { label: "H", value: `${d.height.toFixed(2)}m` },
                    ].map(({ label, value }) => (
                      <div key={label} style={{ background: "#323347", borderRadius: "4px", padding: "5px 6px", textAlign: "center" }}>
                        <div style={{ fontSize: "9px", color: "#646579", letterSpacing: "0.1em" }}>{label}</div>
                        <div style={{ fontSize: "13px", color: "#d1d0c5", marginTop: "1px" }}>{value}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{
                    background: "#323347", borderRadius: "4px", padding: "7px 10px",
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                  }}>
                    <span style={{ fontSize: "9px", color: "#646579", letterSpacing: "0.1em" }}>VOLUME</span>
                    <span style={{ fontSize: "15px", color: `rgb(${r},${g},${b})`, fontWeight: 700 }}>
                      {d.volume_m3.toFixed(3)} m³
                    </span>
                  </div>
                </div>
              ) : (
                <div style={{ marginTop: "6px", fontSize: "11px", color: "#646579" }}>measuring...</div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
