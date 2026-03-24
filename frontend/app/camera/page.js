"use client"

import { useEffect, useRef, useState } from "react"
import ExifReader from "exifreader"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://172.18.192.1:5000"

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
function deriveFocalLength(track, exifTags, streamW, streamH) {
  // Method 1: use zoom from capabilities
  try {
    const capabilities = track.getCapabilities?.() ?? {}
    const settings     = track.getSettings?.()     ?? {}
    const zoom         = settings.zoom ?? 1.0
    const baseHfovRad  = (70 * Math.PI) / 180
    const effHfovRad   = 2 * Math.atan(Math.tan(baseHfovRad / 2) / zoom)
    const fx           = (streamW / 2) / Math.tan(effHfovRad / 2)
    const hasZoomCap   = capabilities.zoom?.min != null
    if (hasZoomCap || zoom !== 1.0) {
      return {
        fx, fy: fx,
        method: `zoom=${zoom.toFixed(2)} hfov=${(effHfovRad * 180 / Math.PI).toFixed(1)}°`,
      }
    }
  } catch (e) {}

  // Method 2: EXIF FocalLength + FocalPlaneXResolution
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

  // Method 3: EXIF FocalLengthIn35mmFilm
  try {
    const f35 = exifTags["FocalLengthIn35mmFilm"]?.value
    if (f35 && f35 > 0) {
      const fx = (f35 * streamW) / 5.6
      return { fx, fy: fx, method: `EXIF f35=${f35}mm` }
    }
  } catch (e) {}

  // Method 4: FOV from stream aspect ratio fallback
  const aspect  = streamW / streamH
  const hfovDeg = aspect > 1.6 ? 75 : 70
  const hfovRad = (hfovDeg * Math.PI) / 180
  const fx      = (streamW / 2) / Math.tan(hfovRad / 2)
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

  // ── Bbox interpolation refs  (STEP 1) ──────────────────────────────────
  const lerpedObjectsRef = useRef({})

  // ── IMU collection refs  (STEP 5) ──────────────────────────────────────
  const imuBufferRef = useRef([])
  const imuActiveRef = useRef(false)

  // ── Scan state  (STEP 6) ───────────────────────────────────────────────
  const scanningRef = useRef(false)

  const [objects,         setObjects]         = useState([])
  const [fps,             setFps]             = useState(0)
  const [connected,       setConnected]       = useState(false)
  const [isMoving,        setIsMoving]        = useState(false)
  const [motion,          setMotion]          = useState(0)
  const [cameraHeight,    setCameraHeight]    = useState(null)
  const [focalInfo,       setFocalInfo]       = useState(null)
  const [showSidebar,     setShowSidebar]     = useState(false)
  const [scanState,       setScanState]       = useState('idle')      // idle | scanning | computing | result
  const [finalDimensions, setFinalDimensions] = useState(null)
  const [scanFrameCount,  setScanFrameCount]  = useState(0)

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

          const track    = stream.getVideoTracks()[0]
          const settings = track.getSettings?.() ?? {}
          const sw       = settings.width  || videoRef.current.videoWidth  || 640
          const sh       = settings.height || videoRef.current.videoHeight || 480
          streamDims.current = { w: sw, h: sh }

          const focal = deriveFocalLength(track, {}, sw, sh)
          focalRef.current = { fx: focal.fx, fy: focal.fy, method: focal.method }
          setFocalInfo(`fx=${focal.fx.toFixed(0)} (${focal.method})`)
          console.log("📷 Focal:", focal)

          if (!loopsRef.current) {
            loopsRef.current = true
            startDetectionLoop()   // STEP 1 — replaces startLoops
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

  // ── EXIF refinement ───────────────────────────────────────────────────
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

  // ── Detection loop  (STEP 1 — fires on response, not fixed interval) ──
  function startDetectionLoop() {
    async function loop() {
      while (loopsRef.current) {
        if (document.visibilityState === 'hidden') {
          await new Promise(res => setTimeout(res, 500))
          continue
        }
        // only run idle detection when not in a scan
        if (!scanningRef.current) {
          await captureAndDetect(true)
        }
        await new Promise(res => setTimeout(res, 200))
      }
    }
    loop()
  }

  // ── Bbox interpolation helpers  (STEP 1) ──────────────────────────────
  function updateDetectionTargets(scene) {
    scene.forEach(obj => {
      if (!lerpedObjectsRef.current[obj.object_id]) {
        lerpedObjectsRef.current[obj.object_id] = {
          ...obj,
          currentBbox: [...obj.bbox],
          targetBbox:  [...obj.bbox],
        }
      } else {
        lerpedObjectsRef.current[obj.object_id].targetBbox  = [...obj.bbox]
        lerpedObjectsRef.current[obj.object_id].dimensions  = obj.dimensions
        lerpedObjectsRef.current[obj.object_id].confidence  = obj.confidence
      }
    })
    const activeIds = new Set(scene.map(o => o.object_id))
    Object.keys(lerpedObjectsRef.current).forEach(id => {
      if (!activeIds.has(parseInt(id))) delete lerpedObjectsRef.current[id]
    })
  }

  function lerpBboxes() {
    Object.values(lerpedObjectsRef.current).forEach(obj => {
      const alpha = 0.2
      obj.currentBbox[0] += (obj.targetBbox[0] - obj.currentBbox[0]) * alpha
      obj.currentBbox[1] += (obj.targetBbox[1] - obj.currentBbox[1]) * alpha
      obj.currentBbox[2] += (obj.targetBbox[2] - obj.currentBbox[2]) * alpha
      obj.currentBbox[3] += (obj.targetBbox[3] - obj.currentBbox[3]) * alpha
    })
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
          const res   = await fetch(`${BACKEND_URL}/detect`, { method: "POST", body: form })
          const data  = await res.json()
          const scene     = data.scene  ?? []
          const motionVal = data.motion ?? 0

          objectsRef.current = scene
          updateDetectionTargets(scene)   // STEP 1
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

  // ── Render loop  (STEP 1 — lerps bboxes at 60fps) ─────────────────────
  function startRenderLoop() {
    function render() {
      lerpBboxes()
      drawOverlays(Object.values(lerpedObjectsRef.current))
      fpsFrames.current++
      requestAnimationFrame(render)
    }
    requestAnimationFrame(render)
  }

  // ── IMU collection  (STEP 5) ───────────────────────────────────────────
  function startIMUCollection() {
    if (typeof DeviceMotionEvent === 'undefined') {
      console.warn('DeviceMotion not available — scan accuracy will be lower')
      return
    }
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
      DeviceMotionEvent.requestPermission()
        .then(state => {
          if (state === 'granted') attachIMUListener()
          else console.warn('IMU permission denied')
        })
        .catch(console.error)
    } else {
      attachIMUListener()
    }
  }

  function stopIMUCollection() {
    imuActiveRef.current = false
  }

  function attachIMUListener() {
    imuActiveRef.current = true
    window.addEventListener('devicemotion', (e) => {
      if (!imuActiveRef.current) return
      imuBufferRef.current.push({
        ts: Date.now(),
        ax: e.acceleration?.x     ?? 0,
        ay: e.acceleration?.y     ?? 0,
        az: e.acceleration?.z     ?? 0,
        gx: e.rotationRate?.alpha ?? 0,
        gy: e.rotationRate?.beta  ?? 0,
        gz: e.rotationRate?.gamma ?? 0,
      })
      if (imuBufferRef.current.length > 30) {
        imuBufferRef.current.shift()
      }
    }, false)
  }

  // ── Scan handlers  (STEPS 6, 7) ───────────────────────────────────────
  async function handleStartScan() {
    imuBufferRef.current = []
    setScanFrameCount(0)
    scanningRef.current = true
    setScanState('scanning')
    startIMUCollection()

    await fetch(`${BACKEND_URL}/start_scan`, { method: 'POST' })
    runScanLoop()
  }

  async function runScanLoop() {
    while (scanningRef.current) {
      const video = videoRef.current
      if (!video) break

      const sw = streamDims.current.w
      const sh = streamDims.current.h

      const tmp = document.createElement('canvas')
      tmp.width  = sw
      tmp.height = sh
      tmp.getContext('2d').drawImage(video, 0, 0, sw, sh)

      const blob = await new Promise(res => tmp.toBlob(res, 'image/jpeg', 0.7))

      const imuSnapshot    = [...imuBufferRef.current]
      imuBufferRef.current = []

      await sendScanFrame(blob, imuSnapshot)
      setScanFrameCount(c => c + 1)

      await new Promise(res => setTimeout(res, 150))
    }
  }

  async function sendScanFrame(blob, imuReadings) {
    const focal = focalRef.current ?? { fx: 554, fy: 554 }
    const form  = new FormData()

    form.append('image',         blob, 'frame.jpg')
    form.append('imu',           JSON.stringify(imuReadings))
    form.append('camera_height', cameraHeight)
    form.append('fx',            focal.fx.toFixed(2))
    form.append('fy',            focal.fy.toFixed(2))
    form.append('img_w',         streamDims.current.w)
    form.append('img_h',         streamDims.current.h)

    try {
      await fetch(`${BACKEND_URL}/scan_frame`, { method: 'POST', body: form })
    } catch (e) {
      console.error('scan frame failed', e)
    }
  }

  async function handleDoneScan() {
    scanningRef.current = false
    stopIMUCollection()
    setScanState('computing')

    try {
      const res  = await fetch(`${BACKEND_URL}/compute_dimensions`, { method: 'POST' })
      const data = await res.json()

      if (data.error) {
        console.error('compute error:', data.error)
        setScanState('idle')
        return
      }

      setFinalDimensions(data.dimensions)
      setScanState('result')
    } catch (e) {
      console.error('compute_dimensions failed', e)
      setScanState('idle')
    }
  }

  function handleScanAgain() {
    setFinalDimensions(null)
    setScanFrameCount(0)
    setScanState('idle')
  }

  // ── Canvas drawing — uses currentBbox from lerped objects ─────────────
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
      // use interpolated bbox (STEP 1)
      const bbox = obj.currentBbox ?? obj.bbox
      if (!bbox) return
      const [x1, y1, x2, y2] = bbox
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

  // ── Layout ──────────────────────────────────────────────────────────────
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
          height: "56.25vw",
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

        {/* ── Scan button — idle, has detections  (STEP 6) ── */}
        {scanState === 'idle' && objects.length > 0 && (
          <button onClick={handleStartScan} style={{
            position: "absolute", bottom: 48, left: "50%", transform: "translateX(-50%)",
            background: "#e2b714", color: "#2b2d3e",
            border: "none", borderRadius: "6px",
            padding: "10px 28px", fontSize: "13px", fontWeight: 700,
            fontFamily: "inherit", cursor: "pointer", letterSpacing: "0.1em",
            boxShadow: "0 2px 12px rgba(0,0,0,0.4)",
          }}>
            START SCAN
          </button>
        )}

        {/* ── Scanning overlay  (STEP 6) ── */}
        {scanState === 'scanning' && (
          <div style={{
            position: "absolute", bottom: 48, left: "50%", transform: "translateX(-50%)",
            background: "rgba(30,31,46,0.95)", border: "1px solid #e2b714",
            borderRadius: "8px", padding: "12px 20px",
            display: "flex", flexDirection: "column", alignItems: "center", gap: "8px",
            minWidth: "220px",
          }}>
            <p style={{ margin: 0, fontSize: "12px", color: "#e2b714", letterSpacing: "0.08em" }}>
              Walk slowly around the box
            </p>
            <p style={{ margin: 0, fontSize: "11px", color: "#646579" }}>
              {scanFrameCount} frames collected
            </p>
            <button onClick={handleDoneScan} style={{
              background: "#e2b714", color: "#2b2d3e",
              border: "none", borderRadius: "4px",
              padding: "7px 24px", fontSize: "12px", fontWeight: 700,
              fontFamily: "inherit", cursor: "pointer", letterSpacing: "0.08em",
            }}>
              DONE
            </button>
          </div>
        )}

        {/* ── Computing overlay  (STEP 6) ── */}
        {scanState === 'computing' && (
          <div style={{
            position: "absolute", inset: 0,
            background: "rgba(30,31,46,0.85)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <div style={{
              fontSize: "14px", color: "#e2b714", letterSpacing: "0.12em",
              display: "flex", alignItems: "center", gap: "10px",
            }}>
              <span style={{
                display: "inline-block", width: "14px", height: "14px",
                border: "2px solid #e2b714", borderTopColor: "transparent",
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
              }} />
              COMPUTING DIMENSIONS…
            </div>
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </div>
        )}

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

        {/* ── Scan result card  (STEP 6) ── */}
        {scanState === 'result' && finalDimensions && (
          <div style={{
            background: "#2b2d3e",
            border: "1px solid rgba(226,183,20,0.6)",
            borderRadius: "8px", padding: "14px",
            marginBottom: "4px",
          }}>
            <div style={{
              fontSize: "10px", color: "#e2b714", letterSpacing: "0.15em",
              marginBottom: "10px", textTransform: "uppercase",
            }}>
              Scan Result
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "6px", marginBottom: "8px" }}>
              {[
                { label: "L", value: `${finalDimensions.length.toFixed(2)}m` },
                { label: "W", value: `${finalDimensions.width.toFixed(2)}m`  },
                { label: "H", value: `${finalDimensions.height.toFixed(2)}m` },
              ].map(({ label, value }) => (
                <div key={label} style={{ background: "#323347", borderRadius: "4px", padding: "6px", textAlign: "center" }}>
                  <div style={{ fontSize: "9px", color: "#646579", letterSpacing: "0.1em" }}>{label}</div>
                  <div style={{ fontSize: "14px", color: "#d1d0c5", marginTop: "2px" }}>{value}</div>
                </div>
              ))}
            </div>
            <div style={{
              background: "#323347", borderRadius: "4px", padding: "8px 10px",
              display: "flex", justifyContent: "space-between", alignItems: "center",
              marginBottom: "10px",
            }}>
              <span style={{ fontSize: "9px", color: "#646579", letterSpacing: "0.1em" }}>VOLUME</span>
              <span style={{ fontSize: "16px", color: "#e2b714", fontWeight: 700 }}>
                {finalDimensions.volume_m3.toFixed(3)} m³
              </span>
            </div>
            <button onClick={handleScanAgain} style={{
              width: "100%",
              background: "#323347", color: "#d1d0c5",
              border: "1px solid #646579", borderRadius: "4px",
              padding: "8px", fontSize: "11px", fontWeight: 600,
              fontFamily: "inherit", cursor: "pointer", letterSpacing: "0.08em",
            }}>
              SCAN AGAIN
            </button>
          </div>
        )}

        {objects.length === 0 && scanState === 'idle' && (
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