"use client"

import { useEffect, useRef, useState, useCallback } from "react"

// ── Config ────────────────────────────────────────────────────────────────────
const BACKEND_URL =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_BACKEND_URL) ||
  "http://172.18.192.1:5000"

const SCAN_MOTION_MIN = 0.3
const SCAN_MOTION_MAX = 150.0
const CARD_WIDTH_M    = 0.0856
const CARD_HEIGHT_M   = 0.05398

const CLASS_COLORS = {
  "box":           { r: 251, g: 191, b: 36  },
  "cardboard box": { r: 251, g: 191, b: 36  },
  carton:          { r: 251, g: 191, b: 36  },
  parcel:          { r: 99,  g: 202, b: 183 },
  package:         { r: 99,  g: 202, b: 183 },
  container:       { r: 167, g: 139, b: 250 },
  "brown box":     { r: 251, g: 146, b: 60  },
}
const colorFor = (label) =>
  CLASS_COLORS[label?.toLowerCase()] ?? { r: 251, g: 191, b: 36 }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// ── Optical flow ──────────────────────────────────────────────────────────────
class FrontendOpticalFlow {
  constructor() { this.prevImageData = null; this.motionSmooth = 0 }
  compute(video) {
    const W = 320, H = 240
    const c = document.createElement("canvas"); c.width = W; c.height = H
    const ctx = c.getContext("2d"); ctx.drawImage(video, 0, 0, W, H)
    const curr = ctx.getImageData(0, 0, W, H)
    if (!this.prevImageData) { this.prevImageData = curr; return 0 }
    const prev = this.prevImageData
    let totalMotion = 0, count = 0
    const step = 16, halfWin = 4
    for (let y = step; y < H - step; y += step) {
      for (let x = step; x < W - step; x += step) {
        const ci = (y * W + x) * 4
        const cg = 0.299 * curr.data[ci] + 0.587 * curr.data[ci + 1] + 0.114 * curr.data[ci + 2]
        let bestDiff = Infinity, bestDx = 0, bestDy = 0
        for (let dy = -halfWin; dy <= halfWin; dy++) {
          for (let dx = -halfWin; dx <= halfWin; dx++) {
            const ny = y + dy, nx = x + dx
            if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue
            const pi = (ny * W + nx) * 4
            const pg = 0.299 * prev.data[pi] + 0.587 * prev.data[pi + 1] + 0.114 * prev.data[pi + 2]
            const diff = Math.abs(cg - pg)
            if (diff < bestDiff) { bestDiff = diff; bestDx = dx; bestDy = dy }
          }
        }
        if (bestDiff < 30) { totalMotion += Math.sqrt(bestDx ** 2 + bestDy ** 2); count++ }
      }
    }
    this.prevImageData = curr
    const raw = count > 0 ? (totalMotion / count) * 5 : 0
    this.motionSmooth = 0.7 * this.motionSmooth + 0.3 * raw
    return this.motionSmooth
  }
  reset() { this.prevImageData = null; this.motionSmooth = 0 }
}

// ── Focal length ──────────────────────────────────────────────────────────────
function deriveFocalLength(track, streamW) {
  try {
    const s = track.getSettings?.() ?? {}
    const c = track.getCapabilities?.() ?? {}
    const zoom = s.zoom ?? 1.0
    if (c.zoom?.min != null || zoom !== 1.0) {
      const base = (65 * Math.PI) / 180
      const eff  = 2 * Math.atan(Math.tan(base / 2) / zoom)
      const fx   = (streamW / 2) / Math.tan(eff / 2)
      return { fx, fy: fx }
    }
  } catch {}
  const fx = (streamW / 2) / Math.tan((65 * Math.PI) / 360)
  return { fx, fy: fx }
}

const DEFAULT_CARD = { x: 0.2, y: 0.35, w: 0.6, h: 0.3 }

// ══════════════════════════════════════════════════════════════════════════════
export default function CameraPage() {
  // refs
  const videoRef       = useRef(null)
  const canvasRef      = useRef(null)
  const loopsRef       = useRef(false)
  const objectsRef     = useRef([])
  const focalRef       = useRef(null)
  const streamDims     = useRef({ w: 640, h: 480 })
  const canvasDims     = useRef({ w: 640, h: 480 })
  const lerpedObjRef   = useRef({})
  const imuBufRef      = useRef([])
  const imuActiveRef   = useRef(false)
  const scanningRef    = useRef(false)
  const flowRef        = useRef(new FrontendOpticalFlow())
  const scanMotionRef  = useRef(0)
  const acceptedRef    = useRef(0)
  const skippedRef     = useRef(0)
  const scaleFactorRef = useRef(null)
  const cardRectRef    = useRef({ ...DEFAULT_CARD })
  const draggingRef    = useRef(null)
  const scanStateRef   = useRef("idle")
  const fpsFrames      = useRef(0)
  const fpsTimer       = useRef(null)
  const lastDetRef     = useRef([])

  // state
  const [objects,        setObjects]        = useState([])
  const [fps,            setFps]            = useState(0)
  const [connected,      setConnected]      = useState(false)
  const [isMoving,       setIsMoving]       = useState(false)
  const [motion,         setMotion]         = useState(0)
  const [focalInfo,      setFocalInfo]      = useState(null)
  const [showSidebar,    setShowSidebar]    = useState(false)
  const [scanState,      setScanState]      = useState("idle")
  const [finalDims,      setFinalDims]      = useState(null)
  const [scanFrameCount, setScanFrameCount] = useState(0)
  const [scanHint,       setScanHint]       = useState("")
  const [scanSkipped,    setScanSkipped]    = useState(0)
  const [scaleFactor,    setScaleFactor]    = useState(null)
  const [camError,       setCamError]       = useState(null)
  const [cardPx,         setCardPx]         = useState({ w: 0, h: 0 })
  const [calStep,        setCalStep]        = useState(0)  // 0=intro,1=align,2=confirm

  useEffect(() => { scanStateRef.current = scanState }, [scanState])

  // ── Camera init ─────────────────────────────────────────────────────────────
  useEffect(() => {
    let stream = null
    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        })
        const vid = videoRef.current
        if (!vid) return
        vid.srcObject = stream

        await new Promise((res, rej) => {
          vid.onloadedmetadata = res
          vid.onerror = rej
          setTimeout(rej, 8000)
        })

        try { await vid.play() } catch {}

        await new Promise((res) => {
          const chk = setInterval(() => {
            if (vid.readyState >= 2) { clearInterval(chk); res() }
          }, 50)
        })

        const track    = stream.getVideoTracks()[0]
        const settings = track.getSettings?.() ?? {}
        const sw = settings.width  || vid.videoWidth  || 640
        const sh = settings.height || vid.videoHeight || 480
        streamDims.current = { w: sw, h: sh }

        const focal = deriveFocalLength(track, sw)
        focalRef.current = focal
        setFocalInfo(`fx=${focal.fx.toFixed(0)}`)

        if (!loopsRef.current) {
          loopsRef.current = true
          startDetectionLoop()
          startRenderLoop()
          fpsTimer.current = setInterval(() => {
            setFps(fpsFrames.current)
            fpsFrames.current = 0
          }, 1000)
        }
      } catch (err) {
        console.error("Camera error:", err)
        setCamError(err.name === "NotAllowedError"
          ? "Camera permission denied. Please allow camera access and refresh."
          : "Could not start camera: " + err.message)
      }
    }
    start()
    return () => {
      loopsRef.current = false
      if (fpsTimer.current) clearInterval(fpsTimer.current)
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  // ── Detection loop ──────────────────────────────────────────────────────────
  function startDetectionLoop() {
    async function loop() {
      while (loopsRef.current) {
        if (document.visibilityState !== "hidden" && !scanningRef.current) {
          await captureAndDetect()
        }
        await sleep(300)
      }
    }
    loop()
  }

  function updateLerp(scene) {
    scene.forEach((obj) => {
      const id = obj.object_id
      if (!lerpedObjRef.current[id]) {
        lerpedObjRef.current[id] = { ...obj, currentBbox: [...obj.bbox], targetBbox: [...obj.bbox] }
      } else {
        lerpedObjRef.current[id].targetBbox  = [...obj.bbox]
        lerpedObjRef.current[id].dimensions  = obj.dimensions
        lerpedObjRef.current[id].confidence  = obj.confidence
      }
    })
    const ids = new Set(scene.map((o) => o.object_id))
    Object.keys(lerpedObjRef.current).forEach((id) => {
      if (!ids.has(parseInt(id))) delete lerpedObjRef.current[id]
    })
  }

  function lerpBboxes() {
    Object.values(lerpedObjRef.current).forEach((obj) => {
      const a = 0.2
      obj.currentBbox[0] += (obj.targetBbox[0] - obj.currentBbox[0]) * a
      obj.currentBbox[1] += (obj.targetBbox[1] - obj.currentBbox[1]) * a
      obj.currentBbox[2] += (obj.targetBbox[2] - obj.currentBbox[2]) * a
      obj.currentBbox[3] += (obj.targetBbox[3] - obj.currentBbox[3]) * a
    })
  }

  async function captureAndDetect() {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return

    try {
      const { w: sw, h: sh } = streamDims.current
      const tmp = document.createElement("canvas")
      tmp.width = sw; tmp.height = sh
      tmp.getContext("2d").drawImage(video, 0, 0, sw, sh)

      const blob = await new Promise((res) => tmp.toBlob(res, "image/jpeg", 0.85))
      if (!blob) return

      const focal = focalRef.current ?? { fx: 920, fy: 920 }
      const form  = new FormData()
      form.append("image",  blob, "frame.jpg")
      form.append("detect", "1")
      form.append("fx",     focal.fx.toFixed(2))
      form.append("fy",     focal.fy.toFixed(2))
      form.append("img_w",  sw)
      form.append("img_h",  sh)
      if (scaleFactorRef.current) form.append("scale", scaleFactorRef.current.toFixed(8))

      const res  = await fetch(`${BACKEND_URL}/detect`, { method: "POST", body: form })
      if (!res.ok) throw new Error(`${res.status}`)
      const data = await res.json()

      const scene = data.scene ?? []
      objectsRef.current = scene
      lastDetRef.current = scene
      updateLerp(scene)
      setObjects(scene)
      setMotion(data.motion ?? 0)
      setIsMoving((data.motion ?? 0) > 2)
      setConnected(true)
    } catch {
      setConnected(false)
    }
  }

  // ── Render loop ─────────────────────────────────────────────────────────────
  function startRenderLoop() {
    const canvas = canvasRef.current
    function render() {
      if (!loopsRef.current) return
      lerpBboxes()
      drawOverlays(Object.values(lerpedObjRef.current))
      fpsFrames.current++
      requestAnimationFrame(render)
    }
    requestAnimationFrame(render)
  }

  // ── Draw ────────────────────────────────────────────────────────────────────
  function drawOverlays(scene) {
    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas || !video || video.readyState < 2) return

    const ctx = canvas.getContext("2d")
    const { w: sw, h: sh } = streamDims.current

    if (canvas.width !== sw || canvas.height !== sh) {
      canvas.width  = sw
      canvas.height = sh
      canvasDims.current = { w: sw, h: sh }
    }

    ctx.drawImage(video, 0, 0, sw, sh)

    if (scanStateRef.current === "calibrate") {
      drawCardOverlay(ctx, sw, sh)
      return
    }

    scene.forEach((obj) => {
      const bbox = obj.currentBbox ?? obj.bbox
      if (!bbox) return
      const [x1, y1, x2, y2] = bbox
      const { r, g, b }       = colorFor(obj.label)
      const bw = x2 - x1, bh = y2 - y1

      ctx.save(); ctx.globalAlpha = 0.18
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillRect(x1, y1, bw, bh)
      ctx.restore()

      ctx.strokeStyle = `rgb(${r},${g},${b})`; ctx.lineWidth = 2.5
      ctx.strokeRect(x1, y1, bw, bh)

      const cs = Math.min(18, bw * 0.18, bh * 0.18); ctx.lineWidth = 3
      ;[[x1, y1+cs, x1, y1, x1+cs, y1],[x2-cs, y1, x2, y1, x2, y1+cs],
        [x1, y2-cs, x1, y2, x1+cs, y2],[x2-cs, y2, x2, y2, x2, y2-cs]]
        .forEach(([ax,ay,bx,by,cx2,cy2]) => {
          ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.lineTo(cx2,cy2); ctx.stroke()
        })

      const label = `${obj.label}  ${(obj.confidence * 100).toFixed(0)}%`
      ctx.font = `700 13px 'JetBrains Mono',monospace`
      const tw = ctx.measureText(label).width
      const py = Math.max(y1 - 28, 0)
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.beginPath(); ctx.roundRect(x1, py, tw+16, 26, 5); ctx.fill()
      ctx.fillStyle = "#0f0f14"; ctx.fillText(label, x1+8, py+18)

      if (obj.dimensions) {
        const d = obj.dimensions
        const lines = [
          `L ${d.length.toFixed(2)}m  W ${d.width.toFixed(2)}m  H ${d.height.toFixed(2)}m`,
          `Vol: ${d.volume_m3.toFixed(3)} m³`,
        ]
        ctx.font = `600 12px 'JetBrains Mono',monospace`
        const lh   = 18
        const boxW = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 16
        const boxH = lines.length * lh + 10
        ctx.fillStyle = "rgba(15,15,20,0.92)"
        ctx.beginPath(); ctx.roundRect(x1, y2+5, boxW, boxH, 5); ctx.fill()
        ctx.strokeStyle = `rgba(${r},${g},${b},0.5)`; ctx.lineWidth = 1; ctx.stroke()
        ctx.fillStyle = "#e8e8e2"
        lines.forEach((line, i) => ctx.fillText(line, x1+8, y2+5+lh*(i+1)))
      } else {
        ctx.font = "500 11px monospace"; ctx.fillStyle = "rgba(15,15,20,0.85)"
        ctx.beginPath(); ctx.roundRect(x1, y2+5, 120, 22, 4); ctx.fill()
        ctx.fillStyle = `rgba(${r},${g},${b},0.9)`; ctx.fillText("measuring…", x1+8, y2+20)
      }
    })
  }

  function drawCardOverlay(ctx, sw, sh) {
    const cr = cardRectRef.current
    const x = cr.x * sw, y = cr.y * sh, w = cr.w * sw, h = cr.h * sh

    ctx.fillStyle = "rgba(0,0,0,0.62)"; ctx.fillRect(0, 0, sw, sh)
    ctx.drawImage(videoRef.current, x, y, w, h, x, y, w, h)
    
    ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 3
    ctx.setLineDash([10, 6]); ctx.strokeRect(x, y, w, h); ctx.setLineDash([])
    
    const cs = 16; ctx.lineWidth = 5; ctx.strokeStyle = "#fbbf24"
    ;[[x,y+cs,x,y,x+cs,y],[x+w-cs,y,x+w,y,x+w,y+cs],
      [x,y+h-cs,x,y+h,x+cs,y+h],[x+w-cs,y+h,x+w,y+h,x+w,y+h-cs]]
      .forEach(([ax,ay,bx,by,cx2,cy2]) => {
        ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.lineTo(cx2,cy2); ctx.stroke()
      })
      
    // Resize Handle
    ctx.fillStyle = "#fbbf24"; ctx.beginPath(); ctx.arc(x+w, y+h, 18, 0, Math.PI*2); ctx.fill()
    ctx.fillStyle = "#0f0f14"; ctx.font = "bold 20px monospace"
    ctx.textAlign = "center"; ctx.textBaseline = "middle"
    ctx.fillText("⤡", x+w, y+h)
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"

    // Set reading
    setCardPx({ w: Math.round(w), h: Math.round(h) })
  }

  // ── Pointer Events (Unified Mouse/Touch) ────────────────────────────────────
  function getCanvasXY(e) {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    // e.clientX / e.clientY works universally for onPointer events
    return {
      x: (e.clientX - rect.left) / rect.width,
      y: (e.clientY - rect.top)  / rect.height,
    }
  }

  function onPointerDown(e) {
    if (scanStateRef.current !== "calibrate") return
    const { x, y } = getCanvasXY(e)
    const cr = cardRectRef.current
    const bx = cr.x + cr.w, by = cr.y + cr.h
    
    // Generous touch tolerance area for grabbing handle (10%)
    const tol = 0.10 
    
    if (Math.abs(x - bx) < tol && Math.abs(y - by) < tol) {
      draggingRef.current = "br"
      e.currentTarget.setPointerCapture(e.pointerId) // Locks drag to element
    } else if (x > cr.x && x < bx && y > cr.y && y < by) {
      draggingRef.current = { type: "move", ox: x - cr.x, oy: y - cr.y }
      e.currentTarget.setPointerCapture(e.pointerId)
    }
  }

  function onPointerMove(e) {
    if (!draggingRef.current || scanStateRef.current !== "calibrate") return
    const { x, y } = getCanvasXY(e)
    const cr = { ...cardRectRef.current }
    
    if (draggingRef.current === "br") {
      cr.w = Math.max(0.1, x - cr.x)
      cr.h = Math.max(0.05, y - cr.y)
    } else if (draggingRef.current?.type === "move") {
      cr.x = Math.max(0, Math.min(1 - cr.w, x - draggingRef.current.ox))
      cr.y = Math.max(0, Math.min(1 - cr.h, y - draggingRef.current.oy))
    }
    cardRectRef.current = cr
  }

  function onPointerUp(e) {
    if (draggingRef.current) {
      try { e.currentTarget.releasePointerCapture(e.pointerId) } catch (err) {}
      draggingRef.current = null
    }
  }

  // ── Calibration flow ────────────────────────────────────────────────────────
  function handleBeginCalibrate() {
    cardRectRef.current = { ...DEFAULT_CARD }
    setCalStep(0)
    setScanState("calibrate")
  }

  function handleConfirmCalibrate() {
    const { w: sw, h: sh } = streamDims.current
    const cr   = cardRectRef.current
    const pixW = cr.w * sw, pixH = cr.h * sh
    const scale = ((CARD_WIDTH_M / pixW) + (CARD_HEIGHT_M / pixH)) / 2
    scaleFactorRef.current = scale
    setScaleFactor(scale)
    setScanState("idle")
    console.log(`✅ scale=${scale.toFixed(8)} m/px  pixW=${pixW.toFixed(0)} pixH=${pixH.toFixed(0)}`)
  }

  // ── IMU ──────────────────────────────────────────────────────────────────────
  function startIMU() {
    if (typeof DeviceMotionEvent === "undefined") return
    const attach = () => {
      imuActiveRef.current = true
      window.addEventListener("devicemotion", (e) => {
        if (!imuActiveRef.current) return
        imuBufRef.current.push({
          ts: Date.now(),
          ax: e.acceleration?.x ?? 0, ay: e.acceleration?.y ?? 0, az: e.acceleration?.z ?? 0,
          gx: e.rotationRate?.alpha ?? 0, gy: e.rotationRate?.beta ?? 0, gz: e.rotationRate?.gamma ?? 0,
        })
        if (imuBufRef.current.length > 30) imuBufRef.current.shift()
      })
    }
    if (typeof DeviceMotionEvent.requestPermission === "function") {
      DeviceMotionEvent.requestPermission().then((s) => { if (s === "granted") attach() }).catch(() => {})
    } else { attach() }
  }
  function stopIMU() { imuActiveRef.current = false }

  // ── Scan ─────────────────────────────────────────────────────────────────────
  async function handleStartScan() {
    if (!scaleFactorRef.current) { alert("Calibrate first"); return }
    imuBufRef.current = []; acceptedRef.current = 0; skippedRef.current = 0
    flowRef.current.reset()
    setScanFrameCount(0); setScanSkipped(0); setScanHint("Move slowly around the box")
    scanningRef.current = true; setScanState("scanning")
    startIMU()
    await fetch(`${BACKEND_URL}/start_scan`, { method: "POST" })
    runScanLoop()
  }

  async function runScanLoop() {
    const video = videoRef.current
    while (scanningRef.current) {
      if (!video) break
      const mv = flowRef.current.compute(video)
      scanMotionRef.current = mv
      setScanHint(getScanHint(mv))
      if (mv >= SCAN_MOTION_MIN && mv <= SCAN_MOTION_MAX) {
        const { w: sw, h: sh } = streamDims.current
        const tmp = document.createElement("canvas"); tmp.width = sw; tmp.height = sh
        tmp.getContext("2d").drawImage(video, 0, 0, sw, sh)
        const blob = await new Promise((r) => tmp.toBlob(r, "image/jpeg", 0.82))
        const imuSnap = [...imuBufRef.current]; imuBufRef.current = []
        const result  = await sendScanFrame(blob, imuSnap)
        if (result?.status === "ok") { acceptedRef.current++; setScanFrameCount(acceptedRef.current) }
        else { skippedRef.current++; setScanSkipped(skippedRef.current) }
      } else { skippedRef.current++; setScanSkipped(skippedRef.current) }
      await sleep(150)
    }
  }

  async function sendScanFrame(blob, imu) {
    const focal = focalRef.current ?? { fx: 920, fy: 920 }
    const form  = new FormData()
    form.append("image",  blob, "frame.jpg")
    form.append("imu",    JSON.stringify(imu))
    form.append("fx",     focal.fx.toFixed(2))
    form.append("fy",     focal.fy.toFixed(2))
    form.append("img_w",  streamDims.current.w)
    form.append("img_h",  streamDims.current.h)
    if (scaleFactorRef.current) form.append("scale", scaleFactorRef.current.toFixed(8))
    try {
      const res = await fetch(`${BACKEND_URL}/scan_frame`, { method: "POST", body: form })
      return await res.json()
    } catch { return null }
  }

  async function handleDoneScan() {
    scanningRef.current = false; stopIMU(); setScanState("computing")
    try {
      const res  = await fetch(`${BACKEND_URL}/compute_dimensions`, { method: "POST" })
      const data = await res.json()
      if (data.error) { alert(`Scan error: ${data.error}`); setScanState("idle"); return }
      setFinalDims(data.dimensions); setScanState("result"); setShowSidebar(true)
    } catch { setScanState("idle") }
  }

  function handleScanAgain() {
    setFinalDims(null); setScanFrameCount(0); setScanSkipped(0); setScanHint(""); setScanState("idle")
  }

  const getScanHint = (mv) => {
    if (mv < SCAN_MOTION_MIN) return "Move — camera too still"
    if (mv > SCAN_MOTION_MAX) return "Slow down — too fast"
    return "✓ Good speed — keep going"
  }

  // ══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ══════════════════════════════════════════════════════════════════════════
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Syne:wght@700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        :root{
          --bg:#0f0f14;--bg2:#16161f;--bg3:#1e1e2a;--surface:#22222e;
          --border:#2e2e3e;--accent:#fbbf24;--accent2:#63c6b4;
          --text:#e8e6dc;--muted:#9090a0;--danger:#f87171;--ok:#4ade80;
          --mono:'JetBrains Mono','Courier New',monospace;
          --sans:'Syne',sans-serif;
        }
        body{background:var(--bg);overscroll-behavior:none}
        .root{display:flex;flex-direction:column;height:100dvh;background:var(--bg);color:var(--text);font-family:var(--mono);overflow:hidden}

        /* Camera wrap - Ensure touch-action none strictly applies here */
        .cam-wrap{position:relative;flex:0 0 auto;background:#000;user-select:none;touch-action:none}
        .cam-wrap canvas{display:block;width:100%;height:auto;aspect-ratio:16/9;max-height:56dvh;object-fit:cover;touch-action:none}

        /* Video hidden but active */
        .cam-wrap video{position:absolute;inset:0;opacity:0;pointer-events:none;width:100%;height:100%}

        /* Camera error */
        .cam-error{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#000;color:var(--danger);font-size:13px;text-align:center;padding:24px;line-height:1.7}

        /* HUD overlay */
        .hud{position:absolute;bottom:56px;left:0;right:0;display:flex;justify-content:center;padding:0 12px;pointer-events:none}
        .hud>*{pointer-events:all}

        /* Panels */
        .panel{background:rgba(15,15,20,0.96);border:1px solid rgba(251,191,36,.5);border-radius:16px;padding:20px;display:flex;flex-direction:column;align-items:center;gap:12px;width:100%;max-width:360px;backdrop-filter:blur(10px)}
        .panel-title{font-family:var(--sans);font-weight:800;font-size:15px;color:var(--accent);text-align:center}
        .panel-note{font-size:11px;color:var(--muted);text-align:center;line-height:1.8;max-width:280px}
        .panel-note strong{color:var(--text)}

        /* Status bar */
        .status-bar{position:absolute;bottom:0;left:0;right:0;padding:5px 12px;background:rgba(15,15,20,0.97);border-top:1px solid var(--border);display:flex;align-items:center;gap:8px;font-size:10px;color:var(--muted);flex-wrap:nowrap;overflow:hidden}
        .pill{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:2px 8px;font-size:10px;color:var(--muted);white-space:nowrap;flex-shrink:0}
        .scale-pill{background:rgba(99,198,180,.12);border:1px solid rgba(99,198,180,.4);border-radius:20px;padding:2px 8px;font-size:10px;color:var(--accent2);flex-shrink:0}
        .motion-dot{width:6px;height:6px;border-radius:50%;background:var(--muted);flex-shrink:0;transition:background .3s}
        .motion-dot.on{background:var(--accent)}

        /* Buttons */
        .btn-y{background:var(--accent);color:var(--bg);border:none;border-radius:10px;padding:12px 24px;font-size:12px;font-weight:800;font-family:var(--mono);cursor:pointer;letter-spacing:.12em;text-transform:uppercase;box-shadow:0 2px 16px rgba(251,191,36,.25);transition:transform .12s,box-shadow .12s;white-space:nowrap}
        .btn-y:active{transform:scale(.97);box-shadow:0 1px 6px rgba(251,191,36,.2)}
        .btn-y:disabled{background:var(--surface);color:var(--muted);box-shadow:none;cursor:not-allowed}
        .btn-s{background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:10px;padding:10px 18px;font-size:11px;font-weight:600;font-family:var(--mono);cursor:pointer;letter-spacing:.08em;white-space:nowrap}
        .btn-g{background:transparent;color:var(--muted);border:1px solid var(--border);border-radius:6px;padding:4px 10px;font-size:11px;font-family:var(--mono);cursor:pointer;flex-shrink:0}
        .btn-g.on{background:rgba(251,191,36,.1);border-color:var(--accent);color:var(--accent)}
        .btn-row{display:flex;gap:10px;width:100%;justify-content:center;flex-wrap:wrap}

        /* Scan hint */
        .hint{font-size:13px;font-weight:700;letter-spacing:.06em;text-align:center;padding:6px 14px;border-radius:8px}
        .hint.ok{color:var(--ok);background:rgba(74,222,128,.1)}
        .hint.warn{color:var(--danger);background:rgba(248,113,113,.1)}
        .hint.info{color:var(--accent);background:rgba(251,191,36,.1)}

        /* Scan stats */
        .scan-stats{display:flex;gap:20px;font-size:11px;color:var(--muted)}
        .scan-stats strong{color:var(--text)}

        /* Computing overlay */
        .computing{position:absolute;inset:0;background:rgba(15,15,20,.88);display:flex;align-items:center;justify-content:center;flex-direction:column;gap:14px;backdrop-filter:blur(4px)}
        .spinner{width:32px;height:32px;border:3px solid var(--accent);border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite}
        .computing-lbl{font-size:12px;color:var(--accent);letter-spacing:.18em;font-weight:700}
        @keyframes spin{to{transform:rotate(360deg)}}

        /* Idle CTA */
        .idle-cta{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}

        /* ── Calibration overlay ── */
        .cal-overlay{position:absolute;inset:0;display:flex;flex-direction:column;pointer-events:none}
        .cal-bottom{position:absolute;bottom:56px;left:0;right:0;display:flex;justify-content:center;padding:0 16px;pointer-events:all}

        /* Calibration steps */
        .cal-step{background:rgba(10,10,16,0.97);border:1px solid rgba(251,191,36,.55);border-radius:18px;padding:22px 20px;display:flex;flex-direction:column;align-items:center;gap:14px;width:100%;max-width:340px;backdrop-filter:blur(12px)}
        .cal-icon{font-size:32px}
        .cal-title{font-family:var(--sans);font-weight:800;font-size:16px;color:var(--accent);text-align:center}
        .cal-body{font-size:12px;color:var(--muted);text-align:center;line-height:1.85;max-width:280px}
        .cal-body strong{color:var(--text)}
        .cal-pxinfo{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.3);border-radius:8px;padding:8px 16px;font-size:11px;color:var(--accent);text-align:center;width:100%;font-family:var(--mono)}
        .cal-dots{display:flex;gap:8px;justify-content:center}
        .cal-dot{width:8px;height:8px;border-radius:50%;background:var(--border);transition:background .2s}
        .cal-dot.active{background:var(--accent)}

        /* Sidebar */
        .sidebar{flex:1;overflow-y:auto;background:var(--bg2);padding:14px;display:flex;flex-direction:column;gap:10px}
        .sidebar::-webkit-scrollbar{width:4px}
        .sidebar::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
        .s-header{display:flex;justify-content:space-between;align-items:center;padding-bottom:10px;border-bottom:1px solid var(--border);margin-bottom:2px}
        .brand{font-family:var(--sans);font-weight:800;font-size:15px;color:var(--text)}
        .s-count{font-size:10px;color:var(--muted);letter-spacing:.1em}
        .nudge{background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.22);border-radius:8px;padding:10px 14px;font-size:11px;color:var(--accent);line-height:1.7;text-align:center}
        .obj-card{background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:12px}
        .obj-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
        .obj-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
        .obj-name{font-size:13px;font-weight:700;font-family:var(--sans);color:var(--text)}
        .obj-id{font-size:10px;color:var(--muted)}
        .conf{font-size:11px;font-weight:600;padding:3px 8px;border-radius:20px}
        .dims-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:6px}
        .dim-cell{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:7px 6px;text-align:center}
        .dim-lbl{font-size:9px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:3px}
        .dim-val{font-size:14px;font-weight:700;font-family:var(--sans);color:var(--text)}
        .dim-unit{font-size:9px;color:var(--muted);font-weight:400}
        .vol-row{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:8px 12px;display:flex;justify-content:space-between;align-items:center}
        .vol-lbl{font-size:9px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase}
        .vol-val{font-size:17px;font-weight:800;font-family:var(--sans)}
        .res-card{background:var(--bg3);border:1px solid rgba(251,191,36,.4);border-radius:12px;padding:16px;margin-bottom:4px}
        .res-title{font-size:9px;color:var(--accent);letter-spacing:.2em;text-transform:uppercase;margin-bottom:12px;font-weight:700}
        .res-actions{display:flex;gap:8px;margin-top:12px}
        .empty{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;gap:10px;color:var(--muted);font-size:13px;text-align:center;padding:24px}
        .empty-icon{font-size:40px;opacity:.25}
      `}</style>

      <div className="root">

        {/* ── CAMERA SECTION ── */}
        <div
          className="cam-wrap"
          onPointerDown={onPointerDown} 
          onPointerMove={onPointerMove} 
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          style={{ touchAction: "none" }}
        >
          {/* Video element — hidden but must be in DOM for drawImage */}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ position: "absolute", top: 0, left: 0, width: "1px", height: "1px", opacity: 0.01, pointerEvents: "none" }}
          />

          {/* Canvas */}
          <canvas ref={canvasRef} />

          {/* Camera error */}
          {camError && <div className="cam-error">{camError}</div>}

          {/* ── CALIBRATE HUD ── */}
          {scanState === "calibrate" && (
            <div className="cal-bottom">
              {calStep === 0 && (
                <div className="cal-step">
                  <div className="cal-icon">💳</div>
                  <div className="cal-title">Card Calibration</div>
                  <p className="cal-body">
                    Place a <strong>standard credit card</strong> flat on top of your box (same distance from camera).
                    We'll use it as a known size reference.
                  </p>
                  <div className="cal-dots">
                    <div className="cal-dot active" /><div className="cal-dot" /><div className="cal-dot" />
                  </div>
                  <div className="btn-row">
                    <button className="btn-s" onClick={() => setScanState("idle")}>Cancel</button>
                    <button className="btn-y" onClick={() => setCalStep(1)}>Next →</button>
                  </div>
                </div>
              )}

              {calStep === 1 && (
                <div className="cal-step">
                  <div className="cal-icon">🎯</div>
                  <div className="cal-title">Align the Box</div>
                  <p className="cal-body">
                    <strong>Drag the yellow box</strong> to match your card.<br />
                    <strong>Drag the ⤡ corner</strong> to resize it.
                  </p>
                  <div className="cal-pxinfo">
                    {cardPx.w} × {cardPx.h} px
                  </div>
                  <div className="cal-dots">
                    <div className="cal-dot" /><div className="cal-dot active" /><div className="cal-dot" />
                  </div>
                  <div className="btn-row">
                    <button className="btn-s" onClick={() => setCalStep(0)}>← Back</button>
                    <button className="btn-y" onClick={() => setCalStep(2)}>Looks Good →</button>
                  </div>
                </div>
              )}

              {calStep === 2 && (
                <div className="cal-step">
                  <div className="cal-icon">✅</div>
                  <div className="cal-title">Confirm Scale</div>
                  <p className="cal-body">
                    The box is aligned to <strong>{cardPx.w}×{cardPx.h}px</strong>.<br />
                    Scale will be set to <strong>
                      {cardPx.w > 0 ? (((CARD_WIDTH_M / cardPx.w) + (CARD_HEIGHT_M / cardPx.h)) / 2 * 1000).toFixed(3) : "—"} mm/px
                    </strong>.
                  </p>
                  <div className="cal-dots">
                    <div className="cal-dot" /><div className="cal-dot" /><div className="cal-dot active" />
                  </div>
                  <div className="btn-row">
                    <button className="btn-s" onClick={() => setCalStep(1)}>← Adjust</button>
                    <button className="btn-y" onClick={handleConfirmCalibrate}>✓ Save Scale</button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── IDLE HUD ── */}
          {scanState === "idle" && (
            <div className="hud">
              <div className="idle-cta">
                {!scaleFactor
                  ? <button className="btn-y" onClick={handleBeginCalibrate}>📐 Calibrate Card</button>
                  : <>
                      <button className="btn-s" onClick={handleBeginCalibrate}>Re-calibrate</button>
                      {objects.length > 0 && (
                        <button className="btn-y" onClick={handleStartScan}>▶ Start Scan</button>
                      )}
                    </>
                }
              </div>
            </div>
          )}

          {/* ── SCANNING HUD ── */}
          {scanState === "scanning" && (
            <div className="hud">
              <div className="panel">
                <p className={`hint ${scanHint.startsWith("✓") ? "ok" : scanHint.includes("too") || scanHint.includes("Slow") ? "warn" : "info"}`}>
                  {scanHint || "Move around the box"}
                </p>
                <ScanMotionBar motionRef={scanMotionRef} min={SCAN_MOTION_MIN} max={SCAN_MOTION_MAX} />
                <div className="scan-stats">
                  <span>✅ <strong>{scanFrameCount}</strong> accepted</span>
                  <span>⏭ <strong>{scanSkipped}</strong> skipped</span>
                </div>
                <button
                  className="btn-y"
                  onClick={handleDoneScan}
                  disabled={scanFrameCount < 3}
                  style={{ width: "100%" }}
                >
                  {scanFrameCount >= 3 ? "Done — Compute Dimensions" : `Need ${3 - scanFrameCount} more frames`}
                </button>
              </div>
            </div>
          )}

          {/* ── COMPUTING ── */}
          {scanState === "computing" && (
            <div className="computing">
              <div className="spinner" />
              <div className="computing-lbl">COMPUTING…</div>
            </div>
          )}

          {/* ── STATUS BAR ── */}
          <div className="status-bar">
            <div className={`motion-dot${isMoving ? " on" : ""}`} />
            <span style={{ fontSize: 10, color: connected ? "var(--accent)" : "var(--danger)", flexShrink: 0 }}>
              {connected ? "live" : "disconnected"}
            </span>
            <span className="pill">{fps} fps</span>
            <span className="pill">{objects.length} obj</span>
            {scaleFactor && (
              <span className="scale-pill">⚖ {(scaleFactor * 1000).toFixed(3)} mm/px</span>
            )}
            <button
              className={`btn-g${showSidebar ? " on" : ""}`}
              style={{ marginLeft: "auto" }}
              onClick={() => setShowSidebar((s) => !s)}
            >
              {showSidebar ? "hide" : `results (${objects.length})`}
            </button>
          </div>
        </div>

        {/* ── SIDEBAR ── */}
        {(showSidebar || objects.length === 0 || scanState === "result") && (
          <div className="sidebar">
            <div className="s-header">
              <span className="brand">ParcelVision</span>
              <span className="s-count">{objects.length} DETECTED</span>
            </div>

            {!scaleFactor && scanState !== "calibrate" && (
              <div className="nudge">
                ⚠️ No scale set — tap <strong>Calibrate Card</strong> for accurate measurements.
              </div>
            )}

            {scanState === "result" && finalDims && (
              <div className="res-card">
                <div className="res-title">📦 Scan Result</div>
                <div className="dims-grid">
                  {[{ l: "L", v: finalDims.length },{ l: "W", v: finalDims.width },{ l: "H", v: finalDims.height }].map(({ l, v }) => (
                    <div className="dim-cell" key={l}>
                      <div className="dim-lbl">{l}</div>
                      <div className="dim-val">{v.toFixed(2)}<span className="dim-unit">m</span></div>
                    </div>
                  ))}
                </div>
                <div className="vol-row">
                  <span className="vol-lbl">Volume</span>
                  <span className="vol-val" style={{ color: "var(--accent)" }}>
                    {finalDims.volume_m3.toFixed(3)} <span style={{ fontSize: 11 }}>m³</span>
                  </span>
                </div>
                <div className="res-actions">
                  <button className="btn-s" style={{ flex: 1 }} onClick={handleScanAgain}>Scan Again</button>
                  <button className="btn-y" style={{ flex: 1 }} onClick={handleBeginCalibrate}>Re-calibrate</button>
                </div>
              </div>
            )}

            {objects.length === 0 && scanState === "idle" && (
              <div className="empty">
                <div className="empty-icon">📦</div>
                <span>Point camera at a box to begin</span>
                {!scaleFactor && (
                  <span style={{ fontSize: 11, color: "var(--muted)" }}>Calibrate first for accurate measurements</span>
                )}
                <button className="btn-y" style={{ marginTop: 8 }} onClick={handleBeginCalibrate}>
                  📐 Calibrate Card
                </button>
              </div>
            )}

            {objects.map((obj) => {
              const { r, g, b } = colorFor(obj.label)
              const d = obj.dimensions
              return (
                <div className="obj-card" key={obj.object_id} style={{ borderColor: `rgba(${r},${g},${b},.35)` }}>
                  <div className="obj-head">
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div className="obj-dot" style={{ background: `rgb(${r},${g},${b})` }} />
                      <span className="obj-name">{obj.label}</span>
                      <span className="obj-id">#{obj.object_id}</span>
                    </div>
                    <span className="conf" style={{ background: `rgba(${r},${g},${b},.12)`, color: `rgb(${r},${g},${b})` }}>
                      {(obj.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  {d ? (
                    <>
                      <div className="dims-grid">
                        {[{ l: "L", v: d.length },{ l: "W", v: d.width },{ l: "H", v: d.height }].map(({ l, v }) => (
                          <div className="dim-cell" key={l}>
                            <div className="dim-lbl">{l}</div>
                            <div className="dim-val" style={{ fontSize: 13 }}>{v.toFixed(2)}<span className="dim-unit">m</span></div>
                          </div>
                        ))}
                      </div>
                      <div className="vol-row">
                        <span className="vol-lbl">Volume</span>
                        <span className="vol-val" style={{ fontSize: 15, color: `rgb(${r},${g},${b})` }}>
                          {d.volume_m3.toFixed(3)} <span style={{ fontSize: 10 }}>m³</span>
                        </span>
                      </div>
                    </>
                  ) : (
                    <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>measuring…</div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </>
  )
}

// ── Scan motion bar ───────────────────────────────────────────────────────────
function ScanMotionBar({ motionRef, min, max }) {
  const barRef = useRef(null)
  useEffect(() => {
    let raf
    function tick() {
      if (barRef.current) {
        const mv  = motionRef.current
        const pct = Math.min(100, (mv / max) * 100)
        barRef.current.style.width      = `${pct}%`
        barRef.current.style.background = mv < min ? "#646579" : mv > max ? "#f87171" : "#4ade80"
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])
  return (
    <div style={{ width: "100%", height: 6, background: "var(--border)", borderRadius: 4, overflow: "hidden" }}>
      <div ref={barRef} style={{ height: "100%", width: "0%", borderRadius: 4, transition: "width .1s" }} />
    </div>
  )
}