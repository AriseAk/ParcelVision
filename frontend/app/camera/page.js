// frontend/app/camera/page.js
"use client"

import { useEffect, useRef, useState } from "react"
import { useRouter } from "next/navigation"

// ── Config ────────────────────────────────────────────────────────────────────
const BACKEND_URL = (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_BACKEND_URL) || "http://127.0.0.1:5000"
const SCAN_MOTION_MIN = 0.3
const SCAN_MOTION_MAX = 150.0
const CARD_WIDTH_M    = 0.0856
const CARD_HEIGHT_M   = 0.05398

const CLASS_COLORS = {
  "box":           { r: 251, g: 191, b: 36  }, // amber
  "cardboard box": { r: 251, g: 191, b: 36  },
  carton:          { r: 251, g: 191, b: 36  },
  parcel:          { r: 99,  g: 202, b: 183 }, // teal
  package:         { r: 99,  g: 202, b: 183 },
  container:       { r: 167, g: 139, b: 250 }, // purple
  "brown box":     { r: 251, g: 146, b: 60  }, // orange
}
const colorFor = (label) => CLASS_COLORS[label?.toLowerCase()] ?? { r: 251, g: 191, b: 36 }
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

export default function CameraPage() {
  const router = useRouter()
  // refs
  const videoRef       = useRef(null)
  const canvasRef      = useRef(null)
  const loopsRef       = useRef(false)
  const objectsRef     = useRef([])
  const focalRef       = useRef(null)
  const streamDims     = useRef({ w: 640, h: 480 })
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
  const [showSidebar,    setShowSidebar]    = useState(false)
  const [scanState,      setScanState]      = useState("idle")
  const [finalDims,      setFinalDims]      = useState(null)
  const [scanFrameCount, setScanFrameCount] = useState(0)
  const [scanHint,       setScanHint]       = useState("")
  const [scaleFactor,    setScaleFactor]    = useState(null)
  const [camError,       setCamError]       = useState(null)
  const [cardPx,         setCardPx]         = useState({ w: 0, h: 0 })
  const [calStep,        setCalStep]        = useState(0)

  useEffect(() => { scanStateRef.current = scanState }, [scanState])

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

        const track    = stream.getVideoTracks()[0]
        const sw = vid.videoWidth  || 640
        const sh = vid.videoHeight || 480
        streamDims.current = { w: sw, h: sh }

        focalRef.current = deriveFocalLength(track, sw)

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
        setCamError("Camera permission denied or device unavailable.")
      }
    }
    start()
    return () => {
      loopsRef.current = false
      if (fpsTimer.current) clearInterval(fpsTimer.current)
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [])

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
      const a = 0.25
      obj.currentBbox[0] += (obj.targetBbox[0] - obj.currentBbox[0]) * a
      obj.currentBbox[1] += (obj.targetBbox[1] - obj.currentBbox[1]) * a
      obj.currentBbox[2] += (obj.targetBbox[2] - obj.currentBbox[2]) * a
      obj.currentBbox[3] += (obj.targetBbox[3] - obj.currentBbox[3]) * a
    })
  }

  async function captureAndDetect() {
    const video  = videoRef.current
    if (!video || video.readyState < 2) return

    try {
      const { w: sw, h: sh } = streamDims.current
      const tmp = document.createElement("canvas")
      tmp.width = sw; tmp.height = sh
      tmp.getContext("2d").drawImage(video, 0, 0, sw, sh)

      const blob = await new Promise((res) => tmp.toBlob(res, "image/jpeg", 0.8))
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
      setIsMoving((data.motion ?? 0) > 2)
      setConnected(true)
    } catch {
      setConnected(false)
    }
  }

  function startRenderLoop() {
    function render() {
      if (!loopsRef.current) return
      lerpBboxes()
      drawOverlays(Object.values(lerpedObjRef.current))
      fpsFrames.current++
      requestAnimationFrame(render)
    }
    requestAnimationFrame(render)
  }

  function drawOverlays(scene) {
    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas || !video || video.readyState < 2) return

    const ctx = canvas.getContext("2d")
    const { w: sw, h: sh } = streamDims.current

    if (canvas.width !== sw || canvas.height !== sh) {
      canvas.width  = sw
      canvas.height = sh
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

      ctx.save(); ctx.globalAlpha = 0.15
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillRect(x1, y1, bw, bh)
      ctx.restore()

      ctx.strokeStyle = `rgb(${r},${g},${b})`; ctx.lineWidth = 2.5
      ctx.strokeRect(x1, y1, bw, bh)

      const cs = Math.min(24, bw * 0.15, bh * 0.15); ctx.lineWidth = 4
      ;[[x1, y1+cs, x1, y1, x1+cs, y1],[x2-cs, y1, x2, y1, x2, y1+cs],
        [x1, y2-cs, x1, y2, x1+cs, y2],[x2-cs, y2, x2, y2, x2, y2-cs]]
        .forEach(([ax,ay,bx,by,cx2,cy2]) => {
          ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.lineTo(cx2,cy2); ctx.stroke()
        })

      const label = `${obj.label}  ${(obj.confidence * 100).toFixed(0)}%`
      ctx.font = `600 14px 'JetBrains Mono',monospace`
      const tw = ctx.measureText(label).width
      const py = Math.max(y1 - 32, 0)
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.beginPath(); ctx.roundRect(x1, py, tw+20, 28, 6); ctx.fill()
      ctx.fillStyle = "#0f0f14"; ctx.fillText(label, x1+10, py+19)

      if (obj.dimensions) {
        const d = obj.dimensions
        const lines = [
          `L ${d.length.toFixed(2)}m W ${d.width.toFixed(2)}m H ${d.height.toFixed(2)}m`,
          `Vol: ${d.volume_m3.toFixed(3)} m³`,
        ]
        ctx.font = `600 13px 'JetBrains Mono',monospace`
        const lh   = 20
        const boxW = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 20
        const boxH = lines.length * lh + 12
        ctx.fillStyle = "rgba(15,15,20,0.95)"
        ctx.beginPath(); ctx.roundRect(x1, y2+8, boxW, boxH, 8); ctx.fill()
        ctx.strokeStyle = `rgba(${r},${g},${b},0.6)`; ctx.lineWidth = 1; ctx.stroke()
        ctx.fillStyle = "#e8e6dc"
        lines.forEach((line, i) => ctx.fillText(line, x1+10, y2+8+lh*(i+1)))
      }
    })
  }

  function drawCardOverlay(ctx, sw, sh) {
    const cr = cardRectRef.current
    const x = cr.x * sw, y = cr.y * sh, w = cr.w * sw, h = cr.h * sh

    ctx.fillStyle = "rgba(0,0,0,0.65)"; ctx.fillRect(0, 0, sw, sh)
    ctx.drawImage(videoRef.current, x, y, w, h, x, y, w, h)
    
    ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2
    ctx.setLineDash([8, 8]); ctx.strokeRect(x, y, w, h); ctx.setLineDash([])
    
    const cs = 20; ctx.lineWidth = 4; ctx.strokeStyle = "#fbbf24"
    ;[[x,y+cs,x,y,x+cs,y],[x+w-cs,y,x+w,y,x+w,y+cs],
      [x,y+h-cs,x,y+h,x+cs,y+h],[x+w-cs,y+h,x+w,y+h,x+w,y+h-cs]]
      .forEach(([ax,ay,bx,by,cx2,cy2]) => {
        ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.lineTo(cx2,cy2); ctx.stroke()
      })
      
    // Resize Handle (Bottom Right)
    ctx.fillStyle = "#fbbf24"; ctx.beginPath(); ctx.arc(x+w, y+h, 24, 0, Math.PI*2); ctx.fill()
    ctx.fillStyle = "#0f0f14"; ctx.font = "bold 24px monospace"
    ctx.textAlign = "center"; ctx.textBaseline = "middle"
    ctx.fillText("⤡", x+w, y+h)
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"

    setCardPx({ w: Math.round(w), h: Math.round(h) })
  }

  // ── Unified Pointer Events (Touch & Mouse) ──────────────────────────────────
  function getCanvasXY(e) {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
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
    
    const tol = 0.15 // Generous 15% touch tolerance for the handle
    
    if (Math.abs(x - bx) < tol && Math.abs(y - by) < tol) {
      draggingRef.current = "br"
      e.currentTarget.setPointerCapture(e.pointerId)
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
      try { e.currentTarget.releasePointerCapture(e.pointerId) } catch {}
      draggingRef.current = null
    }
  }

  // ── Flows ───────────────────────────────────────────────────────────────────
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
  }

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

  async function handleStartScan() {
    if (!scaleFactorRef.current) return
    imuBufRef.current = []; acceptedRef.current = 0; skippedRef.current = 0
    flowRef.current.reset()
    setScanFrameCount(0); setScanHint("Move slowly around the box")
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
      setScanHint(mv < SCAN_MOTION_MIN ? "Move — camera too still" : mv > SCAN_MOTION_MAX ? "Slow down — too fast" : "✓ Good speed")
      
      if (mv >= SCAN_MOTION_MIN && mv <= SCAN_MOTION_MAX) {
        const { w: sw, h: sh } = streamDims.current
        const tmp = document.createElement("canvas"); tmp.width = sw; tmp.height = sh
        tmp.getContext("2d").drawImage(video, 0, 0, sw, sh)
        const blob = await new Promise((r) => tmp.toBlob(r, "image/jpeg", 0.8))
        
        const imuSnap = [...imuBufRef.current]; imuBufRef.current = []
        const form  = new FormData()
        form.append("image", blob, "frame.jpg")
        form.append("imu", JSON.stringify(imuSnap))
        form.append("fx", focalRef.current.fx.toFixed(2))
        form.append("scale", scaleFactorRef.current.toFixed(8))
        
        try {
          const res = await fetch(`${BACKEND_URL}/scan_frame`, { method: "POST", body: form })
          const data = await res.json()
          if (data.status === "ok") { acceptedRef.current++; setScanFrameCount(acceptedRef.current) }
          else skippedRef.current++
        } catch {}
      }
      await sleep(150)
    }
  }

  async function handleDoneScan() {
    scanningRef.current = false; imuActiveRef.current = false; setScanState("computing")
    try {
      const res  = await fetch(`${BACKEND_URL}/compute_dimensions`, { method: "POST" })
      const data = await res.json()
      if (data.error) { alert(`Scan error: ${data.error}`); setScanState("idle"); return }
      setFinalDims(data.dimensions); setScanState("result"); setShowSidebar(true)
    } catch { setScanState("idle") }
  }

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Syne:wght@700;800&display=swap');
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

        .cam-wrap{position:relative;flex:0 0 auto;background:#000;user-select:none;touch-action:none}
        .cam-wrap canvas{display:block;width:100%;height:auto;aspect-ratio:4/3;max-height:60dvh;object-fit:cover;touch-action:none}
        .cam-wrap video{position:absolute;inset:0;opacity:0;pointer-events:none;width:100%;height:100%}

        .cam-error{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#000;color:var(--danger);font-size:14px;text-align:center;padding:24px;line-height:1.7}

        .hud{position:absolute;bottom:64px;left:0;right:0;display:flex;justify-content:center;padding:0 16px;pointer-events:none}
        .hud>*{pointer-events:all}

        .panel{background:rgba(15,15,20,0.95);border:1px solid rgba(251,191,36,.4);border-radius:16px;padding:24px;display:flex;flex-direction:column;align-items:center;gap:16px;width:100%;max-width:380px;backdrop-filter:blur(10px)}
        
        .status-bar{position:absolute;bottom:0;left:0;right:0;padding:8px 16px;background:rgba(15,15,20,0.98);border-top:1px solid var(--border);display:flex;align-items:center;gap:12px;font-size:11px;color:var(--muted);z-index:10}
        .pill{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:4px 10px;font-weight:600;color:var(--text)}
        .scale-pill{background:rgba(99,198,180,.12);border:1px solid rgba(99,198,180,.4);border-radius:20px;padding:4px 10px;font-weight:600;color:var(--accent2)}
        .motion-dot{width:8px;height:8px;border-radius:50%;background:var(--muted);transition:background .3s}
        .motion-dot.on{background:var(--accent)}

        .btn-y{background:var(--accent);color:var(--bg);border:none;border-radius:12px;padding:14px 24px;font-size:14px;font-weight:800;font-family:var(--mono);cursor:pointer;letter-spacing:.1em;text-transform:uppercase;box-shadow:0 4px 16px rgba(251,191,36,.2);transition:transform .1s;white-space:nowrap}
        .btn-y:active{transform:scale(.96)}
        .btn-y:disabled{background:var(--surface);color:var(--muted);box-shadow:none;cursor:not-allowed}
        .btn-s{background:transparent;color:var(--text);border:2px solid var(--border);border-radius:12px;padding:12px 20px;font-size:13px;font-weight:600;font-family:var(--mono);cursor:pointer;letter-spacing:.05em}
        
        .cal-overlay{position:absolute;inset:0;display:flex;flex-direction:column;pointer-events:none}
        .cal-bottom{position:absolute;bottom:64px;left:0;right:0;display:flex;justify-content:center;padding:0 16px;pointer-events:all}
        .cal-step{background:rgba(10,10,16,0.98);border:1px solid rgba(251,191,36,.6);border-radius:20px;padding:24px;display:flex;flex-direction:column;align-items:center;gap:16px;width:100%;max-width:360px;backdrop-filter:blur(12px)}
        .cal-title{font-family:var(--sans);font-weight:800;font-size:18px;color:var(--accent)}
        .cal-body{font-size:13px;color:var(--muted);text-align:center;line-height:1.7}
        .cal-pxinfo{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.3);border-radius:8px;padding:10px;font-size:12px;color:var(--accent);font-family:var(--mono)}

        .sidebar{flex:1;overflow-y:auto;background:var(--bg2);padding:16px;display:flex;flex-direction:column;gap:12px}
        .s-header{display:flex;justify-content:space-between;align-items:center;padding-bottom:12px;border-bottom:1px solid var(--border)}
        .brand{font-family:var(--sans);font-weight:800;font-size:18px;color:var(--text)}
        
        .obj-card{background:var(--bg3);border:1px solid var(--border);border-radius:12px;padding:16px}
        .obj-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
        .obj-name{font-size:16px;font-weight:800;font-family:var(--sans);text-transform:uppercase}
        
        .dims-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px}
        .dim-cell{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center}
        .dim-lbl{font-size:10px;color:var(--muted);letter-spacing:.1em;margin-bottom:4px}
        .dim-val{font-size:16px;font-weight:800;font-family:var(--sans)}
        
        .vol-row{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 16px;display:flex;justify-content:space-between;align-items:center}
        .vol-val{font-size:20px;font-weight:800;font-family:var(--sans)}

        .hint{font-size:14px;font-weight:800;padding:8px 16px;border-radius:8px}
        .hint.ok{color:var(--ok);background:rgba(74,222,128,.1)}
        .hint.warn{color:var(--danger);background:rgba(248,113,113,.1)}
      `}</style>

      <div className="root">
        {/* Top Header */}
        <div style={{ position: "absolute", top: 0, width: "100%", padding: "16px", zIndex: 20, display: "flex", justifyContent: "space-between" }}>
          <div onClick={() => router.push('/')} style={{ cursor: "pointer", background: "rgba(15,15,20,0.8)", padding: "8px 16px", borderRadius: "20px", fontWeight: 800, fontFamily: "var(--sans)" }}>
            ← Home
          </div>
        </div>

        <div className="cam-wrap" onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}>
          <video ref={videoRef} autoPlay playsInline muted style={{ position: "absolute", top: 0, width: "1px", opacity: 0.01 }} />
          <canvas ref={canvasRef} />
          {camError && <div className="cam-error">{camError}</div>}

          {scanState === "calibrate" && (
            <div className="cal-bottom">
              {calStep === 0 && (
                <div className="cal-step">
                  <div className="cal-title">💳 Calibration</div>
                  <p className="cal-body">Place a <strong>standard credit card</strong> flat on top of your box. We use it as a metric scale reference.</p>
                  <div style={{ display: "flex", gap: "12px", width: "100%" }}>
                    <button className="btn-s" style={{ flex: 1 }} onClick={() => setScanState("idle")}>Cancel</button>
                    <button className="btn-y" style={{ flex: 1 }} onClick={() => setCalStep(1)}>Next →</button>
                  </div>
                </div>
              )}
              {calStep === 1 && (
                <div className="cal-step">
                  <div className="cal-title">🎯 Align Box</div>
                  <p className="cal-body"><strong>Drag the box</strong> to match the card.<br/><strong>Drag the corner (⤡)</strong> to resize.</p>
                  <div className="cal-pxinfo">{cardPx.w} × {cardPx.h} px</div>
                  <div style={{ display: "flex", gap: "12px", width: "100%" }}>
                    <button className="btn-s" style={{ flex: 1 }} onClick={() => setCalStep(0)}>← Back</button>
                    <button className="btn-y" style={{ flex: 1 }} onClick={() => setCalStep(2)}>Align →</button>
                  </div>
                </div>
              )}
              {calStep === 2 && (
                <div className="cal-step">
                  <div className="cal-title">✅ Confirm</div>
                  <p className="cal-body">Box locked at <strong>{cardPx.w}×{cardPx.h}px</strong>.</p>
                  <div style={{ display: "flex", gap: "12px", width: "100%" }}>
                    <button className="btn-s" style={{ flex: 1 }} onClick={() => setCalStep(1)}>← Adjust</button>
                    <button className="btn-y" style={{ flex: 1 }} onClick={handleConfirmCalibrate}>✓ Save</button>
                  </div>
                </div>
              )}
            </div>
          )}

          {scanState === "idle" && (
            <div className="hud">
              <div style={{ display: "flex", gap: "12px" }}>
                {!scaleFactor ? (
                  <button className="btn-y" onClick={handleBeginCalibrate}>📐 Calibrate Card</button>
                ) : (
                  <>
                    <button className="btn-s" style={{ background: "rgba(15,15,20,0.9)" }} onClick={handleBeginCalibrate}>Re-calibrate</button>
                    {objects.length > 0 && <button className="btn-y" onClick={handleStartScan}>▶ Start Scan</button>}
                  </>
                )}
              </div>
            </div>
          )}

          {scanState === "scanning" && (
            <div className="hud">
              <div className="panel">
                <p className={`hint ${scanHint.includes("✓") ? "ok" : "warn"}`}>{scanHint}</p>
                <div style={{ display: "flex", justifyContent: "space-between", width: "100%", fontSize: "12px", color: "var(--muted)" }}>
                  <span>Frames: <strong style={{ color: "var(--text)" }}>{scanFrameCount}</strong> / 3 min</span>
                </div>
                <button className="btn-y" style={{ width: "100%" }} onClick={handleDoneScan} disabled={scanFrameCount < 3}>
                  {scanFrameCount >= 3 ? "Done — Compute" : `Need ${3 - scanFrameCount} more`}
                </button>
              </div>
            </div>
          )}

          {scanState === "computing" && (
            <div style={{ position: "absolute", inset: 0, background: "rgba(15,15,20,0.9)", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--accent)", fontWeight: 800, fontSize: "18px", letterSpacing: "0.2em" }}>
              COMPUTING…
            </div>
          )}

          <div className="status-bar">
            <div className={`motion-dot${isMoving ? " on" : ""}`} />
            <span style={{ color: connected ? "var(--accent)" : "var(--danger)" }}>{connected ? "LIVE" : "DISCONNECTED"}</span>
            <span className="pill">{fps} fps</span>
            <span className="pill">{objects.length} obj</span>
            {scaleFactor && <span className="scale-pill">⚖ {(scaleFactor * 1000).toFixed(2)} mm/px</span>}
            <button className="btn-s" style={{ marginLeft: "auto", padding: "4px 12px", fontSize: "11px", border: "1px solid var(--border)" }} onClick={() => setShowSidebar(!showSidebar)}>
              {showSidebar ? "HIDE" : "DATA"}
            </button>
          </div>
        </div>

        {/* Bottom Data View */}
        {(showSidebar || scanState === "result") && (
          <div className="sidebar">
            <div className="s-header">
              <span className="brand">Detected Items</span>
            </div>
            
            {!scaleFactor && scanState !== "calibrate" && (
              <div style={{ background: "rgba(251,191,36,0.1)", border: "1px solid rgba(251,191,36,0.3)", padding: "12px", borderRadius: "8px", color: "var(--accent)", fontSize: "13px", textAlign: "center" }}>
                ⚠️ Calibrate using a standard card first for dimensions.
              </div>
            )}

            {scanState === "result" && finalDims && (
              <div className="obj-card" style={{ borderColor: "var(--accent)", borderWidth: "2px" }}>
                <div className="obj-head">
                  <span className="obj-name" style={{ color: "var(--accent)" }}>Final Scan Result</span>
                </div>
                <div className="dims-grid">
                  {[{ l: "LENGTH", v: finalDims.length },{ l: "WIDTH", v: finalDims.width },{ l: "HEIGHT", v: finalDims.height }].map(({ l, v }) => (
                    <div className="dim-cell" key={l}>
                      <div className="dim-lbl">{l}</div>
                      <div className="dim-val">{v.toFixed(2)}<span style={{ fontSize: "10px", color: "var(--muted)", marginLeft: "2px" }}>m</span></div>
                    </div>
                  ))}
                </div>
                <div className="vol-row" style={{ marginBottom: "16px" }}>
                  <span className="dim-lbl" style={{ margin: 0 }}>VOLUME</span>
                  <span className="vol-val" style={{ color: "var(--accent)" }}>{finalDims.volume_m3.toFixed(3)} <span style={{ fontSize: "12px" }}>m³</span></span>
                </div>
                <div style={{ display: "flex", gap: "12px" }}>
                  <button className="btn-s" style={{ flex: 1 }} onClick={() => setScanState("idle")}>Scan Again</button>
                  <button className="btn-y" style={{ flex: 1 }} onClick={handleBeginCalibrate}>Re-calibrate</button>
                </div>
              </div>
            )}

            {objects.map((obj) => {
              const { r, g, b } = colorFor(obj.label)
              const d = obj.dimensions
              return (
                <div className="obj-card" key={obj.object_id} style={{ borderTop: `4px solid rgb(${r},${g},${b})` }}>
                  <div className="obj-head">
                    <span className="obj-name">{obj.label}</span>
                    <span style={{ background: `rgba(${r},${g},${b},0.15)`, color: `rgb(${r},${g},${b})`, padding: "4px 8px", borderRadius: "8px", fontSize: "12px", fontWeight: 800 }}>
                      {(obj.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  {d ? (
                    <>
                      <div className="dims-grid">
                        <div className="dim-cell"><div className="dim-lbl">L</div><div className="dim-val">{d.length.toFixed(2)}</div></div>
                        <div className="dim-cell"><div className="dim-lbl">W</div><div className="dim-val">{d.width.toFixed(2)}</div></div>
                        <div className="dim-cell"><div className="dim-lbl">H</div><div className="dim-val">{d.height.toFixed(2)}</div></div>
                      </div>
                      <div className="vol-row">
                        <span className="dim-lbl" style={{ margin: 0 }}>VOL</span>
                        <span className="vol-val" style={{ color: `rgb(${r},${g},${b})` }}>{d.volume_m3.toFixed(3)} <span style={{ fontSize: "12px" }}>m³</span></span>
                      </div>
                    </>
                  ) : (
                    <div style={{ color: "var(--muted)", fontSize: "13px", padding: "8px 0" }}>Awaiting Depth Map...</div>
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