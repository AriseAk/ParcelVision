"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { useRouter } from "next/navigation"

// ── Config ────────────────────────────────────────────────────────────────────
const BACKEND_URL    = (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_BACKEND_URL) || "http://127.0.0.1:5000"
const SCAN_MOTION_MIN = 0.3
const SCAN_MOTION_MAX = 150.0
const CARD_WIDTH_M    = 0.0856
const CARD_HEIGHT_M   = 0.05398
const CAL_TARGET_FRAMES = 12

const CLASS_COLORS = {
  "box":           { r: 251, g: 191, b: 36  },
  "cardboard box": { r: 251, g: 191, b: 36  },
  carton:          { r: 251, g: 191, b: 36  },
  parcel:          { r: 99,  g: 202, b: 183 },
  package:         { r: 99,  g: 202, b: 183 },
  container:       { r: 167, g: 139, b: 250 },
  "brown box":     { r: 251, g: 146, b: 60  },
}
const colorFor = (label) => CLASS_COLORS[label?.toLowerCase()] ?? { r: 251, g: 191, b: 36 }
const sleep    = (ms)    => new Promise((r) => setTimeout(r, ms))

const DEFAULT_CARD = { x: 0.2, y: 0.35, w: 0.6, h: 0.3 }

// ── Canvas helpers ───────────────────────────────────────────────────────────
function roundRectPath(ctx, x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2))
  ctx.beginPath()
  ctx.moveTo(x + rr, y)
  ctx.arcTo(x + w, y, x + w, y + h, rr)
  ctx.arcTo(x + w, y + h, x, y + h, rr)
  ctx.arcTo(x, y + h, x, y, rr)
  ctx.arcTo(x, y, x + w, y, rr)
  ctx.closePath()
}

function buildShapePath(ctx, obj, x1, y1, bw, bh, radius) {
  const contour = obj.contour ?? obj.mask_contour
  if (Array.isArray(contour) && contour.length > 2) {
    ctx.beginPath()
    ctx.moveTo(contour[0][0], contour[0][1])
    for (let i = 1; i < contour.length; i++) ctx.lineTo(contour[i][0], contour[i][1])
    ctx.closePath()
    return true
  }
  roundRectPath(ctx, x1, y1, bw, bh, radius)
  return false
}

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

// ── SAM result overlay (replaces AR box) ─────────────────────────────────────
function drawSAMResultOverlay(ctx, bbox, contour, dims, t) {
  if (!bbox) return
  const [x1, y1, x2, y2] = bbox
  const bw = x2 - x1, bh = y2 - y1
  if (bw <= 0 || bh <= 0) return

  // teal — matches the scanning overlay colour so result feels continuous
  const r = 99, g = 202, b = 183
  const pulse = 0.85 + 0.15 * Math.sin(t * 0.04)
  const hasContour = Array.isArray(contour) && contour.length > 2
  const radius = Math.min(14, bw * 0.08, bh * 0.08)

  ctx.save()

  // ── Draw shape path (SAM contour or bbox fallback) ───────────────────────
  if (hasContour) {
    ctx.beginPath()
    ctx.moveTo(contour[0][0], contour[0][1])
    for (let i = 1; i < contour.length; i++) ctx.lineTo(contour[i][0], contour[i][1])
    ctx.closePath()
  } else {
    roundRectPath(ctx, x1, y1, bw, bh, radius)
  }

  // Clipped fill
  ctx.save()
  ctx.clip()
  const grad = ctx.createLinearGradient(x1, y1, x1, y2)
  grad.addColorStop(0, `rgba(${r},${g},${b},0.18)`)
  grad.addColorStop(1, `rgba(${r},${g},${b},0.05)`)
  ctx.fillStyle = grad
  ctx.fillRect(x1 - 4, y1 - 4, bw + 8, bh + 8)

  // subtle scan line to show it's "locked"
  const scanY = y1 + bh * ((t % 80) / 80)
  const sg = ctx.createLinearGradient(x1, scanY - 14, x1, scanY + 14)
  sg.addColorStop(0,   `rgba(${r},${g},${b},0)`)
  sg.addColorStop(0.5, `rgba(${r},${g},${b},0.4)`)
  sg.addColorStop(1,   `rgba(${r},${g},${b},0)`)
  ctx.fillStyle = sg
  ctx.fillRect(x1, scanY - 14, bw, 28)
  ctx.restore()

  // ── Stroke the contour / bbox ─────────────────────────────────────────────
  if (hasContour) {
    ctx.beginPath()
    ctx.moveTo(contour[0][0], contour[0][1])
    for (let i = 1; i < contour.length; i++) ctx.lineTo(contour[i][0], contour[i][1])
    ctx.closePath()
  } else {
    roundRectPath(ctx, x1, y1, bw, bh, radius)
  }
  ctx.globalAlpha = pulse
  ctx.shadowColor = `rgba(${r},${g},${b},0.75)`
  ctx.shadowBlur  = 12
  ctx.strokeStyle = `rgb(${r},${g},${b})`
  ctx.lineWidth   = hasContour ? 2.5 : 2
  ctx.stroke()
  ctx.shadowBlur  = 0
  ctx.globalAlpha = 1

  // ── Corner accents on bbox corners (always, even with contour) ───────────
  if (!hasContour) {
    const cs = Math.min(22, bw * 0.15, bh * 0.15)
    ctx.lineWidth = 3.5; ctx.lineCap = "round"
    ctx.strokeStyle = `rgb(${r},${g},${b})`
    ctx.shadowColor = `rgba(${r},${g},${b},0.6)`
    ctx.shadowBlur  = 8
    ;[[x1,y1+cs,x1,y1,x1+cs,y1],[x2-cs,y1,x2,y1,x2,y1+cs],
      [x1,y2-cs,x1,y2,x1+cs,y2],[x2-cs,y2,x2,y2,x2,y2-cs]]
      .forEach(([ax,ay,bx,by,cx2,cy2]) => {
        ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.lineTo(cx2,cy2); ctx.stroke()
      })
    ctx.lineCap = "butt"; ctx.shadowBlur = 0
  }

  // ── Dimension tags ────────────────────────────────────────────────────────
  if (dims) {
    ctx.font = `700 12px 'JetBrains Mono',monospace`
    ctx.textAlign = "center"

    // Helper: draw a pill tag at (px, py)
    const drawTag = (px, py, label, value, unit) => {
      const fullText = `${label} ${value}${unit}`
      const tw = ctx.measureText(fullText).width + 18
      const th = 22

      // pill bg
      roundRectPath(ctx, px - tw / 2, py - th / 2, tw, th, 6)
      ctx.fillStyle = "rgba(8,10,14,0.92)"
      ctx.fill()
      // teal top border line
      ctx.fillStyle = `rgb(${r},${g},${b})`
      roundRectPath(ctx, px - tw / 2, py - th / 2, tw, 2.5, 2)
      ctx.fill()

      // label text (dimmer) + value (bright)
      ctx.fillStyle = `rgba(${r},${g},${b},0.6)`
      ctx.fillText(label, px - ctx.measureText(fullText).width / 2 + ctx.measureText(label).width / 2 - 2, py + 4)
      const labelW = ctx.measureText(label + " ").width
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillText(`${value}${unit}`, px + labelW / 2, py + 4)
    }

    // Simpler flat tag — just the combined string, cleaner
    const drawFlatTag = (px, py, text) => {
      const tw = ctx.measureText(text).width + 20
      const th = 24
      roundRectPath(ctx, px - tw / 2, py - th / 2, tw, th, 7)
      ctx.fillStyle = "rgba(8,10,14,0.93)"
      ctx.fill()
      ctx.strokeStyle = `rgba(${r},${g},${b},0.55)`
      ctx.lineWidth = 1.2
      ctx.stroke()
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillText(text, px, py + 4.5)
    }

    // L tag — bottom center
    drawFlatTag(
      (x1 + x2) / 2,
      y2 + 20,
      `L ${dims.length.toFixed(2)}m`
    )

    // W tag — right side middle
    drawFlatTag(
      x2 + 38,
      (y1 + y2) / 2,
      `W ${dims.width.toFixed(2)}m`
    )

    // H tag — left side middle
    drawFlatTag(
      x1 - 38,
      (y1 + y2) / 2,
      `H ${dims.height.toFixed(2)}m`
    )

    // Vol tag — top center
    drawFlatTag(
      (x1 + x2) / 2,
      y1 - 18,
      `${dims.volume_m3.toFixed(3)} m³`
    )
  }

  ctx.textAlign = "left"
  ctx.restore()
}

export default function CameraPage() {
  const router = useRouter()

  // refs
  const videoRef         = useRef(null)
  const canvasRef        = useRef(null)
  const loopsRef         = useRef(false)
  const objectsRef       = useRef([])
  const focalRef         = useRef(null)
  const streamDims       = useRef({ w: 640, h: 480 })
  const lerpedObjRef     = useRef({})
  const imuBufRef        = useRef([])
  const imuActiveRef     = useRef(false)
  const scanningRef      = useRef(false)
  const flowRef          = useRef(new FrontendOpticalFlow())
  const scanMotionRef    = useRef(0)
  const acceptedRef      = useRef(0)
  const skippedRef       = useRef(0)
  const scaleFactorRef   = useRef(null)
  const cardRectRef      = useRef({ ...DEFAULT_CARD })
  const draggingRef      = useRef(null)
  const scanStateRef     = useRef("idle")
  const fpsFrames        = useRef(0)
  const fpsTimer         = useRef(null)
  const lastDetRef       = useRef([])
  const frameTimeRef     = useRef(0)
  const calAccumRef      = useRef(0)
  const calIntervalRef   = useRef(null)
  const calActiveRef     = useRef(false)
  const isMovingRef      = useRef(false)

  // scan overlay refs
  const scanOverlayRef   = useRef(null)   // { bbox, contour, frameCount } | null
  const arResultRef      = useRef(null)   // { bbox, contour, dims } | null  ← now stores contour too

  // Track best scan result: largest bbox area + its SAM contour
  const bestScanResultRef = useRef(null)  // { bbox, contour } | null

  // state
  const [objects,         setObjects]         = useState([])
  const [fps,             setFps]             = useState(0)
  const [connected,       setConnected]       = useState(false)
  const [isMoving,        setIsMoving]        = useState(false)
  const [showSidebar,     setShowSidebar]     = useState(false)
  const [scanState,       setScanState]       = useState("idle")
  const [finalDims,       setFinalDims]       = useState(null)
  const [scanFrameCount,  setScanFrameCount]  = useState(0)
  const [scanHint,        setScanHint]        = useState("")
  const [scaleFactor,     setScaleFactor]     = useState(null)
  const [mmPerPx,         setMmPerPx]         = useState(null)
  const [camError,        setCamError]        = useState(null)
  const [cardPx,          setCardPx]          = useState({ w: 0, h: 0 })
  const [calStep,         setCalStep]         = useState(0)
  const [calProgress,     setCalProgress]     = useState(0)
  const [calStatus,       setCalStatus]       = useState("")

  useEffect(() => { scanStateRef.current = scanState }, [scanState])
  useEffect(() => { isMovingRef.current = isMoving }, [isMoving])

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

        const track = stream.getVideoTracks()[0]
        const sw    = vid.videoWidth  || 640
        const sh    = vid.videoHeight || 480
        streamDims.current = { w: sw, h: sh }
        focalRef.current   = deriveFocalLength(track, sw)

        if (!loopsRef.current) {
          loopsRef.current = true
          startDetectionLoop()
          startRenderLoop()
          fpsTimer.current = setInterval(() => {
            setFps(fpsFrames.current)
            fpsFrames.current = 0
          }, 1000)
        }
      } catch {
        setCamError("Camera permission denied or device unavailable.")
      }
    }
    start()
    return () => {
      loopsRef.current = false
      if (fpsTimer.current)       clearInterval(fpsTimer.current)
      if (calIntervalRef.current) clearInterval(calIntervalRef.current)
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
        await sleep(150)
      }
    }
    loop()
  }

  function updateLerp(scene) {
    scene.forEach((obj) => {
      const id = obj.object_id
      if (!lerpedObjRef.current[id]) {
        lerpedObjRef.current[id] = {
          ...obj,
          currentBbox: [...obj.bbox],
          targetBbox:  [...obj.bbox],
          contour:     obj.contour ?? obj.mask_contour ?? null,
          alpha: 0,
        }
      } else {
        lerpedObjRef.current[id].targetBbox  = [...obj.bbox]
        lerpedObjRef.current[id].dimensions  = obj.dimensions
        lerpedObjRef.current[id].confidence  = obj.confidence
        lerpedObjRef.current[id].contour     = obj.contour ?? obj.mask_contour ?? null
      }
    })
    const ids = new Set(scene.map((o) => o.object_id))
    Object.keys(lerpedObjRef.current).forEach((id) => {
      if (!ids.has(parseInt(id))) delete lerpedObjRef.current[id]
    })
  }

  function lerpBboxes() {
    const moving = isMovingRef.current
    Object.values(lerpedObjRef.current).forEach((obj) => {
      if (moving) {
        obj.currentBbox[0] = obj.targetBbox[0]
        obj.currentBbox[1] = obj.targetBbox[1]
        obj.currentBbox[2] = obj.targetBbox[2]
        obj.currentBbox[3] = obj.targetBbox[3]
      } else {
        const a = 0.4
        obj.currentBbox[0] += (obj.targetBbox[0] - obj.currentBbox[0]) * a
        obj.currentBbox[1] += (obj.targetBbox[1] - obj.currentBbox[1]) * a
        obj.currentBbox[2] += (obj.targetBbox[2] - obj.currentBbox[2]) * a
        obj.currentBbox[3] += (obj.targetBbox[3] - obj.currentBbox[3]) * a
      }
      obj.alpha = obj.alpha === undefined ? 1 : obj.alpha + (1 - obj.alpha) * 0.18
    })
  }

  async function captureAndDetect() {
    const video = videoRef.current
    if (!video || video.readyState < 2) return
    try {
      const { w: sw, h: sh } = streamDims.current
      const tmp = document.createElement("canvas")
      tmp.width = sw; tmp.height = sh
      tmp.getContext("2d").drawImage(video, 0, 0, sw, sh)

      const isMovingNow = isMovingRef.current
      const blob = await new Promise((res) => tmp.toBlob(res, "image/jpeg", isMovingNow ? 0.55 : 0.75))
      if (!blob) return

      const focal = focalRef.current ?? { fx: 920, fy: 920 }
      const form  = new FormData()
      form.append("image",  blob, "frame.jpg")
      form.append("detect", "1")
      form.append("fx",     focal.fx.toFixed(2))
      form.append("fy",     focal.fy.toFixed(2))
      form.append("img_w",  sw)
      form.append("img_h",  sh)
      if (isMovingNow) form.append("fast", "1")
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
      frameTimeRef.current += 1
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

    // ── RESULT STATE: draw SAM contour + dimension tags ───────────────────
    if (scanStateRef.current === "result" && arResultRef.current) {
      const { bbox, contour, dims } = arResultRef.current
      drawSAMResultOverlay(ctx, bbox, contour, dims, frameTimeRef.current)
      return
    }

    if (scanStateRef.current === "scanning" && scanOverlayRef.current) {
      drawScanOverlay(ctx, scanOverlayRef.current)
      return
    }

    const t = frameTimeRef.current

    scene.forEach((obj) => {
      const bbox = obj.currentBbox ?? obj.bbox
      if (!bbox) return
      const [x1, y1, x2, y2] = bbox
      const { r, g, b }       = colorFor(obj.label)
      const bw = x2 - x1, bh = y2 - y1
      if (bw <= 0 || bh <= 0) return

      const alpha  = obj.alpha ?? 1
      const radius = Math.min(14, bw * 0.08, bh * 0.08)

      ctx.save()
      ctx.globalAlpha = alpha

      const hasContour = Array.isArray(obj.contour ?? obj.mask_contour) && (obj.contour ?? obj.mask_contour).length > 2

      buildShapePath(ctx, obj, x1, y1, bw, bh, radius)
      ctx.save()
      ctx.clip()

      const grad = ctx.createLinearGradient(x1, y1, x1, y2)
      grad.addColorStop(0, `rgba(${r},${g},${b},0.16)`)
      grad.addColorStop(1, `rgba(${r},${g},${b},0.04)`)
      ctx.fillStyle = grad
      ctx.fillRect(x1, y1, bw, bh)

      ctx.strokeStyle = `rgba(${r},${g},${b},0.08)`
      ctx.lineWidth = 1
      const spacing = 14
      for (let off = -bh; off < bw; off += spacing) {
        ctx.beginPath()
        ctx.moveTo(x1 + off, y1)
        ctx.lineTo(x1 + off + bh, y2)
        ctx.stroke()
      }

      const scanProgress = (Math.sin(t * 0.04 + obj.object_id) + 1) / 2
      const scanY = y1 + bh * scanProgress
      const scanGrad = ctx.createLinearGradient(x1, scanY - 12, x1, scanY + 12)
      scanGrad.addColorStop(0,   `rgba(${r},${g},${b},0)`)
      scanGrad.addColorStop(0.5, `rgba(${r},${g},${b},0.35)`)
      scanGrad.addColorStop(1,   `rgba(${r},${g},${b},0)`)
      ctx.fillStyle = scanGrad
      ctx.fillRect(x1, scanY - 12, bw, 24)

      ctx.restore()

      buildShapePath(ctx, obj, x1, y1, bw, bh, radius)
      ctx.shadowColor = `rgba(${r},${g},${b},0.55)`
      ctx.shadowBlur  = 8
      ctx.strokeStyle = `rgb(${r},${g},${b})`
      ctx.lineWidth   = hasContour ? 1.75 : 2
      ctx.stroke()
      ctx.shadowBlur  = 0

      if (hasContour) {
        ctx.save()
        ctx.globalAlpha = alpha * 0.35
        ctx.strokeStyle = `rgba(${r},${g},${b},0.6)`
        ctx.lineWidth = 1
        ctx.setLineDash([4, 4])
        roundRectPath(ctx, x1, y1, bw, bh, radius)
        ctx.stroke()
        ctx.setLineDash([])
        ctx.restore()
      } else {
        const cs = Math.min(22, bw * 0.15, bh * 0.15)
        ctx.lineWidth = 3.5
        ctx.lineCap   = "round"
        ;[[x1, y1+cs, x1, y1, x1+cs, y1],[x2-cs, y1, x2, y1, x2, y1+cs],
          [x1, y2-cs, x1, y2, x1+cs, y2],[x2-cs, y2, x2, y2, x2, y2-cs]]
          .forEach(([ax,ay,bx,by,cx2,cy2]) => {
            ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.lineTo(cx2,cy2); ctx.stroke()
          })
        ctx.lineCap = "butt"
      }

      const confColor = obj.confidence > 0.66 ? "74,222,128"
                        : obj.confidence > 0.35 ? "251,191,36"
                        : "248,113,113"
      const label   = `${obj.label}  ${(obj.confidence * 100).toFixed(0)}%`
      ctx.font      = `600 14px 'JetBrains Mono',monospace`
      const tw      = ctx.measureText(label).width
      const badgeW  = tw + 34
      const badgeH  = 28
      const py      = Math.max(y1 - badgeH - 4, 0)

      roundRectPath(ctx, x1, py, badgeW, badgeH, 7)
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fill()

      ctx.beginPath()
      ctx.arc(x1 + 14, py + badgeH / 2, 4, 0, Math.PI * 2)
      ctx.fillStyle = `rgb(${confColor})`
      ctx.fill()

      ctx.fillStyle = "#0f0f14"
      ctx.fillText(label, x1 + 24, py + 19)

      if (obj.dimensions) {
        const d = obj.dimensions
        const lines = [
          `L ${d.length.toFixed(2)}m  W ${d.width.toFixed(2)}m  H ${d.height.toFixed(2)}m`,
          `Vol: ${d.volume_m3.toFixed(3)} m³`,
        ]
        ctx.font = `600 13px 'JetBrains Mono',monospace`
        const lh     = 20
        const boxW   = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 22
        const boxH   = lines.length * lh + 14
        const panelY = y2 + 8

        roundRectPath(ctx, x1, panelY, boxW, boxH, 9)
        ctx.fillStyle = "rgba(15,15,20,0.95)"
        ctx.fill()
        ctx.lineWidth   = 1
        ctx.strokeStyle = `rgba(${r},${g},${b},0.6)`
        ctx.stroke()

        ctx.save()
        roundRectPath(ctx, x1, panelY, boxW, boxH, 9)
        ctx.clip()
        ctx.fillStyle = `rgba(${r},${g},${b},0.9)`
        ctx.fillRect(x1, panelY, boxW, 3)
        ctx.restore()

        ctx.fillStyle = "#e8e6dc"
        lines.forEach((line, i) => ctx.fillText(line, x1 + 11, panelY + lh * (i + 1) + 2))
      }

      ctx.restore()
    })
  }

  function drawScanOverlay(ctx, overlay) {
    const { bbox, contour, frameCount } = overlay
    if (!bbox) return
    const [x1, y1, x2, y2] = bbox
    const bw = x2 - x1, bh = y2 - y1
    if (bw <= 0 || bh <= 0) return

    const t = frameTimeRef.current
    const { r, g, b } = { r: 99, g: 202, b: 183 }
    const radius = Math.min(14, bw * 0.08, bh * 0.08)

    ctx.save()

    const hasContour = Array.isArray(contour) && contour.length > 2
    if (hasContour) {
      ctx.beginPath()
      ctx.moveTo(contour[0][0], contour[0][1])
      for (let i = 1; i < contour.length; i++) ctx.lineTo(contour[i][0], contour[i][1])
      ctx.closePath()
    } else {
      roundRectPath(ctx, x1, y1, bw, bh, radius)
    }
    ctx.save()
    ctx.clip()

    ctx.fillStyle = `rgba(${r},${g},${b},0.10)`
    ctx.fillRect(x1, y1, bw, bh)

    const scanY = y1 + bh * ((t % 60) / 60)
    const sg = ctx.createLinearGradient(x1, scanY - 16, x1, scanY + 16)
    sg.addColorStop(0,   `rgba(${r},${g},${b},0)`)
    sg.addColorStop(0.5, `rgba(${r},${g},${b},0.5)`)
    sg.addColorStop(1,   `rgba(${r},${g},${b},0)`)
    ctx.fillStyle = sg
    ctx.fillRect(x1, scanY - 16, bw, 32)

    ctx.restore()

    if (hasContour) {
      ctx.beginPath()
      ctx.moveTo(contour[0][0], contour[0][1])
      for (let i = 1; i < contour.length; i++) ctx.lineTo(contour[i][0], contour[i][1])
      ctx.closePath()
    } else {
      roundRectPath(ctx, x1, y1, bw, bh, radius)
    }
    ctx.shadowColor = `rgba(${r},${g},${b},0.7)`
    ctx.shadowBlur  = 10
    ctx.strokeStyle = `rgb(${r},${g},${b})`
    ctx.lineWidth   = 2
    ctx.stroke()
    ctx.shadowBlur  = 0

    ctx.font = `700 13px 'JetBrains Mono',monospace`
    const badge = `● SCANNING  ${frameCount} frames`
    const bw2   = ctx.measureText(badge).width + 18
    roundRectPath(ctx, x1, y1 - 34, bw2, 26, 6)
    ctx.fillStyle = `rgba(${r},${g},${b},0.9)`
    ctx.fill()
    ctx.fillStyle = "#0f0f14"
    ctx.fillText(badge, x1 + 9, y1 - 16)

    ctx.restore()
  }

  function drawCardOverlay(ctx, sw, sh) {
    const cr = cardRectRef.current
    const x = cr.x * sw, y = cr.y * sh, w = cr.w * sw, h = cr.h * sh

    ctx.fillStyle = "rgba(0,0,0,0.68)"
    ctx.fillRect(0, 0, sw, sh)
    ctx.drawImage(videoRef.current, x, y, w, h, x, y, w, h)

    ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2
    ctx.setLineDash([8, 8])
    ctx.lineDashOffset = -(frameTimeRef.current % 16)
    ctx.strokeRect(x, y, w, h)
    ctx.setLineDash([])
    ctx.lineDashOffset = 0

    const cs = 20; ctx.lineWidth = 4; ctx.strokeStyle = "#fbbf24"
    ;[[x,y+cs,x,y,x+cs,y],[x+w-cs,y,x+w,y,x+w,y+cs],
      [x,y+h-cs,x,y+h,x+cs,y+h],[x+w-cs,y+h,x+w,y+h,x+w,y+h-cs]]
      .forEach(([ax,ay,bx,by,cx2,cy2]) => {
        ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.lineTo(cx2,cy2); ctx.stroke()
      })

    ctx.fillStyle = "#fbbf24"; ctx.beginPath(); ctx.arc(x+w, y+h, 24, 0, Math.PI*2); ctx.fill()
    ctx.fillStyle = "#0f0f14"; ctx.font = "bold 24px monospace"
    ctx.textAlign = "center"; ctx.textBaseline = "middle"
    ctx.fillText("⤡", x+w, y+h)
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"

    setCardPx({ w: Math.round(w), h: Math.round(h) })
  }

  // ── Pointer Events ──────────────────────────────────────────────────────────
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
    const tol = 0.15
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

  // ── Calibration ─────────────────────────────────────────────────────────────
  function startCalibrationCollection() {
    calAccumRef.current = 0
    calActiveRef.current = true
    setCalProgress(0)
    setCalStatus("collecting")

    calIntervalRef.current = setInterval(async () => {
      if (!calActiveRef.current) {
        clearInterval(calIntervalRef.current)
        return
      }
      const { w: sw, h: sh } = streamDims.current
      const cr      = cardRectRef.current
      const pixel_w = cr.w * sw
      const pixel_h = cr.h * sh
      const focal   = focalRef.current ?? { fx: 920, fy: 920 }

      try {
        const form = new FormData()
        form.append("pixel_w", pixel_w.toFixed(1))
        form.append("pixel_h", pixel_h.toFixed(1))
        form.append("img_w",   sw)
        form.append("img_h",   sh)
        form.append("fx",      focal.fx.toFixed(2))

        const res  = await fetch(`${BACKEND_URL}/calibrate_frame`, { method: "POST", body: form })
        const data = await res.json()

        if (data.status === "accepted") {
          calAccumRef.current = data.count
          setCalProgress(Math.min(data.count, CAL_TARGET_FRAMES))
          if (data.ready && data.final_scale) {
            clearInterval(calIntervalRef.current)
            calActiveRef.current = false
            scaleFactorRef.current = data.final_scale
            setScaleFactor(data.final_scale)
            setMmPerPx((data.final_scale * 1000).toFixed(2))
            setCalStatus("ready")
          }
        } else {
          setCalStatus(`rejected: ${data.reason}`)
        }
      } catch {
        setCalStatus("error: backend unreachable")
      }
    }, 200)
  }

  function stopCalibrationCollection() {
    clearInterval(calIntervalRef.current)
    calActiveRef.current = false
  }

  async function handleConfirmCalibrate() {
    stopCalibrationCollection()
    const { w: sw, h: sh } = streamDims.current
    const cr      = cardRectRef.current
    const pixel_w = cr.w * sw
    const pixel_h = cr.h * sh
    const focal   = focalRef.current ?? { fx: 920, fy: 920 }

    try {
      const form = new FormData()
      form.append("pixel_w", pixel_w.toFixed(1))
      form.append("pixel_h", pixel_h.toFixed(1))
      form.append("fx",      focal.fx.toFixed(2))
      form.append("img_w",   sw)

      const res  = await fetch(`${BACKEND_URL}/confirm_calibration`, { method: "POST", body: form })
      const data = await res.json()

      if (data.error) {
        setCalStatus(`error: ${data.error}`)
        return
      }
      scaleFactorRef.current = data.scale
      setScaleFactor(data.scale)
      setMmPerPx(data.mm_per_px)
      setScanState("idle")
    } catch {
      const scale = ((CARD_WIDTH_M / pixel_w) + (CARD_HEIGHT_M / pixel_h)) / 2
      scaleFactorRef.current = scale
      setScaleFactor(scale)
      setMmPerPx((scale * 1000).toFixed(2))
      setScanState("idle")
    }
  }

  function handleBeginCalibrate() {
    cardRectRef.current = { ...DEFAULT_CARD }
    setCalStep(0)
    setCalProgress(0)
    setCalStatus("")
    arResultRef.current = null
    setScanState("calibrate")
  }

  // ── Scan flow ───────────────────────────────────────────────────────────────
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
    scanOverlayRef.current  = null
    arResultRef.current     = null
    bestScanResultRef.current = null   // reset best result tracker
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
      setScanHint(
        mv < SCAN_MOTION_MIN ? "Move — camera too still" :
        mv > SCAN_MOTION_MAX ? "Slow down — too fast"   :
        "✓ Good speed"
      )
      if (mv >= SCAN_MOTION_MIN && mv <= SCAN_MOTION_MAX) {
        const { w: sw, h: sh } = streamDims.current
        const tmp = document.createElement("canvas"); tmp.width = sw; tmp.height = sh
        tmp.getContext("2d").drawImage(video, 0, 0, sw, sh)
        const blob = await new Promise((r) => tmp.toBlob(r, "image/jpeg", 0.8))

        const imuSnap = [...imuBufRef.current]; imuBufRef.current = []
        const form    = new FormData()
        form.append("image", blob, "frame.jpg")
        form.append("imu",   JSON.stringify(imuSnap))
        form.append("fx",    focalRef.current.fx.toFixed(2))
        form.append("img_w", sw)
        form.append("img_h", sh)
        form.append("scale", scaleFactorRef.current.toFixed(8))

        try {
          const res  = await fetch(`${BACKEND_URL}/scan_frame`, { method: "POST", body: form })
          const data = await res.json()

          if (data.bbox) {
            scanOverlayRef.current = {
              bbox:       data.bbox,
              contour:    data.contour ?? null,
              frameCount: data.frame_count ?? acceptedRef.current,
            }

            // Track best result: largest bbox area + its SAM contour
            const [bx1, by1, bx2, by2] = data.bbox
            const area     = (bx2 - bx1) * (by2 - by1)
            const prev     = bestScanResultRef.current
            const prevArea = prev ? (prev[2] - prev[0]) * (prev[3] - prev[1]) : 0
            if (area > prevArea) {
              bestScanResultRef.current = {
                bbox:    data.bbox,
                contour: data.contour ?? null,   // ← store the SAM contour too
              }
            }
          }

          if (data.status === "ok") {
            acceptedRef.current++
            setScanFrameCount(acceptedRef.current)
          } else {
            skippedRef.current++
          }
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

      setFinalDims(data.dimensions)

      // Use the best scan result (largest bbox + its SAM contour) as the anchor
      const best = bestScanResultRef.current
      const anchorBbox = (
        best?.bbox ||
        (lastDetRef.current.length > 0 && lastDetRef.current[0].bbox) ||
        scanOverlayRef.current?.bbox ||
        null
      )
      const anchorContour = (
        best?.contour ||
        scanOverlayRef.current?.contour ||
        (lastDetRef.current.length > 0 && lastDetRef.current[0].contour) ||
        null
      )

      if (anchorBbox) {
        arResultRef.current = {
          bbox:    anchorBbox,
          contour: anchorContour,   // ← pass SAM contour to result renderer
          dims:    data.dimensions,
        }
      }

      setScanState("result")
      setShowSidebar(true)
    } catch { setScanState("idle") }
  }

  // ── derived ─────────────────────────────────────────────────────────────────
  const calProgressPct = Math.min((calProgress / CAL_TARGET_FRAMES) * 100, 100)
  const calReady       = calStatus === "ready" || calProgress >= CAL_TARGET_FRAMES

  const cubeDims = (() => {
    if (!finalDims) return null
    const { length, width, height } = finalDims
    const maxDim = Math.max(length, width, height, 0.01)
    const scale  = 120 / maxDim
    const px     = (v) => Math.max(28, Math.round(v * scale))
    return { w: px(width), h: px(height), l: px(length) }
  })()

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Syne:wght@700;800&display=swap');

        *{box-sizing:border-box;margin:0;padding:0}
        :root{
          --bg:#0a0a0f;
          --bg2:#111118;
          --bg3:#18181f;
          --surface:#1e1e28;
          --surface2:#252532;
          --border:#2a2a38;
          --border2:#333345;
          --accent:#fbbf24;
          --accent-dim:rgba(251,191,36,0.12);
          --accent-glow:rgba(251,191,36,0.25);
          --accent2:#63c6b4;
          --accent2-dim:rgba(99,198,180,0.12);
          --text:#e8e6dc;
          --text2:#b8b6ac;
          --muted:#7070808;
          --muted2:#505060;
          --danger:#f87171;
          --danger-dim:rgba(248,113,113,0.12);
          --ok:#4ade80;
          --ok-dim:rgba(74,222,128,0.12);
          --mono:'JetBrains Mono','Courier New',monospace;
          --sans:'Syne',sans-serif;
          --radius:14px;
          --radius-sm:8px;
          --shadow:0 8px 32px rgba(0,0,0,0.5);
        }
        html,body{height:100%;overscroll-behavior:none}
        body{background:var(--bg);color:var(--text);font-family:var(--mono)}

        .root{
          display:flex;flex-direction:column;height:100dvh;
          background:var(--bg);overflow:hidden;
        }

        .cam-wrap{
          position:relative;flex-shrink:0;background:#000;
          user-select:none;touch-action:none;
        }
        .cam-wrap canvas{
          display:block;width:100%;height:auto;
          aspect-ratio:4/3;max-height:58dvh;object-fit:cover;touch-action:none;
        }
        .cam-wrap video{
          position:absolute;inset:0;opacity:0;pointer-events:none;
          width:1px;height:100%;
        }
        .cam-error{
          position:absolute;inset:0;display:flex;align-items:center;
          justify-content:center;background:#000;color:var(--danger);
          font-size:13px;text-align:center;padding:24px;line-height:1.7;
        }

        .top-bar{
          position:absolute;top:0;left:0;right:0;
          padding:12px 16px;z-index:30;
          display:flex;justify-content:space-between;align-items:center;
          background:linear-gradient(to bottom,rgba(10,10,15,0.9) 0%,transparent 100%);
        }
        .back-btn{
          background:rgba(10,10,15,0.85);border:1px solid var(--border2);
          border-radius:20px;padding:6px 14px;font-size:12px;font-weight:700;
          font-family:var(--sans);color:var(--text2);cursor:pointer;
          display:flex;align-items:center;gap:6px;letter-spacing:.05em;
          backdrop-filter:blur(8px);transition:border-color .2s,color .2s;
        }
        .back-btn:hover{border-color:var(--accent);color:var(--accent)}

        .status-bar{
          position:absolute;bottom:0;left:0;right:0;height:44px;
          padding:0 14px;z-index:20;
          background:rgba(10,10,15,0.97);border-top:1px solid var(--border);
          display:flex;align-items:center;gap:10px;font-size:11px;
        }
        .s-dot{width:7px;height:7px;border-radius:50%;background:var(--muted2);flex-shrink:0;transition:background .3s}
        .s-dot.live{background:var(--ok);box-shadow:0 0 6px var(--ok)}
        .s-dot.moving{background:var(--accent);box-shadow:0 0 6px var(--accent)}
        .tag{
          background:var(--surface);border:1px solid var(--border);
          border-radius:6px;padding:3px 8px;font-size:10px;font-weight:700;
          color:var(--text2);letter-spacing:.06em;
        }
        .tag.green{background:var(--ok-dim);border-color:rgba(74,222,128,.25);color:var(--ok)}
        .tag.red{background:var(--danger-dim);border-color:rgba(248,113,113,.25);color:var(--danger)}
        .tag.yellow{background:var(--accent-dim);border-color:rgba(251,191,36,.3);color:var(--accent)}
        .tag.teal{background:var(--accent2-dim);border-color:rgba(99,198,180,.3);color:var(--accent2)}

        .hud{
          position:absolute;bottom:52px;left:0;right:0;
          display:flex;justify-content:center;padding:0 16px;
          pointer-events:none;
        }
        .hud>*{pointer-events:all}

        .glass-panel{
          background:rgba(10,10,15,0.92);
          border:1px solid rgba(251,191,36,.3);
          border-radius:var(--radius);padding:20px;
          display:flex;flex-direction:column;align-items:center;gap:14px;
          width:100%;max-width:360px;
          backdrop-filter:blur(14px);
          box-shadow:var(--shadow),0 0 0 1px rgba(251,191,36,.06);
        }

        .cal-bottom{
          position:absolute;bottom:52px;left:0;right:0;
          display:flex;justify-content:center;padding:0 14px;
          pointer-events:all;
        }
        .cal-panel{
          background:rgba(8,8,12,0.97);
          border:1px solid rgba(251,191,36,.35);
          border-radius:var(--radius);padding:22px;
          display:flex;flex-direction:column;align-items:center;gap:14px;
          width:100%;max-width:360px;
          backdrop-filter:blur(16px);
          box-shadow:var(--shadow);
        }
        .cal-title{
          font-family:var(--sans);font-weight:800;font-size:17px;
          color:var(--accent);letter-spacing:.05em;
        }
        .cal-body{
          font-size:12px;color:var(--text2);text-align:center;
          line-height:1.7;
        }
        .cal-body strong{color:var(--text)}
        .cal-px{
          background:var(--accent-dim);border:1px solid rgba(251,191,36,.25);
          border-radius:var(--radius-sm);padding:8px 14px;
          font-size:11px;color:var(--accent);font-family:var(--mono);
          letter-spacing:.06em;
        }

        .progress-wrap{width:100%;background:var(--surface);border-radius:999px;height:6px;overflow:hidden}
        .progress-bar{height:6px;border-radius:999px;background:var(--accent);transition:width .3s ease}
        .progress-bar.ready{background:var(--ok)}

        .cal-chip{
          font-size:11px;font-weight:700;letter-spacing:.06em;
          padding:4px 10px;border-radius:6px;
        }
        .cal-chip.ok{background:var(--ok-dim);color:var(--ok)}
        .cal-chip.bad{background:var(--danger-dim);color:var(--danger)}
        .cal-chip.collecting{background:var(--accent-dim);color:var(--accent)}

        .btn-primary{
          background:var(--accent);color:#0a0a0f;
          border:none;border-radius:var(--radius-sm);
          padding:13px 22px;font-size:13px;font-weight:800;
          font-family:var(--mono);cursor:pointer;
          letter-spacing:.1em;text-transform:uppercase;
          box-shadow:0 4px 20px var(--accent-glow);
          transition:transform .12s,box-shadow .12s;white-space:nowrap;
          display:flex;align-items:center;gap:8px;justify-content:center;
        }
        .btn-primary:active{transform:scale(.95);box-shadow:0 2px 8px var(--accent-glow)}
        .btn-primary:disabled{
          background:var(--surface2);color:var(--muted2);
          box-shadow:none;cursor:not-allowed;
        }
        .btn-secondary{
          background:transparent;color:var(--text2);
          border:1px solid var(--border2);border-radius:var(--radius-sm);
          padding:12px 18px;font-size:12px;font-weight:700;
          font-family:var(--mono);cursor:pointer;
          letter-spacing:.06em;transition:border-color .2s,color .2s;
          white-space:nowrap;
        }
        .btn-secondary:hover{border-color:var(--border);color:var(--text)}
        .btn-row{display:flex;gap:10px;width:100%}
        .btn-row .btn-primary,.btn-row .btn-secondary{flex:1}

        .scan-hint{
          font-size:13px;font-weight:800;padding:9px 16px;
          border-radius:var(--radius-sm);letter-spacing:.04em;
          text-transform:uppercase;
        }
        .scan-hint.ok{color:var(--ok);background:var(--ok-dim)}
        .scan-hint.warn{color:var(--danger);background:var(--danger-dim)}

        .frame-counter{
          display:flex;justify-content:space-between;align-items:center;
          width:100%;font-size:11px;color:var(--muted2);
        }
        .frame-counter strong{color:var(--text);font-size:16px;font-family:var(--sans)}

        .computing-overlay{
          position:absolute;inset:0;
          background:rgba(10,10,15,0.93);
          display:flex;flex-direction:column;
          align-items:center;justify-content:center;
          gap:16px;
        }
        .computing-spinner{
          width:40px;height:40px;
          border:3px solid var(--surface2);
          border-top-color:var(--accent);
          border-radius:50%;
          animation:spin .7s linear infinite;
        }
        @keyframes spin{to{transform:rotate(360deg)}}
        .computing-text{
          color:var(--accent);font-weight:800;font-size:14px;
          letter-spacing:.2em;font-family:var(--sans);
        }

        .ar-result-badge{
          position:absolute;bottom:52px;left:0;right:0;
          display:flex;justify-content:center;padding:0 14px;
          pointer-events:all;
        }
        .ar-badge-inner{
          background:rgba(8,8,12,0.96);
          border:1px solid rgba(99,198,180,.5);
          border-radius:var(--radius);padding:16px 20px;
          display:flex;flex-direction:column;align-items:center;gap:10px;
          width:100%;max-width:360px;
          backdrop-filter:blur(16px);
          box-shadow:var(--shadow),0 0 20px rgba(99,198,180,.1);
        }
        .ar-badge-title{
          font-family:var(--sans);font-weight:800;font-size:15px;
          color:var(--accent2);letter-spacing:.08em;
        }
        .ar-dims-row{
          display:flex;gap:8px;width:100%;
        }
        .ar-dim-cell{
          flex:1;background:var(--surface);border:1px solid var(--border);
          border-radius:var(--radius-sm);padding:8px 6px;text-align:center;
        }
        .ar-dim-lbl{font-size:9px;color:var(--muted2);letter-spacing:.1em;margin-bottom:3px;text-transform:uppercase}
        .ar-dim-val{font-size:16px;font-weight:800;font-family:var(--sans);color:var(--accent2)}
        .ar-dim-unit{font-size:9px;color:var(--muted2);margin-left:1px}

        .data-sheet{
          flex:1;overflow-y:auto;
          background:var(--bg2);
          border-top:1px solid var(--border);
        }
        .sheet-inner{padding:14px;display:flex;flex-direction:column;gap:10px}

        .sheet-header{
          display:flex;justify-content:space-between;align-items:center;
          padding-bottom:12px;border-bottom:1px solid var(--border);
        }
        .sheet-title{
          font-family:var(--sans);font-weight:800;font-size:16px;
          color:var(--text);letter-spacing:.03em;
        }

        .obj-card{
          background:var(--bg3);border:1px solid var(--border);
          border-radius:var(--radius);overflow:hidden;
        }
        .obj-card-top{
          padding:12px 14px 10px;
          display:flex;justify-content:space-between;align-items:center;
          border-bottom:1px solid var(--border);
        }
        .obj-label{
          font-family:var(--sans);font-weight:800;font-size:14px;
          text-transform:uppercase;letter-spacing:.06em;
        }
        .conf-badge{
          padding:3px 8px;border-radius:6px;
          font-size:11px;font-weight:800;letter-spacing:.06em;
        }
        .obj-body{padding:12px 14px}
        .dims-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px}
        .dim-cell{
          background:var(--surface);border:1px solid var(--border);
          border-radius:var(--radius-sm);padding:9px 8px;text-align:center;
        }
        .dim-lbl{font-size:9px;color:var(--muted2);letter-spacing:.1em;margin-bottom:4px;text-transform:uppercase}
        .dim-val{font-size:15px;font-weight:800;font-family:var(--sans)}
        .dim-unit{font-size:9px;color:var(--muted2);margin-left:1px}

        .vol-bar{
          background:var(--surface);border:1px solid var(--border);
          border-radius:var(--radius-sm);padding:10px 12px;
          display:flex;justify-content:space-between;align-items:center;
        }
        .vol-label{font-size:9px;color:var(--muted2);letter-spacing:.1em;text-transform:uppercase}
        .vol-val{font-size:18px;font-weight:800;font-family:var(--sans)}

        .no-dims{color:var(--muted2);font-size:12px;padding:6px 0}

        .result-card{
          background:var(--bg3);border:2px solid var(--accent2);
          border-radius:var(--radius);overflow:hidden;
        }
        .result-card-top{
          padding:12px 14px 10px;
          background:var(--accent2-dim);
          display:flex;justify-content:space-between;align-items:center;
          border-bottom:1px solid rgba(99,198,180,.2);
        }
        .result-title{
          font-family:var(--sans);font-weight:800;font-size:14px;
          color:var(--accent2);text-transform:uppercase;letter-spacing:.08em;
        }

        .sfm-stage{
          position:relative;height:200px;
          display:flex;align-items:center;justify-content:center;
          margin-bottom:14px;border-radius:var(--radius);
          background:
            radial-gradient(circle at 50% 55%, rgba(99,198,180,.14), transparent 65%),
            repeating-linear-gradient(0deg, rgba(255,255,255,.02) 0 1px, transparent 1px 22px),
            repeating-linear-gradient(90deg, rgba(255,255,255,.02) 0 1px, transparent 1px 22px),
            var(--bg2);
          border:1px solid var(--border);
          overflow:hidden;
        }
        .sfm-scene{ perspective:900px; }
        .sfm-cube{
          position:relative;
          width:var(--w);height:var(--h);
          transform-style:preserve-3d;
          animation:sfmSpin 18s linear infinite;
          filter:drop-shadow(0 0 14px rgba(99,198,180,.25));
        }
        @keyframes sfmSpin{
          from{ transform:rotateX(-20deg) rotateY(0deg); }
          to  { transform:rotateX(-20deg) rotateY(360deg); }
        }
        .sfm-face{
          position:absolute;
          border:1.5px solid var(--accent2);
          background:
            repeating-linear-gradient(0deg,  rgba(99,198,180,.10) 0 1px, transparent 1px 12px),
            repeating-linear-gradient(90deg, rgba(99,198,180,.10) 0 1px, transparent 1px 12px),
            linear-gradient(135deg, rgba(99,198,180,.18), rgba(99,198,180,.03));
          box-shadow:inset 0 0 18px rgba(99,198,180,.35);
        }
        .sfm-face.front, .sfm-face.back { width:var(--w); height:var(--h); }
        .sfm-face.right, .sfm-face.left { width:var(--l); height:var(--h); }
        .sfm-face.top,   .sfm-face.bottom{ width:var(--w); height:var(--l); }
        .sfm-face.front { transform:translateZ(calc(var(--l) / 2)); }
        .sfm-face.back  { transform:translateZ(calc(var(--l) / -2)) rotateY(180deg); }
        .sfm-face.right { transform:translateX(calc(var(--w) / 2)) rotateY(90deg); }
        .sfm-face.left  { transform:translateX(calc(var(--w) / -2)) rotateY(-90deg); }
        .sfm-face.top   { transform:translateY(calc(var(--h) / -2)) rotateX(90deg); }
        .sfm-face.bottom{ transform:translateY(calc(var(--h) / 2)) rotateX(-90deg); }

        .sfm-tag{
          position:absolute;top:10px;left:10px;
          font-size:10px;font-weight:800;letter-spacing:.14em;
          color:var(--accent2);background:rgba(10,10,15,.85);
          border:1px solid rgba(99,198,180,.35);border-radius:6px;
          padding:3px 8px;font-family:var(--mono);
        }

        .warn-banner{
          background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.2);
          border-radius:var(--radius-sm);padding:12px 14px;
          display:flex;align-items:flex-start;gap:10px;
          font-size:12px;color:var(--text2);line-height:1.6;
        }
        .warn-icon{font-size:16px;flex-shrink:0;margin-top:1px}
      `}</style>

      <div className="root">

        {/* ── Camera View ──────────────────────────────────────────────────── */}
        <div
          className="cam-wrap"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        >
          <video ref={videoRef} autoPlay playsInline muted />
          <canvas ref={canvasRef} />
          {camError && <div className="cam-error">{camError}</div>}

          {/* Top bar */}
          <div className="top-bar">
            <div className="back-btn" onClick={() => router.push('/')}>← Home</div>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              {scaleFactor && (
                <span className="tag teal">⚖ {mmPerPx} mm/px</span>
              )}
            </div>
          </div>

          {/* ── Calibration overlay ─────────────────────────────────────── */}
          {scanState === "calibrate" && (
            <div className="cal-bottom">
              {calStep === 0 && (
                <div className="cal-panel">
                  <div className="cal-title">📐 Card Calibration</div>
                  <p className="cal-body">
                    Place a <strong>standard credit card</strong> flat on top of the box.
                    We'll use it as a metric reference. Keep it parallel to the camera.
                  </p>
                  <div className="btn-row">
                    <button className="btn-secondary" onClick={() => setScanState("idle")}>Cancel</button>
                    <button className="btn-primary" onClick={() => setCalStep(1)}>Next →</button>
                  </div>
                </div>
              )}

              {calStep === 1 && (
                <div className="cal-panel">
                  <div className="cal-title">🎯 Align to Card</div>
                  <p className="cal-body">
                    <strong>Drag</strong> to move · <strong>Corner (⤡)</strong> to resize.<br/>
                    Match the yellow box precisely to the card edges.
                  </p>
                  <div className="cal-px">{cardPx.w} × {cardPx.h} px</div>
                  <div className="btn-row">
                    <button className="btn-secondary" onClick={() => setCalStep(0)}>← Back</button>
                    <button className="btn-primary" onClick={() => { setCalStep(2); startCalibrationCollection() }}>
                      Lock In →
                    </button>
                  </div>
                </div>
              )}

              {calStep === 2 && (
                <div className="cal-panel">
                  <div className="cal-title">
                    {calReady ? "✅ Calibrated" : "⏳ Collecting Frames"}
                  </div>
                  <p className="cal-body">
                    {calReady
                      ? `${CAL_TARGET_FRAMES} stable frames captured. Calibration is solid.`
                      : "Hold the card steady. Gathering stable measurements…"}
                  </p>

                  <div style={{ width: "100%" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--muted2)", marginBottom: "6px" }}>
                      <span>FRAMES CAPTURED</span>
                      <span style={{ color: calReady ? "var(--ok)" : "var(--accent)" }}>
                        {calProgress}/{CAL_TARGET_FRAMES}
                      </span>
                    </div>
                    <div className="progress-wrap">
                      <div className={`progress-bar${calReady ? " ready" : ""}`} style={{ width: `${calProgressPct}%` }} />
                    </div>
                  </div>

                  {calStatus && (
                    <div className={`cal-chip ${calReady ? "ok" : calStatus.startsWith("rejected") ? "bad" : "collecting"}`}>
                      {calReady ? "✓ STABLE" : calStatus.startsWith("rejected") ? `⚠ ${calStatus}` : "● COLLECTING"}
                    </div>
                  )}

                  <div className="cal-px">{cardPx.w} × {cardPx.h} px</div>

                  <div className="btn-row">
                    <button className="btn-secondary" onClick={() => { stopCalibrationCollection(); setCalStep(1) }}>← Readjust</button>
                    <button
                      className="btn-primary"
                      disabled={calProgress < 3}
                      onClick={handleConfirmCalibrate}
                    >
                      {calReady ? "✓ Confirm" : `Need ${Math.max(0, 3 - calProgress)} more`}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── Idle HUD ────────────────────────────────────────────────── */}
          {scanState === "idle" && (
            <div className="hud">
              <div style={{ display: "flex", gap: "10px" }}>
                {!scaleFactor ? (
                  <button className="btn-primary" onClick={handleBeginCalibrate}>
                    📐 Calibrate Card
                  </button>
                ) : (
                  <>
                    <button className="btn-secondary" style={{ background: "rgba(10,10,15,0.88)", backdropFilter: "blur(8px)" }} onClick={handleBeginCalibrate}>
                      Re-cal
                    </button>
                    {objects.length > 0 && (
                      <button className="btn-primary" onClick={handleStartScan}>
                        ▶ Start Scan
                      </button>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {/* ── Scanning HUD ────────────────────────────────────────────── */}
          {scanState === "scanning" && (
            <div className="hud">
              <div className="glass-panel">
                <div className={`scan-hint ${scanHint.includes("✓") ? "ok" : "warn"}`}>{scanHint}</div>
                <div className="frame-counter">
                  <span>FRAMES ACCEPTED</span>
                  <strong>{scanFrameCount}</strong>
                </div>
                <div className="progress-wrap" style={{ width: "100%" }}>
                  <div className="progress-bar" style={{ width: `${Math.min((scanFrameCount / 15) * 100, 100)}%` }} />
                </div>
                <button
                  className="btn-primary"
                  style={{ width: "100%" }}
                  onClick={handleDoneScan}
                  disabled={scanFrameCount < 3}
                >
                  {scanFrameCount >= 3 ? "✓ Done — Compute" : `Need ${3 - scanFrameCount} more frames`}
                </button>
              </div>
            </div>
          )}

          {/* ── Computing overlay ────────────────────────────────────────── */}
          {scanState === "computing" && (
            <div className="computing-overlay">
              <div className="computing-spinner" />
              <div className="computing-text">COMPUTING DIMENSIONS</div>
            </div>
          )}

          {/* ── Result badge — teal to match SAM contour ─────────────────── */}
          {scanState === "result" && finalDims && (
            <div className="ar-result-badge">
              <div className="ar-badge-inner">
                <div className="ar-badge-title">✦ SCAN COMPLETE</div>
                <div className="ar-dims-row">
                  {[["L", finalDims.length], ["W", finalDims.width], ["H", finalDims.height]].map(([lbl, val]) => (
                    <div className="ar-dim-cell" key={lbl}>
                      <div className="ar-dim-lbl">{lbl}</div>
                      <div className="ar-dim-val">{val.toFixed(2)}<span className="ar-dim-unit">m</span></div>
                    </div>
                  ))}
                </div>
                <div className="btn-row">
                  <button className="btn-secondary" onClick={() => {
                    arResultRef.current = null
                    scanOverlayRef.current = null
                    bestScanResultRef.current = null
                    setScanState("idle")
                  }}>Scan Again</button>
                  <button className="btn-primary" onClick={handleBeginCalibrate}>Re-calibrate</button>
                </div>
              </div>
            </div>
          )}

          {/* ── Status bar ───────────────────────────────────────────────── */}
          <div className="status-bar">
            <div className={`s-dot${connected ? " live" : ""}`} />
            <span className={`tag ${connected ? "green" : "red"}`}>{connected ? "LIVE" : "OFFLINE"}</span>
            <span className="tag">{fps} fps</span>
            <span className="tag">{objects.length} obj</span>
            {isMoving && <span className="tag yellow">MOVING</span>}
            <button
              className="btn-secondary"
              style={{
                marginLeft: "auto", padding: "3px 10px",
                fontSize: "10px", background: showSidebar ? "var(--accent2-dim)" : "transparent",
                borderColor: showSidebar ? "rgba(99,198,180,.4)" : "var(--border)",
                color: showSidebar ? "var(--accent2)" : "var(--muted2)",
              }}
              onClick={() => setShowSidebar(!showSidebar)}
            >
              DATA
            </button>
          </div>
        </div>

        {/* ── Data Sheet ───────────────────────────────────────────────────── */}
        {(showSidebar || scanState === "result") && (
          <div className="data-sheet">
            <div className="sheet-inner">
              <div className="sheet-header">
                <span className="sheet-title">Detected Objects</span>
                {scaleFactor && (
                  <span className="tag teal" style={{ fontSize: "10px" }}>
                    ⚖ {mmPerPx} mm/px
                  </span>
                )}
              </div>

              {!scaleFactor && scanState !== "calibrate" && (
                <div className="warn-banner">
                  <span className="warn-icon">⚠️</span>
                  <span>Calibrate using a standard credit card first to enable dimension measurements.</span>
                </div>
              )}

              {/* Final scan result */}
              {scanState === "result" && finalDims && (
                <div className="result-card">
                  <div className="result-card-top">
                    <span className="result-title">✦ Final Scan Result</span>
                    <span className="tag teal">SCAN</span>
                  </div>
                  <div className="obj-body">
                    {cubeDims && (
                      <div className="sfm-stage">
                        <div className="sfm-tag">SFM · 3D</div>
                        <div className="sfm-scene">
                          <div className="sfm-cube" style={{
                            "--w": `${cubeDims.w}px`,
                            "--h": `${cubeDims.h}px`,
                            "--l": `${cubeDims.l}px`,
                          }}>
                            {["front","back","right","left","top","bottom"].map(f => (
                              <div key={f} className={`sfm-face ${f}`} />
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                    <div className="dims-grid">
                      {[["LENGTH", finalDims.length], ["WIDTH", finalDims.width], ["HEIGHT", finalDims.height]].map(([l, v]) => (
                        <div className="dim-cell" key={l}>
                          <div className="dim-lbl">{l}</div>
                          <div className="dim-val" style={{ color: "var(--accent2)" }}>
                            {v.toFixed(2)}<span className="dim-unit">m</span>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="vol-bar" style={{ marginBottom: "14px" }}>
                      <span className="vol-label">Volume</span>
                      <span className="vol-val" style={{ color: "var(--accent2)" }}>
                        {finalDims.volume_m3.toFixed(3)}<span style={{ fontSize: "11px", color: "var(--muted2)", marginLeft: "4px" }}>m³</span>
                      </span>
                    </div>
                    <div className="btn-row">
                      <button className="btn-secondary" onClick={() => {
                        arResultRef.current = null
                        scanOverlayRef.current = null
                        bestScanResultRef.current = null
                        setScanState("idle")
                      }}>Scan Again</button>
                      <button className="btn-primary" onClick={handleBeginCalibrate}>Re-calibrate</button>
                    </div>
                  </div>
                </div>
              )}

              {/* Live object cards */}
              {objects.map((obj) => {
                const { r, g, b } = colorFor(obj.label)
                const d           = obj.dimensions
                return (
                  <div className="obj-card" key={obj.object_id} style={{ borderTop: `3px solid rgb(${r},${g},${b})` }}>
                    <div className="obj-card-top">
                      <span className="obj-label" style={{ color: `rgb(${r},${g},${b})` }}>{obj.label}</span>
                      <span
                        className="conf-badge"
                        style={{
                          background: `rgba(${r},${g},${b},0.12)`,
                          color: `rgb(${r},${g},${b})`,
                          border: `1px solid rgba(${r},${g},${b},0.25)`,
                        }}
                      >
                        {(obj.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="obj-body">
                      {d ? (
                        <>
                          <div className="dims-grid">
                            {[["L", d.length], ["W", d.width], ["H", d.height]].map(([l, v]) => (
                              <div className="dim-cell" key={l}>
                                <div className="dim-lbl">{l}</div>
                                <div className="dim-val" style={{ color: `rgb(${r},${g},${b})` }}>
                                  {v.toFixed(2)}<span className="dim-unit">m</span>
                                </div>
                              </div>
                            ))}
                          </div>
                          <div className="vol-bar">
                            <span className="vol-label">Vol</span>
                            <span className="vol-val" style={{ color: `rgb(${r},${g},${b})` }}>
                              {d.volume_m3.toFixed(3)}<span style={{ fontSize: "11px", color: "var(--muted2)", marginLeft: "4px" }}>m³</span>
                            </span>
                          </div>
                        </>
                      ) : (
                        <div className="no-dims">Awaiting calibration + depth map…</div>
                      )}
                    </div>
                  </div>
                )
              })}

              {objects.length === 0 && scanState !== "result" && (
                <div style={{ color: "var(--muted2)", fontSize: "12px", textAlign: "center", padding: "24px 0" }}>
                  No objects detected. Point camera at a box.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </>
  )
}