"use client"

import { useEffect, useRef, useState } from "react"

export default function CameraPage() {

  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  const [objects, setObjects] = useState([])

  const detectingRef = useRef(false)
  const objectsRef = useRef([])

  useEffect(() => {

    async function startCamera() {

      try {

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false
        })

        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }

        startDetectionLoop()
        startRenderLoop()

      } catch (err) {
        console.error("Camera error:", err)
      }
    }

    startCamera()

  }, [])


  function startDetectionLoop() {

    setInterval(() => {

      if (!detectingRef.current) {
        captureFrame()
      }

    }, 400)

  }


  function startRenderLoop() {

    function render() {

      drawBoxes(objectsRef.current)

      requestAnimationFrame(render)

    }

    requestAnimationFrame(render)

  }


  async function captureFrame() {

    const video = videoRef.current
    const canvas = canvasRef.current

    if (!video || !canvas) return

    detectingRef.current = true

    const ctx = canvas.getContext("2d")

    canvas.width = 640
    canvas.height = 480

    ctx.drawImage(video, 0, 0, 640, 480)

    canvas.toBlob(async (blob) => {

      const formData = new FormData()
      formData.append("image", blob, "frame.jpg")

      try {

        const response = await fetch("http://127.0.0.1:5000/detect", {
          method: "POST",
          body: formData
        })

        const data = await response.json()

        setObjects(data.scene || [])
        objectsRef.current = data.scene || []

      } catch (err) {

        console.error("Detection error:", err)

      }

      detectingRef.current = false

    }, "image/jpeg")

  }


  function drawBoxes(detections) {

    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    ctx.lineWidth = 3
    ctx.strokeStyle = "lime"
    ctx.font = "16px Arial"
    ctx.fillStyle = "lime"

    detections.forEach(obj => {

      const [x1, y1, x2, y2] = obj.bbox

      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

      ctx.fillText(
        `${obj.label} ${(obj.confidence * 100).toFixed(1)}%`,
        x1,
        y1 - 5
      )

    })

  }


  return (
    <div style={{ padding: "20px" }}>

      <h1>ParcelVision Live Detection</h1>

      <div style={{ position: "relative", width: "640px" }}>

        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ width: "640px" }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            width: "640px",
            height: "480px"
          }}
        />

      </div>

      <h3>Detected Objects</h3>

      {objects.map((obj, i) => (
        <div key={i}>
          {obj.label} — {(obj.confidence * 100).toFixed(1)}%
        </div>
      ))}

    </div>
  )
}