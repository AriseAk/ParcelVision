// frontend/app/page.js
"use client"
import { useRouter } from "next/navigation"

export default function Home() {
  const router = useRouter()

  return (
    <div style={{ 
      minHeight: "100vh", 
      background: "#0f0f14", 
      color: "#e8e6dc", 
      fontFamily: "'Syne', sans-serif",
      display: "flex", 
      flexDirection: "column", 
      alignItems: "center", 
      justifyContent: "center",
      padding: "24px",
      textAlign: "center"
    }}>
      <div style={{ 
        width: "64px", 
        height: "64px", 
        background: "rgba(251,191,36,0.1)", 
        border: "2px solid #fbbf24",
        borderRadius: "16px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "32px",
        marginBottom: "24px"
      }}>
        📦
      </div>
      
      <h1 style={{ fontSize: "36px", fontWeight: 800, margin: "0 0 16px 0" }}>
        Parcel<span style={{ color: "#fbbf24" }}>Vision</span>
      </h1>
      
      <p style={{ 
        fontSize: "16px", 
        color: "#9090a0", 
        maxWidth: "400px", 
        lineHeight: 1.6,
        marginBottom: "48px" 
      }}>
        Instant, AI-powered volumetric measurement and 3D bounding box generation for modern logistics.
      </p>

      <button 
        onClick={() => router.push('/camera')}
        style={{
          background: "#fbbf24",
          color: "#0f0f14",
          border: "none",
          borderRadius: "12px",
          padding: "16px 32px",
          fontSize: "16px",
          fontWeight: 800,
          fontFamily: "'JetBrains Mono', monospace",
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          cursor: "pointer",
          boxShadow: "0 8px 32px rgba(251,191,36,0.25)",
          transition: "transform 0.1s",
        }}
        onPointerDown={(e) => e.currentTarget.style.transform = "scale(0.96)"}
        onPointerUp={(e) => e.currentTarget.style.transform = "scale(1)"}
      >
        Launch Scanner →
      </button>
    </div>
  )
}