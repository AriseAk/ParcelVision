"use client"

import Link from "next/link";

export default function Home() {
  return (
    <div style={{
      display: "flex",
      minHeight: "100vh",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      background: "#2b2d3e",
      color: "#d1d0c5",
      fontFamily: "'Roboto Mono', 'Courier New', monospace",
      padding: "24px",
    }}>
      <main style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        textAlign: "center",
        maxWidth: "560px",
        gap: "40px",
      }}>

        {/* Title */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <h1 style={{
            fontSize: "clamp(48px, 10vw, 80px)",
            fontWeight: 800,
            letterSpacing: "-0.02em",
            margin: 0,
            color: "#d1d0c5",
          }}>
            Parcel<span style={{ color: "#e2b714" }}>Vision</span>
          </h1>

          <p style={{
            fontSize: "14px",
            color: "#646579",
            letterSpacing: "0.15em",
            textTransform: "uppercase",
            margin: 0,
          }}>
            ai-powered ar logistics scanner
          </p>

          <p style={{
            fontSize: "15px",
            color: "#878a9c",
            lineHeight: 1.7,
            margin: 0,
            maxWidth: "420px",
          }}>
            point your camera at any box. get real-world dimensions and volume instantly.
          </p>
        </div>

        {/* CTA */}
        <Link href="/camera" style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "10px",
          padding: "14px 32px",
          background: "#e2b714",
          color: "#2b2d3e",
          borderRadius: "6px",
          fontWeight: 700,
          fontSize: "15px",
          letterSpacing: "0.05em",
          textDecoration: "none",
          transition: "opacity 0.15s ease",
        }}
          onMouseEnter={e => e.currentTarget.style.opacity = "0.85"}
          onMouseLeave={e => e.currentTarget.style.opacity = "1"}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
            fill="none" stroke="currentColor" strokeWidth="2.5"
            strokeLinecap="round" strokeLinejoin="round">
            <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
            <circle cx="12" cy="13" r="3" />
          </svg>
          open scanner
        </Link>

        {/* Tech stack */}
        <div style={{
          display: "flex",
          gap: "16px",
          fontSize: "11px",
          color: "#646579",
          letterSpacing: "0.15em",
          textTransform: "uppercase",
        }}>
          <span>YOLOv8</span>
          <span style={{ color: "#404258" }}>·</span>
          <span>SAM</span>
          <span style={{ color: "#404258" }}>·</span>
          <span>DepthAnything</span>
        </div>

      </main>
    </div>
  );
}
