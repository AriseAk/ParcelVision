"use client"
import { useRouter } from "next/navigation"

export default function Home() {
  const router = useRouter()

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Syne:wght@700;800&display=swap');

        *{box-sizing:border-box;margin:0;padding:0}

        :root{
          --bg:#08080d;--bg2:#0f0f16;--bg3:#151520;
          --surface:#1a1a25;--border:#252535;
          --accent:#fbbf24;--accent2:#5eead4;
          --text:#ece9df;--muted:#606075;
          --mono:'JetBrains Mono',monospace;--sans:'Syne',sans-serif;
        }

        html,body{
          height:100%;
          background:var(--bg);
        }

        /* FIX 1: allow scrolling */
        .page{
          min-height:100dvh;
          background:var(--bg);
          display:flex;
          flex-direction:column;
          align-items:center;

          /* KEY CHANGE */
          justify-content:flex-start;

          padding:20px 14px 40px;

          /* KEY CHANGE */
          overflow-y:auto;

          position:relative;
        }

        .page::before{
          content:'';
          position:absolute;
          width:500px;height:500px;
          border-radius:50%;
          background:radial-gradient(circle,rgba(251,191,36,.04) 0%,transparent 70%);
          pointer-events:none;
        }

        .page::after{
          content:'';
          position:absolute;
          inset:0;
          background-image:
            linear-gradient(rgba(251,191,36,.03) 1px,transparent 1px),
            linear-gradient(90deg,rgba(251,191,36,.03) 1px,transparent 1px);
          background-size:52px 52px;
          pointer-events:none;
        }

        /* FIX 2: responsive card */
        .card{
          position:relative;
          z-index:1;
          background:var(--bg2);
          border:1px solid var(--border);
          border-radius:20px;

          /* MOBILE SAFE */
          padding:26px 18px;

          max-width:420px;
          width:100%;

          display:flex;
          flex-direction:column;
          align-items:center;

          margin-top:10px;

          box-shadow:0 24px 60px rgba(0,0,0,.6);
        }

        .icon-wrap{
          width:60px;height:60px;
          font-size:28px;
          margin-bottom:20px;
          display:flex;align-items:center;justify-content:center;
          border-radius:16px;
          background:rgba(251,191,36,.06);
          border:1px solid rgba(251,191,36,.18);
        }

        .headline{
          font-family:var(--sans);
          font-weight:800;
          font-size:28px;
          color:var(--text);
          text-align:center;
          margin-bottom:10px;
        }

        .headline span{color:var(--accent)}

        .tagline{
          font-size:12px;
          text-align:center;
          margin-bottom:18px;
          font-family:var(--mono);
          color:var(--muted);
          line-height:1.5;
        }

        /* FIX 3: compact architecture */
        .arch{
          width:100%;
          background:var(--bg3);
          border:1px solid var(--border);
          border-radius:10px;
          padding:12px;
          margin-bottom:18px;
        }

        .arch-row{
          display:flex;
          gap:6px;
          font-size:10px;
          color:var(--muted);
        }

        .arch-label{
          font-weight:700;
          color:var(--text);
        }

        .arch-sep{
          height:1px;
          background:var(--border);
          margin:4px 0;
        }

        /* FIX 4: tighter pills */
        .features{
          display:flex;
          gap:6px;
          flex-wrap:wrap;
          justify-content:center;
          margin-bottom:20px;
        }

        .feat{
          font-size:8px;
          padding:3px 8px;
          border-radius:999px;
          border:1px solid var(--border);
          color:var(--muted);
          font-family:var(--mono);
        }

        .feat.hi{color:var(--accent2)}
        .feat.acc{color:var(--accent)}

        /* FIX 5: big mobile-friendly button */
        .cta{
          width:100%;
          padding:16px;
          font-size:14px;
          border-radius:12px;
          border:none;
          background:var(--accent);
          color:#000;
          font-weight:800;
          cursor:pointer;
        }

        .footer{
          margin-top:14px;
          font-size:10px;
          color:var(--muted);
          text-align:center;
        }

        /* 🔥 MOST IMPORTANT: camera compatibility */
        @media (max-height:700px){
          .headline{font-size:24px}
          .icon-wrap{width:50px;height:50px}
          .card{padding:20px 14px}
        }

      `}</style>

      <div className="page">
        <div className="card">
          <div className="icon-wrap">📦</div>

          <h1 className="headline">Parcel<span>Vision</span></h1>

          <p className="tagline">
            Metric volumetric measurement from a single camera.
          </p>

          <div className="arch">
            <div className="arch-row">
              <span>🤖 <span className="arch-label">ML</span> — YOLO + SAM</span>
            </div>
            <div className="arch-sep" />
            <div className="arch-row">
              <span>📐 <span className="arch-label">PnP</span> — pose solve</span>
            </div>
            <div className="arch-sep" />
            <div className="arch-row">
              <span>📡 <span className="arch-label">Geometry</span> — 3D</span>
            </div>
          </div>

          <div className="features">
            <span className="feat hi">solvePnP</span>
            <span className="feat acc">No depth</span>
            <span className="feat hi">Ray-plane</span>
            <span className="feat">YOLOv8</span>
          </div>

          <button
            className="cta"
            onClick={() => router.push('/camera')}
          >
            Launch Scanner →
          </button>

          <p className="footer">Point · Calibrate · Measure</p>
        </div>
      </div>
    </>
  )
}