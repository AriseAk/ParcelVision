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
          --bg:#0a0a0f;--bg2:#111118;--bg3:#18181f;
          --surface:#1e1e28;--border:#2a2a38;
          --accent:#fbbf24;--accent2:#63c6b4;
          --text:#e8e6dc;--muted:#7070808;
          --mono:'JetBrains Mono',monospace;--sans:'Syne',sans-serif;
        }
        html,body{height:100%;background:var(--bg)}
        .page{
          min-height:100dvh;background:var(--bg);
          display:flex;flex-direction:column;
          align-items:center;justify-content:center;
          padding:24px;overflow:hidden;position:relative;
        }

        /* Subtle radial glow behind content */
        .page::before{
          content:'';position:absolute;
          width:600px;height:600px;border-radius:50%;
          background:radial-gradient(circle,rgba(251,191,36,.04) 0%,transparent 70%);
          pointer-events:none;
        }

        /* Grid texture */
        .page::after{
          content:'';position:absolute;inset:0;
          background-image:
            linear-gradient(rgba(251,191,36,.04) 1px,transparent 1px),
            linear-gradient(90deg,rgba(251,191,36,.04) 1px,transparent 1px);
          background-size:48px 48px;
          pointer-events:none;mask-image:radial-gradient(ellipse 80% 80% at 50% 50%,black 30%,transparent 100%);
        }

        .card{
          position:relative;z-index:1;
          background:var(--bg2);
          border:1px solid var(--border);
          border-radius:24px;
          padding:40px 32px 36px;
          max-width:400px;width:100%;
          display:flex;flex-direction:column;
          align-items:center;gap:0;
          box-shadow:0 24px 80px rgba(0,0,0,0.6),0 0 0 1px rgba(251,191,36,.06);
        }

        /* Top accent line */
        .card::before{
          content:'';position:absolute;top:0;left:20%;right:20%;height:2px;
          background:linear-gradient(90deg,transparent,rgba(251,191,36,.5),transparent);
          border-radius:999px;
        }

        .icon-wrap{
          width:72px;height:72px;
          background:rgba(251,191,36,.07);
          border:1.5px solid rgba(251,191,36,.2);
          border-radius:20px;
          display:flex;align-items:center;justify-content:center;
          font-size:36px;margin-bottom:28px;
          box-shadow:0 0 40px rgba(251,191,36,.08);
        }

        .headline{
          font-family:var(--sans);font-weight:800;
          font-size:clamp(28px,8vw,38px);
          color:var(--text);letter-spacing:-.01em;
          text-align:center;line-height:1.1;
          margin-bottom:14px;
        }
        .headline span{color:var(--accent)}

        .tagline{
          font-size:13px;color:#8888a0;
          text-align:center;line-height:1.7;
          max-width:300px;margin-bottom:36px;
          font-family:var(--mono);
        }

        /* Feature pills row */
        .features{
          display:flex;gap:8px;flex-wrap:wrap;
          justify-content:center;margin-bottom:32px;
        }
        .feat{
          background:var(--surface);border:1px solid var(--border);
          border-radius:999px;padding:5px 12px;
          font-size:10px;font-weight:700;font-family:var(--mono);
          color:#9090a8;letter-spacing:.08em;text-transform:uppercase;
        }
        .feat.hi{
          background:rgba(99,198,180,.08);border-color:rgba(99,198,180,.2);
          color:var(--accent2);
        }

        .cta{
          width:100%;background:var(--accent);color:#0a0a0f;
          border:none;border-radius:14px;
          padding:17px 24px;
          font-size:14px;font-weight:800;
          font-family:var(--mono);cursor:pointer;
          letter-spacing:.1em;text-transform:uppercase;
          box-shadow:0 8px 32px rgba(251,191,36,.28),0 2px 8px rgba(251,191,36,.15);
          transition:transform .12s,box-shadow .12s;
          display:flex;align-items:center;justify-content:center;gap:10px;
        }
        .cta:active{transform:scale(.97);box-shadow:0 4px 16px rgba(251,191,36,.2)}

        .footer{
          margin-top:20px;font-size:11px;color:#505060;
          font-family:var(--mono);letter-spacing:.04em;text-align:center;
        }
      `}</style>

      <div className="page">
        <div className="card">
          <div className="icon-wrap">📦</div>

          <h1 className="headline">
            Parcel<span>Vision</span>
          </h1>

          <p className="tagline">
            AI-powered volumetric measurement for boxes and parcels — using just your camera.
          </p>

          <div className="features">
            <span className="feat hi">Real-time</span>
            <span className="feat">YOLOv8 + SAM</span>
            <span className="feat hi">Multi-frame cal</span>
            <span className="feat">Card reference</span>
            <span className="feat hi">3D scan</span>
          </div>

          <button
            className="cta"
            onClick={() => router.push('/camera')}
            onPointerDown={(e) => e.currentTarget.style.transform = "scale(0.97)"}
            onPointerUp={(e) => e.currentTarget.style.transform = "scale(1)"}
          >
            Launch Scanner <span style={{ fontSize: "18px" }}>→</span>
          </button>

          <p className="footer">Point · Calibrate · Measure</p>
        </div>
      </div>
    </>
  )
}