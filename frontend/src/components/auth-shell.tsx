"use client"

import { Music2, Radio, Disc2, Disc, Disc3 } from "lucide-react"

export function MusicAuthShell({ children }: { children: React.ReactNode }) {
  return (
    <>
      <style>{`
        @keyframes eq-bar {
          0%, 100% { transform: scaleY(0.3); }
          50% { transform: scaleY(1); }
        }
        @keyframes float-note {
          0% { transform: translateY(0px) rotate(0deg); opacity: 0.6; }
          50% { transform: translateY(-24px) rotate(8deg); opacity: 1; }
          100% { transform: translateY(0px) rotate(0deg); opacity: 0.6; }
        }
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes pulse-glow {
          0%, 100% { opacity: 0.15; }
          50% { opacity: 0.35; }
        }
        @keyframes scan-line {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
        @keyframes fade-up {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .eq-bar { transform-origin: bottom; }
        .auth-card-glow {
          box-shadow: 0 0 60px rgba(124, 58, 237, 0.12), 0 0 120px rgba(124, 58, 237, 0.06);
        }
        .vinyl-disc {
          animation: spin-slow 8s linear infinite;
          transform-origin: center center;
          width: 100%;
          height: 100%;
          border-radius: 50%;
          position: absolute;
          inset: 0;
        }
      `}</style>

      <div className="min-h-screen w-full flex" style={{ background: "#07070f" }}>

        {/* ── LEFT PANEL (desktop only) ─────────────────────────────────── */}
        <div
          className="hidden lg:flex lg:w-[52%] relative overflow-hidden flex-col justify-between p-14"
          style={{ background: "linear-gradient(135deg, #0e0820 0%, #07070f 60%, #0a0515 100%)" }}
        >
          {/* Ambient orbs */}
          <div
            className="absolute top-[-80px] left-[-60px] w-[500px] h-[500px] rounded-full pointer-events-none"
            style={{
              background: "radial-gradient(circle, rgba(139,92,246,0.18) 0%, transparent 70%)",
              animation: "pulse-glow 4s ease-in-out infinite",
            }}
          />
          <div
            className="absolute bottom-[-100px] right-[-80px] w-[420px] h-[420px] rounded-full pointer-events-none"
            style={{
              background: "radial-gradient(circle, rgba(236,72,153,0.12) 0%, transparent 70%)",
              animation: "pulse-glow 5s ease-in-out 1s infinite",
            }}
          />

          {/* Subtle scan-line effect */}
          <div
            className="absolute inset-0 pointer-events-none overflow-hidden opacity-[0.03]"
            style={{ zIndex: 0 }}
          >
            <div
              className="w-full h-[2px]"
              style={{
                background: "linear-gradient(90deg, transparent, rgba(167,139,250,1), transparent)",
                animation: "scan-line 8s linear infinite",
              }}
            />
          </div>

          {/* Grid texture overlay */}
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              backgroundImage:
                "linear-gradient(rgba(139,92,246,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,0.04) 1px, transparent 1px)",
              backgroundSize: "48px 48px",
              zIndex: 0,
            }}
          />

          {/* Logo */}
          <div className="relative z-10 flex items-center gap-3" style={{ animation: "fade-up 0.6s ease both" }}>
            <div
              className="w-11 h-11 rounded-2xl flex items-center justify-center"
              style={{
                background: "linear-gradient(135deg, #7c3aed, #a855f7)",
                boxShadow: "0 4px 20px rgba(124,58,237,0.5)",
              }}
            >
              <Music2 className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="text-white font-bold text-xl tracking-tight">MuseAI</span>
              <div className="text-[10px] tracking-[0.2em] uppercase" style={{ color: "#a855f7" }}>
                AI Music Studio
              </div>
            </div>
          </div>

          {/* Center block */}
          <div className="relative z-10 space-y-10" style={{ animation: "fade-up 0.7s ease 0.1s both" }}>

            {/* Vinyl record + waveform row */}
            <div className="flex items-center gap-6 ">

              {/* Vinyl record */}
              <div className="relative w-48 h-48 flex-shrink-0">
                {/* Spinning disc — isolated element so needle doesn't rotate */}
                <div className="vinyl-disc" style={{
                  background: "radial-gradient(circle at 50% 50%, #1a1a2e 0%, #0a0a15 40%, #1a0a2e 70%, #0d0d1a 100%)",
                  boxShadow: "0 0 40px rgba(124,58,237,0.3), inset 0 0 30px rgba(0,0,0,0.8)",
                }}>
                  {/* Vinyl grooves */}
                  {[60, 72, 84, 92].map((s) => (
                    <div
                      key={s}
                      className="absolute rounded-full border vinyl-disc"
                      style={{
                        width: `${s}%`,
                        height: `${s}%`,
                        top: `${(100 - s) / 2}%`,
                        left: `${(100 - s) / 2}%`,
                        borderColor: "rgba(110, 108, 112, 0.47)",
                      }}
                    />
                  ))}
                  {/* Center label */}
                  <div
                    className="absolute rounded-full flex items-center justify-center"
                    style={{
                      width: "28%",
                      height: "28%",
                      top: "36%",
                      left: "36%",
                      background: "linear-gradient(135deg, #7c3aed, #ec4899)",
                      boxShadow: "0 0 12px rgba(124,58,237,0.6)",
                    }}
                  >
                    <Disc3 className="w-10 h-10 text-white" />
                  </div>
                </div>

                {/* Needle — stays static, outside spinning div */}
                <div
                  className="absolute"
                  style={{ top: "58px", right: "-36px", zIndex: 10, transform: "rotate(65deg)", transformOrigin: "top right" }}
                >
                  <div className="w-[2px] h-14 rounded-full" style={{ background: "linear-gradient(to bottom, #c4b5fd, #7c3aed)" }} />
                  <div className="w-2 h-2 rounded-full mt-0.5 ml-[-3px]" style={{ background: "#ec4899", boxShadow: "0 0 6px #ec4899" }} />
                </div>
              </div>
            </div>

            {/* Equalizer bars */}
            <div className="flex items-end gap-[3px] h-16">
              {[40, 65, 90, 75, 110, 55, 95, 70, 115, 60, 80, 50, 100, 75, 65, 90, 45, 80, 105, 60].map((h, i) => (
                <div
                  key={i}
                  className="eq-bar flex-1"
                  style={{
                    height: `${h}%`,
                    borderRadius: "2px",
                    background: i % 3 === 0
                      ? "linear-gradient(to top, #7c3aed, #a855f7)"
                      : i % 3 === 1
                      ? "linear-gradient(to top, #6d28d9, #ec4899)"
                      : "linear-gradient(to top, #5b21b6, #8b5cf6)",
                    animation: `eq-bar ${0.6 + (i % 5) * 0.15}s ease-in-out ${i * 0.05}s infinite`,
                    opacity: 0.85,
                  }}
                />
              ))}
            </div>

            {/* Headline */}
            <div className="space-y-3">
              <h1 className="text-[2.6rem] font-black leading-[1.1] text-white tracking-tight">
                Your Music,
                <br />
                <span
                  style={{
                    backgroundImage: "linear-gradient(90deg, #a78bfa, #ec4899, #a855f7)",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    backgroundClip: "text",
                  }}
                >
                  Composed by AI
                </span>
              </h1>
              <p className="text-base leading-relaxed max-w-xs" style={{ color: "rgba(255,255,255,0.4)" }}>
                Generate beats, compose melodies, and discover new sounds tailored perfectly to your creative vision.
              </p>
            </div>

            {/* Feature pills */}
            <div className="flex flex-wrap gap-2">
              {["🎵 AI Composition", "🎛️ Beat Studio", "🎤 Vocal FX", "🎸 Genre Fusion"].map((tag) => (
                <span
                  key={tag}
                  className="text-xs px-3 py-1.5 rounded-full"
                  style={{
                    background: "rgba(124,58,237,0.12)",
                    border: "1px solid rgba(124,58,237,0.25)",
                    color: "rgba(196,181,253,0.9)",
                  }}
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

          {/* Bottom */}
          <div className="relative mt-4 z-10 flex items-center gap-2" style={{ color: "rgba(234, 228, 228, 0.6)", fontSize: "12px" }}>
            <Radio className="w-3 h-3" />
            <span>Now streaming: Your next favourite track</span>
          </div>
        </div>

        {/* ── RIGHT PANEL (auth form) ───────────────────────────────────── */}
        <div
          className="w-full lg:w-[48%] flex flex-col items-center justify-center relative px-5 py-12 sm:px-8 min-h-screen"
          style={{ background: "#07070f" }}
        >
          {/* Ambient glow behind form */}
          <div
            className="absolute pointer-events-none"
            style={{
              width: "480px",
              height: "480px",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              background: "radial-gradient(circle, rgba(124,58,237,0.07) 0%, transparent 70%)",
            }}
          />

          {/* Mobile logo */}
          <div
            className="lg:hidden flex items-center gap-2.5 mb-10"
            style={{ animation: "fade-up 0.5s ease both" }}
          >
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{
                background: "linear-gradient(135deg, #7c3aed, #a855f7)",
                boxShadow: "0 4px 16px rgba(124,58,237,0.45)",
              }}
            >
              <Music2 className="w-4 h-4 text-white" />
            </div>
            <div>
              <span className="text-white font-bold text-lg tracking-tight">MuseAI</span>
              <div className="text-[9px] tracking-[0.18em] uppercase" style={{ color: "#a855f7" }}>
                Music Studio
              </div>
            </div>
          </div>

          {/* Auth card wrapper */}
          <div
            className="w-full max-w-sm relative z-10 auth-card-glow rounded-2xl overflow-hidden"
            style={{ animation: "fade-up 0.6s ease 0.15s both" }}
          >
            {/* Top accent bar */}
            <div
              className="h-[2px] w-full"
              style={{ background: "linear-gradient(90deg, #7c3aed, #ec4899, #7c3aed)" }}
            />
            <div
              className="p-1"
              style={{
                background: "rgba(255,255,255,0.02)",
                backdropFilter: "blur(20px)",
                border: "1px solid rgba(139,92,246,0.12)",
                borderTop: "none",
                borderRadius: "0 0 16px 16px",
              }}
            >
              {children}
            </div>
          </div>


          <p
            className="mt-8 text-[11px] text-center lg:hidden"
            style={{ color: "rgba(255,255,255,0.15)" }}
          >
            © 2025 MuseAI · AI-Powered Music Studio
          </p>
        </div>
      </div>
    </>
  )
}