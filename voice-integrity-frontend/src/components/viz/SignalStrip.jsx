import { useEffect, useRef } from "react";

// Oscilloscope-style level meter. Reads the current time-domain waveform each
// animation frame from the monitor (real analyser when live, synthetic trace in
// simulation) and paints a phosphor trail. This is the raw signal, not the
// verdict, so it stays the brand cyan regardless of risk.
export default function SignalStrip({ readWaveform, running, state }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const bufRef = useRef(new Float32Array(1024));
  const rmsRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let w = 0;
    let h = 0;
    const dpr = Math.min(2, window.devicePixelRatio || 1);

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      w = Math.max(1, Math.floor(r.width));
      h = Math.max(1, Math.floor(r.height));
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const draw = () => {
      const buf = bufRef.current;
      const rms = readWaveform(buf) * (running ? 1 : 0.12);
      // exponential meter smoothing
      rmsRef.current = rmsRef.current * 0.8 + rms * 0.2;

      // fade previous frame for a phosphor trail
      ctx.fillStyle = "rgba(10, 13, 20, 0.28)";
      ctx.fillRect(0, 0, w, h);

      // midline
      ctx.strokeStyle = "rgba(53, 224, 208, 0.12)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();

      // waveform
      const unavailable = state === "unavailable";
      const color = unavailable ? "rgba(123,134,158,0.9)" : "rgba(53, 224, 208, 0.95)";
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.75;
      ctx.shadowColor = unavailable ? "transparent" : "rgba(53,224,208,0.55)";
      ctx.shadowBlur = unavailable ? 0 : 8;
      ctx.beginPath();
      const n = buf.length;
      const amp = h * 0.42;
      for (let i = 0; i < n; i++) {
        const x = (i / (n - 1)) * w;
        const y = h / 2 - buf[i] * amp * (running ? 1 : 0.9);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
    };
  }, [readWaveform, running, state]);

  return (
    <div className="scope">
      <canvas ref={canvasRef} className="scope-canvas" />
      <div className="scope-foot">
        <span className="mono scope-tag">INPUT LEVEL</span>
        <Meter rmsRef={rmsRef} running={running} />
      </div>
    </div>
  );
}

function Meter({ rmsRef, running }) {
  const barRef = useRef(null);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const el = barRef.current;
      if (el) {
        const v = Math.min(1, rmsRef.current * 2.4);
        el.style.width = `${v * 100}%`;
        el.style.opacity = running ? "1" : "0.4";
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [rmsRef, running]);
  return (
    <div className="meter">
      <div ref={barRef} className="meter-fill" />
    </div>
  );
}
