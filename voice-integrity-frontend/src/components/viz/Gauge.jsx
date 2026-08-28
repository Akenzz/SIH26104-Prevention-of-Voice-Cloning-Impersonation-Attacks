import { BANDS, pct, riskMeta } from "@/lib/contract.js";

// 270° arc gauge. The track shows the three policy bands; the value arc is drawn
// in the current risk color. Center overlay carries the number + label so a
// viewer never reads the gauge by hue alone.

const CX = 100;
const CY = 100;
const R = 82;
const START = 135; // degrees
const SWEEP = 270;

function polar(cx, cy, r, deg) {
  const a = ((deg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}

function arc(cx, cy, r, startDeg, endDeg) {
  const s = polar(cx, cy, r, endDeg);
  const e = polar(cx, cy, r, startDeg);
  const large = endDeg - startDeg <= 180 ? 0 : 1;
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 0 ${e.x} ${e.y}`;
}

function valToDeg(v) {
  return START + SWEEP * Math.min(1, Math.max(0, v));
}

export default function Gauge({ p, state }) {
  const meta = riskMeta(state);
  const hasValue = p != null && state !== "unavailable";
  const bands = [
    { from: 0, to: BANDS.low_max, color: "var(--r-low)" },
    { from: BANDS.low_max, to: BANDS.uncertain_max, color: "var(--r-uncertain)" },
    { from: BANDS.uncertain_max, to: 1, color: "var(--r-high)" },
  ];
  const valColor = state === "collecting" ? "var(--r-collecting)" : meta.color;

  return (
    <div className="gauge">
      <svg viewBox="0 0 200 200" className="gauge-svg" role="img"
        aria-label={hasValue ? `Smoothed spoof probability ${pct(p)}, ${meta.label}` : `${meta.label}`}>
        {/* base track */}
        <path d={arc(CX, CY, R, START, START + SWEEP)} className="gauge-track" />
        {/* band zones */}
        {bands.map((b, i) => (
          <path
            key={i}
            d={arc(CX, CY, R, valToDeg(b.from), valToDeg(b.to))}
            stroke={b.color}
            className="gauge-band"
          />
        ))}
        {/* value arc */}
        {hasValue && (
          <path
            d={arc(CX, CY, R, START, valToDeg(p))}
            stroke={valColor}
            className="gauge-value"
            style={{ filter: `drop-shadow(0 0 6px ${valColor})` }}
          />
        )}
        {/* tick at each threshold */}
        {[BANDS.low_max, BANDS.uncertain_max].map((t) => {
          const o = polar(CX, CY, R + 9, valToDeg(t));
          const inr = polar(CX, CY, R - 9, valToDeg(t));
          return <line key={t} x1={o.x} y1={o.y} x2={inr.x} y2={inr.y} className="gauge-tick" />;
        })}
        {/* needle */}
        {hasValue && (
          <>
            <line
              x1={CX}
              y1={CY}
              x2={polar(CX, CY, R - 6, valToDeg(p)).x}
              y2={polar(CX, CY, R - 6, valToDeg(p)).y}
              stroke={valColor}
              className="gauge-needle"
            />
            <circle cx={CX} cy={CY} r="5" fill={valColor} />
          </>
        )}
      </svg>
      <div className="gauge-center">
        <div className="gauge-pct mono" style={{ color: valColor }}>
          {hasValue ? pct(p) : "—"}
        </div>
        <div className="gauge-caption mono">spoof prob.</div>
      </div>
    </div>
  );
}
