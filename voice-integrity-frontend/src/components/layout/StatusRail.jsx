import { IconWave, RiskGlyph } from "@/components/ui/icons.jsx";
import { riskMeta, pct } from "@/lib/contract.js";

// Top rail: product identity on the left, a compact live-run chip on the right.
export default function StatusRail({ current, source, running }) {
  const state = current?.risk_state;
  const meta = state ? riskMeta(state) : null;

  return (
    <header className="rail">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <IconWave size={20} />
        </span>
        <div className="brand-text">
          <span className="brand-name">Voice Integrity Monitor</span>
          <span className="brand-sub">Real-time synthetic-speech risk for live calls</span>
        </div>
      </div>

      <div className="rail-right">
        <span className="rail-tag mono">SIH26104</span>
        <span className={`rail-source mono ${source === "sim" ? "rail-source-sim" : "rail-source-live"}`}>
          {source === "sim" ? "SIMULATION" : "LIVE"}
        </span>
        {running && meta && (
          <span className="rail-risk" style={{ color: meta.color, borderColor: meta.color }}>
            <RiskGlyph state={state} size={16} />
            {meta.label}
            {current?.smoothed_probability != null && (
              <b className="mono">{pct(current.smoothed_probability)}</b>
            )}
          </span>
        )}
      </div>
    </header>
  );
}
