import Gauge from "@/components/viz/Gauge.jsx";
import { RiskGlyph } from "@/components/ui/icons.jsx";
import { riskMeta, qualityText } from "@/lib/contract.js";

// The load-bearing readout: current risk band (icon + label + number) and the
// recommended next action in plain language. Text and number never disagree —
// both come from the same message.
export default function VerdictPanel({ current, status }) {
  const idle = !current && status !== "running" && status !== "draining";
  const state = current?.risk_state || (status === "connecting" ? "collecting" : "collecting");
  const meta = riskMeta(state);
  const p = current?.smoothed_probability ?? null;
  const action = current?.recommended_action || meta.blurb;
  const qNote = current?.audio_quality ? qualityText(current.audio_quality) : null;
  const alert = state === "high";

  return (
    <section
      className={`verdict verdict-${state} ${alert ? "verdict-alert" : ""}`}
      style={{ "--risk": meta.color, "--risk-wash": meta.wash }}
      aria-live="polite"
    >
      <div className="verdict-main">
        <div className="verdict-read">
          <span className="verdict-eyebrow mono">Live assessment</span>
          <div className="verdict-band">
            <span className="verdict-glyph" style={{ color: meta.color }}>
              <RiskGlyph state={state} size={34} />
            </span>
            <h1 className="verdict-label" style={{ color: meta.color }}>
              {idle ? "Standby" : meta.label}
            </h1>
          </div>
          <p className="verdict-blurb">
            {idle ? "Choose a source and start the monitor to begin scoring." : meta.blurb}
          </p>
        </div>

        <Gauge p={idle ? null : p} state={idle ? "unavailable" : state} />
      </div>

      <div className="verdict-action">
        <span className="verdict-action-label mono">Recommended action</span>
        <p className="verdict-action-text">
          {idle ? "—" : action}
        </p>
        {qNote && <p className="verdict-quality mono">⚠ {qNote}</p>}
      </div>
    </section>
  );
}
