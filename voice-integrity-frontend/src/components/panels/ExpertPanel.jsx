import { Stat } from "@/components/ui/primitives.jsx";
import { fixed, ms, HOP_SEC, isDummyVersion } from "@/lib/contract.js";

// Everything under the verdict: which experts fired, their raw logits and
// versions, the fusion/calibration provenance, and stream integrity. Also the
// two honesty flags — a "dummy" expert is a random stand-in, and the calibrated
// probability is only valid in-domain.
export default function ExpertPanel({ current, stats }) {
  const scores = current?.raw_per_expert_scores || {};
  const versions = current?.model_version || {};
  const names = Object.keys(scores);
  const hopMs = HOP_SEC * 1000;
  const overBudget = stats.maxLatency > hopMs;
  const anyDummy = names.some((n) => isDummyVersion(versions[n]) || n === "dummy");

  return (
    <div className="experts">
      <div className="experts-list">
        {names.length === 0 && <p className="muted">No expert scores yet.</p>}
        {names.map((n) => {
          const dummy = isDummyVersion(versions[n]) || n === "dummy";
          return (
            <div key={n} className="expert-row">
              <div className="expert-id">
                <span className="expert-name">{n}</span>
                <span className="expert-ver mono">{versions[n] || "—"}</span>
              </div>
              <div className="expert-metric">
                <span className="expert-logit mono">{fixed(scores[n], 2)}</span>
                <span className="expert-logit-label mono">logit</span>
              </div>
              {dummy && <span className="tag tag-warn" title="Random stand-in, not a detection result">stand-in</span>}
            </div>
          );
        })}
      </div>

      <div className="experts-grid">
        <Stat label="fused logit" value={fixed(current?.fused_logit, 2)} />
        <Stat label="fusion" value={current?.fusion_mode || "—"} sub={current?.fusion_version} />
        <Stat label="calibrator" value={current?.calibrator_version ? "platt" : "—"} sub={current?.calibrator_version} />
        <Stat label="policy" value={current?.threshold_version || "—"} />
      </div>

      <div className="integrity">
        <span className="integrity-title mono">STREAM INTEGRITY</span>
        <div className="integrity-row">
          <Stat label="seq" value={stats.lastSeq ?? "—"} sub={stats.seqOk ? "monotonic ✓" : "gap detected ✗"} tone={stats.seqOk ? "var(--r-low)" : "var(--r-high)"} />
          <Stat label="dropped" value={stats.dropped} tone={stats.dropped ? "var(--r-uncertain)" : undefined} />
          <Stat label="max latency" value={ms(stats.maxLatency)} sub={`budget ${hopMs} ms`} tone={overBudget ? "var(--r-high)" : "var(--r-low)"} />
          <Stat label="windows" value={stats.count} />
        </div>
      </div>

      <div className="caveats">
        {anyDummy && (
          <p className="caveat">
            <b>Stand-in expert active.</b> Scores from a <code>dummy</code> expert are random and are
            not detection results.
          </p>
        )}
        <p className="caveat">
          Probability is Platt-calibrated on ASVspoof2019&nbsp;LA (in-domain). On out-of-domain
          audio treat it as unreliable — the honest failure mode, not a fitted one.
        </p>
      </div>
    </div>
  );
}
