// Grounded narration for the reasoning log (task F, local/offline path).
//
// It paraphrases ONLY fields the backend actually sent — risk state, smoothed
// probability, per-expert logits, audio-quality flags, versions. It never
// invents evidence (no "spectral artifacts", no "abnormal prosody"). This is a
// narration convenience, not explainable AI, and the copy stays within that.

import { pct, fixed, qualityText } from "./contract.js";

function expertPhrase(msg) {
  const scores = msg.raw_per_expert_scores || {};
  const names = Object.keys(scores);
  if (names.length === 0) return null;
  const parts = names.map((n) => `${n} logit ${fixed(scores[n], 2)}`);
  return parts.join(", ");
}

// Returns { text, kind } for a line worth showing, or null to stay quiet.
// kind drives styling: collect | info | warn | alert | unavailable | verdict.
export function narrate(prev, msg) {
  const n = msg.window_index;
  const state = msg.risk_state;
  const p = msg.smoothed_probability;
  const prevState = prev?.risk_state;
  const prevP = prev?.smoothed_probability;

  if (state === "unavailable") {
    const reason = qualityText(msg.audio_quality) || "stream integrity failed";
    // only narrate the first unavailable in a run to avoid spam
    if (prevState === "unavailable") return null;
    return {
      text: `Window ${n}: ${reason} Holding at unavailable — not scored as real.`,
      kind: "unavailable",
    };
  }

  if (state === "collecting") {
    if (prevState === "collecting") return null;
    return {
      text: `Window ${n}: collecting audio, ${pct(p)} smoothed spoof probability so far.`,
      kind: "collect",
    };
  }

  const changed = state !== prevState;
  const moved = prevP == null || p == null ? true : Math.abs(p - prevP) >= 0.06;

  if (changed) {
    const ex = expertPhrase(msg);
    const tail = ex ? ` (${ex}).` : ".";
    if (state === "high") {
      return {
        text: `Window ${n}: smoothed probability ${pct(p)} crossed the high threshold${tail}`,
        kind: "alert",
      };
    }
    if (state === "uncertain") {
      return {
        text: `Window ${n}: smoothed probability ${pct(p)} entered the uncertain band${tail}`,
        kind: "warn",
      };
    }
    return {
      text: `Window ${n}: smoothed probability ${pct(p)} — ${state} risk${tail}`,
      kind: "info",
    };
  }

  if (moved) {
    return {
      text: `Window ${n}: ${pct(p)} smoothed, holding ${state}.`,
      kind: state === "high" ? "alert" : state === "uncertain" ? "warn" : "info",
    };
  }
  return null;
}

// A closing verdict line when a stream stops, grounded in the final state.
export function verdictLine(last) {
  if (!last) return null;
  const s = last.risk_state;
  if (s === "high")
    return { text: `Verdict: ended at high risk (${pct(last.smoothed_probability)}). Recommend step-up verification before any sensitive action.`, kind: "verdict" };
  if (s === "uncertain")
    return { text: `Verdict: ended uncertain (${pct(last.smoothed_probability)}). Not cleared — verify out of band.`, kind: "verdict" };
  if (s === "low")
    return { text: `Verdict: ended at low risk (${pct(last.smoothed_probability)}). No synthetic-speech evidence, still not proof of identity.`, kind: "verdict" };
  return { text: `Verdict: ended ${s}.`, kind: "verdict" };
}
