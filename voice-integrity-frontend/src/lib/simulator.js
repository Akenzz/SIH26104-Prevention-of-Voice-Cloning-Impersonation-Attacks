// Simulation source. Emits messages byte-compatible with the task C backend so
// the whole dashboard can be demonstrated with no backend running — the
// "prerecorded backup" path the build plan insists every live demo keep ready.
//
// Nothing here is a detection result. It is a scripted stand-in, surfaced in the
// UI as "Simulation" so no viewer mistakes it for a measured score.

import { BANDS, DEFAULT_VERSIONS, HOP_SEC } from "./contract.js";

// Platt calibrator shipped in artifacts/calibrator.json, inverted so a target
// probability yields the fused logit a real expert would have produced.
const CAL_A = 1.321248982453757;
const CAL_B = -0.5199802458132765;

function logit(p) {
  const q = Math.min(1 - 1e-4, Math.max(1e-4, p));
  return Math.log(q / (1 - q));
}
function fusedLogitForProb(p) {
  return (logit(p) - CAL_B) / CAL_A;
}

const ACTIONS = {
  collecting: "Keep the call going; not enough audio yet to score.",
  low: "Smoothed spoof risk is low. Continue, still verify out-of-band for sensitive steps.",
  uncertain: "Risk is inconclusive. Pause the request and use a known callback or MFA.",
  high: "Smoothed risk is high. Pause the request and verify through a known callback number.",
  unavailable:
    "Audio quality or stream integrity failed. Do not treat this as a real-speech score.",
};

function bandFor(p) {
  if (p < BANDS.low_max) return "low";
  if (p < BANDS.uncertain_max) return "uncertain";
  return "high";
}

// Probability trajectory per scenario, indexed by window number (1-based).
const CURVES = {
  genuine: (n) => 0.12 - 0.02 * Math.min(n, 5) + noise(n, 0.03),
  clone: (n) => {
    const ramp = [0.18, 0.2, 0.24, 0.3, 0.36, 0.44, 0.53, 0.62, 0.71, 0.79, 0.85, 0.89];
    const base = n <= ramp.length ? ramp[n - 1] : 0.88;
    return base + noise(n, 0.02);
  },
  degraded: (n) => {
    const ramp = [0.19, 0.22, 0.27, 0.35, 0.43, 0.52, 0.63, 0.73, 0.82, 0.87];
    const base = n <= ramp.length ? ramp[n - 1] : 0.86;
    return base + noise(n, 0.025);
  },
};

// Deterministic pseudo-noise (no Math.random so runs are reproducible).
function noise(n, amp) {
  const s = Math.sin(n * 12.9898) * 43758.5453;
  return (s - Math.floor(s) - 0.5) * 2 * amp;
}

// Quality faults for the degraded scenario: window index -> reason.
const DEGRADED_FAULTS = {
  5: "silence",
  6: "clipped",
  11: "dropped_or_reordered",
};

export const SCENARIOS = [
  {
    id: "clone",
    label: "AI clone (escalates)",
    blurb: "A cloned voice — risk climbs through the bands and trips the alert.",
  },
  {
    id: "genuine",
    label: "Genuine caller",
    blurb: "A real human voice — risk stays low the whole call.",
  },
  {
    id: "degraded",
    label: "Degraded line",
    blurb: "Silence, clipping and a dropped frame — the fail-safe holds.",
  },
];

export class Simulator {
  constructor({ scenario = "clone", expert = "lfcc", modelVersion = "best_lfcc_lcnn" } = {}) {
    this.scenario = scenario;
    this.expert = expert;
    this.modelVersion = modelVersion;
    this.timer = null;
    this.seq = 0;
    this.window = 0;
    this.onMessage = null;
  }

  start(onMessage) {
    this.onMessage = onMessage;
    this.seq = 0;
    this.window = 0;
    // A short warmup before the first window, then one message per hop.
    this._warmup = setTimeout(() => {
      this._tick();
      this.timer = setInterval(() => this._tick(), HOP_SEC * 1000);
    }, 900);
    // announce the source like a ready frame
    onMessage({ type: "ready", simulated: true, expert: this.expert });
  }

  _tick() {
    this.window += 1;
    const n = this.window;
    const fault = this.scenario === "degraded" ? DEGRADED_FAULTS[n] : null;
    const collecting = n <= 2;

    if (fault) {
      this._emit({
        risk_state: "unavailable",
        smoothed_probability: null,
        fused_logit: null,
        scores: {},
        versions: {},
        dropped_frames: fault === "dropped_or_reordered",
        audio_quality: fault,
        action: ACTIONS.unavailable,
      });
      return;
    }

    let p = CURVES[this.scenario](n);
    p = Math.min(0.97, Math.max(0.02, p));
    const state = collecting ? "collecting" : bandFor(p);
    const fused = fusedLogitForProb(p);
    this._emit({
      risk_state: state,
      smoothed_probability: p,
      fused_logit: fused,
      scores: { [this.expert]: fused + noise(n + 1, 0.15) },
      versions: { [this.expert]: this.modelVersion },
      dropped_frames: false,
      audio_quality: null,
      action: ACTIONS[state],
    });
  }

  _emit({ risk_state, smoothed_probability, fused_logit, scores, versions, dropped_frames, audio_quality, action }) {
    this.seq += 1;
    this.onMessage({
      type: "score",
      sequence_number: this.seq,
      risk_state,
      recommended_action: action,
      smoothed_probability,
      fused_logit,
      raw_per_expert_scores: scores,
      model_version: versions,
      threshold_version: DEFAULT_VERSIONS.threshold_version,
      calibrator_version: DEFAULT_VERSIONS.calibrator_version,
      fusion_version: DEFAULT_VERSIONS.fusion_version,
      fusion_mode: DEFAULT_VERSIONS.fusion_mode,
      dropped_frames,
      audio_quality,
      window_index: this.window,
      latency_ms: 8 + Math.abs(noise(this.seq, 6)),
      simulated: true,
    });
  }

  stop() {
    if (this._warmup) clearTimeout(this._warmup);
    if (this.timer) clearInterval(this.timer);
    this._warmup = null;
    this.timer = null;
  }
}
