// Frontend mirror of the task C backend contract. One source of truth for
// risk-state presentation, audio-quality reasons, and value formatting so no
// component invents its own labels.

// Audio contract (backend config.py). The client declares its own sample_rate
// in the start message; the backend resamples to TARGET_SAMPLE_RATE.
export const TARGET_SAMPLE_RATE = 16000;
export const WINDOW_SEC = 4.0;
export const HOP_SEC = 0.5;

// Policy bands (artifacts/policy.json). Shown as shaded zones on the timeline.
export const BANDS = {
  low_max: 0.35,
  uncertain_max: 0.65,
};

// The five risk states are the whole vocabulary. Each carries an icon key and a
// short line so meaning never rests on color alone (build plan, task E).
export const RISK = {
  collecting: {
    key: "collecting",
    label: "Collecting",
    icon: "collecting",
    color: "var(--r-collecting)",
    wash: "transparent",
    blurb: "Listening. Not enough audio yet to score.",
  },
  low: {
    key: "low",
    label: "Low risk",
    icon: "low",
    color: "var(--r-low)",
    wash: "var(--r-low-wash)",
    blurb: "Speech looks consistent with a real human voice.",
  },
  uncertain: {
    key: "uncertain",
    label: "Uncertain",
    icon: "uncertain",
    color: "var(--r-uncertain)",
    wash: "var(--r-uncertain-wash)",
    blurb: "Inconclusive. Step up verification before proceeding.",
  },
  high: {
    key: "high",
    label: "High risk",
    icon: "high",
    color: "var(--r-high)",
    wash: "var(--r-high-wash)",
    blurb: "Strong evidence of synthetic or manipulated speech.",
  },
  unavailable: {
    key: "unavailable",
    label: "Unavailable",
    icon: "unavailable",
    color: "var(--r-unavailable)",
    wash: "transparent",
    blurb: "Audio quality or stream integrity failed — not a real-speech score.",
  },
};

export const RISK_ORDER = ["collecting", "low", "uncertain", "high", "unavailable"];

export function riskMeta(state) {
  return RISK[state] || RISK.unavailable;
}

// Plain-language reasons for the audio_quality field (backend audio/quality.py).
export const QUALITY_REASON = {
  silence: "Near-silent window — no speech to score.",
  clipped: "Audio is clipping — level too hot to score.",
  decode_failure: "Frame could not be decoded.",
  dropped_or_reordered: "Frames dropped or arrived out of order.",
  protocol_error: "Protocol error — audio arrived before the start handshake.",
};

export function qualityText(reason) {
  if (!reason) return null;
  return QUALITY_REASON[reason] || reason;
}

// Version strings the real backend ships (artifacts/*.json). The simulation
// path reuses them so a viewer sees the same provenance either way.
export const DEFAULT_VERSIONS = {
  threshold_version: "policy-v0",
  calibrator_version: "platt-lfcc-asvspoof19dev-v1",
  fusion_version: "fusion-identity-v0",
  fusion_mode: "single",
};

// --- formatters -----------------------------------------------------------

export function pct(p, digits = 0) {
  if (p == null || Number.isNaN(p)) return "—";
  return `${(p * 100).toFixed(digits)}%`;
}

export function fixed(x, digits = 2) {
  if (x == null || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

export function ms(x) {
  if (x == null || Number.isNaN(x)) return "—";
  return `${Number(x).toFixed(1)} ms`;
}

// An expert whose name/version is "dummy" is a random stand-in, not detection.
export function isDummyVersion(version) {
  return typeof version === "string" && version.toLowerCase().startsWith("dummy");
}
