// Value formatters + audio-quality wording shared by the reasoning narrator
// (lib/narrator.js). This is the trimmed slice of the old Task C frontend
// contract that Task F actually uses — one source of truth so no component
// invents its own labels or number formatting.

// Plain-language reasons for the audio_quality field (backend audio/quality.py).
export const QUALITY_REASON = {
  silence: "Near-silent window — no speech to score.",
  clipped: "Audio is clipping — level too hot to score.",
  clipping: "Audio is clipping — level too hot to score.",
  decode_failure: "Frame could not be decoded.",
  dropped_or_reordered: "Frames dropped or arrived out of order.",
  protocol_error: "Protocol error — audio arrived before the start handshake.",
};

export function qualityText(reason) {
  if (!reason) return null;
  return QUALITY_REASON[reason] || reason;
}

// --- formatters -----------------------------------------------------------

export function pct(p, digits = 0) {
  if (p == null || Number.isNaN(p)) return "N/A";
  return `${(p * 100).toFixed(digits)}%`;
}

export function fixed(x, digits = 2) {
  if (x == null || Number.isNaN(x)) return "N/A";
  return Number(x).toFixed(digits);
}

export function ms(x) {
  if (x == null || Number.isNaN(x)) return "N/A";
  return `${Number(x).toFixed(1)} ms`;
}
