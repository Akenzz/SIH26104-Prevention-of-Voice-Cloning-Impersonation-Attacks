"""
Plain-language, grounded findings from prosody features.
========================================================
`describe_prosody_evidence(features_dict) -> list[str]` returns SHORT
human-readable findings for a single clip. A finding is emitted ONLY when the
feature value falls outside the normal human range established from the training
data's bonafide examples (stored as p5/p95 bands in the trained artifact). This
is what gets handed to the Groq reasoning-trace explainer, so it must never
invent or exaggerate — only report what this specific clip's numbers support.

The human bands are loaded from the model artifact (`bonafide_ranges`). If no
artifact is supplied, a conservative built-in default band is used so the
function still degrades safely — but the trained bands are strongly preferred.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Each rule: (feature, direction, message). direction "low" fires when value is
# below the bonafide p5; "high" fires when above the bonafide p95. Only findings
# with a real human-range violation for THIS clip are surfaced.
_RULES = [
    ("f0_std",           "low",  "pitch variation is unusually flat"),
    ("f0_range",         "low",  "pitch range is unusually narrow"),
    ("f0_delta_var",     "low",  "pitch transitions are unnaturally smooth"),
    ("f0_delta_var",     "high", "pitch jumps abruptly between frames"),
    ("pause_var",        "low",  "pause timing is unusually regular"),
    ("pause_mean",       "low",  "pauses are unusually short or absent"),
    ("rate_var",         "low",  "speaking rate is unusually constant"),
    ("jitter_local",     "low",  "no natural pitch micro-variation (jitter) detected"),
    ("shimmer_local",    "low",  "no natural loudness micro-variation (shimmer) detected"),
    ("spectral_flatness","high", "spectrum is unusually flat / over-smoothed"),
]

# Conservative fallback bands (used only when no trained artifact is loaded).
# Deliberately wide so nothing fires spuriously without real training evidence.
_DEFAULT_BANDS = {
    "f0_std":            {"p5": 5.0,   "p95": 120.0},
    "f0_range":          {"p5": 20.0,  "p95": 400.0},
    "f0_delta_var":      {"p5": 1.0,   "p95": 5000.0},
    "pause_var":         {"p5": 0.0,   "p95": 5.0},
    "pause_mean":        {"p5": 0.0,   "p95": 2.0},
    "rate_var":          {"p5": 0.0,   "p95": 50.0},
    "jitter_local":      {"p5": 0.005, "p95": 0.08},
    "shimmer_local":     {"p5": 0.002, "p95": 0.20},
    "spectral_flatness": {"p5": 0.0,   "p95": 0.5},
}


def load_bonafide_bands(artifact_path: Optional[str] = None) -> dict:
    """Load {feature: {p5, p50, p95}} bands from the trained artifact."""
    if artifact_path is None:
        return _DEFAULT_BANDS
    with open(artifact_path) as f:
        art = json.load(f)
    return art.get("bonafide_ranges", _DEFAULT_BANDS)


def _findings(features: dict, bands: dict) -> list[dict]:
    """Structured findings: only rules whose feature violates the human band."""
    results = []
    seen_features = set()
    for feat, direction, msg in _RULES:
        if feat not in features or feat not in bands:
            continue
        val = float(features[feat])
        band = bands[feat]
        lo, hi = float(band.get("p5", 0.0)), float(band.get("p95", 0.0))
        fired = (direction == "low" and val < lo) or (direction == "high" and val > hi)
        if not fired:
            continue
        # avoid two opposite findings on the same feature; first match wins
        if feat in seen_features:
            continue
        seen_features.add(feat)
        results.append({
            "finding": msg,
            "feature": feat,
            "value": round(val, 5),
            "human_band": {"p5": round(lo, 5), "p95": round(hi, 5)},
            "direction": direction,
        })
    return results


def describe_prosody_evidence(
    features_dict: dict,
    artifact_path: Optional[str] = None,
    bands: Optional[dict] = None,
) -> list[str]:
    """Plain-language findings for this clip. Empty list = nothing anomalous.

    Only findings whose feature is genuinely outside the bonafide human range are
    returned — this never lists all rules, only the ones the numbers support.
    """
    if bands is None:
        bands = load_bonafide_bands(artifact_path)
    return [f["finding"] for f in _findings(features_dict, bands)]


def describe_prosody_evidence_structured(
    features_dict: dict,
    artifact_path: Optional[str] = None,
    bands: Optional[dict] = None,
) -> list[dict]:
    """Same findings as :func:`describe_prosody_evidence` but with the numeric
    evidence attached, for the LLM explainer payload."""
    if bands is None:
        bands = load_bonafide_bands(artifact_path)
    return _findings(features_dict, bands)
