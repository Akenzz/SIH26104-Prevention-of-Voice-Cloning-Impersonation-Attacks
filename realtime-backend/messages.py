"""Outgoing WebSocket JSON. Every field listed in the Task C contract.

`raw_per_expert_scores` and `model_version` are the original contract fields and
are unchanged. `scores` is additive: one entry per expert carrying that expert's
own calibrated probability, its own band, and the calibrator that produced it —
this is what lets the UI show both models side by side without pretending one
expert's Platt fit applies to the other.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from config import EXPERT_LABELS
from experts.protocol import Score
from policy import PolicyConfig, band


class ExpertReport(TypedDict):
    logit: float
    probability: Optional[float]
    risk_state: str
    model_version: str
    label: str
    calibrator_version: Optional[str]


class OutgoingMessage(TypedDict):
    type: str
    sequence_number: int
    risk_state: str
    recommended_action: str
    smoothed_probability: Optional[float]
    fused_logit: Optional[float]
    raw_per_expert_scores: dict[str, float]
    scores: dict[str, ExpertReport]
    model_version: dict[str, str]
    threshold_version: str
    calibrator_version: str
    fusion_version: str
    fusion_mode: str
    dropped_frames: bool
    audio_quality: Optional[str]
    window_index: int
    latency_ms: Optional[float]
    lr_probability: Optional[float]


def expert_logits(scores: dict[str, Score]) -> dict[str, float]:
    return {name: float(score["logit"]) for name, score in scores.items()}


def expert_versions(scores: dict[str, Score]) -> dict[str, str]:
    return {name: str(score["model_version"]) for name, score in scores.items()}


def expert_reports(
    scores: dict[str, Score],
    expert_probabilities: dict[str, float],
    expert_calibrators: dict[str, Any] | None,
    policy: PolicyConfig | None,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name, score in scores.items():
        prob = expert_probabilities.get(name)
        cal = (expert_calibrators or {}).get(name)
        reports[name] = {
            "logit": float(score["logit"]),
            "probability": None if prob is None else float(prob),
            "risk_state": "unavailable" if policy is None else band(prob, policy),
            "model_version": str(score["model_version"]),
            "label": EXPERT_LABELS.get(name, name),
            "calibrator_version": None if cal is None else getattr(cal, "version", None),
        }
    return reports


def build_message(
    *,
    sequence_number: int,
    risk_state: str,
    recommended_action: str,
    smoothed_probability: float | None,
    fused_logit: float | None,
    scores: dict[str, Score],
    threshold_version: str,
    calibrator_version: str,
    fusion_version: str,
    fusion_mode: str,
    dropped_frames: bool,
    audio_quality: str | None,
    window_index: int,
    latency_ms: float | None,
    expert_probabilities: dict[str, float] | None = None,
    lr_probability: float | None = None,
    expert_calibrators: dict[str, Any] | None = None,
    policy: PolicyConfig | None = None,
) -> dict[str, Any]:
    return {
        "type": "score",
        "sequence_number": int(sequence_number),
        "risk_state": risk_state,
        "recommended_action": recommended_action,
        "smoothed_probability": None
        if smoothed_probability is None
        else float(smoothed_probability),
        "fused_logit": None if fused_logit is None else float(fused_logit),
        "raw_per_expert_scores": expert_logits(scores),
        "scores": expert_reports(
            scores, expert_probabilities or {}, expert_calibrators, policy
        ),
        "model_version": expert_versions(scores),
        "threshold_version": threshold_version,
        "calibrator_version": calibrator_version,
        "fusion_version": fusion_version,
        "fusion_mode": fusion_mode,
        "dropped_frames": bool(dropped_frames),
        "audio_quality": audio_quality,
        "window_index": int(window_index),
        "latency_ms": None if latency_ms is None else round(float(latency_ms), 3),
        "lr_probability": None if lr_probability is None else float(lr_probability),
    }
