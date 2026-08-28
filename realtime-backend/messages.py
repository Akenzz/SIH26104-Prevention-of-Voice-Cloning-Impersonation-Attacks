"""Outgoing WebSocket JSON. Every field listed in the Task C contract."""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from experts.protocol import Score


class OutgoingMessage(TypedDict):
    type: str
    sequence_number: int
    risk_state: str
    recommended_action: str
    smoothed_probability: Optional[float]
    fused_logit: Optional[float]
    raw_per_expert_scores: dict[str, float]
    model_version: dict[str, str]
    threshold_version: str
    calibrator_version: str
    fusion_version: str
    fusion_mode: str
    dropped_frames: bool
    audio_quality: Optional[str]
    window_index: int
    latency_ms: Optional[float]


def expert_logits(scores: dict[str, Score]) -> dict[str, float]:
    return {name: float(score["logit"]) for name, score in scores.items()}


def expert_versions(scores: dict[str, Score]) -> dict[str, str]:
    return {name: str(score["model_version"]) for name, score in scores.items()}


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
        "model_version": expert_versions(scores),
        "threshold_version": threshold_version,
        "calibrator_version": calibrator_version,
        "fusion_version": fusion_version,
        "fusion_mode": fusion_mode,
        "dropped_frames": bool(dropped_frames),
        "audio_quality": audio_quality,
        "window_index": int(window_index),
        "latency_ms": None if latency_ms is None else round(float(latency_ms), 3),
    }
