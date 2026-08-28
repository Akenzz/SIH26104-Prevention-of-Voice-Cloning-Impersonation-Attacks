"""Map smoothed probability + quality into one risk band and an action."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RISK_STATES = ("collecting", "low", "uncertain", "high", "unavailable")


@dataclass
class PolicyConfig:
    version: str
    collecting_windows: int
    low_max: float
    uncertain_max: float
    actions: dict[str, str]


def load_policy(path: Path) -> PolicyConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    actions = {str(k): str(v) for k, v in data.get("actions", {}).items()}
    return PolicyConfig(
        version=str(data.get("version", "policy-unversioned")),
        collecting_windows=int(data.get("collecting_windows", 2)),
        low_max=float(data.get("low_max", 0.35)),
        uncertain_max=float(data.get("uncertain_max", 0.65)),
        actions=actions,
    )


def decide(
    *,
    quality_ok: bool,
    quality_reason: str | None,
    windows_scored: int,
    smoothed_probability: float | None,
    config: PolicyConfig,
) -> tuple[str, str, str | None]:
    """Return (risk_state, recommended_action, audio_quality_flag)."""
    if not quality_ok:
        reason = quality_reason or "unavailable"
        action = config.actions.get(
            "unavailable",
            "Audio quality or stream integrity failed. Do not treat this as a real-speech score.",
        )
        return "unavailable", action, reason

    if windows_scored < config.collecting_windows or smoothed_probability is None:
        action = config.actions.get(
            "collecting",
            "Keep the call going; not enough audio yet to score.",
        )
        return "collecting", action, None

    p = float(smoothed_probability)
    if p < config.low_max:
        state = "low"
    elif p < config.uncertain_max:
        state = "uncertain"
    else:
        state = "high"
    action = config.actions.get(state, state)
    return state, action, None
