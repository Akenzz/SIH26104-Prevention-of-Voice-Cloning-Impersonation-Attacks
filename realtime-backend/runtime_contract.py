"""Validate that runtime scoring and fitted artifacts describe the same system."""

from __future__ import annotations

from typing import Any

from config import Settings
from fusion import FusionConfig


def _string_set(value: object) -> set[str]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item) for item in value}


def effective_experts(settings: Settings) -> set[str]:
    """Return the experts whose scores actually reach calibration."""
    configured = {name.strip().lower() for name in settings.experts}
    if settings.fusion_mode == "fused":
        return configured
    selected = settings.single_expert.strip().lower()
    if selected:
        if selected not in configured:
            raise ValueError(f"SINGLE_EXPERT={selected!r} is not present in EXPERTS")
        return {selected}
    return {settings.experts[0].strip().lower()}


def validate_runtime_contract(settings: Settings, calibrator: Any, fusion: FusionConfig) -> None:
    """Fail before serving if scores would be interpreted by the wrong artifact.

    Calibration is model-, preprocessing-, window-, and fusion-specific.  The
    old backend accepted any combination, including dummy logits through a real
    model's Platt scaler, which made the returned probability misleading.
    """
    configured = {name.strip().lower() for name in settings.experts}
    if "dummy" in configured and len(configured) > 1:
        raise ValueError("DummyExpert cannot be combined with real experts")

    selected = effective_experts(settings)
    if "dummy" in selected:
        if not settings.allow_dummy:
            raise ValueError(
                "DummyExpert is disabled for serving. Set ALLOW_DUMMY=1 only for local protocol tests."
            )
        return
    if settings.fusion_mode == "fused" and not bool((fusion.scope or {}).get("fitted")):
        raise ValueError(
            "FUSION_MODE=fused requires a fitted fusion artifact; fusion-identity-v0 is not deployable."
        )

    scope = getattr(calibrator, "scope", None)
    if not isinstance(scope, dict):
        raise ValueError(
            "Calibrator has no compatibility scope. Refit it or add its expert, fusion, "
            "sample-rate, and window metadata before serving."
        )

    expected_experts = _string_set(scope.get("experts"))
    if expected_experts and expected_experts != selected:
        raise ValueError(
            f"Calibrator expects experts={sorted(expected_experts)}, but runtime uses {sorted(selected)}"
        )
    expected_mode = scope.get("fusion_mode")
    if expected_mode and str(expected_mode) != settings.fusion_mode:
        raise ValueError(
            f"Calibrator expects fusion_mode={expected_mode!r}, got {settings.fusion_mode!r}"
        )
    expected_rate = scope.get("target_sample_rate")
    if expected_rate is not None and int(expected_rate) != settings.target_sample_rate:
        raise ValueError(
            f"Calibrator expects target_sample_rate={expected_rate}, got {settings.target_sample_rate}"
        )
    expected_window = scope.get("window_sec")
    if expected_window is not None and float(expected_window) != settings.window_sec:
        raise ValueError(
            f"Calibrator expects window_sec={expected_window}, got {settings.window_sec}"
        )
