"""Factory for creating voice converter instances."""

from __future__ import annotations

from typing import Any
from .base import VoiceConverter
from .mock import MockRVCConverter, PitchFormantFallbackConverter
from .rvc import RealRVCConverter


def create_converter(
    converter_type: str = "mock",
    **kwargs: Any,
) -> VoiceConverter:
    """Instantiate a VoiceConverter by type name."""
    c_type = converter_type.lower().strip()
    if c_type in {"mock", "mock_rvc", "mockrvc"}:
        profile = kwargs.get("profile", "clone_female")
        pitch = kwargs.get("pitch_shift_semitones")
        return MockRVCConverter(default_profile=profile, pitch_shift_semitones=pitch)

    elif c_type in {"fallback", "pitch_formant", "dsp"}:
        pitch = kwargs.get("pitch_shift_semitones", 4.0)
        formant = kwargs.get("formant_shift_ratio", 1.15)
        intensity = kwargs.get("artifact_intensity", 0.12)
        return PitchFormantFallbackConverter(
            pitch_shift_semitones=pitch,
            formant_shift_ratio=formant,
            artifact_intensity=intensity,
        )

    elif c_type in {"rvc", "real_rvc", "torch"}:
        return RealRVCConverter(
            model_path=kwargs.get("model_path", ""),
            index_path=kwargs.get("index_path", ""),
            pitch_shift=kwargs.get("pitch_shift", 0.0),
            f0_method=kwargs.get("f0_method", "pm"),
            device=kwargs.get("device", "cpu"),
            fallback_pitch_shift=kwargs.get("fallback_pitch_shift", 4.0),
        )

    else:
        raise ValueError(
            f"Unknown converter type {converter_type!r}. "
            f"Supported: 'mock', 'fallback', 'rvc'."
        )
