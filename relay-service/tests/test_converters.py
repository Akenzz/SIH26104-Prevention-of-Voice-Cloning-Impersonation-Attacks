"""Unit tests for VoiceConverter implementations: Mock, Fallback, and RVC."""

import numpy as np
import pytest

from converters.base import VoiceConverter
from converters.mock import MockRVCConverter, PitchFormantFallbackConverter
from converters.rvc import RealRVCConverter
from converters.factory import create_converter


def test_pitch_formant_fallback_converter(sample_sine_wave: np.ndarray):
    converter = PitchFormantFallbackConverter(
        pitch_shift_semitones=4.0,
        formant_shift_ratio=1.15,
        artifact_intensity=0.12,
    )
    assert converter.is_ready is True
    assert "PitchFormantFallback" in converter.name

    # Convert 1.0s audio
    converted = converter.convert(sample_sine_wave, sample_rate=16000)
    assert isinstance(converted, np.ndarray)
    assert converted.shape == sample_sine_wave.shape
    assert converted.dtype == np.float32

    # Verify altered signal is not identical to input (pitch/formants shifted)
    diff = np.max(np.abs(converted - sample_sine_wave))
    assert diff > 0.05

    # Energy should be preserved within reasonable bounds
    in_rms = np.sqrt(np.mean(np.square(sample_sine_wave)))
    out_rms = np.sqrt(np.mean(np.square(converted)))
    assert abs(in_rms - out_rms) < 0.2


def test_mock_rvc_converter_profiles(sample_sine_wave: np.ndarray):
    mock = MockRVCConverter(default_profile="clone_female")
    assert mock.is_ready is True
    assert mock.current_profile == "clone_female"

    meta = mock.get_metadata()
    assert "clone_female" in meta["available_profiles"]
    assert "clone_male" in meta["available_profiles"]

    out_female = mock.convert(sample_sine_wave)
    assert out_female.shape == sample_sine_wave.shape

    # Switch profile to clone_deep
    assert mock.set_profile("clone_deep") is True
    assert mock.current_profile == "clone_deep"
    out_deep = mock.convert(sample_sine_wave)
    assert out_deep.shape == sample_sine_wave.shape

    # Different profiles should produce different outputs
    profile_diff = np.max(np.abs(out_female - out_deep))
    assert profile_diff > 0.05


def test_real_rvc_converter_fallback(sample_sine_wave: np.ndarray):
    # Testing when model_path does not exist -> should seamlessly use fallback without raising error
    rvc = RealRVCConverter(model_path="nonexistent_model.pth")
    assert rvc.is_ready is True
    assert rvc.is_using_fallback is True
    assert "Fallback" in rvc.name

    converted = rvc.convert(sample_sine_wave)
    assert converted.shape == sample_sine_wave.shape
    assert np.isfinite(converted).all()


def test_converter_factory():
    # Mock
    c1 = create_converter("mock")
    assert isinstance(c1, MockRVCConverter)

    # Fallback
    c2 = create_converter("fallback")
    assert isinstance(c2, PitchFormantFallbackConverter)

    # RVC
    c3 = create_converter("rvc")
    assert isinstance(c3, RealRVCConverter)
    assert c3.is_using_fallback is True

    # Unknown
    with pytest.raises(ValueError):
        create_converter("invalid_engine_name")
