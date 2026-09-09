"""Pytest configuration and fixtures for relay-service tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import numpy as np


@pytest.fixture
def sample_sine_wave() -> np.ndarray:
    """1.0 second 440 Hz sine wave at 16kHz float32."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def sample_pcm_s16le(sample_sine_wave: np.ndarray) -> bytes:
    """1.0 second int16 PCM bytes at 16kHz."""
    clipped = np.clip(sample_sine_wave, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()
