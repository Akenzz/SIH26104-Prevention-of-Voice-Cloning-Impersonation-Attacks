from __future__ import annotations

import io
import numpy as np
import pytest
import soundfile as sf

from audio.quality import assess_window, decode_pcm
from audio.resample import (
    decode_audio_bytes,
    pad_or_tile_window,
    rms_normalize,
    to_mono,
    to_target_rate,
)
from audio.ring_buffer import RingBuffer


def test_ring_buffer_emits_overlapping_windows():
    buf = RingBuffer(window_samples=8, hop_samples=4)
    first = buf.push(np.arange(8, dtype=np.float32))
    assert len(first) == 1
    assert first[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    second = buf.push(np.arange(8, 12, dtype=np.float32))
    assert len(second) == 1
    assert second[0].tolist() == [4, 5, 6, 7, 8, 9, 10, 11]
    assert len(buf) <= 8


def test_ring_buffer_stays_bounded():
    buf = RingBuffer(window_samples=16, hop_samples=4)
    for i in range(50):
        buf.push(np.ones(8, dtype=np.float32) * i)
    assert len(buf) < 16 + 8


def test_resample_logs_rate_change():
    src = np.linspace(-0.2, 0.2, 48000, dtype=np.float32)
    out, did = to_target_rate(src, 48000, 16000)
    assert did is True
    assert abs(out.size - 16000) <= 1


def test_no_resample_when_already_16k():
    src = np.linspace(-0.2, 0.2, 16000, dtype=np.float32)
    out, did = to_target_rate(src, 16000, 16000)
    assert did is False
    assert out.size == 16000


def test_to_mono_averages_channels():
    stereo = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    mono = to_mono(stereo, channels=2)
    assert mono.tolist() == [0.0, 0.0]


def test_decode_int16_and_reject_odd_bytes():
    raw = np.array([0, 16384, -16384], dtype="<i2").tobytes()
    samples, q = decode_pcm(raw, "pcm_s16le", 1)
    assert q.ok
    assert samples is not None
    assert abs(float(samples[1]) - 0.5) < 1e-3
    bad, q_bad = decode_pcm(b"\x00", "pcm_s16le", 1)
    assert bad is None
    assert q_bad.reason == "decode_failure"


def test_silence_and_clip_fail_quality():
    silence = np.zeros(64000, dtype=np.float32)
    q = assess_window(silence, silence_rms=1e-4, clip_abs=0.99, clip_fraction=0.01)
    assert q.ok is False
    assert q.reason == "silence"

    clipped = np.ones(64000, dtype=np.float32)
    q2 = assess_window(clipped, silence_rms=1e-4, clip_abs=0.99, clip_fraction=0.01)
    assert q2.ok is False
    assert q2.reason == "clipped"

    ok = np.random.default_rng(0).normal(0, 0.1, 64000).astype(np.float32)
    q3 = assess_window(ok, silence_rms=1e-4, clip_abs=0.99, clip_fraction=0.01)
    assert q3.ok is True


def test_rms_normalize_scales_to_target():
    """Verify audio is scaled to target RMS (0.073) and guarded against peak clipping."""
    rng = np.random.default_rng(42)
    quiet = (rng.normal(0, 0.01, 16000)).astype(np.float32)
    normed = rms_normalize(quiet, target_rms=0.073)
    measured_rms = float(np.sqrt(np.mean(np.square(normed))))
    assert abs(measured_rms - 0.073) < 1e-3
    assert np.max(np.abs(normed)) <= 0.99

    # Guard check: huge spike should clamp to 0.99 peak ceiling
    spiky = np.zeros(16000, dtype=np.float32)
    spiky[0] = 5.0
    normed_spiky = rms_normalize(spiky, target_rms=0.073)
    assert np.max(np.abs(normed_spiky)) <= 0.99

    # Near-silence should return without divide-by-zero
    silent = np.zeros(100, dtype=np.float32)
    assert np.all(rms_normalize(silent) == 0.0)


def test_pad_or_tile_window_repeats_short_audio():
    """Verify short audio is circularly tiled instead of padded with cold zeros."""
    short = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tiled = pad_or_tile_window(short, 7)
    assert np.array_equal(tiled, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])

    # Longer or exact audio returns unchanged
    exact = np.ones(10, dtype=np.float32)
    assert np.array_equal(pad_or_tile_window(exact, 10), exact)

    # Empty array returns zeroes of target length
    empty = np.array([], dtype=np.float32)
    assert len(pad_or_tile_window(empty, 16)) == 16


def test_decode_audio_bytes_multi_format():
    """Verify decode_audio_bytes reads in-memory WAV, FLAC, and OGG formats."""
    audio_orig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)

    for fmt in ["WAV", "OGG", "FLAC"]:
        buf = io.BytesIO()
        sf.write(buf, audio_orig, 16000, format=fmt)
        buf.seek(0)
        decoded, sr = decode_audio_bytes(buf.read())
        assert sr == 16000
        assert len(decoded) == 16000
        assert abs(float(np.max(np.abs(decoded))) - 1.0) < 0.05

    # Empty bytes should raise ValueError
    with pytest.raises(ValueError):
        decode_audio_bytes(b"")
