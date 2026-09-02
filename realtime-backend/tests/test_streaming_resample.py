"""Guards for the per-connection streaming resampler.

The browser's AudioWorklet delivers 128 samples per render quantum and the
frontend forwards each block as its own WebSocket message. Resampling those
blocks independently restarts the anti-aliasing FIR and re-rounds the output
length every 2.7 ms, which drifts the stream and corrupts the audio before any
expert sees it. These tests pin the three properties that make streaming safe:
continuity across frames, no length drift, and the same anti-aliasing the
one-shot path gets.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio.resample import StreamingPolyphaseResampler, to_target_rate

SR_IN = 48_000
SR_OUT = 16_000


def _speechlike(n: int, rate: int, seed: int = 0) -> np.ndarray:
    """Broadband-ish signal: a few harmonics plus noise, nothing above Nyquist."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / rate
    sig = (
        0.5 * np.sin(2 * np.pi * 220.0 * t)
        + 0.25 * np.sin(2 * np.pi * 660.0 * t + 0.4)
        + 0.12 * np.sin(2 * np.pi * 1500.0 * t + 1.1)
        + 0.03 * rng.standard_normal(n)
    )
    return (sig * 0.6).astype(np.float32)


def _stream(audio: np.ndarray, frame: int, resampler: StreamingPolyphaseResampler) -> np.ndarray:
    parts = [resampler.process(audio[i : i + frame]) for i in range(0, audio.size, frame)]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def test_browser_frames_match_one_shot_call():
    """128-sample frames must produce the same samples as one big call."""
    audio = _speechlike(SR_IN, SR_IN)

    one_shot = StreamingPolyphaseResampler(SR_IN, SR_OUT).process(audio)
    chunked = _stream(audio, 128, StreamingPolyphaseResampler(SR_IN, SR_OUT))

    assert chunked.size == one_shot.size
    np.testing.assert_allclose(chunked, one_shot, rtol=0, atol=1e-6)


@pytest.mark.parametrize("frame", [128, 160, 480, 1024, 4096])
def test_output_is_independent_of_frame_size(frame):
    """Whatever the client's block size, the resampled stream is identical."""
    audio = _speechlike(SR_IN, SR_IN, seed=1)

    reference = _stream(audio, 128, StreamingPolyphaseResampler(SR_IN, SR_OUT))
    actual = _stream(audio, frame, StreamingPolyphaseResampler(SR_IN, SR_OUT))

    assert actual.size == reference.size
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-6)


def test_no_sample_rate_drift_over_several_seconds():
    """The lag stays constant instead of growing with every frame.

    Per-frame ``resample_poly`` emitted 16125 samples per second of 48 kHz input
    (+0.78%), so a 4 s window kept sliding out of alignment. Here the shortfall
    is a fixed tail held in the filter state, identical at 1 s and 5 s.
    """
    resampler = StreamingPolyphaseResampler(SR_IN, SR_OUT)
    audio = _speechlike(5 * SR_IN, SR_IN, seed=2)

    emitted = []
    for second in range(5):
        chunk = audio[second * SR_IN : (second + 1) * SR_IN]
        emitted.append(_stream(chunk, 128, resampler).size)

    # The first second holds back the FIR tail; every later second is exact.
    pending = SR_OUT - emitted[0]
    assert 0 < pending < 64, f"unexpected pending tail {pending}"
    assert emitted[1:] == [SR_OUT] * 4, emitted
    assert sum(emitted) == 5 * SR_OUT - pending


def test_matches_whole_stream_polyphase_resample():
    """Streaming output equals the offline ``to_target_rate`` result.

    The offline path is what the experts were calibrated against, so the live
    path has to reproduce it rather than merely resemble it. Sample j lands at
    the same instant in both; streaming is only *shorter*, because the last few
    output samples still need input that has not arrived yet and stay in the
    filter state.
    """
    audio = _speechlike(2 * SR_IN, SR_IN, seed=3)

    offline, did = to_target_rate(audio, SR_IN, SR_OUT)
    assert did is True

    streamed = _stream(audio, 128, StreamingPolyphaseResampler(SR_IN, SR_OUT))
    pending = offline.size - streamed.size
    assert 0 < pending < 64, f"unexpected pending tail {pending}"

    np.testing.assert_allclose(streamed, offline[: streamed.size], rtol=0, atol=2e-6)


def test_suppresses_energy_above_target_nyquist():
    """An 11 kHz tone must not fold back into the 0-8 kHz band.

    Linear interpolation has no anti-aliasing filter, so out-of-band energy
    lands inside the band as artifacts no detector saw in training.
    """
    t = np.arange(SR_IN, dtype=np.float64) / SR_IN
    tone = (0.8 * np.sin(2 * np.pi * 11_000.0 * t)).astype(np.float32)

    streamed = _stream(tone, 128, StreamingPolyphaseResampler(SR_IN, SR_OUT))

    # Skip the filter ramp-up, then measure what survived in-band.
    settled = streamed[SR_OUT // 4 :]
    in_band_energy = float(np.sum(settled.astype(np.float64) ** 2))
    assert in_band_energy < 5.0, f"aliased energy {in_band_energy:.1f} leaked in-band"


def test_upsampling_stays_continuous():
    """8 kHz -> 16 kHz uses up=2; the interpolation phase must carry over too."""
    audio = _speechlike(8_000, 8_000, seed=4)

    one_shot = StreamingPolyphaseResampler(8_000, SR_OUT).process(audio)
    chunked = _stream(audio, 128, StreamingPolyphaseResampler(8_000, SR_OUT))

    assert chunked.size == one_shot.size
    np.testing.assert_allclose(chunked, one_shot, rtol=0, atol=1e-6)
    assert 0 < SR_OUT - chunked.size < 128


def test_passthrough_when_rates_match():
    resampler = StreamingPolyphaseResampler(SR_OUT, SR_OUT)
    assert resampler.did_resample is False
    audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    np.testing.assert_array_equal(resampler.process(audio), audio)


def test_reset_clears_filter_state():
    """After a stream gap, a reset resampler behaves like a brand new one."""
    audio = _speechlike(SR_IN // 2, SR_IN, seed=5)

    fresh = _stream(audio, 128, StreamingPolyphaseResampler(SR_IN, SR_OUT))

    reused = StreamingPolyphaseResampler(SR_IN, SR_OUT)
    _stream(_speechlike(SR_IN // 2, SR_IN, seed=6), 128, reused)
    reused.reset()
    after_reset = _stream(audio, 128, reused)

    np.testing.assert_allclose(after_reset, fresh, rtol=0, atol=1e-6)


def test_rejects_invalid_rates_and_non_finite_audio():
    with pytest.raises(ValueError):
        StreamingPolyphaseResampler(0, SR_OUT)
    with pytest.raises(ValueError):
        StreamingPolyphaseResampler(SR_IN, -1)

    resampler = StreamingPolyphaseResampler(SR_IN, SR_OUT)
    with pytest.raises(ValueError):
        resampler.process(np.array([0.1, np.nan, 0.3], dtype=np.float32))


def test_empty_frame_is_a_no_op():
    resampler = StreamingPolyphaseResampler(SR_IN, SR_OUT)
    assert resampler.process(np.zeros(0, dtype=np.float32)).size == 0
