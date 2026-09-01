import numpy as np

from audio.resample import StreamingLinearResampler


def test_streaming_resampler_matches_single_stream_for_browser_frames():
    """48 kHz AudioWorklet frames must not accumulate per-frame rounding drift."""
    audio = np.linspace(-0.8, 0.8, 48_000, dtype=np.float32)

    one_shot = StreamingLinearResampler(48_000, 16_000).process(audio)

    chunked_resampler = StreamingLinearResampler(48_000, 16_000)
    chunked = [
        chunked_resampler.process(audio[start : start + 128])
        for start in range(0, audio.size, 128)
    ]
    chunked_output = np.concatenate(chunked)

    assert one_shot.size == 16_000
    assert chunked_output.size == one_shot.size
    np.testing.assert_allclose(chunked_output, one_shot, rtol=0, atol=1e-6)


def test_streaming_resampler_preserves_audio_when_rates_match():
    audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    output = StreamingLinearResampler(16_000, 16_000).process(audio)
    np.testing.assert_array_equal(output, audio)
