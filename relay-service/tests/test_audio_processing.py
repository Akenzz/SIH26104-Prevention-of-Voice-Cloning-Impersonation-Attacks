"""Unit tests for audio PCM conversion, crossfade processor, soft limiter, and bounded queue."""

import asyncio
import numpy as np
import pytest

from audio.pcm import float32_to_pcm, pcm_to_float32, to_mono
from audio.crossfade import ChunkCrossfadeProcessor, raised_cosine_weights, soft_limit
from audio.buffer import DropOldestQueue, StreamBuffer


def test_pcm_s16le_roundtrip(sample_sine_wave: np.ndarray):
    pcm = float32_to_pcm(sample_sine_wave, encoding="pcm_s16le")
    assert len(pcm) == sample_sine_wave.size * 2

    recovered = pcm_to_float32(pcm, encoding="pcm_s16le")
    assert recovered.shape == sample_sine_wave.shape
    # int16 precision max difference is <= 1/32768 ~ 0.0001
    assert np.max(np.abs(sample_sine_wave - recovered)) < 1e-4


def test_pcm_f32le_roundtrip(sample_sine_wave: np.ndarray):
    pcm = float32_to_pcm(sample_sine_wave, encoding="pcm_f32le")
    assert len(pcm) == sample_sine_wave.size * 4

    recovered = pcm_to_float32(pcm, encoding="pcm_f32le")
    assert np.allclose(sample_sine_wave, recovered)


def test_pcm_sanitization():
    # Odd byte count for int16 (truncate trailing odd byte)
    raw_odd = b"\x00\x10\x00"
    decoded = pcm_to_float32(raw_odd, encoding="pcm_s16le")
    assert decoded.size == 1

    # Empty payload
    assert pcm_to_float32(b"").size == 0
    assert float32_to_pcm(np.empty(0)) == b""


def test_multi_channel_to_mono():
    # 2-channel interleaved
    stereo = np.array([0.5, 0.5, -0.2, 0.8, 0.4, 0.6], dtype=np.float32)
    mono = to_mono(stereo.reshape(-1, 2), channels=2)
    assert mono.shape == (3,)
    assert np.allclose(mono, [0.5, 0.3, 0.5])


def test_soft_limit_guarantee():
    # Signal with extreme peaks well exceeding 1.0
    large_signal = np.array([-2.5, -1.0, 0.0, 0.5, 1.5, 3.0], dtype=np.float32)
    limited = soft_limit(large_signal, max_abs=0.92)

    assert float(np.max(np.abs(limited))) <= 0.92001
    # Check that small values around 0 are linear
    assert abs(limited[2] - 0.0) < 1e-5


def test_raised_cosine_weights():
    length = 400
    fade_in, fade_out = raised_cosine_weights(length)
    assert len(fade_in) == length
    assert len(fade_out) == length
    # Check sum is 1.0 everywhere
    sum_weights = fade_in + fade_out
    assert np.allclose(sum_weights, 1.0, atol=1e-5)
    assert fade_in[0] < fade_in[-1]
    assert fade_out[0] > fade_out[-1]


def test_chunk_crossfade_processor(sample_sine_wave: np.ndarray):
    processor = ChunkCrossfadeProcessor(
        sample_rate=16000,
        chunk_ms=200,      # 3200 samples
        context_ms=150,    # 2400 samples
        crossfade_ms=25,   # 400 samples
        soft_limit_max=0.92,
    )

    # Dummy converter function (e.g. invert phase or pass through)
    def dummy_converter(x: np.ndarray) -> np.ndarray:
        return x * 0.9

    chunks = processor.process(sample_sine_wave, dummy_converter)
    assert len(chunks) >= 3  # 16000 / 3200 = 5 chunks

    flushed = processor.flush(dummy_converter)
    total_out = sum(len(c) for c in chunks) + sum(len(c) for c in flushed)
    assert total_out > 0

    # Ensure every output chunk respects soft_limit <= 0.92
    for c in chunks + flushed:
        assert float(np.max(np.abs(c))) <= 0.92001


@pytest.mark.asyncio
async def test_drop_oldest_queue():
    q: DropOldestQueue[int] = DropOldestQueue(maxsize=3)
    assert q.empty()

    # Put 3 items
    assert not q.put_nowait(1)
    assert not q.put_nowait(2)
    assert not q.put_nowait(3)
    assert q.full()
    assert q.dropped_count == 0

    # Put 4th item -> item 1 must be dropped!
    dropped = q.put_nowait(4)
    assert dropped is True
    assert q.dropped_count == 1
    assert q.full()

    # Retrieved items should be 2, 3, 4
    val1 = await q.get()
    val2 = await q.get()
    val3 = await q.get()
    assert [val1, val2, val3] == [2, 3, 4]
    assert q.empty()


def test_stream_buffer():
    buf = StreamBuffer(chunk_samples=500)
    samples = np.ones(1200, dtype=np.float32)

    chunks = buf.push(samples)
    assert len(chunks) == 2
    assert chunks[0].size == 500
    assert chunks[1].size == 500

    flushed = buf.flush()
    assert len(flushed) == 1
    assert flushed[0].size == 200
