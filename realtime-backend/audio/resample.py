from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger("realtime_backend.audio")


class StreamingLinearResampler:
    """Stateful mono resampler for PCM frames from a single connection.

    ``to_target_rate`` is suitable for a complete file.  Applying it to every
    128-sample browser frame independently, however, rounds each frame's output
    length and resets its interpolation phase.  That creates sample-rate drift
    and discontinuities before the detector sees the audio.  This class keeps
    the fractional source position across calls, so chunked and continuous
    streams produce the same samples (apart from the unavoidable final sample
    held until the next frame).

    The interpolation method is intentionally kept compatible with the legacy
    helper.  It is an integrity fix, not a claim that linear interpolation is a
    replacement for a production polyphase resampler.
    """

    def __init__(self, source_rate: int, target_rate: int):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError(f"invalid sample rates source={source_rate} target={target_rate}")
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self._step = self.source_rate / float(self.target_rate)
        self.reset()

    @property
    def did_resample(self) -> bool:
        return self.source_rate != self.target_rate

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0
        self._input_samples_seen = 0
        self._next_position = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not self.did_resample or audio.size == 0:
            return audio
        if not np.isfinite(audio).all():
            raise ValueError("audio contains non-finite samples")

        if self._buffer.size == 0:
            self._buffer_start = self._input_samples_seen
        self._buffer = np.concatenate((self._buffer, audio))
        self._input_samples_seen += int(audio.size)

        last_available = self._buffer_start + self._buffer.size - 1
        output: list[float] = []
        # We need a sample on each side of a non-integer position.  Holding one
        # frame-edge sample avoids seams without synthesising future audio.
        while self._next_position < last_available:
            left = int(math.floor(self._next_position))
            fraction = self._next_position - left
            offset = left - self._buffer_start
            y0 = float(self._buffer[offset])
            y1 = float(self._buffer[offset + 1])
            output.append(y0 + fraction * (y1 - y0))
            self._next_position += self._step

        # Retain the sample immediately before the next fractional position; it
        # is needed to interpolate against the next incoming frame.
        keep_from = max(self._buffer_start, int(math.floor(self._next_position)) - 1)
        discard = keep_from - self._buffer_start
        if discard > 0:
            self._buffer = self._buffer[discard:]
            self._buffer_start = keep_from

        return np.asarray(output, dtype=np.float32)


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 0:
        raise ValueError(f"channels must be >= 1, got {channels}")
    if samples.ndim == 1:
        if channels == 1:
            return samples.astype(np.float32, copy=False)
        if samples.size % channels != 0:
            raise ValueError(
                f"interleaved PCM length {samples.size} is not divisible by channels={channels}"
            )
        framed = samples.reshape(-1, channels)
        return framed.mean(axis=1).astype(np.float32)
    if samples.ndim == 2:
        if samples.shape[1] == 1:
            return samples[:, 0].astype(np.float32, copy=False)
        return samples.mean(axis=1).astype(np.float32)
    raise ValueError(f"audio must be 1D or 2D, got shape {samples.shape}")


def to_target_rate(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> tuple[np.ndarray, bool]:
    """Linear-resample mono float32 audio. Returns (audio, did_resample)."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(f"invalid sample rates source={source_rate} target={target_rate}")
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return audio, False
    if audio.size == 0:
        return audio, True
    duration = audio.size / float(source_rate)
    n_out = max(1, int(round(duration * target_rate)))
    src_t = np.linspace(0.0, duration, num=audio.size, endpoint=False, dtype=np.float64)
    dst_t = np.linspace(0.0, duration, num=n_out, endpoint=False, dtype=np.float64)
    resampled = np.interp(dst_t, src_t, audio).astype(np.float32)
    logger.debug(
        "Resampled incoming audio %d Hz -> %d Hz (%d samples -> %d samples)",
        source_rate,
        target_rate,
        audio.size,
        resampled.size,
    )
    return resampled, True
