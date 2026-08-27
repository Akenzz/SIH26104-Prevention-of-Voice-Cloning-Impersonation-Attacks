from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("realtime_backend.audio")


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
