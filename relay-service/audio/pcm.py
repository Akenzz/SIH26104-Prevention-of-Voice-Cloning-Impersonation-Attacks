"""PCM conversion and validation utilities."""

from __future__ import annotations

import numpy as np


def pcm_to_float32(
    payload: bytes,
    encoding: str = "pcm_s16le",
    channels: int = 1,
) -> np.ndarray:
    """Decode raw PCM byte payload to float32 in [-1.0, 1.0].
    
    Supports:
      - pcm_s16le / s16le / int16
      - pcm_f32le / f32le / float32
    """
    if not payload:
        return np.empty(0, dtype=np.float32)

    enc = encoding.lower()
    if enc in {"pcm_s16le", "s16le", "int16"}:
        if len(payload) % 2 != 0:
            # Drop truncated trailing byte
            payload = payload[: len(payload) - (len(payload) % 2)]
            if not payload:
                return np.empty(0, dtype=np.float32)
        raw = np.frombuffer(payload, dtype="<i2")
        samples = raw.astype(np.float32) / 32768.0
    elif enc in {"pcm_f32le", "f32le", "float32"}:
        if len(payload) % 4 != 0:
            payload = payload[: len(payload) - (len(payload) % 4)]
            if not payload:
                return np.empty(0, dtype=np.float32)
        samples = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
    else:
        raise ValueError(f"Unsupported PCM encoding: {encoding}")

    if channels > 1:
        # Multi-channel interleaved: average to mono
        rem = samples.size % channels
        if rem != 0:
            samples = samples[: samples.size - rem]
        samples = samples.reshape(-1, channels).mean(axis=1)

    # Sanitize NaN/inf
    if not np.isfinite(samples).all():
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

    return samples.astype(np.float32)


def float32_to_pcm(
    samples: np.ndarray,
    encoding: str = "pcm_s16le",
) -> bytes:
    """Encode float32 samples to raw PCM bytes."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return b""

    enc = encoding.lower()
    if enc in {"pcm_s16le", "s16le", "int16"}:
        # Clip to [-1.0, 1.0] before int16 scaling
        clipped = np.clip(audio, -1.0, 1.0)
        int16_samples = (clipped * 32767.0).astype("<i2")
        return int16_samples.tobytes()
    elif enc in {"pcm_f32le", "f32le", "float32"}:
        return audio.astype("<f4").tobytes()
    else:
        raise ValueError(f"Unsupported PCM encoding: {encoding}")


def to_mono(samples: np.ndarray, channels: int = 1) -> np.ndarray:
    """Ensure audio is 1D mono float32."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if channels > 1 and arr.shape[-1] == channels:
        return arr.mean(axis=-1)
    return arr.reshape(-1)
