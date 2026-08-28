from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    reason: str | None = None

    @property
    def unavailable(self) -> bool:
        return not self.ok


def decode_pcm(
    payload: bytes,
    encoding: str,
    channels: int,
) -> tuple[np.ndarray | None, QualityResult]:
    """Decode a binary audio frame to float32 in [-1, 1], interleaved if multi-channel."""
    if not payload:
        return None, QualityResult(False, "decode_failure")
    encoding = encoding.lower()
    try:
        if encoding in {"pcm_s16le", "s16le", "int16"}:
            if len(payload) % 2 != 0:
                return None, QualityResult(False, "decode_failure")
            raw = np.frombuffer(payload, dtype="<i2")
            samples = raw.astype(np.float32) / 32768.0
        elif encoding in {"pcm_f32le", "f32le", "float32"}:
            if len(payload) % 4 != 0:
                return None, QualityResult(False, "decode_failure")
            samples = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
        else:
            return None, QualityResult(False, "decode_failure")
    except (ValueError, BufferError):
        return None, QualityResult(False, "decode_failure")

    if channels > 1 and samples.size % channels != 0:
        return None, QualityResult(False, "decode_failure")
    if not np.isfinite(samples).all():
        return None, QualityResult(False, "decode_failure")
    return samples, QualityResult(True, None)


def assess_window(
    samples: np.ndarray,
    *,
    silence_rms: float,
    clip_abs: float,
    clip_fraction: float,
) -> QualityResult:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return QualityResult(False, "decode_failure")
    if not np.isfinite(audio).all():
        return QualityResult(False, "decode_failure")
    peak = float(np.max(np.abs(audio)))
    if peak >= clip_abs:
        clipped = float(np.mean(np.abs(audio) >= clip_abs))
        if clipped >= clip_fraction:
            return QualityResult(False, "clipped")
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < silence_rms:
        return QualityResult(False, "silence")
    return QualityResult(True, None)
