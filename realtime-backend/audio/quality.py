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
    active_rms: float = 0.003,
    min_active_audio_sec: float = 0.0,
    activity_frame_ms: int = 30,
    sample_rate: int = 16000,
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

    # A short utterance padded to a four-second model window can have enough
    # global energy to pass the silence check while containing too little speech
    # for a meaningful authenticity score. This lightweight energy gate is not
    # a linguistic VAD; it fails closed until enough active audio is present.
    if min_active_audio_sec > 0.0:
        if active_rms <= 0.0 or activity_frame_ms <= 0 or sample_rate <= 0:
            raise ValueError("activity gate requires positive RMS, frame, and sample-rate values")
        frame_samples = max(1, round(sample_rate * activity_frame_ms / 1000.0))
        usable = (audio.size // frame_samples) * frame_samples
        if usable == 0:
            return QualityResult(False, "insufficient_active_audio")
        frames = audio[:usable].reshape(-1, frame_samples)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
        active_sec = float(np.sum(frame_rms >= active_rms) * frame_samples / sample_rate)
        if active_sec < min_active_audio_sec:
            return QualityResult(False, "insufficient_active_audio")
    return QualityResult(True, None)
