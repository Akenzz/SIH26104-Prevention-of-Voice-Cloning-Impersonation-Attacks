"""
audio_utils.py — Audio standardization helpers.

Operations applied (in order):
  1. Decode audio (any format librosa supports: mp3, wav, flac, ogg…)
  2. Convert to mono (average channels, not drop)
  3. Resample to TARGET_SR (16 000 Hz)
  4. Trim leading/trailing silence (energy threshold)
  5. Loudness-normalize to TARGET_LUFS (-23 LUFS, ITU-R BS.1770-4)
  6. Write 16-bit PCM WAV to output path

Everything is wrapped so a single corrupt file raises an exception that the
caller can catch and log without crashing the whole run.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

# Suppress annoying pyloudnorm clipping warnings
warnings.filterwarnings("ignore", message="Possible clipped samples in output.")

logger = logging.getLogger("pipeline.audio_utils")

TARGET_SR    = 16_000
TARGET_LUFS  = -23.0
SILENCE_DB   = -60.0   # dB threshold for leading/trailing silence trim

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def standardize(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    resample: bool = True,
    skip_if_exists: bool = True,
) -> float:
    """
    Full standardization pipeline for one audio file.

    Returns the duration (seconds) of the written file.
    Raises on any error so callers can log and skip.
    """
    dst_path = Path(dst_path)
    if skip_if_exists and dst_path.exists():
        info = sf.info(str(dst_path))
        return info.duration

    audio, sr = _load_audio(src_path)

    if resample and sr != TARGET_SR:
        audio = _resample(audio, sr, TARGET_SR)
        sr = TARGET_SR

    audio = _trim_silence(audio, sr)
    audio = _loudness_normalize(audio, sr)

    if len(audio) < int(0.1 * sr):
        raise ValueError(f"After processing, audio is < 100ms — skipping: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), audio, sr, subtype="PCM_16")
    return len(audio) / sr


def get_duration(path: str | Path) -> float:
    """Return duration of an audio file in seconds."""
    return sf.info(str(path)).duration


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load audio as float32 mono. Handles MP3 via librosa, all else via soundfile.
    Returns (audio_1d_float32, sample_rate).
    """
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext == ".mp3":
        # librosa handles MP3 via audioread / ffmpeg under the hood
        import librosa  # type: ignore
        audio, sr = librosa.load(path, sr=None, mono=True, dtype=np.float32)
        return audio, int(sr)

    # soundfile handles wav / flac / ogg / opus etc.
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # average channels → mono
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    return data, int(sr)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import librosa  # type: ignore
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq")


def _trim_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Remove leading and trailing silence using a simple energy threshold.
    SILENCE_DB defines the threshold relative to the peak energy in 10ms frames.
    """
    frame_len = int(0.01 * sr)   # 10 ms frames
    hop_len   = frame_len // 2

    # RMS per frame
    n_frames = (len(audio) - frame_len) // hop_len + 1
    if n_frames <= 0:
        return audio

    rms = np.array([
        np.sqrt(np.mean(audio[i * hop_len: i * hop_len + frame_len] ** 2))
        for i in range(n_frames)
    ])

    threshold = 10 ** (SILENCE_DB / 20)
    active = rms > threshold

    if not active.any():
        return audio  # all silent — return as-is, caller can drop it

    first = int(np.argmax(active)) * hop_len
    last  = int(len(active) - 1 - np.argmax(active[::-1])) * hop_len + frame_len
    return audio[first:last]


def _loudness_normalize(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Loudness-normalize to TARGET_LUFS using ITU-R BS.1770-4 (pyloudnorm).
    Falls back to peak normalization if the audio is too short for the meter.
    """
    try:
        import pyloudnorm as pyln  # type: ignore
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if np.isinf(loudness) or np.isnan(loudness):
            raise ValueError("Integrated loudness is inf/nan")
        normalized = pyln.normalize.loudness(audio, loudness, TARGET_LUFS)
        # Guard against extreme clipping after normalization
        peak = np.abs(normalized).max()
        if peak > 1.0:
            normalized = normalized / peak * 0.99
        return normalized
    except Exception as e:
        logger.debug("pyloudnorm failed (%s), falling back to peak normalize", e)
        peak = np.abs(audio).max()
        if peak < 1e-9:
            return audio
        return audio / peak * 0.99
