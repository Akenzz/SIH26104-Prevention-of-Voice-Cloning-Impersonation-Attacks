"""
Prosody / behavioral feature extraction for voice-spoof detection.
======================================================================
Computes a small set of INTERPRETABLE, human-readable signals from a single
audio window that help distinguish natural human speech from TTS / cloned
speech. These are NOT meant to beat the learned experts (WavLM, LFCC-LCNN) on
raw EER — they exist so a downstream LLM explainer can cite concrete, named
evidence ("pitch variation is unusually flat") instead of a bare score.

Contract: `extract_prosody_features(audio_window, sr=16000) -> dict[str, float]`
returns raw numeric values only (no classification). The ordered `FEATURE_NAMES`
list is the single source of truth shared by the trainer, the scorer, and the
describe module.

Micro-variation (jitter/shimmer) uses Praat via `parselmouth` when available and
falls back to a librosa-only proxy otherwise; `jitter_source` records which path
ran (1.0 = praat, 0.0 = librosa fallback).

All features are deterministic and NaN-safe: an unvoiced / silent window yields
sentinel defaults rather than NaN so the classifier never sees a NaN.
"""
from __future__ import annotations

import warnings

import numpy as np

try:
    import librosa
except Exception as e:  # pragma: no cover - librosa is a hard dep
    raise ImportError("prosody_features requires librosa") from e

# Praat jitter/shimmer is optional; degrade gracefully to a librosa proxy.
try:
    import parselmouth
    from parselmouth.praat import call as _praat_call
    _HAVE_PRAAT = True
except Exception:  # pragma: no cover - environment dependent
    parselmouth = None
    _HAVE_PRAAT = False

SR = 16000

# Ordered feature vector. ORDER IS A CONTRACT — the trained artifact stores
# coefficients in this order. Append new features at the END only.
FEATURE_NAMES = [
    "f0_std",          # pitch standard deviation (Hz) over voiced frames
    "f0_range",        # p95-p5 pitch spread (Hz)
    "f0_delta_var",    # variance of frame-to-frame F0 deltas (naturalness of contour)
    "voiced_fraction", # fraction of frames that are voiced
    "pause_mean",      # mean intra-clip silence duration (s)
    "pause_var",       # variance of silence durations (regular TTS timing -> low)
    "pause_count",     # number of distinct silences
    "rate_mean",       # mean energy-peak (syllable-ish) rate (peaks/s) across sub-windows
    "rate_var",        # variance of that rate across sub-windows
    "jitter_local",    # cycle-to-cycle F0 period variation (Praat) or F0-period proxy
    "shimmer_local",   # cycle-to-cycle amplitude variation (Praat) or amplitude proxy
    "spectral_flatness",  # mean spectral flatness (synthesis often over-smooths)
    "jitter_source",   # 1.0 if Praat produced jitter/shimmer, 0.0 if librosa fallback
]

# Sentinel defaults for degenerate (silent / fully-unvoiced) windows. Chosen so
# the values look "human-normal-ish" and don't fire spurious findings.
_DEFAULTS = {
    "f0_std": 0.0,
    "f0_range": 0.0,
    "f0_delta_var": 0.0,
    "voiced_fraction": 0.0,
    "pause_mean": 0.0,
    "pause_var": 0.0,
    "pause_count": 0.0,
    "rate_mean": 0.0,
    "rate_var": 0.0,
    "jitter_local": 0.0,
    "shimmer_local": 0.0,
    "spectral_flatness": 0.0,
    "jitter_source": 0.0,
}

_FRAME = 512      # ~32 ms @ 16 kHz
_HOP = 160        # 10 ms hop
_FMIN = 65.0      # librosa.pyin fmin (Hz)
_FMAX = 400.0     # librosa.pyin fmax (Hz)


def _safe(x: float) -> float:
    """Coerce to a finite python float."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v


def _pitch_features(y: np.ndarray, sr: int) -> dict:
    """F0 contour stats via librosa.pyin. Returns f0_* + voiced_fraction."""
    out = {"f0_std": 0.0, "f0_range": 0.0, "f0_delta_var": 0.0, "voiced_fraction": 0.0}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=_FMIN, fmax=_FMAX, sr=sr,
                frame_length=_FRAME, hop_length=_HOP,
            )
    except Exception:
        return out
    if f0 is None or len(f0) == 0:
        return out
    voiced = f0[np.isfinite(f0)]
    out["voiced_fraction"] = _safe(len(voiced) / len(f0))
    if len(voiced) >= 2:
        out["f0_std"] = _safe(np.std(voiced))
        out["f0_range"] = _safe(np.percentile(voiced, 95) - np.percentile(voiced, 5))
        # frame-to-frame delta over the voiced contour: unnatural TTS pitch
        # either jumps (high) or is glued flat (near zero).
        deltas = np.diff(voiced)
        out["f0_delta_var"] = _safe(np.var(deltas))
    return out


def _pause_features(y: np.ndarray, sr: int) -> dict:
    """Energy-VAD silence durations. Returns pause_mean / pause_var / pause_count."""
    out = {"pause_mean": 0.0, "pause_var": 0.0, "pause_count": 0.0}
    rms = librosa.feature.rms(y=y, frame_length=_FRAME, hop_length=_HOP)[0]
    if rms.size == 0:
        return out
    # Adaptive gate: 15% of the loudest frame (~ -16 dB). Frames below = silence.
    thr = max(float(rms.max()) * 0.15, 1e-4)
    is_speech = rms > thr
    sec_per_frame = _HOP / sr

    # Collect run-lengths of silence that are strictly interior (ignore leading /
    # trailing silence, which is an artifact of windowing, not speaker behavior).
    pauses = []
    run = 0
    started = False
    for flag in is_speech:
        if flag:
            if started and run > 0:
                pauses.append(run * sec_per_frame)
            run = 0
            started = True
        else:
            if started:
                run += 1
    # a trailing run is dropped on purpose (not an interior pause)

    # Only count pauses longer than ~60 ms (below that it's coarticulation, not a pause).
    pauses = [p for p in pauses if p >= 0.06]
    if pauses:
        out["pause_mean"] = _safe(np.mean(pauses))
        out["pause_var"] = _safe(np.var(pauses))
        out["pause_count"] = _safe(len(pauses))
    return out


def _rate_features(y: np.ndarray, sr: int) -> dict:
    """Energy-peak (syllable-ish) rate and its variance across 1 s sub-windows."""
    out = {"rate_mean": 0.0, "rate_var": 0.0}
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
    except Exception:
        return out
    if onset_env.size == 0:
        return out
    frames_per_sec = sr / _HOP
    win = max(int(frames_per_sec), 1)  # ~1 s chunks
    rates = []
    for start in range(0, len(onset_env), win):
        chunk = onset_env[start:start + win]
        if chunk.size < win // 2:
            continue
        thr = float(chunk.mean() + chunk.std())
        peaks = librosa.util.peak_pick(
            chunk, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.0, wait=3
        )
        peaks = [p for p in peaks if chunk[p] >= thr]
        dur = chunk.size / frames_per_sec
        rates.append(len(peaks) / dur if dur > 0 else 0.0)
    if rates:
        out["rate_mean"] = _safe(np.mean(rates))
        out["rate_var"] = _safe(np.var(rates))
    return out


def _micro_features_praat(y: np.ndarray, sr: int) -> dict | None:
    """True Praat local jitter/shimmer. Returns None if Praat can't produce them."""
    if not _HAVE_PRAAT:
        return None
    try:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
        point_process = _praat_call(snd, "To PointProcess (periodic, cc)", _FMIN, _FMAX)
        n_points = _praat_call(point_process, "Get number of points")
        if n_points < 3:
            return None
        jitter = _praat_call(point_process, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3)
        shimmer = _praat_call(
            [snd, point_process], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6
        )
        j, s = _safe(jitter), _safe(shimmer)
        if j == 0.0 and s == 0.0:
            return None
        return {"jitter_local": j, "shimmer_local": s, "jitter_source": 1.0}
    except Exception:
        return None


def _micro_features_librosa(y: np.ndarray, sr: int, pitch: dict) -> dict:
    """librosa-only micro-variation proxy: F0-period jitter + RMS shimmer proxy."""
    # jitter proxy: relative frame-to-frame F0 variation, expressed like Praat's ratio.
    jitter = 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0, _, _ = librosa.pyin(
                y, fmin=_FMIN, fmax=_FMAX, sr=sr, frame_length=_FRAME, hop_length=_HOP
            )
        if f0 is not None:
            voiced = f0[np.isfinite(f0)]
            if len(voiced) >= 3:
                periods = 1.0 / voiced
                jitter = _safe(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9))
    except Exception:
        jitter = 0.0
    # shimmer proxy: relative cycle-to-cycle RMS-envelope variation.
    shimmer = 0.0
    try:
        rms = librosa.feature.rms(y=y, frame_length=_FRAME, hop_length=_HOP)[0]
        rms = rms[rms > (rms.max() * 0.15 if rms.size else 0)]
        if rms.size >= 3:
            shimmer = _safe(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9))
    except Exception:
        shimmer = 0.0
    return {"jitter_local": jitter, "shimmer_local": shimmer, "jitter_source": 0.0}


def extract_prosody_features(audio_window: np.ndarray, sr: int = SR) -> dict:
    """Extract interpretable prosody/behavioral features from one audio window.

    Args:
        audio_window: 1-D waveform (any length), mono. Assumed sampled at ``sr``.
        sr: sample rate of ``audio_window`` (default 16 kHz — the project window).

    Returns:
        dict keyed by every name in ``FEATURE_NAMES`` -> finite float. Raw values
        only; no classification. Silent/unvoiced windows return sentinel defaults.
    """
    y = np.asarray(audio_window, dtype=np.float32).reshape(-1)
    feats = dict(_DEFAULTS)
    if y.size < int(0.1 * sr) or not np.any(np.abs(y) > 1e-5):
        return {k: feats[k] for k in FEATURE_NAMES}

    # normalize peak to stabilize energy-based measures across gain differences
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    pitch = _pitch_features(y, sr)
    feats.update(pitch)
    feats.update(_pause_features(y, sr))
    feats.update(_rate_features(y, sr))

    micro = _micro_features_praat(y, sr)
    if micro is None:
        micro = _micro_features_librosa(y, sr, pitch)
    feats.update(micro)

    try:
        flat = librosa.feature.spectral_flatness(y=y, n_fft=_FRAME, hop_length=_HOP)[0]
        feats["spectral_flatness"] = _safe(np.mean(flat)) if flat.size else 0.0
    except Exception:
        feats["spectral_flatness"] = 0.0

    return {k: _safe(feats.get(k, _DEFAULTS[k])) for k in FEATURE_NAMES}


def features_to_vector(feats: dict) -> np.ndarray:
    """Ordered feature vector matching FEATURE_NAMES (float32)."""
    return np.array([_safe(feats.get(k, _DEFAULTS[k])) for k in FEATURE_NAMES], dtype=np.float32)
