"""Mock and fallback voice converters using DSP techniques.

Provides:
  - PitchFormantFallbackConverter: Pitch shifting, formant warping, and neural vocoder artifact injection.
  - MockRVCConverter: Preset speaker profiles and spoof triggering without requiring GPU or trained weights.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import scipy.signal as signal

from .base import VoiceConverter


def pitch_shift_vocoder(
    x: np.ndarray,
    semitones: float,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop: int = 128,
) -> np.ndarray:
    """Pitch shift a 1D audio signal using a phase vocoder.
    
    Preserves audio duration exactly.
    """
    if x.size < n_fft or abs(semitones) < 0.05:
        return x

    ratio = float(2.0 ** (semitones / 12.0))
    time_stretch_factor = 1.0 / ratio

    # Compute STFT
    _, _, Zxx = signal.stft(
        x,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        padded=True,
    )
    # Zxx has shape (freq_bins, time_frames)
    num_bins, num_frames = Zxx.shape
    if num_frames < 2:
        return x

    # Phase vocoder time-stretch
    phase_advance = np.linspace(0, np.pi * hop, num_bins, endpoint=False)
    time_steps = np.arange(0, num_frames - 1, time_stretch_factor)
    output_frames = len(time_steps)

    new_Zxx = np.zeros((num_bins, output_frames), dtype=np.complex64)
    phi_acc = np.angle(Zxx[:, 0])
    new_Zxx[:, 0] = Zxx[:, 0]

    for i, t_val in enumerate(time_steps[1:], 1):
        t_low = int(np.floor(t_val))
        t_high = min(t_low + 1, num_frames - 1)
        alpha = float(t_val - t_low)

        mag1 = np.abs(Zxx[:, t_low])
        mag2 = np.abs(Zxx[:, t_high])
        mag = (1.0 - alpha) * mag1 + alpha * mag2

        # Phase difference
        phase_diff = np.angle(Zxx[:, t_high]) - np.angle(Zxx[:, t_low])
        # Unwrap phase difference around expected advance
        dphase = phase_diff - phase_advance
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        freq = phase_advance + dphase

        phi_acc += freq * time_stretch_factor
        new_Zxx[:, i] = mag * np.exp(1j * phi_acc)

    # Inverse STFT
    _, x_stretched = signal.istft(
        new_Zxx,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop,
    )

    # Resample stretched audio to match original length exactly
    target_length = x.size
    if x_stretched.size == 0:
        return x
    if x_stretched.size != target_length:
        x_shifted = signal.resample(x_stretched, target_length)
    else:
        x_shifted = x_stretched

    return x_shifted.astype(np.float32)


def apply_formant_filter(
    x: np.ndarray,
    formant_ratio: float = 1.15,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Apply spectral formant alteration via parametric peaking/shelving filters."""
    if abs(formant_ratio - 1.0) < 0.02 or x.size < 64:
        return x

    # Center frequencies for vocal formants (Hz)
    f_center = 2800.0 * formant_ratio
    f_center = min(max(f_center, 800.0), 6500.0)
    q = 2.5
    gain_db = 6.0 if formant_ratio > 1.0 else -4.0

    # Digital peaking biquad filter
    w0 = 2.0 * np.pi * f_center / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    a_val = 10.0 ** (gain_db / 40.0)

    b0 = 1.0 + alpha * a_val
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * a_val
    a0 = 1.0 + alpha / a_val
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / a_val

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)

    filtered = signal.lfilter(b, a, x)
    return filtered.astype(np.float32)


def inject_neural_vocoder_artifacts(
    x: np.ndarray,
    intensity: float = 0.12,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Inject subtle neural vocoder phase/harmonic artifacts characteristic of HiFi-GAN.
    
    This produces cues in spectral envelope and phase consistency that trigger
    LFCC, TakHemlata SSL, and WavLM anti-spoofing classifiers.
    """
    if intensity <= 0.0 or x.size == 0:
        return x

    # 1. Subtle soft harmonic saturation (replicates neural vocoder tanh activations)
    harmonic = np.tanh(1.8 * x) - x

    # 2. High-frequency comb coloration (replicates convolutional upsampler periodic footprint)
    delay_samples = max(2, int(sample_rate / 3500.0))
    comb = np.zeros_like(x)
    comb[delay_samples:] = x[:-delay_samples]

    blended = x + intensity * (0.6 * harmonic + 0.4 * comb)
    return blended.astype(np.float32)


class PitchFormantFallbackConverter(VoiceConverter):
    """DSP fallback converter implementing pitch modulation, formant shifting,
    and simulated spoof artifact injection.
    """

    def __init__(
        self,
        pitch_shift_semitones: float = 4.0,
        formant_shift_ratio: float = 1.15,
        artifact_intensity: float = 0.12,
        name: str = "PitchFormantFallbackConverter",
    ):
        self._pitch_shift = pitch_shift_semitones
        self._formant_ratio = formant_shift_ratio
        self._artifact_intensity = artifact_intensity
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_ready(self) -> bool:
        return True

    def convert(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        if audio.size == 0:
            return audio

        # 1. Pitch shift
        out = pitch_shift_vocoder(audio, self._pitch_shift, sample_rate=sample_rate)

        # 2. Formant shift
        out = apply_formant_filter(out, self._formant_ratio, sample_rate=sample_rate)

        # 3. Neural vocoder artifact injection (triggers spoof classifiers)
        out = inject_neural_vocoder_artifacts(out, self._artifact_intensity, sample_rate=sample_rate)

        # Match input RMS to preserve volume
        in_rms = float(np.sqrt(np.mean(np.square(audio))))
        out_rms = float(np.sqrt(np.mean(np.square(out))))
        if out_rms > 1e-6 and in_rms > 1e-6:
            out = out * (in_rms / out_rms)

        return out.astype(np.float32)

    def set_pitch(self, semitones: float) -> None:
        self._pitch_shift = float(semitones)

    def set_formant(self, ratio: float) -> None:
        self._formant_ratio = float(ratio)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ready": self.is_ready,
            "pitch_shift_semitones": self._pitch_shift,
            "formant_shift_ratio": self._formant_ratio,
            "artifact_intensity": self._artifact_intensity,
        }


class MockRVCConverter(PitchFormantFallbackConverter):
    """Mock RVC Converter with speaker profiles and spoof score trigger simulation."""

    PROFILES = {
        "clone_female": {"pitch": 4.0, "formant": 1.18, "intensity": 0.14},
        "clone_male": {"pitch": -3.5, "formant": 0.88, "intensity": 0.14},
        "clone_deep": {"pitch": -6.0, "formant": 0.80, "intensity": 0.16},
        "clone_child": {"pitch": 6.5, "formant": 1.25, "intensity": 0.15},
        "robotic": {"pitch": 2.0, "formant": 1.10, "intensity": 0.25},
    }

    def __init__(
        self,
        default_profile: str = "clone_female",
        pitch_shift_semitones: float | None = None,
    ):
        profile = self.PROFILES.get(default_profile, self.PROFILES["clone_female"])
        pitch = pitch_shift_semitones if pitch_shift_semitones is not None else profile["pitch"]
        super().__init__(
            pitch_shift_semitones=pitch,
            formant_shift_ratio=profile["formant"],
            artifact_intensity=profile["intensity"],
            name="MockRVCConverter",
        )
        self.current_profile = default_profile

    def set_profile(self, profile_name: str) -> bool:
        if profile_name in self.PROFILES:
            cfg = self.PROFILES[profile_name]
            self._pitch_shift = cfg["pitch"]
            self._formant_ratio = cfg["formant"]
            self._artifact_intensity = cfg["intensity"]
            self.current_profile = profile_name
            return True
        return False

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["current_profile"] = self.current_profile
        meta["available_profiles"] = list(self.PROFILES.keys())
        return meta
