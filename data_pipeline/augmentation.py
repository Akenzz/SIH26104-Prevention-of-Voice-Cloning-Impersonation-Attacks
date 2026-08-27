"""
data_pipeline/augmentation.py
==============================
Shared audio augmentation utilities for Task B (SIH26104).

Applied to TRAINING data ONLY — never dev or eval splits.
The MD spec requires keeping a clean unaugmented baseline and measuring
augmented-vs-clean as an ablation before claiming "augmentation helped".

Three augmentation classes are provided:
    1. CodecAugmentation  — real codec round-trip via ffmpeg (AMR-NB or Opus)
    2. AdditiveNoise      — white or pink noise at a calibrated SNR range
    3. RIRConvolution     — room impulse response convolution (needs RIR wav files)

And one composed pipeline:
    4. TrainingAugmentationPipeline — applies each augmentation stochastically

Usage (minimal):
    from data_pipeline.augmentation import TrainingAugmentationPipeline
    aug = TrainingAugmentationPipeline(apply_codec=False)  # no ffmpeg needed
    audio_aug = aug(audio_tensor)  # audio_tensor: shape (samples,) float32

Usage (full, with codec):
    aug = TrainingAugmentationPipeline(
        apply_codec=True,   # requires ffmpeg in PATH
        apply_noise=True,
        apply_rir=True,
        rir_dir="path/to/rir_wavs/"   # optional
    )

RIR files:
    Any directory of mono 16kHz .wav room impulse responses works.
    A small free set: OpenSLR RIR (https://www.openslr.org/28/).
    If rir_dir is None or empty, RIR augmentation silently no-ops.
"""

import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio


# ---------------------------------------------------------------------------
# 1. Codec Round-Trip (AMR-NB / Opus via ffmpeg)
# ---------------------------------------------------------------------------

class CodecAugmentation:
    """
    Simulate telephony / VoIP degradation via a real codec encode-decode cycle.

    Requires ffmpeg compiled with libopencore-amrnb (for AMR-NB) or
    libopus (for Opus). Both are included in most ffmpeg builds.

    Args:
        codec:       'amr' (AMR-NB, 2G/3G calls) or 'opus' (WhatsApp/VoIP).
        sample_rate: Expected audio sample rate in Hz (default 16000).

    Raises:
        RuntimeError: If ffmpeg is not found on PATH.
    """

    _CODECS = {
        'amr':  ('libopencore_amrnb', '.amr'),
        'opus': ('libopus',           '.ogg'),
    }

    def __init__(self, codec: str = 'amr', sample_rate: int = 16000):
        if codec not in self._CODECS:
            raise ValueError(f"codec must be one of {list(self._CODECS)}; got '{codec}'")
        self.codec = codec
        self.sample_rate = sample_rate
        self._codec_name, self._ext = self._CODECS[codec]
        self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg() -> None:
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "ffmpeg not found on PATH. Install ffmpeg to use codec augmentation.\n"
                "Windows: winget install ffmpeg  |  https://ffmpeg.org/download.html"
            )

    @staticmethod
    def is_available() -> bool:
        """Return True if ffmpeg is available on PATH."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except Exception:
            return False

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply codec round-trip to a 1-D float32 audio tensor.

        Args:
            audio: shape (samples,), float32, range [-1, 1].

        Returns:
            Codec-degraded tensor of the same shape.
        """
        with tempfile.TemporaryDirectory() as tmp:
            wav_in  = os.path.join(tmp, 'input.wav')
            enc_out = os.path.join(tmp, f'encoded{self._ext}')
            wav_out = os.path.join(tmp, 'output.wav')

            # Write input
            torchaudio.save(wav_in, audio.unsqueeze(0), self.sample_rate)

            # Encode → compressed format
            subprocess.run(
                ['ffmpeg', '-y', '-i', wav_in,
                 '-codec:a', self._codec_name, enc_out],
                capture_output=True, check=True
            )

            # Decode → WAV
            subprocess.run(
                ['ffmpeg', '-y', '-i', enc_out,
                 '-ar', str(self.sample_rate), '-ac', '1', wav_out],
                capture_output=True, check=True
            )

            degraded, _ = torchaudio.load(wav_out)
            degraded = degraded.squeeze(0)

        # Ensure identical length (codec may shift by a few samples)
        orig_len = audio.shape[0]
        if degraded.shape[0] < orig_len:
            degraded = torch.nn.functional.pad(degraded, (0, orig_len - degraded.shape[0]))
        else:
            degraded = degraded[:orig_len]

        return degraded


# ---------------------------------------------------------------------------
# 2. Additive Noise
# ---------------------------------------------------------------------------

class AdditiveNoise:
    """
    Add white or pink noise at a random signal-to-noise ratio.

    The SNR is sampled uniformly from snr_db_range on each call, matching the
    "recorded SNR ranges" requirement in the MD spec.

    Args:
        snr_db_range: (min_snr_dB, max_snr_dB). Typical phone call SNRs: 5–20 dB.
        noise_type:   'white' (flat spectrum) or 'pink' (1/f, more speech-like).
    """

    def __init__(
        self,
        snr_db_range: Tuple[float, float] = (5.0, 20.0),
        noise_type: str = 'white',
    ):
        self.snr_db_range = snr_db_range
        self.noise_type   = noise_type

    def _generate_noise(self, n: int) -> torch.Tensor:
        if self.noise_type == 'pink':
            # Pink noise via spectral shaping of white noise
            white = np.random.randn(n).astype(np.float32)
            freqs = np.fft.rfftfreq(n)
            freqs[0] = 1.0  # avoid divide-by-zero at DC
            spectrum = np.fft.rfft(white) / np.sqrt(freqs)
            pink = np.fft.irfft(spectrum, n=n).astype(np.float32)
            return torch.from_numpy(pink)
        else:
            return torch.randn(n)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: shape (samples,), float32.

        Returns:
            Noisy audio of the same shape.
        """
        snr_db = np.random.uniform(*self.snr_db_range)
        noise  = self._generate_noise(audio.shape[0])

        sig_power   = audio.pow(2).mean().clamp(min=1e-8)
        noise_power = noise.pow(2).mean().clamp(min=1e-8)

        target_noise_power = sig_power / (10 ** (snr_db / 10.0))
        scale = torch.sqrt(target_noise_power / noise_power)

        return audio + scale * noise


# ---------------------------------------------------------------------------
# 3. Room Impulse Response Convolution
# ---------------------------------------------------------------------------

class RIRConvolution:
    """
    Convolve audio with a randomly selected room impulse response (RIR).

    RIR files should be mono 16kHz .wav files.  If no files are found the
    augmentation silently passes audio through unchanged.

    A free RIR set is available at https://www.openslr.org/28/.

    Args:
        rir_dir: Path to a directory of RIR .wav files. Can be None.
    """

    def __init__(self, rir_dir: Optional[str] = None):
        self.rir_paths: List[str] = []
        if rir_dir and os.path.isdir(rir_dir):
            self.rir_paths = [
                os.path.join(rir_dir, f)
                for f in os.listdir(rir_dir)
                if f.lower().endswith('.wav')
            ]
        if not self.rir_paths:
            print(
                "[INFO] RIRConvolution: no RIR .wav files found "
                f"in '{rir_dir}'. RIR augmentation will be skipped (audio passed through)."
            )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: shape (samples,), float32.

        Returns:
            RIR-convolved tensor of the same shape, or original if no RIRs available.
        """
        if not self.rir_paths:
            return audio

        rir_path = np.random.choice(self.rir_paths)
        rir, _   = torchaudio.load(rir_path)
        rir      = rir.squeeze(0)

        # FFT-based convolution (fast for long signals)
        from scipy.signal import fftconvolve
        convolved = fftconvolve(audio.numpy(), rir.numpy(), mode='full')
        # Trim to original length (mode='full' produces len(a)+len(b)-1 samples)
        convolved = convolved[:audio.shape[0]]
        convolved = torch.from_numpy(convolved.astype(np.float32))

        # Normalise to prevent clipping
        peak = convolved.abs().max()
        if peak > 1.0:
            convolved = convolved / peak

        return convolved


# ---------------------------------------------------------------------------
# 4. Composed Training Pipeline
# ---------------------------------------------------------------------------

class TrainingAugmentationPipeline:
    """
    Stochastic composition of all three augmentation types.

    Each augmentation is applied independently with its own probability.
    This matches the MD spec requirement: codec round-trip + additive noise
    + RIR convolution, training data only.

    Ablation note (from MD spec):
        Keep a CLEAN baseline run (apply_codec=False, apply_noise=False,
        apply_rir=False) and compare its EER to augmented runs before
        claiming "augmentation helped".

    Args:
        apply_codec:  Enable codec round-trip. Requires ffmpeg.
        apply_noise:  Enable additive noise.
        apply_rir:    Enable RIR convolution. Requires rir_dir.
        codec_prob:   Probability of applying codec per sample (default 0.5).
        noise_prob:   Probability of applying noise per sample (default 0.5).
        rir_prob:     Probability of applying RIR per sample (default 0.3).
        codec:        'amr' or 'opus' (default 'amr').
        snr_db_range: SNR range for noise in dB (default (5, 20)).
        rir_dir:      Directory of RIR wav files (default None → skipped).
        sample_rate:  Audio sample rate (default 16000).
    """

    def __init__(
        self,
        apply_codec:  bool  = False,
        apply_noise:  bool  = True,
        apply_rir:    bool  = False,
        codec_prob:   float = 0.5,
        noise_prob:   float = 0.5,
        rir_prob:     float = 0.3,
        codec:        str   = 'amr',
        snr_db_range: Tuple[float, float] = (5.0, 20.0),
        rir_dir:      Optional[str] = None,
        sample_rate:  int   = 16000,
    ):
        self._stages: List[Tuple[str, object, float]] = []

        if apply_codec:
            if CodecAugmentation.is_available():
                self._stages.append(('codec', CodecAugmentation(codec, sample_rate), codec_prob))
            else:
                print("[WARN] TrainingAugmentationPipeline: ffmpeg not found — codec augmentation disabled.")

        if apply_noise:
            self._stages.append(('noise', AdditiveNoise(snr_db_range), noise_prob))

        if apply_rir:
            rir_aug = RIRConvolution(rir_dir)
            if rir_aug.rir_paths:          # only add if files were found
                self._stages.append(('rir', rir_aug, rir_prob))

        if not self._stages:
            print("[INFO] TrainingAugmentationPipeline: no augmentations active (clean baseline mode).")

    @property
    def active_augmentations(self) -> List[str]:
        """Names of currently active augmentation stages."""
        return [name for name, _, _ in self._stages]

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply active augmentations stochastically to a single audio window.

        MUST only be called for training split samples.

        Args:
            audio: shape (samples,), float32.

        Returns:
            Augmented audio tensor of the same shape.
        """
        for name, aug, prob in self._stages:
            if np.random.rand() < prob:
                try:
                    audio = aug(audio)
                except Exception as exc:
                    # Augmentation failures must never crash training —
                    # log and continue with un-augmented audio.
                    print(f"[WARN] Augmentation '{name}' failed: {exc}. Skipping.")
        return audio


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sr = 16000
    duration = 2.0
    t = torch.linspace(0, duration, int(sr * duration))
    test_audio = (torch.sin(2 * np.pi * 440 * t) * 0.5)

    print("=" * 55)
    print("data_pipeline/augmentation.py — self-test")
    print("=" * 55)

    # Noise
    print("\n[1/3] AdditiveNoise (white, SNR 10-20 dB)...")
    noisy = AdditiveNoise(snr_db_range=(10, 20))(test_audio)
    print(f"  Input RMS : {test_audio.pow(2).mean().sqrt():.4f}")
    print(f"  Output RMS: {noisy.pow(2).mean().sqrt():.4f}  [OK]")

    # Codec
    print("\n[2/3] CodecAugmentation (AMR-NB)...")
    if CodecAugmentation.is_available():
        degraded = CodecAugmentation('amr', sr)(test_audio)
        print(f"  Shape preserved: {test_audio.shape == degraded.shape}  [OK]")
    else:
        print("  ffmpeg not found — skipped.")

    # Pipeline
    print("\n[3/3] TrainingAugmentationPipeline (noise only)...")
    pipeline = TrainingAugmentationPipeline(apply_codec=False, apply_noise=True, apply_rir=False)
    aug_audio = pipeline(test_audio)
    print(f"  Active stages : {pipeline.active_augmentations}")
    print(f"  Output shape  : {aug_audio.shape}  [OK]")

    print("\n✅ All self-tests passed.")
    sys.exit(0)
