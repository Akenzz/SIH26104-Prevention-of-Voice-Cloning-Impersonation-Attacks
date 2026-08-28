"""
Audio augmentation utilities for training robustness.

Augmentations applied:
    - Codec degradation (AMR-NB, Opus)
    - Additive noise (white, environmental)
    - Room impulse response convolution
    - Gain variations
"""

import torch
import torchaudio
import numpy as np
from typing import Optional
import subprocess
import tempfile
import os


class CodecAugmentation:
    """
    Apply codec encoding/decoding to simulate telephony degradation.

    Codecs:
        - AMR-NB (Adaptive Multi-Rate Narrowband): Used in 2G/3G calls
        - Opus: Modern VoIP codec (WhatsApp, Telegram, etc.)
    """

    def __init__(self, codec: str = 'amr', sample_rate: int = 16000):
        """
        Args:
            codec: 'amr' or 'opus'
            sample_rate: Audio sample rate (Hz)
        """
        self.codec = codec
        self.sample_rate = sample_rate

        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ffmpeg not found. Install ffmpeg to use codec augmentation.")

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply codec round-trip to audio.

        Args:
            audio: Tensor of shape (samples,)

        Returns:
            Degraded audio tensor of same shape
        """
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_in:
            input_path = f_in.name

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_out:
            output_path = f_out.name

        # Determine intermediate format
        if self.codec == 'amr':
            intermediate_ext = '.amr'
            codec_name = 'libopencore_amrnb'
        elif self.codec == 'opus':
            intermediate_ext = '.opus'
            codec_name = 'libopus'
        else:
            raise ValueError(f"Unsupported codec: {self.codec}")

        intermediate_path = input_path.replace('.wav', intermediate_ext)

        try:
            # Save input audio
            torchaudio.save(input_path, audio.unsqueeze(0), self.sample_rate)

            # Encode to codec format
            subprocess.run([
                'ffmpeg', '-y', '-i', input_path,
                '-codec:a', codec_name,
                intermediate_path
            ], capture_output=True, check=True)

            # Decode back to WAV
            subprocess.run([
                'ffmpeg', '-y', '-i', intermediate_path,
                '-ar', str(self.sample_rate),
                '-ac', '1',  # Mono
                output_path
            ], capture_output=True, check=True)

            # Load degraded audio
            degraded, sr = torchaudio.load(output_path)
            degraded = degraded.squeeze(0)  # Remove channel dim

            # Ensure same length (codec may change duration slightly)
            if degraded.shape[0] < audio.shape[0]:
                padding = audio.shape[0] - degraded.shape[0]
                degraded = torch.nn.functional.pad(degraded, (0, padding))
            elif degraded.shape[0] > audio.shape[0]:
                degraded = degraded[:audio.shape[0]]

            return degraded

        finally:
            # Clean up temp files
            for path in [input_path, intermediate_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)


class AdditiveNoise:
    """
    Add background noise to audio at a target SNR.
    """

    def __init__(self, snr_db_range: tuple = (5, 20), noise_type: str = 'white'):
        """
        Args:
            snr_db_range: (min_snr, max_snr) in dB. Random SNR sampled from this range.
            noise_type: 'white' or 'environmental' (TODO: add environmental noise)
        """
        self.snr_db_range = snr_db_range
        self.noise_type = noise_type

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Add noise to audio.

        Args:
            audio: Tensor of shape (samples,)

        Returns:
            Noisy audio tensor of same shape
        """
        # Sample target SNR
        snr_db = np.random.uniform(*self.snr_db_range)

        # Generate noise
        if self.noise_type == 'white':
            noise = torch.randn_like(audio)
        else:
            # TODO: Load environmental noise samples (cafeteria, street, etc.)
            noise = torch.randn_like(audio)

        # Compute signal and noise power
        signal_power = audio.pow(2).mean()
        noise_power = noise.pow(2).mean()

        # Compute scaling factor for target SNR
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        scale = torch.sqrt(target_noise_power / (noise_power + 1e-8))

        # Add scaled noise
        noisy_audio = audio + scale * noise

        return noisy_audio


class RIRConvolution:
    """
    Apply room impulse response convolution to simulate acoustic space.

    Placeholder: Requires downloading RIR dataset (e.g., MIT RIR, EchoThief).
    """

    def __init__(self, rir_dir: Optional[str] = None):
        """
        Args:
            rir_dir: Directory containing RIR .wav files
        """
        self.rir_dir = rir_dir
        self.rir_paths = []

        if rir_dir and os.path.exists(rir_dir):
            self.rir_paths = [
                os.path.join(rir_dir, f)
                for f in os.listdir(rir_dir)
                if f.endswith('.wav')
            ]

        if not self.rir_paths:
            print("Warning: No RIR files found. RIR augmentation will be skipped.")

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convolve audio with a random RIR.

        Args:
            audio: Tensor of shape (samples,)

        Returns:
            Convolved audio tensor (may be longer, will be cropped to original length)
        """
        if not self.rir_paths:
            return audio  # No RIRs available, return unchanged

        # Load random RIR
        rir_path = np.random.choice(self.rir_paths)
        rir, sr = torchaudio.load(rir_path)
        rir = rir.squeeze(0)  # Remove channel dim

        # Convolve (in frequency domain for efficiency)
        from scipy.signal import fftconvolve
        convolved = fftconvolve(audio.numpy(), rir.numpy(), mode='same')
        convolved = torch.from_numpy(convolved).float()

        # Normalize to prevent clipping
        max_val = convolved.abs().max()
        if max_val > 1.0:
            convolved = convolved / max_val

        return convolved


class AugmentationPipeline:
    """
    Compose multiple augmentations with random selection.
    """

    def __init__(
        self,
        apply_codec: bool = True,
        apply_noise: bool = True,
        apply_rir: bool = False,
        codec_prob: float = 0.5,
        noise_prob: float = 0.5,
        rir_prob: float = 0.3,
        sample_rate: int = 16000,
        rir_dir: Optional[str] = None
    ):
        """
        Args:
            apply_codec: Enable codec augmentation
            apply_noise: Enable noise augmentation
            apply_rir: Enable RIR augmentation
            codec_prob: Probability of applying codec
            noise_prob: Probability of applying noise
            rir_prob: Probability of applying RIR
            sample_rate: Audio sample rate
            rir_dir: Directory containing RIR files
        """
        self.augmentations = []

        if apply_codec:
            self.augmentations.append(('codec', CodecAugmentation('amr', sample_rate), codec_prob))

        if apply_noise:
            self.augmentations.append(('noise', AdditiveNoise(), noise_prob))

        if apply_rir:
            self.augmentations.append(('rir', RIRConvolution(rir_dir), rir_prob))

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to audio.

        Args:
            audio: Tensor of shape (samples,)

        Returns:
            Augmented audio tensor
        """
        for name, aug, prob in self.augmentations:
            if np.random.rand() < prob:
                try:
                    audio = aug(audio)
                except Exception as e:
                    print(f"Warning: {name} augmentation failed: {e}")

        return audio


if __name__ == '__main__':
    # Test augmentations
    import matplotlib.pyplot as plt

    # Generate test signal (sine wave)
    sample_rate = 16000
    duration = 2.0
    freq = 440  # A4 note
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * np.pi * freq * t) * 0.5

    print("Testing augmentations...")

    # Test noise
    print("\n1. Testing additive noise...")
    noise_aug = AdditiveNoise(snr_db_range=(10, 20))
    noisy = noise_aug(audio)
    print(f"   Original RMS: {audio.pow(2).mean().sqrt():.4f}")
    print(f"   Noisy RMS: {noisy.pow(2).mean().sqrt():.4f}")

    # Test codec (requires ffmpeg)
    print("\n2. Testing codec augmentation...")
    try:
        codec_aug = CodecAugmentation('amr', sample_rate)
        degraded = codec_aug(audio)
        print(f"   Codec augmentation successful")
        print(f"   Shape preserved: {audio.shape == degraded.shape}")
    except RuntimeError as e:
        print(f"   Codec augmentation skipped: {e}")

    # Test pipeline
    print("\n3. Testing augmentation pipeline...")
    pipeline = AugmentationPipeline(
        apply_codec=False,  # Skip codec if ffmpeg not available
        apply_noise=True,
        apply_rir=False,
        sample_rate=sample_rate
    )
    augmented = pipeline(audio)
    print(f"   Pipeline successful, output shape: {augmented.shape}")

    print("\n✅ Augmentation tests complete")
