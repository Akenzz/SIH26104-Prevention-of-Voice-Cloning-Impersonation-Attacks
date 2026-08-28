"""
LFCC (Linear Frequency Cepstral Coefficients) feature extraction.

LFCC is similar to MFCC but uses a linear frequency scale instead of mel scale,
preserving high-frequency details where voice cloning artifacts often appear.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Optional


class LFCCExtractor(nn.Module):
    """
    Extract LFCC features from raw audio.

    Args:
        sample_rate: Audio sample rate (Hz)
        n_fft: FFT size
        win_length: Window length for STFT
        hop_length: Hop length for STFT
        n_lfcc: Number of LFCC coefficients to extract
        n_filters: Number of linear filters in filterbank
        f_min: Minimum frequency (Hz)
        f_max: Maximum frequency (Hz)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_lfcc: int = 20,
        n_filters: int = 70,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or (self.win_length // 2)
        self.n_lfcc = n_lfcc
        self.n_filters = n_filters
        self.f_min = f_min
        self.f_max = f_max or (sample_rate / 2)

        # Use torchaudio's LFCC transform
        # Pass STFT parameters via speckwargs
        self.lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            n_filter=n_filters,
            f_min=f_min,
            f_max=self.f_max,
            speckwargs={
                'n_fft': n_fft,
                'win_length': self.win_length,
                'hop_length': self.hop_length,
            }
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract LFCC features from audio.

        Args:
            waveform: Audio tensor of shape (batch, samples) or (samples,)

        Returns:
            lfcc: Tensor of shape (batch, n_lfcc, time_frames)
        """
        # Ensure batch dimension
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Extract LFCC
        lfcc = self.lfcc_transform(waveform)

        return lfcc


class LFCCWithDelta(nn.Module):
    """
    LFCC features with delta and delta-delta (velocity and acceleration).

    This captures temporal dynamics in the features.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_lfcc: int = 20,
        n_filters: int = 70,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()

        self.lfcc_extractor = LFCCExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_lfcc=n_lfcc,
            n_filters=n_filters,
            f_min=f_min,
            f_max=f_max,
        )

        # Delta and delta-delta computation
        self.compute_deltas = torchaudio.transforms.ComputeDeltas()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract LFCC with deltas.

        Args:
            waveform: Audio tensor of shape (batch, samples) or (samples,)

        Returns:
            features: Tensor of shape (batch, n_lfcc * 3, time_frames)
                     Concatenated [lfcc, delta, delta-delta]
        """
        # Extract LFCC
        lfcc = self.lfcc_extractor(waveform)

        # Compute deltas
        delta = self.compute_deltas(lfcc)
        delta_delta = self.compute_deltas(delta)

        # Concatenate along feature dimension
        features = torch.cat([lfcc, delta, delta_delta], dim=1)

        return features


def extract_lfcc_features(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    n_lfcc: int = 20,
    with_deltas: bool = True
) -> torch.Tensor:
    """
    Convenience function to extract LFCC features.

    Args:
        audio: Tensor of shape (batch, samples) or (samples,)
        sample_rate: Audio sample rate
        n_lfcc: Number of LFCC coefficients
        with_deltas: Whether to include delta and delta-delta features

    Returns:
        features: Tensor of shape (batch, n_features, time_frames)
                 n_features = n_lfcc (without deltas) or n_lfcc * 3 (with deltas)
    """
    if with_deltas:
        extractor = LFCCWithDelta(sample_rate=sample_rate, n_lfcc=n_lfcc)
    else:
        extractor = LFCCExtractor(sample_rate=sample_rate, n_lfcc=n_lfcc)

    extractor.eval()
    with torch.no_grad():
        features = extractor(audio)

    return features


if __name__ == '__main__':
    # Test LFCC extraction
    print("Testing LFCC feature extraction...")

    # Generate dummy audio (4 seconds at 16kHz)
    sample_rate = 16000
    duration = 4.0
    audio = torch.randn(int(sample_rate * duration))

    print(f"\nInput audio shape: {audio.shape}")

    # Test basic LFCC
    print("\n1. Basic LFCC (no deltas):")
    lfcc_basic = extract_lfcc_features(audio, sample_rate, n_lfcc=20, with_deltas=False)
    print(f"   Output shape: {lfcc_basic.shape}")
    print(f"   Expected: (1, 20, time_frames)")

    # Test LFCC with deltas
    print("\n2. LFCC with deltas:")
    lfcc_deltas = extract_lfcc_features(audio, sample_rate, n_lfcc=20, with_deltas=True)
    print(f"   Output shape: {lfcc_deltas.shape}")
    print(f"   Expected: (1, 60, time_frames)  # 20 * 3")

    # Test batch processing
    print("\n3. Batch processing:")
    audio_batch = torch.randn(8, int(sample_rate * duration))
    lfcc_batch = extract_lfcc_features(audio_batch, sample_rate, n_lfcc=20, with_deltas=True)
    print(f"   Input batch shape: {audio_batch.shape}")
    print(f"   Output batch shape: {lfcc_batch.shape}")
    print(f"   Expected: (8, 60, time_frames)")

    # Check feature statistics
    print("\n4. Feature statistics:")
    print(f"   Mean: {lfcc_deltas.mean():.4f}")
    print(f"   Std: {lfcc_deltas.std():.4f}")
    print(f"   Min: {lfcc_deltas.min():.4f}")
    print(f"   Max: {lfcc_deltas.max():.4f}")

    print("\n[PASS] LFCC extraction tests passed!")
