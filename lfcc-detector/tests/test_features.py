"""
Unit tests for LFCC feature extraction and Cepstral Mean and Variance Normalization (CMVN).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np

pytest.importorskip("torchaudio")

from models.features import LFCCExtractor, LFCCWithDelta, extract_lfcc_features, apply_cmvn
from models.lfcc_lcnn import LFCCLCNNWithFeatureExtraction


def test_apply_cmvn_statistics():
    """Verify CMVN normalizes each channel to zero mean and unit variance across time."""
    batch_size = 4
    channels = 60
    time_frames = 200

    # Random features with arbitrary scale and mean per channel
    raw = torch.randn(batch_size, channels, time_frames) * 15.0 + 35.0
    normed = apply_cmvn(raw)

    assert normed.shape == raw.shape

    # Mean across time frames should be ~0.0
    mean_per_channel = normed.mean(dim=-1)
    assert torch.allclose(mean_per_channel, torch.zeros_like(mean_per_channel), atol=1e-5)

    # Standard deviation across time frames should be ~1.0
    std_per_channel = normed.std(dim=-1)
    assert torch.allclose(std_per_channel, torch.ones_like(std_per_channel), atol=1e-4)


def test_cmvn_gain_invariance():
    """
    Verify CMVN eliminates recording gain / volume variance.
    Scaling the waveform amplitude by 0.2x, 1.0x, 3.0x shifts raw LFCCs,
    but CMVN-normalized LFCCs remain invariant.
    """
    sample_rate = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Synthesize multi-tone test signal
    base_wave = torch.sin(2 * np.pi * 440 * t) + 0.5 * torch.sin(2 * np.pi * 1200 * t)

    scales = [0.2, 1.0, 3.0]
    raw_feats = []
    cmvn_feats = []

    for s in scales:
        scaled_wave = base_wave * s
        raw_f = extract_lfcc_features(scaled_wave, sample_rate, with_deltas=True, use_cmvn=False)
        cmvn_f = extract_lfcc_features(scaled_wave, sample_rate, with_deltas=True, use_cmvn=True)
        raw_feats.append(raw_f)
        cmvn_feats.append(cmvn_f)

    # Raw features differ significantly under gain changes (c0 shifts by log gain)
    raw_diff_0_1 = (raw_feats[0] - raw_feats[1]).abs().max().item()
    assert raw_diff_0_1 > 1.0, f"Raw features should vary with gain, diff was {raw_diff_0_1}"

    # CMVN-normalized features should be nearly identical across all scales
    for i in range(len(scales)):
        for j in range(i + 1, len(scales)):
            cmvn_diff = (cmvn_feats[i] - cmvn_feats[j]).abs().max().item()
            assert cmvn_diff < 1e-3, f"CMVN features should be gain-invariant, got max diff {cmvn_diff}"


def test_cmvn_numerical_stability():
    """Verify CMVN does not crash or produce NaNs on flat or silent signals."""
    silent_audio = torch.zeros(16000)
    cmvn_f = extract_lfcc_features(silent_audio, 16000, with_deltas=True, use_cmvn=True)

    assert not torch.isnan(cmvn_f).any()
    assert not torch.isinf(cmvn_f).any()


def test_backward_compatibility():
    """Verify default use_cmvn=False preserves existing unnormalized behavior."""
    audio = torch.randn(16000)

    # Default call without use_cmvn argument
    feat_default = extract_lfcc_features(audio, 16000, with_deltas=True)
    # Explicit use_cmvn=False
    feat_explicit_raw = extract_lfcc_features(audio, 16000, with_deltas=True, use_cmvn=False)

    assert torch.equal(feat_default, feat_explicit_raw)


def test_end_to_end_cmvn_wrapper():
    """Verify LFCCLCNNWithFeatureExtraction works end-to-end with CMVN enabled."""
    model = LFCCLCNNWithFeatureExtraction(use_cmvn=True)
    model.eval()

    waveform = torch.randn(2, 32000)
    with torch.no_grad():
        logit, embed = model(waveform, return_embedding=True)

    assert logit.shape == (2, 1)
    assert embed.shape == (2, 128)
    assert torch.isfinite(logit).all()
    assert torch.isfinite(embed).all()
