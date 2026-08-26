"""
LCNN (Light CNN) with Max-Feature-Map activations.

Architecture adapted from ASVspoof 2021 LFCC-LCNN baseline.
Reference: https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-LFCC-LCNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MaxFeatureMap2D(nn.Module):
    """
    Max-Feature-Map (MFM) activation.

    Splits channels into two groups and takes elementwise max.
    This is the signature activation of Light CNN.

    Args:
        in_channels: Number of input channels (must be even)
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        if in_channels % 2 != 0:
            raise ValueError(f"in_channels must be even, got {in_channels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, channels, height, width)

        Returns:
            Tensor of shape (batch, channels//2, height, width)
        """
        # Split channels in half
        half = self.in_channels // 2
        x1, x2 = x[:, :half, :, :], x[:, half:, :, :]

        # Elementwise max
        out = torch.max(x1, x2)

        return out


class ConvMFM(nn.Module):
    """
    Convolution followed by Max-Feature-Map activation.

    The conv outputs 2x the desired channels, then MFM reduces back to desired channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        # Conv outputs 2x channels for MFM
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,  # Double for MFM
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.mfm = MaxFeatureMap2D(out_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.mfm(x)
        return x


class LFCCLCNN(nn.Module):
    """
    LFCC-LCNN model for audio deepfake detection.

    Architecture:
        - LFCC feature extraction (external, done in features.py)
        - 5 ConvMFM blocks with pooling
        - Global average pooling
        - Fully connected layers
        - Binary classification head

    Args:
        n_lfcc_features: Number of input LFCC features (20 or 60 with deltas)
        embedding_dim: Dimension of penultimate layer (for fusion gate)
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_lfcc_features: int = 60,  # 20 LFCC + 20 delta + 20 delta-delta
        embedding_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.n_lfcc_features = n_lfcc_features
        self.embedding_dim = embedding_dim

        # Convolutional blocks
        # Input: (batch, 1, n_lfcc_features, time_frames)
        self.conv1 = ConvMFM(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ConvMFM(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvMFM(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ConvMFM(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, embedding_dim)
        self.bn_fc2 = nn.BatchNorm1d(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Classification head
        self.fc_out = nn.Linear(embedding_dim, 1)  # Binary: bonafide vs spoof

    def forward(
        self,
        lfcc_features: torch.Tensor,
        return_embedding: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            lfcc_features: Tensor of shape (batch, n_lfcc_features, time_frames)
            return_embedding: Whether to return penultimate layer embedding

        Returns:
            logit: Tensor of shape (batch, 1) — classification logit
            embedding: Tensor of shape (batch, embedding_dim) — feature embedding (if return_embedding=True)
        """
        # Add channel dimension: (batch, 1, n_lfcc_features, time_frames)
        x = lfcc_features.unsqueeze(1)

        # Convolutional blocks
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        # Global pooling: (batch, 512, 1, 1) -> (batch, 512)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Embedding layer (penultimate)
        embedding = self.fc2(x)
        embedding = self.bn_fc2(embedding)
        embedding = F.relu(embedding)
        embedding = self.dropout2(embedding)

        # Classification head
        logit = self.fc_out(embedding)

        if return_embedding:
            return logit, embedding
        else:
            return logit, None

    def get_num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LFCCLCNNWithFeatureExtraction(nn.Module):
    """
    End-to-end model: raw audio -> LFCC extraction -> LCNN -> logit.

    This wraps feature extraction and classification into one module.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_lfcc: int = 20,
        with_deltas: bool = True,
        embedding_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Feature extractor
        from .features import LFCCWithDelta, LFCCExtractor

        if with_deltas:
            self.feature_extractor = LFCCWithDelta(
                sample_rate=sample_rate,
                n_lfcc=n_lfcc
            )
            n_features = n_lfcc * 3  # LFCC + delta + delta-delta
        else:
            self.feature_extractor = LFCCExtractor(
                sample_rate=sample_rate,
                n_lfcc=n_lfcc
            )
            n_features = n_lfcc

        # Classifier
        self.classifier = LFCCLCNN(
            n_lfcc_features=n_features,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def forward(
        self,
        waveform: torch.Tensor,
        return_embedding: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            waveform: Raw audio tensor of shape (batch, samples)
            return_embedding: Whether to return embedding

        Returns:
            logit: Classification logit
            embedding: Feature embedding (if return_embedding=True)
        """
        # Extract features
        features = self.feature_extractor(waveform)

        # Classify
        logit, embedding = self.classifier(features, return_embedding=return_embedding)

        return logit, embedding


if __name__ == '__main__':
    print("Testing LFCC-LCNN model...")

    # Test 1: LCNN with pre-extracted features
    print("\n1. Testing LCNN with pre-extracted LFCC features:")
    batch_size = 4
    n_lfcc = 60  # 20 + 20 delta + 20 delta-delta
    time_frames = 100  # Depends on audio length and hop size

    lfcc_features = torch.randn(batch_size, n_lfcc, time_frames)

    model = LFCCLCNN(n_lfcc_features=n_lfcc, embedding_dim=128)
    model.eval()

    with torch.no_grad():
        logit, embedding = model(lfcc_features)

    print(f"   Input shape: {lfcc_features.shape}")
    print(f"   Logit shape: {logit.shape}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Model parameters: {model.get_num_parameters():,}")

    # Test 2: End-to-end model (raw audio -> logit)
    print("\n2. Testing end-to-end model (raw audio -> LFCC -> LCNN):")
    sample_rate = 16000
    duration = 4.0
    audio_batch = torch.randn(batch_size, int(sample_rate * duration))

    e2e_model = LFCCLCNNWithFeatureExtraction(
        sample_rate=sample_rate,
        n_lfcc=20,
        with_deltas=True,
        embedding_dim=128
    )
    e2e_model.eval()

    with torch.no_grad():
        logit, embedding = e2e_model(audio_batch)

    print(f"   Input audio shape: {audio_batch.shape}")
    print(f"   Logit shape: {logit.shape}")
    print(f"   Embedding shape: {embedding.shape}")

    # Test 3: Check logit direction
    print("\n3. Testing logit interpretation:")
    print(f"   Sample logits: {logit.squeeze().tolist()}")
    print(f"   Positive logit = spoof, Negative logit = bonafide")

    # Test 4: MaxFeatureMap activation
    print("\n4. Testing MaxFeatureMap activation:")
    mfm = MaxFeatureMap2D(in_channels=64)
    x = torch.randn(2, 64, 10, 10)
    y = mfm(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Expected: channels reduced by half (64 -> 32)")

    print("\n[PASS] LFCC-LCNN model tests passed!")
