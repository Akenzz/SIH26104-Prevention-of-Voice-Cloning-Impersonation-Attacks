"""
Official ASVspoof 2019/2021 LFCC-LCNN Baseline Architecture.

Reference:
    Wu et al., "ASVspoof 2021: accelerating progress in spoofed and deepfake speech detection"
    Lavrentyeva et al., "STC Antispoofing Systems for ASVspoof 2019 Challenge"
    Official repo: https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-LFCC-LCNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MaxFeatureMap2D(nn.Module):
    """
    Max-Feature-Map (MFM) activation for 2D inputs (B, C, H, W).
    Splits input channels in half and takes elementwise maximum.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 != 0:
            raise ValueError(f"Channel dimension must be even, got {x.shape[1]}")
        half = x.shape[1] // 2
        return torch.max(x[:, :half, :, :], x[:, half:, :, :])


class MaxFeatureMap1D(nn.Module):
    """
    Max-Feature-Map (MFM) activation for 1D inputs (B, C).
    Splits features in half and takes elementwise maximum.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 != 0:
            raise ValueError(f"Feature dimension must be even, got {x.shape[1]}")
        half = x.shape[1] // 2
        return torch.max(x[:, :half], x[:, half:])


class ConvMFM(nn.Module):
    """
    Convolution 2D followed by BatchNorm and MFM activation.
    Out_channels is internal pre-MFM channel count (doubled).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.mfm = MaxFeatureMap2D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mfm(self.bn(self.conv(x)))


class LFCCLCNN(nn.Module):
    """
    Official ASVspoof LFCC-LCNN 9-layer Architecture.
    """
    def __init__(
        self,
        n_lfcc_features: int = 60,
        embedding_dim: int = 128,
        dropout: float = 0.75
    ):
        super().__init__()
        self.n_lfcc_features = n_lfcc_features
        self.embedding_dim = embedding_dim

        # Block 1
        self.conv1 = ConvMFM(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2a = ConvMFM(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2b = ConvMFM(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3a = ConvMFM(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3b = ConvMFM(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4a = ConvMFM(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = ConvMFM(128, 128, kernel_size=3, stride=1, padding=1)

        # Block 5
        self.conv5a = ConvMFM(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5b = ConvMFM(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # FC Layers
        self.fc1 = nn.Linear(64, 320)
        self.mfm_fc1 = MaxFeatureMap1D()  # 320 -> 160
        self.dropout = nn.Dropout(dropout)

        # Embedding Layer (160 -> 128 for contract compliance)
        self.fc_embed = nn.Linear(160, embedding_dim)
        self.bn_embed = nn.BatchNorm1d(embedding_dim)

        # Classification Output
        self.fc_out = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        lfcc_features: torch.Tensor,
        return_embedding: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Input shape: (B, n_lfcc_features, time_frames) -> Add channel dim: (B, 1, n_lfcc_features, time_frames)
        x = lfcc_features.unsqueeze(1)

        x = self.pool1(self.conv1(x))
        x = self.conv2a(x)
        x = self.pool2(self.conv2b(x))
        x = self.conv3a(x)
        x = self.pool3(self.conv3b(x))
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.conv5a(x)
        x = self.pool4(self.conv5b(x))

        x = self.global_pool(x).view(x.size(0), -1)

        x = self.mfm_fc1(self.fc1(x))
        x = self.dropout(x)

        embedding = F.relu(self.bn_embed(self.fc_embed(x)))
        logit = self.fc_out(embedding)

        if return_embedding:
            return logit, embedding
        else:
            return logit, None


class LFCCLCNNWithFeatureExtraction(nn.Module):
    """
    End-to-end wrapper: Audio -> LFCC -> Official ASVspoof LCNN -> Logit
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_lfcc: int = 20,
        with_deltas: bool = True,
        embedding_dim: int = 128,
        dropout: float = 0.75
    ):
        super().__init__()
        from .features import LFCCWithDelta, LFCCExtractor

        if with_deltas:
            self.feature_extractor = LFCCWithDelta(sample_rate=sample_rate, n_lfcc=n_lfcc)
            n_features = n_lfcc * 3
        else:
            self.feature_extractor = LFCCExtractor(sample_rate=sample_rate, n_lfcc=n_lfcc)
            n_features = n_lfcc

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
        features = self.feature_extractor(waveform)
        return self.classifier(features, return_embedding=return_embedding)


if __name__ == '__main__':
    print("Testing Official ASVspoof LFCC-LCNN Baseline Architecture...")
    b_size = 4
    n_features = 60
    time_frames = 251

    feats = torch.randn(b_size, n_features, time_frames)
    model = LFCCLCNN(n_lfcc_features=n_features, embedding_dim=128)
    model.eval()

    with torch.no_grad():
        logit, embed = model(feats)

    print(f"  Input: {feats.shape}")
    print(f"  Logit shape: {logit.shape}")
    print(f"  Embedding shape: {embed.shape}")
    print(f"  Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("[PASS] Official LFCC-LCNN architecture test passed!")