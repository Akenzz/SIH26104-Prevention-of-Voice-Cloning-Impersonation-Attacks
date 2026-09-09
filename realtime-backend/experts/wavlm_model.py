"""WavLM-based binary speech authenticity classifier vendored for realtime backend."""

from __future__ import annotations

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import numpy as np
import torch
import torch.nn as nn
from transformers import WavLMModel

BACKBONE_NAME = "microsoft/wavlm-base-plus"
HIDDEN_SIZE = 768
HEAD_HIDDEN = 256
DROPOUT_RATE = 0.1
MODEL_VERSION = "wavlm-base-plus-v1"


class WavLMClassifier(nn.Module):
    """Binary deepfake speech detector built on top of frozen WavLM."""

    def __init__(self, backbone_name: str = BACKBONE_NAME):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained(backbone_name, use_safetensors=False)

        # Freeze all backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top 4 encoder layers
        n_unfreeze = 4
        for layer in self.backbone.encoder.layers[-n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.backbone.eval()

        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HEAD_HIDDEN, 1),
        )

    def forward_with_embedding(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.backbone(input_values=waveform)

        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.head(pooled), pooled

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        logit, _ = self.forward_with_embedding(waveform)
        return logit

    def get_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        _, pooled = self.forward_with_embedding(waveform)
        return pooled
