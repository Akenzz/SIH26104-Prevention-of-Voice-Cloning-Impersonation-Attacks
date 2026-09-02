"""
model.py — WavLM-based binary speech authenticity classifier.

Architecture:
    • Backbone : microsoft/wavlm-base-plus (HuggingFace), FROZEN.
      No gradients flow into the backbone weights.
    • Pooling  : Mean-pool the last hidden state across the time axis.
    • Head     : Linear(768 → 256) → GELU → Dropout → Linear(256 → 1)
    • Output   : A single raw logit (no sigmoid applied here).

Label / output convention (IMPORTANT — never flip this):
    logit > 0  →  more likely spoof / AI-generated (label 1)
    logit < 0  →  more likely bonafide / real       (label 0)

Public API used by other team members:
    score(audio_window: np.ndarray) -> dict
        {
            "logit"         : float,        # raw output; higher = more likely fake
            "embedding"     : list[float],  # 768-dim pooled WavLM representation
            "model_version" : str,          # e.g. "wavlm-base-plus-v1"
        }
"""

import numpy as np
import torch
import torch.nn as nn
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from transformers import WavLMModel

# ─── Constants ─────────────────────────────────────────────────────────────────
BACKBONE_NAME  = "microsoft/wavlm-base-plus"
HIDDEN_SIZE    = 768    # WavLM-base hidden dimension
HEAD_HIDDEN    = 256    # Intermediate layer in the classification head
DROPOUT_RATE   = 0.1
MODEL_VERSION  = "wavlm-base-plus-v1"

# ─── Model class ───────────────────────────────────────────────────────────────

class WavLMClassifier(nn.Module):
    """
    Binary deepfake speech detector built on top of frozen WavLM.

    The backbone is loaded once and its parameters are permanently frozen.
    Only the classification head is trained.
    """

    def __init__(self, backbone_name: str = BACKBONE_NAME):
        super().__init__()

        # ── Load frozen backbone ───────────────────────────────────────────────
        print(f"[model] Loading backbone: {backbone_name}")
        self.backbone = WavLMModel.from_pretrained(backbone_name, use_safetensors=False)

        # Freeze ALL backbone parameters — no gradient updates ever.
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()  # keep BN / dropout in eval mode during training too

        # ── Trainable classification head ──────────────────────────────────────
        # Input: mean-pooled last hidden state (HIDDEN_SIZE,)
        # Output: single logit
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HEAD_HIDDEN, 1),
        )

    def forward_with_embedding(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the spoof logit and pooled WavLM representation in one pass.

        Callers that need both (the realtime adapter does, on every window) must
        use this. Calling ``forward`` and ``get_embedding`` separately runs the
        frozen backbone twice for the same audio.
        """
        # Backbone is always in eval mode; no grad through it.
        with torch.no_grad():
            outputs = self.backbone(input_values=waveform)

        # last_hidden_state: (batch, time_frames, hidden)
        # Mean-pool across the time dimension → (batch, hidden)
        pooled = outputs.last_hidden_state.mean(dim=1)

        # Head → single logit per sample
        return self.head(pooled), pooled

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, time) float32 tensor, 16 kHz, range ~[-1, 1]

        Returns:
            logit: (batch, 1)  raw unnormalised score
                   higher value = more evidence of spoof/fake
        """
        logit, _ = self.forward_with_embedding(waveform)
        return logit

    def get_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return the 768-dim pooled embedding (detached from graph)."""
        _, pooled = self.forward_with_embedding(waveform)
        return pooled  # (batch, 768)


# ─── Inference helper — public API ─────────────────────────────────────────────

# Module-level singleton so callers don't reload the model on every call.
_model_singleton: WavLMClassifier | None = None
_device_singleton: torch.device | None  = None


def load_model(checkpoint_path: str | None = None, device: str | None = None):
    """
    Load (or reload) the global model singleton.

    Args:
        checkpoint_path: Path to a .pt file saved by train.py.
                         If None, the model is initialised with random head weights
                         (useful for quick testing without a trained checkpoint).
        device:          "cpu", "cuda", etc.  Defaults to cuda if available.
    """
    global _model_singleton, _device_singleton

    _device_singleton = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = WavLMClassifier()
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=_device_singleton)
        # Support both raw state-dict and wrapped {"model_state_dict": ...} saves
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"[model] Loaded checkpoint: {checkpoint_path}")

    model.to(_device_singleton)
    model.eval()
    _model_singleton = model
    return model


def score(audio_window: np.ndarray) -> dict:
    """
    Score a single audio window.

    Args:
        audio_window: 1-D numpy array of float32 audio samples at 16 kHz.
                      Length should match WINDOW_SAMPLES (64 000 for 4 s default).
                      Values should be in the range [-1.0, 1.0].

    Returns:
        {
            "logit"        : float,         # raw score; higher = more likely fake
            "embedding"    : list[float],   # 768-dim WavLM pooled representation
            "model_version": str,
        }

    NOTE: You must call load_model() at least once before calling score().
    """
    global _model_singleton, _device_singleton

    if _model_singleton is None:
        raise RuntimeError(
            "Model is not loaded. Call model.load_model(checkpoint_path) first."
        )

    # Prepare tensor: (1, time)
    waveform = torch.from_numpy(audio_window).float().unsqueeze(0)
    waveform = waveform.to(_device_singleton)

    with torch.no_grad():
        # One backbone pass for both outputs.
        logit, embedding = _model_singleton.forward_with_embedding(waveform)

    return {
        "logit"        : logit.item(),
        "embedding"    : embedding.squeeze(0).cpu().tolist(),
        "model_version": MODEL_VERSION,
    }
