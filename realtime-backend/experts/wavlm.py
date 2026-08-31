"""
experts/wavlm.py — Adapter for Expert 1 (WavLM-base-plus frozen backbone).

Wiring strategy
---------------
1. LOCAL CHECKPOINT (primary): looks for the checkpoint produced by
   expert1/train.py at a well-known local path relative to the repo root.
   This lets every developer run without a HuggingFace Hub upload.

2. HUB CHECKPOINT (fallback): if the local file does not exist, the adapter
   downloads ``best_model.pt`` from ``Akenzz/Expert-1`` (config.py). Once
   you push the checkpoint there, teammates get it automatically.

Score output (same contract as LFCCLCNNExpert):
    {"logit": float, "embedding": list[float], "model_version": str}

Label convention (IMPORTANT — never flip):
    logit > 0  →  spoof / fake  (label 1)
    logit < 0  →  bonafide / real  (label 0)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

from config import HUB_EXPERTS, TARGET_SAMPLE_RATE, WINDOW_SEC
from .hub import ensure_checkpoint, torch_load_checkpoint
from .protocol import Score

logger = logging.getLogger("realtime_backend.experts")

_WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_SAMPLE_RATE)   # 64 000 for 4 s @ 16 kHz

# Local checkpoint produced by training — no Hub upload needed.
# wavlm-base-plus/checkpoints/best_model_v2.pt is the real trained file.
_LOCAL_CHECKPOINT = (
    Path(__file__).resolve().parent.parent.parent   # repo root
    / "wavlm-base-plus"
    / "checkpoints"
    / "best_model_v2.pt"
)


def _add_wavlm_to_path() -> None:
    """Ensure the repo root is on sys.path so the wavlm-base-plus package
    is importable via importlib (hyphen in name prevents normal import).

    File location:  realtime-backend/experts/wavlm.py
    .parent         realtime-backend/experts/
    .parent.parent  realtime-backend/
    .parent.parent.parent  <repo root>  ← wavlm-base-plus/ lives here
    """
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


class WavLMExpert:
    """Adapter for Expert 1 (WavLM-base-plus + linear classification head).

    The backbone (microsoft/wavlm-base-plus) is permanently frozen; only the
    two-layer linear head is trained.  Higher logit = more spoof evidence.

    Checkpoint resolution order:
        1. expert1/checkpoints/best_model.pt  (local, produced by train.py)
        2. Akenzz/Expert-1 on HuggingFace Hub (for teammates without local ckpt)
    """

    name = "wavlm"
    model_version = "wavlm-base-plus-v1"

    def __init__(self, cache_dir: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None

        # ── Resolve checkpoint ─────────────────────────────────────────────────
        if _LOCAL_CHECKPOINT.exists() and _LOCAL_CHECKPOINT.stat().st_size > 0:
            self.checkpoint_path = _LOCAL_CHECKPOINT
            logger.info("WavLM: using LOCAL checkpoint: %s", self.checkpoint_path)
        else:
            logger.info(
                "WavLM: local checkpoint not found at %s — downloading from Hub.",
                _LOCAL_CHECKPOINT,
            )
            spec = HUB_EXPERTS["wavlm"]
            self.checkpoint_path = ensure_checkpoint(
                repo_id=spec["repo_id"],
                filename=spec["filename"],
                cache_dir=cache_dir,
                local_name=spec["local_name"],
            )

        self._blob = torch_load_checkpoint(
            self.checkpoint_path, map_location=str(self.device)
        )
        self._wire_model()

    def _wire_model(self) -> None:
        """Import WavLMClassifier from wavlm-base-plus package and load weights."""
        _add_wavlm_to_path()
        import importlib
        # Package folder is "wavlm-base-plus" — hyphens are not valid Python
        # identifiers so we use importlib.import_module instead of a bare import.
        pkg = importlib.import_module("wavlm-base-plus.model")
        WavLMClassifier = pkg.WavLMClassifier

        model = WavLMClassifier()

        # The checkpoint is saved as {"model_state_dict": ..., "epoch": ..., ...}
        if isinstance(self._blob, dict):
            state = self._blob.get("model_state_dict", self._blob)
            saved_epoch    = self._blob.get("epoch", "?")
            saved_dev_loss = self._blob.get("dev_loss", float("nan"))
            saved_dev_acc  = self._blob.get("dev_acc",  float("nan"))
        else:
            state = self._blob
            saved_epoch = saved_dev_loss = saved_dev_acc = "?"

        # strict=True: fail loudly on architecture mismatch rather than scoring
        # silently with a half-loaded model.
        model.load_state_dict(state, strict=True)
        model.to(self.device).eval()
        self.model = model
        self.model_version = f"wavlm-base-plus-ep{saved_epoch}"

        logger.info(
            "Loaded expert wavlm  version=%s  from %s "
            "(epoch=%s  dev_loss=%s  dev_acc=%s)",
            self.model_version,
            self.checkpoint_path,
            saved_epoch,
            f"{saved_dev_loss:.4f}" if isinstance(saved_dev_loss, float) else saved_dev_loss,
            f"{saved_dev_acc:.1%}"  if isinstance(saved_dev_acc,  float) else saved_dev_acc,
        )

        # Release the raw blob (optimizer state, etc.) — only the model is needed
        self._blob = None

    def score(self, audio_window: np.ndarray) -> Score:
        """Score one audio window.

        Args:
            audio_window: 1-D float32 numpy array at 16 kHz.
                          Length is expected to be _WINDOW_SAMPLES (64 000);
                          shorter windows are zero-padded, longer ones are cropped.

        Returns:
            Score dict with keys:
                logit        (float)  — raw output; higher = more likely fake
                embedding    (list[float]) — 768-dim pooled WavLM representation
                model_version (str)
        """
        if self.model is None:
            raise RuntimeError(
                "WavLMExpert.model is None — _wire_model() did not complete."
            )

        # ── Defensive pad / crop ───────────────────────────────────────────────
        window = np.asarray(audio_window, dtype=np.float32).reshape(-1)
        if window.size < _WINDOW_SAMPLES:
            window = np.pad(window, (0, _WINDOW_SAMPLES - window.size))
        elif window.size > _WINDOW_SAMPLES:
            window = window[:_WINDOW_SAMPLES]

        tensor = torch.from_numpy(window).to(self.device).unsqueeze(0)  # (1, N)

        use_amp = self.device.type == "cuda"
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=use_amp):
            logit_t   = self.model(tensor)                      # (1, 1)
            embed_t   = self.model.get_embedding(tensor)        # (1, 768)

        logit     = float(logit_t.squeeze().item())
        embedding = embed_t.squeeze(0).float().cpu().numpy().tolist()

        return {
            "logit"        : logit,
            "embedding"    : embedding,
            "model_version": self.model_version,
        }
