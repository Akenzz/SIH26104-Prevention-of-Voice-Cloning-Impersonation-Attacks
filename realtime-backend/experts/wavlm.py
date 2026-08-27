from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config import HUB_EXPERTS
from .hub import ensure_checkpoint, torch_load_checkpoint
from .protocol import Score

logger = logging.getLogger("realtime_backend.experts")


class WavLMExpert:
    """Adapter for Expert 1 (WavLM). Checkpoint is fetched from the Hub.

    The teammate training this expert has not yet handed over the model class
    / state_dict layout. Download and torch.load work today; ``score()`` is
    blocked on the TODO below so we never silently invent a forward pass.
    """

    name = "wavlm"
    model_version = "wavlm-unwired"

    def __init__(self, cache_dir: Path, device: str = "cpu"):
        spec = HUB_EXPERTS["wavlm"]
        self.device = device
        self.checkpoint_path = ensure_checkpoint(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            cache_dir=cache_dir,
            local_name=spec["local_name"],
        )
        self._blob = torch_load_checkpoint(self.checkpoint_path, map_location=device)
        self.model = None
        self._wire_model()

    def _wire_model(self) -> None:
        # ------------------------------------------------------------------
        # TODO(expert-1): import the real WavLM detector class and load weights.
        #
        # Example once the teammate shares the module:
        #     from their_package import WavLMSpoofClassifier
        #     self.model = WavLMSpoofClassifier(...)
        #     state = self._blob["model_state_dict"] if isinstance(self._blob, dict) else self._blob
        #     self.model.load_state_dict(state)
        #     self.model.to(self.device).eval()
        #     self.model_version = "wavlm-base-plus-<date>"
        # ------------------------------------------------------------------
        keys = list(self._blob.keys()) if isinstance(self._blob, dict) else type(self._blob)
        logger.warning(
            "WavLM checkpoint is on disk (%s) but the model class is not wired yet. "
            "Top-level checkpoint keys/type: %s",
            self.checkpoint_path,
            keys,
        )
        raise NotImplementedError(
            "WavLMExpert._wire_model: plug in the Expert-1 model class and "
            "state_dict load here (checkpoint already downloaded)."
        )

    def score(self, audio_window: np.ndarray) -> Score:
        if self.model is None:
            raise NotImplementedError("WavLMExpert is not wired; use DummyExpert or finish _wire_model().")
        window = np.asarray(audio_window, dtype=np.float32).reshape(-1)
        # TODO(expert-1): replace with the real forward:
        #     logit, embedding = self.model(...)
        raise NotImplementedError("WavLMExpert.score")
