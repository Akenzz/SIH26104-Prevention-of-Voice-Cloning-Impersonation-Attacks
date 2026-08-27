from __future__ import annotations

import numpy as np

from .protocol import Score


class DummyExpert:
    """Random-logit stand-in so the rest of C can run before real models exist.

    Swap this for WavLMExpert / LFCCLCNNExpert by changing EXPERTS; nothing
    else in the pipeline should need to change.
    """

    name = "dummy"
    model_version = "dummy-v0"

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def score(self, audio_window: np.ndarray) -> Score:
        window = np.asarray(audio_window, dtype=np.float32).reshape(-1)
        if window.size == 0:
            raise ValueError("audio_window is empty")
        logit = float(self._rng.uniform(-2.0, 2.0))
        return {
            "logit": logit,
            "embedding": None,
            "model_version": self.model_version,
        }
