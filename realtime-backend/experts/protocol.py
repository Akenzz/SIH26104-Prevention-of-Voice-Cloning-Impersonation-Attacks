from __future__ import annotations

from typing import Optional, Protocol, TypedDict

import numpy as np


class Score(TypedDict):
    logit: float
    embedding: Optional[list[float]]
    model_version: str


class Expert(Protocol):
    name: str
    model_version: str

    def score(self, audio_window: np.ndarray) -> Score:
        """Score one 16 kHz mono window. Higher logit = more spoof evidence."""
        ...
