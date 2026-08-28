"""
Milestone 4B: Standardized Model Adapter.
Wraps any deepfake detector adhering to forward(audio_window) -> (logit, embedding).
Runs inference, measures latency, and standardizes score outputs.
"""

import time
import torch
import numpy as np
from typing import Tuple, Dict, Any, List


class ModelAdapter:
    def __init__(self, model_instance: Any, name: str, version: str = "v1.0"):
        self.model = model_instance
        self.name = name
        self.version = version

    def predict_window(self, audio_window: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
        Runs prediction on a single audio window.
        Returns: (logit, embedding, latency_ms)
        """
        t0 = time.perf_counter()
        if hasattr(self.model, 'forward'):
            logit, embedding = self.model.forward(audio_window)
        elif callable(self.model):
            logit, embedding = self.model(audio_window)
        else:
            raise AttributeError("Model must implement forward() or be callable")
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        return float(logit), np.array(embedding), latency_ms