"""Base class for voice conversion engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class VoiceConverter(ABC):
    """Abstract base class for streaming voice converters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the converter engine."""
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the model/engine is loaded and ready for inference."""
        pass

    @abstractmethod
    def convert(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Convert input audio chunk.
        
        Args:
            audio: 1D float32 numpy array in [-1.0, 1.0].
            sample_rate: Sample rate in Hz (default 16000).
            
        Returns:
            1D float32 numpy array with converted audio.
        """
        pass

    def reset(self) -> None:
        """Reset internal phase or buffer state."""
        pass

    def get_metadata(self) -> dict[str, Any]:
        """Return engine metadata and configuration."""
        return {
            "name": self.name,
            "ready": self.is_ready,
        }
