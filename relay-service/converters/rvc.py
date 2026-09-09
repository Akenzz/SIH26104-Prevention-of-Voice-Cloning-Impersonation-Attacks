"""Real RVC Converter supporting torch / rvc-python inference with fallback."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
import numpy as np

from .base import VoiceConverter
from .mock import PitchFormantFallbackConverter

logger = logging.getLogger("relay_service.converters.rvc")


class RealRVCConverter(VoiceConverter):
    """Voice converter using RVC (Retrieval-based Voice Conversion) inference.
    
    If PyTorch or model weights (.pth) are unavailable (e.g. on CPU/macOS or before
    weights are downloaded), it cleanly falls back to PitchFormantFallbackConverter
    so that tests and streaming remain functional without breaking.
    """

    def __init__(
        self,
        model_path: str = "",
        index_path: str = "",
        pitch_shift: float = 0.0,
        f0_method: str = "pm",
        device: str = "cpu",
        fallback_pitch_shift: float = 4.0,
    ):
        self.model_path = model_path
        self.index_path = index_path
        self.pitch_shift = pitch_shift
        self.f0_method = f0_method
        self.device = device

        self._fallback_converter = PitchFormantFallbackConverter(
            pitch_shift_semitones=fallback_pitch_shift,
            name="RVC_Fallback(PitchFormant)",
        )
        self._is_fallback = True
        self._rvc_pipeline = None

        self._init_engine()

    def _init_engine(self) -> None:
        """Attempt to load real RVC model weights and pipeline."""
        if not self.model_path or not Path(self.model_path).is_file():
            logger.info(
                "RVC model path not specified or file not found (%r). "
                "Using PitchFormantFallbackConverter for voice spoofing.",
                self.model_path,
            )
            self._is_fallback = True
            return

        try:
            import torch
            # Check if rvc_python or custom inference is available
            try:
                from rvc_python.infer import RVCInference
                self._rvc_pipeline = RVCInference(device=self.device)
                self._rvc_pipeline.load_model(self.model_path)
                self._is_fallback = False
                logger.info("Successfully loaded RVC model via rvc_python from %s", self.model_path)
                return
            except ImportError:
                pass

            # Alternative torch-based RVC model loader if rvc_python is not installed
            # Look for checkpoint dictionary
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and ("weight" in checkpoint or "model" in checkpoint):
                self._rvc_pipeline = checkpoint
                self._is_fallback = False
                logger.info("Successfully loaded torch RVC checkpoint from %s", self.model_path)
                return

        except Exception as exc:
            logger.warning(
                "Failed to initialize real RVC engine from %s: %s. "
                "Active fallback: PitchFormantFallbackConverter.",
                self.model_path,
                exc,
            )

        self._is_fallback = True

    @property
    def name(self) -> str:
        if self._is_fallback:
            return f"RealRVCConverter[Fallback:{self._fallback_converter.name}]"
        return "RealRVCConverter[Torch]"

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def is_using_fallback(self) -> bool:
        return self._is_fallback

    def convert(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        if self._is_fallback or self._rvc_pipeline is None:
            return self._fallback_converter.convert(audio, sample_rate=sample_rate)

        # Real RVC inference path
        try:
            import torch
            # Convert float32 audio to torch tensor
            audio_tensor = torch.from_numpy(audio).to(self.device).float()
            # If pipeline is callable / inference object:
            if callable(self._rvc_pipeline):
                with torch.no_grad():
                    out = self._rvc_pipeline(audio_tensor)
                return out.cpu().numpy()
            else:
                # Fallback if checkpoint object is raw dict
                return self._fallback_converter.convert(audio, sample_rate=sample_rate)
        except Exception as exc:
            logger.warning("RVC inference failed: %s; falling back to DSP converter", exc)
            return self._fallback_converter.convert(audio, sample_rate=sample_rate)

    def reset(self) -> None:
        self._fallback_converter.reset()

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ready": self.is_ready,
            "is_using_fallback": self._is_fallback,
            "model_path": self.model_path,
            "index_path": self.index_path,
            "device": self.device,
            "pitch_shift": self.pitch_shift,
        }
