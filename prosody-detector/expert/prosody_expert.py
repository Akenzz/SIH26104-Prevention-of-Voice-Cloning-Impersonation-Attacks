"""
ProsodyExpert — backend adapter satisfying the realtime-backend Expert contract.
================================================================================
Lives in the self-contained prosody-detector package (not vendored into the
backend). realtime-backend/experts/loader.py imports ProsodyExpert from here.

Contract (realtime-backend/experts/protocol.py):
    name, model_version, score(audio_window: np.ndarray) -> {logit, embedding, model_version}
    higher logit = more spoof; audio is 1-D float32 mono @ 16 kHz.

The logit is the logistic-regression log-odds on standardized prosody features.
The embedding is the RAW ordered feature vector (interpretable, ~13-dim) so the
downstream explainer / fusion can read named signals directly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
import sys
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from features.prosody_features import extract_prosody_features, features_to_vector, FEATURE_NAMES  # noqa: E402

logger = logging.getLogger("realtime_backend.experts")

_SR = 16000
_WINDOW_SAMPLES = _SR * 4  # 4 s window, matching WINDOW_SEC in backend config
_DEFAULT_ARTIFACT = _HERE / "artifacts" / "prosody_lr_v1.json"


class ProsodyScorer:
    """Loads the trained artifact and maps a feature vector -> spoof logit."""

    def __init__(self, artifact_path: str | Path = _DEFAULT_ARTIFACT):
        with open(artifact_path) as f:
            art = json.load(f)
        self.version = art.get("version", "prosody-lr-v1")
        self.feature_names = art["feature_names"]
        self._mean = np.asarray(art["scaler_mean"], dtype=np.float64)
        self._scale = np.asarray(art["scaler_scale"], dtype=np.float64)
        self._coef = np.asarray(art["coef"], dtype=np.float64)
        self._intercept = float(art["intercept"])
        self.bonafide_ranges = art.get("bonafide_ranges", {})
        if self.feature_names != FEATURE_NAMES:
            logger.warning("Prosody artifact feature order differs from code FEATURE_NAMES")

    def logit_from_vector(self, vec: np.ndarray) -> float:
        z = (vec.astype(np.float64) - self._mean) / np.where(self._scale == 0, 1.0, self._scale)
        return float(np.dot(self._coef, z) + self._intercept)

    def logit_from_features(self, feats: dict) -> float:
        return self.logit_from_vector(features_to_vector(feats))


class ProsodyExpert:
    """Interpretable prosody/behavioral expert."""

    name = "prosody"
    model_version = "prosody-lr-v1"

    def __init__(self, artifact_path: str | Path = _DEFAULT_ARTIFACT, **_ignored):
        self.scorer = ProsodyScorer(artifact_path)
        self.model_version = self.scorer.version
        logger.info("Loaded expert prosody version=%s from %s", self.model_version, artifact_path)

    def score(self, audio_window: np.ndarray) -> dict:
        window = np.asarray(audio_window, dtype=np.float32).reshape(-1)
        if window.size < _WINDOW_SAMPLES:
            window = np.pad(window, (0, _WINDOW_SAMPLES - window.size))
        elif window.size > _WINDOW_SAMPLES:
            window = window[:_WINDOW_SAMPLES]
        feats = extract_prosody_features(window, sr=_SR)
        vec = features_to_vector(feats)
        logit = self.scorer.logit_from_vector(vec)
        return {
            "logit": logit,
            "embedding": vec.astype(np.float32).tolist(),
            "model_version": self.model_version,
        }
