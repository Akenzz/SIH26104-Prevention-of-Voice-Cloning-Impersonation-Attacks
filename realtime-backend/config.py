"""Shared constants for the Task C realtime backend.

Window/hop, Hub repo IDs, fusion/calibration artifact paths, and policy
thresholds live here so teammates can change them without hunting through
the pipeline. Environment variables override defaults at process start.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", str(ROOT / "model_cache")))

TARGET_SAMPLE_RATE = 16000
WINDOW_SEC = float(os.environ.get("WINDOW_SEC", "4.0"))
HOP_SEC = float(os.environ.get("HOP_SEC", "0.5"))

HUB_EXPERTS = {
    "wavlm": {
        "repo_id": "Akenzz/Expert-1",
        "filename": "best_model.pt",
        "local_name": "wavlm_best_model.pt",
    },
    "lfcc": {
        "repo_id": "sarosh22/lfcc-lcnn-asvspoof19",
        "filename": "best_lfcc_lcnn.pth",
        "local_name": "best_lfcc_lcnn.pth",
    },
}


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass
class Settings:
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8000"))
    target_sample_rate: int = TARGET_SAMPLE_RATE
    window_sec: float = WINDOW_SEC
    hop_sec: float = HOP_SEC
    # A real detector is the safe default.  Dummy scoring remains available for
    # local protocol work only through the explicit ALLOW_DUMMY=1 opt-in.
    experts: list[str] = field(default_factory=lambda: _csv_env("EXPERTS", "wavlm"))
    fusion_mode: str = os.environ.get("FUSION_MODE", "single")
    single_expert: str = os.environ.get("SINGLE_EXPERT", "")
    ema_alpha: float = float(os.environ.get("EMA_ALPHA", "0.3"))
    device: str = os.environ.get("DEVICE", "cpu")
    fusion_path: Path = Path(os.environ.get("FUSION_PATH", str(ARTIFACTS_DIR / "fusion.json")))
    calibrator_path: Path = Path(
        os.environ.get("CALIBRATOR_PATH", str(ARTIFACTS_DIR / "platt_v2_combined_dataset.json"))
    )
    policy_path: Path = Path(os.environ.get("POLICY_PATH", str(ARTIFACTS_DIR / "policy.json")))
    model_cache_dir: Path = MODEL_CACHE_DIR
    silence_rms: float = float(os.environ.get("SILENCE_RMS", "1e-4"))
    active_rms: float = float(os.environ.get("ACTIVE_RMS", "0.003"))
    min_active_audio_sec: float = float(os.environ.get("MIN_ACTIVE_AUDIO_SEC", "1.0"))
    activity_frame_ms: int = int(os.environ.get("ACTIVITY_FRAME_MS", "30"))
    clip_abs: float = float(os.environ.get("CLIP_ABS", "0.99"))
    clip_fraction: float = float(os.environ.get("CLIP_FRACTION", "0.01"))
    prefetch_models: bool = os.environ.get("PREFETCH_MODELS", "0") == "1"
    allow_dummy: bool = os.environ.get("ALLOW_DUMMY", "0") == "1"

    @property
    def window_samples(self) -> int:
        return int(self.window_sec * self.target_sample_rate)

    @property
    def hop_samples(self) -> int:
        return int(self.hop_sec * self.target_sample_rate)


def load_settings() -> Settings:
    settings = Settings()
    if settings.fusion_mode not in {"single", "fused"}:
        raise ValueError(f"FUSION_MODE must be 'single' or 'fused', got {settings.fusion_mode!r}")
    if not settings.experts:
        raise ValueError("EXPERTS must list at least one expert")
    return settings
