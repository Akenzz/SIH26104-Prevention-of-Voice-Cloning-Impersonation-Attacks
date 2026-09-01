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
    # Expert 4: LFCC-LCNN retrained on Hindi (V2 debiased checkpoint). Same
    # architecture as "lfcc", different training corpus/checkpoint — wired via
    # the shared LFCCLCNNExpert adapter (experts/loader.py).
    "hindi": {
        "repo_id": "sarosh22/hindi_v2_debiased.pth",
        "filename": "hindi_v2_debiased.pth",
        "local_name": "hindi_v2_debiased.pth",
    },
    # Expert 5: LFCC-LCNN retrained on the multi-corpus V3 mix (Hindi-XTTS +
    # ASVspoof A01-A06 + MLAAD-en/de, 61 spoof generators). Same architecture as
    # "lfcc"; the only LFCC checkpoint whose generalization to unseen generators
    # has been measured (see D:\SIH\multicorpus_v3_summary.md). Wired via the
    # shared LFCCLCNNExpert adapter. Needs its own calibrator
    # (artifacts/calibrator_mc_v3.json) — the shipped one is ASVspoof-only.
    "mc_v3": {
        "repo_id": "sarosh22/Multicorpus",
        "filename": "mc_v3.pth",
        "local_name": "mc_v3.pth",
    },
    # Expert 6: LFCC-LCNN trained on the HYBRID clean-model mix — 6 languages
    # (hi/en/kn/ml/mr/ta), ~20k bonafide / ~20k spoof base clips VAD-sliced to
    # 43.8k 4-10s chunks, 130 spoof generators, bonafide<->spoof paired WITHIN
    # each language so corpus/channel cannot act as a label shortcut, plus
    # equal-on-both-classes train-time channel augmentation.
    # First checkpoint to hold up on UNSEEN generators: dev EER 2.42%, held-out
    # ood_en_mlaad (25 unseen gens) 4.58%, ood_itw (real-world, held-out
    # speakers) 9.73%, pooled eval_ood 5.91% — versus the near-chance
    # cross-corpus collapse of the earlier LFCC checkpoints.
    # Same architecture as "lfcc"; needs its OWN calibrator
    # (artifacts/calibrator_hybrid_clean.json) — mc_v3's a/b would mis-scale it.
    "hybrid": {
        "repo_id": "sarosh22/hybrid-clean",
        "filename": "hybrid_clean.pth",
        "local_name": "hybrid_clean.pth",
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
    experts: list[str] = field(default_factory=lambda: _csv_env("EXPERTS", "dummy"))
    fusion_mode: str = os.environ.get("FUSION_MODE", "single")
    single_expert: str = os.environ.get("SINGLE_EXPERT", "")
    ema_alpha: float = float(os.environ.get("EMA_ALPHA", "0.3"))
    device: str = os.environ.get("DEVICE", "cpu")
    fusion_path: Path = Path(os.environ.get("FUSION_PATH", str(ARTIFACTS_DIR / "fusion.json")))
    calibrator_path: Path = Path(
        os.environ.get("CALIBRATOR_PATH", str(ARTIFACTS_DIR / "calibrator.json"))
    )
    policy_path: Path = Path(os.environ.get("POLICY_PATH", str(ARTIFACTS_DIR / "policy.json")))
    model_cache_dir: Path = MODEL_CACHE_DIR
    silence_rms: float = float(os.environ.get("SILENCE_RMS", "1e-4"))
    clip_abs: float = float(os.environ.get("CLIP_ABS", "0.99"))
    clip_fraction: float = float(os.environ.get("CLIP_FRACTION", "0.01"))
    prefetch_models: bool = os.environ.get("PREFETCH_MODELS", "0") == "1"

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
