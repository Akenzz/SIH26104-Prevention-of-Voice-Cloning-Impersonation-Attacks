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


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from realtime-backend/.env into os.environ.

    Zero-dependency and optional: the file is gitignored and only used to keep
    local secrets (GROQ_API_KEY) out of shell history and out of the repo. Real
    environment variables always win, so this never overrides an explicit
    `GROQ_API_KEY=... python server.py`.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(ROOT / ".env")

MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", str(ROOT / "model_cache")))

TARGET_SAMPLE_RATE = 16000
WINDOW_SEC = float(os.environ.get("WINDOW_SEC", "4.0"))
HOP_SEC = float(os.environ.get("HOP_SEC", "0.5"))

# Checkpoints are pulled from Hugging Face on first use and cached under
# model_cache/. experts/hub.py short-circuits when the file already exists, so a
# fresh clone downloads once and every later run is offline.
#
# Only the two shipped decision experts live here. The earlier LFCC checkpoints
# (lfcc / hindi / mc_v3) and the prosody expert were removed to keep the demo
# surface to exactly two models; recover them from git history if needed.
HUB_EXPERTS = {
    # Expert 1 (Person A): WavLM Base+ front-end + classifier head.
    "wavlm": {
        "repo_id": "Akenzz/SIH-Models",
        "filename": "wavlm_best_model_v5.pt",
        "local_name": "wavlm_best_model_v5.pt",
    },
    # Expert 2: LFCC-LCNN Hybrid, warm-start fine-tuned 5 epochs on the corpus +
    # 12 modern-engine clips. Uploaded to sarosh22/Hybrid_new on HF; ensure_checkpoint
    # pulls it on first run and caches under model_cache/ like the other Hub experts.
    "hybrid": {
        "repo_id": "sarosh22/Hybrid_new",
        "filename": "hybrid_clean_plus_newclips_final.pth",
        "local_name": "hybrid_clean_plus_newclips_final.pth",
    },
    # Expert 3: TakHemlata SSL
    "ssl": {
        "repo_id": "Akenzz/SIH-Models",
        "filename": "best_SSL_model_LA.pth",
        "local_name": "best_SSL_model_LA.pth",
    },
    # Expert 2c: the bandwidth-robust retrain. Same architecture and data as
    # hybrid_nc, but trained with a 7 kHz parity band gate on BOTH classes and
    # ALL splits, which removes the resampler artifact the other LFCC experts
    # decide on: training resampled with librosa/soxr (brickwalls 7.9-8 kHz),
    # this backend resamples with scipy (does not), and the corpus's native
    # sample rates were split by label -- so "hole near Nyquist" was a label
    # proxy and in-training generators read bonafide live.
    # Measured: mean |scipy - librosa| verdict gap 13.35 logits (hybrid) and
    # 5.25 (hybrid_nc) -> 0.03 here, i.e. the verdict no longer depends on the
    # resampler. Local spoof recall 7/7 vs 0/7 and 6/7.
    # Its dev EER (8.88%) looks worse than hybrid's 2.42% because 2.42% was
    # inflated by the artifact -- the same shipped weights score 18.97% once the
    # band is gated. Do not compare the two numbers directly.
    # The 7000 Hz gate is stored INSIDE the checkpoint and applied by
    # experts/lfcc.py at score time; serving this ungated biases it spoofward.
    "hybrid_br": {
        "repo_id": "local",
        "filename": "hybrid_br_best.pth",
        "local_name": "hybrid_br_best.pth",
        "local_only": True,
    },
    # Expert 2d: the CURRENT decision expert. Same LFCC-LCNN architecture and the
    # same 7 kHz parity band gate as hybrid_br, but trained FROM SCRATCH (no warm
    # start) on a much larger corpus: hybrid_max UNIONED with hybrid_br's own
    # training data = 84,271 VAD chunks, 6 languages, 132 spoof generators, at
    # exact chunk-level 1:1 bonafide:spoof WITHIN each language.
    # Gated dev EER 10.20% vs hybrid_br's 8.88%, but on 25x wider generator
    # coverage and a different dev split -- read eval_ood, not dev, to compare.
    # Like hybrid_br the 7000 Hz gate is stored INSIDE the checkpoint and applied
    # by experts/lfcc.py at score time; serving it ungated biases it spoofward.
    "hybrid_maxbr": {
        "repo_id": "sarosh22/Final_LFCC",
        "filename": "hybrid_maxbr_best.pth",
        "local_name": "hybrid_maxbr_best.pth",
    },
}

# Human-facing labels for the frontend, so the UI never has to hardcode names.
# ASCII only — these are echoed straight into JSON.
EXPERT_LABELS = {
    "wavlm":       "Expert-1: WavLM Base+ (v5)",
    "hybrid":      "Expert-2: LFCC-LCNN Hybrid",
    "hybrid_maxbr": "Expert-2: LFCC-LCNN Max (132 generators, bandwidth-robust)",
}

# Per-expert Platt calibrators. Each expert's logits live on their own scale, so
# a calibrator fitted on one expert would misread the other. The band/decision
# still comes from SINGLE_EXPERT via CALIBRATOR_PATH; these are what let the UI
# show a meaningful probability for BOTH models side by side.
EXPERT_CALIBRATORS = {
    "wavlm": ARTIFACTS_DIR / "platt_v6.json",
    "hybrid": ARTIFACTS_DIR / "calibrator_hybrid_newclips.json",
    "ssl": ARTIFACTS_DIR / "platt_ssl.json",
    # hybrid_br is calibrated on GATED dev audio (lfcc-detector/fit_calibrator_br.py,
    # speaker-disjoint half/half: held-out EER 7.84%, ECE 0.061). A calibrator
    # fitted on ungated audio would map logits this model never produces in
    # service, so this file and the checkpoint's gate must stay in step.
    "hybrid_br": ARTIFACTS_DIR / "calibrator_hybrid_br.json",
    # hybrid_maxbr is a different model on a different corpus, so it gets its own
    # Platt fit on ITS OWN gated dev split (hybrid_maxbr_dev.csv, speaker-disjoint
    # half). Reusing calibrator_hybrid_br.json here would read this model's logits
    # on another model's scale -- in-range, plausible, and wrong.
    "hybrid_maxbr": ARTIFACTS_DIR / "calibrator_hybrid_maxbr.json",
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
    experts: list[str] = field(default_factory=lambda: _csv_env("EXPERTS", "wavlm,hybrid_maxbr"))
    fusion_mode: str = os.environ.get("FUSION_MODE", "heuristic_avg")
    single_expert: str = os.environ.get("SINGLE_EXPERT", "hybrid")
    ema_alpha: float = float(os.environ.get("EMA_ALPHA", "0.3"))
    device: str = os.environ.get("DEVICE", "cpu")
    fusion_path: Path = Path(os.environ.get("FUSION_PATH", str(ARTIFACTS_DIR / "fusion.json")))
    calibrator_path: Path = Path(
        os.environ.get("CALIBRATOR_PATH", str(ARTIFACTS_DIR / "calibrator_hybrid_newclips.json"))
    )
    policy_path: Path = Path(os.environ.get("POLICY_PATH", str(ARTIFACTS_DIR / "policy.json")))
    model_cache_dir: Path = MODEL_CACHE_DIR
    silence_rms: float = float(os.environ.get("SILENCE_RMS", "1e-4"))
    clip_abs: float = float(os.environ.get("CLIP_ABS", "0.99"))
    clip_fraction: float = float(os.environ.get("CLIP_FRACTION", "0.01"))
    prefetch_models: bool = os.environ.get("PREFETCH_MODELS", "0") == "1"
    # Task F narration (optional). When GROQ_API_KEY is unset the /narrate
    # endpoint serves 503 and the frontend uses its local template narrator, so
    # the demo runs fully offline. Setting a key turns on LLM rephrasing.
    #
    # Default model is groq/compound-mini: of the chat models Groq currently
    # serves it was the only one that returned a clean one-liner. qwen3.6 leaks
    # <think> chain-of-thought into the content, and openai/gpt-oss-* spend the
    # whole token budget on hidden reasoning and return empty content. Override
    # with GROQ_MODEL if your account has something better (e.g. a llama).
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    groq_model: str = os.environ.get("GROQ_MODEL", "groq/compound-mini")

    @property
    def narration_enabled(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def window_samples(self) -> int:
        return int(self.window_sec * self.target_sample_rate)

    @property
    def hop_samples(self) -> int:
        return int(self.hop_sec * self.target_sample_rate)


def load_settings() -> Settings:
    settings = Settings()
    if settings.fusion_mode not in {"single", "fused", "heuristic", "heuristic_avg", "lr_fusion"}:
        raise ValueError(f"FUSION_MODE must be one of 'single', 'fused', 'heuristic', 'heuristic_avg', 'lr_fusion', got {settings.fusion_mode!r}")
    if not settings.experts:
        raise ValueError("EXPERTS must list at least one expert")
    unknown = [e for e in settings.experts if e not in HUB_EXPERTS and e not in {"dummy", "ssl"}]
    if unknown:
        raise ValueError(
            f"Unknown expert(s) {unknown}. Known: {', '.join(HUB_EXPERTS)}, ssl. "
            "The older lfcc/hindi/mc_v3/prosody/hybrid/hybrid_br experts were removed."
        )
    if settings.fusion_mode == "single":
        if not settings.single_expert:
            raise ValueError("FUSION_MODE=single requires SINGLE_EXPERT")
        if settings.single_expert not in settings.experts:
            raise ValueError(
                f"SINGLE_EXPERT={settings.single_expert!r} is not in EXPERTS={settings.experts}"
            )
    return settings
