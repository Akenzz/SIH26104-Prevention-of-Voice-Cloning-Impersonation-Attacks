from __future__ import annotations

import logging
from pathlib import Path

from config import Settings
from .dummy import DummyExpert
from .protocol import Expert

logger = logging.getLogger("realtime_backend.experts")

def load_experts(settings: Settings) -> dict[str, Expert]:
    """Instantiate experts named in settings.experts.

    Two real experts are wired: `wavlm` (Person A's WavLM Base+) and `hybrid`
    (the 6-language LFCC-LCNN clean-model checkpoint). Both pull their weights
    from Hugging Face on first run and are cached under model_cache/.
    Requesting an unwired name fails at startup instead of silently scoring.
    """
    loaded: dict[str, Expert] = {}
    for name in settings.experts:
        key = name.strip().lower()
        if key == "dummy":
            loaded[key] = DummyExpert()
            logger.info("Loaded expert %s version=%s", key, loaded[key].model_version)
            continue
        if key == "wavlm":
            from .wavlm import WavLMExpert

            loaded[key] = WavLMExpert(cache_dir=settings.model_cache_dir, device=settings.device)
            continue
        if key == "hybrid":
            # LFCC-LCNN, hybrid clean-model checkpoint (6 languages, 130 spoof
            # generators, language-paired to kill the channel=label shortcut).
            # Scored with artifacts/calibrator_hybrid_clean.json — every other
            # calibrator in artifacts/ was fitted on a different expert's logit
            # scale and would misread these.
            from .lfcc import LFCCLCNNExpert

            loaded[key] = LFCCLCNNExpert(
                cache_dir=settings.model_cache_dir,
                device=settings.device,
                hub_key="hybrid",
                name="hybrid",
            )
            continue
        if key == "hybrid_nc":
            # Same LFCC-LCNN architecture as `hybrid`, warm-started 5 epochs on
            # the hybrid mix + 12 modern TTS/VC clips. LOCAL-ONLY checkpoint
            # (config.HUB_EXPERTS["hybrid_nc"]["local_only"] = True): loaded
            # straight from model_cache/, never fetched from the Hub. Reads with
            # its OWN calibrator (artifacts/calibrator_hybrid_newclips.json);
            # its logit scale differs from `hybrid`.
            from .lfcc import LFCCLCNNExpert

            loaded[key] = LFCCLCNNExpert(
                cache_dir=settings.model_cache_dir,
                device=settings.device,
                hub_key="hybrid_nc",
                name="hybrid_nc",
            )
            continue
        raise ValueError(f"Unknown expert {name!r}. Known: dummy, wavlm, hybrid, hybrid_nc")
    if not loaded:
        raise ValueError("No experts loaded")
    return loaded


def prefetch_hub_files(cache_dir: Path) -> None:
    """Download both Hub checkpoints without constructing model classes."""
    from config import HUB_EXPERTS
    from .hub import ensure_checkpoint

    for spec in HUB_EXPERTS.values():
        if spec.get("local_only"):
            # Not on the Hub — nothing to prefetch; it ships in model_cache/.
            continue
        ensure_checkpoint(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            cache_dir=cache_dir,
            local_name=spec["local_name"],
        )
