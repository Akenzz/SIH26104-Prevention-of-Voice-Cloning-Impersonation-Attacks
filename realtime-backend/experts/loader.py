from __future__ import annotations

import logging
from pathlib import Path

from config import Settings
from .dummy import DummyExpert
from .protocol import Expert

logger = logging.getLogger("realtime_backend.experts")

def load_experts(settings: Settings) -> dict[str, Expert]:
    """Instantiate experts named in settings.experts.

    Real Hub experts are registered here once their adapters are wired.
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
        if key == "lfcc":
            from .lfcc import LFCCLCNNExpert

            loaded[key] = LFCCLCNNExpert(cache_dir=settings.model_cache_dir, device=settings.device)
            continue
        raise ValueError(f"Unknown expert {name!r}. Known: dummy, wavlm, lfcc")
    if not loaded:
        raise ValueError("No experts loaded")
    return loaded


def prefetch_hub_files(cache_dir: Path) -> None:
    """Download both Hub checkpoints without constructing model classes."""
    from config import HUB_EXPERTS
    from .hub import ensure_checkpoint

    for spec in HUB_EXPERTS.values():
        ensure_checkpoint(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            cache_dir=cache_dir,
            local_name=spec["local_name"],
        )
