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
        if key == "hindi":
            # Same LFCC-LCNN architecture as "lfcc", different (Hindi V2) checkpoint.
            from .lfcc import LFCCLCNNExpert

            loaded[key] = LFCCLCNNExpert(
                cache_dir=settings.model_cache_dir,
                device=settings.device,
                hub_key="hindi",
                name="hindi",
            )
            continue
        if key == "mc_v3":
            # Same LFCC-LCNN architecture as "lfcc", multi-corpus V3 checkpoint.
            # Pair with CALIBRATOR_PATH=artifacts/calibrator_mc_v3.json — the
            # shipped calibrator is ASVspoof-only and would misread these logits.
            from .lfcc import LFCCLCNNExpert

            loaded[key] = LFCCLCNNExpert(
                cache_dir=settings.model_cache_dir,
                device=settings.device,
                hub_key="mc_v3",
                name="mc_v3",
            )
            continue
        if key == "prosody":
            # Interpretable prosody/behavioral expert. Self-contained in the
            # sibling `prosody-detector/` package (not vendored here); we add it
            # to sys.path and import its adapter. Ships its own JSON artifact, so
            # no Hub download / config entry is needed. Pair with
            # CALIBRATOR_PATH=<prosody-detector>/artifacts/calibrator_prosody.json.
            import sys as _sys

            _prosody_pkg = Path(__file__).resolve().parents[2] / "prosody-detector"
            if str(_prosody_pkg) not in _sys.path:
                _sys.path.insert(0, str(_prosody_pkg))
            from expert.prosody_expert import ProsodyExpert

            loaded[key] = ProsodyExpert()
            logger.info("Loaded expert %s version=%s", key, loaded[key].model_version)
            continue
        raise ValueError(f"Unknown expert {name!r}. Known: dummy, wavlm, lfcc, hindi, mc_v3, prosody")
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
