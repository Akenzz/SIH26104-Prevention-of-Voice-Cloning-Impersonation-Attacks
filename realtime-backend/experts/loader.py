from __future__ import annotations

import logging
from pathlib import Path

from config import Settings
from .dummy import DummyExpert
from .protocol import Expert

logger = logging.getLogger("realtime_backend.experts")

# LFCC-LCNN checkpoints trained WITH the parity band gate. Only hybrid_maxbr
# is the active decision expert; the set is kept as a set so additional
# band-gated checkpoints can be added here without touching load_experts().
_BAND_GATED_LFCC = {"hybrid_maxbr"}


def load_experts(settings: Settings) -> dict[str, Expert]:
    """Instantiate experts named in settings.experts.

    Active expert lineup (3 experts):
      wavlm         WavLM Base+ v5   (Akenzz/SIH-Models :: wavlm_best_model_v5.pt)
      ssl           TakHemlata SSL   (Akenzz/SIH-Models :: best_SSL_model_LA.pth)
      hybrid_maxbr  LFCC-LCNN Max    (sarosh22/Final_LFCC :: hybrid_maxbr_best.pth)
                    -- trained on 84k chunks / 132 generators; applies 7 kHz parity
                       band gate; current decision expert.

    All pull their weights from Hugging Face on first run and are cached under
    model_cache/. Requesting an unwired name fails at startup instead of silently
    scoring. Removed: hybrid, hybrid_br (intermediate LFCC checkpoints).
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
        if key == "ssl":
            # The merge left this calling SSLExpert() with no args, but its
            # __init__ requires cache_dir -> TypeError at startup. Pass the same
            # cache_dir/device every other expert gets.
            from .ssl_antispoof import SSLExpert

            loaded[key] = SSLExpert(cache_dir=settings.model_cache_dir, device=settings.device)
            continue

        if key in _BAND_GATED_LFCC:
            # Same LFCC-LCNN class again, but trained WITH a 7 kHz parity band
            # gate. The gate lives in the checkpoint and is applied inside
            # LFCCLCNNExpert.score(), so nothing here (or in the pipeline) needs
            # to know the cutoff -- and it cannot be applied twice.
            # hybrid_maxbr -- from-scratch on the 84k-chunk / 132-generator
            #                 merged corpus, hosted at sarosh22/Final_LFCC,
            #                 and the current decision expert.
            from .lfcc import LFCCLCNNExpert

            loaded[key] = LFCCLCNNExpert(
                cache_dir=settings.model_cache_dir,
                device=settings.device,
                hub_key=key,
                name=key,
            )
            continue
        raise ValueError(
            f"Unknown expert {name!r}. Known: dummy, wavlm, ssl, "
            f"{', '.join(sorted(_BAND_GATED_LFCC))}"
        )
    if not loaded:
        raise ValueError("No experts loaded")
    _assert_band_gates_applied(loaded)
    return loaded


def _assert_band_gates_applied(loaded: dict[str, Expert]) -> None:
    """Fail startup if a checkpoint that declares a band gate isn't applying it.

    A model trained with the parity band gate and served without it receives an
    octave of energy it never saw in training and reads systematically spoofward.
    That failure is silent -- probabilities stay in range and the stream works --
    so it gets caught here rather than in a demo. The check compares what the
    checkpoint declares against what the loaded expert will actually do.
    """
    import torch

    for key, expert in loaded.items():
        path = getattr(expert, "checkpoint_path", None)
        if path is None:
            continue
        try:
            blob = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:  # unreadable here is not this check's business
            continue
        declared = blob.get("band_gate_hz") if isinstance(blob, dict) else None
        applied = getattr(expert, "band_gate_hz", None)
        if declared and not applied:
            raise RuntimeError(
                f"Expert {key!r} loads {Path(path).name}, which was TRAINED with a "
                f"{float(declared):.0f} Hz parity band gate, but the loaded expert is "
                f"not applying one. Serving it this way biases every verdict toward "
                f"spoof. Fix experts/lfcc.py rather than removing this check."
            )
        if declared and applied and abs(float(declared) - float(applied)) > 1e-6:
            raise RuntimeError(
                f"Expert {key!r} band gate mismatch: checkpoint declares "
                f"{float(declared):.1f} Hz, expert applies {float(applied):.1f} Hz."
            )
        if applied:
            logger.info("Verified expert %s applies its %.0f Hz band gate", key, applied)


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
