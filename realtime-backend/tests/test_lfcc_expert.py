"""Smoke test for the shipped LFCC-LCNN expert (`hybrid`).

Skips cleanly when torchaudio is missing or the Hub checkpoint cannot be
fetched (offline CI), so the default dummy-only suite is unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torchaudio")

from config import Settings


def _load_expert():
    """Load the hybrid clean-model checkpoint through the shared LFCC adapter."""
    from experts.lfcc import LFCCLCNNExpert

    settings = Settings()
    try:
        return LFCCLCNNExpert(
            cache_dir=settings.model_cache_dir,
            device="cpu",
            hub_key="hybrid",
            name="hybrid",
        )
    except Exception as exc:  # network/torchaudio/checkpoint unavailable
        pytest.skip(f"hybrid expert unavailable in this environment: {exc}")


def test_hybrid_expert_returns_finite_logit_and_embedding():
    expert = _load_expert()
    rng = np.random.default_rng(0)
    window = rng.normal(0.0, 0.1, 64000).astype(np.float32)

    score = expert.score(window)

    assert np.isfinite(score["logit"])
    assert isinstance(score["logit"], float)
    assert score["embedding"] is not None
    assert len(score["embedding"]) == 128
    assert all(np.isfinite(v) for v in score["embedding"])
    # `hybrid` now serves the newclips-folded checkpoint (the old standalone
    # `hybrid_nc` expert was removed and its weights promoted). Assert against
    # config rather than a literal so the next checkpoint swap doesn't fail here
    # for the wrong reason -- what matters is that the version tracks the file
    # actually loaded, since EXPERT_CALIBRATORS["hybrid"] is fitted to it.
    from config import HUB_EXPERTS

    assert score["model_version"] == HUB_EXPERTS["hybrid"]["local_name"].removesuffix(".pth")


def test_hybrid_expert_is_deterministic_in_eval():
    expert = _load_expert()
    window = np.random.default_rng(1).normal(0.0, 0.1, 64000).astype(np.float32)
    a = expert.score(window)["logit"]
    b = expert.score(window)["logit"]
    assert abs(a - b) < 1e-5


def test_hybrid_expert_handles_short_and_long_windows():
    expert = _load_expert()
    short = np.random.default_rng(2).normal(0.0, 0.1, 32000).astype(np.float32)
    long = np.random.default_rng(3).normal(0.0, 0.1, 96000).astype(np.float32)
    assert np.isfinite(expert.score(short)["logit"])
    assert np.isfinite(expert.score(long)["logit"])
