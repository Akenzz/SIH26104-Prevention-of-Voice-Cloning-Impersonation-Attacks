"""Smoke test for the real Expert-3 (LFCC-LCNN) adapter.

Skips cleanly when torchaudio is missing or the Hub checkpoint cannot be
fetched (offline CI), so the default dummy-only suite is unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torchaudio")

from config import Settings


def _load_expert():
    from experts.lfcc import LFCCLCNNExpert

    settings = Settings()
    try:
        return LFCCLCNNExpert(cache_dir=settings.model_cache_dir, device="cpu")
    except Exception as exc:  # network/torchaudio/checkpoint unavailable
        pytest.skip(f"LFCC expert unavailable in this environment: {exc}")


def test_lfcc_expert_returns_finite_logit_and_embedding():
    expert = _load_expert()
    rng = np.random.default_rng(0)
    window = rng.normal(0.0, 0.1, 64000).astype(np.float32)

    score = expert.score(window)

    assert np.isfinite(score["logit"])
    assert isinstance(score["logit"], float)
    assert score["embedding"] is not None
    assert len(score["embedding"]) == 128
    assert all(np.isfinite(v) for v in score["embedding"])
    assert score["model_version"] == "best_lfcc_lcnn"


def test_lfcc_expert_is_deterministic_in_eval():
    expert = _load_expert()
    window = np.random.default_rng(1).normal(0.0, 0.1, 64000).astype(np.float32)
    a = expert.score(window)["logit"]
    b = expert.score(window)["logit"]
    assert abs(a - b) < 1e-5


def test_lfcc_expert_handles_short_and_long_windows():
    expert = _load_expert()
    short = np.random.default_rng(2).normal(0.0, 0.1, 32000).astype(np.float32)
    long = np.random.default_rng(3).normal(0.0, 0.1, 96000).astype(np.float32)
    assert np.isfinite(expert.score(short)["logit"])
    assert np.isfinite(expert.score(long)["logit"])


def _load_mc_v3_expert():
    """Load the multi-corpus V3 checkpoint through the shared LFCC adapter."""
    from experts.lfcc import LFCCLCNNExpert

    settings = Settings()
    try:
        return LFCCLCNNExpert(
            cache_dir=settings.model_cache_dir,
            device="cpu",
            hub_key="mc_v3",
            name="mc_v3",
        )
    except Exception as exc:  # network/torchaudio/checkpoint unavailable
        pytest.skip(f"mc_v3 expert unavailable in this environment: {exc}")


def test_mc_v3_expert_returns_finite_logit_and_embedding():
    expert = _load_mc_v3_expert()
    rng = np.random.default_rng(0)
    window = rng.normal(0.0, 0.1, 64000).astype(np.float32)

    score = expert.score(window)

    assert np.isfinite(score["logit"])
    assert isinstance(score["logit"], float)
    assert score["embedding"] is not None
    assert len(score["embedding"]) == 128
    assert all(np.isfinite(v) for v in score["embedding"])
    assert score["model_version"] == "mc_v3"
