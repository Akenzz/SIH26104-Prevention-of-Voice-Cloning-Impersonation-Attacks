"""Smoke test for the ProsodyExpert adapter.

Skips cleanly when librosa or the trained artifact is unavailable, so the
default suite is unaffected in environments where the prosody model hasn't
been trained yet.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("librosa")
pytest.importorskip("sklearn")


def _load_expert():
    # The realtime backend now ships only the wavlm + hybrid experts, so this
    # constructs the adapter directly instead of going through its loader.
    import sys
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[1]
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    try:
        from expert.prosody_expert import ProsodyExpert

        return ProsodyExpert()
    except Exception as exc:  # artifact missing / package not on path
        pytest.skip(f"Prosody expert unavailable in this environment: {exc}")


def test_prosody_expert_returns_finite_logit_and_embedding():
    expert = _load_expert()
    rng = np.random.default_rng(0)
    # a voiced-ish signal so feature extraction has something to chew on
    t = np.linspace(0, 4, 64000, endpoint=False)
    window = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    window += rng.normal(0.0, 0.01, 64000).astype(np.float32)

    score = expert.score(window)

    assert np.isfinite(score["logit"])
    assert isinstance(score["logit"], float)
    assert score["embedding"] is not None
    assert all(np.isfinite(v) for v in score["embedding"])
    assert score["model_version"] == "prosody-lr-v1"


def test_prosody_expert_is_deterministic():
    expert = _load_expert()
    t = np.linspace(0, 4, 64000, endpoint=False)
    window = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    a = expert.score(window)["logit"]
    b = expert.score(window)["logit"]
    assert abs(a - b) < 1e-5


def test_prosody_expert_handles_short_and_long_and_silent_windows():
    expert = _load_expert()
    short = np.random.default_rng(2).normal(0.0, 0.05, 32000).astype(np.float32)
    long = np.random.default_rng(3).normal(0.0, 0.05, 96000).astype(np.float32)
    silent = np.zeros(64000, dtype=np.float32)
    assert np.isfinite(expert.score(short)["logit"])
    assert np.isfinite(expert.score(long)["logit"])
    assert np.isfinite(expert.score(silent)["logit"])
