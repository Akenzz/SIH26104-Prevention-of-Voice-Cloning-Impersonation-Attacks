from __future__ import annotations

from calibration import load_calibrator
from config import ARTIFACTS_DIR
from fusion import fuse_logits, load_fusion
from policy import decide, load_policy
from smoothing import ExponentialMovingAverage


def test_single_mode_ignores_second_expert():
    fusion = load_fusion(ARTIFACTS_DIR / "fusion.json")
    scores = {
        "dummy": {"logit": 2.0, "embedding": None, "model_version": "dummy-v0"},
        "lfcc": {"logit": -9.0, "embedding": None, "model_version": "lfcc"},
    }
    logit, used = fuse_logits(scores, fusion, mode="single", single_expert="dummy")
    assert used == "dummy"
    assert logit == 2.0


def test_fused_mode_uses_artifact_weights():
    fusion = load_fusion(ARTIFACTS_DIR / "fusion.json")
    scores = {
        "dummy": {"logit": 1.0, "embedding": None, "model_version": "dummy-v0"},
        "lfcc": {"logit": 1.0, "embedding": None, "model_version": "lfcc"},
    }
    logit, used = fuse_logits(scores, fusion, mode="fused")
    assert "dummy" in used and "lfcc" in used
    assert abs(logit - 2.0) < 1e-9


def test_identity_calibrator_is_sigmoid():
    cal = load_calibrator(ARTIFACTS_DIR / "calibrator.json")
    assert abs(cal.probability(0.0) - 0.5) < 1e-9
    assert cal.probability(8.0) > 0.99
    assert cal.probability(-8.0) < 0.01


def test_ema_converges_and_resets():
    ema = ExponentialMovingAverage(0.5)
    v1 = ema.update(1.0)
    v2 = ema.update(0.0)
    assert v1 == 1.0
    assert abs(v2 - 0.5) < 1e-9
    ema.reset()
    assert ema.value is None


def test_policy_collecting_then_bands():
    policy = load_policy(ARTIFACTS_DIR / "policy.json")
    state, _, flag = decide(
        quality_ok=True,
        quality_reason=None,
        windows_scored=1,
        smoothed_probability=0.9,
        config=policy,
    )
    assert state == "collecting"
    assert flag is None
    low, _, _ = decide(
        quality_ok=True,
        quality_reason=None,
        windows_scored=5,
        smoothed_probability=0.1,
        config=policy,
    )
    assert low == "low"
    high, _, _ = decide(
        quality_ok=True,
        quality_reason=None,
        windows_scored=5,
        smoothed_probability=0.9,
        config=policy,
    )
    assert high == "high"
    bad, action, reason = decide(
        quality_ok=False,
        quality_reason="silence",
        windows_scored=5,
        smoothed_probability=0.01,
        config=policy,
    )
    assert bad == "unavailable"
    assert reason == "silence"
    assert "Do not treat" in action or "failed" in action.lower()
