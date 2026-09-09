from __future__ import annotations

from pathlib import Path

import pytest

from calibration import Calibrator, load_calibrator, load_expert_calibrators
from config import ARTIFACTS_DIR, EXPERT_CALIBRATORS
from fusion import fuse_logits, load_fusion
from policy import band, decide, load_policy
from smoothing import ExponentialMovingAverage


def test_file_summary_uses_mean_probability_across_windows():
    from server import summarize_file_results

    results = [
        {
            "start_time_sec": 0.0,
            "weighted_probability": 0.10,
            "per_expert_probability": {"wavlm": 0.10, "hybrid": 0.20, "ssl": 0.30},
        },
        {
            "start_time_sec": 1.0,
            "weighted_probability": 0.30,
            "per_expert_probability": {"wavlm": 0.30, "hybrid": 0.40, "ssl": 0.50},
        },
        {
            "start_time_sec": 2.0,
            "weighted_probability": 0.50,
            "per_expert_probability": {"wavlm": 0.50, "hybrid": 0.60, "ssl": 0.70},
        },
    ]

    summary = summarize_file_results(results, ["wavlm", "hybrid", "ssl"], policy=load_policy(ARTIFACTS_DIR / "policy.json"))

    assert summary["weighted_spoof_probability"] == pytest.approx(0.30)
    assert summary["experts"]["wavlm"]["probability"] == pytest.approx(0.30)
    assert summary["experts"]["ssl"]["probability"] == pytest.approx(0.50)
    assert summary["agreement"] == "majority_bonafide"
    assert summary["confidence_level"] == "medium"
    assert summary["peak_time_sec"] == 2.0


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
    """The identity kind must be a plain sigmoid (a=1, b=0)."""
    cal = Calibrator(version="test", kind="identity_sigmoid", a=1.0, b=0.0)
    assert abs(cal.probability(0.0) - 0.5) < 1e-9
    assert cal.probability(8.0) > 0.99
    assert cal.probability(-8.0) < 0.01


def test_shipped_calibrators_are_monotonic_in_logit():
    """Higher logit must mean higher spoof probability for every shipped artifact.

    Each expert's Platt fit has its own a/b, so the absolute values differ; the
    invariant that must never break is direction (bonafide=0, spoof=1).
    """
    for name, path in EXPERT_CALIBRATORS.items():
        if not Path(path).exists():
            continue
        cal = load_calibrator(Path(path))
        probs = [cal.probability(x) for x in (-8.0, -2.0, 0.0, 2.0, 8.0)]
        assert all(0.0 <= p <= 1.0 for p in probs), f"{name} produced out-of-range probability"
        assert probs == sorted(probs), f"{name} calibrator is not monotonic increasing in logit"


def test_per_expert_calibrators_are_distinct_and_loaded():
    """Both experts must get their OWN calibrator, not a shared fallback."""
    fallback = Calibrator(version="fallback", kind="identity_sigmoid", a=1.0, b=0.0)
    available = [n for n, p in EXPERT_CALIBRATORS.items() if Path(p).exists()]
    if len(available) < 2:
        pytest.skip("both calibrator artifacts must be present for this check")
    cals = load_expert_calibrators(EXPERT_CALIBRATORS, available, fallback)
    versions = {n: cals[n].version for n in available}
    assert "fallback" not in versions.values(), f"an expert fell back: {versions}"
    assert len(set(versions.values())) == len(available), f"shared calibrator: {versions}"


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


def test_band_matches_policy_thresholds():
    policy = load_policy(ARTIFACTS_DIR / "policy.json")
    assert band(None, policy) == "unavailable"
    assert band(policy.low_max - 0.01, policy) == "low"
    assert band(policy.low_max, policy) == "uncertain"
    assert band(policy.uncertain_max, policy) == "high"


def test_build_message_reports_each_expert_with_its_own_probability():
    """The contract the UI depends on: one entry per expert, each with its own
    probability, band and calibrator — not one number reused for both."""
    from messages import build_message

    policy = load_policy(ARTIFACTS_DIR / "policy.json")
    scores = {
        "wavlm": {"logit": 3.0, "embedding": None, "model_version": "wavlm-v1"},
        "hybrid": {"logit": -7.0, "embedding": None, "model_version": "hybrid_clean"},
    }
    cals = {
        "wavlm": Calibrator("cal-wavlm", "platt", 1.0, 0.0),
        "hybrid": Calibrator("cal-hybrid", "platt", 1.0, 0.0),
    }
    msg = build_message(
        sequence_number=1,
        risk_state="low",
        recommended_action="continue",
        smoothed_probability=0.02,
        fused_logit=-7.0,
        scores=scores,
        threshold_version=policy.version,
        calibrator_version="cal-hybrid",
        fusion_version="v1",
        fusion_mode="single",
        dropped_frames=False,
        audio_quality=None,
        window_index=1,
        latency_ms=5.0,
        expert_probabilities={"wavlm": 0.95, "hybrid": 0.01},
        expert_calibrators=cals,
        policy=policy,
    )
    assert set(msg["scores"]) == {"wavlm", "hybrid"}
    assert msg["scores"]["wavlm"]["probability"] == 0.95
    assert msg["scores"]["wavlm"]["risk_state"] == "high"
    assert msg["scores"]["hybrid"]["probability"] == 0.01
    assert msg["scores"]["hybrid"]["risk_state"] == "low"
    assert msg["scores"]["wavlm"]["calibrator_version"] == "cal-wavlm"
    assert msg["scores"]["hybrid"]["calibrator_version"] == "cal-hybrid"
    # the original contract fields must stay intact
    assert msg["raw_per_expert_scores"] == {"wavlm": 3.0, "hybrid": -7.0}
    assert msg["model_version"]["hybrid"] == "hybrid_clean"


def test_heuristic_avg_50_50_fusion_and_hybrid_mirroring():
    from pipeline import process_single_window
    from config import Settings
    import numpy as np

    class MockExpert:
        def __init__(self, logit: float, version: str):
            self.logit = logit
            self.model_version = version

        def score(self, window: np.ndarray):
            return {"logit": self.logit, "embedding": None, "model_version": self.model_version}

    settings = Settings(experts=["wavlm", "hybrid_maxbr"], fusion_mode="heuristic_avg")
    fusion = load_fusion(ARTIFACTS_DIR / "fusion.json")
    cal = Calibrator("identity", "identity_sigmoid", 1.0, 0.0)
    policy = load_policy(ARTIFACTS_DIR / "policy.json")

    t = np.linspace(0, 4, 64000, endpoint=False)
    window = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    experts = {
        "wavlm": MockExpert(2.0, "wavlm-v5"),
        "hybrid_maxbr": MockExpert(4.0, "hybrid_maxbr-v1"),
    }

    result = process_single_window(
        window=window,
        settings=settings,
        experts=experts,
        fusion=fusion,
        calibrator=cal,
        policy=policy,
        windows_scored=2,
    )

    # Both hybrid_maxbr and hybrid should be present in expert_probabilities
    assert "hybrid_maxbr" in result.expert_probabilities
    assert "hybrid" in result.expert_probabilities
    assert result.expert_probabilities["hybrid"] == result.expert_probabilities["hybrid_maxbr"]

    # Both should be present in scores
    assert "hybrid_maxbr" in result.scores
    assert "hybrid" in result.scores

    # 50/50 fusion: fused = (2.0 * 0.5 + 4.0 * 0.5) / 1.0 = 3.0
    assert result.fused_logit == pytest.approx(3.0)


def test_heuristic_avg_strong_agreement_override():
    from pipeline import process_single_window
    from config import Settings
    import numpy as np

    class MockExpert:
        def __init__(self, logit: float, version: str):
            self.logit = logit
            self.model_version = version

        def score(self, window: np.ndarray):
            return {"logit": self.logit, "embedding": None, "model_version": self.model_version}

    settings = Settings(experts=["wavlm", "hybrid_maxbr"], fusion_mode="heuristic_avg")
    fusion = load_fusion(ARTIFACTS_DIR / "fusion.json")
    cal = Calibrator("identity", "identity_sigmoid", 1.0, 0.0)
    policy = load_policy(ARTIFACTS_DIR / "policy.json")

    t = np.linspace(0, 4, 64000, endpoint=False)
    window = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # hybrid_maxbr: logit 2.0 -> sigmoid(2.0) = 0.8808 (>0.85)
    # wavlm: logit 0.5 -> sigmoid(0.5) = 0.6225 (>0.60)
    # 50/50 fused logit = 1.25 -> sigmoid(1.25) = 0.7773 (<0.80)
    # Override should kick in and boost to >= 0.80
    experts = {
        "wavlm": MockExpert(0.5, "wavlm-v5"),
        "hybrid_maxbr": MockExpert(2.0, "hybrid_maxbr-v1"),
    }

    result = process_single_window(
        window=window,
        settings=settings,
        experts=experts,
        fusion=fusion,
        calibrator=cal,
        policy=policy,
        windows_scored=2,
    )

    assert result.expert_probabilities["hybrid_maxbr"] > 0.85
    assert result.expert_probabilities["wavlm"] > 0.60
    assert result.probability >= 0.80


