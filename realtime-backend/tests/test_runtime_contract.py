from types import SimpleNamespace

import pytest

from config import Settings
from fusion import FusionConfig
from runtime_contract import validate_runtime_contract


def _calibrator(experts, fusion_mode="single"):
    return SimpleNamespace(
        scope={
            "experts": experts,
            "fusion_mode": fusion_mode,
            "target_sample_rate": 16000,
            "window_sec": 4.0,
        }
    )


def test_real_expert_requires_matching_calibrator_scope():
    settings = Settings(experts=["wavlm"], fusion_mode="single")
    fusion = FusionConfig("unused", 0.0, {})
    validate_runtime_contract(settings, _calibrator(["wavlm"]), fusion)

    with pytest.raises(ValueError, match="expects experts"):
        validate_runtime_contract(settings, _calibrator(["lfcc"]), fusion)


def test_dummy_requires_explicit_opt_in():
    fusion = FusionConfig("unused", 0.0, {})
    with pytest.raises(ValueError, match="DummyExpert is disabled"):
        validate_runtime_contract(Settings(experts=["dummy"]), _calibrator(["dummy"]), fusion)

    validate_runtime_contract(
        Settings(experts=["dummy"], allow_dummy=True), _calibrator(["dummy"]), fusion
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_runtime_contract(
            Settings(experts=["dummy", "wavlm"], allow_dummy=True), _calibrator(["dummy"]), fusion
        )


def test_unfitted_fusion_is_rejected():
    settings = Settings(experts=["wavlm", "lfcc"], fusion_mode="fused")
    fusion = FusionConfig("identity", 0.0, {"wavlm": 1.0, "lfcc": 1.0}, {"fitted": False})
    with pytest.raises(ValueError, match="fitted fusion artifact"):
        validate_runtime_contract(settings, _calibrator(["wavlm", "lfcc"], "fused"), fusion)
