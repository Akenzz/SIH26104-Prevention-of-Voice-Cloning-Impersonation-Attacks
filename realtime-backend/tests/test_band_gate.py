"""Regression tests for the 7 kHz parity band gate.

Every failure this file guards against is SILENT: probabilities stay in [0,1],
the stream keeps working, and only the verdict is wrong. That is why they are
pinned by tests rather than left to a startup log line.

The gate exists because the LFCC training corpus made the near-Nyquist band a
label proxy -- training resampled with librosa/soxr (brickwalls 7.9-8 kHz), this
backend resamples with scipy (does not), and the corpus's native sample rates were
split by label. See CALIBRATION-AND-RESAMPLING.md.

Parametrized over EVERY gated checkpoint (`hybrid_br`, `hybrid_maxbr`) so a new
gated model added to config.HUB_EXPERTS inherits the whole guard set instead of
being trusted. Skips cleanly when a checkpoint is not on this machine.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torchaudio")

from config import EXPERT_CALIBRATORS, HUB_EXPERTS, TARGET_SAMPLE_RATE, Settings
from experts.loader import _BAND_GATED_LFCC

WINDOW = 4 * TARGET_SAMPLE_RATE

# Parametrize over the gated experts the loader knows about, so this file cannot
# silently fall behind config.py.
GATED = sorted(_BAND_GATED_LFCC)


def _load(hub_key: str):
    from experts.lfcc import LFCCLCNNExpert

    settings = Settings()
    try:
        return LFCCLCNNExpert(
            cache_dir=settings.model_cache_dir,
            device="cpu",
            hub_key=hub_key,
            name=hub_key,
        )
    except Exception as exc:  # offline, or a local-only checkpoint absent here
        pytest.skip(f"{hub_key} unavailable in this environment: {exc}")


def _tone(freq_hz: float, amp: float = 0.3, n: int = WINDOW) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / TARGET_SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _speechlike(n: int = WINDOW) -> np.ndarray:
    """A broadband signal with a low-frequency fundamental, plus noise.

    Pure single tones are a poor probe: with almost all energy in one bin, the
    log-energy front-end is dominated by numerical floor effects rather than by
    the band being tested.
    """
    t = np.arange(n, dtype=np.float32) / TARGET_SAMPLE_RATE
    rng = np.random.default_rng(0)
    return (
        0.2 * np.sin(2 * np.pi * 220 * t)
        + 0.1 * np.sin(2 * np.pi * 1400 * t)
        + 0.02 * rng.normal(0.0, 1.0, n)
    ).astype(np.float32)


@pytest.mark.parametrize("hub_key", GATED)
def test_expert_declares_the_checkpoints_gate(hub_key):
    """The expert's applied cutoff must come from the checkpoint, not a constant."""
    import torch

    expert = _load(hub_key)
    blob = torch.load(str(expert.checkpoint_path), map_location="cpu", weights_only=False)
    declared = blob.get("band_gate_hz")
    assert declared, f"{hub_key} checkpoint declares no band_gate_hz"
    assert expert.band_gate_hz == pytest.approx(float(declared))
    # and it must be a usable cutoff, not just non-zero
    assert 0 < expert.band_gate_hz < TARGET_SAMPLE_RATE / 2


@pytest.mark.parametrize("hub_key", GATED)
def test_out_of_band_energy_barely_moves_the_verdict(hub_key):
    """Energy above the cutoff is removed before scoring, so it cannot vote.

    This is the property that makes the verdict resampler-invariant: the two
    resamplers disagree by 59.5 dB at 7.9-8 kHz, and this test asserts that
    disagreement is (almost) unreachable.

    The tolerance is 0.5 logits, NOT ~0, and that is deliberate. Ideal brickwall
    gating (zeroing rFFT bins) leaves ~1% time-domain ringing, and the LFCC
    front-end takes logs of per-frame band energies, so that residual is
    amplified into a small but nonzero logit shift. Measured on speech-like
    audio: hybrid_maxbr 0.03, hybrid_br 0.32, versus 23.5 (hybrid) and 9.2
    (hybrid_nc) ungated. Asserting < 1e-3 here would fail on a correctly gated
    model -- test_gated_experts_are_far_less_sensitive_than_ungated below is the
    assertion that actually has teeth.
    """
    expert = _load(hub_key)
    above = expert.band_gate_hz + 500.0
    assert above < TARGET_SAMPLE_RATE / 2, "need headroom above the cutoff"

    base = _speechlike()
    contaminated = base + _tone(above, amp=0.3)

    a = expert.score(base)["logit"]
    b = expert.score(contaminated)["logit"]
    assert abs(a - b) < 0.5, (
        f"{hub_key}: out-of-band tone at {above:.0f} Hz moved the logit "
        f"{a:.4f} -> {b:.4f}; the gate is not being applied"
    )


def test_gated_experts_are_far_less_sensitive_than_ungated():
    """The whole point of the gate, stated as a comparison.

    An absolute tolerance cannot distinguish "gate applied" from "model happens
    to be flat here". This measures the SAME contamination through a gated and an
    ungated expert and requires at least a 10x reduction -- the ungated experts
    move by 9-24 logits, the gated ones by <0.5, so the real margin is ~30-800x.
    """
    ungated = [k for k in ("hybrid", "hybrid_nc") if k in HUB_EXPERTS]
    base = _speechlike()

    def sensitivity(hub_key: str) -> float:
        expert = _load(hub_key)
        above = (expert.band_gate_hz or 7000.0) + 500.0
        a = expert.score(base)["logit"]
        b = expert.score(base + _tone(above, amp=0.3))["logit"]
        return abs(a - b)

    ungated_deltas = {}
    for key in ungated:
        try:
            ungated_deltas[key] = sensitivity(key)
        except Exception:
            continue
    if not ungated_deltas:
        pytest.skip("no ungated LFCC expert available for comparison")
    worst_ungated = max(ungated_deltas.values())

    for key in GATED:
        delta = sensitivity(key)
        assert delta * 10 < worst_ungated, (
            f"{key} shifted {delta:.4f} logits from out-of-band energy; the ungated "
            f"baseline shifts {worst_ungated:.4f} ({ungated_deltas}). Less than a 10x "
            f"reduction means the gate is not doing its job."
        )


@pytest.mark.parametrize("hub_key", GATED)
def test_in_band_energy_does_change_the_verdict(hub_key):
    """Negative control for the test above.

    Without this, `score()` returning a constant would pass
    test_out_of_band_energy_cannot_move_the_verdict trivially.
    """
    expert = _load(hub_key)
    a = expert.score(_tone(1000.0))["logit"]
    b = expert.score(_tone(1000.0) + _tone(2500.0, amp=0.3))["logit"]
    assert abs(a - b) > 1e-6, f"{hub_key}: in-band change had no effect; is score() constant?"


@pytest.mark.parametrize("hub_key", GATED)
def test_gate_is_idempotent(hub_key):
    """Pre-gating the audio must not change the score.

    Guards against a caller (or a future pipeline stage) applying the gate too,
    which would lowpass twice and shift every logit.
    """
    expert = _load(hub_key)
    window = np.random.default_rng(0).normal(0.0, 0.1, WINDOW).astype(np.float32)

    freqs = np.fft.rfftfreq(WINDOW, 1.0 / TARGET_SAMPLE_RATE)
    spec = np.fft.rfft(window)
    spec[freqs >= expert.band_gate_hz] = 0.0
    pre_gated = np.fft.irfft(spec, n=WINDOW).astype(np.float32)

    assert expert.score(window)["logit"] == pytest.approx(
        expert.score(pre_gated)["logit"], abs=1e-4
    )


@pytest.mark.parametrize("hub_key", GATED)
def test_calibrator_and_checkpoint_agree_on_the_cutoff(hub_key):
    """A calibrator fitted at one cutoff cannot score a model gated at another."""
    path = EXPERT_CALIBRATORS.get(hub_key)
    if path is None or not path.exists():
        pytest.skip(f"no calibrator artifact for {hub_key}")
    data = json.loads(path.read_text(encoding="utf-8"))
    cal_gate = data.get("band_gate_hz")
    assert cal_gate, f"{path.name} does not record band_gate_hz"

    expert = _load(hub_key)
    assert float(cal_gate) == pytest.approx(expert.band_gate_hz)
    # the calibrator must also name the checkpoint it was fitted against
    assert data.get("checkpoint") == HUB_EXPERTS[hub_key]["local_name"]


@pytest.mark.parametrize("hub_key", GATED)
def test_loader_guard_rejects_a_stripped_gate(hub_key):
    """_assert_band_gates_applied must raise when the gate is not applied.

    Simulates the real failure -- a checkpoint that declares a gate served by an
    expert that forgot it -- by blanking the loaded attribute.
    """
    from experts.loader import _assert_band_gates_applied

    expert = _load(hub_key)
    _assert_band_gates_applied({hub_key: expert})  # sanity: passes as loaded

    expert.band_gate_hz = None
    with pytest.raises(RuntimeError, match="band gate|TRAINED with"):
        _assert_band_gates_applied({hub_key: expert})


@pytest.mark.parametrize("hub_key", GATED)
def test_calibrator_guard_rejects_a_stripped_gate(hub_key):
    """assert_calibrator_gates_match must raise on a gated calibrator + ungated expert."""
    from calibration import assert_calibrator_gates_match

    path = EXPERT_CALIBRATORS.get(hub_key)
    if path is None or not path.exists():
        pytest.skip(f"no calibrator artifact for {hub_key}")

    expert = _load(hub_key)
    assert_calibrator_gates_match({hub_key: path}, {hub_key: expert})  # passes as loaded

    expert.band_gate_hz = None
    with pytest.raises(RuntimeError, match="band-gated|meaningless"):
        assert_calibrator_gates_match({hub_key: path}, {hub_key: expert})


def test_shipped_hybrid_expert_has_no_gate():
    """`hybrid` is the ungated baseline; the guards must not fire for it.

    If this starts failing, someone gated the shipped checkpoint without refitting
    calibrator_hybrid_clean.json -- which would make its probabilities invalid.
    """
    from calibration import assert_calibrator_gates_match
    from experts.loader import _assert_band_gates_applied

    expert = _load("hybrid")
    assert expert.band_gate_hz is None
    _assert_band_gates_applied({"hybrid": expert})
    assert_calibrator_gates_match(
        {"hybrid": EXPERT_CALIBRATORS["hybrid"]}, {"hybrid": expert}
    )
