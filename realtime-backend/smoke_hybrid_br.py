"""Ship gate for hybrid_br: prove the whole serving path works before a demo.

The band gate is a silent failure mode -- a gated model served ungated still
produces in-range probabilities, it just reads spoofward on every real voice.
Two startup guards were added for exactly that (experts.loader
._assert_band_gates_applied and calibration.assert_calibrator_gates_match) and
neither had ever been executed. This runs them, then checks the things a passing
import does NOT prove: that the gate actually removes the band from the array the
model sees, that it is applied exactly once, that the calibrator maps this
model's real logits to sane probabilities, and that both guards FAIL when the
gate is removed.

Run from realtime-backend/:
  EXPERTS=hybrid_br FUSION_MODE=single SINGLE_EXPERT=hybrid_br python smoke_hybrid_br.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    from calibration import (
        assert_calibrator_gates_match,
        load_calibrator,
        load_expert_calibrators,
    )
    from config import EXPERT_CALIBRATORS, load_settings
    from experts.loader import load_experts

    settings = load_settings()
    print(f"[1] settings: experts={settings.experts} mode={settings.fusion_mode} "
          f"single={settings.single_expert}")
    check("hybrid_br is in EXPERTS", "hybrid_br" in settings.experts)

    print("\n[2] load_experts() -- runs _assert_band_gates_applied")
    experts = load_experts(settings)
    e = experts.get("hybrid_br")
    check("hybrid_br loaded", e is not None)
    if e is None:
        return 1
    check("expert reports a band gate", bool(getattr(e, "band_gate_hz", None)),
          f"band_gate_hz={getattr(e, 'band_gate_hz', None)}")
    check("gate is 7000 Hz", abs(float(e.band_gate_hz) - 7000.0) < 1e-6)

    print("\n[3] assert_calibrator_gates_match()")
    assert_calibrator_gates_match(EXPERT_CALIBRATORS, experts)
    cals = load_expert_calibrators(EXPERT_CALIBRATORS, settings.experts,
                                   load_calibrator(settings.calibrator_path))
    cal = cals["hybrid_br"]
    check("calibrator is the gated one, not the fallback",
          "hybrid_br" in str(getattr(cal, "version", "")), f"version={cal.version}")

    print("\n[4] the gate actually removes the band from what the model sees")
    sr, n = 16000, 64000
    t = np.arange(n) / sr
    # 7.5 kHz tone lives entirely inside the stop band; 1 kHz entirely inside the
    # pass band. If the gate is wired, only the first disappears.
    tone_hi = (0.3 * np.sin(2 * np.pi * 7500 * t)).astype(np.float32)
    tone_lo = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    def band_energy(x: np.ndarray, lo: float, hi: float) -> float:
        spec = np.abs(np.fft.rfft(x))
        f = np.fft.rfftfreq(len(x), 1.0 / sr)
        return float((spec[(f >= lo) & (f < hi)] ** 2).sum())

    gated_hi = e._apply_gate(tone_hi) if hasattr(e, "_apply_gate") else None
    if gated_hi is None:
        # No helper -- reproduce score()'s gate inline from the expert's own mask.
        spec = np.fft.rfft(tone_hi)
        spec[e._gate_mask] = 0.0
        gated_hi = np.fft.irfft(spec, n=tone_hi.size).astype(np.float32)
    kept = band_energy(gated_hi, 7000, 8000) / max(band_energy(tone_hi, 7000, 8000), 1e-30)
    check("7.5 kHz tone is removed", kept < 1e-6, f"residual {kept:.2e} of original")

    print("\n[5] score() runs and is deterministic + gate-consistent")
    rng = np.random.default_rng(0)
    speechlike = (tone_lo + 0.05 * rng.standard_normal(n)).astype(np.float32)
    # Score is a TypedDict (experts/protocol.py), not an object.
    logit_of = lambda x: float(e.score(x)["logit"])  # noqa: E731
    s1 = logit_of(speechlike)
    s2 = logit_of(speechlike)
    check("score() returns a finite logit", np.isfinite(s1), f"logit={s1:+.4f}")
    check("score() is deterministic", abs(s1 - s2) < 1e-5)
    # Pre-gating the input must be a no-op: gating twice == gating once. If score()
    # forgot the gate, adding out-of-band energy would move the logit instead.
    pre = np.fft.rfft(speechlike)
    pre[e._gate_mask] = 0.0
    pre_gated = np.fft.irfft(pre, n=speechlike.size).astype(np.float32)
    s3 = logit_of(pre_gated)
    check("gate is idempotent (applied exactly once)", abs(s1 - s3) < 1e-3,
          f"{s1:+.4f} vs pre-gated {s3:+.4f}")
    # Out-of-band energy the gate should erase must not change the verdict at all.
    with_hiss = (speechlike + tone_hi).astype(np.float32)
    s4 = logit_of(with_hiss)
    check("out-of-band energy does not move the verdict", abs(s1 - s4) < 1e-3,
          f"{s1:+.4f} vs +7.5kHz tone {s4:+.4f}")

    print("\n[6] calibrator maps this model's real logit range sanely")
    p_at = {z: cal.probability(z) for z in (-10.0, -1.0, 0.0, 1.35, 5.0, 10.0)}
    print("     " + "  ".join(f"logit {z:+.2f}->p {p:.3f}" for z, p in p_at.items()))
    check("probabilities stay in (0,1)", all(0.0 < p < 1.0 for p in p_at.values()))
    check("monotone increasing in logit",
          all(a < b for a, b in zip(list(p_at.values()), list(p_at.values())[1:])))
    # 1.3521 is the held-out EER threshold from fit_calibrator_br.py; p there
    # should sit near the middle of the band, not pinned at an extreme.
    check("EER-threshold logit lands mid-band", 0.35 < p_at[1.35] < 0.65,
          f"p(1.35)={p_at[1.35]:.3f}")

    print("\n[7] both guards FAIL when the gate is stripped (guards are real)")
    saved_gate, saved_mask = e.band_gate_hz, e._gate_mask
    try:
        e.band_gate_hz, e._gate_mask = None, None
        from experts.loader import _assert_band_gates_applied
        raised = False
        try:
            _assert_band_gates_applied({"hybrid_br": e})
        except RuntimeError:
            raised = True
        check("loader guard rejects a stripped gate", raised)
        raised = False
        try:
            assert_calibrator_gates_match(EXPERT_CALIBRATORS, {"hybrid_br": e})
        except RuntimeError:
            raised = True
        check("calibrator guard rejects a stripped gate", raised)
    finally:
        e.band_gate_hz, e._gate_mask = saved_gate, saved_mask
    check("gate restored after the negative test", e.band_gate_hz == saved_gate)

    print()
    if FAILS:
        print(f"[FAIL] {len(FAILS)} check(s) failed: {FAILS}")
        return 1
    print("[OK] hybrid_br serving path verified end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
