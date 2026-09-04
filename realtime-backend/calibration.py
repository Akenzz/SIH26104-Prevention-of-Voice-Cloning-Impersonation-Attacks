"""Logit → probability. Kept outside the experts.

Each expert has its own logit scale, so each needs its OWN calibrator; applying
one expert's Platt a/b to another's logits produces confident nonsense. The
decision band uses the calibrator for SINGLE_EXPERT (settings.calibrator_path);
`load_expert_calibrators` additionally gives every expert its own, so the UI can
report a meaningful probability per model. Do not silently rescale logits inside
the model adapters.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("realtime_backend.calibration")


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Calibrator:
    version: str
    kind: str
    a: float
    b: float

    def probability(self, logit: float) -> float:
        if self.kind in {"identity_sigmoid", "platt"}:
            return _sigmoid(self.a * float(logit) + self.b)
        raise ValueError(f"unsupported calibrator kind {self.kind!r}")


def load_calibrator(path: Path) -> Calibrator | Any:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = str(data.get("kind", "identity_sigmoid"))
    version = str(data.get("version", "calibrator-unversioned"))
    
    if kind == "platt_sklearn":
        import joblib
        import numpy as np
        joblib_path = Path(path).with_suffix(".joblib")
        model = joblib.load(joblib_path)
        
        class SklearnCalibrator:
            def __init__(self, model, version):
                self.model = model
                self.version = version
                self.kind = "platt_sklearn"

            def probability(self, logit: float) -> float:
                return float(self.model.predict_proba(np.array([[float(logit)]]))[0, 1])

        return SklearnCalibrator(model, version)

    return Calibrator(
        version=version,
        kind=kind,
        a=float(data.get("a", 1.0)),
        b=float(data.get("b", 0.0)),
    )


def load_expert_calibrators(
    mapping: dict[str, Path],
    experts: list[str],
    fallback: Calibrator | Any,
) -> dict[str, Calibrator | Any]:
    """One calibrator per expert, so each model reports its own probability.

    An expert with no mapped artifact — or whose artifact is missing on disk —
    falls back to the global calibrator and is logged loudly, because that
    probability is then on the wrong logit scale and must not be trusted.
    """
    out: dict[str, Calibrator | Any] = {}
    for name in experts:
        path = mapping.get(name)
        if path is None:
            logger.warning(
                "Expert %r has no entry in EXPERT_CALIBRATORS; falling back to the global "
                "calibrator. Its per-expert probability is on the wrong scale — do not trust it.",
                name,
            )
            out[name] = fallback
            continue
        if not Path(path).exists():
            logger.warning(
                "Calibrator artifact for expert %r not found at %s; falling back to the global "
                "calibrator. Its per-expert probability is on the wrong scale.",
                name,
                path,
            )
            out[name] = fallback
            continue
        out[name] = load_calibrator(Path(path))
        logger.info("Expert %s calibrator=%s", name, out[name].version)
    return out


def assert_calibrator_gates_match(
    mapping: dict[str, Path],
    loaded_experts: dict[str, Any],
) -> None:
    """Cross-check each calibrator's band gate against its expert's.

    A calibrator fitted on band-gated audio maps logits the ungated model never
    produces (and vice versa). Both directions are silent -- probabilities stay
    in [0,1] -- so mismatches are rejected at startup instead of quietly skewing
    every reported confidence. Only checks artifacts that declare a gate, so
    pre-existing calibrators are unaffected.
    """
    for name, expert in loaded_experts.items():
        path = mapping.get(name)
        if path is None or not Path(path).exists():
            continue
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        cal_gate = data.get("band_gate_hz")
        exp_gate = getattr(expert, "band_gate_hz", None)
        if cal_gate and not exp_gate:
            raise RuntimeError(
                f"Calibrator {Path(path).name} for expert {name!r} was fitted on audio "
                f"band-gated at {float(cal_gate):.0f} Hz, but the expert applies no gate. "
                f"Its probabilities would be meaningless."
            )
        if cal_gate and exp_gate and abs(float(cal_gate) - float(exp_gate)) > 1e-6:
            raise RuntimeError(
                f"Calibrator/expert band gate mismatch for {name!r}: calibrator fitted at "
                f"{float(cal_gate):.1f} Hz, expert applies {float(exp_gate):.1f} Hz."
            )
        if not cal_gate and exp_gate:
            logger.warning(
                "Expert %r applies a %.0f Hz band gate but its calibrator %s does not record "
                "one. If it was fitted on ungated audio its probabilities are on the wrong "
                "scale; refit with lfcc-detector/fit_calibrator_br.py.",
                name, float(exp_gate), Path(path).name,
            )


def assert_single_expert_calibrator(
    fusion_mode: str,
    single_expert: str,
    calibrator_path: Path,
    mapping: dict[str, Path],
) -> None:
    """In `single` mode, the DECISION calibrator must belong to SINGLE_EXPERT.

    pipeline.py computes the headline probability as
    `calibrator.probability(fused)`, where `calibrator` is the global one from
    CALIBRATOR_PATH and `fused` is SINGLE_EXPERT's raw logit. Point those two at
    different experts and every risk band is read off a foreign Platt scale --
    in-range, plausible, and wrong. The per-expert side cards are unaffected
    (they use EXPERT_CALIBRATORS), which is exactly what makes this hard to spot
    by eye: the headline disagrees with its own expert's card.

    Only fires for `single`; the fused modes derive the probability differently.
    """
    if fusion_mode != "single":
        return
    expected = mapping.get(single_expert)
    if expected is None:
        logger.warning(
            "FUSION_MODE=single with SINGLE_EXPERT=%r, which has no entry in "
            "EXPERT_CALIBRATORS -- cannot verify that CALIBRATOR_PATH matches it.",
            single_expert,
        )
        return
    if Path(calibrator_path).resolve() != Path(expected).resolve():
        raise RuntimeError(
            f"FUSION_MODE=single reads the decision band from CALIBRATOR_PATH="
            f"{Path(calibrator_path).name}, but SINGLE_EXPERT={single_expert!r} is "
            f"calibrated by {Path(expected).name}. The headline probability would be "
            f"{single_expert}'s logits on another expert's Platt scale. Set "
            f"CALIBRATOR_PATH={expected}"
        )
    logger.info(
        "Decision calibrator %s matches SINGLE_EXPERT=%s", Path(expected).name, single_expert
    )
