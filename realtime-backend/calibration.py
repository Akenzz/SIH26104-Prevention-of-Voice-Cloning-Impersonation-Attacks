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
