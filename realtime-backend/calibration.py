"""Logit → probability. Kept outside the experts.

Default artifact is an identity sigmoid so the service runs before a real
Platt / isotonic fit exists. Replace artifacts/calibrator.json; do not
silently rescale logits in the model adapters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    scope: dict[str, Any] | None = None

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
            def __init__(self, model, version, scope):
                self.model = model
                self.version = version
                self.scope = scope
                
            def probability(self, logit: float) -> float:
                return float(self.model.predict_proba(np.array([[float(logit)]]))[0, 1])
                
        return SklearnCalibrator(model, version, data.get("scope"))

    return Calibrator(
        version=version,
        kind=kind,
        a=float(data.get("a", 1.0)),
        b=float(data.get("b", 0.0)),
        scope=data.get("scope"),
    )
