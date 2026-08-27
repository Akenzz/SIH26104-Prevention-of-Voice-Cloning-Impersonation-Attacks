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


def load_calibrator(path: Path) -> Calibrator:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Calibrator(
        version=str(data.get("version", "calibrator-unversioned")),
        kind=str(data.get("kind", "identity_sigmoid")),
        a=float(data.get("a", 1.0)),
        b=float(data.get("b", 0.0)),
    )
