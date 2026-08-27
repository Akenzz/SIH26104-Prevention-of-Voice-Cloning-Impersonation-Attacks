"""Optional logistic fusion over expert logits.

Fusion is off by default (FUSION_MODE=single). Weights live in artifacts/fusion.json
so they can be replaced by an offline logistic-regression fit on a locked dev set.
Do not hardcode “average the two models” as if that were a result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experts.protocol import Score


@dataclass
class FusionConfig:
    version: str
    bias: float
    weights: dict[str, float]


def load_fusion(path: Path) -> FusionConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = {str(k): float(v) for k, v in data.get("weights", {}).items()}
    return FusionConfig(
        version=str(data.get("version", "fusion-unversioned")),
        bias=float(data.get("bias", 0.0)),
        weights=weights,
    )


def fuse_logits(
    scores: dict[str, Score],
    config: FusionConfig,
    *,
    mode: str,
    single_expert: str = "",
) -> tuple[float, str]:
    """Return (fused_or_selected_logit, expert_used_label)."""
    if not scores:
        raise ValueError("no expert scores to fuse")

    if mode == "single":
        name = single_expert if single_expert in scores else next(iter(scores))
        return float(scores[name]["logit"]), name

    if mode != "fused":
        raise ValueError(f"unknown fusion mode {mode!r}")

    logit = config.bias
    used: list[str] = []
    for name, score in scores.items():
        weight = config.weights.get(name)
        if weight is None:
            continue
        logit += weight * float(score["logit"])
        used.append(name)
    if not used:
        name = next(iter(scores))
        return float(scores[name]["logit"]), name
    return float(logit), "+".join(used)
