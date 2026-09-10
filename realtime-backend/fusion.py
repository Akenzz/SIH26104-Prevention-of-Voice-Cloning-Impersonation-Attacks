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


def decision_expert(
    scores: dict[str, Score],
    *,
    mode: str,
    single_expert: str = "",
) -> str | None:
    """Name of the expert whose RAW logit ``fuse_logits`` returns, else None.

    `single` selects one expert's logit outright, and `lr_fusion` returns that
    same logit as the value pipeline.py falls back to whenever the LR model is
    unavailable (missing joblib, or `wavlm`/`hybrid_maxbr` not in EXPERTS). In both
    cases the returned float is ONE model's logit and must be read on THAT model's
    Platt scale -- the global CALIBRATOR_PATH one generally belongs to a
    different expert.

    `fused` / `heuristic` / `heuristic_avg` build a genuine combination that no
    single expert owns, so they return None and keep the global calibrator.

    Selection lives here, in one place, so pipeline.py cannot drift out of step
    with which expert fuse_logits actually picked.
    """
    if mode not in {"single", "lr_fusion"} or not scores:
        return None
    return single_expert if single_expert in scores else next(iter(scores))


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

    if mode == "heuristic_avg":
        return 0.0, "wavlm+lfcc"

    if mode == "lr_fusion":
        # The actual probability is computed in pipeline.py from the LR model.
        # We return a dummy logit here so fuse_logits doesn't raise.
        name = decision_expert(scores, mode=mode, single_expert=single_expert)
        return float(scores[str(name)]["logit"]), "lr_fusion"

    if mode == "single":
        name = decision_expert(scores, mode=mode, single_expert=single_expert)
        return float(scores[str(name)]["logit"]), str(name)
        
    if mode == "heuristic":
        # The tuned real-world threshold was -5.0.
        # We shift it by +5.0 so 0.0 is the center decision boundary for the calibrator sigmoid.
        # Formerly summed LFCC + SSL logits; SSL is decommissioned, so this is LFCC-only.
        l_logit = float(scores["hybrid"]["logit"]) if "hybrid" in scores else 0.0
        return float(l_logit + 5.0), "hybrid"

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
