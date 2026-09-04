"""
Shared scoring metrics for the prosody expert.
==============================================
Lives at the package root, not under evaluation/, on purpose: BOTH prosody-detector
and lfcc-detector contain a regular package named `evaluation`, and the sys.path
order every prosody script uses puts lfcc-detector first (so that
`evaluation.metrics.compute_eer` resolves to the repo's canonical EER). That makes
`from evaluation.X import ...` unusable for prosody's own helpers — hence a
top-level module instead.
"""
from __future__ import annotations

import numpy as np


def eer(bona: np.ndarray, spoof: np.ndarray) -> float:
    """EER via the repo's own implementation, so numbers match RESULTS.md."""
    from evaluation.metrics import compute_eer

    return float(compute_eer(np.asarray(bona), np.asarray(spoof))[0])


def auc(bona: np.ndarray, spoof: np.ndarray) -> float:
    """P(spoof score > bonafide score), ties counted as 0.5 — Mann-Whitney AUC.

    Ranks are tie-averaged so a degenerate all-equal case returns exactly 0.500
    instead of drifting to 0 or 1 depending on concatenation order.
    """
    bona, spoof = np.asarray(bona, float), np.asarray(spoof, float)
    scores = np.concatenate([bona, spoof])
    ranks = np.argsort(np.argsort(scores)) + 1.0
    order = np.argsort(scores)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1
    n_b, n_s = len(bona), len(spoof)
    r_spoof = float(np.sum(ranks[n_b:]))
    return (r_spoof - n_s * (n_s + 1) / 2.0) / (n_b * n_s)


def eer_ci(bona: np.ndarray, spoof: np.ndarray, n_boot: int = 300,
           seed: int = 0) -> tuple[float, float]:
    """90% bootstrap interval on EER.

    With ~100-500 clips per slice the point estimate moves several points on
    resampling, so a bare gap between two slices must not be read as a real
    difference without this.
    """
    bona, spoof = np.asarray(bona, float), np.asarray(spoof, float)
    rng = np.random.default_rng(seed)
    vals = [eer(bona[rng.integers(0, len(bona), len(bona))],
                spoof[rng.integers(0, len(spoof), len(spoof))])
            for _ in range(n_boot)]
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def threshold_at_far(bona: np.ndarray, far: float) -> float:
    """Score threshold giving `far` false-alarm rate on this bonafide pool.

    Deployment needs one fixed threshold, so the slice-by-slice EER (which moves
    its own threshold per slice) can hide a score offset between corpora. Pair
    this with `detection_at` to expose that.
    """
    return float(np.percentile(np.asarray(bona, float), 100.0 * (1.0 - far)))


def detection_at(spoof: np.ndarray, thr: float) -> float:
    """Fraction of spoof clips scoring above a FIXED threshold."""
    return float(np.mean(np.asarray(spoof, float) > thr))
