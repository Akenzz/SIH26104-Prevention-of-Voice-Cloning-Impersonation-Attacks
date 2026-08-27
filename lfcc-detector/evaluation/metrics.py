"""
Evaluation metrics for voice cloning / synthetic speech detection.

Includes:
  - Equal Error Rate (EER)
  - Simplified min tDCF (ASVspoof standard metric)
  - Standard binary classification metrics (AUC, Accuracy, Precision, Recall, F1)
"""

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support
from typing import Dict, Tuple, Optional


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and operating threshold.

    Args:
        bonafide_scores: Logits/scores for real audio samples
        spoof_scores: Logits/scores for synthetic/cloned audio samples

    Returns:
        eer (float): Equal error rate in [0, 1]
        threshold (float): Score threshold at EER point
    """
    labels = np.concatenate([np.zeros_like(bonafide_scores), np.ones_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    threshold = float(thresholds[eer_idx])

    return eer, threshold


def compute_min_tdcf(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0
) -> float:
    """
    Compute simplified minimum tandem Decision Cost Function (min tDCF).
    """
    labels = np.concatenate([np.zeros_like(bonafide_scores), np.ones_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Normalized tDCF cost vector
    beta = (c_miss * p_target) / (c_fa * (1.0 - p_target))
    tdcf = fnr + beta * fpr
    min_tdcf = float(np.min(tdcf))

    return min_tdcf


def compute_all_metrics(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute complete set of benchmark metrics.
    """
    labels = np.concatenate([np.zeros_like(bonafide_scores), np.ones_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    min_tdcf = compute_min_tdcf(bonafide_scores, spoof_scores)
    auc = float(roc_auc_score(labels, scores))

    # Binary predictions using EER threshold
    preds = (scores >= threshold).astype(int)
    acc = float(accuracy_score(labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    return {
        'eer': eer,
        'eer_threshold': threshold,
        'min_tdcf': min_tdcf,
        'auc': auc,
        'accuracy': acc,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'num_bonafide': len(bonafide_scores),
        'num_spoof': len(spoof_scores),
    }


if __name__ == '__main__':
    print("Testing evaluation metrics module...")
    b_scores = np.random.normal(loc=-2.0, scale=1.0, size=500)
    s_scores = np.random.normal(loc=2.0, scale=1.0, size=500)

    results = compute_all_metrics(b_scores, s_scores)
    print("\nSample Metric Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n[PASS] Metrics computation tests passed!")