"""
Milestone 4A: Metric functions for voice cloning evaluation.
Calculates EER, operating threshold, min tDCF, ROC-AUC, PR-AUC, and Confusion Matrix.
"""

import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Tuple


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> Tuple[float, float]:
    labels = np.concatenate([np.zeros_like(bonafide_scores), np.ones_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    threshold = float(thresholds[eer_idx])

    return eer, threshold


def compute_pr_auc(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    labels = np.concatenate([np.zeros_like(bonafide_scores), np.ones_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    pr_auc_score = float(auc(recall, precision))
    return pr_auc_score


def compute_confusion_metrics(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    threshold: float
) -> Dict[str, int]:
    labels = np.concatenate([np.zeros_like(bonafide_scores), np.ones_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }


def compute_full_suite(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> Dict:
    labels = np.concatenate([np.zeros_like(bonafide_scores), np.ones_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = compute_pr_auc(bonafide_scores, spoof_scores)
    cm = compute_confusion_metrics(bonafide_scores, spoof_scores, threshold)

    return {
        'eer': eer,
        'threshold': threshold,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'num_bonafide': len(bonafide_scores),
        'num_spoof': len(spoof_scores)
    }