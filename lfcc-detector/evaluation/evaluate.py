"""
Evaluation CLI script for LFCC-LCNN detector.
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import AudioDataset, collate_fn
from models.detector import LFCCLCNNDetector
from evaluation.metrics import compute_all_metrics


def evaluate_checkpoint(checkpoint_path: str, manifest_path: str, batch_size: int = 16):
    """
    Run full evaluation of a model checkpoint against a manifest CSV.
    """
    detector = LFCCLCNNDetector(checkpoint_path=checkpoint_path)

    dataset = AudioDataset(manifest_path, split=None, window_sec=4.0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    bonafide_scores = []
    spoof_scores = []

    print(f"\nEvaluating {len(dataset)} samples...")

    for audios, labels, _ in loader:
        logits, _ = detector.batch_forward(audios)

        # Batch forward returns 0D/1D array for single sample batch
        if logits.ndim == 0:
            logits = np.array([logits.item()])

        for logit, label in zip(logits, labels.tolist()):
            if label == 0:
                bonafide_scores.append(logit)
            else:
                spoof_scores.append(logit)

    b_scores = np.array(bonafide_scores)
    s_scores = np.array(spoof_scores)

    metrics = compute_all_metrics(b_scores, s_scores)

    print("\n" + "="*50)
    print(f"EVALUATION REPORT: {Path(checkpoint_path).name}")
    print("="*50)
    print(f"  Samples:          {len(dataset)} (Bonafide: {metrics['num_bonafide']}, Spoof: {metrics['num_spoof']})")
    print(f"  Equal Error Rate: {metrics['eer']*100:.2f}%")
    print(f"  EER Threshold:    {metrics['eer_threshold']:.4f}")
    print(f"  min tDCF:         {metrics['min_tdcf']:.4f}")
    print(f"  ROC-AUC:          {metrics['auc']:.4f}")
    print(f"  Accuracy:         {metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score:         {metrics['f1_score']:.4f}")
    print("="*50)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate LFCC-LCNN voice cloning detector")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_lfcc_lcnn.pth")
    parser.add_argument("--manifest", type=str, default="../data_pipeline/manifests/asvspoof19_eval.csv")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    metrics = evaluate_checkpoint(args.checkpoint, args.manifest)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()