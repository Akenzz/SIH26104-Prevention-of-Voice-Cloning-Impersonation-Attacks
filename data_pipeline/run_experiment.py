"""
Step 7: Complete Model Experiment Runner.
Evaluates a trained detector model against a dataset manifest and produces
a versioned, reproducible benchmark report (Milestone 4C).
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on Python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from data_pipeline.loader import ManifestAudioDataset
from data_pipeline.evaluation.adapter import ModelAdapter
from data_pipeline.evaluation.metrics import compute_full_suite
from data_pipeline.evaluation.reporter import generate_reports
# Add lfcc-detector folder to sys.path so models.detector resolves cleanly
lfcc_dir = repo_root / "lfcc-detector"
if str(lfcc_dir) not in sys.path:
    sys.path.insert(0, str(lfcc_dir))

from models.detector import LFCCLCNNDetector

def run_experiment(
    checkpoint_path: str,
    manifest_path: str,
    output_report_path: str,
    split: str = 'eval',
    model_name: str = 'LFCC-LCNN'
):
    print("=" * 60)
    print(f"RUNNING EXPERIMENT: {model_name} on {Path(manifest_path).name}")
    print("=" * 60)

    # 1. Load Detector (Model Contract)
    print(f"[1/4] Loading model checkpoint: {checkpoint_path}")
    detector = LFCCLCNNDetector(checkpoint_path=checkpoint_path if Path(checkpoint_path).exists() else None)
    adapter = ModelAdapter(detector, name=model_name, version=detector.model_version)

    # 2. Load Dataset Manifest
    print(f"[2/4] Loading dataset split '{split}' from {manifest_path}")
    dataset = ManifestAudioDataset(manifest_path, split=split)
    print(f"      Loaded {len(dataset)} evaluation samples")

    # 3. Batch Predict & Profile Latency
    print(f"[3/4] Running inference & measuring latency...")
    bonafide_scores = []
    spoof_scores = []
    latencies = []

    for i in range(len(dataset)):
        audio, label, meta = dataset[i]
        logit, embedding, latency_ms = adapter.predict_window(audio.numpy())

        latencies.append(latency_ms)
        if label == 0:
            bonafide_scores.append(logit)
        else:
            spoof_scores.append(logit)

    b_scores = np.array(bonafide_scores)
    s_scores = np.array(spoof_scores)

    # 4. Compute Metrics (4A) & Generate Reports (4C)
    print(f"[4/4] Computing full metric suite & saving reproducible report...")
    metrics = compute_full_suite(b_scores, s_scores)
    latency_stats = {
        'mean_ms': float(np.mean(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99))
    }

    config_dict = {
        'sample_rate': dataset.sample_rate,
        'window_sec': dataset.window_sec,
        'manifest': str(manifest_path),
        'checkpoint': str(checkpoint_path)
    }

    json_path, md_path = generate_reports(
        output_base_path=output_report_path,
        dataset_name=Path(manifest_path).stem,
        split_name=split,
        model_name=model_name,
        model_version=detector.model_version,
        metrics=metrics,
        latency_stats=latency_stats,
        config_dict=config_dict
    )

    print("\n" + "=" * 60)
    print(f"EXPERIMENT COMPLETE")
    print(f"  EER: {metrics['eer']*100:.2f}% | ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Mean Latency: {latency_stats['mean_ms']:.2f} ms")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run complete model benchmark experiment")
    parser.add_argument("--checkpoint", type=str, default="lfcc-detector/checkpoints/best_lfcc_lcnn.pth")
    parser.add_argument("--manifest", type=str, default="data_pipeline/dummy_dataset/manifest.csv")
    parser.add_argument("--output-report", type=str, default="data_pipeline/reports/lfcc_lcnn_eval")
    parser.add_argument("--split", type=str, default="eval")
    args = parser.parse_args()

    run_experiment(args.checkpoint, args.manifest, args.output_report, split=args.split)