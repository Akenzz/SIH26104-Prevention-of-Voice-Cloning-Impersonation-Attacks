"""
Milestone 4C: Reproducible Report Generator.
Generates structured JSON and human-readable Markdown evaluation reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple

def generate_reports(
    output_base_path: str,
    dataset_name: str,
    split_name: str,
    model_name: str,
    model_version: str,
    metrics: Dict[str, Any],
    latency_stats: Dict[str, float],
    config_dict: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Generates both .json and .md evaluation reports.
    """
    report_data = {
        'dataset': dataset_name,
        'split': split_name,
        'model': {
            'name': model_name,
            'version': model_version
        },
        'metrics': metrics,
        'latency_ms': latency_stats,
        'config': config_dict
    }

    json_path = f"{output_base_path}.json"
    md_path = f"{output_base_path}.md"

    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)

    cm = metrics['confusion_matrix']
    md_content = f"""# Benchmark Report: {model_name} ({model_version})

## Metadata
- **Dataset:** {dataset_name}
- **Split:** {split_name}
- **Model:** {model_name} (v{model_version})

## Metrics (4A)
- **Equal Error Rate (EER):** {metrics['eer']*100:.2f}%
- **Operating Threshold:** {metrics['threshold']:.4f}
- **ROC-AUC:** {metrics['roc_auc']:.4f}
- **PR-AUC:** {metrics['pr_auc']:.4f}

## Confusion Matrix
- **True Negatives (Bonafide):** {cm['true_negatives']}
- **False Positives (False Alarm):** {cm['false_positives']}
- **False Negatives (Missed Spoof):** {cm['false_negatives']}
- **True Positives (Detected Spoof):** {cm['true_positives']}

## Latency Profile
- **Mean Latency:** {latency_stats['mean_ms']:.2f} ms
- **P95 Latency:** {latency_stats['p95_ms']:.2f} ms
- **P99 Latency:** {latency_stats['p99_ms']:.2f} ms

---
*Report generated automatically by Shared Data Pipeline Evaluation Harness (Task B).*
"""

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"[OK] Saved JSON report: {json_path}")
    print(f"[OK] Saved Markdown report: {md_path}")
    return json_path, md_path