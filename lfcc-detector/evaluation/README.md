# Evaluation Module

Scripts for evaluating model performance, computing metrics, and checking data integrity.

## Key Files

- `evaluate.py`: Main evaluation script (EER, confusion matrix, per-condition breakdown)
- `leakage_check.py`: Data sanity checks (speaker/utterance leakage detection)
- `metrics.py`: Metric computation utilities (EER, ROC, PR curves)
- `test_contract.py`: Contract compliance tests

## Quick Start

```bash
# Evaluate on test set
python evaluation/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --manifest data/manifests/asvspoof19_eval.csv \
    --output results/asvspoof19_eval.json

# Check for data leakage
python evaluation/leakage_check.py \
    --manifest data/manifests/combined.csv

# Test contract compliance
python evaluation/test_contract.py \
    --checkpoint checkpoints/best_model.pth
```

## Metrics Reported

- **EER (Equal Error Rate)**: Primary metric
- **Confusion matrix**: At fixed threshold from dev set
- **Per-condition breakdown**: By language, generator, codec
- **ROC/PR curves**: Saved as images

## Expected Output

```json
{
  "overall_eer": 0.08,
  "conditions": {
    "asvspoof_eval": {"eer": 0.07, "precision": 0.92, "recall": 0.91},
    "in_the_wild": {"eer": 0.12, "precision": 0.88, "recall": 0.85},
    "indicfake_hindi": {"eer": 0.09, "precision": 0.90, "recall": 0.89}
  },
  "model_version": "lfcc_lcnn_v1_20260826"
}
```
