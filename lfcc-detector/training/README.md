# Training Module

Scripts and configurations for training the LFCC-LCNN detector.

## Key Files

- `train.py`: Main training loop with logging and checkpointing
- `config.py`: Hyperparameters and training configuration
- `utils.py`: Training utilities (EER computation, early stopping, etc.)

## Quick Start

```bash
# Basic training run
python training/train.py \
    --manifest data/manifests/asvspoof19_train.csv \
    --epochs 25 \
    --batch-size 32 \
    --output-dir checkpoints/

# With validation monitoring
python training/train.py \
    --manifest data/manifests/combined_train.csv \
    --val-manifest data/manifests/combined_dev.csv \
    --epochs 30 \
    --early-stopping-patience 5
```

## Training Configuration

Default hyperparameters (see `config.py`):
- Learning rate: 1e-4 (Adam optimizer)
- Batch size: 32
- Epochs: 25
- Early stopping: 5 epochs patience on dev EER
- Checkpointing: Save best model + every 5 epochs

## Monitoring

Training logs include:
- Per-batch loss
- Per-epoch train/dev loss and EER
- Checkpoint saves with model version
