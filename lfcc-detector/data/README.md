# Data Module

This module handles all data loading, preprocessing, and manifest management for LFCC-LCNN training and evaluation.

## Key Files

- `dataset.py`: PyTorch Dataset class that loads audio from manifest CSV
- `manifest_utils.py`: Functions to create/validate/merge manifests
- `augmentation.py`: Codec degradation, noise mixing, RIR convolution

## Manifest Format

Expected CSV columns:
```
path,label,split,source_dataset,speaker_id,utterance_id,generator_id,language,duration_s
```

- `label`: exactly "bonafide" or "spoof"
- `split`: "train", "dev", or "eval"
- `path`: relative or absolute path to audio file

## Usage

```python
from data.dataset import AudioDataset

# Load training data
train_ds = AudioDataset(
    manifest_path='data/manifest_asvspoof19.csv',
    split='train',
    window_sec=4.0
)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
```

## Setup Instructions

1. Download datasets (ASVspoof 2019 LA, etc.)
2. Place raw audio in `data/raw/` (gitignored)
3. Run manifest generation scripts
4. Manifests are saved in `data/manifests/`
