# LFCC-LCNN Detector (Expert 3)

**Owner:** Task D  
**Purpose:** Second expert model for voice clone detection using hand-crafted features (LFCC) and Light CNN architecture.

## What This Module Does

This is Expert 3 in the multi-expert ensemble. It detects synthetic/cloned speech using:
- **LFCC (Linear Frequency Cepstral Coefficients)**: Spectral features that preserve high-frequency artifacts where voice cloning leaves traces
- **LCNN (Light CNN with Max-Feature-Map)**: Small CNN with MFM activations, proven architecture from ASVspoof baselines

## Setup

```bash
cd lfcc-detector
pip install -r requirements.txt
```

## Interface Contract

This module exposes one primary interface for the backend to call:

```python
from models.detector import LFCCLCNNDetector

detector = LFCCLCNNDetector(checkpoint_path='checkpoints/best_model.pth')
logit, embedding = detector.forward(audio_window)
```

- **Input**: `audio_window` — numpy array or torch tensor, shape `(samples,)`, 16kHz mono, ~4 seconds
- **Output**: 
  - `logit` (float): Spoof score, higher = more likely fake
  - `embedding` (128-dim numpy array): Feature vector for fusion gate

See `contracts.md` for full specification.

## Project Structure

```
lfcc-detector/
├── data/              # Data loading and manifest handling
├── models/            # LFCC-LCNN architecture and inference wrapper
├── training/          # Training scripts and configs
├── evaluation/        # Metrics, EER computation, leakage checks
├── checkpoints/       # Saved models (gitignored)
└── notebooks/         # Exploratory analysis (optional)
```

## Quick Start

### Phase 1: Run Pretrained Baseline
```bash
# Download ASVspoof 2019 LA dataset first
python evaluation/evaluate.py --checkpoint pretrained/asvspoof_baseline.pth --data data/asvspoof19_eval.csv
```

### Phase 2: Train on Your Data
```bash
python training/train.py --manifest data/manifest_combined.csv --epochs 25
```

### Phase 3: Evaluate
```bash
python evaluation/evaluate.py --checkpoint checkpoints/best_model.pth --test-splits asvspoof_eval,in_the_wild,indicfake
```

## Current Status

- [ ] Phase 0: Setup and contracts (Aug 26-27)
- [ ] Phase 1: Run pretrained baseline (Aug 28-29)
- [ ] Phase 2: Adapt to manifest (Aug 30-31)
- [ ] Phase 3: Train on combined data (Sep 1-3)
- [ ] Phase 4: Implement model contract (Sep 4-5)
- [ ] Phase 5: Evaluation & ablation (Sep 6-8)
- [ ] Phase 6: Fusion integration (Sep 9-10)
- [ ] Phase 7: Demo readiness (Sep 11-12)

## References

- [ASVspoof 2021 LFCC-LCNN Baseline](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-LFCC-LCNN)
- [ASVspoof 2019 Challenge](https://www.asvspoof.org/asvspoof2019)
- Kartik's Implementation Plan: `../docs/kartik_implementation_plan.md`
