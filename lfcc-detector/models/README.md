# Models Module

LFCC-LCNN architecture and inference wrapper.

## Key Files

- `lfcc_lcnn.py`: PyTorch model architecture (adapted from ASVspoof baseline)
- `detector.py`: Inference wrapper that implements the contract interface
- `features.py`: LFCC feature extraction utilities

## Architecture Overview

**LFCC-LCNN** combines:
1. **LFCC extraction**: Linear frequency cepstral coefficients (preserves high-freq artifacts)
2. **LCNN backbone**: Light CNN with Max-Feature-Map (MFM) activations
3. **Binary classifier head**: Outputs logit (bonafide vs spoof)

## Usage

```python
from models.detector import LFCCLCNNDetector

# Load trained model
detector = LFCCLCNNDetector(checkpoint_path='checkpoints/best_model.pth')

# Inference on 4-second audio window
import numpy as np
audio = np.random.randn(64000).astype(np.float32)  # 4s at 16kHz
logit, embedding = detector.forward(audio)

print(f"Spoof score: {logit:.3f}")  # Higher = more likely fake
print(f"Embedding shape: {embedding.shape}")  # (128,)
```

## Model Contract

See `../contracts.md` for full interface specification.
