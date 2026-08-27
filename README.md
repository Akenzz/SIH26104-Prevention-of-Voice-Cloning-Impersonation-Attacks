# Expert 1 — WavLM Deepfake Speech Detector

Binary classifier (real vs. AI-generated/cloned speech) built on a **frozen** `microsoft/wavlm-base-plus` backbone with a lightweight trainable linear head.

## Label Convention — Never Flip This

| String | Integer | Meaning |
|--------|---------|---------|
| `bonafide` | **0** | Real human speech |
| `spoof` | **1** | AI-generated / voice-cloned / fake speech |

Higher model logit = more evidence of **fake** speech.

---

## File Overview

| File | Purpose |
|------|---------|
| `dataset.py` | `SpeechDataset` — reads manifest CSV, resamples to 16 kHz, pads/crops to fixed window |
| `model.py` | `WavLMClassifier` + `score()` API — frozen backbone + trainable head |
| `train.py` | Training loop — BCEWithLogitsLoss, AdamW on head only, saves best checkpoint |
| `evaluate.py` | Loads checkpoint, computes EER and accuracy on the test split |
| `generate_synthetic_data.py` | Creates `data/audio/*.wav` + `data/manifest.csv` for end-to-end testing |
| `requirements.txt` | Python dependencies |

---

## Quick Start (Synthetic Data — Run Today)

> All commands are run from the **project root** (`/home/akenzz/sih/project`).

```bash
# 1. Activate your virtualenv (if using one)
source .venv/bin/activate          # or: conda activate your-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic placeholder audio + manifest
#    TODO: Skip this step when using your real dataset manifest
python expert1/generate_synthetic_data.py

# 4. Train the model (saves best checkpoint to expert1/checkpoints/best_model.pt)
python -m expert1.train --epochs 10

# 5. Evaluate on the test split (prints Accuracy + EER)
python -m expert1.evaluate
```

### Optional CLI arguments

```bash
# Custom manifest or epochs
python -m expert1.train    --manifest /path/to/real_manifest.csv --epochs 20
python -m expert1.evaluate --manifest /path/to/real_manifest.csv --checkpoint expert1/checkpoints/best_model.pt
```

### Using the score() API from another module

```python
import numpy as np
from expert1.model import load_model, score

# Load once at startup
load_model("expert1/checkpoints/best_model.pt")

# Score a 4-second audio window (64000 samples @ 16 kHz)
audio_np = np.zeros(64000, dtype=np.float32)   # replace with real audio
result = score(audio_np)
# result = {
#   "logit"        : float,        # higher = more likely fake/spoof
#   "embedding"    : list[float],  # 768-dim WavLM representation
#   "model_version": "wavlm-base-plus-v1",
# }
```

---

## Swapping in Your Real Dataset

> **TODO (one change only):** Replace `data/manifest.csv` with a CSV pointing at your
> real audio files — or pass `--manifest /path/to/your_manifest.csv` to `train.py`
> and `evaluate.py`. No other code changes are needed.

The manifest must have **exactly these columns**:

```
path, label, split, source_dataset, speaker_id, utterance_id,
generator_id, language, codec, duration_s, license, consent
```

- `label` must be the string `"bonafide"` or `"spoof"` (exact match, lower-case)
- `split` must be `"train"`, `"dev"`, or `"test"`
- `path` should be the **absolute** path to the WAV file (relative paths work too,
  relative to the directory you run the script from)

### ASVspoof 2019 LA example

Generate a manifest from the ASVspoof2019 LA protocol files:

```python
# Pseudocode — adapt paths to your local ASVspoof installation
# TODO: fill in your actual ASVspoof paths here
import pandas as pd, glob, os

ASVSPOOFDIR = "/data/ASVspoof2019/LA"
rows = []
for split, proto_file in [
    ("train", f"{ASVSPOOFDIR}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"),
    ("dev",   f"{ASVSPOOFDIR}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"),
    ("test",  f"{ASVSPOOFDIR}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"),
]:
    with open(proto_file) as f:
        for line in f:
            parts = line.strip().split()
            speaker_id, utt_id, _, system_id, label_str = parts
            audio_path = os.path.join(
                ASVSPOOFDIR, f"ASVspoof2019_LA_{split}", "flac", f"{utt_id}.flac"
            )
            rows.append({
                "path": audio_path, "label": label_str.lower(),
                "split": split, "source_dataset": "asvspoof2019_la",
                "speaker_id": speaker_id, "utterance_id": utt_id,
                "generator_id": system_id, "language": "en",
                "codec": "flac", "duration_s": -1,
                "license": "asvspoof2019", "consent": "research"
            })
pd.DataFrame(rows).to_csv("data/asvspoof2019_la_manifest.csv", index=False)
```

---

## Using the `score()` API

Other modules should call `score()` — this is the stable contract:

```python
import numpy as np
from model import load_model, score

# Load once (at startup)
load_model("checkpoints/best_model.pt")

# Call for each audio window
audio_np = np.zeros(64000, dtype=np.float32)   # 4 s @ 16 kHz
result = score(audio_np)

# result = {
#   "logit"        : float,        # higher = more likely fake/spoof
#   "embedding"    : list[float],  # 768-dim WavLM representation
#   "model_version": "wavlm-base-plus-v1",
# }
```

---

## Architecture

```
Input waveform (B, 64000)
        │
        ▼
WavLMModel (FROZEN — no grad)
        │  last_hidden_state  (B, T, 768)
        │
        ▼
  Mean pooling across T  →  (B, 768)
        │
        ▼
  Linear(768 → 256) + GELU + Dropout(0.1)
        │
        ▼
  Linear(256 → 1)   →  logit  (B, 1)
```

- **Backbone**: `microsoft/wavlm-base-plus` (~94 M params, all frozen)
- **Head**: ~197 K trainable params
- **Loss**: `BCEWithLogitsLoss`
- **Optimiser**: AdamW on head params only
- **Primary metric**: EER (Equal Error Rate) — lower is better; random chance ≈ 50 %

---

## What's NOT in this version (by design)

- ❌ Attention pooling / specaugment / data augmentation
- ❌ Backbone fine-tuning / unfreezing
- ❌ Calibration / probability scaling
- ❌ Fusion with other models
- ❌ Streaming / real-time inference (handled by the rest of the project)

These are deferred to later iterations.