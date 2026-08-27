# data_pipeline — Task B: Data Pipeline, Manifest & Evaluation Harness

> **Status:** Complete. All ASVspoof 2019 LA manifests are generated, zero-leakage verified, and the evaluation harness is ready to use.
>
> This folder is the **single source of truth** every model trains and is measured against. Person A (WavLM), Person D (LFCC-LCNN), and Person K (AASIST) all consume the same manifests and evaluation harness from here.

---

## Quick Start for Person A (WavLM)

```python
# 1. Load the training dataset
from data_pipeline.loader import ManifestAudioDataset
from data_pipeline.augmentation import TrainingAugmentationPipeline
from torch.utils.data import DataLoader

aug = TrainingAugmentationPipeline(apply_noise=True)  # optional

train_ds = ManifestAudioDataset(
    manifest_path="data_pipeline/manifests/asvspoof19_train.csv",
    split="train",
    window_sec=4.0,
    sample_rate=16000,
    augmentation_pipeline=aug,   # None for clean baseline
)
dev_ds = ManifestAudioDataset(
    manifest_path="data_pipeline/manifests/asvspoof19_dev.csv",
    split="dev",
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
dev_loader   = DataLoader(dev_ds,   batch_size=32, shuffle=False, num_workers=0)

# 2. Each batch returns (audio, label, metadata)
for audio, label, meta in train_loader:
    # audio : Tensor (batch, 64000)  — 4s @ 16kHz, float32
    # label : Tensor (batch,)  long  — 0=bonafide, 1=spoof
    # meta  : list of dicts with speaker_id, utterance_id, generator_id, ...
    pass

# 3. Benchmark your checkpoint
# python data_pipeline/run_experiment.py \
#     --checkpoint path/to/your/wavlm_checkpoint.pth \
#     --manifest   data_pipeline/manifests/asvspoof19_eval.csv \
#     --output-report data_pipeline/reports/wavlm_asvspoof19_eval \
#     --split eval
```

---

## Folder Structure

```
data_pipeline/
├── manifests/                     ← Generated manifest CSVs (locked, do not edit)
│   ├── asvspoof19_train.csv       — 25,380 samples (2,580 bonafide / 22,800 spoof)
│   ├── asvspoof19_dev.csv         — 24,844 samples (2,548 bonafide / 22,296 spoof)
│   ├── asvspoof19_eval.csv        — 71,237 samples (7,355 bonafide / 63,882 spoof)
│   └── asvspoof19_combined.csv    — All splits merged (for leakage audit only)
├── reports/                       ← Benchmark reports written here by run_experiment.py
├── schema.py                      ← 12-column manifest contract + validate_schema()
├── loader.py                      ← ManifestAudioDataset (PyTorch Dataset)
├── augmentation.py                ← TrainingAugmentationPipeline (codec/noise/RIR)
├── convert_asvspoof.py            ← Protocol → manifest CSV converter
├── check_leakage.py               ← LeakageChecker assertion script
├── run_experiment.py              ← Evaluation harness (EER / ROC-AUC / PR-AUC)
└── fetch_asvspoof2019.py          ← Full pipeline runner (manifests + audit + train + bench)
```

---

## Manifest Schema (12 columns)

Every manifest CSV — regardless of dataset or model — uses exactly these columns:

| Column | Type | Values / Notes |
|--------|------|----------------|
| `path` | str | Absolute path to audio file (.flac or .wav) |
| `label` | str | **`bonafide`** or **`spoof`** — nothing else |
| `split` | str | `train`, `dev`, or `eval` |
| `source_dataset` | str | e.g. `ASVspoof2019_LA` |
| `speaker_id` | str | e.g. `LA_0079` |
| `utterance_id` | str | e.g. `LA_T_1000137` |
| `generator_id` | str | Attack system ID (e.g. `A07`) or `none` for bonafide |
| `language` | str | ISO 639-1 code (e.g. `en`) |
| `codec` | str | Source codec (e.g. `pcm_16k`) |
| `duration_s` | float | Duration in seconds |
| `license` | str | Dataset license identifier |
| `consent` | str | `yes` / `no` — speaker consent status |

> **Rule:** `label` is always exactly `bonafide` or `spoof`. Binary, no implicit encoding.

Validated at load time by `schema.py`:
```python
from data_pipeline.schema import validate_schema
errors = validate_schema(df)   # returns [] if valid, list of error strings otherwise
```

---

## Audio Contract

All audio loaded through `ManifestAudioDataset` is delivered as:

- **Sample rate:** 16,000 Hz (resampled automatically if source differs)
- **Channels:** Mono (stereo averaged to mono)
- **Window length:** 4.0 seconds = 64,000 samples (padded with zeros or centre-cropped)
- **Dtype:** `torch.float32`, range approximately [-1, 1]
- **Training windows:** cropped at a random start offset (data augmentation)
- **Dev/eval windows:** cropped at centre (deterministic, reproducible)

---

## Score / Model Contract

Every expert model **must** implement this function signature so Task C (backend) can call any model interchangeably:

```python
def score(audio_window: np.ndarray) -> dict:
    """
    Args:
        audio_window: float32 numpy array, shape (64000,), 16kHz mono.

    Returns:
        {
            'logit':         float  — raw pre-sigmoid score, higher = more spoof,
            'embedding':     np.ndarray or None — (embedding_dim,) vector,
            'model_version': str    — e.g. 'wavlm-base-plus-v1',
        }
    """
```

> **Important:** Higher logit = more evidence of spoof. Document this direction explicitly so nobody flips it in Task C.  
> Do **not** call a raw logit a probability. Apply calibration (Platt scaling or isotonic regression on dev data) before exposing as probability.

---

## Evaluation Harness

`run_experiment.py` is the **single evaluation harness** reused by every model. It:

1. Loads your checkpoint via `ModelAdapter`
2. Runs inference on the locked eval split
3. Reports EER, ROC-AUC, PR-AUC, and latency (mean / p95 / p99)
4. Writes a versioned `.json` + `.md` report to `data_pipeline/reports/`

**Usage:**
```bash
# From repo root
python data_pipeline/run_experiment.py \
    --checkpoint  path/to/your_model.pth \
    --manifest    data_pipeline/manifests/asvspoof19_eval.csv \
    --output-report data_pipeline/reports/your_model_eval \
    --split       eval
```

**Reproducibility requirement (from MD spec):**  
The same checkpoint + same eval manifest must produce the same EER on every run. Do not shuffle the eval set.

---

## Data Leakage Audit

Zero leakage is verified by `check_leakage.py`. Results for ASVspoof 2019 LA:

```
[PASS] Zero data leakage detected across splits!
  train samples : 25,380
  dev   samples : 24,844
  eval  samples : 71,237
```

Checks performed:
- No speaker ID in dev/eval appears in train
- No utterance ID in dev/eval appears in train
- No generator ID marked held-out appears in train
- No duplicate audio file paths across any split

Re-run audit at any time:
```bash
python data_pipeline/check_leakage.py --manifest data_pipeline/manifests/asvspoof19_combined.csv
```

---

## Augmentation (Training Only)

`augmentation.py` provides `TrainingAugmentationPipeline` — applied **only** to training samples, never dev or eval. `ManifestAudioDataset` enforces this automatically.

| Augmentation | What it simulates | Requires |
|---|---|---|
| `CodecAugmentation` | Real codec encode/decode (AMR-NB / Opus) | `ffmpeg` in PATH |
| `AdditiveNoise` | White or pink noise at 5–20 dB SNR | Nothing |
| `RIRConvolution` | Acoustic room reflections | RIR `.wav` files |

**Ablation rule (MD spec):** Before claiming "augmentation improved results", you must:
1. Train a **clean baseline** (`TrainingAugmentationPipeline` with all flags False, or no pipeline)
2. Train with augmentation enabled
3. Compare EER on the **same locked eval split**
4. Only claim improvement if the augmented run is measurably better

```python
# Clean baseline (no augmentation)
aug = TrainingAugmentationPipeline(apply_codec=False, apply_noise=False, apply_rir=False)

# Augmented run
aug = TrainingAugmentationPipeline(
    apply_codec=True,   # needs ffmpeg
    apply_noise=True,
    apply_rir=False,    # set True + rir_dir if you have RIR files
)
```

---

## Dataset Statistics (ASVspoof 2019 LA)

| Split | Total | Bonafide | Spoof | Spoof % |
|-------|-------|----------|-------|---------|
| train | 25,380 | 2,580 | 22,800 | 89.8% |
| dev | 24,844 | 2,548 | 22,296 | 89.7% |
| eval | 71,237 | 7,355 | 63,882 | 89.7% |
| **Total** | **121,461** | **12,483** | **108,978** | — |

> **Note on class imbalance:** The dataset is ~90% spoof. Training accuracy will appear high (90%+) even for a model that predicts everything as spoof. **Always evaluate on EER, not accuracy.** EER is threshold-independent and treats both error directions equally.

---

## Benchmarks to Pass (from MD spec)

| Check | Target |
|-------|--------|
| Zero duplicate paths | Verified ✅ |
| Zero leakage across splits | Verified ✅ |
| EER on ASVspoof19 LA eval | Clearly better than 50% (random guessing) |
| Reproducible EER on re-run | Same checkpoint + same eval = same number |

---

## Running the Full Pipeline

```bash
# From repo root — generates manifests + audit + trains LFCC-LCNN + benchmarks
python data_pipeline/fetch_asvspoof2019.py --epochs 30 --batch-size 64

# Manifests + audit only (no training)
python data_pipeline/fetch_asvspoof2019.py --skip-training --skip-benchmark

# Benchmark existing checkpoint only
python data_pipeline/fetch_asvspoof2019.py --skip-training

# Custom dataset root
python data_pipeline/fetch_asvspoof2019.py --dataset-root D:/MyData/LA
# or:  set ASVSPOOF_LA_ROOT=D:/MyData/LA
```

---

## Files Person A Should NOT Modify

| File | Why |
|------|-----|
| `manifests/asvspoof19_eval.csv` | Locked eval set — modifying it invalidates all reported numbers |
| `schema.py` | Shared contract — changes break every model's loader |
| `check_leakage.py` | Shared assertion — must remain identical for all teams |
| `run_experiment.py` | Shared harness — all EER numbers must come from this script |
