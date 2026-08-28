# data_pipeline — Task B: Data Pipeline, Manifest & Evaluation Harness

> **Status:** ✅ V2 multi-lingual training data ready. All download scripts functional.
>
> This folder is the **single source of truth** every model trains and is measured against. Person A (WavLM), Person D (LFCC-LCNN), Person H (Evaluation), and Person K (AASIST) all consume the same manifests and evaluation harness from here.

---

## Quick Start for All Team Members

### Step 1 — Install dependencies

```powershell
pip install datasets soundfile huggingface_hub pandas pyarrow torchaudio
```

### Step 2 — Generate manifests (run once, takes ~1 minute)

```powershell
# From repo root — regenerates all CSVs in data_pipeline/manifests/
python data_pipeline/fetch_asvspoof2019.py --dataset-root D:\DatasetSIH\LA --skip-training --skip-benchmark
```

### Step 3 — Download the other datasets (one-time, already done by Person D)

```powershell
python data_pipeline/fetch_in_the_wild.py              # 8 GB, eval only
python data_pipeline/fetch_mlaad.py --use-tiny         # 3.5 GB, multilingual spoof
python data_pipeline/fetch_kathbath.py --languages hi  # Hindi bonafide speech
```

### Step 4 — Build the V2 merged training manifest

```powershell
python data_pipeline/build_v2_manifests.py
```

This produces:
| File | Rows | Purpose |
|------|------|---------|
| `v2_train.csv` | ~123,821 | V2 model training |
| `v2_dev.csv` | ~24,844 | V2 early stopping / dev EER |

---

## Manifest Files Reference

> Manifests are in `data_pipeline/manifests/`. They are **git-ignored** (machine-specific paths).
> Every team member regenerates them locally by running the scripts above.

| Manifest | Rows | Split | Dataset | Use |
|----------|------|-------|---------|-----|
| `asvspoof19_train.csv` | 25,380 | train | ASVspoof 2019 LA | English spoof training |
| `asvspoof19_dev.csv` | 24,844 | dev | ASVspoof 2019 LA | Dev / early stopping |
| `asvspoof19_eval.csv` | 71,237 | eval | ASVspoof 2019 LA | English eval benchmark |
| `kathbath_train.csv` | 83,151 | train | Kathbath Hindi | Indic bonafide training |
| `kathbath_eval.csv` | 3,151 | eval | Kathbath Hindi | **H's Indic safety set** (FPR) |
| `mlaad_train.csv` | 15,290 | train | MLAAD-tiny | Multilingual spoof training |
| `in_the_wild_eval_ood.csv` | 31,779 | eval_ood | In-the-Wild | Cross-corpus OOD eval |
| **`v2_train.csv`** | **123,821** | train | ASVspoof19+Kathbath+MLAAD | **V2 model training** |
| **`v2_dev.csv`** | **24,844** | dev | ASVspoof19 | **V2 dev / early stopping** |

---

## For Person A (WavLM) — Using V2 Data

```python
from data_pipeline.loader import ManifestAudioDataset
from data_pipeline.augmentation import TrainingAugmentationPipeline
from torch.utils.data import DataLoader

# Load V2 training data (multilingual: English + Hindi)
train_ds = ManifestAudioDataset(
    manifest_path="data_pipeline/manifests/v2_train.csv",
    split="train",
    window_sec=4.0,
    sample_rate=16000,
    augmentation_pipeline=TrainingAugmentationPipeline(apply_noise=True),
)
dev_ds = ManifestAudioDataset(
    manifest_path="data_pipeline/manifests/v2_dev.csv",
    split="dev",
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
dev_loader   = DataLoader(dev_ds,   batch_size=32, shuffle=False, num_workers=0)

# Each batch: (audio, label, metadata)
# audio : Tensor (batch, 64000) — 4s @ 16kHz float32
# label : Tensor (batch,) long  — 0=bonafide, 1=spoof
```

> **Note on class balance in V2:** V2 has more bonafide rows (83k Hindi + 2.5k English = ~86k)
> vs spoof (23k English + 15k MLAAD = ~38k). This is intentional — it reduces false positives on
> real Indic speech. Use `BCEWithLogitsLoss` with `pos_weight` if needed.

---

## For Person H — Running Sliced Evaluation

Person H uses `evaluate_slices.py` to produce the per-condition results table.
**Never pool slices** — MD spec requirement.

### V1 Model (baseline, already run)
```powershell
python data_pipeline/evaluate_slices.py `
    --checkpoint  lfcc-detector/checkpoints/best_lfcc_lcnn.pth `
    --manifests   data_pipeline/manifests/kathbath_eval.csv `
                  data_pipeline/manifests/in_the_wild_eval_ood.csv `
    --output-dir  data_pipeline/reports/sliced_eval_v1 `
    --model-name  "LFCC-LCNN (V1)"
```

### V2 Model (run after training completes)
```powershell
python data_pipeline/evaluate_slices.py `
    --checkpoint  lfcc-detector/checkpoints/best_lfcc_lcnn_v2.pth `
    --manifests   data_pipeline/manifests/asvspoof19_eval.csv `
                  data_pipeline/manifests/kathbath_eval.csv `
                  data_pipeline/manifests/in_the_wild_eval_ood.csv `
    --output-dir  data_pipeline/reports/sliced_eval_v2 `
    --model-name  "LFCC-LCNN (V2)"
```

**Key findings from V1 sliced eval:**

| Condition | EER / FP Rate | Status |
|-----------|--------------|--------|
| English (In-the-Wild) | FP = 5.2% | ✅ Acceptable |
| Hindi bonafide (Kathbath) | **FP = 45.0%** | ❌ V1 bias — fixed by V2 training |
| Full OOD pooled | EER = 37.35% | ❌ V1 fails cross-corpus |

---

## For Person D (LFCC-LCNN) — Training V2

```powershell
cd lfcc-detector
python training/train.py `
    --train-manifest  ../data_pipeline/manifests/v2_train.csv `
    --dev-manifest    ../data_pipeline/manifests/v2_dev.csv `
    --output-dir      checkpoints `
    --epochs          20 `
    --batch-size      32 `
    --lr              1e-4 `
    --log-every       100 `
    --checkpoint-name best_lfcc_lcnn_v2.pth `
    2>&1 | Tee-Object -FilePath ../data_pipeline/reports/v2_training_log.txt
```

V1 checkpoint (`best_lfcc_lcnn.pth`) is preserved. V2 saves to `best_lfcc_lcnn_v2.pth`.

---

## Dataset Download Guide

Audio files go to `E:\DatasetSIH\` or `D:\DatasetSIH\`. Manifests (CSVs, ~MBs) stay local.

| Dataset | Size | Access | Purpose |
|---------|------|--------|---------|
| ASVspoof 2019 LA | 7 GB | ✅ Already at `D:\DatasetSIH\LA` | Primary English train/eval |
| In-the-Wild | 8.16 GB | ✅ Open (HuggingFace) | OOD eval ONLY — never train |
| MLAAD-tiny | 3.54 GB | ✅ Open (HuggingFace) | Multilingual spoof train |
| Kathbath (Hindi) | ~5 GB | 🔒 Gated (HF login) | Indic bonafide train + safety set |
| ASVspoof 2021 LA eval | ~8 GB | Downloaded to `E:\DatasetSIH\` | Channel robustness eval (future) |

---

## Manifest Schema (12 columns — unchanged)

| Column | Type | Values |
|--------|------|--------|
| `path` | str | Absolute path to `.flac` or `.wav` |
| `label` | str | `bonafide` or `spoof` |
| `split` | str | `train`, `dev`, `eval`, or `eval_ood` |
| `source_dataset` | str | e.g. `ASVspoof2019_LA`, `Kathbath`, `MLAAD-tiny` |
| `speaker_id` | str | Speaker identifier |
| `utterance_id` | str | Utterance identifier |
| `generator_id` | str | TTS system ID or `none` for bonafide |
| `language` | str | ISO 639-1 code (`en`, `hi`, `de`, …) |
| `codec` | str | Source codec (`pcm_16k`, `flac`, …) |
| `duration_s` | float | Duration in seconds |
| `license` | str | Dataset license |
| `consent` | str | `yes` / `no` |

Validated at load time:
```python
from data_pipeline.schema import validate_schema
errors = validate_schema(df)  # [] if valid
```

---

## Files Not to Modify

| File | Why |
|------|-----|
| `manifests/asvspoof19_eval.csv` | Locked eval — modifying invalidates all reported numbers |
| `schema.py` | Shared contract — changes break every model's loader |
| `check_leakage.py` | Shared assertion — must remain identical for all teams |
| `run_experiment.py` | Shared harness — all EER numbers must come from this script |
