# Prosody / Behavioral Expert

An interpretable, human-readable feature expert for the voice-spoof detection
project. It runs **alongside** the learned experts (WavLM, LFCC-LCNN
`lfcc`/`hindi`/`mc_v3`) — not as a replacement. Its purpose is to give the
downstream Groq/LLM explainer **concrete, named evidence** ("pitch variation is
unusually flat") instead of a bare score.

## What it extracts (per 4 s / 64000-sample window)

| Signal | Features |
|---|---|
| Pitch (F0) contour — `librosa.pyin` | `f0_std`, `f0_range`, `f0_delta_var` (contour smoothness), `voiced_fraction` |
| Pause / silence — energy VAD (`librosa.feature.rms`) | `pause_mean`, `pause_var` (TTS timing is suspiciously regular → low var), `pause_count` |
| Speaking rate — onset/energy peaks | `rate_mean`, `rate_var` |
| Micro-variation — Praat (`parselmouth`) primary, librosa proxy fallback | `jitter_local`, `shimmer_local`, `spectral_flatness`, `jitter_source` |

`extract_prosody_features(audio_window, sr=16000) -> dict` returns raw numeric
values only (NaN-safe; silent/unvoiced windows → sentinel defaults). Ordered
`FEATURE_NAMES` is the single source of truth shared by trainer, scorer, and
describe.

## Components

- `features/prosody_features.py` — extractor + `FEATURE_NAMES`
- `features/describe.py` — `describe_prosody_evidence(features) -> list[str]`; emits a
  finding **only** when the value is outside the bonafide human range (p5/p95
  bands from training). Never invents or lists everything.
- `training/train_prosody.py` — StandardScaler + LogisticRegression on the same
  `multicorpus_final.csv` train split; writes `artifacts/prosody_lr_v1.json`.
- `evaluation/evaluate_prosody.py` — EER/AUC/min-tDCF on the exact same slices
  (in-domain `eval`, held-out `eval_ood` per `group`, In-the-Wild, POOLED),
  formatted identically to `lfcc-detector/evaluation/evaluate_multicorpus.py`.
- `expert/prosody_expert.py` — `ProsodyExpert` satisfying the backend `Expert`
  contract (`score(window) -> {logit, embedding, model_version}`; higher logit =
  more spoof; embedding = raw feature vector).

## Run

```bash
pip install -r requirements.txt

# 1. train (feature extraction is cached to artifacts/prosody_features_cache.npz)
python training/train_prosody.py --manifest ../data_pipeline/manifests/multicorpus_final.csv --out artifacts/prosody_lr_v1.json

# 2. evaluate on the shared held-out slices
python evaluation/evaluate_prosody.py --artifact artifacts/prosody_lr_v1.json --manifest ../data_pipeline/manifests/multicorpus_final.csv --output-json artifacts/prosody_eval.json
```

## Wire into the realtime backend

`realtime-backend/experts/loader.py` has a `prosody` branch that imports
`ProsodyExpert` from this package. Enable it:

```bash
EXPERTS=wavlm,lfcc,mc_v3,prosody SINGLE_EXPERT=mc_v3 python server.py
```

`prosody` is a complementary evidence expert — it is **not** expected to beat
`mc_v3` on raw EER. Keep the decision expert as `mc_v3`; the prosody logit +
`describe_prosody_evidence` feed the explainer and (optionally) fusion.
