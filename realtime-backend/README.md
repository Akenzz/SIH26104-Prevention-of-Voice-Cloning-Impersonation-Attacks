# Task C — Real-time voice-clone detection backend

Own directory. Does not live inside `data_pipeline/` or `lfcc-detector/`.

Receives live audio over WebSocket, windows it, scores available experts, optionally fuses, calibrates, smooths, and emits a risk band plus recommended action.

This is **not** identity verification, **not** the dashboard (task E), and **not** the LLM explainer (task F).

## Status

| Piece | State |
|---|---|
| FastAPI `/health` + WebSocket `/ws` | done |
| Start message → resample to 16 kHz mono (logged, not silent) | done |
| Ring buffer, 4.0 s window / 0.5 s hop (configurable) | done |
| DummyExpert so the full path runs without Hub models | done |
| Hub download + cache for WavLM and LFCC-LCNN | done (load on demand) |
| LFCC-LCNN (Expert 3) real model wired + `state_dict` load | **done + tested** |
| WavLM (Expert 1) real model wiring | **done + tested** |
| Fusion scaffolding (`FUSION_MODE=single` default) | done |
| Platt calibrator fitted on ASVspoof19 dev (in-domain) + EMA + policy | **done + tested** |
| Fail-safe: silence / clip / decode / seq gap → `unavailable` | done + tested |
| WAV client, soak (memory + p95 latency) | done |

Default `EXPERTS=dummy`. **`EXPERTS=wavlm` now runs the real WavLM detector** — it loads `expert1/checkpoints/best_model.pt` (produced by `expert1/train.py`) directly from the repo, or falls back to downloading from `Akenzz/Expert-1` on the Hub. The adapter imports `WavLMClassifier` from the `expert1` package, loads the `model_state_dict` with `strict=True`, and runs inference with fp16 autocast for ~2× GPU throughput. The 768-dim mean-pooled embedding and a single spoof logit are returned on every window. **`EXPERTS=lfcc`** runs the LFCC-LCNN detector (see existing docs). The calibrator is a **Platt fit on ASVspoof2019 LA dev** — see the cross-corpus caveat below.

## Layout

```
realtime-backend/
  config.py              window/hop, Hub IDs, env overrides
  server.py              FastAPI: GET /health, WS /ws
  pipeline.py            per-connection decode → window → score → policy
  messages.py            outgoing JSON schema
  fusion.py              loadable logistic fusion (off by default)
  calibration.py         loadable logit → probability
  smoothing.py           EMA
  policy.py              collecting | low | uncertain | high | unavailable
  audio/                 resample, ring buffer, quality gates
  experts/
    protocol.py          score(window) -> {logit, embedding, model_version}
    dummy.py             random logit, version dummy-v0
    hub.py               hf_hub_download into ./model_cache/ (gitignored)
    wavlm.py             Expert 1 adapter — TODO: model class
    lfcc.py              Expert 3 adapter — wired to vendored LFCC-LCNN
    lfcc_model/          vendored LFCC-LCNN architecture (copy of lfcc-detector/models)
    loader.py            EXPERTS=dummy,wavlm,lfcc
  artifacts/             fusion.json, calibrator.json, policy.json
  scripts/               wav_client.py, soak_test.py, inspect_checkpoint.py, fit_calibrator.py
  tests/
```

## Quick start

```bash
cd realtime-backend
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Health: `GET http://127.0.0.1:8000/health`

Feed audio (generates a tone if you have no WAV):

```bash
python scripts/wav_client.py --url ws://127.0.0.1:8000/ws
python scripts/wav_client.py --wav path/to/file.wav
```

Unit + fail-safe + WavLM smoke tests (no server required):

```bash
cd realtime-backend
pytest                          # all fast tests (slow EER test excluded)
pytest tests/test_wavlm_expert.py -v   # WavLM-specific tests
```

**WavLM EER benchmark** against 5 000-sample balanced subset of ASVspoof 2019 LA test:

```bash
# Requires: expert1/data/asvspoof_manifest.csv + expert1/checkpoints/best_model.pt
cd realtime-backend
pytest tests/test_wavlm_expert.py -v -m slow -s
```

Soak (in-process DummyExpert, RSS + p95 latency vs hop):

```bash
python scripts/soak_test.py --minutes 2
python scripts/soak_test.py --minutes 10
```

## Fitting the calibrator

The shipped `artifacts/calibrator.json` is a **Platt fit** (`kind: "platt"`,
`a=1.32`, `b=-0.52`, version `platt-lfcc-asvspoof19dev-v1`) of the LFCC-LCNN
logit, fitted on the **ASVspoof2019 LA dev** split — the same held-out split the
checkpoint was selected on. `smoothed_probability = sigmoid(a*logit + b)` is
therefore a real calibrated probability, not an identity passthrough.

Reproduce it (needs `soundfile` + the ASVspoof2019 LA dev audio; the manifest
paths point at `D:\DatasetSIH\LA`):

```bash
# prove the fitting math with no audio/model (synthetic, numpy only):
python scripts/fit_calibrator.py --self-test

# the real in-domain fit that produced the shipped artifact:
pip install soundfile
python scripts/fit_calibrator.py \
    --manifest ../data_pipeline/manifests/asvspoof19_dev.csv \
    --split dev --expert lfcc --balance --limit 3000 --eval-frac 0.2 \
    --version platt-lfcc-asvspoof19dev-v1 --out artifacts/calibrator.json
```

`--balance` subsamples to equal bonafide/spoof (ASVspoof dev is ~9:1 spoof) so
the dataset prior is **not** baked into `b`; the policy bands set the operating
point. It reuses the backend's own resampler and the real expert adapter, so the
logits it fits on come from the same code path that runs live. On the shipped fit
(3000 clips, 2400 fit / 600 held out): classes separate cleanly (spoof logit
mean +4.71, bonafide −10.69), **held-out EER 0.0000**, and the fit **halves**
held-out calibration error vs the identity map (ECE 0.0084→0.0043, log-loss
0.0101→0.0059). Re-tune `artifacts/policy.json` and **refit** if the expert or
`FUSION_MODE=fused` output changes (the calibrator must see the same logit the
policy does).

> **Cross-corpus caveat — read before demoing.** This calibrator (and the LFCC
> model under it) is validated **in-domain only**. The same checkpoint scores
> **chance-level on MLAAD** (EER 0.442, real audiobook speech scored *more*
> spoof-like than TTS), so a Platt fit there produced an *inverted* slope. If the
> demo audio is not ASVspoof2019-like, treat the probability as unreliable — the
> honest failure mode, not a fitted one. Do not fit the calibrator on a corpus
> the model does not actually separate.

## Available Endpoints (for Postman/Testing)

The backend exposes the following endpoints on `http://127.0.0.1:8000` (or your configured host/port).

### 1. Health Check
* **URL**: `http://127.0.0.1:8000/health`
* **Method**: `GET`
* **Description**: Returns the current status of the backend, which experts are loaded, configuration parameters, and versions of the policy, calibrator, and fusion modules.
* **Example Response**:
  ```json
  {
    "status": "ok",
    "experts": ["wavlm"],
    "fusion_mode": "single",
    "window_sec": 4.0,
    "hop_sec": 0.5,
    "target_sample_rate": 16000,
    "threshold_version": "policy-v0",
    "calibrator_version": "platt-lfcc-asvspoof19dev-v1",
    "fusion_version": "fusion-identity-v0"
  }
  ```

### 2. Single File Prediction (REST)
* **URL**: `http://127.0.0.1:8000/predict-file`
* **Method**: `POST`
* **Body**: `multipart/form-data` with a single field named `file` containing the audio file (e.g., .wav, .flac, .mp3).
* **Description**: A convenience endpoint for testing individual files in Postman. It accepts an audio file, resamples it to 16kHz mono, and processes the *entire* file using the exact same overlapping-window and EMA smoothing logic as the streaming path. Short files are padded to a minimum of one 4.0-second window.
* **Example Response**:
  ```json
  {
    "summary": {
      "overall_risk_state": "low",
      "final_smoothed_probability": 0.00958,
      "max_probability": 0.10755,
      "max_probability_window_index": 4,
      "expert_risk_states": {
        "wavlm": "low",
        "lfcc": "low"
      }
    },
    "windows": [
      {
        "window_index": 1,
        "start_time_sec": 0.0,
        "risk_state": "collecting",
        "calibrated_probability": 0.00004,
        "raw_per_expert_scores": {
          "wavlm": -7.199,
          "lfcc": -8.145
        }
      },
      {
        "window_index": 2,
        "start_time_sec": 0.5,
        "risk_state": "low",
        "calibrated_probability": 0.00004,
        "raw_per_expert_scores": {
          "wavlm": -7.261,
          "lfcc": -8.201
        }
      }
    ],
    "model_version": {
      "wavlm": "wavlm-base-plus-ep4",
      "lfcc": "best_lfcc_lcnn"
    },
    "calibrator_version": "platt-lfcc-asvspoof19dev-v1",
    "threshold_version": "policy-v0",
    "fusion_version": "fusion-identity-v0",
    "audio_quality": null
  }
  ```

### 3. Audio Streaming (WebSocket)
* **URL**: `ws://127.0.0.1:8000/ws`
* **Method**: WebSocket
* **Description**: The primary endpoint for real-time audio streaming. Postman supports WebSocket connections. You can connect to this endpoint, send a JSON `start` message, and then send binary PCM frames or JSON `frame` messages. The server will stream back JSON `score` messages. (See "WebSocket protocol" below for exact message formats).

## WebSocket protocol

1. Client connects to `/ws`.
2. Client sends JSON **start**:

```json
{
  "type": "start",
  "sample_rate": 48000,
  "encoding": "pcm_s16le",
  "channels": 1
}
```

`encoding` is `pcm_s16le` or `pcm_f32le`. Declared rate is trusted; if it is not 16 kHz the server **resamples and logs a warning**.

3. Client sends **binary PCM frames** (same encoding). Optional: `"binary_seq": true` on start, then each binary frame is `uint32le sequence_number || pcm`.
4. Optional JSON frames: `{"type":"frame","sequence_number":0,"pcm":[...floats]}`. Gaps or reorders emit `unavailable` with `dropped_frames: true`.

Outgoing score message (every completed window):

```json
{
  "type": "score",
  "sequence_number": 1,
  "risk_state": "collecting",
  "recommended_action": "...",
  "smoothed_probability": 0.42,
  "fused_logit": 0.1,
  "raw_per_expert_scores": {"dummy": 0.1},
  "model_version": {"dummy": "dummy-v0"},
  "threshold_version": "policy-v0",
  "calibrator_version": "platt-lfcc-asvspoof19dev-v1",
  "fusion_version": "fusion-identity-v0",
  "fusion_mode": "single",
  "dropped_frames": false,
  "audio_quality": null,
  "window_index": 1,
  "latency_ms": 12.3
}
```

`risk_state` is exactly one of `collecting`, `low`, `uncertain`, `high`, `unavailable`.

Silence, clipping, decode failure, or a sequence gap **never** produce `low`. They set `risk_state=unavailable`, `smoothed_probability=null`, and `audio_quality` to the reason.

## Config (env)

| Variable | Default | Meaning |
|---|---|---|
| `EXPERTS` | `dummy` | Comma list: `dummy`, `wavlm`, `lfcc` |
| `FUSION_MODE` | `single` | `single` or `fused` |
| `SINGLE_EXPERT` | first loaded | Which expert when mode is `single` |
| `WINDOW_SEC` | `4.0` | Window length |
| `HOP_SEC` | `0.5` | Hop length (p95 latency must stay below this) |
| `EMA_ALPHA` | `0.3` | Smoothing |
| `DEVICE` | `cpu` | Torch device for real experts |
| `MODEL_CACHE_DIR` | `./model_cache` | Hub downloads |
| `PREFETCH_MODELS` | `0` | `1` = download both Hub files at startup |
| `SILENCE_RMS` | `1e-4` | Below this → `unavailable` / silence |
| `CLIP_ABS` / `CLIP_FRACTION` | `0.99` / `0.01` | Clipping gate |

Fusion / calibrator / policy math is **not** hardcoded. Replace:

- `artifacts/fusion.json` — logistic weights from an offline dev-set fit
- `artifacts/calibrator.json` — Platt `a`,`b` (kind `platt` or `identity_sigmoid`)
- `artifacts/policy.json` — band cuts and recommended-action text

Do not turn on `FUSION_MODE=fused` until an offline eval shows both experts beat baseline, errors differ, and fusion improves the locked metric including unseen-generator slices.

## Hugging Face checkpoints (not in git)

| Expert | Repo | File |
|---|---|---|
| WavLM (A) | `Akenzz/Expert-1` | `best_model.pt` |
| LFCC-LCNN (D) | `sarosh22/lfcc-lcnn-asvspoof19` | `best_lfcc_lcnn.pth` |

Public repos, no token. First use calls `hf_hub_download` into `model_cache/` and logs start/finish. Later boots skip the download if the file is already there.

Inspect a checkpoint without wiring a class:

```bash
python scripts/inspect_checkpoint.py wavlm
python scripts/inspect_checkpoint.py lfcc
```

### Wiring a real expert

**LFCC-LCNN (Expert 3) is already wired.** Run it with:

```bash
EXPERTS=lfcc FUSION_MODE=single uvicorn server:app --port 8000
```

Its architecture is vendored byte-for-byte from `lfcc-detector/models/` into
`experts/lfcc_model/` so the backend runs standalone. The Hub checkpoint loads
into `LFCCLCNNWithFeatureExtraction(sample_rate=16000, n_lfcc=20,
with_deltas=True, embedding_dim=128)` with **zero missing/unexpected keys**
(verified). The checkpoint pickles a `training.config.TrainingConfig` dataclass;
`experts/hub.py` registers a stub `training.config` module so `torch.load`
succeeds without the trainer package on the path (the config is discarded — only
`model_state_dict` is used). If the trainer changes the architecture, re-vendor
those two files and confirm `load_state_dict` still matches.

**WavLM (Expert 1) is wired and tested.**

`experts/wavlm.py` imports `WavLMClassifier` from the sibling `expert1/` package, loads `model_state_dict` with `strict=True`, runs with fp16 autocast on CUDA, and returns `{logit, embedding (768-dim), model_version}` on every window. Checkpoint resolution:

1. **Local** — `expert1/checkpoints/best_model.pt` (produced by `expert1/train.py`)
2. **Hub fallback** — `Akenzz/Expert-1 / best_model.pt` (for teammates without a local checkpoint)

Run it with:

```bash
# From repo root — train first (one-time, ~1 hr for 10 epochs on RTX 2060):
python -m expert1.train \
    --manifest expert1/data/asvspoof_manifest.csv \
    --epochs 10 --batch-size 32 --dev-batches 200

# Then start the backend with WavLM:
cd realtime-backend
EXPERTS=wavlm DEVICE=cuda uvicorn server:app --host 0.0.0.0 --port 8000

# Smoke tests (fast, no server needed):
pytest tests/test_wavlm_expert.py -v

# EER benchmark (5 000-sample balanced subset, ~5 min):
pytest tests/test_wavlm_expert.py -v -m slow -s
```

## Fail-safe (required)

| Input | Result |
|---|---|
| Near-zero RMS window | `unavailable` / `silence` |
| Heavily clipped window | `unavailable` / `clipped` |
| Odd-length / bad PCM | `unavailable` / `decode_failure` |
| Client seq gap or reorder | `unavailable` / `dropped_or_reordered`, `dropped_frames=true` |
| Audio before `start` | `unavailable` / `protocol_error` |

Never default to “real” / `low` on a failure path.

## Claims this service will not make

- It does not prove caller identity.
- Dummy scores are not detection results.
- Fusion is not “better because two models exist.”
- The calibrated probability is trustworthy **only in-domain**: it is fitted and validated on ASVspoof2019 LA, and the same model is chance-level cross-corpus (MLAAD EER 0.442). Out-of-domain, the number is not a real probability.
