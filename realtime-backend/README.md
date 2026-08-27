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
| WavLM (Expert 1) real model wiring | **TODO in `experts/wavlm.py`** |
| Fusion scaffolding (`FUSION_MODE=single` default) | done |
| Identity-sigmoid calibrator + EMA + policy | done (calibrator not yet fitted) |
| Fail-safe: silence / clip / decode / seq gap → `unavailable` | done + tested |
| WAV client, soak (memory + p95 latency) | done |

Default `EXPERTS=dummy`. **`EXPERTS=lfcc` now runs the real LFCC-LCNN detector** — it downloads `best_lfcc_lcnn.pth`, loads it into the vendored architecture (`experts/lfcc_model/`), and emits real spoof logits with a 128-dim embedding. WavLM (`experts/wavlm.py`) is still stubbed: requesting `wavlm` downloads the file, `torch.load`s it, then raises `NotImplementedError` with the checkpoint keys printed. The calibrator is still the identity stub, so `smoothed_probability` is an honest-but-unfitted mapping of the real logit until `artifacts/calibrator.json` is replaced from a dev-set fit.

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
  scripts/               wav_client.py, soak_test.py, inspect_checkpoint.py
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

Unit + fail-safe tests (no server required):

```bash
pytest
```

Soak (in-process DummyExpert, RSS + p95 latency vs hop):

```bash
python scripts/soak_test.py --minutes 2
python scripts/soak_test.py --minutes 10
```

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
  "calibrator_version": "identity-sigmoid-v0",
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

**WavLM (Expert 1) is still a stub.** To wire it, only touch `experts/wavlm.py`:

1. Fill `_wire_model()` (import class, `load_state_dict`, `.eval()`).
2. Fill `score()` so it returns `{logit, embedding, model_version}`. Higher logit = more spoof.
3. Set `EXPERTS=wavlm` (or `wavlm,lfcc` with `FUSION_MODE` still `single` until eval says otherwise).

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
- The identity calibrator is not a fitted probability until `calibrator.json` is replaced from a real dev-set fit.
