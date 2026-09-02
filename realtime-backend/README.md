# Realtime voice-clone detection backend

FastAPI service. Receives live audio over WebSocket (or a whole file over REST),
windows it, scores it with two experts, calibrates each expert's logit into its
own probability, smooths with an EMA, and emits a risk band plus a recommended
action.

Not identity verification, not the dashboard, not an LLM explainer.

Label convention: `bonafide` = 0 (negative logit), `spoof` = 1 (positive logit).
Higher score always means more likely synthetic. Never flip this.

## Quick start

Zero configuration. `config.py` already defaults to both shipped experts with
`hybrid` as the decision expert and its matched calibrator.

```bash
cd realtime-backend
pip install -r requirements.txt
python server.py
```

The first run downloads both checkpoints from Hugging Face into `model_cache/`
(~400 MB, needs internet). Later runs load from disk and work offline. If the
download fails, the error names the repo, the file and the local path to drop it
into manually.

The service binds `0.0.0.0:8000` with **no authentication**. Demo only — do not
put it on an untrusted network.

## Experts

| Key | Repo | File | Label shown in UI | Calibrator |
|---|---|---|---|---|
| `wavlm` | `Akenzz/Expert-1` | `best_model.pt` | Expert-1: WavLM Base+ | `artifacts/platt_v2_combined_dataset.json` |
| `hybrid` | `sarosh22/Expert2` | `hybrid_clean.pth` | Expert-2: LFCC-LCNN Hybrid | `artifacts/calibrator_hybrid_clean.json` |

`hybrid` is the `SINGLE_EXPERT`: its calibrated probability alone produces the
risk band. Both experts' logits, probabilities and per-expert bands are reported
on every window so the UI can show them side by side.

Earlier experts (`lfcc`, `hindi`, `mc_v3`, `prosody`) were removed to keep the
demo surface at exactly two models. `load_settings()` rejects them by name.
Recover them from git history if needed. A `dummy` expert (random logit) still
exists for exercising the pipeline without any checkpoint.

## Per-expert calibration

Each expert's logits sit on their own scale, so each gets its own Platt fit
(`p = sigmoid(a*logit + b)`). Applying one expert's calibrator to the other's
logit is a real bug, not a rounding issue — it is how a model ends up reporting
1.000 on genuine speech.

If you change `SINGLE_EXPERT` you **must** change `CALIBRATOR_PATH` to that
expert's calibrator:

| `SINGLE_EXPERT` | `CALIBRATOR_PATH` |
|---|---|
| `hybrid` (default) | `artifacts/calibrator_hybrid_clean.json` |
| `wavlm` | `artifacts/platt_v2_combined_dataset.json` |

`EXPERT_CALIBRATORS` in `config.py` maps every expert to its calibrator for the
side-by-side display; `CALIBRATOR_PATH` is only the decision path. An expert with
a missing calibrator file falls back to the decision calibrator and logs a
warning — treat that probability as meaningless.

Refit with `scripts/fit_calibrator.py` (`--self-test` proves the math with no
audio or model). `--balance` subsamples to equal classes so the dataset prior is
not baked into `b`. Refit whenever the expert changes or `FUSION_MODE=fused` is
turned on, because the calibrator must see the same logit the policy does.

## Endpoints

### `GET /health`

Reports loaded experts, `decision_expert`, per-expert `expert_details`
(key, label, calibrator version, `is_decision_expert`), window/hop, and the
policy / calibrator / fusion versions.

### `POST /predict-file`

`multipart/form-data`, field `file` (wav/flac/mp3). Resamples to 16 kHz mono and
runs the whole file through the same windowing and EMA logic as the stream. Short
files are padded to one 4.0 s window.

`summary` contains `overall_risk_state`, `final_smoothed_probability`,
`max_probability`, `expert_probabilities`, `expert_risk_states`, an `experts`
list (label, probability, risk state, calibrator version, decision flag) and
`decision_expert`. Each entry in `windows` carries `raw_per_expert_scores` and
`per_expert_probability`.

### `WS /ws`

1. Client sends a JSON start frame:

```json
{"type": "start", "sample_rate": 48000, "encoding": "pcm_s16le", "channels": 1}
```

`encoding` is `pcm_s16le` or `pcm_f32le`. A declared rate other than 16 kHz is
resampled and logged.

Resampling is stateful per connection. The browser's AudioWorklet hands the
frontend 128 samples at a time and each block is forwarded as its own message, so
each frame continues the previous frame's anti-aliasing filter and decimation
phase instead of being resampled in isolation. Frames of any size produce the
same 16 kHz stream as resampling the whole recording offline. Because that filter
state is tied to one rate pair, `sample_rate` cannot change mid-stream — send a
new `start` message instead.

2. Client sends binary PCM frames in that encoding. With `"binary_seq": true` on
   start, each frame is `uint32le sequence_number || pcm`. JSON frames
   (`{"type":"frame","sequence_number":0,"pcm":[...]}`) also work; gaps or
   reorders emit `unavailable` with `dropped_frames: true`, and drop the buffered
   tail, resampler state, and smoothed score so no later window spans the gap.

3. Server replies with one `score` message per completed window:

```json
{
  "type": "score",
  "risk_state": "low",
  "recommended_action": "...",
  "smoothed_probability": 0.02,
  "fused_logit": -7.0,
  "scores": {
    "wavlm":  {"logit": 3.0, "probability": 0.95, "risk_state": "high",
               "model_version": "wavlm-base-plus-ep4",
               "label": "Expert-1: WavLM Base+", "calibrator_version": "..."},
    "hybrid": {"logit": -7.0, "probability": 0.01, "risk_state": "low",
               "model_version": "hybrid_clean",
               "label": "Expert-2: LFCC-LCNN Hybrid", "calibrator_version": "..."}
  },
  "raw_per_expert_scores": {"wavlm": 3.0, "hybrid": -7.0},
  "model_version": {"wavlm": "wavlm-base-plus-ep4", "hybrid": "hybrid_clean"},
  "window_index": 1,
  "latency_ms": 12.3
}
```

`scores` is what the frontend reads. `raw_per_expert_scores` and `model_version`
are kept for backward compatibility.

`risk_state` is exactly one of `collecting`, `low`, `uncertain`, `high`,
`unavailable`.

## Fail-safe

Never default to "real" on a failure path.

| Input | Result |
|---|---|
| Near-zero RMS window | `unavailable` / `silence` |
| Heavily clipped window | `unavailable` / `clipped` |
| Odd-length / bad PCM | `unavailable` / `decode_failure` |
| Client sequence gap or reorder | `unavailable` / `dropped_or_reordered` |
| Audio before `start` | `unavailable` / `protocol_error` |

These set `risk_state=unavailable`, `smoothed_probability=null`, and
`audio_quality` to the reason.

## Layout

```
config.py         window/hop, HUB_EXPERTS, EXPERT_LABELS, EXPERT_CALIBRATORS, env overrides
server.py         FastAPI: /health, /predict-file, /ws
pipeline.py       per-connection decode -> window -> score -> calibrate -> policy
messages.py       outgoing JSON schema, incl. per-expert `scores`
calibration.py    load_calibrator, load_expert_calibrators (platt / platt_sklearn / identity)
fusion.py         logistic fusion, off by default
smoothing.py      EMA (per-expert and decision tracks)
policy.py         collecting | low | uncertain | high | unavailable, plus band()
audio/            resample, ring buffer, quality gates
experts/          protocol, dummy, hub (HF download+cache), wavlm, lfcc, lfcc_model/, loader
artifacts/        calibrators, fusion.json, policy.json
scripts/          wav_client.py, soak_test.py, inspect_checkpoint.py, fit_calibrator.py
tests/
```

`experts/hub.py` also registers a stub `training.config` module so `torch.load`
can unpickle the LFCC checkpoints without the trainer package on the path (the
pickled config is discarded; only `model_state_dict` is used).

## Config (env) — all optional

| Variable | Default | Meaning |
|---|---|---|
| `EXPERTS` | `wavlm,hybrid` | Comma list: `wavlm`, `hybrid`, `dummy` |
| `SINGLE_EXPERT` | `hybrid` | Which expert's probability drives the band |
| `CALIBRATOR_PATH` | `artifacts/calibrator_hybrid_clean.json` | Must match `SINGLE_EXPERT` |
| `FUSION_MODE` | `single` | `single` or `fused` |
| `DEVICE` | `cpu` | Torch device; `cuda` needs a CUDA torch build |
| `WINDOW_SEC` / `HOP_SEC` | `4.0` / `0.5` | p95 latency must stay under the hop |
| `EMA_ALPHA` | `0.3` | Smoothing |
| `MODEL_CACHE_DIR` | `./model_cache` | Hub download cache |
| `PREFETCH_MODELS` | `0` | `1` downloads both checkpoints at startup |
| `SILENCE_RMS` | `1e-4` | Below this -> `unavailable` / silence |
| `CLIP_ABS` / `CLIP_FRACTION` | `0.99` / `0.01` | Clipping gate |

`artifacts/fusion.json` has no weight for `hybrid` yet — add one before ever
setting `FUSION_MODE=fused`. In `single` mode the weights are ignored. Do not
enable fusion until an offline eval shows both experts beat baseline, their
errors differ, and fusion improves the locked metric *including* unseen-generator
slices.

## Tests

```bash
cd realtime-backend
pytest
```

Covers calibration monotonicity, that both experts get distinct calibrators, the
policy bands, the per-expert message contract, fail-safe paths, streaming
resampler equivalence with the offline path, and expert smoke tests.
`pytest -m slow -s` adds the EER benchmarks (needs local eval manifests).

```bash
python scripts/wav_client.py --wav path/to/file.wav   # feed the WebSocket
python scripts/soak_test.py --minutes 2               # RSS + p95 latency
python scripts/inspect_checkpoint.py hybrid           # peek at a checkpoint
```

## Claims this service will not make

- It does not prove caller identity or prevent fraud.
- `dummy` scores are not detection results.
- Fusion is not "better because two models exist".
- A calibrated probability is trustworthy only on audio resembling the corpus it
  was fitted on. Out of domain, a confident number can still be wrong — check
  `/health` for which calibrator is actually in use before quoting a probability.



