# SIH26104 — Prevention of Voice Cloning / Impersonation Attacks

Detects synthetic and voice-cloned speech in real time. A React dashboard streams
microphone audio (or uploads a file) to a FastAPI backend, which windows the audio,
scores it with several independently trained detectors, calibrates each score into
a probability, smooths it, and returns a risk band.

**Not** identity verification. It answers "does this audio look machine-generated?",
not "is this person who they claim to be".

## Label convention — never flip this

| String | Integer | Meaning |
|---|---|---|
| `bonafide` | **0** | Real human speech |
| `spoof` | **1** | AI-generated / voice-cloned speech |

Higher logit and higher probability always mean **more likely synthetic**.

## The models

The two primary experts are downloaded automatically from Hugging Face on first
run and cached in `realtime-backend/model_cache/`. No token, no manual download,
no env vars. `realtime-backend/config.py::HUB_EXPERTS` is the authoritative list.

| # | Key | Hugging Face | Architecture | Role |
|---|---|---|---|---|
| Expert-1 | `wavlm` | [Akenzz/Expert-1](https://huggingface.co/Akenzz/Expert-1) | Frozen WavLM Base+ backbone + linear head | reported |
| Expert-2 | `hybrid` | [sarosh22/Expert2](https://huggingface.co/sarosh22/Expert2) | LFCC-LCNN, 6 languages (hi/en/kn/ml/mr/ta) | **drives the risk band** |

Each expert has its own Platt calibrator, because their logits live on different
scales — one shared calibrator would misread the other model. The displayed
risk band comes from the decision expert only; the other experts' probabilities
are shown alongside for comparison.

> **Read [CALIBRATION-AND-RESAMPLING.md](CALIBRATION-AND-RESAMPLING.md) before
> retraining a model, changing any audio preprocessing, or swapping a calibrator
> artifact.** Those two areas have produced every silent failure this project has
> had — wrong answers that still return a valid probability, keep the stream
> running, and look completely normal on the dashboard. It also lists which
> calibrator artifact belongs to which expert, which is not guessable from the
> filenames.

### Expert-2 variants (LFCC-LCNN)

Three checkpoints share one architecture. `hybrid` is on the Hub; the other two
are local-only and ship in `model_cache/`.

| Key | What is different | Verdict changes when the resampler changes |
|---|---|---|
| `hybrid` | the published baseline | **13.35 logits** |
| `hybrid_nc` | + fine-tuned on 12 modern engines | 5.25 logits |
| `hybrid_br` | + 7 kHz parity band gate | **0.03 logits** |

That last column is the one that matters in service. `hybrid`'s reported dev EER
of **2.42% is artifact-inflated**: it was measured on librosa-resampled audio,
and on the audio this backend actually produces the same weights score
**27.52%**. `hybrid_br` scores **8.27% on both paths**. Do not compare 2.42% to
8.27% — they measure different things, and only the second one predicts field
behaviour. The full argument, with the measurement that settles it, is in
[CALIBRATION-AND-RESAMPLING.md](CALIBRATION-AND-RESAMPLING.md).

At the shipped policy bands, `hybrid_br` on 27 generators unseen by both the
training run and the calibrator fit: **88.1% of spoofs caught at a 2.9% false
alarm rate**, with no generator below 50% recall.

## Run it

The first backend start needs internet and downloads ~400 MB; every later start
is offline. One command brings up both servers from the repo root:

```powershell
.\start_all.ps1
```

```bash
./start_all.sh
```

It installs dependencies, frees ports 8000/5173 if a stale server is still
holding them, waits for `/health`, prints which experts loaded, then starts the
frontend. Add `-SkipInstall` / `--skip-install` for fast restarts, or `-Stop` /
`--stop` to just kill both servers.

To run them by hand instead — backend:

```bash
cd realtime-backend && pip install -r requirements.txt && python server.py
```

Frontend (separate terminal):

```bash
cd voice-integrity-frontend && npm install && npm run dev
```

Backend on <http://localhost:8000> (health: `/health`), frontend on
<https://localhost:5173> — the dev server uses a self-signed certificate, so
click through the browser warning. It proxies `/health`, `/ws` and
`/predict-file`, so start the backend first.

Full instructions, including PowerShell specifics, are in
[how-to-run-backend-and-frontend.txt](how-to-run-backend-and-frontend.txt).

The backend binds `0.0.0.0` and has **no authentication** — it is a demo service.
Do not expose it to an untrusted network.

## Repository layout

| Path | What it is |
|---|---|
| `realtime-backend/` | FastAPI service: WebSocket `/ws`, `POST /predict-file`, `GET /health`. See its [README](realtime-backend/README.md). |
| `voice-integrity-frontend/` | React + Vite dashboard (live monitor, file analysis, settings) |
| `wavlm-base-plus/` | Expert-1 training/eval code ([README](wavlm-base-plus/README.md)) |
| `lfcc-detector/` | Expert-2 training/eval code (LFCC-LCNN) |
| `prosody-detector/` | Experimental interpretable prosody expert — not loaded by the backend |
| `data_pipeline/` | Manifest building, VAD slicing, dataset prep |
| [CALIBRATION-AND-RESAMPLING.md](CALIBRATION-AND-RESAMPLING.md) | **Required reading** before retraining or changing preprocessing: the calibrator↔expert mapping, the resampler bug, the three startup guards |
| [realtime-backend/.env.example](realtime-backend/.env.example) | Every supported backend config, including how to make the bandwidth-robust expert the decision model |

## What this project does not claim

- It does not prevent fraud, prove identity, or verify a speaker.
- The calibrated probability is only as good as the corpora it was fitted on;
  out-of-domain audio can be scored confidently and wrongly.
- Held-out (unseen-generator) numbers are reported separately from in-domain
  numbers above, and only the held-out ones predict field behaviour.
- Voices were cloned only with the consent of team members; no public figures.
