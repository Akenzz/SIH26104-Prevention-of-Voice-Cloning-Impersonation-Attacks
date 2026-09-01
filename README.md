# SIH26104 — Prevention of Voice Cloning / Impersonation Attacks

Detects synthetic and voice-cloned speech in real time. A React dashboard streams
microphone audio (or uploads a file) to a FastAPI backend, which windows the audio,
scores it with two independently trained detectors, calibrates each score into a
probability, smooths it, and returns a risk band.

**Not** identity verification. It answers "does this audio look machine-generated?",
not "is this person who they claim to be".

## Label convention — never flip this

| String | Integer | Meaning |
|---|---|---|
| `bonafide` | **0** | Real human speech |
| `spoof` | **1** | AI-generated / voice-cloned speech |

Higher logit and higher probability always mean **more likely synthetic**.

## The two models

Both are downloaded automatically from Hugging Face on first run and cached in
`realtime-backend/model_cache/`. No token, no manual download, no env vars.

| # | Key | Hugging Face | Architecture | Role |
|---|---|---|---|---|
| Expert-1 | `wavlm` | [Akenzz/Expert-1](https://huggingface.co/Akenzz/Expert-1) | Frozen WavLM Base+ backbone + linear head | reported |
| Expert-2 | `hybrid` | [sarosh22/Expert2](https://huggingface.co/sarosh22/Expert2) | LFCC-LCNN, 6 languages (hi/en/kn/ml/mr/ta) | **drives the risk band** |

Each expert has its own Platt calibrator, because their logits live on different
scales — one shared calibrator would misread the other model. The displayed
risk band comes from the decision expert (`hybrid`) only; Expert-1's probability
is shown alongside for comparison.

Expert-2 numbers: dev EER 2.42%, held-out **unseen-generator** MLAAD 4.58%,
real-world in-the-wild 9.73%, pooled out-of-domain 5.91%. Trained on ~20k
bonafide / ~20k spoof base clips (43.8k VAD chunks, 130 spoof generators) with
bonafide↔spoof paired *within* each language so corpus or channel cannot act as
a label shortcut.

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

## What this project does not claim

- It does not prevent fraud, prove identity, or verify a speaker.
- The calibrated probability is only as good as the corpora it was fitted on;
  out-of-domain audio can be scored confidently and wrongly.
- Held-out (unseen-generator) numbers are reported separately from in-domain
  numbers above, and only the held-out ones predict field behaviour.
- Voices were cloned only with the consent of team members; no public figures.
