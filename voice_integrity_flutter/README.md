# Voice Integrity Flutter client

A responsive, judge-ready client for the voice-cloning detection backend. It is
separate from the existing React dashboard, so both clients can be used during
the project transition.

## What is implemented

- An executive-friendly overview with decision, next action, evidence, and
  system-readiness cards.
- Audio-file review for WAV, MP3, and FLAC through the FastAPI
  `POST /predict-file` endpoint.
- Typed parsing of the backend's server-sent `window_scored` and `summary`
  events into a timeline and per-model evidence.
- A reliable local judge walkthrough, explicitly labelled as showcase data.
- A model and system-review page which reads the backend's `GET /health`
  payload and explains the model, calibration, fusion, and policy rules.
- Responsive desktop navigation rail and mobile bottom navigation.
- Mobile-first information order below 768 px: verdict and action first,
  full-width touch controls, and expandable evidence rather than a compressed
  desktop dashboard.

Microphone capture and binary PCM transport for the realtime `WS /ws` endpoint
are deliberately not presented as complete. They are the next implementation
module; see [the checklist](docs/flutter/IMPLEMENTATION_CHECKLIST.md).

## Run locally

Start the FastAPI backend in one terminal:

```bash
cd ../realtime-backend
python -m pip install -r requirements.txt
python server.py
```

Then start Flutter in another terminal:

```bash
flutter pub get
flutter run -d chrome
```

The client defaults to `http://127.0.0.1:8000`. Change it in **Models → Backend
connection** when testing from a physical device or a different host. The
backend currently allows CORS for all origins, but it has no authentication and
must remain on a trusted demonstration network.

## Verify

```bash
dart format lib
flutter analyze
flutter build web
```

## Documentation

- [Implementation checklist](docs/flutter/IMPLEMENTATION_CHECKLIST.md)
- [Model integration guide](docs/models/MODEL_INTEGRATION.md)
- [Project calibration and resampling guide](../CALIBRATION-AND-RESAMPLING.md)

The model-integration guide documents the exact backend-facing behaviour of
this client, including model roles, calibration mapping, policy bands,
file-analysis ensemble weights, and known limitations. It should be reviewed
before changing models, preprocessing, or calibration artifacts.
