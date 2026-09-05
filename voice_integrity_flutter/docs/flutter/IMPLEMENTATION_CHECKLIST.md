# Flutter client implementation checklist

This checklist is updated as modules are completed. The Flutter client lives in
`voice_integrity_flutter/` and is deliberately separate from the React demo so
the original project remains runnable during migration.

## Product and architecture

- [x] Define and save the judge-facing design context in `../.impeccable.md`.
- [x] Map the backend contract: `GET /health`, `POST /predict-file` (SSE), and
  the live-score WebSocket contract.
- [x] Create a Flutter application targeting Android, iOS, macOS, Windows,
  Linux, and web.
- [x] Create a responsive navigation shell: overview, file review, guided live
  demo, and model/system review.

## Experience modules

- [x] Judge overview with a glanceable risk card, next action, evidence summary,
  backend state, and risk timeline.
- [x] File review UI with native audio selection and a real multipart upload to
  `/predict-file`.
- [x] Parse server-sent `window_scored` and `summary` events into typed Flutter
  models for a post-analysis verdict, model evidence, and timeline.
- [x] Guided live walkthrough using explicitly-labelled showcase data. It does
  not pretend to capture or score a microphone.
- [x] Model and system-review screen with the architecture, weighting, policy,
  calibration, and safety constraints.
- [x] Responsive desktop rail and mobile bottom navigation.
- [x] Risk states are communicated by icon and text as well as colour.
- [x] Mobile-first adaptation: the phone view prioritizes verdict and next
  action, uses full-width touch controls, scales page titles, condenses the
  header, and reveals evidence progressively without removing it.

## Remaining integration work

- [ ] Add platform microphone capture and PCM framing for the backend `/ws`
  protocol. This needs a product decision on supported platforms and permission
  handling; it must send the backend's required `start` frame before audio.
- [ ] Stream `/predict-file` events into the UI while the upload is processing,
  rather than presenting them after the SSE response completes.
- [ ] Add widget tests for unavailable, collecting, low, uncertain, and high
  states, plus API parser tests with captured backend fixtures.
- [ ] Run on a physical Android/iOS device and a desktop target before demo day.
- [ ] Resolve the runtime/documentation drift described in
  `../models/MODEL_INTEGRATION.md` before a public deployment.

## Validation record

- [x] `flutter pub get`
- [x] `dart format lib`
- [x] `flutter analyze`
- [x] `flutter build web`
- [x] `flutter test` (3 model-contract tests)
- [ ] Manual file-analysis smoke test against a running FastAPI backend

## Module update log

| Module | Status | Notes |
| --- | --- | --- |
| Core shell + visual system | Complete | Editorial trust-cockpit design with responsive navigation. |
| Typed backend contract | Complete | Health JSON and file-analysis SSE parsing implemented. |
| Overview | Complete | Decision, action, system status, evidence, and timeline. |
| File review | Complete | Native picker and actual multipart upload path. |
| Guided demo | Complete | Presentation-safe local simulation with explicit disclosure. |
| Mobile-first adaptation | Complete | Phone-first information order below 768 px; tablet/desktop progressively add density. |
| Live microphone transport | Planned | Requires a capture/permission implementation. |
| Verification | Complete | Dependency resolution, formatting, static analysis, web build, and three model-contract tests passed. |
