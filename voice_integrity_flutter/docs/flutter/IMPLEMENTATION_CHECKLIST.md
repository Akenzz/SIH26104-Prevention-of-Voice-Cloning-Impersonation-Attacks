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

- [x] Add platform microphone capture and PCM framing for live call demo (`/ws/caller` and `/ws/receiver`).
  Includes native permissions handling, 16kHz PCM 16-bit mono streaming, and pure-Dart linear resampler for 44.1/48kHz inputs.
- [x] Add Caller Mode (Attacker) with unmistakable 'Talk as Teammate 3' spoof toggle, glowing red visual beacon/border, timers, and live VU meter.
- [x] Add Receiver Mode (Victim) with speaker playback (~120ms jitter queue, loud-speaker routing), 4-band real-time risk assessment, 1.5s Amber buffer analyzing transition state, live probability meter, and reasoning trace.
- [x] Add interactive simulation fallback mode for reliable offline judge demonstrations.
- [ ] Stream `/predict-file` events into the UI while the upload is processing,
  rather than presenting them after the SSE response completes.
- [ ] Run on a physical Android/iOS device and a desktop target before demo day.
- [ ] Resolve the runtime/documentation drift described in
  `../models/MODEL_INTEGRATION.md` before a public deployment.

## Validation record

- [x] `flutter pub get`
- [x] `dart format lib test`
- [x] `flutter analyze` (Zero issues found)
- [x] `flutter build web` (Successfully compiled to build/web in 24s)
- [x] `flutter test` (15 tests passed across voice models and live call suites)
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
| Live Call: Caller Mode (Attacker) | Complete | 16kHz PCM mono capture via `record`, pure-Dart resampler, unmistakable "Talk as Teammate 3" spoof toggle, pulsating red border/badge, duration timers, and live VU meter. |
| Live Call: Receiver Mode (Victim) | Complete | Speaker playback via `audioplayers` with ~120ms jitter queue and iOS loud-speaker routing; 4 risk bands (Green/Amber/Red/Gray); 1.5s Amber transition state ("Switching Voice Stream — Analyzing Buffer..."); live probability gauge and real-time reasoning trace. |
| Offline Demo Simulation | Complete | Built-in interactive simulator enabling judge walkthroughs without external server dependencies. |
| Verification | Complete | Dependencies resolved (`record`, `audioplayers`, `web_socket_channel`), formatting clean, static analysis 100% clean, web build passed, 15 unit/contract tests passed. |
