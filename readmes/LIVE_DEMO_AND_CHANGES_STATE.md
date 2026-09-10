# Voice Integrity — Live Demo System State & Changelog
> **Document Purpose:** Complete record of architecture, failure mode mitigations, changelog, and operational state for the SIH26104 Live Two-Phone Voice Cloning Demonstration system.

---

## 1. Executive Summary

To make voice cloning fraud tangible for hackathon judges rather than abstract, we designed and built a live **two-phone impersonation attack and real-time defense demonstration**.

```
[Phone 1: Caller / Attacker]
  - Captures 16 kHz 16-bit mono PCM microphone audio
  - Interactive "Talk as Teammate 3 (Spoof)" toggle button with glowing red visual feedback
  - Streams raw PCM frames over WebSocket to Relay (:8001/ws/caller)
          │
          ▼
[Host Laptop: Python Audio Relay (:8001)]
  - Context-preserving audio chunk processor (200ms blocks, 150ms context, 25ms crossfade)
  - Modular Voice Conversion Engine (Mock pitch-shift / Real RVC inference)
  - Bounded leaky queue (maxsize=3) preventing TCP buffer bloat
  - Fast-transition signal: flushes backend RingBuffer & EMA on spoof toggle
          ├──► Streams audio chunks to Realtime Detection Backend (:8000/ws)
          │    Receives real-time probability & multi-expert scores
          ▼
[Phone 2: Receiver / Victim]
  - Receives audio + risk telemetry over WebSocket (:8001/ws/receiver)
  - Plays audio via device loudspeaker using a 120ms jitter queue and RIFF containerization
  - Visualizes real-time 4-band risk gauge (Green / Amber / Red / Gray) + 1.5s Amber transition
  - Multi-expert breakdown (WavLM Base+, LFCC-LCNN Hybrid) & forensic reasoning trace
```

---

## 2. Red Team Failure Modes Identified & Solved

Prior to implementation, a thorough Failure Mode Analysis identified 5 showstoppers in the initial concept, all of which have been resolved:

| # | Failure Mode | Cause | Engineering Solution |
|---|---|---|---|
| **1** | **RVC Audio Destruction on Small Chunks** | Neural vocoders (HuBERT + HiFi-GAN) fail on isolated 20–50ms chunks (F₀ → 0, robotic buzz, clicks). | **`ChunkCrossfadeProcessor`:** Batches audio into 200ms inference blocks with 150ms acoustic context, applies 25ms raised-cosine crossfading at boundaries, and soft-limits via `tanh` ($\pm 0.92$). |
| **2** | **Mobile PCM Streaming Playback & iOS Earpiece Trap** | Standard Flutter audio plugins require container headers; mobile OS defaults `playAndRecord` to the earpiece. | **`AudioPlaybackService`:** Prepends dynamic 44-byte RIFF-WAV headers to audio chunks, buffers ~120ms jitter queue, and forces routing to the loudspeaker. |
| **3** | **5-Second Transition Lag on Spoof Toggle** | Backend's 4.0s RingBuffer and EMA smoothing ($\alpha=0.3$) took ~5s to reflect spoof toggle. | **Fast Transition Protocol:** Toggling spoof sends a control message that immediately flushes the backend RingBuffer and resets the EMA filter. |
| **4** | **TCP Buffer Bloat & Latency Drift** | Network jitter caused TCP queues to accumulate seconds of lag during live streaming. | **`DropOldestQueue(maxsize=3)`:** Bounded leaky queue drops older frames when processing falls behind, capping latency below 250ms. |
| **5** | **Acoustic Feedback Loop** | Caller mic picking up Receiver loudspeaker creates howling and corrupts detector input. | **Acoustic Isolation:** Teammates must test with physical separation ($\ge 2\text{m}$) or use headphones on Phone 1. |

---

## 3. Comprehensive File Changelog

### A. Python Audio Relay Service (`relay-service/`)
Newly created standalone FastAPI microservice mediating between phones and the detection backend:
- [`relay-service/server.py`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/server.py): FastAPI app with endpoints:
  - `WS /ws/caller`: Accepts Caller stream, handles audio frames and spoof toggle commands.
  - `WS /ws/receiver`: Feeds converted/genuine audio and telemetry to Receiver.
  - `GET /health`: Returns schema matching Flutter's `BackendHealth.fromJson`.
- [`relay-service/session.py`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/session.py): `CallSession` coordinator managing caller/receiver pairing, spoof state, audio transformation, and broadcasting.
- [`relay-service/backend_client.py`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/backend_client.py): WebSocket client communicating with `realtime-backend` (`ws://127.0.0.1:8000/ws`), handling handshakes, audio pushes, and fast-transition resets.
- [`relay-service/audio/crossfade.py`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/audio/crossfade.py): `ChunkCrossfadeProcessor` implementing context overlap and raised-cosine crossfades.
- [`relay-service/audio/buffer.py`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/audio/buffer.py): `DropOldestQueue` bounded async leaky queue.
- [`relay-service/audio/pcm.py`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/audio/pcm.py): 16-bit PCM conversion, RMS audio level calculations, and silence gating.
- [`relay-service/converters/base.py`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/converters/base.py), [`mock.py`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/converters/mock.py), [`rvc.py`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/converters/rvc.py), [`factory.py`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/converters/factory.py): Extensible voice conversion engine (Mock pitch-shifter + Real RVC hook).
- [`relay-service/tests/`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/tests/): 17 comprehensive unit/integration tests verifying all audio pipelines and WebSocket connections.
- [`relay-service/requirements.txt`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/requirements.txt), [`README.md`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/README.md), [`pytest.ini`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/relay-service/pytest.ini).

### B. Flutter Mobile Application (`voice_integrity_flutter/`)
- [`lib/features/live_call/live_call_screen.dart`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/voice_integrity_flutter/lib/features/live_call/live_call_screen.dart):
  - Primary interactive live call defense console.
  - Mode selector pill: Caller Mode (Attacker) vs. Receiver Mode (Victim).
  - Responsive connection control card with collapsible host/port configuration.
  - Caller UI: Glowing red animated border during attack, spoof duration timers, VU audio level meter, and responsive spoof toggle button.
  - Receiver UI: 4-band risk banner (Low/Amber/High/Unavailable), 1.5s amber transition state, calibrated probability gauge, loudspeaker playback controls, and multi-expert breakdown.
- [`lib/core/services/live_call_service.dart`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/voice_integrity_flutter/lib/core/services/live_call_service.dart): WebSocket streaming client managing capture/playback lifecycles and fallback simulation.
- [`lib/core/services/audio_capture_service.dart`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/voice_integrity_flutter/lib/core/services/audio_capture_service.dart): High-frequency 16kHz PCM audio recorder.
- [`lib/core/services/audio_playback_service.dart`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/voice_integrity_flutter/lib/core/services/audio_playback_service.dart): Streaming playback service with RIFF header builder and loudspeaker forcing.
- [`lib/core/services/pcm_resampler.dart`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/voice_integrity_flutter/lib/core/services/pcm_resampler.dart): Polyphase downsampler ensuring uniform 16kHz mono audio.
- [`lib/core/models/live_call_models.dart`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/voice_integrity_flutter/lib/core/models/live_call_models.dart): Immutable state models (`CallMode`, `LiveRiskLevel`, `LiveRiskAssessment`, `LiveExpertScore`, `CallStats`).
- [`lib/features/dashboard/voice_integrity_shell.dart`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/voice_integrity_flutter/lib/features/dashboard/voice_integrity_shell.dart):
  - Updated to incorporate `_Destination.liveCall` navigation.
  - Enhanced error handling for socket/connection exceptions to show clean diagnostic cards.
  - Styled system readiness card to match design tokens.
- Platform Manifests Updated:
  - Android: `RECORD_AUDIO`, `INTERNET`, `MODIFY_AUDIO_SETTINGS` in `AndroidManifest.xml`.
  - iOS: `NSMicrophoneUsageDescription` in `Info.plist`.
  - macOS: `com.apple.security.device.audio-input` and `network.client` entitlements.
- [`test/live_call_test.dart`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/voice_integrity_flutter/test/live_call_test.dart): 15 unit and widget tests passing.

### C. Real-Time Detection Backend (`realtime-backend/`)
- [`realtime-backend/.env`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/realtime-backend/.env):
  - Configured `EXPERTS=wavlm,hybrid` with `FUSION_MODE=heuristic_avg` and `SINGLE_EXPERT=hybrid`.
  - Resolves Python 3.12 compatibility by omitting legacy `fairseq` while keeping the two primary HuggingFace models (`WavLM Base+` and `LFCC-LCNN Hybrid`).

### D. Documentation & Team Guides
- [`TEAM_TESTING_GUIDE.md`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/TEAM_TESTING_GUIDE.md): End-to-end testing guide covering local setup, port reversals (`adb reverse`), mobile hotspot configuration, 3 test scenarios, and troubleshooting.
- [`LIVE_DEMO_AND_CHANGES_STATE.md`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/LIVE_DEMO_AND_CHANGES_STATE.md): This state document.

---

## 4. UI Bugs Fixed & Verification Details

1. **Live Call Flex Overflow (`RenderFlex overflowed by 62 pixels on the right`):**
   - Fixed by changing rigid `Row` layout into responsive `Wrap` with aligned children, preventing overflows on narrow widths.
2. **`ClientException: Connection closed before full header was received`:**
   - Handled in `voice_integrity_shell.dart` (`_friendlyError` and `_EndpointCard`), converting socket connection resets into actionable instructions.
3. **Probability Gauge Legends Overflow:**
   - Wrapped legend items in `Expanded` widgets to ensure 3-column legend wraps cleanly on small device screens.
4. **Spoof Button Label Truncation:**
   - Wrapped button text in `FittedBox(fit: BoxFit.scaleDown)` to scale smoothly rather than ellipsis.
5. **System Readiness Card Error Display:**
   - Replaced plain text error with a themed alert container matching Material 3 tokens.
6. **State Rebuild Anti-Pattern:**
   - Replaced `(context as Element).markNeedsBuild()` with `onToggleSimulation` reactive callback in `_ConnectionControlCard`.

---

## 5. Verification & Live Test Results

- **Relay Tests:** 17/17 tests passing (`pytest tests/`).
- **Flutter Tests:** 15/15 tests passing (`flutter test`).
- **Flutter Linter:** `flutter analyze` reports 0 issues.
- **Android Live Verification (Emulator API 36):**
  - App launched cleanly, connected to backend (`http://127.0.0.1:8000/health`) showing **🟢 Backend ready**.
  - Connected to relay (`ws://127.0.0.1:8001/ws/caller`), completed backend handshake.
  - Tapped **Talk as Teammate 3 (Spoof)**: visual glow activated, relay confirmed `Fast transition triggered: flushing backend RingBuffer & EMA`.
- **Git Commit:** Rebased and pushed to `origin main` as commit [`d043fd0`](https://github.com/Akenzz/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/commit/d043fd0).
- **Background Tasks:** Both ports `8000` and `8001` were verified stopped and freed.

---

## 6. How to Resume or Re-run the System

To spin up the system again:

```bash
# 1. Start Detection Backend (:8000)
cd realtime-backend
.venv/bin/python server.py

# 2. Start Relay Service (:8001)
cd relay-service
./.venv/bin/python server.py

# 3. For Android Emulator: Reverse ports
adb reverse tcp:8000 tcp:8000 && adb reverse tcp:8001 tcp:8001

# 4. Launch Flutter Mobile App
cd voice_integrity_flutter
flutter run -d <device_id>
```

For two physical phones on a Wi-Fi network, refer to §3 Option B in [`TEAM_TESTING_GUIDE.md`](file:///Users/kartiknhm/stuff/sih/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/TEAM_TESTING_GUIDE.md).
