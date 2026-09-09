# Voice Integrity Relay Service (`relay-service/`)

High-performance real-time Python Relay Service connecting **Caller phones**, **Receiver phones**, and the **ML Realtime Detection Backend** (`realtime-backend/`), featuring live voice conversion (RVC engine hook & DSP fallback) and continuous deepfake detection score streaming.

---

## Architecture Overview

```
                          ┌─────────────────────────────┐
                          │         Caller Phone        │
                          │   (Microphone Audio / PCM)  │
                          └──────────────┬──────────────┘
                                         │ WS /ws/caller
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                     Relay Service                                      │
│                                                                                        │
│   ┌──────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────┐   │
│   │   Bounded Queue      │    │  VoiceConverter Engine  │    │ Raised-Cosine Fade  │   │
│   │ (maxsize=3, drop-old)│───►│ • MockRVC / DSP Fallback│───►│ 25ms crossfade      │   │
│   │  Latency Prevention  │    │ • Real RVC (Torch)      │    │ Soft-limit <= 0.92  │   │
│   └──────────────────────┘    └─────────────────────────┘    └──────────┬──────────┘   │
│                                                                         │              │
│                ┌────────────────────────────────────────────────────────┴────┐         │
│                │                                                             │         │
└────────────────┼─────────────────────────────────────────────────────────────┼─────────┘
                 │ (Forwarded Audio Chunks)                                    │ (Forwarded Audio)
                 ▼                                                             ▼
┌─────────────────────────────────┐                       ┌──────────────────────────────┐
│         Receiver Phone          │                       │   Realtime Backend /ws       │
│  • Plays audio (clean or clone) │                       │  • RingBuffer & VAD          │
│  • Displays live spoof scores   │◄──────────────────────│  • LFCC, WavLM, TakHemlata   │
│  • Visualizes risk state        │  (Score Stream JSON)  │  • Calibrated Ensemble Score │
└─────────────────────────────────┘                       └──────────────────────────────┘
```

---

## Key Features & Red Team Mitigations

1. **Strict Handshake & Readiness Guard**:
   - Maintains connection to the ML backend (`ws://127.0.0.1:8000/ws`).
   - Sends initial handshake `{"type": "start", "sample_rate": 16000, "encoding": "pcm_s16le", "channels": 1, "binary_seq": false}` and awaits `{"type": "ready"}` before streaming binary frames.
2. **Latency Drift Prevention**:
   - Caller stream buffered using `DropOldestQueue(maxsize=3)` to guarantee latency cannot accumulate if network or compute jitters.
   - `TCP_NODELAY` enabled on all WebSocket transport sockets to prevent Nagle buffering.
3. **Fast Transition on Spoof Toggle**:
   - When Caller toggles spoof (`{"type": "toggle_spoof", "enabled": bool}`), relay issues a fresh `start` handshake to the backend.
   - Flushes backend's 4.0s RingBuffer and EMA smoothing, slashing spoof transition detection latency from **~5.0s down to ~1.2s**!
   - Broadcasts `{"type": "transition_state", "status": "switching", "spoof_enabled": bool}` to the receiver.
4. **RVC Chunking & Crossfading**:
   - 150ms historical context prepended to conversion blocks to give pitch trackers and vocoders continuous phase.
   - 25ms raised-cosine crossfading applied across chunk boundaries to eliminate clicks and phase glitches.
   - Peak amplitude soft-limited to `[-0.92, 0.92]` using smooth hyperbolic tangent saturation, strictly preventing backend `quality.py`'s 1% clipping rejection (`clip_abs = 0.99`, `clip_fraction = 0.01`).
5. **Modular Voice Conversion (`VoiceConverter`)**:
   - `MockRVCConverter`: Pure NumPy/SciPy phase-vocoder pitch shifting, formant warping, and HiFi-GAN neural vocoder artifact injection. Provides preset speaker profiles (`clone_female`, `clone_male`, `clone_deep`, `clone_child`, `robotic`). Runs immediately on CPU/macOS without GPU or model weights.
   - `PitchFormantFallbackConverter`: Parametric DSP fallback engine.
   - `RealRVCConverter`: Production wrapper for `rvc-python` or PyTorch RVC `.pth` inference, with seamless DSP fallback when weights or CUDA are not available.

---

## WebSocket Protocol Specification

### 1. Caller Endpoint (`/ws/caller?session_id=<id>`)

- **Inbound Audio**:
  - Binary frames: raw PCM bytes (16-bit signed little-endian, 16kHz, mono).
  - JSON frames: `{"type": "audio", "data": "<base64_pcm>"}`.
- **Inbound Controls**:
  - `{"type": "toggle_spoof", "enabled": true}`: Enable voice clone attack simulation.
  - `{"type": "toggle_spoof", "enabled": false}`: Revert to clean bonafide audio.
  - `{"type": "set_profile", "profile": "clone_male"}`: Select active clone profile.
- **Outbound Responses**:
  - `{"type": "ready", "session_id": "...", "sample_rate": 16000, "encoding": "pcm_s16le", "spoof_enabled": false}`
  - `{"type": "spoof_status", "enabled": bool}`

### 2. Receiver Endpoint (`/ws/receiver?session_id=<id>`)

- **Outbound Audio**:
  - Binary PCM frames (16-bit signed LE, 16kHz, mono) forwarded in real-time.
- **Outbound Detection Scores (JSON)**:
  ```json
  {
    "type": "score",
    "event": "window_scored",
    "window_index": 12,
    "risk_state": "spoof",
    "calibrated_probability": 0.865,
    "weighted_probability": 0.865,
    "per_expert_probability": {
      "wavlm": 0.62,
      "hybrid": 0.91,
      "ssl": 0.88
    },
    "latency_ms": 14.8
  }
  ```
- **Outbound Transitions & Status**:
  - `{"type": "transition_state", "status": "switching", "spoof_enabled": true, "timestamp": 1725880000.0}`
  - `{"type": "caller_status", "event": "caller_connected", "caller_connected": true}`

---

## Quickstart & Usage

### 1. Installation

Using `uv` (recommended) or standard `python -m venv`:

```bash
cd relay-service
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Running the Relay Service

```bash
# Default port 8080, connecting to detection backend at ws://127.0.0.1:8000/ws
python server.py

# Custom host/port or backend URL
RELAY_PORT=8080 BACKEND_WS_URL=ws://127.0.0.1:8000/ws python server.py
```

### 3. Running Zero-Packet-Drop Verification CLI (`client_sim.py`)

Simulates a caller and receiver running in parallel, streams audio, toggles spoof mid-call, and verifies zero packet drop:

```bash
# Full automated simulation with synthetic speech sweep
python client_sim.py --relay-url ws://127.0.0.1:8080 --duration 5.0 --toggle-spoof-at 2.5

# Test with a specific WAV clip
python client_sim.py --relay-url ws://127.0.0.1:8080 --wav path/to/sample.wav --duration 6.0
```

### 4. Running Unit & Integration Tests

```bash
pytest -v
```

All 17 tests verify audio conversions, bounded queues, raised-cosine crossfading, soft-limiting, backend handshake, fast transitions, and end-to-end integration.
