# Voice Integrity — Team Testing & Demo Guide
> **SIH26104: Prevention of Voice Cloning & Impersonation Attacks**  
> Live Two-Phone Voice Cloning Fraud Demonstration & Detection System

---

## 1. System Architecture Overview

This project implements a complete, end-to-end defense system against real-time voice cloning fraud. The architecture is split across three clean layers:

```
┌─────────────────────────┐                ┌────────────────────────┐                ┌─────────────────────────┐
│   Caller Phone (Fraud)  │                │   Host Laptop Relay    │                │  Receiver Phone (Victim)│
│  • 16 kHz Mono Mic Rec  │  Raw PCM/WS    │  • Crossfade Processor │  Audio + Risk  │  • 120ms Jitter Queue   │
│  • "Talk as Teammate 3" ├───────────────►│  • Voice Converter     ├───────────────►│  • Forced Loudspeaker   │
│    Spoof Toggle Button  │  (:8001/ws/    │  • RingBuffer Reset    │  (:8001/ws/    │  • 4-Band Risk Gauge    │
│  • Visual Red Glow UI   │   caller)      │  • Dropping Queue      │   receiver)    │  • 1.5s Amber Transition│
└─────────────────────────┘                └───────────┬────────────┘                └─────────────────────────┘
                                                       │
                                                       │ Audio Frames
                                                       ▼
                                           ┌────────────────────────┐
                                           │ Realtime ML Backend    │
                                           │  • WavLM Base+ (20%)   │
                                           │  • LFCC-LCNN (20%)     │
                                           │  • Heuristic Fusion    │
                                           │  • Platt Calibrators   │
                                           │  (:8000/ws & /health)  │
                                           └────────────────────────┘
```

### Why This Design? (Key Engineering Decisions)
1. **Decoupled Heavy Compute:** Heavy ML inference (HuBERT, WavLM, RVC) runs on the host laptop rather than mobile devices, ensuring sub-200ms latency and high battery efficiency.
2. **Context-Preserving Audio Relay:** Processes continuous 200ms blocks with 150ms acoustic context, 25ms raised-cosine crossfades, and soft-limiting to eliminate clicks, pops, and phase artifacts.
3. **Instantaneous Spoof Detection:** When the Caller toggles Spoof, the relay sends a fast-transition reset command that flushes the backend's 4.0s RingBuffer and EMA smoothing, switching the risk score from Low Risk to High Risk without a 5-second lag.
4. **Resilient Mobile Audio Playback:** The Flutter app uses custom RIFF-WAV stream chunking with a 120ms jitter queue and forces audio to the device loudspeaker (`playAndRecord` mode).

---

## 2. Quick Start: Running the Entire Stack

### Prerequisites
- **Python:** 3.10, 3.11, or 3.12
- **Package Manager:** `uv` (recommended) or standard `pip`
- **Flutter SDK:** 3.19 or higher
- **Android Platform Tools:** `adb` (installed with Android Studio or Command Line Tools)

---

### Step 1: Start Real-Time Detection Backend (`:8000`)

The detection backend runs the multi-expert speech analysis models (WavLM Base+ and LFCC-LCNN Hybrid).

```bash
cd realtime-backend

# 1. Create and activate virtual environment (using uv or python)
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Start the server (binds to 0.0.0.0:8000)
python server.py
```

*Verification:*
In another terminal, test the health endpoint:
```bash
curl http://127.0.0.1:8000/health
```
You should receive `{"status":"ok","experts":["wavlm","hybrid"]...}`.

---

### Step 2: Start Audio Relay & Conversion Service (`:8001`)

The relay service mediates communication between the two phones and the detection backend.

```bash
cd relay-service

# 1. Create and activate virtual environment
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Start the relay (binds to 0.0.0.0:8001)
python server.py
```

*Verification:*
```bash
curl http://127.0.0.1:8001/health
```
You should receive `{"status":"ok","experts":["mock_rvc"]...}`.

---

### Step 3: Run the Flutter Mobile Application

```bash
cd voice_integrity_flutter

# Get dependencies and verify code integrity
flutter pub get
flutter test
```

---

## 3. Testing Options & Device Setup

Choose the setup matching your hardware:

### Option A: Single Laptop + Android Emulator (Quick Testing)

1. **Reverse ADB Ports:**
   Android emulators cannot access the host machine via `127.0.0.1` by default unless port forwarding is configured:
   ```bash
   adb reverse tcp:8000 tcp:8000
   adb reverse tcp:8001 tcp:8001
   ```
2. **Launch App on Emulator:**
   ```bash
   flutter run -d emulator-5554
   ```
3. The app is pre-configured to point to `127.0.0.1:8000` (detection) and `127.0.0.1:8001` (relay).

---

### Option B: Two Physical Phones + Laptop (Full Live Demo)

This is the ideal presentation setup for judges.

1. **Network Setup:**
   - Connect the laptop and both physical phones to the **same local Wi-Fi** or a **Mobile Hotspot** hosted on one phone.
   - Note the laptop's local IP address:
     - **macOS:** `ipconfig getifaddr en0` (e.g., `192.168.1.150`)
     - **Linux:** `hostname -I | awk '{print $1}'`
     - **Windows:** `ipconfig` (IPv4 address under Wireless LAN adapter)

2. **Install the APK on Both Phones:**
   Build the debug APK:
   ```bash
   cd voice_integrity_flutter
   flutter build apk --debug
   ```
   The APK is located at:
   `voice_integrity_flutter/build/app/outputs/flutter-apk/app-debug.apk`.  
   Transfer and install it on Phone 1 and Phone 2.

3. **Configure Phone 1 (Caller / Attacker):**
   - Open app -> Navigate to **Live call** tab (phone icon in bottom nav).
   - Tap **Caller Mode**.
   - Tap the gear icon (⚙) next to connection status.
   - Set **WebSocket Host** to your laptop's local IP (e.g., `192.168.1.150`).
   - Port is `8001`. Tap **Save**.

4. **Configure Phone 2 (Receiver / Victim):**
   - Open app -> Navigate to **Live call** tab.
   - Tap **Receiver Mode**.
   - Tap the gear icon (⚙).
   - Set **WebSocket Host** to your laptop's local IP (e.g., `192.168.1.150`).
   - Port is `8001`. Tap **Save**.

---

### Option C: Offline Presentation Mode (Zero Network / Pitch Backup)

If no Wi-Fi or local server is available during a presentation:
1. Open the app -> Navigate to **Live call** tab.
2. Tap the gear icon (⚙).
3. Switch on **Offline Presentation Simulation**.
4. Tap **Start Demo**.
   - This executes a fully realistic offline simulation generating live VU meters, calibrated probability transitions, reasoning logs, and multi-expert evidence.

---

## 4. Test Scenarios & Verification Walkthrough

### Scenario 1: Live Fraud Call Impersonation & Defense Walkthrough

| Step | Action on Caller Phone | Action on Receiver Phone | Expected System Behavior |
| :--- | :--- | :--- | :--- |
| **1. Standby** | Mode: **Caller Mode** | Mode: **Receiver Mode** | Both devices show status `Disconnected`. |
| **2. Connect** | Tap **Connect Call** | Tap **Connect Call** | Both phones show `Connected to ws://<host>:8001`. Receiver speaker initializes with 120ms jitter queue. |
| **3. Authentic Speech** | Caller talks normally: *"Hello, this is Kartik from the dev team."* | Receiver listens via speaker | Receiver hears genuine human voice. Live risk gauge stays in **Low Risk** (🟢 12–25%). Recommended Action: *"Allow normal transaction"*. |
| **4. Trigger Spoof Attack** | Tap **Talk as Teammate 3 (Spoof)** | Watch screen & listen to audio | • Caller button turns dark with **pulsing red outer glow**, badge shows `AI SPOOF: TEAMMATE 3`, timer counts spoof duration.<br>• Receiver immediately shows **1.5s Amber Transition Warning** (⏳ *"Model adapting to voice transition..."*).<br>• Receiver risk meter jumps to **High Risk** (🔴 ≥ 65%). Audio output transitions into the synthetic voice target.<br>• Action card triggers: *"Pause request and trigger secondary MFA / callback"*. |
| **5. Revert Attack** | Tap **Stop Spoof (Revert to Genuine)** | Watch screen & listen to audio | • Caller glow disappears; badge reverts to `GENUINE STREAM`.<br>• Receiver immediately transitions back to **Low Risk** (🟢 < 35%) authentic speech. |
| **6. Disconnect** | Tap **End Call** | Tap **End Call** | Microphone capture stops; speaker output terminates cleanly. |

---

### Scenario 2: Forensic Audio File Review

1. Tap **Review audio** tab in the bottom navigation.
2. Tap **Choose audio** to pick any local audio file (`.wav`, `.mp3`, `.flac`), or tap **Load judge demo**.
3. **Verify:**
   - Calibrated synthetic risk percentage gauge (0–100%).
   - Action recommendations: *Allow*, *Review Needed*, or *High Risk / Block*.
   - Multi-expert score breakdown:
     - **WavLM Base+** probability and raw logit.
     - **LFCC-LCNN Hybrid** probability and raw logit.
   - Interactive risk timeline graph showing probability per window over time.

---

### Scenario 3: Model & System Review Console

1. Tap **Models** tab in the bottom navigation.
2. **Backend Connection Card:**
   - Displays Backend URL (`http://127.0.0.1:8000` or custom IP).
   - Tap **Check connection**: Shows green badge `● Backend ready` with zero error alerts.
3. **The Three Experts Panel:**
   - Summarizes WavLM Base+, LFCC-LCNN, and TakHemlata SSL.
4. **Decision Formula:**
   - Calibrated probability weights (20% WavLM, 20% LFCC-LCNN, 60% SSL) and decision threshold logic (<35% Low, 35–64% Review, ≥65% High).

---

## 5. Troubleshooting & Frequently Asked Questions

### Q1: `ClientException: Connection closed before full header was received` or `SocketException: Connection refused`
- **Cause:** The app is attempting to connect to port 8000 or 8001, but the Python server is not running or the port is not accessible from the emulator/device.
- **Fix:**
  1. Confirm the backend process is running: `curl http://127.0.0.1:8000/health` and `curl http://127.0.0.1:8001/health`.
  2. If using the Android emulator, run:
     ```bash
     adb reverse tcp:8000 tcp:8000
     adb reverse tcp:8001 tcp:8001
     ```
  3. If using physical phones, ensure both the phone and laptop are on the same Wi-Fi and you have entered the laptop's LAN IP (not `127.0.0.1`) in the settings gear (⚙).

### Q2: Why does `realtime-backend` use `EXPERTS=wavlm,hybrid`?
- On modern Python (3.12+), Facebook's legacy `fairseq` library has compatibility issues with Python 3.12 dataclasses.
- We have configured `.env` with `EXPERTS=wavlm,hybrid`, loading the two primary state-of-the-art HuggingFace checkpoints (`WavLM Base+` and `LFCC-LCNN Hybrid`). This setup loads cleanly in ~5 seconds and delivers robust real-time scoring.

### Q3: Audio feedback or howling sound during testing
- **Cause:** If Phone 1 (Caller mic) is too close to Phone 2 (Receiver loudspeaker), the microphone will pick up the speaker audio and create an acoustic feedback loop.
- **Fix:** Keep Phone 1 and Phone 2 at least 2 meters apart, test in separate rooms, or wear headphones on Phone 1.

### Q4: Microphone permission denied on Android
- **Fix:** Go to Android **Settings -> Apps -> Voice Integrity -> Permissions -> Microphone -> Allow**.

---

## 6. Ports & Services Reference

| Port | Service | Repository Path | Primary Endpoints |
| :--- | :--- | :--- | :--- |
| **8000** | Real-Time Detection Backend | `realtime-backend/` | `GET /health`<br>`WS /ws`<br>`POST /analyze` |
| **8001** | Audio Relay & Voice Conversion | `relay-service/` | `GET /health`<br>`WS /ws/caller`<br>`WS /ws/receiver` |
| **Emulator** | Android App Runner | `voice_integrity_flutter/` | Device connection via `adb` |

---

*Prepared for the SIH26104 development team.*
