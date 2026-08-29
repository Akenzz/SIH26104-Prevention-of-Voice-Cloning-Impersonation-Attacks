# Voice Integrity Monitor — Frontend

The demo dashboard for **SIH26104 · Prevention of Voice-Cloning & Impersonation Attacks**.
It shows, in real time, how likely live call audio is to be synthetic — and recommends
stepping up verification. This is the whole-project demo surface (build-plan **Task E**),
plus the live reasoning panel (**Task F**) and the call simulator (**Task G**).

> **Honest framing (please keep it):** this is *decision support, not proof of identity*.
> It does not prove who is calling and does not stop fraud on its own. Scores are only
> meaningful on audio similar to what the detector was measured on (ASVspoof2019 LA).
> The UI states this on purpose — see [`Disclaimer.jsx`](src/components/layout/Disclaimer.jsx).

---

## Quick start

```bash
npm install
npm run dev        # http://localhost:5173 (Vite prints the exact port)
```

The dashboard runs on its own. Pick **Simulation** as the source and hit **Run simulation** —
no backend needed. That path is also the demo-day backup.

To score **real audio**, run the backend too (separate terminal, from the repo root):

```bash
cd realtime-backend
python -m pip install huggingface_hub transformers soundfile     # one-time
EXPERTS=wavlm,lfcc SINGLE_EXPERT=lfcc FUSION_MODE=single DEVICE=cpu python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

That loads **both** teammates' detectors — `wavlm` (role A) and `lfcc` (role D) — so the
Experts panel shows a logit from each. Model weights auto-download on first start (~380 MB
checkpoint + ~377 MB WavLM backbone), so the first boot takes a few minutes; later boots read
the cache. For the LFCC-only fast path, use `EXPERTS=lfcc` and drop `SINGLE_EXPERT`.

> **Two things to know before you read a number.** `SINGLE_EXPERT` picks which expert's logit
> becomes the calibrated probability; with it unset the backend silently uses the *first* name in
> `EXPERTS`. The shipped calibrator is `platt-lfcc-asvspoof19dev-v1` — fitted on LFCC logits — so
> pointing it at WavLM pins the probability at 100% on everything. And WavLM on CPU costs
> ~2.5–3 s per 4 s window against the 500 ms budget the console displays, so **MAX LATENCY will
> read over budget**; that is the honest number, not a bug in the dashboard.

> **PowerShell note:** `EXPERTS=lfcc python ...` does **not** work in PowerShell (that syntax
> is bash/macOS/Linux only). In PowerShell set the vars first:
> ```powershell
> $env:EXPERTS="wavlm,lfcc"; $env:SINGLE_EXPERT="lfcc"; $env:FUSION_MODE="single"; $env:DEVICE="cpu"
> python -m uvicorn server:app --host 127.0.0.1 --port 8000
> ```
> Or use Git Bash / WSL, where the one-line form works.

The dev server proxies `/health` and `/ws` to `127.0.0.1:8000`, so the browser talks to the
backend same-origin (no CORS). Override the target with `BACKEND_ORIGIN` — see
[`vite.config.js`](vite.config.js).

---

## Where is what (file map)

```
voice-integrity-frontend/
├─ index.html              ← <head>: fonts, favicon, title, theme-color
├─ vite.config.js          ← dev-server proxy + the "@" → /src import alias
├─ test-samples/
│  ├─ *.wav                ← synthetic clips (silence, clipped, tone, sweep, noise) — plumbing only
│  └─ real-calls/          ← real ASVspoof2019 LA speech: genuine vs cloned, same speaker
├─ scripts/
│  ├─ probe_experts.py     ← CLI: stream a clip over /ws, print each expert's logit
│  └─ fetch_real_calls.py  ← downloads ASVspoof LA dev clips, stitches call-length WAVs
│
└─ src/
   ├─ main.jsx             ← React entry. Loads styles/tokens.css. Don't edit.
   ├─ App.jsx              ← PAGE LAYOUT — the 3-column grid; where panels are placed
   │
   ├─ styles/                        ★ DESIGN TEAM STARTS HERE
   │  ├─ tokens.css        ← colors, fonts, radii, shadows (CSS variables). Retheme here.
   │  └─ app.css           ← all layout + per-component styling
   │
   ├─ components/
   │  ├─ layout/           ← page frame
   │  │  ├─ StatusRail.jsx    top bar: brand + live risk chip
   │  │  └─ Disclaimer.jsx    honesty banner (keep the wording)
   │  ├─ panels/           ← the four console panels (content + readouts)
   │  │  ├─ SourcePanel.jsx    control deck: source / scenario / mic / file / backend
   │  │  ├─ VerdictPanel.jsx   the hero readout: risk band + gauge + action
   │  │  ├─ ExpertPanel.jsx    per-expert logits, fusion/calibration, stream integrity
   │  │  └─ ReasoningLog.jsx   streaming grounded narration (Task F)
   │  ├─ viz/              ← data visualizations
   │  │  ├─ Gauge.jsx          270° spoof-probability arc gauge (SVG)
   │  │  ├─ TimelineChart.jsx  risk-over-time strip chart (recharts)
   │  │  └─ SignalStrip.jsx    live waveform + input-level meter (canvas)
   │  └─ ui/              ← reusable primitives
   │     ├─ primitives.jsx     Panel, Pill, Stat, Segmented
   │     └─ icons.jsx          all inline SVG icons + <RiskGlyph>
   │
   └─ lib/                 ← app logic. Rarely touched for design work.
      ├─ useMonitor.js         orchestration hook — the app's single source of state
      ├─ contract.js           risk bands, colors, labels, formatters (shared vocabulary)
      ├─ simulator.js          scripted scenarios for the Simulation source
      ├─ narrator.js           turns detector numbers into narration lines
      ├─ audio.js              mic + file capture, PCM framing (Web Audio)
      └─ backendClient.js      /health fetch + /ws WebSocket client
```

---

## Design team: how to work here

- **Reskin colors, type, spacing:** edit [`src/styles/tokens.css`](src/styles/tokens.css) first —
  everything reads from those CSS variables. Component-specific rules live in
  [`src/styles/app.css`](src/styles/app.css), grouped by component with `/* ---- name ---- */`
  banners.
- **Change how a panel looks:** open its `.jsx` for the markup/class names, then style those
  classes in `app.css`. The `.jsx` files hold structure and logic; keep visual values in CSS.
- **Rearrange the page:** [`App.jsx`](src/App.jsx) is the only place panels are positioned
  (the `.console` grid). The responsive breakpoints are at the bottom of `app.css`.
- **Icons:** all hand-rolled inline SVG in [`icons.jsx`](src/components/ui/icons.jsx) (stroke =
  `currentColor`, so they inherit text color). Add new ones there — we intentionally don't use an
  icon package.
- **Accessibility floor to preserve:** risk is always shown by **icon + label**, never color
  alone; there's a visible focus ring; `prefers-reduced-motion` is respected. Please don't
  regress these.
- **Don't hand-edit** `lib/` unless you're changing behavior — the panels get all their data as
  props from `useMonitor`.

### Import paths

Use the `@` alias for cross-folder imports so paths don't break when files move:

```js
import { pct } from "@/lib/contract.js";
import { Panel } from "@/components/ui/primitives.jsx";
```

`@` resolves to `src/` (configured in `vite.config.js`). Same-folder imports can stay relative.

---

## Testing with real data

### Get real speech first

`test-samples/*.wav` are generated tones, sweeps and noise. They exercise the fail-safe and the
plumbing; they are **not speech**, so any verdict from them is meaningless. For a real
genuine-vs-cloned demo, build call-length clips from ASVspoof2019 LA:

```bash
python -m pip install huggingface_hub soundfile numpy
python scripts/fetch_real_calls.py
```

That writes `test-samples/real-calls/` — a 24 s genuine call and a 24 s cloned call **from the same
speaker** (defaults: speaker `LA_0076`, spoof system `A06`, spectral-filtering voice conversion),
plus a few single utterances and a `MANIFEST.txt` naming every source utterance. Pass
`--attack A01` for neural TTS instead, or `--speaker`/`--seconds` to vary it.

> These come from the LA **dev** partition — the same partition the shipped calibrator
> `platt-lfcc-asvspoof19dev-v1` was fitted on. Scores are in-domain and flattering. They prove the
> system works end to end; they are **not** a held-out accuracy measurement. That needs the LA
> **eval** partition.

### Run it

1. Start the backend (above) and the dev server.
2. In the dashboard: **Source → Live backend**, leave the backend field empty, click **Check** →
   expect `online · experts: wavlm, lfcc`.
3. Choose **Microphone** and **Start monitor** (grant the mic prompt), or **Audio file** →
   **Analyze file**. A file uploads far faster than the detector scores it, so when playback ends
   the console shows `finishing last windows…` and keeps the socket open until the backend goes
   quiet — that is the drain, not a hang.

Measured on this machine with `SINGLE_EXPERT=lfcc`, both files through the browser:

| File | Verdict | wavlm | lfcc | Windows |
|---|---|---|---|---|
| `call_genuine_LA_0076.wav` | **Low risk · 0%** | −0.03 | −6.56 | 13 |
| `call_cloned_A06_LA_0076.wav` | **High risk · 99%** | +2.35 | +4.48 | 23 |

Want the raw per-expert numbers without the UI? Stream a clip straight at the socket:

```bash
python scripts/probe_experts.py --wav test-samples/real-calls/call_cloned_A06_LA_0076.wav
```

It prints one line per window with each expert's logit, so you can confirm role A and role D are
both scoring. (The backend's own `scripts/wav_client.py` drains with 50–200 ms timeouts and gives
up with "No score messages received" when WavLM is in the mix — that is why this exists.)

On the cloned call, `lfcc` stayed firmly positive (+2.7…+6.0) across every window while `wavlm`
briefly dipped negative at one window — a concrete example of why two complementary experts help.

**What the number means depends on the audio:**

| Audio | Result |
|---|---|
| **ASVspoof2019 LA** clip | Trustworthy — the only domain the calibrator was validated on |
| Your mic / own recordings / music | Real model output but **out-of-domain** → treat the probability as unreliable |
| Near-silence or heavily clipped | Should show **unavailable** (fail-safe), not a fake "low" |

> **Don't change the capture sample rate.** [`audio.js`](src/lib/audio.js) asks for a 16 kHz
> `AudioContext` on purpose. Left at the hardware default (48 kHz) the browser upsamples the clip
> and the backend decimates it back to 16 kHz; that round trip smears exactly the high-frequency
> detail the spoof detectors key on. It flipped the genuine call above from **low** to **high**
> — a silently wrong verdict, with nothing in the UI to hint at it.

---

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Dev server with HMR + backend proxy |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Serve the built `dist/` locally |
| `npm run lint` | ESLint |

## Stack

Vite 8 · React 19 · Tailwind CSS v4 · recharts 3. Fonts: Space Grotesk (display),
IBM Plex Sans (body), IBM Plex Mono (telemetry), loaded in `index.html`.
