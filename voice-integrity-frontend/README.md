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
python -m pip install huggingface_hub                       # one-time
EXPERTS=lfcc FUSION_MODE=single DEVICE=cpu python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

> **PowerShell note:** `EXPERTS=lfcc python ...` does **not** work in PowerShell (that syntax
> is bash/macOS/Linux only). In PowerShell set the vars first:
> ```powershell
> $env:EXPERTS="lfcc"; $env:FUSION_MODE="single"; $env:DEVICE="cpu"
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
├─ test-samples/           ← generated .wav clips for exercising the live path
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

1. Start the backend (above) and the dev server.
2. In the dashboard: **Source → Live backend**, leave the backend field empty, click **Check** →
   expect `online · experts: lfcc`.
3. Choose **Microphone** and **Start monitor** (grant the mic prompt), or **Audio file** →
   **Analyze file**.

**What the number means depends on the audio:**

| Audio | Result |
|---|---|
| **ASVspoof2019 LA** clip (`.flac`) | Trustworthy — the only domain the calibrator was validated on |
| Your mic / own recordings / music | Real model output but **out-of-domain** → treat the probability as unreliable |
| Near-silence or heavily clipped | Should show **unavailable** (fail-safe), not a fake "low" |

`test-samples/` holds generated clips (silence, clipped, tone, sweep, white noise) — these are
**not** speech, so they're for exercising the fail-safe and the plumbing, not for a real verdict.
For a meaningful bonafide-vs-spoof demo you need an ASVspoof2019 LA file.

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
