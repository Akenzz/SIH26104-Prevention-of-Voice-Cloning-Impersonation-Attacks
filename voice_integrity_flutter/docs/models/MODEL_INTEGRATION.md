# Voice Integrity model integration guide

This is the review document for the Flutter client. It describes what the
**current backend code** uses—not a product claim that the models can identify
who is speaking. The source of truth is `../../realtime-backend/config.py`,
`pipeline.py`, `server.py`, and the matched artifacts in
`../../realtime-backend/artifacts/`.

## What the product predicts

The label convention is fixed across the project:

| Label | Integer | Meaning |
| --- | ---: | --- |
| `bonafide` | 0 | Human / real speech |
| `spoof` | 1 | AI-generated or cloned speech |

A higher raw logit and a higher calibrated probability mean **more likely
synthetic audio**. This is a classifier for synthetic-speech characteristics;
it is not speaker verification, identity proof, or fraud prevention by itself.

## Audio path before the models

1. The backend converts uploaded/stereo audio to mono and resamples it to
   **16 kHz**.
2. The realtime path uses **4.0-second windows** with a **0.5-second hop**;
   file analysis extracts the same window and hop sizes.
3. Quality gates reject silence, clipping, malformed PCM, protocol errors, and
   dropped/reordered frames as `unavailable`. The Flutter UI must never render
   that condition as “low risk.”
4. Models score each clean window with a raw logit. The backend calibrates the
   logits and applies the decision policy after smoothing/aggregation.

## Runtime experts

The current `HUB_EXPERTS` map defines these available experts:

| Runtime key | Checkpoint source / file | Role | How the client presents it |
| --- | --- | --- | --- |
| `wavlm` | `Akenzz/SIH-Models` / `wavlm_best_model_v5.pt` | WavLM Base+ representation with a classifier head. It adds an independent speech-representation signal. | “Expert 1 · WavLM Base+” |
| `hybrid` | `sarosh22/Hybrid_new` / `hybrid_clean_plus_newclips_final.pth` | LFCC-LCNN spectral-forensics model, fine-tuned with 12 modern engine clips. | “Expert 2 · LFCC-LCNN Hybrid” |
| `ssl` | `Akenzz/SIH-Models` / `best_SSL_model_LA.pth` | TakHemlata SSL expert, providing a third independent signal. | “Expert 3 · TakHemlata SSL” |
| `hybrid_br` | local-only / `hybrid_br_best.pth` | Optional LFCC-LCNN retrain with a 7 kHz parity-band gate. It reduces the prior resampler shortcut. | “Expert 2c · bandwidth-robust” when the backend enables it |

At the time this guide was written, the default `Settings` configuration loads
`wavlm,hybrid,ssl`, uses `FUSION_MODE=heuristic_avg`, and lists `hybrid` as the
single-expert fallback. Do not infer the active production path from this table:
the Flutter app calls `GET /health` and displays the backend’s reported active
configuration.

## Calibration: the non-negotiable rule

Raw logits from different models are **not** probabilities on the same scale.
Each expert has its own Platt calibrator:

| Expert | Current mapped artifact | Notes |
| --- | --- | --- |
| `wavlm` | `platt_v5.json` | Metadata identifies `wavlm-base-plus-ep6`; a refit follows any weight or corpus change. |
| `hybrid` | `calibrator_hybrid_newclips.json` | The artifact metadata uses the historical name `hybrid_nc`; the runtime key remains `hybrid`. Treat the mapping in `config.py` as authoritative. |
| `ssl` | `platt_ssl.json` | Sklearn Platt calibrator for the TakHemlata SSL model. |
| `hybrid_br` | `calibrator_hybrid_br.json` | Must only be used with the matching 7 kHz gated checkpoint. |

The mapping is important enough to be enforced by backend startup checks. A
foreign calibrator can still return a number between zero and one, but that
number can be confidently wrong. For this reason, the Flutter UI labels
per-model probabilities as **independent evidence** and does not recalculate
probabilities on-device.

## How a risk band is derived

The policy in `artifacts/policy.json` is:

| Probability | Risk state | User-facing action |
| ---: | --- | --- |
| first two clean windows | `collecting` | Keep the call going; there is not enough audio yet. |
| `< 0.35` | `low` | Continue, while using out-of-band verification for sensitive steps. |
| `0.35–0.649…` | `uncertain` | Pause the request and use a known callback or MFA. |
| `≥ 0.65` | `high` | Pause and verify through a known callback number. |
| invalid quality / stream integrity | `unavailable` | Do not treat it as a real-speech score. |

### File-analysis endpoint used by the Flutter app

The Flutter file-review screen uses `POST /predict-file`. Per valid window,
`server.py` first takes a **cumulative mean of each expert’s raw logits**, then
calibrates each mean using that expert’s own calibrator. It combines the
calibrated probabilities in probability space using these current constants:

```
final = 0.20 × WavLM + 0.20 × LFCC-LCNN Hybrid + 0.60 × SSL
```

If an expert is absent, the available weights are re-normalized. A strong
agreement safeguard promotes the result to at least `0.80` when Hybrid is over
`0.85` and WavLM or SSL also exceeds `0.60`. The endpoint emits SSE frames:

- `window_scored`: time, decision probability, risk state, raw logits, and
  calibrated probability for each expert;
- `summary`: overall risk state, final probability, agreement/confidence, peak
  time, counts, and expert summaries.

The Flutter client parses those frames into `AnalysisReport`, `WindowScore`, and
`ExpertScore`; it uses the backend’s returned probability rather than recreating
the ensemble locally.

### Realtime endpoint

The backend also exposes `WS /ws`. It needs a JSON `start` frame declaring PCM
format and sample rate before binary or JSON audio frames. The server applies
stateful resampling, scores each completed window, smooths probabilities with
EMA, and emits score messages. The current Flutter app has a **guided local
walkthrough**, clearly marked as showcase data; microphone capture and the
WebSocket transport are intentionally listed as the next module rather than
being implied as complete.

## Known model and documentation risks

1. **Resampling mismatch is a known silent failure.** Training previously used
   librosa/soxr while serving used scipy; the 7.9–8 kHz difference became a
   label proxy. `hybrid_br` addresses it by low-passing at 7 kHz during both
   training and inference. See `../../CALIBRATION-AND-RESAMPLING.md`.
2. **The 6–7 kHz corpus imbalance remains open.** It may contribute to false
   alarms; do not describe the output as certain.
3. **`hybrid_br` is local-only.** A fresh clone cannot download it until the
   checkpoint is published, so it is not a turnkey default.
4. **Repository documentation has drift.** Some README text says “two
   experts,” some uses older Hub IDs, and `/health` currently labels
   `heuristic_avg` as “LFCC + SSL Average,” while `pipeline.py` actually uses
   WavLM, Hybrid, and SSL with 20/20/60 weights. Review runtime code and the
   health payload before presenting a configuration as final.

## Reviewer checklist before demo or deployment

- Start the backend and use the Flutter Models page to check `/health`.
- Confirm the active experts, fusion mode, and calibrator versions match the
  run you are about to demonstrate.
- Use actual speech—not generated tones—to test the end-to-end path.
- Keep the limitations message visible in screenshots and presentations.
- After every new checkpoint, refit the appropriate calibrator and rerun the
  resampling parity/operating-point checks documented in the repository.
