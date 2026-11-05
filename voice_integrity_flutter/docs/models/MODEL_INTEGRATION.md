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
2. The realtime path uses **4.0-second windows** with a **1.0-second / 0.5-second hop**;
   file analysis extracts the same window and hop sizes.
3. Quality gates reject silence, clipping, malformed PCM, protocol errors, and
   dropped/reordered frames as `unavailable`. The Flutter UI must never render
   that condition as “low risk.”
4. Models score each clean window with a raw logit. The backend calibrates the
   logits and applies the decision policy after smoothing/aggregation.

## Runtime experts

The current `HUB_EXPERTS` map in `config.py` defines the active dual-expert ensemble:

| Runtime key | Checkpoint source / file | Role | How the client presents it |
| --- | --- | --- | --- |
| `wavlm` | `sarosh22/wavLM-Hybrid` / `wavlm_maxbr_best.pt` | WavLM Base+ transformer backbone with fine-tuned 2-layer classification head. Catches deep temporal anomalies and neural TTS patterns. | “Expert 1 · WavLM Base+” |
| `hybrid_maxbr` | `sarosh22/Final_LFCC` / `hybrid_maxbr_best.pth` | LFCC-LCNN Hybrid trained from scratch on 84,271 chunks across 6 Indian languages + 132 generators, with a strict 7,000 Hz parity gate. | “Expert 2 · LFCC-LCNN Hybrid (Decision Expert)” |
| `hybrid_br` | local-only / `hybrid_br_best.pth` | Bandwidth-robust warm-start baseline. | “Expert 2c · bandwidth-robust” |

*Note: Legacy `ssl` (TakHemlata) and `fairseq` dependencies were removed to ensure Python 3.12+ compatibility.*

The shipped default `Settings` configuration loads `wavlm,hybrid_maxbr`, uses `FUSION_MODE=heuristic_avg` (0.75 LFCC + 0.25 WavLM), and lists `hybrid_maxbr` as the decision expert. The Flutter app calls `GET /health` and displays the backend’s reported active configuration.

## Calibration: the non-negotiable rule

Raw logits from different models are **not** probabilities on the same scale.
Each expert has its own Platt calibrator:

| Expert | Current mapped artifact | Notes |
| --- | --- | --- |
| `wavlm` | `platt_v5.json` / `platt_v6.json` | Platt calibrator for the WavLM MaxBR head. |
| `hybrid_maxbr` | `calibrator_hybrid_maxbr.json` | Fitted on hybrid_maxbr gated dev split (band_gate_hz: 7000.0). |
| `hybrid_br` | `calibrator_hybrid_br.json` | Used only with the matching 7 kHz gated hybrid_br checkpoint. |

The mapping is enforced by backend startup checks. A foreign calibrator can still return a number between zero and one, but that number is uncalibrated and misleading.

## How a risk band is derived

The policy in `artifacts/policy.json` is:

| Probability | Risk state | User-facing action |
| ---: | --- | --- |
| first two clean windows | `collecting` | Keep the call going; there is not enough audio yet. |
| `< 0.35` | `low` | Continue, while using out-of-band verification for sensitive steps. |
| `0.35–0.649…` | `uncertain` | Pause the request and use a known callback or MFA. |
| `≥ 0.65` | `high` | Pause and verify through a known callback number. |
| invalid quality / stream integrity | `unavailable` | Do not treat it as a real-speech score. |

### File-analysis endpoint (`POST /predict-file`)

Per valid window, `server.py` evaluates the dual-expert models and fuses their calibrated probabilities using the configured fusion mode (`heuristic_avg`: 0.75 LFCC + 0.25 WavLM).

The endpoint emits SSE frames:
- `window_scored`: time, decision probability, risk state, raw logits, and calibrated probability for each expert;
- `summary`: overall risk state, final probability, agreement/confidence, peak time, counts, and expert summaries.
