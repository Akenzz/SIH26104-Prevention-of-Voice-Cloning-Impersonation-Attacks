# Calibration and Resampling — the two things that keep breaking

Read this before you retrain a model, change any audio preprocessing, or swap a
calibrator artifact. Both classes of bug below have already shipped once, and
both share the same property:

> **They never raise, never log, and never look wrong.** The probability stays in
> `[0, 1]`, the stream keeps running, the dashboard keeps updating. Only the
> answer is wrong. There is no test you can skip your way past — you have to
> measure the thing on purpose.

Everything here is measured on this repo, not inherited advice. Commands to
reproduce each number are in [§5](#5-command-reference).

---

## 0. The rules, if you read nothing else

0. **The shipped decision expert is `hybrid_maxbr`**, gated at 7000 Hz and scored
   with `calibrator_hybrid_maxbr.json`. `hybrid_br` is still loaded as a side card
   for comparison. See [§2](#2-calibration-which-artifact-belongs-to-which-model).
1. **One calibrator per expert. Never share.** Every model's logits live on their
   own scale. The mapping is `EXPERT_CALIBRATORS` in
   [config.py](realtime-backend/config.py) — it is the only source of truth.
2. **Whatever preprocessing training used, inference must use exactly that.**
   Same resampler, same filters, same cutoffs. "Close enough" is how the
   bandwidth bug happened.
3. **A retrain invalidates its calibrator.** New weights ⇒ new logit scale ⇒
   refit, or every probability you report is on a stale scale.
4. **Never quote an in-domain EER as a deployment number.** See
   [§3](#3-reading-the-numbers-honestly). Our headline 2.42% was really 27.52%.
5. **If you touch preprocessing, re-run `diag_band_gate.py`** and require the
   scipy-vs-librosa verdict gap to stay near 0.

---

## 1. Resampling: how a filter became the classifier

### What happened

Training preprocessing resampled to 16 kHz with **librosa/soxr**. The realtime
backend resamples with **scipy `resample_poly`**
([audio/resample.py](realtime-backend/audio/resample.py)). Those two do not
have the same rolloff:

| Band | librosa/soxr vs scipy | Verdict |
|---|---|---|
| below 7 kHz | agree to **0.02–0.7 dB** | safe |
| 7.9–8 kHz | differ by **59.5 dB** | librosa brickwalls it, scipy keeps it |

On its own that would be a small domain shift. What made it fatal is that the
corpus's **native sample rates were split by label**: 73.3% of bonafide clips
were already 16 kHz (so never resampled, so full band), against 31.4% of spoof.
Combine the two and "energy present near Nyquist" became a label proxy —
corr(label) = **−0.254** in the 7.9–8 kHz band.

So the model learned: *full band ⇒ real*. In training that scored 2.42% EER. In
service, scipy hands it a full band for **everything**, including spoofs — and
in-training generators read bonafide. This is the single most misleading demo
failure we have had: the model was most confident exactly where it was wrong.

### Augmentation cannot fix this — don't retry it

The manifests point at WAVs librosa **already** wrote at 16 kHz, so the
brickwall is baked into the files on disk. Augmentation can only *add* holes,
never restore a deleted band. Measured over 400 train windows, only 1.6% of
spoof windows are natively full-band vs 13.6% of bonafide, so "full band ⇒
bonafide" survives at any augmentation strength. The first attempt cut
label/top-band correlation from −0.204 to −0.045 by driving full-band spoof from
25% to ~4% — it "fixed" the shortcut by deleting the exact case the live path
produces.

### The fix: a parity band gate

**Delete the untrustworthy band from both classes, all splits, at train AND
inference.** Lowpass at **7000 Hz**, comfortably below where the resamplers
diverge.

The cutoff is stored **inside the checkpoint** (`band_gate_hz`) and applied in
`LFCCLCNNExpert.score()` ([experts/lfcc.py](realtime-backend/experts/lfcc.py)),
so nothing hardcodes a second copy that can drift, and it cannot be applied
twice. A gated model served ungated is biased **spoofward** on every real voice.

### What it bought

`eval_raw_sources.py` is the decisive measurement: it starts from the **raw**
sources at native rate and builds each window twice — librosa (what training
saw) vs scipy (what the backend does). Same clip, same content, only the
resampler differs. 992 clips:

| model | TRAIN path (librosa) | DEPLOY path (scipy) | cost | mean \|logit\| gap | verdict flips |
|---|---|---|---|---|---|
| `hybrid` (ungated) | **2.42%** EER | **27.52%** EER | **+25.10 pts** | 5.90 | **17.5%** |
| `hybrid_br` (gated 7 kHz) | 8.27% | **8.27%** | **+0.00 pts** | 0.02 | **0.0%** |

Same story from the diagnostic angle — mean |scipy − librosa| verdict gap:
`hybrid` **13.35** logits → `hybrid_nc` **5.25** → `hybrid_br` **0.03**.

`hybrid_maxbr` inherits the property by construction (same 7000 Hz gate, read
from its own checkpoint) and is measured directly by
`tests/test_band_gate.py::test_gated_experts_are_far_less_sensitive_than_ungated`,
which pushes an out-of-band tone through every expert and compares the shift:

| model | logit shift from out-of-band energy |
|---|---|
| `hybrid` (ungated) | **23.55** |
| `hybrid_nc` (ungated) | **9.23** |
| `hybrid_br` (gated) | 0.32 |
| `hybrid_maxbr` (gated) | **0.03** |

The verdict no longer depends on which resampler produced the audio. Note that
"gated" means *small*, not zero: brickwall rFFT gating leaves ~1% time-domain
ringing and the log-energy LFCC front-end amplifies it, so the test asserts a
≥10× reduction against the ungated baseline rather than ≈0. An absolute epsilon
would fail on a correctly gated model.

### Still open: the 6–7 kHz imbalance

The 6–7 kHz band has corr(label) = **+0.275** — *stronger* than the bug, and the
opposite sign. But the resamplers **agree** there, so it is a real corpus
imbalance that is consistent train-to-serve. The gate does not fix it and is not
meant to. This is now the largest known shortcut and the likely cause of the
residual false alarms in [§4](#4-known-open-issues).

---

## 2. Calibration: which artifact belongs to which model

A calibrator is a two-parameter Platt map, `p = sigmoid(a·logit + b)`. `a` and
`b` are fitted **for one specific set of weights on one specific corpus**. Use
them on anything else and you get a confident number with no meaning.

### The mapping — this table is the whole point of this section

| Artifact in `realtime-backend/artifacts/` | Belongs to | kind | a | b | Fitted on |
|---|---|---|---|---|---|
| `platt_v4.json` | `wavlm` | sklearn | — | — | unified_manifest dev (Person A) |
| `platt_ssl.json` | `ssl` | sklearn | — | — | unified_manifest dev (Person A) |
| `calibrator_hybrid_clean.json` | `hybrid` | platt | 0.4491 | −1.3645 | hybrid dev, **ungated** |
| `calibrator_hybrid_newclips.json` | `hybrid_nc` | platt | 0.5129 | −0.9220 | hybrid dev, **ungated** |
| `calibrator_hybrid_br.json` | `hybrid_br` | platt | 0.4702 | −0.3206 | hybrid dev, **GATED 7000 Hz** |
| `calibrator_hybrid_maxbr.json` | `hybrid_maxbr` — **the decision expert** | platt | 0.6854 | +0.1428 | hybrid_maxbr dev, **GATED 7000 Hz** |
| `calibrator_heuristic.json` | `FUSION_MODE=heuristic` only | identity | 1.0 | 0.0 | not a model calibrator |
| `calibrator.json` | the **retired** `lfcc` expert | platt | 1.3212 | −0.5200 | ASVspoof19 dev — **do not use** |

`calibrator.json` is the trap in that list: it is the shortest, most
default-looking filename in the directory and it belongs to an expert that no
longer exists. It was fitted on ASVspoof19 and reports `EER 0.0` on its own
held-out split, which is exactly the kind of number that gets pasted into a
slide.

**The two gated calibrators are the newer trap.** `calibrator_hybrid_br.json` and
`calibrator_hybrid_maxbr.json` both record `band_gate_hz: 7000.0`, so guard 2
below — which compares gates — **passes when they are swapped**. Only guard 3
catches it, by checking the `checkpoint` field against `SINGLE_EXPERT`. Their
Platt scales are genuinely different (a 0.4702/b −0.3206 vs a 0.6854/b +0.1428):
at the shipped `p ≥ 0.65` band, `hybrid_br`'s scale puts the boundary at logit
**1.98** and `hybrid_maxbr`'s at logit **0.69**. Reading maxbr's logits through
br's calibrator would under-report every spoof by roughly a full band.

`hybrid_maxbr`'s `a` is larger and its `b` is *positive* — it is a from-scratch
model on a different corpus (132 generators, 74,672 train chunks), so nothing
about its scale is comparable to a warm-start fine-tune's. That is the whole
reason `fit_calibrator_maxbr.py` exists as a separate script instead of a flag on
`fit_calibrator_br.py`, which is hardcoded to hybrid_br's dev split.

### Failure mode A — one expert's logits on another's scale

Concretely, reading `hybrid`'s logits through `hybrid_nc`'s calibrator (a
one-character config mistake) moves the policy boundaries:

| Boundary | With hybrid's own calibrator | With hybrid_nc's |
|---|---|---|
| low / uncertain (p=0.35) | logit **1.66** | logit **0.59** |
| uncertain / high (p=0.65) | logit **4.42** | logit **3.00** |

The two disagree about the band over **15.5%** of the useful logit range, with a
max probability error of 0.154. A logit of 3.5 reads `uncertain` on the right
scale and `high` on the wrong one. Nothing about the output looks broken.

### Failure mode B — gated model, ungated calibrator (or vice versa)

`hybrid_br` and `hybrid_maxbr` are both calibrated on band-gated audio. Serve
either ungated, or serve a gated model with an ungated calibrator, and the Platt
map is being applied to logits that model never produces. Both directions are
silent. This is why each gated calibrator records `band_gate_hz: 7000.0` — so the
mismatch is *detectable* rather than a matter of remembering.

### Failure mode C — the headline read off a foreign scale

[pipeline.py](realtime-backend/pipeline.py) derives the headline probability from
the fused logit. In `single` mode that "fused" logit **is** `SINGLE_EXPERT`'s raw
logit — and in `lr_fusion` mode it is the value used whenever the LR model is
unavailable (no `fusion_lr.joblib`, or `ssl`/`wavlm` not in `EXPERTS` — note
`ssl` needs fairseq, so dropping it is a normal thing to do).

The per-expert side cards always use `EXPERT_CALIBRATORS` and stay correct. That
is what makes this one nasty: **the headline silently disagrees with its own
expert's side card**, and both look plausible.

The committed default (`FUSION_MODE=lr_fusion`, `SINGLE_EXPERT=hybrid`,
`CALIBRATOR_PATH=calibrator_hybrid_newclips.json`) hits exactly this whenever the
LR model is unavailable — which is the two-expert `wavlm,hybrid` config. Fixed by
having the pipeline calibrate a single-expert logit with **that expert's own**
calibrator (`fusion.decision_expert()` names the owner; genuine combinations keep
the global calibrator). Pinned by
`test_lr_fusion_fallback_calibrates_on_the_right_experts_scale`.

### The three startup guards

All in [server.py](realtime-backend/server.py)'s `lifespan`, all raising
`RuntimeError` — a wrong probability must break the boot, not the demo.

| # | Rejects | Implemented in |
|---|---|---|
| 1 | checkpoint declares a band gate, expert applies none | `experts/loader.py::_assert_band_gates_applied` |
| 2 | calibrator's gate ≠ its expert's gate | `calibration.py::assert_calibrator_gates_match` |
| 3 | `FUSION_MODE=single` where `CALIBRATOR_PATH` belongs to another expert | `calibration.py::assert_single_expert_calibrator` |

An expert with no entry in `EXPERT_CALIBRATORS`, or a missing artifact, falls back
to the global calibrator and logs a **warning** — that probability is on the
wrong scale and must not be trusted. Grep the startup log for `falling back`.

---

## 3. Reading the numbers honestly

**Never compare an ungated EER to a gated one.** `hybrid` scores 2.42% dev EER
ungated. The *same weights*, with only the 7–8 kHz band removed, score
**18.97%** — so roughly 87% of its apparent accuracy was riding that band.
`hybrid_br`'s 8.88% gated dev EER looks worse than 2.42% and is in fact far
better; the two measure different things.

The same applies across the two gated models, for a different reason.
`hybrid_maxbr`'s best gated dev EER is **10.20%** and its held-out calibrator EER
**9.09%**, against `hybrid_br`'s 7.84% — yet maxbr is the better model at the
shipped operating point. Its dev set is a *single* generator and a different
corpus, so those two EERs are not comparable either. Compare on eval_ood only.

Also worth knowing before you read `train_hybrid_maxbr.log`: dev EER was
**non-monotonic** across the 8 epochs (19.49 → 16.87 → 16.26 → 15.15 → 12.63 →
15.25 → 17.47 → **10.20**). The best epoch was the last one, after two epochs of
getting worse — patience-based early stopping would have stopped at epoch 5 and
cost 2.4 points.

**Quote the operating point, not the EER.** The backend bands the calibrated
probability at `low < 0.35` / `high ≥ 0.65`
([artifacts/policy.json](realtime-backend/artifacts/policy.json)). The EER
threshold (p = 0.578) is a different decision. `eval_operating_point.py --model
hybrid_maxbr` measures the real chain — gate → model → its own calibrator →
policy. The shipped decision expert against the previous one:

| Set | model | Spoof caught (p≥0.65) | Real false alarm | Abstain (spoof) |
|---|---|---|---|---|
| **eval_ood** | **`hybrid_maxbr`** | **91.5%** | **1.7%** | 3.1% |
| **eval_ood** | `hybrid_br` | 88.1% | 2.9% | 5.1% |
| dev-heldout | `hybrid_maxbr` | 94.6% | 13.2% | 4.1% |
| dev-heldout | `hybrid_br` | 93.0% | 11.2% | 3.7% |

`hybrid_maxbr` is strictly better on eval_ood — **+3.4 points of recall for −1.2
points of false alarm** — which is why it replaced `hybrid_br` as the decision.
The two rows are *not* the same eval set: each model's checkpoint, calibrator, dev
split and eval_ood split move together (`MODELS` in `eval_operating_point.py`),
because hybrid_maxbr trained on a different corpus and scoring it against
hybrid_br's splits would measure part of its own training set. Both eval_ood sets
happen to have 26 spoof generators; maxbr's is 7,163 rows against br's 3,232.

**eval_ood is the headline**; dev-heldout has only 7 bonafide speakers, so its
13.2% is small-sample and speaker-specific, and its single dev generator makes it
useless as a generalization claim.

**"Unseen generators" means 25, not 26 — and never 27.** Of maxbr's 26 eval_ood
spoof generators, `itw-unknown` is *also* a training source, so only **25** are
genuinely held out. Do not round this up in a slide. (The older "27 generators"
figure in earlier drafts of this file was wrong for `hybrid` too.)

**One real regression: `optispeech` recall is 26.9%** (n=193) — the only generator
below 50%, where `hybrid_br` had none. MeloTTS is second-worst at 53.7% (it was
62.9% under br). So maxbr's better aggregate hides one generator it handles worse;
if optispeech-family audio matters for a demo, `hybrid_br` is the safer side card
and is still loaded.

**Do not retune `policy.json`.** The sweep is nearly flat for maxbr too — Youden J
is 0.908 at 0.35 versus **0.899** at the shipped 0.65 — so 0.65 costs under one
point of J and buys a much lower false-alarm rate. That is the right trade for
this product, and the choice is a measurement, not a guess.

---

## 4. Known open issues

| Issue | Status |
|---|---|
| 6–7 kHz corr(label) = **+0.275** corpus imbalance | **Untouched.** Largest known remaining shortcut. |
| `optispeech` recall **26.9%** under `hybrid_maxbr` (was fine under `hybrid_br`) | Open, and a genuine regression. The only eval_ood generator below 50%. |
| `sudhanva.wav` false-alarms at p=0.870; 13.2% dev-heldout FA | Open, pre-existing. The 6–7 kHz imbalance is the prime suspect. The other two local real clips are fine (0.004 `low`, 0.500 `uncertain`). |
| `hybrid_br_best.pth` is **local-only** — not on the Hub | A fresh clone cannot download it. `hybrid_maxbr` does not have this problem (`sarosh22/Final_LFCC`), so the turnkey path is now the decision expert; br is the one that would go missing. |
| Neither gated model is the committed default | Requires a `.env` (see [.env.example](realtime-backend/.env.example) Config C). Changing the default disables `lr_fusion`, so it is a team decision. |
| `wavlm` reads p=0.880 on a real clip | Person A's lane: calibrator/checkpoint mismatch, not a bug in this doc's scope. |
| Guard 2 cannot tell the two gated calibrators apart | Both declare 7000 Hz. Only guard 3 catches a swap, and only in `single` mode. |

---

## 5. Command reference

Resampler parity — the number that must stay near 0:

```bash
cd lfcc-detector && python diag_band_gate.py
```

The decisive train-path vs deploy-path comparison (needs the raw sources on `E:`):

```bash
cd lfcc-detector && python eval_raw_sources.py
```

Where the model lands at the *shipped* policy bands, plus the threshold sweep.
`--model` moves the checkpoint, calibrator, dev split and eval_ood split together;
mixing them silently reports numbers for a chain that is never served:

```bash
cd lfcc-detector && python eval_operating_point.py --model hybrid_maxbr
```

Refit a gated calibrator on gated audio (speaker-disjoint half/half). One script
per model on purpose — each is pinned to its own dev manifest:

```bash
cd lfcc-detector && python fit_calibrator_maxbr.py
```

Regression suite. The gate and calibrator invariants are in
`tests/test_band_gate.py`, which is **parametrized over every gated checkpoint**
in `experts/loader.py::_BAND_GATED_LFCC` — adding a name there gives the new model
the whole guard set instead of trusting it:

```bash
cd realtime-backend && python -m pytest
```

End-to-end ship gate for `hybrid_br` — 17 checks including a negative test that
strips the gate and requires the startup guards to raise. There is **no
`smoke_hybrid_maxbr.py` twin yet**; the parametrized pytest file covers the same
invariants for maxbr:

```bash
cd realtime-backend && python smoke_hybrid_br.py
```

Consolidated evidence for the `hybrid_br` numbers, in one tracked file — the raw
JSONs live under `data_pipeline/manifests/`, which is gitignored:
[lfcc-detector/hybrid_br_evidence.json](lfcc-detector/hybrid_br_evidence.json).
The `hybrid_maxbr` equivalents are `lfcc-detector/eval_op_maxbr.log`,
`lfcc-detector/fit_calibrator_maxbr.log` and
`lfcc-detector/train_hybrid_maxbr.log`.

---

## 6. Serving `hybrid_maxbr` — what the wiring actually is

Nothing about the new model needed new serving code. It follows the same path as
every other LFCC checkpoint:

| Step | Mechanism | Model-specific? |
|---|---|---|
| download | `HUB_EXPERTS["hybrid_maxbr"]` → `sarosh22/Final_LFCC` via `ensure_checkpoint` | one config entry |
| adapt | the same `LFCCLCNNExpert` class | no |
| gate | `band_gate_hz` read from the checkpoint in `_wire_model()` | no — read, not hardcoded |
| calibrate | `EXPERT_CALIBRATORS["hybrid_maxbr"]` | one config entry |
| guard | all three startup guards, unchanged | no |

`experts/loader.py` routes both gated checkpoints through one branch keyed on the
`_BAND_GATED_LFCC` set, so the previous copy-pasted per-model branch is gone.

The fresh-clone path was verified empirically: deleting the cached checkpoint and
re-loading downloads from the Hub to a byte-identical file (13,753,026 bytes,
sha256 `d8877e2156c2ec85…`, `band_gate_hz=7000.0`).

To serve it, `realtime-backend/.env`:

```
EXPERTS=wavlm,hybrid,hybrid_nc,hybrid_br,hybrid_maxbr
FUSION_MODE=single
SINGLE_EXPERT=hybrid_maxbr
CALIBRATOR_PATH=<abs path>/artifacts/calibrator_hybrid_maxbr.json
```

`CALIBRATOR_PATH` must be the **decision** expert's own artifact — in `single`
mode `pipeline.py` applies the global calibrator to `SINGLE_EXPERT`'s raw logit,
so a mismatch reads the band off another model's Platt scale. Guard 3 enforces it
at startup.

### A trap in the WebSocket path, if you write a test client

`apply_start` defaults `encoding` to **`pcm_s16le`**. Send float32 frames without
declaring `encoding: "pcm_f32le"` and the server decodes them as int16 — which is
garbage, and garbage reads **spoofward**: every window on both a real and a spoof
clip came back `high` at p≈0.95–0.97, looking exactly like a model regression.
With the encoding declared, `/ws` matches `/predict-file` (real clip p=0.0045
`low`, spoof p=0.94 `high`), and int16 vs float32 differ by <0.001. The streaming
resampler itself is exact: `StreamingPolyphaseResampler` fed 4800-sample frames
reproduces `to_target_rate`'s one-shot output logit-for-logit.


