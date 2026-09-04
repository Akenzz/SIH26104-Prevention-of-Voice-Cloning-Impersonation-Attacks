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
`hybrid` **13.35** logits → `hybrid_nc` **5.25** → `hybrid_br` **0.03**. The
verdict no longer depends on which resampler produced the audio.

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
| `calibrator_heuristic.json` | `FUSION_MODE=heuristic` only | identity | 1.0 | 0.0 | not a model calibrator |
| `calibrator.json` | the **retired** `lfcc` expert | platt | 1.3212 | −0.5200 | ASVspoof19 dev — **do not use** |

`calibrator.json` is the trap in that list: it is the shortest, most
default-looking filename in the directory and it belongs to an expert that no
longer exists. It was fitted on ASVspoof19 and reports `EER 0.0` on its own
held-out split, which is exactly the kind of number that gets pasted into a
slide.

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

`hybrid_br` is calibrated on band-gated audio. Serve it ungated, or serve a gated
model with an ungated calibrator, and the Platt map is being applied to logits
that model never produces. Both directions are silent. This is why
`calibrator_hybrid_br.json` records `band_gate_hz: 7000.0` — so the mismatch is
*detectable* rather than a matter of remembering.

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

**Quote the operating point, not the EER.** The backend bands the calibrated
probability at `low < 0.35` / `high ≥ 0.65`
([artifacts/policy.json](realtime-backend/artifacts/policy.json)). The EER
threshold (p = 0.578) is a different decision. `eval_operating_point.py` measures
the real chain — gate → model → calibrator → policy:

| Set | Spoof caught (p≥0.65) | Real false alarm | Abstain |
|---|---|---|---|
| **eval_ood** — 27 generators unseen by both training and the calibrator fit | **88.1%** | **2.9%** | 5.1% |
| dev-heldout — speaker-disjoint half the calibrator did not fit | 93.0% | 11.2% | — |

**eval_ood is the headline**; dev-heldout has only 7 bonafide speakers, so its
11.2% is small-sample and speaker-specific. Per-generator, **0 of 26 fall below
50% recall** (worst: Qwen2.5-Omni 51.2%, MeloTTS 62.9%).

**Do not retune `policy.json`.** The sweep is nearly flat — Youden J is 0.887 at
0.35 versus 0.852 at the shipped 0.65 — so 0.65 buys a much lower false-alarm
rate for about 5 points of recall. That is the right trade for this product, and
the choice is a measurement, not a guess.

---

## 4. Known open issues

| Issue | Status |
|---|---|
| 6–7 kHz corr(label) = **+0.275** corpus imbalance | **Untouched.** Largest known remaining shortcut. |
| 2 of 3 local real clips read as spoof (0.760, 0.706); 11.2% dev-heldout FA | Open. The 6–7 kHz imbalance is the prime suspect. |
| `hybrid_br_best.pth` is **local-only** — not on the Hub | A fresh clone cannot download it. Publish to `sarosh22/Expert2` to make the install turnkey. |
| `hybrid_br` is not the default | Requires a `.env` (see [.env.example](realtime-backend/.env.example) Config B). Changing the default disables `lr_fusion`, so it is a team decision. |
| `wavlm` reads p=0.880 on a real clip | Person A's lane: calibrator/checkpoint mismatch, not a bug in this doc's scope. |

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

Where the model lands at the *shipped* policy bands, plus the threshold sweep:

```bash
cd lfcc-detector && python eval_operating_point.py
```

Refit `hybrid_br`'s calibrator on gated audio (speaker-disjoint half/half):

```bash
cd lfcc-detector && python fit_calibrator_br.py
```

End-to-end ship gate — 17 checks including a negative test that strips the gate
and requires the startup guards to raise:

```bash
cd realtime-backend && python smoke_hybrid_br.py
```

Regression suite (the gate and calibrator invariants are pinned in
`tests/test_band_gate.py` and `tests/test_fusion_calibration.py`):

```bash
cd realtime-backend && python -m pytest
```

Consolidated evidence for every number above, in one tracked file — the raw
JSONs live under `data_pipeline/manifests/`, which is gitignored:
[lfcc-detector/hybrid_br_evidence.json](lfcc-detector/hybrid_br_evidence.json).


