# Fix 2 — prosody expert: result and recommendation

Date: 2026-09-02 · branch `task-d` · all numbers reproducible from
`prosody-detector/artifacts/manip_experiment.log` and `.json`

---

## Short version

Fix 2 asked for a prosody expert to cover the blind spot the two spectral experts
have on **non-neural manipulation** (cheap voice changers: pitch/formant shift and
crude resynthesis applied to a real recording). Two things came out of building
and testing it:

1. **The shipped prosody artifact does not cover that blind spot.** It is near
   chance on non-neural attacks (mean AUC 0.570). Its only real signal is on
   classic parametric vocoders, and even there it is unstable between two systems
   using the *same* vocoder.
2. **Retraining it on real manipulation data works, but only for the resynthesis
   engine it trained on.** On unseen speakers with *seen* transforms it goes
   49.90% → **30.83% EER** (AUC 0.502 → 0.767). On the three withheld transform
   families it is **43.80% EER** — i.e. no better than the model that never saw a
   manipulation (43.07%).

**Recommendation: do not ship prosody as a scoring expert, and do not add it to
the fusion.** Keep the manipulation corpus — it is the real asset, and it belongs
to the LFCC/hybrid retrain (Fix 1), not to a 13-feature linear model.

Prosody is currently *not* wired into the backend (`realtime-backend/experts/loader.py`
knows only `dummy`, `wavlm`, `hybrid`), so there is no live exposure to undo.

---

## 1. What we measured first, and why the original hypothesis failed

`prosody-detector/evaluation/diagnose_by_generator.py` scores every spoof
generator individually against the bonafide pool of its own group, then aggregates
by how the waveform was actually produced. ASVspoof A01–A19 assignments are the
attack descriptions from the ASVspoof 2019 database paper (arXiv:1911.01601),
not something inferred from the audio.

| family | generators | clips | mean AUC | mean EER |
|---|---|---|---|---|
| non_neural_waveform | 9 | 1,077 | 0.570 | 44.34% |
| dsp_vocoder | 7 | 912 | **0.729** | **31.65%** |
| neural_waveform | 21 | 2,311 | 0.598 | 42.67% |
| unknown | 4 | 450 | 0.590 | 42.87% |

The family Fix 2 was written to cover is the one prosody is *worst* on. And the
family it is best on is not internally consistent: **A02 (WORLD) 17.57% EER vs
A03 (WORLD) 51.90%** — same vocoder, same corpus. A04 (unit selection) is 56.43%
and A18 (parametric VC) 64.63% / AUC 0.298, i.e. actively inverted.

That pattern says the signal is *system-specific prosody habits* (rate and pause
were the top-separating features for the DSP family — plausibly TTS-script
confounds), not a physical resynthesis artifact. So the honest read was: the model
had never been trained on the target attack class. That is what the retrain tests.

---

## 2. The missing data, built and verified

No public corpus contains voice-changer manipulations. It does not need one — a
voice changer is a deterministic DSP chain, so it can be reproduced from bonafide
audio we already have (`data_pipeline/generate_manipulations.py`, Praat via
parselmouth + librosa, offline).

**7,447 pairs / 14,894 rows.** Every manipulated clip is emitted alongside its own
source as bonafide, so both classes contain the same speakers, channels and
sentences — the only difference is the transform. Six Praat `Change gender`
transforms in train/dev; **three transform families withheld from train/dev and
present only in eval**, which is what produced finding 2 above.

Verified, not assumed:

- Every transform measurably does what it claims
  (`data_pipeline/verify_manipulations.py`, 225 pairs: pitch family moves F0
  ×0.77/×1.26, formant family moves spectral centre of gravity ×0.87/×1.11,
  monotone crushes F0 std to ×0.267, and no clip is identical to its source).
- Class balance exactly 1:1 in every group; duration means matched to 1e-6.
- **0** speakers span splits, **0** source stems span splits (of 7,447),
  **0** withheld transforms outside eval, **0** duplicate paths.

---

## 3. Retrain result

Three variants, identical cached features, identical slices. `shipped` =
`artifacts/prosody_lr_v1.json` unchanged; `manip_only` = fitted on the
manipulation train split; `combined` = manipulations + a per-generator-capped
neural sample.

| slice | n (bona/spoof) | shipped | manip_only | combined |
|---|---|---|---|---|
| manip_dev — seen transforms, seen speakers | 495/495 | 51.31% / 0.491 | **26.67% / 0.819** | 30.51% / 0.745 |
| manip_eval_seen — seen transforms, **unseen speakers** | 480/480 | 49.90% / 0.502 | **30.83% / 0.767** | 36.04% / 0.689 |
| **manip_eval_heldout — unseen transforms + unseen speakers** | 476/476 | 43.07% / 0.636 | **43.80% / 0.572** | 51.26% / 0.460 |
| neural_dev | 495/150 | 49.62% / 0.524 | 51.32% / 0.470 | 41.17% / 0.626 |
| neural_eval | 500/450 | 66.11% / **0.288** | 48.52% / 0.490 | 36.84% / 0.678 |
| neural_eval_ood | 500/1837 | 61.64% / 0.369 | 45.61% / 0.554 | 33.18% / 0.722 |
| asv_dev | 500/900 | 40.06% / 0.641 | 43.69% / 0.597 | 51.00% / 0.509 |
| ood_en_asv | 500/1500 | 36.73% / 0.661 | 47.40% / 0.542 | 52.03% / 0.484 |

`manip_eval_heldout` is the generalization number. Quote that row, not `manip_dev`.

Two secondary results in this table are worth flagging to the team:

- **The shipped prosody artifact is inverted on the operational corpus**
  (neural_eval AUC 0.288, neural_eval_ood 0.369). It was fitted on
  `multicorpus_final`, where it still reads 0.641/0.661. Its "signal" does not
  transfer to `hybrid_vad_chunks` — it flips sign. RESULTS.md's pooled 43.94%
  averaged this away. If prosody were ever added to the fusion with a positive
  weight, it would actively hurt.
- **`combined` is the only variant above chance on neural attacks**
  (neural_eval_ood 33.18% / 0.722) — but it pays for it by collapsing to chance on
  ASVspoof (0.509 / 0.484) *and* on unseen manipulations (0.460).

---

## 4. What the retrained model actually learned

Per-transform on the withheld eval slice:

| withheld transform | resynthesis maths | manip_only EER |
|---|---|---|
| `manip_praat_time_stretch` | Praat PSOLA — **same engine as training** | **26.6%** |
| `manip_pv_pitch_shift` | librosa phase vocoder | 58.0% |
| `manip_resample_speed` | scipy polyphase resample | 47.0% |

Transfer happens *within* a resynthesis engine and not *across* one. What the
model learned is the PSOLA fingerprint, not "manipulation" as a class. The
withheld-family design is what exposed this; a conventional random split would
have reported 30.83% and called it generalization.

(Note the mirror image: `shipped` catches `pv_pitch_shift` at **13.0%** — a phase
vocoder smears phase much like the neural vocoders it trained on. The two models
are complementary across resynthesis types, which is consistent with the same
mechanism.)

### Why one artifact cannot cover both attack classes

This is mechanical, not a tuning problem. `jitter_local` is the dominant weight of
the retrained model at **−1.14** (lower jitter ⇒ spoof), because PSOLA re-imposes
regular periodicity. But the diagnostic's Cohen's d shows non-neural ASVspoof
attacks have jitter **+0.33** (*higher* than human), while the DSP-vocoder family
has **−0.46**. A single linear weight cannot carry both signs. `combined`'s
collapse on ASVspoof is exactly that contradiction resolving to zero.

| variant | top weights (standardized, + drives toward spoof) |
|---|---|
| shipped | f0_std −0.37, voiced_fraction +0.25, f0_range +0.25, pause_mean +0.19 |
| manip_only | **jitter_local −1.14**, voiced_fraction −0.63, f0_std −0.21, spectral_flatness −0.20 |
| combined | jitter_local −0.73, voiced_fraction −0.69, shimmer_local −0.37, pause_var +0.31 |

---

## 5. Deployment reality: one threshold, applied everywhere

Per-slice EER moves its own threshold per slice, which hides score offsets between
corpora. Deployment gets one threshold. Set it on `manip_dev` bonafide, then leave
it alone:

| variant | FAR target | manipulations detected (seen / **unseen** family) | real voices wrongly flagged (asv_dev / ood_en_asv) |
|---|---|---|---|
| shipped | 5% | 20.6% / **37.2%** | 1.0% / 0.2% |
| manip_only | 5% | 33.3% / **17.4%** | 9.4% / 10.8% |
| manip_only | 10% | 46.2% / **24.6%** | 15.2% / 18.4% |
| combined | 5% | 18.5% / **9.7%** | 19.6% / 19.2% |

At any usable false-alarm rate, no variant catches novel manipulations often
enough to gate on. The best case is `manip_only` at 10% FAR: fewer than 1 in 4
unseen-family manipulations caught, while flagging roughly 1 in 6 genuine human
recordings from another corpus. That is not a shippable detector.

---

## 6. One confound checked and ruled out

`verify_manipulations.py` found a real level offset in the corpus: peak matching
leaves the manipulated side **0.5–1.8 dB quieter in RMS** (median, per transform),
because PSOLA raises crest factor. If the prosody features could see loudness,
every number above would be worthless.

They cannot. `prosody-detector/evaluation/check_gain_invariance.py` rescales the
same waveform by ±1, ±1.78 and ±6 dB and re-extracts: worst-case drift across all
13 features is **8×10⁻⁷ relative** — floating-point noise. Every feature is a
frequency, a duration, a ratio, or gated on the clip's own loudest frame.

**This does matter for Fix 1.** An LFCC/log-magnitude model reads a gain change as
a constant offset in its input — a free label bit. `generate_manipulations.py` now
defaults to `--level-match rms` (verified: exact RMS parity, with a clipping
guard; `--level-match peak` reproduces the old corpus). **The corpus must be
regenerated before the hybrid retrain.**

---

## 7. Recommendation

1. **Do not ship prosody as a scoring expert; do not add it to `fusion.json`.**
   It is at chance on unseen manipulations, and the currently shipped artifact is
   inverted on the operational corpus.
2. **Keep the manipulation corpus and point it at Fix 1.** Non-neural coverage
   should come from the hybrid LFCC model, which has the capacity to learn a
   resynthesis fingerprint. 13 hand features and a linear model do not.
   Regenerate RMS-matched first.
3. **Keep the withheld-transform-family split as standard practice.** It is the
   only reason we know the 30.83% was memorization.
4. If prosody features are wanted in the UI, show them as measured statistics
   displayed next to the decision — never as a score, a verdict, or a
   contribution to one.

### Still open

- Regenerate the corpus RMS-matched, then retrain hybrid on it (deferred to
  overnight). Two decisions pending: separate `hybrid_manip.pth` vs overwriting
  `hybrid_clean.pth`; whether to keep the double-weighted paired-bonafide rows.
- `RESULTS.md` still benchmarks prosody against the retired `mc_v3` and quotes the
  pooled 43.94% that hides the sign flip in §3. Needs updating.
- `fusion.json` has no `hybrid` weight, so `FUSION_MODE=fused` remains unsafe.
