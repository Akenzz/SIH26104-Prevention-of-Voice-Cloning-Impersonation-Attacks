# Prosody Expert — Results

Trained on `data_pipeline/manifests/multicorpus_final.csv` (train split: 15,797
windows, 7,900 bonafide / 7,897 spoof). Model: StandardScaler + class-balanced
LogisticRegression over 13 interpretable prosody features. Evaluated on the
**exact same held-out slices** as WavLM/LFCC/mc_v3, reported identically.

## EER / AUC — prosody vs the decision expert (mc_v3)

| Slice | Prosody EER | Prosody AUC | mc_v3 EER | mc_v3 AUC |
|---|---|---|---|---|
| **in-domain** hi_xtts | 50.00% | 0.508 | 4.58% | 0.992 |
| ood_en_asv (unseen attacks) | 36.40% | 0.659 | 24.53% | 0.835 |
| ood_hi_indicf5 (unseen Hindi gen) | 42.66% | 0.597 | 26.08% | 0.820 |
| ood_en_mlaad (15 held-out EN gens) | 44.25% | 0.583 | 15.81% | 0.914 |
| ood_de_mlaad (4 held-out DE gens) | 45.13% | 0.559 | 34.25% | 0.716 |
| ood_itw (In-the-Wild) | 50.13% | 0.497 | 34.39% | 0.719 |
| **POOLED** eval_ood | 43.94% | 0.575 | 27.18% | 0.800 |

Full JSON: `artifacts/prosody_eval.json` (adds min-tDCF, accuracy, P/R/F1 per slice).

## Honest read

Prosody-only is a **weak standalone detector** on this diverse, multilingual,
many-generator set — near chance in-domain and on In-the-Wild, modestly
better-than-chance on ASVspoof unseen attacks (AUC 0.66) and unseen Hindi
(AUC 0.60). This is expected and matches the literature: modern TTS reproduces
gross prosody well. **It is not a decision-maker — keep `mc_v3` as the decision
expert.** Its value is the *named, grounded evidence* it exposes to the Task-F
explainer, plus a small complementary signal for fusion.

Feature weights (|coef|, spoof-positive): `f0_std` (−0.37), `voiced_fraction`
(+0.25), `f0_range` (+0.25), `pause_mean` (+0.19), `shimmer_local` (+0.16). The
`jitter_source` weight (−0.18) is a mild corpus artifact (whether Praat
succeeded); harmless for the explainer but worth pruning if prosody is ever
promoted into fusion weighting.

## Calibrator

`artifacts/calibrator_prosody.json` — Platt fit on the dev split (balanced,
2,988 windows). Held-out ECE 0.092 → 0.044; EER unchanged (monotonic). Use with
`CALIBRATOR_PATH=../prosody-detector/artifacts/calibrator_prosody.json` only if
prosody is the single decision expert (not recommended).

## Describe sanity check (real eval clips)

Findings fire **only** when a value is outside the bonafide p5/p95 human band —
never a fixed list.

- `[bonafide]` clean Hindi clip → *(no anomalies)*
- `[bonafide]` clip with flat delivery → "speaking rate is unusually constant" [rate_var=0.0 vs human p5=0.19]
- `[spoof]` coqui-xtts-v2 → *(no anomalies — this generator's prosody looks human, consistent with the 50% in-domain EER)*
- `[spoof]` coqui-xtts-v2 → "pitch transitions are unnaturally smooth" [f0_delta_var=5.21 vs human p5=10.01]

Reproduce: `python evaluation/describe_examples.py --n 3`.

## Verification status
- Backend unit tests: `realtime-backend/tests/test_prosody_expert.py` — 3/3 pass.
- Integration: `EXPERTS=lfcc,mc_v3,prosody` load + score together — OK (prosody
  returns 13-dim interpretable embedding).
