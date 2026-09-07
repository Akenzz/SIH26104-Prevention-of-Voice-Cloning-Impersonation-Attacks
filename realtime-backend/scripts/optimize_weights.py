#!/usr/bin/env python3
"""
Find the optimal (w_wavlm, w_lfcc, w_ssl) weights and decision threshold
for the current testdata using brute-force grid search.
"""
import numpy as np

# Per-expert calibrated probabilities from the diagnostic
# Format: (filename, true_label, wavlm%, lfcc%, ssl%)
DATA = [
    # BONAFIDE (label = 0)
    ("bonafied1.mp3",                  0, 97.5,  1.8, 26.5),
    ("bonafied2.wav",                  0,  0.1,  0.6, 65.4),
    ("bonafied3.wav",                  0, 26.9,  0.7, 54.6),
    ("bonafied4.wav",                  0,  0.4, 91.8, 41.1),
    ("bonafied5.wav",                  0, 74.7, 80.9, 44.8),  # suspicious but user says bonafide
    ("bonafied6.wav",                  0,  5.1,  0.5, 29.5),
    # SPOOF (label = 1)
    ("SAROSH_SOPRO_V2_SPOOF.wav",      1, 67.4, 26.9, 75.8),
    ("generated_1788276145.wav",       1, 99.9, 99.3, 11.9),
    ("me_spoofed_qwen.wav",            1,  9.1, 75.2, 51.0),
    ("my_spoof_eng.wav",               1, 21.5, 17.1, 26.6),
    ("sarosh_chatterbox_spoof.wav",    1,  2.5, 99.9, 52.9),
    ("sarosh_fireredtts.wav",          1,  1.3, 96.9, 26.5),
    ("sarosh_omni.wav",                1,  2.1, 90.7, 52.8),
    ("sarosh_spoof.wav",               1,  2.0, 99.1, 55.7),
    ("sarosh_styletts.wav",            1, 99.9, 59.5, 50.2),
    ("spk_1788276914.wav",             1, 57.9,  2.6, 74.3),
    ("spoof.wav",                      1, 18.1,  2.3, 14.0),
    ("sudhu_eng_spoof.wav",            1, 99.7, 88.8, 31.5),
    ("temp_output.mp3",                1,  5.8,  0.9, 74.8),
    ("tmp7y7ms46r.wav",                1,  0.6,  0.9, 80.5),
]

labels  = np.array([d[1]   for d in DATA])
wavlm   = np.array([d[2]/100 for d in DATA])
lfcc    = np.array([d[3]/100 for d in DATA])
ssl     = np.array([d[4]/100 for d in DATA])
names   = [d[0] for d in DATA]

# --- Grid search over weights + threshold ---
best_acc   = -1
best_cfg   = None
results    = []

step = 0.05
ws = np.arange(0, 1.01, step)
for w_w in ws:
    for w_l in ws:
        w_s = round(1.0 - w_w - w_l, 6)
        if w_s < 0 or w_s > 1.001:
            continue
        if abs(w_w + w_l + w_s - 1.0) > 0.01:
            continue
        fused = w_w * wavlm + w_l * lfcc + w_s * ssl
        for threshold in np.arange(0.30, 0.71, 0.05):
            preds = (fused >= threshold).astype(int)
            acc   = np.mean(preds == labels)
            correct = np.sum(preds == labels)
            fa    = np.sum((preds == 1) & (labels == 0))  # false alarms (bonafide → spoof)
            miss  = np.sum((preds == 0) & (labels == 1))  # misses (spoof → bonafide)
            results.append((acc, correct, fa, miss, w_w, w_l, w_s, threshold))

results.sort(key=lambda x: (-x[0], x[2], x[3]))  # best acc, then fewest false alarms

print("=" * 80)
print("TOP 10 WEIGHT CONFIGURATIONS")
print("=" * 80)
print(f"  {'WavLM':>6} {'LFCC':>6} {'SSL':>5}  {'Thr':>5}  {'Correct':>8} {'Acc':>5}  {'FalseAlarm':>10} {'Missed':>7}")
print("-" * 80)
for cfg in results[:10]:
    acc, correct, fa, miss, w_w, w_l, w_s, thr = cfg
    print(f"  {w_w:>5.0%}  {w_l:>5.0%} {w_s:>5.0%}  {thr:>4.0%}  "
          f"{correct:>5}/{len(DATA)}  {acc:>4.0%}  {fa:>6} bonafide→spoof  {miss:>3} spoof→bonafide")

# Show detailed breakdown for the best config
print("\n" + "=" * 80)
best = results[0]
acc, correct, fa, miss, w_w, w_l, w_s, thr = best
print(f"BEST: WavLM={w_w:.0%}  LFCC={w_l:.0%}  SSL={w_s:.0%}  threshold={thr:.0%}")
print(f"      Accuracy={acc:.0%}  ({correct}/{len(DATA)} correct,  {fa} false alarms, {miss} missed spoofs)")
print("=" * 80)

fused = w_w * wavlm + w_l * lfcc + w_s * ssl
preds = (fused >= thr).astype(int)
print(f"\n  {'File':<45} {'WavLM%':>7} {'LFCC%':>6} {'SSL%':>6} {'Fused%':>7}  Result")
print("  " + "-" * 85)
for i, name in enumerate(names):
    p   = fused[i] * 100
    ok  = "✅ correct" if preds[i] == labels[i] else ("❌ FP (bona→spoof)" if preds[i] == 1 and labels[i] == 0 else "❌ FN (spoof→bona)")
    lbl = "BONAFIDE" if labels[i] == 0 else "SPOOF"
    print(f"  {name:<45} {wavlm[i]*100:>6.1f}% {lfcc[i]*100:>5.1f}% {ssl[i]*100:>5.1f}% {p:>6.1f}%  {ok}")
