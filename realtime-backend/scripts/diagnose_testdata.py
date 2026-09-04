"""
diagnose_testdata.py — Offline diagnostic: score every file in testdata/.
Prints per-file mean logits from all 3 experts and LR fusion probability.

Run from repo root:
    cd /home/akenzz/sih/project
    cd realtime-backend && ../.venv/bin/python scripts/diagnose_testdata.py
"""
import sys, os

# Make realtime-backend the working directory and ensure its imports work
_SCRIPT   = os.path.abspath(__file__)
_SCRIPTS  = os.path.dirname(_SCRIPT)
_BACKEND  = os.path.dirname(_SCRIPTS)           # realtime-backend/
_REPO     = os.path.dirname(_BACKEND)           # repo root

os.chdir(_BACKEND)
for p in [_BACKEND, _REPO]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import joblib
import librosa
import torch
from pathlib import Path

TESTDATA       = Path(_REPO) / "testdata"
WINDOW_SEC     = 4.0
HOP_SEC        = 0.5
SR             = 16000
WINDOW_SAMPLES = int(WINDOW_SEC * SR)
HOP_SAMPLES    = int(HOP_SEC * SR)

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models (this takes ~90s for SSL)…\n")

from config import load_settings, HUB_EXPERTS
settings    = load_settings()
cache_dir   = settings.model_cache_dir

from experts.loader import load_experts
all_experts   = load_experts(settings)   # loads wavlm + hybrid + ssl
wavlm_expert  = all_experts["wavlm"]
hybrid_expert = all_experts["hybrid"]
ssl_expert    = all_experts["ssl"]

lr_model = joblib.load(Path(_BACKEND) / "artifacts" / "fusion_lr.joblib")

print(f"LR coefficients: wavlm={lr_model.coef_[0][0]:.3f}  hybrid={lr_model.coef_[0][1]:.3f}  ssl={lr_model.coef_[0][2]:.3f}")
print(f"LR intercept   : {lr_model.intercept_[0]:.3f}")
print(f"  (spoof if: {lr_model.coef_[0][0]:.3f}*wavlm + {lr_model.coef_[0][1]:.3f}*hybrid + {lr_model.coef_[0][2]:.3f}*ssl > {-lr_model.intercept_[0]:.3f})\n")

# ── Score each file ───────────────────────────────────────────────────────────
SPOOF_KEYWORDS = {"spoof", "generated", "chatterbox", "firered", "omni",
                  "styletts", "sopro", "spk_17", "qwen"}

files = sorted(TESTDATA.glob("*.*"))
files = [f for f in files if f.suffix.lower() in {".wav", ".mp3", ".flac"}]

hdr = f"{'FILE':<45} {'TRUTH':>8} | {'wavlm_μ':>8} {'hybrid_μ':>8} {'ssl_μ':>8} | {'LR_prob':>8} {'VERDICT':>10} {'OK?':>6}"
print(hdr)
print("─" * len(hdr))

n_correct = 0
for fpath in files:
    name  = fpath.name
    truth = "spoof" if any(k in name.lower() for k in SPOOF_KEYWORDS) else "bonafide"

    try:
        audio, _ = librosa.load(str(fpath), sr=SR, mono=True)
    except Exception as e:
        print(f"  ERROR loading {name}: {e}")
        continue

    w_logs, h_logs, s_logs, lr_ps = [], [], [], []

    for start in range(0, max(1, len(audio) - WINDOW_SAMPLES + 1), HOP_SAMPLES):
        w = audio[start: start + WINDOW_SAMPLES]
        if len(w) < WINDOW_SAMPLES:
            w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))

        wl = float(wavlm_expert.score(w)["logit"])
        hl = float(hybrid_expert.score(w)["logit"])
        sl = float(ssl_expert.score(w)["logit"])
        lp = float(lr_model.predict_proba([[wl, hl, sl]])[0, 1])

        w_logs.append(wl); h_logs.append(hl)
        s_logs.append(sl); lr_ps.append(lp)

    if not lr_ps:
        print(f"  {name:<43} — too short to score")
        continue

    avg_w  = np.mean(w_logs)
    avg_h  = np.mean(h_logs)
    avg_s  = np.mean(s_logs)
    avg_lr = np.mean(lr_ps)
    verdict = "SPOOF" if avg_lr >= 0.5 else "bonafide"
    correct = (verdict == "SPOOF") == (truth == "spoof")
    n_correct += int(correct)

    print(f"  {name:<43} {truth:>8} | {avg_w:>8.3f} {avg_h:>8.3f} {avg_s:>8.3f} | {avg_lr:>8.1%} {verdict:>10} {'✓' if correct else '✗ WRONG':>6}")

print(f"\nAccuracy: {n_correct}/{len(files)} = {n_correct/len(files):.0%}")
