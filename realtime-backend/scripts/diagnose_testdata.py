#!/usr/bin/env python3
"""Quick diagnostic: score all testdata files with each expert separately."""
import sys, os
# Run from realtime-backend/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

# Suppress noisy loggers
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("realtime_backend").setLevel(logging.WARNING)
logging.getLogger("realtime_backend.experts").setLevel(logging.WARNING)
logging.getLogger("realtime_backend.calibration").setLevel(logging.WARNING)

from audio.resample import to_mono, to_target_rate
from config import Settings, TARGET_SAMPLE_RATE, EXPERT_CALIBRATORS
from experts.loader import load_experts
from calibration import load_expert_calibrators, load_calibrator

WINDOW_SAMPLES = int(4.0 * TARGET_SAMPLE_RATE)  # 4s @ 16kHz

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def score_file(path, experts, calibrators):
    try:
        audio, sr = sf.read(path, always_2d=False)
    except Exception as e:
        return None, str(e)
    audio = to_mono(audio, audio.ndim)
    result = to_target_rate(audio, sr, TARGET_SAMPLE_RATE)
    audio = result[0] if isinstance(result, tuple) else result
    audio = audio.astype(np.float32)

    # Take up to 3 windows
    results = {}
    n_windows = max(1, min(3, len(audio) // WINDOW_SAMPLES))
    logits_per_expert = {k: [] for k in experts}

    for i in range(n_windows):
        window = audio[i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES]
        if len(window) < WINDOW_SAMPLES:
            window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))
        for name, expert in experts.items():
            score = expert.score(window)
            logits_per_expert[name].append(float(score["logit"]))

    for name, logits in logits_per_expert.items():
        mean_logit = np.mean(logits)
        cal = calibrators.get(name)
        prob = float(cal.probability(mean_logit)) if cal else sigmoid(mean_logit)
        results[name] = {"logit": round(mean_logit, 3), "prob": round(prob * 100, 1)}
    return results, None

def main():
    print("Loading experts...")
    settings = Settings(experts=["wavlm", "hybrid", "ssl"], fusion_mode="heuristic_avg")
    experts = load_experts(settings)
    # load calibrators
    global_cal = load_calibrator(settings.calibrator_path)
    calibrators = load_expert_calibrators(EXPERT_CALIBRATORS, settings.experts, global_cal)
    print("Experts loaded.\n")

    testdata = sorted(glob.glob("/home/akenzz/sih/project/testdata/*"))
    bonafide = [f for f in testdata if "bonafied" in os.path.basename(f).lower()]
    spoof    = [f for f in testdata if "bonafied" not in os.path.basename(f).lower()]

    def run_group(files, label):
        print(f"{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        print(f"  {'File':<38} {'WavLM%':>7} {'LFCC%':>7} {'SSL%':>7}  {'Weighted%':>10}")
        print(f"  {'-'*75}")
        for f in files:
            res, err = score_file(f, experts, calibrators)
            if err:
                print(f"  {os.path.basename(f):<38}  ERROR: {err}")
                continue
            pw = res.get("wavlm", {}).get("prob", 0)
            pl = res.get("hybrid", {}).get("prob", 0)
            ps = res.get("ssl", {}).get("prob", 0)
            weighted = round(pw * 0.60 + pl * 0.30 + ps * 0.10, 1)
            flag = "  ← WRONG" if (label.startswith("BONAFIDE") and weighted > 50) else (
                   "  ← missed" if (label.startswith("SPOOF") and weighted < 50) else "")
            print(f"  {os.path.basename(f):<38} {pw:>6.1f}% {pl:>6.1f}% {ps:>6.1f}%  {weighted:>8.1f}%{flag}")
        print()

    run_group(bonafide, "BONAFIDE (should all be < 50%)")
    run_group(spoof,    "SPOOF (should all be > 50%)")

if __name__ == "__main__":
    main()
