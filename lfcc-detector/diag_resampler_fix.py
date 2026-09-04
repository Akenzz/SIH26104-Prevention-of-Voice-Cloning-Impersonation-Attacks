"""Does matching the TRAINING resampler at inference actually help?

diag_preproc_ablation.py proved the backend's scipy `resample_poly` erases the
band (~7.5-8 kHz) the LFCC models key on, because the training corpus was
resampled with librosa/soxr, which brickwalls 7.9-8 kHz (~-61 dB) while scipy
leaves it only ~3 dB down.

This scores the real local test clips BOTH ways so the fix can be judged on the
thing that matters: spoof recall AND bonafide false-alarm rate.
"""

from __future__ import annotations

import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "realtime-backend"))

TARGET_SR = 16000
WINDOW = 4 * TARGET_SR
MAX_SEC = 30

CLIPS = [
    ("sarosh_real_3sept.wav", "REAL"),
    ("sarosh_ZsVicqBh_original.wav", "REAL"),
    ("sudhanva.wav", "REAL"),
    ("sarosh_fireredtts.wav", "spoof"),
    ("sarosh_omni.wav", "spoof"),
    ("sarosh_chatterbox_spoof.wav", "spoof"),
    ("sarosh_styletts.wav", "spoof"),
    ("sarosh_spoof.wav", "spoof"),
    ("SAROSH_SOPRO_V2_SPOOF.wav", "spoof"),
]
ROOT = Path(r"E:\DatasetSIH\Local_test")


def wins(a: np.ndarray) -> list[np.ndarray]:
    if len(a) < WINDOW:
        return [np.pad(a, (0, WINDOW - len(a)))]
    return [a[i : i + WINDOW] for i in range(0, len(a) - WINDOW + 1, WINDOW // 2)]


def stats(expert, a: np.ndarray, cal) -> tuple[float, float, float]:
    ls = [expert.score(w)["logit"] for w in wins(a)]
    return float(np.mean(ls)), float(np.max(ls)), float(cal.probability(float(np.mean(ls))))


def two_ways(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    import librosa
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x[: sr * MAX_SEC]
    if sr == TARGET_SR:
        return x, x, sr
    g = gcd(int(sr), TARGET_SR)
    a = resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float32)
    b = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
    return a, b, sr


def main() -> None:
    from calibration import load_calibrator
    from config import EXPERT_CALIBRATORS, MODEL_CACHE_DIR
    from experts.lfcc import LFCCLCNNExpert

    print("loading experts ...", flush=True)
    models = {
        "hybrid": (
            LFCCLCNNExpert(cache_dir=MODEL_CACHE_DIR, hub_key="hybrid", name="hybrid"),
            load_calibrator(EXPERT_CALIBRATORS["hybrid"]),
        ),
        "hybrid_nc": (
            LFCCLCNNExpert(cache_dir=MODEL_CACHE_DIR, hub_key="hybrid_nc", name="hybrid_nc"),
            load_calibrator(EXPERT_CALIBRATORS["hybrid_nc"]),
        ),
    }

    print(
        f"\n{'clip':30s} {'truth':6s} {'sr':>6s} | "
        f"{'scipy (BACKEND NOW)':>21s} | {'librosa (TRAINING)':>21s}"
    )
    print(f"{'':30s} {'':6s} {'':6s} | {'hyb P':>9s} {'nc P':>10s} | {'hyb P':>9s} {'nc P':>10s}")
    tally = {"scipy": [0, 0], "librosa": [0, 0]}  # [spoof caught, real false-alarmed]
    for fname, truth in CLIPS:
        p = ROOT / fname
        if not p.exists():
            print(f"{fname:30s} <missing>")
            continue
        a, b, sr = two_ways(p)
        row = []
        for arr, key in ((a, "scipy"), (b, "librosa")):
            probs = {}
            for mname, (m, cal) in models.items():
                _, _, pr = stats(m, arr, cal)
                probs[mname] = pr
            row.append(probs)
            flagged = probs["hybrid_nc"] >= 0.5
            if truth == "spoof" and flagged:
                tally[key][0] += 1
            if truth == "REAL" and flagged:
                tally[key][1] += 1
        print(
            f"{fname:30s} {truth:6s} {sr:6d} | "
            f"{row[0]['hybrid']:9.3f} {row[0]['hybrid_nc']:10.3f} | "
            f"{row[1]['hybrid']:9.3f} {row[1]['hybrid_nc']:10.3f}"
        )

    n_spoof = sum(1 for _, t in CLIPS if t == "spoof")
    n_real = sum(1 for _, t in CLIPS if t == "REAL")
    print(f"\nhybrid_nc @ P>=0.5   spoof caught / real false-alarmed  (of {n_spoof} spoof, {n_real} real)")
    for k, (c, f) in tally.items():
        print(f"  {k:8s}  caught {c}/{n_spoof}   false alarms {f}/{n_real}")


if __name__ == "__main__":
    main()
