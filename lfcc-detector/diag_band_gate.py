"""Does an identical 7 kHz band gate collapse the scipy-vs-librosa verdict gap?

diag_resampler_fix.py established the problem: the same clip scores bonafide via
the backend's scipy resampler and spoof via training's librosa, because they
differ by ~59 dB at 7.9-8 kHz and the corpus made that band a label proxy.

Randomizing the band cannot fix it. Measured over 400 train windows, only 1.6% of
spoof windows are natively full-band (the librosa brickwall is already baked into
the preprocessed WAVs on disk and is irreversible), versus 13.6% of bonafide. So
"full band => bonafide" stays a valid rule no matter how many extra holes
augmentation stamps on; augmentation can only ADD holes, never restore a band.

The alternative is to DELETE the untrustworthy region from both classes, at train
time and at inference: lowpass at 7000 Hz, below which the two resamplers agree to
0.02-0.7 dB. This script tests the premise on the SHIPPED models, before spending
a retrain: if the gate is the right fix, the scipy-vs-librosa gap must collapse
even for a model that never trained with it.

Reports per clip: logit via scipy and via librosa, ungated and gated, plus the
gap. Success = |gap| shrinks toward 0 under the gate.
"""

from __future__ import annotations

import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "realtime-backend"))
sys.path.insert(0, str(REPO))

TARGET_SR = 16000
WINDOW = 4 * TARGET_SR
MAX_SEC = 30
GATE_HZ = 7000.0

CLIPS = [
    # the clip that started this: in training, yet reads bonafide live
    (r"E:\DatasetSIH\mlaad\fake\ml\Edge-TTS\dorothy_and_wizard_oz_01_f000031.wav", "spoof"),
    (r"E:\DatasetSIH\Local_test\sarosh_real_3sept.wav", "REAL"),
    (r"E:\DatasetSIH\Local_test\sarosh_ZsVicqBh_original.wav", "REAL"),
    (r"E:\DatasetSIH\Local_test\sudhanva.wav", "REAL"),
    (r"E:\DatasetSIH\Local_test\sarosh_fireredtts.wav", "spoof"),
    (r"E:\DatasetSIH\Local_test\sarosh_omni.wav", "spoof"),
    (r"E:\DatasetSIH\Local_test\sarosh_chatterbox_spoof.wav", "spoof"),
    (r"E:\DatasetSIH\Local_test\sarosh_styletts.wav", "spoof"),
    (r"E:\DatasetSIH\Local_test\sarosh_spoof.wav", "spoof"),
    (r"E:\DatasetSIH\Local_test\SAROSH_SOPRO_V2_SPOOF.wav", "spoof"),
]


def lowpass(x: np.ndarray, cutoff: float, sr: int = TARGET_SR) -> np.ndarray:
    if cutoff >= sr / 2:
        return x.astype(np.float32)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    spec[freqs >= cutoff] = 0.0
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def two_ways(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (scipy-resampled, librosa-resampled, native_sr)."""
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
    b = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
    return a, b, sr


def wins(a: np.ndarray) -> list[np.ndarray]:
    if len(a) < WINDOW:
        return [np.pad(a, (0, WINDOW - len(a)))]
    return [a[i : i + WINDOW] for i in range(0, len(a) - WINDOW + 1, WINDOW // 2)]


def mean_logit(expert, a: np.ndarray) -> float:
    return float(np.mean([expert.score(w)["logit"] for w in wins(a)]))


def main() -> None:
    from config import MODEL_CACHE_DIR
    from experts.lfcc import LFCCLCNNExpert

    print("loading experts ...", flush=True)
    models = {
        "hybrid": LFCCLCNNExpert(cache_dir=MODEL_CACHE_DIR, hub_key="hybrid", name="hybrid"),
        "hybrid_nc": LFCCLCNNExpert(
            cache_dir=MODEL_CACHE_DIR, hub_key="hybrid_nc", name="hybrid_nc"
        ),
    }

    for mname, expert in models.items():
        print(f"\n=== {mname} — mean logit (>0 = spoof) ===")
        print(
            f"{'clip':34s} {'truth':6s} | {'UNGATED':>17s} | {'GATED 7 kHz':>17s}"
        )
        print(f"{'':34s} {'':6s} | {'scipy':>7s} {'librosa':>7s} {'gap':>7s}"[:60]
              + f" | {'scipy':>7s} {'librosa':>7s} {'gap':>7s}")
        gaps_un, gaps_gt = [], []
        for path_s, truth in CLIPS:
            p = Path(path_s)
            if not p.exists():
                print(f"{p.name[:34]:34s} <missing>")
                continue
            a, b, _ = two_ways(p)
            u_s, u_l = mean_logit(expert, a), mean_logit(expert, b)
            g_s = mean_logit(expert, lowpass(a, GATE_HZ))
            g_l = mean_logit(expert, lowpass(b, GATE_HZ))
            gaps_un.append(abs(u_s - u_l))
            gaps_gt.append(abs(g_s - g_l))
            print(
                f"{p.name[:34]:34s} {truth:6s} | "
                f"{u_s:7.2f} {u_l:7.2f} {u_s - u_l:7.2f} | "
                f"{g_s:7.2f} {g_l:7.2f} {g_s - g_l:7.2f}"
            )
        if gaps_un:
            print(
                f"\n  mean |scipy - librosa| gap:  ungated {np.mean(gaps_un):6.2f}"
                f"   gated {np.mean(gaps_gt):6.2f}"
                f"   -> {100 * (1 - np.mean(gaps_gt) / max(np.mean(gaps_un), 1e-9)):.0f}% smaller"
            )


if __name__ == "__main__":
    main()
