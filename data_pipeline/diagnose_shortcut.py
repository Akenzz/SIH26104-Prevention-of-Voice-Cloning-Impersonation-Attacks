"""
data_pipeline/diagnose_shortcut.py
==================================
canary_silence.py fires (AUC 0.899) but reports only one combined number on the
WHOLE file. This pins down (a) WHICH trivial feature separates the classes and
(b) how much of it survives into the exact 4 s window the model actually sees
(center-crop if >4 s, zero-pad if <4 s — same as data/dataset.py at eval).

If the separability lives in `silence_frac` and survives windowing, the shortcut
is the studio-vs-field noise-floor gap and it reaches the model -> EER suspect.
If it lives in `dur` and collapses after windowing, the 4 s crop already kills it.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

SR = 16000
WIN = 4 * SR


def feats(x):
    """[dur, lead_sil, trail_sil, silence_frac, zero_frac] for a mono signal."""
    n = len(x)
    if n == 0:
        return None
    absx = np.abs(x)
    thr = max(absx.max() * 0.02, 1e-4)
    voiced = absx > thr
    dur = n / SR
    if voiced.any():
        lead = int(np.argmax(voiced)) / SR
        trail = (n - 1 - int(np.argmax(voiced[::-1]))) / SR
        trailing = dur - trail
    else:
        lead = trailing = dur
    silence_frac = 1.0 - float(voiced.mean())
    zero_frac = float(np.mean(absx < 1e-6))   # exact-zero (digital) samples, e.g. pad
    return [dur, lead, trailing, silence_frac, zero_frac]


def _trim_silence(x):
    """Frame-RMS silence gate, matching lfcc-detector/data/dataset.py._trim_silence."""
    fl = max(int(0.02 * SR), 1)
    if len(x) < 2 * fl:
        return x
    n_frames = len(x) // fl
    frames = x[:n_frames * fl].reshape(n_frames, fl)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-9)
    thr = max(float(rms.max()) * 0.1, 1e-4)
    keep = np.where(rms > thr)[0]
    if len(keep) == 0:
        return x
    trimmed = x[keep[0] * fl: (keep[-1] + 1) * fl]
    if len(trimmed) < int(0.2 * SR):
        return x
    return trimmed


def to_window(x):
    """Mirror data/dataset.py eval path: trim silence, tile-pad <WIN, center-crop >WIN."""
    x = _trim_silence(x)
    n = len(x)
    if n < WIN:
        reps = -(-WIN // max(n, 1))
        return np.tile(x, reps)[:WIN]
    if n > WIN:
        s = (n - WIN) // 2
        return x[s:s + WIN]
    return x


NAMES = ["dur", "lead_sil", "trail_sil", "silence_frac", "zero_frac"]


def univariate(X, y):
    out = {}
    for j, nm in enumerate(NAMES):
        a = roc_auc_score(y, X[:, j])
        out[nm] = max(a, 1 - a)   # direction-agnostic separability
    return out


def combined(X, y):
    p = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5,
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default=None, help="restrict to one split (e.g. eval)")
    ap.add_argument("--limit", type=int, default=6000)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, encoding="utf-8-sig")
    if args.split:
        df = df[df["split"] == args.split]
    per = args.limit // 2
    parts = [g.sample(min(len(g), per), random_state=0) for _, g in df.groupby("label")]
    df = pd.concat(parts, ignore_index=True)

    Xw, Xwin, y = [], [], []
    for r in df.itertuples(index=False):
        try:
            x, sr = sf.read(r.path, dtype="float32", always_2d=False)
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        fw = feats(x)
        fwin = feats(to_window(x))
        if fw is None or fwin is None:
            continue
        Xw.append(fw)
        Xwin.append(fwin)
        y.append(1 if r.label == "spoof" else 0)

    Xw, Xwin, y = np.array(Xw), np.array(Xwin), np.array(y)
    print(f"n={len(y):,}  (spoof={int(y.sum())}, bonafide={int(len(y)-y.sum())})  "
          f"split={args.split or 'ALL'}\n")

    for tag, X in [("WHOLE FILE", Xw), ("4s WINDOW (what the model sees)", Xwin)]:
        print(f"== {tag} ==")
        uni = univariate(X, y)
        for nm in NAMES:
            print(f"   {nm:14s} separability AUC = {uni[nm]:.3f}")
        print(f"   {'ALL (logreg)':14s} combined     AUC = {combined(X, y):.3f}\n")


if __name__ == "__main__":
    main()
