"""
data_pipeline/canary_silence.py
===============================
The shortcut detector. Train a classifier on ONE trivial, non-content feature
family — duration and silence geometry — and measure how well it separates
bonafide from spoof. A real spoof detector must NOT be beatable by this.

  Run it BEFORE preprocess_parity.py : may score high (raw TTS has tell-tale
                                        silence/loudness) — that's the warning.
  Run it AFTER  preprocess_parity.py : should collapse to ~chance (AUC ~0.5).

If AUC stays high after parity, your preprocessing didn't remove the shortcut and
any downstream EER is suspect.

Deps: scikit-learn (pip install scikit-learn).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def features(path: str):
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
    except Exception:
        return None
    n = len(x)
    if n == 0:
        return None
    absx = np.abs(x)
    thr = max(absx.max() * 0.02, 1e-4)
    voiced = absx > thr
    dur = n / sr
    if voiced.any():
        lead = int(np.argmax(voiced)) / sr
        trail = (n - 1 - int(np.argmax(voiced[::-1]))) / sr
        trailing_sil = dur - trail
    else:
        lead = trailing_sil = dur
    silence_frac = 1.0 - float(voiced.mean())
    return [dur, lead, trailing_sil, silence_frac]


def eer(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s)
    y = y[order]
    P, N = int(y.sum()), int(len(y) - y.sum())
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / max(P, 1)
    fpr = fp / max(N, 1)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2)


def main():
    ap = argparse.ArgumentParser(description="Silence/duration-only shortcut canary.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=6000, help="max clips (balanced) to sample")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, encoding="utf-8-sig")
    per = args.limit // 2
    # Balanced sample per class. Avoid groupby(...).apply() — in pandas 2.x it can
    # push the grouping column ('label') into the index, so itertuples drops it.
    parts = [g.sample(min(len(g), per), random_state=0) for _, g in df.groupby("label")]
    df = pd.concat(parts, ignore_index=True)

    X, y = [], []
    for r in df.itertuples(index=False):
        f = features(r.path)
        if f:
            X.append(f)
            y.append(1 if r.label == "spoof" else 0)
    X, y = np.array(X), np.array(y)
    if len(set(y.tolist())) < 2:
        raise SystemExit("Need both bonafide and spoof rows.")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    clf = LogisticRegression(max_iter=1000)
    p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    e = eer(y, p)
    print(f"silence/duration-only  AUC={auc:.3f}  EER={e:.3f}  (n={len(y):,})")
    if auc > 0.65:
        print("[WARN] classes are separable by silence/duration ALONE -> shortcut risk.")
        print("       Fix preprocessing (parity) before trusting any model EER.")
    else:
        print("[OK] silence/duration alone is ~uninformative; no obvious shortcut.")


if __name__ == "__main__":
    main()
