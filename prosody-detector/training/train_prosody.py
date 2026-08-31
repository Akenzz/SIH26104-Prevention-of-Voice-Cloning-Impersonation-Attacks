"""
Train the prosody/behavioral classifier.
=========================================
Extracts interpretable prosody features over the SAME manifest + splits the
other experts use (data_pipeline/manifests/multicorpus_final.csv), fits a small
StandardScaler + LogisticRegression, and writes a single portable JSON artifact
that both the backend adapter and the describe module read.

Artifact (prosody_lr_v1.json):
  feature_names, scaler_mean, scaler_scale, coef, intercept,
  bonafide_ranges  {feat: {p5, p50, p95}} over bonafide TRAIN rows (for describe),
  metadata

Feature extraction is the slow part, so the raw feature matrix is cached to
prosody_features_cache.npz and reused on reruns (delete it to force re-extract).

Run from inside prosody-detector/:
  python training/train_prosody.py \
      --manifest ../data_pipeline/manifests/multicorpus_final.csv \
      --out artifacts/prosody_lr_v1.json
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
# reuse the project's dataset loader so windows match every other expert exactly
sys.path.insert(0, str(_HERE.parent / "lfcc-detector"))

from features.prosody_features import extract_prosody_features, features_to_vector, FEATURE_NAMES  # noqa: E402


def _extract_split(manifest: str, split: str):
    """Return (X [n, d], y [n], groups [n]) for one split via AudioDataset."""
    from data.dataset import AudioDataset  # noqa

    ds = AudioDataset(manifest, split=split, window_sec=4.0)
    groups = ds.df["group"].tolist() if "group" in ds.df.columns else ["all"] * len(ds)
    X, y, g = [], [], []
    for i in range(len(ds)):
        audio, label, meta = ds[i]
        feats = extract_prosody_features(audio.numpy(), sr=16000)
        X.append(features_to_vector(feats))
        y.append(int(label))
        g.append(groups[meta["idx"]])
        if (i + 1) % 500 == 0:
            print(f"  [{split}] extracted {i + 1}/{len(ds)}", flush=True)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="../data_pipeline/manifests/multicorpus_final.csv")
    ap.add_argument("--out", default="artifacts/prosody_lr_v1.json")
    ap.add_argument("--cache", default="artifacts/prosody_features_cache.npz")
    ap.add_argument("--force", action="store_true", help="ignore feature cache")
    args = ap.parse_args()

    cache = Path(args.cache)
    if cache.exists() and not args.force:
        print(f"[cache] loading features from {cache}")
        d = np.load(cache, allow_pickle=True)
        X, y = d["X"], d["y"]
    else:
        print("[extract] train split (this is the slow step)...")
        X, y, _ = _extract_split(args.manifest, "train")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=X, y=y)
        print(f"[cache] wrote {cache}  X={X.shape}")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Xs, y)
    train_acc = clf.score(Xs, y)
    print(f"[fit] train accuracy={train_acc:.4f}  (bonafide={int((y==0).sum())}, spoof={int((y==1).sum())})")

    # bonafide human ranges (drive describe.py) — computed on bonafide TRAIN rows,
    # in RAW (unscaled) feature space so describe compares like-for-like.
    bona = X[y == 0]
    ranges = {}
    for j, name in enumerate(FEATURE_NAMES):
        col = bona[:, j]
        ranges[name] = {
            "p5": float(np.percentile(col, 5)),
            "p50": float(np.percentile(col, 50)),
            "p95": float(np.percentile(col, 95)),
        }

    artifact = {
        "version": "prosody-lr-v1",
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coef": clf.coef_[0].astype(float).tolist(),
        "intercept": float(clf.intercept_[0]),
        "bonafide_ranges": ranges,
        "metadata": {
            "manifest": str(args.manifest),
            "n_train": int(len(y)),
            "n_bonafide": int((y == 0).sum()),
            "n_spoof": int((y == 1).sum()),
            "train_accuracy": float(train_acc),
            "note": "Higher logit = more spoof. Trained on multicorpus_final train split.",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[OK] wrote {out}")
    # quick sanity: which features carry weight
    order = np.argsort(-np.abs(clf.coef_[0]))
    print("[weights] |coef| ranking:")
    for j in order:
        print(f"    {FEATURE_NAMES[j]:20s} {clf.coef_[0][j]:+.3f}")


if __name__ == "__main__":
    main()
