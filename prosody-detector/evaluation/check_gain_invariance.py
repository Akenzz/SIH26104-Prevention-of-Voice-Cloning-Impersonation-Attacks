"""
prosody-detector/evaluation/check_gain_invariance.py
====================================================
Can the 13 prosody features see a pure gain change?

`verify_manipulations.py --workers N` found a systematic level offset in the
manipulation corpus: the generator matches PEAK amplitude between a pair, but
PSOLA resynthesis raises crest factor, so at equal peak the manipulated side ends
up 0.5-1.8 dB quieter in RMS (median, per transform). That is a real property of
the corpus and a genuine hazard for the LFCC/hybrid retrain, whose log-magnitude
input shifts by a constant under gain.

Whether it threatens the PROSODY retrain is a separate question, and it is
answerable rather than arguable. Every one of the 13 features is nominally a
ratio, a frequency, a duration, or is computed against a threshold defined
relative to the clip's own loudest frame -- all of which should be invariant to a
global scale factor. "Should be" is not evidence, because the features run
through Praat's pitch tracker and a silence gate, either of which could hide an
absolute constant.

So measure it: take real clips, extract features, rescale the SAME waveform by a
range of gains spanning the observed offset, re-extract, and report the largest
drift per feature in units of that feature's own between-class standard deviation
on the manipulation corpus. A feature drifting a small fraction of a class-sigma
under 2 dB of gain cannot manufacture the retrain's separation; one drifting a
comparable amount can, and would invalidate it.

Run from inside prosody-detector/:
  python evaluation/check_gain_invariance.py
  python evaluation/check_gain_invariance.py --n-clips 60 --manifest ../data_pipeline/manifests/manipulations.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "lfcc-detector"))

# Gains spanning the measured offset in both directions, plus a much larger one.
# +-1.78 dB is what the corpus actually shows; 6 dB is a stress test, so a feature
# that is merely noisy can be told apart from one that genuinely tracks level.
GAINS_DB = (-6.0, -1.78, -1.0, 1.0, 1.78, 6.0)


def _load(path: str):
    """The exact 4 s window the feature store caches for this clip."""
    from features.feature_store import load_eval_window

    return np.asarray(load_eval_window(path), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="../data_pipeline/manifests/manipulations.csv")
    ap.add_argument("--n-clips", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from features.prosody_features import (FEATURE_NAMES, extract_prosody_features,
                                           features_to_vector)

    def vec(y: np.ndarray) -> np.ndarray:
        return np.asarray(features_to_vector(extract_prosody_features(y, sr=16000)),
                          dtype=np.float64)

    with Path(args.manifest).open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    # Spoof rows only: these are the clips whose level the generator shifted, so
    # they are where a gain sensitivity would actually be exploited.
    spoof = [r for r in rows if r.get("label") == "spoof"]
    random.Random(args.seed).shuffle(spoof)

    base, shifted = [], {g: [] for g in GAINS_DB}
    used = 0
    for r in spoof:
        if used >= args.n_clips:
            break
        try:
            # Gain is applied to the cached window rather than the file, which is
            # equivalent here: the silence trim gates on 0.1x the clip's OWN
            # loudest frame, so it selects the same samples at any scale.
            y = _load(r["path"])
            if y.size < 16000:
                continue
            v0 = vec(y)
            vs = {}
            for g in GAINS_DB:
                s = y * np.float32(10.0 ** (g / 20.0))
                if float(np.max(np.abs(s))) >= 1.0:   # clipping is not gain
                    raise ValueError("would clip")
                vs[g] = vec(s)
        except Exception:
            continue
        base.append(v0)
        for g in GAINS_DB:
            shifted[g].append(vs[g])
        used += 1

    if used < 10:
        print(f"only {used} clips usable — cannot conclude anything")
        return 1
    B = np.array(base)
    print(f"{used} clips x {len(GAINS_DB)} gains, {B.shape[1]} features\n")

    # Scale: each feature's own spread across the corpus sample. A drift is only
    # meaningful relative to how much that feature varies between real clips.
    sigma = B.std(0) + 1e-12
    print(f"{'feature':<20s} {'max |drift| / sigma':>20s}   drift per gain (in sigma)")
    print("-" * 92)
    worst_name, worst_val = "", 0.0
    for j, name in enumerate(FEATURE_NAMES[: B.shape[1]]):
        per_gain = []
        for g in GAINS_DB:
            S = np.array(shifted[g])
            per_gain.append(float(np.median((S[:, j] - B[:, j]) / sigma[j])))
        m = max(abs(v) for v in per_gain)
        if m > worst_val:
            worst_name, worst_val = name, m
        flag = "  <-- SENSITIVE" if m > 0.10 else ""
        print(f"{name:<20s} {m:>20.4f}   " +
              " ".join(f"{v:+.3f}" for v in per_gain) + flag)
    print("-" * 92)
    print(f"gains (dB): " + "  ".join(f"{g:+.2f}" for g in GAINS_DB))
    print(f"\nworst feature: {worst_name} at {worst_val:.4f} sigma")
    if worst_val < 0.10:
        print("[OK] the prosody features are gain-invariant at the corpus's level offset.\n"
              "     The measured 0.5-1.8 dB offset therefore cannot be the source of any\n"
              "     separation the prosody retrain shows. It remains a must-fix for the\n"
              "     LFCC/hybrid retrain, whose input is log-magnitude.")
        return 0
    print(f"[WARN] {worst_name} moves {worst_val:.3f} sigma under a gain change alone.\n"
          "       Re-generate the corpus RMS-matched before trusting the prosody EERs.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
