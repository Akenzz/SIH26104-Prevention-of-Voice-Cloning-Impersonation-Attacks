"""
data_pipeline/finalize_multicorpus.py
=====================================
Merge the Hindi passthrough manifest with the freshly-processed non-Hindi
manifest into the single training manifest, and HARD-VALIDATE it before we
spend hours training on it.

Checks:
  1. split x label balance (per-pool 1:1 is the design; a few clips <0.4s get
     dropped by preprocess, so tiny deltas are fine — pos_weight/EER absorb them)
  2. exact-file leakage: no utterance_id shared across splits (HARD, must be 0)
  3. real-speaker leakage for corpora with genuine speakers (Kathbath+XTTS,
     ASVspoof2019_LA):
       - train<->dev  and  train<->eval  : HARD FAIL (would inflate the
         early-stopping / in-domain headline number)
       - train<->eval_ood                : INFO ONLY. Hindi IndicF5 is a
         generator-OOD probe whose speakers are (by construction) the original
         train speakers; that is expected, not leakage of the in-domain metric.
  4. every path exists on disk (drop the rare preprocess-skipped clip).

Writes data_pipeline/manifests/multicorpus_final.csv; exits non-zero if a hard
check fails so the training launch can gate on it.
"""

import sys
from pathlib import Path

import pandas as pd

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
HINDI = MANIFEST_DIR / "mc_hindi_processed.csv"
NEW = MANIFEST_DIR / "mc_new_processed.csv"
OUT = MANIFEST_DIR / "multicorpus_final.csv"

REAL_SPEAKER_CORPORA = {"Kathbath+XTTS", "ASVspoof2019_LA"}


def main():
    hi = pd.read_csv(HINDI, encoding="utf-8-sig", dtype={"speaker_id": str})
    new = pd.read_csv(NEW, encoding="utf-8-sig", dtype={"speaker_id": str})
    df = pd.concat([hi, new], ignore_index=True)
    print(f"Merged: {len(hi):,} Hindi + {len(new):,} new = {len(df):,} rows")

    hard_fail = []

    # 1. balance
    print("\nsplit x label:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())

    # 2. exact-file leakage across splits (HARD)
    dup = df.groupby("utterance_id")["split"].nunique()
    leaked = dup[dup > 1]
    if len(leaked):
        hard_fail.append(f"{len(leaked)} utterance_ids span >1 split (file leakage)")
        print("\n[FAIL] utterance leakage examples:", list(leaked.index[:5]))
    else:
        print("\n[OK] no utterance_id spans multiple splits (no exact-file leakage)")

    # 3. real-speaker leakage
    print("\nspeaker disjointness (real-speaker corpora):")
    for corpus in REAL_SPEAKER_CORPORA:
        sub = df[df.source_dataset == corpus]
        if len(sub) == 0:
            continue
        tr = set(sub[sub.split == "train"].speaker_id)
        for other in ("dev", "eval", "eval_ood"):
            ov = tr & set(sub[sub.split == other].speaker_id)
            if ov and other in ("dev", "eval"):
                hard_fail.append(f"{corpus}: {len(ov)} speakers shared train<->{other}")
                flag = "LEAK-HARD"
            elif ov:
                flag = "info (generator-OOD probe, expected)"
            else:
                flag = "OK"
            print(f"  {corpus:18s} train<->{other:8s}: {len(ov):4d} shared  [{flag}]")

    # 4. path existence -> drop the rare preprocess-skipped clip
    exists = df.path.map(lambda p: Path(p).exists())
    if (~exists).any():
        print(f"\n[WARN] dropping {int((~exists).sum())} rows with missing audio:")
        print(df[~exists].groupby(["source_dataset", "split"]).size().to_string())
        df = df[exists].reset_index(drop=True)
    else:
        print("\n[OK] all audio paths exist")

    print("\nFINAL split x label:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print("\nFINAL group x label:")
    print(df.groupby(["group", "label"]).size().unstack(fill_value=0).to_string())
    print("\nTRAIN spoof generators:",
          df[(df.split == "train") & (df.label == "spoof")].generator_id.nunique())

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n[OK] wrote {OUT}  ({len(df):,} rows)")

    if hard_fail:
        print("\n" + "!" * 60)
        for h in hard_fail:
            print("HARD-FAIL:", h)
        print("!" * 60)
        sys.exit(1)
    print("\n[PASS] all hard checks green - safe to train.")


if __name__ == "__main__":
    main()
