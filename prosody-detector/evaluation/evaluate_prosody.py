"""
prosody-detector/evaluation/evaluate_prosody.py
===============================================
Evaluate the prosody classifier on the SAME held-out slices as the other
experts, reporting each held-out family SEPARATELY from the in-domain number
(claims-ledger rule), formatted identically to
lfcc-detector/evaluation/evaluate_multicorpus.py so numbers are directly
comparable.

  IN-DOMAIN : split=='eval'      (Hindi XTTS, held-out speakers)
  HELD-OUT  : split=='eval_ood', bucketed by `group`:
                ood_hi_indicf5, ood_en_asv, ood_en_mlaad, ood_de_mlaad, ood_itw
              POOLED eval_ood

Run from inside prosody-detector/:
  python evaluation/evaluate_prosody.py \
      --artifact artifacts/prosody_lr_v1.json \
      --manifest ../data_pipeline/manifests/multicorpus_final.csv \
      --output-json artifacts/prosody_eval.json
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "lfcc-detector"))

from features.prosody_features import extract_prosody_features, features_to_vector  # noqa: E402
from expert.prosody_expert import ProsodyScorer  # noqa: E402
from data.dataset import AudioDataset  # noqa: E402
from evaluation.metrics import compute_all_metrics  # noqa: E402


def score_split(scorer, manifest, split):
    """Return {group: {'bona': [logits], 'spoof': [logits]}} for one split."""
    ds = AudioDataset(manifest, split=split, window_sec=4.0)
    if len(ds) == 0:
        return {}
    groups = ds.df["group"].tolist() if "group" in ds.df.columns else ["all"] * len(ds)
    buckets = defaultdict(lambda: {"bona": [], "spoof": []})
    for i in range(len(ds)):
        audio, label, meta = ds[i]
        feats = extract_prosody_features(audio.numpy(), sr=16000)
        logit = scorer.logit_from_vector(features_to_vector(feats))
        g = groups[meta["idx"]]
        buckets[g]["spoof" if label == 1 else "bona"].append(float(logit))
        if (i + 1) % 1000 == 0:
            print(f"  scored {i + 1}/{len(ds)} ({split}) ...", flush=True)
    return buckets


def report(name, bona, spoof):
    b, s = np.array(bona), np.array(spoof)
    if len(b) == 0 or len(s) == 0:
        print(f"  {name:18s}  SKIP (bona={len(b)}, spoof={len(s)})")
        return None
    m = compute_all_metrics(b, s)
    print(f"  {name:18s}  EER={m['eer']*100:6.2f}%  AUC={m['auc']:.4f}  "
          f"min-tDCF={m['min_tdcf']:.4f}  (bona={len(b)}, spoof={len(s)})")
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/prosody_lr_v1.json")
    ap.add_argument("--manifest", default="../data_pipeline/manifests/multicorpus_final.csv")
    ap.add_argument("--output-json", default="artifacts/prosody_eval.json")
    args = ap.parse_args()

    scorer = ProsodyScorer(args.artifact)
    results = {}

    print("\n" + "=" * 68)
    print("IN-DOMAIN  (split='eval' — trained-on distribution, held-out speakers)")
    print("=" * 68)
    ind = score_split(scorer, args.manifest, "eval")
    for g, d in sorted(ind.items()):
        results[f"indomain::{g}"] = report(g, d["bona"], d["spoof"])

    print("\n" + "=" * 68)
    print("HELD-OUT  (split='eval_ood' — generalization gap, per family)")
    print("=" * 68)
    ood = score_split(scorer, args.manifest, "eval_ood")
    pooled_b, pooled_s = [], []
    for g, d in sorted(ood.items()):
        results[f"ood::{g}"] = report(g, d["bona"], d["spoof"])
        pooled_b += d["bona"]; pooled_s += d["spoof"]
    print("  " + "-" * 60)
    results["ood::POOLED"] = report("POOLED_ood", pooled_b, pooled_s)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] wrote {args.output_json}")


if __name__ == "__main__":
    main()
