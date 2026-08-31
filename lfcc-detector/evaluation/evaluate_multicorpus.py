"""
lfcc-detector/evaluation/evaluate_multicorpus.py
================================================
Evaluate a checkpoint on the multi-corpus manifest, reporting EACH held-out
family SEPARATELY from the in-domain number (claims-ledger rule: never merge
in-domain skill with generalization).

Reports:
  IN-DOMAIN   : split=='eval'      (Hindi XTTS, held-out speakers)   -- the "skill" number
  HELD-OUT    : split=='eval_ood', broken down by the `group` column:
                  ood_hi_indicf5  -- unseen generator, Hindi (speakers may overlap; weak probe)
                  ood_en_asv      -- ASVspoof A07-A19 (unseen attacks, disjoint speakers)
                  ood_en_mlaad    -- 15 held-out English MLAAD generators
                  ood_de_mlaad    -- 4 held-out German MLAAD generators
                  ood_itw         -- In-the-Wild real-world (public-figure audio)
                POOLED eval_ood   -- all held-out bonafide vs all held-out spoof

The `group` column is not surfaced by AudioDataset's metadata, so we recover it
via metadata['idx'] -> dataset.df.iloc[idx]['group'] (shuffle=False keeps order,
but we key on idx to be safe).

Run from inside lfcc-detector/ so the checkpoint's pickled TrainingConfig and the
data/models packages import cleanly:
  python evaluation/evaluate_multicorpus.py --checkpoint checkpoints/mc_v3.pth \
      --manifest ../data_pipeline/manifests/multicorpus_final.csv \
      --output-json ../data_pipeline/manifests/mc_v3_eval.json
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import AudioDataset, collate_fn
from models.detector import LFCCLCNNDetector
from evaluation.metrics import compute_all_metrics


def score_split(detector, manifest, split, batch_size=16):
    """Return {group: {'bona': [...], 'spoof': [...]}} for one split."""
    ds = AudioDataset(manifest, split=split, window_sec=4.0)
    if len(ds) == 0:
        return {}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    buckets = defaultdict(lambda: {"bona": [], "spoof": []})
    groups = ds.df["group"].tolist() if "group" in ds.df.columns else ["all"] * len(ds)

    n = 0
    for audios, labels, metas in loader:
        logits, _ = detector.batch_forward(audios)
        if logits.ndim == 0:
            logits = np.array([logits.item()])
        for logit, label, meta in zip(logits, labels.tolist(), metas):
            g = groups[meta["idx"]]
            buckets[g]["spoof" if label == 1 else "bona"].append(float(logit))
            n += 1
        if n % 2000 < batch_size:
            print(f"  scored {n}/{len(ds)} ({split}) ...", flush=True)
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    detector = LFCCLCNNDetector(checkpoint_path=args.checkpoint)
    results = {}

    print("\n" + "=" * 68)
    print("IN-DOMAIN  (split='eval' — the trained-on distribution, held-out speakers)")
    print("=" * 68)
    ind = score_split(detector, args.manifest, "eval", args.batch_size)
    for g, d in sorted(ind.items()):
        results[f"indomain::{g}"] = report(g, d["bona"], d["spoof"])

    print("\n" + "=" * 68)
    print("HELD-OUT  (split='eval_ood' — generalization gap, reported per family)")
    print("=" * 68)
    ood = score_split(detector, args.manifest, "eval_ood", args.batch_size)
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
