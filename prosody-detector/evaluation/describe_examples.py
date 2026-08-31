"""
Sanity check for describe_prosody_evidence: print example findings on real
bonafide and spoof eval clips, so we can eyeball that the plain-language output
matches what the numbers actually say for each clip.

Run from inside prosody-detector/ AFTER training:
  python evaluation/describe_examples.py --n 3
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "lfcc-detector"))

from features.prosody_features import extract_prosody_features  # noqa: E402
from features.describe import describe_prosody_evidence_structured, load_bonafide_bands  # noqa: E402
from expert.prosody_expert import ProsodyScorer  # noqa: E402
from features.prosody_features import features_to_vector  # noqa: E402
from data.dataset import AudioDataset  # noqa: E402


def _pick(ds, label, n):
    idxs = [i for i in range(len(ds)) if ds.df.iloc[i]["label"] == label]
    return idxs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/prosody_lr_v1.json")
    ap.add_argument("--manifest", default="../data_pipeline/manifests/multicorpus_final.csv")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    bands = load_bonafide_bands(args.artifact)
    scorer = ProsodyScorer(args.artifact)
    ds = AudioDataset(args.manifest, split=args.split, window_sec=4.0)

    for label in ("bonafide", "spoof"):
        print("\n" + "=" * 68)
        print(f"{label.upper()} example clips ({args.split})")
        print("=" * 68)
        for i in _pick(ds, label, args.n):
            audio, _, meta = ds[i]
            feats = extract_prosody_features(audio.numpy(), 16000)
            logit = scorer.logit_from_vector(features_to_vector(feats))
            prob = 1.0 / (1.0 + np.exp(-logit))
            findings = describe_prosody_evidence_structured(feats, bands=bands)
            gen = meta.get("generator_id", "?")
            print(f"\n[{label}] idx={i} gen={gen}  spoof_logit={logit:+.2f} P(spoof)={prob:.2f}")
            if not findings:
                print("   (no prosody anomalies outside the human range)")
            for f in findings:
                print(f"   - {f['finding']}  [{f['feature']}={f['value']} vs human "
                      f"p5={f['human_band']['p5']} p95={f['human_band']['p95']}]")


if __name__ == "__main__":
    main()
