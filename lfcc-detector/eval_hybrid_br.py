"""Ship-readiness evaluation for hybrid_br_best.pth.

Reports the numbers that decide whether the model can be deployed, on the audio
the backend actually produces:

  * in-domain dev / eval EER (regression guard)
  * UNSEEN-GENERATOR eval_ood EER (27 generators held out of training) -- the
    number that actually predicts live behaviour
  * per-generator recall at the shipped operating threshold, so a good average
    can't hide a generator the model never catches
  * bonafide false-alarm rate at that same threshold

Each model is scored in the configuration it is valid in: gated checkpoints get
their gate (read from the checkpoint), ungated ones do not. Comparing `hybrid`
here is therefore comparing deployments, not features.

Run from lfcc-detector/:
  python eval_hybrid_br.py 2>&1 | tee eval_hybrid_br.log
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from data.dataset import AudioDataset, collate_fn          # noqa: E402
from models.lfcc_lcnn import LFCCLCNNWithFeatureExtraction  # noqa: E402
from training.config import TrainingConfig                  # noqa: E402

MANIFESTS = {
    "dev (in-domain)": ("hybrid_vad_chunks_dev.csv", "dev"),
    "eval (in-domain)": ("hybrid_vad_chunks_eval.csv", "eval"),
    "eval_ood (UNSEEN gens)": ("hybrid_vad_chunks_eval_ood.csv", "eval_ood"),
}
MODELS = {
    "hybrid (shipped)": "hybrid_clean.pth",
    "hybrid_br (new)": "hybrid_br_best.pth",
}


def eer_and_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        return float("nan"), float("nan")
    cands = np.unique(s)
    best = (2.0, 0.0, 0.0)
    for t in cands:
        far = float((s[y == 0] >= t).sum()) / n_neg
        frr = float((s[y == 1] < t).sum()) / n_pos
        d = abs(far - frr)
        if d < best[0]:
            best = (d, (far + frr) / 2, float(t))
    return best[1], best[2]


def load(name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(str(HERE / "checkpoints" / name), map_location=device, weights_only=False)
    cfg = ck["config"]
    m = LFCCLCNNWithFeatureExtraction(
        sample_rate=cfg.sample_rate, n_lfcc=cfg.n_lfcc, with_deltas=cfg.with_deltas,
        embedding_dim=cfg.embedding_dim, dropout=cfg.dropout,
    ).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, device, ck.get("band_gate_hz"), cfg


def score_manifest(model, device, gate, cfg, fname: str, split: str):
    ds = AudioDataset(str(REPO / "data_pipeline" / "manifests" / fname), split=split,
                      window_sec=cfg.window_sec, band_gate_hz=gate)
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)
    logits, labels, gens = [], [], []
    with torch.no_grad():
        for i, (x, y, meta) in enumerate(dl, 1):
            out = model(x.to(device), return_embedding=False)
            if isinstance(out, tuple):
                out = out[0]
            logits.extend(np.atleast_1d(out.squeeze(-1).cpu().numpy()).tolist())
            labels.extend(y.numpy().tolist())
            gens.extend([str(m.get("generator_id", "?")) for m in meta])
            if i % 40 == 0:
                print(f"    {min(i*16, len(ds))}/{len(ds)}", flush=True)
    return np.asarray(logits), np.asarray(labels), np.asarray(gens)


def main() -> int:
    results: dict = {}
    for mlabel, ckname in MODELS.items():
        if not (HERE / "checkpoints" / ckname).is_file():
            print(f"[WARN] missing {ckname}")
            continue
        model, device, gate, cfg = load(ckname)
        tag = f"gated {gate:.0f} Hz" if gate else "ungated"
        print(f"\n{'='*70}\n{mlabel}  ({ckname}, {tag})\n{'='*70}")
        results[mlabel] = {"checkpoint": ckname, "band_gate_hz": gate, "sets": {}}

        dev_thr = None
        for slabel, (fname, split) in MANIFESTS.items():
            print(f"  scoring {slabel} ...", flush=True)
            s, y, g = score_manifest(model, device, gate, cfg, fname, split)
            e, t = eer_and_threshold(y, s)
            if dev_thr is None:
                dev_thr = t  # operating point chosen on DEV, applied everywhere
            far = float((s[y == 0] >= dev_thr).mean()) if (y == 0).any() else float("nan")
            rec = float((s[y == 1] >= dev_thr).mean()) if (y == 1).any() else float("nan")
            print(f"    EER {e*100:6.2f}%   (own thresh {t:+.2f})   "
                  f"@dev-thresh {dev_thr:+.2f}: recall {rec*100:5.1f}%  false-alarm {far*100:5.1f}%")
            results[mlabel]["sets"][slabel] = {
                "n": int(len(y)), "eer": e, "own_threshold": t,
                "recall_at_dev_thresh": rec, "far_at_dev_thresh": far,
            }
            if split == "eval_ood":
                print("    per-generator recall @dev-thresh (unseen generators):")
                per = {}
                for gid in sorted(set(g[y == 1])):
                    m = (y == 1) & (g == gid)
                    r = float((s[m] >= dev_thr).mean())
                    per[gid] = {"n": int(m.sum()), "recall": r}
                for gid, v in sorted(per.items(), key=lambda kv: kv[1]["recall"]):
                    flag = "  <-- MISSED" if v["recall"] < 0.5 else ""
                    print(f"      {gid[:34]:34s} n={v['n']:4d}  recall {v['recall']*100:5.1f}%{flag}")
                results[mlabel]["sets"][slabel]["per_generator"] = per
                results[mlabel]["dev_threshold"] = dev_thr
                weak = [k for k, v in per.items() if v["recall"] < 0.5]
                print(f"    generators below 50% recall: {len(weak)}/{len(per)}")

    out = REPO / "data_pipeline" / "manifests" / "hybrid_br_eval.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\n[OK] wrote {out}")
    print("\nNOTE: thresholds are chosen on DEV and applied unchanged to eval/eval_ood,")
    print("      which is what deployment does. Per-set 'own thresh' is shown for reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
