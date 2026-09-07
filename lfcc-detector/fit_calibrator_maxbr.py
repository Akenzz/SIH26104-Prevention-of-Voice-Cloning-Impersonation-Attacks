"""Fit a Platt calibrator for hybrid_maxbr_best.pth on ITS OWN gated dev set.

Same procedure as fit_calibrator_br.py -- that script is hardcoded to
hybrid_br_best.pth + hybrid_vad_chunks_dev.csv, and pointing it at a different
model would fit on the wrong dev split, so this is the hybrid_maxbr twin rather
than a flag on that one.

Why a NEW calibrator is mandatory: hybrid_maxbr is a from-scratch model on a
different corpus, so its logits sit on their own scale. Serving it against
calibrator_hybrid_br.json would read one model's logits through another's Platt
fit -- probabilities stay in [0,1], the stream works, every risk band is wrong.

The 7 kHz gate is part of the model's input contract, so the fit runs on GATED
audio and the cutoff is READ FROM THE CHECKPOINT, never hardcoded. dev is split
in half by SPEAKER (not by row) so the reported EER/ECE are held out.

Run from lfcc-detector/:
  python fit_calibrator_maxbr.py 2>&1 | tee fit_calibrator_maxbr.log
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

CKPT = HERE / "checkpoints" / "hybrid_maxbr_best.pth"
DEV_MANIFEST = REPO / "data_pipeline" / "manifests" / "hybrid_maxbr_dev.csv"
OUT = REPO / "realtime-backend" / "artifacts" / "calibrator_hybrid_maxbr.json"
VERSION_TAG = "hybrid_maxbr"


def eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores)
    s, y = scores[order], labels[order]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        return float("nan"), float("nan")
    rates = []
    for t in np.unique(s):
        far = float((s[y == 0] >= t).sum()) / n_neg
        frr = float((s[y == 1] < t).sum()) / n_pos
        rates.append((abs(far - frr), (far + frr) / 2, float(t)))
    rates.sort()
    return rates[0][1], rates[0][2]


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1] if i < bins - 1 else probs <= 1.0)
        if not m.any():
            continue
        total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(total)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(str(CKPT), map_location=device, weights_only=False)
    cfg = ck["config"]
    gate = ck.get("band_gate_hz")
    print(f"[INFO] {CKPT.name}: epoch {ck.get('epoch')}, gated dev EER "
          f"{float(ck.get('best_eer', float('nan')))*100:.4f}%, band gate {gate} Hz")
    if not gate:
        print("[ERR] checkpoint has no band_gate_hz; refusing to guess")
        return 1

    ds = AudioDataset(str(DEV_MANIFEST), split="dev",
                      window_sec=cfg.window_sec, band_gate_hz=gate)
    model = LFCCLCNNWithFeatureExtraction(
        sample_rate=cfg.sample_rate, n_lfcc=cfg.n_lfcc,
        with_deltas=cfg.with_deltas, embedding_dim=cfg.embedding_dim,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    logits, labels, speakers = [], [], []
    with torch.no_grad():
        for i, (x, y, meta) in enumerate(loader, 1):
            out = model(x.to(device), return_embedding=False)
            if isinstance(out, tuple):
                out = out[0]
            logits.extend(out.squeeze(-1).cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
            speakers.extend([str(m.get("speaker_id", "?")) for m in meta])
            if i % 40 == 0:
                print(f"  scored {i*8}/{len(ds)}", flush=True)
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    speakers = np.asarray(speakers)

    # speaker-disjoint half/half so the reported numbers are held out
    uniq = np.unique(speakers)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    fit_spk = set(uniq[: max(1, len(uniq) // 2)].tolist())
    fit_m = np.array([s in fit_spk for s in speakers])
    held_m = ~fit_m
    if held_m.sum() == 0 or len(np.unique(labels[fit_m])) < 2:
        print("[WARN] speaker split degenerate; falling back to row split")
        fit_m = np.zeros(len(labels), dtype=bool)
        fit_m[::2] = True
        held_m = ~fit_m

    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logits[fit_m].reshape(-1, 1), labels[fit_m])
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])

    def prob(z):
        return 1.0 / (1.0 + np.exp(-(a * z + b)))

    h_eer, h_thr = eer(labels[held_m], logits[held_m])
    h_ece = ece(prob(logits[held_m]), labels[held_m])
    print(f"\n[FIT]  a={a:.10f}  b={b:.10f}   (fit on {int(fit_m.sum())} rows, "
          f"{len(fit_spk)} speakers)")
    print(f"[HELD-OUT] n={int(held_m.sum())}  EER={h_eer*100:.4f}%  ECE={h_ece:.5f}  "
          f"logit thresh at EER={h_thr:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "version": f"{VERSION_TAG}-gated{int(gate)}-platt-v1",
        "kind": "platt",
        "a": a,
        "b": b,
        "checkpoint": CKPT.name,
        "band_gate_hz": gate,
        "fitted_on": str(DEV_MANIFEST.name) + " (dev split, GATED, speaker-disjoint half)",
        "n_fit": int(fit_m.sum()),
        "n_heldout": int(held_m.sum()),
        "heldout_eer": h_eer,
        "heldout_ece": h_ece,
        "note": ("Fitted on band-gated audio. The model MUST be served with a "
                 f"{gate:.0f} Hz lowpass or these probabilities are invalid."),
    }, indent=2), encoding="utf-8")
    print(f"[OK] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
