"""Which model is better on audio resampled the way the BACKEND resamples it?

eval_hybrid_br.py scores the preprocessed corpus WAVs, which were written by the
training preprocessing (librosa/soxr). On that audio `hybrid` wins (eval_ood EER
5.91% vs 7.46%). But that is the artifact's home turf: those files carry the
near-Nyquist hole that `hybrid` keys on, and live backend audio does not.

This re-scores the SAME eval_ood rows after a round trip through the deployment
resampler -- upsample to 48 kHz and back down with scipy resample_poly, exactly
what audio/resample.py does to a browser's 48 kHz stream. The audio is otherwise
identical, so any EER change is attributable to the resampler fingerprint alone.

If `hybrid`'s advantage is real it survives the round trip. If the advantage was
the artifact, it does not.

Run from lfcc-detector/:
  python eval_deployment_path.py 2>&1 | tee eval_deployment_path.log
"""

from __future__ import annotations

import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from models.lfcc_lcnn import LFCCLCNNWithFeatureExtraction  # noqa: E402
from training.config import TrainingConfig                  # noqa: E402

import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

MANIFEST = REPO / "data_pipeline" / "manifests" / "hybrid_vad_chunks_eval_ood.csv"
TARGET_SR = 16000
WINDOW = 4 * TARGET_SR
VIA_RATE = 48000          # what a browser stream arrives at
N_PER_CLASS = 700         # subsample: the round trip is the expensive part
MODELS = {"hybrid (shipped)": "hybrid_clean.pth", "hybrid_br (new)": "hybrid_br_best.pth"}


def backend_round_trip(x: np.ndarray) -> np.ndarray:
    """16k -> 48k -> 16k using the backend's resampler, both directions."""
    from scipy.signal import resample_poly

    g1 = gcd(TARGET_SR, VIA_RATE)
    up = resample_poly(x, VIA_RATE // g1, TARGET_SR // g1)
    g2 = gcd(VIA_RATE, TARGET_SR)
    return resample_poly(up, TARGET_SR // g2, VIA_RATE // g2).astype(np.float32)


def lowpass(x: np.ndarray, cutoff: float | None) -> np.ndarray:
    if not cutoff:
        return x.astype(np.float32)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / TARGET_SR)
    spec[freqs >= cutoff] = 0.0
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def eer_of(y: np.ndarray, s: np.ndarray) -> float:
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        return float("nan")
    best = (2.0, 1.0)
    for t in np.unique(s):
        far = float((s[y == 0] >= t).sum()) / n_neg
        frr = float((s[y == 1] < t).sum()) / n_pos
        d = abs(far - frr)
        if d < best[0]:
            best = (d, (far + frr) / 2)
    return best[1]


def center_window(x: np.ndarray) -> np.ndarray:
    if len(x) < WINDOW:
        reps = -(-WINDOW // max(len(x), 1))
        x = np.tile(x, reps)[:WINDOW]
    elif len(x) > WINDOW:
        s = (len(x) - WINDOW) // 2
        x = x[s : s + WINDOW]
    return x.astype(np.float32)


def main() -> int:
    df = pd.read_csv(MANIFEST, low_memory=False)
    rng = np.random.default_rng(0)
    parts = []
    for lab in ("bonafide", "spoof"):
        sub = df[df.label == lab]
        take = min(N_PER_CLASS, len(sub))
        parts.append(sub.iloc[rng.choice(len(sub), take, replace=False)])
    df = pd.concat(parts, ignore_index=True)
    print(f"[INFO] {len(df)} rows ({(df.label=='bonafide').sum()} bona / "
          f"{(df.label=='spoof').sum()} spoof) from eval_ood")

    print(f"[INFO] loading audio + {TARGET_SR}->{VIA_RATE}->{TARGET_SR} scipy round trip ...")
    native, deployed, labels, gens = [], [], [], []
    for i, row in enumerate(df.itertuples(index=False), 1):
        try:
            x, sr = sf.read(row.path, dtype="float32", always_2d=False)
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != TARGET_SR:
            continue
        w = center_window(x)
        native.append(w)
        deployed.append(center_window(backend_round_trip(w)))
        labels.append(1 if row.label == "spoof" else 0)
        gens.append(str(row.generator_id))
        if i % 300 == 0:
            print(f"    {i}/{len(df)}", flush=True)
    labels = np.asarray(labels)
    gens = np.asarray(gens)
    print(f"[INFO] prepared {len(labels)} windows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for mlabel, ckname in MODELS.items():
        ck = torch.load(str(HERE / "checkpoints" / ckname), map_location=device,
                        weights_only=False)
        cfg = ck["config"]
        gate = ck.get("band_gate_hz")
        model = LFCCLCNNWithFeatureExtraction(
            sample_rate=cfg.sample_rate, n_lfcc=cfg.n_lfcc, with_deltas=cfg.with_deltas,
            embedding_dim=cfg.embedding_dim, dropout=cfg.dropout,
        ).to(device)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()

        row_out = {}
        for cond, arrs in (("corpus (librosa)", native), ("deployment (scipy round trip)", deployed)):
            scores = []
            with torch.no_grad():
                for j in range(0, len(arrs), 32):
                    batch = np.stack([lowpass(a, gate) for a in arrs[j : j + 32]])
                    t = torch.from_numpy(batch).float().to(device)
                    out = model(t, return_embedding=False)
                    if isinstance(out, tuple):
                        out = out[0]
                    scores.extend(np.atleast_1d(out.squeeze(-1).cpu().numpy()).tolist())
            s = np.asarray(scores)
            row_out[cond] = {"eer": eer_of(labels, s), "mean_logit": float(s.mean())}
        tag = f"gated {gate:.0f} Hz" if gate else "ungated"
        d = row_out["deployment (scipy round trip)"]["eer"] - row_out["corpus (librosa)"]["eer"]
        print(f"\n{mlabel}  ({tag})")
        for cond, v in row_out.items():
            print(f"    {cond:32s} EER {v['eer']*100:6.2f}%   mean logit {v['mean_logit']:+7.2f}")
        print(f"    -> resampler costs {d*100:+.2f} EER points")
        results[mlabel] = {"band_gate_hz": gate, "conditions": row_out,
                           "eer_delta_from_resampler": d}

    out = REPO / "data_pipeline" / "manifests" / "hybrid_br_deployment_eval.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\n[OK] wrote {out}")
    print("\nThe LAST column is the deployment number. A model whose EER jumps under the")
    print("round trip was reading the training resampler, not the voice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
