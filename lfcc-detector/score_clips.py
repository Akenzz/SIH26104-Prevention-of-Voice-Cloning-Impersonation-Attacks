"""
score_clips.py — score whole clips with an LFCC-LCNN checkpoint.

Used to measure the fine-tune's effect: run it with the shipped checkpoint
(before) and the fine-tuned one (after) on the SAME clips and diff the numbers.

Scoring mirrors deployment, not training:
  * audio is scored at its NATIVE level (no RMS-normalization — the deployed
    detector gets whatever level the caller sends; normalization was only a
    TRAIN-time anti-shortcut fix).
  * each clip is silence-trimmed (same gate as the trainer's AudioDataset), then
    cut into non-overlapping 4 s windows; each window is scored.
  * higher logit = more spoof. The model's own boundary is logit>0 (train.py
    thresholds there); we also report the calibrated probability.

Run from lfcc-detector/:
  python score_clips.py --checkpoint checkpoints/hybrid_clean.pth --out before.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.lfcc_lcnn import LFCCLCNNWithFeatureExtraction  # noqa: E402

SRC_DIR = Path(r"E:\DatasetSIH\spoof_amogh_sentg\spoof")
TARGET_SR = 16000
WIN = TARGET_SR * 4
# Platt calibrator for hybrid_clean (artifacts/calibrator_hybrid_clean.json)
CAL_A, CAL_B = 0.44907376680772687, -1.3644504886434792


def trim_silence(x: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Frame-RMS gate ~-20 dB from the loudest frame — same as AudioDataset."""
    fl = max(int(0.02 * sr), 1)
    if len(x) < 2 * fl:
        return x
    nf = len(x) // fl
    frames = x[:nf * fl].reshape(nf, fl)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-9)
    thr = max(float(rms.max()) * 0.1, 1e-4)
    keep = np.where(rms > thr)[0]
    if len(keep) == 0:
        return x
    trimmed = x[keep[0] * fl:(keep[-1] + 1) * fl]
    return trimmed if len(trimmed) >= int(0.2 * sr) else x


def load_16k_mono(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != TARGET_SR:
        from math import gcd
        import scipy.signal as ss
        g = gcd(sr, TARGET_SR)
        x = ss.resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)
    return x


def windows(x: np.ndarray):
    x = trim_silence(x)
    if len(x) < WIN:
        reps = -(-WIN // max(len(x), 1))
        yield np.tile(x, reps)[:WIN]
        return
    for i in range(len(x) // WIN):
        yield x[i * WIN:(i + 1) * WIN]


def load_model(ckpt_path: str, device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
    model = LFCCLCNNWithFeatureExtraction(
        sample_rate=TARGET_SR, n_lfcc=20, with_deltas=True,
        embedding_dim=128, dropout=0.0,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--src-dir", default=str(SRC_DIR))
    ap.add_argument("--out", default=None, help="write per-clip results to JSON")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    results = {}
    print(f"\ncheckpoint: {args.checkpoint}   device: {device}")
    print("-" * 84)
    print(f"{'clip':30s} {'n':>3s} {'meanLogit':>10s} {'medLogit':>9s} {'meanProb':>9s} {'%spoof':>7s}")
    print("-" * 84)
    for f in sorted(Path(args.src_dir).iterdir()):
        if not f.is_file():
            continue
        try:
            x = load_16k_mono(f)
        except Exception as e:  # noqa: BLE001
            print(f"{f.name:30s}  load failed: {e}")
            continue
        logits = []
        with torch.no_grad():
            for w in windows(x):
                t = torch.from_numpy(w.astype(np.float32)).to(device).unsqueeze(0)
                lo, _ = model(t, return_embedding=False)
                logits.append(float(lo.squeeze().item()))
        logits = np.array(logits, dtype=np.float64)
        probs = 1.0 / (1.0 + np.exp(-(CAL_A * logits + CAL_B)))
        rec = {
            "n_windows": int(len(logits)),
            "mean_logit": round(float(logits.mean()), 4),
            "median_logit": round(float(np.median(logits)), 4),
            "mean_prob": round(float(probs.mean()), 4),
            "frac_spoof_logit_gt0": round(float((logits > 0).mean()), 4),
        }
        results[f.name] = rec
        print(f"{f.name:30s} {rec['n_windows']:3d} {rec['mean_logit']:10.3f} "
              f"{rec['median_logit']:9.3f} {rec['mean_prob']:9.3f} "
              f"{rec['frac_spoof_logit_gt0']*100:6.0f}%")
    print("-" * 84)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
