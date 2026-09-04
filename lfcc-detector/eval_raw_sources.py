"""The decisive test: score RAW sources, not preprocessed corpus WAVs.

eval_deployment_path.py round-trips the already-preprocessed 16 kHz corpus files,
which understates the problem: those WAVs were written by librosa, so the
near-Nyquist brickwall is already baked in and a scipy round trip cannot restore
the band. The model still sees the hole either way.

This starts from the ORIGINAL files on E: (native 22050/24000/44100/48000 Hz) and
builds each window twice:
  * TRAIN path   -- librosa.load(sr=16000), i.e. what data_pipeline wrote to disk
  * DEPLOY path  -- scipy resample_poly, i.e. what realtime-backend/audio/resample.py
                    actually does to live audio
Same source clip, same content, only the resampler differs. This is the exact
train/serve mismatch, measured on the audio the backend will really receive.

`hybrid` is scored ungated (how it ships); `hybrid_br` gated at its checkpoint
cutoff (how it ships). Each model in the configuration it is valid in.

Run from lfcc-detector/:
  python eval_raw_sources.py 2>&1 | tee eval_raw_sources.log
"""

from __future__ import annotations

import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from models.lfcc_lcnn import LFCCLCNNWithFeatureExtraction  # noqa: E402
from training.config import TrainingConfig                  # noqa: E402

RAW_MANIFEST = REPO / "data_pipeline" / "manifests" / "hybrid_new_raw.csv"
TARGET_SR = 16000
WINDOW = 4 * TARGET_SR
N_PER_CLASS = 500
MODELS = {"hybrid (shipped)": "hybrid_clean.pth", "hybrid_br (new)": "hybrid_br_best.pth"}


def load_train_path(path: str) -> np.ndarray | None:
    """What preprocess_vad_slice.py did: librosa/soxr down to 16 kHz."""
    import librosa

    try:
        x, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    except Exception:
        return None
    return x.astype(np.float32)


def load_deploy_path(path: str) -> np.ndarray | None:
    """What the backend does: read native, scipy resample_poly to 16 kHz."""
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(int(sr), TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, int(sr) // g)
    return np.ascontiguousarray(x, dtype=np.float32)


def center_window(x: np.ndarray) -> np.ndarray:
    if len(x) < WINDOW:
        x = np.tile(x, -(-WINDOW // max(len(x), 1)))[:WINDOW]
    elif len(x) > WINDOW:
        s = (len(x) - WINDOW) // 2
        x = x[s : s + WINDOW]
    return x.astype(np.float32)


def lowpass(x: np.ndarray, cutoff: float | None) -> np.ndarray:
    if not cutoff:
        return x.astype(np.float32)
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(len(x), 1.0 / TARGET_SR) >= cutoff] = 0.0
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def eer_of(y: np.ndarray, s: np.ndarray) -> float:
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        return float("nan")
    best = (2.0, 1.0)
    for t in np.unique(s):
        far = float((s[y == 0] >= t).sum()) / n_neg
        frr = float((s[y == 1] < t).sum()) / n_pos
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]


def main() -> int:
    df = pd.read_csv(RAW_MANIFEST, low_memory=False)
    pathcol = "path" if "path" in df.columns else df.columns[0]
    rng = np.random.default_rng(0)
    parts = []
    for lab in ("bonafide", "spoof"):
        sub = df[df.label == lab]
        parts.append(sub.iloc[rng.choice(len(sub), min(N_PER_CLASS, len(sub)), replace=False)])
    df = pd.concat(parts, ignore_index=True)
    print(f"[INFO] {len(df)} raw source clips "
          f"({(df.label=='bonafide').sum()} bona / {(df.label=='spoof').sum()} spoof)")

    train_w, deploy_w, labels, rates = [], [], [], []
    for i, row in enumerate(df.itertuples(index=False), 1):
        p = getattr(row, pathcol)
        a, b = load_train_path(p), load_deploy_path(p)
        if a is None or b is None or len(a) < TARGET_SR or len(b) < TARGET_SR:
            continue
        train_w.append(center_window(a))
        deploy_w.append(center_window(b))
        labels.append(1 if row.label == "spoof" else 0)
        try:
            import soundfile as sf
            rates.append(sf.info(p).samplerate)
        except Exception:
            rates.append(-1)
        if i % 200 == 0:
            print(f"    {i}/{len(df)}", flush=True)
    labels = np.asarray(labels)
    rates = np.asarray(rates)
    print(f"[INFO] {len(labels)} clips loaded both ways")
    for lab, nm in ((0, "bonafide"), (1, "spoof")):
        m = rates[labels == lab]
        native = float((m == TARGET_SR).mean()) * 100 if len(m) else 0.0
        print(f"    {nm:9s}: {native:5.1f}% already 16 kHz (never resampled -> full band)")

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

        per = {}
        raw = {}
        for cond, arrs in (("TRAIN path (librosa)", train_w),
                           ("DEPLOY path (scipy)", deploy_w)):
            scores = []
            with torch.no_grad():
                for j in range(0, len(arrs), 32):
                    batch = np.stack([lowpass(a, gate) for a in arrs[j : j + 32]])
                    out = model(torch.from_numpy(batch).float().to(device),
                                return_embedding=False)
                    if isinstance(out, tuple):
                        out = out[0]
                    scores.extend(np.atleast_1d(out.squeeze(-1).cpu().numpy()).tolist())
            s = np.asarray(scores)
            raw[cond] = s
            per[cond] = {
                "eer": eer_of(labels, s),
                "mean_logit_bonafide": float(s[labels == 0].mean()),
                "mean_logit_spoof": float(s[labels == 1].mean()),
            }
        flip = float((np.sign(raw["TRAIN path (librosa)"]) !=
                      np.sign(raw["DEPLOY path (scipy)"])).mean())
        gap = float(np.abs(raw["TRAIN path (librosa)"] -
                           raw["DEPLOY path (scipy)"]).mean())
        tag = f"gated {gate:.0f} Hz" if gate else "ungated"
        print(f"\n{mlabel}  ({tag})")
        for cond, v in per.items():
            print(f"    {cond:22s} EER {v['eer']*100:6.2f}%   "
                  f"mean logit bona {v['mean_logit_bonafide']:+7.2f} / "
                  f"spoof {v['mean_logit_spoof']:+7.2f}")
        d = per["DEPLOY path (scipy)"]["eer"] - per["TRAIN path (librosa)"]["eer"]
        print(f"    -> train->deploy costs {d*100:+.2f} EER points; "
              f"mean |logit| gap {gap:.2f}; verdict FLIPS on {flip*100:.1f}% of clips")
        results[mlabel] = {"band_gate_hz": gate, "conditions": per,
                           "eer_delta_train_to_deploy": d,
                           "mean_abs_logit_gap": gap, "verdict_flip_rate": flip}

    out = REPO / "data_pipeline" / "manifests" / "hybrid_br_raw_source_eval.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\n[OK] wrote {out}")
    print("\nThe DEPLOY row is the only one that describes live behaviour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
