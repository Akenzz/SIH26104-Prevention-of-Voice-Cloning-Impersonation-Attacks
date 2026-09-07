"""Where does a gated LFCC model actually land at the SHIPPED policy bands?

Every number so far used an EER-optimal threshold, which is not what the backend
does. policy.json bands the CALIBRATED probability at low<0.35, uncertain, and
high>=0.65, so the deployed operating point is p>=0.65 -- not the EER point
(p=0.578 at logit 1.35). Those are different decisions and give different
recall/false-alarm rates.

This applies the real serving chain -- gate -> model -> that model's own Platt
calibrator -> policy bands -- to two held-out sets and reports the confusion at
the bands the backend will actually use, plus a sweep so the choice of high_min is
a measurement rather than a guess.

eval_ood is the honest headline: generators the model never trained on AND never
seen by the calibrator fit. dev-heldout is the speaker-disjoint half that the
calibrator fit excluded (same rng(0) speaker shuffle, reproduced here).

--model picks the chain. The checkpoint, calibrator, dev split and eval_ood split
move TOGETHER: hybrid_maxbr trained on a different corpus, so measuring it against
hybrid_br's dev/ood would score it on data that is partly in its own training set.

Run from lfcc-detector/:
  python eval_operating_point.py --model hybrid_maxbr 2>&1 | tee eval_op_maxbr.log
"""

from __future__ import annotations

import json
import sys
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

import soundfile as sf  # noqa: E402

# Defaults reproduce the original hybrid_br run byte-for-byte; --model selects a
# different (checkpoint, calibrator, manifest) triple. They must move together --
# scoring one model through another's calibrator or dev split silently reports
# numbers for a chain that is never served.
MODELS = {
    "hybrid_br": {
        "ckpt": "hybrid_br_best.pth",
        "cal": "calibrator_hybrid_br.json",
        "dev": "hybrid_vad_chunks_dev.csv",
        "ood": "hybrid_vad_chunks_eval_ood.csv",
    },
    "hybrid_maxbr": {
        "ckpt": "hybrid_maxbr_best.pth",
        "cal": "calibrator_hybrid_maxbr.json",
        "dev": "hybrid_maxbr_dev.csv",
        "ood": "hybrid_maxbr_eval_ood.csv",
    },
}

MODEL = "hybrid_br"
CKPT = HERE / "checkpoints" / MODELS[MODEL]["ckpt"]
CAL = REPO / "realtime-backend" / "artifacts" / MODELS[MODEL]["cal"]
POLICY = REPO / "realtime-backend" / "artifacts" / "policy.json"
DEV = REPO / "data_pipeline" / "manifests" / MODELS[MODEL]["dev"]
OOD = REPO / "data_pipeline" / "manifests" / MODELS[MODEL]["ood"]
TARGET_SR = 16000
WINDOW = 4 * TARGET_SR


def center_window(x: np.ndarray) -> np.ndarray:
    if len(x) < WINDOW:
        x = np.tile(x, -(-WINDOW // max(len(x), 1)))[:WINDOW]
    elif len(x) > WINDOW:
        s = (len(x) - WINDOW) // 2
        x = x[s : s + WINDOW]
    return x.astype(np.float32)


def score_manifest(model, gate, df, pathcol="path"):
    logits, labels, gens, spks = [], [], [], []
    batch, meta = [], []

    def flush():
        if not batch:
            return
        arr = np.stack(batch)
        spec = np.fft.rfft(arr, axis=-1)
        spec[:, np.fft.rfftfreq(arr.shape[-1], 1.0 / TARGET_SR) >= gate] = 0.0
        arr = np.fft.irfft(spec, n=arr.shape[-1], axis=-1).astype(np.float32)
        with torch.no_grad():
            out = model(torch.from_numpy(arr).float(), return_embedding=False)
            if isinstance(out, tuple):
                out = out[0]
            vals = np.atleast_1d(out.squeeze(-1).cpu().numpy())
        for v, (lab, g, s) in zip(vals, meta):
            logits.append(float(v)); labels.append(lab); gens.append(g); spks.append(s)
        batch.clear(); meta.clear()

    for i, row in enumerate(df.itertuples(index=False), 1):
        try:
            x, sr = sf.read(getattr(row, pathcol), dtype="float32", always_2d=False)
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != TARGET_SR:
            continue
        batch.append(center_window(x))
        meta.append((1 if row.label == "spoof" else 0,
                     str(getattr(row, "generator_id", "?")),
                     str(getattr(row, "speaker_id", "?"))))
        if len(batch) == 32:
            flush()
        if i % 500 == 0:
            print(f"    {i}/{len(df)}", flush=True)
    flush()
    return (np.asarray(logits), np.asarray(labels),
            np.asarray(gens), np.asarray(spks))


def confusion(p: np.ndarray, y: np.ndarray, low: float, high: float) -> dict:
    spoof, real = y == 1, y == 0
    return {
        "n": int(len(y)),
        "spoof_high_recall": float((p[spoof] >= high).mean()),
        "spoof_uncertain": float(((p[spoof] >= low) & (p[spoof] < high)).mean()),
        "spoof_missed_low": float((p[spoof] < low).mean()),
        "real_false_alarm_high": float((p[real] >= high).mean()),
        "real_uncertain": float(((p[real] >= low) & (p[real] < high)).mean()),
        "real_correct_low": float((p[real] < low).mean()),
    }


def show(name: str, c: dict, low: float, high: float) -> None:
    print(f"\n  {name}  (n={c['n']}, bands low<{low} / uncertain / high>={high})")
    print(f"    SPOOF: {c['spoof_high_recall']*100:5.1f}% high (caught)   "
          f"{c['spoof_uncertain']*100:5.1f}% uncertain   "
          f"{c['spoof_missed_low']*100:5.1f}% low (MISSED)")
    print(f"    REAL : {c['real_false_alarm_high']*100:5.1f}% high (FALSE ALARM)   "
          f"{c['real_uncertain']*100:5.1f}% uncertain   "
          f"{c['real_correct_low']*100:5.1f}% low (correct)")


def main() -> int:
    global CKPT, CAL, DEV, OOD
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODELS), default=MODEL,
                    help="Which (checkpoint, calibrator, dev, eval_ood) chain to "
                         "measure. These move together by design.")
    ap.add_argument("--out", default=None,
                    help="Where to write the JSON summary "
                         "(default: manifests/<model>_operating_point.json)")
    args = ap.parse_args()

    spec = MODELS[args.model]
    CKPT = HERE / "checkpoints" / spec["ckpt"]
    CAL = REPO / "realtime-backend" / "artifacts" / spec["cal"]
    DEV = REPO / "data_pipeline" / "manifests" / spec["dev"]
    OOD = REPO / "data_pipeline" / "manifests" / spec["ood"]
    print(f"[INFO] model={args.model}  ckpt={CKPT.name}  cal={CAL.name}\n"
          f"       dev={DEV.name}  eval_ood={OOD.name}")
    for p in (CKPT, CAL, DEV, OOD):
        if not p.exists():
            print(f"[ERR] missing {p}")
            return 1

    cal = json.loads(CAL.read_text(encoding="utf-8"))
    pol = json.loads(POLICY.read_text(encoding="utf-8"))
    a, b = float(cal["a"]), float(cal["b"])
    low, high = float(pol["low_max"]), float(pol["high_min"])
    print(f"[INFO] calibrator {cal['version']}  a={a:.6f} b={b:.6f}")
    print(f"[INFO] policy {pol['version']}  low_max={low}  high_min={high}")

    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg, gate = ck["config"], float(ck["band_gate_hz"])
    model = LFCCLCNNWithFeatureExtraction(
        sample_rate=cfg.sample_rate, n_lfcc=cfg.n_lfcc, with_deltas=cfg.with_deltas,
        embedding_dim=cfg.embedding_dim, dropout=cfg.dropout,
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"[INFO] {CKPT.name}: gate {gate:.0f} Hz")

    def prob(z):
        return 1.0 / (1.0 + np.exp(-(a * z + b)))

    results = {}

    print("\n[1] eval_ood -- unseen generators, never seen by the calibrator fit")
    ood = pd.read_csv(OOD, low_memory=False)
    lo, yo, go, _ = score_manifest(model, gate, ood)
    po = prob(lo)
    c_ood = confusion(po, yo, low, high)
    show("eval_ood", c_ood, low, high)
    results["eval_ood"] = c_ood

    print("\n    per-generator recall at the SHIPPED band (p >= "
          f"{high}), worst 8 of {len(set(go[yo==1]))}:")
    rec = []
    for g in sorted(set(go[yo == 1])):
        m = (go == g) & (yo == 1)
        rec.append((float((po[m] >= high).mean()), g, int(m.sum())))
    rec.sort()
    for r, g, n in rec[:8]:
        print(f"      {r*100:5.1f}%  {g}  (n={n})")
    dead = [(g, r) for r, g, _ in rec if r < 0.5]
    print(f"    generators below 50% recall at the shipped band: {len(dead)}/{len(rec)}"
          + (f"  -> {[g for g, _ in dead]}" if dead else ""))
    results["eval_ood_generators_below_50pct"] = [g for g, _ in dead]
    results["eval_ood_worst_generator"] = {"name": rec[0][1], "recall": rec[0][0]}

    print("\n[2] dev, speaker-disjoint HELD-OUT half (the calibrator's fit excluded it)")
    dev = pd.read_csv(DEV, low_memory=False)
    ld, yd, _, sd = score_manifest(model, gate, dev)
    uniq = np.unique(sd)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    fit_spk = set(uniq[: max(1, len(uniq) // 2)].tolist())
    held = np.array([s not in fit_spk for s in sd])
    pd_ = prob(ld[held])
    c_dev = confusion(pd_, yd[held], low, high)
    show("dev-heldout", c_dev, low, high)
    results["dev_heldout"] = c_dev

    print("\n[3] threshold sweep on eval_ood -- is 0.65 the right high_min?")
    print(f"    {'high_min':>9s} {'spoof caught':>13s} {'false alarm':>12s} {'youden J':>9s}")
    best = (-1.0, None)
    for t in [0.35, 0.45, 0.50, 0.578, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        r = float((po[yo == 1] >= t).mean())
        f = float((po[yo == 0] >= t).mean())
        j = r - f
        mark = "  <- shipped" if abs(t - high) < 1e-9 else ""
        print(f"    {t:9.3f} {r*100:12.1f}% {f*100:11.1f}% {j:9.3f}{mark}")
        if j > best[0]:
            best = (j, t)
    print(f"    best Youden J on eval_ood at high_min={best[1]} (J={best[0]:.3f}); "
          f"shipped {high} gives J={c_ood['spoof_high_recall']-c_ood['real_false_alarm_high']:.3f}")
    results["sweep_best_high_min"] = {"threshold": best[1], "youden_j": best[0]}
    results["shipped_high_min"] = high

    out = (Path(args.out) if args.out else
           REPO / "data_pipeline" / "manifests" / f"{args.model}_operating_point.json")
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\n[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
