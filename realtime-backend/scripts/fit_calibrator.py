"""Fit a Platt calibrator for a real expert on a held-out dev split.

The live pipeline maps a single-window logit to a probability with
``sigmoid(a * logit + b)`` (see calibration.py) and only then smooths + bands
it. This script fits ``a, b`` on real (logit, label) pairs so that probability
is honest instead of the identity passthrough shipped by default.

It deliberately reuses the backend's own resampler and the real expert adapter,
so the logits it fits on come from the same code path that runs live.

Usage:
    # prove the fitting math with no audio/model needed:
    python scripts/fit_calibrator.py --self-test

    # real fit against Task B's dev manifest:
    python scripts/fit_calibrator.py \
        --manifest ../data_pipeline/manifests/asvspoof19_dev.csv \
        --split dev --expert lfcc --out artifacts/calibrator.json

Manifest must have at least ``path`` and ``label`` columns; ``label`` is
``bonafide`` or ``spoof`` (spoof = 1 = higher logit). A ``split`` column is
filtered to --split when present.

Audio loading needs ``soundfile`` (pip install soundfile); it is imported lazily
so --self-test runs with numpy alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WINDOW_SAMPLES = 64000  # 4.0 s @ 16 kHz, matches the live window


# --------------------------------------------------------------------------- #
# Platt scaling: fit p = sigmoid(a*logit + b) by Newton-Raphson on the
# cross-entropy with Platt's smoothed targets (Platt 1999), avoiding the
# hard-0/1 overfit. Higher logit => higher P(spoof), so a should come out > 0.
# --------------------------------------------------------------------------- #
def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_platt(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200,
              tol: float = 1e-7) -> tuple[float, float]:
    """Robust Platt scaling (Lin, Lin & Weng 2007) with backtracking line search.

    Fits P(spoof) = 1/(1+exp(A*f+B)); returns it in this backend's convention
    p = sigmoid(a*f+b), i.e. a = -A, b = -B. The line search guarantees the
    negative-log-likelihood decreases every step, so it cannot diverge even when
    the scores saturate or are perfectly separable (a plain Newton step can).
    """
    f = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n_pos = float(np.sum(y > 0.5))
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both bonafide and spoof examples to fit a calibrator")
    hi = (n_pos + 1.0) / (n_pos + 2.0)
    lo = 1.0 / (n_neg + 2.0)
    t = np.where(y > 0.5, hi, lo)

    def nll(A: float, B: float) -> float:
        z = f * A + B
        # numerically stable t*z + log(1+exp(-z)) branch
        pos = z >= 0
        out = np.empty_like(z)
        out[pos] = t[pos] * z[pos] + np.log1p(np.exp(-z[pos]))
        out[~pos] = (t[~pos] - 1.0) * z[~pos] + np.log1p(np.exp(z[~pos]))
        return float(np.sum(out))

    A = 0.0
    B = float(np.log((n_neg + 1.0) / (n_pos + 1.0)))  # base-rate-aware init
    fval = nll(A, B)
    for _ in range(max_iter):
        z = f * A + B
        p = _sigmoid(-z)  # P(y=1) under 1/(1+exp(z))
        d2 = p * (1.0 - p)
        h11 = float(np.sum(f * f * d2)) + 1e-12
        h22 = float(np.sum(d2)) + 1e-12
        h21 = float(np.sum(f * d2))
        g1 = float(np.sum(f * (t - p)))
        g2 = float(np.sum(t - p))
        if abs(g1) < 1e-5 and abs(g2) < 1e-5:
            break
        det = h11 * h22 - h21 * h21
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB
        step = 1.0
        while step >= 1e-10:
            newA, newB = A + step * dA, B + step * dB
            newf = nll(newA, newB)
            if newf < fval + 1e-4 * step * gd:
                A, B, fval = newA, newB, newf
                break
            step *= 0.5
        if step < 1e-10 or max(abs(step * dA), abs(step * dB)) < tol:
            break
    return -A, -B


# --------------------------------------------------------------------------- #
# Reliability metrics
# --------------------------------------------------------------------------- #
def _clip01(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-12, 1.0 - 1e-12)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = _clip01(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        total += (np.sum(mask) / len(p)) * abs(acc - conf)
    return float(total)


def eer(logits: np.ndarray, y: np.ndarray) -> float:
    """Equal error rate from raw scores (monotonic calibration leaves it fixed)."""
    order = np.argsort(logits)
    s = logits[order]
    yy = y[order]
    n_pos = float(np.sum(yy > 0.5))
    n_neg = float(len(yy) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    best = 1.0
    thresholds = np.concatenate([[s[0] - 1.0], s, [s[-1] + 1.0]])
    for thr in thresholds:
        pred_pos = logits >= thr
        fa = float(np.sum(pred_pos & (y < 0.5))) / n_neg  # false accept (bona->spoof)
        miss = float(np.sum(~pred_pos & (y > 0.5))) / n_pos  # miss (spoof->bona)
        if abs(fa - miss) < best:
            best = abs(fa - miss)
            eer_val = (fa + miss) / 2.0
    return float(eer_val)


def report(tag: str, logits: np.ndarray, y: np.ndarray, a: float, b: float) -> None:
    p = _sigmoid(a * logits + b)
    print(
        f"  [{tag}] a={a:+.4f} b={b:+.4f}  ECE={ece(p, y):.4f}  "
        f"Brier={brier(p, y):.4f}  logloss={log_loss(p, y):.4f}"
    )


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #
def _label_to_y(raw: str) -> float | None:
    v = raw.strip().lower()
    if v in {"spoof", "fake", "1", "synthetic"}:
        return 1.0
    if v in {"bonafide", "bona-fide", "genuine", "real", "0"}:
        return 0.0
    return None


def _windows(audio: np.ndarray, max_windows: int) -> list[np.ndarray]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size <= WINDOW_SAMPLES or max_windows <= 1:
        return [audio[:WINDOW_SAMPLES] if audio.size > WINDOW_SAMPLES else audio]
    starts = np.linspace(0, audio.size - WINDOW_SAMPLES, num=max_windows, dtype=int)
    return [audio[s : s + WINDOW_SAMPLES] for s in starts]


def collect_scores(args) -> tuple[np.ndarray, np.ndarray, str]:
    import soundfile as sf  # lazy: only needed for real runs

    from audio.resample import to_mono, to_target_rate
    from config import Settings
    from experts.loader import load_experts

    settings = Settings(experts=[args.expert], fusion_mode="single")
    expert = load_experts(settings)[args.expert]
    print(f"expert={args.expert} version={expert.model_version} device={args.device}")

    manifest = Path(args.manifest).resolve()
    base = manifest.parent
    rows: list[tuple[Path, float]] = []
    # utf-8-sig strips a leading BOM if the manifest was written with one
    # (else the first column becomes "﻿path" and path lookups silently miss).
    with manifest.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if args.split and "split" in row and row["split"].strip().lower() != args.split.lower():
                continue
            y = _label_to_y(row.get("label", ""))
            if y is None:
                continue
            raw_path = row.get("path", "").strip()
            if not raw_path:
                continue
            p = Path(raw_path)
            if not p.is_absolute():
                p = (base / p).resolve()
            rows.append((p, y))

    if args.shuffle:
        import random

        random.Random(args.seed).shuffle(rows)
    if args.balance:
        import random

        pos = [r for r in rows if r[1] > 0.5]
        neg = [r for r in rows if r[1] <= 0.5]
        k = min(len(pos), len(neg))
        if args.limit:
            k = min(k, args.limit // 2)
        rows = pos[:k] + neg[:k]
        random.Random(args.seed + 1).shuffle(rows)  # interleave classes for a balanced eval tail
    elif args.limit:
        rows = rows[: args.limit]
    n_pos_rows = sum(1 for _, y in rows if y > 0.5)
    print(f"manifest={manifest.name} rows={len(rows)} (spoof={n_pos_rows} "
          f"bonafide={len(rows) - n_pos_rows}) split={args.split!r} "
          f"shuffle={args.shuffle} balance={args.balance} seed={args.seed}")

    logits: list[float] = []
    labels: list[float] = []
    skipped = 0
    for i, (path, y) in enumerate(rows):
        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"  skip {path.name}: {exc}")
            continue
        mono = to_mono(np.asarray(audio, dtype=np.float32), 1 if audio.ndim == 1 else audio.shape[1])
        resampled, _ = to_target_rate(mono, int(sr), settings.target_sample_rate)
        for w in _windows(resampled, args.max_windows_per_clip):
            logits.append(float(expert.score(w)["logit"]))
            labels.append(y)
        if (i + 1) % 200 == 0:
            print(f"  scored {i + 1}/{len(rows)} clips ({len(logits)} windows)")

    if skipped:
        print(f"  WARNING: skipped {skipped} unreadable/missing files")
    return np.asarray(logits), np.asarray(labels), expert.model_version


# --------------------------------------------------------------------------- #
# Self-test: synthetic miscalibrated logits, no audio/model needed.
# --------------------------------------------------------------------------- #
def self_test() -> int:
    rng = np.random.default_rng(0)
    n = 4000
    # "true" well-separated logits, then distort scale+shift to miscalibrate.
    bona = rng.normal(-2.0, 1.3, n)
    spoof = rng.normal(+2.2, 1.3, n)
    true_logits = np.concatenate([bona, spoof])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    observed = 3.1 * true_logits + 1.7  # miscalibrated (over-confident, biased)

    a, b = fit_platt(observed, y)
    print("self-test: fit on synthetic miscalibrated logits")
    report("identity", observed, y, 1.0, 0.0)
    report("fitted  ", observed, y, a, b)
    p_id = _sigmoid(observed)
    p_fit = _sigmoid(a * observed + b)
    ok = (a > 0) and (ece(p_fit, y) < ece(p_id, y)) and (log_loss(p_fit, y) < log_loss(p_id, y))
    # EER is unchanged by monotonic calibration:
    e_before, e_after = eer(observed, y), eer(a * observed + b, y)
    print(f"  EER identity={e_before:.4f} fitted={e_after:.4f} (should match)")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="dev-split manifest CSV (path,label[,split])")
    parser.add_argument("--split", default="dev", help="filter manifest split column")
    parser.add_argument("--expert", default="hybrid", help="expert to calibrate: hybrid, wavlm")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "calibrator.json"))
    parser.add_argument("--version", default=None, help="calibrator version label")
    parser.add_argument("--max-windows-per-clip", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="cap clips (quick trials)")
    parser.add_argument("--shuffle", dest="shuffle", action="store_true", default=True,
                        help="deterministically shuffle rows before --limit/eval split (default on)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--seed", type=int, default=0, help="shuffle/eval-split seed")
    parser.add_argument("--balance", action="store_true",
                        help="subsample to equal bonafide/spoof before scoring, so the "
                             "calibrator is not biased by a skewed dataset prior (e.g. "
                             "ASVspoof dev is ~9:1 spoof). Policy bands set the operating point.")
    parser.add_argument("--eval-frac", type=float, default=0.2,
                        help="fraction held out to report honest (out-of-fit) metrics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.manifest:
        parser.error("--manifest is required (or pass --self-test)")

    logits, labels, model_version = collect_scores(args)
    if len(logits) == 0:
        raise SystemExit("no scored windows; check manifest paths and labels")

    lg_pos = logits[labels > 0.5]
    lg_neg = logits[labels <= 0.5]
    print(f"logit separation  spoof: mean={lg_pos.mean():+.3f} std={lg_pos.std():.3f}   "
          f"bonafide: mean={lg_neg.mean():+.3f} std={lg_neg.std():.3f}   "
          f"AUC-proxy(EER)={eer(logits, labels):.4f}")

    n_pos = int(np.sum(labels > 0.5))
    n_neg = int(len(labels) - n_pos)
    print(f"collected {len(logits)} windows (spoof={n_pos} bonafide={n_neg})")

    # Hold out a fold for honest (out-of-fit) metric reporting. Rows were already
    # shuffled in collect_scores, so a tail slice is a random stratified-ish split.
    n_eval = int(len(logits) * args.eval_frac) if args.eval_frac > 0 else 0
    if n_eval < 20 or (len(logits) - n_eval) < 20:
        n_eval = 0  # too small to hold out; fit + report in-sample
    fit_lg, fit_lb = logits[: len(logits) - n_eval], labels[: len(labels) - n_eval]
    ev_lg, ev_lb = logits[len(logits) - n_eval:], labels[len(labels) - n_eval:]

    a, b = fit_platt(fit_lg, fit_lb)
    print(f"fit on {len(fit_lg)} windows; eval on {len(ev_lg)} held-out windows")
    if n_eval:
        print("calibration quality (HELD-OUT eval fold):")
        report("identity", ev_lg, ev_lb, 1.0, 0.0)
        report("fitted  ", ev_lg, ev_lb, a, b)
        print(f"  EER (held-out, threshold-free) = {eer(ev_lg, ev_lb):.4f}")
    else:
        print("calibration quality (in-sample; dataset too small to hold out):")
        report("identity", fit_lg, fit_lb, 1.0, 0.0)
        report("fitted  ", fit_lg, fit_lb, a, b)
    print("calibration quality (fit fold, for reference):")
    report("identity", fit_lg, fit_lb, 1.0, 0.0)
    report("fitted  ", fit_lg, fit_lb, a, b)

    ev_metrics = None
    if n_eval:
        p_ev = _sigmoid(a * ev_lg + b)
        ev_metrics = {
            "n_eval_windows": int(n_eval),
            "ece": round(ece(p_ev, ev_lb), 5),
            "brier": round(brier(p_ev, ev_lb), 5),
            "log_loss": round(log_loss(p_ev, ev_lb), 5),
            "eer": round(eer(ev_lg, ev_lb), 5),
            "ece_identity": round(ece(_sigmoid(ev_lg), ev_lb), 5),
        }

    version = args.version or f"platt-{args.expert}-{datetime.now(timezone.utc):%Y%m%d}"
    artifact = {
        "version": version,
        "kind": "platt",
        "a": a,
        "b": b,
        "note": (
            f"Platt fit for expert '{args.expert}' ({model_version}) on "
            f"{Path(args.manifest).name} split={args.split}; "
            f"fit on {len(fit_lg)} windows, {n_eval} held out for eval "
            f"(total spoof={n_pos}, bonafide={n_neg})"
            + (", class-balanced so the dataset prior is not baked in (policy bands "
               "set the operating point)" if args.balance else "")
            + ". Refit if the expert, the calibration corpus, or the fusion output "
            "changes; validity depends on the fit corpus matching the deployment "
            "distribution."
        ),
        "metadata": {
            "expert": args.expert,
            "model_version": model_version,
            "manifest": Path(args.manifest).name,
            "split": args.split,
            "balanced": bool(args.balance),
            "n_windows_total": len(logits),
            "n_windows_fit": len(fit_lg),
            "n_spoof": n_pos,
            "n_bonafide": n_neg,
            "max_windows_per_clip": args.max_windows_per_clip,
            "shuffle_seed": args.seed,
            "held_out_eval": ev_metrics,
            "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}  (kind=platt a={a:+.4f} b={b:+.4f} version={version})")
    print("NOTE: policy bands in artifacts/policy.json may need re-tuning against "
          "the calibrated probability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
