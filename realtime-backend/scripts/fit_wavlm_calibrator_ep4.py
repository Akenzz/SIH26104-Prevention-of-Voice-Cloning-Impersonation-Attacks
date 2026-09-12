"""
fit_wavlm_calibrator_ep4.py
===========================
Refit the Platt calibrator for the WavLM expert using the locally-trained
epoch-4 checkpoint (wavlm_maxbr_best.pt, dev_eer=15.05%).

Mirrors exactly what fit_calibrator.py does, but is self-contained and runs
from the realtime-backend/ directory without needing a separate val.csv on
a remote machine.

Data used: data_pipeline/manifests/asvspoof19_dev.csv
    - 24,845 rows (bonafide + spoof), local Windows paths under D:\\DatasetSIH
    - The dev split is speaker-disjoint from training, so the Platt fit is
      honest (out-of-training-distribution).

    NOTE: The wavlm_maxbr training corpus (74,672 chunks) includes ASVspoof19
    train + many other generators. ASVspoof19 dev uses different speakers and
    a subset of the same generators -- appropriate for calibration but not for
    OOD evaluation. Use asvspoof19_eval.csv or in_the_wild_eval_ood.csv for OOD.

Usage (from realtime-backend/):
    python scripts/fit_wavlm_calibrator_ep4.py
    python scripts/fit_wavlm_calibrator_ep4.py --limit 2000 --device cuda
    python scripts/fit_wavlm_calibrator_ep4.py --out artifacts/platt_v7.json --no-update-v5

Output:
    artifacts/platt_v7.json        -- the new calibrator artifact
    artifacts/platt_v5.json        -- overwritten in-place (EXPERT_CALIBRATORS["wavlm"])
                                       unless --no-update-v5 is passed

The script intentionally reuses the backend's expert adapter and audio
resampling so the logit distribution it fits on is identical to the one
the pipeline produces at inference time.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

# -- Ensure realtime-backend/ is on sys.path so config / experts are importable
ROOT = Path(__file__).resolve().parents[1]   # realtime-backend/
REPO = ROOT.parent                            # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WINDOW_SAMPLES = 64_000   # 4.0 s @ 16 kHz


# ---------------------------------------------------------------------------
# Platt scaling -- verbatim from fit_calibrator.py (Lin, Lin & Weng 2007)
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_platt(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> tuple[float, float]:
    """Robust Platt scaling with backtracking line search.

    Returns (a, b) such that P(spoof) = sigmoid(a * logit + b).
    """
    f = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n_pos = float(np.sum(y > 0.5))
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both bonafide and spoof examples to fit calibrator")
    hi = (n_pos + 1.0) / (n_pos + 2.0)
    lo = 1.0 / (n_neg + 2.0)
    t  = np.where(y > 0.5, hi, lo)

    def nll(A: float, B: float) -> float:
        z = f * A + B
        pos = z >= 0
        out = np.empty_like(z)
        out[pos]  = t[pos]  * z[pos]  + np.log1p(np.exp(-z[pos]))
        out[~pos] = (t[~pos] - 1.0) * z[~pos] + np.log1p(np.exp(z[~pos]))
        return float(np.sum(out))

    A    = 0.0
    B    = float(np.log((n_neg + 1.0) / (n_pos + 1.0)))
    fval = nll(A, B)
    for _ in range(max_iter):
        z  = f * A + B
        p  = _sigmoid(-z)
        d2 = p * (1.0 - p)
        h11 = float(np.sum(f * f * d2)) + 1e-12
        h22 = float(np.sum(d2))         + 1e-12
        h21 = float(np.sum(f * d2))
        g1  = float(np.sum(f * (t - p)))
        g2  = float(np.sum(t - p))
        if abs(g1) < 1e-5 and abs(g2) < 1e-5:
            break
        det = h11 * h22 - h21 * h21
        dA  = -(h22 * g1 - h21 * g2) / det
        dB  = -(-h21 * g1 + h11 * g2) / det
        gd  = g1 * dA + g2 * dB
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


# ---------------------------------------------------------------------------
# Reliability metrics
# ---------------------------------------------------------------------------

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
        lo_e, hi_e = edges[i], edges[i + 1]
        mask = (p > lo_e) & (p <= hi_e) if i > 0 else (p >= lo_e) & (p <= hi_e)
        if not np.any(mask):
            continue
        conf  = float(np.mean(p[mask]))
        acc   = float(np.mean(y[mask]))
        total += (np.sum(mask) / len(p)) * abs(acc - conf)
    return float(total)


def eer_metric(logits: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(logits)
    s   = logits[order]
    yy  = y[order]
    n_pos = float(np.sum(yy > 0.5))
    n_neg = float(len(yy) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    best    = 1.0
    eer_val = 0.5
    thresholds = np.concatenate([[s[0] - 1.0], s, [s[-1] + 1.0]])
    for thr in thresholds:
        pred_pos = logits >= thr
        fa   = float(np.sum(pred_pos & (y < 0.5))) / n_neg
        miss = float(np.sum(~pred_pos & (y > 0.5))) / n_pos
        if abs(fa - miss) < best:
            best    = abs(fa - miss)
            eer_val = (fa + miss) / 2.0
    return float(eer_val)


def report(tag: str, logits: np.ndarray, y: np.ndarray, a: float, b: float) -> None:
    p = _sigmoid(a * logits + b)
    print(
        f"  [{tag}]  a={a:+.4f}  b={b:+.4f}  "
        f"ECE={ece(p, y):.4f}  Brier={brier(p, y):.4f}  "
        f"logloss={log_loss(p, y):.4f}"
    )


# ---------------------------------------------------------------------------
# Score collection -- identical code path to the live backend
# ---------------------------------------------------------------------------

def _label_to_y(raw: str) -> "float | None":
    v = raw.strip().lower()
    if v in {"spoof", "fake", "1", "synthetic"}:
        return 1.0
    if v in {"bonafide", "bona-fide", "genuine", "real", "0"}:
        return 0.0
    return None


def _windows(audio: np.ndarray, max_windows: int) -> list:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size <= WINDOW_SAMPLES or max_windows <= 1:
        return [audio[:WINDOW_SAMPLES] if audio.size > WINDOW_SAMPLES else audio]
    starts = np.linspace(0, audio.size - WINDOW_SAMPLES, num=max_windows, dtype=int)
    return [audio[s: s + WINDOW_SAMPLES] for s in starts]


def collect_scores(args):
    import soundfile as sf
    from audio.resample import to_mono, to_target_rate
    from config import Settings
    from experts.loader import load_experts

    settings = Settings(experts=["wavlm"], fusion_mode="single", device=args.device)
    expert   = load_experts(settings)["wavlm"]
    print(f"\nexpert=wavlm  version={expert.model_version}  device={args.device}")

    manifest = Path(args.manifest).resolve()
    rows = []
    with manifest.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            split_col = row.get("split", "").strip().lower()
            if args.split and split_col and split_col != args.split.lower():
                continue
            y = _label_to_y(row.get("label", ""))
            if y is None:
                continue
            raw_path = row.get("path", "").strip()
            if not raw_path:
                continue
            p = Path(raw_path)
            if not p.is_absolute():
                p = (manifest.parent / p).resolve()
            rows.append((p, y))

    # Balanced subsample so the calibrator is not biased by a skewed prior.
    # platt_v5 used this same strategy; platt_v6 did not (hence slightly worse ECE).
    pos = [r for r in rows if r[1] > 0.5]
    neg = [r for r in rows if r[1] <= 0.5]
    rng = random.Random(args.seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(len(pos), len(neg))
    if args.limit:
        k = min(k, args.limit // 2)
    rows = pos[:k] + neg[:k]
    rng.shuffle(rows)

    print(
        f"manifest={manifest.name}  total_rows={len(rows)}  "
        f"(spoof={k}  bonafide={k})  split={args.split!r}"
    )

    logits: list = []
    labels: list = []
    skipped = 0
    pbar = tqdm(enumerate(rows), total=len(rows), desc="Scoring clips")
    for i, (path, y) in pbar:
        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                tqdm.write(f"  skip {path.name}: {exc}")
            continue
        mono = to_mono(
            np.asarray(audio, dtype=np.float32),
            1 if audio.ndim == 1 else audio.shape[1],
        )
        resampled, _ = to_target_rate(mono, int(sr), settings.target_sample_rate)
        for w in _windows(resampled, args.max_windows_per_clip):
            logits.append(float(expert.score(w)["logit"]))
            labels.append(y)
        pbar.set_postfix(windows=len(logits))

    if skipped:
        print(f"  WARNING: skipped {skipped} unreadable/missing files")
    return np.asarray(logits), np.asarray(labels), expert.model_version


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(REPO / "data_pipeline" / "manifests" / "asvspoof19_dev.csv"),
        help="Calibration manifest CSV (path,label[,split])",
    )
    parser.add_argument(
        "--split", default="dev",
        help="Value of the 'split' column to keep (default: dev)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Cap clips PER CLASS after balancing (0 = use all available)",
    )
    parser.add_argument("--max-windows-per-clip", type=int, default=1)
    parser.add_argument(
        "--eval-frac", type=float, default=0.2,
        help="Fraction of windows held out for honest metric reporting (default: 0.2)",
    )
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "platt_v7.json"),
        help="Output path for the new calibrator artifact (default: artifacts/platt_v7.json)",
    )
    parser.add_argument(
        "--no-update-v5", action="store_true",
        help=(
            "Do NOT overwrite artifacts/platt_v5.json. "
            "By default platt_v5.json IS overwritten because that is what "
            "EXPERT_CALIBRATORS['wavlm'] points to in config.py."
        ),
    )
    args = parser.parse_args()

    logits, labels, model_version = collect_scores(args)
    if len(logits) == 0:
        raise SystemExit("no scored windows -- check manifest paths and labels")

    lg_pos = logits[labels > 0.5]
    lg_neg = logits[labels <= 0.5]
    print(
        f"\nlogit separation:"
        f"\n  spoof    mean={lg_pos.mean():+.3f}  std={lg_pos.std():.3f}"
        f"\n  bonafide mean={lg_neg.mean():+.3f}  std={lg_neg.std():.3f}"
        f"\n  EER={eer_metric(logits, labels):.4f}"
    )

    n_pos = int(np.sum(labels > 0.5))
    n_neg = int(len(labels) - n_pos)
    print(f"total windows: {len(logits)}  (spoof={n_pos}  bonafide={n_neg})")

    # Train / eval split -- rows already shuffled in collect_scores
    n_eval = int(len(logits) * args.eval_frac) if args.eval_frac > 0 else 0
    if n_eval < 20 or (len(logits) - n_eval) < 20:
        n_eval = 0  # too small to hold out; fit + report in-sample
    fit_lg = logits[:len(logits) - n_eval]
    fit_lb = labels[:len(labels) - n_eval]
    ev_lg  = logits[len(logits) - n_eval:]
    ev_lb  = labels[len(labels) - n_eval:]

    a, b = fit_platt(fit_lg, fit_lb)
    print(f"\nfitted on {len(fit_lg)} windows; eval on {len(ev_lg)} held-out windows")

    if n_eval:
        print("calibration quality (HELD-OUT eval fold):")
        report("identity", ev_lg, ev_lb, 1.0, 0.0)
        report("fitted  ", ev_lg, ev_lb, a, b)
        print(f"  EER (held-out, threshold-free) = {eer_metric(ev_lg, ev_lb):.4f}")
    print("calibration quality (fit fold, for reference):")
    report("identity", fit_lg, fit_lb, 1.0, 0.0)
    report("fitted  ", fit_lg, fit_lb, a, b)

    ev_metrics = None
    if n_eval:
        p_ev = _sigmoid(a * ev_lg + b)
        ev_metrics = {
            "n_eval_windows": int(n_eval),
            "ece":            round(float(ece(p_ev, ev_lb)), 5),
            "brier":          round(float(brier(p_ev, ev_lb)), 5),
            "log_loss":       round(float(log_loss(p_ev, ev_lb)), 5),
            "eer":            round(float(eer_metric(ev_lg, ev_lb)), 5),
            "ece_identity":   round(float(ece(_sigmoid(ev_lg), ev_lb)), 5),
        }

    version = f"platt-wavlm-ep4-{datetime.now(timezone.utc):%Y%m%d}"
    artifact = {
        "version": version,
        "kind":    "platt",
        "a":       a,
        "b":       b,
        "note": (
            f"Platt fit for expert 'wavlm' ({model_version}) on "
            f"{Path(args.manifest).name} split={args.split}; "
            f"fit on {len(fit_lg)} windows, {n_eval} held out for eval "
            f"(total spoof={n_pos}, bonafide={n_neg}), class-balanced so the "
            "dataset prior is not baked in (policy bands set the operating point). "
            "Refit if the expert, the calibration corpus, or the fusion output "
            "changes; validity depends on the fit corpus matching the deployment "
            "distribution."
        ),
        "metadata": {
            "expert":               "wavlm",
            "model_version":        model_version,
            "manifest":             Path(args.manifest).name,
            "split":                args.split,
            "balanced":             True,
            "n_windows_total":      len(logits),
            "n_windows_fit":        len(fit_lg),
            "n_spoof":              n_pos,
            "n_bonafide":           n_neg,
            "max_windows_per_clip": args.max_windows_per_clip,
            "band_gate_hz":         None,
            "shuffle_seed":         args.seed,
            "held_out_eval":        ev_metrics,
            "fitted_utc":           datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"  kind=platt  a={a:+.4f}  b={b:+.4f}  version={version}")

    # Also overwrite platt_v5.json -- that is what EXPERT_CALIBRATORS["wavlm"]
    # points to in config.py, so the live backend picks it up on next restart.
    v5_path = ROOT / "artifacts" / "platt_v5.json"
    if not args.no_update_v5:
        v5_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"also updated {v5_path}  <- EXPERT_CALIBRATORS['wavlm'] live entry")
        print("Restart the backend to load the new calibrator.")
    else:
        print(f"--no-update-v5 set: {v5_path} was NOT overwritten.")
        print(f"To activate: copy {out_path.name} over platt_v5.json manually.")

    print(
        "\nNOTE: policy bands in artifacts/policy.json may need re-tuning "
        "against the new calibrated probability if the logit scale shifted "
        "between the previous model and ep4."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
