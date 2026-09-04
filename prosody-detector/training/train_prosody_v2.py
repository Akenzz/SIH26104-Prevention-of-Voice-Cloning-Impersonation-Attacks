"""
Train and evaluate the manipulation-aware prosody expert (Fix 2).
================================================================
WHY THIS EXISTS. `evaluation/diagnose_by_generator.py` tested the hypothesis that
the shipped prosody expert's weak discrimination concentrates on the attack class
the spectral experts miss (non-neural DSP manipulation). It REFUTED it:

    non_neural_waveform   9 generators   mean AUC 0.570   mean EER 44.34%
    dsp_vocoder           7 generators   mean AUC 0.729   mean EER 31.65%
    neural_waveform      21 generators   mean AUC 0.598   mean EER 42.67%

So the shipped artifact is near chance on exactly the class it was supposed to
cover, and its only real signal is on classic parametric vocoders (WORLD /
STRAIGHT / Vocaine) -- and even there it swings from 17.6% to 64.6% EER between
generators, i.e. it is reading system-specific prosody habits, not a physical
resynthesis artifact.

But that artifact was trained on `multicorpus_final` train, which is ~all neural
TTS. It has never seen a DSP manipulation. The open question this script answers:

    trained ON manipulations, does prosody detect manipulations --
    including transform families it never saw?

Three variants are fitted on identical cached features and scored on identical
slices, so the comparison is apples to apples:

    shipped     artifacts/prosody_lr_v1.json, unchanged (the baseline)
    manip_only  fitted on the manipulation train split alone (a specialist)
    combined    fitted on manipulations + a neural sample (one artifact for both)

and every slice is reported separately, in particular:

    manip_eval_heldout   unseen speakers AND three transform families withheld
                         from training -- the only honest generalization number

Nothing shipped is overwritten. New artifacts go to artifacts/prosody_lr_*.json
and the full result set to artifacts/manip_experiment.json.

Run from inside prosody-detector/:
  python training/train_prosody_v2.py
  python training/train_prosody_v2.py --workers 6 --neural-train-cap 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
# lfcc-detector must come first so `evaluation.metrics` resolves to the repo's
# canonical EER; prosody's own helpers live in the top-level prosody_metrics.
sys.path.insert(0, str(_HERE.parent / "lfcc-detector"))

MANIP = "../data_pipeline/manifests/manipulations.csv"
NEURAL = "../data_pipeline/manifests/hybrid_vad_chunks.csv"
MULTI = "../data_pipeline/manifests/multicorpus_final.csv"

# Transform families withheld from train/dev by generate_manipulations.py. Kept
# as a literal so this script fails loudly if that list ever changes underneath.
HELDOUT_TRANSFORMS = ("manip_pv_pitch_shift", "manip_resample_speed",
                      "manip_praat_time_stretch")


def _read(path: str, keep) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh) if keep(r)]


def _tag(rows: list[dict], name: str, gen_key: str = "generator_id") -> list[dict]:
    """Attach the slice name, binary label and sub-group key used for breakdowns."""
    for r in rows:
        r["_slice"] = name
        r["_y"] = 0 if r["label"].strip().lower() == "bonafide" else 1
        r["_gen"] = (r.get(gen_key) or "?").strip()
    return rows


def _cap(rows: list[dict], bona_cap: int, spoof_cap: int, seed: int) -> list[dict]:
    """Cap spoof per generator and bonafide as one pool.

    Two caps, not one: spoof needs a per-generator cap so a single large generator
    cannot dominate a slice, while bonafide carries generator_id='none' throughout
    and so must be capped as a single pool -- capping it "per generator" would
    silently shrink it to the spoof cap and make every EER unstable.
    """
    rng = random.Random(seed)
    by: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by[(r["_y"], r["_gen"] if r["_y"] == 1 else "")].append(r)
    out: list[dict] = []
    for (y, _), v in by.items():
        rng.shuffle(v)
        cap = spoof_cap if y == 1 else bona_cap
        out.extend(v if cap <= 0 else v[:cap])
    return out


def build_slices(args) -> dict[str, list[dict]]:
    """Every train/eval pool used by the experiment, as tagged manifest rows.

    Manipulation slices come from the paired corpus, so within a slice the two
    classes share speakers, corpus, channel, language and duration distribution:
    any separation the model achieves has to come from the transform itself.
    """
    heldout = set(HELDOUT_TRANSFORMS)
    manip = _read(args.manip, lambda r: True)
    if not any(r["label"] == "spoof" for r in manip):
        raise SystemExit(f"no spoof rows in {args.manip}")
    present = {r["generator_id"] for r in manip if r["label"] == "spoof"}
    if not heldout & present:
        raise SystemExit(f"none of the withheld transforms {HELDOUT_TRANSFORMS} "
                         f"appear in {args.manip}; the eval slice would be empty")

    def manip_slice(split: str, want_heldout: bool) -> list[dict]:
        """Rows for one manipulation slice, keeping each spoof clip's own pair.

        Selecting on the spoof row's transform is not enough: the paired bonafide
        rows carry generator_id='none', so filtering by transform alone would
        drop them and leave a spoof-only slice. Pairs are recovered through the
        shared source stem the generator wrote into both filenames.
        """
        spoof = [r for r in manip if r["split"] == split and r["label"] == "spoof"
                 and (r["generator_id"] in heldout) == want_heldout]
        stems = {Path(r["path"]).name.split("__")[0] for r in spoof}
        bona = [r for r in manip if r["split"] == split and r["label"] == "bonafide"
                and Path(r["path"]).name.split("__")[0] in stems]
        return spoof + bona

    slices: dict[str, list[dict]] = {
        "manip_train": _tag(manip_slice("train", False), "manip_train"),
        "manip_dev": _tag(manip_slice("dev", False), "manip_dev"),
        "manip_eval_seen": _tag(manip_slice("eval", False), "manip_eval_seen"),
        "manip_eval_heldout": _tag(manip_slice("eval", True), "manip_eval_heldout"),
    }
    if args.manip_frac < 1.0:
        # Subsample by SOURCE STEM, never by row: dropping rows independently would
        # split pairs apart and destroy the property the whole corpus exists for.
        rng = random.Random(args.seed)
        for name, rows in slices.items():
            stems = sorted({Path(r["path"]).name.split("__")[0] for r in rows})
            rng.shuffle(stems)
            keep = set(stems[: max(2, int(len(stems) * args.manip_frac))])
            slices[name] = [r for r in rows
                            if Path(r["path"]).name.split("__")[0] in keep]

    # Neural corpus: the operational manifest the shipped hybrid expert trained on.
    for name, split in (("neural_dev", "dev"), ("neural_eval", "eval"),
                        ("neural_eval_ood", "eval_ood")):
        rows = _tag(_read(args.neural, lambda r, s=split: r["split"] == s), name)
        slices[name] = _cap(rows, args.eval_bona_cap, args.eval_spoof_cap, args.seed)
    # 130 generators in the neural train split, so the spoof cap is per generator
    # and deliberately small: the point is coverage of many systems, not volume.
    slices["neural_train"] = _cap(
        _tag(_read(args.neural, lambda r: r["split"] == "train"), "neural_train"),
        args.train_bona_cap, args.train_spoof_cap, args.seed)

    # ASVspoof slices live only in multicorpus_final, and the diagnostic already
    # cached their features, so including them is nearly free.
    asv = _tag(_read(args.multi, lambda r: r["split"] in ("dev", "eval_ood")
                     and (r.get("group") or "").strip() in ("asv_dev", "ood_en_asv")),
               "asv")
    for grp in ("asv_dev", "ood_en_asv"):
        rows = [dict(r, _slice=grp) for r in asv if (r.get("group") or "").strip() == grp]
        if rows:
            slices[grp] = _cap(rows, args.eval_bona_cap, args.eval_spoof_cap, args.seed)
    return slices


def fit_lr(X: np.ndarray, y: np.ndarray, feature_names: list[str],
           note: str, version: str) -> dict:
    """StandardScaler + class-balanced LogisticRegression -> portable artifact.

    Same shape as prosody_lr_v1.json so ProsodyScorer loads either without a code
    change: feature_names, scaler_mean, scaler_scale, coef, intercept,
    bonafide_ranges (raw feature space, for describe.py), metadata.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
    clf.fit(scaler.transform(X), y)
    bona = X[y == 0]
    ranges = {n: {"p5": float(np.percentile(bona[:, j], 5)),
                  "p50": float(np.percentile(bona[:, j], 50)),
                  "p95": float(np.percentile(bona[:, j], 95))}
              for j, n in enumerate(feature_names)}
    return {
        "version": version,
        "feature_names": list(feature_names),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coef": clf.coef_[0].astype(float).tolist(),
        "intercept": float(clf.intercept_[0]),
        "bonafide_ranges": ranges,
        "metadata": {
            "n_train": int(len(y)), "n_bonafide": int((y == 0).sum()),
            "n_spoof": int((y == 1).sum()),
            "train_accuracy": float(clf.score(scaler.transform(X), y)),
            "note": note,
        },
    }


def score_with(artifact: dict, X: np.ndarray) -> np.ndarray:
    """Vectorised ProsodyScorer.logit_from_vector — higher logit = more spoof."""
    mean = np.asarray(artifact["scaler_mean"], float)
    scale = np.asarray(artifact["scaler_scale"], float)
    scale = np.where(scale == 0, 1.0, scale)
    return ((X.astype(np.float64) - mean) / scale) @ np.asarray(artifact["coef"], float) \
        + float(artifact["intercept"])


def slice_metrics(logits: np.ndarray, y: np.ndarray, seed: int = 0) -> dict | None:
    from prosody_metrics import auc, eer, eer_ci

    b, s = logits[y == 0], logits[y == 1]
    if len(b) < 15 or len(s) < 15:
        return None
    lo, hi = eer_ci(b, s, seed=seed)
    return {"eer": eer(b, s), "eer_ci90": [lo, hi], "auc": auc(b, s),
            "n_bona": int(len(b)), "n_spoof": int(len(s)),
            "mean_bona": float(b.mean()), "mean_spoof": float(s.mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manip", default=MANIP)
    ap.add_argument("--neural", default=NEURAL)
    ap.add_argument("--multi", default=MULTI)
    ap.add_argument("--baseline", default="artifacts/prosody_lr_v1.json")
    ap.add_argument("--cache", default="artifacts/feature_store.npz")
    ap.add_argument("--out-json", default="artifacts/manip_experiment.json")
    ap.add_argument("--train-bona-cap", type=int, default=2500,
                    help="neural bonafide sampled for the combined variant (0 = all)")
    ap.add_argument("--train-spoof-cap", type=int, default=40,
                    help="neural spoof per generator for the combined variant")
    ap.add_argument("--eval-bona-cap", type=int, default=500, help="0 = every bonafide")
    ap.add_argument("--eval-spoof-cap", type=int, default=150,
                    help="spoof per generator on the neural eval slices")
    ap.add_argument("--manip-frac", type=float, default=1.0,
                    help="subsample manipulation slices by source stem (smoke tests)")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (cpu-2, max 8)")
    ap.add_argument("--plan-only", action="store_true",
                    help="print slice sizes and the extraction cost, then stop")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from features.feature_store import FeatureStore
    from features.prosody_features import FEATURE_NAMES

    slices = build_slices(args)
    print("slices:")
    for name, rows in slices.items():
        c = Counter(r["_y"] for r in rows)
        print(f"   {name:<20s} bonafide={c[0]:<6d} spoof={c[1]:<6d} total={len(rows)}")

    # one extraction pass over every path any slice needs
    all_paths = list(dict.fromkeys(r["path"] for rows in slices.values() for r in rows))
    print(f"\n[features] {len(all_paths)} unique clips")
    store = FeatureStore(args.cache)
    if args.plan_only:
        import os
        have = sum(1 for p in all_paths
                   if os.path.normcase(os.path.normpath(p)) in store._feats)
        n_w = args.workers or max(1, min(8, (os.cpu_count() or 4) - 2))
        print(f"[plan] {have} already cached, {len(all_paths) - have} to extract "
              f"(~{(len(all_paths) - have) * 1.1 / n_w / 60:.0f} min on {n_w} workers)")
        return 0
    X_all, ok_all = store.get_or_extract(all_paths, workers=args.workers or None)
    idx = {p: i for i, p in enumerate(all_paths)}

    def matrix(name: str):
        rows = [r for r in slices[name] if ok_all[idx[r["path"]]]]
        if len(rows) < len(slices[name]):
            print(f"   [{name}] dropped {len(slices[name]) - len(rows)} unreadable")
        X = (np.stack([X_all[idx[r["path"]]] for r in rows]) if rows
             else np.zeros((0, X_all.shape[1]), dtype=np.float32))
        return rows, X, np.array([r["_y"] for r in rows], dtype=np.int64)

    variants: dict[str, dict] = {}
    with open(args.baseline) as fh:
        variants["shipped"] = json.load(fh)

    tr_rows, Xtr, ytr = matrix("manip_train")
    variants["manip_only"] = fit_lr(
        Xtr, ytr, FEATURE_NAMES, version="prosody-lr-manip-v1",
        note="Higher logit = more spoof. Fitted on the PAIRED non-neural "
             "manipulation train split only (data_pipeline/manifests/manipulations.csv). "
             "A DSP-manipulation specialist: expect near-chance on neural TTS/VC.")

    nt_rows, Xnt, ynt = matrix("neural_train")
    variants["combined"] = fit_lr(
        np.concatenate([Xtr, Xnt]), np.concatenate([ytr, ynt]), FEATURE_NAMES,
        version="prosody-lr-combined-v1",
        note="Higher logit = more spoof. Fitted on the paired manipulation train "
             "split PLUS a per-generator-capped sample of the neural hybrid train "
             "split, so one artifact covers both attack classes.")

    return _report(args, slices, matrix, variants, FEATURE_NAMES)


EVAL_ORDER = ["manip_dev", "manip_eval_seen", "manip_eval_heldout",
              "neural_dev", "neural_eval", "neural_eval_ood", "asv_dev", "ood_en_asv"]


def _report(args, slices, matrix, variants, feature_names) -> int:
    results: dict[str, dict] = {v: {} for v in variants}
    cached = {name: matrix(name) for name in EVAL_ORDER if name in slices}

    print("\n" + "=" * 100)
    print("EER% / AUC per slice (lower EER, higher AUC = better; 50% / 0.500 = chance)")
    print("=" * 100)
    names = list(variants)
    print(f"  {'slice':<22s} {'n':>11s}   " + "".join(f"{v:>22s}" for v in names))
    for name in EVAL_ORDER:
        if name not in cached:
            continue
        rows, X, y = cached[name]
        cells = []
        for v in names:
            m = slice_metrics(score_with(variants[v], X), y, seed=args.seed)
            results[v][name] = m
            cells.append("        too small     " if m is None else
                         f"   {m['eer'] * 100:5.2f}%  {m['auc']:.3f}   ")
        print(f"  {name:<22s} {f'{int((y == 0).sum())}/{int((y == 1).sum())}':>11s}   "
              + "".join(cells))
    print("\n  manip_eval_heldout = unseen speakers AND three transform families never "
          "trained on.\n  That row, not manip_dev, is the number to quote.")

    # ---- per-transform breakdown: which manipulations are actually caught -----
    print("\n" + "=" * 100)
    print("PER-TRANSFORM on the manipulation eval slices (each vs its OWN paired bonafide)")
    print("=" * 100)
    per_tf: dict[str, dict] = {v: {} for v in variants}
    for name in ("manip_eval_seen", "manip_eval_heldout"):
        if name not in cached:
            continue
        rows, X, y = cached[name]
        tfs = sorted({r["_gen"] for r in rows if r["_y"] == 1})
        print(f"\n  [{name}]")
        for v in names:
            logits = score_with(variants[v], X)
            bona = logits[y == 0]
            line, skipped = [], []
            for tf in tfs:
                sel = np.array([r["_gen"] == tf and r["_y"] == 1 for r in rows])
                m = slice_metrics(np.concatenate([bona, logits[sel]]),
                                  np.concatenate([np.zeros(len(bona), int),
                                                  np.ones(int(sel.sum()), int)]),
                                  seed=args.seed)
                if m:
                    per_tf[v][f"{name}/{tf}"] = m
                    line.append(f"{tf.replace('manip_', ''):>22s} {m['eer'] * 100:5.1f}%")
                else:
                    skipped.append(tf.replace("manip_", ""))
            print(f"    {v:<12s} " + "  ".join(line) +
                  (f"   [too few clips: {', '.join(skipped)}]" if skipped else ""))
    return _report_tail(args, cached, variants, results, per_tf, feature_names)


def _report_tail(args, cached, variants, results, per_tf, feature_names) -> int:
    """Fixed-threshold operating points, the cross-corpus offset check, weights."""
    from prosody_metrics import detection_at, threshold_at_far

    print("\n" + "=" * 100)
    print("FIXED THRESHOLD (set on manip_dev bonafide, then applied unchanged elsewhere)")
    print("=" * 100)
    print("  A per-slice EER moves its own threshold per slice and so hides score")
    print("  offsets between corpora. Deployment gets ONE threshold, so: pick it for a")
    print("  target false-alarm rate on manip_dev bonafide, then measure what it does.")
    ops: dict[str, dict] = {v: {} for v in variants}
    for v in variants:
        if "manip_dev" not in cached:
            break
        _, Xd, yd = cached["manip_dev"]
        bona_dev = score_with(variants[v], Xd)[yd == 0]
        print(f"\n  [{v}]")
        for far in (0.01, 0.05, 0.10):
            thr = threshold_at_far(bona_dev, far)
            cells = []
            for name, (rows, X, y) in cached.items():
                logits = score_with(variants[v], X)
                if name.startswith("manip"):
                    val = detection_at(logits[y == 1], thr)
                    cells.append(f"{name.replace('manip_', 'det ')}={val * 100:5.1f}%")
                else:
                    val = detection_at(logits[y == 0], thr)  # false alarms
                    cells.append(f"FA {name.replace('neural_', '')}={val * 100:5.1f}%")
                ops[v][f"far{int(far * 100)}/{name}"] = float(val)
            print(f"    FAR={far * 100:4.1f}%  thr={thr:+7.3f}   " + "  ".join(cells))
    print("\n  'det' = manipulations detected. 'FA' = clean human voices on OTHER corpora")
    print("  wrongly flagged at that same threshold. A large FA gap means the score is")
    print("  corpus-dependent and one global threshold is not usable as-is.")

    print("\n" + "=" * 100)
    print("FEATURE WEIGHTS (standardized; + drives the score toward spoof)")
    print("=" * 100)
    for v, art in variants.items():
        coef = np.asarray(art["coef"], float)
        order = np.argsort(-np.abs(coef))[:6]
        print(f"  {v:<12s} " + "  ".join(f"{feature_names[j]}={coef[j]:+.2f}"
                                         for j in order))

    out_dir = Path(args.baseline).parent
    written = []
    for v in ("manip_only", "combined"):
        if v not in variants:
            continue
        p = out_dir / f"prosody_lr_{v}_v1.json"
        p.write_text(json.dumps(variants[v], indent=2) + "\n", encoding="utf-8")
        written.append(str(p))
        print(f"[OK] wrote {p}")
    res = Path(args.out_json)
    res.write_text(json.dumps(
        {"slices": results, "per_transform": per_tf, "operating_points": ops,
         "artifacts_written": written, "baseline": args.baseline,
         "manifests": {"manip": args.manip, "neural": args.neural, "multi": args.multi}},
        indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
