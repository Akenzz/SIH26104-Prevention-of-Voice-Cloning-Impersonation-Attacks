"""
prosody-detector/evaluation/diagnose_by_generator.py
===================================================
Does the prosody expert's (weak) discrimination concentrate on NON-NEURAL
attacks?

RESULTS.md reports prosody at 43.94% EER pooled over eval_ood, i.e. near chance
-- but every slice there except the ASVspoof one contains only neural TTS/VC,
where prosody is reproduced well. The one slice with classic signal-processing
attacks (`ood_en_asv`) is also prosody's best at 36.40% EER / AUC 0.659. That
could be the hypothesis showing through, or it could be a corpus effect.

This script separates the two by scoring every spoof generator individually
against the bonafide pool of its own group, then aggregating by how the waveform
was actually produced:

  non_neural_waveform : no neural net in waveform generation at all
                        (unit-selection concatenation, spectral/waveform
                        filtering, Griffin-Lim phase reconstruction)
  dsp_vocoder         : classic parametric vocoder (WORLD, STRAIGHT, Vocaine)
                        driven by a neural acoustic model
  neural_waveform     : neural vocoder / end-to-end (WaveNet, WaveRNN, HiFi-GAN,
                        modern hosted TTS)
  unknown             : not classifiable (e.g. in-the-wild, generator unlabelled)

The ASVspoof A01-A19 assignments are the attack descriptions from the ASVspoof
2019 database paper (arXiv:1911.01601), not something inferred from the audio.

Read-only: no new audio, no artifact is rewritten. Features land in a shared
path-keyed store so later experiments reuse them.

Run from inside prosody-detector/:
  python evaluation/diagnose_by_generator.py
  python evaluation/diagnose_by_generator.py --max-per-generator 200 --workers 12
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "lfcc-detector"))

# --------------------------------------------------------------------------- #
# Attack families. ASVspoof A01-A19 follow the ASVspoof 2019 database paper
# (arXiv:1911.01601, Table 2) -- classified by WAVEFORM GENERATION, which is what
# leaves (or fails to leave) a neural-vocoder fingerprint for the spectral
# experts. The acoustic model may still be a neural net in the dsp_vocoder rows.
# --------------------------------------------------------------------------- #
ATTACK_FAMILY = {
    "A01": "neural_waveform",       # TTS, WaveNet
    "A02": "dsp_vocoder",           # TTS, WORLD
    "A03": "dsp_vocoder",           # TTS, WORLD
    "A04": "non_neural_waveform",   # TTS, waveform concatenation (unit selection)
    "A05": "dsp_vocoder",           # VC, VAE acoustic model + WORLD
    "A06": "non_neural_waveform",   # VC, spectral filtering (transfer function)
    "A07": "dsp_vocoder",           # TTS, WORLD + GAN post-filter
    "A08": "neural_waveform",       # TTS, neural source-filter
    "A09": "dsp_vocoder",           # TTS, Vocaine
    "A10": "neural_waveform",       # TTS, Tacotron2 + WaveRNN
    "A11": "non_neural_waveform",   # TTS, Tacotron2 + Griffin-Lim
    "A12": "neural_waveform",       # TTS, WaveNet
    "A13": "non_neural_waveform",   # TTS+VC, waveform filtering
    "A14": "dsp_vocoder",           # TTS+VC, STRAIGHT
    "A15": "neural_waveform",       # TTS+VC, WaveNet
    "A16": "non_neural_waveform",   # TTS, waveform concatenation
    "A17": "non_neural_waveform",   # VC, waveform filtering
    "A18": "dsp_vocoder",           # VC, parametric vocoder
    "A19": "non_neural_waveform",   # VC, spectral filtering (GMM-UBM)
    "griffin_lim": "non_neural_waveform",
}

# Substrings that mark a locally generated signal-processing manipulation, so
# manipulation sets added later are grouped correctly without editing this table.
_MANIPULATION_HINTS = ("pitchshift", "pitch_shift", "formant", "changer", "praat",
                       "resample_shift", "phasevocoder", "dsp_manip")
# Anything hosted / diffusion / GAN / flow based. Kept explicit rather than
# "everything else is neural" so a new generator shows up as unknown, loudly.
_NEURAL_HINTS = (
    "xtts", "vits", "tacotron", "wavenet", "wavernn", "hifigan", "hifi-gan", "melgan",
    "waveglow", "bark", "f5", "indicf5", "edge-tts", "elevenlabs", "kokoro", "chatterbox",
    "sovits", "rvc", "voxtral", "higgs", "qwen", "llasa", "matcha", "megatts", "outetts",
    "fishtts", "cartesia", "magpie", "resemble", "marvis", "kani", "neutts", "inworld",
    "styletts", "parler", "speecht5", "yourtts", "openvoice", "cosyvoice", "seedtts",
    "fast_pitch", "fastpitch", "overflow", "glow", "diff", "vall", "spark", "orpheus",
    "svaratts", "coqui", "piper", "mms-tts", "sonic", "gpt-sovits", "tortoise",
)


def attack_family(generator_id: str) -> str:
    """Family for one generator_id. Unrecognised -> 'unknown' (never guessed)."""
    g = (generator_id or "").strip()
    if g in ATTACK_FAMILY:
        return ATTACK_FAMILY[g]
    low = g.lower().replace(" ", "").replace("-", "").replace("_", "")
    if any(h.replace("-", "").replace("_", "") in low for h in _MANIPULATION_HINTS):
        return "non_neural_waveform"
    if any(h.replace("-", "").replace("_", "") in low for h in _NEURAL_HINTS):
        return "neural_waveform"
    return "unknown"


# --------------------------------------------------------------------------- #
# Metrics — shared with the trainer via the top-level prosody_metrics module
# (see its docstring for why it cannot live under evaluation/).
# --------------------------------------------------------------------------- #
from prosody_metrics import auc as _auc  # noqa: E402
from prosody_metrics import eer as _eer  # noqa: E402
from prosody_metrics import eer_ci as _eer_ci_impl  # noqa: E402


def _eer_ci(bona: np.ndarray, spoof: np.ndarray, n_boot: int = 300,
            seed: int = 0) -> tuple[float, float]:
    return _eer_ci_impl(bona, spoof, n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------- #
# Manifest selection
# --------------------------------------------------------------------------- #
def select_rows(manifest: Path, splits: list[str], max_per_generator: int,
                max_bona_per_group: int, seed: int):
    """Return (rows, per-group index) with spoof capped per generator.

    Bonafide is capped per group and spoof per generator, because the point is to
    compare generators to each other on an equal footing — an unbalanced pool
    would let one large generator dominate its group's aggregate.
    """
    with manifest.open(newline="", encoding="utf-8-sig") as fh:
        all_rows = [r for r in csv.DictReader(fh) if r.get("split") in splits]

    by_group_bona: dict[str, list] = defaultdict(list)
    by_group_gen: dict[tuple[str, str], list] = defaultdict(list)
    for r in all_rows:
        group = (r.get("group") or r.get("split") or "all").strip()
        if r.get("label", "").strip().lower() == "bonafide":
            by_group_bona[group].append(r)
        else:
            by_group_gen[(group, (r.get("generator_id") or "?").strip())].append(r)

    rng = random.Random(seed)
    kept: list[dict] = []
    for group, rows in by_group_bona.items():
        rng.shuffle(rows)
        for r in rows[:max_bona_per_group]:
            r["_group"], r["_gen"], r["_y"] = group, "bonafide", 0
            kept.append(r)
    for (group, gen), rows in by_group_gen.items():
        rng.shuffle(rows)
        for r in rows[:max_per_generator]:
            r["_group"], r["_gen"], r["_y"] = group, gen, 1
            kept.append(r)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default="artifacts/prosody_lr_v1.json")
    ap.add_argument("--manifest", default="../data_pipeline/manifests/multicorpus_final.csv")
    ap.add_argument("--splits", default="eval_ood,dev",
                    help="comma list; dev included for ASVspoof A01-A06")
    ap.add_argument("--cache", default="artifacts/feature_store.npz")
    ap.add_argument("--output-json", default="artifacts/prosody_by_generator.json")
    ap.add_argument("--max-per-generator", type=int, default=150)
    ap.add_argument("--max-bona-per-group", type=int, default=500)
    ap.add_argument("--min-spoof", type=int, default=40,
                    help="skip generators with fewer scored clips than this")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (cpu-2)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from expert.prosody_expert import ProsodyScorer
    from features.feature_store import FeatureStore

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    rows = select_rows(Path(args.manifest), splits, args.max_per_generator,
                       args.max_bona_per_group, args.seed)
    print(f"selected {len(rows)} clips from splits={splits} "
          f"(<={args.max_per_generator}/generator, <={args.max_bona_per_group} bonafide/group)")

    store = FeatureStore(args.cache)
    X, ok = store.get_or_extract([r["path"] for r in rows],
                                 workers=args.workers or None)
    if not ok.any():
        print("no clips could be read — check the manifest paths")
        return 1
    if (~ok).any():
        print(f"dropping {int((~ok).sum())} unreadable clip(s)")

    scorer = ProsodyScorer(args.artifact)
    logits = np.array([scorer.logit_from_vector(X[i]) if ok[i] else np.nan
                       for i in range(len(rows))])
    return _report(rows, logits, ok, args, X)


def _report(rows, logits, ok, args, X) -> int:
    from features.prosody_features import FEATURE_NAMES

    groups = sorted({r["_group"] for r in rows})
    bona_by_group = {
        g: logits[[i for i, r in enumerate(rows)
                   if r["_group"] == g and r["_y"] == 0 and ok[i]]]
        for g in groups
    }
    per_gen: list[dict] = []

    print("\n" + "=" * 92)
    print("PER-GENERATOR (each generator vs the bonafide pool of its OWN group)")
    print("=" * 92)
    for g in groups:
        bona = bona_by_group[g]
        gens = sorted({r["_gen"] for r in rows if r["_group"] == g and r["_y"] == 1})
        print(f"\n  [{g}]  bonafide={len(bona)}")
        if len(bona) < 20:
            print("     too few bonafide clips in this group to score against - skipped")
            continue
        entries = []
        for gen in gens:
            idx = [i for i, r in enumerate(rows)
                   if r["_group"] == g and r["_gen"] == gen and ok[i]]
            spoof = logits[idx]
            if len(spoof) < args.min_spoof:
                continue
            fam = attack_family(gen)
            e, a = _eer(bona, spoof), _auc(bona, spoof)
            lo, hi = _eer_ci(bona, spoof, seed=args.seed)
            entries.append((a, gen, fam, e, lo, hi, len(spoof), idx))
        for a, gen, fam, e, lo, hi, n, idx in sorted(entries, reverse=True):
            mark = "*" if fam == "non_neural_waveform" else (
                "+" if fam == "dsp_vocoder" else " ")
            print(f"   {mark} {gen:<34s} {fam:<20s} EER={e * 100:6.2f}% "
                  f"[{lo * 100:5.1f},{hi * 100:5.1f}]  AUC={a:.3f}  n={n}")
            per_gen.append({"group": g, "generator": gen, "family": fam,
                            "eer": e, "eer_ci90": [lo, hi], "auc": a, "n_spoof": n,
                            "n_bona": int(len(bona))})
    print("\n   * = non-neural waveform generation   + = classic DSP vocoder")

    # ---- family macro-average. AUC is rank-based, so averaging per-generator
    # AUCs (each already scored against its own group's bonafide) avoids pooling
    # logits across corpora with different score offsets.
    print("\n" + "=" * 92)
    print("BY ATTACK FAMILY  (macro-average over generators; 0.500 AUC = chance)")
    print("=" * 92)
    fam_summary = {}
    for fam in ("non_neural_waveform", "dsp_vocoder", "neural_waveform", "unknown"):
        sel = [d for d in per_gen if d["family"] == fam]
        if not sel:
            continue
        aucs = np.array([d["auc"] for d in sel])
        eers = np.array([d["eer"] for d in sel])
        fam_summary[fam] = {"n_generators": len(sel), "mean_auc": float(aucs.mean()),
                            "mean_eer": float(eers.mean()), "min_auc": float(aucs.min()),
                            "max_auc": float(aucs.max()),
                            "n_spoof": int(sum(d["n_spoof"] for d in sel))}
        print(f"  {fam:<22s} generators={len(sel):<3d} clips={fam_summary[fam]['n_spoof']:<5d} "
              f"mean AUC={aucs.mean():.3f} (min {aucs.min():.3f} / max {aucs.max():.3f})  "
              f"mean EER={eers.mean() * 100:5.2f}%")

    # ---- which features actually move, per family (Cohen's d vs all bonafide).
    print("\n" + "=" * 92)
    print("FEATURE SEPARATION vs bonafide  (Cohen's d, + = higher than human)")
    print("=" * 92)
    bona_idx = [i for i, r in enumerate(rows) if r["_y"] == 0 and ok[i]]
    B = X[bona_idx]
    fam_feats = {}
    for fam in ("non_neural_waveform", "dsp_vocoder", "neural_waveform"):
        idx = [i for i, r in enumerate(rows)
               if r["_y"] == 1 and ok[i] and attack_family(r["_gen"]) == fam]
        if len(idx) < 50:
            continue
        S = X[idx]
        pooled = np.sqrt((B.var(0) + S.var(0)) / 2.0) + 1e-9
        d = (S.mean(0) - B.mean(0)) / pooled
        fam_feats[fam] = {n: float(v) for n, v in zip(FEATURE_NAMES, d)}
        top = np.argsort(-np.abs(d))[:5]
        print(f"  {fam:<22s} n={len(idx):<5d} " +
              "  ".join(f"{FEATURE_NAMES[j]}={d[j]:+.2f}" for j in top))

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_generator": per_gen, "by_family": fam_summary,
                               "feature_cohens_d": fam_feats,
                               "artifact": args.artifact, "splits": args.splits},
                              indent=2) + "\n", encoding="utf-8")
    print(f"\n[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
