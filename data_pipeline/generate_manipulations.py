"""
Generate the non-neural manipulation corpus (Fix 2's missing attack class).
==========================================================================
Every spoof clip in this project's manifests comes from a neural TTS/VC system,
so nothing on disk tests the attack a cheap voice changer actually performs:
pitch/formant shift and crude resynthesis applied to a REAL recording. No public
corpus covers it either. It does not need to — a voice changer is a deterministic
DSP chain, so the attack can be reproduced exactly from bonafide audio we already
have, using Praat (parselmouth) and librosa, offline.

Design decisions that matter:

* PAIRED. Each manipulated clip is emitted alongside its own source clip as
  bonafide, so both classes contain the same speakers, channels and sentences.
  The only difference between a bonafide row and a spoof row is the transform
  itself, which removes every corpus/speaker/channel shortcut by construction.
  (`data_pipeline/check_leakage.py` conventions; same reasoning as the
  within-language pairing used for the hybrid manifest.)

* SPLIT-FAITHFUL. A manipulated clip inherits its source clip's split, so the
  speaker disjointness already established in the source manifest is preserved
  and no manipulated version of a training clip can appear in eval.

* UNSEEN-MANIPULATION EVAL. Three transform families are withheld from
  train/dev entirely and appear only in eval, tagged `manip_heldout_eval`.
  Reporting on those separately is the honest generalization number, exactly as
  unseen-generator slices are reported separately for the neural experts.

Nothing is overwritten: audio goes to --out-dir, rows to a new manifest.

  python data_pipeline/generate_manipulations.py --n-train 6000 --n-dev 800 --n-eval 800

NOTE ON THE CORPUS ALREADY ON DISK (2026-09-02, 7,447 pairs). It was built when
level matching equalised PEAK amplitude. `verify_manipulations.py` later measured
that this leaves the manipulated side 0.5-1.8 dB quieter in RMS, because PSOLA
raises crest factor. The prosody experiment run against it is unaffected -- its
features are gain-invariant to 8e-7 relative, verified directly by
`prosody-detector/evaluation/check_gain_invariance.py` -- but an LFCC/hybrid model
would read that offset as a free label bit. The default here is now
--level-match rms; the LFCC retrain needs a regenerated corpus, and
--level-match peak reproduces the old one exactly if a comparison is wanted.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SR = 16000
_PITCH_FLOOR, _PITCH_CEIL = 75.0, 600.0


# --------------------------------------------------------------------------- #
# Transforms. Each takes (y float32 @16k, rng) and returns float32 @16k, or
# raises/returns None when the clip cannot carry the transform (e.g. Praat finds
# no pitch). A skipped clip is dropped — never emitted unmodified with a spoof
# label, which would be a mislabelled training example.
# --------------------------------------------------------------------------- #
def _praat_sound(y: np.ndarray):
    import parselmouth

    return parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=SR)


def _median_f0(snd) -> float | None:
    from parselmouth.praat import call

    pitch = call(snd, "To Pitch", 0.0, _PITCH_FLOOR, _PITCH_CEIL)
    med = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
    if med is None or not np.isfinite(med) or med <= 0:
        return None
    return float(med)


def _change_gender(y: np.ndarray, formant_ratio: float, new_median: float,
                   range_factor: float = 1.0, duration_factor: float = 1.0):
    """Praat 'Change gender' — the canonical voice-changer transform.

    Args mirror Praat's own: formant shift ratio (vocal-tract length), new pitch
    median in Hz (0 = leave alone), pitch range factor (1 = leave alone),
    duration factor. Uses PSOLA resynthesis, so micro-variation is re-imposed
    rather than preserved — which is the artifact we expect prosody to catch.
    """
    from parselmouth.praat import call

    snd = _praat_sound(y)
    out = call(snd, "Change gender", _PITCH_FLOOR, _PITCH_CEIL,
               float(formant_ratio), float(new_median), float(range_factor),
               float(duration_factor))
    return np.asarray(out.values, dtype=np.float32).reshape(-1)


def t_pitch_up(y, rng):
    snd = _praat_sound(y)
    med = _median_f0(snd)
    return None if med is None else _change_gender(y, 1.0, med * rng.uniform(1.18, 1.35))


def t_pitch_down(y, rng):
    snd = _praat_sound(y)
    med = _median_f0(snd)
    return None if med is None else _change_gender(y, 1.0, med * rng.uniform(0.72, 0.85))


def t_formant_up(y, rng):
    return _change_gender(y, rng.uniform(1.10, 1.22), 0.0)


def t_formant_down(y, rng):
    return _change_gender(y, rng.uniform(0.82, 0.91), 0.0)


def t_monotone(y, rng):
    """Flatten intonation without touching median pitch — the 'robotic' setting."""
    return _change_gender(y, 1.0, 0.0, range_factor=rng.uniform(0.15, 0.40))


def t_gender_swap(y, rng):
    """Full masculinise/feminise: vocal-tract length AND pitch median together."""
    snd = _praat_sound(y)
    med = _median_f0(snd)
    if med is None:
        return None
    if med < 165.0:  # low-pitched source -> shift upward
        return _change_gender(y, rng.uniform(1.15, 1.28), rng.uniform(180.0, 220.0))
    return _change_gender(y, rng.uniform(0.78, 0.88), rng.uniform(100.0, 125.0))


# ---- withheld from train/dev: unseen-manipulation eval only ---------------- #
def t_pv_pitch_shift(y, rng):
    """librosa phase-vocoder pitch shift — a different resynthesis than PSOLA."""
    import librosa

    steps = rng.choice([-4.0, -3.0, 3.0, 4.0]) + rng.uniform(-0.4, 0.4)
    return librosa.effects.pitch_shift(y=np.asarray(y, dtype=np.float32), sr=SR,
                                       n_steps=float(steps)).astype(np.float32)


def t_resample_speed(y, rng):
    """Naive playback-rate change: pitch and tempo move together (crudest changer)."""
    from math import gcd

    from scipy.signal import resample_poly

    r = rng.choice([0.86, 0.90, 1.11, 1.16])
    num, den = int(round(r * 1000)), 1000
    g = gcd(num, den)
    return resample_poly(np.asarray(y, dtype=np.float32), den // g,
                         num // g).astype(np.float32)


def t_time_stretch(y, rng):
    """Praat duration factor only: tempo changes, pitch does not."""
    return _change_gender(y, 1.0, 0.0, duration_factor=rng.choice([0.82, 0.88, 1.14, 1.22]))


SEEN_TRANSFORMS = {
    "manip_praat_pitch_up": t_pitch_up,
    "manip_praat_pitch_down": t_pitch_down,
    "manip_praat_formant_up": t_formant_up,
    "manip_praat_formant_down": t_formant_down,
    "manip_praat_monotone": t_monotone,
    "manip_praat_gender_swap": t_gender_swap,
}
# Withheld from train/dev. Different resynthesis maths from the PSOLA family
# above, so eval on these measures unseen-manipulation generalization.
HELDOUT_TRANSFORMS = {
    "manip_pv_pitch_shift": t_pv_pitch_shift,
    "manip_resample_speed": t_resample_speed,
    "manip_praat_time_stretch": t_time_stretch,
}
ALL_TRANSFORMS = {**SEEN_TRANSFORMS, **HELDOUT_TRANSFORMS}


def _match_level(y: np.ndarray, out: np.ndarray, mode: str = "rms") -> np.ndarray:
    """Scale `out` so loudness cannot separate the classes.

    mode="peak" was the original choice and it is NOT sufficient. PSOLA and the
    phase vocoder both raise crest factor, so equalising peaks leaves the
    manipulated side 0.5-1.8 dB quieter in RMS -- measured, per transform, by
    `verify_manipulations.py`. The 13 prosody features are provably blind to that
    (`prosody-detector/evaluation/check_gain_invariance.py`: worst-case drift 8e-7
    relative under +-6 dB, because every feature is a frequency, a duration, a
    ratio, or gated on the clip's own loudest frame). An LFCC/log-magnitude model
    is NOT blind to it: a gain change is a constant offset in its input, i.e. a
    free label bit. So mode="rms" is the default.

    RMS matching can demand a gain that clips, in which case fall back to the
    largest gain that does not, and let the residual offset stand for that clip
    rather than distorting the waveform.
    """
    rms = lambda v: float(np.sqrt(np.mean(np.asarray(v, dtype=np.float64) ** 2)))
    p_in, p_out = float(np.max(np.abs(y))), float(np.max(np.abs(out)))
    if p_out <= 0:
        return out
    if mode == "peak":
        return out * min(p_in / p_out, 0.99 / p_out)
    r_in, r_out = rms(y), rms(out)
    if r_out <= 0:
        return out * min(p_in / p_out, 0.99 / p_out)
    return out * min(r_in / r_out, 0.99 / p_out)


def _make_one(job: dict):
    """Worker: read source, apply one transform, write BOTH sides at equal length.

    The paired bonafide side is re-written rather than referenced in place so the
    two classes have identical length distributions. Without that, a transform
    that shortens the clip (time stretch, resample) would make spoof rows more
    likely to hit the trainer's tile-pad branch, and periodic repetition would
    become a label shortcut.
    """
    import soundfile as sf

    # librosa's phase vocoder emits deprecation warnings for its own internal
    # kwargs; at 7k clips that is thousands of identical lines over the real log.
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    src = job["src"]
    try:
        y, sr = sf.read(src, dtype="float32", always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if int(sr) != SR:
            return {"ok": False, "src": src, "why": f"unexpected sample rate {sr}"}
        rng = random.Random(job["seed"])
        out = ALL_TRANSFORMS[job["transform"]](y, rng)
        if out is None:
            return {"ok": False, "src": src, "why": "no pitch found for transform"}
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(out)):
            return {"ok": False, "src": src, "why": "transform produced non-finite audio"}
        n = int(min(y.size, out.size))
        if n < SR:  # under 1 s of usable audio after the transform
            return {"ok": False, "src": src, "why": f"too short after transform ({n} samples)"}
        y, out = y[:n], out[:n]
        out = _match_level(y, out, job.get("level_match", "rms"))
        sf.write(job["dst_spoof"], out, SR, subtype="PCM_16")
        sf.write(job["dst_bona"], y, SR, subtype="PCM_16")
    except Exception as exc:
        return {"ok": False, "src": src, "why": f"{type(exc).__name__}: {exc}"[:160]}
    return {"ok": True, "src": src, "transform": job["transform"],
            "dst_spoof": job["dst_spoof"], "dst_bona": job["dst_bona"],
            "duration_s": round(n / SR, 3)}


COLUMNS = ["path", "label", "split", "source_dataset", "speaker_id", "utterance_id",
           "generator_id", "language", "codec", "duration_s", "license", "consent",
           "held_out", "role", "group"]


def plan_jobs(manifest: Path, out_dir: Path, counts: dict[str, int], seed: int,
              make_dirs: bool = True, level_match: str = "rms"):
    """Pick bonafide sources per split and assign one transform to each."""
    with manifest.open(newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("label", "").strip().lower() == "bonafide"
                and r.get("split") in counts]
    by_split: dict[str, list] = {s: [] for s in counts}
    seen_paths: set[str] = set()
    for r in rows:
        p = r.get("path", "").strip()
        if p and p not in seen_paths:
            seen_paths.add(p)
            by_split[r["split"]].append(r)

    jobs: list[dict] = []
    for split, want in counts.items():
        pool = by_split[split]
        random.Random(seed).shuffle(pool)
        # eval also carries the withheld transforms, and carries them at double
        # weight: the unseen-manipulation EER is the headline number, so it gets
        # the sample size (~2x per transform), while seen-transform eval only has
        # to corroborate dev. train/dev never see the withheld family at all.
        names = list(SEEN_TRANSFORMS)
        if split == "eval":
            names += list(HELDOUT_TRANSFORMS) * 2
        take = pool if want < 0 else pool[: min(want, len(pool))]
        if want > len(pool):
            print(f"  [{split}] only {len(pool)} bonafide sources available (wanted {want})")
        for i, row in enumerate(take):
            name = names[i % len(names)]
            stem = Path(row["path"]).stem
            spoof_dir = out_dir / split / name
            bona_dir = out_dir / split / "paired_bonafide"
            if make_dirs:
                spoof_dir.mkdir(parents=True, exist_ok=True)
                bona_dir.mkdir(parents=True, exist_ok=True)
            jobs.append({
                "src": row["path"], "transform": name, "row": row, "split": split,
                "dst_spoof": str(spoof_dir / f"{stem}__{name}.wav"),
                "dst_bona": str(bona_dir / f"{stem}__paired.wav"),
                "seed": seed * 1_000_003 + i,
                "level_match": level_match,
            })
    return jobs


def _rows_for(res: dict, job: dict) -> list[dict]:
    """Two manifest rows — the manipulated clip and its paired bonafide source.

    Every column except path/label/generator_id/utterance_id is IDENTICAL between
    the pair, inherited from the source row: same speaker, corpus, language,
    licence, split and duration. That is deliberate. If `source_dataset` (or any
    other field) marked the spoof side, anyone slicing the manifest by metadata
    would get a class-correlated split, and the leakage checker could not tell a
    real shortcut from this construction.
    """
    src = job["row"]
    name, split = job["transform"], job["split"]
    heldout = name in HELDOUT_TRANSFORMS
    # `group` names the eval slice; both classes of a pair share it so the slice
    # is self-contained — scoring never has to borrow bonafide from another
    # corpus, which would add a channel offset on top of the transform effect.
    group = f"manip_{'heldout' if heldout else 'seen'}_{split}"
    base = {c: (src.get(c) or "") for c in COLUMNS}
    base.update({"split": split, "codec": "pcm_16k", "role": "matched",
                 "duration_s": f"{res['duration_s']:.3f}", "group": group})
    uid = base["utterance_id"] or Path(job["src"]).stem
    spoof = {**base, "path": res["dst_spoof"], "label": "spoof",
             "generator_id": name, "utterance_id": f"{uid}__{name}",
             "held_out": "yes" if heldout else "no"}
    bona = {**base, "path": res["dst_bona"], "label": "bonafide",
            "generator_id": "none", "utterance_id": f"{uid}__paired",
            "held_out": "no"}
    return [spoof, bona]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data_pipeline/manifests/hybrid_vad_chunks.csv",
                    help="bonafide source; each clip's split is inherited from here")
    ap.add_argument("--out-dir", default=r"E:\DatasetSIH\processed_manipulations")
    ap.add_argument("--out-manifest", default="data_pipeline/manifests/manipulations.csv")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-dev", type=int, default=-1, help="-1 = every bonafide source")
    ap.add_argument("--n-eval", type=int, default=-1, help="-1 = every bonafide source")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (cpu-2, capped at 8)")
    ap.add_argument("--level-match", choices=("rms", "peak"), default="rms",
                    help="rms (default) leaves no loudness cue; peak reproduces the "
                         "original 2026-09-02 corpus, which is 0.5-1.8 dB quieter on "
                         "the spoof side")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and report only — writes no audio and no manifest")
    args = ap.parse_args()

    manifest, out_dir = Path(args.manifest), Path(args.out_dir)
    if not manifest.exists():
        print(f"manifest not found: {manifest}")
        return 1
    counts = {k: v for k, v in (("train", args.n_train), ("dev", args.n_dev),
                                ("eval", args.n_eval)) if v != 0}
    print(f"[plan] {manifest}  ->  {out_dir}")
    jobs = plan_jobs(manifest, out_dir, counts, args.seed, make_dirs=not args.dry_run,
                     level_match=args.level_match)
    if not jobs:
        print("no bonafide sources matched the requested splits")
        return 1

    per_split, per_tf = Counter(j["split"] for j in jobs), Counter(j["transform"] for j in jobs)
    print(f"[plan] {len(jobs)} manipulations -> {2 * len(jobs)} manifest rows "
          f"({'/'.join(f'{s}:{per_split[s]}' for s in counts)})")
    for name in ALL_TRANSFORMS:
        tag = "  <- withheld from train/dev" if name in HELDOUT_TRANSFORMS else ""
        print(f"         {name:<28s} {per_tf[name]:>6d}{tag}")
    if args.dry_run:
        print("[dry-run] nothing written")
        return 0
    return _run(jobs, args, counts)


def _run(jobs: list[dict], args, counts: dict[str, int]) -> int:
    """Generate the audio, then write the manifest and the parity self-checks."""
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    by_src = {j["src"]: j for j in jobs}       # plan_jobs de-duplicates paths globally
    n_workers = args.workers or max(1, min(8, (os.cpu_count() or 4) - 2))
    rows: list[dict] = []
    failed: list[dict] = []
    seen: set[str] = set()
    print(f"[gen] {len(jobs)} clips on {n_workers} workers", flush=True)
    while True:
        todo = [j for j in jobs if j["src"] not in seen]
        if not todo:
            break
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for res in pool.map(_make_one, todo, chunksize=8):
                    seen.add(res["src"])
                    if res["ok"]:
                        rows.extend(_rows_for(res, by_src[res["src"]]))
                    else:
                        failed.append(res)
                    if len(seen) % 500 == 0:
                        print(f"[gen]   {len(seen)}/{len(jobs)}", flush=True)
        except BrokenProcessPool:
            # Same failure mode as the prosody feature store: each worker holds
            # Praat + librosa, so the ceiling is RAM. Halve and resume; work
            # already written to disk is not repeated.
            if n_workers == 1:
                raise
            n_workers = max(1, n_workers // 2)
            print(f"[gen] worker pool died (likely RAM) — resuming "
                  f"{len(jobs) - len(seen)} clips on {n_workers} workers", flush=True)
            continue
        break

    out_man = Path(args.out_manifest)
    out_man.parent.mkdir(parents=True, exist_ok=True)
    with out_man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] wrote {out_man}  ({len(rows)} rows from {len(jobs) - len(failed)} pairs, "
          f"{len(failed)} sources skipped)")
    if failed:
        for why, n in Counter(f["why"].split(":")[0] for f in failed).most_common(6):
            print(f"       {n:>5d}  {why}")
    _self_check(rows, counts)
    return 0


def _self_check(rows: list[dict], counts: dict[str, int]) -> None:
    """Verify the anti-shortcut properties actually hold in what was written.

    Cheap, and worth printing on every run: the entire value of this corpus is
    that label is decorrelated from length, speaker, corpus and channel. A silent
    regression there yields a model that scores well and has learnt nothing.
    """
    print("\n[check] per-split parity - the two classes must match exactly")
    for split in counts:
        sel = [r for r in rows if r["split"] == split]
        b, s = (np.array([float(r["duration_s"]) for r in sel if r["label"] == lab])
                for lab in ("bonafide", "spoof"))
        if not len(b) or not len(s):
            continue
        ok = len(b) == len(s) and abs(float(b.mean()) - float(s.mean())) < 1e-6
        print(f"   [{split:<5s}] n={len(b)}/{len(s)}  mean={b.mean():.3f}/{s.mean():.3f}s"
              f"  >=4s {100 * (b >= 4).mean():.1f}%/{100 * (s >= 4).mean():.1f}%"
              f"{'' if ok else '   <-- MISMATCH'}")

    leaked = [r for r in rows
              if r["split"] != "eval" and r["generator_id"] in HELDOUT_TRANSFORMS]
    dup = len(rows) - len({r["path"] for r in rows})
    splits_per_spk: dict[str, set] = {}
    for r in rows:
        if r["speaker_id"]:
            splits_per_spk.setdefault(r["speaker_id"], set()).add(r["split"])
    cross = sum(1 for v in splits_per_spk.values() if len(v) > 1)
    print(f"[check] withheld transforms outside eval: {len(leaked)}  (must be 0)")
    print(f"[check] duplicate output paths:           {dup}  (must be 0)")
    print(f"[check] speakers spanning splits:         {cross}  "
          f"(inherited from the source manifest, not introduced here)")
    for split in counts:
        c = Counter(r["generator_id"] for r in rows
                    if r["split"] == split and r["label"] == "spoof")
        print(f"[check] [{split:<5s}] " +
              " ".join(f"{k.replace('manip_', '')}={v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    raise SystemExit(main())
