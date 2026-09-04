"""
Verify a manipulation manifest measures what it claims.
=======================================================
`generate_manipulations.py` checks bookkeeping (class parity, no withheld
transform in train, no duplicate paths). It cannot check that the DSP actually
happened: a Praat call that silently returned its input would pass every one of
those checks and would poison training with mislabelled clips.

So measure the pairs directly. For each spoof row, load it and its paired
bonafide and report, as ratios spoof/bonafide:

  f0    median F0                    - the pitch-shift family should move this
  cog   spectral centre of gravity   - the formant family should move this
  std   F0 standard deviation        - the monotone transform should crush this
  diff  rms(spoof - bona)/rms(bona)  - 0 only if the clip was never altered

`cog` is used instead of tracked F1/F2 deliberately. Praat's Burg tracker,
averaged over a whole clip, is dominated by unvoiced frames and is not even
monotonic in the true shift: on a real clip, formant ratio 0.60 measured x1.00
and 0.85 measured x1.17, while CoG tracked the request to within 1%.

A second, cheaper pass then scans EVERY pair for a loudness shortcut
(rms spoof/bona in dB). The generator matches peaks, not RMS, so this is the one
remaining way the label could be readable without hearing the transform at all.

  python data_pipeline/verify_manipulations.py --manifest data_pipeline/manifests/manipulations.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

SR = 16000
_FLOOR, _CEIL = 75.0, 600.0

# Predicted signature of each transform. "." = expected unchanged (ratio ~1.0).
EXPECTED = {
    "manip_praat_pitch_up":     "f0 1.18-1.35  cog .          std .",
    "manip_praat_pitch_down":   "f0 0.72-0.85  cog .          std .",
    "manip_praat_formant_up":   "f0 .          cog 1.10-1.22  std .",
    "manip_praat_formant_down": "f0 .          cog 0.82-0.91  std .",
    "manip_praat_monotone":     "f0 .          cog .          std 0.15-0.40",
    "manip_praat_gender_swap":  "f0 and cog both move, direction depends on source sex",
    # +-3/4 semitones with a +-0.4 semitone dither, so 2^((3-0.4)/12)..2^((4+0.4)/12)
    "manip_pv_pitch_shift":     "f0 0.78-0.86 or 1.16-1.29   cog follows f0 partially",
    "manip_resample_speed":     "f0 0.86-1.16 (inverse of the rate) and cog with it",
    "manip_praat_time_stretch": "f0 .          cog .          std .   (only timing moves)",
}


def _measure(path: str):
    """(median F0, spectral centre of gravity, F0 std, waveform) for one clip."""
    import parselmouth
    import soundfile as sf
    from parselmouth.praat import call

    y, sr = sf.read(path, dtype="float64", always_2d=False)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    snd = parselmouth.Sound(y, sampling_frequency=int(sr))
    pitch = call(snd, "To Pitch", 0.0, _FLOOR, _CEIL)
    med = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
    sd = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    cog = call(call(snd, "To Spectrum", "yes"), "Get centre of gravity", 2.0)
    if med is None or not np.isfinite(med) or med <= 0:
        return None
    return (float(med), float(cog), float(sd) if sd and np.isfinite(sd) else np.nan, y)


def _pairs(manifest: Path):
    """spoof path -> paired bonafide path, matched on the shared source stem."""
    with manifest.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    bona = {Path(r["path"]).stem.replace("__paired", ""): r["path"]
            for r in rows if r["label"] == "bonafide"}
    by_tf: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["label"] != "spoof":
            continue
        stem = Path(r["path"]).stem
        key = stem[: stem.rindex("__")] if "__" in stem else stem
        if key in bona:
            by_tf[r["generator_id"]].append((r["path"], bona[key]))
    return rows, by_tf


def _level_scan(by_tf: dict, workers: int):
    """Fast pass over EVERY pair: does loudness separate the classes?

    Peaks are matched in the generator, but PSOLA and the phase vocoder both
    change RMS at a fixed peak, so peak parity does not imply level parity. If
    rms(spoof)/rms(bona) sat consistently off 1.0, a detector could score well on
    gain alone and every EER measured on this corpus would be worthless. Uses
    soundfile only -- no Praat -- so all 7.4k pairs are affordable rather than the
    25/transform the pitch measurements are capped at.

    Reported as the median ratio in dB plus the fraction of pairs beyond 1 dB. The
    honest target is a median near 0 dB; a systematic offset here is a finding, not
    a rounding error.
    """
    from concurrent.futures import ProcessPoolExecutor

    print("\nlevel parity over ALL pairs  (rms spoof/bona; peaks are matched at "
          "generation, rms is not)")
    jobs = [(tf, sp, bo) for tf in sorted(by_tf) for sp, bo in by_tf[tf]]
    out: dict[str, list] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for tf, r in zip((j[0] for j in jobs), pool.map(_rms_ratio, jobs, chunksize=64)):
            if r is not None:
                out[tf].append(r)
    worst = 0.0
    for tf in sorted(out):
        db = 20.0 * np.log10(np.array(out[tf]))
        frac = float(np.mean(np.abs(db) > 1.0))
        worst = max(worst, abs(float(np.median(db))))
        print(f"  {tf:<28s} n={len(db):<5d} median {np.median(db):+5.2f} dB   "
              f"p05 {np.percentile(db, 5):+5.2f}  p95 {np.percentile(db, 95):+5.2f}   "
              f"|>1dB| {100 * frac:4.1f}%")
    print(f"  -> largest median offset {worst:.2f} dB "
          + ("(no usable level shortcut)" if worst < 0.5 else
             "(LEVEL SHORTCUT: a detector can score on gain alone)"))
    return worst


def _rms_ratio(job):
    """rms(spoof)/rms(bona) over the overlapping region of one pair."""
    import soundfile as sf

    _, sp, bo = job
    try:
        a, _ = sf.read(sp, dtype="float32", always_2d=False)
        b, _ = sf.read(bo, dtype="float32", always_2d=False)
    except Exception:
        return None
    n = min(np.size(a), np.size(b))
    if n == 0:
        return None
    ra = float(np.sqrt(np.mean(np.asarray(a).reshape(-1)[:n] ** 2)))
    rb = float(np.sqrt(np.mean(np.asarray(b).reshape(-1)[:n] ** 2)))
    return None if rb <= 0 or ra <= 0 else ra / rb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data_pipeline/manifests/manipulations.csv")
    ap.add_argument("--per-transform", type=int, default=25,
                    help="pairs to measure per transform (Praat is slow)")
    ap.add_argument("--workers", type=int, default=8, help="for the level scan only")
    ap.add_argument("--skip-level-scan", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows, by_tf = _pairs(Path(args.manifest))
    print(f"{len(rows)} rows; {sum(len(v) for v in by_tf.values())} spoof clips paired "
          f"to their source\n")
    unchanged_total = 0
    for tf in sorted(by_tf):
        pairs = by_tf[tf]
        random.Random(args.seed).shuffle(pairs)
        r_f0, r_cog, r_std, r_diff = [], [], [], []
        for sp, bo in pairs[: args.per_transform]:
            a, b = _measure(sp), _measure(bo)
            if a is None or b is None:
                continue
            r_f0.append(a[0] / b[0])
            r_cog.append(a[1] / b[1])
            if np.isfinite(a[2]) and np.isfinite(b[2]) and b[2] > 0:
                r_std.append(a[2] / b[2])
            n = min(a[3].size, b[3].size)
            den = float(np.sqrt(np.mean(b[3][:n] ** 2))) + 1e-12
            r_diff.append(float(np.sqrt(np.mean((a[3][:n] - b[3][:n]) ** 2))) / den)
        if not r_f0:
            print(f"  {tf:<28s} no measurable pairs")
            continue
        n_same = sum(1 for d in r_diff if d < 1e-3)
        unchanged_total += n_same
        std = np.median(r_std) if r_std else float("nan")
        print(f"  {tf:<28s} n={len(r_f0):<3d} f0 x{np.median(r_f0):.3f}  "
              f"cog x{np.median(r_cog):.3f}  std x{std:.3f}  diff {np.median(r_diff):.3f}"
              + (f"   <-- {n_same} UNCHANGED" if n_same else ""))
        print(f"  {'':<28s} expect: {EXPECTED.get(tf, '?')}")
    print("\n" + ("[OK] every measured clip differs from its source"
                  if not unchanged_total else
                  f"[WARN] {unchanged_total} clip(s) are identical to their source"))
    level_ok = True
    if not args.skip_level_scan:
        level_ok = _level_scan(by_tf, args.workers) < 0.5
    return 0 if (not unchanged_total and level_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
