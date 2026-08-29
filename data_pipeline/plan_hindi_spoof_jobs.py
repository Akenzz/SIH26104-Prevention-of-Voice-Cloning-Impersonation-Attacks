"""
data_pipeline/plan_hindi_spoof_jobs.py
======================================
Turn 86k bonafide Kathbath-Hindi clips into a *small, balanced, leakage-free*
spoofing plan. This script GENERATES NO AUDIO. It only decides:

  1. Speaker-disjoint split of the 101 speakers -> train / dev / eval
     (a speaker's clips live in exactly ONE split, so bonafide + its future
     spoof + everything from that speaker never crosses a split boundary).
  2. Balanced, round-robin-capped sampling within each split (spreads thin
     across speakers first, so no single speaker dominates).
  3. Generator assignment:
       - train / dev : DISJOINT  -> each clip spoofed by ONE generator
                                     (IndicF5 / XTTS), ~50/50 within each speaker
       - eval        : OVERLAP   -> each clip spoofed by EVERY held-out generator
                                     (IndicF5, XTTS, RVC) so per-generator EER is
                                     apples-to-apples on identical content.
     RVC appears ONLY in eval  => it is the held-out / unseen generator.

Outputs (all under data_pipeline/):
  manifests/hindi_spoof_jobs.csv        -> the Colab work order (what to generate)
  manifests/hindi_bonafide_selected.csv -> the REAL half of the final dataset
  reports/hindi_spoof_plan_summary.json -> counts / speaker split / budgets
  manifests/hindi_refs_to_upload.txt    -> the ~7.6k wavs to zip up for Colab
  (optional) --stage-refs copies those wavs into <stage>/refs/<split>/<utt>.wav

Matched-pair note: every spoofed clip also exists as its own bonafide row, so
the detector cannot cheat on speaker/content/gender — which is also why we do
NOT force a 50/50 gender split (matched pairs neutralize gender as a shortcut).

Usage (defaults produce ~6000/1000/600 train/dev/eval + 3000 real-only):
  python data_pipeline/plan_hindi_spoof_jobs.py
  python data_pipeline/plan_hindi_spoof_jobs.py --stage-refs
"""

import os
import sys
import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
MANIFEST_DIR = repo_root / "data_pipeline" / "manifests"
REPORT_DIR = repo_root / "data_pipeline" / "reports"

DEFAULT_KATHBATH_ROOT = Path(os.environ.get("KATHBATH_ROOT", r"E:\DatasetSIH\kathbath"))
DEFAULT_WAV_DIR = DEFAULT_KATHBATH_ROOT / "hindi"
DEFAULT_STAGE = DEFAULT_KATHBATH_ROOT.parent / "hindi_spoof"

# generators used per split
TRAIN_GENERATORS = ["indicf5", "xtts"]      # disjoint across clips
EVAL_GENERATORS = ["indicf5", "xtts", "rvc"]  # overlap on same clips; rvc = held-out


def norm_gender(g) -> str:
    if g is None:
        return "u"
    s = str(g).strip().lower()
    return s[0] if s and s[0] in ("f", "m") else "u"


def allocate_speakers(spk_gender: dict, n_eval: int, n_dev: int, seed: int):
    """Assign speakers to eval/dev/train, disjoint, keeping both genders in each
    of eval and dev. Returns dict split -> set(speakers)."""
    rng = random.Random(seed)
    by_g = defaultdict(list)
    for spk, g in spk_gender.items():
        by_g[g].append(spk)
    for g in by_g:
        by_g[g].sort()          # determinism before shuffle
        rng.shuffle(by_g[g])

    genders = [g for g in ("f", "m", "u") if by_g.get(g)]
    total = sum(len(by_g[g]) for g in genders)

    def take(n_target):
        """Take n_target speakers, spread across genders by their proportion,
        with a floor of 1 per available gender when possible."""
        picked = []
        # proportional target per gender
        alloc = {}
        for g in genders:
            alloc[g] = max(1, round(n_target * len(by_g[g]) / total)) if len(by_g[g]) else 0
        # trim/pad to exactly n_target
        while sum(alloc.values()) > n_target:
            g = max(alloc, key=lambda k: alloc[k])
            alloc[g] -= 1
        while sum(alloc.values()) < n_target:
            g = max(genders, key=lambda k: len(by_g[k]))
            alloc[g] += 1
        for g in genders:
            for _ in range(alloc[g]):
                if by_g[g]:
                    picked.append(by_g[g].pop())
        return picked

    eval_spk = set(take(n_eval))
    dev_spk = set(take(n_dev))
    train_spk = set(s for g in genders for s in by_g[g])   # whatever remains
    return {"train": train_spk, "dev": dev_spk, "eval": eval_spk}


def round_robin_sample(spk_to_utts: dict, budget: int, max_per_speaker: int, seed: int):
    """Spread thin across speakers: pop one clip per speaker per round until the
    budget is met (or clips run out). Bounds any single speaker's contribution."""
    rng = random.Random(seed)
    queues = {}
    for spk, utts in spk_to_utts.items():
        u = list(utts)
        rng.shuffle(u)
        queues[spk] = u[:max_per_speaker] if max_per_speaker else u
    order = sorted(queues.keys())
    picked = []
    while len(picked) < budget:
        progressed = False
        for spk in order:
            if queues[spk]:
                picked.append((spk, queues[spk].pop()))
                progressed = True
                if len(picked) >= budget:
                    break
        if not progressed:
            break
    return picked


def main():
    ap = argparse.ArgumentParser(description="Plan a leakage-free multi-generator Hindi spoofing job set.")
    ap.add_argument("--transcripts", default=str(MANIFEST_DIR / "kathbath_hindi_transcripts.csv"))
    ap.add_argument("--wav-dir", default=str(DEFAULT_WAV_DIR))
    ap.add_argument("--train-budget", type=int, default=6000)
    ap.add_argument("--dev-budget", type=int, default=1000)
    ap.add_argument("--eval-budget", type=int, default=600, help="matched eval clips (each spoofed by every eval generator)")
    ap.add_argument("--eval-realonly", type=int, default=3000, help="extra bonafide-only eval clips for false-positive rate")
    ap.add_argument("--n-eval-speakers", type=int, default=15)
    ap.add_argument("--n-dev-speakers", type=int, default=15)
    ap.add_argument("--max-per-speaker", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--stage-refs", action="store_true", help="copy the ref wavs into <stage>/refs/ for upload")
    ap.add_argument("--stage-dir", default=str(DEFAULT_STAGE))
    ap.add_argument("--spoof-root", default="spoof", help="relative root recorded in out_relpath for generated audio")
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    if not wav_dir.is_dir():
        raise SystemExit(f"[FAIL] --wav-dir not found: {wav_dir}")

    # ---- load transcripts, keep only utts that actually have a wav on disk ----
    tx = pd.read_csv(args.transcripts, encoding="utf-8-sig", dtype={"utt_id": str, "speaker_id": str})
    wav_stems = {Path(n).stem for n in os.listdir(wav_dir) if n.lower().endswith(".wav")}
    tx = tx[tx["utt_id"].isin(wav_stems)].copy()
    tx["gender"] = tx["gender"].map(norm_gender)
    tx["duration_s"] = pd.to_numeric(tx["duration_s"], errors="coerce").fillna(0.0)
    print(f"[1/5] {len(tx):,} transcribed clips with audio on disk.")

    # speaker -> gender (majority), speaker -> utts
    spk_gender = {}
    spk_to_utts = defaultdict(list)
    dur = {}
    text = {}
    for r in tx.itertuples(index=False):
        spk_to_utts[r.speaker_id].append(r.utt_id)
        dur[r.utt_id] = float(r.duration_s)
        text[r.utt_id] = r.text
        spk_gender.setdefault(r.speaker_id, r.gender)
    nF = sum(1 for g in spk_gender.values() if g == "f")
    nM = sum(1 for g in spk_gender.values() if g == "m")
    print(f"      {len(spk_gender)} speakers ({nF} F / {nM} M / {len(spk_gender)-nF-nM} U)")

    # ---- speaker-disjoint split ----
    split_spk = allocate_speakers(spk_gender, args.n_eval_speakers, args.n_dev_speakers, args.seed)
    for s in ("train", "dev", "eval"):
        clips = sum(len(spk_to_utts[k]) for k in split_spk[s])
        print(f"      {s:5}: {len(split_spk[s]):>3} speakers, {clips:,} clips available")

    # ---- sample clips per split ----
    def utts_for(split):
        return {spk: spk_to_utts[spk] for spk in split_spk[split]}

    train_pick = round_robin_sample(utts_for("train"), args.train_budget, args.max_per_speaker, args.seed + 1)
    dev_pick   = round_robin_sample(utts_for("dev"),   args.dev_budget,   args.max_per_speaker, args.seed + 2)
    eval_total = round_robin_sample(utts_for("eval"),  args.eval_budget + args.eval_realonly,
                                    args.max_per_speaker * 20, args.seed + 3)  # eval: few speakers, allow more each
    eval_matched  = eval_total[:args.eval_budget]
    eval_realonly = eval_total[args.eval_budget:]
    print(f"[2/5] sampled  train={len(train_pick)}  dev={len(dev_pick)}  "
          f"eval_matched={len(eval_matched)}  eval_realonly={len(eval_realonly)}")

    # ---- build job rows + bonafide rows ----
    job_rows = []
    bona_rows = []
    refs_to_upload = []
    jid = 0

    def bona_row(spk, utt, split, role):
        return {
            "path": str(wav_dir / f"{utt}.wav"),
            "label": "bonafide", "split": split, "source_dataset": "Kathbath",
            "speaker_id": spk, "utterance_id": utt, "generator_id": "none",
            "language": "hi", "codec": "pcm_16k", "duration_s": round(dur.get(utt, 0.0), 3),
            "license": "CC-BY-4.0", "consent": "yes", "held_out": "no",
            "role": role, "text": text.get(utt, ""),
        }

    def add_job(utt, spk, gender, split, gen):
        nonlocal jid
        jid += 1
        job_rows.append({
            "job_id": f"j{jid:06d}",
            "utt_id": utt,
            "src_wav": str(wav_dir / f"{utt}.wav"),
            "text": text.get(utt, ""),
            "speaker_id": spk,
            "gender": gender,
            "split": split,
            "target_generator": gen,
            "out_relpath": f"{args.spoof_root}/{split}/{gen}/{utt}.wav",
        })

    # train / dev : disjoint generators, alternating within each speaker
    for split, picks, gens in (("train", train_pick, TRAIN_GENERATORS),
                               ("dev", dev_pick, TRAIN_GENERATORS)):
        per_spk_counter = defaultdict(int)
        for spk, utt in picks:
            bona_rows.append(bona_row(spk, utt, split, "matched"))
            refs_to_upload.append(str(wav_dir / f"{utt}.wav"))
            g = gens[per_spk_counter[spk] % len(gens)]   # 50/50 within speaker
            per_spk_counter[spk] += 1
            add_job(utt, spk, spk_gender.get(spk, "u"), split, g)

    # eval matched : overlap -> one job per eval generator on the SAME clip
    for spk, utt in eval_matched:
        bona_rows.append(bona_row(spk, utt, "eval", "matched"))
        refs_to_upload.append(str(wav_dir / f"{utt}.wav"))
        for g in EVAL_GENERATORS:
            add_job(utt, spk, spk_gender.get(spk, "u"), "eval", g)

    # eval real-only : bonafide only, no jobs (measures false-positive rate)
    for spk, utt in eval_realonly:
        bona_rows.append(bona_row(spk, utt, "eval", "realonly"))

    # ---- write artifacts ----
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    jobs_df = pd.DataFrame(job_rows)
    jobs_csv = MANIFEST_DIR / "hindi_spoof_jobs.csv"
    jobs_df.to_csv(jobs_csv, index=False, encoding="utf-8-sig")

    bona_df = pd.DataFrame(bona_rows)
    bona_csv = MANIFEST_DIR / "hindi_bonafide_selected.csv"
    bona_df.to_csv(bona_csv, index=False, encoding="utf-8-sig")

    refs = sorted(set(refs_to_upload))
    refs_txt = MANIFEST_DIR / "hindi_refs_to_upload.txt"
    refs_txt.write_text("\n".join(refs), encoding="utf-8")

    gen_counts = jobs_df["target_generator"].value_counts().to_dict() if len(jobs_df) else {}
    split_gen = (jobs_df.groupby(["split", "target_generator"]).size().unstack(fill_value=0).to_dict()
                 if len(jobs_df) else {})
    summary = {
        "seed": args.seed,
        "speakers_total": len(spk_gender),
        "speakers_per_split": {s: sorted(split_spk[s]) for s in ("train", "dev", "eval")},
        "budgets": {"train": args.train_budget, "dev": args.dev_budget,
                    "eval_matched": args.eval_budget, "eval_realonly": args.eval_realonly},
        "bonafide_selected": int(len(bona_df)),
        "bonafide_by_split_role": ({f"{s}/{r}": int(n) for (s, r), n in bona_df.groupby(["split", "role"]).size().items()}
                                   if len(bona_df) else {}),
        "spoof_jobs_total": int(len(jobs_df)),
        "spoof_jobs_by_generator": {str(k): int(v) for k, v in gen_counts.items()},
        "spoof_jobs_by_split_generator": {str(k): v for k, v in split_gen.items()},
        "refs_to_upload": len(refs),
        "train_generators": TRAIN_GENERATORS,
        "eval_generators": EVAL_GENERATORS,
        "held_out_generator": "rvc",
    }
    (REPORT_DIR / "hindi_spoof_plan_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"[3/5] wrote {jobs_csv.name} ({len(jobs_df):,} jobs), "
          f"{bona_csv.name} ({len(bona_df):,} bonafide rows)")
    print(f"[4/5] spoof jobs by generator: {gen_counts}")
    print(f"      refs to upload: {len(refs):,}  -> {refs_txt.name}")

    # ---- optional staging copy ----
    if args.stage_refs:
        stage = Path(args.stage_dir) / "refs"
        n = 0
        # map utt -> split for folder layout
        utt_split = {}
        for r in bona_rows:
            if r["role"] == "matched":
                utt_split[r["utterance_id"]] = r["split"]
        for src in refs:
            utt = Path(src).stem
            dst = stage / utt_split.get(utt, "train") / f"{utt}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                    n += 1
                except Exception as e:
                    print(f"      [WARN] copy failed {src}: {e}")
        print(f"[5/5] staged {n:,} ref wavs under {stage}")
    else:
        print(f"[5/5] (skipped staging; re-run with --stage-refs to copy the {len(refs):,} ref wavs)")


if __name__ == "__main__":
    main()
