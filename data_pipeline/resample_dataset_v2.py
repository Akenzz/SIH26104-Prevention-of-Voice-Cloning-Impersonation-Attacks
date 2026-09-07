#!/usr/bin/env python3
"""
resample_dataset_v2.py
──────────────────────
Builds a balanced, generator-fair training corpus from the full raw dataset.

Key fix vs v1: instead of capping total spoof, we cap EACH generator to
CAP_PER_GENERATOR files so every TTS engine (Omni, Qwen3, Chatterbox...) is
equally represented regardless of how many files it has.

Usage:
    python3 resample_dataset_v2.py [--cap N] [--seed S] [--out-dir PATH] [--dry-run]

Outputs:
    <out_dir>/train.csv   - 70% of each generator
    <out_dir>/val.csv     - 15% of each generator
    <out_dir>/test.csv    - 15% remaining

CSV columns: path, label, generator, language, split
    label    : "spoof" or "bonafide"
    generator: TTS model name or "bonafide_la" / "bonafide_gv" etc.
"""

import argparse
import csv
import os
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_ROOT   = Path("/media/akenzz/D/DataSet")
MLAAD_FAKE     = DATASET_ROOT / "MLAAD/MLAAD/fake"
LA_PROTO_DIR   = DATASET_ROOT / "LA/ASVspoof2019_LA_cm_protocols"
LA_AUDIO_DIRS  = {
    "train": DATASET_ROOT / "LA/ASVspoof2019_LA_train/flac",
    "dev":   DATASET_ROOT / "LA/ASVspoof2019_LA_dev/flac",
    "eval":  DATASET_ROOT / "LA/ASVspoof2019_LA_eval/flac",
}
GV_AUDIO_DIR   = DATASET_ROOT / "GV_Train_100h/Audio"
HINDI_AUDIO    = DATASET_ROOT / "Hindi/v1/train"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 1 - TRAIN_FRAC - VAL_FRAC  (remainder)


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_audio(folder: Path) -> list[Path]:
    """Return all audio files under folder (non-recursive one level)."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )


def list_audio_recursive(folder: Path) -> list[Path]:
    """Return all audio files under folder (recursive)."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )


def split_files(files: list, seed: int) -> tuple[list, list, list]:
    """Randomly split into (train, val, test) using TRAIN/VAL fracs."""
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def make_row(path: Path, label: str, generator: str, language: str) -> dict:
    return {
        "path":      str(path),
        "label":     label,
        "generator": generator,
        "language":  language,
    }


# ── Spoof: MLAAD ──────────────────────────────────────────────────────────────

def collect_mlaad_spoof(cap: int, seed: int) -> dict[str, list[dict]]:
    """
    Returns {'train': [row,...], 'val': [...], 'test': [...]}.
    Each generator is independently capped and split.
    """
    rng = random.Random(seed)
    splits: dict[str, list] = {"train": [], "val": [], "test": []}

    langs = sorted(d.name for d in MLAAD_FAKE.iterdir() if d.is_dir())
    for lang in langs:
        lang_dir = MLAAD_FAKE / lang
        generators = sorted(d.name for d in lang_dir.iterdir() if d.is_dir())
        for gen_name in generators:
            gen_dir = lang_dir / gen_name
            files = list_audio(gen_dir)
            if not files:
                print(f"  [SKIP] {lang}/{gen_name}: 0 audio files")
                continue

            # Cap to at most `cap` files
            if len(files) > cap:
                rng.seed(seed)  # deterministic per generator
                files = rng.sample(files, cap)
            else:
                files = list(files)

            tr, va, te = split_files(files, seed=hash((lang, gen_name, seed)) & 0xFFFFFF)
            for split_name, split_files_ in [("train", tr), ("val", va), ("test", te)]:
                for p in split_files_:
                    splits[split_name].append(
                        make_row(p, "spoof", gen_name, lang)
                    )

            print(f"  {lang}/{gen_name}: {len(files)} sampled  "
                  f"(train={len(tr)}, val={len(va)}, test={len(te)})")

    return splits


# ── Bonafide: ASVspoof 2019 LA ────────────────────────────────────────────────

def collect_la_bonafide(seed: int) -> list[dict]:
    """Read protocol files, extract bonafide entries, return rows (no split yet)."""
    rows = []
    proto_files = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    for split_name, proto_file in proto_files.items():
        proto_path = LA_PROTO_DIR / proto_file
        audio_dir  = LA_AUDIO_DIRS[split_name]
        if not proto_path.exists():
            print(f"  [WARN] LA protocol missing: {proto_path}")
            continue
        with open(proto_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # format: speaker  utt_id  -  -  label
                utt_id = parts[1]
                label  = parts[4]
                if label != "bonafide":
                    continue
                audio_path = audio_dir / f"{utt_id}.flac"
                if audio_path.exists():
                    rows.append(make_row(audio_path, "bonafide", "bonafide_la", "en"))
    print(f"  LA bonafide total: {len(rows)}")
    return rows


# ── Bonafide: GramVaani (Hindi) ───────────────────────────────────────────────

def collect_gv_bonafide() -> list[dict]:
    files = list_audio_recursive(GV_AUDIO_DIR)
    print(f"  GV bonafide total: {len(files)}")
    return [make_row(p, "bonafide", "bonafide_gv", "hi") for p in files]


def collect_hindi_bonafide() -> list[dict]:
    files = list_audio_recursive(HINDI_AUDIO)
    print(f"  Hindi bonafide total: {len(files)}")
    return [make_row(p, "bonafide", "bonafide_hi", "hi") for p in files]


def collect_gramvaani_extra() -> list[dict]:
    extra_dirs = [
        DATASET_ROOT / "GV_Eval_3h/Audio",
        DATASET_ROOT / "Gramvaani_1000hrData_Part5",
    ]
    rows = []
    for d in extra_dirs:
        files = list_audio_recursive(d)
        print(f"  GV extra {d.name}: {len(files)} files")
        rows.extend(make_row(p, "bonafide", "bonafide_gv_extra", "hi") for p in files)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap",     type=int, default=300, help="Max files per generator (default 300)")
    parser.add_argument("--seed",    type=int, default=42,  help="Random seed (default 42)")
    parser.add_argument("--out-dir", type=str,
                        default="/media/akenzz/D/DataSet_processed/v2",
                        help="Output directory for CSVs")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, do not write files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Dataset Resampler v2")
    print(f" cap={args.cap} seed={args.seed} out={args.out_dir}")
    print(f"{'='*60}\n")

    # ── Collect spoof ─────────────────────────────────────────────────────────
    print("[1/3] Collecting MLAAD spoof audio...")
    spoof_splits = collect_mlaad_spoof(cap=args.cap, seed=args.seed)
    n_spoof = sum(len(v) for v in spoof_splits.values())
    print(f"\n  Total spoof: {n_spoof}  "
          f"(train={len(spoof_splits['train'])}, "
          f"val={len(spoof_splits['val'])}, "
          f"test={len(spoof_splits['test'])})\n")

    # ── Collect bonafide ──────────────────────────────────────────────────────
    print("[2/3] Collecting bonafide audio...")
    all_bonafide = []
    all_bonafide += collect_la_bonafide(seed=args.seed)
    all_bonafide += collect_gv_bonafide()
    all_bonafide += collect_hindi_bonafide()
    all_bonafide += collect_gramvaani_extra()
    print(f"\n  Total bonafide available: {len(all_bonafide)}")

    # Balance bonafide to match spoof total
    rng = random.Random(args.seed)
    if len(all_bonafide) > n_spoof:
        all_bonafide = rng.sample(all_bonafide, n_spoof)
        print(f"  Bonafide sampled down to {n_spoof} to match spoof count")
    else:
        print(f"  [WARN] Bonafide ({len(all_bonafide)}) < spoof ({n_spoof}), using all bonafide")

    # Split bonafide
    b_train, b_val, b_test = split_files(all_bonafide, seed=args.seed)
    bonafide_splits = {"train": b_train, "val": b_val, "test": b_test}
    print(f"  Bonafide split: train={len(b_train)}, val={len(b_val)}, test={len(b_test)}\n")

    # ── Merge and write CSVs ──────────────────────────────────────────────────
    print("[3/3] Writing CSVs...")
    fieldnames = ["path", "label", "generator", "language", "split"]
    total_rows = 0

    for split_name in ["train", "val", "test"]:
        rows = []
        for r in spoof_splits[split_name]:
            r["split"] = split_name
            rows.append(r)
        for r in bonafide_splits[split_name]:
            r["split"] = split_name
            rows.append(r)

        rng.shuffle(rows)

        if not args.dry_run:
            out_path = out_dir / f"{split_name}.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Wrote {split_name}.csv: {len(rows)} rows → {out_path}")
        else:
            print(f"  [DRY RUN] {split_name}: {len(rows)} rows")

        total_rows += len(rows)

    print(f"\n{'='*60}")
    print(f" Done! Total rows: {total_rows}")
    n_sp = len(spoof_splits['train']) + len(spoof_splits['val']) + len(spoof_splits['test'])
    n_bo = len(bonafide_splits['train']) + len(bonafide_splits['val']) + len(bonafide_splits['test'])
    print(f" Spoof: {n_sp}  |  Bonafide: {n_bo}  |  Balance: {n_sp/(n_sp+n_bo)*100:.1f}% / {n_bo/(n_sp+n_bo)*100:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
