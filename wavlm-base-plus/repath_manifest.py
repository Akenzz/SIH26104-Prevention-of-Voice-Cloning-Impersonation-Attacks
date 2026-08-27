"""
repath_manifest.py — Fix Windows paths in the existing manifests so they
point at the actual audio files on this Linux machine.

The manifests in data_pipeline/manifests/ were generated on Windows and contain
paths like:
    D:\\DatasetSIH\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_1138215.flac

This script rewrites them to Linux paths like:
    /path/to/LA/ASVspoof2019_LA_train/flac/LA_T_1138215.flac

It also:
  - Renames the 'eval' split value → 'test'  (our SpeechDataset expects
    'train', 'dev', 'test' — the data_pipeline schema used 'eval' instead).
  - Merges the three per-split CSVs into one combined manifest that
    expert1/train.py + expert1/evaluate.py can consume directly.

Usage:
    python expert1/repath_manifest.py --dataset-root /path/to/LA
    
    # Example (replace with wherever you extracted the Kaggle zip):
    python expert1/repath_manifest.py --dataset-root /home/akenzz/DatasetSIH/LA
"""

import argparse
import os
import pandas as pd
from pathlib import Path

# Where the per-split manifests are
MANIFEST_DIR   = Path(__file__).resolve().parent.parent / "data_pipeline" / "manifests"
OUT_DIR        = Path(__file__).resolve().parent / "data"   # expert1/data/

# Windows path segment that maps to the dataset root on the original machine
WINDOWS_ROOT   = "D:\\DatasetSIH\\LA"  # everything before this is stripped

SPLIT_FILES = {
    "train" : MANIFEST_DIR / "asvspoof19_train.csv",
    "dev"   : MANIFEST_DIR / "asvspoof19_dev.csv",
    "test"  : MANIFEST_DIR / "asvspoof19_eval.csv",   # 'eval' → renamed to 'test'
}

OUTPUT_CSV = OUT_DIR / "asvspoof_manifest.csv"


def repath(old_path: str, dataset_root: Path) -> str:
    """
    Convert a Windows-style absolute path to a Linux path under dataset_root.
    
    Strategy: strip everything up to and including the Windows root prefix,
    then join to dataset_root, replacing backslashes with forward slashes.
    
    e.g.:
        D:\\DatasetSIH\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_1.flac
        → <dataset_root>/ASVspoof2019_LA_train/flac/LA_T_1.flac
    """
    # Normalise separator differences
    p = old_path.replace("\\", "/")
    # Strip the Windows prefix (case-insensitive)
    win_root_norm = WINDOWS_ROOT.replace("\\", "/")
    idx = p.lower().find(win_root_norm.lower())
    if idx != -1:
        # Everything after the root prefix
        relative = p[idx + len(win_root_norm):].lstrip("/")
    else:
        # Fallback: just take the filename and try to reconstruct
        relative = "/".join(p.split("/")[-3:])   # e.g. ASVspoof.../flac/LA_T_1.flac
    
    return str(dataset_root / relative)


def main(dataset_root: Path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    frames = []
    for split_name, csv_path in SPLIT_FILES.items():
        if not csv_path.exists():
            print(f"[WARN] Manifest not found, skipping: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path)
        
        # Fix paths
        df["path"] = df["path"].apply(lambda p: repath(p, dataset_root))
        
        # Fix split name: 'eval' → 'test'
        df["split"] = df["split"].replace({"eval": "test"})
        # Force the correct split name regardless of what's in the CSV
        df["split"] = split_name
        
        n_bonafide = (df["label"] == "bonafide").sum()
        n_spoof    = (df["label"] == "spoof").sum()
        print(f"  {split_name:5s}: {len(df):,} rows  ({n_bonafide:,} bonafide, {n_spoof:,} spoof)")
        
        # Quick sanity-check: does the first audio file actually exist?
        first_path = df["path"].iloc[0]
        if os.path.exists(first_path):
            print(f"         ✓ First file exists: {first_path}")
        else:
            print(f"         ✗ First file NOT found: {first_path}")
            print(f"           Check that --dataset-root is correct.")
        
        frames.append(df)
    
    if not frames:
        print("[FAIL] No manifests processed. Nothing to write.")
        return
    
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n[OK] Combined manifest written to: {OUTPUT_CSV}")
    print(f"     Total rows : {len(combined):,}")
    print(f"\nNow run training with:")
    print(f"  python -m expert1.train --manifest expert1/data/asvspoof_manifest.csv --epochs 10")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repath Windows manifest paths to Linux and merge into one CSV"
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Linux path to the ASVspoof 2019 LA root directory (contains "
             "ASVspoof2019_LA_train/, ASVspoof2019_LA_dev/, ASVspoof2019_LA_eval/)"
    )
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root).resolve()
    print(f"[repath] Dataset root : {dataset_root}")
    print(f"[repath] Output CSV   : {OUTPUT_CSV}\n")
    
    main(dataset_root)
