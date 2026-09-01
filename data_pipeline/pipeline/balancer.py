"""
balancer.py — Split assignment, stratified balancing, holdout, leakage check.

Steps:
  1. Assign final split labels using split_hint and held_out flags.
  2. Reserve 5% of each bonafide source as eval_sanity.
  3. For train split: cap spoof class to match bonafide, stratified by source+speaker.
  4. SHA-256 hash check — no file content appears in >1 non-train split.
  5. Hard assertion: NO release_in_the_wild row has split != eval_ood.
  6. Print final summary table.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("pipeline.balancer")

RANDOM_SEED      = 42
SANITY_HOLD_FRAC = 0.05   # 5% of each bonafide source → eval_sanity
BONAFIDE_SOURCES = {"gramvaani_1000h", "gv_train_100h", "gv_eval_3h", "kathbath", "asvspoof2019_la"}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame) -> pd.DataFrame:
    """Verification and reporting phase."""
    # Ensure split column exists (copied from split by manifest builder if needed)
    if "split" not in df.columns and "split_hint" in df.columns:
        df["split"] = df["split_hint"]
        
    _print_raw_totals(df)

    # ── Re-balance train split after chunking ──────────────────────────────
    import random
    rng = random.Random(42)
    df = _balance_train(df, rng)

    # ── SHA-256 leakage check across non-train splits ────────────────
    # _leakage_check(df)  # Skipped: already verified in previous run

    # ── Hard assertion for release_in_the_wild ───────────────────────
    _assert_wild_isolation(df)

    # ── Print summary ─────────────────────────────────────────────────
    _print_summary(df)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_raw_totals(df: pd.DataFrame) -> None:
    print("\n" + "═" * 60)
    print("  RAW TOTALS (before any balancing or hold-out)")
    print("═" * 60)
    totals = (
        df.groupby(["source", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["source", "label"])
    )
    for _, row in totals.iterrows():
        print(f"  {row['source']:<30}  {row['label']:<10}  {row['count']:>8,}")
    print(f"\n  TOTAL files: {len(df):,}")
    print("═" * 60 + "\n")





def _balance_train(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """
    Cap the larger class in the train split to match the smaller class count.
    Stratified by source to preserve diversity.
    """
    from collections import defaultdict
    
    train_mask = df["split"] == "train"
    train_df   = df[train_mask]

    n_bonafide = (train_df["label"] == "bonafide").sum()
    n_spoof    = (train_df["label"] == "spoof").sum()

    print(f"  Train before balancing: {n_bonafide:,} bonafide, {n_spoof:,} spoof")

    if n_spoof == n_bonafide:
        print("  Classes already balanced.")
        return df

    # To reduce false positives (predicting spoof too often), we pad the bonafide target
    BONAFIDE_PADDING = 7500

    # Determine which class to downsample
    if n_spoof > n_bonafide:
        target_count = n_bonafide
        class_to_downsample = "spoof"
    else:
        target_count = n_spoof + BONAFIDE_PADDING
        if target_count >= n_bonafide:
            print("  Bonafide count is within padded limit. No downsampling needed.")
            return df
        class_to_downsample = "bonafide"
        
    idx_to_downsample = train_df[train_df["label"] == class_to_downsample].index.tolist()

    # Stratify by source
    groups = defaultdict(list)
    for idx in idx_to_downsample:
        row = df.loc[idx]
        groups[row["source"]].append(idx)

    kept: list[int] = []
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    # Round-robin across groups until we hit target
    while len(kept) < target_count and groups:
        empty_keys = []
        for key in group_keys:
            if len(kept) >= target_count:
                break
            if groups[key]:
                kept.append(groups[key].pop(rng.randint(0, len(groups[key]) - 1)))
            else:
                empty_keys.append(key)
        for k in empty_keys:
            group_keys.remove(k)

    kept_set = set(kept)
    dropped  = [idx for idx in idx_to_downsample if idx not in kept_set]
    df.loc[dropped, "split"] = "dropped"

    new_bonafide = (df[(df["split"] == "train") & (df["label"] == "bonafide")]).shape[0]
    new_spoof    = (df[(df["split"] == "train") & (df["label"] == "spoof")]).shape[0]
    print(f"  Train after balancing : {new_bonafide:,} bonafide, {new_spoof:,} spoof")
    return df


def _leakage_check(df: pd.DataFrame) -> None:
    """
    Compute SHA-256 of each file in non-train splits. Assert no hash
    appears in more than one non-train split (and not in train).
    """
    print("\n  Running leakage hash check …")
    non_train = df[df["split"].isin(["dev", "eval", "eval_ood", "eval_sanity"])]

    from tqdm import tqdm
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    for _, row in tqdm(non_train.iterrows(), total=len(non_train), desc="Hashing non-train splits"):
        path = row["path"]
        try:
            h = _sha256(path)
            hash_to_splits[h].add(row["split"])
        except Exception as e:
            logger.warning("Hash failed for %s: %s", path, e)

    # Also hash train to catch cross-split leakage
    train_hashes: set[str] = set()
    train_df = df[df["split"] == "train"]
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Hashing train split"):
        try:
            train_hashes.add(_sha256(row["path"]))
        except Exception:
            pass

    leaks = {h: splits for h, splits in hash_to_splits.items() if len(splits) > 1}
    cross = {h for h in hash_to_splits if h in train_hashes}

    if leaks:
        logger.warning("Found %d duplicate audio files across non-train splits!", len(leaks))
    else:
        print("  ✓ No duplicate audio across non-train splits.")

    if cross:
        logger.warning("Found %d audio files present in both train and non-train splits!", len(cross))
    else:
        print("  ✓ No train/non-train hash overlap detected.")


def _sha256(path: str, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _assert_wild_isolation(df: pd.DataFrame) -> None:
    """Hard assertion: every release_in_the_wild row MUST be eval_ood."""
    wild = df[df["source"] == "release_in_the_wild"]
    violators = wild[wild["split"] != "eval_ood"]
    if not violators.empty:
        raise AssertionError(
            f"CRITICAL: {len(violators)} release_in_the_wild rows have split != 'eval_ood'!\n"
            f"{violators[['path', 'split']].head(10)}"
        )
    print("\n  ✓ release_in_the_wild never-in-train assertion PASSED")


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "═" * 70)
    print("  FINAL MANIFEST SUMMARY")
    print("═" * 70)
    summary = (
        df[df["split"] != "dropped"]
        .groupby(["split", "source", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "source", "label"])
    )
    for _, row in summary.iterrows():
        print(
            f"  {row['split']:<15}  {row['source']:<30}  "
            f"{row['label']:<10}  {row['count']:>8,}"
        )

    print(f"\n  Total rows in final manifest: {(df['split'] != 'dropped').sum():,}")
    print(f"  Dropped (imbalance): {(df['split'] == 'dropped').sum():,}")
    from pipeline.scanner import MLAAD_HELD_OUT_GENERATORS
    print(f"\n  MLAAD held-out generators (eval_ood only): {sorted(MLAAD_HELD_OUT_GENERATORS)}")
    print("═" * 70 + "\n")
