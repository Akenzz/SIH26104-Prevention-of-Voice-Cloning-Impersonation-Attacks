"""
pre_selector.py — Stratified sampling to drastically reduce dataset size before processing.

Filters the raw_index.csv down to selected_files.csv.
"""

import logging
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger("pipeline.pre_selector")

RANDOM_SEED = 42

TRAIN_BONAFIDE_TARGET = 20_000
TRAIN_SPOOF_TARGET    = 20_000
DEV_TARGET_PER_CLASS  = 1_500
SANITY_TARGET         = 3_000

SANITY_SOURCES = {"gramvaani_1000h", "gv_train_100h", "gv_eval_3h", "kathbath", "asvspoof2019_la"}


def run(raw_index_path: Path, output_path: Path) -> pd.DataFrame:
    logger.info("Running Pre-Selection...")
    df = pd.read_csv(raw_index_path)
    
    # We will build a list of chosen indices, then subset the DataFrame.
    # At the same time, we overwrite the `split_hint` with the final assigned `split`.
    df["split"] = df["split_hint"]  # Default
    
    random.seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)
    
    # 1. PRESERVE ALL release_in_the_wild
    wild_mask = df["source"] == "release_in_the_wild"
    df.loc[wild_mask, "split"] = "eval_ood"
    chosen_wild = df[wild_mask].index.tolist()
    
    available_mask = ~wild_mask & ~df["held_out"]  # don't use held_out MLAAD generators for train/dev/sanity
    
    # Add held_out MLAAD generators to eval_ood automatically
    mlaad_heldout_mask = (df["source"] == "mlaad") & df["held_out"]
    df.loc[mlaad_heldout_mask, "split"] = "eval_ood"
    chosen_heldout = df[mlaad_heldout_mask].index.tolist()
    
    # 2. EVAL_SANITY (3,000 bonafide strictly from SANITY_SOURCES)
    sanity_mask = available_mask & df["source"].isin(SANITY_SOURCES) & (df["label"] == "bonafide")
    sanity_candidates = df[sanity_mask].index.tolist()
    chosen_sanity = _stratified_sample(df.loc[sanity_candidates], SANITY_TARGET, rng)
    df.loc[chosen_sanity, "split"] = "eval_sanity"
    available_mask &= ~df.index.isin(chosen_sanity)
    
    # 3. DEV (~1,500 bonafide, ~1,500 spoof from split_hint == 'dev')
    dev_b_mask = available_mask & (df["split_hint"] == "dev") & (df["label"] == "bonafide")
    dev_s_mask = available_mask & (df["split_hint"] == "dev") & (df["label"] == "spoof")
    
    chosen_dev_b = _stratified_sample(df[dev_b_mask], DEV_TARGET_PER_CLASS, rng)
    chosen_dev_s = _stratified_sample(df[dev_s_mask], DEV_TARGET_PER_CLASS, rng)
    
    df.loc[chosen_dev_b + chosen_dev_s, "split"] = "dev"
    available_mask &= ~df.index.isin(chosen_dev_b + chosen_dev_s)
    
    # 4. TRAIN (20k bonafide, 20k spoof from remaining)
    train_b_mask = available_mask & (df["label"] == "bonafide")
    train_s_mask = available_mask & (df["label"] == "spoof")
    
    chosen_train_b = _stratified_sample(df[train_b_mask], TRAIN_BONAFIDE_TARGET, rng)
    chosen_train_s = _stratified_sample(df[train_s_mask], TRAIN_SPOOF_TARGET, rng)
    
    df.loc[chosen_train_b + chosen_train_s, "split"] = "train"
    
    # Combine all chosen indices
    all_chosen = (
        chosen_wild + 
        chosen_heldout + 
        chosen_sanity + 
        chosen_dev_b + chosen_dev_s + 
        chosen_train_b + chosen_train_s
    )
    
    df_selected = df.loc[all_chosen].copy()
    
    # Print summary
    print("\n" + "═" * 60)
    print("  PRE-SELECTION COUNTS")
    print("═" * 60)
    summary = (
        df_selected.groupby(["split", "source", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "source", "label"])
    )
    for _, row in summary.iterrows():
        print(f"  {row['split']:<12}  {row['source']:<22}  {row['label']:<10}  {row['count']:>8,}")
    print(f"\n  Total Selected Files: {len(df_selected):,}")
    print("═" * 60 + "\n")
    
    df_selected.to_csv(output_path, index=False)
    logger.info("Wrote selected subset to %s", output_path)
    return df_selected


def _stratified_sample(df_subset: pd.DataFrame, target_count: int, rng: random.Random) -> list[int]:
    """Sample target_count rows stratified evenly across source + speaker/generator."""
    if len(df_subset) <= target_count:
        return df_subset.index.tolist()
        
    groups = defaultdict(list)
    for idx, row in df_subset.iterrows():
        # Stratify by source, and then by speaker (if bonafide) or generator (if spoof)
        sec_key = row["generator_id"] if row["label"] == "spoof" else str(row.get("speaker_id", ""))
        key = (row["source"], sec_key)
        groups[key].append(idx)
        
    kept = []
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    
    # Round-robin
    while len(kept) < target_count and groups:
        empty_keys = []
        for key in group_keys:
            if len(kept) >= target_count:
                break
            if groups[key]:
                # Pop randomly from the group
                kept.append(groups[key].pop(rng.randint(0, len(groups[key]) - 1)))
            else:
                empty_keys.append(key)
                
        for k in empty_keys:
            group_keys.remove(k)
            
    return kept
