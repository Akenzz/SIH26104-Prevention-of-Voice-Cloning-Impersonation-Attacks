"""
manifest_builder.py — Assembles unified_manifest_raw.csv from processed files + index.

Reads the raw index (Phase 1) and cross-references it with the actual
written files in the output directory (Phase 2 + Phase 3).
If a file was chunked, it replaces the original row with N rows for the chunks.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import soundfile as sf

logger = logging.getLogger("pipeline.manifest_builder")

def build_raw_manifest(
    index_csv: Path,
    out_dir: Path,
    output_csv: Path
) -> pd.DataFrame:
    """Builds the raw, pre-balanced unified manifest."""
    logger.info("Building raw manifest from %s and %s", index_csv.name, out_dir.name)
    
    if not index_csv.exists():
        raise FileNotFoundError(f"Missing index: {index_csv}")
        
    df_index = pd.read_csv(index_csv)
    
    # We will build a list of dicts for the final dataframe
    final_rows = []
    
    missing_count = 0
    
    from tqdm import tqdm
    # For each row in the index, figure out what the actual output file(s) is
    for idx, row in tqdm(df_index.iterrows(), total=len(df_index), desc="Building Manifest"):
        src_path = Path(row["src_path"])
        
        # Source path relative to /media/akenzz/D/DataSet
        # OR if it's already in processed folder, we handle it slightly differently?
        # Actually, the pipeline saves things under DataSet_processed preserving relative paths.
        
        # The expected processed path is out_dir / source_name / relative_path
        # But wait! For processed folder, we just copy? 
        # The build_unified_manifest.py handles writing. Let's just assume the 
        # base processed path is passed to us, or we just glob the chunks.
        
        # We need a robust way to map src_path to the processed path(s).
        # We know the base name of the source file.
        # It's better to just reconstruct the expected path.
        dataset_root = Path("/media/akenzz/D/DataSet")
        try:
            rel_path = src_path.relative_to(dataset_root)
            # The destination dir was `out_dir / rel_path.parent`
            # The base filename was `rel_path.stem`
            base_expected_path = out_dir / rel_path.parent / f"{rel_path.stem}.wav"
        except ValueError:
            # If not relative to dataset root, just use its name
            base_expected_path = out_dir / row["source"] / f"{src_path.stem}.wav"
            
        parent_dir = base_expected_path.parent
        base_stem = base_expected_path.stem
        
        # Check for chunks first
        chunks = sorted(parent_dir.glob(f"{base_stem}_chunk*.wav"))
        
        if chunks:
            # Add one row per chunk
            for chunk_path in chunks:
                try:
                    dur = sf.info(str(chunk_path)).duration
                    new_row = row.copy()
                    new_row["path"] = str(chunk_path)
                    new_row["duration_s"] = dur
                    final_rows.append(new_row)
                except Exception as e:
                    logger.warning("Failed to read chunk %s: %s", chunk_path, e)
        elif base_expected_path.exists():
            # No chunks, just the single processed file
            try:
                dur = sf.info(str(base_expected_path)).duration
                new_row = row.copy()
                new_row["path"] = str(base_expected_path)
                new_row["duration_s"] = dur
                final_rows.append(new_row)
            except Exception as e:
                logger.warning("Failed to read processed file %s: %s", base_expected_path, e)
        else:
            # This file didn't survive processing (e.g., corrupt, zero-byte, etc)
            missing_count += 1
            if missing_count < 10:
                logger.debug("Skipping missing processed file for %s", src_path)

    if missing_count > 0:
        logger.info("%d source files were dropped during processing", missing_count)

    df_raw = pd.DataFrame(final_rows)
    # Reorder/clean columns
    expected_cols = [
        "path", "label", "split_hint", "source", "speaker_id", 
        "language", "generator_id", "duration_s", "held_out"
    ]
    # Drop src_path, keep others
    for c in expected_cols:
        if c not in df_raw.columns:
            df_raw[c] = None
            
    df_raw = df_raw[expected_cols]
    df_raw.to_csv(output_csv, index=False)
    logger.info("Wrote raw manifest with %d rows to %s", len(df_raw), output_csv)
    
    return df_raw
