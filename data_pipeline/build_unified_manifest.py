"""
build_unified_manifest.py — Main orchestrator for the multi-source unified dataset pipeline.

Phases:
  1. Scan & Index -> raw_index.csv
  2. Pre-Selection -> selected_files.csv
  3. Standardize Audio -> /media/akenzz/D/DataSet_processed/...
  4. VAD Chunking -> chunks > 12s files
  5. Build unified manifest -> unified_manifest_raw.csv
  6. Verify & Report -> unified_manifest_final.csv
"""

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pipeline import scanner, audio_utils, vad_chunker, manifest_builder, balancer

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pipeline.main")

DATASET_ROOT     = Path("/media/akenzz/D/DataSet")
PROCESSED_ROOT   = Path("/media/akenzz/D/DataSet_processed")
MANIFEST_DIR     = Path(__file__).resolve().parent / "manifests"
STATE_FILE       = MANIFEST_DIR / "pipeline_state.json"
INDEX_CSV        = MANIFEST_DIR / "raw_index.csv"
SELECTED_CSV     = MANIFEST_DIR / "selected_files.csv"
RAW_MANIFEST     = MANIFEST_DIR / "unified_manifest_raw.csv"
FINAL_MANIFEST   = MANIFEST_DIR / "unified_manifest_final.csv"


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"completed_phases": []}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def phase_1_scan():
    """Run all scanners and dump to raw_index.csv"""
    logger.info("--- PHASE 1: Scan & Index ---")
    rows = []
    
    for source_name, scan_func in scanner.ALL_SCANNERS:
        logger.info("Scanning %s...", source_name)
        count = 0
        for row in scan_func():
            rows.append(row)
            count += 1
            if count % 10000 == 0:
                logger.info("  ...scanned %d files", count)
        logger.info("Finished %s: %d files found", source_name, count)
        
    df = pd.DataFrame(rows)
    df.to_csv(INDEX_CSV, index=False)
    logger.info("Phase 1 complete. Wrote %d rows to %s", len(df), INDEX_CSV.name)


def phase_2_preselect():
    """Stratified sampling to reduce raw index to a selected subset."""
    from pipeline import pre_selector
    logger.info("--- PHASE 2: Pre-Selection ---")
    pre_selector.run(INDEX_CSV, SELECTED_CSV)


def _process_file(args):
    """Worker function for phase 3."""
    idx, src_path_str, rel_path_str, resample = args
    src_path = Path(src_path_str)
    dst_path = PROCESSED_ROOT / Path(rel_path_str).parent / f"{Path(rel_path_str).stem}.wav"
    
    error = None
    try:
        audio_utils.standardize(src_path, dst_path, resample=resample, skip_if_exists=True)
    except Exception as e:
        error = f"{src_path}: {e}"
        
    return error


def phase_3_standardize():
    """Resample, mono, normalize, silence trim all audio in the selected index."""
    logger.info("--- PHASE 3: Audio Standardization ---")
    df = pd.read_csv(SELECTED_CSV)
    
    total = len(df)
    
    tasks = []
    for idx, row in df.iterrows():
        src_path = Path(row["src_path"])
        try:
            rel_path = src_path.relative_to(DATASET_ROOT)
        except ValueError:
            rel_path = Path(row["source"]) / src_path.name
            
        resample = True
        if row["source"] in ["asvspoof2019_la", "release_in_the_wild", "processed_v2"]:
            resample = False
            
        tasks.append((idx, str(src_path), str(rel_path), resample))
        
    error_count = 0
    max_workers = max(1, os.cpu_count() - 2)
    logger.info("Starting standardization with %d workers...", max_workers)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_file, t) for t in tasks]
        
        from tqdm import tqdm
        for future in tqdm(as_completed(futures), total=total, desc="Standardizing Audio"):
            err = future.result()
            if err:
                error_count += 1
                if error_count < 10:
                    logger.warning("Standardize failed: %s", err)
                    
    logger.info("Phase 3 complete. Total: %d, Errors: %d", total, error_count)


def _chunk_file(args):
    """Worker function for phase 4."""
    idx, src_path_str, rel_path_str = args
    src_path = Path(src_path_str)
    dst_path = PROCESSED_ROOT / Path(rel_path_str).parent / f"{Path(rel_path_str).stem}.wav"
    
    chunked = 0
    if dst_path.exists():
        try:
            dur = audio_utils.get_duration(dst_path)
            if dur > vad_chunker.MAX_CHUNK_S:
                audio, sr = audio_utils._load_audio(dst_path)
                chunks = vad_chunker.chunk_file(
                    audio, sr, 
                    dst_dir=dst_path.parent, 
                    base_id=dst_path.stem, 
                    skip_if_exists=True
                )
                if chunks:
                    chunked = 1
        except Exception as e:
            pass
            
    return chunked


def phase_4_chunking():
    """Run Silero VAD on files > 12s."""
    logger.info("--- PHASE 4: VAD Chunking ---")
    df = pd.read_csv(SELECTED_CSV)
    total = len(df)
    
    tasks = []
    for idx, row in df.iterrows():
        src_path = Path(row["src_path"])
        try:
            rel_path = src_path.relative_to(DATASET_ROOT)
        except ValueError:
            rel_path = Path(row["source"]) / src_path.name
        tasks.append((idx, str(src_path), str(rel_path)))
        
    chunked_files = 0
    max_workers = max(1, os.cpu_count() - 2)
    logger.info("Starting chunking with %d workers...", max_workers)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_chunk_file, t) for t in tasks]
        
        from tqdm import tqdm
        for future in tqdm(as_completed(futures), total=total, desc="VAD Chunking"):
            chunked_files += future.result()

    logger.info("Phase 4 complete. Chunked %d long files.", chunked_files)


def phase_5_build_manifest():
    """Merge selected_files + actual output files into raw manifest."""
    logger.info("--- PHASE 5: Build Raw Manifest ---")
    manifest_builder.build_raw_manifest(SELECTED_CSV, PROCESSED_ROOT, RAW_MANIFEST)


def phase_6_verify():
    """Verify holdouts, leakage checks, and write final CSV."""
    logger.info("--- PHASE 6: Verify & Finalize ---")
    df_raw = pd.read_csv(RAW_MANIFEST)
    df_final = balancer.run(df_raw)
    
    df_final = df_final[df_final["split"] != "dropped"].copy()
    
    df_final.to_csv(FINAL_MANIFEST, index=False)
    logger.info("Phase 6 complete. Final manifest written to %s", FINAL_MANIFEST.name)


def main():
    parser = argparse.ArgumentParser(description="Unified Audio Pipeline")
    parser.add_argument("--force-phase", type=int, choices=[1, 2, 3, 4, 5, 6], 
                        help="Force run a specific phase regardless of state")
    args = parser.parse_args()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    state = load_state()
    completed = set(state["completed_phases"])

    phases = [
        (1, phase_1_scan),
        (2, phase_2_preselect),
        (3, phase_3_standardize),
        (4, phase_4_chunking),
        (5, phase_5_build_manifest),
        (6, phase_6_verify),
    ]

    for phase_num, phase_func in phases:
        if args.force_phase and args.force_phase != phase_num:
            continue
            
        if phase_num in completed and not args.force_phase:
            logger.info("Skipping Phase %d (already completed)", phase_num)
            continue
            
        phase_func()
        
        if phase_num not in completed:
            completed.add(phase_num)
            state["completed_phases"] = list(completed)
            save_state(state)

    logger.info("Pipeline finished successfully!")


if __name__ == "__main__":
    main()
