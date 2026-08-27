"""
data_pipeline/fetch_in_the_wild.py
======================================
Downloads In-The-Wild dataset from HuggingFace.
Source: mueller91/In-The-Wild
"""

import os
import sys
import argparse
import shutil
import zipfile
from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from data_pipeline.schema import REQUIRED_COLUMNS, validate_schema

DEFAULT_AUDIO_ROOT = Path(os.environ.get("IN_THE_WILD_ROOT", r"E:\DatasetSIH\in_the_wild"))
MANIFEST_DIR       = repo_root / "data_pipeline" / "manifests"
HF_DATASET_ID      = "mueller91/In-The-Wild"

def download_and_build_manifest(audio_root: Path) -> Path:
    audio_root.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    hf_cache = audio_root.parent / ".hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache)
    os.environ["HF_HOME"]           = str(hf_cache)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    print(f"\n[1/3] Processing In-The-Wild from HuggingFace...")
    print(f"      Dataset   : {HF_DATASET_ID}")
    print(f"      Audio dir : {audio_root}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        sys.exit(1)

    print(f"      Locating datasets in cache...")
    try:
        snapshot_download(
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            cache_dir=str(hf_cache / "hub"),
            ignore_patterns=["*.md", "*.json", "*.txt"],
        )
    except Exception as exc:
        print(f"      [NOTE] snapshot_download info: {exc}")

    hub_dir = hf_cache / "hub" / f"datasets--{HF_DATASET_ID.replace('/', '--')}" / "snapshots"
    if not hub_dir.exists():
        print(f"[FAIL] Could not find snapshot cache at {hub_dir}")
        sys.exit(1)
        
    snapshots = list(hub_dir.iterdir())
    if not snapshots:
        print(f"[FAIL] No snapshots found in {hub_dir}")
        sys.exit(1)
        
    snapshot_path = snapshots[0]
    print(f"      Found snapshot at {snapshot_path}")

    zip_files = list(snapshot_path.rglob("*.zip"))
    if zip_files:
        print(f"      Extracting {len(zip_files)} zip archive(s)... (This may take a few minutes for 8GB+)")
        for zf_path in zip_files:
            try:
                with zipfile.ZipFile(zf_path, 'r') as zf:
                    zf.extractall(audio_root)
            except Exception as e:
                print(f"      [WARN] Zip extraction error: {e}")
    else:
        print(f"      [WARN] No zip files found in snapshot! Trying to scan for raw wavs...")

    all_wavs = list(audio_root.rglob("*.wav")) + list(snapshot_path.rglob("*.wav"))
    all_csvs = list(audio_root.rglob("*.csv")) + list(snapshot_path.rglob("*.csv"))
    
    if not all_wavs:
        print("[FAIL] No .wav files found after extraction.")
        sys.exit(1)
        
    print(f"      Found {len(all_wavs)} .wav files.")

    # Try to load metadata if a CSV exists
    meta_df = None
    if all_csvs:
        print(f"      Found metadata CSV: {all_csvs[0].name}")
        try:
            meta_df = pd.read_csv(all_csvs[0])
        except Exception as e:
            print(f"      [WARN] Could not read metadata CSV: {e}")
            
    # If no metadata CSV, we'll try to find parquets that might have metadata
    parquet_files = list(snapshot_path.rglob("*.parquet"))
    if not meta_df and parquet_files:
        print(f"      Reading metadata from {len(parquet_files)} parquet files...")
        try:
            dfs = [pd.read_parquet(p) for p in parquet_files]
            meta_df = pd.concat(dfs, ignore_index=True)
        except Exception as e:
            print(f"      [WARN] Could not read Parquet metadata: {e}")

    # Build a lookup for metadata
    meta_dict = {}
    if meta_df is not None:
        # Standardize columns
        cols = {c.lower(): c for c in meta_df.columns}
        file_col = cols.get("file") or cols.get("filename") or cols.get("path") or cols.get("audio_filepath")
        label_col = cols.get("label") or cols.get("is_spoof")
        sys_col = cols.get("system") or cols.get("generator")
        spk_col = cols.get("speaker") or cols.get("speaker_id")
        
        if file_col:
            for _, row in meta_df.iterrows():
                fname = str(row[file_col])
                stem = Path(fname).stem
                meta_dict[stem] = {
                    "label": str(row[label_col]) if label_col else "spoof",
                    "generator": str(row[sys_col]) if sys_col else "unknown",
                    "speaker": str(row[spk_col]) if spk_col else "unknown",
                }

    rows = []
    i = 0
    try:
        import soundfile as sf
    except ImportError:
        sf = None

    for wav_path in all_wavs:
        stem = wav_path.stem
        meta = meta_dict.get(stem, {})
        
        raw_label = str(meta.get("label", "spoof")).strip().lower()
        label  = "bonafide" if raw_label in ("real", "bonafide", "genuine", "0", "false") else "spoof"
        gen_id = str(meta.get("generator", "unknown"))
        if label == "bonafide":
            gen_id = "none"
            
        speaker_id = str(meta.get("speaker", f"spk_{i:05d}"))

        dur = 0.0
        if sf:
            try:
                dur = round(sf.info(str(wav_path)).duration, 3)
            except Exception:
                pass

        # Ensure the file is actually inside audio_root for consistency
        out_path = audio_root / speaker_id / f"{stem}.wav"
        if not out_path.exists() and wav_path != out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav_path, out_path)
            
        final_path = out_path if out_path.exists() else wav_path

        rows.append({
            "path":           str(final_path.resolve()),
            "label":          label,
            "split":          "eval_ood",
            "source_dataset": "In-the-Wild",
            "speaker_id":     speaker_id,
            "utterance_id":   stem,
            "generator_id":   gen_id,
            "language":       "en",
            "codec":          "pcm_16k",
            "duration_s":     dur,
            "license":        "CC-BY-4.0",
            "consent":        "yes",
        })

        i += 1
        if i % 1000 == 0:
            print(f"  [{i:>6} done]  samples={len(rows):,}", flush=True)

    if not rows:
        print("[FAIL] 0 rows extracted.")
        sys.exit(1)

    df_csv = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    csv_path = MANIFEST_DIR / "in_the_wild_eval_ood.csv"
    df_csv.to_csv(csv_path, index=False)
    
    return csv_path

def validate_manifest(csv_path: Path) -> None:
    df     = pd.read_csv(csv_path)
    errors = validate_schema(df)
    if errors:
        print("[FAIL] In-The-Wild schema errors:")
        for e in errors:
            print(f"  {e}")
    else:
        gens = df["generator_id"].nunique()
        print(f"  [PASS] In-The-Wild schema valid! ({len(df):,} rows, {gens} TTS systems)")

def main():
    parser = argparse.ArgumentParser(description="Download In-The-Wild and build eval_ood manifest.")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_AUDIO_ROOT))
    args = parser.parse_args()

    audio_root = Path(args.dataset_root)

    print("=" * 65)
    print("In-The-Wild — AI4Bharat Indic Speech Pipeline")
    print("=" * 65)
    
    csv_path = download_and_build_manifest(audio_root=audio_root)

    print("\n[3/3] Validating manifests ...")
    validate_manifest(csv_path)

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)

if __name__ == "__main__":
    main()
