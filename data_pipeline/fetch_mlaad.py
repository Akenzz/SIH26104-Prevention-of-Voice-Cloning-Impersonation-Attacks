"""
data_pipeline/fetch_mlaad.py
==============================
Downloads MLAAD (Multi-Language Audio Anti-Spoofing Dataset) from HuggingFace
and converts it to the standard 12-column manifest format.

Source  : HuggingFace  mueller91/MLAAD   (GATED — requires HF login + acceptance)
          MLAAD-tiny   mueller91/MLAAD-tiny  (~3.54 GB, open, good for prototyping)
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from data_pipeline.schema import REQUIRED_COLUMNS, validate_schema

DEFAULT_AUDIO_ROOT    = Path(os.environ.get("MLAAD_ROOT", r"E:\DatasetSIH\mlaad"))
MANIFEST_DIR          = repo_root / "data_pipeline" / "manifests"

HF_DATASET_FULL       = "mueller91/MLAAD"
HF_DATASET_TINY       = "mueller91/MLAAD-tiny"

DEFAULT_HELD_OUT_GENERATORS = {"VITS", "YourTTS", "vits", "yourtts", "vits-tts"}


def download_and_build_manifest(
    audio_root: Path,
    use_tiny: bool,
    held_out_generators: set,
) -> tuple:
    audio_root.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    hf_id = HF_DATASET_TINY if use_tiny else HF_DATASET_FULL
    size_note = "~3.54 GB (tiny)" if use_tiny else "1000+ hours (full — needs HF login)"

    hf_cache = audio_root.parent / ".hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache)
    os.environ["HF_HOME"]           = str(hf_cache)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["DISABLE_TORCHCODEC"]              = "1"

    print(f"\n[1/3] Downloading MLAAD from HuggingFace...")
    print(f"      Dataset    : {hf_id}")
    print(f"      Size       : {size_note}")
    print(f"      Audio      : {audio_root}")
    print(f"      HF cache   : {hf_cache}  (all downloads go here, not C:)")
    print(f"      Held-out (eval_ood): {sorted(held_out_generators)}\n")

    try:
        import soundfile as sf
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        print("       Run: pip install soundfile pandas huggingface_hub")
        sys.exit(1)

    print(f"      Downloading audio files to cache...")
    try:
        local_dir = hf_cache / "raw" / hf_id.replace("/", "--")
        local_dir.mkdir(parents=True, exist_ok=True)
        # MLAAD-tiny has raw audio directly in Hugging Face, no parquet
        snapshot_download(
            repo_id=hf_id,
            repo_type="dataset",
            cache_dir=str(hf_cache / "hub"),
            local_dir=str(local_dir),
            ignore_patterns=["*.md", "*.json", "*.txt", "*.py"],
        )
    except Exception as exc:
        print(f"      [NOTE] snapshot_download info: {exc}")

    # For snapshot download without local_dir hard-linking properly, search the hub cache
    hub_dir = hf_cache / "hub" / f"datasets--{hf_id.replace('/', '--')}" / "snapshots"
    if not hub_dir.exists():
        print(f"[FAIL] Could not find snapshot cache at {hub_dir}")
        sys.exit(1)

    # Get the latest snapshot
    snapshots = list(hub_dir.iterdir())
    if not snapshots:
        print(f"[FAIL] No snapshots found in {hub_dir}")
        sys.exit(1)
        
    snapshot_path = snapshots[0]
    print(f"      Found snapshot at {snapshot_path}")

    # Grab all .wav and .flac files
    audio_files = list(snapshot_path.rglob("*.wav")) + list(snapshot_path.rglob("*.flac"))
    if not audio_files:
        print(f"[FAIL] No audio files found in {snapshot_path}")
        sys.exit(1)

    print(f"      Found {len(audio_files)} audio file(s). Processing...\n")

    train_rows    = []
    eval_ood_rows = []
    failed        = 0
    i             = 0

    for src_path in audio_files:
        try:
            rel_path = src_path.relative_to(snapshot_path)
            parts = rel_path.parts
            
            # parts will be something like: ('fake', 'en', 'VITS', 'filename.wav')
            # or ('original', 'en', 'filename.wav')
            if len(parts) < 3:
                continue
                
            label_type = parts[0].lower()
            language = parts[1].lower()
            
            if label_type == "fake":
                label = "spoof"
                tts_system = parts[2]
            else:
                label = "bonafide"
                tts_system = "none"

            speaker_id = f"spk_{i:05d}"
            stem = src_path.stem

            is_held_out = any(h.lower() in tts_system.lower() for h in held_out_generators)
            row_split   = "eval_ood" if is_held_out else "train"

            lang_dir = audio_root / language / tts_system
            lang_dir.mkdir(parents=True, exist_ok=True)
            out_path = lang_dir / f"{stem}{src_path.suffix}"

            if not out_path.exists():
                shutil.copy2(src_path, out_path)

            if out_path.exists():
                try:
                    info = sf.info(str(out_path))
                    dur = round(info.duration, 3)
                except Exception:
                    dur = 0.0

                row = {
                    "path":           str(out_path.resolve()),
                    "label":          label,
                    "split":          row_split,
                    "source_dataset": "MLAAD-tiny" if use_tiny else "MLAAD",
                    "speaker_id":     speaker_id,
                    "utterance_id":   stem,
                    "generator_id":   tts_system,
                    "language":       language,
                    "codec":          "pcm_16k" if out_path.suffix == ".wav" else out_path.suffix[1:],
                    "duration_s":     dur,
                    "license":        "CC-BY-NC-4.0",
                    "consent":        "no",
                }
                
                if row_split == "train":
                    train_rows.append(row)
                else:
                    eval_ood_rows.append(row)

            i += 1
            if i % 1000 == 0:
                print(f"  [{i:>6} done]  train={len(train_rows):,}  eval_ood={len(eval_ood_rows):,}", flush=True)

        except Exception as exc:
            failed += 1
            if failed <= 5:
                print(f"  [WARN] Row {i}: {exc}")
            i += 1

    train_csv = eval_ood_csv = None

    if train_rows:
        df_train  = pd.DataFrame(train_rows, columns=REQUIRED_COLUMNS)
        train_csv = MANIFEST_DIR / "mlaad_train.csv"
        df_train.to_csv(train_csv, index=False)
        print(f"\n  [OK] Train manifest   : {train_csv}  ({len(df_train):,} rows)")

    if eval_ood_rows:
        df_ood    = pd.DataFrame(eval_ood_rows, columns=REQUIRED_COLUMNS)
        eval_ood_csv = MANIFEST_DIR / "mlaad_eval_ood.csv"
        df_ood.to_csv(eval_ood_csv, index=False)
        print(f"  [OK] Eval-OOD manifest: {eval_ood_csv}  ({len(df_ood):,} rows)")
        print(f"       Held-out systems  : {df_ood['generator_id'].unique().tolist()}")
    else:
        print(f"\n  [WARN] No held-out rows found. Check --held-out-generators names.")

    if failed:
        print(f"  [WARN] {failed} items failed")

    return train_csv, eval_ood_csv


def validate_manifest(csv_path: Path, label: str) -> None:
    df     = pd.read_csv(csv_path)
    errors = validate_schema(df)
    if errors:
        print(f"[FAIL] {label} schema errors:")
        for e in errors:
            print(f"  {e}")
    else:
        langs  = df["language"].unique().tolist()
        gens   = df["generator_id"].nunique()
        splits = df["split"].unique().tolist()
        print(f"  [PASS] {label}: {len(df):,} rows  |  {len(langs)} languages  |  {gens} TTS systems  |  splits={splits}")


def main():
    parser = argparse.ArgumentParser(description="Download MLAAD and build train/eval_ood manifests.")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_AUDIO_ROOT))
    parser.add_argument("--use-tiny", action="store_true")
    parser.add_argument("--held-out-generators", nargs="+", default=list(DEFAULT_HELD_OUT_GENERATORS))
    args = parser.parse_args()

    audio_root = Path(args.dataset_root)
    held_out   = set(args.held_out_generators)

    print("=" * 65)
    print("MLAAD — Multilingual Deepfake Dataset Pipeline")
    print("=" * 65)
    
    train_csv, ood_csv = download_and_build_manifest(
        audio_root=audio_root,
        use_tiny=args.use_tiny,
        held_out_generators=held_out,
    )

    print("\n[3/3] Validating manifests ...")
    if train_csv and train_csv.exists():
        validate_manifest(train_csv, "MLAAD train")
    if ood_csv and ood_csv.exists():
        validate_manifest(ood_csv, "MLAAD eval_ood")

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
