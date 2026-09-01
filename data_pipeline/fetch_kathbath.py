"""
data_pipeline/fetch_kathbath.py
=================================
Downloads AI4Bharat Kathbath (real Indic speech, 12 Indian languages)
from HuggingFace and builds the 12-column manifest.

Direct Parquet reader approach — completely bypasses datasets library Audio encoding.
"""

import os
import sys
import argparse
import io as _io
from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from data_pipeline.schema import REQUIRED_COLUMNS, validate_schema

DEFAULT_AUDIO_ROOT = Path(os.environ.get("KATHBATH_ROOT", r"E:\DatasetSIH\kathbath"))
MANIFEST_DIR       = repo_root / "data_pipeline" / "manifests"
HF_DATASET_ID      = "ai4bharat/Kathbath"

KATHBATH_LANGUAGES = {
    "bengali":   "bn",
    "gujarati":  "gu",
    "hindi":     "hi",
    "kannada":   "kn",
    "malayalam": "ml",
    "marathi":   "mr",
    "odia":      "or",
    "punjabi":   "pa",
    "sanskrit":  "sa",
    "tamil":     "ta",
    "telugu":    "te",
    "urdu":      "ur",
}


def download_and_build_manifest(
    audio_root: Path,
    languages: list,
    out_suffix: str = "",
    max_shards: int = 0,
    max_per_lang: int = 0,
    valid_only: bool = False,
) -> tuple:
    audio_root.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    if not languages:
        target_langs = list(KATHBATH_LANGUAGES.keys())
    else:
        iso_to_name = {v: k for k, v in KATHBATH_LANGUAGES.items()}
        target_langs = []
        for l in languages:
            l_lower = l.lower()
            if l_lower in KATHBATH_LANGUAGES:
                target_langs.append(l_lower)
            elif l_lower in iso_to_name:
                target_langs.append(iso_to_name[l_lower])

    hf_cache = audio_root.parent / ".hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache)
    os.environ["HF_HOME"]           = str(hf_cache)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["DISABLE_TORCHCODEC"]              = "1"

    print(f"\n[1/3] Processing Kathbath from HuggingFace...")
    print(f"      Dataset   : {HF_DATASET_ID}")
    print(f"      Languages : {target_langs}")
    print(f"      Audio dir : {audio_root}\n")

    try:
        import soundfile as sf
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        print("       Run: pip install soundfile pandas huggingface_hub")
        sys.exit(1)

    all_train_rows = []
    all_eval_rows  = []

    for lang_name in target_langs:
        iso_code = KATHBATH_LANGUAGES[lang_name]
        print(f"\n  Language: {lang_name} ({iso_code})")

        local_dir = hf_cache / "parquet" / f"kathbath_{lang_name}"
        local_dir.mkdir(parents=True, exist_ok=True)

        # Only grab as many train shards as needed to satisfy max_per_lang, plus
        # the (small) valid shards. Each Kathbath train shard holds ~2.5k
        # utterances, so 1-2 shards is plenty for a few-hundred-clip balance set;
        # downloading all 33-40 shards would waste tens of GB per language.
        if valid_only:
            allow = [f"{lang_name}/valid-*.parquet"]
            print(f"    Downloading ONLY the small valid split for {lang_name}")
        elif max_shards and max_shards > 0:
            allow = [f"{lang_name}/train-{i:05d}-of-*.parquet" for i in range(max_shards)]
            allow += [f"{lang_name}/valid-*.parquet"]
            print(f"    Limiting download to first {max_shards} train shard(s) + valid shards")
        else:
            allow = [f"data/{lang_name}/*", f"*{lang_name}*"]

        try:
            snapshot_download(
                repo_id=HF_DATASET_ID,
                repo_type="dataset",
                cache_dir=str(hf_cache / "hub"),
                local_dir=str(local_dir),
                allow_patterns=allow,
                ignore_patterns=["*.md", "*.json", "*.txt"],
            )
        except Exception as exc:
            print(f"    [WARN] Download/snapshot error for {lang_name}: {exc}")

        parquet_files = sorted(local_dir.rglob("*.parquet"))
        if not parquet_files:
            # Fallback: search anywhere in hf_cache for matching parquets
            parquet_files = sorted((hf_cache / "hub").rglob(f"*{lang_name}*.parquet"))

        if not parquet_files:
            print(f"    [WARN] No Parquet files found for {lang_name}")
            continue

        print(f"    Found {len(parquet_files)} Parquet file(s). Reading directly...\n")
        lang_dir = audio_root / lang_name
        lang_dir.mkdir(exist_ok=True)

        import pyarrow.parquet as pq

        i = 0
        lang_written = 0            # bonafide clips kept for THIS language
        for pq_file in parquet_files:
            if max_per_lang and lang_written >= max_per_lang:
                print(f"    [cap] reached {max_per_lang} clips for {lang_name}; stopping")
                break
            row_split = "eval" if ("valid" in pq_file.name.lower() or "test" in pq_file.name.lower()) else "train"
            target_list = all_eval_rows if row_split == "eval" else all_train_rows

            try:
                table = pq.read_table(str(pq_file))
                # Convert pyarrow table to list of dicts for easy iteration
                batch = table.to_pylist()
            except Exception as exc:
                print(f"    [WARN] Could not read {pq_file.name}: {exc}")
                continue

            for row_data in batch:
                try:
                    # audio bytes are stored in the 'audio_filepath' column for Kathbath!
                    audio_col = row_data.get("audio_filepath") or row_data.get("audio")
                    arr, sr   = None, 16000
                    
                    if audio_col and isinstance(audio_col, dict):
                        raw_bytes = audio_col.get("bytes")
                        raw_path  = audio_col.get("path")
                        if raw_bytes:
                            try:
                                arr, sr = sf.read(_io.BytesIO(raw_bytes), dtype="float32")
                            except Exception:
                                import torchaudio
                                tensor, sr = torchaudio.load(_io.BytesIO(raw_bytes))
                                arr = tensor.numpy().T
                                if tensor.shape[0] > 1:
                                    arr = tensor.mean(dim=0).numpy()
                        elif raw_path and Path(raw_path).exists():
                            try:
                                arr, sr = sf.read(raw_path, dtype="float32")
                            except Exception:
                                import torchaudio
                                tensor, sr = torchaudio.load(raw_path)
                                arr = tensor.numpy().T
                                if tensor.shape[0] > 1:
                                    arr = tensor.mean(dim=0).numpy()

                    speaker_id = str(row_data.get("speaker_id", row_data.get("speaker", f"spk_{i:05d}")))
                    
                    # Some datasets use fname, some use path
                    file_name = row_data.get("fname") or row_data.get("path") or row_data.get("file")
                    if not file_name and audio_col and isinstance(audio_col, dict):
                        file_name = audio_col.get("path")
                    if not file_name:
                        file_name = f"{lang_name}_{i:06d}"
                        
                    stem = Path(file_name).stem
                    out_path = lang_dir / f"{stem}.wav"
                    
                    if not out_path.exists() and arr is not None:
                        sf.write(str(out_path), arr, sr)

                    if out_path.exists():
                        dur = round(len(arr) / sr, 3) if arr is not None else 0.0
                        target_list.append({
                            "path":           str(out_path.resolve()),
                            "label":          "bonafide",
                            "split":          row_split,
                            "source_dataset": "Kathbath",
                            "speaker_id":     speaker_id,
                            "utterance_id":   stem,
                            "generator_id":   "none",
                            "language":       iso_code,
                            "codec":          "pcm_16k",
                            "duration_s":     dur,
                            "license":        "CC-BY-4.0",
                            "consent":        "yes",
                        })
                        lang_written += 1

                    i += 1
                    if i % 2000 == 0:
                        print(f"      [{i:>6} done] train={len(all_train_rows):,}  eval={len(all_eval_rows):,}", flush=True)
                    if max_per_lang and lang_written >= max_per_lang:
                        break

                except Exception as exc:
                    if i < 5:
                        print(f"      [WARN] Row {i} failed: {exc}")
                    i += 1

    train_csv = eval_csv = None

    if all_train_rows:
        df = pd.DataFrame(all_train_rows, columns=REQUIRED_COLUMNS)
        train_csv = MANIFEST_DIR / f"kathbath{out_suffix}_train.csv"
        df.to_csv(train_csv, index=False)
        print(f"\n  [OK] Train manifest: {train_csv}  ({len(df):,} rows)")

    if all_eval_rows:
        df = pd.DataFrame(all_eval_rows, columns=REQUIRED_COLUMNS)
        eval_csv = MANIFEST_DIR / f"kathbath{out_suffix}_eval.csv"
        df.to_csv(eval_csv, index=False)
        print(f"  [OK] Eval manifest : {eval_csv}  ({len(df):,} rows)")

    return train_csv, eval_csv


def validate_manifest(csv_path: Path, label: str) -> None:
    df     = pd.read_csv(csv_path)
    errors = validate_schema(df)
    if errors:
        print(f"  [FAIL] {label}:")
        for e in errors:
            print(f"    {e}")
    else:
        langs  = df["language"].unique().tolist()
        n_rows = len(df)
        print(f"  [PASS] {label}: {n_rows:,} rows  |  languages={langs}")


def main():
    parser = argparse.ArgumentParser(description="Download AI4Bharat Kathbath and build train/eval manifests.")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_AUDIO_ROOT))
    parser.add_argument("--languages", nargs="*", default=["hi"])
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Append to manifest name, e.g. '_indic4' -> kathbath_indic4_train.csv "
                             "(keeps the Hindi kathbath_train.csv from being overwritten)")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="Only download the first N train shards per language (+valid). "
                             "0 = all shards. Use 1-2 for a small balance set.")
    parser.add_argument("--max-per-lang", type=int, default=0,
                        help="Stop after writing N bonafide clips per language. 0 = no cap.")
    parser.add_argument("--valid-only", action="store_true",
                        help="Download ONLY the small 'valid' split per language (~2.4k clips) "
                             "instead of any train shards — leanest way to get a balance set.")
    args = parser.parse_args()

    audio_root = Path(args.dataset_root)

    print("=" * 65)
    print("Kathbath — AI4Bharat Indic Speech Pipeline")
    print("=" * 65)

    train_csv, eval_csv = download_and_build_manifest(
        audio_root=audio_root,
        languages=args.languages,
        out_suffix=args.out_suffix,
        max_shards=args.max_shards,
        max_per_lang=args.max_per_lang,
        valid_only=args.valid_only,
    )

    print("\n[3/3] Validating manifests ...")
    if train_csv and train_csv.exists():
        validate_manifest(train_csv, "Kathbath train")
    if eval_csv and eval_csv.exists():
        validate_manifest(eval_csv, "Kathbath eval (H's real-only safety set)")

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
