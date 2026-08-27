"""
data_pipeline/fetch_kathbath.py
=================================
Downloads AI4Bharat Kathbath (real Indic speech, 12 Indian languages)
from HuggingFace and builds the 12-column manifest.

Source  : HuggingFace  ai4bharat/Kathbath  (GATED — requires HF login + acceptance)
Size    : ~1,684 hours across 12 Indian languages
Languages: Bengali, Gujarati, Kannada, Hindi, Malayalam, Marathi, Odia,
           Punjabi, Sanskrit, Tamil, Telugu, Urdu
Labels  : ALL rows are "bonafide" — this is a REAL speech corpus, no fakes.

Purpose (from MD spec):
  1. Expose the model to real Indic speech during TRAINING so it doesn't learn
     "English accent = real, Indian accent = suspicious."
  2. Become the "real-only safety set" for Person H → measures false positive
     rate on real Indian speech (MD spec Task H line 170).

HF login: huggingface-cli login     (or HUGGING_FACE_HUB_TOKEN env var)

Audio saved to  : E:\\DatasetSIH\\kathbath\\  (default)
Manifests:
    data_pipeline/manifests/kathbath_train.csv   (train split)
    data_pipeline/manifests/kathbath_eval.csv    (eval split — H's safety set)

Run:
    # Step 1: accept terms at https://huggingface.co/datasets/ai4bharat/Kathbath
    # Step 2: login
    huggingface-cli login

    # Step 3: download (per-language to save space if needed)
    python data_pipeline/fetch_kathbath.py
    python data_pipeline/fetch_kathbath.py --languages hi ta te  # Hindi, Tamil, Telugu only
    python data_pipeline/fetch_kathbath.py --dataset-root F:\\MyData\\kathbath
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from data_pipeline.schema import REQUIRED_COLUMNS, validate_schema

DEFAULT_AUDIO_ROOT = Path(os.environ.get("KATHBATH_ROOT", r"E:\DatasetSIH\kathbath"))
MANIFEST_DIR       = repo_root / "data_pipeline" / "manifests"
HF_DATASET_ID      = "ai4bharat/Kathbath"

# 12 Kathbath languages with ISO 639-1 codes
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


# ---------------------------------------------------------------------------
# Download + build manifest
# ---------------------------------------------------------------------------

def download_and_build_manifest(
    audio_root: Path,
    languages: list,  # list of Kathbath language names or ISO codes
) -> tuple:
    """Returns (train_csv_path, eval_csv_path)."""

    audio_root.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve language filter
    if not languages:
        target_langs = list(KATHBATH_LANGUAGES.keys())
    else:
        # Accept both full names and ISO codes
        iso_to_name = {v: k for k, v in KATHBATH_LANGUAGES.items()}
        target_langs = []
        for l in languages:
            l_lower = l.lower()
            if l_lower in KATHBATH_LANGUAGES:
                target_langs.append(l_lower)
            elif l_lower in iso_to_name:
                target_langs.append(iso_to_name[l_lower])
            else:
                print(f"  [WARN] Unknown language '{l}' — skipping.")

    print(f"\n[1/3] Downloading Kathbath from HuggingFace...")
    print(f"      Dataset   : {HF_DATASET_ID}")
    print(f"      Languages : {target_langs}")
    print(f"      Audio dir : {audio_root}\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("[FAIL] Run: pip install datasets soundfile")
        sys.exit(1)

    try:
        import soundfile as sf
    except ImportError:
        print("[FAIL] Run: pip install soundfile")
        sys.exit(1)

    all_train_rows = []
    all_eval_rows  = []

    for lang_name in target_langs:
        iso_code = KATHBATH_LANGUAGES[lang_name]
        print(f"\n  Language: {lang_name} ({iso_code})")

        try:
            # Kathbath is organised by language as config/subset name
            ds = load_dataset(HF_DATASET_ID, lang_name, trust_remote_code=True)
        except Exception as exc:
            err = str(exc)
            if "gated" in err.lower() or "401" in err or "403" in err:
                print(f"    [FAIL] Gated dataset — HuggingFace login required.")
                print(f"           1. Visit: https://huggingface.co/datasets/{HF_DATASET_ID}")
                print(f"           2. Accept terms, then: huggingface-cli login")
                sys.exit(1)
            elif "not found" in err.lower() or "config" in err.lower():
                print(f"    [WARN] Language config '{lang_name}' not found in HF dataset. Trying 'train' split ...")
                try:
                    ds = load_dataset(HF_DATASET_ID, trust_remote_code=True)
                except Exception as e2:
                    print(f"    [FAIL] {e2}")
                    continue
            else:
                print(f"    [FAIL] {exc}")
                continue

        lang_dir = audio_root / lang_name
        lang_dir.mkdir(exist_ok=True)

        for hf_split_name, hf_split_data in ds.items():
            # Map HF split names to our schema splits
            if "train" in hf_split_name.lower():
                row_split = "train"
                target_list = all_train_rows
            else:
                row_split = "eval"
                target_list = all_eval_rows

            n_this = len(hf_split_data)
            print(f"    {hf_split_name} → split='{row_split}'  {n_this:,} samples")

            for i, item in enumerate(hf_split_data):
                try:
                    audio_info = item.get("audio", {})
                    arr        = audio_info.get("array")
                    sr         = audio_info.get("sampling_rate", 16000)
                    speaker_id = str(item.get("speaker_id", item.get("speaker", f"spk_{i:05d}")))
                    file_name  = str(item.get("path", item.get("file", f"{lang_name}_{i:06d}")))
                    stem       = Path(file_name).stem

                    out_path = lang_dir / f"{stem}.wav"
                    if not out_path.exists() and arr is not None:
                        sf.write(str(out_path), arr, sr)

                    if out_path.exists():
                        target_list.append({
                            "path":           str(out_path.resolve()),
                            "label":          "bonafide",   # Kathbath is ALL real speech
                            "split":          row_split,
                            "source_dataset": "Kathbath",
                            "speaker_id":     speaker_id,
                            "utterance_id":   stem,
                            "generator_id":   "none",       # bonafide, no generator
                            "language":       iso_code,
                            "codec":          "pcm_16k",
                            "duration_s":     round(len(arr) / sr, 3) if arr is not None else 0.0,
                            "license":        "CC-BY-4.0",
                            "consent":        "yes",
                        })

                    if (i + 1) % 1000 == 0:
                        print(f"      [{i+1:>6}/{n_this}] saved {len(target_list):,} rows", flush=True)

                except Exception as exc:
                    if i < 5:
                        print(f"      [WARN] Item {i}: {exc}")

        print(f"    Done — train={len(all_train_rows):,}  eval={len(all_eval_rows):,}")

    # Write manifests
    train_csv = eval_csv = None

    if all_train_rows:
        df = pd.DataFrame(all_train_rows, columns=REQUIRED_COLUMNS)
        train_csv = MANIFEST_DIR / "kathbath_train.csv"
        df.to_csv(train_csv, index=False)
        langs_in = df["language"].unique().tolist()
        print(f"\n  [OK] Train manifest: {train_csv}  ({len(df):,} rows)  languages={langs_in}")

    if all_eval_rows:
        df = pd.DataFrame(all_eval_rows, columns=REQUIRED_COLUMNS)
        eval_csv = MANIFEST_DIR / "kathbath_eval.csv"
        df.to_csv(eval_csv, index=False)
        langs_in = df["language"].unique().tolist()
        print(f"  [OK] Eval manifest : {eval_csv}  ({len(df):,} rows)  languages={langs_in}")

    return train_csv, eval_csv


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

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
        spoof_count = (df["label"] == "spoof").sum()
        print(f"  [PASS] {label}: {n_rows:,} rows  |  {len(langs)} languages  |  "
              f"{spoof_count} spoof rows (should be 0)")
        if spoof_count > 0:
            print(f"  [WARN] Kathbath should contain only bonafide rows!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download AI4Bharat Kathbath and build train/eval manifests."
    )
    parser.add_argument(
        "--dataset-root", type=str, default=str(DEFAULT_AUDIO_ROOT),
        help=f"Directory to save audio files (default: {DEFAULT_AUDIO_ROOT}). Set KATHBATH_ROOT env var."
    )
    parser.add_argument(
        "--languages", nargs="*", default=[],
        help="Specific languages to download. Default: all 12. "
             "Use full names (hindi, tamil) or ISO codes (hi, ta). "
             "Example: --languages hi ta te bn kn"
    )
    args = parser.parse_args()

    audio_root = Path(args.dataset_root)

    print("=" * 65)
    print("Kathbath — AI4Bharat Indic Speech Pipeline")
    print("=" * 65)
    print(f"  Audio root : {audio_root}")
    print(f"  Languages  : {args.languages if args.languages else 'all 12'}")
    print(f"  Note       : ALL labels are bonafide (real speech corpus)")
    print(f"  Purpose    : Indic training exposure + H's real-only safety set")

    if not args.languages:
        print("\n  Available languages:")
        for name, iso in KATHBATH_LANGUAGES.items():
            print(f"    {iso}  {name}")
        print("\n  Tip: Start with --languages hi ta te to test before downloading all.\n")

    train_csv, eval_csv = download_and_build_manifest(
        audio_root=audio_root,
        languages=args.languages,
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
