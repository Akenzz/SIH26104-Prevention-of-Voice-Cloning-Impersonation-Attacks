"""
data_pipeline/fetch_mlaad.py
==============================
Downloads MLAAD (Multi-Language Audio Anti-Spoofing Dataset) from HuggingFace
and converts it to the standard 12-column manifest format.

Source  : HuggingFace  mueller91/MLAAD   (GATED — requires HF login + acceptance)
          MLAAD-tiny   mueller91/MLAAD-tiny  (~3.54 GB, open, good for prototyping)
Size    : Full MLAAD: 1000+ hours, 50+ languages, 175+ TTS systems (very large)
          MLAAD-tiny : ~3.54 GB  ← default, recommended first

HF login: huggingface-cli login     (or set env var HUGGING_FACE_HUB_TOKEN)

Held-out generators (MD spec, Task H "unseen generator" slice):
    The MD requires at least 1-2 TTS systems to be held out of training entirely
    and used only for H's "unseen generator" test. These get split="eval_ood".
    Default held-out: VITS, YourTTS (most capable, best test of generalization).
    Override with --held-out-generators.

Audio saved to  : E:\\DatasetSIH\\mlaad\\  (default)
Manifests:
    data_pipeline/manifests/mlaad_train.csv      (training-eligible rows)
    data_pipeline/manifests/mlaad_eval_ood.csv   (held-out TTS systems)

Run:
    # Tiny version (recommended first run):
    python data_pipeline/fetch_mlaad.py --use-tiny

    # Full version (needs HF login + gated access approval):
    huggingface-cli login
    python data_pipeline/fetch_mlaad.py

    # Custom held-out generators:
    python data_pipeline/fetch_mlaad.py --use-tiny --held-out-generators VITS YourTTS
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from data_pipeline.schema import REQUIRED_COLUMNS, validate_schema

DEFAULT_AUDIO_ROOT    = Path(os.environ.get("MLAAD_ROOT", r"E:\DatasetSIH\mlaad"))
MANIFEST_DIR          = repo_root / "data_pipeline" / "manifests"

HF_DATASET_FULL       = "mueller91/MLAAD"
HF_DATASET_TINY       = "mueller91/MLAAD-tiny"

# Default TTS systems held out of training (MD spec: "at least one or two")
DEFAULT_HELD_OUT_GENERATORS = {"VITS", "YourTTS", "vits", "yourtts"}


# ---------------------------------------------------------------------------
# Download + build manifest in one streaming pass
# ---------------------------------------------------------------------------

def download_and_build_manifest(
    audio_root: Path,
    use_tiny: bool,
    held_out_generators: set,
) -> tuple:
    """
    Returns (train_csv_path, eval_ood_csv_path).
    """
    audio_root.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    hf_id = HF_DATASET_TINY if use_tiny else HF_DATASET_FULL
    size_note = "~3.54 GB (tiny)" if use_tiny else "1000+ hours (full — needs HF login)"

    print(f"\n[1/3] Downloading MLAAD from HuggingFace...")
    print(f"      Dataset : {hf_id}")
    print(f"      Size    : {size_note}")
    print(f"      Audio   : {audio_root}")
    print(f"      Held-out (eval_ood): {sorted(held_out_generators)}\n")

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

    try:
        ds = load_dataset(hf_id, trust_remote_code=True)
    except Exception as exc:
        if "gated" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
            print("\n[FAIL] MLAAD is a gated dataset on HuggingFace.")
            print("       Steps to access:")
            print("       1. Create account at https://huggingface.co")
            print(f"       2. Visit https://huggingface.co/datasets/{hf_id}")
            print("          and accept the terms of use.")
            print("       3. Run: huggingface-cli login")
            print("       4. Re-run this script.")
            print("\n       Alternatively: use MLAAD-tiny (open access):")
            print("       python data_pipeline/fetch_mlaad.py --use-tiny")
        else:
            print(f"[FAIL] {exc}")
        sys.exit(1)

    split_name = list(ds.keys())[0]
    data       = ds[split_name]
    total      = len(data)
    print(f"      {total:,} samples in HuggingFace split '{split_name}'\n")

    train_rows   = []
    eval_ood_rows = []
    failed       = 0

    for i, item in enumerate(data):
        try:
            audio_info = item.get("audio", {})
            arr        = audio_info.get("array")
            sr         = audio_info.get("sampling_rate", 16000)

            # MLAAD field names (may vary between full and tiny)
            language   = str(item.get("language", item.get("lang", "unknown")))
            tts_system = str(item.get("tts_system", item.get("system", item.get("generator", "unknown"))))
            file_name  = str(item.get("file",   item.get("path", f"{i:06d}")))
            speaker_id = str(item.get("speaker", f"spk_{i:05d}"))

            # MLAAD is ALL spoof — no bonafide samples
            label  = "spoof"
            gen_id = tts_system

            # Determine split: held-out generators → eval_ood
            is_held_out = any(h.lower() in tts_system.lower() for h in held_out_generators)
            row_split   = "eval_ood" if is_held_out else "train"

            # Save audio
            lang_dir = audio_root / language / tts_system
            lang_dir.mkdir(parents=True, exist_ok=True)
            stem     = Path(file_name).stem
            out_path = lang_dir / f"{stem}.wav"

            if not out_path.exists() and arr is not None:
                sf.write(str(out_path), arr, sr)

            if out_path.exists():
                row = {
                    "path":           str(out_path.resolve()),
                    "label":          label,
                    "split":          row_split,
                    "source_dataset": "MLAAD-tiny" if use_tiny else "MLAAD",
                    "speaker_id":     speaker_id,
                    "utterance_id":   stem,
                    "generator_id":   gen_id,
                    "language":       language,
                    "codec":          "pcm_16k",
                    "duration_s":     round(len(arr) / sr, 3) if arr is not None else 0.0,
                    "license":        "CC-BY-NC-4.0",
                    "consent":        "no",   # synthetic speech
                }
                if row_split == "train":
                    train_rows.append(row)
                else:
                    eval_ood_rows.append(row)

            if (i + 1) % 500 == 0:
                print(f"  [{i+1:>6}/{total}]  train={len(train_rows):,}  eval_ood={len(eval_ood_rows):,}",
                      flush=True)

        except Exception as exc:
            failed += 1
            if failed <= 5:
                print(f"  [WARN] Item {i}: {exc}")

    # Write manifests
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


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download MLAAD and build train/eval_ood manifests.")
    parser.add_argument(
        "--dataset-root", type=str, default=str(DEFAULT_AUDIO_ROOT),
        help=f"Directory to save audio (default: {DEFAULT_AUDIO_ROOT}). Set MLAAD_ROOT env var."
    )
    parser.add_argument(
        "--use-tiny", action="store_true",
        help="Use MLAAD-tiny (~3.54 GB, open access) instead of full MLAAD (gated, 1000+ hours)."
    )
    parser.add_argument(
        "--held-out-generators", nargs="+",
        default=list(DEFAULT_HELD_OUT_GENERATORS),
        help="TTS system names to hold out of training (split=eval_ood). "
             f"Default: {sorted(DEFAULT_HELD_OUT_GENERATORS)}"
    )
    args = parser.parse_args()

    audio_root      = Path(args.dataset_root)
    held_out        = set(args.held_out_generators)

    print("=" * 65)
    print("MLAAD — Multilingual Deepfake Dataset Pipeline")
    print("=" * 65)
    print(f"  Audio root  : {audio_root}")
    print(f"  Mode        : {'MLAAD-tiny (open)' if args.use_tiny else 'Full MLAAD (gated)'}")
    print(f"  Held-out    : {sorted(held_out)}  → split=eval_ood (H's unseen-generator slice)")

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
