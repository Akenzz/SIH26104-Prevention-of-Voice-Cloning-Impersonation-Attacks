"""
ASVspoof 2019 LA — Local Dataset Manifest Builder & Pipeline Runner (Task B).

The dataset is assumed to be already downloaded at D:/DatasetSIH/LA (or
overridden via the --dataset-root CLI argument / ASVSPOOF_LA_ROOT env var).

Steps performed:
  1. Validates local dataset directory structure.
  2. Generates official 12-column manifests (train / dev / eval) under
     data_pipeline/manifests/.
  3. Produces a combined manifest and runs the Data Leakage Audit via
     check_leakage.py.
  4. Trains the LFCC-LCNN detector on the real train/dev splits.
  5. Benchmarks the trained checkpoint against the real eval split using
     run_experiment.py.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap: ensure repo root is importable regardless of CWD
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from data_pipeline.convert_asvspoof import convert_asvspoof2019_protocol
from data_pipeline.check_leakage import LeakageChecker

# ---------------------------------------------------------------------------
# Defaults (overridable via env var or CLI)
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = Path(os.environ.get("ASVSPOOF_LA_ROOT", r"D:\DatasetSIH\LA"))
MANIFEST_DIR = repo_root / "data_pipeline" / "manifests"
REPORTS_DIR  = repo_root / "data_pipeline" / "reports"
CHECKPOINT_DIR = repo_root / "lfcc-detector" / "checkpoints"

# Protocol and audio sub-directory names as they exist in the official release
SPLITS = [
    {
        "name":       "train",
        "proto_name": "ASVspoof2019.LA.cm.train.trn.txt",
        "audio_dir":  "ASVspoof2019_LA_train/flac",
    },
    {
        "name":       "dev",
        "proto_name": "ASVspoof2019.LA.cm.dev.trl.txt",
        "audio_dir":  "ASVspoof2019_LA_dev/flac",
    },
    {
        "name":       "eval",
        "proto_name": "ASVspoof2019.LA.cm.eval.trl.txt",
        "audio_dir":  "ASVspoof2019_LA_eval/flac",
    },
]

PROTOCOL_SUBDIR = "ASVspoof2019_LA_cm_protocols"


# ---------------------------------------------------------------------------
# Step 1 — Validate local dataset root
# ---------------------------------------------------------------------------
def validate_dataset_root(dataset_root: Path) -> bool:
    """
    Check that the expected sub-directories and protocol files exist.
    Returns True on success, prints diagnostics and returns False on failure.
    """
    ok = True

    proto_dir = dataset_root / PROTOCOL_SUBDIR
    if not proto_dir.is_dir():
        print(f"[FAIL] Protocol directory not found: {proto_dir}")
        ok = False
    else:
        for split in SPLITS:
            proto_file = proto_dir / split["proto_name"]
            if not proto_file.is_file():
                print(f"[FAIL] Protocol file missing: {proto_file}")
                ok = False

    for split in SPLITS:
        audio_dir = dataset_root / split["audio_dir"]
        if not audio_dir.is_dir():
            print(f"[WARN] Audio directory not found: {audio_dir}  (split '{split['name']}' will be skipped)")

    return ok


# ---------------------------------------------------------------------------
# Step 2 — Generate official manifests
# ---------------------------------------------------------------------------
def generate_manifests(dataset_root: Path) -> dict[str, Path]:
    """
    Convert each split's protocol file into a 12-column manifest CSV.

    Returns a dict  {split_name -> output_csv_path}  for every split that
    was successfully processed.
    """
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    proto_dir = dataset_root / PROTOCOL_SUBDIR

    manifest_paths: dict[str, Path] = {}

    print("\n" + "=" * 65)
    print("STEP 2 — Generating Official Manifests")
    print("=" * 65)

    for split in SPLITS:
        split_name = split["name"]
        proto_file = proto_dir / split["proto_name"]
        audio_dir  = dataset_root / split["audio_dir"]

        if not proto_file.is_file():
            print(f"[SKIP] Protocol file missing for split '{split_name}': {proto_file}")
            continue
        if not audio_dir.is_dir():
            print(f"[SKIP] Audio directory missing for split '{split_name}': {audio_dir}")
            continue

        output_csv = MANIFEST_DIR / f"asvspoof19_{split_name}.csv"

        print(f"\n  Processing '{split_name}' split...")
        print(f"    Protocol : {proto_file.name}")
        print(f"    Audio dir: {audio_dir}")

        convert_asvspoof2019_protocol(
            protocol_path=str(proto_file),
            audio_dir=str(audio_dir),
            output_csv=str(output_csv),
            split=split_name,
            source_dataset="ASVspoof2019_LA",
        )
        manifest_paths[split_name] = output_csv

    return manifest_paths


# ---------------------------------------------------------------------------
# Step 3 — Data Leakage Audit
# ---------------------------------------------------------------------------
def run_leakage_audit(manifest_paths: dict[str, Path]) -> bool:
    """
    Concatenate all per-split manifests into a single combined CSV, then run
    LeakageChecker on it.  Returns True if audit passes (zero leakage).
    """
    if len(manifest_paths) < 2:
        print("\n[SKIP] Leakage audit requires at least 2 splits. Skipping.")
        return True

    print("\n" + "=" * 65)
    print("STEP 3 — Data Leakage Audit")
    print("=" * 65)

    combined_csv = MANIFEST_DIR / "asvspoof19_combined.csv"
    frames = [pd.read_csv(p) for p in manifest_paths.values()]
    combined_df = pd.concat(frames, ignore_index=True)
    combined_df.to_csv(combined_csv, index=False)
    print(f"  Combined manifest written to: {combined_csv}")
    print(f"  Total rows: {len(combined_df)}")

    checker = LeakageChecker(str(combined_csv))
    passed, report = checker.check_all()

    print("\n  " + "-" * 60)
    print("  DATA LEAKAGE AUDIT REPORT")
    print("  " + "-" * 60)
    if passed:
        print("  [PASS] Zero data leakage detected across splits!")
        for split_name in ["train", "dev", "eval"]:
            count = len(combined_df[combined_df["split"] == split_name])
            if count:
                print(f"    {split_name:5s} samples : {count:,}")
    else:
        print(f"  [FAIL] Data leakage violations detected ({len(report)} issues):")
        for err in report:
            print(f"    {err}")

    return passed


# ---------------------------------------------------------------------------
# Step 4 — Train LFCC-LCNN on real data
# ---------------------------------------------------------------------------
def run_training(manifest_paths: dict[str, Path], epochs: int, batch_size: int) -> Path | None:
    """
    Invoke lfcc-detector/training/train.py as a subprocess using the real
    train and dev manifests.  Returns the checkpoint path on success.
    """
    if "train" not in manifest_paths or "dev" not in manifest_paths:
        print("\n[SKIP] Training requires both 'train' and 'dev' manifests.")
        return None

    print("\n" + "=" * 65)
    print("STEP 4 — Training LFCC-LCNN on Real ASVspoof 2019 LA Data")
    print("=" * 65)

    train_script = repo_root / "lfcc-detector" / "training" / "train.py"
    checkpoint_dir = repo_root / "lfcc-detector" / "checkpoints"
    checkpoint_path = checkpoint_dir / "best_lfcc_lcnn.pth"

    cmd = [
        sys.executable, str(train_script),
        "--train-manifest", str(manifest_paths["train"]),
        "--dev-manifest",   str(manifest_paths["dev"]),
        "--output-dir",     str(checkpoint_dir),
        "--epochs",         str(epochs),
        "--batch-size",     str(batch_size),
    ]

    print(f"  Running: {' '.join(cmd)}\n")

    # PYTHONUNBUFFERED=1 ensures print() inside the subprocess flushes
    # immediately to the terminal without block-buffering.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(cmd, cwd=str(repo_root), env=env)

    if result.returncode != 0:
        print(f"\n[FAIL] Training exited with code {result.returncode}.")
        return None

    if checkpoint_path.exists():
        print(f"\n[OK] Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    else:
        print(f"\n[WARN] Training completed but checkpoint not found at {checkpoint_path}.")
        return None


# ---------------------------------------------------------------------------
# Step 5 — Benchmark against eval split
# ---------------------------------------------------------------------------
def run_benchmark(manifest_paths: dict[str, Path], checkpoint_path: Path | None) -> None:
    """
    Invoke data_pipeline/run_experiment.py against the eval manifest.
    Uses the best checkpoint produced by training, or a dummy run if no
    checkpoint exists (will still exercise the full pipeline).
    """
    if "eval" not in manifest_paths:
        print("\n[SKIP] Benchmarking requires an 'eval' manifest.")
        return

    print("\n" + "=" * 65)
    print("STEP 5 — Benchmarking LFCC-LCNN on ASVspoof 2019 LA Eval")
    print("=" * 65)

    experiment_script = repo_root / "data_pipeline" / "run_experiment.py"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_base = REPORTS_DIR / "lfcc_lcnn_asvspoof19_eval"

    ckpt_arg = str(checkpoint_path) if checkpoint_path and checkpoint_path.exists() \
               else "lfcc-detector/checkpoints/best_lfcc_lcnn.pth"

    cmd = [
        sys.executable, str(experiment_script),
        "--checkpoint",     ckpt_arg,
        "--manifest",       str(manifest_paths["eval"]),
        "--output-report",  str(report_base),
        "--split",          "eval",
    ]

    print(f"  Running: {' '.join(cmd)}\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(cmd, cwd=str(repo_root), env=env)

    if result.returncode != 0:
        print(f"\n[FAIL] Experiment runner exited with code {result.returncode}.")
    else:
        json_report = Path(str(report_base) + ".json")
        md_report   = Path(str(report_base) + ".md")
        print(f"\n[OK] Benchmark reports written:")
        if json_report.exists():
            print(f"    JSON : {json_report}")
        if md_report.exists():
            print(f"    MD   : {md_report}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build ASVspoof 2019 LA manifests from a local download, audit for "
            "data leakage, then train and benchmark the LFCC-LCNN detector."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help=(
            f"Path to the ASVspoof 2019 LA root directory "
            f"(default: {DEFAULT_DATASET_ROOT}). "
            "Can also be set via the ASVSPOOF_LA_ROOT environment variable."
        ),
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the LFCC-LCNN training step (manifests + audit still run).",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the benchmark / experiment step.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    print("=" * 65)
    print("ASVspoof 2019 LA — Local Pipeline")
    print("=" * 65)
    print(f"  Dataset root : {dataset_root}")
    print(f"  Manifest dir : {MANIFEST_DIR}")
    print(f"  Reports dir  : {REPORTS_DIR}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")

    # ------------------------------------------------------------------
    # Step 1 — Validate
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 1 — Validating Local Dataset Root")
    print("=" * 65)

    if not dataset_root.is_dir():
        print(
            f"[FAIL] Dataset root does not exist: {dataset_root}\n"
            f"       Set --dataset-root or the ASVSPOOF_LA_ROOT environment "
            f"variable to the correct path."
        )
        sys.exit(1)

    dataset_ok = validate_dataset_root(dataset_root)
    if not dataset_ok:
        print(
            "\n[FAIL] Dataset validation failed. "
            "Please check the paths above and re-run."
        )
        sys.exit(1)

    print("  [OK] Dataset root validated successfully.")

    # ------------------------------------------------------------------
    # Step 2 — Generate manifests
    # ------------------------------------------------------------------
    manifest_paths = generate_manifests(dataset_root)

    if not manifest_paths:
        print("\n[FAIL] No manifests were generated. Aborting.")
        sys.exit(1)

    print(f"\n  [OK] Manifests generated for splits: {list(manifest_paths.keys())}")

    # ------------------------------------------------------------------
    # Step 3 — Leakage audit
    # ------------------------------------------------------------------
    audit_passed = run_leakage_audit(manifest_paths)
    if not audit_passed:
        print(
            "\n[WARN] Leakage audit failed. Proceeding is NOT recommended.\n"
            "       Inspect the report above, then re-run after fixing the dataset."
        )
        # Do not hard-exit here — let the user decide by inspecting output.

    # ------------------------------------------------------------------
    # Step 4 — Training
    # ------------------------------------------------------------------
    checkpoint_path: Path | None = None
    if not args.skip_training:
        checkpoint_path = run_training(manifest_paths, args.epochs, args.batch_size)
    else:
        print("\n[SKIP] Training skipped (--skip-training flag set).")
        # Still point to checkpoint if it already exists
        existing = repo_root / "lfcc-detector" / "checkpoints" / "best_lfcc_lcnn.pth"
        if existing.exists():
            checkpoint_path = existing
            print(f"  Using existing checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------------
    # Step 5 — Benchmark
    # ------------------------------------------------------------------
    if not args.skip_benchmark:
        run_benchmark(manifest_paths, checkpoint_path)
    else:
        print("\n[SKIP] Benchmarking skipped (--skip-benchmark flag set).")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)
    print("\nSummary:")
    for split_name, csv_path in manifest_paths.items():
        df = pd.read_csv(csv_path)
        n_bonafide = (df["label"] == "bonafide").sum()
        n_spoof    = (df["label"] == "spoof").sum()
        print(f"  {split_name:5s}: {len(df):,} total  ({n_bonafide:,} bonafide, {n_spoof:,} spoof)")
    if checkpoint_path and checkpoint_path.exists():
        print(f"\n  Checkpoint : {checkpoint_path}")
    print(f"  Manifests  : {MANIFEST_DIR}")
    print(f"  Reports    : {REPORTS_DIR}")


if __name__ == "__main__":
    main()