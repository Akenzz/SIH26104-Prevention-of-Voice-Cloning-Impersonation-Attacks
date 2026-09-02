"""
data_pipeline/evaluate_slices.py
==================================
Per-condition evaluation harness for Task H (SIH26104).

Person H uses this script to produce the results table the pitch deck needs.
Each condition is reported SEPARATELY — never pooled — per MD spec.

Slices produced automatically from a manifest:
  1. Per attack/generator system  (e.g. A07-A19 on ASVspoof eval)
  2. Per source dataset           (within-corpus vs cross-corpus)
  3. Per language                 (one EER per named language)
  4. Real-only safety set         (bonafide rows only → false positive rate)
  5. Full-set baseline            (pooled, for sanity check only)

Usage:
    python data_pipeline/evaluate_slices.py \\
        --checkpoint  lfcc-detector/checkpoints/best_lfcc_lcnn.pth \\
        --manifests   data_pipeline/manifests/asvspoof19_eval.csv \\
                      data_pipeline/manifests/in_the_wild_eval_ood.csv \\
        --output-dir  data_pipeline/reports/sliced_eval \\
        --model-name  LFCC-LCNN

Outputs (in --output-dir):
    sliced_eval_results.json   Machine-readable full results
    sliced_eval_report.md      Human-readable table for the pitch slide
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

lfcc_dir = repo_root / "lfcc-detector"
if str(lfcc_dir) not in sys.path:
    sys.path.insert(0, str(lfcc_dir))

from data_pipeline.loader import ManifestAudioDataset
from data_pipeline.evaluation.adapter import ModelAdapter
from data_pipeline.evaluation.metrics import compute_full_suite
from models.detector import LFCCLCNNDetector


# ---------------------------------------------------------------------------
# Core: score a sub-slice of a manifest
# ---------------------------------------------------------------------------

def score_slice(
    df_slice: pd.DataFrame,
    adapter: ModelAdapter,
    sample_rate: int = 16000,
    window_sec: float = 4.0,
) -> Optional[dict]:
    """
    Run inference on a pandas DataFrame slice and return metric dict.
    Returns None if slice has < 2 samples or only one class.
    """
    if len(df_slice) < 2:
        return None

    labels_present = df_slice["label"].unique()
    if len(labels_present) < 2:
        # Real-only or spoof-only slice → compute FPR / FNR instead of EER
        return _score_single_class(df_slice, adapter, sample_rate, window_sec)

    bonafide_scores, spoof_scores, latencies = [], [], []
    window_samples = int(window_sec * sample_rate)

    import torchaudio, soundfile as sf

    for _, row in df_slice.iterrows():
        audio_path = str(row["path"])
        try:
            try:
                audio_np, sr = sf.read(audio_path, dtype="float32")
                audio = torch.from_numpy(audio_np)
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)
            except Exception:
                audio, sr = torchaudio.load(audio_path)

            if sr != sample_rate:
                audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            audio = audio.squeeze(0)

            # Centre-crop / pad
            if audio.shape[0] < window_samples:
                audio = torch.nn.functional.pad(audio, (0, window_samples - audio.shape[0]))
            elif audio.shape[0] > window_samples:
                start = (audio.shape[0] - window_samples) // 2
                audio = audio[start: start + window_samples]

            logit, _, lat = adapter.predict_window(audio.numpy())
            latencies.append(lat)

            if row["label"] == "bonafide":
                bonafide_scores.append(logit)
            else:
                spoof_scores.append(logit)

        except Exception as exc:
            print(f"  [WARN] Could not load {audio_path}: {exc}")
            continue

    if not bonafide_scores or not spoof_scores:
        return None

    metrics = compute_full_suite(np.array(bonafide_scores), np.array(spoof_scores))
    metrics["num_bonafide"] = len(bonafide_scores)
    metrics["num_spoof"]    = len(spoof_scores)
    metrics["mean_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
    return metrics


def _score_single_class(df_slice, adapter, sample_rate, window_sec):
    """For real-only or spoof-only slices: compute error rate at EER threshold."""
    import torchaudio, soundfile as sf
    window_samples = int(window_sec * sample_rate)
    scores, is_spoof = [], []

    for _, row in df_slice.iterrows():
        try:
            try:
                audio_np, sr = sf.read(str(row["path"]), dtype="float32")
                audio = torch.from_numpy(audio_np)
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)
            except Exception:
                audio, sr = torchaudio.load(str(row["path"]))
            if sr != sample_rate:
                audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            audio = audio.squeeze(0)
            if audio.shape[0] < window_samples:
                audio = torch.nn.functional.pad(audio, (0, window_samples - audio.shape[0]))
            elif audio.shape[0] > window_samples:
                start = (audio.shape[0] - window_samples) // 2
                audio = audio[start: start + window_samples]
            logit, _, _ = adapter.predict_window(audio.numpy())
            scores.append(logit)
            is_spoof.append(row["label"] == "spoof")
        except Exception:
            continue

    if not scores:
        return None

    n = len(scores)
    label_str = "spoof" if is_spoof[0] else "bonafide"
    return {
        "single_class": label_str,
        "n_samples": n,
        "mean_logit": float(np.mean(scores)),
        "std_logit":  float(np.std(scores)),
        # Bonafide-only slices report false positives; spoof-only slices report
        # false negatives.  Keeping the type explicit prevents a markdown report
        # from accidentally presenting one as the other.
        "error_rate_at_zero_threshold": float(np.mean(np.array(scores) > 0)) if label_str == "bonafide"
                                        else float(np.mean(np.array(scores) <= 0)),
        "error_type": "FP" if label_str == "bonafide" else "FN",
    }


# ---------------------------------------------------------------------------
# Slice runner
# ---------------------------------------------------------------------------

def run_sliced_evaluation(
    manifest_paths: list,
    adapter: ModelAdapter,
    output_dir: Path,
    model_name: str,
) -> dict:

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all manifests into one frame (eval / eval_ood only — never train or dev)
    frames = []
    for p in manifest_paths:
        df = pd.read_csv(p)
        eval_rows = df[df["split"].isin(["eval", "eval_ood"])]
        if len(eval_rows) == 0:
            print(f"  [WARN] {Path(p).name}: no eval/eval_ood rows found — skipping.")
        else:
            frames.append(eval_rows)
            print(f"  Loaded {len(eval_rows):,} eval rows from {Path(p).name}")

    if not frames:
        print("[FAIL] No evaluation data found in any manifest.")
        sys.exit(1)

    all_eval = pd.concat(frames, ignore_index=True)
    print(f"  Total eval rows: {len(all_eval):,}\n")

    results = {}

    # ------------------------------------------------------------------
    # Slice 1: Full baseline (pooled — sanity check only)
    # ------------------------------------------------------------------
    print("[1/6] Full baseline (pooled) ...")
    m = score_slice(all_eval, adapter)
    results["full_pooled"] = {"description": "All eval rows pooled (sanity check)", "metrics": m}
    if m:
        print(f"       EER: {m['eer']*100:.2f}%  ROC-AUC: {m['roc_auc']:.4f}")

    # ------------------------------------------------------------------
    # Slice 2: Per source dataset (within-corpus / cross-corpus)
    # ------------------------------------------------------------------
    print("\n[2/6] Per source dataset ...")
    results["per_dataset"] = {}
    for ds_name, grp in all_eval.groupby("source_dataset"):
        print(f"       {ds_name} ({len(grp):,} rows) ...")
        m = score_slice(grp, adapter)
        results["per_dataset"][ds_name] = m
        if m and "eer" in m:
            print(f"         EER: {m['eer']*100:.2f}%  n_bonafide={m.get('num_bonafide',0)}  n_spoof={m.get('num_spoof',0)}")

    # ------------------------------------------------------------------
    # Slice 3: Per generator / attack system
    # ------------------------------------------------------------------
    print("\n[3/6] Per generator/attack system ...")
    results["per_generator"] = {}
    for gen_id, grp in all_eval.groupby("generator_id"):
        if gen_id in ("none", "-", ""):
            continue    # bonafide group — not an attack system
        m = score_slice(grp, adapter)
        results["per_generator"][gen_id] = m
        if m and "eer" in m:
            print(f"       {gen_id:6s}: EER={m['eer']*100:.2f}%  ({m.get('num_spoof',0)} spoof samples)")

    # ------------------------------------------------------------------
    # Slice 4: Per language
    # ------------------------------------------------------------------
    print("\n[4/6] Per language ...")
    results["per_language"] = {}
    for lang, grp in all_eval.groupby("language"):
        m = score_slice(grp, adapter)
        results["per_language"][lang] = m
        if m:
            if "eer" in m:
                print(f"       {lang}: EER={m['eer']*100:.2f}%  ({len(grp):,} samples)")
            elif "single_class" in m:
                print(f"       {lang}: {m['single_class']}-only  FP/FN rate={m['error_rate_at_zero_threshold']*100:.1f}%")

    # ------------------------------------------------------------------
    # Slice 5: OOD only (In-the-Wild, MLAAD held-out, etc.)
    # ------------------------------------------------------------------
    ood_rows = all_eval[all_eval["split"] == "eval_ood"]
    print(f"\n[5/6] OOD cross-corpus (eval_ood rows: {len(ood_rows):,}) ...")
    if len(ood_rows) > 0:
        m = score_slice(ood_rows, adapter)
        results["ood_cross_corpus"] = m
        if m and "eer" in m:
            print(f"       EER: {m['eer']*100:.2f}%  ROC-AUC: {m['roc_auc']:.4f}")
    else:
        results["ood_cross_corpus"] = None
        print("       [SKIP] No eval_ood rows yet. Add In-the-Wild / MLAAD manifests.")

    # ------------------------------------------------------------------
    # Slice 6: Real-only safety set (false positive rate)
    # ------------------------------------------------------------------
    real_only = all_eval[all_eval["label"] == "bonafide"]
    print(f"\n[6/6] Real-only safety set ({len(real_only):,} bonafide samples) ...")
    m = score_slice(real_only, adapter)
    results["real_only_safety_set"] = m
    if m:
        if "error_rate_at_zero_threshold" in m:
            print(f"       FP rate (threshold=0): {m['error_rate_at_zero_threshold']*100:.2f}%")
        elif "eer" in m:
            # If there are somehow spoof rows in the bonafide slice
            print(f"       EER: {m['eer']*100:.2f}%")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    json_path = output_dir / "sliced_eval_results.json"
    with open(json_path, "w") as f:
        json.dump({"model": model_name, "slices": results}, f, indent=2, default=str)
    print(f"\n[OK] JSON results: {json_path}")

    # ------------------------------------------------------------------
    # Generate Markdown table (Person H's pitch slide)
    # ------------------------------------------------------------------
    md_path = output_dir / "sliced_eval_report.md"
    _write_markdown_report(results, md_path, model_name)
    print(f"[OK] Markdown report: {md_path}")

    return results


def _write_markdown_report(results: dict, path: Path, model_name: str):
    lines = [
        f"# Sliced Evaluation Report: {model_name}",
        "",
        "> Generated by `data_pipeline/evaluate_slices.py`  ",
        "> Each condition is reported separately — never pooled (MD spec requirement).",
        "",
    ]

    def eer_row(label, m):
        if m is None:
            return f"| {label} | — | — | — | No data |"
        if "eer" in m:
            return (f"| {label} | {m['eer']*100:.2f}% | {m['roc_auc']:.4f} | "
                    f"{m.get('num_bonafide',0):,} / {m.get('num_spoof',0):,} | |")
        if "error_rate_at_zero_threshold" in m:
            single_class = m.get("single_class", "unknown")
            error_type = m.get("error_type", "FP" if single_class == "bonafide" else "FN")
            return (f"| {label} | {error_type}={m['error_rate_at_zero_threshold']*100:.2f}% | — | "
                    f"{m.get('n_samples',0):,} {single_class} | single-class |")
        return f"| {label} | — | — | — | |"

    header = "| Condition | EER | ROC-AUC | Bonafide / Spoof | Notes |"
    sep    = "|-----------|-----|---------|------------------|-------|"

    lines += ["## Per Source Dataset (Within-corpus vs Cross-corpus)", "", header, sep]
    for ds, m in results.get("per_dataset", {}).items():
        lines.append(eer_row(ds, m))

    lines += ["", "## Per Generator / Attack System", "", header, sep]
    for gen, m in results.get("per_generator", {}).items():
        lines.append(eer_row(gen, m))

    lines += ["", "## Per Language", "", header, sep]
    for lang, m in results.get("per_language", {}).items():
        lines.append(eer_row(lang, m))

    lines += ["", "## OOD Cross-Corpus (In-the-Wild / MLAAD held-out)", "", header, sep]
    lines.append(eer_row("OOD (all eval_ood)", results.get("ood_cross_corpus")))

    lines += ["", "## Real-Only Safety Set (False Positive Rate)", "", header, sep]
    lines.append(eer_row("Real speech only", results.get("real_only_safety_set")))

    lines += [
        "",
        "---",
        "",
        "> **Claims ledger note:** Only report numbers from this table in the pitch deck.",
        "> Never pool across conditions. A strong average that hides a failed language",
        "> is a claim you cannot defend in Q&A (MD spec, Task H).",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-condition sliced evaluation for Task H."
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="lfcc-detector/checkpoints/best_lfcc_lcnn.pth",
        help="Path to model checkpoint (.pth)."
    )
    parser.add_argument(
        "--manifests", nargs="+",
        default=["data_pipeline/manifests/asvspoof19_eval.csv"],
        help="One or more manifest CSVs to evaluate against. eval/eval_ood rows only."
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data_pipeline/reports/sliced_eval",
        help="Directory to write results JSON and Markdown report."
    )
    parser.add_argument(
        "--model-name", type=str, default="LFCC-LCNN",
        help="Model name for report headers."
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"SLICED EVALUATION: {args.model_name}")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Manifests  : {args.manifests}")
    print(f"  Output dir : {args.output_dir}\n")

    detector = LFCCLCNNDetector(
        checkpoint_path=args.checkpoint if Path(args.checkpoint).exists() else None
    )
    adapter = ModelAdapter(detector, name=args.model_name, version=detector.model_version)

    run_sliced_evaluation(
        manifest_paths=args.manifests,
        adapter=adapter,
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
