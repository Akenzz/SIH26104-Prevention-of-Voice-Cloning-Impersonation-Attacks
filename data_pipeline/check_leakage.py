"""
Data Leakage Prevention Checker for Voice Cloning Manifests.

Strictly verifies that:
  1. No speaker ID in 'dev' or 'eval' appears in 'train'.
  2. No source utterance ID in 'dev' or 'eval' appears in 'train'.
  3. No generator ID marked as 'held-out' appears in 'train'.
  4. Audio file paths are unique across all splits.
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from schema import validate_schema
except ImportError:
    from data_pipeline.schema import validate_schema


class LeakageChecker:
    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)

        # Validate schema first
        errors = validate_schema(self.df)
        if errors:
            raise ValueError(f"Schema validation failed for {manifest_path}:\n" + "\n".join(errors))

    def check_all(self, held_out_generators: List[str] = None) -> Tuple[bool, List[str]]:
        """
        Run complete data leakage check suite.

        Returns:
            passed (bool): True if zero leakage detected.
            report (List[str]): Failure log messages.
        """
        report = []
        train_df = self.df[self.df['split'] == 'train']
        dev_df = self.df[self.df['split'] == 'dev']
        eval_df = self.df[self.df['split'] == 'eval']

        # 1. Path Uniqueness Check
        all_paths = self.df['path'].tolist()
        if len(all_paths) != len(set(all_paths)):
            duplicates = self.df[self.df.duplicated(subset=['path'])]['path'].tolist()
            report.append(f"[FAIL] Duplicate audio file paths found ({len(duplicates)} duplicates): {duplicates[:5]}")

        # 2. Speaker Leakage Check
        train_spks = set(train_df['speaker_id'].unique())
        dev_spks = set(dev_df['speaker_id'].unique())
        eval_spks = set(eval_df['speaker_id'].unique())

        dev_spk_leak = train_spks.intersection(dev_spks)
        if dev_spk_leak:
            report.append(f"[FAIL] Speaker leakage detected between train and dev ({len(dev_spk_leak)} speakers): {list(dev_spk_leak)[:5]}")

        eval_spk_leak = train_spks.intersection(eval_spks)
        if eval_spk_leak:
            report.append(f"[FAIL] Speaker leakage detected between train and eval ({len(eval_spk_leak)} speakers): {list(eval_spk_leak)[:5]}")

        # 3. Utterance Leakage Check
        train_utts = set(train_df['utterance_id'].unique())
        dev_utts = set(dev_df['utterance_id'].unique())
        eval_utts = set(eval_df['utterance_id'].unique())

        dev_utt_leak = train_utts.intersection(dev_utts)
        if dev_utt_leak:
            report.append(f"[FAIL] Utterance leakage detected between train and dev ({len(dev_utt_leak)} utterances): {list(dev_utt_leak)[:5]}")

        eval_utt_leak = train_utts.intersection(eval_utts)
        if eval_utt_leak:
            report.append(f"[FAIL] Utterance leakage detected between train and eval ({len(eval_utt_leak)} utterances): {list(eval_utt_leak)[:5]}")

        # 4. Held-Out Generator Leakage Check
        if held_out_generators:
            train_gens = set(train_df['generator_id'].unique())
            for gen in held_out_generators:
                if gen in train_gens:
                    report.append(f"[FAIL] Held-out generator '{gen}' found in training split!")

        passed = (len(report) == 0)
        return passed, report


def main():
    parser = argparse.ArgumentParser(description="Check voice cloning dataset manifest for data leakage")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--held-out-gens", nargs="*", default=[], help="List of generator IDs that must be held out from train")
    args = parser.parse_args()

    checker = LeakageChecker(args.manifest)
    passed, report = checker.check_all(held_out_generators=args.held_out_gens)

    print("=" * 60)
    print(f"DATA LEAKAGE AUDIT REPORT: {args.manifest}")
    print("=" * 60)
    if passed:
        print("[PASS] Zero data leakage detected across splits!")
        print(f"  Train samples: {len(checker.df[checker.df['split']=='train'])}")
        print(f"  Dev samples:   {len(checker.df[checker.df['split']=='dev'])}")
        print(f"  Eval samples:  {len(checker.df[checker.df['split']=='eval'])}")
    else:
        print(f"[FAIL] Data leakage violations detected ({len(report)} issues):")
        for err in report:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == '__main__':
    main()