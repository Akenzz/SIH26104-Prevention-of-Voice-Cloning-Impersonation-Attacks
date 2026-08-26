"""
Utilities for creating, validating, and merging manifest files.
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Optional, Dict
import hashlib


def create_manifest_from_protocol(
    protocol_file: str,
    audio_dir: str,
    output_path: str,
    dataset_name: str,
    split: str
) -> pd.DataFrame:
    """
    Create manifest from ASVspoof protocol file format.

    ASVspoof protocol format:
        speaker_id audio_filename - system_id label
    Example:
        LA_0079 LA_E_5000296.flac - A17 spoof

    Args:
        protocol_file: Path to protocol .txt file
        audio_dir: Directory containing audio files
        output_path: Where to save manifest CSV
        dataset_name: Name of dataset (e.g., 'asvspoof19_la')
        split: 'train', 'dev', or 'eval'

    Returns:
        DataFrame with manifest
    """
    records = []

    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            speaker_id = parts[0]
            audio_filename = parts[1]
            system_id = parts[3]  # Generator/attack ID
            label = parts[4]  # 'bonafide' or 'spoof'

            # Construct path
            audio_path = os.path.join(audio_dir, audio_filename)

            # Extract utterance ID from filename (e.g., LA_E_5000296)
            utterance_id = audio_filename.replace('.flac', '').replace('.wav', '')

            records.append({
                'path': audio_path,
                'label': label,
                'split': split,
                'source_dataset': dataset_name,
                'speaker_id': speaker_id,
                'utterance_id': utterance_id,
                'generator_id': system_id if label == 'spoof' else 'none',
                'attack_type': 'tts_vc' if label == 'spoof' else 'none',
                'language': 'en',  # ASVspoof is English
                'codec': 'none',
                'channel': 'studio',
                'duration_s': 0.0,  # Will be computed if needed
                'license': 'research',
                'consent': 'yes'
            })

    df = pd.DataFrame(records)

    # Check if files exist and compute durations (optional, can be slow)
    print(f"Created manifest with {len(df)} entries")

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved manifest to {output_path}")

    return df


def validate_manifest(manifest_path: str, check_files: bool = True) -> Dict[str, any]:
    """
    Validate manifest for common issues.

    Checks:
        - Required columns present
        - Labels are valid ('bonafide' or 'spoof')
        - No duplicate paths
        - Files exist (if check_files=True)
        - No speaker/utterance leakage between splits

    Returns:
        Dict with validation results and warnings
    """
    df = pd.read_csv(manifest_path)
    issues = []
    warnings = []

    # Check required columns
    required_cols = ['path', 'label', 'split']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    # Check labels
    valid_labels = {'bonafide', 'spoof'}
    invalid_labels = set(df['label'].unique()) - valid_labels
    if invalid_labels:
        issues.append(f"Invalid labels: {invalid_labels}. Must be 'bonafide' or 'spoof'")

    # Check for duplicate paths
    duplicates = df[df.duplicated(subset='path', keep=False)]
    if not duplicates.empty:
        warnings.append(f"Found {len(duplicates)} duplicate paths")

    # Check if files exist
    if check_files:
        missing_files = []
        for idx, row in df.iterrows():
            if not os.path.exists(row['path']):
                missing_files.append(row['path'])
                if len(missing_files) >= 10:  # Limit output
                    break

        if missing_files:
            issues.append(f"Missing files (showing first 10): {missing_files}")

    # Check for speaker leakage between splits
    if 'speaker_id' in df.columns and 'split' in df.columns:
        for split1 in df['split'].unique():
            for split2 in df['split'].unique():
                if split1 >= split2:
                    continue

                speakers1 = set(df[df['split'] == split1]['speaker_id'].unique())
                speakers2 = set(df[df['split'] == split2]['speaker_id'].unique())
                overlap = speakers1 & speakers2

                if overlap:
                    warnings.append(
                        f"Speaker leakage: {len(overlap)} speakers appear in both "
                        f"'{split1}' and '{split2}' splits"
                    )

    # Check for utterance leakage between splits
    if 'utterance_id' in df.columns and 'split' in df.columns:
        for split1 in df['split'].unique():
            for split2 in df['split'].unique():
                if split1 >= split2:
                    continue

                utts1 = set(df[df['split'] == split1]['utterance_id'].unique())
                utts2 = set(df[df['split'] == split2]['utterance_id'].unique())
                overlap = utts1 & utts2

                if overlap:
                    issues.append(
                        f"CRITICAL: Utterance leakage! {len(overlap)} utterances appear in both "
                        f"'{split1}' and '{split2}' splits. This will inflate results."
                    )

    result = {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'num_samples': len(df),
        'splits': df['split'].value_counts().to_dict(),
        'labels': df['label'].value_counts().to_dict(),
    }

    return result


def merge_manifests(manifest_paths: List[str], output_path: str) -> pd.DataFrame:
    """
    Merge multiple manifest files into one.

    Args:
        manifest_paths: List of paths to manifest CSVs
        output_path: Where to save merged manifest

    Returns:
        Merged DataFrame
    """
    dfs = []

    for path in manifest_paths:
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"Loaded {len(df)} samples from {path}")

    merged = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (based on path)
    before = len(merged)
    merged = merged.drop_duplicates(subset='path', keep='first')
    after = len(merged)

    if before != after:
        print(f"Removed {before - after} duplicate entries")

    # Save
    merged.to_csv(output_path, index=False)
    print(f"Saved merged manifest with {len(merged)} samples to {output_path}")

    return merged


def compute_manifest_stats(manifest_path: str) -> Dict:
    """
    Compute statistics for a manifest file.

    Returns summary statistics useful for understanding the dataset.
    """
    df = pd.read_csv(manifest_path)

    stats = {
        'total_samples': len(df),
        'splits': df['split'].value_counts().to_dict() if 'split' in df.columns else {},
        'labels': df['label'].value_counts().to_dict() if 'label' in df.columns else {},
    }

    # Per-split label distribution
    if 'split' in df.columns and 'label' in df.columns:
        stats['split_label_dist'] = {}
        for split in df['split'].unique():
            split_df = df[df['split'] == split]
            stats['split_label_dist'][split] = split_df['label'].value_counts().to_dict()

    # Language distribution
    if 'language' in df.columns:
        stats['languages'] = df['language'].value_counts().to_dict()

    # Generator distribution (for spoof samples only)
    if 'generator_id' in df.columns:
        spoof_df = df[df['label'] == 'spoof']
        stats['generators'] = spoof_df['generator_id'].value_counts().to_dict()

    # Dataset sources
    if 'source_dataset' in df.columns:
        stats['datasets'] = df['source_dataset'].value_counts().to_dict()

    return stats


if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 2:
        print("Manifest utilities")
        print("\nUsage:")
        print("  Validate: python manifest_utils.py validate <manifest.csv>")
        print("  Stats:    python manifest_utils.py stats <manifest.csv>")
        print("  Merge:    python manifest_utils.py merge <output.csv> <input1.csv> <input2.csv> ...")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'validate':
        manifest_path = sys.argv[2]
        print(f"Validating {manifest_path}...")
        result = validate_manifest(manifest_path, check_files=False)

        print(f"\n{'='*60}")
        print(f"Validation Result: {'PASS' if result['valid'] else 'FAIL'}")
        print(f"{'='*60}")

        print(f"\nSamples: {result['num_samples']}")
        print(f"Splits: {result['splits']}")
        print(f"Labels: {result['labels']}")

        if result['issues']:
            print(f"\n❌ ISSUES ({len(result['issues'])}):")
            for issue in result['issues']:
                print(f"  - {issue}")

        if result['warnings']:
            print(f"\n⚠️  WARNINGS ({len(result['warnings'])}):")
            for warning in result['warnings']:
                print(f"  - {warning}")

        if result['valid'] and not result['warnings']:
            print(f"\n✅ Manifest is valid!")

    elif command == 'stats':
        manifest_path = sys.argv[2]
        print(f"Computing stats for {manifest_path}...")
        stats = compute_manifest_stats(manifest_path)
        print(json.dumps(stats, indent=2))

    elif command == 'merge':
        output_path = sys.argv[2]
        input_paths = sys.argv[3:]
        print(f"Merging {len(input_paths)} manifests...")
        merge_manifests(input_paths, output_path)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
