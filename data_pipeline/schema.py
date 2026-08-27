"""
Manifest Schema Contract for SIH26104 Voice Cloning Detection.

Defines the required 12-column manifest CSV format used by all models and benchmarks:
    path, label, split, source_dataset, speaker_id, utterance_id, generator_id, language, codec, duration_s, license, consent
"""

import pandas as pd
from typing import List, Set, Dict

# Required columns for every manifest
REQUIRED_COLUMNS: List[str] = [
    'path',
    'label',
    'split',
    'source_dataset',
    'speaker_id',
    'utterance_id',
    'generator_id',
    'language',
    'codec',
    'duration_s',
    'license',
    'consent'
]

VALID_LABELS: Set[str] = {'bonafide', 'spoof'}
VALID_SPLITS: Set[str] = {'train', 'dev', 'eval', 'eval_ood'}
# eval_ood = cross-corpus / OOD evaluation only — NEVER loaded by any training DataLoader


def validate_schema(df: pd.DataFrame) -> List[str]:
    """
    Validate that a DataFrame conforms strictly to the Manifest Schema.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    # Column check
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if errors:
        return errors

    # Label check
    invalid_labels = set(df['label'].dropna().unique()) - VALID_LABELS
    if invalid_labels:
        errors.append(f"Invalid labels found: {invalid_labels}. Must be in {VALID_LABELS}")

    # Split check
    invalid_splits = set(df['split'].dropna().unique()) - VALID_SPLITS
    if invalid_splits:
        errors.append(f"Invalid splits found: {invalid_splits}. Must be in {VALID_SPLITS}")

    # Null value check for critical fields
    for col in ['path', 'label', 'split', 'speaker_id']:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null/missing values.")

    return errors


def create_empty_manifest() -> pd.DataFrame:
    """Create an empty DataFrame with the exact manifest schema columns."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS)