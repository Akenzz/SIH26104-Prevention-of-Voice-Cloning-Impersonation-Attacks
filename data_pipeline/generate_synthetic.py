"""
Synthetic Audio Dataset and Manifest Generator for Integration Testing.

Generates:
  - 16 kHz WAV dummy audio files for train, dev, and eval.
  - Multi-speaker setup with zero speaker leakage across splits.
  - Full 12-column compliant manifest CSV.
"""

import os
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

try:
    from schema import REQUIRED_COLUMNS, validate_schema
except ImportError:
    from data_pipeline.schema import REQUIRED_COLUMNS, validate_schema


def generate_tiny_dataset(
    output_dir: str = "data_pipeline/dummy_dataset",
    manifest_csv: str = "data_pipeline/dummy_dataset/manifest.csv",
    sample_rate: int = 16000,
    duration_s: float = 4.0
):
    """
    Generate dummy dataset files and manifest with distinct speakers per split.
    """
    base_dir = Path(output_dir)
    audio_dir = base_dir / "wavs"
    audio_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(sample_rate * duration_s)

    # Configuration for splits with zero speaker leakage
    split_configs = [
        # (split, num_bonafide, num_spoof, speaker_prefix, generator_ids)
        ('train', 4, 4, 'spk_tr', ['gen_A', 'gen_B']),
        ('dev',   2, 2, 'spk_dv', ['gen_A', 'gen_B']),
        ('eval',  2, 2, 'spk_ev', ['gen_C']),  # gen_C is held-out generator
    ]

    rows = []
    file_idx = 1

    for split, num_bona, num_spof, spk_prefix, generators in split_configs:
        # Create bonafide samples
        for i in range(num_bona):
            spk_id = f"{spk_prefix}_{i+1:02d}"
            utt_id = f"utt_{file_idx:04d}"
            filename = f"{utt_id}.wav"
            file_path = audio_dir / filename

            # Generate synthetic sine wave audio (bonafide mock)
            t = np.linspace(0, duration_s, num_samples)
            freq = 220 + i * 20
            audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.randn(num_samples)
            sf.write(str(file_path), audio.astype(np.float32), sample_rate)

            rows.append({
                'path': str(file_path),
                'label': 'bonafide',
                'split': split,
                'source_dataset': 'Synthetic_TestBench',
                'speaker_id': spk_id,
                'utterance_id': utt_id,
                'generator_id': 'none',
                'language': 'en',
                'codec': 'pcm_16k',
                'duration_s': duration_s,
                'license': 'MIT',
                'consent': 'yes'
            })
            file_idx += 1

        # Create spoof samples
        for i in range(num_spof):
            spk_id = f"{spk_prefix}_{i+1:02d}"  # Same source speaker
            utt_id = f"utt_{file_idx:04d}"
            filename = f"{utt_id}.wav"
            file_path = audio_dir / filename

            gen_id = generators[i % len(generators)]

            # Generate noise-heavy audio (spoof mock)
            audio = 0.3 * np.random.randn(num_samples)
            sf.write(str(file_path), audio.astype(np.float32), sample_rate)

            rows.append({
                'path': str(file_path),
                'label': 'spoof',
                'split': split,
                'source_dataset': 'Synthetic_TestBench',
                'speaker_id': spk_id,
                'utterance_id': utt_id,
                'generator_id': gen_id,
                'language': 'en',
                'codec': 'pcm_16k',
                'duration_s': duration_s,
                'license': 'MIT',
                'consent': 'yes'
            })
            file_idx += 1

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    errors = validate_schema(df)
    if errors:
        raise ValueError(f"Generated manifest failed schema check:\n" + "\n".join(errors))

    Path(manifest_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_csv, index=False)

    print(f"[OK] Generated {len(df)} dummy audio files in '{audio_dir}'")
    print(f"[OK] Saved compliant 12-column manifest to '{manifest_csv}'")
    return manifest_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic audio dataset and manifest")
    parser.add_argument("--output-dir", type=str, default="data_pipeline/dummy_dataset")
    parser.add_argument("--manifest-csv", type=str, default="data_pipeline/dummy_dataset/manifest.csv")
    args = parser.parse_args()

    generate_tiny_dataset(args.output_dir, args.manifest_csv)