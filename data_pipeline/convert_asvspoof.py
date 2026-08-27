"""
Converter utility to transform official ASVspoof 2019/2021 protocol text files into the standard 12-column Manifest CSV schema.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    from schema import REQUIRED_COLUMNS, validate_schema
except ImportError:
    from data_pipeline.schema import REQUIRED_COLUMNS, validate_schema


def convert_asvspoof2019_protocol(
    protocol_path: str,
    audio_dir: str,
    output_csv: str,
    split: str = 'train',
    source_dataset: str = 'ASVspoof2019_LA'
) -> pd.DataFrame:
    """
    Convert ASVspoof 2019 LA protocol file to standard manifest format.

    ASVspoof 2019 protocol format:
        speaker_id utterance_id environment attack_id key
        Example: LA_0079 LA_E_1000137 - - bonafide
        Example: LA_0079 LA_E_1000138 - A07 spoof
    """
    protocol_path = Path(protocol_path)
    audio_dir = Path(audio_dir)

    rows = []
    with open(protocol_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            spk_id = parts[0]
            utt_id = parts[1]
            attack_id = parts[3]
            key = parts[4]  # 'bonafide' or 'spoof'

            # Audio path resolution (handles .flac or .wav)
            audio_path = audio_dir / f"{utt_id}.flac"
            if not audio_path.exists():
                audio_path = audio_dir / f"{utt_id}.wav"

            generator_id = 'none' if key == 'bonafide' else attack_id

            rows.append({
                'path': str(audio_path),
                'label': key,
                'split': split,
                'source_dataset': source_dataset,
                'speaker_id': spk_id,
                'utterance_id': utt_id,
                'generator_id': generator_id,
                'language': 'en',
                'codec': 'pcm_16k',
                'duration_s': 4.0,  # Default standard window duration
                'license': 'ASVspoof_EULA',
                'consent': 'yes'
            })

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    errors = validate_schema(df)

    if errors:
        raise ValueError(f"Schema validation failed during conversion:\n" + "\n".join(errors))

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[OK] Converted {len(df)} samples from {protocol_path.name} -> {output_csv}")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert ASVspoof protocol to Manifest CSV")
    parser.add_argument("--protocol", type=str, required=True, help="Path to ASVspoof protocol txt file")
    parser.add_argument("--audio-dir", type=str, required=True, help="Path to directory containing audio files")
    parser.add_argument("--output", type=str, required=True, help="Output manifest CSV path")
    parser.add_argument("--split", type=str, default="train", choices=["train", "dev", "eval"])
    args = parser.parse_args()

    convert_asvspoof2019_protocol(args.protocol, args.audio_dir, args.output, split=args.split)