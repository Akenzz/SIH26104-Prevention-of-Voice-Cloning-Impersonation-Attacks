"""
Shared PyTorch Dataset for loading audio from the 12-column manifest CSV schema.
"""

import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict

try:
    from schema import validate_schema
except ImportError:
    from data_pipeline.schema import validate_schema


class ManifestAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: Optional[str] = 'train',
        window_sec: float = 4.0,
        sample_rate: int = 16000,
        root_dir: Optional[str] = None
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self.split = split
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.window_samples = int(window_sec * sample_rate)
        self.root_dir = Path(root_dir) if root_dir else None

        # Schema check
        errors = validate_schema(self.df)
        if errors:
            raise ValueError(f"Manifest schema invalid:\n" + "\n".join(errors))

        # Filter split
        if split:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.label_map = {'bonafide': 0, 'spoof': 1}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        row = self.df.iloc[idx]
        audio_path = Path(row['path'])
        if self.root_dir and not audio_path.is_absolute():
            audio_path = self.root_dir / audio_path

        # Load audio using soundfile first (PyTorch 2.6 Windows stability)
        try:
            import soundfile as sf
            audio_np, sr = sf.read(str(audio_path), dtype='float32')
            audio = torch.from_numpy(audio_np)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
        except Exception:
            audio, sr = torchaudio.load(str(audio_path))

        # Resample if needed
        if sr != self.sample_rate:
            resample = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resample(audio)

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.squeeze(0)

        # Pad / Crop to fixed window length
        if audio.shape[0] < self.window_samples:
            padding = self.window_samples - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
        elif audio.shape[0] > self.window_samples:
            if self.split == 'train':
                start = torch.randint(0, audio.shape[0] - self.window_samples + 1, (1,)).item()
            else:
                start = (audio.shape[0] - self.window_samples) // 2
            audio = audio[start:start + self.window_samples]

        label = self.label_map[row['label']]

        metadata = {
            'path': str(audio_path),
            'speaker_id': row['speaker_id'],
            'utterance_id': row['utterance_id'],
            'generator_id': row['generator_id'],
            'language': row['language'],
            'codec': row['codec'],
            'source_dataset': row['source_dataset']
        }

        return audio, label, metadata


def collate_manifest_batch(batch):
    audios, labels, metadatas = zip(*batch)
    audio_batch = torch.stack(audios)
    label_batch = torch.tensor(labels, dtype=torch.long)
    return audio_batch, label_batch, list(metadatas)