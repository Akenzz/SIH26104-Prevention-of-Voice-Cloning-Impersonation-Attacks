"""
PyTorch Dataset for loading audio from manifest CSV.

Expected manifest format:
    path,label,split,source_dataset,speaker_id,utterance_id,generator_id,language,duration_s
"""

import os
import pandas as pd
import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class AudioDataset(torch.utils.data.Dataset):
    """
    Dataset for loading audio samples from a manifest CSV.

    Args:
        manifest_path: Path to manifest CSV file
        split: Which split to load ('train', 'dev', 'eval')
        window_sec: Target audio window duration in seconds
        sample_rate: Target sample rate (default 16000 Hz)
        root_dir: Optional root directory to prepend to relative paths
        augment: Whether to apply augmentation (only for training)
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = 'train',
        window_sec: float = 4.0,
        sample_rate: int = 16000,
        root_dir: Optional[str] = None,
        augment: bool = False
    ):
        self.manifest_path = manifest_path
        self.split = split
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.window_samples = int(window_sec * sample_rate)
        self.root_dir = Path(root_dir) if root_dir else None
        self.augment = augment

        # Load manifest
        self.df = pd.read_csv(manifest_path)

        # Filter by split
        if split:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        print(f"Loaded {len(self.df)} samples for split='{split}' from {manifest_path}")

        # Validate required columns
        required_cols = ['path', 'label']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

        # Validate labels
        valid_labels = {'bonafide', 'spoof'}
        invalid = set(self.df['label'].unique()) - valid_labels
        if invalid:
            raise ValueError(f"Invalid labels found: {invalid}. Must be 'bonafide' or 'spoof'")

        # Label encoding: bonafide=0, spoof=1
        self.label_to_idx = {'bonafide': 0, 'spoof': 1}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        """
        Returns:
            audio: Tensor of shape (window_samples,), float32
            label: 0 (bonafide) or 1 (spoof)
            metadata: Dict with path, speaker_id, generator_id, etc.
        """
        row = self.df.iloc[idx]

        # Construct full path
        audio_path = row['path']
        if self.root_dir and not os.path.isabs(audio_path):
            audio_path = self.root_dir / audio_path

        # Load audio
        try:
            audio, sr = torchaudio.load(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {audio_path}: {e}")

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Remove channel dimension: (1, samples) -> (samples,)
        audio = audio.squeeze(0)

        # Pad or crop to fixed window length
        if audio.shape[0] < self.window_samples:
            # Pad with zeros
            padding = self.window_samples - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
        elif audio.shape[0] > self.window_samples:
            # Random crop during training, center crop during eval
            if self.split == 'train':
                start = torch.randint(0, audio.shape[0] - self.window_samples + 1, (1,)).item()
            else:
                start = (audio.shape[0] - self.window_samples) // 2
            audio = audio[start:start + self.window_samples]

        # Apply augmentation if enabled (placeholder for now)
        if self.augment and self.split == 'train':
            audio = self._augment(audio)

        # Get label
        label = self.label_to_idx[row['label']]

        # Collect metadata
        metadata = {
            'path': str(audio_path),
            'idx': idx,
            'label_str': row['label'],
        }

        # Add optional columns if present
        optional_cols = ['speaker_id', 'utterance_id', 'generator_id', 'language', 'source_dataset']
        for col in optional_cols:
            if col in row:
                metadata[col] = row[col]

        return audio, label, metadata

    def _augment(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply audio augmentation (placeholder for now).
        Will be implemented in augmentation.py module.
        """
        # TODO: Import and apply augmentations from augmentation.py
        # - Codec degradation (AMR-NB, Opus)
        # - Additive noise
        # - RIR convolution
        return audio

    def get_label_distribution(self) -> dict:
        """Returns count of bonafide vs spoof samples."""
        counts = self.df['label'].value_counts().to_dict()
        return counts

    def get_duration_stats(self) -> dict:
        """Returns duration statistics if duration_s column exists."""
        if 'duration_s' not in self.df.columns:
            return {}

        durations = self.df['duration_s']
        return {
            'min': durations.min(),
            'max': durations.max(),
            'mean': durations.mean(),
            'median': durations.median(),
        }


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Args:
        batch: List of (audio, label, metadata) tuples

    Returns:
        audio_batch: Tensor of shape (batch_size, window_samples)
        label_batch: Tensor of shape (batch_size,)
        metadata_batch: List of metadata dicts
    """
    audios, labels, metadatas = zip(*batch)

    # Stack tensors
    audio_batch = torch.stack(audios)
    label_batch = torch.tensor(labels, dtype=torch.long)

    return audio_batch, label_batch, list(metadatas)


if __name__ == '__main__':
    # Test the dataset
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <manifest_path>")
        print("Example: python dataset.py data/manifests/asvspoof19_train.csv")
        sys.exit(1)

    manifest_path = sys.argv[1]

    # Load dataset
    dataset = AudioDataset(manifest_path, split='train', window_sec=4.0)

    print(f"\nDataset loaded: {len(dataset)} samples")
    print(f"Label distribution: {dataset.get_label_distribution()}")
    print(f"Duration stats: {dataset.get_duration_stats()}")

    # Test loading one sample
    audio, label, metadata = dataset[0]
    print(f"\nSample 0:")
    print(f"  Audio shape: {audio.shape}")
    print(f"  Label: {label} ({metadata['label_str']})")
    print(f"  Path: {metadata['path']}")

    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    audio_batch, label_batch, metadata_batch = next(iter(loader))
    print(f"\nBatch test:")
    print(f"  Audio batch shape: {audio_batch.shape}")
    print(f"  Label batch shape: {label_batch.shape}")
    print(f"  Labels: {label_batch.tolist()}")
