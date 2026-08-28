"""
dataset.py — PyTorch Dataset for the voice-clone / deepfake speech detector.

Reads a manifest CSV with these columns:
    path, label, split, source_dataset, speaker_id, utterance_id,
    generator_id, language, codec, duration_s, license, consent

Label convention (IMPORTANT — never flip this):
    "bonafide" → 0  (real, human speech)
    "spoof"    → 1  (AI-generated / cloned / fake speech)

Audio is:
    • Resampled to TARGET_SR (16 kHz) mono
    • Padded with zeros or cropped to exactly WINDOW_SAMPLES samples
      (WINDOW_SAMPLES = TARGET_SR * WINDOW_SECONDS)

# TODO: To swap in your real dataset, change MANIFEST_CSV to point at your
#       ASVspoof2019-LA (or any other) manifest.  No code changes required.
"""

import os
import torch
import soundfile as sf
import numpy as np
from torchaudio.transforms import Resample
import pandas as pd
from torch.utils.data import Dataset

# ─── Configurable constants ────────────────────────────────────────────────────
TARGET_SR      = 16_000          # All audio resampled to 16 kHz mono
WINDOW_SECONDS = 4               # Fixed window length in seconds
WINDOW_SAMPLES = TARGET_SR * WINDOW_SECONDS   # 64 000 samples per clip

# Label string → integer index
# IMPORTANT: bonafide=0, spoof=1  — higher model output = more evidence of fake
LABEL_MAP = {"bonafide": 0, "spoof": 1}

# ─── Dataset class ─────────────────────────────────────────────────────────────

class SpeechDataset(Dataset):
    """
    Reads a manifest CSV and returns (waveform, label) pairs.

    Args:
        manifest_csv (str): Path to the manifest CSV file.
        split        (str): One of "train", "dev", or "test".
        window_samples (int): Number of audio samples per clip (after
                              resampling). Clips are zero-padded or cropped.
    """

    def __init__(
        self,
        manifest_csv: str,
        split: str,
        window_samples: int = WINDOW_SAMPLES,
    ):
        # Accept 'eval' as an alias for 'test' (data_pipeline uses 'eval',
        # our pipeline uses 'test' — both are valid here).
        assert split in ("train", "dev", "test", "eval"), (
            f"split must be 'train', 'dev', 'test' (or 'eval'), got '{split}'"
        )

        df = pd.read_csv(manifest_csv)
        # Treat 'eval' rows as 'test' so a single split argument finds both.
        df["split"] = df["split"].replace({"eval": "test"})
        if split == "eval":
            split = "test"
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.window_samples = window_samples

        if len(self.data) == 0:
            raise ValueError(
                f"No rows found for split='{split}' in {manifest_csv}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        audio_path = row["path"]
        label_str  = row["label"]

        # ── Validate label ─────────────────────────────────────────────────────
        assert label_str in LABEL_MAP, (
            f"Unknown label '{label_str}' in row {idx}. "
            f"Expected 'bonafide' or 'spoof'."
        )
        label = LABEL_MAP[label_str]  # 0 or 1

        # ── Load audio ─────────────────────────────────────────────────────
        # Use soundfile directly to avoid the torchaudio torchcodec backend
        # dependency that was introduced in torchaudio 2.13.
        audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        # audio_np shape: (samples, channels)  — soundfile is channels-last
        waveform = torch.from_numpy(audio_np.T)   # (channels, samples)

        # ── Convert to mono ──────────────────────────────────────────────────
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)   # (1, samples)

        # ── Resample to TARGET_SR if needed ──────────────────────────────────
        if sr != TARGET_SR:
            resampler = Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform  = resampler(waveform)

        # ── Pad or crop to fixed window length ─────────────────────────────────
        waveform = _pad_or_crop(waveform, self.window_samples)   # (1, W)

        # Remove the channel dimension; model expects (W,)
        waveform = waveform.squeeze(0)

        return waveform.float(), torch.tensor(label, dtype=torch.long)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _pad_or_crop(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Pad with zeros on the right, or crop from the left, so that
    waveform has exactly target_len samples.

    Shape: (channels, samples) → (channels, target_len)
    """
    n = waveform.shape[-1]
    if n < target_len:
        pad_amount = target_len - n
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif n > target_len:
        waveform = waveform[..., :target_len]
    return waveform
