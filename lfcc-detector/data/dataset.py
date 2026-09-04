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
        bandwidth_augment: Randomize resampler fingerprint + bandwidth per window,
            identically for bonafide and spoof (train split only). Reduces but does
            not remove the near-Nyquist shortcut -- see band_gate_hz.
        band_gate_hz: Lowpass EVERY window (all splits, both classes) at this
            cutoff, deleting the frequency region where the training resampler
            (librosa/soxr) and the serving resampler (scipy) disagree. 7000.0 is
            the validated value. Whatever is set here MUST also be applied at
            inference; a model trained with the gate and served without it is
            biased spoofward. See diag_band_gate.py.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = 'train',
        window_sec: float = 4.0,
        sample_rate: int = 16000,
        root_dir: Optional[str] = None,
        augment: bool = False,
        bandwidth_augment: bool = False,
        band_gate_hz: Optional[float] = None
    ):
        # Make repo-root imports (data_pipeline.*) work whether the caller runs
        # from lfcc-detector/ or from the repo root.
        import sys, pathlib
        _repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))

        self.manifest_path = manifest_path
        self.split = split
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.window_samples = int(window_sec * sample_rate)
        self.root_dir = Path(root_dir) if root_dir else None
        self.augment = augment and (split == 'train')  # never augment dev/eval

        # Bandwidth/resampler-fingerprint randomization. Off by default so the
        # existing training scripts are byte-identical in behaviour; opt in with
        # bandwidth_augment=True (finetune_bandwidth_robust.py does).
        # This is the fix for the near-Nyquist label shortcut: the corpus's native
        # sample rates are split by label (spoof 52.5% @22050 -> librosa-resampled
        # -> 7.9-8 kHz hole; bonafide 75.8% already @16k -> full band), so the
        # models learned "hole => spoof" and read live scipy-resampled audio as
        # bonafide. Randomizing the top band IDENTICALLY on both classes removes
        # the label information. See memory/lfcc-resampler-bandwidth-shortcut.md.
        # Parity band gate. Unlike the augmentation below this applies to EVERY
        # split, because it is not augmentation -- it defines the band the model
        # is allowed to see, and dev/eval must match train or the numbers are
        # meaningless. Measured effect on the shipped models: the
        # scipy-vs-librosa verdict gap falls from 13.35 to 0.05 logits.
        self.band_gate_hz = float(band_gate_hz) if band_gate_hz else None
        if self.band_gate_hz:
            print(f"[INFO] Parity band gate ACTIVE at {self.band_gate_hz:.0f} Hz "
                  f"(split='{split}', both classes) -- inference MUST use the same cutoff")

        self._bw_aug = None
        if bandwidth_augment and split == 'train':
            from data_pipeline.bandwidth_augment import BandwidthAugment

            self._bw_aug = BandwidthAugment(sample_rate=sample_rate)
            print("[INFO] Bandwidth augmentation ACTIVE (resampler-fingerprint + lowpass "
                  "randomization, applied identically to bonafide and spoof)")

        # Build augmentation pipeline once (reused per sample)
        self._aug_pipeline = None
        if self.augment:
            try:
                import sys, pathlib
                # Allow import from either lfcc-detector or repo root context
                repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))
                from data_pipeline.augmentation import TrainingAugmentationPipeline
                self._aug_pipeline = TrainingAugmentationPipeline(
                    apply_codec=False,   # disable by default (requires ffmpeg)
                    apply_noise=True,
                    apply_rir=False,
                    sample_rate=sample_rate,
                )
                print(f"[INFO] Augmentation active: {self._aug_pipeline.active_augmentations}")
            except Exception as e:
                print(f"[WARN] Could not load augmentation pipeline: {e}. Running without augmentation.")

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

        # Load audio (try soundfile first, fallback to torchaudio)
        try:
            import soundfile as sf
            audio_np, sr = sf.read(audio_path, dtype='float32')
            audio = torch.from_numpy(audio_np)
            # soundfile returns (samples,) for mono, ensure (1, samples) for consistency
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
        except Exception as e_sf:
            # Fallback to torchaudio if soundfile fails
            try:
                audio, sr = torchaudio.load(audio_path)
            except Exception as e_ta:
                print(f"[WARN] Failed to load {audio_path}. Returning silent audio. Error: {e_ta}")
                audio = torch.zeros(1, self.window_samples, dtype=torch.float32)
                sr = self.sample_rate

        # Resample if needed.
        # Use the SAME resampler the realtime backend uses (scipy resample_poly,
        # realtime-backend/audio/resample.py) instead of torchaudio's, which was a
        # third distinct filter with a third distinct near-Nyquist rolloff. Any
        # remaining fingerprint is then randomized by bandwidth_augment below, so
        # the band can't encode the label either way.
        if sr != self.sample_rate:
            audio = self._resample_backend_parity(audio, sr)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Remove channel dimension: (1, samples) -> (samples,)
        audio = audio.squeeze(0)

        # Fit to a fixed window WITHOUT leaking a trivial shortcut.
        # The silence canary showed bonafide clips are more often <4s than spoof,
        # so zero-padding taught the model "trailing digital zeros / leading
        # silence => real". Fix, applied identically to both classes:
        #   1. trim leading/trailing silence (equalizes onset -> kills lead_sil)
        #   2. repeat/tile-pad clips shorter than the window (no zero region -> kills zero_frac)
        #   3. random-crop (train) / center-crop (eval) longer clips, as before
        audio = self._trim_silence(audio)
        W = self.window_samples
        n = audio.shape[0]
        if n < W:
            # Tile the actual signal to fill the window (standard anti-spoofing pad).
            reps = -(-W // max(n, 1))  # ceil division
            audio = audio.repeat(reps)[:W]
        elif n > W:
            if self.split == 'train':
                start = torch.randint(0, n - W + 1, (1,)).item()
            else:
                start = (n - W) // 2
            audio = audio[start:start + W]

        # Bandwidth / resampler-fingerprint randomization. Deliberately applied
        # BEFORE the noise+codec pipeline and AFTER cropping (bounded cost), and
        # WITHOUT looking at the label -- conditioning on the label here would
        # re-create the very shortcut this removes.
        if self._bw_aug is not None:
            audio = torch.from_numpy(
                self._bw_aug(audio.numpy())
            ).to(dtype=torch.float32)

        # Apply augmentation if enabled (placeholder for now)
        if self.augment and self.split == 'train':
            audio = self._augment(audio)

        # Parity band gate LAST, so nothing downstream can reintroduce energy
        # above the cutoff (the noise augmentation is broadband and would).
        if self.band_gate_hz:
            from data_pipeline.bandwidth_augment import _lowpass

            audio = torch.from_numpy(
                _lowpass(audio.numpy(), self.band_gate_hz, self.sample_rate)
            ).to(dtype=torch.float32)

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

    def _resample_backend_parity(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Resample (1, n) audio to self.sample_rate the way the backend does.

        scipy ``resample_poly`` with the default Kaiser design, matching
        realtime-backend/audio/resample.py. Falls back to torchaudio if scipy is
        missing so training never dies on this.
        """
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(int(sr), int(self.sample_rate))
            up = int(self.sample_rate) // g
            down = int(sr) // g
            x = audio.numpy()
            y = resample_poly(x, up, down, axis=-1)
            return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        except ImportError:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            return resampler(audio)

    def _augment(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation via data_pipeline.augmentation.TrainingAugmentationPipeline."""
        if self._aug_pipeline is not None:
            return self._aug_pipeline(audio)
        return audio

    def _trim_silence(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Trim leading/trailing silence with a frame-RMS gate (~-20 dB from the
        loudest frame), applied identically to bonafide and spoof. This equalizes
        speech onset so the model can't shortcut on "how much leading silence".
        Falls back to the original clip if trimming would leave almost nothing.
        """
        x = audio.detach().cpu().numpy()
        fl = max(int(0.02 * self.sample_rate), 1)  # 20 ms frames
        if len(x) < 2 * fl:
            return audio
        n_frames = len(x) // fl
        frames = x[:n_frames * fl].reshape(n_frames, fl)
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-9)
        thr = max(float(rms.max()) * 0.1, 1e-4)  # -20 dB below loudest frame
        keep = np.where(rms > thr)[0]
        if len(keep) == 0:
            return audio
        trimmed = audio[keep[0] * fl: (keep[-1] + 1) * fl]
        # guard: keep original if the gate nuked the clip (e.g. very quiet audio)
        if trimmed.shape[0] < int(0.2 * self.sample_rate):
            return audio
        return trimmed

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
