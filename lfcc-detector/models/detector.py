"""
Inference wrapper for LFCC-LCNN detector.

This module implements the contract interface defined in contracts.md.
"""

import torch
import numpy as np
from typing import Tuple, Union
from pathlib import Path
import warnings

from .lfcc_lcnn import LFCCLCNNWithFeatureExtraction


class LFCCLCNNDetector:
    """
    Production inference wrapper for LFCC-LCNN deepfake detector.

    Implements the contract interface:
        forward(audio_window) -> (logit, embedding)

    Where:
        - audio_window: 4-second audio at 16kHz (64,000 samples)
        - logit: float, higher = more spoof evidence
        - embedding: 128-dim numpy array for fusion

    Args:
        checkpoint_path: Path to trained model checkpoint (.pth file)
        device: 'cuda', 'cpu', or None (auto-detect)
        sample_rate: Expected audio sample rate (default 16000)
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = None,
        sample_rate: int = 16000,
    ):
        self.checkpoint_path = checkpoint_path
        self.sample_rate = sample_rate
        self.model_name = "LFCC-LCNN"

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = LFCCLCNNWithFeatureExtraction(
            sample_rate=sample_rate,
            n_lfcc=20,
            with_deltas=True,
            embedding_dim=128,
            dropout=0.0  # No dropout at inference
        )

        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            warnings.warn("No checkpoint provided. Model initialized with random weights.")

        self.model.to(self.device)
        self.model.eval()

        # Extract model version from checkpoint path
        if checkpoint_path:
            self.model_version = Path(checkpoint_path).stem
        else:
            self.model_version = "untrained"

        print(f"[OK] Loaded {self.model_name} detector")
        print(f"  Version: {self.model_version}")
        print(f"  Device: {self.device}")
        print(f"  Sample rate: {self.sample_rate} Hz")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

        # Print checkpoint info if available
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"  Checkpoint epoch: {checkpoint['epoch']}")
            if 'best_eer' in checkpoint:
                print(f"  Best EER: {checkpoint['best_eer']:.4f}")

    def forward(
        self,
        audio_window: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[float, np.ndarray]:
        """
        Score an audio window for synthetic speech detection.

        Args:
            audio_window: Audio samples, shape (samples,) or (batch, samples)
                         Expected: 16kHz mono, ~4 seconds (64,000 samples)
                         Accepts numpy array or torch tensor

        Returns:
            logit (float): Spoof score. Higher = more evidence of synthetic speech.
                          Typical range: [-5.0, +5.0] but unbounded.
                          Direction: positive = spoof, negative = bonafide

            embedding (np.ndarray): 128-dimensional feature vector from penultimate layer.
                                   Shape: (128,)

        Raises:
            ValueError: If audio_window has wrong shape or dtype
        """
        # Input validation and conversion
        audio_tensor = self._validate_and_convert_input(audio_window)

        # Inference
        with torch.no_grad():
            logit_tensor, embedding_tensor = self.model(audio_tensor, return_embedding=True)

        # Convert to contract output format
        logit = logit_tensor.squeeze().item()  # Single float
        embedding = embedding_tensor.squeeze().cpu().numpy()  # (128,) numpy array

        return logit, embedding

    def _validate_and_convert_input(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Validate input audio and convert to expected format.

        Expected: (samples,) or (batch, samples), 16kHz mono
        """
        # Convert numpy to torch
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        elif isinstance(audio, torch.Tensor):
            audio = audio.float()
        else:
            raise ValueError(f"audio must be numpy array or torch tensor, got {type(audio)}")

        # Ensure 2D: (batch, samples)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim == 2:
            pass  # Already (batch, samples)
        else:
            raise ValueError(f"audio must be 1D or 2D, got shape {audio.shape}")

        # Check sample count (4 seconds at 16kHz = 64,000 samples)
        expected_samples = int(4.0 * self.sample_rate)
        actual_samples = audio.shape[1]

        if actual_samples != expected_samples:
            # Pad or crop to expected length
            if actual_samples < expected_samples:
                padding = expected_samples - actual_samples
                audio = torch.nn.functional.pad(audio, (0, padding))
            else:
                audio = audio[:, :expected_samples]

        # Move to device
        audio = audio.to(self.device)

        return audio

    def predict_probability(self, logit: float, temperature: float = 1.0) -> float:
        """
        Convert logit to probability using sigmoid with optional temperature scaling.

        Note: This is uncalibrated. For calibrated probabilities, use a fitted
        calibrator on dev set (isotonic regression or Platt scaling).

        Args:
            logit: Raw model output
            temperature: Temperature for scaling (1.0 = no scaling)

        Returns:
            Probability in [0, 1], where higher = more likely spoof
        """
        scaled_logit = logit / temperature
        probability = 1.0 / (1.0 + np.exp(-scaled_logit))
        return probability

    def batch_forward(
        self,
        audio_batch: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch inference for multiple audio windows.

        Args:
            audio_batch: Tensor of shape (batch_size, samples)

        Returns:
            logits: Array of shape (batch_size,)
            embeddings: Array of shape (batch_size, 128)
        """
        # Validate batch input
        audio_tensor = self._validate_and_convert_input(audio_batch)

        # Inference
        with torch.no_grad():
            logit_tensor, embedding_tensor = self.model(audio_tensor, return_embedding=True)

        # Convert to numpy
        logits = logit_tensor.squeeze().cpu().numpy()
        embeddings = embedding_tensor.cpu().numpy()

        return logits, embeddings

    def __call__(self, audio_window):
        """Alias for forward() to make detector callable."""
        return self.forward(audio_window)


def test_detector_contract():
    """
    Test that detector follows the contract specification.
    """
    print("Testing LFCC-LCNN detector contract compliance...")

    # Initialize detector (no checkpoint, random weights)
    detector = LFCCLCNNDetector(checkpoint_path=None, device='cpu')

    # Test 1: Correct output types
    print("\n1. Testing output types and shapes:")
    audio = np.random.randn(64000).astype(np.float32) * 0.1
    logit, embedding = detector.forward(audio)

    assert isinstance(logit, float), f"logit must be float, got {type(logit)}"
    assert isinstance(embedding, np.ndarray), f"embedding must be numpy array, got {type(embedding)}"
    assert embedding.shape == (128,), f"embedding must be 128-dim, got {embedding.shape}"
    print("   [OK] Output types correct")
    print(f"     Logit: {logit:.4f} (type: {type(logit).__name__})")
    print(f"     Embedding shape: {embedding.shape}")

    # Test 2: Deterministic (no dropout at inference)
    print("\n2. Testing deterministic inference:")
    logit2, _ = detector.forward(audio)
    assert abs(logit - logit2) < 1e-6, "Model must be deterministic"
    print(f"   [OK] Deterministic (logit1={logit:.6f}, logit2={logit2:.6f})")

    # Test 3: Batch handling
    print("\n3. Testing batch inference:")
    audio_batch = np.random.randn(8, 64000).astype(np.float32) * 0.1
    logits, embeddings = detector.batch_forward(audio_batch)
    assert logits.shape == (8,), f"Expected (8,), got {logits.shape}"
    assert embeddings.shape == (8, 128), f"Expected (8, 128), got {embeddings.shape}"
    print(f"   [OK] Batch processing works")
    print(f"     Input: {audio_batch.shape}")
    print(f"     Logits: {logits.shape}")
    print(f"     Embeddings: {embeddings.shape}")

    # Test 4: Different input lengths (padding/cropping)
    print("\n4. Testing automatic padding/cropping:")
    # Too short
    short_audio = np.random.randn(32000).astype(np.float32) * 0.1  # 2 seconds
    logit_short, _ = detector.forward(short_audio)
    print(f"   [OK] Short audio (2s) handled: logit={logit_short:.4f}")

    # Too long
    long_audio = np.random.randn(96000).astype(np.float32) * 0.1  # 6 seconds
    logit_long, _ = detector.forward(long_audio)
    print(f"   [OK] Long audio (6s) handled: logit={logit_long:.4f}")

    # Test 5: Input format conversion (numpy vs torch)
    print("\n5. Testing input format conversion:")
    audio_torch = torch.from_numpy(audio).float()
    logit_torch, _ = detector.forward(audio_torch)
    assert abs(logit - logit_torch) < 1e-6, "Numpy and torch inputs should give same result"
    print(f"   [OK] Torch tensor input works (logit={logit_torch:.4f})")

    # Test 6: Probability conversion
    print("\n6. Testing probability conversion:")
    prob = detector.predict_probability(logit)
    assert 0 <= prob <= 1, f"Probability must be in [0, 1], got {prob}"
    print(f"   [OK] Probability conversion works")
    print(f"     Logit: {logit:.4f} -> Probability: {prob:.4f}")

    print("\n" + "="*60)
    print("[PASS] All contract tests PASSED!")
    print("="*60)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run contract tests
        test_detector_contract()

    elif len(sys.argv) > 1 and sys.argv[1] == '--demo':
        # Demo usage
        print("="*60)
        print("LFCC-LCNN Detector Demo")
        print("="*60)

        # Initialize detector
        detector = LFCCLCNNDetector(checkpoint_path=None, device='cpu')

        # Generate fake audio (sine wave + noise)
        sample_rate = 16000
        duration = 4.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) + np.random.randn(len(t)) * 0.1
        audio = audio.astype(np.float32)

        # Inference
        logit, embedding = detector(audio)

        print(f"\nInput audio: {len(audio)} samples ({duration}s at {sample_rate}Hz)")
        print(f"Spoof score (logit): {logit:.4f}")
        print(f"Probability (uncalibrated): {detector.predict_probability(logit):.4f}")
        print(f"Embedding L2 norm: {np.linalg.norm(embedding):.4f}")

        # Interpretation
        if logit > 0:
            verdict = "SPOOF (synthetic)"
        else:
            verdict = "BONAFIDE (real)"

        print(f"\nVerdict: {verdict}")
        print("\nNote: Model has random weights (no training). Scores are meaningless.")

    else:
        print("LFCC-LCNN Detector")
        print("\nUsage:")
        print("  python detector.py --test    # Run contract compliance tests")
        print("  python detector.py --demo    # Run demo inference")
        print("\nFor production use:")
        print("  from models.detector import LFCCLCNNDetector")
        print("  detector = LFCCLCNNDetector(checkpoint_path='checkpoints/best.pth')")
        print("  logit, embedding = detector.forward(audio_window)")
