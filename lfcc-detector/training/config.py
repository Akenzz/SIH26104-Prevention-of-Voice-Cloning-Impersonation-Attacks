"""
Training configuration parameters for LFCC-LCNN detector.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    # Dataset and paths
    train_manifest: str = 'data/manifests/train.csv'
    dev_manifest: str = 'data/manifests/dev.csv'
    output_dir: str = 'checkpoints'
    checkpoint_name: str = 'best_lfcc_lcnn.pth'
    
    # Audio parameters
    sample_rate: int = 16000
    window_sec: float = 4.0
    
    # LFCC Feature parameters
    n_lfcc: int = 20
    n_filters: int = 70
    n_fft: int = 512
    with_deltas: bool = True
    
    # Model parameters
    embedding_dim: int = 128
    dropout: float = 0.5
    
    # Optimization parameters
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    grad_clip: float = 1.0
    
    # Augmentation
    use_augmentation: bool = True
    
    # Environment & Reproducibility
    seed: int = 42
    device: Optional[str] = None
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)