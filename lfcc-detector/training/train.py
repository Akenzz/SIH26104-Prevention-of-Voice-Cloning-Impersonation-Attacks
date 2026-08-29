"""
Training script for LFCC-LCNN Voice Cloning / Synthetic Speech Detector.
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

# Devanagari metadata + the box/★ glyphs below crash Windows cp1252 stdout — force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import AudioDataset, collate_fn
from models.lfcc_lcnn import LFCCLCNNWithFeatureExtraction
from training.config import TrainingConfig


def set_seed(seed: int):
    """Make a run reproducible across python/numpy/torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray):
    """
    Compute Equal Error Rate (EER) given bonafide and spoof score distributions.
    Higher score = more spoof.
    """
    labels = np.concatenate([np.zeros_like(bonafide_scores), np.ones_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # EER is where FPR == FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold = thresholds[eer_idx]

    return eer, threshold


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip=1.0,
                    log_every: int = 50):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(dataloader)

    for batch_idx, (audios, labels, _) in enumerate(dataloader):
        audios = audios.to(device)
        labels = labels.to(device).float().unsqueeze(1)  # (batch, 1)

        optimizer.zero_grad()
        logits, _ = model(audios, return_embedding=False)
        loss = criterion(logits, labels)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)

        # Real-time per-batch progress
        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == num_batches:
            running_loss = total_loss / total
            running_acc  = correct / total * 100
            print(
                f"  Batch [{batch_idx+1:>4}/{num_batches}]  "
                f"loss: {running_loss:.4f}  acc: {running_acc:.1f}%",
                flush=True
            )

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy  = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, log_every: int = 100):
    model.eval()
    total_loss = 0.0
    bonafide_scores = []
    spoof_scores = []
    num_batches = len(dataloader)

    print(f"  [Dev eval — {num_batches} batches]", flush=True)

    with torch.no_grad():
        for batch_idx, (audios, labels, _) in enumerate(dataloader):
            audios = audios.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            logits, _ = model(audios, return_embedding=False)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)

            scores = logits.squeeze(1).cpu().numpy()
            target_labels = labels.squeeze(1).cpu().numpy()

            for score, label in zip(scores, target_labels):
                if label == 0:
                    bonafide_scores.append(score)
                else:
                    spoof_scores.append(score)

            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == num_batches:
                print(
                    f"  Dev batch [{batch_idx+1:>4}/{num_batches}]  "
                    f"processed: {len(bonafide_scores) + len(spoof_scores)} samples",
                    flush=True
                )

    total_samples = len(bonafide_scores) + len(spoof_scores)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    bonafide_scores = np.array(bonafide_scores)
    spoof_scores = np.array(spoof_scores)

    if len(bonafide_scores) > 0 and len(spoof_scores) > 0:
        eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    else:
        eer, threshold = 0.5, 0.0

    return avg_loss, eer, threshold


def main():
    parser = argparse.ArgumentParser(description="Train LFCC-LCNN voice cloning detector")
    parser.add_argument("--train-manifest", type=str, default="../data_pipeline/manifests/asvspoof19_train.csv")
    parser.add_argument("--dev-manifest", type=str, default="../data_pipeline/manifests/asvspoof19_dev.csv")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes (0 = serial, starves the GPU on Windows)")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print batch loss/acc every N batches (default: 50)")
    parser.add_argument("--checkpoint-name", type=str, default="best_lfcc_lcnn.pth",
                        help="Filename to save the best checkpoint (default: best_lfcc_lcnn.pth)")
    parser.add_argument("--pin-memory", action="store_true",
                        help="Pin host memory for faster H->D copies. OFF by default: "
                             "Windows+CUDA can raise 'CUDA error: resource already mapped' "
                             "in the pin-memory thread. Disk I/O is the real bottleneck here.")
    args = parser.parse_args()

    config = TrainingConfig(
        train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    log_every       = args.log_every
    checkpoint_name = args.checkpoint_name
    num_workers     = args.num_workers

    set_seed(config.seed)
    print(f"[INFO] Seed set to {config.seed}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[INFO] Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(
            "[WARN] CUDA not available — training on CPU will be very slow.\n"
            "       Install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/"
        )

    # Load datasets
    train_dataset = AudioDataset(config.train_manifest, split='train', window_sec=config.window_sec)
    dev_dataset = AudioDataset(config.dev_manifest, split='dev', window_sec=config.window_sec)

    # On Windows the main-guard (if __name__=='__main__') lets num_workers>0 spawn
    # cleanly. num_workers=0 decodes serially and starves the GPU (~1 batch/s).
    loader_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(args.pin_memory and device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, **loader_kwargs
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs
    )

    # Initialize model
    model = LFCCLCNNWithFeatureExtraction(
        sample_rate=config.sample_rate,
        n_lfcc=config.n_lfcc,
        with_deltas=config.with_deltas,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout
    ).to(device)

    # Class imbalance: upweight the positive (spoof) class so the model isn't
    # rewarded for predicting "real" by default.
    label_counts = train_dataset.df['label'].value_counts()
    n_bonafide = int(label_counts.get('bonafide', 0))
    n_spoof    = int(label_counts.get('spoof', 0))
    if n_spoof > 0:
        pos_weight_val = n_bonafide / n_spoof
        pos_weight = torch.tensor([pos_weight_val], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[INFO] pos_weight = {pos_weight_val:.4f}  (bonafide={n_bonafide:,} / spoof={n_spoof:,})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("[WARN] No spoof samples found in train split; using unweighted BCE.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    print(f"\n--- Starting Training ---")
    print(f"  Train rows  : {len(train_dataset):,}")
    print(f"  Dev rows    : {len(dev_dataset):,}")
    print(f"  Epochs      : {config.epochs}")
    print(f"  Batch size  : {config.batch_size}")
    print(f"  LR          : {config.learning_rate}")
    print(f"  Log every   : {log_every} batches")
    print(f"  Checkpoint  : {Path(config.output_dir) / checkpoint_name}")
    print(f"{'-'*60}")

    best_eer = 1.0
    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            config.grad_clip, log_every=log_every
        )
        dev_loss, dev_eer, dev_thresh = evaluate(
            model, dev_loader, criterion, device, log_every=log_every * 2
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{config.epochs:02d} [{elapsed:.1f}s]  "
            f"Train loss={train_loss:.4f}  acc={train_acc*100:.1f}%  "
            f"| Dev loss={dev_loss:.4f}  EER={dev_eer*100:.2f}%  (thresh={dev_thresh:.4f})",
            flush=True
        )

        if dev_eer < best_eer:
            best_eer = dev_eer
            checkpoint_path = Path(config.output_dir) / checkpoint_name
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_eer': best_eer,
                'config': config
            }, checkpoint_path)
            print(f"  * New best! EER={best_eer*100:.4f}%  -> saved {checkpoint_path}", flush=True)

    print(f"\nTraining Complete! Best Dev EER: {best_eer*100:.2f}%")


if __name__ == '__main__':
    main()
