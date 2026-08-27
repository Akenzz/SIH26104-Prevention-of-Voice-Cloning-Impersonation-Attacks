"""
Training script for LFCC-LCNN Voice Cloning / Synthetic Speech Detector.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import AudioDataset, collate_fn
from models.lfcc_lcnn import LFCCLCNNWithFeatureExtraction
from training.config import TrainingConfig


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
    args = parser.parse_args()

    config = TrainingConfig(
        train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

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

    # num_workers=0 is required on Windows to avoid DataLoader multiprocessing issues
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == 'cuda')
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == 'cuda')
    )

    # Initialize model
    model = LFCCLCNNWithFeatureExtraction(
        sample_rate=config.sample_rate,
        n_lfcc=config.n_lfcc,
        with_deltas=config.with_deltas,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    print("\n--- Starting Training ---")
    best_eer = 1.0

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, config.grad_clip)
        dev_loss, dev_eer, dev_thresh = evaluate(model, dev_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{config.epochs:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.1f}% | Dev Loss: {dev_loss:.4f}, Dev EER: {dev_eer*100:.2f}% (thresh: {dev_thresh:.4f})")

        if dev_eer < best_eer:
            best_eer = dev_eer
            checkpoint_path = Path(config.output_dir) / config.checkpoint_name
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_eer': best_eer,
                'config': config
            }, checkpoint_path)
            print(f"  [OK] Saved new best checkpoint to {checkpoint_path}")

    print(f"\nTraining Complete! Best Dev EER: {best_eer*100:.2f}%")


if __name__ == '__main__':
    main()