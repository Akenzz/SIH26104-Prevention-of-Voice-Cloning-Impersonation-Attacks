"""
train.py — Training loop for the WavLM-based deepfake speech detector.

What this script does:
    1. Loads "train" and "dev" splits from the manifest CSV.
    2. Builds a WavLMClassifier (frozen backbone + trainable head).
    3. Trains using binary cross-entropy loss with AdamW — gradients only
       flow through the head; the backbone never updates.
    4. After every epoch, evaluates on the dev split.
    5. Saves the best checkpoint (lowest dev loss) to CHECKPOINT_PATH.

Label convention (IMPORTANT — never flip):
    bonafide = 0  (real)
    spoof    = 1  (fake)

# TODO: To use your real dataset, change MANIFEST_CSV below.
#       Everything else stays the same.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpeechDataset, WINDOW_SAMPLES
from model import WavLMClassifier

# ─── Hyperparameters / paths (edit these or pass via CLI) ─────────────────────
# TODO: Change MANIFEST_CSV to point at your real dataset manifest.
MANIFEST_CSV    = "wavlm-base-plus/data/manifest.csv"
CHECKPOINT_PATH = "wavlm-base-plus/checkpoints/best_model_v2.pt"
BATCH_SIZE      = 8
NUM_EPOCHS      = 10
LEARNING_RATE   = 1e-3      # Head-only learning rate
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 0         # Set > 0 if your system has multiple CPU cores
SEED            = 42


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    """Run one full pass over the training set. Returns mean loss."""
    model.train()
    # Keep backbone frozen / in eval mode even during model.train()
    model.backbone.eval()

    total_loss  = 0.0
    total_items = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:>3}/{total_epochs} [train]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    for waveform, labels in pbar:
        waveform = waveform.to(device)          # (B, T)
        labels   = labels.float().to(device)    # (B,)  — BCEWithLogitsLoss wants float

        optimizer.zero_grad()

        logits = model(waveform).squeeze(1)     # (B,)
        loss   = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss  += loss.item() * len(labels)
        total_items += len(labels)

        # Live loss readout in the progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_items


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a data split. Returns (mean_loss, accuracy, eer)."""
    from sklearn.metrics import roc_curve
    import numpy as np

    model.eval()

    total_loss    = 0.0
    total_items   = 0
    total_correct = 0
    
    all_labels = []
    all_logits = []

    pbar = tqdm(loader, desc="              [dev]  ", unit="batch", leave=False, dynamic_ncols=True)
    for waveform, labels in pbar:
        waveform = waveform.to(device)
        labels   = labels.to(device)

        logits = model(waveform).squeeze(1)            # (B,)
        loss   = criterion(logits, labels.float())

        preds = (logits > 0).long()                    # threshold at 0
        total_correct += (preds == labels).sum().item()
        total_loss    += loss.item() * len(labels)
        total_items   += len(labels)
        
        all_labels.append(labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    mean_loss = total_loss / total_items
    accuracy  = total_correct / total_items
    
    # Compute EER
    all_labels_arr = np.concatenate(all_labels)
    all_logits_arr = np.concatenate(all_logits)
    fpr, tpr, thresholds = roc_curve(all_labels_arr, all_logits_arr, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    return mean_loss, accuracy, eer


def main(args):
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[train] Loading datasets ...")
    train_ds = SpeechDataset(args.manifest, split="train")
    dev_manifest = args.dev_manifest if args.dev_manifest else args.manifest
    dev_ds   = SpeechDataset(dev_manifest, split="dev")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    print(f"[train] Train samples: {len(train_ds)}  |  Dev samples: {len(dev_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = WavLMClassifier().to(device)

    # Optimizer only sees the head parameters — backbone is frozen.
    head_params = list(model.head.parameters())
    print(f"[train] Trainable head parameters: "
          f"{sum(p.numel() for p in head_params):,}")

    optimizer = torch.optim.AdamW(
        head_params, lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # ── Checkpoint directory ───────────────────────────────────────────────────
    ckpt_dir = os.path.dirname(args.checkpoint)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    # ── Resume Logic ───────────────────────────────────────────────────────────
    start_epoch = 1
    best_dev_eer = float("inf")
    
    if args.resume and os.path.exists(args.checkpoint):
        print(f"\n[train] Resuming from checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            last_epoch = state.get("epoch", 0)
            start_epoch = last_epoch + 1
            best_dev_eer = state.get("dev_eer", float("inf"))
            dev_loss = state.get("dev_loss", 0.0)
            print(f"         ✓ Restored from Epoch {last_epoch} (dev_eer={best_dev_eer:.2%}, dev_loss={dev_loss:.4f})")
        else:
            print("         ⚠ Checkpoint is old format, loading weights only. Starting from Epoch 1.")
            model.load_state_dict(state)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Dev Loss':>9}  {'Dev Acc':>8}  {'Dev EER':>9}")
    print("─" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss        = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs
        )
        dev_loss, dev_acc, dev_eer = evaluate(model, dev_loader, criterion, device)

        print(f"{epoch:>5}  {train_loss:>10.4f}  {dev_loss:>9.4f}  {dev_acc:>7.1%}  {dev_eer:>8.2%}")

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            torch.save(
                {
                    "epoch"            : epoch,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dev_loss"         : dev_loss,
                    "dev_acc"          : dev_acc,
                    "dev_eer"          : dev_eer,
                },
                args.checkpoint,
            )
            print(f"         ✓ Saved best checkpoint (dev_eer={dev_eer:.2%}, dev_loss={dev_loss:.4f})")

    print("─" * 72)
    print(f"\n[train] Done. Best dev EER: {best_dev_eer:.2%}")
    print(f"[train] Checkpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the WavLM deepfake speech detector"
    )
    parser.add_argument(
        "--manifest", default=MANIFEST_CSV,
        help="Path to manifest CSV  [default: %(default)s]"
    )
    parser.add_argument(
        "--dev-manifest", default=None,
        help="Path to a separate dev manifest CSV (if your train and dev splits are in separate files)"
    )
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH,
        help="Where to save the best checkpoint  [default: %(default)s]"
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs",     type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--resume",     action="store_true", help="Resume training from the checkpoint")

    main(parser.parse_args())
