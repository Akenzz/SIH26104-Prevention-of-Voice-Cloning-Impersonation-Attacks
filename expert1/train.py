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

from expert1.dataset import SpeechDataset, WINDOW_SAMPLES
from expert1.model import WavLMClassifier

# ─── Hyperparameters / paths (edit these or pass via CLI) ─────────────────────
# TODO: Change MANIFEST_CSV to point at your real dataset manifest.
# Paths are relative to the project root (where you run `python -m expert1.train`).
MANIFEST_CSV    = "expert1/data/manifest.csv"
CHECKPOINT_PATH = "expert1/checkpoints/best_model.pt"
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
    """Evaluate on a data split. Returns (mean_loss, accuracy)."""
    model.eval()

    total_loss    = 0.0
    total_items   = 0
    total_correct = 0

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

    mean_loss = total_loss / total_items
    accuracy  = total_correct / total_items
    return mean_loss, accuracy


def main(args):
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[train] Loading datasets ...")
    train_ds = SpeechDataset(args.manifest, split="train")
    dev_ds   = SpeechDataset(args.manifest, split="dev")

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

    # ── Training loop ─────────────────────────────────────────────────────────
    best_dev_loss = float("inf")
    print("\n" + "─" * 60)
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Dev Loss':>9}  {'Dev Acc':>8}")
    print("─" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss        = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs
        )
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)

        print(f"{epoch:>5}  {train_loss:>10.4f}  {dev_loss:>9.4f}  {dev_acc:>7.1%}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(
                {
                    "epoch"            : epoch,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dev_loss"         : dev_loss,
                    "dev_acc"          : dev_acc,
                },
                args.checkpoint,
            )
            print(f"         ✓ Saved best checkpoint (dev_loss={dev_loss:.4f})")

    print("─" * 60)
    print(f"\n[train] Done. Best dev loss: {best_dev_loss:.4f}")
    print(f"[train] Checkpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the WavLM deepfake speech detector"
    )
    parser.add_argument(
        "--manifest", default=MANIFEST_CSV,
        help="Path to manifest CSV  [default: %(default)s]"
        # TODO: Pass --manifest /path/to/real_manifest.csv to use your real data
    )
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH,
        help="Where to save the best checkpoint  [default: %(default)s]"
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs",     type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)

    main(parser.parse_args())
