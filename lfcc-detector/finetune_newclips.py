"""
finetune_newclips.py — warm-start fine-tune of the shipped hybrid LFCC-LCNN on
the base corpus + Amogh's modern-engine clips.

Replicates the friend's recipe ("same data as the latest hybrid + these new
clips, train 4-5 epochs") but adapted for a FULL-CNN LFCC model rather than
WavLM's frozen head:

  * we WARM-START from checkpoints/hybrid_clean.pth (the shipped Expert-2) instead
    of training from scratch, because 4-5 epochs from scratch would badly
    undertrain this CNN (it needed ~40). Warm-start + few epochs = a gentle nudge
    toward the new engines without forgetting the 43.8k-clip corpus.
  * FRESH AdamW at LR 5e-5 (half the 1e-4 used for the from-scratch run) — a
    fine-tune LR. We deliberately do NOT resume the old optimizer momentum.
  * dev set is the UNCHANGED hybrid dev manifest, so the dev EER here is directly
    comparable to the shipped model's 3.03%. We measure dev EER BEFORE training
    (epoch 0) as an explicit regression guard.

Outputs (checkpoints/, never touches hybrid_clean.pth or realtime-backend):
  * hybrid_clean_plus_newclips_best.pth   — best dev EER across the fine-tune
  * hybrid_clean_plus_newclips_final.pth  — latest completed epoch (crash recovery)

Run from lfcc-detector/:
  python finetune_newclips.py 2>&1 | tee finetune_newclips.log
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data.dataset import AudioDataset, collate_fn          # noqa: E402
from models.lfcc_lcnn import LFCCLCNNWithFeatureExtraction  # noqa: E402
from training.config import TrainingConfig                  # noqa: E402
from training.train import train_one_epoch, evaluate, set_seed  # noqa: E402

REPO = HERE.parent
TRAIN_MANIFEST = REPO / "data_pipeline" / "manifests" / "hybrid_plus_newclips_train.csv"
DEV_MANIFEST = REPO / "data_pipeline" / "manifests" / "hybrid_vad_chunks_dev.csv"
RESUME_FROM = HERE / "checkpoints" / "hybrid_clean.pth"
OUT_DIR = HERE / "checkpoints"
BEST_PATH = OUT_DIR / "hybrid_clean_plus_newclips_best.pth"
FINAL_PATH = OUT_DIR / "hybrid_clean_plus_newclips_final.pth"

# --- fine-tune hyperparameters ---
EPOCHS = 5            # friend's recipe: 4-5
BATCH_SIZE = 8        # this RTX-3050 box: batch 8 / num_workers 0 is the safe recipe
NUM_WORKERS = 0       # avoids the 1455/ptxas/OOM crashes seen with workers>0 here
LR = 5e-5             # half the from-scratch 1e-4 — a fine-tune LR
LOG_EVERY = 200


def _save(path, model, optimizer, epoch, dev_eer, config, note):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eer": dev_eer,
        "config": config,
        "note": note,
    }, path)


def main() -> int:
    config = TrainingConfig(
        train_manifest=str(TRAIN_MANIFEST),
        dev_manifest=str(DEV_MANIFEST),
        output_dir=str(OUT_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    set_seed(config.seed)

    if not RESUME_FROM.is_file():
        print(f"[ERR] warm-start checkpoint not found: {RESUME_FROM}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"[INFO] warm-start: {RESUME_FROM.name}   LR={LR}   epochs={EPOCHS}   batch={BATCH_SIZE}")

    # --- datasets: train = base+newclips (augmented), dev = unchanged hybrid dev ---
    train_ds = AudioDataset(str(TRAIN_MANIFEST), split="train",
                            window_sec=config.window_sec, augment=True)
    dev_ds = AudioDataset(str(DEV_MANIFEST), split="dev", window_sec=config.window_sec)

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=NUM_WORKERS,
                         pin_memory=False, persistent_workers=(NUM_WORKERS > 0))
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(dev_ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs)

    # --- model + warm-started weights ---
    model = LFCCLCNNWithFeatureExtraction(
        sample_rate=config.sample_rate, n_lfcc=config.n_lfcc,
        with_deltas=config.with_deltas, embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    ckpt = torch.load(str(RESUME_FROM), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    prior_best = float(ckpt.get("best_eer", float("nan")))
    print(f"[INFO] loaded warm-start weights (stored best_eer={prior_best*100:.4f}%)")

    # pos_weight from the AUGMENTED train manifest (base + new clips)
    label_counts = train_ds.df["label"].value_counts()
    n_bona = int(label_counts.get("bonafide", 0))
    n_spoof = int(label_counts.get("spoof", 0))
    pos_weight = torch.tensor([n_bona / max(n_spoof, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[INFO] pos_weight={n_bona / max(n_spoof, 1):.4f}  (bonafide={n_bona:,} / spoof={n_spoof:,})")

    # FRESH AdamW — do NOT resume old momentum; this is a gentle fine-tune.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=config.weight_decay)

    print(f"[INFO] train rows: {len(train_ds):,}   dev rows: {len(dev_ds):,}")

    # --- regression baseline: dev EER of the warm-start model BEFORE any step ---
    t0 = time.time()
    _, dev_eer0, _ = evaluate(model, dev_loader, criterion, device, log_every=LOG_EVERY * 2)
    print(f"[BASELINE] dev EER before fine-tune = {dev_eer0*100:.4f}%  "
          f"({time.time()-t0:.0f}s)  <-- must not regress much above this", flush=True)

    best_eer = dev_eer0
    _save(BEST_PATH, model, optimizer, 0, dev_eer0, config, "baseline (pre-finetune weights)")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                          device, config.grad_clip, log_every=LOG_EVERY)
        dev_loss, dev_eer, dev_thresh = evaluate(model, dev_loader, criterion, device,
                                                 log_every=LOG_EVERY * 2)
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} [{time.time()-t0:.0f}s]  "
              f"train loss={tr_loss:.4f} acc={tr_acc*100:.1f}%  |  "
              f"dev loss={dev_loss:.4f} EER={dev_eer*100:.4f}% thresh={dev_thresh:.4f}", flush=True)

        # per-epoch crash-recovery checkpoint (always overwrite)
        _save(FINAL_PATH, model, optimizer, epoch, dev_eer, config, f"epoch {epoch} of finetune")
        if dev_eer < best_eer:
            best_eer = dev_eer
            _save(BEST_PATH, model, optimizer, epoch, dev_eer, config, f"best dev EER at epoch {epoch}")
            print(f"  * new best dev EER {best_eer*100:.4f}% -> {BEST_PATH.name}", flush=True)

    print("\n" + "=" * 64)
    print("FINE-TUNE COMPLETE")
    print("=" * 64)
    print(f"dev EER before      : {dev_eer0*100:.4f}%")
    print(f"dev EER best (5 ep) : {best_eer*100:.4f}%")
    print(f"best checkpoint     : {BEST_PATH}")
    print(f"final checkpoint    : {FINAL_PATH}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
