"""
finetune_bandwidth_robust.py — retrain the hybrid LFCC-LCNN so its verdict does
not depend on which resampler produced the audio.

THE BUG THIS FIXES
------------------
Training preprocessing resampled with librosa/soxr, which brickwalls 7.9-8 kHz to
about -61 dB. The realtime backend resamples with scipy resample_poly, which
leaves that band only ~3 dB down. The corpus's native sample rates are split by
label (bonafide 75.8% already 16 kHz -> never resampled -> full band; spoof 52.5%
at 22050 Hz -> resampled -> hole), so "hole near Nyquist" became a label proxy.
Consequence: generators that were IN training read bonafide live. Measured on the
shipped models, the same clip scores -9.16 (bonafide) through scipy and +20.12
(SPOOF) through librosa -- a 29-logit swing from the resampler alone.

WHY AUGMENTATION ALONE IS NOT THE FIX
-------------------------------------
The obvious move is to randomize the resampler fingerprint on both classes. It
helps (label/top-band correlation -0.204 -> -0.045 over 300 windows) but cannot
close the hole, because the preprocessed WAVs on disk are ALREADY brickwalled:
augmentation can add holes, never restore a band. Only 1.6% of spoof windows are
natively full-band versus 13.6% of bonafide, so "full band => bonafide" survives
any amount of randomization.

THE FIX
-------
Delete the untrustworthy region from both classes and every split: lowpass at
7000 Hz, below which the two resamplers agree to 0.02-0.7 dB. Verified on the
shipped models before spending this retrain (diag_band_gate.py): the mean
|scipy - librosa| verdict gap collapses from 13.35 to 0.05 logits. Bandwidth
augmentation is kept ON TOP, so the model also stops keying on the exact band
edge and tolerates narrowband/telephony input.

The gate costs the model the 7-8 kHz octave. That is deliberate: that octave was
not carrying voice evidence it could use at serving time, it was carrying a
preprocessing artifact.

*** DEPLOYMENT REQUIREMENT ***
A model trained with the gate MUST be served with the same gate. Serving it
ungated feeds it an octave of energy it never saw and biases it spoofward (the
diagnostic shows exactly this on the ungated shipped models). The matching
inference-side change is BAND_GATE_HZ in the LFCC expert.

Outputs (checkpoints/, never touches hybrid_clean.pth or
hybrid_clean_plus_newclips_final.pth, which is gitignored, absent from the Hub,
and therefore unrecoverable if overwritten):
  * hybrid_br_best.pth   — best dev EER across the run
  * hybrid_br_final.pth  — latest completed epoch (crash recovery)

Run from lfcc-detector/:
  python finetune_bandwidth_robust.py 2>&1 | tee finetune_bandwidth_robust.log
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data.dataset import AudioDataset, collate_fn          # noqa: E402
from models.lfcc_lcnn import LFCCLCNNWithFeatureExtraction  # noqa: E402
from training.config import TrainingConfig                  # noqa: E402
from training.train import train_one_epoch, evaluate, set_seed  # noqa: E402

REPO = HERE.parent
# Base corpus + Amogh's modern-engine clips: the newclips fine-tune win is real
# (it was measured through the scipy deployment path), so keep that data.
TRAIN_MANIFEST = REPO / "data_pipeline" / "manifests" / "hybrid_plus_newclips_train.csv"
DEV_MANIFEST = REPO / "data_pipeline" / "manifests" / "hybrid_vad_chunks_dev.csv"
RESUME_FROM = HERE / "checkpoints" / "hybrid_clean.pth"
OUT_DIR = HERE / "checkpoints"
BEST_PATH = OUT_DIR / "hybrid_br_best.pth"
FINAL_PATH = OUT_DIR / "hybrid_br_final.pth"

# --- the fix ---
BAND_GATE_HZ = 7000.0     # both classes, ALL splits, and required at inference
BANDWIDTH_AUGMENT = True  # randomize resampler fingerprint, label-independently

# --- hyperparameters (this RTX-3050 box recipe) ---
EPOCHS = 6
BATCH_SIZE = 8        # batch 8 / num_workers 0 avoids the 1455/ptxas/OOM crashes
NUM_WORKERS = 0
LR = 5e-5             # fine-tune LR, half the from-scratch 1e-4
LOG_EVERY = 200


def _save(path, model, optimizer, epoch, dev_eer, config, note):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eer": dev_eer,
        "config": config,
        "note": note,
        # Persisted so inference cannot silently forget the gate. The expert
        # should read this rather than hardcoding a second copy of the number.
        "band_gate_hz": BAND_GATE_HZ,
        "bandwidth_augment": BANDWIDTH_AUGMENT,
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
    for p in (BEST_PATH, FINAL_PATH):
        if p.exists():
            print(f"[INFO] will overwrite previous run output {p.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"[INFO] warm-start: {RESUME_FROM.name}   LR={LR}   epochs={EPOCHS}   batch={BATCH_SIZE}")
    print(f"[INFO] band gate: {BAND_GATE_HZ:.0f} Hz (all splits)   "
          f"bandwidth augment: {BANDWIDTH_AUGMENT}")

    # Gate applies to dev too -- dev must match train or the EER is meaningless.
    train_ds = AudioDataset(str(TRAIN_MANIFEST), split="train",
                            window_sec=config.window_sec, augment=True,
                            bandwidth_augment=BANDWIDTH_AUGMENT,
                            band_gate_hz=BAND_GATE_HZ)
    dev_ds = AudioDataset(str(DEV_MANIFEST), split="dev",
                          window_sec=config.window_sec,
                          band_gate_hz=BAND_GATE_HZ)

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=NUM_WORKERS,
                         pin_memory=False, persistent_workers=(NUM_WORKERS > 0))
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(dev_ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs)

    model = LFCCLCNNWithFeatureExtraction(
        sample_rate=config.sample_rate, n_lfcc=config.n_lfcc,
        with_deltas=config.with_deltas, embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    ckpt = torch.load(str(RESUME_FROM), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] loaded warm-start weights "
          f"(stored best_eer={float(ckpt.get('best_eer', float('nan')))*100:.4f}%)")

    label_counts = train_ds.df["label"].value_counts()
    n_bona = int(label_counts.get("bonafide", 0))
    n_spoof = int(label_counts.get("spoof", 0))
    pos_weight = torch.tensor([n_bona / max(n_spoof, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[INFO] pos_weight={n_bona / max(n_spoof, 1):.4f}  "
          f"(bonafide={n_bona:,} / spoof={n_spoof:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=config.weight_decay)
    print(f"[INFO] train rows: {len(train_ds):,}   dev rows: {len(dev_ds):,}")

    # Baseline: the warm-start weights measured ON GATED dev. This is expected to
    # look BAD -- those weights were trained ungated, and the gate removes the
    # band they lean on. It is the number the fine-tune has to beat, not a
    # regression threshold against the shipped 3.03%.
    t0 = time.time()
    _, dev_eer0, _ = evaluate(model, dev_loader, criterion, device, log_every=LOG_EVERY * 2)
    print(f"[BASELINE] gated dev EER before fine-tune = {dev_eer0*100:.4f}%  "
          f"({time.time()-t0:.0f}s)", flush=True)

    best_eer = float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                          device, config.grad_clip, log_every=LOG_EVERY)
        dev_loss, dev_eer, dev_thresh = evaluate(model, dev_loader, criterion, device,
                                                 log_every=LOG_EVERY * 2)
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} [{time.time()-t0:.0f}s]  "
              f"train loss={tr_loss:.4f} acc={tr_acc*100:.1f}%  |  "
              f"dev loss={dev_loss:.4f} EER={dev_eer*100:.4f}% thresh={dev_thresh:.4f}", flush=True)

        _save(FINAL_PATH, model, optimizer, epoch, dev_eer, config,
              f"epoch {epoch}, band gate {BAND_GATE_HZ:.0f} Hz")
        if dev_eer < best_eer:
            best_eer = dev_eer
            _save(BEST_PATH, model, optimizer, epoch, dev_eer, config,
                  f"best gated dev EER at epoch {epoch}, band gate {BAND_GATE_HZ:.0f} Hz")
            print(f"  * new best gated dev EER {best_eer*100:.4f}% -> {BEST_PATH.name}", flush=True)

    print("\n" + "=" * 64)
    print("BANDWIDTH-ROBUST FINE-TUNE COMPLETE")
    print("=" * 64)
    print(f"band gate            : {BAND_GATE_HZ:.0f} Hz  (MUST be applied at inference)")
    print(f"gated dev EER before : {dev_eer0*100:.4f}%")
    print(f"gated dev EER best   : {best_eer*100:.4f}%")
    print(f"best checkpoint      : {BEST_PATH}")
    print(f"final checkpoint     : {FINAL_PATH}")
    print("\nNext: python diag_bandwidth_verdict.py  (scipy-vs-librosa gap + local clips)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
