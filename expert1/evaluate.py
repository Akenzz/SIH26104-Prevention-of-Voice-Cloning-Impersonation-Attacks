"""
evaluate.py — Evaluate a trained checkpoint on the "test" split.

Metrics printed:
    • Accuracy          — sanity-check; threshold at logit=0
    • EER (Equal Error Rate) — the primary metric for anti-spoofing systems.

EER is computed from the ROC curve (using sklearn) by finding the threshold
where False Accept Rate ≈ False Reject Rate.

Label convention (IMPORTANT — never flip):
    bonafide = 0  (real / negative class)
    spoof    = 1  (fake / positive class)
    Higher logit = more evidence of spoof.

Speed tips:
    Quick sanity check  (< 3 min):   --max-samples 5000
    Balanced quick check             --max-samples 5000 --batch-size 64
    Full official eval  (~ 25 min):  no flag  (default runs all 71K samples)

# TODO: Change MANIFEST_CSV below (or pass --manifest) to use your real dataset.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve
from tqdm import tqdm

from expert1.dataset import SpeechDataset, WINDOW_SAMPLES
from expert1.model import WavLMClassifier

# ─── Defaults ─────────────────────────────────────────────────────────────────
MANIFEST_CSV    = "expert1/data/asvspoof_manifest.csv"
CHECKPOINT_PATH = "expert1/checkpoints/best_model.pt"
BATCH_SIZE      = 64    # no grads during eval → large batches are fine
NUM_WORKERS     = 4     # parallel FLAC loading


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
    Compute Equal Error Rate (EER) from binary labels and raw scores.

    The EER is the threshold at which the False Positive Rate (FAR)
    equals the False Negative Rate (FRR).

    Args:
        labels: Ground-truth binary labels (0 = bonafide, 1 = spoof).
        scores: Continuous scores (higher = more likely spoof).

    Returns:
        (eer, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    eer_idx   = np.argmin(np.abs(fpr - fnr))
    eer       = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    threshold = float(thresholds[eer_idx])
    return eer, threshold


@torch.no_grad()
def run_inference(model, loader, device):
    """
    Forward-pass the entire loader with fp16 autocast for ~2x GPU speedup.
    Returns (all_labels, all_logits) as 1-D numpy arrays.
    """
    model.eval()
    all_labels = []
    all_logits = []

    # torch.autocast runs the heavy WavLM attention in fp16 on GPU,
    # cutting each batch roughly in half time-wise with no accuracy loss.
    use_amp = device.type == "cuda"

    pbar = tqdm(
        loader,
        desc="[evaluate] Inference",
        unit="batch",
        dynamic_ncols=True,
    )
    for waveform, labels in pbar:
        waveform = waveform.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(waveform).squeeze(1)   # (B,)

        all_labels.append(labels.numpy())
        all_logits.append(logits.float().cpu().numpy())

        samples_done = sum(len(x) for x in all_labels)
        pbar.set_postfix(samples=f"{samples_done:,}/{len(loader.dataset):,}")

    return np.concatenate(all_labels), np.concatenate(all_logits)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Using device : {device}")
    if device.type == "cuda":
        print(f"[evaluate] GPU          : {torch.cuda.get_device_name(0)}")
        print(f"[evaluate] fp16 autocast: enabled (faster inference)")

    # ── Load test data ─────────────────────────────────────────────────────────
    print("[evaluate] Loading test split ...")
    test_ds = SpeechDataset(args.manifest, split="test")

    # --max-samples: take a stratified random subset for quick sanity checks
    if args.max_samples and args.max_samples < len(test_ds):
        # Keep class balance: sample equally from bonafide and spoof
        import random
        random.seed(42)
        bonafide_idx = [i for i in range(len(test_ds))
                        if test_ds.data.iloc[i]["label"] == "bonafide"]
        spoof_idx    = [i for i in range(len(test_ds))
                        if test_ds.data.iloc[i]["label"] == "spoof"]
        half = args.max_samples // 2
        chosen = (random.sample(bonafide_idx, min(half, len(bonafide_idx))) +
                  random.sample(spoof_idx,    min(half, len(spoof_idx))))
        random.shuffle(chosen)
        test_ds = Subset(test_ds, chosen)
        print(f"[evaluate] Using {len(test_ds):,} samples "
              f"(quick mode, balanced subset of test split)")
    else:
        print(f"[evaluate] Test samples : {len(test_ds):,}  "
              f"(full eval — use --max-samples N to go faster)")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"[evaluate] Loading checkpoint: {args.checkpoint}")
    model = WavLMClassifier().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        saved_epoch    = state.get("epoch", "?")
        saved_dev_loss = state.get("dev_loss", float("nan"))
        saved_dev_acc  = state.get("dev_acc",  float("nan"))
        print(f"[evaluate] Checkpoint     : epoch {saved_epoch}  "
              f"dev_loss={saved_dev_loss:.4f}  dev_acc={saved_dev_acc:.1%}")
    else:
        model.load_state_dict(state)

    # ── Inference ──────────────────────────────────────────────────────────────
    print("[evaluate] Running inference ...")
    labels, logits = run_inference(model, test_loader, device)

    # ── Accuracy (threshold at logit=0) ───────────────────────────────────────
    preds    = (logits > 0).astype(int)
    accuracy = (preds == labels).mean()

    # ── EER ───────────────────────────────────────────────────────────────────
    eer, eer_threshold = compute_eer(labels, logits)

    # ── Print results ──────────────────────────────────────────────────────────
    n_bonafide = (labels == 0).sum()
    n_spoof    = (labels == 1).sum()

    print("\n" + "═" * 50)
    print("  EVALUATION RESULTS")
    print("═" * 50)
    print(f"  Evaluated on  : {len(labels):,} samples  "
          f"(bonafide={n_bonafide:,}, spoof={n_spoof:,})")
    print(f"  Accuracy      : {accuracy:.1%}  (threshold at logit=0)")
    print(f"  EER           : {eer * 100:.2f}%  "
          f"(threshold={eer_threshold:.4f})")
    print("═" * 50)
    print("\n  EER is the primary anti-spoofing metric.")
    print("  Lower EER = better.  Random chance ≈ 50%.")
    if args.max_samples:
        print(f"\n  ⚠  This was a quick eval on {len(labels):,} samples.")
        print("     Run without --max-samples for the official full-set EER.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the WavLM deepfake speech detector"
    )
    parser.add_argument(
        "--manifest", default=MANIFEST_CSV,
        help="Path to manifest CSV  [default: %(default)s]"
    )
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH,
        help="Path to a .pt checkpoint  [default: %(default)s]"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Batch size for inference  [default: %(default)s]"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        metavar="N",
        help="Evaluate on N balanced samples instead of the full test set. "
             "e.g. --max-samples 5000 finishes in ~3 min and gives a "
             "reliable EER estimate."
    )

    main(parser.parse_args())
