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

# TODO: Change MANIFEST_CSV below (or pass --manifest) to use your real dataset.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

from expert1.dataset import SpeechDataset, WINDOW_SAMPLES
from expert1.model import WavLMClassifier

# ─── Defaults ─────────────────────────────────────────────────────────────────
# TODO: Change MANIFEST_CSV to point at your real manifest.
# Paths are relative to the project root (where you run `python -m expert1.evaluate`).
MANIFEST_CSV    = "expert1/data/manifest.csv"
CHECKPOINT_PATH = "expert1/checkpoints/best_model.pt"
BATCH_SIZE      = 8
NUM_WORKERS     = 0


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
    Compute Equal Error Rate (EER) from binary labels and raw scores.

    The EER is the threshold at which the False Positive Rate (FAR / false
    accept rate) equals the False Negative Rate (FRR / false reject rate).

    Args:
        labels: Ground-truth binary labels (0 = bonafide, 1 = spoof).
        scores: Continuous scores (higher = more likely spoof).

    Returns:
        (eer, threshold)
    """
    # sklearn's roc_curve treats the *positive* class (spoof=1) as the signal.
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr   # False Negative Rate = 1 - True Positive Rate

    # Find the index where |FPR - FNR| is minimised
    eer_idx = np.argmin(np.abs(fpr - fnr))

    # Interpolate for a slightly smoother estimate
    eer       = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    threshold = float(thresholds[eer_idx])

    return eer, threshold


@torch.no_grad()
def run_inference(model, loader, device):
    """
    Forward-pass the entire loader. Returns (all_labels, all_logits) as
    1-D numpy arrays.
    """
    model.eval()
    all_labels = []
    all_logits = []

    for waveform, labels in loader:
        waveform = waveform.to(device)
        logits   = model(waveform).squeeze(1)  # (B,)

        all_labels.append(labels.numpy())
        all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_logits)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Using device: {device}")

    # ── Load test data ─────────────────────────────────────────────────────────
    print("[evaluate] Loading test split ...")
    test_ds = SpeechDataset(args.manifest, split="test")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS,
    )
    print(f"[evaluate] Test samples: {len(test_ds)}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"[evaluate] Loading checkpoint: {args.checkpoint}")
    model = WavLMClassifier().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        saved_epoch    = state.get("epoch", "?")
        saved_dev_loss = state.get("dev_loss", float("nan"))
        print(f"[evaluate] Checkpoint from epoch {saved_epoch}, "
              f"dev_loss={saved_dev_loss:.4f}")
    else:
        model.load_state_dict(state)

    # ── Inference ──────────────────────────────────────────────────────────────
    print("[evaluate] Running inference on test split ...")
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
    print(f"  Test samples  : {len(labels)}  "
          f"(bonafide={n_bonafide}, spoof={n_spoof})")
    print(f"  Accuracy      : {accuracy:.1%}  (threshold at logit=0)")
    print(f"  EER           : {eer * 100:.2f}%  "
          f"(threshold={eer_threshold:.4f})")
    print("═" * 50)
    print("\n  NOTE: EER is the primary anti-spoofing metric.")
    print("        Lower EER = better detector.")
    print("        Random chance EER ≈ 50 %.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the WavLM deepfake speech detector on the test split"
    )
    parser.add_argument(
        "--manifest", default=MANIFEST_CSV,
        help="Path to manifest CSV  [default: %(default)s]"
        # TODO: Pass --manifest /path/to/real_manifest.csv to use your real data
    )
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH,
        help="Path to a .pt checkpoint  [default: %(default)s]"
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    main(parser.parse_args())
