import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "wavlm-base-plus"))

from model import WavLMClassifier
from dataset import WINDOW_SAMPLES
import librosa

def compute_eer(labels, scores):
    if len(set(labels)) == 1:
        # If there's only one class (e.g. all spoofs), we can't compute EER.
        # Instead, we just return the False Negative Rate (or Error Rate).
        # Assuming threshold is 0 for the logit.
        preds = (np.array(scores) > 0).astype(int)
        error_rate = np.mean(preds != np.array(labels))
        return float(error_rate)
        
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

def score_df(model, df, device):
    labels = []
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
        try:
            y, sr = librosa.load(PROJECT_ROOT / row["path"], sr=16000, mono=True)
            if len(y) > WINDOW_SAMPLES:
                y = y[:WINDOW_SAMPLES]
            else:
                num_repeats = int(WINDOW_SAMPLES / len(y)) + 1
                y = np.tile(y, (1, num_repeats))[:, :WINDOW_SAMPLES][0]
            
            tensor = torch.from_numpy(y).float().to(device).unsqueeze(0)
            with torch.no_grad():
                logit = model(tensor).squeeze(1).item()
                
            labels.append(1 if row["label"] == "spoof" else 0)
            scores.append(logit)
        except Exception:
            continue
    return compute_eer(labels, scores) if len(labels) > 0 else 0.0

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_v3 = WavLMClassifier().to(device)
    state_v3 = torch.load(PROJECT_ROOT / "wavlm-base-plus/checkpoints/best_model_unified_epoch8_backup.pt", map_location=device)
    model_v3.load_state_dict(state_v3.get("model_state_dict", state_v3))
    model_v3.eval()
    
    model_v4 = WavLMClassifier().to(device)
    state_v4 = torch.load(PROJECT_ROOT / "wavlm-base-plus/checkpoints/best_model_v4.pt", map_location=device)
    model_v4.load_state_dict(state_v4.get("model_state_dict", state_v4))
    model_v4.eval()
    
    df = pd.read_csv(PROJECT_ROOT / "data_pipeline/manifests/unified_manifest_final.csv")
    
    slices = {
        "Eval: In-Domain": df[df["split_hint"] == "eval"],
        "Eval: Held-Out Generators": df[df["split_hint"] == "eval_ood"],
        "Eval: Internet Deepfakes": df[(df["split_hint"] == "eval") & (df["source"] == "release_in_the_wild")],
        "Hard Negative: Synthetic": df[df["split_hint"] == "hard_negative_eval_synthetic"],
        "Hard Negative: Real (Manual)": df[df["split_hint"] == "hard_negative_eval_real"]
    }
    
    print("\n==================== V3 vs V4 EER COMPARISON ====================")
    print(f"{'Slice Name':<30} | {'V3 (Epoch 8)':<12} | {'V4 (Epoch 12)':<12} | {'Delta':<10}")
    print("-" * 75)
    
    for name, slice_df in slices.items():
        if len(slice_df) == 0: continue
        
        # We only evaluate a sample if the slice is huge (e.g., eval_ood is 35k clips)
        if len(slice_df) > 2000:
            slice_df = slice_df.sample(2000, random_state=42)
            
        print(f"Scoring {name} (N={len(slice_df)})...")
        eer_v3 = score_df(model_v3, slice_df, device)
        eer_v4 = score_df(model_v4, slice_df, device)
        
        delta = eer_v4 - eer_v3
        flag = "🔴 REGRESSION" if delta > 0.01 else ("🟢 IMPROVED" if delta < -0.01 else "⚪ NEUTRAL")
        
        print(f"\r{name:<30} | {eer_v3:>11.2%} | {eer_v4:>11.2%} | {delta:>+9.2%} {flag}")
    
    print("\nDone.")

if __name__ == "__main__":
    run()
