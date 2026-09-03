import sys
import os
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import joblib
from tqdm import tqdm
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "realtime-backend"))

# Import dataset
spec_dataset = importlib.util.spec_from_file_location("dataset", str(PROJECT_ROOT / "wavlm-base-plus" / "dataset.py"))
dataset = importlib.util.module_from_spec(spec_dataset)
spec_dataset.loader.exec_module(dataset)

from experts.ssl_antispoof import SSLExpert

def run_inference(expert, loader):
    all_labels = []
    all_logits = []
    
    pbar = tqdm(loader, desc="[evaluate] Inference", unit="batch", dynamic_ncols=True)
    # Loader batch size must be 1 for SSLExpert adapter unless we modify it, but let's just do batch_size=1
    for waveform, labels in pbar:
        # waveform is (B, T). SSLExpert expects a 1D numpy array.
        for i in range(len(labels)):
            window_np = waveform[i].numpy()
            label = labels[i].item()
            
            try:
                score = expert.score(window_np)
                all_logits.append(score["logit"])
                all_labels.append(label)
            except Exception as e:
                pass
                
    return np.array(all_labels), np.array(all_logits)

def fit_platt():
    manifest_path = PROJECT_ROOT / "data_pipeline/manifests/unified_manifest_final.csv"
    
    print(f"Loading Dev Split from {manifest_path}")
    dev_ds = dataset.SpeechDataset(str(manifest_path), split="dev")
    # Using batch size 8 just to read efficiently, but we iterate individually inside run_inference
    dev_loader = DataLoader(dev_ds, batch_size=8, shuffle=False, num_workers=0)
    
    print("Loading SSL Expert...")
    expert = SSLExpert()
    
    print("Running Inference...")
    labels, logits = run_inference(expert, dev_loader)
    
    # We need to map labels back to their sources for the sanity check
    # Since we skipped exceptions, we'll just match sizes or assume none were skipped
    # Actually, SSLExpert shouldn't throw exceptions on valid shape (64000)
    sources = dev_ds.data["source"].values
    
    # Fit Platt Scaling
    print("Fitting Logistic Regression (Platt Scaling)...")
    lr = LogisticRegression(class_weight="balanced", random_state=42)
    X = logits.reshape(-1, 1)
    y = labels
    lr.fit(X, y)
    
    # Save the model
    out_dir = PROJECT_ROOT / "realtime-backend/artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = out_dir / "platt_ssl.joblib"
    json_path = out_dir / "platt_ssl.json"
    
    joblib.dump(lr, joblib_path)
    
    metadata = {
        "version": "platt_ssl_v1",
        "kind": "platt_sklearn",
        "metadata": {
            "model": "TakHemlata_SSL",
            "manifest": str(manifest_path),
            "split": "dev",
            "dev_size": len(labels),
            "n_bonafide": int((labels == 0).sum()),
            "n_spoof": int((labels == 1).sum()),
            "fitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved {joblib_path} and {json_path}")
    
    # Sanity Check
    print("\n--- SANITY CHECK ---")
    probs = lr.predict_proba(X)[:, 1]
    
    mean_bona = probs[y == 0].mean() if (y == 0).sum() > 0 else float("nan")
    mean_spoof = probs[y == 1].mean() if (y == 1).sum() > 0 else float("nan")
    
    print(f"Global Mean Bonafide Probability: {mean_bona:.1%}")
    print(f"Global Mean Spoof Probability   : {mean_spoof:.1%}")
    
    if mean_bona > 0.3 or mean_spoof < 0.7:
        print("WARNING: The calibrated probabilities are not well separated globally!")
        
    print("\nBreakdown by Source (Bonafide):")
    unique_sources = np.unique(sources)
    for src in unique_sources:
        mask = (sources[:len(y)] == src) & (y == 0)
        if mask.sum() > 0:
            print(f"  {src:<20}: {probs[mask].mean():.1%} (n={mask.sum()})")
            
    print("\nBreakdown by Source (Spoof):")
    for src in unique_sources:
        mask = (sources[:len(y)] == src) & (y == 1)
        if mask.sum() > 0:
            print(f"  {src:<20}: {probs[mask].mean():.1%} (n={mask.sum()})")

if __name__ == "__main__":
    fit_platt()
