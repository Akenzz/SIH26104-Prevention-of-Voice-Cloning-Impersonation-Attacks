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

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import wavlm-base-plus modules using importlib because of the hyphens
spec_dataset = importlib.util.spec_from_file_location("dataset", str(PROJECT_ROOT / "wavlm-base-plus" / "dataset.py"))
dataset = importlib.util.module_from_spec(spec_dataset)
spec_dataset.loader.exec_module(dataset)

spec_model = importlib.util.spec_from_file_location("model", str(PROJECT_ROOT / "wavlm-base-plus" / "model.py"))
model_module = importlib.util.module_from_spec(spec_model)
spec_model.loader.exec_module(model_module)

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_labels = []
    all_logits = []
    use_amp = device.type == "cuda"
    pbar = tqdm(loader, desc="[evaluate] Inference", unit="batch", dynamic_ncols=True)
    for waveform, labels in pbar:
        waveform = waveform.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(waveform).squeeze(1)
        all_labels.append(labels.numpy())
        all_logits.append(logits.float().cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_logits)

def fit_platt():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = PROJECT_ROOT / "data_pipeline/manifests/unified_manifest_final.csv"
    checkpoint_path = PROJECT_ROOT / "wavlm-base-plus/checkpoints/best_model_unified.pt"
    
    print(f"Loading Dev Split from {manifest_path}")
    dev_ds = dataset.SpeechDataset(str(manifest_path), split="dev")
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Loading Checkpoint {checkpoint_path}")
    model = model_module.WavLMClassifier().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    
    print("Running Inference...")
    labels, logits = run_inference(model, dev_loader, device)
    
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
    joblib_path = out_dir / "platt_v2_combined_dataset.joblib"
    json_path = out_dir / "platt_v2_combined_dataset.json"
    
    joblib.dump(lr, joblib_path)
    
    metadata = {
        "version": "platt_v2_combined_dataset",
        "kind": "platt_sklearn",
        "metadata": {
            "checkpoint": str(checkpoint_path),
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
        mask = (sources == src) & (y == 0)
        if mask.sum() > 0:
            print(f"  {src:<20}: {probs[mask].mean():.1%} (n={mask.sum()})")
            
    print("\nBreakdown by Source (Spoof):")
    for src in unique_sources:
        mask = (sources == src) & (y == 1)
        if mask.sum() > 0:
            print(f"  {src:<20}: {probs[mask].mean():.1%} (n={mask.sum()})")

if __name__ == "__main__":
    fit_platt()
