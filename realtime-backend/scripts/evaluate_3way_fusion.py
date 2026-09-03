import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "realtime-backend"))

from config import Settings
from experts.loader import load_experts
import librosa

WINDOW_SAMPLES = 64000
CACHE_FILE = PROJECT_ROOT / "realtime-backend/artifacts/fusion_logits_cache.csv"

def compute_eer(labels, scores):
    if len(set(labels)) == 1:
        preds = (np.array(scores) > 0).astype(int)
        error_rate = np.mean(preds != np.array(labels))
        return float(error_rate)
        
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

def load_audio_window(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    if len(y) > WINDOW_SAMPLES:
        y = y[:WINDOW_SAMPLES]
    elif len(y) < WINDOW_SAMPLES:
        num_repeats = int(WINDOW_SAMPLES / len(y)) + 1
        y = np.tile(y, (1, num_repeats))[:, :WINDOW_SAMPLES][0]
    return y

def evaluate_testdata(lr_model):
    testdata_dir = PROJECT_ROOT / "testdata"
    if not testdata_dir.exists():
        print("testdata/ folder not found. Skipping smoke test.")
        return
        
    print("\n==================== 16-FILE SMOKE TEST ====================")
    settings = Settings(experts=["wavlm", "hybrid", "ssl"], device="cuda" if torch.cuda.is_available() else "cpu")
    experts = load_experts(settings)
    
    files = list(testdata_dir.glob("*.*"))
    correct_lr = 0
    correct_fixed = 0
    total = len(files)
    
    for f in files:
        label = 0 if "bonafi" in f.name.lower() else 1
        
        y, sr = librosa.load(f, sr=16000, mono=True)
        # Process in 4s windows
        windows = []
        for i in range(0, len(y), WINDOW_SAMPLES):
            window = y[i:i+WINDOW_SAMPLES]
            if len(window) < WINDOW_SAMPLES:
                window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))
            windows.append(window)
            
        avg_wavlm, avg_lfcc, avg_ssl = [], [], []
        for w in windows:
            try:
                avg_wavlm.append(experts["wavlm"].score(w)["logit"])
                avg_lfcc.append(experts["hybrid"].score(w)["logit"])
                avg_ssl.append(experts["ssl"].score(w)["logit"])
            except Exception:
                pass
                
        if not avg_wavlm: continue
        
        w_logit = np.mean(avg_wavlm)
        l_logit = np.mean(avg_lfcc)
        s_logit = np.mean(avg_ssl)
        
        # Predictions
        lr_prob = lr_model.predict_proba([[w_logit, l_logit, s_logit]])[0, 1]
        lr_pred = 1 if lr_prob > 0.5 else 0
        fixed_pred = 1 if (l_logit + s_logit > 0) else 0
        
        if lr_pred == label: correct_lr += 1
        if fixed_pred == label: correct_fixed += 1
        
        actual_str = "SPOOF" if label == 1 else "BONA "
        lr_str = "SPOOF" if lr_pred == 1 else "BONA "
        fix_str = "SPOOF" if fixed_pred == 1 else "BONA "
        
        mark_lr = "🟢" if lr_pred == label else "🔴"
        mark_fix = "🟢" if fixed_pred == label else "🔴"
        
        print(f"[{actual_str}] {f.name[:25]:<25} | LR Fusion: {lr_str} {mark_lr} | Fixed: {fix_str} {mark_fix}")
        
    print("-" * 75)
    print(f"LR Fusion Accuracy  : {correct_lr}/{total} ({correct_lr/total:.1%})")
    print(f"Fixed Rule Accuracy : {correct_fixed}/{total} ({correct_fixed/total:.1%})")


def run():
    print("Preparing Datasets...")
    df = pd.read_csv(PROJECT_ROOT / "data_pipeline/manifests/unified_manifest_final.csv")
    
    slices = {
        "Dev Set (Fit Data)": df[df["split_hint"] == "dev"],
        "Eval: In-Domain": df[df["split_hint"] == "eval"],
        "Eval: Held-Out Gen": df[df["split_hint"] == "eval_ood"],
        "Eval: Internet OOD": df[(df["split_hint"] == "eval") & (df["source"] == "release_in_the_wild")],
        "Hard Negative: Synth": df[df["split_hint"] == "hard_negative_eval_synthetic"],
        "Hard Negative: Real": df[df["split_hint"] == "hard_negative_eval_real"]
    }
    
    worklist_dfs = []
    for name, slice_df in slices.items():
        if len(slice_df) > 2000 and name != "Dev Set (Fit Data)":
            # Cap eval slices at 2000 to prevent 8 hour runs, Dev must remain full size for LR fit
            slice_df = slice_df.sample(2000, random_state=42)
        slice_df = slice_df.copy()
        slice_df["slice_name"] = name
        worklist_dfs.append(slice_df)
        
    worklist = pd.concat(worklist_dfs)
    print(f"Total items to score: {len(worklist)} (Dev: {len(slices['Dev Set (Fit Data)'])}, Eval sampled cap: 2000)")
    
    # Load cache
    if CACHE_FILE.exists():
        cache_df = pd.read_csv(CACHE_FILE)
        print(f"Loaded {len(cache_df)} cached logits from {CACHE_FILE.name}")
    else:
        cache_df = pd.DataFrame(columns=["path", "label", "slice_name", "wavlm_logit", "lfcc_logit", "ssl_logit"])
        
    missing = worklist[~worklist["path"].isin(cache_df["path"])]
    if len(missing) > 0:
        print(f"Need to score {len(missing)} missing items...")
        settings = Settings(experts=["wavlm", "hybrid", "ssl"], device="cuda" if torch.cuda.is_available() else "cpu")
        experts = load_experts(settings)
        
        new_records = []
        for _, row in tqdm(missing.iterrows(), total=len(missing), desc="Scoring Missing Items"):
            try:
                y = load_audio_window(PROJECT_ROOT / row["path"])
                w_logit = experts["wavlm"].score(y)["logit"]
                l_logit = experts["hybrid"].score(y)["logit"]
                s_logit = experts["ssl"].score(y)["logit"]
                
                new_records.append({
                    "path": row["path"],
                    "label": 1 if row["label"] == "spoof" else 0,
                    "slice_name": row["slice_name"],
                    "wavlm_logit": w_logit,
                    "lfcc_logit": l_logit,
                    "ssl_logit": s_logit
                })
            except Exception:
                continue
                
            # Save incrementally every 100 to prevent data loss
            if len(new_records) % 100 == 0:
                pd.DataFrame(new_records).to_csv(CACHE_FILE, mode='a', header=not CACHE_FILE.exists(), index=False)
                new_records = []
                
        if new_records:
            pd.DataFrame(new_records).to_csv(CACHE_FILE, mode='a', header=not CACHE_FILE.exists(), index=False)
            
        cache_df = pd.read_csv(CACHE_FILE)
    
    # --- STEP 1: FIT LR ON DEV SET ONLY ---
    dev_data = cache_df[cache_df["slice_name"] == "Dev Set (Fit Data)"]
    print(f"\n[STEP 1] Fitting Logistic Regression on strictly Dev Set (N={len(dev_data)})")
    
    X_dev = dev_data[["wavlm_logit", "lfcc_logit", "ssl_logit"]].values
    y_dev = dev_data["label"].values
    
    lr = LogisticRegression(class_weight="balanced", random_state=42)
    lr.fit(X_dev, y_dev)
    
    # Save the fusion model just in case it's the winner
    joblib.dump(lr, PROJECT_ROOT / "realtime-backend/artifacts/fusion_lr.joblib")
    
    # --- STEP 2: EVALUATE ON HELD-OUT SLICES ---
    print("\n========================= 3-WAY FUSION EVALUATION =========================")
    print(f"{'Slice Name':<22} | {'WavLM':<7} | {'LFCC':<7} | {'SSL':<7} | {'LR Fusion':<9} | {'LFCC+SSL':<9}")
    print("-" * 75)
    
    results = {}
    
    for slice_name in slices.keys():
        if slice_name == "Dev Set (Fit Data)": continue
        
        slice_data = cache_df[cache_df["slice_name"] == slice_name]
        if len(slice_data) == 0: continue
        
        X = slice_data[["wavlm_logit", "lfcc_logit", "ssl_logit"]].values
        y = slice_data["label"].values
        
        # Predictions
        w_eer = compute_eer(y, X[:, 0])
        l_eer = compute_eer(y, X[:, 1])
        s_eer = compute_eer(y, X[:, 2])
        
        lr_probs = lr.predict_proba(X)[:, 1]
        lr_eer = compute_eer(y, lr_probs)
        
        # Fixed Rule: lfcc_logit + ssl_logit > 0
        fixed_logits = X[:, 1] + X[:, 2]
        fixed_eer = compute_eer(y, fixed_logits)
        
        results[slice_name] = {
            "wavlm": w_eer, "lfcc": l_eer, "ssl": s_eer,
            "lr": lr_eer, "fixed": fixed_eer
        }
        
        print(f"{slice_name:<22} | {w_eer:>6.2%} | {l_eer:>6.2%} | {s_eer:>6.2%} | {lr_eer:>8.2%} | {fixed_eer:>8.2%}")
        
    print("-" * 75)
    print("NOTE: Single-class slices (Hard Negatives) report Error Rate (FNR) instead of EER.")
    
    # --- STEP 3: SMOKE TEST ---
    evaluate_testdata(lr)
    
if __name__ == "__main__":
    run()
