import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "realtime-backend"))

from config import Settings
import librosa

WINDOW_SAMPLES = 64000
CACHE_FILE = PROJECT_ROOT / "realtime-backend/artifacts/fusion_logits_cache.csv"
TESTDATA_CACHE = PROJECT_ROOT / "realtime-backend/artifacts/testdata_logits_cache.csv"

def compute_eer(labels, scores):
    if len(set(labels)) == 1:
        preds = (np.array(scores) > 0).astype(int)
        error_rate = np.mean(preds != np.array(labels))
        return float(error_rate)
        
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

def compute_eer_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return thresholds[eer_idx]

def compute_high_precision_threshold(labels, scores, target=0.98):
    precisions, recalls, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    for p, r, t in zip(precisions, recalls, thresholds):
        if p >= target and r > 0.05:
            return t
    # Fallback to EER threshold if target precision is unreachable
    return compute_eer_threshold(labels, scores)

def process_testdata():
    from experts.loader import load_experts
    
    testdata_dir = PROJECT_ROOT / "testdata"
    files = sorted(list(testdata_dir.glob("*.*")))
    if not files:
        print("No files found in testdata/!")
        return pd.DataFrame()
        
    if TESTDATA_CACHE.exists():
        return pd.read_csv(TESTDATA_CACHE)
        
    print(f"Running inference on {len(files)} testdata files. This may take a few minutes for large files...")
    settings = Settings(experts=["wavlm", "hybrid", "ssl"], device="cuda" if torch.cuda.is_available() else "cpu")
    experts = load_experts(settings)
    
    records = []
    for f in tqdm(files, desc="Scoring Testdata"):
        label = 0 if "bonafi" in f.name.lower() else 1
        y, sr = librosa.load(f, sr=16000, mono=True)
        
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
        
        records.append({
            "path": f.name,
            "label": label,
            "wavlm_logit": np.mean(avg_wavlm),
            "lfcc_logit": np.mean(avg_lfcc),
            "ssl_logit": np.mean(avg_ssl)
        })
        
    df = pd.DataFrame(records)
    df.to_csv(TESTDATA_CACHE, index=False)
    return df

def run():
    if not CACHE_FILE.exists():
        print("ERROR: fusion_logits_cache.csv not found. Run evaluate_3way_fusion.py first.")
        return
        
    cache_df = pd.read_csv(CACHE_FILE)
    dev_data = cache_df[cache_df["slice_name"] == "Dev Set (Fit Data)"]
    
    # 1. Train LR model
    X_dev = dev_data[["wavlm_logit", "lfcc_logit", "ssl_logit"]].values
    y_dev = dev_data["label"].values
    lr = LogisticRegression(class_weight="balanced", random_state=42)
    lr.fit(X_dev, y_dev)
    
    # 2. Get thresholds
    T_w_eer = compute_eer_threshold(y_dev, X_dev[:, 0])
    T_w_high = compute_high_precision_threshold(y_dev, X_dev[:, 0], 0.98)
    T_l_high = compute_high_precision_threshold(y_dev, X_dev[:, 1], 0.98)
    T_s_high = compute_high_precision_threshold(y_dev, X_dev[:, 2], 0.98)
    
    print("\n==================== STEP 1: COMPLEMENTARITY ANALYSIS ====================")
    slices_to_eval = [s for s in cache_df["slice_name"].unique() if s != "Dev Set (Fit Data)"]
    
    for slice_name in slices_to_eval:
        slice_data = cache_df[cache_df["slice_name"] == slice_name]
        
        fixed_preds = (slice_data["lfcc_logit"] + slice_data["ssl_logit"] > 0).astype(int)
        wavlm_preds = (slice_data["wavlm_logit"] > T_w_eer).astype(int)
        labels = slice_data["label"].values
        
        fixed_wrong = (fixed_preds != labels)
        wavlm_wrong = (wavlm_preds != labels)
        
        both_wrong_count = (fixed_wrong & wavlm_wrong).sum()
        
        # A. LFCC+SSL Wrong, WavLM Right
        lfcc_wrong_wavlm_right = fixed_wrong & (~wavlm_wrong)
        count_A = lfcc_wrong_wavlm_right.sum()
        pct_A = count_A / fixed_wrong.sum() if fixed_wrong.sum() > 0 else 0
        
        # B. WavLM Wrong, LFCC+SSL Right
        wavlm_wrong_lfcc_right = wavlm_wrong & (~fixed_wrong)
        count_B = wavlm_wrong_lfcc_right.sum()
        
        print(f"\n--- {slice_name} ---")
        print(f"LFCC+SSL made {fixed_wrong.sum()} total errors.")
        print(f"Of those, WavLM independently got {count_A} right ({pct_A:.1%}).")
        
        if count_A > 0:
            examples = slice_data[lfcc_wrong_wavlm_right]["path"].head(3).tolist()
            print("  Examples where WavLM saved the day:")
            for ex in examples: print(f"    - {ex}")
            
        print(f"Conversely, WavLM made {wavlm_wrong.sum()} total errors.")
        print(f"Of those, LFCC+SSL independently got {count_B} right.")
        print(f"Cases where ALL THREE models failed: {both_wrong_count}")
        
    print("\n==================== STEP 2: ESCALATION/OR-RULE EVALUATION ====================")
    print(f"High-Precision thresholds derived from Dev Set (Target ~98% precision):")
    print(f"  WavLM : > {T_w_high:.3f}")
    print(f"  LFCC  : > {T_l_high:.3f}")
    print(f"  SSL   : > {T_s_high:.3f}")
    
    
    print("\n==================== STEP 3: REAL-WORLD CALIBRATION & TUNING ====================")
    test_df = process_testdata()
    if len(test_df) == 0: return
    
    # Split 50/50 randomly but deterministically
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    mid = len(test_df) // 2
    calib_df = test_df.iloc[:mid]
    final_df = test_df.iloc[mid:]
    
    print(f"Real-World Split: {len(calib_df)} Calibration files, {len(final_df)} Final Test files.")
    
    # Evaluate on Calibration
    c_labels = calib_df["label"].values
    c_fixed = calib_df["lfcc_logit"] + calib_df["ssl_logit"]
    
    # Find optimal threshold for LFCC+SSL on calibration set to minimize errors
    best_t, best_err = 0, 1.0
    # Sweep threshold from -5 to 5
    for t in np.linspace(-5, 5, 100):
        preds = (c_fixed > t).astype(int)
        err = np.mean(preds != c_labels)
        if err < best_err:
            best_err = err
            best_t = t
            
    print(f"Tuning `LFCC + SSL > T`: Original T=0.0 error rate on Calibration: {np.mean((c_fixed > 0).astype(int) != c_labels):.1%}")
    print(f"Tuning `LFCC + SSL > T`: Optimal T={best_t:.2f} error rate on Calibration: {best_err:.1%}")
    
    
    print("\n==================== STEP 4: FINAL MASSIVE REPORT ====================")
    print(f"{'Slice Name':<25} | {'LR Fusion':<9} | {'LFCC+SSL':<9} | {'Tuned LFCC+SSL':<14} | {'Escalation OR-Rule':<18}")
    print("-" * 88)
    
    def score_row(name, df, is_single_class=False):
        if len(df) == 0: return
        labels = df["label"].values
        X = df[["wavlm_logit", "lfcc_logit", "ssl_logit"]].values
        
        lr_probs = lr.predict_proba(X)[:, 1]
        fixed_logits = X[:, 1] + X[:, 2]
        
        # Escalation Rule
        escalation_preds = ((X[:, 0] > T_w_high) | (X[:, 1] > T_l_high) | (X[:, 2] > T_s_high)).astype(int)
        
        if is_single_class:
            lr_val = np.mean((lr_probs > 0.5).astype(int) != labels)
            fix_val = np.mean((fixed_logits > 0).astype(int) != labels)
            tune_val = np.mean((fixed_logits > best_t).astype(int) != labels)
            esc_val = np.mean(escalation_preds != labels)
        else:
            lr_val = compute_eer(labels, lr_probs)
            # EER is threshold independent, so Fixed and Tuned Fixed have the same EER!
            fix_val = compute_eer(labels, fixed_logits)
            tune_val = fix_val 
            esc_val = np.mean(escalation_preds != labels) # It's a hard rule, so just error rate
            
        print(f"{name:<25} | {lr_val:>8.2%} | {fix_val:>8.2%} | {tune_val:>12.2%} | {esc_val:>16.2%}")
        
    for slice_name in slices_to_eval:
        slice_data = cache_df[cache_df["slice_name"] == slice_name]
        is_single = "Hard Negative" in slice_name
        score_row(slice_name, slice_data, is_single)
        
    print("-" * 88)
    score_row("Realworld Calibration", calib_df, is_single_class=False)
    score_row("Realworld Final Test", final_df, is_single_class=False)
    print("-" * 88)
    
if __name__ == "__main__":
    run()
