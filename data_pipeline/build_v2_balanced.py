import os
from pathlib import Path
import pandas as pd
import random

# Fix random seed for reproducibility
random.seed(42)

repo_root = Path(__file__).resolve().parent.parent
manifest_dir = repo_root / "data_pipeline" / "manifests"
processed_dir = repo_root / "data1" / "processed"

# Required columns based on the schema
COLUMNS = [
    'path', 'label', 'split', 'source_dataset', 'speaker_id', 'utterance_id',
    'generator_id', 'language', 'codec', 'duration_s', 'license', 'consent'
]

def fix_windows_paths(df):
    def repath(p):
        p = p.replace("\\", "/")
        win_root = "D:/DatasetSIH/LA"
        idx = p.lower().find(win_root.lower())
        if idx != -1:
            relative = p[idx + len(win_root):].lstrip("/")
            return "/home/akenzz/sih/project/data1/LA/" + relative
        return p
    df['path'] = df['path'].apply(repath)
    return df

def scan_processed_split(split_name):
    """Scan data1/processed/{split_name} and build a dataframe."""
    split_dir = processed_dir / split_name
    if not split_dir.exists():
        print(f"[WARN] Processed split '{split_name}' not found at {split_dir}")
        return pd.DataFrame(columns=COLUMNS)
        
    rows = []
    # Known folders in processed
    for category in split_dir.iterdir():
        if not category.is_dir():
            continue
            
        generator = category.name
        label = "bonafide" if generator == "bonafide" else "spoof"
        
        for audio_file in category.glob("*.*"):
            if audio_file.suffix.lower() not in [".wav", ".flac", ".mp3", ".ogg"]:
                continue
                
            rows.append({
                'path': str(audio_file),
                'label': label,
                'split': split_name,
                'source_dataset': 'processed',
                'speaker_id': 'unknown',
                'utterance_id': audio_file.stem,
                'generator_id': generator,
                'language': 'unknown',
                'codec': 'unknown',
                'duration_s': 4.0,  # default
                'license': 'unknown',
                'consent': 'yes'
            })
            
    df = pd.DataFrame(rows, columns=COLUMNS)
    print(f"Scanned {len(df)} files from processed/{split_name}")
    return df

def balance_and_merge():
    print("--- Building Balanced V2 Dataset ---")
    
    # 1. Process TRAIN
    asv_train_path = manifest_dir / "asvspoof19_train.csv"
    if not asv_train_path.exists():
        print(f"[FAIL] Missing {asv_train_path}")
        return
        
    asv_train = pd.read_csv(asv_train_path)
    asv_train = fix_windows_paths(asv_train)
    print(f"Loaded ASVspoof19 train: {len(asv_train)} rows")
    
    bonafide = asv_train[asv_train['label'] == 'bonafide']
    spoofed = asv_train[asv_train['label'] == 'spoof']
    
    print(f"  -> Bonafide count: {len(bonafide)}")
    print(f"  -> Spoofed count (original): {len(spoofed)}")
    
    # Balance
    if len(spoofed) > len(bonafide):
        spoofed = spoofed.sample(n=len(bonafide), random_state=42)
    print(f"  -> Spoofed count (balanced): {len(spoofed)}")
    
    balanced_asv_train = pd.concat([bonafide, spoofed], ignore_index=True)
    
    # Load processed train
    processed_train = scan_processed_split("train")
    
    # Combine
    v2_train = pd.concat([balanced_asv_train, processed_train], ignore_index=True)
    
    out_train = manifest_dir / "v2_balanced_train.csv"
    v2_train.to_csv(out_train, index=False)
    print(f"\n[OK] Saved V2 Train to {out_train.name} (Total rows: {len(v2_train)})")
    
    
    # 2. Process DEV (validation)
    asv_dev_path = manifest_dir / "asvspoof19_dev.csv"
    if asv_dev_path.exists():
        asv_dev = pd.read_csv(asv_dev_path)
        asv_dev = fix_windows_paths(asv_dev)
        print(f"\nLoaded ASVspoof19 dev: {len(asv_dev)} rows")
        
        processed_dev = scan_processed_split("dev")
        v2_dev = pd.concat([asv_dev, processed_dev], ignore_index=True)
        
        out_dev = manifest_dir / "v2_balanced_dev.csv"
        v2_dev.to_csv(out_dev, index=False)
        print(f"[OK] Saved V2 Dev to {out_dev.name} (Total rows: {len(v2_dev)})")
    else:
        print(f"[WARN] Missing {asv_dev_path}")

    # 3. Process EVAL (test)
    asv_eval_path = manifest_dir / "asvspoof19_eval.csv"
    if asv_eval_path.exists():
        asv_eval = pd.read_csv(asv_eval_path)
        asv_eval = fix_windows_paths(asv_eval)
        print(f"\nLoaded ASVspoof19 eval: {len(asv_eval)} rows")
        
        processed_eval = scan_processed_split("eval")
        v2_eval = pd.concat([asv_eval, processed_eval], ignore_index=True)
        
        # evaluation script expects split="test", but it handles "eval" by default in dataset.py
        # however, it's safer to ensure they match if needed. The dataset.py replaces 'eval' with 'test'.
        out_eval = manifest_dir / "v2_balanced_eval.csv"
        v2_eval.to_csv(out_eval, index=False)
        print(f"[OK] Saved V2 Eval to {out_eval.name} (Total rows: {len(v2_eval)})")
    else:
        print(f"[WARN] Missing {asv_eval_path}")

if __name__ == '__main__':
    balance_and_merge()
