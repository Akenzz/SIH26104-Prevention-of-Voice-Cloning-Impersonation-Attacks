import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
manifest_dir = repo_root / "data_pipeline" / "manifests"

def merge_manifests():
    train_files = [
        manifest_dir / "asvspoof19_train.csv",
        manifest_dir / "kathbath_train.csv",
        manifest_dir / "mlaad_train.csv"
    ]
    
    dev_files = [
        manifest_dir / "asvspoof19_dev.csv"
        # We can use ASVspoof dev as the primary dev set for early stopping,
        # or we could carve out 10% of kathbath/mlaad if we wanted to.
        # For simplicity in V2, we'll just use ASVspoof dev + whatever else is explicitly 'dev'
    ]
    
    # 1. Build V2 Train
    train_dfs = []
    for f in train_files:
        if f.exists():
            df = pd.read_csv(f)
            train_dfs.append(df)
            print(f"Loaded {len(df):,} rows from {f.name}")
        else:
            print(f"[WARN] Missing train manifest: {f.name}")
            
    if train_dfs:
        v2_train = pd.concat(train_dfs, ignore_index=True)
        out_train = manifest_dir / "v2_train.csv"
        v2_train.to_csv(out_train, index=False)
        print(f"\n[OK] Built {out_train.name} with {len(v2_train):,} total rows.")
        
    # 2. Build V2 Dev
    dev_dfs = []
    for f in dev_files:
        if f.exists():
            df = pd.read_csv(f)
            dev_dfs.append(df)
            print(f"Loaded {len(df):,} rows from {f.name}")
            
    if dev_dfs:
        v2_dev = pd.concat(dev_dfs, ignore_index=True)
        out_dev = manifest_dir / "v2_dev.csv"
        v2_dev.to_csv(out_dev, index=False)
        print(f"[OK] Built {out_dev.name} with {len(v2_dev):,} total rows.")

if __name__ == "__main__":
    merge_manifests()
