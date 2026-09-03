import pandas as pd
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run():
    print("Merging new hard negative manifests...")
    unified_path = PROJECT_ROOT / "data_pipeline" / "manifests" / "unified_manifest_final.csv"
    synth_path = PROJECT_ROOT / "data_pipeline" / "manifests" / "synthetic_hn_manifest.csv"
    real_path = PROJECT_ROOT / "data_pipeline" / "manifests" / "real_hn_manifest.csv"
    
    df_unified = pd.read_csv(unified_path)
    
    try:
        df_synth = pd.read_csv(synth_path)
    except FileNotFoundError:
        df_synth = pd.DataFrame()
        
    try:
        df_real = pd.read_csv(real_path)
    except FileNotFoundError:
        df_real = pd.DataFrame()
        
    def split_data(df, eval_split_name):
        df = df.copy()
        if len(df) == 0:
            return df
            
        # Group by generator_id so each tool/manipulation gets 80/20 split
        generators = df["generator_id"].unique()
        for gen in generators:
            idx = df[df["generator_id"] == gen].index
            np.random.seed(42)
            eval_idx = np.random.choice(idx, size=int(len(idx) * 0.2), replace=False)
            train_idx = list(set(idx) - set(eval_idx))
            
            df.loc[train_idx, "split"] = "train"
            df.loc[train_idx, "split_hint"] = "train"
            
            df.loc[eval_idx, "split"] = "eval"
            df.loc[eval_idx, "split_hint"] = eval_split_name
            
        return df
        
    df_synth_split = split_data(df_synth, "hard_negative_eval_synthetic")
    df_real_split = split_data(df_real, "hard_negative_eval_real")
    
    combined = pd.concat([df_unified, df_synth_split, df_real_split], ignore_index=True)
    combined.to_csv(unified_path, index=False)
    
    print("\n--- NEW MANIFEST BALANCE ---")
    print(f"Total size: {len(combined)}")
    print(combined.groupby(["split", "split_hint"]).size())
    print("\nSynthetic Hard Negatives added:", len(df_synth))
    print("Real Hard Negatives added:", len(df_real))

if __name__ == "__main__":
    run()
