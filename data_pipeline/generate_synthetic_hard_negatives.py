import sys
import os
import uuid
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def preprocess_audio(y, sr):
    # Same as standard project preprocessing
    y, _ = librosa.effects.trim(y, top_db=30)
    # Loudness norm using ffmpeg-normalize logic
    # Simplified here to peak normalization to match standard pipeline if ffmpeg-normalize isn't directly callable on arrays
    if np.abs(y).max() > 0:
        y = y / np.abs(y).max() * 0.9
    return y

def run():
    out_dir = PROJECT_ROOT / "data1" / "hard_negatives" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = PROJECT_ROOT / "data_pipeline" / "manifests" / "unified_manifest_final.csv"
    df = pd.read_csv(manifest_path)
    
    # Get high quality bonafide clips (processed_v2 or kathbath)
    bonafides = df[(df["label"] == "bonafide") & (df["source"].isin(["processed_v2", "kathbath"]))]
    if len(bonafides) > 200:
        bonafides = bonafides.sample(200, random_state=42)
        
    shifts = [-2, -1, 1, 2] # semitones
    
    records = []
    
    print(f"Generating synthetic hard negatives from {len(bonafides)} source files...")
    
    for _, row in tqdm(bonafides.iterrows(), total=len(bonafides)):
        src_path = str(PROJECT_ROOT / row["path"] if not str(row["path"]).startswith("/") else row["path"])
        if not os.path.exists(src_path):
            continue
            
        try:
            y, sr = librosa.load(src_path, sr=16000, mono=True)
        except Exception:
            continue
            
        src_clip_id = os.path.basename(src_path)
        
        for shift in shifts:
            try:
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
                y_processed = preprocess_audio(y_shifted, sr)
                
                new_filename = f"synth_pitch_{shift}_{uuid.uuid4().hex[:8]}.wav"
                out_path = out_dir / new_filename
                
                sf.write(str(out_path), y_processed, sr)
                
                records.append({
                    "path": str(out_path.absolute()),
                    "label": "spoof",
                    "split": "train", # will split properly later
                    "split_hint": "train",
                    "source": "synthetic_hard_negative",
                    "generator_id": f"pitch_shift_semitone_{shift}",
                    "matched_source_clip_id": src_clip_id
                })
            except Exception as e:
                pass
                
    out_df = pd.DataFrame(records)
    out_df.to_csv(PROJECT_ROOT / "data_pipeline" / "manifests" / "synthetic_hn_manifest.csv", index=False)
    print(f"Generated {len(out_df)} synthetic hard negatives!")

if __name__ == "__main__":
    run()
