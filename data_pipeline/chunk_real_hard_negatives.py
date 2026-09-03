import sys
import os
import uuid
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def preprocess_audio(y, sr):
    y, _ = librosa.effects.trim(y, top_db=30)
    if np.abs(y).max() > 0:
        y = y / np.abs(y).max() * 0.9
    return y

def get_speech_timestamps(y, sr=16000):
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        wav = torch.from_numpy(y).float()
        timestamps = get_speech_timestamps(wav, model, sampling_rate=sr)
        return timestamps
    except Exception as e:
        print(f"VAD error: {e}")
        # fallback: just split into 5s chunks
        return [{"start": i*sr*5, "end": (i+1)*sr*5} for i in range(0, len(y)//(sr*5))]

def run():
    spoof_dir = PROJECT_ROOT / "spoof"
    out_dir = PROJECT_ROOT / "data1" / "hard_negatives" / "real"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    files = [f for f in os.listdir(spoof_dir) if f.endswith(('.wav', '.mp3'))]
    
    print(f"Chunking {len(files)} real failing audio files with Silero VAD...")
    
    for filename in tqdm(files):
        if "bonafide" in filename.lower():
            continue # Skip bonafide files in this folder
            
        src_path = spoof_dir / filename
        try:
            y, sr = librosa.load(src_path, sr=16000, mono=True)
        except Exception:
            continue
            
        timestamps = get_speech_timestamps(y, sr)
        
        # Merge small chunks to target 4-10s
        target_len_min = 4.0 * sr
        target_len_max = 10.0 * sr
        
        current_chunk = []
        current_len = 0
        merged_chunks = []
        
        for t in timestamps:
            chunk_y = y[t['start']:t['end']]
            chunk_len = len(chunk_y)
            
            if current_len + chunk_len < target_len_max:
                current_chunk.append(chunk_y)
                current_len += chunk_len
            else:
                if current_len > target_len_min:
                    merged_chunks.append(np.concatenate(current_chunk))
                current_chunk = [chunk_y]
                current_len = chunk_len
                
        if current_len > target_len_min:
            merged_chunks.append(np.concatenate(current_chunk))
            
        generator_name = os.path.splitext(filename)[0]
            
        for chunk in merged_chunks:
            chunk_processed = preprocess_audio(chunk, sr)
            if len(chunk_processed) < sr * 1.0: # Skip sub 1s
                continue
                
            new_filename = f"{generator_name}_chunk_{uuid.uuid4().hex[:8]}.wav"
            out_path = out_dir / new_filename
            sf.write(str(out_path), chunk_processed, sr)
            
            records.append({
                "path": str(out_path.absolute()),
                "label": "spoof",
                "split": "train",
                "split_hint": "train",
                "source": "manual_collected_hard_negative",
                "generator_id": generator_name,
                "matched_source_clip_id": "none"
            })
            
    out_df = pd.DataFrame(records)
    out_df.to_csv(PROJECT_ROOT / "data_pipeline" / "manifests" / "real_hn_manifest.csv", index=False)
    print(f"Generated {len(out_df)} real hard negative chunks!")

if __name__ == "__main__":
    run()
