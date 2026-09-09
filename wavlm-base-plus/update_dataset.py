import os
import pandas as pd
from pathlib import Path
import random

print("Loading existing datasets...")
train_csv = '/media/akenzz/D1/DataSet_processed/v2_extended/train.csv'
val_csv = '/media/akenzz/D1/DataSet_processed/v2_extended/val.csv'

df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)

existing_paths = set(df_train['path']).union(set(df_val['path']))

print("Scanning for all MLAAD spoofs...")
mlaad_fake_dir = Path('/media/akenzz/D1/DataSet/MLAAD/MLAAD/fake')
all_spoofs = []
for root, _, files in os.walk(mlaad_fake_dir):
    for f in files:
        if f.endswith('.wav') or f.endswith('.flac'):
            all_spoofs.append(os.path.join(root, f))

print(f"Found {len(all_spoofs)} total MLAAD spoofs.")

new_rows = []
for path in all_spoofs:
    if path not in existing_paths:
        # path is like /media/akenzz/D1/DataSet/MLAAD/MLAAD/fake/{language}/{generator}/{filename}
        parts = Path(path).parts
        fake_idx = parts.index('fake')
        language = parts[fake_idx + 1]
        generator = parts[fake_idx + 2]
        
        new_rows.append({
            'path': path,
            'label': 'spoof',
            'generator': generator,
            'language': language,
            'split': 'train'
        })

print(f"Adding {len(new_rows)} new spoofs to the train set...")
df_new_spoofs = pd.DataFrame(new_rows)
df_train = pd.concat([df_train, df_new_spoofs], ignore_index=True)

# Balance bonafides in train set
spoof_count = len(df_train[df_train['label'] == 'spoof'])
bonafide_count = len(df_train[df_train['label'] == 'bonafide'])

print(f"Current train set: {spoof_count} spoofs, {bonafide_count} bonafides.")

if spoof_count > bonafide_count:
    print(f"Upsampling bonafides to {spoof_count}...")
    df_bonafides = df_train[df_train['label'] == 'bonafide']
    
    # We need spoof_count - bonafide_count MORE bonafides
    extra_needed = spoof_count - bonafide_count
    df_extra_bonafides = df_bonafides.sample(n=extra_needed, replace=True, random_state=42)
    
    df_train = pd.concat([df_train, df_extra_bonafides], ignore_index=True)

print(f"Final train set distribution:")
print(df_train['label'].value_counts())

# Save to v3_extended
out_dir = '/media/akenzz/D1/DataSet_processed/v3_extended'
os.makedirs(out_dir, exist_ok=True)

print("Saving to", out_dir)
df_train.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
df_val.to_csv(os.path.join(out_dir, 'val.csv'), index=False)
print("Done!")
