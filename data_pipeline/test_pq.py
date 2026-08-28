import sys
import pandas as pd
from pathlib import Path

cache_dir = Path(r"E:\DatasetSIH\.hf_cache\hub\datasets--ai4bharat--Kathbath")
pq_files = list(cache_dir.rglob("*.parquet"))

if not pq_files:
    print("No parquet files found.")
    sys.exit(0)

pq_file = pq_files[0]
print(f"Reading {pq_file}")
df = pd.read_parquet(str(pq_file))

print(f"Columns: {list(df.columns)}")
print(f"Number of rows: {len(df)}")
print(f"First row:")
row = df.iloc[0]
for col in df.columns:
    val = row[col]
    val_type = type(val)
    if isinstance(val, (bytes, bytearray)):
        print(f"  {col}: <bytes, len={len(val)}>")
    elif isinstance(val, dict):
        print(f"  {col}: <dict, keys={list(val.keys())}>")
        for k, v in val.items():
            if isinstance(v, (bytes, bytearray)):
                print(f"    {k}: <bytes, len={len(v)}>")
            else:
                print(f"    {k}: {v}")
    else:
        print(f"  {col}: {val}")
