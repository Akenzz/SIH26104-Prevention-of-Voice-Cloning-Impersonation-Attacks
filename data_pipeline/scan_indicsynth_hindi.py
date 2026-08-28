"""
Scan the downloaded IndicSynth Hindi parquet shards to build the generator and
speaker inventory needed to plan a leakage-free v2.1 split.

Shards are organized (roughly) one generative-model-per-shard, so this maps:
  - generator -> clip count
  - generator -> which shards it lives in
  - target speaker -> clip count (this is the IndicSUPERB/Kathbath speaker id space)

Run AFTER:
  hf download vdivyasharma/IndicSynth --repo-type dataset --include "Hindi/*" \
      --local-dir E:\\DatasetSIH\\indicsynth

Usage:
  python scan_indicsynth_hindi.py --hindi-dir E:/DatasetSIH/indicsynth/Hindi
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

META_COLS = ["Generative Model", "Target Speaker ID", "Source Speaker_ID", "Gender"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindi-dir", required=True,
                    help="Folder holding the Hindi/*.parquet shards")
    ap.add_argument("--out", default="reports/indicsynth_hindi_inventory.json")
    args = ap.parse_args()

    hindi_dir = Path(args.hindi_dir)
    shards = sorted(hindi_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No .parquet shards found under {hindi_dir}")

    print(f"Scanning {len(shards)} shards under {hindi_dir}")

    gen_counts = Counter()
    spk_counts = Counter()
    gen_to_shards = defaultdict(set)
    gen_speakers = defaultdict(set)
    total_rows = 0

    for i, shard in enumerate(shards):
        t = pq.read_table(shard, columns=META_COLS)
        n = t.num_rows
        total_rows += n
        gens = t.column("Generative Model").to_pylist()
        spks = t.column("Target Speaker ID").to_pylist()
        for g, s in zip(gens, spks):
            gen_counts[g] += 1
            spk_counts[s] += 1
            gen_to_shards[g].add(shard.name)
            gen_speakers[g].add(s)
        if (i + 1) % 10 == 0 or (i + 1) == len(shards):
            print(f"  scanned {i+1}/{len(shards)} shards, {total_rows:,} rows so far",
                  flush=True)

    print(f"\nTotal clips: {total_rows:,}")
    print(f"Distinct target speakers: {len(spk_counts):,}")
    print("\n== Generative Model distribution ==")
    for g, c in gen_counts.most_common():
        print(f"  {g:<24} {c:>7,}  ({len(gen_speakers[g])} speakers, "
              f"{len(gen_to_shards[g])} shards)")

    inventory = {
        "total_rows": total_rows,
        "num_speakers": len(spk_counts),
        "generators": {
            g: {
                "clips": gen_counts[g],
                "speakers": sorted(int(x) for x in gen_speakers[g]),
                "shards": sorted(gen_to_shards[g]),
            }
            for g in gen_counts
        },
        "speaker_clip_counts": {str(k): v for k, v in spk_counts.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventory, indent=2))
    print(f"\nWrote inventory -> {out}")


if __name__ == "__main__":
    main()
