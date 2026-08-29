"""
data_pipeline/recover_kathbath_transcripts.py
=============================================
Recover the Hindi *transcripts* that fetch_kathbath.py threw away.

fetch_kathbath.py wrote only the audio (E:\\DatasetSIH\\kathbath\\hindi\\<stem>.wav)
and never saved the `text` column from the Kathbath parquet shards. IndicF5 and
XTTS-v2 need that text to regenerate MATCHED-CONTENT spoofs (same speaker saying
the same sentence, cloned), so we read the cached shards and join them back to
the wav files by filename stem.

The parquet lives in the HF cache created by fetch_kathbath.py, e.g.:
  E:\\DatasetSIH\\.hf_cache\\hub\\datasets--ai4bharat--Kathbath\\snapshots\\<hash>\\hindi\\train-*-of-*.parquet

Each parquet row (non-audio columns):
  fname:      "844424930703439-329-f.m4a"   -> stem joins to <stem>.wav
  text:       "<Hindi transcript>"          (Devanagari)
  lang, duration, gender, speaker_id(int)

Output (UTF-8, Devanagari-safe):
  data_pipeline/manifests/kathbath_hindi_transcripts.csv
    columns: utt_id, text, speaker_id, gender, duration_s, language

Usage:
  python data_pipeline/recover_kathbath_transcripts.py
  python data_pipeline/recover_kathbath_transcripts.py --wav-dir E:/DatasetSIH/kathbath/hindi
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

repo_root = Path(__file__).resolve().parent.parent
MANIFEST_DIR = repo_root / "data_pipeline" / "manifests"

# Only the columns we need — projecting these SKIPS decoding the audio bytes,
# so reading all 32 shards is fast and low-memory.
WANT_COLS = ["fname", "text", "lang", "duration", "gender", "speaker_id"]

DEFAULT_KATHBATH_ROOT = Path(os.environ.get("KATHBATH_ROOT", r"E:\DatasetSIH\kathbath"))
DEFAULT_WAV_DIR = DEFAULT_KATHBATH_ROOT / "hindi"
# .hf_cache is created alongside the kathbath/ folder by fetch_kathbath.py
DEFAULT_HF_CACHE = DEFAULT_KATHBATH_ROOT.parent / ".hf_cache"


def find_parquet_shards(hf_cache: Path, explicit_dir: str | None) -> list[Path]:
    if explicit_dir:
        shards = sorted(Path(explicit_dir).glob("*.parquet"))
        if shards:
            return shards
        raise SystemExit(f"[FAIL] No .parquet shards under --parquet-dir {explicit_dir}")

    # snapshot hash is unknown/variable -> glob for it
    patterns = [
        hf_cache / "hub" / "datasets--ai4bharat--Kathbath" / "snapshots" / "*" / "hindi" / "*.parquet",
        hf_cache / "**" / "hindi" / "*.parquet",
    ]
    for pat in patterns:
        hits = sorted(Path(p) for p in glob.glob(str(pat), recursive=True))
        if hits:
            return hits
    raise SystemExit(
        f"[FAIL] Could not locate Kathbath Hindi parquet shards under {hf_cache}.\n"
        f"       Pass --parquet-dir pointing at the folder that holds train-*-of-*.parquet"
    )


def main():
    ap = argparse.ArgumentParser(description="Recover Kathbath Hindi transcripts from cached parquet shards.")
    ap.add_argument("--hf-cache", default=str(DEFAULT_HF_CACHE),
                    help="HF cache root created by fetch_kathbath.py")
    ap.add_argument("--parquet-dir", default=None,
                    help="Directly point at the folder holding the Hindi *.parquet shards (overrides --hf-cache glob)")
    ap.add_argument("--wav-dir", default=str(DEFAULT_WAV_DIR),
                    help="Folder of extracted <stem>.wav files, for a coverage cross-check")
    ap.add_argument("--out", default=str(MANIFEST_DIR / "kathbath_hindi_transcripts.csv"))
    args = ap.parse_args()

    shards = find_parquet_shards(Path(args.hf_cache), args.parquet_dir)
    print(f"[1/3] Found {len(shards)} parquet shard(s).")

    rows = []
    empty_text = 0
    for i, shard in enumerate(shards, 1):
        pf = pq.ParquetFile(shard)
        present = [c for c in WANT_COLS if c in pf.schema_arrow.names]
        table = pf.read(columns=present)
        cols = {c: table.column(c).to_pylist() for c in present}
        n = table.num_rows
        for j in range(n):
            fname = cols.get("fname", [None] * n)[j]
            text = cols.get("text", [None] * n)[j]
            if not fname:
                continue
            if text is None or str(text).strip() == "":
                empty_text += 1
            rows.append({
                "utt_id": Path(str(fname)).stem,             # "844424930703439-329-f"
                "text": ("" if text is None else str(text)),
                "speaker_id": cols.get("speaker_id", [None] * n)[j],
                "gender": cols.get("gender", [None] * n)[j],
                "duration_s": cols.get("duration", [None] * n)[j],
                "language": cols.get("lang", ["hi"] * n)[j],
            })
        print(f"      shard {i:>2}/{len(shards)}: +{n:,} rows (total {len(rows):,})", flush=True)

    df = pd.DataFrame(rows).drop_duplicates(subset=["utt_id"])
    print(f"[2/3] {len(df):,} unique utterances | {empty_text:,} rows had empty text.")

    # Coverage cross-check against the actual wav files (optional but cheap).
    wav_dir = Path(args.wav_dir)
    if wav_dir.is_dir():
        wav_stems = {Path(n).stem for n in os.listdir(wav_dir) if n.lower().endswith(".wav")}
        tx_stems = set(df["utt_id"])
        have_both = wav_stems & tx_stems
        wav_no_text = wav_stems - tx_stems
        print(f"      wav files: {len(wav_stems):,} | with transcript: {len(have_both):,} "
              f"| wav missing transcript: {len(wav_no_text):,}")
        if wav_no_text:
            ex = list(wav_no_text)[:3]
            print(f"      e.g. missing-transcript stems: {ex}")
    else:
        print(f"      [WARN] wav-dir not found ({wav_dir}); skipped coverage check.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel also renders Devanagari; pandas/downstream read it fine.
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[3/3] Wrote {args.out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
