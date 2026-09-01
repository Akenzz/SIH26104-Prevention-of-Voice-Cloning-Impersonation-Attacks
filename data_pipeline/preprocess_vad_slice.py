"""
data_pipeline/preprocess_vad_slice.py
=====================================
Silero-VAD 4-10s chunking pass, applied IDENTICALLY to bonafide and spoof.

For every manifest row:
  load mono 16k -> trim -> loudness-normalize (SAME parity as preprocess_parity)
  -> Silero VAD speech regions -> greedily pack speech into MIN_SEC..MAX_SEC
  chunks -> write each chunk as its own wav -> one manifest row per chunk.

Why: the LFCC-LCNN trains on 4s windows (config.window_sec). Long GV/ITW clips
otherwise contribute a single random 4s crop per epoch; slicing them into
several 4-10s speech chunks yields many distinct windows and strips leading/inter
silence (a known label shortcut). Chunks INHERIT the source row's split/label/
speaker/generator/group, so the speaker-disjoint splits are preserved with zero
chunk leakage across train/dev/eval.

Deterministic. Chunks per clip are capped (MAX_CHUNKS) so one long file can't
dominate the balance.

Outputs the combined chunk manifest AND per-split siblings (…_train.csv etc.)
so training can point --train-manifest / --dev-manifest straight at them.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

import sys
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
from data_pipeline.preprocess_parity import load_mono_16k, trim_silence, normalize, TARGET_SR

MIN_SEC = 4.0
MAX_SEC = 10.0
MAX_CHUNKS = 6          # per source clip, to keep the pools balanced
MIN_KEEP_SEC = 1.0      # emit a short single chunk rather than drop the clip


def _load_vad():
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def pack_chunks(regions, sr, total_len):
    """Greedily pack VAD speech regions into MIN..MAX-second (start,end) chunks."""
    max_s = int(MAX_SEC * sr)
    chunks, cur_s, cur_e = [], None, None
    for r in regions:
        s, e = r['start'], r['end']
        if cur_s is None:
            cur_s, cur_e = s, e
        elif (e - cur_s) <= max_s:
            cur_e = e                      # bridge the inter-speech gap
        else:
            chunks.append((cur_s, cur_e)); cur_s, cur_e = s, e
        if cur_e - cur_s >= max_s:
            chunks.append((cur_s, cur_e)); cur_s = cur_e = None
    if cur_s is not None:
        chunks.append((cur_s, cur_e))

    # hard-split any region longer than MAX_SEC into <=MAX_SEC windows
    out = []
    for s, e in chunks:
        if e - s > max_s:
            n = math.ceil((e - s) / max_s)
            step = (e - s) // n
            for k in range(n):
                out.append((s + k * step, min(e, s + (k + 1) * step)))
        else:
            out.append((s, e))
    return out


def rel_for(row, cidx) -> str:
    tag = "bonafide" if row.label == "bonafide" else str(row.generator_id)
    return f"{row.split}/{tag}/{row.utterance_id}_c{cidx:02d}.wav"


def main():
    ap = argparse.ArgumentParser(description="Silero-VAD 4-10s chunking with parity preprocessing.")
    ap.add_argument("--manifest", required=True, action="append",
                    help="Input manifest(s); pass multiple --manifest to merge sources.")
    ap.add_argument("--out-audio", required=True, help="root of the chunk mirror to create")
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--limit", type=int, default=0, help="process only first N rows (smoke test)")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(m, encoding="utf-8-sig", dtype={"speaker_id": str})
                    for m in args.manifest], ignore_index=True)
    if args.limit:
        df = df.head(args.limit)

    print(f"[INFO] loading Silero VAD ...")
    model, get_speech_timestamps = _load_vad()

    outroot = Path(args.out_audio)
    rows, bad, n_chunks = [], 0, 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        try:
            x = normalize(trim_silence(load_mono_16k(row.path)))
            if len(x) < int(MIN_KEEP_SEC * TARGET_SR):
                bad += 1
                continue
            wav = torch.from_numpy(x.astype(np.float32))
            ts = get_speech_timestamps(wav, model, sampling_rate=TARGET_SR)
            if not ts:                       # no speech detected -> keep whole clip
                spans = [(0, len(x))]
            else:
                spans = pack_chunks(ts, TARGET_SR, len(x))
            spans = spans[:MAX_CHUNKS]
            for cidx, (s, e) in enumerate(spans):
                seg = x[s:e]
                if len(seg) < int(MIN_KEEP_SEC * TARGET_SR):
                    continue
                dst = outroot / rel_for(row, cidx)
                dst.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(dst), seg, TARGET_SR, subtype="PCM_16")
                d = dict(row._asdict())
                d["path"] = str(dst.resolve())
                d["utterance_id"] = f"{row.utterance_id}_c{cidx:02d}"
                d["duration_s"] = round(len(seg) / TARGET_SR, 3)
                d["codec"] = "pcm_16k"
                rows.append(d); n_chunks += 1
        except Exception as e:
            bad += 1
            if bad <= 5:
                print(f"[WARN] {getattr(row,'path','?')}: {e}")
        if i % 1000 == 0:
            print(f"  {i:,}/{len(df):,} clips -> {n_chunks:,} chunks ({bad} skipped)", flush=True)

    out = pd.DataFrame(rows)
    outm = Path(args.out_manifest)
    outm.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outm, index=False, encoding="utf-8-sig")
    # per-split siblings for the trainer
    for sp, g in out.groupby("split"):
        g.to_csv(outm.with_name(outm.stem + f"_{sp}.csv"), index=False, encoding="utf-8-sig")

    print(f"\n[OK] {len(df):,} clips -> {len(out):,} chunks ({bad:,} skipped) -> {outm}")
    print(out.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()

