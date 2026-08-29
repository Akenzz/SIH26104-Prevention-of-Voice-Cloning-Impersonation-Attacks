"""
data_pipeline/generate_xtts_local.py
====================================
Finish XTTS-v2 spoof generation on a LOCAL GPU (or CPU) instead of Colab.

Reads hindi_spoof_jobs.csv, filters to the target generator, and writes each
job to its out_relpath under --spoof-root. No upload/download: the reference
clip is the original Kathbath wav (src_wav), already on disk.

Resumable & crash-safe:
  - skips a job whose output already exists AND is readable AND > 0.3 s
    (so truncated files from a killed Colab run are regenerated, not trusted)
  - writes to <out>.tmp then atomically renames, so a kill mid-write never
    leaves a half-written wav that a later run would wrongly skip

Usage (from repo root):
  python data_pipeline/generate_xtts_local.py                 # auto: cuda if available
  python data_pipeline/generate_xtts_local.py --device cpu    # force CPU
  python data_pipeline/generate_xtts_local.py --limit 50      # smoke test first
"""

import os
import sys
import argparse
from pathlib import Path

# Devanagari in job text would crash Windows cp1252 stdout on print — force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
# Skip the interactive Coqui license prompt on first model download.
os.environ.setdefault("COQUI_TOS_AGREED", "1")
# Reduce CUDA fragmentation on small (4 GB) cards.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import soundfile as sf

repo_root = Path(__file__).resolve().parent.parent
MANIFEST_DIR = repo_root / "data_pipeline" / "manifests"
DEFAULT_SPOOF_ROOT = Path(os.environ.get("HINDI_SPOOF_ROOT", r"E:\DatasetSIH\hindi_spoof"))
MIN_OK_SEC = 0.3


def output_ok(path: Path) -> bool:
    """True if a prior output exists and looks complete (not truncated)."""
    if not path.exists():
        return False
    try:
        info = sf.info(str(path))
        return (info.frames / info.samplerate) >= MIN_OK_SEC
    except Exception:
        return False


def resolve_ref(job, refs_fallback: Path | None) -> Path | None:
    p = Path(job.src_wav)
    if p.exists():
        return p
    if refs_fallback:
        cand = refs_fallback / job.split / f"{job.utt_id}.wav"
        if cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description="Local XTTS-v2 finisher for the Hindi spoof jobs.")
    ap.add_argument("--jobs", default=str(MANIFEST_DIR / "hindi_spoof_jobs.csv"))
    ap.add_argument("--spoof-root", default=str(DEFAULT_SPOOF_ROOT))
    ap.add_argument("--generator", default="xtts")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--refs-fallback", default=str(DEFAULT_SPOOF_ROOT / "refs"),
                    help="folder of staged refs, used only if src_wav is missing")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else stop after N generated")
    args = ap.parse_args()

    import torch
    
    # [MONKEY-PATCH] coqui-tts relies on an old transformers function that was removed.
    # Instead of compiling old transformers from source (which requires Rust on Python 3.12),
    # we just patch it back in right before importing TTS.
    import transformers.pytorch_utils
    if not hasattr(transformers.pytorch_utils, 'isin_mps_friendly'):
        transformers.pytorch_utils.isin_mps_friendly = torch.isin
        
    from TTS.api import TTS

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    jobs = pd.read_csv(args.jobs, encoding="utf-8-sig", dtype={"speaker_id": str})
    jobs = jobs[jobs.target_generator == args.generator].reset_index(drop=True)
    root = Path(args.spoof_root)
    refs_fallback = Path(args.refs_fallback) if args.refs_fallback else None

    # Pre-count what's left so you see the real workload up front.
    todo = [j for j in jobs.itertuples(index=False)
            if not output_ok(root / j.out_relpath)]
    print(f"{args.generator}: {len(jobs):,} total | {len(jobs) - len(todo):,} done | {len(todo):,} to do")
    if not todo:
        print("nothing to generate — all outputs present.")
        return

    print("loading XTTS-v2 (first run downloads ~1.8 GB) ...", flush=True)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    made = failed = noref = 0
    for i, j in enumerate(todo, 1):
        ref = resolve_ref(j, refs_fallback)
        if ref is None:
            noref += 1
            continue
        out = root / j.out_relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.wav")
        try:
            tts.tts_to_file(text=str(j.text), speaker_wav=str(ref),
                            language="hi", file_path=str(tmp))
            os.replace(tmp, out)            # atomic: only a complete file gets the real name
            made += 1
        except Exception as e:
            failed += 1
            tmp.unlink(missing_ok=True)
            msg = str(e).splitlines()[0][:160]
            print(f"  FAIL {j.job_id} ({j.utt_id}): {msg}")
            if "out of memory" in str(e).lower() and device == "cuda":
                torch.cuda.empty_cache()
                print("  [hint] CUDA OOM — if this repeats, rerun with --device cpu")
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  made={made} failed={failed} noref={noref}", flush=True)
        if args.limit and made >= args.limit:
            print(f"  reached --limit {args.limit}, stopping.")
            break

    print("=" * 48)
    print(f"{args.generator} local run: made={made:,} failed={failed:,} noref={noref:,}")
    print("re-run this same command to retry any failures (already-done files are skipped).")
    print("=" * 48)


if __name__ == "__main__":
    main()
