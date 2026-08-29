"""
data_pipeline/preprocess_parity.py
==================================
Apply the SAME preprocessing to EVERY clip — bonafide and spoof alike — and write
the results into a processed/ mirror, emitting a manifest that points at them.

  16 kHz mono  ->  energy silence-trim  ->  loudness-normalize  ->  peak guard

Identical treatment is the entire point. Real Kathbath recordings and TTS/VC
output differ in leading silence, loudness, and file fingerprints; if you don't
flatten those, a detector can learn "less leading silence => fake" instead of a
real spoof cue. Run canary_silence.py before/after to prove it worked.

Deps (recommended): librosa (resample + trim), pyloudnorm (LUFS). Both degrade
gracefully to fallbacks if missing, but install them for real results:
  pip install librosa pyloudnorm
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False
try:
    import pyloudnorm as pyln
    HAVE_PYLN = True
except Exception:
    HAVE_PYLN = False

TARGET_SR = 16000
TARGET_LUFS = -23.0
TRIM_TOP_DB = 30


def load_mono_16k(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != TARGET_SR:
        if HAVE_LIBROSA:
            x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
        else:
            raise RuntimeError(f"{sr} Hz needs librosa to resample to {TARGET_SR}")
    return x


def trim_silence(x: np.ndarray) -> np.ndarray:
    if HAVE_LIBROSA:
        xt, _ = librosa.effects.trim(x, top_db=TRIM_TOP_DB)
        return xt if len(xt) else x
    fl = int(0.02 * TARGET_SR)
    if len(x) < fl:
        return x
    rms = np.array([np.sqrt(np.mean(x[i:i + fl] ** 2) + 1e-9)
                    for i in range(0, len(x) - fl, fl)])
    thr = max(rms.max() * 0.05, 1e-4)
    keep = np.where(rms > thr)[0]
    if len(keep) == 0:
        return x
    return x[keep[0] * fl: (keep[-1] + 1) * fl]


def normalize(x: np.ndarray) -> np.ndarray:
    if HAVE_PYLN:
        meter = pyln.Meter(TARGET_SR)
        loud = meter.integrated_loudness(x)
        if np.isfinite(loud):
            x = pyln.normalize.loudness(x, loud, TARGET_LUFS)
    else:
        rms = np.sqrt(np.mean(x ** 2) + 1e-9)
        x = x * (10 ** (TARGET_LUFS / 20) / rms)   # coarse RMS proxy for LUFS
    peak = float(np.max(np.abs(x)) + 1e-9)
    if peak > 0.98:
        x = x * (0.98 / peak)
    return x.astype(np.float32)


def rel_for(row) -> str:
    tag = "bonafide" if row.label == "bonafide" else str(row.generator_id)
    return f"{row.split}/{tag}/{row.utterance_id}.wav"


def main():
    ap = argparse.ArgumentParser(description="Uniform 16k/trim/loudness pass over a manifest (bonafide + spoof).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-audio", required=True, help="root of the processed/ mirror to create")
    ap.add_argument("--out-manifest", required=True)
    args = ap.parse_args()

    if not HAVE_LIBROSA:
        print("[WARN] librosa missing: cannot resample non-16k, trim is coarse.")
    if not HAVE_PYLN:
        print("[WARN] pyloudnorm missing: using RMS-normalize fallback (not true LUFS).")

    df = pd.read_csv(args.manifest, encoding="utf-8-sig", dtype={"speaker_id": str})
    outroot = Path(args.out_audio)
    rows, bad = [], 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        try:
            x = normalize(trim_silence(load_mono_16k(row.path)))
            if len(x) < int(0.4 * TARGET_SR):
                bad += 1
                continue
            dst = outroot / rel_for(row)
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), x, TARGET_SR, subtype="PCM_16")
            d = dict(row._asdict())
            d["path"] = str(dst.resolve())
            d["duration_s"] = round(len(x) / TARGET_SR, 3)
            d["codec"] = "pcm_16k"
            rows.append(d)
        except Exception as e:
            bad += 1
            if bad <= 5:
                print(f"[WARN] {row.path}: {e}")
        if i % 2000 == 0:
            print(f"  {i:,}/{len(df):,} processed ...", flush=True)

    out = pd.DataFrame(rows)
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_manifest, index=False, encoding="utf-8-sig")
    print(f"[OK] processed {len(out):,} clips ({bad:,} skipped) -> {args.out_manifest}")


if __name__ == "__main__":
    main()
