"""
prep_newclips.py — fold Amogh's modern-engine spoof clips into the hybrid corpus.

What it does (see the friend's recipe: "same data as the latest hybrid + these
new clips, train 4-5 epochs"):

  1. Reads the 12 clips in SRC_DIR.
  2. Classifies each: clearly-spoof (engine in the name) / bonafide (the one
     mislabeled-by-location real clip) / EXCLUDED (ambiguous, no engine tag and
     corpus-like level -> too risky to auto-label as spoof).
  3. Resamples to 16 kHz mono and **RMS-normalizes to the corpus median (0.073)**
     so the model cannot shortcut on loudness (the new clips are ~1.4-1.8x louder
     than the corpus; the corpus itself is level-matched across classes).
  4. Slices each clip into non-overlapping 4 s chunks (drops a <2 s remainder).
  5. Writes chunks to OUT_DIR and appends rows to a COPY of the hybrid train
     manifest -> hybrid_plus_newclips_train.csv (dev/eval manifests untouched).

Nothing here trains anything; it only produces audio + a manifest. Idempotent:
re-running overwrites OUT_DIR chunks and rewrites the manifest.

Run from lfcc-detector/:  python prep_newclips.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC_DIR = Path(r"E:\DatasetSIH\spoof_amogh_sentg\spoof")
OUT_DIR = Path(r"E:\DatasetSIH\processed_newclips")
REPO = Path(__file__).resolve().parent.parent
BASE_TRAIN = REPO / "data_pipeline" / "manifests" / "hybrid_vad_chunks_train.csv"
OUT_TRAIN = REPO / "data_pipeline" / "manifests" / "hybrid_plus_newclips_train.csv"

TARGET_SR = 16000
CHUNK_SEC = 4.0
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SEC)
MIN_REMAINDER_SEC = 2.0
TARGET_RMS = 0.073  # corpus median RMS (bonafide 0.0738 / spoof 0.0719)

# Manifest columns, in the exact order the hybrid manifest uses.
COLUMNS = [
    "path", "label", "split", "source_dataset", "speaker_id", "utterance_id",
    "generator_id", "language", "codec", "duration_s", "license", "consent",
    "held_out", "role", "text", "group",
]


def classify(name: str):
    """Return (label, generator_id) or (None, reason) to exclude."""
    n = name.lower()
    if "bonafide" in n:
        return "bonafide", "real_reference"
    if "qwen" in n:
        return "spoof", "qwen"
    if "chatterbox" in n:
        return "spoof", "chatterbox"
    if "firered" in n:
        return "spoof", "fireredtts"
    if "omni" in n:
        return "spoof", "omni"
    if "styletts" in n:
        return "spoof", "styletts2"
    if "sopro" in n:
        return "spoof", "sopro_v2"
    if "my_spoof" in n or "sarosh_spoof" in n:
        return "spoof", "newclip_generic"
    # audio(1).wav, spk_*.wav, tmp*.wav -> no engine tag; too risky to call spoof.
    return None, "ambiguous-no-engine-tag"


def load_16k_mono(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)  # -> mono (samples,)
    if sr != TARGET_SR:
        # lightweight polyphase resample without pulling in torch here
        from math import gcd
        import scipy.signal as ss
        g = gcd(sr, TARGET_SR)
        x = ss.resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)
    return x


def rms_normalize(x: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt((x ** 2).mean() + 1e-12))
    if rms < 1e-6:
        return x
    y = x * (TARGET_RMS / rms)
    peak = float(np.abs(y).max())
    if peak > 0.99:  # guard against clipping introduced by the gain-up
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def main() -> int:
    if not SRC_DIR.is_dir():
        print(f"[ERR] source dir not found: {SRC_DIR}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    kept, excluded = [], []
    per_engine = {}

    for f in sorted(SRC_DIR.iterdir()):
        if not f.is_file():
            continue
        label, gen = classify(f.name)
        if label is None:
            excluded.append((f.name, gen))
            continue
        try:
            x = load_16k_mono(f)
        except Exception as e:  # noqa: BLE001
            excluded.append((f.name, f"load-failed: {e}"))
            continue
        x = rms_normalize(x)

        eng_dir = OUT_DIR / gen
        eng_dir.mkdir(parents=True, exist_ok=True)
        stem = f.stem.replace(" ", "_").replace("(", "").replace(")", "")

        n = len(x)
        n_full = n // CHUNK_SAMPLES
        made = 0
        for i in range(n_full):
            chunk = x[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
            _emit(rows, eng_dir, stem, i, chunk, label, gen)
            made += 1
        rem = n - n_full * CHUNK_SAMPLES
        if rem >= int(MIN_REMAINDER_SEC * TARGET_SR):
            chunk = x[n_full * CHUNK_SAMPLES:]
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))  # tail-pad last piece
            _emit(rows, eng_dir, stem, n_full, chunk, label, gen)
            made += 1

        kept.append((f.name, label, gen, made))
        per_engine[gen] = per_engine.get(gen, 0) + made

    # ---- assemble augmented train manifest ----
    base = pd.read_csv(BASE_TRAIN, low_memory=False)
    add = pd.DataFrame(rows, columns=COLUMNS)
    # keep only columns base has, in base's order (base is the source of truth)
    add = add.reindex(columns=base.columns)
    combined = pd.concat([base, add], ignore_index=True)
    combined.to_csv(OUT_TRAIN, index=False, encoding="utf-8")

    # ---- report ----
    print("=" * 64)
    print("PREP SUMMARY")
    print("=" * 64)
    print(f"source        : {SRC_DIR}")
    print(f"chunks out    : {OUT_DIR}")
    print(f"new manifest  : {OUT_TRAIN}")
    print()
    print("KEPT:")
    for name, label, gen, made in kept:
        print(f"  {name:32s} -> {label:8s} [{gen:16s}] {made:3d} chunks")
    print()
    print("EXCLUDED (not added — verify manually):")
    for name, why in excluded:
        print(f"  {name:32s} -- {why}")
    print()
    n_new_spoof = sum(1 for r in rows if r[1] == "spoof")
    n_new_bona = sum(1 for r in rows if r[1] == "bonafide")
    print(f"new chunks    : spoof={n_new_spoof}  bonafide={n_new_bona}  (per-engine: {per_engine})")
    print(f"base rows     : {len(base):,}")
    print(f"combined rows : {len(combined):,}  (+{len(add)})")
    bl = base["label"].value_counts().to_dict()
    cl = combined["label"].value_counts().to_dict()
    print(f"label dist    : base {bl}  ->  combined {cl}")
    print("=" * 64)
    return 0


def _emit(rows, eng_dir, stem, i, chunk, label, gen):
    out = eng_dir / f"{stem}_c{i:03d}.wav"
    sf.write(str(out), chunk, TARGET_SR, subtype="PCM_16")
    rows.append([
        str(out), label, "train", "amogh_newclips", "sarosh",
        f"{stem}_c{i:03d}", gen, "en", "pcm_16k", round(len(chunk) / TARGET_SR, 3),
        "internal", "research", "no", "newclip", "", "en_newclips",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
