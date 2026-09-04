"""Diagnose why Edge-TTS/ml (an IN-TRAIN generator) reads bonafide at inference.

Scores the SAME utterances two ways:
  raw       E:\\DatasetSIH\\mlaad\\fake\\ml\\Edge-TTS\\<name>.wav        (what the user uploaded)
  processed E:\\DatasetSIH\\processed_hybrid_vad\\train\\Edge-TTS\\mlml_Edge-TTS_<name>_cNN.wav
            (the exact bytes the model trained on)

If processed scores spoof and raw scores bonafide on the same utterance, the gap
is preprocessing (loudness / VAD / resample), not the model forgetting the generator.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "realtime-backend"))

RAW_DIR = Path(r"E:\DatasetSIH\mlaad\fake\ml\Edge-TTS")
TRAIN_CSV = REPO / "data_pipeline" / "manifests" / "hybrid_vad_chunks_train.csv"
EVAL_CSV = REPO / "data_pipeline" / "manifests" / "hybrid_vad_chunks_eval.csv"

TARGET_SR = 16000
WINDOW = 4 * TARGET_SR


def load_16k_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    native_sr = sr
    if sr != TARGET_SR:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, int(sr) // g).astype(np.float32)
    return audio, native_sr


def windows_of(audio: np.ndarray) -> list[np.ndarray]:
    if len(audio) < WINDOW:
        return [np.pad(audio, (0, WINDOW - len(audio)))]
    hop = WINDOW // 2
    return [audio[i : i + WINDOW] for i in range(0, len(audio) - WINDOW + 1, hop)]


def score_all(expert, audio: np.ndarray) -> tuple[float, float, int]:
    """Return (mean_logit, max_logit, n_windows)."""
    ls = [expert.score(w)["logit"] for w in windows_of(audio)]
    return float(np.mean(ls)), float(np.max(ls)), len(ls)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def main() -> None:
    from config import MODEL_CACHE_DIR
    from experts.lfcc import LFCCLCNNExpert

    print("loading experts...", flush=True)
    hybrid = LFCCLCNNExpert(cache_dir=MODEL_CACHE_DIR, hub_key="hybrid", name="hybrid")
    nc = LFCCLCNNExpert(cache_dir=MODEL_CACHE_DIR, hub_key="hybrid_nc", name="hybrid_nc")

    # Map raw utterance base -> one processed chunk path + which split it lives in
    proc: dict[str, tuple[str, str]] = {}
    for csv_path, split in ((TRAIN_CSV, "train"), (EVAL_CSV, "eval")):
        with open(csv_path, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("generator_id") != "Edge-TTS" or r.get("language") != "ml":
                    continue
                m = re.match(r"mlml_Edge-TTS_(.+)_c\d+$", r["utterance_id"])
                if m and m.group(1) not in proc:
                    proc[m.group(1)] = (r["path"], split)

    raw_files = sorted(p for p in RAW_DIR.glob("*.wav"))
    picks = [p for p in raw_files if p.stem in proc][:8]
    picks += [p for p in raw_files if p.stem not in proc][:4]

    hdr = f"{'utterance':44s} {'split':6s} {'sr':>6s} {'rms':>7s} {'hyb':>7s} {'hyb_nc':>7s} {'verdict':>9s}"
    print("\n=== RAW file as uploaded (native loudness, no VAD) ===")
    print(hdr)
    raw_rows = {}
    for p in picks:
        audio, sr = load_16k_mono(p)
        h_mean, _, n = score_all(hybrid, audio)
        c_mean, _, _ = score_all(nc, audio)
        split = proc.get(p.stem, ("", "UNSEEN"))[1]
        raw_rows[p.stem] = (h_mean, c_mean)
        v = "SPOOF" if c_mean > 0 else "bonafide"
        print(f"{p.stem:44s} {split:6s} {sr:6d} {rms(audio):7.4f} {h_mean:+7.2f} {c_mean:+7.2f} {v:>9s}")

    print("\n=== SAME utterances, PROCESSED chunk the model actually trained on ===")
    print(hdr)
    for p in picks:
        if p.stem not in proc:
            continue
        ppath, split = proc[p.stem]
        if not os.path.exists(ppath):
            print(f"{p.stem:44s} {split:6s}  <processed file missing on disk>")
            continue
        audio, sr = load_16k_mono(Path(ppath))
        h_mean, _, _ = score_all(hybrid, audio)
        c_mean, _, _ = score_all(nc, audio)
        v = "SPOOF" if c_mean > 0 else "bonafide"
        rh, rc = raw_rows[p.stem]
        print(
            f"{p.stem:44s} {split:6s} {sr:6d} {rms(audio):7.4f} {h_mean:+7.2f} {c_mean:+7.2f} {v:>9s}"
            f"   (raw was hyb {rh:+.2f} / nc {rc:+.2f})"
        )

    # Loudness sweep on one raw clip: is the miss a level effect?
    print("\n=== loudness sweep on one raw in-train clip (hybrid_nc mean logit) ===")
    target = picks[0]
    audio, _ = load_16k_mono(target)
    base = rms(audio)
    print(f"clip={target.stem}  native rms={base:.4f}")
    for tgt in (base, 0.02, 0.04, 0.073, 0.10, 0.15):
        scaled = audio if abs(tgt - base) < 1e-9 else audio * (tgt / max(base, 1e-9))
        scaled = np.clip(scaled, -1.0, 1.0)
        h_mean, _, _ = score_all(hybrid, scaled)
        c_mean, _, _ = score_all(nc, scaled)
        tag = " <- native" if abs(tgt - base) < 1e-9 else (" <- corpus median" if tgt == 0.073 else "")
        print(f"  rms={tgt:6.4f}  hyb {h_mean:+7.2f}   hyb_nc {c_mean:+7.2f}{tag}")


if __name__ == "__main__":
    main()
