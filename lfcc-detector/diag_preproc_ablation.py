"""Which preprocessing step makes an in-corpus spoof clip readable?

diag_edge_ml.py showed the SAME Edge-TTS/ml utterance scores bonafide as a raw
22.05k file (-9) and confidently spoof as its processed training chunk (+19).
Level is not the cause (a 7.5x loudness sweep moved the logit ~2).

The training corpus went through data_pipeline/preprocess_vad_slice.py:
    librosa.resample -> librosa.effects.trim(top_db=30)
    -> pyloudnorm -23 LUFS + peak guard -> Silero-VAD repack
The realtime backend does only scipy resample_poly. This script adds one training
step at a time to the raw file to find which one carries the spoof evidence.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "realtime-backend"))
sys.path.insert(0, str(REPO))

RAW_DIR = Path(r"E:\DatasetSIH\mlaad\fake\ml\Edge-TTS")
TARGET_SR = 16000
WINDOW = 4 * TARGET_SR


def windows_of(a: np.ndarray) -> list[np.ndarray]:
    if len(a) < WINDOW:
        return [np.pad(a, (0, WINDOW - len(a)))]
    hop = WINDOW // 2
    return [a[i : i + WINDOW] for i in range(0, len(a) - WINDOW + 1, hop)]


def mean_logit(expert, a: np.ndarray) -> float:
    return float(np.mean([expert.score(w)["logit"] for w in windows_of(a)]))


def variants(path: Path) -> dict[str, np.ndarray]:
    """Raw file under each cumulative preprocessing chain."""
    import librosa
    from math import gcd
    from scipy.signal import resample_poly

    from data_pipeline.preprocess_parity import normalize, trim_silence

    x0, sr = sf.read(str(path), dtype="float32")
    if x0.ndim > 1:
        x0 = x0.mean(axis=1)

    g = gcd(int(sr), TARGET_SR)
    scipy_rs = resample_poly(x0, TARGET_SR // g, int(sr) // g).astype(np.float32)
    libr_rs = librosa.resample(x0, orig_sr=sr, target_sr=TARGET_SR)

    out = {
        "A scipy resample (what the BACKEND does)": scipy_rs,
        "B librosa resample only": libr_rs,
        "C librosa + trim": trim_silence(libr_rs),
        "D librosa + trim + LUFS  (= training parity)": normalize(trim_silence(libr_rs)),
        "E scipy resample + trim + LUFS": normalize(trim_silence(scipy_rs)),
    }
    return out


def main() -> None:
    from config import MODEL_CACHE_DIR
    from experts.lfcc import LFCCLCNNExpert

    print("loading experts ...", flush=True)
    hyb = LFCCLCNNExpert(cache_dir=MODEL_CACHE_DIR, hub_key="hybrid", name="hybrid")
    nc = LFCCLCNNExpert(cache_dir=MODEL_CACHE_DIR, hub_key="hybrid_nc", name="hybrid_nc")

    # The processed chunk for the same utterance, for reference.
    proc: dict[str, str] = {}
    for name in ("hybrid_vad_chunks_train.csv", "hybrid_vad_chunks_eval.csv"):
        with open(REPO / "data_pipeline" / "manifests" / name, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("generator_id") == "Edge-TTS" and r.get("language") == "ml":
                    m = re.match(r"mlml_Edge-TTS_(.+)_c\d+$", r["utterance_id"])
                    if m:
                        proc.setdefault(m.group(1), r["path"])

    picks = [p for p in sorted(RAW_DIR.glob("*.wav")) if p.stem in proc][:4]

    for p in picks:
        print(f"\n--- {p.stem} ---")
        print(f"{'chain':46s} {'rms':>7s} {'hybrid':>8s} {'hybrid_nc':>10s}  verdict(nc)")
        for tag, a in variants(p).items():
            r = float(np.sqrt(np.mean(a**2))) if len(a) else 0.0
            h, c = mean_logit(hyb, a), mean_logit(nc, a)
            print(f"{tag:46s} {r:7.4f} {h:+8.2f} {c:+10.2f}  {'SPOOF' if c > 0 else 'bonafide'}")
        pa, psr = sf.read(proc[p.stem], dtype="float32")
        if pa.ndim > 1:
            pa = pa.mean(axis=1)
        h, c = mean_logit(hyb, pa), mean_logit(nc, pa)
        r = float(np.sqrt(np.mean(pa**2)))
        print(f"{'F processed chunk on disk (+Silero VAD)':46s} {r:7.4f} {h:+8.2f} {c:+10.2f}  "
              f"{'SPOOF' if c > 0 else 'bonafide'}")


if __name__ == "__main__":
    main()
