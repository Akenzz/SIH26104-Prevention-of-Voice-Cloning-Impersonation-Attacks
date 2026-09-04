"""Did the bandwidth-robust retrain actually fix the resampler dependence?

Compares the shipped models against hybrid_br_best.pth on the three things that
matter, in order of importance:

  1. RESAMPLER INVARIANCE (the actual bug). Score each clip through the backend's
     scipy path and through training's librosa path. A model whose verdict depends
     on the resampler is broken regardless of its EER. Target: gap -> 0.
     Shipped models measure 13.35 (hybrid) and 5.25 (hybrid_nc) mean |logit| gap.

  2. THE CLIP THAT STARTED THIS. mlaad/fake/ml/Edge-TTS was IN training yet read
     bonafide live (-9.16). It must now read spoof through the scipy path.

  3. LOCAL CLIPS — recall AND false alarms together. diag_resampler_fix.py showed
     the naive "just use librosa at inference" fix moved every clip spoofward
     (real 0.003 -> 0.830) without separating the classes, which a recall-only
     table would have scored as a win. So both are always printed.

The new model is scored WITH its band gate (read from the checkpoint, not
hardcoded) because that is how it must be served; the old models are scored
ungated because that is how they were trained and are currently deployed. This is
deliberately not an apples-to-apples feature comparison -- it compares each model
in the configuration it is actually valid in.

Note the local "REAL" clips are labelled by filename only and appear in no
manifest, so treat their absolute numbers as indicative.
"""

from __future__ import annotations

import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "realtime-backend"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

TARGET_SR = 16000
WINDOW = 4 * TARGET_SR
MAX_SEC = 30

EDGE_TTS = Path(r"E:\DatasetSIH\mlaad\fake\ml\Edge-TTS\dorothy_and_wizard_oz_01_f000031.wav")
LOCAL = Path(r"E:\DatasetSIH\Local_test")
CLIPS: list[tuple[Path, str]] = [
    (EDGE_TTS, "spoof"),
    (LOCAL / "sarosh_real_3sept.wav", "REAL"),
    (LOCAL / "sarosh_ZsVicqBh_original.wav", "REAL"),
    (LOCAL / "sudhanva.wav", "REAL"),
    (LOCAL / "sarosh_fireredtts.wav", "spoof"),
    (LOCAL / "sarosh_omni.wav", "spoof"),
    (LOCAL / "sarosh_chatterbox_spoof.wav", "spoof"),
    (LOCAL / "sarosh_styletts.wav", "spoof"),
    (LOCAL / "sarosh_spoof.wav", "spoof"),
    (LOCAL / "SAROSH_SOPRO_V2_SPOOF.wav", "spoof"),
]


def lowpass(x: np.ndarray, cutoff: float | None, sr: int = TARGET_SR) -> np.ndarray:
    if not cutoff or cutoff >= sr / 2:
        return x.astype(np.float32)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    spec[freqs >= cutoff] = 0.0
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def two_ways(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x[: sr * MAX_SEC]
    if sr == TARGET_SR:
        return x, x
    g = gcd(int(sr), TARGET_SR)
    a = resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float32)
    b = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
    return a, b


def wins(a: np.ndarray) -> list[np.ndarray]:
    if len(a) < WINDOW:
        return [np.pad(a, (0, WINDOW - len(a)))]
    return [a[i : i + WINDOW] for i in range(0, len(a) - WINDOW + 1, WINDOW // 2)]


class LocalCheckpoint:
    """Score raw audio with a checkpoint straight off disk, gate included."""

    def __init__(self, path: Path):
        from models.lfcc_lcnn import LFCCLCNNWithFeatureExtraction
        from training.config import TrainingConfig  # noqa: F401  (pickle shim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(str(path), map_location=self.device, weights_only=False)
        cfg = ck.get("config")
        self.band_gate_hz = ck.get("band_gate_hz")
        self.dev_eer = ck.get("best_eer")
        self.epoch = ck.get("epoch")
        self.model = LFCCLCNNWithFeatureExtraction(
            sample_rate=getattr(cfg, "sample_rate", 16000),
            n_lfcc=getattr(cfg, "n_lfcc", 60),
            with_deltas=getattr(cfg, "with_deltas", True),
            embedding_dim=getattr(cfg, "embedding_dim", 256),
            dropout=getattr(cfg, "dropout", 0.3),
        ).to(self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()

    def mean_logit(self, a: np.ndarray) -> float:
        a = lowpass(a, self.band_gate_hz)
        out = []
        with torch.no_grad():
            for w in wins(a):
                t = torch.from_numpy(w).float().unsqueeze(0).to(self.device)
                # forward() returns (logit, embedding) unless told otherwise
                y = self.model(t, return_embedding=False)
                if isinstance(y, tuple):
                    y = y[0]
                out.append(float(y.squeeze().item()))
        return float(np.mean(out))


def main() -> int:
    from config import MODEL_CACHE_DIR
    from experts.lfcc import LFCCLCNNExpert

    scorers: dict[str, tuple[object, float | None]] = {}
    for key in ("hybrid", "hybrid_nc"):
        try:
            e = LFCCLCNNExpert(cache_dir=MODEL_CACHE_DIR, hub_key=key, name=key)
            scorers[key] = (lambda a, _e=e: float(np.mean([_e.score(w)["logit"] for w in wins(a)])), None)
        except Exception as exc:
            print(f"[WARN] could not load {key}: {exc}")

    br = HERE / "checkpoints" / "hybrid_br_best.pth"
    if br.is_file():
        c = LocalCheckpoint(br)
        gate = c.band_gate_hz
        print(f"[INFO] hybrid_br_best.pth: epoch {c.epoch}, gated dev EER "
              f"{(c.dev_eer or float('nan'))*100:.4f}%, band gate {gate} Hz")
        scorers["hybrid_br"] = (c.mean_logit, gate)
    else:
        print(f"[WARN] {br.name} not found -- run finetune_bandwidth_robust.py first")

    rows = []
    for path, truth in CLIPS:
        if not path.exists():
            print(f"[WARN] missing {path}")
            continue
        a, b = two_ways(path)
        rows.append((path.name, truth, a, b))

    for name, (fn, gate) in scorers.items():
        tag = f"gated {gate:.0f} Hz" if gate else "ungated"
        print(f"\n=== {name}  ({tag})   mean logit, >0 = spoof ===")
        print(f"{'clip':34s} {'truth':6s} {'scipy':>8s} {'librosa':>8s} {'gap':>7s}  verdict(scipy)")
        gaps, caught, fa, n_sp, n_re = [], 0, 0, 0, 0
        for cname, truth, a, b in rows:
            ls, ll = fn(a), fn(b)
            gaps.append(abs(ls - ll))
            flagged = ls > 0
            if truth == "spoof":
                n_sp += 1
                caught += int(flagged)
            else:
                n_re += 1
                fa += int(flagged)
            mark = "SPOOF" if flagged else "bonafide"
            ok = "ok" if (flagged == (truth == "spoof")) else "MISS" if truth == "spoof" else "FALSE ALARM"
            print(f"{cname[:34]:34s} {truth:6s} {ls:8.2f} {ll:8.2f} {ls-ll:7.2f}  {mark:9s} {ok}")
        print(f"  mean |scipy-librosa| gap = {np.mean(gaps):.2f} logits   "
              f"(0 = resampler-invariant)")
        print(f"  spoof caught {caught}/{n_sp}   false alarms {fa}/{n_re}   @ logit>0")

    print("\nReminder: the local REAL clips are filename-labelled and in no manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
