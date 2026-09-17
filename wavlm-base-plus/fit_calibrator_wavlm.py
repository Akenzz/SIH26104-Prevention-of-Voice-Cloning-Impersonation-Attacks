"""Fit a Platt calibrator for wavlm_maxbr_best.pt on the gated val set.

Uses /media/akenzz/D1/DataSet_processed/v3_extended/val.csv which has
correct absolute paths on the D1 drive.  Reads the 7 kHz band gate from
the checkpoint (must match dataset.py BAND_GATE_HZ) and applies it when
scoring every window.

Sampling: class-balanced 50%/50% (3 000 spoof + 3 000 bonafide = 6 000
windows).  A random 50% of those are held out for honest ECE/EER reporting;
the other 50% are used to fit the Platt a/b.

Run from the REPO ROOT:

    source .venv/bin/activate
    python wavlm-base-plus/fit_calibrator_wavlm.py 2>&1 | tee fit_calibrator_wavlm.log

Takes ~5-15 min on GPU depending on drive speed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as F_audio
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent   # wavlm-base-plus/
REPO = HERE.parent                        # repo root

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from model import WavLMClassifier  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────────
CKPT         = HERE / "checkpoints" / "wavlm_maxbr_best.pt"
VAL_CSV      = Path("/media/akenzz/D1/DataSet_processed/v3_extended/val.csv")
OUT          = REPO / "realtime-backend" / "artifacts" / "platt_v5.json"

TARGET_SR    = 16_000
WINDOW_SECS  = 4
WINDOW_SAMP  = TARGET_SR * WINDOW_SECS   # 64 000 samples

# 3 000 per class → 6 000 total (50 / 50 spoof / bonafide)
MAX_PER_CLASS = 3_000

LABEL_MAP = {"bonafide": 0, "spoof": 1}


# ── Dataset ───────────────────────────────────────────────────────────────────────

class _ValDataset(Dataset):
    """Reads a flat val.csv (path, label, ...) and applies the band gate."""

    def __init__(self, rows: pd.DataFrame, gate_hz: int | None):
        self._rows    = rows.reset_index(drop=True)
        self._gate_hz = gate_hz

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        row   = self._rows.iloc[idx]
        label = LABEL_MAP[str(row["label"]).strip()]
        path  = str(row["path"]).strip()

        try:
            audio_np, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            return None  # sentinel — filtered by collate

        wav = torch.from_numpy(audio_np.T)          # (channels, samples)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)     # mono
        if sr != TARGET_SR:
            from torchaudio.transforms import Resample
            wav = Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
        if self._gate_hz:
            wav = F_audio.lowpass_biquad(wav, TARGET_SR, float(self._gate_hz))

        # Pad / crop to fixed length
        n = wav.shape[-1]
        if n < WINDOW_SAMP:
            wav = torch.nn.functional.pad(wav, (0, WINDOW_SAMP - n))
        elif n > WINDOW_SAMP:
            wav = wav[..., :WINDOW_SAMP]

        return wav.squeeze(0).float(), label   # (WINDOW_SAMP,), int


def _collate(batch):
    valid = [b for b in batch if b is not None]
    if not valid:
        return None
    waveforms = torch.stack([b[0] for b in valid])
    labels    = torch.tensor([b[1] for b in valid], dtype=torch.long)
    return waveforms, labels


# ── Metric helpers ────────────────────────────────────────────────────────────────

def _eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores)
    s, y  = scores[order], labels[order]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        return float("nan"), float("nan")
    rates = []
    for t in np.unique(s):
        far = float((s[y == 0] >= t).sum()) / n_neg
        frr = float((s[y == 1] <  t).sum()) / n_pos
        rates.append((abs(far - frr), (far + frr) / 2, float(t)))
    rates.sort()
    return rates[0][1], rates[0][2]


def _ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (probs >= edges[i]) & (
            probs < edges[i + 1] if i < bins - 1 else probs <= 1.0
        )
        if not m.any():
            continue
        total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(total)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────────
    print(f"[INFO] loading checkpoint {CKPT} ...")
    ck = torch.load(str(CKPT), map_location=device, weights_only=False)
    if not isinstance(ck, dict):
        print("[ERR] checkpoint is not a dict — wrong file?")
        return 1

    gate      = ck.get("band_gate_hz")
    epoch     = ck.get("epoch", "?")
    dev_eer_c = ck.get("dev_eer", float("nan"))
    print(
        f"[INFO] wavlm_maxbr_best.pt: epoch={epoch}, "
        f"dev_eer={float(dev_eer_c)*100:.2f}%, band_gate_hz={gate}"
    )
    print(f"[INFO] will apply {gate} Hz lowpass gate when scoring." if gate
          else "[WARN] no band_gate_hz in checkpoint; scoring without gate.")

    model = WavLMClassifier()
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()

    # ── Load and balance val.csv ──────────────────────────────────────────────────
    print(f"[INFO] reading {VAL_CSV} ...")
    if not VAL_CSV.exists():
        print(f"[ERR] val.csv not found at {VAL_CSV}")
        return 1

    df = pd.read_csv(VAL_CSV)
    print(f"[INFO] total rows: {len(df)}")

    # Normalise the label column name
    if "label" not in df.columns:
        print(f"[ERR] no 'label' column found. Columns: {list(df.columns)}")
        return 1

    spoof_df    = df[df["label"] == "spoof"].sample(
        n=min(MAX_PER_CLASS, (df["label"] == "spoof").sum()),
        random_state=42,
    )
    bonafide_df = df[df["label"] == "bonafide"].sample(
        n=min(MAX_PER_CLASS, (df["label"] == "bonafide").sum()),
        random_state=42,
    )
    balanced = pd.concat([spoof_df, bonafide_df]).sample(frac=1, random_state=0).reset_index(drop=True)
    print(
        f"[INFO] balanced sample: {len(balanced)} rows "
        f"(spoof={len(spoof_df)}, bonafide={len(bonafide_df)})"
    )

    ds     = _ValDataset(balanced, gate_hz=gate)
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        collate_fn=_collate, num_workers=0)

    # ── Score ─────────────────────────────────────────────────────────────────────
    logits_list: list[float] = []
    labels_list: list[int]   = []
    skipped = 0

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if batch is None:
                skipped += 8
                continue
            x, y = batch
            x = x.to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            logits_list.extend(out.squeeze(-1).cpu().numpy().tolist())
            labels_list.extend(y.numpy().tolist())
            if i % 50 == 0:
                print(f"  scored {len(logits_list)}/{len(balanced)} windows "
                      f"(skipped {skipped} broken)", flush=True)

    if skipped:
        print(f"[WARN] skipped {skipped} windows (missing/unreadable files).")

    if len(logits_list) < 200:
        print(f"[ERR] only {len(logits_list)} windows scored — too few to fit. "
              "Verify the D1 drive is mounted.")
        return 1

    logits = np.asarray(logits_list, dtype=np.float64)
    labels = np.asarray(labels_list, dtype=np.float64)
    print(
        f"[INFO] scored {len(logits)} windows | "
        f"spoof={int((labels==1).sum())}  bonafide={int((labels==0).sum())}"
    )

    # ── 50 / 50 row split for honest held-out metrics ────────────────────────────
    rng = np.random.default_rng(0)
    idx = np.arange(len(logits))
    rng.shuffle(idx)
    half   = max(1, len(idx) // 2)
    fit_m  = np.zeros(len(logits), dtype=bool)
    fit_m[idx[:half]] = True
    held_m = ~fit_m

    if held_m.sum() == 0 or len(np.unique(labels[fit_m])) < 2:
        print("[ERR] degenerate split — need both classes in fit half.")
        return 1

    # ── Platt fit ─────────────────────────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logits[fit_m].reshape(-1, 1), labels[fit_m])
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])

    def prob(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(a * z + b)))

    h_eer, h_thr = _eer(labels[held_m], logits[held_m])
    h_ece        = _ece(prob(logits[held_m]), labels[held_m])

    print(f"\n[FIT]  a={a:.10f}  b={b:.10f}   (fit on {int(fit_m.sum())} rows)")
    print(
        f"[HELD-OUT]  n={int(held_m.sum())}  "
        f"EER={h_eer*100:.4f}%  ECE={h_ece:.5f}  "
        f"logit_thresh_at_EER={h_thr:.4f}"
    )

    # ── Write ─────────────────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    gate_tag = int(gate) if gate else "ungated"
    payload = {
        "version":     f"platt-wavlm-ep{epoch}-gated{gate_tag}-refit",
        "kind":        "platt",
        "a":           a,
        "b":           b,
        "band_gate_hz": int(gate) if gate else None,
        "checkpoint":  CKPT.name,
        "fitted_on":   str(VAL_CSV) + " (class-balanced 50/50, row-50/50 held-out)",
        "n_fit":       int(fit_m.sum()),
        "n_heldout":   int(held_m.sum()),
        "heldout_eer": h_eer,
        "heldout_ece": h_ece,
        "note": (
            f"Platt calibrator refit for wavlm_maxbr_best.pt (epoch {epoch}) "
            f"on gated ({gate} Hz) audio. "
            "Rerun wavlm-base-plus/fit_calibrator_wavlm.py whenever the checkpoint changes."
        ),
        "metadata": {
            "expert":        "wavlm",
            "model_version": f"wavlm-base-plus-ep{epoch}",
            "manifest":      str(VAL_CSV.name),
            "split":         "val",
            "balanced":      True,
            "band_gate_hz":  int(gate) if gate else None,
            "n_windows_fit": int(fit_m.sum()),
            "n_spoof":       int((labels[fit_m] == 1).sum()),
            "n_bonafide":    int((labels[fit_m] == 0).sum()),
            "max_windows_per_class": MAX_PER_CLASS,
            "held_out_eval": {
                "n_eval_windows": int(held_m.sum()),
                "eer":            h_eer,
                "ece":            h_ece,
            },
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {OUT}")
    print("[OK] restart the backend — the startup warning should be gone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
