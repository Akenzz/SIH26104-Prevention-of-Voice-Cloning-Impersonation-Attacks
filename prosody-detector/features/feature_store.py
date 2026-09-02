"""
Path-keyed prosody feature cache with multiprocess extraction.
==============================================================
Feature extraction is the expensive step (~1.1 s/clip: librosa.pyin twice plus
Praat), so every experiment that touches the same audio should pay for it once.

This replaces the implicit cache in training/train_prosody.py, which stored only
(X, y) under a fixed filename and was therefore keyed by *nothing* — pointing the
trainer at a different manifest silently reused the previous manifest's features.
Here the key is the absolute audio path, so mixing manifests is safe and adding
new clips only extracts the new ones.

`load_eval_window` reproduces the eval-time window that lfcc-detector's
AudioDataset produces (trim silence -> tile-pad short -> centre-crop long), so
logits from this store are directly comparable to the numbers in RESULTS.md.
Resampling uses scipy.signal.resample_poly rather than torchaudio to keep worker
processes light; `n_resampled` in the extract summary reports how many clips it
touched (0 for the project's own 16 kHz pipeline output).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

SR = 16000
WINDOW_SAMPLES = SR * 4  # 4.0 s, matches WINDOW_SEC everywhere else in the repo


def _trim_silence(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """20 ms frame-RMS gate at -20 dB from the loudest frame.

    Mirrors AudioDataset._trim_silence exactly, including its fallbacks: an
    all-silent clip and a clip the gate would reduce below 0.2 s are returned
    untouched. Applied identically to both classes — it exists to stop
    "how much leading silence" from becoming a label shortcut.
    """
    fl = max(int(0.02 * sr), 1)
    if x.size < 2 * fl:
        return x
    n_frames = x.size // fl
    frames = x[: n_frames * fl].reshape(n_frames, fl)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-9)
    thr = max(float(rms.max()) * 0.1, 1e-4)
    keep = np.where(rms > thr)[0]
    if keep.size == 0:
        return x
    trimmed = x[keep[0] * fl : (keep[-1] + 1) * fl]
    if trimmed.size < int(0.2 * sr):
        return x
    return trimmed


_n_resampled = 0


def load_eval_window(path: str | Path, window_samples: int = WINDOW_SAMPLES,
                     sr: int = SR) -> np.ndarray:
    """One eval-time 4 s window from an audio file, as AudioDataset would make it."""
    global _n_resampled
    import soundfile as sf

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if int(file_sr) != sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(file_sr), sr)
        y = resample_poly(y, sr // g, int(file_sr) // g).astype(np.float32)
        _n_resampled += 1
    y = _trim_silence(y, sr)
    n = y.size
    W = window_samples
    if n == 0:
        return np.zeros(W, dtype=np.float32)
    if n < W:
        y = np.tile(y, -(-W // n))[:W]
    elif n > W:
        start = (n - W) // 2  # centre crop: the eval branch, never the random one
        y = y[start : start + W]
    return np.ascontiguousarray(y, dtype=np.float32)


def _extract_one(path: str):
    """Worker: path -> (path, feature vector or None on failure, n_resampled)."""
    from features.prosody_features import extract_prosody_features, features_to_vector

    try:
        window = load_eval_window(path)
        vec = features_to_vector(extract_prosody_features(window, sr=SR))
    except Exception as exc:  # unreadable/corrupt file: drop it, never guess features
        return path, None, str(exc)[:120]
    return path, vec.astype(np.float32), None


def _key(path: str | Path) -> str:
    """Canonical cache key. normcase because Windows manifests mix drive casing."""
    import os

    return os.path.normcase(os.path.normpath(str(path)))


class FeatureStore:
    """npz-backed ``{audio path -> 13-dim prosody vector}``, extracted in parallel.

    Paths that fail to load are remembered too, so a rerun does not retry a
    corrupt file 16 times in parallel just to fail again.
    """

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self._feats: dict[str, np.ndarray] = {}
        self._failed: dict[str, str] = {}
        if self.cache_path.exists():
            d = np.load(self.cache_path, allow_pickle=True)
            X = d["X"]
            for i, p in enumerate(d["paths"]):
                self._feats[str(p)] = X[i]
            if "failed_paths" in d:
                for p, why in zip(d["failed_paths"], d["failed_reasons"]):
                    self._failed[str(p)] = str(why)
            print(f"[store] {self.cache_path.name}: {len(self._feats)} cached "
                  f"({len(self._failed)} known-bad)")

    def __len__(self) -> int:
        return len(self._feats)

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        paths = list(self._feats)
        X = (np.stack([self._feats[p] for p in paths]).astype(np.float32)
             if paths else np.zeros((0, 13), dtype=np.float32))
        np.savez(
            self.cache_path,
            paths=np.array(paths, dtype=object),
            X=X,
            failed_paths=np.array(list(self._failed), dtype=object),
            failed_reasons=np.array(list(self._failed.values()), dtype=object),
        )
        print(f"[store] wrote {self.cache_path}  ({len(paths)} vectors)")

    def get_or_extract(self, paths, workers: int | None = None,
                       progress_every: int = 250, save: bool = True):
        """Return ``(X, ok)`` for ``paths``: X[i] is the vector, ok[i] False if unreadable.

        Only uncached paths are extracted. Rows for failures are NaN and flagged
        in ``ok`` — the caller drops them rather than scoring a zero vector, which
        would look like a confident prediction on a file we never read.
        """
        import os
        from concurrent.futures import ProcessPoolExecutor
        # BrokenProcessPool is not re-exported by the package root on any CPython
        # version — it lives in the process submodule.
        from concurrent.futures.process import BrokenProcessPool

        keys = [_key(p) for p in paths]
        todo = [k for k in dict.fromkeys(keys)
                if k not in self._feats and k not in self._failed]
        # Each worker imports librosa+numba+Praat (~400 MB resident), so the cap is
        # memory, not cores: 14 workers on a 16-core box breaks the pool outright.
        n_workers = workers or max(1, min(8, (os.cpu_count() or 4) - 2))
        if todo:
            print(f"[store] extracting {len(todo)} new clips on {n_workers} workers "
                  f"(~{len(todo) * 1.1 / n_workers / 60:.1f} min)", flush=True)
        done = 0
        while todo:
            try:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    for path, vec, err in pool.map(_extract_one, todo, chunksize=8):
                        if vec is None:
                            self._failed[path] = err or "unknown"
                        else:
                            self._feats[path] = vec
                        done += 1
                        if done % progress_every == 0:
                            print(f"[store]   {done} extracted", flush=True)
            except BrokenProcessPool:
                todo = [k for k in todo
                        if k not in self._feats and k not in self._failed]
                if n_workers == 1:
                    raise
                n_workers = max(1, n_workers // 2)
                print(f"[store] worker pool died (likely RAM) — retrying {len(todo)} "
                      f"clips on {n_workers} workers", flush=True)
                if save:
                    self.save()  # never throw away work already paid for
                continue
            break
        if todo:
            if self._failed:
                print(f"[store] {len(self._failed)} unreadable file(s); first: "
                      f"{next(iter(self._failed.items()))}")
            if save:
                self.save()

        dim = len(next(iter(self._feats.values()))) if self._feats else 13
        X = np.full((len(keys), dim), np.nan, dtype=np.float32)
        ok = np.zeros(len(keys), dtype=bool)
        for i, k in enumerate(keys):
            vec = self._feats.get(k)
            if vec is not None:
                X[i] = vec
                ok[i] = True
        return X, ok
