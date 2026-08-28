from __future__ import annotations

import numpy as np


class RingBuffer:
    """Bounded overlapping-window buffer.

    Keeps at most ``window_samples + last_chunk`` floats, so memory cannot
    grow with session length. Each full window is copied out; the buffer then
    drops ``hop_samples`` from the front.
    """

    def __init__(self, window_samples: int, hop_samples: int):
        if window_samples <= 0 or hop_samples <= 0:
            raise ValueError("window_samples and hop_samples must be positive")
        if hop_samples > window_samples:
            raise ValueError("hop_samples cannot exceed window_samples")
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self._buf = np.empty(0, dtype=np.float32)

    def __len__(self) -> int:
        return int(self._buf.size)

    def reset(self) -> None:
        self._buf = np.empty(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size:
            self._buf = np.concatenate((self._buf, chunk))
        windows: list[np.ndarray] = []
        while self._buf.size >= self.window_samples:
            windows.append(self._buf[: self.window_samples].copy())
            self._buf = self._buf[self.hop_samples :]
        return windows
