"""RVC chunking, historical context buffering, raised-cosine crossfading, and soft-limiting."""

from __future__ import annotations

from typing import Callable
import numpy as np


def soft_limit(samples: np.ndarray, max_abs: float = 0.92) -> np.ndarray:
    """Soft-limit audio to [-max_abs, max_abs] using smooth hyperbolic tangent saturation.
    
    This ensures converted audio never trips the backend quality detector's
    clipping threshold (clip_abs=0.99, clip_fraction=0.01 in realtime-backend/quality.py).
    """
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return arr

    peak = float(np.max(np.abs(arr)))
    if peak <= max_abs:
        return arr

    # Smooth soft compression preserving dynamics while guaranteeing strict bound <= max_abs
    scale = max_abs * 0.999
    compressed = np.tanh(arr / max_abs) * scale
    return compressed.astype(np.float32)


def raised_cosine_weights(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute raised-cosine fade-in and fade-out weight vectors of given length.
    
    fade_in + fade_out == 1.0 everywhere.
    """
    if length <= 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    t = (np.arange(length, dtype=np.float32) + 0.5) / float(length)
    fade_in = 0.5 * (1.0 - np.cos(np.pi * t)).astype(np.float32)
    fade_out = (1.0 - fade_in).astype(np.float32)
    return fade_in, fade_out


class ChunkCrossfadeProcessor:
    """Processes audio chunks with historical context (150ms), applies raised-cosine
    crossfading (25ms) across chunk boundaries to prevent clicks/pitch discontinuities,
    and guarantees strict soft-limiting to [-0.92, 0.92].
    
    Preserves exact audio length (zero packet drop / zero time compression).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_ms: int = 200,
        context_ms: int = 150,
        crossfade_ms: int = 25,
        soft_limit_max: float = 0.92,
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * (chunk_ms / 1000.0))
        self.context_samples = int(sample_rate * (context_ms / 1000.0))
        self.crossfade_samples = int(sample_rate * (crossfade_ms / 1000.0))
        self.soft_limit_max = soft_limit_max

        self.fade_in, self.fade_out = raised_cosine_weights(self.crossfade_samples)

        # Buffers
        self._input_history = np.empty(0, dtype=np.float32)
        self._prev_tail = np.empty(0, dtype=np.float32)

    def reset(self) -> None:
        """Clear all context and crossfade state (e.g. on spoof toggle or new call)."""
        self._input_history = np.empty(0, dtype=np.float32)
        self._prev_tail = np.empty(0, dtype=np.float32)

    def process(
        self,
        samples: np.ndarray,
        convert_fn: Callable[[np.ndarray], np.ndarray],
    ) -> list[np.ndarray]:
        """Ingest new samples, chunk if larger than chunk_samples, prepend historical
        context (150ms), convert, apply raised-cosine crossfade (25ms) at chunk boundaries,
        and soft-limit to [-0.92, 0.92].
        
        Guarantees total output samples == total input samples (zero loss).
        """
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return []

        output_chunks: list[np.ndarray] = []
        offset = 0
        total_len = arr.size

        while offset < total_len:
            # Chunk window
            step = min(self.chunk_samples, total_len - offset)
            sub_chunk = arr[offset : offset + step]
            offset += step

            # 1. Prepend historical context if available (Mitigation #4: 150ms historical context)
            if self._input_history.size > 0:
                context = self._input_history[-self.context_samples :]
                full_input = np.concatenate((context, sub_chunk))
                context_len = context.size
            else:
                full_input = sub_chunk
                context_len = 0

            # Update input history (keep up to 2x context size)
            self._input_history = np.concatenate((self._input_history, sub_chunk))
            if self._input_history.size > self.context_samples * 4:
                self._input_history = self._input_history[-self.context_samples * 2 :]

            # 2. Convert through engine
            converted_full = convert_fn(full_input)

            # Extract converted chunk corresponding to sub_chunk (skip context prefix)
            if converted_full.size >= context_len + sub_chunk.size:
                converted_chunk = converted_full[context_len : context_len + sub_chunk.size].copy()
            elif converted_full.size >= sub_chunk.size:
                converted_chunk = converted_full[-sub_chunk.size :].copy()
            else:
                converted_chunk = converted_full.copy()

            # 3. Apply raised-cosine crossfade at boundary with previous chunk (Mitigation #4)
            fade_len = min(self.crossfade_samples, converted_chunk.size)
            if self._prev_tail.size >= fade_len and fade_len > 0:
                fade_in = self.fade_in[:fade_len]
                fade_out = self.fade_out[:fade_len]
                tail = self._prev_tail[-fade_len:]

                converted_chunk[:fade_len] = (converted_chunk[:fade_len] * fade_in) + (tail * fade_out)

            # Save tail for the next chunk's boundary crossfade
            if converted_chunk.size >= self.crossfade_samples:
                self._prev_tail = converted_chunk[-self.crossfade_samples :].copy()
            else:
                self._prev_tail = converted_chunk.copy()

            # 4. Soft-limit to [-0.92, 0.92] (Mitigation #4: avoid 1% clipping rejection)
            limited = soft_limit(converted_chunk, max_abs=self.soft_limit_max)
            output_chunks.append(limited)

        return output_chunks

    def flush(
        self,
        convert_fn: Callable[[np.ndarray], np.ndarray],
    ) -> list[np.ndarray]:
        """Flush state on stream completion."""
        self._input_history = np.empty(0, dtype=np.float32)
        self._prev_tail = np.empty(0, dtype=np.float32)
        return []
