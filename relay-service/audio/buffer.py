"""Audio buffering and bounded drop-oldest queuing to prevent latency drift."""

from __future__ import annotations

import asyncio
from typing import Any, Generic, TypeVar
import numpy as np

T = TypeVar("T")


class DropOldestQueue(Generic[T]):
    """An asyncio Queue with bounded capacity that drops the OLDEST item
    when full instead of blocking or growing unboundedly.
    
    This ensures latency never drifts even if network throughput momentarily dips.
    """

    def __init__(self, maxsize: int = 3):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self.dropped_count: int = 0

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()

    def put_nowait(self, item: T) -> bool:
        """Enqueues an item immediately. If full, drops the oldest item first.
        Returns True if an item was dropped, False otherwise.
        """
        dropped = False
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.dropped_count += 1
                dropped = True
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            # Fallback if racing
            self._queue.get_nowait()
            self._queue.put_nowait(item)
            self.dropped_count += 1
            dropped = True
        return dropped

    async def put(self, item: T) -> bool:
        """Async variant of put_nowait."""
        return self.put_nowait(item)

    async def get(self) -> T:
        """Retrieves next item from the queue."""
        return await self._queue.get()

    def get_nowait(self) -> T:
        """Retrieves next item immediately or raises QueueEmpty."""
        return self._queue.get_nowait()

    def clear(self) -> int:
        """Clears all items and returns count of cleared items."""
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        return count


class StreamBuffer:
    """Buffers arbitrary-length audio samples and chunks them into uniform window slices."""

    def __init__(self, chunk_samples: int):
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        self.chunk_samples = chunk_samples
        self._buf = np.empty(0, dtype=np.float32)

    def reset(self) -> None:
        self._buf = np.empty(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size:
            self._buf = np.concatenate((self._buf, arr))
        chunks: list[np.ndarray] = []
        while self._buf.size >= self.chunk_samples:
            chunks.append(self._buf[: self.chunk_samples].copy())
            self._buf = self._buf[self.chunk_samples :]
        return chunks

    def flush(self) -> list[np.ndarray]:
        if self._buf.size == 0:
            return []
        chunks = [self._buf.copy()]
        self._buf = np.empty(0, dtype=np.float32)
        return chunks
