"""Exponential moving average over calibrated probabilities."""

from __future__ import annotations


class ExponentialMovingAverage:
    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.value: float | None = None

    def reset(self) -> None:
        self.value = None

    def update(self, probability: float) -> float:
        p = float(probability)
        if self.value is None:
            self.value = p
        else:
            self.value = self.alpha * p + (1.0 - self.alpha) * self.value
        return self.value
