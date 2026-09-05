import numpy as np

def aggregate_logits(logits: list[float], method: str = "mean") -> float:
    """
    Aggregates a list of logits.
    method: "mean" or "median"
    """
    if not logits:
        raise ValueError("Cannot aggregate empty list of logits")
    
    if method == "mean":
        return float(np.mean(logits))
    elif method == "median":
        return float(np.median(logits))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

class LogitEMA:
    """
    Exponential Moving Average tracker for logit space.
    """
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def update(self, logit: float) -> float:
        if self.value is None:
            self.value = logit
        else:
            self.value = self.alpha * logit + (1.0 - self.alpha) * self.value
        return self.value
