"""Training and evaluation metrics utilities."""

from typing import Dict
import torch
import torch.nn.functional as F


def parameter_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def metrics_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def security_penalty(security_score_mean: float) -> float:
    # Encourage higher security scores by penalizing low averages
    return max(0.0, 0.5 - security_score_mean)


class MetricTracker:
    def __init__(self):
        self.storage: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, key: str, value: float):
        self.storage[key] = self.storage.get(key, 0.0) + value
        self.counts[key] = self.counts.get(key, 0) + 1

    def average(self, key: str) -> float:
        if self.counts.get(key, 0) == 0:
            return 0.0
        return self.storage[key] / self.counts[key]

    def to_dict(self) -> Dict[str, float]:
        return {k: self.average(k) for k in self.storage.keys()}

__all__ = [
    "parameter_mse",
    "metrics_mse",
    "security_penalty",
    "MetricTracker",
]
