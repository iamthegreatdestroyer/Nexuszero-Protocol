"""Higher-level validation orchestrator.

Provides batch-oriented verification utilities that can be integrated inside
training/evaluation loops to compute aggregated security metrics.
"""

from typing import Dict, List
import torch
from .soundness import SoundnessVerifier


class BatchSoundnessValidator:
    def __init__(self, verifier: SoundnessVerifier):
        self.verifier = verifier

    def evaluate_batch(self, params_batch: torch.Tensor) -> Dict[str, float]:
        """Compute aggregate security statistics for a batch.

        Args:
            params_batch: Tensor [batch, 3] normalized
        Returns:
            Aggregated metrics.
        """
        scores: List[float] = []
        bit_scores: List[float] = []
        hardness_scores: List[float] = []
        passes = 0
        for row in params_batch:
            res = self.verifier.verify_tensor(row)
            scores.append(res.security_score)
            bit_scores.append(res.bit_security)
            hardness_scores.append(res.hardness_score)
            passes += int(res.passed)
        scores_t = torch.tensor(scores, dtype=torch.float)
        bit_t = torch.tensor(bit_scores, dtype=torch.float)
        hard_t = torch.tensor(hardness_scores, dtype=torch.float)
        return {
            "security_score_mean": float(scores_t.mean()),
            "security_score_min": float(scores_t.min()),
            "security_pass_rate": passes / len(scores) if scores else 0.0,
            "bit_security_mean": float(bit_t.mean()),
            "hardness_mean": float(hard_t.mean()),
        }

__all__ = ["BatchSoundnessValidator"]

