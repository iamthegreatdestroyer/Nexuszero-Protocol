"""Soundness verification utilities for predicted proof parameters.

The SoundnessVerifier validates predicted (normalized) parameters and computes
security-related metrics. It can also suggest adjustments to bring parameters
within acceptable cryptographic and performance constraints.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SoundnessResult:
    """Structured result of a soundness verification pass.

    Attributes:
        passed: Overall pass/fail heuristic
        security_score: Normalized (0-1) composite security score
        bit_security: Approximate bit security estimate
        hardness_score: Normalized hardness proxy (0-1)
        issues: Map of failing constraints -> description
        suggestions: Map of constraint -> suggested value
        denormalized: Original parameter values
    """
    passed: bool
    security_score: float
    bit_security: float
    hardness_score: float
    issues: Dict[str, str]
    suggestions: Dict[str, Any]
    denormalized: Dict[str, Any]


class SoundnessVerifier:
    """Verify predicted (normalized) lattice parameters for soundness.

    Validation Rules (heuristic placeholders – replace later):
    - n: power of two in [256, 4096]
    - q: q > 2*n and in [4096, 131072]
    - sigma: in [2.0, 5.0]
    - Security score grows with n, q; mild sigma penalty
    """

    def __init__(
        self,
        n_min: int = 256,
        n_max: int = 4096,
        q_min: int = 4096,
        q_max: int = 131072,
        sigma_min: float = 2.0,
        sigma_max: float = 5.0,
    ):
        self.n_min = n_min
        self.n_max = n_max
        self.q_min = q_min
        self.q_max = q_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    # --- Normalization helpers ---
    def denormalize(
        self,
        n_norm: float,
        q_norm: float,
        sigma_norm: float,
    ) -> Dict[str, Any]:
        n = int(round(n_norm * (self.n_max - self.n_min) + self.n_min))
        q = int(round(q_norm * (self.q_max - self.q_min) + self.q_min))
        sigma = sigma_norm * (self.sigma_max - self.sigma_min) + self.sigma_min
        return {"n": n, "q": q, "sigma": sigma}

    # --- Core Verification ---
    def is_power_of_two(self, value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    def compute_security_score(self, n: int, q: int, sigma: float) -> float:
        n_factor = (n - self.n_min) / (self.n_max - self.n_min)
        q_factor = (q - self.q_min) / (self.q_max - self.q_min)
        sigma_factor = 1.0 - (
            (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)
        )
        score = 0.6 * n_factor + 0.35 * q_factor + 0.05 * sigma_factor
        return max(0.0, min(score, 1.0))

    def compute_bit_security(self, n: int, q: int) -> float:
        # Very rough proxy: log2(q) * (n / n_max)
        import math
        raw = (math.log2(max(q, 2)) * (n / self.n_max))
        # Normalize to [0,1] assuming 256 bits target ceiling
        return max(0.0, min(raw / 256.0, 1.0))

    def compute_hardness(self, n: int, q: int, sigma: float) -> float:
        # Proxy hardness ~ n * log2(q) / (sigma^2 * constant)
        import math
        raw = n * math.log2(max(q, 2)) / (sigma * sigma * 1000.0)
        return max(0.0, min(raw, 1.0))

    def verify(self, params_norm: Dict[str, float]) -> SoundnessResult:
        """Verify normalized parameters and return structured result.

        Args:
            params_norm: Dict with normalized "n", "q", "sigma" in [0,1]
        """
        denorm = self.denormalize(
            params_norm["n"], params_norm["q"], params_norm["sigma"]
        )

        issues = {}
        suggestions = {}

        n = denorm["n"]
        q = denorm["q"]
        sigma = denorm["sigma"]

        # Validate n
        if not self.is_power_of_two(n):
            issues["n"] = "n must be a power of two"
            # Suggest closest power of two
            candidate = 1
            while candidate < n:
                candidate <<= 1
            lower = candidate >> 1
            # Choose closer of lower/upper
            if abs(lower - n) <= abs(candidate - n):
                suggestions["n"] = lower
            else:
                suggestions["n"] = candidate

        if n < self.n_min or n > self.n_max:
            issues["n_range"] = "n out of allowed range"
            suggestions["n_range"] = max(self.n_min, min(n, self.n_max))

        # Validate q
        if q < 2 * n:
            issues["q_small"] = "q should be > 2*n for security margin"
            suggestions["q_small"] = 2 * n + 1
        if q < self.q_min or q > self.q_max:
            issues["q_range"] = "q out of allowed range"
            suggestions["q_range"] = max(self.q_min, min(q, self.q_max))

        # sigma range
        if sigma < self.sigma_min or sigma > self.sigma_max:
            issues["sigma_range"] = "sigma out of allowed range"
            suggestions["sigma_range"] = max(
                self.sigma_min, min(sigma, self.sigma_max)
            )

        security_score = self.compute_security_score(n, q, sigma)
        bit_sec = self.compute_bit_security(n, q)
        hardness = self.compute_hardness(n, q, sigma)
        passed = (
            len(issues) == 0
            and security_score >= 0.5
            and bit_sec >= 0.3
            and hardness >= 0.2
        )
        return SoundnessResult(
            passed=passed,
            security_score=security_score,
            bit_security=bit_sec,
            hardness_score=hardness,
            issues=issues,
            suggestions=suggestions,
            denormalized=denorm,
        )

    # Convenience wrapper for model output tensor
    def verify_tensor(self, params_tensor) -> SoundnessResult:
        params_norm = {
            "n": float(params_tensor[0]),
            "q": float(params_tensor[1]),
            "sigma": float(params_tensor[2]),
        }
        return self.verify(params_norm)


__all__ = ["SoundnessVerifier", "SoundnessResult"]
