"""
Adaptive Privacy Morphing (APM) Engine for NexusZero Protocol.

Implements the 6-level privacy spectrum:
- Level 0: Transparent - Public blockchain parity
- Level 1: Pseudonymous - Address obfuscation
- Level 2: Confidential - Encrypted amounts
- Level 3: Private - Full transaction privacy
- Level 4: Anonymous - Unlinkable transactions
- Level 5: Sovereign - Maximum privacy, ZK everything
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class PrivacyLevel(IntEnum):
    """The 6-level privacy spectrum."""

    TRANSPARENT = 0  # Public blockchain parity
    PSEUDONYMOUS = 1  # Address obfuscation
    CONFIDENTIAL = 2  # Encrypted amounts
    PRIVATE = 3  # Full transaction privacy
    ANONYMOUS = 4  # Unlinkable transactions
    SOVEREIGN = 5  # Maximum privacy, ZK everything

    @property
    def description(self) -> str:
        """Human-readable description of the privacy level."""
        descriptions = {
            0: "Transparent: Public blockchain parity, no privacy enhancements",
            1: "Pseudonymous: Address obfuscation with basic decoys",
            2: "Confidential: Encrypted amounts, visible addresses",
            3: "Private: Full transaction privacy with ZK proofs",
            4: "Anonymous: Unlinkable transactions, large anonymity set",
            5: "Sovereign: Maximum privacy, quantum-resistant ZK proofs",
        }
        return descriptions[self.value]

    @property
    def security_bits(self) -> int:
        """Security level in bits."""
        security = {0: 0, 1: 80, 2: 128, 3: 192, 4: 256, 5: 256}
        return security[self.value]

    @property
    def estimated_proof_time_ms(self) -> int:
        """Estimated proof generation time in milliseconds."""
        times = {0: 0, 1: 50, 2: 100, 3: 250, 4: 500, 5: 1000}
        return times[self.value]


@dataclass
class PrivacyParameters:
    """Configuration parameters for a specific privacy level."""

    lattice_n: int  # Lattice dimension
    modulus_q: int  # Modulus
    sigma: float  # Error distribution standard deviation
    security_bits: int
    proof_strategy: str
    anonymity_set_size: Optional[int] = None
    decoy_count: Optional[int] = None


@dataclass
class TransactionContext:
    """Context for privacy level recommendation."""

    value_usd: float
    requires_compliance: bool = False
    preferred_level: Optional[int] = None
    risk_score: float = 0.0
    jurisdiction: str = "US"
    counterparty_known: bool = False


@dataclass
class PrivacyRecommendation:
    """Recommendation from the APM engine."""

    level: PrivacyLevel
    parameters: PrivacyParameters
    reasons: list[str] = field(default_factory=list)
    estimated_proof_time_ms: int = 0
    estimated_cost_gas: int = 0


class PrivacyEngine:
    """
    Adaptive Privacy Morphing (APM) Engine.

    Provides dynamic privacy level management and recommendations
    based on transaction context, regulatory requirements, and user preferences.
    """

    # Default parameters for each privacy level
    LEVEL_PARAMETERS: dict[int, PrivacyParameters] = {
        0: PrivacyParameters(
            lattice_n=0,
            modulus_q=0,
            sigma=0.0,
            security_bits=0,
            proof_strategy="none",
        ),
        1: PrivacyParameters(
            lattice_n=256,
            modulus_q=12289,
            sigma=3.2,
            security_bits=80,
            proof_strategy="bulletproofs",
            decoy_count=3,
        ),
        2: PrivacyParameters(
            lattice_n=512,
            modulus_q=12289,
            sigma=3.2,
            security_bits=128,
            proof_strategy="bulletproofs",
            decoy_count=7,
        ),
        3: PrivacyParameters(
            lattice_n=1024,
            modulus_q=40961,
            sigma=3.2,
            security_bits=192,
            proof_strategy="quantum_lattice_pkc",
            anonymity_set_size=16,
            decoy_count=15,
        ),
        4: PrivacyParameters(
            lattice_n=2048,
            modulus_q=65537,
            sigma=3.2,
            security_bits=256,
            proof_strategy="quantum_lattice_pkc",
            anonymity_set_size=64,
            decoy_count=31,
        ),
        5: PrivacyParameters(
            lattice_n=4096,
            modulus_q=786433,
            sigma=3.2,
            security_bits=256,
            proof_strategy="hybrid_zksnark_lattice",
            anonymity_set_size=256,
            decoy_count=63,
        ),
    }

    GAS_COSTS: dict[int, int] = {
        0: 21000,
        1: 50000,
        2: 100000,
        3: 200000,
        4: 350000,
        5: 500000,
    }

    def __init__(self) -> None:
        """Initialize the APM engine with default parameters."""
        self.parameters = self.LEVEL_PARAMETERS.copy()

    def get_parameters(self, level: int) -> PrivacyParameters:
        """Get parameters for a specific privacy level."""
        if level not in self.parameters:
            raise ValueError(f"Invalid privacy level: {level}")
        return self.parameters[level]

    def recommend(self, context: TransactionContext) -> PrivacyRecommendation:
        """
        Recommend optimal privacy level based on transaction context.

        Args:
            context: Transaction context including value, compliance needs, etc.

        Returns:
            PrivacyRecommendation with level, parameters, and reasoning.
        """
        recommended_level = 3  # Default to Private
        reasons: list[str] = []
        compliance_cap: int | None = None

        # Regulatory considerations - track as cap to enforce at end
        if context.requires_compliance:
            compliance_cap = 3
            reasons.append("Regulatory compliance limits maximum privacy to Level 3")

        # Transaction value considerations
        if context.value_usd > 100_000.0:
            recommended_level = 5
            reasons.append("Very high-value transaction ($100k+) warrants Sovereign privacy")
        elif context.value_usd > 10_000.0:
            recommended_level = max(recommended_level, 4)
            reasons.append("High-value transaction ($10k+) benefits from Anonymous privacy")

        # User preference
        if context.preferred_level is not None:
            recommended_level = context.preferred_level
            reasons.append(f"User preference: Level {context.preferred_level}")

        # Enforce compliance cap after all other adjustments
        if compliance_cap is not None and recommended_level > compliance_cap:
            recommended_level = compliance_cap

        # Risk score adjustment
        if context.risk_score > 0.7:
            recommended_level = min(recommended_level, 2)
            reasons.append("Elevated risk score (>0.7) reduces maximum privacy")

        # Counterparty considerations
        if context.counterparty_known:
            reasons.append("Known counterparty may allow reduced privacy overhead")

        level = PrivacyLevel(recommended_level)
        params = self.get_parameters(recommended_level)

        return PrivacyRecommendation(
            level=level,
            parameters=params,
            reasons=reasons,
            estimated_proof_time_ms=level.estimated_proof_time_ms,
            estimated_cost_gas=self.GAS_COSTS.get(recommended_level, 0),
        )

    def can_morph(self, from_level: int, to_level: int) -> tuple[bool, str]:
        """
        Check if morphing between privacy levels is possible.

        Args:
            from_level: Current privacy level
            to_level: Target privacy level

        Returns:
            Tuple of (can_morph, reason)
        """
        if from_level < 0 or from_level > 5:
            return False, f"Invalid source level: {from_level}"
        if to_level < 0 or to_level > 5:
            return False, f"Invalid target level: {to_level}"

        # Increasing privacy is always allowed (new proof)
        if to_level >= from_level:
            return True, "Increasing or maintaining privacy level is always allowed"

        # Decreasing privacy requires incremental steps
        return True, "Decreasing privacy must be done incrementally to preserve guarantees"

    def get_morph_path(self, from_level: int, to_level: int) -> list[int]:
        """
        Get the morphing path between privacy levels.

        Args:
            from_level: Current privacy level
            to_level: Target privacy level

        Returns:
            List of levels to transition through
        """
        if to_level > from_level:
            # Increasing privacy - can do in one step
            return [to_level]
        else:
            # Decreasing privacy - must do incrementally
            return list(range(from_level - 1, to_level - 1, -1))
