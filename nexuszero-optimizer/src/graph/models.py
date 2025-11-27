"""
Graph Models for Cryptographic Primitive Knowledge Base
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class PrimitiveCategory(Enum):
    COMMITMENT = "commitment"
    ENCRYPTION = "encryption"
    SIGNATURE = "signature"
    HASH = "hash"
    ZK_PROOF = "zk_proof"
    MPC = "mpc"
    ARITHMETIC = "arithmetic"
    POLYNOMIAL = "polynomial"
    ELLIPTIC_CURVE = "elliptic_curve"
    LATTICE = "lattice"


class SecurityAssumption(Enum):
    DISCRETE_LOG = "discrete_log"
    CDH = "computational_diffie_hellman"
    DDH = "decisional_diffie_hellman"
    LWE = "learning_with_errors"
    RLWE = "ring_lwe"
    RANDOM_ORACLE = "random_oracle"
    AGM = "algebraic_group_model"


class ProofSystem(Enum):
    GROTH16 = "groth16"
    PLONK = "plonk"
    BULLETPROOFS = "bulletproofs"
    STARK = "stark"
    HALO2 = "halo2"
    NOVA = "nova"


class RelationshipType(Enum):
    DEPENDS_ON = "DEPENDS_ON"
    COMPOSED_OF = "COMPOSED_OF"
    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    OPTIMIZES = "OPTIMIZES"
    IMPLEMENTS = "IMPLEMENTS"
    COMPATIBLE_WITH = "COMPATIBLE_WITH"


@dataclass
class CryptographicPrimitive:
    primitive_id: str
    name: str
    category: PrimitiveCategory
    description: str
    security_assumptions: List[SecurityAssumption] = field(default_factory=list)
    security_level_bits: int = 128
    time_complexity: str = "O(n)"
    space_complexity: str = "O(n)"
    field_type: Optional[str] = None
    requires_trusted_setup: bool = False
    quantum_resistant: bool = False
    common_use_cases: List[str] = field(default_factory=list)
    compatible_proof_systems: List[ProofSystem] = field(default_factory=list)
    usage_count: int = 0
    success_rate: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "security_assumptions": [sa.value for sa in self.security_assumptions],
            "security_level_bits": self.security_level_bits,
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "field_type": self.field_type,
            "requires_trusted_setup": self.requires_trusted_setup,
            "quantum_resistant": self.quantum_resistant,
            "common_use_cases": self.common_use_cases,
            "compatible_proof_systems": [ps.value for ps in self.compatible_proof_systems],
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
        }


@dataclass
class PrimitiveRelationship:
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    weight: float = 1.0
    confidence: float = 1.0
    context: Optional[str] = None


@dataclass
class ProofConstructionPattern:
    pattern_id: str
    name: str
    description: str
    required_primitives: List[str] = field(default_factory=list)
    proof_system: ProofSystem = ProofSystem.GROTH16
    typical_constraint_count: int = 10000
    use_cases: List[str] = field(default_factory=list)


# Core primitives seeded for the graph KB
CORE_PRIMITIVES: List[CryptographicPrimitive] = [
    CryptographicPrimitive(
        primitive_id="pedersen_commitment",
        name="Pedersen Commitment",
        category=PrimitiveCategory.COMMITMENT,
        description="Computationally hiding and perfectly binding commitment scheme.",
        security_assumptions=[SecurityAssumption.DISCRETE_LOG],
        common_use_cases=["range proofs", "confidential transactions"],
        compatible_proof_systems=[ProofSystem.BULLETPROOFS, ProofSystem.GROTH16, ProofSystem.PLONK],
    ),
    CryptographicPrimitive(
        primitive_id="poseidon_hash",
        name="Poseidon Hash",
        category=PrimitiveCategory.HASH,
        description="ZK-friendly hash function optimized for arithmetic circuits.",
        security_assumptions=[SecurityAssumption.RANDOM_ORACLE],
        common_use_cases=["merkle trees", "nullifiers"],
        compatible_proof_systems=[ProofSystem.GROTH16, ProofSystem.PLONK, ProofSystem.HALO2],
    ),
    CryptographicPrimitive(
        primitive_id="ntt",
        name="Number Theoretic Transform",
        category=PrimitiveCategory.POLYNOMIAL,
        description="FFT over finite fields for polynomial multiplication.",
        time_complexity="O(n log n)",
        common_use_cases=["polynomial multiplication", "proof generation"],
        compatible_proof_systems=[ProofSystem.GROTH16, ProofSystem.PLONK, ProofSystem.STARK],
    ),
    CryptographicPrimitive(
        primitive_id="ring_lwe",
        name="Ring Learning With Errors",
        category=PrimitiveCategory.LATTICE,
        description="Lattice-based cryptographic assumption for post-quantum security.",
        security_assumptions=[SecurityAssumption.RLWE],
        quantum_resistant=True,
        common_use_cases=["post-quantum encryption", "FHE"],
    ),
    CryptographicPrimitive(
        primitive_id="kzg_commitment",
        name="KZG Polynomial Commitment",
        category=PrimitiveCategory.COMMITMENT,
        description="Constant-size polynomial commitment using pairings.",
        security_assumptions=[SecurityAssumption.AGM, SecurityAssumption.DDH],
        requires_trusted_setup=True,
        field_type="Pairing-friendly curves",
        common_use_cases=["polynomial commitments", "rollups"],
        compatible_proof_systems=[ProofSystem.PLONK, ProofSystem.HALO2],
    ),
    CryptographicPrimitive(
        primitive_id="bulletproofs_ipa",
        name="Bulletproofs Inner Product Argument",
        category=PrimitiveCategory.ZK_PROOF,
        description="Logarithmic-size range proofs without trusted setup.",
        security_assumptions=[SecurityAssumption.DISCRETE_LOG],
        requires_trusted_setup=False,
        common_use_cases=["range proofs", "confidential assets"],
        compatible_proof_systems=[ProofSystem.BULLETPROOFS],
    ),
    CryptographicPrimitive(
        primitive_id="msm",
        name="Multi-Scalar Multiplication",
        category=PrimitiveCategory.ELLIPTIC_CURVE,
        description="Efficient batch scalar multiplication on elliptic curves.",
        time_complexity="O(n / log n)",
        common_use_cases=["proof generation", "KZG evaluation"],
        compatible_proof_systems=[ProofSystem.GROTH16, ProofSystem.PLONK, ProofSystem.BULLETPROOFS],
    ),
    CryptographicPrimitive(
        primitive_id="fri",
        name="Fast Reed-Solomon IOP of Proximity",
        category=PrimitiveCategory.ZK_PROOF,
        description="Transparent polynomial commitment for STARKs.",
        security_assumptions=[SecurityAssumption.RANDOM_ORACLE],
        requires_trusted_setup=False,
        quantum_resistant=True,
        common_use_cases=["STARKs", "transparent proofs"],
        compatible_proof_systems=[ProofSystem.STARK],
    ),
]
