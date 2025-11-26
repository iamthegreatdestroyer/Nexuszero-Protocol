"""
NexusZero Protocol Python SDK

High-level Python interface for:
- Privacy-preserving transactions
- Zero-knowledge proof generation
- Cross-chain operations
- Compliance proof generation
- Adaptive Privacy Morphing (APM)

Example:
    >>> from nexuszero import NexusZeroClient, PrivacyLevel
    >>> async with NexusZeroClient() as client:
    ...     tx = await client.create_transaction(
    ...         recipient="0x...",
    ...         amount=1000,
    ...         privacy_level=PrivacyLevel.PRIVATE,
    ...     )
    ...     print(f"Transaction ID: {tx.id}")
"""

from .client import NexusZeroClient
from .privacy import PrivacyLevel, PrivacyEngine, PrivacyRecommendation
from .proof import ProofGenerator, ProofVerifier, ProofResult
from .bridge import CrossChainBridge, BridgeTransfer
from .compliance import ComplianceProver, ComplianceProofType
from .types import (
    Transaction,
    TransactionRequest,
    TransactionStatus,
    ChainId,
    ProofMetadata,
)
from .errors import (
    NexusZeroError,
    AuthenticationError,
    ProofGenerationError,
    VerificationError,
    BridgeError,
    ComplianceError,
)

__version__ = "0.1.0"
__all__ = [
    # Client
    "NexusZeroClient",
    # Privacy
    "PrivacyLevel",
    "PrivacyEngine",
    "PrivacyRecommendation",
    # Proofs
    "ProofGenerator",
    "ProofVerifier",
    "ProofResult",
    # Bridge
    "CrossChainBridge",
    "BridgeTransfer",
    # Compliance
    "ComplianceProver",
    "ComplianceProofType",
    # Types
    "Transaction",
    "TransactionRequest",
    "TransactionStatus",
    "ChainId",
    "ProofMetadata",
    # Errors
    "NexusZeroError",
    "AuthenticationError",
    "ProofGenerationError",
    "VerificationError",
    "BridgeError",
    "ComplianceError",
]
