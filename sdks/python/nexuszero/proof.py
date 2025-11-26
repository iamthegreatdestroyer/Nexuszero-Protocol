"""
Proof generation and verification for NexusZero Protocol.
"""

from dataclasses import dataclass
from typing import Optional

from .errors import ProofGenerationError, VerificationError
from .privacy import PrivacyLevel, PrivacyParameters


@dataclass
class ProofResult:
    """Result of proof generation."""

    proof_id: str
    proof_data: bytes
    privacy_level: int
    generation_time_ms: int
    quality_score: float
    verified: bool = False


class ProofGenerator:
    """
    Local proof generation using NexusZero crypto primitives.

    For production use, proofs are typically generated via the API
    using the Distributed Proof Generation Network (DPGN).
    This class provides local generation for testing and offline use.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        """
        Initialize the proof generator.

        Args:
            use_gpu: Whether to attempt GPU acceleration
        """
        self.use_gpu = use_gpu
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure the generator is initialized."""
        if not self._initialized:
            # In a real implementation, this would load native libraries
            self._initialized = True

    def generate(
        self,
        circuit_data: bytes,
        privacy_level: PrivacyLevel,
        parameters: Optional[PrivacyParameters] = None,
    ) -> ProofResult:
        """
        Generate a zero-knowledge proof locally.

        Args:
            circuit_data: Input data for the proof circuit
            privacy_level: Determines proof strength and parameters
            parameters: Optional custom parameters (uses defaults if None)

        Returns:
            ProofResult with proof bytes and metadata

        Raises:
            ProofGenerationError: If proof generation fails
        """
        self._ensure_initialized()

        import hashlib
        import time
        import uuid

        start_time = time.time()

        try:
            # Simulate proof generation based on privacy level
            # In production, this would call nexuszero-crypto FFI
            proof_size = 32 * (privacy_level.value + 1)
            proof_hash = hashlib.sha256(circuit_data + bytes([privacy_level.value]))

            # Simulate generation time based on level
            level_time_ms = privacy_level.estimated_proof_time_ms
            # Add some variance
            import random

            actual_time_ms = level_time_ms + random.randint(-10, 50)

            # Generate mock proof (in production, real ZK proof)
            proof_data = proof_hash.digest() * (proof_size // 32 + 1)
            proof_data = proof_data[:proof_size]

            generation_time_ms = int((time.time() - start_time) * 1000)

            return ProofResult(
                proof_id=str(uuid.uuid4()),
                proof_data=proof_data,
                privacy_level=privacy_level.value,
                generation_time_ms=max(generation_time_ms, actual_time_ms),
                quality_score=self._calculate_quality(proof_data, privacy_level),
                verified=False,
            )
        except Exception as e:
            raise ProofGenerationError(f"Local proof generation failed: {e}")

    def _calculate_quality(self, proof: bytes, level: PrivacyLevel) -> float:
        """Calculate quality score based on proof size efficiency."""
        expected_size = 32 * (level.value + 1)
        actual_size = len(proof)

        if actual_size <= expected_size:
            return 1.0
        return min(1.0, expected_size / actual_size)


class ProofVerifier:
    """
    Proof verification for NexusZero Protocol.

    Supports both local verification and on-chain verification status checks.
    """

    def __init__(self) -> None:
        """Initialize the proof verifier."""
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure the verifier is initialized."""
        if not self._initialized:
            self._initialized = True

    def verify(
        self,
        proof: bytes,
        public_inputs: Optional[bytes] = None,
        privacy_level: Optional[PrivacyLevel] = None,
    ) -> bool:
        """
        Verify a zero-knowledge proof locally.

        Args:
            proof: The proof bytes to verify
            public_inputs: Optional public inputs for the proof
            privacy_level: Expected privacy level (for parameter selection)

        Returns:
            True if the proof is valid, False otherwise

        Raises:
            VerificationError: If verification process fails
        """
        self._ensure_initialized()

        try:
            # Basic structure validation
            if len(proof) < 32:
                return False

            # Check proof format based on privacy level
            if privacy_level is not None:
                expected_min_size = 32 * (privacy_level.value + 1)
                if len(proof) < expected_min_size:
                    return False

            # In production, this would perform actual cryptographic verification
            # using nexuszero-crypto FFI bindings
            return True

        except Exception as e:
            raise VerificationError(f"Proof verification failed: {e}")

    def verify_batch(self, proofs: list[bytes]) -> list[bool]:
        """
        Verify multiple proofs in batch.

        Args:
            proofs: List of proof bytes to verify

        Returns:
            List of boolean verification results
        """
        return [self.verify(proof) for proof in proofs]
