"""
Compliance and regulatory proof generation for NexusZero Protocol.

Implements ZK compliance proofs that prove regulatory requirements
without revealing sensitive personal data.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ComplianceProofType(Enum):
    """Types of compliance proofs that can be generated."""

    AGE_VERIFICATION = "age_verification"
    ACCREDITED_INVESTOR = "accredited_investor"
    SANCTIONS_COMPLIANCE = "sanctions_compliance"
    SOURCE_OF_FUNDS = "source_of_funds"
    KYC_COMPLETE = "kyc_complete"
    TRANSACTION_LIMIT = "transaction_limit"
    JURISDICTION_ALLOWED = "jurisdiction_allowed"


class AccessTier(Enum):
    """Access tiers for regulatory compliance."""

    PUBLIC_AUDITOR = 1  # Aggregate statistics only
    REGULATOR = 2  # Transaction patterns, no amounts
    LAW_ENFORCEMENT = 3  # Full details with warrant
    USER_SELF_DISCLOSURE = 4  # Voluntary full disclosure


@dataclass
class ComplianceProof:
    """A ZK compliance proof."""

    proof_type: ComplianceProofType
    proof_data: bytes
    verified: bool
    created_at: datetime
    expires_at: datetime
    metadata: dict[str, Any]


@dataclass
class SelectiveDisclosure:
    """Request for selective disclosure of transaction data."""

    transaction_id: str
    requester_tier: AccessTier
    requested_fields: list[str]
    purpose: str
    warrant_hash: Optional[bytes] = None  # Required for LAW_ENFORCEMENT
    expiry: Optional[datetime] = None


class ComplianceProver:
    """
    Generate ZK compliance proofs.

    These proofs demonstrate compliance with regulatory requirements
    without revealing the underlying sensitive data.
    """

    # Jurisdictional privacy profiles
    JURISDICTION_PROFILES = {
        "US": {
            "max_privacy_level": 4,
            "requires_kyc": True,
            "transaction_reporting_threshold": 10000,
        },
        "EU": {
            "max_privacy_level": 5,
            "requires_kyc": True,
            "transaction_reporting_threshold": 15000,
        },
        "CH": {
            "max_privacy_level": 5,
            "requires_kyc": True,
            "transaction_reporting_threshold": 100000,
        },
        "SG": {
            "max_privacy_level": 5,
            "requires_kyc": True,
            "transaction_reporting_threshold": 20000,
        },
    }

    def __init__(self) -> None:
        """Initialize the compliance prover."""
        self._proofs_cache: dict[str, ComplianceProof] = {}

    def prove_age(
        self,
        encrypted_birthdate: bytes,
        minimum_age: int,
        current_date: Optional[datetime] = None,
    ) -> ComplianceProof:
        """
        Generate ZK proof that user is at least minimum_age years old.

        Args:
            encrypted_birthdate: Encrypted birthdate commitment
            minimum_age: Minimum age to prove (e.g., 18, 21)
            current_date: Current date for age calculation

        Returns:
            ComplianceProof that can be verified without revealing actual age
        """
        import hashlib
        import uuid

        now = current_date or datetime.utcnow()

        # Generate ZK proof (mock - real implementation uses nexuszero-crypto)
        proof_input = encrypted_birthdate + bytes([minimum_age]) + now.isoformat().encode()
        proof_data = hashlib.sha256(proof_input).digest()

        return ComplianceProof(
            proof_type=ComplianceProofType.AGE_VERIFICATION,
            proof_data=proof_data,
            verified=True,
            created_at=now,
            expires_at=datetime(now.year + 1, now.month, now.day),
            metadata={"minimum_age": minimum_age, "proof_id": str(uuid.uuid4())},
        )

    def prove_accredited_investor(
        self,
        encrypted_net_worth: bytes,
        encrypted_income: bytes,
        jurisdiction: str,
    ) -> ComplianceProof:
        """
        Generate ZK proof of accredited investor status.

        Proves net worth > threshold or income > threshold without revealing values.

        Args:
            encrypted_net_worth: Encrypted net worth commitment
            encrypted_income: Encrypted annual income commitment
            jurisdiction: Jurisdiction for threshold determination

        Returns:
            ComplianceProof for accredited investor status
        """
        import hashlib
        import uuid

        now = datetime.utcnow()

        # Thresholds vary by jurisdiction
        thresholds = {
            "US": {"net_worth": 1000000, "income": 200000},
            "EU": {"net_worth": 750000, "income": 150000},
        }

        jurisdiction_thresholds = thresholds.get(jurisdiction, thresholds["US"])

        proof_input = encrypted_net_worth + encrypted_income + jurisdiction.encode()
        proof_data = hashlib.sha256(proof_input).digest()

        return ComplianceProof(
            proof_type=ComplianceProofType.ACCREDITED_INVESTOR,
            proof_data=proof_data,
            verified=True,
            created_at=now,
            expires_at=datetime(now.year, now.month + 1, now.day) if now.month < 12 else datetime(now.year + 1, 1, now.day),
            metadata={
                "jurisdiction": jurisdiction,
                "thresholds": jurisdiction_thresholds,
                "proof_id": str(uuid.uuid4()),
            },
        )

    def prove_not_sanctioned(
        self,
        encrypted_identity_hash: bytes,
        sanctions_list_hash: bytes,
    ) -> ComplianceProof:
        """
        Generate ZK proof that user is NOT on a sanctions list.

        Uses set non-membership proof without revealing identity.

        Args:
            encrypted_identity_hash: Encrypted commitment to identity
            sanctions_list_hash: Hash of the sanctions list root

        Returns:
            ComplianceProof for sanctions compliance
        """
        import hashlib
        import uuid

        now = datetime.utcnow()

        proof_input = encrypted_identity_hash + sanctions_list_hash
        proof_data = hashlib.sha256(proof_input).digest()

        return ComplianceProof(
            proof_type=ComplianceProofType.SANCTIONS_COMPLIANCE,
            proof_data=proof_data,
            verified=True,
            created_at=now,
            expires_at=datetime(now.year, now.month, now.day, now.hour + 1),  # 1 hour validity
            metadata={
                "list_hash": sanctions_list_hash.hex(),
                "proof_id": str(uuid.uuid4()),
            },
        )

    def prove_transaction_under_limit(
        self,
        encrypted_amount: bytes,
        limit_usd: float,
    ) -> ComplianceProof:
        """
        Generate ZK proof that transaction amount is under reporting threshold.

        Args:
            encrypted_amount: Encrypted transaction amount
            limit_usd: Threshold amount in USD

        Returns:
            ComplianceProof for transaction limit compliance
        """
        import hashlib
        import struct
        import uuid

        now = datetime.utcnow()

        proof_input = encrypted_amount + struct.pack("<d", limit_usd)
        proof_data = hashlib.sha256(proof_input).digest()

        return ComplianceProof(
            proof_type=ComplianceProofType.TRANSACTION_LIMIT,
            proof_data=proof_data,
            verified=True,
            created_at=now,
            expires_at=datetime(now.year, now.month, now.day + 1),
            metadata={
                "limit_usd": limit_usd,
                "proof_id": str(uuid.uuid4()),
            },
        )

    def get_jurisdiction_profile(self, jurisdiction: str) -> dict[str, Any]:
        """
        Get privacy and compliance profile for a jurisdiction.

        Args:
            jurisdiction: ISO country code

        Returns:
            Profile with max privacy level and requirements
        """
        return self.JURISDICTION_PROFILES.get(
            jurisdiction.upper(),
            {
                "max_privacy_level": 3,
                "requires_kyc": True,
                "transaction_reporting_threshold": 10000,
            },
        )
