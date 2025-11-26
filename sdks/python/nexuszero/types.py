"""
Core types for NexusZero Protocol SDK.
"""

from datetime import datetime
from enum import IntEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChainId(IntEnum):
    """Supported blockchain networks."""

    ETHEREUM = 1
    POLYGON = 137
    ARBITRUM = 42161
    OPTIMISM = 10
    BASE = 8453
    SOLANA = 0  # Non-EVM, uses different addressing
    BITCOIN = 0  # Non-EVM, uses different addressing
    COSMOS = 0  # Non-EVM, uses different addressing


class TransactionStatus(IntEnum):
    """Transaction lifecycle states."""

    CREATED = 0
    PRIVACY_SELECTED = 1
    PROOF_GENERATING = 2
    PROOF_GENERATED = 3
    SUBMITTED = 4
    CONFIRMED = 5
    FAILED = 6


class ProofMetadata(BaseModel):
    """Metadata associated with a zero-knowledge proof."""

    privacy_level: int = Field(ge=0, le=5, description="Privacy level 0-5")
    proof_type: str = Field(description="Type of proof (e.g., 'lattice', 'bulletproof')")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sender_commitment: bytes = Field(description="32-byte commitment to sender")
    recipient_commitment: bytes = Field(description="32-byte commitment to recipient")
    generation_time_ms: Optional[int] = Field(default=None, description="Proof generation time")
    prover_node_id: Optional[str] = Field(default=None, description="ID of prover node")

    class Config:
        json_encoders = {bytes: lambda v: v.hex()}


class TransactionRequest(BaseModel):
    """Request to create a privacy-preserving transaction."""

    recipient: str = Field(description="Recipient address or public key")
    amount: int = Field(ge=0, description="Transaction amount in base units")
    privacy_level: int = Field(default=3, ge=0, le=5, description="Privacy level 0-5")
    chain: str = Field(default="ethereum", description="Target blockchain")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Additional metadata")


class Transaction(BaseModel):
    """A privacy-preserving transaction."""

    id: str = Field(description="Unique transaction ID (UUID)")
    sender_commitment: bytes = Field(description="Commitment to sender identity")
    recipient_commitment: bytes = Field(description="Commitment to recipient identity")
    amount_commitment: Optional[bytes] = Field(default=None, description="Encrypted amount")
    privacy_level: int = Field(ge=0, le=5)
    proof_id: Optional[str] = Field(default=None, description="Associated proof ID")
    chain: str = Field(description="Target blockchain")
    chain_tx_hash: Optional[str] = Field(default=None, description="On-chain transaction hash")
    status: TransactionStatus = Field(default=TransactionStatus.CREATED)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            bytes: lambda v: v.hex(),
            datetime: lambda v: v.isoformat(),
        }


class FeeEstimate(BaseModel):
    """Gas/fee estimate for a blockchain operation."""

    gas_units: int = Field(description="Estimated gas units")
    gas_price_gwei: float = Field(description="Gas price in gwei")
    total_fee_native: float = Field(description="Total fee in native currency")
    total_fee_usd: float = Field(description="Total fee in USD")


class ProofResult(BaseModel):
    """Result of proof generation."""

    proof_id: str = Field(description="Unique proof ID")
    proof_data: bytes = Field(description="The actual proof bytes")
    privacy_level: int = Field(ge=0, le=5)
    generation_time_ms: int = Field(description="Time to generate proof in milliseconds")
    quality_score: float = Field(ge=0.0, le=1.0, description="Proof quality score")
    verified: bool = Field(default=False, description="Whether proof was verified")

    class Config:
        json_encoders = {bytes: lambda v: v.hex()}
