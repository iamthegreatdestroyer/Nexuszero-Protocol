"""
Main NexusZero Protocol client.
"""

from typing import Any, Optional

import httpx

from .errors import (
    AuthenticationError,
    BridgeError,
    NetworkError,
    NexusZeroError,
    ProofGenerationError,
    RateLimitError,
    VerificationError,
)
from .privacy import PrivacyEngine, PrivacyLevel, PrivacyRecommendation, TransactionContext
from .types import FeeEstimate, ProofResult, Transaction, TransactionRequest


class NexusZeroClient:
    """
    Main client for NexusZero Protocol operations.

    Provides a high-level async interface for:
    - Creating privacy-preserving transactions
    - Generating and verifying ZK proofs
    - Cross-chain bridge operations
    - Privacy level management

    Example:
        >>> async with NexusZeroClient(api_key="your-key") as client:
        ...     tx = await client.create_transaction(
        ...         recipient="0x...",
        ...         amount=1000,
        ...         privacy_level=PrivacyLevel.PRIVATE,
        ...     )
    """

    DEFAULT_API_URL = "https://api.nexuszero.io"
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize the NexusZero client.

        Args:
            api_url: Base URL for the NexusZero API. Defaults to production.
            api_key: API key for authentication. Required for most operations.
            timeout: Request timeout in seconds.
        """
        self.api_url = api_url or self.DEFAULT_API_URL
        self.api_key = api_key
        self.timeout = timeout
        self._privacy_engine = PrivacyEngine()

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.api_url,
            headers=headers,
            timeout=timeout,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "NexusZeroClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle API response and raise appropriate errors."""
        if response.status_code == 401:
            raise AuthenticationError("Invalid or expired API key")
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                "Rate limit exceeded",
                retry_after=int(retry_after) if retry_after else None,
            )
        if response.status_code >= 400:
            try:
                error_data = response.json()
                raise NexusZeroError(
                    error_data.get("message", "Unknown error"),
                    code=error_data.get("code"),
                    details=error_data.get("details"),
                )
            except (ValueError, KeyError):
                raise NexusZeroError(f"HTTP {response.status_code}: {response.text}")

        return response.json()

    # ========== Transaction Operations ==========

    async def create_transaction(
        self,
        recipient: str,
        amount: int,
        privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE,
        chain: str = "ethereum",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Transaction:
        """
        Create a privacy-preserving transaction.

        Args:
            recipient: Recipient address or public key
            amount: Transaction amount in base units (wei, lamports, etc.)
            privacy_level: Desired privacy level (0-5)
            chain: Target blockchain (ethereum, polygon, solana, etc.)
            metadata: Optional additional metadata

        Returns:
            Transaction object with ID and status
        """
        request = TransactionRequest(
            recipient=recipient,
            amount=amount,
            privacy_level=privacy_level.value,
            chain=chain,
            metadata=metadata,
        )

        try:
            response = await self._client.post(
                "/api/v1/transactions",
                json=request.model_dump(),
            )
            data = self._handle_response(response)
            return Transaction(**data)
        except httpx.RequestError as e:
            raise NetworkError(f"Failed to create transaction: {e}")

    async def get_transaction(self, transaction_id: str) -> Transaction:
        """
        Get transaction details by ID.

        Args:
            transaction_id: UUID of the transaction

        Returns:
            Transaction object with current status
        """
        try:
            response = await self._client.get(f"/api/v1/transactions/{transaction_id}")
            data = self._handle_response(response)
            return Transaction(**data)
        except httpx.RequestError as e:
            raise NetworkError(f"Failed to get transaction: {e}")

    async def list_transactions(
        self,
        limit: int = 50,
        offset: int = 0,
        chain: Optional[str] = None,
    ) -> list[Transaction]:
        """
        List transactions for the authenticated user.

        Args:
            limit: Maximum number of transactions to return
            offset: Offset for pagination
            chain: Optional filter by blockchain

        Returns:
            List of Transaction objects
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if chain:
            params["chain"] = chain

        try:
            response = await self._client.get("/api/v1/transactions", params=params)
            data = self._handle_response(response)
            return [Transaction(**tx) for tx in data.get("transactions", [])]
        except httpx.RequestError as e:
            raise NetworkError(f"Failed to list transactions: {e}")

    # ========== Proof Operations ==========

    async def generate_proof(
        self,
        data: bytes,
        privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE,
    ) -> ProofResult:
        """
        Generate a zero-knowledge proof.

        Args:
            data: Input data to prove
            privacy_level: Privacy level determines proof strength

        Returns:
            ProofResult with proof bytes and metadata
        """
        try:
            response = await self._client.post(
                "/api/v1/proofs/generate",
                json={
                    "data": data.hex(),
                    "privacy_level": privacy_level.value,
                },
            )
            result = self._handle_response(response)
            return ProofResult(
                proof_id=result["proof_id"],
                proof_data=bytes.fromhex(result["proof"]),
                privacy_level=result["privacy_level"],
                generation_time_ms=result["generation_time_ms"],
                quality_score=result.get("quality_score", 1.0),
                verified=result.get("verified", False),
            )
        except httpx.RequestError as e:
            raise ProofGenerationError(f"Failed to generate proof: {e}")

    async def verify_proof(self, proof: bytes) -> bool:
        """
        Verify a zero-knowledge proof.

        Args:
            proof: Proof bytes to verify

        Returns:
            True if proof is valid, False otherwise
        """
        try:
            response = await self._client.post(
                "/api/v1/proofs/verify",
                json={"proof": proof.hex()},
            )
            result = self._handle_response(response)
            return result.get("valid", False)
        except httpx.RequestError as e:
            raise VerificationError(f"Failed to verify proof: {e}")

    async def batch_generate_proofs(
        self,
        data_list: list[bytes],
        privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE,
    ) -> list[ProofResult]:
        """
        Generate multiple proofs in a batch for efficiency.

        Args:
            data_list: List of input data to prove
            privacy_level: Privacy level for all proofs

        Returns:
            List of ProofResult objects
        """
        try:
            response = await self._client.post(
                "/api/v1/proofs/batch",
                json={
                    "data": [d.hex() for d in data_list],
                    "privacy_level": privacy_level.value,
                },
            )
            result = self._handle_response(response)
            return [
                ProofResult(
                    proof_id=p["proof_id"],
                    proof_data=bytes.fromhex(p["proof"]),
                    privacy_level=p["privacy_level"],
                    generation_time_ms=p["generation_time_ms"],
                    quality_score=p.get("quality_score", 1.0),
                    verified=p.get("verified", False),
                )
                for p in result.get("proofs", [])
            ]
        except httpx.RequestError as e:
            raise ProofGenerationError(f"Failed to batch generate proofs: {e}")

    # ========== Privacy Operations ==========

    def recommend_privacy(
        self,
        value_usd: float,
        requires_compliance: bool = False,
        preferred_level: Optional[int] = None,
        risk_score: float = 0.0,
    ) -> PrivacyRecommendation:
        """
        Get a privacy level recommendation based on context.

        This is computed locally without an API call.

        Args:
            value_usd: Transaction value in USD
            requires_compliance: Whether regulatory compliance is needed
            preferred_level: User's preferred privacy level
            risk_score: Risk score (0.0 to 1.0)

        Returns:
            PrivacyRecommendation with level, parameters, and reasoning
        """
        context = TransactionContext(
            value_usd=value_usd,
            requires_compliance=requires_compliance,
            preferred_level=preferred_level,
            risk_score=risk_score,
        )
        return self._privacy_engine.recommend(context)

    async def morph_privacy(
        self,
        transaction_id: str,
        target_level: PrivacyLevel,
    ) -> Transaction:
        """
        Morph transaction privacy level.

        Args:
            transaction_id: ID of transaction to morph
            target_level: New privacy level

        Returns:
            Updated Transaction object
        """
        try:
            response = await self._client.post(
                "/api/v1/privacy/morph",
                json={
                    "transaction_id": transaction_id,
                    "target_level": target_level.value,
                },
            )
            data = self._handle_response(response)
            return Transaction(**data)
        except httpx.RequestError as e:
            raise NetworkError(f"Failed to morph privacy: {e}")

    # ========== Bridge Operations ==========

    async def get_bridge_quote(
        self,
        from_chain: str,
        to_chain: str,
        amount: int,
        privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE,
    ) -> FeeEstimate:
        """
        Get a quote for cross-chain bridge operation.

        Args:
            from_chain: Source blockchain
            to_chain: Destination blockchain
            amount: Amount to bridge
            privacy_level: Privacy level for the bridge

        Returns:
            FeeEstimate with gas and USD costs
        """
        try:
            response = await self._client.post(
                "/api/v1/bridge/quote",
                json={
                    "from_chain": from_chain,
                    "to_chain": to_chain,
                    "amount": amount,
                    "privacy_level": privacy_level.value,
                },
            )
            data = self._handle_response(response)
            return FeeEstimate(**data)
        except httpx.RequestError as e:
            raise BridgeError(f"Failed to get bridge quote: {e}")

    async def initiate_bridge(
        self,
        from_chain: str,
        to_chain: str,
        amount: int,
        recipient: str,
        privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE,
    ) -> str:
        """
        Initiate a cross-chain bridge transfer.

        Args:
            from_chain: Source blockchain
            to_chain: Destination blockchain
            amount: Amount to bridge
            recipient: Recipient address on destination chain
            privacy_level: Privacy level for the bridge

        Returns:
            Transfer ID for tracking
        """
        try:
            response = await self._client.post(
                "/api/v1/bridge/initiate",
                json={
                    "from_chain": from_chain,
                    "to_chain": to_chain,
                    "amount": amount,
                    "recipient": recipient,
                    "privacy_level": privacy_level.value,
                },
            )
            data = self._handle_response(response)
            return data["transfer_id"]
        except httpx.RequestError as e:
            raise BridgeError(f"Failed to initiate bridge: {e}")

    # ========== Health & Status ==========

    async def health_check(self) -> dict[str, Any]:
        """
        Check API health status.

        Returns:
            Health status information
        """
        try:
            response = await self._client.get("/health")
            return self._handle_response(response)
        except httpx.RequestError as e:
            raise NetworkError(f"Health check failed: {e}")
