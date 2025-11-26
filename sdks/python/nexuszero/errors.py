"""
Exception classes for NexusZero Protocol SDK.
"""

from typing import Any, Optional


class NexusZeroError(Exception):
    """Base exception for all NexusZero SDK errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, code={self.code!r})"


class AuthenticationError(NexusZeroError):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        code: str = "AUTH_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class ProofGenerationError(NexusZeroError):
    """Raised when proof generation fails."""

    def __init__(
        self,
        message: str = "Proof generation failed",
        code: str = "PROOF_GEN_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class VerificationError(NexusZeroError):
    """Raised when proof verification fails."""

    def __init__(
        self,
        message: str = "Proof verification failed",
        code: str = "VERIFY_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class BridgeError(NexusZeroError):
    """Raised when cross-chain bridge operations fail."""

    def __init__(
        self,
        message: str = "Bridge operation failed",
        code: str = "BRIDGE_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class ComplianceError(NexusZeroError):
    """Raised when compliance operations fail."""

    def __init__(
        self,
        message: str = "Compliance check failed",
        code: str = "COMPLIANCE_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class RateLimitError(NexusZeroError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        code: str = "RATE_LIMIT",
        retry_after: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)
        self.retry_after = retry_after


class NetworkError(NexusZeroError):
    """Raised when network communication fails."""

    def __init__(
        self,
        message: str = "Network error",
        code: str = "NETWORK_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class InvalidPrivacyLevelError(NexusZeroError):
    """Raised when an invalid privacy level is specified."""

    def __init__(
        self,
        level: int,
        message: Optional[str] = None,
    ) -> None:
        msg = message or f"Invalid privacy level: {level}. Must be 0-5."
        super().__init__(msg, "INVALID_PRIVACY_LEVEL", {"level": level})
        self.level = level
