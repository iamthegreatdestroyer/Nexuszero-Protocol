"""
Enterprise JWT Authentication System for Nexuszero Protocol
Short-lived access tokens with secure refresh rotation.
"""

from __future__ import annotations

import os
import secrets
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger("NexusAuth")


class TokenType(Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    BIOMETRIC = "biometric"


class AuthError(Exception):
    pass


class TokenExpiredError(AuthError):
    pass


class TokenInvalidError(AuthError):
    pass


class TokenRevokedError(AuthError):
    pass


class RateLimitError(AuthError):
    pass


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    access_expires_at: datetime
    refresh_expires_at: datetime
    token_type: str = "Bearer"

    def to_dict(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": int((self.access_expires_at - now).total_seconds()),
            "refresh_expires_in": int((self.refresh_expires_at - now).total_seconds()),
        }


@dataclass
class TokenClaims:
    sub: str
    iat: datetime
    exp: datetime
    jti: str
    iss: str = "nexuszero-auth"
    aud: str = "nexuszero-wallet"
    token_type: TokenType = TokenType.ACCESS
    wallet_address: Optional[str] = None
    device_id: Optional[str] = None
    permissions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub": self.sub,
            "iat": int(self.iat.timestamp()),
            "exp": int(self.exp.timestamp()),
            "jti": self.jti,
            "iss": self.iss,
            "aud": self.aud,
            "type": self.token_type.value,
            "wallet_address": self.wallet_address,
            "device_id": self.device_id,
            "permissions": self.permissions,
        }


class TokenRevocationStore:
    def __init__(self, max_entries: int = 100000):
        self._revoked: Dict[str, datetime] = {}
        self._max = max_entries
        self._lock = asyncio.Lock()

    async def revoke(self, jti: str, expires_at: datetime):
        async with self._lock:
            if len(self._revoked) >= self._max:
                now = datetime.now(timezone.utc)
                self._revoked = {k: v for k, v in self._revoked.items() if v > now}
            self._revoked[jti] = expires_at

    async def is_revoked(self, jti: str) -> bool:
        # No lock needed for read in this simple in-memory fallback
        return jti in self._revoked


class RefreshTokenStore:
    def __init__(self):
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._families: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()

    async def store(self, jti: str, family_id: str, user_id: str, expires_at: datetime):
        async with self._lock:
            self._tokens[jti] = {
                "family_id": family_id,
                "user_id": user_id,
                "expires_at": expires_at,
                "used": False,
            }
            if family_id not in self._families:
                self._families[family_id] = []
            self._families[family_id].append(jti)

    async def validate_and_use(self, jti: str) -> Tuple[bool, Optional[str], Optional[str]]:
        async with self._lock:
            if jti not in self._tokens:
                return False, None, None

            token = self._tokens[jti]
            if token["expires_at"] < datetime.now(timezone.utc):
                del self._tokens[jti]
                return False, None, None

            if token["used"]:
                # Token reuse attack - revoke entire family
                family_id = token["family_id"]
                for t in self._families.get(family_id, []):
                    self._tokens.pop(t, None)
                self._families.pop(family_id, None)
                logger.warning(f"Token reuse detected!  Family {family_id} revoked")
                return False, None, None

            token["used"] = True
            return True, token["user_id"], token["family_id"]


class RateLimiter:
    def __init__(self, requests_per_minute: int = 10):
        self._rpm = requests_per_minute
        self._buckets: Dict[str, List[datetime]] = {}

    async def check(self, identifier: str) -> bool:
        now = datetime.now(timezone.utc)
        minute_ago = now - timedelta(minutes=1)

        if identifier not in self._buckets:
            self._buckets[identifier] = []

        self._buckets[identifier] = [t for t in self._buckets[identifier] if t > minute_ago]

        if len(self._buckets[identifier]) >= self._rpm:
            raise RateLimitError(f"Rate limit exceeded for {identifier}")

        self._buckets[identifier].append(now)
        return True


class JWTAuthManager:
    def __init__(
        self,
        private_key_path: Optional[str] = None,
        access_lifetime_minutes: int = 15,
        refresh_lifetime_days: int = 7,
    ):
        self.access_lifetime = timedelta(minutes=access_lifetime_minutes)
        self.refresh_lifetime = timedelta(days=refresh_lifetime_days)

        if private_key_path and os.path.exists(private_key_path):
            with open(private_key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        else:
            self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

        self._public_key = self._private_key.public_key()
        self._revocation_store = TokenRevocationStore()
        self._refresh_store = RefreshTokenStore()
        self._rate_limiter = RateLimiter()

    def _get_private_pem(self) -> bytes:
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def _get_public_pem(self) -> bytes:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    async def authenticate(
        self,
        user_id: str,
        wallet_address: Optional[str] = None,
        device_id: Optional[str] = None,
        permissions: Optional[List[str]] = None,
    ) -> TokenPair:
        await self._rate_limiter.check(f"auth:{user_id}")

        now = datetime.now(timezone.utc)
        family_id = secrets.token_urlsafe(16)

        # Create access token
        access_claims = TokenClaims(
            sub=user_id,
            iat=now,
            exp=now + self.access_lifetime,
            jti=secrets.token_urlsafe(32),
            token_type=TokenType.ACCESS,
            wallet_address=wallet_address,
            device_id=device_id,
            permissions=permissions or ["read", "sign"],
        )

        # Create refresh token
        refresh_jti = secrets.token_urlsafe(32)
        refresh_claims = TokenClaims(
            sub=user_id, iat=now, exp=now + self.refresh_lifetime, jti=refresh_jti, token_type=TokenType.REFRESH,
            wallet_address=wallet_address, device_id=device_id,
        )

        access_token = jwt.encode(access_claims.to_dict(), self._get_private_pem(), algorithm="RS256")
        refresh_token = jwt.encode({**refresh_claims.to_dict(), "family_id": family_id}, self._get_private_pem(), algorithm="RS256")

        await self._refresh_store.store(refresh_jti, family_id, user_id, now + self.refresh_lifetime)

        logger.info(f"Tokens generated for user: {user_id}")
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            access_expires_at=now + self.access_lifetime,
            refresh_expires_at=now + self.refresh_lifetime,
        )

    async def verify_access_token(self, token: str) -> TokenClaims:
        try:
            payload = jwt.decode(
                token,
                self._get_public_pem(),
                algorithms=["RS256"],
                issuer="nexuszero-auth",
                audience="nexuszero-wallet",
                leeway=0,
                options={"verify_iat": False, "verify_exp": True},
            )

            if payload.get("type") != TokenType.ACCESS.value:
                raise TokenInvalidError("Invalid token type")

            if await self._revocation_store.is_revoked(payload["jti"]):
                raise TokenRevokedError("Token revoked")

            return TokenClaims(
                sub=payload["sub"],
                iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
                exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
                jti=payload["jti"],
                wallet_address=payload.get("wallet_address"),
                device_id=payload.get("device_id"),
                permissions=payload.get("permissions", []),
            )
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token expired")
        except jwt.InvalidTokenError as e:
            raise TokenInvalidError(f"Invalid token: {e}")

    async def refresh_tokens(self, refresh_token: str) -> TokenPair:
        try:
            payload = jwt.decode(
                refresh_token,
                self._get_public_pem(),
                algorithms=["RS256"],
                issuer="nexuszero-auth",
                audience="nexuszero-wallet",
                options={"verify_iat": False},
            )

            if payload.get("type") != TokenType.REFRESH.value:
                raise TokenInvalidError("Invalid token type")

            valid, user_id, family_id = await self._refresh_store.validate_and_use(payload["jti"])
            if not valid:
                raise TokenRevokedError("Refresh token invalid or reused")

            # Issue new tokens and rotate
            return await self.authenticate(
                user_id=user_id,
                wallet_address=payload.get("wallet_address"),
                device_id=payload.get("device_id"),
                permissions=payload.get("permissions", ["read", "sign"]),
            )
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Refresh token expired")
        except jwt.InvalidTokenError as e:
            raise TokenInvalidError(f"Invalid refresh token: {e}")

    async def revoke_token(self, token: str):
        try:
            payload = jwt.decode(
                token,
                self._get_public_pem(),
                algorithms=["RS256"],
                issuer="nexuszero-auth",
                audience="nexuszero-wallet",
                options={"verify_exp": False, "verify_iat": False},
            )
            logger.debug(f"Revoke token payload jti: {payload.get('jti')} exp: {payload.get('exp')}")
            await self._revocation_store.revoke(
                payload["jti"], datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Failed to decode token for revocation: {e}")
            pass

    def get_public_key_pem(self) -> bytes:
        return self._get_public_pem()


def create_auth_manager(private_key_path: Optional[str] = None) -> JWTAuthManager:
    return JWTAuthManager(private_key_path=private_key_path)

# Global default for simple apps: singleton manager to share signing keys across modules
DEFAULT_AUTH_MANAGER: JWTAuthManager = create_auth_manager()
