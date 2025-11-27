"""
Unit tests for JWT auth flows
"""

import asyncio
from datetime import timedelta, datetime
import os
import sys

import pytest

# Ensure repo root is importable so `src` package can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.auth.jwt_auth import (
    JWTAuthManager,
    TokenRevokedError,
    TokenExpiredError,
    TokenInvalidError,
)
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from src.auth import auth_router, JWTAuthMiddleware


@pytest.mark.asyncio
async def test_auth_flow_basic():
    manager = JWTAuthManager(access_lifetime_minutes=1, refresh_lifetime_days=1)

    tokens = await manager.authenticate(user_id="testuser", wallet_address="0xabc")

    # Verify access token ok
    claims = await manager.verify_access_token(tokens.access_token)
    assert claims.sub == "testuser"
    assert claims.wallet_address == "0xabc"

    # Refresh tokens
    new_pair = await manager.refresh_tokens(tokens.refresh_token)

    assert new_pair.access_token != tokens.access_token
    assert new_pair.refresh_token != tokens.refresh_token

    # Reuse refresh token should be treated as breach (invalid after use)
    with pytest.raises(TokenRevokedError):
        await manager.refresh_tokens(tokens.refresh_token)

    # Revoke access token -> subsequent verify raises
    await manager.revoke_token(new_pair.access_token)
    with pytest.raises(TokenRevokedError):
        await manager.verify_access_token(new_pair.access_token)


def test_auth_routes_via_testclient():
    # Sanity check: verify login, refresh and protected endpoint via TestClient
    app = FastAPI()
    app.include_router(auth_router, prefix="/auth")
    app.add_middleware(JWTAuthMiddleware, exempt_paths=["/auth/login", "/auth/refresh"])

    @app.get("/protected")
    async def protected(request: Request):
        claims = getattr(request.state, "user", None)
        return {"ok": True, "sub": claims.sub if claims else None}

    client = TestClient(app)
    resp = client.post("/auth/login", json={"user_id": "tcuser"})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("access_token")
    token = data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    r = client.get("/protected", headers=headers)
    assert r.status_code == 200
    assert r.json()["sub"] == "tcuser"


@pytest.mark.asyncio
async def test_invalid_tokens():
    import time
    manager = JWTAuthManager()
    fake_token = "not.a.token"

    with pytest.raises(TokenInvalidError):
        await manager.verify_access_token(fake_token)

    # Expired tokens handled - use negative lifetime to guarantee expiry
    # We need to create a token that's already expired
    # Since we can't set negative lifetime, we'll use a fresh manager and wait
    m2 = JWTAuthManager(access_lifetime_minutes=0)  # exp = iat
    pair = await m2.authenticate(user_id="u2")
    # Wait 2 seconds to ensure we're past the exp time
    time.sleep(2)
    with pytest.raises(TokenExpiredError):
        await m2.verify_access_token(pair.access_token)
