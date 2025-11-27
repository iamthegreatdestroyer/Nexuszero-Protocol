"""
Routes for JWT Auth (FastAPI)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from .jwt_auth import DEFAULT_AUTH_MANAGER, JWTAuthManager, TokenPair

router = APIRouter()

# Configuration: use default shared manager for routes
AUTH_MANAGER: JWTAuthManager = DEFAULT_AUTH_MANAGER


class LoginRequest(BaseModel):
    user_id: str
    wallet_address: Optional[str] = None
    device_id: Optional[str] = None


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    refresh_expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest):
    try:
        tokens: TokenPair = await AUTH_MANAGER.authenticate(payload.user_id, payload.wallet_address, payload.device_id)
        return tokens.to_dict()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post("/refresh", response_model=LoginResponse)
async def refresh(payload: RefreshRequest):
    try:
        tokens: TokenPair = await AUTH_MANAGER.refresh_tokens(payload.refresh_token)
        return tokens.to_dict()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post("/revoke")
async def revoke(payload: RefreshRequest):
    await AUTH_MANAGER.revoke_token(payload.refresh_token)
    return {"ok": True}
