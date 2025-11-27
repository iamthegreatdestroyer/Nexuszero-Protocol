"""
Authentication middleware for FastAPI / Starlette
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger("NexusAuth")

from .jwt_auth import DEFAULT_AUTH_MANAGER, JWTAuthManager, TokenClaims

AUTH_MANAGER: JWTAuthManager = DEFAULT_AUTH_MANAGER


class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, exempt_paths: Optional[list] = None):
        super().__init__(app)
        self._exempt_paths = exempt_paths or ["/login", "/refresh", "/open-api"]

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in self._exempt_paths:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format")

        token = parts[1]
        try:
            claims: TokenClaims = await AUTH_MANAGER.verify_access_token(token)
            # Attach claims to state for route handlers
            request.state.user = claims
            return await call_next(request)
        except Exception as e:
            logger.warning(f"Auth failed: {e}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
