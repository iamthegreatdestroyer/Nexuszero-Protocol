# Nexuszero Auth

JWT-based authentication module for Nexuszero Wallet and APIs.

Features:

- Short-lived RS256 signed access tokens
- Refresh tokens with rotation and family revocation on reuse
- In-memory revocation store with async lock (fallback for production)
- Rate limiting per user for token creation
- FastAPI router and middleware for easy integration

Usage:

- Include `src.auth` router in your app and add `JWTAuthMiddleware`.
- Secure endpoints will receive `request.state.user` populated with token claims.

For tests and demo see `scripts/demo_auth.py` and `tests/test_auth.py`.
