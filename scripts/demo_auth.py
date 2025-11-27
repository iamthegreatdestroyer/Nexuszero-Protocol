"""
Demo script for auth using FastAPI TestClient
"""

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel

from src.auth import auth_router, JWTAuthMiddleware

app = FastAPI()
app.include_router(auth_router, prefix="/auth")
# Mount middleware for auth - leave /auth/login & /auth/refresh open
app.add_middleware(JWTAuthMiddleware, exempt_paths=["/auth/login", "/auth/refresh"])


@app.get("/protected")
async def protected_endpoint(request: Request):
    claims = getattr(request.state, "user", None)
    return {"ok": True, "user": claims.sub if claims else None}


class Demo:
    def run(self):
        client = TestClient(app)

        # Login
        login_resp = client.post("/auth/login", json={"user_id": "user123"})
        assert (
            login_resp.status_code == 200
        ), f"Login failed: {login_resp.text}"
        data = login_resp.json()
        access = data["access_token"]
        refresh = data["refresh_token"]

        # Access protected route
        headers = {"Authorization": f"Bearer {access}"}
        r_prot = client.get("/protected", headers=headers)
        print("Protected status:", r_prot.status_code, r_prot.json())

        # Refresh tokens
        r_refresh = client.post(
            "/auth/refresh", json={"refresh_token": refresh}
        )
        print("Refresh status:", r_refresh.status_code, r_refresh.json())


if __name__ == "__main__":
    d = Demo()
    d.run()
