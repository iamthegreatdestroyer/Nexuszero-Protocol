import asyncio
import os
import sys
import jwt
from datetime import datetime

# Make sure repo root is on sys.path so `src` package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.auth.jwt_auth import JWTAuthManager

async def run_test():
    m = JWTAuthManager()
    pair = await m.authenticate('test1')
    print('access_token:', pair.access_token[:60] + '...')
    # decode payload without verifying signature so we can inspect fields quickly
    payload = jwt.decode(pair.access_token, options={'verify_signature': False})
    print('jti access:', payload['jti'])

    # Call revoke_token which uses jwt.decode with signature verification
    try:
        await m.revoke_token(pair.access_token)
        print('revoke_token executed')
    except Exception as e:
        print('revoke_token raised', type(e), e)
    print('revoked store keys', list(m._revocation_store._revoked.keys()))

    # As a control, revoke manually with the same jti and check map
    await m._revocation_store.revoke(
        payload['jti'], datetime.fromtimestamp(payload['exp'])
    )
    print('revoked store keys after manual revoke', list(m._revocation_store._revoked.keys()))
    print('revoked?', await m._revocation_store.is_revoked(payload['jti']))

    # Verify that signature-verified decode works (revoke_token uses this)
    try:
        payload_verify = jwt.decode(
            pair.access_token, m._get_public_pem(), algorithms=['RS256'], options={'verify_exp': False}
        )
        print('verified decode jti:', payload_verify['jti'])
    except Exception as e:
        print('verified decode failed:', type(e), e)

    try:
        await m.verify_access_token(pair.access_token)
        print('Verify succeeded unexpectedly')
    except Exception as e:
        print('Verify raised as expected:', type(e), e)

if __name__ == '__main__':
    asyncio.run(run_test())
