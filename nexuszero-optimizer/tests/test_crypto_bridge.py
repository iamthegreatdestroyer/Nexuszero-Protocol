import pytest
from nexuszero_optimizer.utils.crypto_bridge import CryptoBridge


def test_crypto_bridge_load():
    bridge = CryptoBridge()
    assert bridge is not None
    # Availability may depend on build; ensure object exists
    # Attempt to access version symbol if library loaded
    if bridge.is_available():
        try:
            version_func = bridge.lib.nexuszero_crypto_version
            version = version_func()
            assert version == 100
        except AttributeError:
            pytest.skip(
                "Version symbol not found; library may be partial build"
            )
    else:
        pytest.skip("Rust library not available; running in simulation mode")
