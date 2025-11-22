import pytest

def test_ray_tune_import():
    try:
        from ray import tune  # noqa: F401
    except Exception:
        pytest.skip("Ray Tune not installed")

    # Minimal smoke test: ensure trainable import
    from nexuszero_optimizer.training.tuner import ray_trainable  # noqa: F401
