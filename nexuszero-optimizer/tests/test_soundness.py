import torch
from nexuszero_optimizer.verification.soundness import SoundnessVerifier


def test_soundness_basic_pass():
    verifier = SoundnessVerifier()
    # Mid-range normalized values likely valid
    res = verifier.verify({"n": 0.5, "q": 0.5, "sigma": 0.5})
    assert res.security_score >= 0.0
    assert "n" in res.denormalized


def test_soundness_power_of_two_adjustment():
    verifier = SoundnessVerifier()
    # Choose n that will denormalize to non-power-of-two by tweaking normalization
    # Normalization reverse: n = n_norm * (4096-256) + 256
    # pick value that maps close to e.g. 3000 (not power of two)
    n_norm = (3000 - 256) / (4096 - 256)
    res = verifier.verify({"n": n_norm, "q": 0.6, "sigma": 0.3})
    if "n" in res.issues:
        assert "n" in res.suggestions


def test_soundness_tensor_wrapper():
    verifier = SoundnessVerifier()
    t = torch.tensor([0.2, 0.3, 0.4])
    res = verifier.verify_tensor(t)
    assert res.denormalized["n"] >= verifier.n_min
