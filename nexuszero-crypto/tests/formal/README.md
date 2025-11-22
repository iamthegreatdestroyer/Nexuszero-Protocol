# Formal Verification Setup

This directory contains Kani proof harnesses for formally verifying constant-time properties of cryptographic primitives.

## Harnesses

- **`ct_modpow_kani.rs`**: Verifies functional correctness of Montgomery ladder modular exponentiation against `num_bigint::modpow` for arbitrary 64-bit inputs.
- **`ct_in_range_kani.rs`**: Validates constant-time range checking against reference implementation.
- **`ct_dot_product_kani.rs`**: Checks dot product correctness for small bounded vectors (length ≤ 6).

## Local Execution (Linux/WSL Only)

Kani requires Linux. Install via:

```bash
cargo install --locked kani-verifier
cargo kani setup
```

Run all harnesses:

```bash
cd nexuszero-crypto
cargo kani --tests
```

Run specific harness:

```bash
cargo kani --tests --harness verify_ct_modpow_small_equivalence
```

## CI Integration

GitHub Actions workflow (`.github/workflows/kani.yml`) runs Kani on Ubuntu 20.04:

- **Matrix**: Tests against `stable` and `beta` Rust toolchains.
- **Modes**:
  - `quick`: Single harness (`ct_modpow`) for fast PR feedback.
  - `full`: All harnesses for comprehensive verification.
- **Triggers**:
  - Push/PR affecting `nexuszero-crypto/**` or workflow file.
  - Weekly scheduled run (Mondays 3 AM UTC).

## Limitations

- **State Space**: Harnesses use symbolic execution; large bit widths or unbounded loops may time out.
- **Constant-Time Guarantees**: These harnesses verify functional correctness, not strict timing properties. True constant-time assurance requires side-channel analysis (e.g., dudect, differential power analysis).
- **Platform**: Kani is Linux-only; Windows users must use WSL/Docker or rely on CI.

## Future Work

- Add harnesses for higher-level protocols (Bulletproofs commitment, LWE operations).
- Integrate `enable-propproof` for property-based test verification.
- Explore CBMC-based timing analysis.

## Resources

- [Kani Documentation](https://model-checking.github.io/kani/)
- [Kani GitHub Action](https://github.com/model-checking/kani-github-action)
- [Constant-Time Crypto Best Practices](https://github.com/veorq/cryptocoding)
