# Nexuszero Crypto

A quantum-resistant zero-knowledge proof system based on lattice cryptography.

## Overview

**Nexuszero Crypto** is a Rust library implementing post-quantum cryptographic primitives for zero-knowledge proofs. It leverages the hardness of lattice problems (specifically Learning With Errors - LWE and Ring-LWE) to provide security against both classical and quantum adversaries.

## Features

- ✅ **LWE Encryption**: Learning With Errors based encryption scheme
- 🚧 **Ring-LWE**: Efficient ring-based variant with NTT optimization
- 🚧 **Zero-Knowledge Proofs**: Complete statement/witness/proof system
- ✅ **Parameter Selection**: Standard NIST security levels (128/192/256-bit)
- ✅ **Memory Security**: Automatic zeroization of sensitive data
- 🚧 **Comprehensive Testing**: Unit tests, integration tests, and benchmarks

Legend: ✅ Implemented | 🚧 In Progress | ⏳ Planned

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
nexuszero-crypto = "0.1.0"
```

## Quick Start

### LWE Encryption

```rust
use nexuszero_crypto::lattice::{LWEParameters, keygen, encrypt, decrypt};

// Create parameters for 128-bit security
let params = LWEParameters::new(256, 512, 12289, 3.2);
let mut rng = rand::thread_rng();

// Generate keys
let (secret_key, public_key) = keygen(&params, &mut rng)?;

// Encrypt a message bit
let message = true;
let ciphertext = encrypt(&public_key, message, &params, &mut rng)?;

// Decrypt
let decrypted = decrypt(&secret_key, &ciphertext, &params)?;
assert_eq!(message, decrypted);
```

### Security Parameters

```rust
use nexuszero_crypto::{CryptoParameters, SecurityLevel};

// Select security level
let params = CryptoParameters::from_security_level(SecurityLevel::Bit128);

// Or use convenience methods
let params_128 = CryptoParameters::new_128bit_security();
let params_192 = CryptoParameters::new_192bit_security();
let params_256 = CryptoParameters::new_256bit_security();
```

### Zero-Knowledge Proofs (Coming Soon)

```rust
use nexuszero_crypto::proof::{Statement, Witness, prove, verify};

// Create a statement (public)
let statement = StatementBuilder::new()
    .preimage(HashFunction::SHA3_256, hash_output)
    .build()?;

// Create a witness (secret)
let witness = Witness::preimage(preimage);

// Generate proof
let proof = prove(&statement, &witness)?;

// Verify proof
verify(&statement, &proof)?;
```

## Project Structure

```
nexuszero-crypto/
├── src/
│   ├── lib.rs                  # Main library entry
│   ├── lattice/
│   │   ├── mod.rs              # Lattice module
│   │   ├── lwe.rs              # LWE primitives
│   │   ├── ring_lwe.rs         # Ring-LWE operations
│   │   └── sampling.rs         # Error sampling
│   ├── proof/
│   │   ├── mod.rs              # Proof module
│   │   ├── statement.rs        # Statement structures
│   │   ├── witness.rs          # Witness structures
│   │   └── proof.rs            # Proof generation/verification
│   ├── params/
│   │   ├── mod.rs              # Parameters module
│   │   └── security.rs         # Security level configurations
│   └── utils/
│       ├── mod.rs              # Utilities
│       └── math.rs             # Mathematical primitives
├── tests/
│   ├── integration_tests.rs    # Integration tests
│   └── test_vectors/
│       └── nist_vectors.json   # NIST test vectors
├── benches/
│   └── crypto_benchmarks.rs    # Benchmarks
└── Cargo.toml
```

## Security Levels

| Level   | NIST Category | Parameters      | Proof Size | Prove Time | Verify Time |
| ------- | ------------- | --------------- | ---------- | ---------- | ----------- |
| 128-bit | Level 1       | n=512, q=12289  | ~8 KB      | ~80 ms     | ~40 ms      |
| 192-bit | Level 3       | n=1024, q=40961 | ~16 KB     | ~150 ms    | ~75 ms      |
| 256-bit | Level 5       | n=2048, q=65537 | ~32 KB     | ~300 ms    | ~150 ms     |

## Development

### Building

```bash
# Build the library
cargo build

# Build with optimizations
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Testing

```bash
# Run all tests
cargo test --all-features

# Run integration tests
cargo test --test integration_tests

# Run with verbose output
cargo test -- --nocapture
```

### Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench lwe_encrypt
```

## Roadmap

### Week 1: Core Cryptography (Current)

- [x] Project structure and dependencies
- [x] LWE primitives implementation
- [ ] Ring-LWE with NTT optimization
- [ ] Proof structures (Statement, Witness, Proof)
- [ ] Parameter selection algorithms
- [ ] Comprehensive unit tests

### Week 2: Neural Optimizer

- [ ] GNN-based parameter optimization
- [ ] PyTorch integration
- [ ] Real-time proof size estimation

### Week 3+: Advanced Features

- [ ] Multi-party computation support
- [ ] Threshold schemes
- [ ] On-chain verification optimization

## Contributing

Contributions are welcome! This is an active research and development project.

### Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Security

⚠️ **Warning**: This library is under active development and has not been audited. Do NOT use in production for security-critical applications.

### Reporting Security Issues

If you discover a security vulnerability, please email security@nexuszero.dev with details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [Learning With Errors (LWE)](https://cims.nyu.edu/~regev/papers/lwesurvey.pdf)
- [NIST Post-Quantum Cryptography Standards](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [Ring-LWE](https://eprint.iacr.org/2012/230.pdf)
- [Zero-Knowledge Proofs](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)

## Acknowledgments

Built with:

- **Rust** - Systems programming language
- **ndarray** - N-dimensional arrays
- **sha3** - Cryptographic hashing
- **zeroize** - Secure memory management

---

**Created:** November 20, 2025  
**Status:** 🟢 Active Development  
**Maintainer:** Steve (@iamthegreatdestroyer)
