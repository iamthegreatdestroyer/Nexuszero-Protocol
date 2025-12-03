# NexusZero Zero-Knowledge Proof API Reference

This document provides comprehensive documentation for the NexusZero ZK proof system API, including all types, traits, and functions for creating and verifying zero-knowledge proofs.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Types](#core-types)
4. [Statement API](#statement-api)
5. [Witness API](#witness-api)
6. [Proof API](#proof-api)
7. [Prover API](#prover-api)
8. [Verifier API](#verifier-api)
9. [Error Handling](#error-handling)
10. [Security Considerations](#security-considerations)

---

## Overview

The NexusZero ZK proof system provides a modular, trait-based architecture for generating and verifying zero-knowledge proofs. It supports multiple proof types including:

- **Discrete Log Proofs** - Prove knowledge of exponent `x` where `g^x = h`
- **Preimage Proofs** - Prove knowledge of `x` where `H(x) = y`
- **Range Proofs** - Prove a value lies within a range `[min, max]`
- **Custom Proofs** - Extensible for application-specific statements

### Key Features

- **Quantum-Resistant** - Lattice-based cryptography with post-quantum security
- **Modular Architecture** - Pluggable provers, verifiers, and proof systems
- **Constant-Time Operations** - Side-channel resistant implementations
- **Hardware Acceleration** - Optional GPU/TPU support
- **Distributed Verification** - Byzantine-fault-tolerant verification

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ZK PROOF SYSTEM ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐      │
│  │ Statement  │───▶│  Witness   │───▶│   Prover   │───▶│   Proof    │      │
│  │  Builder   │    │ Generator  │    │            │    │            │      │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘      │
│        │                                                      │             │
│        │                                                      ▼             │
│        │           ┌────────────┐    ┌────────────┐    ┌────────────┐      │
│        └──────────▶│  Verifier  │◀───│   Config   │    │  Result    │      │
│                    │            │    │            │    │ (bool/err) │      │
│                    └────────────┘    └────────────┘    └────────────┘      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  PLUGINS: Schnorr | Bulletproofs | Groth16 | PLONK | Ring-LWE              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
nexuszero-crypto/src/proof/
├── mod.rs              # Module exports and re-exports
├── statement.rs        # Statement types and builder
├── witness.rs          # Witness structures (secret data)
├── proof.rs            # Proof generation and verification
├── prover.rs           # Prover trait and implementations
├── verifier.rs         # Verifier trait and implementations
├── circuit.rs          # Circuit abstraction layer
├── witness_dsl.rs      # Witness generation DSL
├── witness_manager.rs  # Witness lifecycle management
├── bulletproofs.rs     # Bulletproofs range proofs
└── plugins/            # Proof system plugins
    ├── schnorr.rs
    ├── bulletproofs.rs
    ├── groth16.rs
    └── plonk.rs
```

---

## Core Types

### Statement

A `Statement` represents the public claim being proven. It contains no secret information.

```rust
use nexuszero_crypto::proof::{Statement, StatementBuilder, StatementType};

/// Types of statements that can be proven
pub enum StatementType {
    /// Prove knowledge of discrete log: g^x = h
    DiscreteLog {
        generator: Vec<u8>,
        public_value: Vec<u8>,
    },

    /// Prove knowledge of hash preimage: H(x) = y
    Preimage {
        hash_function: HashFunction,
        hash_output: Vec<u8>,
    },

    /// Prove x ∈ [min, max] (range proof)
    Range {
        min: u64,
        max: u64,
        commitment: Vec<u8>,
    },

    /// Custom statement (extensible)
    Custom {
        description: String,
    },
}

/// Complete statement structure
pub struct Statement {
    pub statement_type: StatementType,
    pub version: u8,
}
```

### Witness

A `Witness` contains the secret data proving a statement. **MUST be kept secure.**

```rust
use nexuszero_crypto::proof::{Witness, WitnessType};

/// Witness type indicator
pub enum WitnessType {
    DiscreteLog,
    Preimage,
    Range,
    Custom,
}

/// Witness structure (implements ZeroizeOnDrop for security)
pub struct Witness {
    secret_data: SecretData,  // Automatically zeroized on drop
    randomness: Vec<u8>,
    witness_type: WitnessType,
}
```

### Proof

A `Proof` is the cryptographic evidence that a statement is true without revealing the witness.

```rust
use nexuszero_crypto::proof::{Proof, ProofMetadata};

/// Zero-knowledge proof structure
pub struct Proof {
    pub commitments: Vec<Vec<u8>>,
    pub challenge: Vec<u8>,
    pub responses: Vec<Vec<u8>>,
    pub metadata: ProofMetadata,
    pub bulletproof: Option<BulletproofRangeProof>,
}

/// Proof metadata
pub struct ProofMetadata {
    pub proof_type: ProofType,
    pub timestamp: u64,
    pub prover_id: String,
    pub size: usize,
}
```

---

## Statement API

### StatementBuilder

Use the builder pattern to construct statements:

```rust
use nexuszero_crypto::proof::{StatementBuilder, HashFunction};

// Discrete Log Statement
let discrete_log_stmt = StatementBuilder::new()
    .discrete_log(
        generator.to_vec(),    // Generator g
        public_value.to_vec()  // Public value h = g^x
    )
    .build()?;

// Preimage Statement
let preimage_stmt = StatementBuilder::new()
    .preimage(
        HashFunction::SHA3_256,
        hash_output.to_vec()   // H(x) = y
    )
    .build()?;

// Range Statement
let range_stmt = StatementBuilder::new()
    .range(
        0,                     // min
        1000000,               // max
        commitment.to_vec()    // Pedersen commitment to value
    )
    .build()?;
```

### Statement Methods

| Method              | Description                    | Return Type               |
| ------------------- | ------------------------------ | ------------------------- |
| `validate()`        | Validate statement consistency | `CryptoResult<()>`        |
| `to_bytes()`        | Serialize to bytes             | `CryptoResult<Vec<u8>>`   |
| `from_bytes(bytes)` | Deserialize from bytes         | `CryptoResult<Statement>` |
| `hash()`            | Compute SHA3-256 hash          | `CryptoResult<[u8; 32]>`  |

### Supported Hash Functions

```rust
pub enum HashFunction {
    SHA3_256,  // Keccak-based SHA-3
    SHA256,    // NIST SHA-256
    Blake3,    // BLAKE3 (fast, secure)
}
```

---

## Witness API

### Creating Witnesses

```rust
use nexuszero_crypto::proof::Witness;

// Discrete Log Witness (exponent x where g^x = h)
let dlog_witness = Witness::discrete_log(exponent.to_vec());

// Preimage Witness (x where H(x) = y)
let preimage_witness = Witness::preimage(preimage.to_vec());

// Range Witness (value and blinding factor)
let range_witness = Witness::range(
    42u64,                    // The secret value
    blinding_factor.to_vec()  // Randomness for commitment
);

// Custom Witness
let custom_witness = Witness::custom(secret_data.to_vec());
```

### Witness Methods

| Method                       | Description                          | Return Type    |
| ---------------------------- | ------------------------------------ | -------------- |
| `satisfies_statement(&stmt)` | Check if witness satisfies statement | `bool`         |
| `witness_type()`             | Get the witness type                 | `&WitnessType` |

### Security: Automatic Zeroization

Witnesses implement `ZeroizeOnDrop`, ensuring secret data is securely erased from memory:

```rust
{
    let witness = Witness::discrete_log(secret.to_vec());
    // Use witness...
} // witness dropped here, secret data automatically zeroized
```

---

## Proof API

### Generating Proofs

```rust
use nexuszero_crypto::proof::{prove, Statement, Witness};

// Basic proof generation
let proof = prove(&statement, &witness)?;

// With configuration
let prover_config = ProverConfig {
    security_level: SecurityLevel::Bit128,
    optimizations: HashMap::new(),
    backend_params: HashMap::new(),
};

let prover = DirectProver;
let proof = prover.prove(&statement, &witness, &prover_config).await?;
```

### Verifying Proofs

```rust
use nexuszero_crypto::proof::{verify, Statement, Proof};

// Basic verification
verify(&statement, &proof)?;  // Returns CryptoResult<()>

// With verifier trait
let verifier_config = VerifierConfig {
    security_level: SecurityLevel::Bit128,
    optimizations: HashMap::new(),
    backend_params: HashMap::new(),
};

let verifier = DirectVerifier;
let is_valid = verifier.verify(&statement, &proof, &verifier_config).await?;
```

### Batch Operations

```rust
use nexuszero_crypto::proof::verify_batch;

// Batch verification (more efficient than individual verification)
let statements_and_proofs: Vec<(Statement, Proof)> = vec![
    (stmt1, proof1),
    (stmt2, proof2),
    (stmt3, proof3),
];

verify_batch(&statements_and_proofs)?;
```

### Proof Serialization

```rust
// Serialize proof to bytes
let proof_bytes = bincode::serialize(&proof)?;

// Deserialize proof from bytes
let recovered_proof: Proof = bincode::deserialize(&proof_bytes)?;

// Serialize to JSON
let proof_json = serde_json::to_string(&proof)?;
```

---

## Prover API

### Prover Trait

```rust
#[async_trait]
pub trait Prover: Send + Sync {
    /// Unique prover identifier
    fn id(&self) -> &str;

    /// Statement types this prover supports
    fn supported_statements(&self) -> Vec<StatementType>;

    /// Generate a proof
    async fn prove(
        &self,
        statement: &Statement,
        witness: &Witness,
        config: &ProverConfig
    ) -> CryptoResult<Proof>;

    /// Batch proof generation
    async fn prove_batch(
        &self,
        statements: &[Statement],
        witnesses: &[Witness],
        config: &ProverConfig
    ) -> CryptoResult<Vec<Proof>>;

    /// Get prover capabilities
    fn capabilities(&self) -> ProverCapabilities;
}
```

### Prover Configuration

```rust
pub struct ProverConfig {
    /// Security level (128, 192, or 256 bits)
    pub security_level: SecurityLevel,

    /// Optimization hints
    pub optimizations: HashMap<String, serde_json::Value>,

    /// Backend-specific parameters
    pub backend_params: HashMap<String, serde_json::Value>,
}
```

### Prover Capabilities

```rust
pub struct ProverCapabilities {
    /// Maximum statement size supported
    pub max_statement_size: usize,

    /// Average proof generation time (ms)
    pub avg_proof_time_ms: u64,

    /// Security guarantee level
    pub zk_guarantee: ZKGuarantee,

    /// Supported optimizations
    pub supported_optimizations: Vec<String>,
}

pub enum ZKGuarantee {
    Perfect,        // Information-theoretic ZK
    Computational,  // Computational ZK
    HonestVerifier, // Honest-verifier ZK
}
```

### Available Provers

| Prover                | Description                | Use Case            |
| --------------------- | -------------------------- | ------------------- |
| `LegacyProver`        | Original implementation    | Default fallback    |
| `DirectProver`        | Trait-based direct proving | Standard operations |
| `CircuitProver`       | Circuit-based proving      | Complex statements  |
| `HardwareProver`      | GPU/TPU acceleration       | High throughput     |
| `ParallelBatchProver` | Parallel batch proving     | Batch operations    |

---

## Verifier API

### Verifier Trait

```rust
#[async_trait]
pub trait Verifier: Send + Sync {
    /// Unique verifier identifier
    fn id(&self) -> &str;

    /// Statement types this verifier supports
    fn supported_statements(&self) -> Vec<StatementType>;

    /// Verify a proof
    async fn verify(
        &self,
        statement: &Statement,
        proof: &Proof,
        config: &VerifierConfig
    ) -> CryptoResult<bool>;

    /// Batch verification
    async fn verify_batch(
        &self,
        statements: &[Statement],
        proofs: &[Proof],
        config: &VerifierConfig
    ) -> CryptoResult<Vec<bool>>;

    /// Get verifier capabilities
    fn capabilities(&self) -> VerifierCapabilities;
}
```

### Verifier Configuration

```rust
pub struct VerifierConfig {
    /// Security level
    pub security_level: SecurityLevel,

    /// Verification optimizations
    pub optimizations: HashMap<String, serde_json::Value>,

    /// Backend-specific parameters
    pub backend_params: HashMap<String, serde_json::Value>,
}
```

### Available Verifiers

| Verifier                 | Description             | Use Case           |
| ------------------------ | ----------------------- | ------------------ |
| `DirectVerifier`         | Standard verification   | Default            |
| `HardwareVerifier`       | GPU/TPU accelerated     | High throughput    |
| `DistributedVerifier`    | Multi-node verification | Decentralized apps |
| `ProbabilisticVerifier`  | Sampling-based          | Very high volume   |
| `OptimizedBatchVerifier` | Optimized batching      | Batch operations   |

---

## Error Handling

### CryptoError Types

```rust
pub enum CryptoError {
    /// Invalid security parameter
    InvalidParameter(String),

    /// Encryption/Decryption error
    EncryptionError(String),

    /// Proof generation failed
    ProofError(String),

    /// Verification failed
    VerificationError(String),

    /// Serialization error
    SerializationError(String),

    /// Mathematical operation error
    MathError(String),

    /// Hardware backend error
    HardwareError(String),

    /// Invalid input
    InvalidInput(String),

    /// Internal error
    InternalError(String),

    /// Network error
    NetworkError(String),

    /// Not implemented
    NotImplemented(String),
}
```

### Error Handling Pattern

```rust
use nexuszero_crypto::{CryptoResult, CryptoError};

fn generate_proof() -> CryptoResult<Proof> {
    let statement = StatementBuilder::new()
        .discrete_log(g, h)
        .build()
        .map_err(|e| CryptoError::InvalidParameter(e.to_string()))?;

    let witness = Witness::discrete_log(secret);

    if !witness.satisfies_statement(&statement) {
        return Err(CryptoError::ProofError(
            "Witness does not satisfy statement".to_string()
        ));
    }

    prove(&statement, &witness)
}

// Usage
match generate_proof() {
    Ok(proof) => println!("Proof generated: {:?}", proof.metadata),
    Err(CryptoError::InvalidParameter(msg)) => eprintln!("Invalid input: {}", msg),
    Err(CryptoError::ProofError(msg)) => eprintln!("Proof failed: {}", msg),
    Err(e) => eprintln!("Error: {}", e),
}
```

---

## Security Considerations

### ⚠️ Security Warning

**This library is under active development and has not been independently audited.**

Do NOT use in production without:

1. ✅ Independent third-party security review
2. ✅ Comprehensive side-channel analysis
3. ✅ Formal threat modeling for your use case
4. ✅ Infrastructure hardening

### Constant-Time Operations

All witness operations use constant-time algorithms to prevent timing attacks:

```rust
// Constant-time modular exponentiation (Montgomery ladder)
let result = ct_modpow(&base, &exponent, &modulus);

// Constant-time byte comparison
let equal = ct_bytes_eq(&a, &b);

// Constant-time range check
let in_range = ct_in_range(value, min, max);
```

### Memory Security

- Witnesses implement `ZeroizeOnDrop` for automatic secure erasure
- Secret data never serialized to disk without encryption
- No witness data in proof (zero-knowledge property)

### Trusted Setup

Some proof systems (Groth16) require a trusted setup:

```rust
// Check if trusted setup is required
let caps = prover.capabilities();
if caps.trusted_setup_required {
    // Use ceremony-generated parameters
    let setup_params = load_trusted_setup("ceremony_params.bin")?;
    // ...
}
```

---

## Quick Reference

### Common Operations

```rust
use nexuszero_crypto::proof::*;

// 1. Build a statement
let stmt = StatementBuilder::new()
    .range(0, 1000, commitment)
    .build()?;

// 2. Create a witness
let witness = Witness::range(value, blinding);

// 3. Validate witness satisfies statement
assert!(witness.satisfies_statement(&stmt));

// 4. Generate proof
let proof = prove(&stmt, &witness)?;

// 5. Verify proof
verify(&stmt, &proof)?;

// 6. Serialize for transmission
let proof_bytes = bincode::serialize(&proof)?;
```

### Type Aliases

```rust
/// Result type for all crypto operations
pub type CryptoResult<T> = Result<T, CryptoError>;
```

---

## See Also

- [Integration Guide](./INTEGRATION_GUIDE.md) - Step-by-step integration
- [Circuit DSL](./CIRCUIT_DSL.md) - Circuit definition language
- [Security Audit](../SECURITY_AUDIT.md) - Security analysis
- [Proof Mechanism](./proof-mechanism.md) - Design decisions
