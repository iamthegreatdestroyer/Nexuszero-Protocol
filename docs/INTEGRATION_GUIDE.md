# NexusZero ZK Proof System Integration Guide

This guide provides step-by-step instructions for integrating the NexusZero zero-knowledge proof system into your application.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Integration Patterns](#integration-patterns)
5. [TypeScript/JavaScript SDK](#typescriptjavascript-sdk)
6. [Rust Integration](#rust-integration)
7. [On-Chain Verification](#on-chain-verification)
8. [Testing Your Integration](#testing-your-integration)
9. [Production Considerations](#production-considerations)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Rust**: 1.70+ (for native integration)
- **Node.js**: 18+ (for TypeScript SDK)
- **Memory**: 4GB+ RAM for proof generation
- **Storage**: 500MB+ for cryptographic parameters

### Dependencies

```toml
# Cargo.toml
[dependencies]
nexuszero-crypto = { path = "../nexuszero-crypto" }
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
```

```json
// package.json
{
  "dependencies": {
    "@nexuszero/sdk": "^0.1.0"
  }
}
```

---

## Installation

### Rust (Native)

```bash
# Clone the repository
git clone https://github.com/nexuszero/protocol.git
cd protocol

# Build the library
cargo build --release -p nexuszero-crypto

# Run tests
cargo test -p nexuszero-crypto
```

### TypeScript/JavaScript

```bash
# Install the SDK
npm install @nexuszero/sdk

# Or with yarn
yarn add @nexuszero/sdk
```

### Docker

```dockerfile
FROM rust:1.70-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release -p nexuszero-crypto

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/libnexuszero_crypto.so /usr/lib/
```

---

## Quick Start

### 1. Generate Your First Proof (Rust)

```rust
use nexuszero_crypto::proof::{StatementBuilder, Witness, prove, verify};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Create a statement (public claim)
    // "I know x such that H(x) = this_hash"
    let hash_output = vec![0xab; 32]; // Your expected hash

    let statement = StatementBuilder::new()
        .preimage(
            nexuszero_crypto::proof::HashFunction::SHA3_256,
            hash_output
        )
        .build()?;

    // Step 2: Create the witness (your secret)
    let secret_preimage = b"my_secret_value".to_vec();
    let witness = Witness::preimage(secret_preimage);

    // Step 3: Verify witness satisfies statement (optional but recommended)
    assert!(witness.satisfies_statement(&statement), "Invalid witness");

    // Step 4: Generate the proof
    let proof = prove(&statement, &witness)?;
    println!("Proof generated! Size: {} bytes", proof.metadata.size);

    // Step 5: Verify the proof
    verify(&statement, &proof)?;
    println!("Proof verified successfully!");

    Ok(())
}
```

### 2. Generate Your First Proof (TypeScript)

```typescript
import { ProofBuilder, StatementType } from "@nexuszero/sdk";

async function main() {
  // Create a range proof: prove value is between 0 and 100
  const proof = await new ProofBuilder()
    .setStatement(StatementType.Range, {
      min: 0n,
      max: 100n,
      bitLength: 7,
    })
    .setWitness({ value: 42n })
    .generate();

  console.log("Proof generated:", proof);

  // Verify the proof
  const isValid = await proof.verify();
  console.log("Proof valid:", isValid);
}

main().catch(console.error);
```

---

## Integration Patterns

### Pattern 1: Client-Side Proof Generation

Use when the secret data should never leave the client.

```
┌─────────────┐                    ┌─────────────┐
│   Client    │                    │   Server    │
│             │                    │             │
│ 1. Create   │                    │             │
│    witness  │                    │             │
│             │                    │             │
│ 2. Generate │                    │             │
│    proof    │                    │             │
│             │    3. Send proof   │             │
│             │───────────────────▶│ 4. Verify   │
│             │                    │    proof    │
│             │    5. Response     │             │
│             │◀───────────────────│             │
└─────────────┘                    └─────────────┘
```

```rust
// Client side
async fn client_prove(secret: &[u8], public_hash: &[u8]) -> CryptoResult<Vec<u8>> {
    let statement = StatementBuilder::new()
        .preimage(HashFunction::SHA3_256, public_hash.to_vec())
        .build()?;

    let witness = Witness::preimage(secret.to_vec());
    let proof = prove(&statement, &witness)?;

    // Serialize and send to server
    Ok(bincode::serialize(&proof)?)
}

// Server side
async fn server_verify(public_hash: &[u8], proof_bytes: &[u8]) -> CryptoResult<bool> {
    let statement = StatementBuilder::new()
        .preimage(HashFunction::SHA3_256, public_hash.to_vec())
        .build()?;

    let proof: Proof = bincode::deserialize(proof_bytes)?;
    verify(&statement, &proof)?;

    Ok(true)
}
```

### Pattern 2: Batch Verification

Efficient for verifying multiple proofs simultaneously.

```rust
use nexuszero_crypto::proof::verify_batch;

async fn verify_multiple_proofs(
    statements: Vec<Statement>,
    proofs: Vec<Proof>
) -> CryptoResult<()> {
    let pairs: Vec<(Statement, Proof)> = statements
        .into_iter()
        .zip(proofs.into_iter())
        .collect();

    // ~40% faster than individual verification
    verify_batch(&pairs)?;

    Ok(())
}
```

### Pattern 3: Async Proof Pipeline

For high-throughput applications.

```rust
use tokio::sync::mpsc;
use nexuszero_crypto::proof::*;

struct ProofPipeline {
    prover: Box<dyn Prover>,
    config: ProverConfig,
}

impl ProofPipeline {
    async fn run(
        &self,
        mut rx: mpsc::Receiver<(Statement, Witness)>,
        tx: mpsc::Sender<Proof>
    ) {
        while let Some((statement, witness)) = rx.recv().await {
            match self.prover.prove(&statement, &witness, &self.config).await {
                Ok(proof) => {
                    tx.send(proof).await.ok();
                }
                Err(e) => {
                    eprintln!("Proof generation failed: {}", e);
                }
            }
        }
    }
}
```

### Pattern 4: Hardware-Accelerated Verification

For maximum throughput.

```rust
use nexuszero_crypto::proof::{HardwareVerifier, HardwareType};

async fn gpu_verification(
    statements: &[Statement],
    proofs: &[Proof]
) -> CryptoResult<Vec<bool>> {
    let gpu_verifier = HardwareVerifier::new(HardwareType::GPU);

    let config = VerifierConfig {
        security_level: SecurityLevel::Bit128,
        optimizations: [("batch_size".to_string(), json!(1000))].into(),
        backend_params: HashMap::new(),
    };

    gpu_verifier.verify_batch(statements, proofs, &config).await
}
```

---

## TypeScript/JavaScript SDK

### Complete Example: Age Verification

```typescript
import {
  ProofBuilder,
  StatementType,
  CommitmentScheme,
  Proof,
} from "@nexuszero/sdk";

class AgeVerifier {
  private minimumAge: bigint;

  constructor(minimumAge: number) {
    this.minimumAge = BigInt(minimumAge);
  }

  /**
   * Generate proof that user is at least minimumAge years old
   * without revealing actual age
   */
  async generateAgeProof(actualAge: number): Promise<Proof> {
    if (actualAge < Number(this.minimumAge)) {
      throw new Error("Age does not meet minimum requirement");
    }

    // Create range proof: actualAge >= minimumAge
    // We prove: actualAge - minimumAge >= 0
    // Which means: value ∈ [0, MAX_AGE - minimumAge]
    const normalizedAge = BigInt(actualAge) - this.minimumAge;
    const maxNormalizedAge = 150n - this.minimumAge; // Max human age ~150

    const proof = await new ProofBuilder()
      .setStatement(StatementType.Range, {
        min: 0n,
        max: maxNormalizedAge,
        bitLength: 8,
      })
      .setWitness({ value: normalizedAge })
      .generate();

    return proof;
  }

  /**
   * Verify an age proof
   */
  async verifyAgeProof(proof: Proof): Promise<boolean> {
    return proof.verify();
  }
}

// Usage
async function main() {
  const verifier = new AgeVerifier(21);

  // User is 25, proves they're over 21 without revealing exact age
  const proof = await verifier.generateAgeProof(25);
  console.log("Age proof generated");

  // Verifier checks proof
  const isValid = await verifier.verifyAgeProof(proof);
  console.log("Is over 21?", isValid); // true
}
```

### SDK API Reference

```typescript
// Statement Types
enum StatementType {
  Range = "range",
  DiscreteLog = "discrete_log",
  Preimage = "preimage",
  Custom = "custom",
}

// ProofBuilder
class ProofBuilder {
  setStatement(type: StatementType, data: any): ProofBuilder;
  setWitness(data: any): ProofBuilder;
  generate(): Promise<Proof>;
}

// Proof
interface Proof {
  data: Uint8Array;
  statement: Statement;
  commitment: Uint8Array;
  verify(): Promise<boolean>;
  serialize(): Uint8Array;
}

// Configuration
interface CryptoParameters {
  n: number; // Dimension
  q: number; // Modulus
  sigma: number; // Error distribution parameter
  securityLevel: SecurityLevel;
}
```

---

## Rust Integration

### Full Application Example

```rust
use nexuszero_crypto::{
    proof::{
        StatementBuilder, Witness, Proof,
        prove, verify, HashFunction,
        ProverConfig, VerifierConfig,
        DirectProver, DirectVerifier,
        Prover, Verifier,
    },
    SecurityLevel, CryptoResult, CryptoError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Application-level proof manager
pub struct ProofManager {
    prover: Box<dyn Prover>,
    verifier: Box<dyn Verifier>,
    prover_config: ProverConfig,
    verifier_config: VerifierConfig,
}

impl ProofManager {
    pub fn new(security_level: SecurityLevel) -> Self {
        Self {
            prover: Box::new(DirectProver),
            verifier: Box::new(DirectVerifier),
            prover_config: ProverConfig {
                security_level,
                optimizations: HashMap::new(),
                backend_params: HashMap::new(),
            },
            verifier_config: VerifierConfig {
                security_level,
                optimizations: HashMap::new(),
                backend_params: HashMap::new(),
            },
        }
    }

    /// Prove knowledge of a hash preimage
    pub async fn prove_preimage(
        &self,
        secret: &[u8],
        hash_output: &[u8]
    ) -> CryptoResult<Proof> {
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash_output.to_vec())
            .build()?;

        let witness = Witness::preimage(secret.to_vec());

        if !witness.satisfies_statement(&statement) {
            return Err(CryptoError::ProofError(
                "Witness does not satisfy statement".to_string()
            ));
        }

        self.prover.prove(&statement, &witness, &self.prover_config).await
    }

    /// Prove a value is in a range
    pub async fn prove_range(
        &self,
        value: u64,
        min: u64,
        max: u64,
        blinding: &[u8]
    ) -> CryptoResult<(Proof, Vec<u8>)> {
        // Create Pedersen commitment
        let commitment = nexuszero_crypto::proof::bulletproofs::pedersen_commit(
            value, blinding
        )?;

        let statement = StatementBuilder::new()
            .range(min, max, commitment.clone())
            .build()?;

        let witness = Witness::range(value, blinding.to_vec());

        let proof = self.prover.prove(
            &statement,
            &witness,
            &self.prover_config
        ).await?;

        Ok((proof, commitment))
    }

    /// Verify any proof
    pub async fn verify_proof(
        &self,
        statement: &Statement,
        proof: &Proof
    ) -> CryptoResult<bool> {
        self.verifier.verify(statement, proof, &self.verifier_config).await
    }
}

// REST API integration example
#[derive(Serialize, Deserialize)]
pub struct ProofRequest {
    pub proof_type: String,
    pub public_inputs: HashMap<String, Vec<u8>>,
}

#[derive(Serialize, Deserialize)]
pub struct ProofResponse {
    pub success: bool,
    pub proof_data: Option<Vec<u8>>,
    pub error: Option<String>,
}

impl ProofManager {
    pub async fn handle_verification_request(
        &self,
        proof_bytes: &[u8],
        statement_bytes: &[u8]
    ) -> ProofResponse {
        // Deserialize
        let proof: Result<Proof, _> = bincode::deserialize(proof_bytes);
        let statement: Result<Statement, _> = bincode::deserialize(statement_bytes);

        match (proof, statement) {
            (Ok(proof), Ok(statement)) => {
                match self.verify_proof(&statement, &proof).await {
                    Ok(true) => ProofResponse {
                        success: true,
                        proof_data: None,
                        error: None,
                    },
                    Ok(false) => ProofResponse {
                        success: false,
                        proof_data: None,
                        error: Some("Verification failed".to_string()),
                    },
                    Err(e) => ProofResponse {
                        success: false,
                        proof_data: None,
                        error: Some(e.to_string()),
                    },
                }
            }
            _ => ProofResponse {
                success: false,
                proof_data: None,
                error: Some("Invalid proof or statement format".to_string()),
            },
        }
    }
}
```

### Async Integration with Tokio

```rust
use tokio::time::{timeout, Duration};
use nexuszero_crypto::proof::*;

pub struct AsyncProofService {
    proof_timeout: Duration,
    verify_timeout: Duration,
}

impl AsyncProofService {
    pub fn new() -> Self {
        Self {
            proof_timeout: Duration::from_secs(30),
            verify_timeout: Duration::from_secs(5),
        }
    }

    pub async fn generate_with_timeout(
        &self,
        statement: Statement,
        witness: Witness
    ) -> CryptoResult<Proof> {
        timeout(self.proof_timeout, async {
            prove(&statement, &witness)
        })
        .await
        .map_err(|_| CryptoError::InternalError("Proof generation timeout".to_string()))?
    }

    pub async fn verify_with_timeout(
        &self,
        statement: Statement,
        proof: Proof
    ) -> CryptoResult<bool> {
        timeout(self.verify_timeout, async {
            verify(&statement, &proof).map(|_| true)
        })
        .await
        .map_err(|_| CryptoError::InternalError("Verification timeout".to_string()))?
    }
}
```

---

## On-Chain Verification

### Ethereum/EVM Integration

The NexusZero protocol uses Groth16 for on-chain verification due to gas efficiency.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./INexuszeroVerifier.sol";

contract NexuszeroVerifier is INexuszeroVerifier {
    // Verification key components (set during deployment)
    uint256[2] public alfa1;
    uint256[2][2] public beta2;
    uint256[2][2] public gamma2;
    uint256[2][2] public delta2;
    uint256[2][] public IC;

    /**
     * @notice Verify a Groth16 proof
     * @param proof The proof components [A, B, C]
     * @param publicInputs The public inputs to the circuit
     * @return True if the proof is valid
     */
    function verifyProof(
        uint256[8] calldata proof,
        uint256[] calldata publicInputs
    ) external view override returns (bool) {
        // Proof components
        uint256[2] memory a = [proof[0], proof[1]];
        uint256[2][2] memory b = [[proof[2], proof[3]], [proof[4], proof[5]]];
        uint256[2] memory c = [proof[6], proof[7]];

        // Compute linear combination of public inputs
        uint256[2] memory vk_x = IC[0];
        for (uint256 i = 0; i < publicInputs.length; i++) {
            vk_x = addition(vk_x, scalar_mul(IC[i + 1], publicInputs[i]));
        }

        // Pairing check
        return pairing(
            negate(a),
            b,
            alfa1,
            beta2,
            vk_x,
            gamma2,
            c,
            delta2
        );
    }

    // ... pairing implementation using precompiles
}
```

### Submitting Proofs On-Chain

```typescript
import { ethers } from "ethers";
import { ProofBuilder, StatementType, serializeForEVM } from "@nexuszero/sdk";

async function submitProofOnChain(
  proof: Proof,
  verifierContract: ethers.Contract
) {
  // Serialize proof for EVM
  const evmProof = serializeForEVM(proof);

  // Extract public inputs
  const publicInputs = extractPublicInputs(proof.statement);

  // Submit to contract
  const tx = await verifierContract.verifyProof(evmProof, publicInputs, {
    gasLimit: 500000,
  });

  const receipt = await tx.wait();
  return receipt.status === 1;
}
```

---

## Testing Your Integration

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use nexuszero_crypto::proof::*;

    #[tokio::test]
    async fn test_preimage_proof_roundtrip() {
        // Setup
        use sha3::{Digest, Sha3_256};

        let secret = b"test_secret_123";
        let mut hasher = Sha3_256::new();
        hasher.update(secret);
        let hash_output = hasher.finalize().to_vec();

        // Generate proof
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash_output.clone())
            .build()
            .expect("Statement creation failed");

        let witness = Witness::preimage(secret.to_vec());
        assert!(witness.satisfies_statement(&statement));

        let proof = prove(&statement, &witness)
            .expect("Proof generation failed");

        // Verify
        verify(&statement, &proof)
            .expect("Verification failed");
    }

    #[tokio::test]
    async fn test_range_proof() {
        let value = 42u64;
        let min = 0u64;
        let max = 100u64;
        let blinding = vec![0u8; 32];

        // Create commitment
        let commitment = bulletproofs::pedersen_commit(value, &blinding)
            .expect("Commitment failed");

        let statement = StatementBuilder::new()
            .range(min, max, commitment)
            .build()
            .expect("Statement creation failed");

        let witness = Witness::range(value, blinding);
        assert!(witness.satisfies_statement(&statement));

        let proof = prove(&statement, &witness)
            .expect("Proof generation failed");

        verify(&statement, &proof)
            .expect("Verification failed");
    }

    #[tokio::test]
    async fn test_invalid_witness_rejected() {
        let statement = StatementBuilder::new()
            .range(0, 100, vec![0u8; 32])
            .build()
            .expect("Statement creation failed");

        // Value outside range
        let witness = Witness::range(150, vec![0u8; 32]);
        assert!(!witness.satisfies_statement(&statement));
    }
}
```

### Integration Tests

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_proof_serialization_roundtrip() {
        let statement = StatementBuilder::new()
            .discrete_log(vec![1, 2, 3], vec![4, 5, 6])
            .build()
            .unwrap();

        let witness = Witness::discrete_log(vec![7, 8, 9]);
        let proof = prove(&statement, &witness).unwrap();

        // Serialize
        let proof_bytes = bincode::serialize(&proof).unwrap();
        let statement_bytes = statement.to_bytes().unwrap();

        // Deserialize
        let recovered_proof: Proof = bincode::deserialize(&proof_bytes).unwrap();
        let recovered_statement = Statement::from_bytes(&statement_bytes).unwrap();

        // Verify with recovered data
        verify(&recovered_statement, &recovered_proof).unwrap();
    }

    #[tokio::test]
    async fn test_batch_verification_performance() {
        use std::time::Instant;

        let count = 100;
        let mut pairs = Vec::with_capacity(count);

        for i in 0..count {
            let statement = StatementBuilder::new()
                .discrete_log(vec![i as u8], vec![i as u8 + 1])
                .build()
                .unwrap();
            let witness = Witness::discrete_log(vec![i as u8]);
            let proof = prove(&statement, &witness).unwrap();
            pairs.push((statement, proof));
        }

        // Individual verification
        let start = Instant::now();
        for (stmt, proof) in &pairs {
            verify(stmt, proof).unwrap();
        }
        let individual_time = start.elapsed();

        // Batch verification
        let start = Instant::now();
        verify_batch(&pairs).unwrap();
        let batch_time = start.elapsed();

        println!("Individual: {:?}, Batch: {:?}", individual_time, batch_time);
        assert!(batch_time < individual_time);
    }
}
```

---

## Production Considerations

### Security Checklist

- [ ] Use 256-bit security level for high-value applications
- [ ] Implement rate limiting on proof generation endpoints
- [ ] Never log or persist witness data
- [ ] Use TLS for all proof transmission
- [ ] Implement proof replay protection
- [ ] Regular security audits

### Performance Optimization

```rust
// Use batch operations when possible
let proofs = prover.prove_batch(&statements, &witnesses, &config).await?;

// Enable hardware acceleration
let config = ProverConfig {
    optimizations: [
        ("use_gpu".to_string(), json!(true)),
        ("parallel_workers".to_string(), json!(8)),
    ].into(),
    ..Default::default()
};

// Cache verification keys
lazy_static! {
    static ref VERIFICATION_KEY: VerificationKey = load_verification_key();
}
```

### Monitoring

```rust
use nexuszero_crypto::metrics::{ZkMetrics, ProofType, SecurityLevel};

// Initialize metrics
ZkMetrics::init().expect("Metrics init failed");

// Track proof generation
let guard = ZkMetrics::global().proof_generation_timer(
    ProofType::Groth16,
    SecurityLevel::Bit128
);
let proof = prove(&statement, &witness)?;
guard.complete();

// Monitor errors
ZkMetrics::global().record_error("proof_generation", "error", &error_msg);
```

---

## Troubleshooting

### Common Issues

#### 1. "Witness does not satisfy statement"

**Cause**: The secret data doesn't mathematically satisfy the public claim.

**Solution**:

```rust
// Always validate before proving
if !witness.satisfies_statement(&statement) {
    // Debug: check your inputs
    println!("Statement type: {:?}", statement.statement_type);
    println!("Witness type: {:?}", witness.witness_type());
}
```

#### 2. "Proof generation timeout"

**Cause**: Complex statements or insufficient resources.

**Solution**:

- Increase timeout for complex proofs
- Use hardware acceleration
- Reduce proof complexity

#### 3. "Verification failed"

**Cause**: Proof tampering, network corruption, or wrong statement.

**Solution**:

```rust
// Verify statement hash matches
let expected_hash = statement.hash()?;
let actual_hash = compute_hash_from_proof(&proof);
assert_eq!(expected_hash, actual_hash);
```

#### 4. Memory issues during batch operations

**Cause**: Too many proofs in memory simultaneously.

**Solution**:

```rust
// Process in chunks
for chunk in proofs.chunks(100) {
    verify_batch(chunk)?;
}
```

### Getting Help

- **Documentation**: [docs.nexuszero.io](https://docs.nexuszero.io)
- **GitHub Issues**: [github.com/nexuszero/protocol/issues](https://github.com/nexuszero/protocol/issues)
- **Discord**: [discord.gg/nexuszero](https://discord.gg/nexuszero)

---

## Next Steps

1. **[ZK Proof API Reference](./ZK_PROOF_API.md)** - Complete API documentation
2. **[Circuit DSL Guide](./CIRCUIT_DSL.md)** - Build custom circuits
3. **[Security Audit](../SECURITY_AUDIT.md)** - Security considerations
4. **[Examples](./EXAMPLES.md)** - More code examples
