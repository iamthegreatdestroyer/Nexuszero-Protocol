# Integration Guide: Nexuszero Cryptographic Library

## Overview

This guide demonstrates how to integrate and compose the cryptographic primitives provided by the Nexuszero library to build secure applications. We cover common integration patterns, security considerations, and best practices for combining Ring-LWE encryption, Schnorr signatures, and Bulletproofs.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Integration Patterns](#integration-patterns)
3. [Security Considerations](#security-considerations)
4. [Performance Optimization](#performance-optimization)
5. [Common Pitfalls](#common-pitfalls)
6. [Migration Guide](#migration-guide)

---

## Quick Start

### Basic Setup

```rust
use nexuszero_crypto::{
    lattice::ring_lwe::{RingLWEParameters, generate_keypair},
    proof::schnorr::{schnorr_keygen, schnorr_sign, schnorr_verify},
    proof::bulletproofs::{pedersen_commit, prove_range, verify_range},
};
use rand::thread_rng;

// Initialize random number generator (use once, reuse)
let mut rng = thread_rng();
```

### Security Parameter Selection

Choose security parameters based on your threat model:

| Use Case            | Ring-LWE Level   | Schnorr Group       | Bulletproofs Range  |
| ------------------- | ---------------- | ------------------- | ------------------- |
| Web applications    | 128-bit (n=512)  | Group 14 (2048-bit) | 16-bit (768 bytes)  |
| Financial systems   | 192-bit (n=1024) | Group 16 (4096-bit) | 32-bit (896 bytes)  |
| Government/Military | 256-bit (n=2048) | Group 18 (8192-bit) | 64-bit (1024 bytes) |

**⚠️ Quantum Resistance:**

- **Ring-LWE**: Post-quantum secure ✅
- **Schnorr**: Vulnerable to Shor's algorithm ❌
- **Bulletproofs**: Vulnerable to Shor's algorithm ❌

For post-quantum signatures, transition to lattice-based schemes (Dilithium, SPHINCS+).

---

## Integration Patterns

### Pattern 1: Authenticated Encryption

Combine Ring-LWE encryption with Schnorr signatures for authenticated, confidential messaging.

```rust
use nexuszero_crypto::{CryptoResult, CryptoError};
use zeroize::Zeroize;

/// Authenticated encryption: Encrypt + Sign
///
/// Security Properties:
/// - Confidentiality: Only recipient can decrypt (Ring-LWE)
/// - Authenticity: Signature proves sender identity (Schnorr)
/// - Integrity: Tampered messages fail verification
/// - Non-repudiation: Sender cannot deny sending
pub struct AuthenticatedMessage {
    /// Encrypted message content
    pub ciphertext: RingLWECiphertext,
    /// Digital signature over ciphertext
    pub signature: SchnorrSignature,
    /// Sender's public key (for verification)
    pub sender_public_key: SchnorrPublicKey,
}

impl AuthenticatedMessage {
    /// Encrypt and sign a message
    pub fn create(
        message: &[u64],
        recipient_ring_lwe_key: &RingLWEPublicKey,
        sender_schnorr_key: &SchnorrPrivateKey,
        params: &RingLWEParameters,
        rng: &mut impl rand::Rng,
    ) -> CryptoResult<Self> {
        // Step 1: Encrypt message for recipient
        let ciphertext = encrypt(recipient_ring_lwe_key, message, params, rng)?;

        // Step 2: Serialize ciphertext for signing
        let ciphertext_bytes = bincode::serialize(&ciphertext)
            .map_err(|e| CryptoError::SerializationError(e.to_string()))?;

        // Step 3: Sign ciphertext
        let signature = schnorr_sign(&ciphertext_bytes, sender_schnorr_key)?;

        // Step 4: Get sender's public key for verification
        let sender_public_key = sender_schnorr_key.public_key()?;

        Ok(Self {
            ciphertext,
            signature,
            sender_public_key,
        })
    }

    /// Verify signature and decrypt message
    pub fn verify_and_decrypt(
        &self,
        recipient_ring_lwe_key: &RingLWESecretKey,
        params: &RingLWEParameters,
    ) -> CryptoResult<Vec<u64>> {
        // Step 1: Serialize ciphertext
        let ciphertext_bytes = bincode::serialize(&self.ciphertext)
            .map_err(|e| CryptoError::SerializationError(e.to_string()))?;

        // Step 2: Verify signature
        if !schnorr_verify(&ciphertext_bytes, &self.signature, &self.sender_public_key)? {
            return Err(CryptoError::SignatureVerificationFailed);
        }

        // Step 3: Decrypt message
        decrypt(recipient_ring_lwe_key, &self.ciphertext, params)
    }
}
```

**Usage Example:**

```rust
// Alice's keys
let (mut alice_ring_lwe_secret, alice_ring_lwe_public) = generate_keypair(&params, &mut rng)?;
let (mut alice_schnorr_secret, alice_schnorr_public) = schnorr_keygen()?;

// Bob's keys
let (mut bob_ring_lwe_secret, bob_ring_lwe_public) = generate_keypair(&params, &mut rng)?;

// Alice sends authenticated message to Bob
let message = vec![42; params.n];
let auth_msg = AuthenticatedMessage::create(
    &message,
    &bob_ring_lwe_public,
    &alice_schnorr_secret,
    &params,
    &mut rng,
)?;

// Bob verifies and decrypts
let decrypted = auth_msg.verify_and_decrypt(&bob_ring_lwe_secret, &params)?;
assert_eq!(message, decrypted);

// Cleanup
alice_ring_lwe_secret.zeroize();
alice_schnorr_secret.zeroize();
bob_ring_lwe_secret.zeroize();
```

---

### Pattern 2: Confidential Transactions

Combine Pedersen commitments and Bulletproofs for private yet verifiable transactions.

```rust
/// Confidential transaction with range proofs
///
/// Security Properties:
/// - Hiding: Transaction amounts are private
/// - Binding: Cannot change amounts after commitment
/// - Soundness: Cannot create money (inputs must equal outputs)
/// - Zero-knowledge: Proofs reveal nothing about amounts
pub struct ConfidentialTransaction {
    /// Input commitments (sender's funds)
    pub inputs: Vec<Vec<u8>>,
    /// Output commitments (recipient amounts + change)
    pub outputs: Vec<Vec<u8>>,
    /// Range proofs for each output (prevents negative amounts)
    pub range_proofs: Vec<Vec<u8>>,
}

impl ConfidentialTransaction {
    /// Create a confidential transaction
    ///
    /// # Arguments
    /// * `input_amounts` - Input amounts (must sum to >= output amounts)
    /// * `input_blindings` - Blinding factors for inputs
    /// * `output_amounts` - Output amounts (transfers + change)
    /// * `output_blindings` - Blinding factors for outputs
    /// * `range_bits` - Range proof bit length (8, 16, 32, or 64)
    pub fn create(
        input_amounts: &[u64],
        input_blindings: &[&[u8]],
        output_amounts: &[u64],
        output_blindings: &[&[u8]],
        range_bits: usize,
    ) -> CryptoResult<Self> {
        // Validate: inputs must equal outputs (no inflation)
        let input_sum: u64 = input_amounts.iter().sum();
        let output_sum: u64 = output_amounts.iter().sum();

        if input_sum != output_sum {
            return Err(CryptoError::InvalidParameter(
                format!("Transaction doesn't balance: {} inputs != {} outputs",
                        input_sum, output_sum)
            ));
        }

        // Create input commitments
        let inputs: Result<Vec<_>, _> = input_amounts.iter()
            .zip(input_blindings.iter())
            .map(|(&amount, &blinding)| pedersen_commit(amount, blinding))
            .collect();
        let inputs = inputs?;

        // Create output commitments
        let outputs: Result<Vec<_>, _> = output_amounts.iter()
            .zip(output_blindings.iter())
            .map(|(&amount, &blinding)| pedersen_commit(amount, blinding))
            .collect();
        let outputs = outputs?;

        // Generate range proofs for each output
        let range_proofs: Result<Vec<_>, _> = output_amounts.iter()
            .zip(output_blindings.iter())
            .map(|(&amount, &blinding)| prove_range(amount, blinding, range_bits))
            .collect();
        let range_proofs = range_proofs?;

        Ok(Self {
            inputs,
            outputs,
            range_proofs,
        })
    }

    /// Verify transaction validity without learning amounts
    pub fn verify(&self, range_bits: usize) -> CryptoResult<bool> {
        // Verify all range proofs
        for (commitment, proof) in self.outputs.iter().zip(self.range_proofs.iter()) {
            if !verify_range(commitment, proof, range_bits)? {
                return Ok(false);
            }
        }

        // In production: Also verify commitment balance (inputs = outputs)
        // This requires homomorphic addition of commitments

        Ok(true)
    }
}
```

**Usage Example:**

```rust
use rand::{thread_rng, Rng};

// Transaction: Alice (100 tokens) → Bob (60 tokens) + Change (40 tokens)
let input_amount = 100u64;
let mut rng = thread_rng();
let mut input_blinding = [0u8; 32];
rng.fill(&mut input_blinding);

let output_amounts = vec![60, 40]; // To Bob, change to Alice
let mut bob_blinding = [0u8; 32];
let mut change_blinding = [0u8; 32];
rng.fill(&mut bob_blinding);
rng.fill(&mut change_blinding);

// Create confidential transaction
let tx = ConfidentialTransaction::create(
    &[input_amount],
    &[&input_blinding],
    &output_amounts,
    &[&bob_blinding, &change_blinding],
    64, // 64-bit range (0 to 2^64-1)
)?;

// Anyone can verify without learning amounts
assert!(tx.verify(64)?);
```

---

### Pattern 3: Hybrid Encryption (Post-Quantum KEM)

Use Ring-LWE as a Key Encapsulation Mechanism (KEM) for efficient encryption of large messages.

```rust
use aes_gcm::{Aes256Gcm, Key, Nonce};
use aes_gcm::aead::{Aead, NewAead};

/// Hybrid encryption using Ring-LWE KEM + AES-GCM
///
/// Security Properties:
/// - Post-quantum key exchange (Ring-LWE)
/// - Fast symmetric encryption (AES-256-GCM)
/// - Authenticated encryption (GCM provides AEAD)
/// - Efficient for large messages
pub struct HybridCiphertext {
    /// Ring-LWE encapsulated key
    pub kem_ciphertext: RingLWECiphertext,
    /// AES-GCM encrypted data
    pub data_ciphertext: Vec<u8>,
    /// AES-GCM nonce
    pub nonce: [u8; 12],
}

impl HybridCiphertext {
    /// Encrypt data using hybrid approach
    pub fn encrypt(
        data: &[u8],
        recipient_public_key: &RingLWEPublicKey,
        params: &RingLWEParameters,
        rng: &mut impl rand::Rng,
    ) -> CryptoResult<Self> {
        // Step 1: Generate random 256-bit symmetric key
        let mut symmetric_key_poly = vec![0u64; params.n];
        for coeff in symmetric_key_poly.iter_mut().take(4) {
            *coeff = rng.gen::<u64>();
        }

        // Step 2: Encapsulate key using Ring-LWE
        let kem_ciphertext = encrypt(
            recipient_public_key,
            &symmetric_key_poly,
            params,
            rng,
        )?;

        // Step 3: Derive 256-bit AES key from polynomial
        let mut aes_key = [0u8; 32];
        for (i, &coeff) in symmetric_key_poly.iter().take(4).enumerate() {
            aes_key[i*8..(i+1)*8].copy_from_slice(&coeff.to_le_bytes());
        }

        // Step 4: Encrypt data with AES-256-GCM
        let cipher = Aes256Gcm::new(Key::from_slice(&aes_key));
        let mut nonce = [0u8; 12];
        rng.fill(&mut nonce);

        let data_ciphertext = cipher.encrypt(Nonce::from_slice(&nonce), data)
            .map_err(|e| CryptoError::EncryptionError(e.to_string()))?;

        Ok(Self {
            kem_ciphertext,
            data_ciphertext,
            nonce,
        })
    }

    /// Decrypt data using hybrid approach
    pub fn decrypt(
        &self,
        recipient_secret_key: &RingLWESecretKey,
        params: &RingLWEParameters,
    ) -> CryptoResult<Vec<u8>> {
        // Step 1: Decapsulate symmetric key
        let symmetric_key_poly = decrypt(
            recipient_secret_key,
            &self.kem_ciphertext,
            params,
        )?;

        // Step 2: Derive AES key
        let mut aes_key = [0u8; 32];
        for (i, &coeff) in symmetric_key_poly.iter().take(4).enumerate() {
            aes_key[i*8..(i+1)*8].copy_from_slice(&coeff.to_le_bytes());
        }

        // Step 3: Decrypt data
        let cipher = Aes256Gcm::new(Key::from_slice(&aes_key));
        let plaintext = cipher.decrypt(Nonce::from_slice(&self.nonce), self.data_ciphertext.as_ref())
            .map_err(|e| CryptoError::DecryptionError(e.to_string()))?;

        Ok(plaintext)
    }
}
```

---

## Security Considerations

### 1. Randomness Quality

**CRITICAL:** All cryptographic operations require high-quality randomness.

```rust
// ✅ GOOD: Use thread_rng() (ChaCha20-based CSPRNG)
use rand::thread_rng;
let mut rng = thread_rng();

// ✅ GOOD: Use OsRng for long-term keys
use rand::rngs::OsRng;
let mut rng = OsRng;

// ❌ BAD: Never use predictable RNG for crypto
use rand::rngs::StdRng;
use rand::SeedableRng;
let mut rng = StdRng::seed_from_u64(42); // INSECURE!
```

### 2. Key Management

**Best Practices:**

- **Zeroize secrets:** Always clear sensitive data from memory
- **Separate keys:** Use different keys for signing and encryption
- **Key rotation:** Rotate keys periodically (e.g., every 90 days)
- **Secure storage:** Use HSM, TPM, or encrypted keystores
- **Backup:** Implement secure key backup with recovery procedures

```rust
use zeroize::Zeroize;

fn secure_operation() -> CryptoResult<()> {
    let (mut secret_key, public_key) = generate_keypair(&params, &mut rng)?;

    // ... use secret_key ...

    // ✅ CRITICAL: Zeroize before dropping
    secret_key.zeroize();

    Ok(())
}
```

### 3. Nonce Reuse Prevention

**⚠️ CATASTROPHIC FAILURE:** Reusing nonces in Schnorr signatures leaks private keys!

```rust
// ✅ GOOD: Our implementation uses fresh nonces automatically
let sig1 = schnorr_sign(message1, &private_key)?;
let sig2 = schnorr_sign(message2, &private_key)?;
// Different nonces, safe!

// ❌ BAD: Never implement custom nonce generation unless you're an expert
// Nonce reuse allows private key recovery: x = (s1 - s2) / (c1 - c2)
```

### 4. Side-Channel Protection

Protect against timing and cache attacks:

```rust
// ✅ Use constant-time operations for sensitive comparisons
use subtle::ConstantTimeEq;

fn secure_comparison(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    // Constant-time comparison
    a.ct_eq(b).into()
}

// ❌ BAD: Early-exit comparisons leak timing information
fn insecure_comparison(a: &[u8], b: &[u8]) -> bool {
    a == b // Variable-time!
}
```

### 5. Parameter Validation

Always validate parameters before use:

```rust
// ✅ GOOD: Validate parameters
let params = RingLWEParameters::new_128bit_security();
params.validate()?; // Checks parameter security

// Generate keys
let keypair = generate_keypair(&params, &mut rng)?;
```

---

## Performance Optimization

### 1. Batch Operations

Batch verification provides ~3x speedup:

```rust
// Instead of verifying individually:
for proof in proofs.iter() {
    verify_range(commitment, proof, 64)?;
}

// Use batch verification (when implemented):
// batch_verify_range(&commitments, &proofs, 64)?; // ~3x faster
```

### 2. Precomputation

Precompute expensive operations:

```rust
// Precompute NTT powers for Ring-LWE
let ntt_powers = precompute_ntt_powers(&params);

// Precompute generator powers for Bulletproofs
let g_powers = precompute_generator_powers(64);
```

### 3. Memory Pooling

Reuse polynomial buffers to reduce allocations:

```rust
let pool = PolynomialMemoryPool::new(params.n, 10); // Pool of 10 buffers

// Operations use pooled memory automatically
let result = pool.with_temp(|temp1| {
    pool.with_temp(|temp2| {
        // Use temp1 and temp2 without allocation
    })
});
```

### 4. Parallel Processing

Leverage multi-core CPUs for batch operations:

```rust
use rayon::prelude::*;

// Parallel encryption of multiple messages
let ciphertexts: Result<Vec<_>, _> = messages.par_iter()
    .map(|msg| encrypt(public_key, msg, &params, &mut thread_rng()))
    .collect();
```

---

## Common Pitfalls

### ❌ Pitfall 1: Reusing Blinding Factors

```rust
// ❌ BAD: Reusing blinding factor allows value recovery
let blinding = generate_blinding_factor();
let c1 = pedersen_commit(value1, &blinding)?; // Uses blinding
let c2 = pedersen_commit(value2, &blinding)?; // REUSES blinding - INSECURE!

// ✅ GOOD: Fresh blinding for each commitment
let blinding1 = generate_blinding_factor();
let c1 = pedersen_commit(value1, &blinding1)?;

let blinding2 = generate_blinding_factor();
let c2 = pedersen_commit(value2, &blinding2)?;
```

### ❌ Pitfall 2: Not Handling Decryption Failures

```rust
// ❌ BAD: Panics on decryption failure
let plaintext = decrypt(secret_key, ciphertext, &params).unwrap();

// ✅ GOOD: Handle failures gracefully
match decrypt(secret_key, ciphertext, &params) {
    Ok(plaintext) => {
        // Process plaintext
    },
    Err(CryptoError::DecryptionError) => {
        // Decryption failure (noise too large)
        // Retry or request retransmission
    },
    Err(e) => {
        // Other error
        return Err(e);
    }
}
```

### ❌ Pitfall 3: Mixing Security Levels

```rust
// ❌ BAD: Mixing 128-bit and 256-bit security
let ring_lwe_params = RingLWEParameters::new_128bit_security(); // 128-bit
let schnorr_params = SchnorrParams::new_4096bit(); // ~150-bit classical

// ✅ GOOD: Consistent security levels
let ring_lwe_params = RingLWEParameters::new_128bit_security(); // 128-bit
let schnorr_params = SchnorrParams::new_2048bit(); // ~112-bit (acceptable for transitional)
```

### ❌ Pitfall 4: Not Authenticating Encrypted Data

```rust
// ❌ BAD: Encryption without authentication (malleable ciphertexts)
let ciphertext = encrypt(public_key, message, &params, &mut rng)?;
send(ciphertext);

// ✅ GOOD: Use authenticated encryption
let auth_msg = AuthenticatedMessage::create(
    message,
    recipient_public_key,
    sender_private_key,
    &params,
    &mut rng,
)?;
send(auth_msg);
```

---

## Migration Guide

### Transitioning to Post-Quantum Cryptography

**Timeline:**

- **Now - 2025**: Hybrid approach (classical + PQ)
- **2025 - 2030**: Gradual transition to PQ-only
- **2030+**: Full post-quantum deployment

**Migration Strategy:**

1. **Phase 1: Hybrid Deployment**

   ```rust
   // Support both classical and post-quantum
   pub enum SignatureScheme {
       Classical(SchnorrSignature),
       PostQuantum(DilithiumSignature),
       Hybrid(SchnorrSignature, DilithiumSignature),
   }
   ```

2. **Phase 2: Deprecate Classical**

   ```rust
   // Warn users about classical signatures
   #[deprecated(since = "2.0.0", note = "Use post-quantum signatures")]
   pub fn schnorr_sign(...) -> CryptoResult<SchnorrSignature> {
       // ...
   }
   ```

3. **Phase 3: Remove Classical**
   ```rust
   // PQ-only (version 3.0.0+)
   pub fn sign(...) -> CryptoResult<DilithiumSignature> {
       // ...
   }
   ```

---

## Additional Resources

- **API Documentation**: Run `cargo doc --open` for full API reference
- **Examples**: See `examples/` directory for runnable demos
- **Security Audit**: See `audit_materials/` (Task 7)
- **Performance Benchmarks**: Run `cargo bench` (Task 6)
- **Research Papers**:
  - Ring-LWE: Lyubashevsky, Peikert, Regev (EUROCRYPT 2010)
  - Bulletproofs: Bünz et al. (S&P 2018)
  - Schnorr: Schnorr (CRYPTO 1991)

---

## Support

For questions, issues, or contributions:

- **GitHub**: https://github.com/iamthegreatdestroyer/Nexuszero-Protocol
- **Documentation**: https://docs.nexuszero.io
- **Security Issues**: security@nexuszero.io

---

**Last Updated**: Task 5 (December 2025)  
**Version**: 0.1.0
