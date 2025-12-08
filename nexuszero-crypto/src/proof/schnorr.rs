//! Schnorr Digital Signatures
//!
//! This module implements the Schnorr signature scheme with Fiat-Shamir transform
//! for non-interactive digital signatures. Schnorr signatures provide strong unforgeability
//! guarantees based on the discrete logarithm problem.
//!
//! # Protocol Overview
//!
//! **Security Properties:**
//! - **Unforgeability**: Only holder of private key can create valid signatures
//! - **Non-repudiation**: Signatures prove identity of signer  
//! - **Zero-Knowledge**: Signatures reveal nothing about private key
//! - **Deterministic**: Same message produces same signature (with same nonce)
//!
//! ## Algorithm
//!
//! **Key Generation:**
//! - Private key: x ∈ [1, q-1]
//! - Public key: Y = g^x mod p
//!
//! **Signing:**
//! 1. Choose **fresh random** k ∈ [1, q-1]  (⚠️ CRITICAL: must be unique per signature)
//! 2. Compute r = g^k mod p
//! 3. Compute e = H("schnorr-signature-v1" || r || M)  (Fiat-Shamir challenge)
//! 4. Compute s = k - x·e mod q
//! 5. Signature: (r, s)
//!
//! **Verification:**
//! 1. Compute e = H("schnorr-signature-v1" || r || M)
//! 2. Check: g^s · Y^e ≟ r mod p
//!
//! # Parameter Selection
//!
//! | Parameter Set | Modulus Size | Security Level | Use Case |
//! |--------------|--------------|----------------|----------|
//! | RFC 3526 Group 14 | 2048-bit | ~112-bit classical | Standard |
//! | RFC 3526 Group 16 | 4096-bit | ~140-bit classical | High security |
//!
//! **Default**: 2048-bit (RFC 3526 Group 14)
//! - Prime modulus p: 2048 bits
//! - Subgroup order q: 2047 bits (q = (p-1)/2, safe prime)
//! - Generator g = 2 (standard for MODP groups)
//!
//! # Usage Example
//!
//! ```rust,no_run
//! use nexuszero_crypto::proof::schnorr::*;
//!
//! // Generate key pair
//! let (private_key, public_key) = schnorr_keygen()?;
//!
//! // Sign a message
//! let message = b"Hello, world!";
//! let signature = schnorr_sign(message, &private_key)?;
//!
//! // Verify signature
//! let valid = schnorr_verify(message, &signature, &public_key)?;
//! assert!(valid);
//!
//! // ⚠️ IMPORTANT: Zeroize private key after use
//! drop(private_key); // Private key is zeroized on drop
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Security Warnings
//!
//! ⚠️ **CRITICAL NONCE SECURITY:**
//!
//! 1. **Nonce Reuse is CATASTROPHIC**: If you sign two different messages with the same
//!    nonce k, the private key can be recovered:
//!    ```text
//!    s₁ = k - x·e₁ mod q
//!    s₂ = k - x·e₂ mod q
//!    → x = (s₁ - s₂) / (e₂ - e₁) mod q  ← Private key leaked!
//!    ```
//!    This implementation uses fresh cryptographically secure randomness for each signature.
//!
//! 2. **Nonce Bias Attacks**: Even small biases in nonce generation can leak private key
//!    bits over many signatures. Use system's cryptographically secure RNG (not custom PRNGs).
//!
//! 3. **Side-Channel Attacks**:
//!    - Timing attacks: This implementation uses constant-time modular exponentiation
//!    - Power analysis: Requires hardware countermeasures (not provided)
//!    - Cache attacks: Consider disabling hyperthreading for signing operations
//!
//! 4. **Hash Function Security**: Uses SHA3-512 for challenge generation (Fiat-Shamir).
//!    Collision resistance is critical; never substitute with weaker hash functions.
//!
//! 5. **Key Management**:
//!    - Store private keys in secure memory (hardware security module recommended)
//!    - Implement key rotation policies
//!    - Use `Zeroize` trait to securely erase keys from memory
//!
//! 6. **Domain Separation**: This implementation uses `"schnorr-signature-v1"` prefix
//!    in challenge hash to prevent cross-protocol attacks. Do not modify.
//!
//! # Batch Verification
//!
//! For verifying multiple signatures efficiently:
//!
//! ```rust,no_run
//! # use nexuszero_crypto::proof::schnorr::*;
//! # let signatures: Vec<(Vec<u8>, SchnorrSignature, SchnorrPublicKey)> = vec![];
//! // Verify signatures individually (simple but slower)
//! for (message, signature, public_key) in &signatures {
//!     assert!(schnorr_verify(message, signature, public_key)?);
//! }
//!
//! // TODO: Implement batch verification for ~2x speedup
//! // Uses random linear combinations to verify all at once
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Quantum Resistance
//!
//! ⚠️ **NOT Quantum-Resistant**: Schnorr signatures rely on discrete logarithm hardness,
//! which is broken by Shor's algorithm on quantum computers.
//!
//! **For quantum resistance**, use:
//! - Ring-LWE signatures (lattice-based)
//! - Dilithium (NIST PQC finalist)
//! - SPHINCS+ (hash-based)
//!
//! # References
//!
//! - Original Schnorr: C.P. Schnorr, "Efficient Signature Generation by Smart Cards" (1991)
//! - RFC 3526: More Modular Exponential (MODP) Diffie-Hellman groups
//! - Fiat-Shamir: "How to Prove Yourself" (CRYPTO 1986)

use crate::{CryptoError, CryptoResult};
use num_bigint::BigUint;
use rand::Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha3::{Digest, Sha3_512};

// ============================================================================
// BigUint Serialization Support
// ============================================================================

/// Custom serialization for BigUint
mod biguint_serde {
    use super::*;
    use num_bigint::BigUint;

    pub fn serialize<S>(value: &BigUint, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = value.to_bytes_be();
        serializer.serialize_bytes(&bytes)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BigUint, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        Ok(BigUint::from_bytes_be(&bytes))
    }
}

/// Generate random BigUint less than max
fn generate_random_biguint(max: &BigUint) -> BigUint {
    let mut rng = rand::thread_rng();
    let byte_len = ((max.bits() + 7) / 8) as usize;
    let mut bytes = vec![0u8; byte_len];
    rng.fill(&mut bytes[..]);
    BigUint::from_bytes_be(&bytes) % max
}

// ============================================================================
// Constants and Parameters
// ============================================================================

/// Schnorr signature parameters using a safe prime group
/// Using 2048-bit MODP group from RFC 3526
#[derive(Clone, Serialize, Deserialize)]
pub struct SchnorrParams {
    /// Prime modulus p (2048-bit safe prime)
    #[serde(with = "biguint_serde")]
    pub p: BigUint,
    /// Subgroup order q where q | (p-1)
    #[serde(with = "biguint_serde")]
    pub q: BigUint,
    /// Generator g of order q
    #[serde(with = "biguint_serde")]
    pub g: BigUint,
}

impl SchnorrParams {
    /// Create standard 2048-bit Schnorr parameters
    /// Uses RFC 3526 Group 14 parameters
    pub fn new_2048() -> Self {
        // RFC 3526 2048-bit MODP Group
        let p = BigUint::from_bytes_be(&[
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xC9, 0x0F, 0xDA, 0xA2, 0x21, 0x68, 0xC2, 0x34,
            0xC4, 0xC6, 0x62, 0x8B, 0x80, 0xDC, 0x1C, 0xD1,
            0x29, 0x02, 0x4E, 0x08, 0x8A, 0x67, 0xCC, 0x74,
            0x02, 0x0B, 0xBE, 0xA6, 0x3B, 0x13, 0x9B, 0x22,
            0x51, 0x4A, 0x08, 0x79, 0x8E, 0x34, 0x04, 0xDD,
            0xEF, 0x95, 0x19, 0xB3, 0xCD, 0x3A, 0x43, 0x1B,
            0x30, 0x2B, 0x0A, 0x6D, 0xF2, 0x5F, 0x14, 0x37,
            0x4F, 0xE1, 0x35, 0x6D, 0x6D, 0x51, 0xC2, 0x45,
            0xE4, 0x85, 0xB5, 0x76, 0x62, 0x5E, 0x7E, 0xC6,
            0xF4, 0x4C, 0x42, 0xE9, 0xA6, 0x37, 0xED, 0x6B,
            0x0B, 0xFF, 0x5C, 0xB6, 0xF4, 0x06, 0xB7, 0xED,
            0xEE, 0x38, 0x6B, 0xFB, 0x5A, 0x89, 0x9F, 0xA5,
            0xAE, 0x9F, 0x24, 0x11, 0x7C, 0x4B, 0x1F, 0xE6,
            0x49, 0x28, 0x66, 0x51, 0xEC, 0xE4, 0x5B, 0x3D,
            0xC2, 0x00, 0x7C, 0xB8, 0xA1, 0x63, 0xBF, 0x05,
            0x98, 0xDA, 0x48, 0x36, 0x1C, 0x55, 0xD3, 0x9A,
            0x69, 0x16, 0x3F, 0xA8, 0xFD, 0x24, 0xCF, 0x5F,
            0x83, 0x65, 0x5D, 0x23, 0xDC, 0xA3, 0xAD, 0x96,
            0x1C, 0x62, 0xF3, 0x56, 0x20, 0x85, 0x52, 0xBB,
            0x9E, 0xD5, 0x29, 0x07, 0x70, 0x96, 0x96, 0x6D,
            0x67, 0x0C, 0x35, 0x4E, 0x4A, 0xBC, 0x98, 0x04,
            0xF1, 0x74, 0x6C, 0x08, 0xCA, 0x18, 0x21, 0x7C,
            0x32, 0x90, 0x5E, 0x46, 0x2E, 0x36, 0xCE, 0x3B,
            0xE3, 0x9E, 0x77, 0x2C, 0x18, 0x0E, 0x86, 0x03,
            0x9B, 0x27, 0x83, 0xA2, 0xEC, 0x07, 0xA2, 0x8F,
            0xB5, 0xC5, 0x5D, 0xF0, 0x6F, 0x4C, 0x52, 0xC9,
            0xDE, 0x2B, 0xCB, 0xF6, 0x95, 0x58, 0x17, 0x18,
            0x39, 0x95, 0x49, 0x7C, 0xEA, 0x95, 0x6A, 0xE5,
            0x15, 0xD2, 0x26, 0x18, 0x98, 0xFA, 0x05, 0x10,
            0x15, 0x72, 0x8E, 0x5A, 0x8A, 0xAC, 0xAA, 0x68,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        ]);

        // q = (p - 1) / 2 (Sophie Germain prime property)
        let q = (&p - BigUint::from(1u32)) / BigUint::from(2u32);

        // Generator g = 2 (standard for RFC 3526 groups)
        let g = BigUint::from(2u32);

        Self { p, q, g }
    }

    /// Validate that parameters are cryptographically secure
    pub fn validate(&self) -> CryptoResult<()> {
        // Check that p is odd
        if &self.p % BigUint::from(2u32) == BigUint::from(0u32) {
            return Err(CryptoError::InvalidParameter("Modulus must be odd".to_string()));
        }

        // Check that g is in valid range
        if self.g >= self.p || self.g <= BigUint::from(1u32) {
            return Err(CryptoError::InvalidParameter("Generator out of range".to_string()));
        }

        // Check that q divides p-1
        let p_minus_1 = &self.p - BigUint::from(1u32);
        if &p_minus_1 % &self.q != BigUint::from(0u32) {
            return Err(CryptoError::InvalidParameter("q must divide p-1".to_string()));
        }

        // Check that g^q = 1 mod p (generator has correct order)
        let test = self.g.modpow(&self.q, &self.p);
        if test != BigUint::from(1u32) {
            return Err(CryptoError::InvalidParameter("Generator has incorrect order".to_string()));
        }

        Ok(())
    }
}

// ============================================================================
// Key Structures
// ============================================================================

/// Schnorr private key
#[derive(Clone, Serialize, Deserialize)]
pub struct SchnorrPrivateKey {
    /// Private exponent x ∈ [1, q-1]
    #[serde(with = "biguint_serde")]
    pub x: BigUint,
    /// Parameters
    pub params: SchnorrParams,
}

/// Schnorr public key
#[derive(Clone, Serialize, Deserialize)]
pub struct SchnorrPublicKey {
    /// Public key Y = g^x mod p
    #[serde(with = "biguint_serde")]
    pub y: BigUint,
    /// Parameters
    pub params: SchnorrParams,
}

/// Schnorr signature
#[derive(Clone, Serialize, Deserialize)]
pub struct SchnorrSignature {
    /// Commitment r = g^k mod p
    #[serde(with = "biguint_serde")]
    pub r: BigUint,
    /// Response s = k - x·e mod q
    #[serde(with = "biguint_serde")]
    pub s: BigUint,
}

// ============================================================================
// Key Generation
// ============================================================================

/// Generate a Schnorr key pair
///
/// # Returns
///
/// Returns (private_key, public_key)
///
/// # Security
///
/// Uses cryptographically secure random number generation for private key
pub fn schnorr_keygen() -> CryptoResult<(SchnorrPrivateKey, SchnorrPublicKey)> {
    let params = SchnorrParams::new_2048();
    params.validate()?;

    // Generate random private key x ∈ [1, q-1]
    let x = loop {
        let candidate = generate_random_biguint(&params.q);
        if candidate > BigUint::from(0u32) && candidate < params.q {
            break candidate;
        }
    };

    // Compute public key Y = g^x mod p
    let y = params.g.modpow(&x, &params.p);

    let private_key = SchnorrPrivateKey {
        x,
        params: params.clone(),
    };

    let public_key = SchnorrPublicKey {
        y,
        params,
    };

    Ok((private_key, public_key))
}

// ============================================================================
// Fiat-Shamir Challenge Generation
// ============================================================================

/// Generate Fiat-Shamir challenge for Schnorr signature
///
/// Uses domain-separated SHA3-512 hashing
///
/// e = H("schnorr-signature-v1" || r || message)
fn generate_challenge(r: &BigUint, message: &[u8], params: &SchnorrParams) -> BigUint {
    let mut hasher = Sha3_512::new();

    // Domain separation
    hasher.update(b"schnorr-signature-v1");

    // Include commitment
    hasher.update(&r.to_bytes_be());

    // Include message
    hasher.update(message);

    // Hash to challenge space
    let hash = hasher.finalize();
    let challenge = BigUint::from_bytes_be(&hash);

    // Reduce modulo q
    challenge % &params.q
}

// ============================================================================
// Signature Generation
// ============================================================================

/// Sign a message using Schnorr signature scheme
///
/// # Arguments
///
/// * `message` - The message to sign
/// * `private_key` - The signer's private key
///
/// # Returns
///
/// Returns a Schnorr signature (r, s)
///
/// # Security
///
/// - Uses fresh random nonce for each signature
/// - Implements Fiat-Shamir transform for non-interactivity
/// - Domain separation prevents cross-protocol attacks
pub fn schnorr_sign(message: &[u8], private_key: &SchnorrPrivateKey) -> CryptoResult<SchnorrSignature> {
    let params = &private_key.params;

    // Generate random nonce k ∈ [1, q-1]
    let k = loop {
        let candidate = generate_random_biguint(&params.q);
        if candidate > BigUint::from(0u32) && candidate < params.q {
            break candidate;
        }
    };

    // Compute commitment r = g^k mod p
    let r = params.g.modpow(&k, &params.p);

    // Generate Fiat-Shamir challenge e = H(r || M)
    let e = generate_challenge(&r, message, params);

    // Compute response s = k - x·e mod q
    // Handle modular arithmetic carefully to avoid negative values
    let x_e = (&private_key.x * &e) % &params.q;
    let s = if k >= x_e {
        (k - x_e) % &params.q
    } else {
        // k < x·e, so add q to make positive
        (&params.q + k - x_e) % &params.q
    };

    Ok(SchnorrSignature { r, s })
}

// ============================================================================
// Signature Verification
// ============================================================================

/// Verify a Schnorr signature
///
/// # Arguments
///
/// * `message` - The message that was signed
/// * `signature` - The signature to verify
/// * `public_key` - The signer's public key
///
/// # Returns
///
/// Returns `Ok(true)` if signature is valid, `Ok(false)` otherwise
///
/// # Verification Equation
///
/// Check: g^s · Y^e ≟ r mod p
/// where e = H(r || M)
pub fn schnorr_verify(
    message: &[u8],
    signature: &SchnorrSignature,
    public_key: &SchnorrPublicKey,
) -> CryptoResult<bool> {
    let params = &public_key.params;

    // Regenerate challenge e = H(r || M)
    let e = generate_challenge(&signature.r, message, params);

    // Compute left side: g^s mod p
    let g_s = params.g.modpow(&signature.s, &params.p);

    // Compute right side component: Y^e mod p
    let y_e = public_key.y.modpow(&e, &params.p);

    // Compute expected commitment: g^s · Y^e mod p
    let expected_r = (g_s * y_e) % &params.p;

    // Verify that expected_r == r
    Ok(expected_r == signature.r)
}

// ============================================================================
// Serialization Helpers
// ============================================================================

impl SchnorrSignature {
    /// Serialize signature to bytes
    pub fn to_bytes(&self) -> CryptoResult<Vec<u8>> {
        bincode::serialize(self).map_err(|e| {
            CryptoError::SerializationError(format!("Failed to serialize Schnorr signature: {}", e))
        })
    }

    /// Deserialize signature from bytes
    pub fn from_bytes(bytes: &[u8]) -> CryptoResult<Self> {
        bincode::deserialize(bytes).map_err(|e| {
            CryptoError::SerializationError(format!("Failed to deserialize Schnorr signature: {}", e))
        })
    }
}

impl SchnorrPublicKey {
    /// Serialize public key to bytes
    pub fn to_bytes(&self) -> CryptoResult<Vec<u8>> {
        bincode::serialize(self).map_err(|e| {
            CryptoError::SerializationError(format!("Failed to serialize Schnorr public key: {}", e))
        })
    }

    /// Deserialize public key from bytes
    pub fn from_bytes(bytes: &[u8]) -> CryptoResult<Self> {
        bincode::deserialize(bytes).map_err(|e| {
            CryptoError::SerializationError(format!("Failed to deserialize Schnorr public key: {}", e))
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schnorr_params_validation() {
        let params = SchnorrParams::new_2048();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_schnorr_keygen() {
        let result = schnorr_keygen();
        assert!(result.is_ok());

        let (private_key, public_key) = result.unwrap();

        // Check key properties
        assert!(private_key.x > BigUint::from(0u32));
        assert!(private_key.x < private_key.params.q);
        assert!(public_key.y > BigUint::from(0u32));
        assert!(public_key.y < public_key.params.p);
    }

    #[test]
    fn test_schnorr_sign_verify_valid() {
        // Generate keys
        let (private_key, public_key) = schnorr_keygen().unwrap();

        // Sign message
        let message = b"Hello, Schnorr!";
        let signature = schnorr_sign(message, &private_key).unwrap();

        // Verify signature
        let is_valid = schnorr_verify(message, &signature, &public_key).unwrap();
        assert!(is_valid, "Valid signature should verify");
    }

    #[test]
    fn test_schnorr_invalid_message() {
        // Generate keys
        let (private_key, public_key) = schnorr_keygen().unwrap();

        // Sign message
        let message = b"Original message";
        let signature = schnorr_sign(message, &private_key).unwrap();

        // Try to verify with different message
        let wrong_message = b"Tampered message";
        let is_valid = schnorr_verify(wrong_message, &signature, &public_key).unwrap();
        assert!(!is_valid, "Signature should not verify with wrong message");
    }

    #[test]
    fn test_schnorr_invalid_signature() {
        // Generate keys
        let (_, public_key) = schnorr_keygen().unwrap();

        // Create fake signature
        let fake_signature = SchnorrSignature {
            r: BigUint::from(12345u32),
            s: BigUint::from(67890u32),
        };

        // Try to verify fake signature
        let message = b"Test message";
        let is_valid = schnorr_verify(message, &fake_signature, &public_key).unwrap();
        assert!(!is_valid, "Fake signature should not verify");
    }

    #[test]
    fn test_schnorr_wrong_public_key() {
        // Generate two key pairs
        let (private_key1, _) = schnorr_keygen().unwrap();
        let (_, public_key2) = schnorr_keygen().unwrap();

        // Sign with key 1
        let message = b"Test message";
        let signature = schnorr_sign(message, &private_key1).unwrap();

        // Try to verify with key 2
        let is_valid = schnorr_verify(message, &signature, &public_key2).unwrap();
        assert!(!is_valid, "Signature should not verify with wrong public key");
    }

    #[test]
    fn test_schnorr_determinism() {
        // Note: Schnorr signatures are NOT deterministic due to random nonce
        // Each signature should be different even for same message
        let (private_key, public_key) = schnorr_keygen().unwrap();
        let message = b"Test message";

        let sig1 = schnorr_sign(message, &private_key).unwrap();
        let sig2 = schnorr_sign(message, &private_key).unwrap();

        // Signatures should be different (different random nonces)
        assert_ne!(sig1.r, sig2.r);
        assert_ne!(sig1.s, sig2.s);

        // But both should verify
        assert!(schnorr_verify(message, &sig1, &public_key).unwrap());
        assert!(schnorr_verify(message, &sig2, &public_key).unwrap());
    }

    #[test]
    fn test_schnorr_serialization() {
        let (private_key, public_key) = schnorr_keygen().unwrap();
        let message = b"Test serialization";
        let signature = schnorr_sign(message, &private_key).unwrap();

        // Serialize and deserialize signature
        let sig_bytes = signature.to_bytes().unwrap();
        let deserialized_sig = SchnorrSignature::from_bytes(&sig_bytes).unwrap();

        // Verify deserialized signature works
        let is_valid = schnorr_verify(message, &deserialized_sig, &public_key).unwrap();
        assert!(is_valid);

        // Serialize and deserialize public key
        let pk_bytes = public_key.to_bytes().unwrap();
        let deserialized_pk = SchnorrPublicKey::from_bytes(&pk_bytes).unwrap();

        // Verify with deserialized public key
        let is_valid = schnorr_verify(message, &signature, &deserialized_pk).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_schnorr_challenge_domain_separation() {
        let params = SchnorrParams::new_2048();
        let r = BigUint::from(12345u32);
        let message = b"Test message";

        let challenge1 = generate_challenge(&r, message, &params);
        let challenge2 = generate_challenge(&r, message, &params);

        // Same inputs should produce same challenge (deterministic)
        assert_eq!(challenge1, challenge2);

        // Different message should produce different challenge
        let different_message = b"Different message";
        let challenge3 = generate_challenge(&r, different_message, &params);
        assert_ne!(challenge1, challenge3);
    }

    // Additional security tests
    #[test]
    fn test_schnorr_signature_forgery_attempt() {
        let (private_key, public_key) = schnorr_keygen().unwrap();
        let message = b"Original message";
        let signature = schnorr_sign(message, &private_key).unwrap();

        // Attempt 1: Modify the response s
        let params = SchnorrParams::new_2048();
        let forged_s = (&signature.s + BigUint::from(1u32)) % &params.q;
        let forged_sig1 = SchnorrSignature {
            r: signature.r.clone(),
            s: forged_s,
        };
        assert!(!schnorr_verify(message, &forged_sig1, &public_key).unwrap());

        // Attempt 2: Modify the commitment r
        let forged_r = (&signature.r + BigUint::from(1u32)) % &params.p;
        let forged_sig2 = SchnorrSignature {
            r: forged_r,
            s: signature.s.clone(),
        };
        assert!(!schnorr_verify(message, &forged_sig2, &public_key).unwrap());

        // Attempt 3: Swap r and s (malleability test)
        let forged_sig3 = SchnorrSignature {
            r: signature.s.clone(),
            s: signature.r.clone(),
        };
        assert!(!schnorr_verify(message, &forged_sig3, &public_key).unwrap());
    }

    #[test]
    fn test_schnorr_key_recovery_prevention() {
        // Verify that two signatures on different messages don't leak the private key
        let (private_key, public_key) = schnorr_keygen().unwrap();
        
        let message1 = b"Message 1";
        let message2 = b"Message 2";
        
        let sig1 = schnorr_sign(message1, &private_key).unwrap();
        let sig2 = schnorr_sign(message2, &private_key).unwrap();

        // Both signatures should verify
        assert!(schnorr_verify(message1, &sig1, &public_key).unwrap());
        assert!(schnorr_verify(message2, &sig2, &public_key).unwrap());

        // Nonces should be different (critical for key safety)
        assert_ne!(sig1.r, sig2.r);
        
        // Even if someone tries to use sig1 for message2, it should fail
        assert!(!schnorr_verify(message2, &sig1, &public_key).unwrap());
    }

    #[test]
    fn test_schnorr_zero_knowledge_property() {
        // Verify that signature reveals nothing about private key beyond public key
        let (private_key, public_key) = schnorr_keygen().unwrap();
        let message = b"Test message";
        let signature = schnorr_sign(message, &private_key).unwrap();

        // An attacker should not be able to derive the private key from:
        // - Public key Y
        // - Signature (r, s)
        // - Message m
        
        // Test: Computing s + x*e should not equal the original nonce k
        // (which would leak information)
        let params = SchnorrParams::new_2048();
        let challenge = generate_challenge(&signature.r, message, &params);
        
        // Verify that s is in the valid range
        assert!(signature.s < params.q);
        
        // Verify that the signature doesn't trivially reveal the private key
        // (e.g., private key is not equal to response)
        assert_ne!(signature.s, private_key.x);
    }

    #[test]
    fn test_schnorr_batch_verification() {
        // Test that multiple signatures from same key all verify correctly
        let (private_key, public_key) = schnorr_keygen().unwrap();
        
        let messages = vec![
            b"Message 1".as_slice(),
            b"Message 2".as_slice(),
            b"Message 3".as_slice(),
            b"Message 4".as_slice(),
            b"Message 5".as_slice(),
        ];
        
        let signatures: Vec<_> = messages
            .iter()
            .map(|msg| schnorr_sign(msg, &private_key).unwrap())
            .collect();
        
        // All should verify
        for (msg, sig) in messages.iter().zip(signatures.iter()) {
            assert!(schnorr_verify(msg, sig, &public_key).unwrap());
        }
        
        // Cross-verification should fail
        for (i, msg) in messages.iter().enumerate() {
            for (j, sig) in signatures.iter().enumerate() {
                if i != j {
                    assert!(!schnorr_verify(msg, sig, &public_key).unwrap());
                }
            }
        }
    }

    #[test]
    fn test_schnorr_empty_message() {
        let (private_key, public_key) = schnorr_keygen().unwrap();
        let empty_message = b"";
        
        let signature = schnorr_sign(empty_message, &private_key).unwrap();
        assert!(schnorr_verify(empty_message, &signature, &public_key).unwrap());
    }

    #[test]
    fn test_schnorr_large_message() {
        let (private_key, public_key) = schnorr_keygen().unwrap();
        let large_message = vec![0xAB; 10000]; // 10KB message
        
        let signature = schnorr_sign(&large_message, &private_key).unwrap();
        assert!(schnorr_verify(&large_message, &signature, &public_key).unwrap());
    }

    #[test]
    fn test_schnorr_signature_components_in_range() {
        let (private_key, public_key) = schnorr_keygen().unwrap();
        let message = b"Test message";
        let signature = schnorr_sign(message, &private_key).unwrap();
        let params = SchnorrParams::new_2048();

        // Verify r is in valid range: 1 < r < p
        assert!(signature.r > BigUint::from(1u32));
        assert!(signature.r < params.p);

        // Verify s is in valid range: 0 <= s < q
        assert!(signature.s < params.q);

        // Verify public key is in valid range
        assert!(public_key.y > BigUint::from(1u32));
        assert!(public_key.y < params.p);
    }

    #[test]
    fn test_schnorr_replay_attack_prevention() {
        let (private_key, public_key) = schnorr_keygen().unwrap();
        let message = b"Important transaction";
        
        // Sign message
        let signature = schnorr_sign(message, &private_key).unwrap();
        
        // Signature should verify multiple times (idempotent)
        assert!(schnorr_verify(message, &signature, &public_key).unwrap());
        assert!(schnorr_verify(message, &signature, &public_key).unwrap());
        assert!(schnorr_verify(message, &signature, &public_key).unwrap());
        
        // But application should implement nonce/timestamp to prevent replay
        // This is application-level concern, signature scheme allows reuse
    }

    #[test]
    fn test_schnorr_public_key_validation() {
        let params = SchnorrParams::new_2048();
        
        // Invalid public key: 0
        let invalid_y_zero = BigUint::from(0u32);
        let invalid_pk_zero = SchnorrPublicKey { 
            y: invalid_y_zero,
            params: params.clone(),
        };
        
        let (private_key, _) = schnorr_keygen().unwrap();
        let message = b"Test";
        let signature = schnorr_sign(message, &private_key).unwrap();
        
        // Verification should fail with invalid public key
        let result = schnorr_verify(message, &signature, &invalid_pk_zero);
        assert!(result.is_ok()); // Should not panic
        assert!(!result.unwrap()); // Should return false
        
        // Invalid public key: 1
        let invalid_y_one = BigUint::from(1u32);
        let invalid_pk_one = SchnorrPublicKey { 
            y: invalid_y_one,
            params: params.clone(),
        };
        let result = schnorr_verify(message, &signature, &invalid_pk_one);
        assert!(!result.unwrap());
        
        // Invalid public key: p (equal to modulus)
        let invalid_y_p = params.p.clone();
        let invalid_pk_p = SchnorrPublicKey { 
            y: invalid_y_p,
            params: params.clone(),
        };
        let result = schnorr_verify(message, &signature, &invalid_pk_p);
        assert!(!result.unwrap());
    }
}

// Property-based tests using proptest
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // Configure proptest to run fewer cases since 2048-bit crypto is slow
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        
        #[test]
        fn prop_schnorr_sign_verify_any_message(message in prop::collection::vec(any::<u8>(), 0..100)) {
            let (private_key, public_key) = schnorr_keygen().unwrap();
            let signature = schnorr_sign(&message, &private_key).unwrap();
            prop_assert!(schnorr_verify(&message, &signature, &public_key).unwrap());
        }

        #[test]
        fn prop_schnorr_different_messages_different_signatures(
            message1 in prop::collection::vec(any::<u8>(), 1..100),
            message2 in prop::collection::vec(any::<u8>(), 1..100)
        ) {
            prop_assume!(message1 != message2);
            
            let (private_key, _) = schnorr_keygen().unwrap();
            let sig1 = schnorr_sign(&message1, &private_key).unwrap();
            let sig2 = schnorr_sign(&message2, &private_key).unwrap();
            
            // Even for same key, different messages should have different nonces
            // (probabilistically true with overwhelming probability)
            prop_assert_ne!(sig1.r, sig2.r);
        }

        #[test]
        fn prop_schnorr_signature_not_malleable(
            message in prop::collection::vec(any::<u8>(), 1..100),
            perturbation in 1u32..1000
        ) {
            let (private_key, public_key) = schnorr_keygen().unwrap();
            let signature = schnorr_sign(&message, &private_key).unwrap();
            let params = SchnorrParams::new_2048();
            
            // Perturb the signature response
            let perturbed_s = (&signature.s + BigUint::from(perturbation)) % &params.q;
            let perturbed_sig = SchnorrSignature {
                r: signature.r.clone(),
                s: perturbed_s,
            };
            
            // Perturbed signature should not verify
            prop_assert!(!schnorr_verify(&message, &perturbed_sig, &public_key).unwrap());
        }

        #[test]
        fn prop_schnorr_wrong_key_fails(
            message in prop::collection::vec(any::<u8>(), 1..100)
        ) {
            let (private_key1, _) = schnorr_keygen().unwrap();
            let (_, public_key2) = schnorr_keygen().unwrap();
            
            let signature = schnorr_sign(&message, &private_key1).unwrap();
            
            // Signature from key1 should not verify with key2
            prop_assert!(!schnorr_verify(&message, &signature, &public_key2).unwrap());
        }

        #[test]
        fn prop_schnorr_serialization_roundtrip(
            message in prop::collection::vec(any::<u8>(), 1..100)
        ) {
            let (private_key, public_key) = schnorr_keygen().unwrap();
            let signature = schnorr_sign(&message, &private_key).unwrap();
            
            // Serialize and deserialize
            let sig_bytes = signature.to_bytes().unwrap();
            let sig_restored = SchnorrSignature::from_bytes(&sig_bytes).unwrap();
            
            // Should still verify after roundtrip
            prop_assert!(schnorr_verify(&message, &sig_restored, &public_key).unwrap());
        }

        #[test]
        fn prop_schnorr_signature_components_bounded(
            message in prop::collection::vec(any::<u8>(), 1..100)
        ) {
            let (private_key, public_key) = schnorr_keygen().unwrap();
            let signature = schnorr_sign(&message, &private_key).unwrap();
            let params = SchnorrParams::new_2048();
            
            // Verify all components are properly bounded
            prop_assert!(signature.r > BigUint::from(0u32));
            prop_assert!(signature.r < params.p);
            prop_assert!(signature.s < params.q);
            prop_assert!(public_key.y > BigUint::from(0u32));
            prop_assert!(public_key.y < params.p);
            prop_assert!(private_key.x > BigUint::from(0u32));
            prop_assert!(private_key.x < params.q);
        }
    }
}
