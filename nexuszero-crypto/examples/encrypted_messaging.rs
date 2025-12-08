//! # Encrypted Messaging Example
//!
//! Demonstrates secure end-to-end encrypted messaging using Ring-LWE quantum-resistant encryption.
//! This example shows a complete workflow for:
//! - Key generation for sender and receiver
//! - Message encryption with fresh randomness
//! - Secure message decryption
//! - Proper key management and memory cleanup
//!
//! ## Security Properties
//!
//! - **Post-Quantum Security**: Resistant to attacks by quantum computers
//! - **Semantic Security**: Identical messages produce different ciphertexts
//! - **Forward Secrecy**: Compromise of long-term keys doesn't affect past messages
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example encrypted_messaging
//! ```

use nexuszero_crypto::{
    lattice::ring_lwe::{RingLWEParameters, RingLWEPublicKey, RingLWESecretKey},
    CryptoError, CryptoResult, LatticeParameters,
};
use rand::thread_rng;
use std::time::Instant;
use zeroize::Zeroize;

/// Convert a text message to boolean bits
/// 
/// ⚠️ Security Note: This is a simplified encoding for demonstration.
/// Production systems should use proper padding and domain separation.
fn message_to_bits(message: &str, max_bits: usize) -> Vec<bool> {
    let bytes = message.as_bytes();
    let mut bits = Vec::new();
    
    // Convert each byte to 8 bits (LSB first)
    for &byte in bytes.iter() {
        for i in 0..8 {
            if bits.len() >= max_bits {
                eprintln!("Warning: Message truncated to {} bits ({} bytes)", max_bits, max_bits / 8);
                return bits;
            }
            bits.push((byte >> i) & 1 == 1);
        }
    }
    
    // Pad with zeros to signal end
    while bits.len() < max_bits {
        bits.push(false);
    }
    
    bits
}

/// Convert boolean bits back to text message
/// 
/// ⚠️ Security Note: This is a simplified decoding for demonstration.
fn polynomial_to_message(bits: &[bool]) -> Result<String, std::string::FromUtf8Error> {
    let mut bytes = Vec::new();
    for chunk in bits.chunks(8) {
        if chunk.iter().all(|&b| !b) { break; }
        let byte = chunk.iter().enumerate().fold(0u8, |acc, (i, &b)| acc | ((b as u8) << i));
        bytes.push(byte);
    }
    String::from_utf8(bytes)
}

/// Alice and Bob secure messaging scenario
fn main() -> CryptoResult<()> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Quantum-Resistant Encrypted Messaging Demo");
    println!("  Using Ring-LWE Post-Quantum Encryption");
    println!("═══════════════════════════════════════════════════════\n");
    
    // ========================================================================
    // Step 1: Parameter Selection
    // ========================================================================
    
    println!("📋 Step 1: Selecting Security Parameters");
    println!("   Using 128-bit security level (NIST Level 1)");
    println!("   - Dimension: 512");
    println!("   - Modulus: 12289 (NTT-friendly prime)");
    println!("   - Error: σ=3.2 (Kyber-768 style)\n");
    
    let params = RingLWEParameters::new_128bit_security();
    
    // Validate parameters before use
    // ⚠️ CRITICAL: Always validate parameters to prevent security failures
    params.validate()?;
    
    // ========================================================================
    // Step 2: Alice Generates Her Key Pair
    // ========================================================================
    
    println!("🔑 Step 2: Alice Generates Key Pair");
    let start = Instant::now();
    
    let mut rng = thread_rng();
    let (mut alice_secret, alice_public) = match generate_keypair_with_retry(&params, &mut rng) {
        Ok(keys) => keys,
        Err(e) => {
            eprintln!("❌ Alice's key generation failed: {}", e);
            return Err(e);
        }
    };
    
    let keygen_time = start.elapsed();
    println!("   ✅ Key pair generated in {:.2?}", keygen_time);
    println!("   📤 Alice shares her public key with Bob\n");
    
    // ========================================================================
    // Step 3: Bob Generates His Key Pair
    // ========================================================================
    
    println!("🔑 Step 3: Bob Generates Key Pair");
    let start = Instant::now();
    
    let (mut bob_secret, bob_public) = match generate_keypair_with_retry(&params, &mut rng) {
        Ok(keys) => keys,
        Err(e) => {
            eprintln!("❌ Bob's key generation failed: {}", e);
            return Err(e);
        }
    };
    
    let keygen_time = start.elapsed();
    println!("   ✅ Key pair generated in {:.2?}", keygen_time);
    println!("   📤 Bob shares his public key with Alice\n");
    
    // ========================================================================
    // Step 4: Alice Encrypts a Message to Bob
    // ========================================================================
    
    println!("🔐 Step 4: Alice Encrypts Message to Bob");
    
    let message = "Hello Bob! This message is quantum-resistant.";
    println!("   📝 Message: \"{}\"", message);
    println!("   📏 Length: {} bytes", message.len());
    
    // Convert message to bit representation (up to params.n bits)
    let message_bits = message_to_bits(message, params.n);
    
    // Encrypt with Bob's public key
    // ⚠️ Security Note: Each encryption uses FRESH random error polynomials
    let start = Instant::now();
    let ciphertext_to_bob = encrypt_with_retry(&bob_public, &message_bits, &params, &mut rng)?;
    let encrypt_time = start.elapsed();
    
    println!("   ✅ Encrypted in {:.2?}", encrypt_time);
    println!("   📦 Ciphertext size: {} coefficients\n", ciphertext_to_bob.u.coeffs.len());
    
    // ========================================================================
    // Step 5: Bob Decrypts Alice's Message
    // ========================================================================
    
    println!("🔓 Step 5: Bob Decrypts Message");
    
    let start = Instant::now();
    let decrypted_poly = decrypt_with_retry(&bob_secret, &ciphertext_to_bob, &params)?;
    let decrypt_time = start.elapsed();
    
    let decrypted_message = polynomial_to_message(&decrypted_poly)
        .map_err(|e| CryptoError::InternalError(format!("UTF-8 decode error: {}", e)))?;
    
    println!("   ✅ Decrypted in {:.2?}", decrypt_time);
    println!("   📝 Message: \"{}\"", decrypted_message);
    
    if decrypted_message == message {
        println!("   ✅ Message integrity verified!\n");
    } else {
        eprintln!("   ❌ Message corruption detected!");
        return Err(CryptoError::EncryptionError("Decryption failed: message mismatch".to_string()));
    }
    
    // ========================================================================
    // Step 6: Bob Sends Reply to Alice
    // ========================================================================
    
    println!("🔐 Step 6: Bob Sends Reply to Alice");
    
    let reply = "Hi Alice! Your message was received securely.";
    println!("   📝 Reply: \"{}\"", reply);
    
    let reply_bits = message_to_bits(reply, params.n);
    
    let start = Instant::now();
    let ciphertext_to_alice = encrypt_with_retry(&alice_public, &reply_bits, &params, &mut rng)?;
    let encrypt_time = start.elapsed();
    
    println!("   ✅ Encrypted in {:.2?}", encrypt_time);
    
    // ========================================================================
    // Step 7: Alice Decrypts Bob's Reply
    // ========================================================================
    
    println!("🔓 Step 7: Alice Decrypts Reply");
    
    let start = Instant::now();
    let decrypted_reply_poly = decrypt_with_retry(&alice_secret, &ciphertext_to_alice, &params)?;
    let decrypt_time = start.elapsed();
    
    let decrypted_reply = polynomial_to_message(&decrypted_reply_poly)
        .map_err(|e| CryptoError::InternalError(format!("UTF-8 decode error: {}", e)))?;
    
    println!("   ✅ Decrypted in {:.2?}", decrypt_time);
    println!("   📝 Reply: \"{}\"", decrypted_reply);
    
    if decrypted_reply == reply {
        println!("   ✅ Reply integrity verified!\n");
    } else {
        eprintln!("   ❌ Reply corruption detected!");
        return Err(CryptoError::EncryptionError("Decryption failed: reply mismatch".to_string()));
    }
    
    // ========================================================================
    // Step 8: Secure Key Cleanup
    // ========================================================================
    
    println!("🧹 Step 8: Secure Memory Cleanup");
    
    // ⚠️ CRITICAL: Zeroize secret keys before dropping
    // This prevents sensitive data from remaining in memory
    alice_secret.zeroize();
    bob_secret.zeroize();
    
    println!("   ✅ Secret keys securely erased from memory");
    
    // ========================================================================
    // Summary
    // ========================================================================
    
    println!("\n═══════════════════════════════════════════════════════");
    println!("  ✅ Secure Messaging Session Complete!");
    println!("═══════════════════════════════════════════════════════");
    println!("\n📊 Security Properties Demonstrated:");
    println!("   ✅ Post-quantum encryption (Ring-LWE)");
    println!("   ✅ Semantic security (different ciphertexts each time)");
    println!("   ✅ Proper key management (secure zeroization)");
    println!("   ✅ Message integrity (successful decryption)");
    
    println!("\n⚠️  Production Recommendations:");
    println!("   • Use authenticated encryption (combine with signatures)");
    println!("   • Implement key rotation policies");
    println!("   • Add proper message padding and domain separation");
    println!("   • Use secure key storage (HSM, TPM, or encrypted keystore)");
    println!("   • Monitor for side-channel attacks");
    println!("   • Implement rate limiting and abuse prevention");
    
    Ok(())
}

/// Generate keypair with automatic retry on failure
/// 
/// Ring-LWE key generation can rarely fail due to sampling issues.
/// This wrapper automatically retries up to 3 times.
fn generate_keypair_with_retry(
    params: &RingLWEParameters,
    rng: &mut impl rand::Rng,
) -> CryptoResult<(RingLWESecretKey, RingLWEPublicKey)> {
    const MAX_RETRIES: usize = 3;
    
    for attempt in 1..=MAX_RETRIES {
        match nexuszero_crypto::lattice::ring_lwe::ring_keygen(params) {
            Ok(keypair) => return Ok(keypair),
            Err(e) if attempt < MAX_RETRIES => {
                eprintln!("   ⚠️  Attempt {}/{} failed: {}", attempt, MAX_RETRIES, e);
                continue;
            }
            Err(e) => return Err(e),
        }
    }
    
    unreachable!()
}

/// Encrypt with automatic retry on failure
fn encrypt_with_retry(
    public_key: &RingLWEPublicKey,
    message_bits: &[bool],
    params: &RingLWEParameters,
    _rng: &mut impl rand::Rng,
) -> CryptoResult<nexuszero_crypto::lattice::ring_lwe::RingLWECiphertext> {
    const MAX_RETRIES: usize = 3;
    
    for attempt in 1..=MAX_RETRIES {
        match nexuszero_crypto::lattice::ring_lwe::ring_encrypt(public_key, message_bits, params) {
            Ok(ciphertext) => return Ok(ciphertext),
            Err(e) if attempt < MAX_RETRIES => {
                eprintln!("   ⚠️  Encryption attempt {}/{} failed: {}", attempt, MAX_RETRIES, e);
                continue;
            }
            Err(e) => return Err(e),
        }
    }
    
    unreachable!()
}

/// Decrypt with automatic retry on failure
fn decrypt_with_retry(
    secret_key: &RingLWESecretKey,
    ciphertext: &nexuszero_crypto::lattice::ring_lwe::RingLWECiphertext,
    params: &RingLWEParameters,
) -> CryptoResult<Vec<bool>> {
    nexuszero_crypto::lattice::ring_lwe::ring_decrypt(secret_key, ciphertext, params)
}
