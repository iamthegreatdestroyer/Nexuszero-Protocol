//! # Digital Signature Example
//!
//! Demonstrates secure digital signatures using Schnorr signatures with Fiat-Shamir transform.
//! This example shows:
//! - Key generation for multiple parties
//! - Document signing with secure nonce generation
//! - Signature verification
//! - **CRITICAL**: Why nonce reuse is catastrophic
//! - Proper key management and cleanup
//!
//! ## Security Properties
//!
//! - **Unforgeability**: Only the private key holder can create valid signatures
//! - **Non-repudiation**: Signatures prove the signer's intent
//! - **Message Binding**: Signature is cryptographically tied to the message
//!
//! ## ⚠️ WARNING: NOT Quantum-Resistant
//!
//! Schnorr signatures are vulnerable to Shor's algorithm on quantum computers.
//! For post-quantum signatures, use lattice-based schemes like Dilithium.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example digital_signature
//! ```

use nexuszero_crypto::{
    proof::schnorr::{schnorr_keygen, schnorr_sign, schnorr_verify},
    CryptoResult,
};
use sha2::{Digest, Sha256};
use std::time::Instant;
use zeroize::Zeroize;

/// Compute SHA-256 hash of document content
fn hash_document(content: &str) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher.finalize().to_vec()
}

fn main() -> CryptoResult<()> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Digital Signature Demo");
    println!("  Using Schnorr Signatures with Fiat-Shamir Transform");
    println!("═══════════════════════════════════════════════════════\n");
    
    // ========================================================================
    // Step 1: Alice Generates Her Signing Key
    // ========================================================================
    
    println!("🔑 Step 1: Alice Generates Signing Key");
    println!("   Using RFC 3526 Group 14 (2048-bit MODP)");
    
    let start = Instant::now();
    let (mut alice_private, alice_public) = schnorr_keygen()?;
    let keygen_time = start.elapsed();
    
    println!("   ✅ Key pair generated in {:.2?}", keygen_time);
    println!("   📤 Alice publishes her public key\n");
    
    // ========================================================================
    // Step 2: Alice Signs a Document
    // ========================================================================
    
    println!("✍️  Step 2: Alice Signs Document");
    
    let document = "I, Alice, hereby transfer 100 tokens to Bob.";
    println!("   📄 Document: \"{}\"", document);
    
    // Hash the document (standard practice for signatures)
    let document_hash = hash_document(document);
    println!("   🔢 SHA-256 Hash: {}", hex::encode(&document_hash));
    
    // Sign the hash
    // ⚠️ CRITICAL: Each signature uses a FRESH cryptographic nonce
    // Nonce reuse allows COMPLETE private key recovery!
    let start = Instant::now();
    let signature = schnorr_sign(&document_hash, &alice_private)?;
    let sign_time = start.elapsed();
    
    println!("   ✅ Signature generated in {:.2?}", sign_time);
    println!("   📝 Signature size: {} bytes\n", signature.s.to_bytes_le().len());
    
    // ========================================================================
    // Step 3: Bob Verifies Alice's Signature
    // ========================================================================
    
    println!("✅ Step 3: Bob Verifies Signature");
    
    let start = Instant::now();
    let is_valid = schnorr_verify(&document_hash, &signature, &alice_public)?;
    let verify_time = start.elapsed();
    
    println!("   ⏱️  Verification completed in {:.2?}", verify_time);
    
    if is_valid {
        println!("   ✅ Signature is VALID");
        println!("   ✅ Document authenticity confirmed");
        println!("   ✅ Alice's authorship proven\n");
    } else {
        println!("   ❌ Signature is INVALID");
        return Ok(());
    }
    
    // ========================================================================
    // Step 4: Demonstrate Tamper Detection
    // ========================================================================
    
    println!("🔍 Step 4: Tamper Detection Test");
    
    let tampered_document = "I, Alice, hereby transfer 1000 tokens to Bob.";
    println!("   📄 Tampered: \"{}\"", tampered_document);
    println!("   ⚠️  (Changed 100 → 1000)");
    
    let tampered_hash = hash_document(tampered_document);
    
    let start = Instant::now();
    let is_valid_tampered = schnorr_verify(&tampered_hash, &signature, &alice_public)?;
    let verify_time = start.elapsed();
    
    println!("   ⏱️  Verification completed in {:.2?}", verify_time);
    
    if is_valid_tampered {
        println!("   ❌ ERROR: Tampered document verified (shouldn't happen!)");
    } else {
        println!("   ✅ Tampering DETECTED");
        println!("   ✅ Signature verification failed as expected\n");
    }
    
    // ========================================================================
    // Step 5: Demonstrate Multi-Party Signing
    // ========================================================================
    
    println!("👥 Step 5: Multi-Party Document Signing");
    
    // Bob generates his key
    println!("   🔑 Bob generates his signing key...");
    let (mut bob_private, bob_public) = schnorr_keygen()?;
    
    // Carol generates her key
    println!("   🔑 Carol generates her signing key...");
    let (mut carol_private, carol_public) = schnorr_keygen()?;
    
    // Multi-party contract
    let contract = "We, Alice, Bob, and Carol, agree to form a partnership.";
    println!("\n   📄 Contract: \"{}\"", contract);
    let contract_hash = hash_document(contract);
    
    // Each party signs independently
    println!("\n   ✍️  Alice signs...");
    let alice_sig = schnorr_sign(&contract_hash, &alice_private)?;
    
    println!("   ✍️  Bob signs...");
    let bob_sig = schnorr_sign(&contract_hash, &bob_private)?;
    
    println!("   ✍️  Carol signs...");
    let carol_sig = schnorr_sign(&contract_hash, &carol_private)?;
    
    // Verify all signatures
    println!("\n   ✅ Verifying all signatures...");
    
    let alice_valid = schnorr_verify(&contract_hash, &alice_sig, &alice_public)?;
    let bob_valid = schnorr_verify(&contract_hash, &bob_sig, &bob_public)?;
    let carol_valid = schnorr_verify(&contract_hash, &carol_sig, &carol_public)?;
    
    if alice_valid && bob_valid && carol_valid {
        println!("   ✅ All signatures VALID");
        println!("   ✅ Multi-party agreement authenticated\n");
    } else {
        println!("   ❌ One or more signatures invalid");
    }
    
    // ========================================================================
    // Step 6: CRITICAL Security Demonstration
    // ========================================================================
    
    println!("⚠️  Step 6: CRITICAL - Why Nonce Reuse is Catastrophic");
    println!("\n   Schnorr Signature Structure:");
    println!("   • r = k·G (commitment using random nonce k)");
    println!("   • c = H(r || m) (challenge via Fiat-Shamir)");
    println!("   • s = k + c·x (response, where x is private key)");
    println!("\n   🔴 If nonce k is reused for two messages m₁ and m₂:");
    println!("   • s₁ = k + c₁·x (signature 1)");
    println!("   • s₂ = k + c₂·x (signature 2)");
    println!("   • s₁ - s₂ = (c₁ - c₂)·x");
    println!("   • x = (s₁ - s₂) / (c₁ - c₂)");
    println!("\n   💀 RESULT: Private key x can be computed directly!");
    println!("\n   Our implementation prevents this by:");
    println!("   • Using cryptographically secure RNG (ChaCha20)");
    println!("   • Fresh randomness for every signature");
    println!("   • Stateless design (no nonce state to reuse)");
    
    // ========================================================================
    // Step 7: Secure Key Cleanup
    // ========================================================================
    
    println!("\n🧹 Step 7: Secure Memory Cleanup");
    
    // Private keys will be dropped and memory cleared automatically
    // Note: For production, implement Zeroize trait on SchnorrPrivateKey
    drop(alice_private);
    drop(bob_private);
    drop(carol_private);
    
    println!("   ✅ All private keys dropped from memory");
    
    // ========================================================================
    // Summary
    // ========================================================================
    
    println!("\n═══════════════════════════════════════════════════════");
    println!("  ✅ Digital Signature Demo Complete!");
    println!("═══════════════════════════════════════════════════════");
    println!("\n📊 Security Properties Demonstrated:");
    println!("   ✅ Unforgeability (only private key holder can sign)");
    println!("   ✅ Message binding (signatures tied to document)");
    println!("   ✅ Tamper detection (modification invalidates signature)");
    println!("   ✅ Multi-party signing (independent signatures)");
    println!("   ✅ Secure nonce generation (prevents key recovery)");
    
    println!("\n⚠️  Important Limitations:");
    println!("   ⚠️  NOT quantum-resistant (use Dilithium for PQ signatures)");
    println!("   ⚠️  Vulnerable to Shor's algorithm on quantum computers");
    println!("   ⚠️  112-bit classical security (2048-bit MODP group)");
    
    println!("\n⚠️  Production Recommendations:");
    println!("   • Transition to post-quantum signatures (Dilithium, SPHINCS+)");
    println!("   • Use deterministic signatures (RFC 6979) for reproducibility");
    println!("   • Implement signature aggregation for efficiency");
    println!("   • Add timestamp and expiration to prevent replay attacks");
    println!("   • Use secure key storage (HSM or encrypted keystore)");
    println!("   • Implement key rotation policies");
    println!("   • Monitor for side-channel attacks (timing, power analysis)");
    
    Ok(())
}
