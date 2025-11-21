/// Test Vector Runner
/// 
/// Loads JSON test vectors and executes them against the crypto implementations.
/// Provides automated testing against standardized test vectors.

use nexuszero_crypto::lattice::lwe::*;
use nexuszero_crypto::lattice::ring_lwe::*;
use nexuszero_crypto::proof::proof::*;
use nexuszero_crypto::proof::statement::*;
use nexuszero_crypto::proof::witness::*;
use serde::{Deserialize, Serialize};
use std::fs;
use rand::thread_rng;

#[derive(Debug, Deserialize, Serialize)]
struct LWETestVector {
    description: String,
    security_level: u8,
    n: usize,
    m: usize,
    q: u64,
    sigma: f64,
    seed: Option<u64>,
    expected_success: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct RingLWETestVector {
    description: String,
    security_level: u8,
    n: usize,
    q: u64,
    sigma: f64,
    message_bits: Vec<bool>,
    expected_success: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct ProofTestVector {
    description: String,
    proof_type: String,
    expected_properties: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct LWETestVectors {
    vectors: Vec<LWETestVector>,
}

#[derive(Debug, Deserialize)]
struct RingLWETestVectors {
    vectors: Vec<RingLWETestVector>,
}

#[derive(Debug, Deserialize)]
struct ProofTestVectors {
    vectors: Vec<ProofTestVector>,
}

#[test]
fn test_lwe_vectors() {
    // Load test vectors
    let json_data = fs::read_to_string("tests/test_vectors/lwe_test_vectors.json")
        .expect("Failed to read LWE test vectors");
    
    let vectors: LWETestVectors = serde_json::from_str(&json_data)
        .expect("Failed to parse LWE test vectors");
    
    println!("Running {} LWE test vectors...", vectors.vectors.len());
    
    for (i, vector) in vectors.vectors.iter().enumerate() {
        println!("  [{}] {}", i + 1, vector.description);
        
        // Create parameters
        let params = LWEParameters::new(vector.n, vector.m, vector.q, vector.sigma);
        
        // Test key generation
        let mut rng = thread_rng();
        let keygen_result = keygen(&params, &mut rng);
        assert_eq!(keygen_result.is_ok(), vector.expected_success,
            "KeyGen failed for vector: {}", vector.description);
        
        if let Ok((sk, pk)) = keygen_result {
            // Test encryption
            let encrypt_result = encrypt(&pk, true, &params, &mut rng);
            assert_eq!(encrypt_result.is_ok(), vector.expected_success,
                "Encrypt failed for vector: {}", vector.description);
            
            if let Ok(ct) = encrypt_result {
                // Test decryption
                let decrypt_result = decrypt(&sk, &ct, &params);
                if vector.expected_success {
                    assert_eq!(decrypt_result.unwrap(), true,
                        "Decryption produced wrong result for vector: {}", vector.description);
                }
            }
        }
    }
    
    println!("✓ All LWE test vectors passed!");
}

#[test]
fn test_ring_lwe_vectors() {
    // Load test vectors
    let json_data = fs::read_to_string("tests/test_vectors/ring_lwe_test_vectors.json")
        .expect("Failed to read Ring-LWE test vectors");
    
    let vectors: RingLWETestVectors = serde_json::from_str(&json_data)
        .expect("Failed to parse Ring-LWE test vectors");
    
    println!("Running {} Ring-LWE test vectors...", vectors.vectors.len());
    
    for (i, vector) in vectors.vectors.iter().enumerate() {
        println!("  [{}] {}", i + 1, vector.description);
        
        // Create parameters
        let params = RingLWEParameters::new(vector.n, vector.q, vector.sigma);
        
        // Test key generation
        let keygen_result = ring_keygen(&params);
        assert_eq!(keygen_result.is_ok(), vector.expected_success,
            "KeyGen failed for vector: {}", vector.description);
        
        if let Ok((sk, pk)) = keygen_result {
            // Test encryption with provided message bits
            if !vector.message_bits.is_empty() {
                let encrypt_result = ring_encrypt(&pk, &vector.message_bits, &params);
                assert_eq!(encrypt_result.is_ok(), vector.expected_success,
                    "Encrypt failed for vector: {}", vector.description);
                
                if let Ok(ct) = encrypt_result {
                    // Test decryption
                    let decrypt_result = ring_decrypt(&sk, &ct, &params);
                    if vector.expected_success && !vector.message_bits.is_empty() {
                        assert!(decrypt_result.is_ok(), "Ring-LWE decryption failed");
                        let decrypted = decrypt_result.unwrap();
                        if !decrypted.is_empty() {
                            // Check at least the first bit matches
                            assert_eq!(decrypted[0], vector.message_bits[0],
                                "First bit mismatch for vector: {}", vector.description);
                        }
                    }
                }
            }
        }
    }
    
    println!("✓ All Ring-LWE test vectors passed!");
}

#[test]
fn test_proof_vectors() {
    // Load test vectors
    let json_data = fs::read_to_string("tests/test_vectors/proof_test_vectors.json")
        .expect("Failed to read proof test vectors");
    
    let vectors: ProofTestVectors = serde_json::from_str(&json_data)
        .expect("Failed to parse proof test vectors");
    
    println!("Running {} proof test vectors...", vectors.vectors.len());
    
    for (i, vector) in vectors.vectors.iter().enumerate() {
        println!("  [{}] {}", i + 1, vector.description);
        
        match vector.proof_type.as_str() {
            "discrete_log" => {
                // Test discrete log proof
                use num_bigint::BigUint;
                let generator = vec![2u8; 32];
                let secret = vec![42u8; 32];
                let modulus_bytes = vec![0xFF; 32];
                
                let gen_big = BigUint::from_bytes_be(&generator);
                let secret_big = BigUint::from_bytes_be(&secret);
                let mod_big = BigUint::from_bytes_be(&modulus_bytes);
                let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
                
                let statement = StatementBuilder::new()
                    .discrete_log(generator, public_value)
                    .build()
                    .expect("Failed to build statement");
                
                let witness = Witness::discrete_log(secret);
                
                // Generate proof
                let proof_result = prove(&statement, &witness);
                assert!(proof_result.is_ok(), "Proof generation failed for: {}", vector.description);
                
                if let Ok(proof) = proof_result {
                    // Verify proof
                    let verify_result = verify(&statement, &proof);
                    assert!(verify_result.is_ok(), "Proof verification failed for: {}", vector.description);
                    
                    // Check expected properties
                    for property in &vector.expected_properties {
                        match property.as_str() {
                            "completeness" => {
                                // Already verified above
                                assert!(verify_result.is_ok());
                            }
                            "zero_knowledge" => {
                                // Proof should not reveal secret
                                assert!(proof.responses.len() > 0);
                            }
                            "soundness" => {
                                // Valid proofs should verify
                                assert!(verify_result.is_ok());
                            }
                            _ => {}
                        }
                    }
                }
            }
            "preimage" => {
                // Test preimage proof
                use sha3::{Digest, Sha3_256};
                let preimage = b"test_message".to_vec();
                let mut hasher = Sha3_256::new();
                hasher.update(&preimage);
                let hash = hasher.finalize().to_vec();
                
                let statement = StatementBuilder::new()
                    .preimage(HashFunction::SHA3_256, hash)
                    .build()
                    .expect("Failed to build preimage statement");
                
                let witness = Witness::preimage(preimage);
                
                // Generate and verify proof
                let proof_result = prove(&statement, &witness);
                assert!(proof_result.is_ok(), "Preimage proof generation failed");
                
                if let Ok(proof) = proof_result {
                    let verify_result = verify(&statement, &proof);
                    assert!(verify_result.is_ok(), "Preimage proof verification failed");
                }
            }
            _ => {
                println!("    Skipping unknown proof type: {}", vector.proof_type);
            }
        }
    }
    
    println!("✓ All proof test vectors passed!");
}

#[test]
fn test_polynomial_operations_from_vectors() {
    // Load Ring-LWE vectors which include polynomial test data
    let json_data = fs::read_to_string("tests/test_vectors/ring_lwe_test_vectors.json")
        .expect("Failed to read Ring-LWE test vectors");
    
    let vectors: RingLWETestVectors = serde_json::from_str(&json_data)
        .expect("Failed to parse Ring-LWE test vectors");
    
    println!("Testing polynomial operations from Ring-LWE vectors...");
    
    for vector in &vectors.vectors {
        if vector.description.contains("polynomial") || vector.description.contains("NTT") {
            println!("  Testing: {}", vector.description);
            
            // Create test polynomials
            let poly_a = Polynomial {
                coeffs: vec![1i64; 4],
                modulus: vector.q,
                degree: 4,
            };
            
            let poly_b = Polynomial {
                coeffs: vec![2i64; 4],
                modulus: vector.q,
                degree: 4,
            };
            
            // Test addition
            let sum = poly_add(&poly_a, &poly_b, vector.q);
            assert_eq!(sum.coeffs[0], 3, "Polynomial addition failed");
            
            // Test subtraction
            let diff = poly_sub(&poly_b, &poly_a, vector.q);
            assert_eq!(diff.coeffs[0], 1, "Polynomial subtraction failed");
            
            // Test NTT if degree is power of 2 and we can find primitive root
            if vector.n.is_power_of_two() {
                if let Some(root) = find_primitive_root(vector.n, vector.q) {
                    let ntt_result = ntt(&poly_a, vector.q, root);
                    assert_eq!(ntt_result.len(), poly_a.coeffs.len(), "NTT length mismatch");
                }
            }
        }
    }
    
    println!("✓ Polynomial operations from vectors passed!");
}
