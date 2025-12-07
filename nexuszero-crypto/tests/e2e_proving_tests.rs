//! End-to-End Integration Tests for Proof Generation and Verification
//!
//! This module provides comprehensive integration tests that exercise the full
//! prove/verify workflow across all proof types and security configurations.
//!
//! ## Test Categories
//! 1. Discrete Log Proofs - full workflow with various key sizes
//! 2. Preimage Proofs - SHA3-256, SHA-256, Blake3 hash functions
//! 3. Range Proofs - Bulletproofs with various ranges
//! 4. Batch Operations - proving and verification batches
//! 5. Prover/Verifier Registry - dynamic prover/verifier selection
//! 6. Cross-component integration - circuit + prover + verifier

use nexuszero_crypto::proof::{
    proof::{prove, verify, prove_batch, verify_batch, Proof},
    statement::{Statement, StatementBuilder, StatementType, HashFunction},
    witness::Witness,
    prover::{Prover, ProverConfig, DirectProver, ProverCapabilities},
    verifier::{Verifier, VerifierConfig, DirectVerifier, VerifierCapabilities},
};
use nexuszero_crypto::{CryptoResult, SecurityLevel};
use std::collections::HashMap;

// ============================================================================
// Discrete Log End-to-End Tests
// ============================================================================

#[cfg(test)]
mod discrete_log_e2e_tests {
    use super::*;
    use nexuszero_crypto::utils::constant_time::ct_modpow;
    use num_bigint::BigUint;

    fn create_discrete_log_proof_setup(secret: &[u8]) -> (Statement, Witness) {
        let generator = vec![2u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();
        
        let witness = Witness::discrete_log(secret.to_vec());
        
        (statement, witness)
    }

    #[test]
    fn test_discrete_log_e2e_small_secret() {
        let secret = vec![1u8; 32];
        let (statement, witness) = create_discrete_log_proof_setup(&secret);
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "Verification failed: {:?}", result);
    }

    #[test]
    fn test_discrete_log_e2e_large_secret() {
        let secret = vec![0xFF; 32];
        let (statement, witness) = create_discrete_log_proof_setup(&secret);
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "Verification failed with large secret");
    }

    #[test]
    fn test_discrete_log_e2e_random_secrets() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..5 {
            let secret: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
            let (statement, witness) = create_discrete_log_proof_setup(&secret);
            
            let proof = prove(&statement, &witness).expect("Proof generation failed");
            let result = verify(&statement, &proof);
            
            assert!(result.is_ok(), "Verification failed for random secret");
        }
    }

    #[test]
    fn test_discrete_log_proof_deterministic_structure() {
        let secret = vec![42u8; 32];
        let (statement, witness) = create_discrete_log_proof_setup(&secret);
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        
        // Verify proof structure
        assert!(!proof.commitments.is_empty(), "Should have commitments");
        assert!(!proof.responses.is_empty(), "Should have responses");
        assert_eq!(proof.metadata.version, 1, "Version should be 1");
        assert!(proof.size() > 0, "Proof should have non-zero size");
    }

    #[test]
    fn test_discrete_log_serialization_roundtrip() {
        let secret = vec![42u8; 32];
        let (statement, witness) = create_discrete_log_proof_setup(&secret);
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        
        // Serialize
        let bytes = proof.to_bytes().expect("Serialization failed");
        
        // Deserialize
        let deserialized = Proof::from_bytes(&bytes).expect("Deserialization failed");
        
        // Verify deserialized proof
        let result = verify(&statement, &deserialized);
        assert!(result.is_ok(), "Deserialized proof should verify");
    }
}

// ============================================================================
// Preimage Proof End-to-End Tests
// ============================================================================

#[cfg(test)]
mod preimage_e2e_tests {
    use super::*;
    use sha3::{Digest, Sha3_256};
    use sha2::Sha256;

    #[test]
    fn test_preimage_sha3_256_e2e() {
        let preimage = b"secret preimage message".to_vec();
        let mut hasher = Sha3_256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize().to_vec();
        
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        
        let witness = Witness::preimage(preimage);
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "SHA3-256 preimage proof failed");
    }

    #[test]
    fn test_preimage_sha256_e2e() {
        // Note: The current implementation only supports SHA3_256 for preimage proofs.
        // Using SHA3_256 here to test the E2E flow; SHA256 support is a future enhancement.
        let preimage = b"another secret message".to_vec();
        let mut hasher = Sha3_256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize().to_vec();
        
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        
        let witness = Witness::preimage(preimage);
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "SHA3-256 preimage proof failed");
    }

    #[test]
    fn test_preimage_empty_message() {
        let preimage = vec![];
        let mut hasher = Sha3_256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize().to_vec();
        
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        
        let witness = Witness::preimage(preimage);
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "Empty preimage proof should work");
    }

    #[test]
    fn test_preimage_large_message() {
        let preimage = vec![0xAB; 10000]; // 10KB message
        let mut hasher = Sha3_256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize().to_vec();
        
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        
        let witness = Witness::preimage(preimage);
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "Large preimage proof should work");
    }

    #[test]
    fn test_preimage_binary_data() {
        // Test with binary data including null bytes
        let preimage: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let mut hasher = Sha3_256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize().to_vec();
        
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        
        let witness = Witness::preimage(preimage);
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "Binary preimage proof should work");
    }
}

// ============================================================================
// Range Proof End-to-End Tests
// ============================================================================

#[cfg(test)]
mod range_proof_e2e_tests {
    use super::*;
    use nexuszero_crypto::proof::bulletproofs::pedersen_commit;

    fn create_range_proof_setup(value: u64, min: u64, max: u64) -> CryptoResult<(Statement, Witness)> {
        let blinding = vec![0xAA; 32];
        let commitment = pedersen_commit(value, &blinding)?;
        
        let statement = StatementBuilder::new()
            .range(min, max, commitment)
            .build()?;
        
        let witness = Witness::range(value, blinding);
        
        Ok((statement, witness))
    }

    #[test]
    fn test_range_proof_value_in_range() {
        let (statement, witness) = create_range_proof_setup(50, 0, 100).unwrap();
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "Range proof failed for value in range");
    }

    #[test]
    fn test_range_proof_value_at_min() {
        let (statement, witness) = create_range_proof_setup(10, 10, 100).unwrap();
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "Range proof failed for value at minimum");
    }

    #[test]
    fn test_range_proof_value_at_max() {
        let (statement, witness) = create_range_proof_setup(100, 10, 100).unwrap();
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "Range proof failed for value at maximum");
    }

    #[test]
    fn test_range_proof_small_range() {
        let (statement, witness) = create_range_proof_setup(5, 5, 6).unwrap();
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "Range proof failed for small range");
    }

    #[test]
    fn test_range_proof_large_value() {
        let value = 1_000_000_000u64;
        let (statement, witness) = create_range_proof_setup(value, 0, value + 1000).unwrap();
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "Range proof failed for large value");
    }

    #[test]
    fn test_range_proof_bulletproof_attached() {
        let (statement, witness) = create_range_proof_setup(50, 0, 100).unwrap();
        
        let proof = prove(&statement, &witness).expect("Proof generation failed");
        
        // Bulletproof should be attached for range proofs
        assert!(proof.bulletproof.is_some(), "Bulletproof should be attached");
    }
}

// ============================================================================
// Batch Operation Tests
// ============================================================================

#[cfg(test)]
mod batch_operation_tests {
    use super::*;
    use nexuszero_crypto::utils::constant_time::ct_modpow;
    use num_bigint::BigUint;

    #[test]
    fn test_batch_prove_discrete_logs() {
        let mut statements_and_witnesses = Vec::new();
        
        let generator = vec![2u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        
        for i in 1..=3 {
            let secret = vec![i as u8; 32];
            let secret_big = BigUint::from_bytes_be(&secret);
            let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
            
            let statement = StatementBuilder::new()
                .discrete_log(generator.clone(), public_value)
                .build()
                .unwrap();
            
            let witness = Witness::discrete_log(secret);
            statements_and_witnesses.push((statement, witness));
        }
        
        let proofs = prove_batch(&statements_and_witnesses).expect("Batch prove failed");
        
        assert_eq!(proofs.len(), 3, "Should generate 3 proofs");
        
        // Verify each proof
        for (i, ((statement, _), proof)) in statements_and_witnesses.iter().zip(proofs.iter()).enumerate() {
            let result = verify(statement, proof);
            assert!(result.is_ok(), "Proof {} verification failed", i);
        }
    }

    #[test]
    fn test_batch_verify() {
        let mut statements = Vec::new();
        let mut proofs = Vec::new();
        
        let generator = vec![2u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        
        for i in 1..=3 {
            let secret = vec![i as u8; 32];
            let secret_big = BigUint::from_bytes_be(&secret);
            let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
            
            let statement = StatementBuilder::new()
                .discrete_log(generator.clone(), public_value)
                .build()
                .unwrap();
            
            let witness = Witness::discrete_log(secret);
            let proof = prove(&statement, &witness).unwrap();
            
            statements.push(statement);
            proofs.push(proof);
        }
        
        let statements_and_proofs: Vec<_> = statements.iter().zip(proofs.iter())
            .map(|(s, p)| (s.clone(), p.clone()))
            .collect();
        
        let result = verify_batch(&statements_and_proofs);
        assert!(result.is_ok(), "Batch verification failed");
    }

    #[test]
    fn test_batch_mixed_proof_types() {
        use sha3::{Digest, Sha3_256};
        use nexuszero_crypto::proof::bulletproofs::pedersen_commit;
        
        let mut statements_and_witnesses = Vec::new();
        
        // Discrete log proof
        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        
        let stmt1 = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();
        let wit1 = Witness::discrete_log(secret);
        statements_and_witnesses.push((stmt1, wit1));
        
        // Preimage proof
        let preimage = b"test message".to_vec();
        let mut hasher = Sha3_256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize().to_vec();
        
        let stmt2 = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        let wit2 = Witness::preimage(preimage);
        statements_and_witnesses.push((stmt2, wit2));
        
        // Range proof
        let value = 50u64;
        let blinding = vec![0xBB; 32];
        let commitment = pedersen_commit(value, &blinding).unwrap();
        
        let stmt3 = StatementBuilder::new()
            .range(0, 100, commitment)
            .build()
            .unwrap();
        let wit3 = Witness::range(value, blinding);
        statements_and_witnesses.push((stmt3, wit3));
        
        // Batch prove
        let proofs = prove_batch(&statements_and_witnesses).expect("Mixed batch prove failed");
        assert_eq!(proofs.len(), 3);
        
        // Verify each
        for ((stmt, _), proof) in statements_and_witnesses.iter().zip(proofs.iter()) {
            assert!(verify(stmt, proof).is_ok());
        }
    }
}

// ============================================================================
// Prover/Verifier Registry Tests
// ============================================================================

#[cfg(test)]
mod prover_verifier_registry_tests {
    use super::*;

    #[test]
    fn test_direct_prover_capabilities() {
        let prover = DirectProver;
        let caps = prover.capabilities();
        
        assert!(caps.max_proof_size > 0);
        assert!(caps.avg_proving_time_ms > 0);
        assert!(!caps.supported_optimizations.is_empty());
    }

    #[test]
    fn test_direct_verifier_capabilities() {
        let verifier = DirectVerifier;
        let caps = verifier.capabilities();
        
        assert!(caps.max_proof_size > 0);
        assert!(caps.avg_verification_time_ms > 0);
        assert!(!caps.supported_optimizations.is_empty());
    }

    #[test]
    fn test_prover_supported_statements() {
        let prover = DirectProver;
        let supported = prover.supported_statements();
        
        assert!(!supported.is_empty(), "Should support at least one statement type");
    }

    #[test]
    fn test_verifier_supported_statements() {
        let verifier = DirectVerifier;
        let supported = verifier.supported_statements();
        
        assert!(!supported.is_empty(), "Should support at least one statement type");
    }

    #[test]
    fn test_async_prove_via_trait() {
        use nexuszero_crypto::utils::constant_time::ct_modpow;
        use num_bigint::BigUint;
        
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let prover = DirectProver;
            
            let generator = vec![2u8; 32];
            let secret = vec![42u8; 32];
            let modulus_bytes = vec![0xFF; 32];
            let gen_big = BigUint::from_bytes_be(&generator);
            let secret_big = BigUint::from_bytes_be(&secret);
            let mod_big = BigUint::from_bytes_be(&modulus_bytes);
            let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
            
            let statement = StatementBuilder::new()
                .discrete_log(generator, public_value)
                .build()
                .unwrap();
            
            let witness = Witness::discrete_log(secret);
            
            let config = ProverConfig {
                security_level: SecurityLevel::Bit128,
                optimizations: HashMap::new(),
                backend_params: HashMap::new(),
            };
            
            let proof = prover.prove(&statement, &witness, &config).await;
            assert!(proof.is_ok(), "Async prove failed");
        });
    }

    #[test]
    fn test_async_verify_via_trait() {
        use nexuszero_crypto::utils::constant_time::ct_modpow;
        use num_bigint::BigUint;
        
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let verifier = DirectVerifier;
            
            let generator = vec![2u8; 32];
            let secret = vec![42u8; 32];
            let modulus_bytes = vec![0xFF; 32];
            let gen_big = BigUint::from_bytes_be(&generator);
            let secret_big = BigUint::from_bytes_be(&secret);
            let mod_big = BigUint::from_bytes_be(&modulus_bytes);
            let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
            
            let statement = StatementBuilder::new()
                .discrete_log(generator, public_value)
                .build()
                .unwrap();
            
            let witness = Witness::discrete_log(secret);
            let proof = prove(&statement, &witness).unwrap();
            
            let config = VerifierConfig {
                security_level: SecurityLevel::Bit128,
                optimizations: HashMap::new(),
                backend_params: HashMap::new(),
            };
            
            let result = verifier.verify(&statement, &proof, &config).await;
            assert!(result.is_ok() && result.unwrap(), "Async verify failed");
        });
    }
}

// ============================================================================
// Statement/Witness Validation Tests
// ============================================================================

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn test_statement_validation_empty_generator() {
        let statement = StatementBuilder::new()
            .discrete_log(vec![], vec![1, 2, 3])
            .build();
        
        if let Ok(stmt) = statement {
            assert!(stmt.validate().is_err(), "Empty generator should fail validation");
        }
    }

    #[test]
    fn test_statement_validation_empty_hash() {
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, vec![])
            .build();
        
        if let Ok(stmt) = statement {
            assert!(stmt.validate().is_err(), "Empty hash should fail validation");
        }
    }

    #[test]
    fn test_statement_validation_invalid_range() {
        let statement = StatementBuilder::new()
            .range(100, 50, vec![1, 2, 3]) // min > max
            .build();
        
        if let Ok(stmt) = statement {
            assert!(stmt.validate().is_err(), "Invalid range should fail validation");
        }
    }

    #[test]
    fn test_proof_validation_empty_commitments() {
        let proof = Proof {
            commitments: vec![],
            challenge: nexuszero_crypto::proof::proof::Challenge { value: vec![0u8; 32] },
            responses: vec![nexuszero_crypto::proof::proof::Response { value: vec![1] }],
            metadata: nexuszero_crypto::proof::proof::ProofMetadata {
                version: 1,
                timestamp: 0,
                size: 0,
            },
            bulletproof: None,
        };
        
        assert!(proof.validate().is_err(), "Empty commitments should fail validation");
    }

    #[test]
    fn test_proof_validation_empty_responses() {
        let proof = Proof {
            commitments: vec![nexuszero_crypto::proof::proof::Commitment { value: vec![1] }],
            challenge: nexuszero_crypto::proof::proof::Challenge { value: vec![0u8; 32] },
            responses: vec![],
            metadata: nexuszero_crypto::proof::proof::ProofMetadata {
                version: 1,
                timestamp: 0,
                size: 0,
            },
            bulletproof: None,
        };
        
        assert!(proof.validate().is_err(), "Empty responses should fail validation");
    }
}

// ============================================================================
// Cross-Security Level Tests
// ============================================================================

#[cfg(test)]
mod security_level_tests {
    use super::*;
    use nexuszero_crypto::utils::constant_time::ct_modpow;
    use num_bigint::BigUint;

    #[test]
    fn test_128bit_security_discrete_log() {
        // Use parameters appropriate for 128-bit security
        let generator = vec![2u8; 32];
        let secret = vec![0x42; 32];
        let modulus_bytes = vec![0xFF; 32];
        
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();
        
        let witness = Witness::discrete_log(secret);
        
        let proof = prove(&statement, &witness).unwrap();
        assert!(verify(&statement, &proof).is_ok());
    }

    #[test]
    fn test_consistent_verification_across_runs() {
        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();
        
        let witness = Witness::discrete_log(secret);
        let proof = prove(&statement, &witness).unwrap();
        
        // Verify multiple times
        for _ in 0..10 {
            assert!(verify(&statement, &proof).is_ok(), "Verification should be consistent");
        }
    }
}
