use crate::{
    lattice::{lwe::{LWEParameters, keygen, encrypt, decrypt}, ring_lwe::RingLWEParameters},
    proof::bulletproofs::{prove_range, verify_range, pedersen_commit},
    proof::schnorr::{schnorr_keygen, schnorr_sign, schnorr_verify, SchnorrSignature},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use num_bigint::BigUint;

/// Comprehensive test vector generation for cryptographic operations
#[derive(Debug, Serialize, Deserialize)]
pub struct TestVectors {
    pub metadata: TestVectorMetadata,
    pub lwe_vectors: LWEVectors,
    pub bulletproof_vectors: BulletproofVectors,
    pub schnorr_vectors: SchnorrVectors,
    pub hash_vectors: HashVectors,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestVectorMetadata {
    pub version: String,
    pub generated_at: String,
    pub security_level: String,
    pub seed: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LWEVectors {
    pub keygen_tests: Vec<LWEKeygenTest>,
    pub encrypt_decrypt_tests: Vec<LWEEncryptDecryptTest>,
    pub soundness_tests: Vec<LWESoundnessTest>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LWEKeygenTest {
    pub test_id: String,
    pub seed: String,
    pub public_key_hash: String,
    pub secret_key_hash: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LWEEncryptDecryptTest {
    pub test_id: String,
    pub message: bool,
    pub ciphertext_hash: String,
    pub decrypted_message: bool,
    pub success: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LWESoundnessTest {
    pub test_id: String,
    pub wrong_key_used: bool,
    pub decryption_success: bool,
    pub expected_failure: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BulletproofVectors {
    pub valid_range_proofs: Vec<BulletproofRangeTest>,
    pub invalid_range_proofs: Vec<BulletproofInvalidTest>,
    pub edge_case_proofs: Vec<BulletproofEdgeCaseTest>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BulletproofRangeTest {
    pub test_id: String,
    pub value: u64,
    pub blinding_hex: String,
    pub proof_size_bytes: usize,
    pub verification_result: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BulletproofInvalidTest {
    pub test_id: String,
    pub invalid_value: u64,
    pub reason: String,
    pub verification_result: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BulletproofEdgeCaseTest {
    pub test_id: String,
    pub value: u64,
    pub description: String,
    pub verification_result: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SchnorrVectors {
    pub valid_proofs: Vec<SchnorrValidTest>,
    pub invalid_proofs: Vec<SchnorrInvalidTest>,
    pub soundness_tests: Vec<SchnorrSoundnessTest>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SchnorrValidTest {
    pub test_id: String,
    pub witness_hex: String,
    pub statement_hex: String,
    pub proof_response_hex: String,
    pub verification_result: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SchnorrInvalidTest {
    pub test_id: String,
    pub witness_hex: String,
    pub statement_hex: String,
    pub modified_response_hex: String,
    pub verification_result: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SchnorrSoundnessTest {
    pub test_id: String,
    pub false_witness_used: bool,
    pub verification_result: bool,
    pub expected_failure: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HashVectors {
    pub sha256_tests: Vec<HashTest>,
    pub consistency_tests: Vec<HashConsistencyTest>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HashTest {
    pub test_id: String,
    pub input_hex: String,
    pub output_hex: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HashConsistencyTest {
    pub test_id: String,
    pub input_hex: String,
    pub hash1_hex: String,
    pub hash2_hex: String,
    pub consistent: bool,
}

/// Generate comprehensive test vectors for security audit
pub fn generate_test_vectors() -> Result<TestVectors, Box<dyn std::error::Error>> {
    let seed_string = "NEXUSZERO_SECURITY_AUDIT_2024_SEED";
    
    // Hash the seed string to get a 32-byte array for ChaCha20Rng
    let mut hasher = Sha3_256::new();
    hasher.update(seed_string.as_bytes());
    let seed_bytes: [u8; 32] = hasher.finalize().into();
    
    let mut rng = ChaCha20Rng::from_seed(seed_bytes);

    let metadata = TestVectorMetadata {
        version: "1.0".to_string(),
        generated_at: chrono::Utc::now().to_rfc3339(),
        security_level: "128-bit".to_string(),
        seed: hex::encode(seed_bytes),
    };

    // Generate LWE test vectors
    let lwe_vectors = generate_lwe_vectors(&mut rng)?;

    // Generate Bulletproof test vectors
    let bulletproof_vectors = generate_bulletproof_vectors(&mut rng)?;

    // Generate Schnorr test vectors
    let schnorr_vectors = generate_schnorr_vectors(&mut rng)?;

    // Generate hash test vectors
    let hash_vectors = generate_hash_vectors(&mut rng)?;

    Ok(TestVectors {
        metadata,
        lwe_vectors,
        bulletproof_vectors,
        schnorr_vectors,
        hash_vectors,
    })
}

fn generate_lwe_vectors(rng: &mut ChaCha20Rng) -> Result<LWEVectors, Box<dyn std::error::Error>> {
    let params = LWEParameters::new(32, 64, 257, 2.0); // Smaller parameters for testing
    let mut keygen_tests = Vec::new();
    let mut encrypt_decrypt_tests = Vec::new();
    let mut soundness_tests = Vec::new();

    // Generate keygen test vectors
    for i in 0..10 {
        let seed_bytes: [u8; 32] = rng.gen();
        let mut seeded_rng = ChaCha20Rng::from_seed(seed_bytes);

        let (sk, pk) = keygen(&params, &mut seeded_rng)?;

        // Hash the keys for test vectors (since they contain large arrays)
        let mut hasher = Sha3_256::new();
        hasher.update(format!("{:?}", pk.a));
        hasher.update(format!("{:?}", pk.b));
        let pk_hash = hex::encode(hasher.finalize());

        let mut hasher = Sha3_256::new();
        hasher.update(format!("{:?}", sk.s));
        let sk_hash = hex::encode(hasher.finalize());

        keygen_tests.push(LWEKeygenTest {
            test_id: format!("lwe_keygen_{}", i),
            seed: hex::encode(seed_bytes),
            public_key_hash: pk_hash,
            secret_key_hash: sk_hash,
        });
    }

    // Generate encrypt/decrypt test vectors
    for i in 0..20 {
        let seed_bytes: [u8; 32] = rng.gen();
        let mut seeded_rng = ChaCha20Rng::from_seed(seed_bytes);

        let (sk, pk) = keygen(&params, &mut seeded_rng)?;
        let message = rng.gen_bool(0.5);

        let ciphertext = encrypt(&pk, message, &params, &mut seeded_rng)?;
        let decrypted = decrypt(&sk, &ciphertext, &params)?;

        // Hash the ciphertext
        let mut hasher = Sha3_256::new();
        hasher.update(format!("{:?}", ciphertext.u));
        hasher.update(&ciphertext.v.to_le_bytes());
        let ct_hash = hex::encode(hasher.finalize());

        encrypt_decrypt_tests.push(LWEEncryptDecryptTest {
            test_id: format!("lwe_encrypt_decrypt_{}", i),
            message,
            ciphertext_hash: ct_hash,
            decrypted_message: decrypted,
            success: message == decrypted,
        });
    }

    // Generate soundness test vectors (wrong key tests)
    for i in 0..10 {
        let seed_bytes: [u8; 32] = rng.gen();
        let mut seeded_rng = ChaCha20Rng::from_seed(seed_bytes);

        let (_sk1, pk) = keygen(&params, &mut seeded_rng)?;
        let (sk2, _pk2) = keygen(&params, &mut seeded_rng)?;
        let message = rng.gen_bool(0.5);

        let ciphertext = encrypt(&pk, message, &params, &mut seeded_rng)?;
        let decrypted = decrypt(&sk2, &ciphertext, &params); // Wrong key

        soundness_tests.push(LWESoundnessTest {
            test_id: format!("lwe_soundness_{}", i),
            wrong_key_used: true,
            decryption_success: decrypted.is_ok(),
            expected_failure: true, // Should fail with wrong key
        });
    }

    Ok(LWEVectors {
        keygen_tests,
        encrypt_decrypt_tests,
        soundness_tests,
    })
}

fn generate_bulletproof_vectors(rng: &mut ChaCha20Rng) -> Result<BulletproofVectors, Box<dyn std::error::Error>> {
    let mut valid_range_proofs = Vec::new();
    let mut invalid_range_proofs = Vec::new();
    let mut edge_case_proofs = Vec::new();

    // Generate valid range proofs
    for i in 0..15 {
        let value = rng.gen_range(0..(1u64 << 32)); // Values up to 2^32
        let blinding: [u8; 32] = rng.gen();

        let proof = prove_range(value, &blinding, 64)?;
        let commitment = pedersen_commit(value, &blinding)?;
        let verification = verify_range(&proof, &commitment, 64).is_ok();

        valid_range_proofs.push(BulletproofRangeTest {
            test_id: format!("bulletproof_valid_{}", i),
            value,
            blinding_hex: hex::encode(blinding),
            proof_size_bytes: serde_json::to_string(&proof)?.len(),
            verification_result: verification,
        });
    }

    // Generate invalid range proofs
    let invalid_values = vec![
        (1u64 << 33, "Value exceeds maximum range"),
        (u64::MAX, "Maximum u64 value"),
    ];

    for (i, (value, reason)) in invalid_values.iter().enumerate() {
        let blinding: [u8; 32] = rng.gen();

        let proof_result = prove_range(*value, &blinding, 64);
        let verification = if let Ok(proof) = proof_result {
            let commitment = pedersen_commit(*value, &blinding).unwrap_or_default();
            verify_range(&proof, &commitment, 64).is_ok()
        } else {
            false
        };

        invalid_range_proofs.push(BulletproofInvalidTest {
            test_id: format!("bulletproof_invalid_{}", i),
            invalid_value: *value,
            reason: reason.to_string(),
            verification_result: verification,
        });
    }

    // Generate edge case proofs
    let edge_cases = vec![
        (0u64, "Zero value"),
        (1u64, "Minimum positive value"),
        ((1u64 << 32) - 1, "Maximum valid value"),
    ];

    for (i, (value, description)) in edge_cases.iter().enumerate() {
        let blinding: [u8; 32] = rng.gen();

        let proof = prove_range(*value, &blinding, 64)?;
        let commitment = pedersen_commit(*value, &blinding)?;
        let verification = verify_range(&proof, &commitment, 64).is_ok();

        edge_case_proofs.push(BulletproofEdgeCaseTest {
            test_id: format!("bulletproof_edge_{}", i),
            value: *value,
            description: description.to_string(),
            verification_result: verification,
        });
    }

    Ok(BulletproofVectors {
        valid_range_proofs,
        invalid_range_proofs,
        edge_case_proofs,
    })
}

fn generate_schnorr_vectors(rng: &mut ChaCha20Rng) -> Result<SchnorrVectors, Box<dyn std::error::Error>> {
    let mut valid_proofs = Vec::new();
    let mut invalid_proofs = Vec::new();
    let mut soundness_tests = Vec::new();

    // Generate valid proof test vectors
    for i in 0..15 {
        let (private_key, public_key) = schnorr_keygen()?;
        
        let message_len = rng.gen_range(10..=100);
        let message: Vec<u8> = (0..message_len).map(|_| rng.gen()).collect();
        
        let signature = schnorr_sign(&message, &private_key)?;
        let verification_result = schnorr_verify(&message, &signature, &public_key)?;

        valid_proofs.push(SchnorrValidTest {
            test_id: format!("schnorr_valid_{}", i),
            witness_hex: hex::encode(private_key.x.to_bytes_be()),
            statement_hex: hex::encode(&message),
            proof_response_hex: hex::encode(signature.s.to_bytes_be()),
            verification_result,
        });
    }

    // Generate invalid proof test vectors (modified signatures)
    for i in 0..10 {
        let (private_key, public_key) = schnorr_keygen()?;
        
        let message: Vec<u8> = (0..50).map(|_| rng.gen()).collect();
        let signature = schnorr_sign(&message, &private_key)?;
        
        // Modify the signature response
        let modified_s = (&signature.s + BigUint::from(rng.gen_range(1u32..=100u32))) % &public_key.y;
        let modified_signature = SchnorrSignature {
            r: signature.r.clone(),
            s: modified_s,
        };
        
        let verification_result = schnorr_verify(&message, &modified_signature, &public_key)?;

        invalid_proofs.push(SchnorrInvalidTest {
            test_id: format!("schnorr_invalid_{}", i),
            witness_hex: hex::encode(private_key.x.to_bytes_be()),
            statement_hex: hex::encode(&message),
            modified_response_hex: hex::encode(modified_signature.s.to_bytes_be()),
            verification_result,
        });
    }

    // Generate soundness test vectors (wrong keys)
    for i in 0..10 {
        let (private_key1, _) = schnorr_keygen()?;
        let (_, public_key2) = schnorr_keygen()?;
        
        let message: Vec<u8> = (0..50).map(|_| rng.gen()).collect();
        let signature = schnorr_sign(&message, &private_key1)?;
        
        // Try to verify with wrong public key
        let verification_result = schnorr_verify(&message, &signature, &public_key2)?;

        soundness_tests.push(SchnorrSoundnessTest {
            test_id: format!("schnorr_soundness_{}", i),
            false_witness_used: true,
            verification_result,
            expected_failure: true,
        });
    }

    Ok(SchnorrVectors {
        valid_proofs,
        invalid_proofs,
        soundness_tests,
    })
}

fn generate_hash_vectors(rng: &mut ChaCha20Rng) -> Result<HashVectors, Box<dyn std::error::Error>> {
    let mut sha256_tests = Vec::new();
    let mut consistency_tests = Vec::new();

    // Generate SHA256 test vectors
    for i in 0..20 {
        let input_len = rng.gen_range(1..=100);
        let input: Vec<u8> = (0..input_len).map(|_| rng.gen()).collect();

        let mut hasher = Sha3_256::new();
        hasher.update(&input);
        let hash = hasher.finalize();

        sha256_tests.push(HashTest {
            test_id: format!("hash_sha256_{}", i),
            input_hex: hex::encode(&input),
            output_hex: hex::encode(hash),
        });
    }

    // Generate consistency tests
    for i in 0..10 {
        let input_len = rng.gen_range(1..=100);
        let input: Vec<u8> = (0..input_len).map(|_| rng.gen()).collect();

        let mut hasher1 = Sha3_256::new();
        hasher1.update(&input);
        let hash1 = hasher1.finalize();

        let mut hasher2 = Sha3_256::new();
        hasher2.update(&input);
        let hash2 = hasher2.finalize();

        consistency_tests.push(HashConsistencyTest {
            test_id: format!("hash_consistency_{}", i),
            input_hex: hex::encode(&input),
            hash1_hex: hex::encode(hash1),
            hash2_hex: hex::encode(hash2),
            consistent: hash1 == hash2,
        });
    }

    Ok(HashVectors {
        sha256_tests,
        consistency_tests,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_test_vectors() {
        let vectors = generate_test_vectors().unwrap();

        // Verify metadata
        assert_eq!(vectors.metadata.version, "1.0");
        assert_eq!(vectors.metadata.security_level, "128-bit");

        // Verify LWE vectors
        assert_eq!(vectors.lwe_vectors.keygen_tests.len(), 10);
        assert_eq!(vectors.lwe_vectors.encrypt_decrypt_tests.len(), 20);
        assert_eq!(vectors.lwe_vectors.soundness_tests.len(), 10);

        // Verify Bulletproof vectors
        assert_eq!(vectors.bulletproof_vectors.valid_range_proofs.len(), 15);
        assert!(!vectors.bulletproof_vectors.invalid_range_proofs.is_empty());
        assert!(!vectors.bulletproof_vectors.edge_case_proofs.is_empty());

        // Verify Schnorr vectors
        assert_eq!(vectors.schnorr_vectors.valid_proofs.len(), 15);
        assert_eq!(vectors.schnorr_vectors.invalid_proofs.len(), 10);
        assert_eq!(vectors.schnorr_vectors.soundness_tests.len(), 10);
        
        // Verify all valid Schnorr proofs passed
        for proof in &vectors.schnorr_vectors.valid_proofs {
            assert!(proof.verification_result);
        }
        
        // Verify all invalid Schnorr proofs failed
        for proof in &vectors.schnorr_vectors.invalid_proofs {
            assert!(!proof.verification_result);
        }
        
        // Verify all soundness tests failed as expected
        for test in &vectors.schnorr_vectors.soundness_tests {
            assert!(!test.verification_result);
            assert!(test.expected_failure);
        }

        // Verify hash vectors
        assert_eq!(vectors.hash_vectors.sha256_tests.len(), 20);
        assert_eq!(vectors.hash_vectors.consistency_tests.len(), 10);
    }
}