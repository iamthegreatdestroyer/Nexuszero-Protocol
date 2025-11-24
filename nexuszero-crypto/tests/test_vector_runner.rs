/// Unified Test Vector Runner
/// Adapts to nested JSON schemas for LWE, Ring-LWE, and Proof vectors.
use nexuszero_crypto::lattice::lwe::*;
use nexuszero_crypto::lattice::ring_lwe::*;
use nexuszero_crypto::proof::proof::*;
use nexuszero_crypto::proof::statement::*;
use nexuszero_crypto::proof::witness::*;
use serde::Deserialize;
use std::fs;
use rand::thread_rng;
use sha3::{Digest, Sha3_256};

#[derive(Debug, Deserialize)]
struct GenericVectors<T> { vectors: Vec<T> }

// LWE schema
#[derive(Debug, Deserialize)]
struct LWEParametersJson { dimension: usize, modulus: u64, sigma: f64 }
#[derive(Debug, Deserialize)]
struct LWETestCaseJson { plaintext_bit: u8, decryption_result: Option<u8> }
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct LWEVectorJson { name: String, security_level: u32, parameters: LWEParametersJson, test_cases: Vec<LWETestCaseJson> }

// Ring-LWE schema
#[derive(Debug, Deserialize)]
struct RingLWEParametersJson { degree: usize, modulus: u64, sigma: f64 }
#[derive(Debug, Deserialize)]
struct EncryptionTestJson { message: Vec<u8> }
#[derive(Debug, Deserialize)]
struct PolynomialTestJson { operation: Option<String>, poly_a: Option<Vec<i64>>, poly_b: Option<Vec<i64>>, expected: Option<Vec<i64>> }
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct RingLWEVectorJson { name: String, security_level: u32, parameters: RingLWEParametersJson, encryption_tests: Option<Vec<EncryptionTestJson>>, polynomial_tests: Option<Vec<PolynomialTestJson>> }

// Proof schema
#[derive(Debug, Deserialize, Clone)]
struct ProofVectorJson { name: String }

#[test]
fn test_lwe_vectors() {
    let json_data = fs::read_to_string("tests/test_vectors/lwe_test_vectors.json").expect("read LWE vectors");
    let suite: GenericVectors<LWEVectorJson> = serde_json::from_str(&json_data).expect("parse LWE vectors");
    for vector in &suite.vectors {
        let p = &vector.parameters;
        let m = p.dimension * 2; // derive samples
        let params = LWEParameters::new(p.dimension, m, p.modulus, p.sigma);
        let mut rng = thread_rng();
        let (sk, pk) = keygen(&params, &mut rng).expect("keygen");
        for case in &vector.test_cases {
            let bit = case.plaintext_bit != 0;
            let ct = encrypt(&pk, bit, &params, &mut rng).expect("encrypt");
            let dec = decrypt(&sk, &ct, &params).expect("decrypt");
            if let Some(expected) = case.decryption_result { assert_eq!(dec, expected != 0); } else { assert_eq!(dec, bit); }
        }
    }
    println!("✓ LWE vectors OK");
}

#[test]
fn test_ring_lwe_vectors() {
    let json_data = fs::read_to_string("tests/test_vectors/ring_lwe_test_vectors.json").expect("read Ring-LWE vectors");
    let suite: GenericVectors<RingLWEVectorJson> = serde_json::from_str(&json_data).expect("parse Ring-LWE vectors");
    for vector in &suite.vectors {
        let p = &vector.parameters;
        let params = RingLWEParameters::new(p.degree, p.modulus, p.sigma);
        let (sk, pk) = ring_keygen(&params).expect("ring keygen");
        if let Some(enc_tests) = &vector.encryption_tests {
            for t in enc_tests {
                let bits: Vec<bool> = t.message.iter().map(|b| (b & 1)==1).collect();
                let ct = ring_encrypt(&pk, &bits, &params).expect("ring encrypt");
                let dec = ring_decrypt(&sk, &ct, &params).expect("ring decrypt");
                if !bits.is_empty() { assert_eq!(dec[0], bits[0]); }
            }
        }
        if let Some(poly_tests) = &vector.polynomial_tests {
            for pt in poly_tests {
                if pt.operation.as_deref() == Some("addition") {
                    if let (Some(a), Some(b), Some(exp)) = (&pt.poly_a, &pt.poly_b, &pt.expected) {
                        let pa = Polynomial { coeffs: a.clone(), modulus: p.modulus, degree: a.len() };
                        let pb = Polynomial { coeffs: b.clone(), modulus: p.modulus, degree: b.len() };
                        let sum = poly_add(&pa, &pb, p.modulus);
                        for (i, e) in exp.iter().enumerate() { if i < sum.coeffs.len() { assert_eq!(sum.coeffs[i], *e); } }
                    }
                }
            }
        }
    }
    println!("✓ Ring-LWE vectors OK");
}

#[test]
fn test_proof_vectors() {
    let json_data = fs::read_to_string("tests/test_vectors/proof_test_vectors.json").expect("read proof vectors");
    let suite: GenericVectors<ProofVectorJson> = serde_json::from_str(&json_data).expect("parse proof vectors");
    for vector in &suite.vectors {
        if vector.name.contains("Discrete-Log") {
            let generator = vec![2u8; 32];
            let secret = vec![42u8; 32];
            use num_bigint::BigUint; let modulus_bytes = vec![0xFF; 32];
            let gen_big = BigUint::from_bytes_be(&generator); let secret_big = BigUint::from_bytes_be(&secret); let mod_big = BigUint::from_bytes_be(&modulus_bytes);
            let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
            let statement = StatementBuilder::new().discrete_log(generator, public_value).build().expect("statement");
            let witness = Witness::discrete_log(secret);
            let proof = prove(&statement, &witness).expect("prove");
            verify(&statement, &proof).expect("verify");
        } else if vector.name.contains("Preimage") {
            let preimage = b"test_message".to_vec(); let mut hasher = Sha3_256::new(); hasher.update(&preimage); let hash = hasher.finalize().to_vec();
            let statement = StatementBuilder::new().preimage(HashFunction::SHA3_256, hash).build().expect("preimage statement");
            let witness = Witness::preimage(preimage);
            let proof = prove(&statement, &witness).expect("prove preimage");
            verify(&statement, &proof).expect("verify preimage");
        } else if vector.name.contains("Serialization") {
            let preimage = b"serialization".to_vec(); let mut hasher = Sha3_256::new(); hasher.update(&preimage); let hash = hasher.finalize().to_vec();
            let statement = StatementBuilder::new().preimage(HashFunction::SHA3_256, hash).build().expect("ser statement");
            let witness = Witness::preimage(preimage);
            let proof = prove(&statement, &witness).expect("prove ser");
            let bytes = proof.to_bytes().expect("serialize");
            let restored = Proof::from_bytes(&bytes).expect("deserialize");
            verify(&statement, &restored).expect("verify restored");
        }
    }
    println!("✓ Proof vectors OK");
}

#[test]
fn test_polynomial_operations_from_vectors() {
    let json_data = fs::read_to_string("tests/test_vectors/ring_lwe_test_vectors.json").expect("read Ring-LWE vectors");
    let suite: GenericVectors<RingLWEVectorJson> = serde_json::from_str(&json_data).expect("parse Ring-LWE vectors");
    for vector in &suite.vectors { let p=&vector.parameters; if p.degree.is_power_of_two() { if let Some(root)=find_primitive_root(p.degree,p.modulus){ let poly=Polynomial{coeffs:vec![1i64;8],modulus:p.modulus,degree:8}; let t=ntt(&poly,p.modulus,root); assert_eq!(t.len(),poly.coeffs.len()); } } }
    println!("✓ Polynomial ops OK");
}
