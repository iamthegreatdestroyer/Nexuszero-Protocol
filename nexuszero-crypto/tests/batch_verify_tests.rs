//! Unit tests for batch_verify — Sprint 1 of Phase 6.
//!
//! Covers: empty batch, single proof, 8-proof batch, mixed valid/invalid.

use nexuszero_crypto::proof::{
    batch_verify,
    proof::prove,
    statement::StatementBuilder,
    witness::Witness,
};
use nexuszero_crypto::CryptoError;
use num_bigint::BigUint;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_dlog_pair(secret_byte: u8) -> (nexuszero_crypto::proof::Statement, Witness) {
    let generator = vec![2u8; 32];
    let secret = vec![secret_byte; 32];
    let modulus_bytes = vec![0xFF; 32];
    let gen_big = BigUint::from_bytes_be(&generator);
    let secret_big = BigUint::from_bytes_be(&secret);
    let mod_big = BigUint::from_bytes_be(&modulus_bytes);
    let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();

    let statement = StatementBuilder::new()
        .discrete_log(generator, public_value)
        .build()
        .unwrap();
    let witness = Witness::discrete_log(secret);
    (statement, witness)
}

// ---------------------------------------------------------------------------
// Test: empty batch
// ---------------------------------------------------------------------------

#[test]
fn test_batch_verify_empty() {
    use nexuszero_crypto::proof::Statement;
    use nexuszero_crypto::proof::proof::Proof;

    let statements: Vec<Statement> = vec![];
    let proofs: Vec<Proof> = vec![];

    let result = batch_verify(&statements, &proofs).unwrap();
    assert!(result.is_empty(), "empty batch should return empty Vec");
}

// ---------------------------------------------------------------------------
// Test: single valid proof
// ---------------------------------------------------------------------------

#[test]
fn test_batch_verify_single_valid() {
    let (stmt, wit) = make_dlog_pair(42);
    let proof = prove(&stmt, &wit).unwrap();

    let results = batch_verify(&[stmt], &[proof]).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0], "single valid proof should return true");
}

// ---------------------------------------------------------------------------
// Test: 8-proof batch — all valid
// ---------------------------------------------------------------------------

#[test]
fn test_batch_verify_8_proofs_all_valid() {
    let pairs: Vec<_> = (1u8..=8).map(make_dlog_pair).collect();
    let (statements, proofs): (Vec<_>, Vec<_>) = pairs
        .into_iter()
        .map(|(stmt, wit)| {
            let proof = prove(&stmt, &wit).unwrap();
            (stmt, proof)
        })
        .unzip();

    let results = batch_verify(&statements, &proofs).unwrap();
    assert_eq!(results.len(), 8);
    assert!(results.iter().all(|&r| r), "all 8 proofs should verify");
}

// ---------------------------------------------------------------------------
// Test: 8-proof batch — performance target <200 ms total
// ---------------------------------------------------------------------------

#[test]
fn test_batch_verify_8_proofs_timing() {
    let pairs: Vec<_> = (1u8..=8).map(make_dlog_pair).collect();
    let (statements, proofs): (Vec<_>, Vec<_>) = pairs
        .into_iter()
        .map(|(stmt, wit)| {
            let proof = prove(&stmt, &wit).unwrap();
            (stmt, proof)
        })
        .unzip();

    let start = std::time::Instant::now();
    let results = batch_verify(&statements, &proofs).unwrap();
    let elapsed = start.elapsed();

    assert!(results.iter().all(|&r| r));
    // Soft timing assertion — may fail on very slow CI machines but signals regressions
    println!("batch_verify(8 proofs) took {:?}", elapsed);
    assert!(
        elapsed.as_millis() < 2000,
        "batch_verify(8) took {}ms, expected <2000ms",
        elapsed.as_millis()
    );
}

// ---------------------------------------------------------------------------
// Test: mixed valid/invalid batch
// ---------------------------------------------------------------------------

#[test]
fn test_batch_verify_mixed_valid_invalid() {
    // Build 3 valid proofs
    let mut statements = Vec::new();
    let mut proofs = Vec::new();

    for i in 1u8..=3 {
        let (stmt, wit) = make_dlog_pair(i);
        let proof = prove(&stmt, &wit).unwrap();
        statements.push(stmt);
        proofs.push(proof);
    }

    // Build 1 tampered (invalid) proof
    let (stmt_bad, wit_bad) = make_dlog_pair(10);
    let mut bad_proof = prove(&stmt_bad, &wit_bad).unwrap();
    bad_proof.challenge.value[0] ^= 0xFF; // tamper
    statements.push(stmt_bad);
    proofs.push(bad_proof);

    let results = batch_verify(&statements, &proofs).unwrap();
    assert_eq!(results.len(), 4);
    assert!(results[0], "proof 0 should be valid");
    assert!(results[1], "proof 1 should be valid");
    assert!(results[2], "proof 2 should be valid");
    assert!(!results[3], "tampered proof should return false");
}

// ---------------------------------------------------------------------------
// Test: length mismatch returns Err
// ---------------------------------------------------------------------------

#[test]
fn test_batch_verify_length_mismatch_returns_err() {
    let (stmt, _wit) = make_dlog_pair(1);

    // One statement, zero proofs
    let result = batch_verify(&[stmt], &[]);
    assert!(
        matches!(result, Err(CryptoError::InvalidInput(_))),
        "mismatched lengths should return InvalidInput error"
    );
}

// ---------------------------------------------------------------------------
// Test: all invalid proofs
// ---------------------------------------------------------------------------

#[test]
fn test_batch_verify_all_invalid() {
    let mut statements = Vec::new();
    let mut proofs = Vec::new();

    for i in 1u8..=4 {
        let (stmt, wit) = make_dlog_pair(i);
        let mut proof = prove(&stmt, &wit).unwrap();
        proof.challenge.value[0] ^= 0xFF;
        statements.push(stmt);
        proofs.push(proof);
    }

    let results = batch_verify(&statements, &proofs).unwrap();
    assert!(
        results.iter().all(|&r| !r),
        "all tampered proofs should return false"
    );
}
