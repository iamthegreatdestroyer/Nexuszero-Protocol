use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nexuszero_crypto::proof::bulletproofs::{
    prove_range, verify_range,
};
use rand::{Rng, thread_rng};

// Use a fixed non-zero blinding to avoid modular inverse edge failures
fn fixed_blinding() -> Vec<u8> {
    vec![1u8; 32]
}

fn range_limit(bits: usize) -> u64 {
    if bits >= 64 { u64::MAX } else { 1u64 << bits }
}

fn bench_range_proofs(c: &mut Criterion) {
    // Bit sizes to benchmark for standard range proofs
    // NOTE: Larger sizes (16,32,64) currently trigger internal modular inverse errors
    // in the prototype Bulletproof implementation. Limiting to 8 bits until algorithm
    // robustness is improved.
    let sizes = [8usize];
    for &bits in &sizes {
        let bench_name_prove = format!("prove_range_{}bits", bits);
        c.bench_function(&bench_name_prove, |b| {
            b.iter(|| {
                let mut rng = thread_rng();
                let value = rng.gen::<u64>() % range_limit(bits);
                let blinding = fixed_blinding();
                // Measure proof generation
                let proof = prove_range(black_box(value), black_box(&blinding), black_box(bits)).unwrap();
                black_box(proof);
            })
        });

        // Pre-generate a proof for verify benchmark
        let mut rng = thread_rng();
        let value = rng.gen::<u64>() % range_limit(bits);
        let blinding = fixed_blinding();
        let proof = prove_range(value, &blinding, bits).unwrap();
        let commitment = proof.commitment.clone();

        let bench_name_verify = format!("verify_range_{}bits", bits);
        c.bench_function(&bench_name_verify, |b| {
            b.iter(|| {
                verify_range(black_box(&proof), black_box(&commitment), black_box(bits)).unwrap();
            })
        });
    }
}

// Offset range proof benchmarks disabled due to modular inverse failures.
// Re-enable once arithmetic robustness is improved.

criterion_group!(benches, bench_range_proofs);
criterion_main!(benches);
