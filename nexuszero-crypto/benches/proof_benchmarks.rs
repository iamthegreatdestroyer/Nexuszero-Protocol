use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nexuszero_crypto::proof::statement::{StatementBuilder, HashFunction};
use nexuszero_crypto::proof::witness::Witness;
use nexuszero_crypto::proof::proof::{prove, verify, Proof};
use num_bigint::BigUint;
use sha3::{Digest, Sha3_256};

fn bench_discrete_log(c: &mut Criterion) {
    let generator = vec![2u8; 32];
    let secret = vec![42u8; 32];
    let modulus_bytes = vec![0xFF; 32];
    let gen_big = BigUint::from_bytes_be(&generator);
    let secret_big = BigUint::from_bytes_be(&secret);
    let mod_big = BigUint::from_bytes_be(&modulus_bytes);
    let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();

    let statement = StatementBuilder::new()
        .discrete_log(generator.clone(), public_value)
        .build()
        .unwrap();
    let witness = Witness::discrete_log(secret.clone());

    c.bench_function("prove_discrete_log_micro", |b| {
        b.iter(|| prove(black_box(&statement), black_box(&witness)))
    });

    let proof = prove(&statement, &witness).unwrap();
    c.bench_function("verify_discrete_log_micro", |b| {
        b.iter(|| verify(black_box(&statement), black_box(&proof)))
    });

    // Serialize / Deserialize micro benches
    c.bench_function("serialize_discrete_log_proof", |b| {
        b.iter(|| proof.to_bytes())
    });

    let serialized = proof.to_bytes().unwrap();
    c.bench_function("deserialize_discrete_log_proof", |b| {
        b.iter(|| Proof::from_bytes(black_box(&serialized)))
    });
}

fn bench_preimage(c: &mut Criterion) {
    let preimage = b"benchmark_message_micro".to_vec();
    let mut hasher = Sha3_256::new();
    hasher.update(&preimage);
    let hash = hasher.finalize().to_vec();

    let statement = StatementBuilder::new()
        .preimage(HashFunction::SHA3_256, hash)
        .build()
        .unwrap();
    let witness = Witness::preimage(preimage);

    c.bench_function("prove_preimage_micro", |b| {
        b.iter(|| prove(black_box(&statement), black_box(&witness)))
    });

    let proof = prove(&statement, &witness).unwrap();
    c.bench_function("verify_preimage_micro", |b| {
        b.iter(|| verify(black_box(&statement), black_box(&proof)))
    });

    c.bench_function("serialize_preimage_proof", |b| {
        b.iter(|| proof.to_bytes())
    });

    let serialized = proof.to_bytes().unwrap();
    c.bench_function("deserialize_preimage_proof", |b| {
        b.iter(|| Proof::from_bytes(black_box(&serialized)))
    });
}

criterion_group!(benches, bench_discrete_log, bench_preimage);
criterion_main!(benches);
