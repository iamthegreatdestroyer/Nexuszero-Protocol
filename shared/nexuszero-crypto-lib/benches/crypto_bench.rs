//! Crypto benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nexuszero_crypto_lib::prelude::*;

fn bench_hashing(c: &mut Criterion) {
    let data = vec![0u8; 1024];
    
    c.bench_function("sha256_1kb", |b| {
        b.iter(|| sha256(black_box(&data)))
    });
    
    c.bench_function("sha3_256_1kb", |b| {
        b.iter(|| sha3_256(black_box(&data)))
    });
    
    c.bench_function("blake3_1kb", |b| {
        b.iter(|| blake3_hash(black_box(&data)))
    });
}

fn bench_encryption(c: &mut Criterion) {
    let key = AesGcm::generate_key();
    let cipher = AesGcm::new(&key);
    let data = vec![0u8; 1024];
    
    c.bench_function("aes_gcm_encrypt_1kb", |b| {
        b.iter(|| cipher.encrypt(black_box(&data)))
    });
}

fn bench_signing(c: &mut Criterion) {
    let keypair = Ed25519KeyPair::generate();
    let message = vec![0u8; 256];
    
    c.bench_function("ed25519_sign", |b| {
        b.iter(|| keypair.sign(black_box(&message)))
    });
}

criterion_group!(benches, bench_hashing, bench_encryption, bench_signing);
criterion_main!(benches);
