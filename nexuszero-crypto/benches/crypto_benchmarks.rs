use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nexuszero_crypto::lattice::LWEParameters;
use nexuszero_crypto::lattice::lwe::{keygen, encrypt, decrypt};
use rand::thread_rng;

fn bench_lwe_encryption(c: &mut Criterion) {
    let params = LWEParameters::new(256, 512, 12289, 3.2);
    let mut rng = thread_rng();
    let (_, pk) = keygen(&params, &mut rng).unwrap();

    c.bench_function("lwe_encrypt_128bit", |b| {
        b.iter(|| {
            encrypt(black_box(&pk), black_box(true), black_box(&params), black_box(&mut rng))
        });
    });
}

fn bench_lwe_decryption(c: &mut Criterion) {
    let params = LWEParameters::new(256, 512, 12289, 3.2);
    let mut rng = thread_rng();
    let (sk, pk) = keygen(&params, &mut rng).unwrap();
    let ct = encrypt(&pk, true, &params, &mut rng).unwrap();

    c.bench_function("lwe_decrypt_128bit", |b| {
        b.iter(|| {
            decrypt(black_box(&sk), black_box(&ct), black_box(&params))
        });
    });
}

criterion_group!(benches, bench_lwe_encryption, bench_lwe_decryption);
criterion_main!(benches);
