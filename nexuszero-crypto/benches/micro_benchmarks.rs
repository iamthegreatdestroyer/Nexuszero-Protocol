use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nexuszero_crypto::utils::constant_time::ct_dot_product;
use nexuszero_crypto::proof::bulletproofs::pedersen_commit;
use num_bigint::BigUint;
use rand::{thread_rng, Rng};

/// Microbenchmark: Constant-time dot product (used in LWE decrypt)
fn bench_ct_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("ct_dot_product");
    
    for size in [32, 64, 128, 256, 512].iter() {
        let a: Vec<i64> = (0..*size).map(|i| i as i64).collect();
        let b: Vec<i64> = (0..*size).map(|i| (i + 1) as i64).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    ct_dot_product(black_box(&a), black_box(&b))
                });
            },
        );
    }
    group.finish();
}

/// Microbenchmark: Regular (non-constant-time) dot product for comparison
fn bench_regular_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("regular_dot_product");
    
    for size in [32, 64, 128, 256, 512].iter() {
        let a: Vec<i64> = (0..*size).map(|i| i as i64).collect();
        let b: Vec<i64> = (0..*size).map(|i| (i + 1) as i64).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result: i64 = a.iter()
                        .zip(b.iter())
                        .map(|(x, y)| x.wrapping_mul(*y))
                        .sum();
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

/// Microbenchmark: Pedersen commitment (used heavily in Bulletproofs)
fn bench_pedersen_commit(c: &mut Criterion) {
    let mut rng = thread_rng();
    let blinding: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
    
    c.bench_function("pedersen_commit", |b| {
        b.iter(|| {
            pedersen_commit(black_box(42u64), black_box(&blinding))
        });
    });
}

/// Microbenchmark: Inner product with BigUint (computational core of prove_inner_product)
fn bench_biguint_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("biguint_inner_product");
    
    for size in [8, 16, 32, 64].iter() {
        let a: Vec<BigUint> = (0..*size).map(|i| BigUint::from(i as u64)).collect();
        let b: Vec<BigUint> = (0..*size).map(|i| BigUint::from((i + 1) as u64)).collect();
        let modulus = BigUint::from(12289u64);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = a.iter()
                        .zip(b.iter())
                        .fold(BigUint::from(0u32), |acc, (ai, bi)| {
                            (acc + (ai * bi)) % &modulus
                        });
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

/// Microbenchmark: Vector folding operations (critical in recursive halving)
fn bench_vector_folding(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_folding");
    let p = BigUint::from(12289u64);
    
    for size in [8, 16, 32, 64].iter() {
        let a_left: Vec<BigUint> = (0..size/2).map(|i| BigUint::from(i as u64)).collect();
        let a_right: Vec<BigUint> = (size/2..*size).map(|i| BigUint::from(i as u64)).collect();
        let x = BigUint::from(123u32);
        let x_inv = BigUint::from(100u32); // Simplified for benchmark
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    let result: Vec<BigUint> = a_left
                        .iter()
                        .zip(&a_right)
                        .map(|(al, ar)| ((al * &x) + (ar * &x_inv)) % &p)
                        .collect();
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

/// Microbenchmark: Modular exponentiation (used in commitments)
fn bench_modpow(c: &mut Criterion) {
    let base = BigUint::from(3u32);
    let exp = BigUint::from(65537u32);
    let modulus = BigUint::from(12289u64);
    
    c.bench_function("biguint_modpow", |b| {
        b.iter(|| {
            base.modpow(black_box(&exp), black_box(&modulus))
        });
    });
}

/// Microbenchmark: Bit decomposition (used in range proof generation)
fn bench_bit_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_decomposition");
    
    for num_bits in [8, 16, 32, 64].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_bits),
            num_bits,
            |bencher, &bits| {
                let value = (1u64 << (bits - 1)) - 1; // Max value for bit range
                bencher.iter(|| {
                    let decomposed: Vec<u8> = (0..bits)
                        .map(|i| ((value >> i) & 1) as u8)
                        .collect();
                    black_box(decomposed)
                });
            },
        );
    }
    group.finish();
}

/// Microbenchmark: REM_EUCLID operations (used in LWE decrypt)
fn bench_rem_euclid(c: &mut Criterion) {
    let mut group = c.benchmark_group("rem_euclid_ops");
    let q = 12289i64;
    
    for iterations in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(iterations),
            iterations,
            |bencher, &iters| {
                bencher.iter(|| {
                    let mut result = 0i64;
                    for i in 0..iters {
                        let val = (i as i64 * 123 - 456) % q;
                        result = result.wrapping_add(val.rem_euclid(q));
                    }
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ct_dot_product,
    bench_regular_dot_product,
    bench_pedersen_commit,
    bench_biguint_inner_product,
    bench_vector_folding,
    bench_modpow,
    bench_bit_decomposition,
    bench_rem_euclid
);
criterion_main!(benches);
