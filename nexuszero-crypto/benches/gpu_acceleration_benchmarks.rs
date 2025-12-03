//! GPU acceleration benchmarks
//!
//! This module benchmarks the performance improvements achieved through
//! GPU-accelerated modular arithmetic operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "gpu")]
use nexuszero_crypto::utils::{
    gpu_acceleration_available, init_gpu_acceleration,
    gpu_montgomery_mul_batch, gpu_modular_exponentiation,
    gpu_batch_modular_multiplication,
};

use nexuszero_crypto::utils::{
    montgomery_modmul, montgomery_modpow,
};
use num_bigint::BigUint;
use num_traits::identities::{Zero, One};
use rand::Rng;

#[cfg(feature = "gpu")]
use std::sync::Once;

/// Initialize GPU acceleration once for all benchmarks
#[cfg(feature = "gpu")]
static GPU_INIT: Once = Once::new();

#[cfg(feature = "gpu")]
fn ensure_gpu_initialized() {
    GPU_INIT.call_once(|| {
        // Initialize GPU acceleration if available
        if let Ok(runtime) = tokio::runtime::Runtime::new() {
            runtime.block_on(async {
                let _ = init_gpu_acceleration().await;
            });
        }
    });
}

/// Benchmark CPU vs GPU Montgomery multiplication
fn bench_montgomery_multiplication(c: &mut Criterion) {
    #[cfg(feature = "gpu")]
    ensure_gpu_initialized();
    
    let mut group = c.benchmark_group("montgomery_multiplication");

    // Test data
    let modulus = BigUint::from(0xFFFFFFFFFFFFFFFFu64); // 64-bit modulus
    let a = BigUint::from(0x123456789ABCDEF0u64);
    let b = BigUint::from(0xFEDCBA9876543210u64);

    group.bench_function("cpu_montgomery_mul", |bencher| {
        bencher.iter(|| {
            black_box(montgomery_modmul(&a, &b, &modulus));
        });
    });

    // Only benchmark GPU if feature is enabled and available
    #[cfg(feature = "gpu")]
    if gpu_acceleration_available() {
        let a_u32 = a.to_u32_digits();
        let b_u32 = b.to_u32_digits();
        let mod_u32 = modulus.to_u32_digits()[0];

        // Calculate Montgomery parameters
        let r = BigUint::from(1u32) << 64; // R = 2^64 for 64-bit modulus
        let r_squared = (&r * &r) % &modulus;
        let r_inv = montgomery_mod_inverse(&r, &modulus);

        let mont_r = r.to_u32_digits()[0];
        let mont_r_squared = r_squared.to_u32_digits()[0];
        let mont_r_inv = r_inv.to_u32_digits()[0];

        group.bench_function("gpu_montgomery_mul_batch", |bencher| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            bencher.iter(|| {
                runtime.block_on(async {
                    black_box(gpu_montgomery_mul_batch(
                        &[a_u32[0]],
                        &[b_u32[0]],
                        mod_u32,
                        mont_r,
                        mont_r_squared,
                        mont_r_inv,
                    ).await.unwrap());
                });
            });
        });
    }

    group.finish();
}

/// Benchmark CPU vs GPU modular exponentiation
fn bench_modular_exponentiation(c: &mut Criterion) {
    #[cfg(feature = "gpu")]
    ensure_gpu_initialized();
    
    let mut group = c.benchmark_group("modular_exponentiation");

    // Test data - smaller for reasonable benchmark times
    let modulus = BigUint::from(0xFFFFFFFFu64); // 32-bit modulus
    let base = BigUint::from(12345u32);
    let exponent = BigUint::from(0xFFFFu32); // 16-bit exponent

    group.bench_function("cpu_montgomery_modpow", |bencher| {
        bencher.iter(|| {
            black_box(montgomery_modpow(&base, &exponent, &modulus));
        });
    });

    // Only benchmark GPU if feature is enabled and available
    #[cfg(feature = "gpu")]
    if gpu_acceleration_available() {
        let base_u32 = base.to_u32_digits()[0];
        let mod_u32 = modulus.to_u32_digits()[0];

        group.bench_function("gpu_modular_exponentiation", |bencher| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            bencher.iter(|| {
                runtime.block_on(async {
                    black_box(gpu_modular_exponentiation(
                        base_u32,
                        &exponent,
                        mod_u32,
                    ).await.unwrap());
                });
            });
        });
    }

    group.finish();
}

/// Benchmark batch modular multiplication performance
fn bench_batch_modular_multiplication(c: &mut Criterion) {
    #[cfg(feature = "gpu")]
    ensure_gpu_initialized();
    
    let mut group = c.benchmark_group("batch_modular_multiplication");

    // Generate test data
    let mut rng = rand::thread_rng();
    let batch_size = 1024;

    let a_values: Vec<u32> = (0..batch_size).map(|_| rng.gen()).collect();
    let b_values: Vec<u32> = (0..batch_size).map(|_| rng.gen()).collect();
    let moduli: Vec<u32> = (0..batch_size).map(|_| {
        loop {
            let m = rng.gen::<u32>() | 1; // Ensure odd modulus
            if m > 1 { return m; }
        }
    }).collect();

    // CPU benchmark (simulated batch using individual operations)
    group.bench_function("cpu_batch_modmul", |bencher| {
        bencher.iter(|| {
            let results: Vec<u32> = a_values.iter().zip(b_values.iter()).zip(moduli.iter())
                .map(|((&a, &b), &m)| {
                    let a_big = BigUint::from(a);
                    let b_big = BigUint::from(b);
                    let m_big = BigUint::from(m);
                    let result = montgomery_modmul(&a_big, &b_big, &m_big);
                    if result.to_u32_digits().is_empty() {
                        0
                    } else {
                        result.to_u32_digits()[0]
                    }
                })
                .collect();
            black_box(results);
        });
    });

    // GPU benchmark
    #[cfg(feature = "gpu")]
    if gpu_acceleration_available() {
        group.bench_function("gpu_batch_modmul", |bencher| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            bencher.iter(|| {
                runtime.block_on(async {
                    black_box(gpu_batch_modular_multiplication(
                        &a_values,
                        &b_values,
                        &moduli,
                    ).await.unwrap());
                });
            });
        });
    }

    group.finish();
}

/// Helper function to compute Montgomery modular inverse
#[allow(dead_code)]
fn montgomery_mod_inverse(r: &BigUint, modulus: &BigUint) -> BigUint {
    // For R = 2^k, R^-1 mod modulus can be computed efficiently
    // This is a simplified implementation for benchmarking
    let mut result = BigUint::one();
    let mut base = r.clone();
    let exp = modulus - BigUint::one();

    let mut exp_bits = exp.clone();
    while exp_bits > BigUint::zero() {
        if &exp_bits % 2u32 == BigUint::one() {
            result = (result * &base) % modulus;
        }
        base = (&base * &base) % modulus;
        exp_bits >>= 1;
    }

    result
}

criterion_group!(
    name = gpu_benchmarks;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(10));
    targets =
        bench_montgomery_multiplication,
        bench_modular_exponentiation,
        bench_batch_modular_multiplication
);

criterion_main!(gpu_benchmarks);
