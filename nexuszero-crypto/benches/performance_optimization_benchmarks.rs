//! Performance benchmarks for zero-knowledge proof optimizations
//!
//! This module provides comprehensive benchmarks for measuring the performance
//! improvements from parallel processing, caching, and adaptive algorithms.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nexuszero_crypto::proof::*;
use nexuszero_crypto::proof::performance_optimization::*;
use nexuszero_crypto::proof::statement::StatementBuilder;
use nexuszero_crypto::proof::witness::Witness;
use nexuszero_crypto::proof::proof::{prove, LegacyProver, LegacyVerifier};
use nexuszero_crypto::SecurityLevel;
use std::collections::HashMap;
use num_bigint::BigUint;
use nexuszero_crypto::utils::constant_time::ct_modpow;

/// Create valid test data for discrete log proof
fn create_discrete_log_test_data(count: usize) -> (Vec<Statement>, Vec<Witness>) {
    let modulus = vec![0xFFu8; 32];
    let mod_big = BigUint::from_bytes_be(&modulus);

    let mut statements = Vec::new();
    let mut witnesses = Vec::new();

    for i in 0..count {
        // Choose a generator and secret
        let generator = vec![2u8; 32];
        let secret_value = ((i % 256) as u8).max(1); // Ensure secret is not zero
        let secret = vec![secret_value; 32];

        // Compute public_value = generator^secret mod modulus
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);

        let public_value_big = ct_modpow(&gen_big, &secret_big, &mod_big);
        let mut public_value = public_value_big.to_bytes_be();

        // Ensure public_value is exactly 32 bytes
        while public_value.len() < 32 {
            public_value.insert(0, 0);
        }
        public_value.truncate(32);

        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();

        let witness = Witness::discrete_log(secret);

        statements.push(statement);
        witnesses.push(witness);
    }

    (statements, witnesses)
}

/// Benchmark parallel batch proving
pub fn bench_parallel_batch_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_batch_prover");

    // Create test data - discrete log statements
    let (statements, witnesses) = create_discrete_log_test_data(100);

    let config = ProverConfig {
        security_level: SecurityLevel::Bit128,
        optimizations: HashMap::new(),
        backend_params: HashMap::new(),
    };

    // Test different worker counts
    for worker_count in [1, 2, 4].iter() {
        let parallel_prover = ParallelBatchProver::new(LegacyProver, *worker_count, 100);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("workers_{}", worker_count)),
            worker_count,
            |b, _| {
                b.iter(|| {
                    let prover = &parallel_prover;
                    let statements = &statements;
                    let witnesses = &witnesses;
                    let config = &config;
                    tokio::runtime::Runtime::new().unwrap().block_on(async {
                        black_box(prover.prove_large_batch(statements, witnesses, config).await.unwrap())
                    })
                });
            }
        );
    }

    group.finish();
}

/// Benchmark optimized batch verification with caching
pub fn bench_optimized_batch_verifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimized_batch_verifier");

    // Create test data
    let (statements, witnesses) = create_discrete_log_test_data(100);
    let mut proofs = Vec::new();

    for (stmt, wit) in statements.iter().zip(witnesses.iter()) {
        let proof = prove(stmt, wit).unwrap();
        proofs.push(proof);
    }

    let config = VerifierConfig {
        security_level: SecurityLevel::Bit128,
        optimizations: HashMap::new(),
        backend_params: HashMap::new(),
    };

    // Test different cache sizes
    for cache_size in [10, 50, 100].iter() {
        let verifier = OptimizedBatchVerifier::new(LegacyVerifier, *cache_size, 4);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("cache_{}", cache_size)),
            cache_size,
            |b, _| {
                b.iter(|| {
                    let verifier = &verifier;
                    let statements = &statements;
                    let proofs = &proofs;
                    let config = &config;
                    tokio::runtime::Runtime::new().unwrap().block_on(async {
                        black_box(verifier.verify_batch_optimized(statements, proofs, config).await.unwrap())
                    })
                });
            }
        );
    }

    group.finish();
}

/// Benchmark adaptive prover strategy selection
pub fn bench_adaptive_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_prover");

    // Create test data
    let (statements, witnesses) = create_discrete_log_test_data(50);

    let config = ProverConfig {
        security_level: SecurityLevel::Bit128,
        optimizations: HashMap::new(),
        backend_params: HashMap::new(),
    };

    let mut adaptive_prover = AdaptiveProver::new();

    // Add some strategies (using the base prove function)
    adaptive_prover.add_strategy("direct".to_string(), Box::new(LegacyProver));
    adaptive_prover.add_strategy("optimized".to_string(), Box::new(LegacyProver));

    group.bench_function("adaptive_strategy_selection", |b| {
        b.iter(|| {
            let adaptive_prover = &adaptive_prover;
            let statements = &statements;
            let witnesses = &witnesses;
            let config = &config;
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                for (stmt, wit) in statements.iter().zip(witnesses.iter()) {
                    black_box(adaptive_prover.prove_adaptive(stmt, wit, config).await.unwrap());
                }
            })
        });
    });

    group.finish();
}

/// Benchmark performance monitoring
pub fn bench_performance_monitor(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_monitor");

    let thresholds = PerformanceThresholds {
        max_proving_time_ms: 100.0,
        max_verification_time_ms: 50.0,
        min_proofs_per_second: 10.0,
        max_memory_usage_bytes: 1024 * 1024 * 1024, // 1GB
        max_cpu_utilization_percent: 80.0,
        min_cache_hit_rate: 0.8,
    };

    let mut monitor = PerformanceMonitor::new(thresholds);

    group.bench_function("record_and_check_metrics", |b| {
        b.iter(|| {
            let monitor = &mut monitor;
            // Simulate some performance data
            let metrics = PerformanceMetrics {
                avg_proving_time_ms: 50.0,
                avg_verification_time_ms: 10.0,
                proofs_per_second: 20.0,
                memory_usage_bytes: 1024 * 1024,
                cpu_utilization_percent: 50.0,
                cache_hit_rate: 0.9,
            };
            black_box(monitor.check_and_alert(&metrics));
        });
    });

    group.finish();
}

/// Benchmark memory scaling
pub fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");

    for size in [10, 50, 100, 500].iter() {
        // Create test data
        let (statements, witnesses) = create_discrete_log_test_data(*size);

        let config = ProverConfig {
            security_level: SecurityLevel::Bit128,
            optimizations: HashMap::new(),
            backend_params: HashMap::new(),
        };

        let parallel_prover = ParallelBatchProver::new(LegacyProver, 4, 100);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("batch_size_{}", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let prover = &parallel_prover;
                    let statements = &statements;
                    let witnesses = &witnesses;
                    let config = &config;
                    tokio::runtime::Runtime::new().unwrap().block_on(async {
                        black_box(prover.prove_large_batch(statements, witnesses, config).await.unwrap())
                    })
                });
            }
        );
    }

    group.finish();
}

/// Benchmark cache performance
pub fn bench_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");

    let verifier = OptimizedBatchVerifier::new(LegacyVerifier, 100, 4);

    // Create test data
    let (statements, witnesses) = create_discrete_log_test_data(50);

    let config = ProverConfig {
        security_level: SecurityLevel::Bit128,
        optimizations: HashMap::new(),
        backend_params: HashMap::new(),
    };

    // Generate proofs
    let mut proofs = Vec::new();
    for (stmt, wit) in statements.iter().zip(witnesses.iter()) {
        let proof = prove(stmt, wit).unwrap();
        proofs.push(proof);
    }

    let config = VerifierConfig {
        security_level: SecurityLevel::Bit128,
        optimizations: HashMap::new(),
        backend_params: HashMap::new(),
    };

    group.bench_function("cache_hit_ratio", |b| {
        b.iter(|| {
            let verifier = &verifier;
            let statements = &statements;
            let proofs = &proofs;
            let config = &config;
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                // First run - cache miss
                black_box(verifier.verify_batch_optimized(statements, proofs, config).await.unwrap());
                // Second run - cache hit
                black_box(verifier.verify_batch_optimized(statements, proofs, config).await.unwrap())
            })
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_batch_prover,
    bench_optimized_batch_verifier,
    bench_adaptive_prover,
    bench_performance_monitor,
    bench_memory_scaling,
    bench_cache_performance
);
criterion_main!(benches);