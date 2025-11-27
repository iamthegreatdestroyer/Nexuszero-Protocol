//! Integration Performance Benchmarks
//!
//! Comprehensive performance validation suite for NexusZero Protocol integration layer.
//! Validates Phase 3 performance targets: <100ms proof generation, <50ms verification.

use criterion::{
    black_box, criterion_group, criterion_main, Criterion, BenchmarkId,
    Throughput, SamplingMode
};
use nexuszero_integration::{
    NexuszeroAPI, ProtocolConfig,
    metrics::{MetricsCollector, ComprehensiveProofMetrics},
    optimization::{CircuitAnalysis, HeuristicOptimizer, Optimizer},
};
use nexuszero_crypto::SecurityLevel;
use std::time::{Duration, Instant};

/// Performance targets validation
const PROOF_GENERATION_TARGET_MS: f64 = 100.0;
const PROOF_VERIFICATION_TARGET_MS: f64 = 50.0;

/// Create test data for discrete log proof
fn create_discrete_log_test_data() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    use num_bigint::BigUint;

    // Use the same modulus as the crypto library (2^256 - 1)
    let modulus = vec![0xFFu8; 32];
    let mod_big = BigUint::from_bytes_be(&modulus);

    // Choose a generator and secret
    let generator = vec![2u8; 32];
    let secret = vec![5u8; 32];

    // Compute public_value = generator^secret mod modulus
    let gen_big = BigUint::from_bytes_be(&generator);
    let secret_big = BigUint::from_bytes_be(&secret);

    let public_value_big = gen_big.modpow(&secret_big, &mod_big);
    let mut public_value = public_value_big.to_bytes_be();

    // Ensure public_value is exactly 32 bytes (pad with zeros if needed)
    while public_value.len() < 32 {
        public_value.insert(0, 0);
    }
    public_value.truncate(32);

    (generator, public_value, secret, modulus)
}
const COMPRESSION_RATIO_TARGET: f64 = 1.0;

/// Benchmark proof generation performance against targets
fn bench_proof_generation_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_generation_performance");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100); // Large sample for accurate percentiles
    group.measurement_time(Duration::from_secs(10));

    // Test configurations that stress performance
    let test_configs = vec![
        ("small_circuit", 64, SecurityLevel::Bit128),
        ("medium_circuit", 256, SecurityLevel::Bit128),
        ("large_circuit", 1024, SecurityLevel::Bit192),
        ("xl_circuit", 4096, SecurityLevel::Bit256),
    ];

    for (name, circuit_size, security_level) in test_configs {
        let config = ProtocolConfig {
            security_level,
            use_compression: true,
            use_optimizer: true,
            max_proof_size: Some(10_000),
            max_verify_time: Some(50.0),
            verify_after_generation: false,
        };

        let mut api = NexuszeroAPI::with_config(config);
        let secret = &[42u8; 32];
        let base = &[7u8; 32];
        let modulus = &1000000007u64.to_le_bytes();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("generation", name),
            &(secret, base, modulus, circuit_size),
            |b, &(s, ba, m, _)| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let _ = black_box(api.prove_discrete_log(s, ba, m));
                    }
                    start.elapsed()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark verification performance against targets
fn bench_verification_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("verification_performance");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));

    let configs = vec![
        ("small_proof", 64),
        ("medium_proof", 256),
        ("large_proof", 1024),
        ("xl_proof", 4096),
    ];

    for (name, circuit_size) in configs {
        let config = ProtocolConfig::default();
        let mut api = NexuszeroAPI::with_config(config);

        // Pre-generate proofs for verification benchmarking
        let proofs: Vec<_> = (0..10).map(|_| {
            let (generator, public_value, secret, _modulus) = create_discrete_log_test_data();
            api.prove_discrete_log(&generator, &public_value, &secret).unwrap()
        }).collect();

        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::new("verification", name), |b| {
            let mut proof_iter = proofs.iter().cycle();
            b.iter(|| {
                let proof = proof_iter.next().unwrap();
                let _ = black_box(api.verify(proof));
            });
        });
    }

    group.finish();
}

/// Benchmark compression performance and ratios
fn bench_compression_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_performance");
    group.sampling_mode(SamplingMode::Flat);

    let compression_configs = vec![
        ("no_compression", false),
        ("with_compression", true),
    ];

    for (name, enabled) in compression_configs {
        let config = ProtocolConfig {
            security_level: SecurityLevel::Bit128,
            use_compression: enabled,
            use_optimizer: true,
            max_proof_size: Some(10_000),
            max_verify_time: Some(50.0),
            verify_after_generation: false,
        };

        let mut api = NexuszeroAPI::with_config(config);

        group.bench_function(BenchmarkId::new("compression", name), |b| {
            b.iter_custom(|iters| {
                let mut total_compressed_size = 0u64;
                let mut total_original_size = 0u64;
                let start = Instant::now();

                for i in 0..iters {
                    let (generator, public_value, secret, _modulus) = create_discrete_log_test_data();
                    let result = api.prove_discrete_log(&generator, &public_value, &secret).unwrap();

                    // Measure sizes
                    total_original_size += result.original_size() as u64;
                    if let Some(compressed) = &result.compressed {
                        total_compressed_size += compressed.data.len() as u64;
                    } else {
                        total_compressed_size += result.original_size() as u64;
                    }
                }

                let elapsed = start.elapsed();

                // Validate compression ratio target
                if enabled && total_original_size > 0 {
                    let ratio = total_compressed_size as f64 / total_original_size as f64;
                    assert!(ratio <= COMPRESSION_RATIO_TARGET,
                           "Compression ratio {:.3} exceeds target {:.3}",
                           ratio, COMPRESSION_RATIO_TARGET);
                }

                elapsed
            });
        });
    }

    group.finish();
}

/// Benchmark optimization performance
fn bench_optimization_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_performance");

    let circuit_sizes = vec![64, 256, 1024, 4096];

    for size in circuit_sizes {
        let optimizer = HeuristicOptimizer::new(SecurityLevel::Bit128);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("circuit_{}", size)),
            &size,
            |b, &circuit_size| {
                let analysis = CircuitAnalysis::from_statement_size(circuit_size);

                b.iter(|| {
                    let _ = black_box(optimizer.optimize(&analysis));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch processing efficiency
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    group.sampling_mode(SamplingMode::Flat);

    let batch_sizes = vec![1, 5, 10, 25, 50, 100];

    for batch_size in batch_sizes {
        let config = ProtocolConfig::default();
        let mut api = NexuszeroAPI::with_config(config);

        // Create batch data
        let batch: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..batch_size)
            .map(|i| (
                vec![42 + i as u8; 32],
                vec![7u8; 32],
                1000000007u64.to_le_bytes().to_vec()
            ))
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch,
            |b, batch| {
                b.iter(|| {
                    for (secret, base, modulus) in batch {
                        let _ = black_box(api.prove_discrete_log(&secret, &base, &modulus));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark metrics collection overhead
fn bench_metrics_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_overhead");

    let config = ProtocolConfig::default();
    let mut api = NexuszeroAPI::with_config(config);

    // Benchmark without metrics
    group.bench_function("no_metrics", |b| {
        b.iter(|| {
            let _ = black_box(api.prove_discrete_log(&[42u8; 32], &[7u8; 32], &1000000007u64.to_le_bytes()));
        });
    });

    // Benchmark with full metrics collection
    group.bench_function("full_metrics", |b| {
        b.iter(|| {
            let mut collector = MetricsCollector::new();
            collector.start();

            let (generator, public_value, secret, _modulus) = create_discrete_log_test_data();
            let result = api.prove_discrete_log(&generator, &public_value, &secret).unwrap();

            // Use the metrics from the result
            let metrics = result.comprehensive_metrics.as_ref().unwrap_or(&ComprehensiveProofMetrics::default()).clone();

            let _ = black_box(collector.finalize());
            let _ = black_box(metrics);
        });
    });

    group.finish();
}

/// Generate a large modulus for testing
fn generate_large_modulus(size_hint: usize) -> u64 {
    // Generate a modulus that's large enough for the circuit size
    // In practice, this would be cryptographically secure
    let base = 1_000_000_007u64;
    let multiplier = (size_hint / 64).max(1) as u64;
    base + multiplier * 1_000_000
}

/// Custom benchmark for target validation
fn validate_performance_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("target_validation");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(50);

    // Test that meets <100ms generation target
    group.bench_function("generation_target_check", |b| {
        let config = ProtocolConfig {
            security_level: SecurityLevel::Bit128,
            use_compression: true,
            use_optimizer: true,
            max_proof_size: Some(10_000),
            max_verify_time: Some(50.0),
            verify_after_generation: false,
        };

        let mut api = NexuszeroAPI::with_config(config);

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = api.prove_discrete_log(&[42u8; 32], &[7u8; 32], &1000000007u64.to_le_bytes());
            }
            let elapsed = start.elapsed();
            let avg_time = elapsed.as_millis() as f64 / iters as f64;

            // Assert target is met
            assert!(avg_time < PROOF_GENERATION_TARGET_MS,
                   "Generation time {:.2}ms exceeds target {:.2}ms",
                   avg_time, PROOF_GENERATION_TARGET_MS);

            elapsed
        });
    });

    // Test that meets <50ms verification target
    group.bench_function("verification_target_check", |b| {
        let config = ProtocolConfig::default();
        let mut api = NexuszeroAPI::with_config(config);
        let (generator, public_value, secret, _modulus) = create_discrete_log_test_data();
        let result = api.prove_discrete_log(&generator, &public_value, &secret).unwrap();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = api.verify(&result);
            }
            let elapsed = start.elapsed();
            let avg_time = elapsed.as_millis() as f64 / iters as f64;

            // Assert target is met
            assert!(avg_time < PROOF_VERIFICATION_TARGET_MS,
                   "Verification time {:.2}ms exceeds target {:.2}ms",
                   avg_time, PROOF_VERIFICATION_TARGET_MS);

            elapsed
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_proof_generation_performance,
    bench_verification_performance,
    bench_compression_performance,
    bench_optimization_performance,
    bench_batch_processing,
    bench_metrics_overhead,
    validate_performance_targets,
);

criterion_main!(benches);
