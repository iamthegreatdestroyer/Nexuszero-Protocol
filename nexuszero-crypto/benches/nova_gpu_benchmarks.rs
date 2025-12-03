// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// GPU Acceleration Benchmarks for Nova Operations
// Measures performance of MSM, NTT, and commitment operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use nexuszero_crypto::proof::nova::gpu::{
    GPUAccelerationManager, GPUConfig, NovaGPU, ScalarPoint, NTTDomain,
};
use std::time::Duration;
use tokio::runtime::Runtime;

/// Generate random scalar points for MSM benchmarks
fn generate_scalar_points(n: usize) -> Vec<ScalarPoint> {
    (0..n)
        .map(|i| ScalarPoint {
            scalar: (0..32).map(|j| ((i + j) % 256) as u8).collect(),
            point: (0..64).map(|j| ((i * 2 + j) % 256) as u8).collect(),
        })
        .collect()
}

/// Generate random field elements for NTT benchmarks
fn generate_field_elements(n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| (0..32).map(|j| ((i + j) % 256) as u8).collect())
        .collect()
}

fn benchmark_gpu_initialization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("gpu_manager_initialization", |b| {
        b.iter(|| {
            rt.block_on(async {
                let manager = GPUAccelerationManager::new(false).await.unwrap();
                black_box(manager.is_gpu_available())
            })
        })
    });

    c.bench_function("gpu_config_creation_default", |b| {
        b.iter(|| {
            black_box(GPUConfig::default())
        })
    });

    c.bench_function("gpu_config_creation_high_performance", |b| {
        b.iter(|| {
            black_box(GPUConfig::high_performance())
        })
    });

    c.bench_function("gpu_config_creation_memory_efficient", |b| {
        b.iter(|| {
            black_box(GPUConfig::memory_efficient())
        })
    });
}

fn benchmark_msm_cpu_fallback(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("msm_cpu_fallback");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [10, 100, 500, 1000].iter() {
        let scalar_points = generate_scalar_points(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &scalar_points,
            |b, points| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut manager = GPUAccelerationManager::new(false).await.unwrap();
                        let result = manager.msm(black_box(points)).await.unwrap();
                        black_box(result)
                    })
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_ntt_cpu_fallback(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("ntt_cpu_fallback");
    group.measurement_time(Duration::from_secs(10));
    
    for log_size in [8, 10, 12, 14].iter() {
        let size = 1 << log_size;
        let coefficients = generate_field_elements(size);
        let domain = NTTDomain::new(size).unwrap();
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("forward", log_size),
            &(coefficients, domain),
            |b, (coeffs, dom)| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut manager = GPUAccelerationManager::new(false).await.unwrap();
                        let result = manager.ntt_forward(black_box(coeffs), black_box(dom)).await.unwrap();
                        black_box(result)
                    })
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_domain_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_domain");
    
    for log_size in [8, 10, 12, 14, 16].iter() {
        let size = 1 << log_size;
        
        group.bench_with_input(
            BenchmarkId::new("creation", log_size),
            &size,
            |b, &s| {
                b.iter(|| {
                    black_box(NTTDomain::new(s).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_scalar_point_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_point_ops");
    
    for size in [10, 100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("generation", size),
            size,
            |b, &s| {
                b.iter(|| {
                    black_box(generate_scalar_points(s))
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_field_element_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_element_ops");
    
    for log_size in [8, 10, 12, 14].iter() {
        let size = 1 << log_size;
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("generation", log_size),
            &size,
            |b, &s| {
                b.iter(|| {
                    black_box(generate_field_elements(s))
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_commitment_cpu_fallback(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("commitment_cpu_fallback");
    group.measurement_time(Duration::from_secs(5));
    
    let value: Vec<u8> = (0..32).collect();
    let blinding: Vec<u8> = (32..64).collect();
    
    group.bench_function("pedersen_single", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut gpu = NovaGPU::new(GPUConfig::default()).await.unwrap();
                let result = gpu.pedersen_commit(black_box(&value), black_box(&blinding)).await.unwrap();
                black_box(result)
            })
        })
    });

    // Batch commitment benchmarks
    for batch_size in [10, 50, 100].iter() {
        let values: Vec<Vec<u8>> = (0..*batch_size)
            .map(|i| (0..32).map(|j| ((i + j) % 256) as u8).collect())
            .collect();
        let blindings: Vec<Vec<u8>> = (0..*batch_size)
            .map(|i| (0..32).map(|j| ((i + j + 100) % 256) as u8).collect())
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("pedersen_batch", batch_size),
            &(values, blindings),
            |b, (vals, blinds)| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut gpu = NovaGPU::new(GPUConfig::default()).await.unwrap();
                        let result = gpu.batch_pedersen_commit(black_box(vals), black_box(blinds)).await.unwrap();
                        black_box(result)
                    })
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_gpu_metrics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("gpu_metrics");
    
    group.bench_function("metrics_retrieval", |b| {
        b.iter(|| {
            rt.block_on(async {
                let gpu = NovaGPU::new(GPUConfig::default()).await.unwrap();
                black_box(gpu.metrics().clone())
            })
        })
    });

    group.bench_function("metrics_reset", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut gpu = NovaGPU::new(GPUConfig::default()).await.unwrap();
                gpu.reset_metrics();
                black_box(gpu.metrics().clone())
            })
        })
    });
    
    group.finish();
}

fn benchmark_end_to_end_workflow(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("e2e_gpu_workflow");
    group.measurement_time(Duration::from_secs(15));
    
    // Simulate a realistic Nova proving workflow
    group.bench_function("typical_nova_msm_workflow", |b| {
        let scalar_points = generate_scalar_points(256);
        
        b.iter(|| {
            rt.block_on(async {
                // Initialize manager
                let mut manager = GPUAccelerationManager::new(false).await.unwrap();
                
                // Perform MSM (typical in commitment generation)
                let msm_result = manager.msm(black_box(&scalar_points)).await.unwrap();
                
                black_box(msm_result)
            })
        })
    });

    group.bench_function("typical_nova_ntt_workflow", |b| {
        let coefficients = generate_field_elements(1024);
        let domain = NTTDomain::new(1024).unwrap();
        
        b.iter(|| {
            rt.block_on(async {
                let mut manager = GPUAccelerationManager::new(false).await.unwrap();
                
                // Forward NTT
                let ntt_result = manager.ntt_forward(black_box(&coefficients), black_box(&domain)).await.unwrap();
                
                black_box(ntt_result)
            })
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_gpu_initialization,
    benchmark_msm_cpu_fallback,
    benchmark_ntt_cpu_fallback,
    benchmark_domain_creation,
    benchmark_scalar_point_operations,
    benchmark_field_element_operations,
    benchmark_commitment_cpu_fallback,
    benchmark_gpu_metrics,
    benchmark_end_to_end_workflow,
);

criterion_main!(benches);
