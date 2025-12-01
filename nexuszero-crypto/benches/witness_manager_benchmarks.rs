use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

use nexuszero_crypto::proof::{
    witness_manager::{WitnessManager, DefaultWitnessManager, WitnessGenerationConfig, ValidationConstraints, RandomnessConfig},
    statement::{StatementBuilder},
    witness::{WitnessType},
};

// Create a shared Tokio runtime for all benchmarks
lazy_static::lazy_static! {
    static ref RUNTIME: Runtime = Runtime::new().unwrap();
}

fn bench_witness_creation(c: &mut Criterion) {
    println!("Starting bench_witness_creation");
    let manager = DefaultWitnessManager::new(1000);
    let statement = StatementBuilder::new().discrete_log(vec![1, 2, 3], vec![4, 5, 6]).build().unwrap();
    let secret_data = b"test_secret_data";
    let config = WitnessGenerationConfig {
        parallel_generation: false,
        max_parallel_tasks: 4,
        zero_copy: true,
        randomness_config: RandomnessConfig {
            length: 32,
            secure_random: true,
            custom_entropy: None,
        },
        validation_constraints: ValidationConstraints {
            max_range_value: Some(1000),
            min_range_value: Some(0),
            max_preimage_length: Some(1024),
            required_hash_function: None,
            custom_constraints: HashMap::new(),
        },
    };

    println!("About to run bench_function");
    c.bench_function("witness_creation", |b| {
        println!("Inside bench_function closure");
        b.iter(|| {
            println!("Starting iteration");
            RUNTIME.block_on(async {
                let _result = manager.create_witness(&statement, secret_data, &config).await;
                println!("Witness creation completed");
            });
        });
    });
}

fn bench_cache_hit_performance(c: &mut Criterion) {
    let manager = DefaultWitnessManager::new(1000);
    let statement = StatementBuilder::new().discrete_log(vec![1, 2, 3], vec![4, 5, 6]).build().unwrap();
    let secret_data = b"test_secret_data";
    let config = WitnessGenerationConfig {
        parallel_generation: false,
        max_parallel_tasks: 4,
        zero_copy: true,
        randomness_config: RandomnessConfig {
            length: 32,
            secure_random: true,
            custom_entropy: None,
        },
        validation_constraints: ValidationConstraints {
            max_range_value: Some(1000),
            min_range_value: Some(0),
            max_preimage_length: Some(1024),
            required_hash_function: None,
            custom_constraints: HashMap::new(),
        },
    };

    // Pre-populate cache with a witness
    let _cache_id = RUNTIME.block_on(async {
        let witness = manager.create_witness(&statement, secret_data, &config).await.unwrap();
        manager.cache_witness(witness, Some(Duration::from_secs(300)), None).await.unwrap()
    });

    c.bench_function("cache_hit_performance", |b| {
        b.iter(|| {
            // Measure cache stats retrieval (simulating cache hit monitoring)
            let _stats = RUNTIME.block_on(async {
                manager.cache_stats().await.unwrap()
            });
        });
    });
    println!("bench_cache_hit_performance completed");
}

fn bench_validation_performance(c: &mut Criterion) {
    let manager = DefaultWitnessManager::new(1000);
    let statement = StatementBuilder::new().discrete_log(vec![1, 2, 3], vec![4, 5, 6]).build().unwrap();
    let secret_data = b"test_secret_data";
    let config = WitnessGenerationConfig {
        parallel_generation: false,
        max_parallel_tasks: 4,
        zero_copy: true,
        randomness_config: RandomnessConfig {
            length: 32,
            secure_random: true,
            custom_entropy: None,
        },
        validation_constraints: ValidationConstraints {
            max_range_value: Some(1000),
            min_range_value: Some(0),
            max_preimage_length: Some(1024),
            required_hash_function: None,
            custom_constraints: HashMap::new(),
        },
    };
    let constraints = ValidationConstraints {
        max_range_value: Some(1000),
        min_range_value: Some(0),
        max_preimage_length: Some(1024),
        required_hash_function: None,
        custom_constraints: HashMap::new(),
    };

    let witness = RUNTIME.block_on(async {
        manager.create_witness(&statement, secret_data, &config).await.unwrap()
    });

    c.bench_function("validation_performance", |b| {
        b.iter(|| {
            RUNTIME.block_on(async {
                let _result = manager.validate_witness(&witness, &statement, &constraints).await;
            });
        });
    });
}

fn bench_transformation_performance(c: &mut Criterion) {
    let manager = DefaultWitnessManager::new(1000);
    let statement = StatementBuilder::new().discrete_log(vec![1, 2, 3], vec![4, 5, 6]).build().unwrap();
    let secret_data = b"test_secret_data";
    let config = WitnessGenerationConfig {
        parallel_generation: false,
        max_parallel_tasks: 4,
        zero_copy: true,
        randomness_config: RandomnessConfig {
            length: 32,
            secure_random: true,
            custom_entropy: None,
        },
        validation_constraints: ValidationConstraints {
            max_range_value: Some(1000),
            min_range_value: Some(0),
            max_preimage_length: Some(1024),
            required_hash_function: None,
            custom_constraints: HashMap::new(),
        },
    };

    let witness = RUNTIME.block_on(async {
        manager.create_witness(&statement, secret_data, &config).await.unwrap()
    });

    let options = HashMap::new();

    c.bench_function("transformation_performance", |b| {
        b.iter(|| {
            RUNTIME.block_on(async {
                let _result = manager.transform_witness(witness.clone(), WitnessType::DiscreteLog, &options).await;
            });
        });
    });
}

fn bench_memory_usage_scaling(c: &mut Criterion) {
    let config = WitnessGenerationConfig {
        parallel_generation: false,
        max_parallel_tasks: 4,
        zero_copy: true,
        randomness_config: RandomnessConfig {
            length: 32,
            secure_random: true,
            custom_entropy: None,
        },
        validation_constraints: ValidationConstraints {
            max_range_value: Some(1000),
            min_range_value: Some(0),
            max_preimage_length: Some(1024),
            required_hash_function: None,
            custom_constraints: HashMap::new(),
        },
    };
    let secret_data = b"test_secret_data";

    let mut group = c.benchmark_group("memory_usage_scaling");

    for size in [100, 500, 1000, 2000].iter() {
        let manager = DefaultWitnessManager::new(*size);
        let statement = StatementBuilder::new().discrete_log(vec![1, 2, 3], vec![4, 5, 6]).build().unwrap();

        group.bench_with_input(format!("cache_size_{}", size), size, |b, _size| {
            b.iter(|| {
                RUNTIME.block_on(async {
                    let witness = manager.create_witness(&statement, secret_data, &config).await.unwrap();
                    manager.cache_witness(witness, Some(Duration::from_secs(300)), None).await.unwrap();
                });
            });
        });
    }
    group.finish();
}

fn bench_ttl_eviction(c: &mut Criterion) {
    let manager = DefaultWitnessManager::new(1000);
    let statement = StatementBuilder::new().discrete_log(vec![1, 2, 3], vec![4, 5, 6]).build().unwrap();
    let secret_data = b"test_secret_data";
    let config = WitnessGenerationConfig {
        parallel_generation: false,
        max_parallel_tasks: 4,
        zero_copy: true,
        randomness_config: RandomnessConfig {
            length: 32,
            secure_random: true,
            custom_entropy: None,
        },
        validation_constraints: ValidationConstraints {
            max_range_value: Some(1000),
            min_range_value: Some(0),
            max_preimage_length: Some(1024),
            required_hash_function: None,
            custom_constraints: HashMap::new(),
        },
    };

    c.bench_function("ttl_eviction", |b| {
        b.iter(|| {
            RUNTIME.block_on(async {
                let witness = manager.create_witness(&statement, secret_data, &config).await.unwrap();
                manager.cache_witness(witness, Some(Duration::from_millis(1)), None).await.unwrap();
                tokio::time::sleep(Duration::from_millis(10)).await;
                let _stats = manager.cache_stats().await;
            });
        });
    });
}

fn bench_zero_copy_operations(c: &mut Criterion) {
    let manager = DefaultWitnessManager::new(1000);
    let statement = StatementBuilder::new().discrete_log(vec![1, 2, 3], vec![4, 5, 6]).build().unwrap();
    let secret_data = b"test_secret_data";
    let config = WitnessGenerationConfig {
        parallel_generation: false,
        max_parallel_tasks: 4,
        zero_copy: true,
        randomness_config: RandomnessConfig {
            length: 32,
            secure_random: true,
            custom_entropy: None,
        },
        validation_constraints: ValidationConstraints {
            max_range_value: Some(1000),
            min_range_value: Some(0),
            max_preimage_length: Some(1024),
            required_hash_function: None,
            custom_constraints: HashMap::new(),
        },
    };

    let witness = RUNTIME.block_on(async {
        manager.create_witness(&statement, secret_data, &config).await.unwrap()
    });

    c.bench_function("zero_copy_operations", |b| {
        b.iter(|| {
            RUNTIME.block_on(async {
                let _arc_clone = Arc::clone(&witness);
            });
        });
    });
}

fn bench_concurrent_access(c: &mut Criterion) {
    let manager = Arc::new(DefaultWitnessManager::new(1000));
    let statement = StatementBuilder::new().discrete_log(vec![1, 2, 3], vec![4, 5, 6]).build().unwrap();
    let secret_data = b"test_secret_data";
    let config = WitnessGenerationConfig {
        parallel_generation: false,
        max_parallel_tasks: 4,
        zero_copy: true,
        randomness_config: RandomnessConfig {
            length: 32,
            secure_random: true,
            custom_entropy: None,
        },
        validation_constraints: ValidationConstraints {
            max_range_value: Some(1000),
            min_range_value: Some(0),
            max_preimage_length: Some(1024),
            required_hash_function: None,
            custom_constraints: HashMap::new(),
        },
    };

    c.bench_function("concurrent_access", |b| {
        b.iter(|| {
            RUNTIME.block_on(async {
                let tasks: Vec<_> = (0..10).map(|_| {
                    let manager_clone = Arc::clone(&manager);
                    let stmt = statement.clone();
                    let cfg = config.clone();
                    let data = secret_data;
                    tokio::spawn(async move {
                        let witness = manager_clone.create_witness(&stmt, data, &cfg).await.unwrap();
                        manager_clone.cache_witness(witness, Some(Duration::from_secs(300)), None).await.unwrap();
                    })
                }).collect();

                for task in tasks {
                    task.await.unwrap();
                }
            });
        });
    });
}

criterion_group!(
    benches,
    bench_witness_creation,
    bench_cache_hit_performance,
    bench_validation_performance,
    bench_transformation_performance,
    bench_memory_usage_scaling,
    bench_ttl_eviction,
    bench_zero_copy_operations,
    bench_concurrent_access
);
criterion_main!(benches);