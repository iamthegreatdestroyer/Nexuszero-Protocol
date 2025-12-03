// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Nova Proving System Benchmarks
// Comprehensive benchmarks for the Nova IVC proving system

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;

#[cfg(feature = "nova")]
mod nova_benchmarks {
    use super::*;
    use nexuszero_crypto::proof::nova::{
        NovaConfig, NovaProver, CompressionLevel,
        FoldingEngine, FoldingConfig, FoldedInstance,
        IVCChain, RecursiveConfig, RecursiveProver,
        TrivialCircuit, HashChainCircuit, MinRootCircuit, HashType,
        R1CSConverter, R1CSConstraintSystem,
        NovaSecurityLevel,
    };

    pub fn benchmark_folding_engine(c: &mut Criterion) {
        let mut group = c.benchmark_group("nova_folding_engine");
        group.measurement_time(Duration::from_secs(5));

        // Benchmark folding initialization with different configs
        group.bench_function("init_default", |b| {
            b.iter(|| {
                black_box(FoldingEngine::new(16, true))
            })
        });

        group.bench_function("init_serial", |b| {
            b.iter(|| {
                black_box(FoldingEngine::new(16, false))
            })
        });

        group.bench_function("init_with_config", |b| {
            b.iter(|| {
                let config = FoldingConfig::default();
                black_box(FoldingEngine::with_config(config))
            })
        });

        // Benchmark different batch sizes
        for batch_size in [8, 16, 32, 64].iter() {
            group.bench_with_input(
                BenchmarkId::new("init_batch_size", batch_size),
                batch_size,
                |b, &batch_size| {
                    b.iter(|| {
                        black_box(FoldingEngine::new(batch_size, true))
                    })
                },
            );
        }

        group.finish();
    }

    pub fn benchmark_nova_prover(c: &mut Criterion) {
        let mut group = c.benchmark_group("nova_prover");
        group.measurement_time(Duration::from_secs(5));

        // Benchmark prover initialization
        group.bench_function("init_default_config", |b| {
            b.iter(|| {
                let config = NovaConfig::default();
                black_box(NovaProver::new(config))
            })
        });

        // Benchmark different compression levels
        for level in [CompressionLevel::None, CompressionLevel::Standard, CompressionLevel::Maximum].iter() {
            let config = NovaConfig {
                compression_level: level.clone(),
                ..Default::default()
            };

            group.bench_with_input(
                BenchmarkId::new("init_compression", format!("{:?}", level)),
                &config,
                |b, config| {
                    b.iter(|| {
                        black_box(NovaProver::new(config.clone()))
                    })
                },
            );
        }

        // Benchmark different security levels
        for security in [NovaSecurityLevel::Bit128, NovaSecurityLevel::Bit192, NovaSecurityLevel::Bit256].iter() {
            let config = NovaConfig {
                security_level: security.clone(),
                ..Default::default()
            };

            group.bench_with_input(
                BenchmarkId::new("init_security", format!("{:?}", security)),
                &config,
                |b, config| {
                    b.iter(|| {
                        black_box(NovaProver::new(config.clone()))
                    })
                },
            );
        }

        group.finish();
    }

    pub fn benchmark_ivc_chain(c: &mut Criterion) {
        let mut group = c.benchmark_group("nova_ivc_chain");
        group.measurement_time(Duration::from_secs(5));

        // Benchmark IVC chain creation
        group.bench_function("create_chain", |b| {
            b.iter(|| {
                let config = RecursiveConfig::default();
                black_box(IVCChain::new(config))
            })
        });

        // Benchmark IVC chain with different max depths
        for max_depth in [10, 50, 100, 500].iter() {
            let config = RecursiveConfig {
                max_depth: *max_depth,
                ..Default::default()
            };
            
            group.bench_with_input(
                BenchmarkId::new("create_max_depth", max_depth),
                &config,
                |b, config| {
                    b.iter(|| {
                        black_box(IVCChain::new(config.clone()))
                    })
                },
            );
        }

        // Benchmark adding steps to IVC chain
        for num_steps in [5, 10, 20].iter() {
            group.throughput(Throughput::Elements(*num_steps as u64));
            group.bench_with_input(
                BenchmarkId::new("add_steps", num_steps),
                num_steps,
                |b, &num| {
                    b.iter(|| {
                        let config = RecursiveConfig::default();
                        let mut chain = IVCChain::new(config);
                        let circuit = TrivialCircuit::new(2);
                        for i in 0..num {
                            let input = vec![vec![i as u8; 32], vec![(i + 1) as u8; 32]];
                            let _ = chain.add_step(&circuit, input);
                        }
                        black_box(chain)
                    })
                },
            );
        }

        group.finish();
    }

    pub fn benchmark_recursive_prover(c: &mut Criterion) {
        let mut group = c.benchmark_group("nova_recursive_prover");
        group.measurement_time(Duration::from_secs(5));

        // Benchmark recursive prover initialization
        group.bench_function("init_default", |b| {
            b.iter(|| {
                black_box(RecursiveProver::new())
            })
        });

        group.bench_function("init_with_config", |b| {
            b.iter(|| {
                let config = RecursiveConfig {
                    max_depth: 100,
                    parallel: true,
                    ..Default::default()
                };
                black_box(RecursiveProver::with_config(config))
            })
        });

        // Benchmark creating chains from prover
        group.bench_function("create_chain_from_prover", |b| {
            b.iter(|| {
                let prover = RecursiveProver::new();
                black_box(prover.create_chain())
            })
        });

        group.finish();
    }

    pub fn benchmark_circuits(c: &mut Criterion) {
        let mut group = c.benchmark_group("nova_circuits");

        // Benchmark TrivialCircuit creation with different arities
        for arity in [2, 4, 8, 16].iter() {
            group.bench_with_input(
                BenchmarkId::new("trivial_circuit", arity),
                arity,
                |b, &arity| {
                    b.iter(|| {
                        black_box(TrivialCircuit::new(arity))
                    })
                },
            );
        }

        // Benchmark MinRootCircuit with different iteration counts
        for iterations in [5, 10, 20, 50].iter() {
            group.bench_with_input(
                BenchmarkId::new("minroot_circuit", iterations),
                iterations,
                |b, &iterations| {
                    b.iter(|| {
                        black_box(MinRootCircuit::new(iterations))
                    })
                },
            );
        }

        // Benchmark HashChainCircuit with different hash types
        for hash_type in [HashType::Poseidon, HashType::MiMC, HashType::Rescue, HashType::SHA256].iter() {
            group.bench_with_input(
                BenchmarkId::new("hashchain_circuit", format!("{:?}", hash_type)),
                hash_type,
                |b, hash_type| {
                    b.iter(|| {
                        black_box(HashChainCircuit::new(*hash_type))
                    })
                },
            );
        }

        // Benchmark HashChainCircuit::poseidon() convenience
        group.bench_function("hashchain_poseidon", |b| {
            b.iter(|| {
                black_box(HashChainCircuit::poseidon())
            })
        });

        group.finish();
    }

    pub fn benchmark_r1cs_conversion(c: &mut Criterion) {
        let mut group = c.benchmark_group("nova_r1cs");
        group.measurement_time(Duration::from_secs(5));

        // Benchmark R1CS constraint system creation at different security levels
        for security in [NovaSecurityLevel::Bit128, NovaSecurityLevel::Bit192, NovaSecurityLevel::Bit256].iter() {
            group.bench_with_input(
                BenchmarkId::new("create_constraint_system", format!("{:?}", security)),
                security,
                |b, security| {
                    b.iter(|| {
                        black_box(R1CSConstraintSystem::new(security.clone()))
                    })
                },
            );
        }

        // Benchmark R1CS converter at different security levels
        for security in [NovaSecurityLevel::Bit128, NovaSecurityLevel::Bit192, NovaSecurityLevel::Bit256].iter() {
            group.bench_with_input(
                BenchmarkId::new("create_converter", format!("{:?}", security)),
                security,
                |b, security| {
                    b.iter(|| {
                        black_box(R1CSConverter::new(security.clone()))
                    })
                },
            );
        }

        group.finish();
    }

    pub fn benchmark_end_to_end_workflow(c: &mut Criterion) {
        let mut group = c.benchmark_group("nova_e2e_workflow");
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(20);

        // Simple workflow: Create chain, add steps, finalize
        group.bench_function("simple_ivc_workflow", |b| {
            b.iter(|| {
                // 1. Create config
                let config = RecursiveConfig::default();
                
                // 2. Create IVC chain
                let mut chain = IVCChain::new(config);
                
                // 3. Create circuit
                let circuit = TrivialCircuit::new(2);
                
                // 4. Add 5 steps
                for i in 0..5u8 {
                    let input = vec![vec![i; 32], vec![i + 1; 32]];
                    let _ = chain.add_step(&circuit, input);
                }
                
                // 5. Finalize
                black_box(chain.finalize())
            })
        });

        // Full prover workflow
        group.bench_function("full_prover_workflow", |b| {
            b.iter(|| {
                // 1. Create prover config
                let config = NovaConfig::default();
                let prover = NovaProver::new(config);
                
                // 2. Create recursive prover
                let recursive_prover = RecursiveProver::new();
                
                black_box((prover, recursive_prover))
            })
        });

        // Combined workflow with different compression levels
        for (label, compression) in [
            ("fast", CompressionLevel::None),
            ("standard", CompressionLevel::Standard),
            ("max_compress", CompressionLevel::Maximum),
        ].iter() {
            group.bench_with_input(
                BenchmarkId::new("workflow", *label),
                compression,
                |b, compression| {
                    b.iter(|| {
                        let config = NovaConfig {
                            compression_level: compression.clone(),
                            ..Default::default()
                        };
                        let prover = NovaProver::new(config);
                        
                        let recursive_config = RecursiveConfig::default();
                        let mut chain = IVCChain::new(recursive_config);
                        let circuit = TrivialCircuit::new(2);
                        
                        for i in 0..3u8 {
                            let input = vec![vec![i; 32], vec![i + 1; 32]];
                            let _ = chain.add_step(&circuit, input);
                        }
                        
                        black_box((prover, chain.finalize()))
                    })
                },
            );
        }

        // Workflow with different circuit types
        group.bench_function("workflow_trivial_circuit", |b| {
            b.iter(|| {
                let config = RecursiveConfig::default();
                let mut chain = IVCChain::new(config);
                let circuit = TrivialCircuit::new(4);
                
                for i in 0..5u8 {
                    let input = vec![vec![i; 32]; 4];
                    let _ = chain.add_step(&circuit, input);
                }
                
                black_box(chain.finalize())
            })
        });

        group.bench_function("workflow_minroot_circuit", |b| {
            b.iter(|| {
                let config = RecursiveConfig::default();
                let mut chain = IVCChain::new(config);
                let circuit = MinRootCircuit::new(5);
                
                for i in 0..3u8 {
                    let input = vec![vec![i; 32]; circuit.arity()];
                    let _ = chain.add_step(&circuit, input);
                }
                
                black_box(chain.finalize())
            })
        });

        group.finish();
    }

    // Aggregate all benchmark groups
    criterion_group!(
        nova_benches,
        benchmark_folding_engine,
        benchmark_nova_prover,
        benchmark_ivc_chain,
        benchmark_recursive_prover,
        benchmark_circuits,
        benchmark_r1cs_conversion,
        benchmark_end_to_end_workflow,
    );
}

#[cfg(feature = "nova")]
criterion_main!(nova_benchmarks::nova_benches);

#[cfg(not(feature = "nova"))]
fn main() {
    eprintln!("Nova benchmarks require the 'nova' feature flag. Run with:");
    eprintln!("  cargo bench --features nova --bench nova_proving_benchmarks");
}
