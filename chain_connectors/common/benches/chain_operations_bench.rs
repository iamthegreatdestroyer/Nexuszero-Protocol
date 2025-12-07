//! Chain Connector Performance Benchmarks
//!
//! Comprehensive benchmarks for cross-chain operations, proof verification,
//! and bridge workflows.

use criterion::{
    black_box, criterion_group, criterion_main, Criterion, BenchmarkId,
    Throughput, SamplingMode,
};
use chain_connectors_common::prelude::*;
use std::time::{Duration, Instant};

// =============================================================================
// ChainId Operations Benchmarks
// =============================================================================

fn bench_chain_id_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("ChainId Operations");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(1000);
    
    let chains = vec![
        ChainId::Ethereum,
        ChainId::Polygon,
        ChainId::Bitcoin,
        ChainId::Solana,
        ChainId::Cosmos,
        ChainId::Avalanche,
        ChainId::Arbitrum,
        ChainId::Optimism,
    ];
    
    // Benchmark EVM detection
    group.bench_function("is_evm_check", |b| {
        let chains_clone = chains.clone();
        b.iter(|| {
            for chain in &chains_clone {
                let _ = black_box(chain.is_evm());
            }
        });
    });
    
    // Benchmark native symbol lookup
    group.bench_function("native_symbol_lookup", |b| {
        let chains_clone = chains.clone();
        b.iter(|| {
            for chain in &chains_clone {
                let _ = black_box(chain.native_symbol());
            }
        });
    });
    
    // Benchmark decimals lookup
    group.bench_function("decimals_lookup", |b| {
        let chains_clone = chains.clone();
        b.iter(|| {
            for chain in &chains_clone {
                let _ = black_box(chain.native_decimals());
            }
        });
    });
    
    // Benchmark custom chain creation
    group.bench_function("custom_chain_creation", |b| {
        b.iter(|| {
            let _ = black_box(ChainId::Custom(99999));
        });
    });
    
    group.finish();
}

// =============================================================================
// ProofMetadata Benchmarks
// =============================================================================

fn bench_proof_metadata_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("ProofMetadata Operations");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(500);
    
    // Benchmark metadata creation for different chains
    let chains = vec![
        ChainId::Ethereum,
        ChainId::Polygon,
        ChainId::Bitcoin,
        ChainId::Solana,
        ChainId::Cosmos,
    ];
    
    for chain in chains {
        group.bench_with_input(
            BenchmarkId::new("creation", format!("{:?}", chain)),
            &chain,
            |b, chain| {
                b.iter(|| {
                    let metadata = ProofMetadata {
                        chain_id: chain.clone(),
                        proof_type: "range".to_string(),
                        version: 1,
                        timestamp: chrono::Utc::now(),
                        nullifier: Some(vec![0u8; 32]),
                    };
                    black_box(metadata)
                });
            },
        );
    }
    
    // Benchmark serialization
    let metadata = ProofMetadata {
        chain_id: ChainId::Ethereum,
        proof_type: "transfer".to_string(),
        version: 1,
        timestamp: chrono::Utc::now(),
        nullifier: Some(vec![42u8; 32]),
    };
    
    group.bench_function("serialize_json", |b| {
        b.iter(|| {
            let serialized = serde_json::to_string(&metadata).unwrap();
            black_box(serialized)
        });
    });
    
    let serialized = serde_json::to_string(&metadata).unwrap();
    group.bench_function("deserialize_json", |b| {
        b.iter(|| {
            let deserialized: ProofMetadata = serde_json::from_str(&serialized).unwrap();
            black_box(deserialized)
        });
    });
    
    group.finish();
}

// =============================================================================
// BlockInfo Benchmarks
// =============================================================================

fn bench_block_info_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("BlockInfo Operations");
    group.sampling_mode(SamplingMode::Flat);
    
    // Benchmark block info creation with varying transaction counts
    let tx_counts = vec![0, 100, 1000, 5000, 10000];
    
    for tx_count in tx_counts {
        group.bench_with_input(
            BenchmarkId::new("creation", tx_count),
            &tx_count,
            |b, &count| {
                b.iter(|| {
                    let block = BlockInfo {
                        number: 18500000,
                        hash: "0xabc123".to_string(),
                        parent_hash: "0xdef456".to_string(),
                        timestamp: chrono::Utc::now(),
                        transaction_count: count,
                    };
                    black_box(block)
                });
            },
        );
    }
    
    // Benchmark serialization
    let block = BlockInfo {
        number: 18500000,
        hash: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef".to_string(),
        parent_hash: "0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321".to_string(),
        timestamp: chrono::Utc::now(),
        transaction_count: 500,
    };
    
    group.bench_function("serialize_json", |b| {
        b.iter(|| {
            let serialized = serde_json::to_string(&block).unwrap();
            black_box(serialized)
        });
    });
    
    group.finish();
}

// =============================================================================
// TransactionReceipt Benchmarks
// =============================================================================

fn bench_transaction_receipt_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("TransactionReceipt Operations");
    group.sampling_mode(SamplingMode::Flat);
    
    // Benchmark receipt creation
    group.bench_function("success_receipt_creation", |b| {
        b.iter(|| {
            let receipt = TransactionReceipt {
                tx_hash: "0xabc123def456".to_string(),
                block_number: 18500000,
                block_hash: "0xblock123".to_string(),
                status: TransactionStatus::Confirmed,
                gas_used: 21000,
                effective_gas_price: 30_000_000_000u64,
                logs: vec![],
            };
            black_box(receipt)
        });
    });
    
    // Benchmark receipt with logs
    let log_counts = vec![0, 5, 10, 25, 50];
    
    for log_count in log_counts {
        group.bench_with_input(
            BenchmarkId::new("receipt_with_logs", log_count),
            &log_count,
            |b, &count| {
                let logs: Vec<String> = (0..count).map(|i| format!("log_{}", i)).collect();
                b.iter(|| {
                    let receipt = TransactionReceipt {
                        tx_hash: "0xabc123def456".to_string(),
                        block_number: 18500000,
                        block_hash: "0xblock123".to_string(),
                        status: TransactionStatus::Confirmed,
                        gas_used: 200000,
                        effective_gas_price: 30_000_000_000u64,
                        logs: logs.clone(),
                    };
                    black_box(receipt)
                });
            },
        );
    }
    
    // Benchmark serialization
    let receipt = TransactionReceipt {
        tx_hash: "0xabc123def456".to_string(),
        block_number: 18500000,
        block_hash: "0xblock123".to_string(),
        status: TransactionStatus::Confirmed,
        gas_used: 21000,
        effective_gas_price: 30_000_000_000u64,
        logs: vec!["Transfer".to_string(), "Approval".to_string()],
    };
    
    group.bench_function("serialize_json", |b| {
        b.iter(|| {
            let serialized = serde_json::to_string(&receipt).unwrap();
            black_box(serialized)
        });
    });
    
    group.finish();
}

// =============================================================================
// ChainError Benchmarks
// =============================================================================

fn bench_chain_error_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("ChainError Operations");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(1000);
    
    // Create various error types
    let errors = vec![
        ChainError::connection_failed("Failed to connect to Ethereum RPC"),
        ChainError::timeout("Transaction confirmation timeout"),
        ChainError::rate_limited(Duration::from_secs(60)),
        ChainError::insufficient_funds(1000000, 500000),
        ChainError::invalid_address("Invalid address format"),
        ChainError::transaction_failed("Execution reverted"),
    ];
    
    // Benchmark is_retryable check
    group.bench_function("is_retryable_check", |b| {
        let errors_clone = errors.clone();
        b.iter(|| {
            for error in &errors_clone {
                let _ = black_box(error.is_retryable());
            }
        });
    });
    
    // Benchmark retry_delay calculation
    group.bench_function("retry_delay_calculation", |b| {
        let errors_clone = errors.clone();
        b.iter(|| {
            for error in &errors_clone {
                let _ = black_box(error.retry_delay());
            }
        });
    });
    
    // Benchmark error creation
    group.bench_function("error_creation_connection", |b| {
        b.iter(|| {
            let _ = black_box(ChainError::connection_failed("Connection refused"));
        });
    });
    
    group.bench_function("error_creation_insufficient_funds", |b| {
        b.iter(|| {
            let _ = black_box(ChainError::insufficient_funds(1000000, 500000));
        });
    });
    
    group.finish();
}

// =============================================================================
// FeeEstimate Benchmarks
// =============================================================================

fn bench_fee_estimate_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("FeeEstimate Operations");
    group.sampling_mode(SamplingMode::Flat);
    
    // Benchmark fee estimate creation
    group.bench_function("fee_estimate_creation", |b| {
        b.iter(|| {
            let estimate = FeeEstimate {
                base_fee: 25_000_000_000u64,
                priority_fee: 2_000_000_000u64,
                gas_limit: 21000,
                total_fee: 567_000_000_000_000u64,
                confidence: FeeConfidence::High,
            };
            black_box(estimate)
        });
    });
    
    // Benchmark confidence level operations
    let confidences = vec![
        FeeConfidence::Low,
        FeeConfidence::Medium,
        FeeConfidence::High,
    ];
    
    group.bench_function("confidence_comparison", |b| {
        let confs = confidences.clone();
        b.iter(|| {
            for conf in &confs {
                let _ = black_box(matches!(conf, FeeConfidence::High));
            }
        });
    });
    
    // Benchmark serialization
    let estimate = FeeEstimate {
        base_fee: 25_000_000_000u64,
        priority_fee: 2_000_000_000u64,
        gas_limit: 21000,
        total_fee: 567_000_000_000_000u64,
        confidence: FeeConfidence::High,
    };
    
    group.bench_function("serialize_json", |b| {
        b.iter(|| {
            let serialized = serde_json::to_string(&estimate).unwrap();
            black_box(serialized)
        });
    });
    
    group.finish();
}

// =============================================================================
// EventFilter Benchmarks
// =============================================================================

fn bench_event_filter_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("EventFilter Operations");
    group.sampling_mode(SamplingMode::Flat);
    
    // Benchmark filter creation
    group.bench_function("filter_creation", |b| {
        b.iter(|| {
            let filter = EventFilter {
                address: Some("0x1234567890abcdef".to_string()),
                event_type: Some(EventType::ProofSubmitted),
                from_block: Some(18000000),
                to_block: Some(18500000),
            };
            black_box(filter)
        });
    });
    
    // Benchmark proof shortcut filters
    group.bench_function("proof_submitted_filter", |b| {
        b.iter(|| {
            let filter = EventFilter::proof_submitted("0xcontract".to_string());
            black_box(filter)
        });
    });
    
    group.bench_function("proof_verified_filter", |b| {
        b.iter(|| {
            let filter = EventFilter::proof_verified("0xcontract".to_string());
            black_box(filter)
        });
    });
    
    // Benchmark event type signature generation
    let event_types = vec![
        EventType::ProofSubmitted,
        EventType::ProofVerified,
        EventType::Transfer,
        EventType::BridgeInitiated,
        EventType::BridgeCompleted,
    ];
    
    group.bench_function("event_signature_generation", |b| {
        let types = event_types.clone();
        b.iter(|| {
            for event_type in &types {
                let _ = black_box(event_type.evm_signature());
            }
        });
    });
    
    group.finish();
}

// =============================================================================
// ChainAddress Benchmarks
// =============================================================================

fn bench_chain_address_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("ChainAddress Operations");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(500);
    
    // Benchmark address creation for different chains
    let chains = vec![
        (ChainId::Ethereum, vec![0xabu8; 20]),
        (ChainId::Bitcoin, vec![0x00u8; 25]),
        (ChainId::Solana, vec![0xffu8; 32]),
        (ChainId::Cosmos, vec![0x42u8; 20]),
    ];
    
    for (chain, bytes) in chains.clone() {
        group.bench_with_input(
            BenchmarkId::new("creation", format!("{:?}", chain)),
            &(chain, bytes),
            |b, (chain, bytes)| {
                b.iter(|| {
                    let addr = ChainAddress::new(chain.clone(), bytes.clone());
                    black_box(addr)
                });
            },
        );
    }
    
    // Benchmark to_hex conversion
    for (chain, bytes) in chains.clone() {
        let addr = ChainAddress::new(chain.clone(), bytes);
        group.bench_with_input(
            BenchmarkId::new("to_hex", format!("{:?}", chain)),
            &addr,
            |b, addr| {
                b.iter(|| {
                    let hex = addr.to_hex();
                    black_box(hex)
                });
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// Bridge Workflow Benchmarks
// =============================================================================

fn bench_bridge_workflows(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bridge Workflows");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(200);
    
    // Benchmark bridge route validation (checking supported chains)
    let supported_routes = vec![
        (ChainId::Ethereum, ChainId::Polygon),
        (ChainId::Ethereum, ChainId::Arbitrum),
        (ChainId::Ethereum, ChainId::Optimism),
        (ChainId::Polygon, ChainId::Ethereum),
        (ChainId::Bitcoin, ChainId::Ethereum),
        (ChainId::Solana, ChainId::Ethereum),
    ];
    
    group.bench_function("route_validation", |b| {
        let routes = supported_routes.clone();
        b.iter(|| {
            for (source, dest) in &routes {
                // Simulate route validation
                let source_evm = source.is_evm();
                let dest_evm = dest.is_evm();
                let compatible = source_evm || dest_evm;
                black_box(compatible)
            }
        });
    });
    
    // Benchmark proof metadata creation for bridge
    group.bench_function("bridge_proof_metadata", |b| {
        b.iter(|| {
            let source_meta = ProofMetadata {
                chain_id: ChainId::Ethereum,
                proof_type: "bridge_lock".to_string(),
                version: 1,
                timestamp: chrono::Utc::now(),
                nullifier: Some(vec![0x42u8; 32]),
            };
            let dest_meta = ProofMetadata {
                chain_id: ChainId::Polygon,
                proof_type: "bridge_mint".to_string(),
                version: 1,
                timestamp: chrono::Utc::now(),
                nullifier: Some(vec![0x42u8; 32]),
            };
            black_box((source_meta, dest_meta))
        });
    });
    
    // Benchmark complete bridge initiation workflow
    group.bench_function("bridge_initiation_workflow", |b| {
        b.iter(|| {
            // 1. Create source chain operation
            let source_op = ChainOperation::BridgeOut {
                destination_chain: ChainId::Polygon,
                recipient: "0xrecipient".to_string(),
                amount: 1000000000000000000u128,
            };
            
            // 2. Create proof metadata
            let metadata = ProofMetadata {
                chain_id: ChainId::Ethereum,
                proof_type: "bridge".to_string(),
                version: 1,
                timestamp: chrono::Utc::now(),
                nullifier: Some(vec![0x00u8; 32]),
            };
            
            // 3. Create fee estimate
            let fee = FeeEstimate {
                base_fee: 30_000_000_000u64,
                priority_fee: 2_000_000_000u64,
                gas_limit: 150000,
                total_fee: 4_800_000_000_000_000u64,
                confidence: FeeConfidence::High,
            };
            
            black_box((source_op, metadata, fee))
        });
    });
    
    group.finish();
}

// =============================================================================
// Multi-Chain Batch Operations Benchmarks
// =============================================================================

fn bench_multi_chain_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multi-Chain Batch");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    
    let batch_sizes = vec![10, 50, 100, 500, 1000];
    
    for batch_size in batch_sizes {
        // Benchmark batch proof metadata creation
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("metadata_batch", batch_size),
            &batch_size,
            |b, &size| {
                let chains = vec![
                    ChainId::Ethereum,
                    ChainId::Polygon,
                    ChainId::Arbitrum,
                    ChainId::Optimism,
                    ChainId::Bitcoin,
                ];
                b.iter(|| {
                    let metadata: Vec<_> = (0..size).map(|i| {
                        ProofMetadata {
                            chain_id: chains[i % chains.len()].clone(),
                            proof_type: "transfer".to_string(),
                            version: 1,
                            timestamp: chrono::Utc::now(),
                            nullifier: Some(vec![(i % 256) as u8; 32]),
                        }
                    }).collect();
                    black_box(metadata)
                });
            },
        );
        
        // Benchmark batch serialization
        group.bench_with_input(
            BenchmarkId::new("serialize_batch", batch_size),
            &batch_size,
            |b, &size| {
                let chains = vec![
                    ChainId::Ethereum,
                    ChainId::Polygon,
                    ChainId::Bitcoin,
                ];
                let metadata: Vec<_> = (0..size).map(|i| {
                    ProofMetadata {
                        chain_id: chains[i % chains.len()].clone(),
                        proof_type: "transfer".to_string(),
                        version: 1,
                        timestamp: chrono::Utc::now(),
                        nullifier: Some(vec![(i % 256) as u8; 32]),
                    }
                }).collect();
                
                b.iter(|| {
                    let serialized = serde_json::to_string(&metadata).unwrap();
                    black_box(serialized)
                });
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// Cross-Chain Proof Verification Simulation
// =============================================================================

fn bench_cross_chain_proof_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cross-Chain Proof Simulation");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(5));
    
    // Simulate complete cross-chain verification workflow
    group.bench_function("full_verification_workflow", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            
            for _ in 0..iters {
                // 1. Parse source chain proof
                let source_metadata = ProofMetadata {
                    chain_id: ChainId::Ethereum,
                    proof_type: "zk_transfer".to_string(),
                    version: 1,
                    timestamp: chrono::Utc::now(),
                    nullifier: Some(vec![0xabu8; 32]),
                };
                
                // 2. Validate chain compatibility
                let source_evm = source_metadata.chain_id.is_evm();
                
                // 3. Create relay proof for destination
                let relay_metadata = ProofMetadata {
                    chain_id: ChainId::Polygon,
                    proof_type: "relay".to_string(),
                    version: 1,
                    timestamp: chrono::Utc::now(),
                    nullifier: source_metadata.nullifier.clone(),
                };
                
                // 4. Create receipt simulation
                let receipt = TransactionReceipt {
                    tx_hash: "0xsimulated".to_string(),
                    block_number: 50000000,
                    block_hash: "0xblock".to_string(),
                    status: TransactionStatus::Confirmed,
                    gas_used: 150000,
                    effective_gas_price: 50_000_000_000u64,
                    logs: vec!["ProofVerified".to_string()],
                };
                
                black_box((source_metadata, relay_metadata, receipt, source_evm))
            }
            
            start.elapsed()
        });
    });
    
    // Benchmark multi-hop verification (ETH -> L2 -> L2)
    group.bench_function("multi_hop_verification", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            
            for _ in 0..iters {
                let hops = vec![
                    (ChainId::Ethereum, ChainId::Arbitrum),
                    (ChainId::Arbitrum, ChainId::Optimism),
                    (ChainId::Optimism, ChainId::Polygon),
                ];
                
                let mut current_nullifier = vec![0x00u8; 32];
                
                for (source, dest) in &hops {
                    let metadata = ProofMetadata {
                        chain_id: source.clone(),
                        proof_type: "relay".to_string(),
                        version: 1,
                        timestamp: chrono::Utc::now(),
                        nullifier: Some(current_nullifier.clone()),
                    };
                    
                    // Update nullifier for next hop
                    current_nullifier[0] = current_nullifier[0].wrapping_add(1);
                    
                    black_box((metadata, dest));
                }
            }
            
            start.elapsed()
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_chain_id_operations,
    bench_proof_metadata_operations,
    bench_block_info_operations,
    bench_transaction_receipt_operations,
    bench_chain_error_operations,
    bench_fee_estimate_operations,
    bench_event_filter_operations,
    bench_chain_address_operations,
    bench_bridge_workflows,
    bench_multi_chain_batch_operations,
    bench_cross_chain_proof_simulation,
);

criterion_main!(benches);
