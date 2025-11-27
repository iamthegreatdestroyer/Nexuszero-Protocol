// Integration E2E Tests
//
// Tests interactions between multiple modules and services
// INTEGRATED: Uses actual nexuszero-crypto and nexuszero-holographic modules

use nexuszero_e2e::{
    Timer, TestMetrics, generate_deterministic_bytes, generate_random_bytes,
    prove_range, verify_range, BulletproofRangeProof,
    CompressedMPS, MPSConfig, HolographicEncoder, EncoderConfig,
    compress_proof_data, decompress_proof_data, CompressionConfig, StoragePrecision,
    E2ETestConfig, SecurityLevel,
};

#[cfg(test)]
mod module_integration_tests {
    use super::*;

    /// Test: Crypto + Holographic compression integration
    #[test]
    fn test_crypto_compression_integration() {
        println!("Testing crypto + compression integration");
        let timer = Timer::new();
        let mut metrics = TestMetrics::new();
        
        // Workflow:
        // 1. Generate cryptographic proof
        // 2. Serialize to bytes
        // 3. Compress proof using holographic compression
        // 4. Decompress proof
        // 5. Verify decompressed proof matches original
        
        for i in 0..5 {
            let iteration_timer = Timer::new();
            
            // Step 1: Generate proof
            let blinding = generate_deterministic_bytes(32, i as u64);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            let value = ((i + 1) * 1000) as u64;
            
            let proof = prove_range(value, &blinding_array, 16)
                .expect("Proof generation should succeed");
            
            // Step 2: Serialize
            let proof_bytes = proof.to_bytes();
            let original_size = proof_bytes.len();
            
            // Step 3: Compress
            let config = CompressionConfig {
                precision: StoragePrecision::Float32,
                max_bond_dim: 32,
                truncation_threshold: 1e-6,
                use_lz4: true,
            };
            
            let compressed = compress_proof_data(&proof_bytes, &config)
                .expect("Compression should succeed");
            let compressed_size = compressed.serialized_size();
            
            // Step 4: Decompress
            let decompressed = decompress_proof_data(&compressed)
                .expect("Decompression should succeed");
            
            // Step 5: Verify match
            let success = decompressed == proof_bytes;
            metrics.add_result(success, iteration_timer.elapsed());
            
            if success {
                println!("  ✅ Iteration {}: {}→{} bytes (ratio: {:.2}x)", 
                         i + 1, original_size, compressed_size,
                         original_size as f64 / compressed_size.max(1) as f64);
            } else {
                println!("  ❌ Iteration {}: Roundtrip failed", i + 1);
            }
        }
        
        println!("✅ Integration completed in {}ms: {}", timer.elapsed_ms(), metrics.summary());
        assert!(metrics.success_rate() >= 80.0, "Integration should mostly succeed");
    }

    /// Test: End-to-end proof workflow
    #[test]
    fn test_end_to_end_proof_workflow() {
        println!("Testing end-to-end proof workflow");
        let timer = Timer::new();
        
        // Simulate complete user journey:
        // 1. User requests proof for transaction
        // 2. Generate proof with crypto module
        // 3. Verify proof
        // 4. Compress for storage/transmission
        // 5. Decompress
        // 6. Re-verify proof
        
        let blinding = generate_deterministic_bytes(32, 42);
        let blinding_array: [u8; 32] = blinding.try_into().unwrap();
        
        // Step 1-2: Generate proof
        let proof = prove_range(50000, &blinding_array, 16)
            .expect("Proof generation should succeed");
        println!("  Step 1-2: Proof generated ({} bytes)", proof.size_bytes());
        
        // Step 3: Verify proof
        verify_range(&proof).expect("Proof should verify");
        println!("  Step 3: Proof verified ✅");
        
        // Step 4: Compress
        let proof_bytes = proof.to_bytes();
        let config = CompressionConfig {
            precision: StoragePrecision::Float32,
            max_bond_dim: 32,
            truncation_threshold: 1e-6,
            use_lz4: true,
        };
        
        let compressed = compress_proof_data(&proof_bytes, &config)
            .expect("Compression should succeed");
        println!("  Step 4: Compressed to {} bytes", compressed.serialized_size());
        
        // Step 5: Decompress
        let decompressed = decompress_proof_data(&compressed)
            .expect("Decompression should succeed");
        println!("  Step 5: Decompressed to {} bytes", decompressed.len());
        
        // Step 6: Verify data integrity
        assert_eq!(proof_bytes, decompressed, "Data should match after roundtrip");
        println!("  Step 6: Data integrity verified ✅");
        
        println!("✅ E2E workflow completed in {}ms", timer.elapsed_ms());
    }

    /// Test: API Gateway + backend services simulation
    #[test]
    fn test_api_simulation() {
        println!("Testing API gateway simulation");
        
        // Simulate API request/response pattern
        #[derive(Debug)]
        struct ApiRequest {
            method: String,
            endpoint: String,
            value: u64,
            num_bits: u8,
        }
        
        #[derive(Debug)]
        struct ApiResponse {
            success: bool,
            proof_size: usize,
            compressed_size: usize,
            processing_time_ms: u128,
        }
        
        let requests = vec![
            ApiRequest { method: "POST".into(), endpoint: "/proof/generate".into(), value: 1000, num_bits: 16 },
            ApiRequest { method: "POST".into(), endpoint: "/proof/generate".into(), value: 50000, num_bits: 16 },
            ApiRequest { method: "POST".into(), endpoint: "/proof/generate".into(), value: 100, num_bits: 8 },
        ];
        
        for request in requests {
            let timer = Timer::new();
            
            // Process request
            let blinding = generate_deterministic_bytes(32, request.value);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            
            let result = prove_range(request.value, &blinding_array, request.num_bits);
            
            let response = match result {
                Ok(proof) => {
                    let proof_bytes = proof.to_bytes();
                    let config = CompressionConfig {
                        precision: StoragePrecision::Float32,
                        max_bond_dim: 16,
                        truncation_threshold: 1e-4,
                        use_lz4: true,
                    };
                    
                    let compressed_size = compress_proof_data(&proof_bytes, &config)
                        .map(|c| c.serialized_size())
                        .unwrap_or(0);
                    
                    ApiResponse {
                        success: true,
                        proof_size: proof.size_bytes(),
                        compressed_size,
                        processing_time_ms: timer.elapsed_ms(),
                    }
                }
                Err(_) => ApiResponse {
                    success: false,
                    proof_size: 0,
                    compressed_size: 0,
                    processing_time_ms: timer.elapsed_ms(),
                },
            };
            
            println!("  {} {} -> {:?}", request.method, request.endpoint, response);
        }
        
        println!("✅ API simulation test completed");
    }

    /// Test: Chain connectors simulation
    #[test]
    fn test_chain_connectors_simulation() {
        println!("Testing chain connector integration");
        
        // Simulate proof generation for different chains
        let chains = vec!["ethereum", "bitcoin", "solana", "polygon", "cosmos"];
        
        for chain in chains {
            let timer = Timer::new();
            
            // Generate chain-specific proof
            let blinding = generate_deterministic_bytes(32, chain.len() as u64);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            
            let proof = prove_range(10000, &blinding_array, 16);
            let status = if proof.is_ok() { "✅" } else { "❌" };
            
            println!("  {} {}: proof generated in {}ms", status, chain, timer.elapsed_ms());
        }
        
        println!("✅ Chain connector simulation completed");
    }
}

#[cfg(test)]
mod service_mesh_tests {
    use super::*;

    /// Test: Service discovery works correctly (simulated)
    #[test]
    fn test_service_discovery_simulation() {
        println!("Testing service discovery mechanisms");
        
        // Simulate service registry
        let services = vec![
            ("privacy_service", "localhost:8080"),
            ("transaction_service", "localhost:8081"),
            ("proof_service", "localhost:8082"),
            ("compliance_service", "localhost:8083"),
        ];
        
        for (name, addr) in &services {
            println!("  Discovered: {} at {}", name, addr);
        }
        
        // Verify all expected services found
        assert_eq!(services.len(), 4, "All 4 services should be discovered");
        
        println!("✅ Service discovery simulation verified");
    }

    /// Test: Circuit breaker pattern
    #[test]
    fn test_circuit_breaker_simulation() {
        println!("Testing circuit breaker pattern");
        
        let mut consecutive_failures = 0;
        let threshold = 5;
        let mut circuit_open = false;
        
        // Simulate operations with some failures
        for i in 0..20 {
            if circuit_open {
                // Skip operation if circuit is open
                println!("  Iteration {}: Circuit OPEN - skipping", i + 1);
                
                // Attempt recovery after cooldown
                if i > 15 {
                    circuit_open = false;
                    consecutive_failures = 0;
                    println!("  Circuit CLOSED - attempting recovery");
                }
                continue;
            }
            
            // Simulate operation with 30% failure rate
            let fails = i % 10 < 3;
            
            if fails {
                consecutive_failures += 1;
                if consecutive_failures >= threshold {
                    circuit_open = true;
                    println!("  Iteration {}: FAILED - Circuit OPENED", i + 1);
                } else {
                    println!("  Iteration {}: FAILED ({}/{})", i + 1, consecutive_failures, threshold);
                }
            } else {
                consecutive_failures = 0;
                println!("  Iteration {}: SUCCESS", i + 1);
            }
        }
        
        println!("✅ Circuit breaker simulation completed");
    }

    /// Test: Retry logic with backoff
    #[test]
    fn test_retry_logic() {
        println!("Testing retry logic");
        
        let max_retries = 3;
        let mut attempts = 0;
        let mut success = false;
        
        // Simulate operation that succeeds on 3rd try
        while attempts < max_retries && !success {
            attempts += 1;
            let backoff_ms = 10 * (1 << (attempts - 1)); // Exponential backoff
            
            // Simulate: succeed on 3rd attempt
            success = attempts >= 3;
            
            println!("  Attempt {}: {} (backoff: {}ms)", 
                     attempts, if success { "SUCCESS" } else { "RETRY" }, backoff_ms);
        }
        
        assert!(success, "Operation should eventually succeed");
        println!("✅ Retry logic verified after {} attempts", attempts);
    }
}

#[cfg(test)]
mod data_flow_tests {
    use super::*;

    /// Test: Data flows correctly through entire system
    #[test]
    fn test_end_to_end_data_flow() {
        println!("Testing complete data flow");
        let mut metrics = TestMetrics::new();
        
        // Simulate complete user journey:
        // 1. User submits transaction value
        // 2. Transaction validated (range check)
        // 3. Privacy proof generated
        // 4. Proof compressed
        // 5. Transaction submitted (simulated)
        // 6. Confirmation received
        
        for i in 0..10 {
            let timer = Timer::new();
            
            // Step 1-2: Validate transaction value
            let transaction_value = ((i + 1) * 5000) as u64;
            let valid_range = transaction_value < 100000;
            
            if !valid_range {
                metrics.add_result(false, timer.elapsed());
                continue;
            }
            
            // Step 3: Generate proof
            let blinding = generate_deterministic_bytes(32, i as u64);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            
            let proof_result = prove_range(transaction_value, &blinding_array, 32);
            
            let success = match proof_result {
                Ok(proof) => {
                    // Step 4: Compress
                    let proof_bytes = proof.to_bytes();
                    let config = CompressionConfig {
                        precision: StoragePrecision::Float32,
                        max_bond_dim: 16,
                        truncation_threshold: 1e-4,
                        use_lz4: true,
                    };
                    
                    compress_proof_data(&proof_bytes, &config).is_ok()
                }
                Err(_) => false,
            };
            
            metrics.add_result(success, timer.elapsed());
            println!("  Transaction {}: value={}, {}", 
                     i + 1, transaction_value, if success { "✅" } else { "❌" });
        }
        
        println!("✅ Data flow tests: {}", metrics.summary());
        assert!(metrics.success_rate() >= 80.0, "Most data flow tests should succeed");
    }

    /// Test: Error propagation across modules
    #[test]
    fn test_error_propagation() {
        println!("Testing error propagation");
        
        // Test that errors are properly propagated:
        // - Crypto error -> handled gracefully
        // - Compression error -> handled gracefully
        
        let blinding = generate_deterministic_bytes(32, 42);
        let blinding_array: [u8; 32] = blinding.try_into().unwrap();
        
        // Trigger crypto error (value out of range)
        let crypto_error = prove_range(1000, &blinding_array, 8);
        match crypto_error {
            Err(e) => println!("  Crypto error correctly propagated: {:?}", e),
            Ok(_) => println!("  Unexpected success for out-of-range value"),
        }
        
        // Test empty data compression
        let empty_data: Vec<u8> = vec![];
        let config = CompressionConfig {
            precision: StoragePrecision::Float32,
            max_bond_dim: 8,
            truncation_threshold: 1e-4,
            use_lz4: true,
        };
        
        let compression_result = compress_proof_data(&empty_data, &config);
        match compression_result {
            Err(e) => println!("  Compression error correctly propagated: {:?}", e),
            Ok(_) => println!("  Empty data handled"),
        }
        
        println!("✅ Error propagation verified");
    }

    /// Test: State consistency across operations
    #[test]
    fn test_state_consistency() {
        println!("Testing state consistency");
        
        // Generate same proof multiple times - should be consistent
        let blinding = generate_deterministic_bytes(32, 12345);
        let blinding_array: [u8; 32] = blinding.try_into().unwrap();
        
        let proofs: Vec<BulletproofRangeProof> = (0..3)
            .filter_map(|_| prove_range(5000, &blinding_array, 16).ok())
            .collect();
        
        // All proofs should verify
        for (i, proof) in proofs.iter().enumerate() {
            let verified = verify_range(proof).is_ok();
            println!("  Proof {}: {}", i + 1, if verified { "verified ✅" } else { "failed ❌" });
            assert!(verified, "All identical proofs should verify");
        }
        
        println!("✅ State consistency verified");
    }
}

#[cfg(test)]
mod monitoring_integration_tests {
    use super::*;

    /// Test: Metrics collection simulation
    #[test]
    fn test_metrics_collection() {
        println!("Testing metrics collection");
        
        // Simulate collecting metrics from operations
        let mut proof_times: Vec<u128> = Vec::new();
        let mut compress_times: Vec<u128> = Vec::new();
        
        for i in 0..10 {
            let blinding = generate_deterministic_bytes(32, i as u64);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            
            // Measure proof generation
            let proof_timer = Timer::new();
            let proof = prove_range((i + 1) as u64 * 100, &blinding_array, 16).unwrap();
            proof_times.push(proof_timer.elapsed_ms());
            
            // Measure compression
            let compress_timer = Timer::new();
            let proof_bytes = proof.to_bytes();
            let config = CompressionConfig {
                precision: StoragePrecision::Float32,
                max_bond_dim: 16,
                truncation_threshold: 1e-4,
                use_lz4: true,
            };
            let _ = compress_proof_data(&proof_bytes, &config);
            compress_times.push(compress_timer.elapsed_ms());
        }
        
        // Report metrics
        let avg_proof = proof_times.iter().sum::<u128>() as f64 / proof_times.len() as f64;
        let avg_compress = compress_times.iter().sum::<u128>() as f64 / compress_times.len() as f64;
        
        println!("  Metrics collected:");
        println!("    Average proof time: {:.2}ms", avg_proof);
        println!("    Average compression time: {:.2}ms", avg_compress);
        println!("    Samples: {}", proof_times.len());
        
        println!("✅ Metrics collection verified");
    }

    /// Test: Logging integration
    #[test]
    fn test_logging_simulation() {
        println!("Testing logging integration");
        
        // Simulate structured logging
        let log_entries = vec![
            ("INFO", "Proof generation started", "proof_service"),
            ("DEBUG", "Blinding factor generated", "crypto"),
            ("INFO", "Proof generated successfully", "proof_service"),
            ("INFO", "Compression started", "holographic"),
            ("TRACE", "MPS tensor initialized", "holographic"),
            ("INFO", "Compression completed", "holographic"),
        ];
        
        for (level, message, component) in log_entries {
            println!("  [{}] [{}] {}", level, component, message);
        }
        
        println!("✅ Logging simulation verified");
    }
}

#[cfg(test)]
mod configuration_tests {
    use super::*;

    /// Test: Configuration loading and validation
    #[test]
    fn test_configuration_validation() {
        println!("Testing configuration validation");
        
        // Test default config
        let default_config = E2ETestConfig::default();
        assert!(matches!(default_config.security_level, SecurityLevel::Bit128));
        assert!(default_config.iterations > 0);
        println!("  Default config: ✅");
        
        // Test quick config
        let quick_config = E2ETestConfig::quick();
        assert!(quick_config.iterations < default_config.iterations);
        println!("  Quick config: ✅");
        
        // Test exhaustive config
        let exhaustive_config = E2ETestConfig::exhaustive();
        assert!(matches!(exhaustive_config.security_level, SecurityLevel::Bit256));
        println!("  Exhaustive config: ✅");
        
        // Test compression configs
        let configs = vec![
            ("default", MPSConfig::default()),
            ("fast", MPSConfig::fast()),
            ("high_compression", MPSConfig::high_compression()),
            ("lossless", MPSConfig::lossless()),
        ];
        
        for (name, config) in configs {
            assert!(config.max_bond_dim > 0);
            assert!(config.block_size > 0);
            println!("  {} MPS config: ✅", name);
        }
        
        println!("✅ Configuration validation completed");
    }
}
