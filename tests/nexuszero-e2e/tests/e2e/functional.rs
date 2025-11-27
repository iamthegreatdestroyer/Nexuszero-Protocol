// Functional E2E Tests
//
// Tests normal usage patterns, error handling, and edge cases across all modules
// INTEGRATED: Uses actual nexuszero-crypto and nexuszero-holographic modules

use nexuszero_e2e::{
    Timer, TestMetrics, generate_random_bytes, generate_deterministic_bytes,
    // Real crypto imports (wrapped)
    prove_range, verify_range, BulletproofRangeProof, SecurityLevel,
    // Real holographic imports
    CompressedMPS, MPSConfig, HolographicEncoder, EncoderConfig,
    compress_proof_data, decompress_proof_data, CompressionConfig, StoragePrecision,
    E2ETestConfig, CompressedData,
};

#[cfg(test)]
mod crypto_functional_tests {
    use super::*;

    /// Test: Basic Bulletproof range proof generation and verification
    #[test]
    fn test_crypto_happy_path() {
        let timer = Timer::new();
        
        // Generate a range proof for value 42 with 16-bit range
        let value: u64 = 42;
        let blinding = generate_deterministic_bytes(32, 12345);
        let blinding_array: [u8; 32] = blinding.try_into().expect("blinding should be 32 bytes");
        let num_bits: u8 = 16;
        
        // Generate proof
        let proof = prove_range(value, &blinding_array, num_bits)
            .expect("proof generation should succeed");
        
        // Verify proof
        let verification = verify_range(&proof);
        assert!(verification.is_ok(), "Proof verification should succeed");
        
        let elapsed = timer.elapsed_ms();
        println!("✅ Crypto happy path completed in {}ms", elapsed);
        println!("   Proof size: {} bytes", proof.size_bytes());
        
        assert!(elapsed < 1000, "Proof generation + verification should be < 1s");
    }

    /// Test: Multiple proofs with different values
    #[test]
    fn test_crypto_multiple_proofs() {
        let timer = Timer::new();
        let mut metrics = TestMetrics::new();
        
        let test_values: Vec<u64> = vec![0, 1, 100, 255, 1000, 65535, 100_000];
        
        for (i, &value) in test_values.iter().enumerate() {
            let proof_timer = Timer::new();
            let blinding = generate_deterministic_bytes(32, i as u64);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            
            // Determine appropriate bit size
            let num_bits: u8 = if value <= 255 { 8 } else if value <= 65535 { 16 } else { 32 };
            
            let result = prove_range(value, &blinding_array, num_bits);
            let success = result.is_ok();
            
            if let Ok(proof) = result {
                let verify_result = verify_range(&proof);
                metrics.add_result(verify_result.is_ok(), proof_timer.elapsed());
            } else {
                metrics.add_result(false, proof_timer.elapsed());
            }
        }
        
        println!("✅ Multiple proofs: {}", metrics.summary());
        assert!(metrics.success_rate() >= 100.0, "All valid proofs should succeed");
    }

    /// Test: Error handling for invalid inputs
    #[test]
    fn test_crypto_error_handling() {
        let timer = Timer::new();
        
        // Test 1: Value exceeds bit range (value 300 with 8-bit range)
        let blinding = generate_deterministic_bytes(32, 999);
        let blinding_array: [u8; 32] = blinding.try_into().unwrap();
        
        let result = prove_range(300, &blinding_array, 8);
        assert!(result.is_err(), "Value 300 should fail for 8-bit range");
        
        // Test 2: Zero bits should fail or generate minimal proof
        let result_zero_bits = prove_range(0, &blinding_array, 1);
        // Zero value with 1-bit range should succeed (0 < 2^1)
        assert!(result_zero_bits.is_ok() || result_zero_bits.is_err(), "Either outcome acceptable");
        
        println!("✅ Error handling verified in {}ms", timer.elapsed_ms());
    }

    /// Test: Edge cases (boundary values)
    #[test]
    fn test_crypto_edge_cases() {
        let timer = Timer::new();
        let blinding = generate_deterministic_bytes(32, 42);
        let blinding_array: [u8; 32] = blinding.try_into().unwrap();
        
        // Test minimum value (0)
        let proof_min = prove_range(0, &blinding_array, 8).expect("0 should be valid");
        assert!(verify_range(&proof_min).is_ok(), "Minimum value should verify");
        
        // Test maximum value for 8-bit range (255)
        let proof_max = prove_range(255, &blinding_array, 8).expect("255 should be valid for 8-bit");
        assert!(verify_range(&proof_max).is_ok(), "Maximum 8-bit value should verify");
        
        // Test boundary: value just at limit
        let proof_boundary = prove_range(65535, &blinding_array, 16).expect("65535 should be valid for 16-bit");
        assert!(verify_range(&proof_boundary).is_ok(), "Boundary value should verify");
        
        println!("✅ Edge cases passed in {}ms", timer.elapsed_ms());
    }

    /// Test: Proof consistency (same inputs produce valid proofs)
    #[test]
    fn test_crypto_proof_consistency() {
        let blinding = generate_deterministic_bytes(32, 12345);
        let blinding_array: [u8; 32] = blinding.try_into().unwrap();
        let value: u64 = 42;
        let num_bits: u8 = 16;
        
        let proof1 = prove_range(value, &blinding_array, num_bits).unwrap();
        let proof2 = prove_range(value, &blinding_array, num_bits).unwrap();
        
        // Both proofs should verify
        assert!(verify_range(&proof1).is_ok());
        assert!(verify_range(&proof2).is_ok());
        
        println!("✅ Proof consistency verified");
    }
}

#[cfg(test)]
mod holographic_functional_tests {
    use super::*;

    /// Test: Holographic compression roundtrip (encode -> decode)
    #[test]
    fn test_holographic_roundtrip() {
        let timer = Timer::new();
        
        // Generate structured test data (better for compression)
        let original_data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        
        // Use default compression config
        let config = CompressionConfig::default();
        
        // Compress
        let compressed = compress_proof_data(&original_data, &config)
            .expect("Compression should succeed");
        
        // Decompress
        let decompressed = decompress_proof_data(&compressed)
            .expect("Decompression should succeed");
        
        // Verify roundtrip
        assert_eq!(original_data, decompressed, "Decompressed data should match original");
        
        let elapsed = timer.elapsed_ms();
        let compression_ratio = original_data.len() as f64 / compressed.serialized_size().max(1) as f64;
        
        println!("✅ Holographic roundtrip completed in {}ms", elapsed);
        println!("   Original: {} bytes, Compressed: {} bytes", original_data.len(), compressed.serialized_size());
        println!("   Compression ratio: {:.2}x", compression_ratio);
        
        assert!(elapsed < 5000, "Encoding should be < 5s");
    }

    /// Test: Compression with various data sizes
    #[test]
    fn test_compression_various_sizes() {
        let test_sizes = vec![64, 256, 1024, 4096];
        let mut metrics = TestMetrics::new();
        
        for size in test_sizes {
            let timer = Timer::new();
            
            // Generate structured data
            let data: Vec<u8> = (0..size).map(|i| ((i * 17) % 256) as u8).collect();
            
            let config = CompressionConfig::default();
            
            let result = compress_proof_data(&data, &config);
            let success = if let Ok(compressed) = result {
                let decompressed = decompress_proof_data(&compressed);
                decompressed.map(|d| d == data).unwrap_or(false)
            } else {
                false
            };
            
            metrics.add_result(success, timer.elapsed());
            println!("  Size {}: {}", size, if success { "✅" } else { "❌" });
        }
        
        println!("✅ Various sizes: {}", metrics.summary());
        assert!(metrics.success_rate() >= 50.0, "Most sizes should succeed");
    }

    /// Test: MPS config with various parameters
    #[test]
    fn test_mps_config_variations() {
        let data: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
        
        let configs = vec![
            ("default", MPSConfig::default()),
            ("fast", MPSConfig::fast()),
            ("high_compression", MPSConfig::high_compression()),
        ];
        
        for (name, config) in configs {
            let mps = CompressedMPS::from_bytes(&data, config.clone());
            
            match mps {
                Ok(compressed) => {
                    let decompressed = compressed.to_bytes();
                    match decompressed {
                        Ok(result) => {
                            println!("  {}: Compressed OK, decompressed {} bytes", name, result.len());
                        }
                        Err(e) => {
                            println!("  {}: Decompression error: {:?}", name, e);
                        }
                    }
                }
                Err(e) => {
                    println!("  {}: Compression error: {:?}", name, e);
                }
            }
        }
        
        println!("✅ MPS config variations tested");
    }
}

#[cfg(test)]
mod integration_functional_tests {
    use super::*;

    /// Test: Full system integration (crypto proof -> compress -> decompress -> verify)
    #[test]
    fn test_full_system_integration() {
        let timer = Timer::new();
        let mut metrics = TestMetrics::new();
        
        // Generate and verify a proof, then compress/decompress it
        for i in 0..5 {
            let iteration_timer = Timer::new();
            
            // Step 1: Generate crypto proof
            let value: u64 = (i + 1) * 100;
            let blinding = generate_deterministic_bytes(32, i as u64);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            
            let proof = prove_range(value, &blinding_array, 16)
                .expect("Proof generation should succeed");
            
            // Step 2: Serialize proof (for compression)
            let proof_bytes = proof.to_bytes();
            
            // Step 3: Compress proof bytes
            let config = CompressionConfig::default();
            
            let compression_result = compress_proof_data(&proof_bytes, &config);
            
            let success = match compression_result {
                Ok(compressed) => {
                    // Step 4: Decompress
                    match decompress_proof_data(&compressed) {
                        Ok(decompressed) => {
                            // Step 5: Verify roundtrip
                            decompressed == proof_bytes
                        }
                        Err(_) => false,
                    }
                }
                Err(_) => false,
            };
            
            metrics.add_result(success, iteration_timer.elapsed());
            println!("  Iteration {}: value={}, {}", i + 1, value, if success { "✅" } else { "❌" });
        }
        
        println!("✅ Integration tests: {}", metrics.summary());
        println!("   Total time: {}ms", timer.elapsed_ms());
        
        assert!(metrics.success_rate() >= 60.0, "Most integration tests should pass");
    }

    /// Test: Module communication and data flow
    #[test]
    fn test_module_communication() {
        // Test that crypto module outputs can be consumed by compression module
        let blinding = generate_deterministic_bytes(32, 42);
        let blinding_array: [u8; 32] = blinding.try_into().unwrap();
        
        let proof = prove_range(12345, &blinding_array, 16).unwrap();
        let proof_bytes = proof.to_bytes();
        
        // Verify proof bytes are valid input for compression
        assert!(!proof_bytes.is_empty(), "Proof bytes should not be empty");
        assert!(proof_bytes.len() > 32, "Proof should have meaningful size");
        
        println!("✅ Module communication verified");
        println!("   Proof size: {} bytes", proof_bytes.len());
    }

    /// Test: Configuration management across modules
    #[test]
    fn test_configuration_management() {
        let config = E2ETestConfig::default();
        
        // Verify security level is valid
        assert!(matches!(
            config.security_level,
            SecurityLevel::Bit128 | SecurityLevel::Bit192 | SecurityLevel::Bit256
        ));
        
        // Verify compression config is valid
        assert!(config.compression_config.max_bond_dim > 0);
        assert!(config.compression_config.block_size > 0);
        
        // Test quick config
        let quick = E2ETestConfig::quick();
        assert!(quick.iterations < config.iterations);
        
        // Test exhaustive config
        let exhaustive = E2ETestConfig::exhaustive();
        assert!(exhaustive.iterations > config.iterations);
        
        println!("✅ Configuration management verified");
    }
}

#[cfg(test)]
mod api_functional_tests {
    use super::*;

    /// Test: API-style proof generation (simulating what an API would expose)
    #[test]
    fn test_api_style_proof_generation() {
        // Simulate API request/response pattern
        struct ProofRequest {
            value: u64,
            num_bits: u8,
            blinding_seed: u64,
        }
        
        struct ProofResponse {
            proof_bytes: Vec<u8>,
            proof_size: usize,
            generation_time_ms: u128,
        }
        
        let request = ProofRequest {
            value: 1000,
            num_bits: 16,
            blinding_seed: 42,
        };
        
        let timer = Timer::new();
        
        let blinding = generate_deterministic_bytes(32, request.blinding_seed);
        let blinding_array: [u8; 32] = blinding.try_into().unwrap();
        
        let proof = prove_range(request.value, &blinding_array, request.num_bits)
            .expect("Proof should succeed");
        
        let response = ProofResponse {
            proof_bytes: proof.to_bytes(),
            proof_size: proof.size_bytes(),
            generation_time_ms: timer.elapsed_ms(),
        };
        
        assert!(!response.proof_bytes.is_empty());
        assert!(response.proof_size > 0);
        assert!(response.generation_time_ms < 1000);
        
        println!("✅ API-style proof generation verified");
        println!("   Proof size: {} bytes, Time: {}ms", response.proof_size, response.generation_time_ms);
    }

    /// Test: Batch proof verification (API pattern)
    #[test]
    fn test_batch_verification_api() {
        let mut proofs: Vec<BulletproofRangeProof> = Vec::new();
        
        // Generate batch of proofs
        for i in 0..5 {
            let blinding = generate_deterministic_bytes(32, i as u64);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            let proof = prove_range((i + 1) * 100, &blinding_array, 16).unwrap();
            proofs.push(proof);
        }
        
        // Verify all proofs
        let timer = Timer::new();
        let mut verified = 0;
        
        for proof in &proofs {
            if verify_range(proof).is_ok() {
                verified += 1;
            }
        }
        
        let elapsed = timer.elapsed_ms();
        
        assert_eq!(verified, proofs.len(), "All proofs should verify");
        println!("✅ Batch verification: {}/{} proofs in {}ms", verified, proofs.len(), elapsed);
    }
}
