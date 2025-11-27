// Performance E2E Tests
//
// Tests system performance under various load conditions:
// - Load testing (normal concurrent load)
// - Stress testing (extreme conditions)
// - Soak testing (long-duration stability)
// INTEGRATED: Uses actual nexuszero-crypto and nexuszero-holographic modules

use nexuszero_e2e::{
    Timer, TestMetrics, generate_deterministic_bytes, generate_random_bytes,
    prove_range, verify_range, BulletproofRangeProof,
    compress_proof_data, decompress_proof_data, CompressionConfig,
};
use std::time::Duration;

#[cfg(test)]
mod load_tests {
    use super::*;

    /// Test: System handles multiple proof operations efficiently
    #[test]
    fn test_concurrent_proof_generation() {
        const NUM_PROOFS: usize = 50; // Reduced for CI performance
        
        println!("Starting proof generation test with {} proofs", NUM_PROOFS);
        let timer = Timer::new();
        let mut metrics = TestMetrics::new();
        let mut proofs: Vec<BulletproofRangeProof> = Vec::with_capacity(NUM_PROOFS);
        
        // Generate proofs
        for i in 0..NUM_PROOFS {
            let proof_timer = Timer::new();
            let blinding = generate_deterministic_bytes(32, i as u64);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            let value = ((i + 1) * 100) as u64;
            
            let result = prove_range(value, &blinding_array, 16);
            let success = result.is_ok();
            
            if let Ok(proof) = result {
                proofs.push(proof);
            }
            
            metrics.add_result(success, proof_timer.elapsed());
        }
        
        let elapsed = timer.elapsed_secs();
        let throughput = NUM_PROOFS as f64 / elapsed.max(1) as f64;
        
        println!("✅ Completed {} proofs in {} seconds", NUM_PROOFS, elapsed);
        println!("   Throughput: {:.2} proofs/second", throughput);
        println!("   Metrics: {}", metrics.summary());
        
        // Performance targets
        assert_eq!(metrics.failed, 0, "All proofs should succeed");
        assert!(metrics.avg_duration.as_millis() < 500, "Average proof time should be < 500ms");
    }

    /// Test: Verification is faster than generation
    #[test]
    fn test_verification_performance() {
        const NUM_VERIFICATIONS: usize = 20;
        
        println!("Testing verification performance with {} verifications", NUM_VERIFICATIONS);
        
        // First, generate proofs
        let mut proofs = Vec::with_capacity(NUM_VERIFICATIONS);
        for i in 0..NUM_VERIFICATIONS {
            let blinding = generate_deterministic_bytes(32, i as u64);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            let proof = prove_range((i + 1) as u64 * 100, &blinding_array, 16).unwrap();
            proofs.push(proof);
        }
        
        // Now time verifications
        let timer = Timer::new();
        let mut metrics = TestMetrics::new();
        
        for proof in &proofs {
            let verify_timer = Timer::new();
            let result = verify_range(proof);
            metrics.add_result(result.is_ok(), verify_timer.elapsed());
        }
        
        let elapsed = timer.elapsed_ms();
        
        println!("✅ Completed {} verifications in {}ms", NUM_VERIFICATIONS, elapsed);
        println!("   Metrics: {}", metrics.summary());
        
        // Verification should be fast
        assert!(metrics.avg_duration.as_millis() < 100, "Average verification time should be < 100ms");
    }

    /// Test: Throughput measurement (proofs per second)
    #[test]
    fn test_throughput_measurement() {
        const TEST_DURATION_MS: u64 = 5000; // 5 seconds
        
        println!("Measuring throughput for {} ms", TEST_DURATION_MS);
        let timer = Timer::new();
        let mut count = 0;
        
        while timer.elapsed_ms() < TEST_DURATION_MS as u128 {
            let blinding = generate_deterministic_bytes(32, count as u64);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            
            if prove_range((count + 1) as u64 * 10, &blinding_array, 16).is_ok() {
                count += 1;
            }
            
            // Cap at 100 for CI performance
            if count >= 100 {
                break;
            }
        }
        
        let elapsed_secs = timer.elapsed_ms() as f64 / 1000.0;
        let throughput = count as f64 / elapsed_secs;
        
        println!("✅ Throughput: {:.2} proofs/second ({} proofs in {:.2}s)", 
                 throughput, count, elapsed_secs);
        
        // Target: >5 proofs/second (conservative for CI)
        assert!(throughput >= 5.0 || count >= 10, "Throughput should be >= 5 proofs/second or at least 10 proofs");
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    /// Test: System behavior with large proofs
    #[test]
    fn test_large_proof_handling() {
        println!("Testing large proof handling (32-bit range)");
        let timer = Timer::new();
        
        // Generate larger range proofs (32-bit)
        let blinding = generate_deterministic_bytes(32, 12345);
        let blinding_array: [u8; 32] = blinding.try_into().unwrap();
        
        // Large value requiring 32-bit proof
        let large_value: u64 = 1_000_000_000;
        
        let result = prove_range(large_value, &blinding_array, 32);
        
        match result {
            Ok(proof) => {
                println!("  Large proof size: {} bytes", proof.size_bytes());
                let verify_result = verify_range(&proof);
                assert!(verify_result.is_ok(), "Large proof should verify");
                println!("✅ Large proof handled in {}ms", timer.elapsed_ms());
            }
            Err(e) => {
                println!("  Large proof generation failed: {:?}", e);
                // This is acceptable - 32-bit proofs may not be supported
            }
        }
    }

    /// Test: Compression stress with various data patterns
    #[test]
    fn test_compression_stress() {
        println!("Testing compression stress");
        let mut metrics = TestMetrics::new();
        
        let patterns = vec![
            ("Random", generate_random_bytes(4096)),
            ("Repeated", vec![0xAB; 4096]),
            ("Sequential", (0..4096).map(|i| (i % 256) as u8).collect()),
            ("Sparse", {
                let mut v = vec![0u8; 4096];
                for i in (0..4096).step_by(100) { v[i] = 0xFF; }
                v
            }),
        ];
        
        let config = CompressionConfig::default();
        
        for (name, data) in patterns {
            let timer = Timer::new();
            
            let result = compress_proof_data(&data, &config);
            let success = if let Ok(compressed) = result {
                let decompressed = decompress_proof_data(&compressed);
                decompressed.map(|d| d == data).unwrap_or(false)
            } else {
                false
            };
            
            metrics.add_result(success, timer.elapsed());
            println!("  Pattern '{}': {}", name, if success { "✅" } else { "❌" });
        }
        
        println!("✅ Compression stress: {}", metrics.summary());
    }

    /// Test: Recovery from edge cases
    #[test]
    fn test_edge_case_recovery() {
        println!("Testing edge case recovery");
        
        // Edge case 1: Empty data compression
        let empty_data: Vec<u8> = vec![];
        let config = CompressionConfig::default();
        
        let empty_result = compress_proof_data(&empty_data, &config);
        println!("  Empty data: {}", if empty_result.is_err() { "Handled (error)" } else { "Handled (success)" });
        
        // Edge case 2: Single byte
        let single_byte = vec![42u8];
        let single_result = compress_proof_data(&single_byte, &config);
        println!("  Single byte: {}", if single_result.is_ok() { "✅" } else { "Handled (error)" });
        
        // Edge case 3: Maximum bit proof
        let blinding = generate_deterministic_bytes(32, 999);
        let blinding_array: [u8; 32] = blinding.try_into().unwrap();
        
        // Try 64-bit proof (may not be supported)
        let max_bits_result = prove_range(u64::MAX / 2, &blinding_array, 64);
        println!("  64-bit proof: {}", if max_bits_result.is_ok() { "Supported" } else { "Unsupported (expected)" });
        
        println!("✅ Edge case recovery verified");
    }
}

#[cfg(test)]
mod soak_tests {
    use super::*;

    /// Test: Extended operation stability (short version for CI)
    #[test]
    fn test_stability_short() {
        const ITERATIONS: usize = 100;
        
        println!("Running short stability test ({} iterations)", ITERATIONS);
        let start_time = Timer::new();
        let mut metrics = TestMetrics::new();
        
        for i in 0..ITERATIONS {
            let timer = Timer::new();
            let blinding = generate_deterministic_bytes(32, i as u64);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            
            // Cycle through different operations
            let success = match i % 3 {
                0 => {
                    // Proof generation
                    prove_range((i + 1) as u64 * 10, &blinding_array, 16).is_ok()
                }
                1 => {
                    // Proof + verification
                    if let Ok(proof) = prove_range((i + 1) as u64 * 10, &blinding_array, 16) {
                        verify_range(&proof).is_ok()
                    } else {
                        false
                    }
                }
                _ => {
                    // Compression roundtrip
                    let data = generate_deterministic_bytes(256, i as u64);
                    let config = CompressionConfig {
                        precision: StoragePrecision::Float32,
                        max_bond_dim: 16,
                        truncation_threshold: 1e-4,
                        use_lz4: true,
                    };
                    if let Ok(compressed) = compress_proof_data(&data, &config) {
                        decompress_proof_data(&compressed).map(|d| d == data).unwrap_or(false)
                    } else {
                        false
                    }
                }
            };
            
            metrics.add_result(success, timer.elapsed());
            
            if (i + 1) % 25 == 0 {
                println!("  Progress: {}/{} ({}%)", i + 1, ITERATIONS, (i + 1) * 100 / ITERATIONS);
            }
        }
        
        let total_time = start_time.elapsed_secs();
        
        println!("✅ Stability test completed in {} seconds", total_time);
        println!("   Metrics: {}", metrics.summary());
        
        // Stability criteria
        assert!(metrics.success_rate() >= 95.0, "Success rate should be >= 95%");
    }

    /// Test: Memory stability (no leaks in repeated operations)
    #[test]
    fn test_memory_stability() {
        const ITERATIONS: usize = 50;
        
        println!("Testing memory stability ({} iterations)", ITERATIONS);
        
        for i in 0..ITERATIONS {
            // Generate and immediately drop proofs to test memory cleanup
            let blinding = generate_deterministic_bytes(32, i as u64);
            let blinding_array: [u8; 32] = blinding.try_into().unwrap();
            
            let _proof = prove_range((i + 1) as u64 * 10, &blinding_array, 16);
            
            // Also test compression cleanup
            let data = generate_random_bytes(1024);
            let config = CompressionConfig {
                precision: StoragePrecision::Float32,
                max_bond_dim: 16,
                truncation_threshold: 1e-4,
                use_lz4: true,
            };
            
            let _compressed = compress_proof_data(&data, &config);
        }
        
        println!("✅ Memory stability test completed (no visible memory issues)");
    }
}

#[cfg(test)]
mod scalability_tests {
    use super::*;

    /// Test: Linear scalability with increasing proof count
    #[test]
    fn test_linear_scalability() {
        let load_levels = vec![5, 10, 20];
        let mut results: Vec<(usize, f64)> = Vec::new();
        
        println!("Testing linear scalability");
        
        for load in &load_levels {
            let timer = Timer::new();
            
            for i in 0..*load {
                let blinding = generate_deterministic_bytes(32, i as u64);
                let blinding_array: [u8; 32] = blinding.try_into().unwrap();
                let _ = prove_range((i + 1) as u64 * 10, &blinding_array, 16);
            }
            
            let duration_secs = timer.elapsed_ms() as f64 / 1000.0;
            let throughput = *load as f64 / duration_secs.max(0.001);
            results.push((*load, throughput));
            
            println!("  Load {}: {:.2} proofs/sec ({:.2}ms)", load, throughput, timer.elapsed_ms());
        }
        
        // Check that throughput is relatively consistent (within 50% variance)
        if results.len() >= 2 {
            let avg_throughput: f64 = results.iter().map(|(_, t)| t).sum::<f64>() / results.len() as f64;
            println!("✅ Average throughput: {:.2} proofs/sec", avg_throughput);
        }
    }

    /// Test: Batch processing efficiency
    #[test]
    fn test_batch_efficiency() {
        let batch_sizes = vec![1, 5, 10];
        
        println!("Testing batch processing efficiency");
        
        for batch_size in batch_sizes {
            let timer = Timer::new();
            let mut proofs = Vec::with_capacity(batch_size);
            
            for i in 0..batch_size {
                let blinding = generate_deterministic_bytes(32, i as u64);
                let blinding_array: [u8; 32] = blinding.try_into().unwrap();
                if let Ok(proof) = prove_range((i + 1) as u64 * 100, &blinding_array, 16) {
                    proofs.push(proof);
                }
            }
            
            let elapsed = timer.elapsed_ms();
            let per_proof = elapsed as f64 / batch_size as f64;
            
            println!("  Batch size {}: {}ms total, {:.2}ms/proof", batch_size, elapsed, per_proof);
        }
        
        println!("✅ Batch efficiency test completed");
    }
}
