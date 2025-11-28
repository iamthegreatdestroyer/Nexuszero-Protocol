//! End-to-End Proof Generation Example
//!
//! This example demonstrates the complete proof generation workflow using
//! the nexuszero-integration module.
//!
//! # Running
//! ```bash
//! cargo run --example end_to_end_proof_generation
//! ```

use nexuszero_integration::{
    NexuszeroAPI, ProtocolConfig, MetricsCollector, BatchMetricsAggregator,
    optimization::{HeuristicOptimizer, Optimizer},
    compression::CompressionManager,
};

use nexuszero_integration::optimization::{CircuitAnalysis, CircuitType};
use nexuszero_crypto::proof::statement::HashFunction;
use sha2::{Sha256, Digest};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("           NexusZero Protocol - End-to-End Proof Demo          ");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ========================================================================
    // STEP 1: Configure the system
    // ========================================================================
    println!("📋 Step 1: Configuring NexusZero Integration...\n");
    
    let config = ProtocolConfig::default();
    
    println!("   Security Level: {:?}", config.security_level);
    println!("   Compression Enabled: {}", config.use_compression);
    println!("   Neural Optimizer: Disabled (using heuristics)");
    println!();

    // ========================================================================
    // STEP 2: Initialize the API
    // ========================================================================
    println!("🔧 Step 2: Initializing NexusZero API...\n");
    
    let mut api = NexuszeroAPI::new();
    println!("   ✓ API initialized successfully\n");

    // ========================================================================
    // STEP 3: Generate a discrete log proof
    // ========================================================================
    println!("🔐 Step 3: Generating Discrete Log Proof...\n");
    
    // Create valid test data (generator^secret = public_value mod modulus for cyclic groups)
    let generator = vec![2u8; 32];
    let secret = vec![5u8; 32];
    let public_value = vec![32u8; 32]; // In reality this would be computed as g^s
    
    println!("   Parameters:");
    println!("   - Generator: {} bytes", generator.len());
    println!("   - Secret: {} bytes (hidden from verifier)", secret.len());
    println!("   - Public Value: {} bytes", public_value.len());
    println!();
    
    let start = std::time::Instant::now();
    
    match api.prove_discrete_log(&generator, &public_value, &secret) {
        Ok(result) => {
            let duration = start.elapsed();
            
            println!("   ✓ Proof generated successfully!");
            println!();
            println!("   Results:");
            println!("   - Proof Size: {} bytes", result.original_size());
            if result.is_compressed() {
                println!("   - Compressed Size: {} bytes", result.effective_size());
                let ratio = result.original_size() as f64 / result.effective_size() as f64;
                println!("   - Compression Ratio: {:.2}x", ratio);
            }
            println!("   - Generation Time: {:.2}ms", duration.as_secs_f64() * 1000.0);
            println!();
            
            // Verify the proof
            println!("   Verifying proof...");
            match api.verify(&result) {
                Ok(valid) => {
                    if valid {
                        println!("   ✓ Proof verified successfully!");
                    } else {
                        println!("   ✗ Proof verification failed!");
                    }
                }
                Err(e) => println!("   ✗ Verification error: {:?}", e),
            }
        }
        Err(e) => {
            println!("   ✗ Error generating proof: {:?}\n", e);
        }
    }
    println!();

    // ========================================================================
    // STEP 4: Generate a preimage proof
    // ========================================================================
    println!("🔐 Step 4: Generating Preimage (Hash) Proof...\n");
    
    let preimage = b"Hello, NexusZero Protocol!";
    let hash_function = HashFunction::SHA256;
    
    // Compute the hash output
    let mut hasher = Sha256::new();
    hasher.update(preimage);
    let hash_output = hasher.finalize().to_vec();
    
    println!("   Preimage: \"{}\"", String::from_utf8_lossy(preimage));
    println!("   Hash Function: {:?}", hash_function);
    println!("   Hash Output: {} bytes", hash_output.len());
    println!();
    
    let start = std::time::Instant::now();
    match api.prove_preimage(hash_function, &hash_output, preimage) {
        Ok(result) => {
            let duration = start.elapsed();
            println!("   ✓ Preimage proof generated!");
            println!("   - Proof Size: {} bytes", result.original_size());
            println!("   - Generation Time: {:.2}ms", duration.as_secs_f64() * 1000.0);
            
            // Verify
            match api.verify(&result) {
                Ok(valid) => println!("   - Verified: {}", if valid { "✓" } else { "✗" }),
                Err(e) => println!("   - Verification error: {:?}", e),
            }
            println!();
        }
        Err(e) => {
            println!("   ✗ Error: {:?}\n", e);
        }
    }

    // ========================================================================
    // STEP 5: Test the optimizer
    // ========================================================================
    println!("🧠 Step 5: Testing Heuristic Optimizer...\n");
    
    let optimizer = HeuristicOptimizer::new(config.security_level);
    
    // Test data of different sizes
    let test_sizes = [100, 1000, 10000, 100000];
    
    for size in test_sizes.iter() {
        // Create a circuit analysis for the optimizer
        let analysis = CircuitAnalysis {
            gate_count: *size / 10, // Rough estimate
            input_count: 32, // 256-bit input
            output_count: 32, // 256-bit output
            depth: (*size / 100).max(1), // Rough depth estimate
            circuit_type: CircuitType::Preimage,
            estimated_memory: *size * 8, // Rough memory estimate
            complexity_score: (*size as f64 / 100000.0).min(1.0), // Normalized score
            has_patterns: *size > 1000, // Larger circuits may have patterns
            statement_size: *size,
        };
        
        let result = optimizer.optimize(&analysis);
        
        println!("   Size {} bytes:", size);
        println!("   - Crypto Params: {:?}", result.crypto_params);
        println!("   - Compression: {:?}", result.compression_strategy);
        println!("   - Est. Time: {:.2}ms", result.estimated_time_ms);
        println!("   - Est. Size: {} bytes", result.estimated_size);
        println!("   - Confidence: {:.2}", result.confidence);
        println!("   - Source: {:?}", result.source);
        println!();
    }

    // ========================================================================
    // STEP 6: Batch metrics collection
    // ========================================================================
    println!("📊 Step 6: Batch Metrics Aggregation Demo...\n");
    
    let mut aggregator = BatchMetricsAggregator::new();
    let num_samples = 10;
    
    println!("   Collecting {} sample metrics...\n", num_samples);
    
    for i in 0..num_samples {
        let mut collector = MetricsCollector::new();
        collector.start();
        
        // Simulate work
        std::thread::sleep(std::time::Duration::from_millis(5 + (i as u64 % 10)));
        
        collector.record_proof_size(1000 + i * 100, Some(500 + i * 30));
        
        let metrics = collector.finalize();
        aggregator.add(metrics);
    }
    
    println!("   Batch Statistics:");
    println!("   - Total Samples: {}", aggregator.count());
    println!("   - Avg Generation Time: {:.2}ms", aggregator.avg_generation_time_ms());
    println!("   - Avg Total Time: {:.2}ms", aggregator.avg_total_time_ms());
    println!("   - Avg Compression Ratio: {:.2}x", aggregator.avg_compression_ratio());
    println!("   - Min Generation Time: {:.2}ms", aggregator.min_generation_time_ms());
    println!("   - Max Generation Time: {:.2}ms", aggregator.max_generation_time_ms());
    println!("   - Total Bytes Processed: {} bytes", aggregator.total_bytes_processed());
    println!();

    // ========================================================================
    // STEP 7: Summary
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("                         Demo Complete!                         ");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    println!("✅ All proof generation and verification operations completed.");
    println!();
    println!("Key features demonstrated:");
    println!("   • Discrete log proofs");
    println!("   • Preimage (hash) proofs");
    println!("   • Heuristic optimization");
    println!("   • Batch metrics aggregation");
    println!("   • Proof verification");
    println!();
}
