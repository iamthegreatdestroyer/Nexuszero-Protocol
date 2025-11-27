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
    optimization::{CompressionStrategy, HeuristicOptimizer, Optimizer},
};

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
    println!("   Compression Enabled: {}", config.enable_compression);
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
            println!("   - Proof Size: {} bytes", result.base_proof.len());
            if let Some(compressed) = &result.compressed {
                println!("   - Compressed Size: {} bytes", compressed.len());
                let ratio = result.base_proof.len() as f64 / compressed.len() as f64;
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
    
    println!("   Preimage: \"{}\"", String::from_utf8_lossy(preimage));
    println!();
    
    let start = std::time::Instant::now();
    match api.prove_preimage(preimage) {
        Ok(result) => {
            let duration = start.elapsed();
            println!("   ✓ Preimage proof generated!");
            println!("   - Proof Size: {} bytes", result.base_proof.len());
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
    
    let optimizer = HeuristicOptimizer::new();
    
    // Test data of different sizes
    let test_sizes = [100, 1000, 10000, 100000];
    
    for size in test_sizes.iter() {
        let test_data = vec![0xABu8; *size];
        let result = optimizer.optimize(&test_data);
        
        println!("   Data Size: {} bytes", size);
        println!("   - Strategy: {:?}", result.strategy);
        println!("   - Complexity Score: {:.2}", result.complexity_score);
        println!("   - Est. Compression: {:.2}x", result.estimated_compression_ratio);
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
        collector.start_generation();
        
        // Simulate work
        std::thread::sleep(std::time::Duration::from_millis(5 + (i as u64 % 10)));
        
        collector.end_generation();
        collector.record_proof_size(1000 + i * 100, Some(500 + i * 30));
        
        let metrics = collector.finalize();
        aggregator.add_metrics(metrics);
    }
    
    println!("   Batch Statistics:");
    println!("   - Total Samples: {}", aggregator.count());
    
    if let Some(stats) = aggregator.total_time_stats() {
        println!("   - Mean Generation Time: {:.2}ms", stats.mean);
        println!("   - Std Dev: {:.2}ms", stats.std_dev);
        println!("   - Min: {:.2}ms", stats.min);
        println!("   - Max: {:.2}ms", stats.max);
    }
    
    if let Some(stats) = aggregator.compression_ratio_stats() {
        println!("   - Mean Compression Ratio: {:.2}x", stats.mean);
    }
    
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
