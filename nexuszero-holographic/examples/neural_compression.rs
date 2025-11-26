//! Neural-Enhanced Holographic Compression Example
//!
//! This example demonstrates using the neural enhancement feature for
//! optimized compression parameters.
//!
//! Run with: `cargo run --example neural_compression`
//! With neural feature: `cargo run --example neural_compression --features neural`

use nexuszero_holographic::{
    NeuralCompressor, NeuralConfig, CompressedTensorTrain, CompressionConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     NexusZero Holographic - Neural Compression Example       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Check if neural feature is available
    #[cfg(feature = "neural")]
    {
        println!("🧠 Neural feature: ENABLED (PyTorch bindings available)");
    }
    #[cfg(not(feature = "neural"))]
    {
        println!("⚠️  Neural feature: DISABLED (using heuristic fallback)");
        println!("   To enable: cargo run --example neural_compression --features neural\n");
    }

    // =========================================================================
    // Generate test data
    // =========================================================================
    println!("\n━━━ Generating Test Data ━━━\n");
    
    // Structured data that benefits from neural optimization
    let proof_data: Vec<u8> = (0..20480)
        .map(|i| {
            // Mix of patterns to simulate real proof data
            let block = (i / 64) as u8;
            let offset = (i % 64) as u8;
            block.wrapping_add(offset.wrapping_mul(7))
        })
        .collect();
    
    println!("📊 Test data: {} bytes", proof_data.len());
    println!("   Pattern: block-structured with local variations\n");

    // =========================================================================
    // Standard compression (baseline)
    // =========================================================================
    println!("━━━ Baseline: Standard Compression ━━━\n");
    
    let config = CompressionConfig::default();
    
    let start = std::time::Instant::now();
    let standard = CompressedTensorTrain::compress(&proof_data, config)?;
    let standard_time = start.elapsed();
    
    let standard_stats = standard.stats();
    let standard_ratio = standard_stats.compression_ratio();
    println!("📦 Standard compression:");
    println!("   Compressed size: {} bytes", standard_stats.compressed_bytes);
    println!("   Compression ratio: {:.2}x", standard_ratio);
    println!("   Time: {:?}", standard_time);

    // =========================================================================
    // Neural-enhanced compression
    // =========================================================================
    println!("\n━━━ Neural-Enhanced Compression ━━━\n");
    
    // Configure neural compressor with fallback
    let neural_config = NeuralConfig::default();
    
    println!("⚙️  Neural config:");
    println!("   Model path: {:?}", neural_config.model_path);
    println!("   Use GPU: {}", neural_config.use_gpu);
    println!("   Fallback on error: {}", neural_config.fallback_on_error);
    
    let compressor = NeuralCompressor::from_config(&neural_config)?;
    
    println!("\n🔧 Compressor status: {}", 
             if compressor.is_enabled() { "Neural model loaded" } 
             else { "Using heuristic fallback" });
    
    let start = std::time::Instant::now();
    let neural = compressor.compress_v2(&proof_data)?;
    let neural_time = start.elapsed();
    
    let neural_stats = neural.stats();
    println!("\n📦 Neural compression:");
    println!("   Neural enhanced: {}", neural.neural_enhanced);
    println!("   Compressed size: {} bytes", neural_stats.compressed_bytes);
    println!("   Time: {:?}", neural_time);
    
    // Get compression analysis
    let analysis = compressor.analyze(&proof_data);
    println!("\n📊 Neural analysis:");
    println!("   Predicted scale: {:.4}", analysis.predicted_params.scale);
    println!("   Predicted zero point: {:.4}", analysis.predicted_params.zero_point);
    println!("   Neural enabled: {}", analysis.neural_enabled);
    if let Some(bond_hint) = analysis.predicted_params.bond_dim_hint {
        println!("   Suggested bond dim: {}", bond_hint);
    }

    // =========================================================================
    // Comparison
    // =========================================================================
    println!("\n━━━ Comparison: Standard vs Neural ━━━\n");
    
    let neural_ratio = neural_stats.compression_ratio();
    let improvement = if standard_ratio > 0.0 {
        ((neural_ratio / standard_ratio) - 1.0) * 100.0
    } else {
        0.0
    };
    
    println!("| Metric            | Standard | Neural   | Diff      |");
    println!("|-------------------|----------|----------|-----------|");
    println!("| Compressed (B)    | {:>8} | {:>8} | {:>+8} |",
             standard_stats.compressed_bytes,
             neural_stats.compressed_bytes,
             neural_stats.compressed_bytes as i64 - standard_stats.compressed_bytes as i64);
    println!("| Ratio             | {:>7.2}x | {:>7.2}x | {:>+8.1}% |",
             standard_ratio,
             neural_ratio,
             improvement);
    println!("| Encode time       | {:>8?} | {:>8?} |           |",
             standard_time,
             neural_time);

    // =========================================================================
    // Verify reconstruction
    // =========================================================================
    println!("\n━━━ Verification ━━━\n");
    
    // Standard reconstruction
    let start = std::time::Instant::now();
    let standard_reconstructed = standard.decompress()?;
    let standard_decomp_time = start.elapsed();
    
    // Neural reconstruction
    let start = std::time::Instant::now();
    let neural_reconstructed = compressor.decompress_v2(&neural)?;
    let neural_decomp_time = start.elapsed();
    
    // Calculate reconstruction errors
    let standard_error: f64 = proof_data.iter()
        .zip(standard_reconstructed.iter())
        .map(|(&a, &b)| ((a as f64) - (b as f64)).abs())
        .sum::<f64>() / proof_data.len() as f64;
    
    let neural_error: f64 = proof_data.iter()
        .zip(neural_reconstructed.iter())
        .map(|(&a, &b)| ((a as f64) - (b as f64)).abs())
        .sum::<f64>() / proof_data.len() as f64;
    
    println!("Standard reconstruction:");
    println!("   Time: {:?}", standard_decomp_time);
    println!("   Mean error: {:.4}", standard_error);
    
    println!("\nNeural reconstruction:");
    println!("   Time: {:?}", neural_decomp_time);
    println!("   Mean error: {:.4}", neural_error);
    println!("   Length match: {}", proof_data.len() == neural_reconstructed.len());

    // =========================================================================
    // Different data patterns
    // =========================================================================
    println!("\n━━━ Neural Performance by Pattern ━━━\n");
    
    let patterns: Vec<(&str, Vec<u8>)> = vec![
        ("Uniform", vec![128u8; 4096]),
        ("Sequential", (0..4096).map(|i| (i % 256) as u8).collect()),
        ("Block-64", (0..4096).map(|i| ((i / 64) % 256) as u8).collect()),
        ("Block-256", (0..4096).map(|i| ((i / 256) % 256) as u8).collect()),
        ("Pseudo-random", (0..4096).map(|i| ((i * 17 + 31) % 256) as u8).collect()),
    ];
    
    println!("| Pattern      | Standard | Neural   | Improvement |");
    println!("|--------------|----------|----------|-------------|");
    
    for (name, data) in patterns {
        let std_comp = CompressedTensorTrain::compress(&data, CompressionConfig::default())?;
        let neural_comp = compressor.compress_v2(&data)?;
        
        let std_ratio = std_comp.stats().compression_ratio();
        let neural_ratio = neural_comp.stats().compression_ratio();
        let improvement = if std_ratio > 0.0 {
            ((neural_ratio / std_ratio) - 1.0) * 100.0
        } else {
            0.0
        };
        
        println!("| {:12} | {:>7.2}x | {:>7.2}x | {:>+10.1}% |",
                 name, std_ratio, neural_ratio, improvement);
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                      Summary                                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  • Neural enhancement provides data-adaptive optimization    ║");
    println!("║  • Best improvements on structured, non-uniform data         ║");
    println!("║  • Graceful fallback when PyTorch unavailable               ║");
    println!("║  • Heuristic mode still provides reasonable parameters       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
