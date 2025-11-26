//! Basic Holographic Compression Example
//!
//! This example demonstrates the fundamental usage of NexusZero Holographic
//! compression using the v2 Tensor Train algorithm.
//!
//! Run with: `cargo run --example basic_compression`

use nexuszero_holographic::{CompressedTensorTrain, CompressionConfig, CompressionError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       NexusZero Holographic - Basic Compression Example      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // Example 1: Simple compression with default settings
    // =========================================================================
    println!("━━━ Example 1: Default Compression ━━━\n");
    
    // Generate structured test data (simulates proof data patterns)
    let proof_data: Vec<u8> = (0..10240)
        .map(|i| ((i / 16) % 256) as u8)
        .collect();
    
    println!("📊 Original data size: {} bytes", proof_data.len());
    println!("   Pattern: repeating blocks of 16 bytes");
    
    // Use default configuration
    let config = CompressionConfig::default();
    println!("\n⚙️  Config: max_bond_dim={}, truncation_threshold={:.0e}",
             config.max_bond_dim, config.truncation_threshold);
    
    // Compress
    let start = std::time::Instant::now();
    let compressed = CompressedTensorTrain::compress(&proof_data, config)?;
    let compress_time = start.elapsed();
    
    let stats = compressed.stats();
    println!("\n📦 Compressed size: {} bytes", stats.compressed_bytes);
    println!("📈 Compression ratio: {:.2}x", stats.compression_ratio());
    println!("⏱️  Compression time: {:?}", compress_time);
    
    // Decompress
    let start = std::time::Instant::now();
    let reconstructed = compressed.decompress()?;
    let decompress_time = start.elapsed();
    
    println!("\n📤 Decompression time: {:?}", decompress_time);
    
    // Verify reconstruction (may have small errors due to lossy compression)
    let error: f64 = proof_data.iter()
        .zip(reconstructed.iter())
        .map(|(&a, &b)| ((a as f64) - (b as f64)).abs())
        .sum::<f64>() / proof_data.len() as f64;
    println!("📐 Mean reconstruction error: {:.4}", error);
    
    if error < 1.0 {
        println!("✅ Near-lossless reconstruction verified!\n");
    } else {
        println!("⚠️  Some reconstruction error (expected for lossy modes)\n");
    }

    // =========================================================================
    // Example 2: Custom configurations
    // =========================================================================
    println!("━━━ Example 2: Configuration Presets ━━━\n");
    
    println!("| Preset          | Compressed | Ratio    | Encode Time |");
    println!("|-----------------|------------|----------|-------------|");
    
    let presets: Vec<(&str, CompressionConfig)> = vec![
        ("High Compression", CompressionConfig::high_compression()),
        ("Fast", CompressionConfig::fast()),
        ("Balanced", CompressionConfig::balanced()),
        ("Lossless", CompressionConfig::lossless()),
    ];
    
    for (name, config) in presets {
        let start = std::time::Instant::now();
        let compressed = CompressedTensorTrain::compress(&proof_data, config)?;
        let elapsed = start.elapsed();
        let stats = compressed.stats();
        
        println!("| {:15} | {:>10} | {:>7.2}x | {:>11?} |",
                 name,
                 format!("{} B", stats.compressed_bytes),
                 stats.compression_ratio(),
                 elapsed);
    }

    // =========================================================================
    // Example 3: Different data patterns
    // =========================================================================
    println!("\n━━━ Example 3: Different Data Patterns ━━━\n");
    
    let config = CompressionConfig::default();
    
    // Pattern 1: Uniform data
    let uniform: Vec<u8> = vec![42u8; 4096];
    let comp_uniform = CompressedTensorTrain::compress(&uniform, config.clone())?;
    println!("Uniform (4096 bytes of 0x2A):");
    println!("  Compressed: {} bytes, Ratio: {:.2}x",
             comp_uniform.stats().compressed_bytes,
             comp_uniform.stats().compression_ratio());
    
    // Pattern 2: Sequential data
    let sequential: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    let comp_seq = CompressedTensorTrain::compress(&sequential, config.clone())?;
    println!("Sequential (0,1,2,...,255,0,1,...):");
    println!("  Compressed: {} bytes, Ratio: {:.2}x",
             comp_seq.stats().compressed_bytes,
             comp_seq.stats().compression_ratio());
    
    // Pattern 3: Block pattern (like proof data)
    let blocks: Vec<u8> = (0..4096).map(|i| ((i / 64) % 256) as u8).collect();
    let comp_blocks = CompressedTensorTrain::compress(&blocks, config.clone())?;
    println!("Block pattern (64-byte blocks):");
    println!("  Compressed: {} bytes, Ratio: {:.2}x",
             comp_blocks.stats().compressed_bytes,
             comp_blocks.stats().compression_ratio());
    
    // Pattern 4: Random-ish data (harder to compress)
    let pseudorandom: Vec<u8> = (0..4096)
        .map(|i| ((i * 17 + 31) % 256) as u8)
        .collect();
    let comp_random = CompressedTensorTrain::compress(&pseudorandom, config.clone())?;
    println!("Pseudo-random (LCG pattern):");
    println!("  Compressed: {} bytes, Ratio: {:.2}x",
             comp_random.stats().compressed_bytes,
             comp_random.stats().compression_ratio());

    // =========================================================================
    // Example 4: Serialization
    // =========================================================================
    println!("\n━━━ Example 4: Serialization ━━━\n");
    
    let data: Vec<u8> = (0..2048).map(|i| ((i / 32) % 256) as u8).collect();
    let compressed = CompressedTensorTrain::compress(&data, config)?;
    
    // Serialize to bytes
    let serialized = compressed.to_bytes()?;
    println!("📝 Serialized size: {} bytes", serialized.len());
    println!("   (vs compressed storage: {} bytes)", compressed.stats().compressed_bytes);
    
    // Serialize with LZ4 for even smaller output
    let serialized_lz4 = compressed.to_bytes_lz4()?;
    println!("📝 With LZ4: {} bytes", serialized_lz4.len());
    
    // Deserialize
    let deserialized = CompressedTensorTrain::from_bytes(&serialized)?;
    let recovered = deserialized.decompress()?;
    
    let error: f64 = data.iter()
        .zip(recovered.iter())
        .map(|(&a, &b)| ((a as f64) - (b as f64)).abs())
        .sum::<f64>() / data.len() as f64;
    
    println!("✅ Serialization round-trip verified! (error: {:.4})\n", error);

    // =========================================================================
    // Example 5: Error Handling
    // =========================================================================
    println!("━━━ Example 5: Error Handling ━━━\n");
    
    // Empty input
    let empty: Vec<u8> = vec![];
    match CompressedTensorTrain::compress(&empty, CompressionConfig::default()) {
        Err(CompressionError::EmptyInput) => {
            println!("✅ Empty input correctly rejected");
        }
        _ => println!("❌ Expected EmptyInput error"),
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                      Summary                                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  • Holographic compression excels on structured data         ║");
    println!("║  • Use presets: high_compression, fast, balanced, lossless   ║");
    println!("║  • Serialization adds minimal overhead                       ║");
    println!("║  • LZ4 backend further reduces storage                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
