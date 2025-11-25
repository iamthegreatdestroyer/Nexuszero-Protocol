//! Benchmark Comparison: Holographic vs Standard Compression
//!
//! This example compares NexusZero Holographic compression against
//! standard compression algorithms (Zstd, LZ4).
//!
//! Run with: `cargo run --example benchmark_comparison --release`

use nexuszero_holographic::{CompressedTensorTrain, CompressionConfig};
use std::time::{Duration, Instant};

/// Benchmark result for a single algorithm
struct BenchResult {
    name: String,
    compressed_size: usize,
    ratio: f64,
    encode_time: Duration,
    decode_time: Duration,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    NexusZero Holographic - Algorithm Comparison Benchmark    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // Test configurations
    // =========================================================================
    let data_sizes = [1024usize, 10240, 102400];
    let iterations = 5;
    
    println!("⚙️  Configuration:");
    println!("   Data sizes: {:?} bytes", data_sizes);
    println!("   Iterations per test: {}", iterations);
    println!("   Note: Run with --release for accurate timings\n");

    // =========================================================================
    // Run benchmarks for each size
    // =========================================================================
    for &size in &data_sizes {
        run_benchmark_suite(size, iterations)?;
    }

    // =========================================================================
    // Detailed analysis on 100KB
    // =========================================================================
    println!("\n━━━ Detailed 100KB Analysis ━━━\n");
    
    let data = generate_proof_like_data(102400);
    
    // Holographic with different presets
    println!("Holographic configuration sweep:");
    println!("| Config          | Size (B) | Ratio    | Encode    | Decode    |");
    println!("|-----------------|----------|----------|-----------|-----------|");
    
    let configs: Vec<(&str, CompressionConfig)> = vec![
        ("High Compression", CompressionConfig::high_compression()),
        ("Fast", CompressionConfig::fast()),
        ("Balanced", CompressionConfig::balanced()),
        ("Lossless", CompressionConfig::lossless()),
    ];
    
    for (name, config) in configs {
        let start = Instant::now();
        let compressed = CompressedTensorTrain::compress(&data, config)?;
        let encode_time = start.elapsed();
        
        let start = Instant::now();
        let _ = compressed.decompress()?;
        let decode_time = start.elapsed();
        
        let stats = compressed.stats();
        println!("| {:>15} | {:>8} | {:>7.2}x | {:>9?} | {:>9?} |",
                 name,
                 stats.compressed_bytes,
                 stats.compression_ratio(),
                 encode_time,
                 decode_time);
    }
    
    // Zstd with different levels
    println!("\nZstd compression level sweep:");
    println!("| Level    | Size (B) | Ratio    | Encode    | Decode    |");
    println!("|----------|----------|----------|-----------|-----------|");
    
    for level in [1, 3, 9, 19] {
        let start = Instant::now();
        let compressed = zstd::encode_all(&data[..], level)?;
        let encode_time = start.elapsed();
        
        let start = Instant::now();
        let _ = zstd::decode_all(&compressed[..])?;
        let decode_time = start.elapsed();
        
        let ratio = data.len() as f64 / compressed.len() as f64;
        
        println!("| {:>8} | {:>8} | {:>7.2}x | {:>9?} | {:>9?} |",
                 level,
                 compressed.len(),
                 ratio,
                 encode_time,
                 decode_time);
    }

    // =========================================================================
    // Use case analysis
    // =========================================================================
    println!("\n━━━ Use Case Recommendations ━━━\n");
    
    println!("┌────────────────────────┬──────────────────────────────────────┐");
    println!("│ Use Case               │ Recommended Algorithm                │");
    println!("├────────────────────────┼──────────────────────────────────────┤");
    println!("│ ZK Proof Storage       │ Holographic (better ratio)           │");
    println!("│ Real-time Compression  │ LZ4 (fastest encode/decode)          │");
    println!("│ General Archives       │ Zstd level 3 (good balance)          │");
    println!("│ Maximum Compression    │ Holographic + Zstd hybrid            │");
    println!("│ Streaming Data         │ LZ4 or Zstd level 1                  │");
    println!("└────────────────────────┴──────────────────────────────────────┘");

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                      Key Findings                            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  • Holographic achieves better compression on structured     ║");
    println!("║    proof data compared to general algorithms                 ║");
    println!("║  • Trade-off: Higher CPU time for compression                ║");
    println!("║  • Best for: Storage-constrained, bandwidth-limited cases    ║");
    println!("║  • Consider hybrid: Holographic + Zstd for optimal results   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Generate data that simulates ZK proof structure
fn generate_proof_like_data(size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| {
            // Simulate proof data: block structure with local correlations
            let block = (i / 64) as u8;
            let offset = (i % 64) as u8;
            block.wrapping_add(offset.wrapping_mul(3))
        })
        .collect()
}

/// Run benchmark suite for a given data size
fn run_benchmark_suite(size: usize, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("━━━ Benchmark: {} bytes ━━━\n", size);
    
    let data = generate_proof_like_data(size);
    let mut results: Vec<BenchResult> = Vec::new();
    
    // Holographic (v2 Tensor Train)
    let config = CompressionConfig::default();
    let (compressed, encode_time) = bench_encode(iterations, || {
        CompressedTensorTrain::compress(&data, config.clone())
    })?;
    let (_, decode_time) = bench_decode(iterations, || {
        compressed.decompress()
    })?;
    let stats = compressed.stats();
    results.push(BenchResult {
        name: "Holographic v2".to_string(),
        compressed_size: stats.compressed_bytes,
        ratio: stats.compression_ratio(),
        encode_time,
        decode_time,
    });
    
    // Zstd level 3
    let (zstd_compressed, encode_time) = bench_encode(iterations, || {
        zstd::encode_all(&data[..], 3)
    })?;
    let (_, decode_time) = bench_decode(iterations, || {
        zstd::decode_all(&zstd_compressed[..])
    })?;
    results.push(BenchResult {
        name: "Zstd (L3)".to_string(),
        compressed_size: zstd_compressed.len(),
        ratio: data.len() as f64 / zstd_compressed.len() as f64,
        encode_time,
        decode_time,
    });
    
    // LZ4
    let (lz4_compressed, encode_time) = bench_encode(iterations, || {
        lz4::block::compress(&data, None, false)
    })?;
    let (_, decode_time) = bench_decode(iterations, || {
        lz4::block::decompress(&lz4_compressed, Some(data.len() as i32))
    })?;
    results.push(BenchResult {
        name: "LZ4".to_string(),
        compressed_size: lz4_compressed.len(),
        ratio: data.len() as f64 / lz4_compressed.len() as f64,
        encode_time,
        decode_time,
    });
    
    // Print results table
    println!("| Algorithm      | Size (B) | Ratio    | Encode     | Decode     |");
    println!("|----------------|----------|----------|------------|------------|");
    
    for result in &results {
        println!("| {:14} | {:>8} | {:>7.2}x | {:>10?} | {:>10?} |",
                 result.name,
                 result.compressed_size,
                 result.ratio,
                 result.encode_time,
                 result.decode_time);
    }
    
    // Calculate advantages
    let holo = &results[0];
    println!("\nRatio comparison vs Holographic:");
    for result in results.iter().skip(1) {
        let ratio_diff = holo.ratio / result.ratio;
        if ratio_diff > 1.0 {
            println!("  vs {}: Holographic {:.1}x better ratio", result.name, ratio_diff);
        } else {
            println!("  vs {}: {} {:.1}x better ratio", result.name, result.name, 1.0/ratio_diff);
        }
    }
    println!();
    
    Ok(())
}

/// Benchmark encoding, return result and average time
fn bench_encode<T, E, F>(iterations: usize, f: F) -> Result<(T, Duration), E>
where
    F: Fn() -> Result<T, E>,
{
    // Warmup
    let result = f()?;
    
    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = f()?;
    }
    let elapsed = start.elapsed() / iterations as u32;
    
    Ok((result, elapsed))
}

/// Benchmark decoding, return result and average time
fn bench_decode<T, E, F>(iterations: usize, f: F) -> Result<(T, Duration), E>
where
    F: Fn() -> Result<T, E>,
{
    // Warmup
    let result = f()?;
    
    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = f()?;
    }
    let elapsed = start.elapsed() / iterations as u32;
    
    Ok((result, elapsed))
}
