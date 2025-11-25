//! Comprehensive Compression Benchmark Suite
//!
//! This benchmark compares NexusZero holographic compression against standard algorithms:
//! - Tensor Train (MPS v2) with multi-precision quantization
//! - LZ4 (fast, moderate compression)
//! - Zstd (balanced speed/compression)
//! - Brotli (high compression, slower)
//!
//! Run: `cargo bench --bench compression_bench`
//! Results: `target/criterion/report/index.html`

use criterion::{
    black_box, criterion_group, criterion_main,
    BenchmarkId, Criterion, Throughput,
};
use std::io::Write;

// Import v2 tensor train implementation (the REAL compression)
use nexuszero_holographic::{
    CompressedTensorTrain, CompressionConfig, StoragePrecision,
    analyze_compression_potential, CompressionRecommendation,
};

// External compressors for comparison
use zstd;
use brotli;
use lz4;

// ============================================================================
// DATA GENERATORS
// ============================================================================

/// Generate structured data with repeating patterns (compressible)
fn generate_compressible_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| ((i / 16) % 256) as u8).collect()
}

/// Generate ZK proof-like structured data (highly compressible)
fn generate_zk_proof_data(size: usize) -> Vec<u8> {
    // Simulate ZK proof structure with headers, field elements, commitments
    let mut data = Vec::with_capacity(size);
    
    // Header pattern (16 bytes repeated)
    let header = [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03,
                  0xCA, 0xFE, 0xBA, 0xBE, 0x10, 0x20, 0x30, 0x40];
    
    while data.len() < size {
        // Add header
        data.extend_from_slice(&header[..header.len().min(size - data.len())]);
        if data.len() >= size { break; }
        
        // Add "field elements" (structured 32-byte chunks)
        for j in 0..8 {
            if data.len() >= size { break; }
            let field_elem: Vec<u8> = (0..32).map(|k| ((j * 32 + k) % 256) as u8).collect();
            data.extend_from_slice(&field_elem[..field_elem.len().min(size - data.len())]);
        }
    }
    
    data.truncate(size);
    data
}

/// Generate random data (incompressible baseline)
fn generate_random_data(size: usize) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    (0..size).map(|i| {
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        (hasher.finish() % 256) as u8
    }).collect()
}

// ============================================================================
// V2 TENSOR TRAIN BENCHMARKS (THE REAL COMPRESSION)
// ============================================================================

fn bench_v2_compression_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("v2_compression_speed");
    let sizes = vec![1024, 10 * 1024, 100 * 1024];

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        let data = generate_zk_proof_data(size);
        
        // Default config
        group.bench_with_input(
            BenchmarkId::new("tensor_train_default", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let compressed = CompressedTensorTrain::compress(
                        black_box(data),
                        CompressionConfig::default()
                    ).unwrap();
                    black_box(compressed);
                });
            },
        );
        
        // High compression config
        group.bench_with_input(
            BenchmarkId::new("tensor_train_high", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let compressed = CompressedTensorTrain::compress(
                        black_box(data),
                        CompressionConfig::high_compression()
                    ).unwrap();
                    black_box(compressed);
                });
            },
        );
        
        // Fast config
        group.bench_with_input(
            BenchmarkId::new("tensor_train_fast", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let compressed = CompressedTensorTrain::compress(
                        black_box(data),
                        CompressionConfig::fast()
                    ).unwrap();
                    black_box(compressed);
                });
            },
        );
    }

    group.finish();
}

fn bench_v2_decompression_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("v2_decompression_speed");
    let sizes = vec![1024, 10 * 1024, 100 * 1024];

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        let data = generate_zk_proof_data(size);
        let compressed = CompressedTensorTrain::compress(&data, CompressionConfig::default()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("tensor_train_decompress", size),
            &compressed,
            |b, compressed| {
                b.iter(|| {
                    let decompressed = compressed.decompress().unwrap();
                    black_box(decompressed);
                });
            },
        );
    }

    group.finish();
}

fn bench_v2_compression_ratio(_: &mut Criterion) {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║          V2 TENSOR TRAIN COMPRESSION RATIO MEASUREMENTS              ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    
    let sizes = vec![1024, 10 * 1024, 100 * 1024];
    let configs = vec![
        ("default", CompressionConfig::default()),
        ("high_compression", CompressionConfig::high_compression()),
        ("fast", CompressionConfig::fast()),
        ("balanced", CompressionConfig::balanced()),
    ];
    
    println!("║ {:^68} ║", "ZK Proof-like Data (Structured)");
    println!("╟──────────────────────────────────────────────────────────────────────╢");
    
    for size in &sizes {
        let data = generate_zk_proof_data(*size);
        println!("║ Size: {:>6} bytes                                                   ║", size);
        
        for (name, config) in &configs {
            let compressed = CompressedTensorTrain::compress(&data, config.clone()).unwrap();
            let stats = compressed.stats();
            let ratio = stats.compression_ratio();
            let compressed_size = stats.compressed_bytes;
            
            println!("║   {:20}: {:>6}B → {:>6}B  ratio: {:>6.2}x              ║", 
                name, size, compressed_size, ratio);
        }
        println!("╟──────────────────────────────────────────────────────────────────────╢");
    }
    
    println!("║ {:^68} ║", "Random Data (Baseline - Not Compressible)");
    println!("╟──────────────────────────────────────────────────────────────────────╢");
    
    for size in &[1024usize, 10 * 1024] {
        let data = generate_random_data(*size);
        let compressed = CompressedTensorTrain::compress(&data, CompressionConfig::default()).unwrap();
        let stats = compressed.stats();
        println!("║   {:>6}B random: {:>6}B → {:>6}B  ratio: {:>6.2}x                    ║",
            size, size, stats.compressed_bytes, stats.compression_ratio());
    }
    
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn bench_v2_vs_zstd(c: &mut Criterion) {
    let mut group = c.benchmark_group("v2_vs_zstd");
    let size = 100 * 1024;
    let data = generate_zk_proof_data(size);

    // V2 Tensor Train compress
    group.bench_function("tensor_train_compress", |b| {
        b.iter(|| {
            let compressed = CompressedTensorTrain::compress(
                black_box(&data),
                CompressionConfig::high_compression()
            ).unwrap();
            black_box(compressed);
        });
    });

    // Zstd compress (level 3 = fast)
    group.bench_function("zstd_level3", |b| {
        b.iter(|| {
            let compressed = zstd::stream::encode_all(black_box(&data[..]), 3).unwrap();
            black_box(compressed);
        });
    });

    // Zstd compress (level 9 = high compression)
    group.bench_function("zstd_level9", |b| {
        b.iter(|| {
            let compressed = zstd::stream::encode_all(black_box(&data[..]), 9).unwrap();
            black_box(compressed);
        });
    });

    group.finish();

    // Print comparison
    let tt_compressed = CompressedTensorTrain::compress(&data, CompressionConfig::high_compression()).unwrap();
    let tt_bytes = tt_compressed.to_bytes().unwrap();
    let tt_ratio = size as f64 / tt_bytes.len() as f64;
    
    let zstd3 = zstd::stream::encode_all(&data[..], 3).unwrap();
    let zstd9 = zstd::stream::encode_all(&data[..], 9).unwrap();
    let zstd3_ratio = size as f64 / zstd3.len() as f64;
    let zstd9_ratio = size as f64 / zstd9.len() as f64;

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                    100KB ZK DATA: ZSTD COMPARISON                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Tensor Train (high):  {:>6} bytes  ratio: {:>6.2}x                    ║", tt_bytes.len(), tt_ratio);
    println!("║ Zstd level 3:         {:>6} bytes  ratio: {:>6.2}x                    ║", zstd3.len(), zstd3_ratio);
    println!("║ Zstd level 9:         {:>6} bytes  ratio: {:>6.2}x                    ║", zstd9.len(), zstd9_ratio);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn bench_v2_vs_brotli(c: &mut Criterion) {
    let mut group = c.benchmark_group("v2_vs_brotli");
    let size = 100 * 1024;
    let data = generate_zk_proof_data(size);

    // V2 Tensor Train compress
    group.bench_function("tensor_train_compress", |b| {
        b.iter(|| {
            let compressed = CompressedTensorTrain::compress(
                black_box(&data),
                CompressionConfig::high_compression()
            ).unwrap();
            black_box(compressed);
        });
    });

    // Brotli compress
    group.bench_function("brotli_quality5", |b| {
        b.iter(|| {
            let mut dst: Vec<u8> = Vec::new();
            {
                let mut compressor = brotli::CompressorWriter::new(&mut dst, 4096, 5, 22);
                compressor.write_all(&data).unwrap();
            }
            black_box(dst);
        });
    });

    group.finish();

    // Print comparison
    let tt_compressed = CompressedTensorTrain::compress(&data, CompressionConfig::high_compression()).unwrap();
    let tt_bytes = tt_compressed.to_bytes().unwrap();
    let tt_ratio = size as f64 / tt_bytes.len() as f64;
    
    let mut brotli_dst: Vec<u8> = Vec::new();
    {
        let mut compressor = brotli::CompressorWriter::new(&mut brotli_dst, 4096, 5, 22);
        compressor.write_all(&data).unwrap();
    }
    let brotli_ratio = size as f64 / brotli_dst.len() as f64;

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                   100KB ZK DATA: BROTLI COMPARISON                   ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Tensor Train (high):  {:>6} bytes  ratio: {:>6.2}x                    ║", tt_bytes.len(), tt_ratio);
    println!("║ Brotli quality 5:     {:>6} bytes  ratio: {:>6.2}x                    ║", brotli_dst.len(), brotli_ratio);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn bench_v2_vs_lz4(c: &mut Criterion) {
    let mut group = c.benchmark_group("v2_vs_lz4");
    let size = 100 * 1024;
    let data = generate_zk_proof_data(size);

    // V2 Tensor Train compress
    group.bench_function("tensor_train_compress", |b| {
        b.iter(|| {
            let compressed = CompressedTensorTrain::compress(
                black_box(&data),
                CompressionConfig::high_compression()
            ).unwrap();
            black_box(compressed);
        });
    });

    // LZ4 compress
    group.bench_function("lz4_default", |b| {
        b.iter(|| {
            let compressed = lz4::block::compress(
                black_box(&data),
                Some(lz4::block::CompressionMode::DEFAULT),
                false
            ).unwrap();
            black_box(compressed);
        });
    });

    group.finish();

    // Print comparison
    let tt_compressed = CompressedTensorTrain::compress(&data, CompressionConfig::high_compression()).unwrap();
    let tt_bytes = tt_compressed.to_bytes().unwrap();
    let tt_ratio = size as f64 / tt_bytes.len() as f64;
    
    let lz4_compressed = lz4::block::compress(&data, Some(lz4::block::CompressionMode::DEFAULT), false).unwrap();
    let lz4_ratio = size as f64 / lz4_compressed.len() as f64;

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                    100KB ZK DATA: LZ4 COMPARISON                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Tensor Train (high):  {:>6} bytes  ratio: {:>6.2}x                    ║", tt_bytes.len(), tt_ratio);
    println!("║ LZ4 default:          {:>6} bytes  ratio: {:>6.2}x                    ║", lz4_compressed.len(), lz4_ratio);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn bench_precision_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_sweep");
    let size = 10 * 1024;
    let data = generate_zk_proof_data(size);
    
    let precisions = vec![
        ("F64", StoragePrecision::F64),
        ("F32", StoragePrecision::F32),
        ("F16", StoragePrecision::F16),
        ("I8", StoragePrecision::I8),
    ];

    for (name, precision) in &precisions {
        let config = CompressionConfig {
            precision: *precision,
            ..CompressionConfig::default()
        };
        
        group.bench_with_input(
            BenchmarkId::new("compress", *name),
            &data,
            |b, data| {
                b.iter(|| {
                    let compressed = CompressedTensorTrain::compress(
                        black_box(data),
                        config.clone()
                    ).unwrap();
                    black_box(compressed);
                });
            },
        );
    }

    group.finish();

    // Print precision comparison
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                    10KB: PRECISION LEVEL COMPARISON                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    
    for (name, precision) in &precisions {
        let config = CompressionConfig {
            precision: *precision,
            ..CompressionConfig::default()
        };
        let compressed = CompressedTensorTrain::compress(&data, config).unwrap();
        let stats = compressed.stats();
        
        // Test roundtrip
        let decompressed = compressed.decompress().unwrap();
        let error: f64 = data.iter()
            .zip(decompressed.iter())
            .map(|(&a, &b)| ((a as f64) - (b as f64)).abs())
            .sum::<f64>() / data.len() as f64;
        
        println!("║ {:4}: {:>6}B → {:>6}B  ratio: {:>5.2}x  avg_error: {:>6.3}          ║",
            name, size, stats.compressed_bytes, stats.compression_ratio(), error);
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn bench_bond_dimension_sweep_v2(c: &mut Criterion) {
    let mut group = c.benchmark_group("v2_bond_dimension_sweep");
    let size = 10 * 1024;
    let data = generate_zk_proof_data(size);
    let bond_dims = vec![4usize, 8, 16, 32, 64];

    for bd in &bond_dims {
        let config = CompressionConfig {
            max_bond_dim: *bd,
            ..CompressionConfig::default()
        };
        
        group.bench_with_input(
            BenchmarkId::new("tensor_train", *bd),
            &data,
            |b, data| {
                b.iter(|| {
                    let compressed = CompressedTensorTrain::compress(
                        black_box(data),
                        config.clone()
                    ).unwrap();
                    black_box(compressed);
                });
            },
        );
    }

    group.finish();

    // Print bond dimension comparison
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                  10KB: BOND DIMENSION COMPARISON                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    
    for bd in &bond_dims {
        let config = CompressionConfig {
            max_bond_dim: *bd,
            ..CompressionConfig::default()
        };
        let compressed = CompressedTensorTrain::compress(&data, config).unwrap();
        let stats = compressed.stats();
        
        println!("║ bond_dim={:>2}: {:>6}B → {:>6}B  ratio: {:>5.2}x  max_used: {:>2}         ║",
            bd, size, stats.compressed_bytes, stats.compression_ratio(), stats.max_bond_dim);
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn bench_entropy_analysis(_: &mut Criterion) {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                      ENTROPY ANALYSIS RESULTS                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    
    let test_cases = vec![
        ("ZK Proof Data (structured)", generate_zk_proof_data(10240)),
        ("Repeating Patterns", generate_compressible_data(10240)),
        ("Random Data", generate_random_data(10240)),
        ("All Zeros", vec![0u8; 10240]),
        ("Alternating", (0..10240).map(|i| if i % 2 == 0 { 0xAA } else { 0x55 }).collect()),
    ];
    
    for (name, data) in &test_cases {
        let analysis = analyze_compression_potential(data);
        let recommendation = match analysis.recommendation {
            CompressionRecommendation::TensorTrain => "TensorTrain",
            CompressionRecommendation::Hybrid => "Hybrid",
            CompressionRecommendation::StandardOnly => "StandardOnly",
        };
        
        println!("║ {:25} entropy: {:>5.2} bits  est_ratio: {:>5.2}x  rec: {:12} ║",
            name, analysis.entropy, analysis.estimated_ratio, recommendation);
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn bench_memory_usage_v2(_: &mut Criterion) {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                      MEMORY USAGE ESTIMATES                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    
    let sizes = vec![1024, 10 * 1024, 100 * 1024];
    
    for size in sizes {
        let data = generate_zk_proof_data(size);
        
        // Measure compressed size with different configs
        let default_compressed = CompressedTensorTrain::compress(&data, CompressionConfig::default()).unwrap();
        let high_compressed = CompressedTensorTrain::compress(&data, CompressionConfig::high_compression()).unwrap();
        
        println!("║ {:>6}B input:                                                       ║", size);
        println!("║   default config:      {:>8}B compressed  ({:>5.1}% of original)     ║", 
            default_compressed.stats().compressed_bytes,
            100.0 * default_compressed.stats().compressed_bytes as f64 / size as f64);
        println!("║   high_compression:    {:>8}B compressed  ({:>5.1}% of original)     ║",
            high_compressed.stats().compressed_bytes,
            100.0 * high_compressed.stats().compressed_bytes as f64 / size as f64);
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

// Utility for formatting sizes
#[allow(dead_code)]
fn format_size(bytes: usize) -> String {
    if bytes < 1024 { return format!("{}B", bytes); }
    if bytes < 1024 * 1024 { return format!("{}KB", bytes / 1024); }
    format!("{}MB", bytes / (1024 * 1024))
}

criterion_group!(
    benches,
    bench_v2_compression_by_size,
    bench_v2_decompression_by_size,
    bench_v2_compression_ratio,
    bench_v2_vs_zstd,
    bench_v2_vs_brotli,
    bench_v2_vs_lz4,
    bench_precision_sweep,
    bench_bond_dimension_sweep_v2,
    bench_entropy_analysis,
    bench_memory_usage_v2
);
criterion_main!(benches);
