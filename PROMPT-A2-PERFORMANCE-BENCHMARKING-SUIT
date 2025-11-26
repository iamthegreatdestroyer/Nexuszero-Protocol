Agent: Quinn Quality + Morgan Rustico
File: nexuszero-holographic/benches/compression_bench.rs
Time Estimate: 2 hours
Current State: Benchmark file doesn't exist
Target State: Comprehensive benchmark suite with comparison data
CONTEXT
I am implementing the performance benchmarking suite for NexusZero holographic compression. I need to create a comprehensive benchmark that verifies our 1000x-100000x compression claims and compares against standard algorithms.
Why this matters:

Must prove "holographic advantage" over standard compression
Week 3.4 integration requires baseline performance data
Marketing claims need empirical backing

New file to create: nexuszero-holographic/benches/compression_bench.rs
YOUR TASK
Create comprehensive benchmark suite with:

Compression Speed by Size - Measure encoding time for 1KB, 10KB, 100KB, 1MB
Decompression Speed - Measure decoding time
Compression Ratio Measurement - Document actual ratios achieved
Comparison vs Zstd - Direct comparison
Comparison vs Brotli - Direct comparison
Comparison vs LZ4 - Direct comparison
Bond Dimension Tuning - Find optimal bond dimensions
Memory Usage Profiling - Peak memory consumption

DEPENDENCIES TO ADD
Add to nexuszero-holographic/Cargo.toml under [dev-dependencies]:
tomlcriterion = { version = "0.5", features = ["html_reports"] }
zstd = "0.13"
brotli = "3.4"
lz4 = "1.24"
CODE STRUCTURE
rustuse criterion::{
    black_box, criterion_group, criterion_main, 
    BenchmarkId, Criterion, Throughput
};
use nexuszero_holographic::MPS;

// === CORE BENCHMARKS ===
fn bench_compression_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_speed");
    
    let sizes = vec![1024, 10*1024, 100*1024, 1024*1024];
    
    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        let data = generate_compressible_data(size);
        let bond_dim = calculate_optimal_bond_dim(size);
        
        group.bench_with_input(
            BenchmarkId::new("holographic", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mps = MPS::from_proof_data(
                        black_box(data), 
                        black_box(bond_dim)
                    ).unwrap();
                    black_box(mps);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_decompression_by_size(c: &mut Criterion) {
    // YOUR CODE: Similar structure for decompression
}

fn bench_compression_ratio_by_size(c: &mut Criterion) {
    // YOUR CODE: Measure and print ratios
    println!("\n=== COMPRESSION RATIO MEASUREMENTS ===");
    // For each size: create MPS, calculate ratio, print
    println!("1KB: {:.2}x compression", ratio);
    // ...
}

fn bench_vs_zstd(c: &mut Criterion) {
    let mut group = c.benchmark_group("holographic_vs_zstd");
    let size = 100 * 1024;
    let data = generate_compressible_data(size);
    
    // Benchmark holographic
    group.bench_function("holographic_compress", |b| {
        b.iter(|| {
            let mps = MPS::from_proof_data(black_box(&data), 16).unwrap();
            black_box(mps);
        });
    });
    
    // Benchmark zstd
    group.bench_function("zstd_compress", |b| {
        b.iter(|| {
            let compressed = zstd::encode_all(black_box(&data[..]), 3).unwrap();
            black_box(compressed);
        });
    });
    
    // Print comparison
    let holo_mps = MPS::from_proof_data(&data, 16).unwrap();
    let holo_ratio = holo_mps.compression_ratio();
    let zstd_compressed = zstd::encode_all(&data[..], 3).unwrap();
    let zstd_ratio = data.len() as f64 / zstd_compressed.len() as f64;
    
    println!("\n=== COMPARISON ===");
    println!("Holographic: {:.2}x", holo_ratio);
    println!("Zstd: {:.2}x", zstd_ratio);
    println!("Advantage: {:.2}x better\n", holo_ratio / zstd_ratio);
    
    group.finish();
}

fn bench_vs_brotli(c: &mut Criterion) {
    // YOUR CODE: Similar to zstd comparison
}

fn bench_vs_lz4(c: &mut Criterion) {
    // YOUR CODE: Similar to zstd comparison
}

fn bench_bond_dimension_sweep(c: &mut Criterion) {
    // YOUR CODE: Test bond dims 2, 4, 8, 16, 32, 64
    // Print ratio for each
}

fn bench_memory_usage(c: &mut Criterion) {
    // YOUR CODE: Estimate memory usage for different sizes
}

// === HELPER FUNCTIONS ===
fn generate_compressible_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| ((i / 16) % 256) as u8).collect()
}

fn calculate_optimal_bond_dim(size: usize) -> usize {
    match size {
        0..=1024 => 4,
        1025..=10240 => 8,
        10241..=102400 => 16,
        102401..=1048576 => 32,
        _ => 64,
    }
}

criterion_group!(
    benches,
    bench_compression_by_size,
    bench_decompression_by_size,
    bench_compression_ratio_by_size,
    bench_vs_zstd,
    bench_vs_brotli,
    bench_vs_lz4,
    bench_bond_dimension_sweep,
    bench_memory_usage
);

criterion_main!(benches);
VERIFICATION COMMANDS
bashcd nexuszero-holographic
cargo bench --bench compression_bench
SUCCESS CRITERIA

 All benchmarks execute successfully
 Compression ratios verified:

1KB → 10-15x
10KB → 100-150x
100KB → 1000-1500x
1MB → 10000-15000x


 Holographic proves 50-150x better than standard algorithms
 Performance acceptable (<500ms for 100KB encoding)
 HTML report generated at target/criterion/report/index.html

EXPECTED OUTPUT
=== COMPRESSION RATIO MEASUREMENTS ===
1KB: 12.50x compression
10KB: 125.00x compression
100KB: 1250.00x compression
1MB: 12500.00x compression

=== 100KB COMPARISON ===
Holographic: 1250.00x
Zstd: 12.50x
Advantage: 100.00x better

=== BROTLI COMPARISON ===
Holographic: 1250.00x
Brotli: 15.00x
Advantage: 83.33x better

=== LZ4 COMPARISON ===
Holographic: 1250.00x
LZ4: 8.00x
Advantage: 156.25x better

Benchmarking completed. HTML report: target/criterion/report/index.html
NOW GENERATE THE COMPLETE BENCHMARK SUITE IMPLEMENTATION.
