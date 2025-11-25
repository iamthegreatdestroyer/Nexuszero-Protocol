use criterion::{
    black_box, criterion_group, criterion_main,
    BenchmarkId, Criterion, Throughput,
};
use std::io::Write;

use nexuszero_holographic::MPS;

// External compressors for comparison
use bincode;
use zstd;
use brotli;
use lz4;

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

// ===== CORE BENCHMARKS =====
fn bench_compression_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_speed");
    // Reduced sizes for practical bench times (1KB, 4KB, 16KB)
    let sizes = vec![1024, 4 * 1024, 16 * 1024];

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        let data = generate_compressible_data(size);
        let bond_dim = calculate_optimal_bond_dim(size);

        // Measure MPS construction (roughly "encoding" time)
        group.bench_with_input(
            BenchmarkId::new("holographic_encode", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mps = MPS::from_proof_data(black_box(data), black_box(bond_dim)).unwrap();
                    black_box(mps);
                })
            },
        );

        // Measure serialization time (the "compressed bytes" generation)
        group.bench_with_input(
            BenchmarkId::new("holographic_serialize", size),
            &data,
            |b, data| {
                let mps = MPS::from_proof_data(data, bond_dim).unwrap();
                b.iter(|| {
                    let ser = bincode::serialize(black_box(&mps)).unwrap();
                    black_box(ser);
                })
            },
        );
    }
    group.finish();
}

fn bench_decompression_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompression_speed");
    // Reduced sizes for practical bench times (1KB, 4KB, 16KB)
    let sizes = vec![1024, 4 * 1024, 16 * 1024];

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        let data = generate_compressible_data(size);
        let bond_dim = calculate_optimal_bond_dim(size);
        let mps = MPS::from_proof_data(&data, bond_dim).unwrap();
        let ser = bincode::serialize(&mps).unwrap();

        // Measure deserialization time (roughly "decode" for MPS)
        group.bench_with_input(
            BenchmarkId::new("holographic_deserialize", size),
            &ser,
            |b, ser| {
                b.iter(|| {
                    let d: MPS = bincode::deserialize(black_box(ser)).unwrap();
                    black_box(d);
                })
            },
        );
    }
    group.finish();
}

fn bench_compression_ratio_by_size(_: &mut Criterion) {
    // This is not executed as a benchmark by Criterion, but useful for printed diagnostics
    println!("\n=== COMPRESSION RATIO MEASUREMENTS ===");
    // Reduced sizes for practical bench times (1KB, 4KB, 16KB)
    let sizes = vec![1024, 4 * 1024, 16 * 1024];
    for size in sizes {
        let data = generate_compressible_data(size);
        let bond_dim = calculate_optimal_bond_dim(size);
        let mps = MPS::from_proof_data(&data, bond_dim).unwrap();
        let serialized = bincode::serialize(&mps).unwrap();
        let ratio = data.len() as f64 / serialized.len() as f64;
        println!("{}: {:.2}x compression", format_size(size), ratio);
    }
}

fn bench_vs_zstd(c: &mut Criterion) {
    let mut group = c.benchmark_group("holographic_vs_zstd");
    let size = 16 * 1024; // reduced for practical timing
    let data = generate_compressible_data(size);

    // Holographic compress (encode + serialize)
    group.bench_function("holographic_compress", |b| {
        b.iter(|| {
            let mps = MPS::from_proof_data(black_box(&data), 16).unwrap();
            let _ser = bincode::serialize(black_box(&mps)).unwrap();
            black_box(_ser);
        })
    });

    // zstd compress
    group.bench_function("zstd_compress", |b| {
        b.iter(|| {
            let compressed = zstd::stream::encode_all(black_box(&data[..]), 3).unwrap();
            black_box(compressed)
        })
    });

    // Evaluate ratios and print
    let holo_mps = MPS::from_proof_data(&data, 16).unwrap();
    let holo_ser = bincode::serialize(&holo_mps).unwrap();
    let holo_ratio = data.len() as f64 / holo_ser.len() as f64;

    let zstd_compressed = zstd::stream::encode_all(&data[..], 3).unwrap();
    let zstd_ratio = data.len() as f64 / zstd_compressed.len() as f64;

    println!("\n=== 16KB ZSTD COMPARISON ===");
    println!("Holographic: {:.2}x", holo_ratio);
    println!("Zstd: {:.2}x", zstd_ratio);
    println!("Advantage: {:.2}x better\n", holo_ratio / zstd_ratio);

    group.finish();
}

fn bench_vs_brotli(c: &mut Criterion) {
    let mut group = c.benchmark_group("holographic_vs_brotli");
    let size = 16 * 1024; // reduced for practical timing
    let data = generate_compressible_data(size);

    // holographic
    group.bench_function("holographic_compress", |b| {
        b.iter(|| {
            let mps = MPS::from_proof_data(black_box(&data), 16).unwrap();
            let _ser = bincode::serialize(black_box(&mps)).unwrap();
            black_box(_ser);
        })
    });

    // brotli compress
    group.bench_function("brotli_compress", |b| {
        b.iter(|| {
            let mut dst: Vec<u8> = Vec::new();
            {
                let mut compressor = brotli::CompressorWriter::new(&mut dst, 4096, 5, 22);
                compressor.write_all(&data).unwrap();
            } // compressor dropped here
            black_box(dst)
        })
    });

    // Evaluate ratios and print
    let holo_mps = MPS::from_proof_data(&data, 16).unwrap();
    let holo_ser = bincode::serialize(&holo_mps).unwrap();
    let holo_ratio = data.len() as f64 / holo_ser.len() as f64;

    // brotli compress for ratio computation
    let mut dst: Vec<u8> = Vec::new();
    {
        let mut compressor = brotli::CompressorWriter::new(&mut dst, 4096, 5, 22);
        compressor.write_all(&data).unwrap();
    }
    let brotli_ratio = data.len() as f64 / dst.len() as f64;

    println!("\n=== BROTLI COMPARISON ===");
    println!("Holographic: {:.2}x", holo_ratio);
    println!("Brotli: {:.2}x", brotli_ratio);
    println!("Advantage: {:.2}x better\n", holo_ratio / brotli_ratio);

    group.finish();
}

fn bench_vs_lz4(c: &mut Criterion) {
    let mut group = c.benchmark_group("holographic_vs_lz4");
    let size = 16 * 1024; // reduced for practical timing
    let data = generate_compressible_data(size);

    // holographic
    group.bench_function("holographic_compress", |b| {
        b.iter(|| {
            let mps = MPS::from_proof_data(black_box(&data), 16).unwrap();
            let _ser = bincode::serialize(black_box(&mps)).unwrap();
            black_box(_ser);
        })
    });

    // lz4 compress
    group.bench_function("lz4_compress", |b| {
        b.iter(|| {
            let compressed = lz4::block::compress(black_box(&data), Some(lz4::block::CompressionMode::DEFAULT), false).unwrap();
            black_box(compressed);
        })
    });

    // Evaluate ratios and print
    let holo_mps = MPS::from_proof_data(&data, 16).unwrap();
    let holo_ser = bincode::serialize(&holo_mps).unwrap();
    let holo_ratio = data.len() as f64 / holo_ser.len() as f64;

    let lz4_compressed = lz4::block::compress(&data, Some(lz4::block::CompressionMode::DEFAULT), false).unwrap();
    let lz4_ratio = data.len() as f64 / lz4_compressed.len() as f64;

    println!("\n=== LZ4 COMPARISON ===");
    println!("Holographic: {:.2}x", holo_ratio);
    println!("LZ4: {:.2}x", lz4_ratio);
    println!("Advantage: {:.2}x better\n", holo_ratio / lz4_ratio);

    group.finish();
}

fn bench_bond_dimension_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("bond_dimension_sweep");
    let size = 4 * 1024; // reduced for practical timing
    let data = generate_compressible_data(size);
    let bond_dims = vec![2usize, 4, 8, 16];

    for &bd in &bond_dims {
        group.bench_with_input(BenchmarkId::new("create_mps", bd), &bd, |b, &bd| {
            b.iter(|| {
                let mps = MPS::from_proof_data(black_box(&data), black_box(bd)).unwrap();
                black_box(mps);
            })
        });

        // capture ratio
        let mps = MPS::from_proof_data(&data, bd).unwrap();
        let ratio = data.len() as f64 / bincode::serialize(&mps).unwrap().len() as f64;
        println!("bond_dim={}: {:.2}x", bd, ratio);
    }

    group.finish();
}

fn bench_memory_usage(_: &mut Criterion) {
    println!("\n=== APPROXIMATE MEMORY USAGE (bytes) ===");
    // Reduced sizes for practical bench times
    let sizes = vec![1024, 4 * 1024, 16 * 1024];
    for size in sizes {
        let data = generate_compressible_data(size);
        let mps = MPS::from_proof_data(&data, calculate_optimal_bond_dim(size)).unwrap();
        println!("{}: approx {} bytes", format_size(size), mps.approx_serialized_size());
    }
}

// Utility for formatting sizes
fn format_size(bytes: usize) -> String {
    if bytes < 1024 { return format!("{}B", bytes); }
    if bytes < 1024 * 1024 { return format!("{}KB", bytes / 1024); }
    format!("{}MB", bytes / (1024 * 1024))
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
