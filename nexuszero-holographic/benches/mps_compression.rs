use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use nexuszero_holographic::compression::encoder::encode_proof;

fn bench_mps_encode(c: &mut Criterion) {
    let sizes = [256usize, 1024, 4096, 16384];
    let max_bond = 8usize;
    for size in sizes {
        c.bench_with_input(BenchmarkId::new("mps_encode", size), &size, |b, &sz| {
            let data: Vec<u8> = (0..sz).map(|i| (i % 251) as u8).collect();
            b.iter(|| {
                let mps = encode_proof(&data, max_bond).expect("encode failed");
                criterion::black_box(mps.compression_ratio());
            });
        });
    }
}

criterion_group!(benches, bench_mps_encode);
criterion_main!(benches);