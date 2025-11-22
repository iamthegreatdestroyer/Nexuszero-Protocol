use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use nexuszero_crypto::lattice::{ring_lwe::{Polynomial, poly_mult_schoolbook, poly_mult_ntt}};
use rand::Rng;

fn random_poly(n: usize, q: u64) -> Polynomial { Polynomial::from_coeffs((0..n).map(|_| rand::thread_rng().gen_range(0..q as i64)).collect(), q) }

fn bench_ntt_vs_schoolbook(c: &mut Criterion) {
    let mut group = c.benchmark_group("poly_mult");
    let sizes = [256usize, 512, 1024];
    let q = 12289u64; // uses known primitive root for 512; others may fall back
    std::env::set_var("NEXUSZERO_USE_NTT", "1");

    for &n in &sizes {
        let a = random_poly(n, q);
        let b = random_poly(n, q);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("schoolbook", n), &n, |bencher, &_n| {
            bencher.iter(|| { poly_mult_schoolbook(&a, &b, q) });
        });
        group.bench_with_input(BenchmarkId::new("ntt", n), &n, |bencher, &_n| {
            bencher.iter(|| { poly_mult_ntt(&a, &b, q) });
        });
    }
    group.finish();
}

criterion_group!(name=ntt_benches; config=Criterion::default(); targets=bench_ntt_vs_schoolbook);
criterion_main!(ntt_benches);
