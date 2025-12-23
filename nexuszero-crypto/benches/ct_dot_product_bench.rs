//! Micro-benchmarks comparing O(n²) vs O(n) constant-time dot product
//!
//! Run with: cargo bench --bench ct_dot_product_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// Import both implementations
// Original O(n²) implementation
fn ct_array_access_original(array: &[i64], index: usize) -> i64 {
    let mut result = 0i64;
    for (i, val) in array.iter().enumerate() {
        // Constant-time conditional: select val if i == index, else keep result
        let mask = ((i == index) as i64).wrapping_neg();
        result = (result & !mask) | (*val & mask);
    }
    result
}

fn ct_dot_product_original(a: &[i64], b: &[i64]) -> i64 {
    assert_eq!(a.len(), b.len());
    let mut result = 0i64;
    for i in 0..a.len() {
        // O(n) per element access → O(n²) total
        let a_val = ct_array_access_original(a, i);
        let b_val = ct_array_access_original(b, i);
        result = result.wrapping_add(a_val.wrapping_mul(b_val));
    }
    result
}

// Optimized O(n) implementation
#[inline(never)]
fn ct_dot_product_fast(a: &[i64], b: &[i64]) -> i64 {
    assert_eq!(a.len(), b.len());
    let mut result = 0i64;
    for (a_val, b_val) in a.iter().zip(b.iter()) {
        result = result.wrapping_add(a_val.wrapping_mul(*b_val));
    }
    result
}

fn bench_dot_products(c: &mut Criterion) {
    let mut group = c.benchmark_group("ct_dot_product");
    
    // Test sizes matching LWE security parameters
    // 128-bit: n=256, 192-bit: n=384, 256-bit: n=512
    for size in [32, 64, 128, 256, 384, 512] {
        let a: Vec<i64> = (0..size).map(|i| (i as i64) % 1000 - 500).collect();
        let b: Vec<i64> = (0..size).map(|i| ((i * 7) as i64) % 1000 - 500).collect();
        
        // Verify both produce same result
        let original_result = ct_dot_product_original(&a, &b);
        let fast_result = ct_dot_product_fast(&a, &b);
        assert_eq!(original_result, fast_result, "Mismatch at size {}", size);
        
        group.bench_with_input(
            BenchmarkId::new("O(n²)_original", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| ct_dot_product_original(black_box(a), black_box(b)))
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("O(n)_optimized", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| ct_dot_product_fast(black_box(a), black_box(b)))
            },
        );
    }
    
    group.finish();
}

fn bench_array_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_access_pattern");
    
    // Demonstrate the O(n) vs O(1) access pattern difference
    let size = 256; // LWE 128-bit security parameter
    let array: Vec<i64> = (0..size).map(|i| i as i64).collect();
    
    group.bench_function("sequential_direct", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for i in 0..size {
                sum = sum.wrapping_add(black_box(array[i]));
            }
            sum
        })
    });
    
    group.bench_function("sequential_ct_access", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for i in 0..size {
                sum = sum.wrapping_add(ct_array_access_original(black_box(&array), i));
            }
            sum
        })
    });
    
    group.finish();
}

fn bench_lwe_decrypt_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("lwe_decrypt_simulation");
    
    // Simulate LWE decrypt hot path: dot(sk, ct) mod q
    // Security levels: 128-bit (n=256), 192-bit (n=384), 256-bit (n=512)
    let sizes = [
        (256, "128-bit"),
        (384, "192-bit"),
        (512, "256-bit"),
    ];
    
    for (size, security) in sizes {
        let secret_key: Vec<i64> = (0..size).map(|i| (i as i64) % 3 - 1).collect(); // {-1, 0, 1}
        let ciphertext: Vec<i64> = (0..size).map(|i| (i as i64 * 17) % 1000).collect();
        
        group.bench_with_input(
            BenchmarkId::new("original", security),
            &(&secret_key, &ciphertext),
            |bench, (sk, ct)| {
                bench.iter(|| ct_dot_product_original(black_box(sk), black_box(ct)))
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("optimized", security),
            &(&secret_key, &ciphertext),
            |bench, (sk, ct)| {
                bench.iter(|| ct_dot_product_fast(black_box(sk), black_box(ct)))
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_dot_products,
    bench_array_access_patterns,
    bench_lwe_decrypt_simulation,
);

criterion_main!(benches);
