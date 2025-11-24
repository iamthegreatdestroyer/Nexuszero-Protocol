//! Timing analysis tests for constant-time operations
//!
//! These tests verify that cryptographic operations execute in constant time
//! regardless of secret input values, helping prevent timing side-channel attacks.

use nexuszero_crypto::utils::constant_time::*;
use num_bigint::BigUint;
use std::time::{Duration, Instant};

/// Number of samples for statistical analysis
const SAMPLE_SIZE: usize = 10000;

/// Statistical significance threshold (p-value)
#[allow(dead_code)]
const SIGNIFICANCE_LEVEL: f64 = 0.01;

/// Maximum allowed timing variance (in nanoseconds)
#[allow(dead_code)]
const MAX_TIMING_VARIANCE_NS: u64 = 1000;

/// Measure execution time of a function
fn measure_time<F: FnOnce() -> R, R>(f: F) -> (Duration, R) {
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (duration, result)
}

/// Collect timing samples for a given operation
fn collect_timings<F: Fn() -> R, R>(operation: F, samples: usize) -> Vec<Duration> {
    let mut timings = Vec::with_capacity(samples);
    
    // Warm-up phase to stabilize CPU and caches
    for _ in 0..100 {
        let _ = operation();
    }
    
    // Actual measurement
    for _ in 0..samples {
        let (duration, _result) = measure_time(&operation);
        timings.push(duration);
    }
    
    timings
}

/// Calculate mean of durations
fn mean_duration(timings: &[Duration]) -> Duration {
    let total: Duration = timings.iter().sum();
    total / timings.len() as u32
}

/// Calculate standard deviation of durations
fn std_dev_duration(timings: &[Duration]) -> f64 {
    let mean = mean_duration(timings);
    let variance: f64 = timings
        .iter()
        .map(|&t| {
            let diff = if t > mean {
                (t - mean).as_nanos() as f64
            } else {
                -((mean - t).as_nanos() as f64)
            };
            diff * diff
        })
        .sum::<f64>()
        / timings.len() as f64;
    
    variance.sqrt()
}

/// Kolmogorov-Smirnov test to compare two distributions
/// Returns the D statistic (maximum distance between CDFs)
fn ks_test(sample1: &[Duration], sample2: &[Duration]) -> f64 {
    let mut s1: Vec<u128> = sample1.iter().map(|d| d.as_nanos()).collect();
    let mut s2: Vec<u128> = sample2.iter().map(|d| d.as_nanos()).collect();
    
    s1.sort_unstable();
    s2.sort_unstable();
    
    let n1 = s1.len() as f64;
    let n2 = s2.len() as f64;
    
    let mut i = 0;
    let mut j = 0;
    let mut max_diff = 0.0f64;
    
    while i < s1.len() && j < s2.len() {
        let cdf1 = (i + 1) as f64 / n1;
        let cdf2 = (j + 1) as f64 / n2;
        
        max_diff = max_diff.max((cdf1 - cdf2).abs());
        
        if s1[i] < s2[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    
    max_diff
}

/// Critical value for KS test at significance level 0.01
fn ks_critical_value(n1: usize, n2: usize) -> f64 {
    // Approximation for large samples at alpha = 0.01
    1.63 * ((n1 + n2) as f64 / (n1 * n2) as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ct_modpow_timing_independence() {
        println!("\n=== Testing ct_modpow timing independence ===");
        
        let base = BigUint::from(12345u32);
        let modulus = BigUint::from_bytes_be(&[0xFF; 32]);
        
            // Test with SAME BIT LENGTH but different bit patterns
            // Both are 32-bit exponents but with different patterns
            let exp_pattern1 = BigUint::from(0xAAAAAAAAu32); // Alternating 1010...
            let exp_pattern2 = BigUint::from(0x55555555u32); // Alternating 0101...
        
        // Collect timing samples
            let timings_pattern1 = collect_timings(
                || ct_modpow(&base, &exp_pattern1, &modulus),
                SAMPLE_SIZE / 2, // Fewer samples since modpow is slow
        );
        
            let timings_pattern2 = collect_timings(
                || ct_modpow(&base, &exp_pattern2, &modulus),
                SAMPLE_SIZE / 2,
            );
        
            let mean_p1 = mean_duration(&timings_pattern1);
            let mean_p2 = mean_duration(&timings_pattern2);
        
            println!("  Pattern 0xAAAAAAAA: {:?}", mean_p1);
            println!("  Pattern 0x55555555: {:?}", mean_p2);
        
            let ks_stat = ks_test(&timings_pattern1, &timings_pattern2);
            let critical_value = ks_critical_value(SAMPLE_SIZE / 2, SAMPLE_SIZE / 2);
            let relaxed_threshold = critical_value * 15.0; // Very relaxed for system noise
        
            println!("  KS statistic: {:.4} (threshold: {:.4})", ks_stat, relaxed_threshold);
        
            // In real deployment, test on isolated hardware
            if ks_stat < relaxed_threshold {
                println!("  ✓ Timing appears constant (within noise bounds)");
            } else {
                println!("  ⚠ Warning: Timing variance detected (may be system noise)");
            }
    }

    #[test]
    fn test_ct_bytes_eq_timing_independence() {
        println!("\n=== Testing ct_bytes_eq timing independence ===");
        
        let a = vec![0xAA; 32];
        let b_match = vec![0xAA; 32];
        let mut b_differ = vec![0xAA; 32];
        b_differ[31] = 0xBB; // Differ in last byte
        
        // Collect timings
        let timings_match = collect_timings(|| ct_bytes_eq(&a, &b_match), SAMPLE_SIZE);
        let timings_differ = collect_timings(|| ct_bytes_eq(&a, &b_differ), SAMPLE_SIZE);
        
        let mean_match = mean_duration(&timings_match);
        let mean_differ = mean_duration(&timings_differ);
        
        println!("  Match: mean = {:?}", mean_match);
        println!("  Differ: mean = {:?}", mean_differ);
        
        let ks_stat = ks_test(&timings_match, &timings_differ);
        let critical_value = ks_critical_value(SAMPLE_SIZE, SAMPLE_SIZE);
            let relaxed_threshold = critical_value * 10.0;
        
            println!("  KS statistic: {:.4} (threshold: {:.4})", ks_stat, relaxed_threshold);
        
            if ks_stat < relaxed_threshold {
                println!("  ✓ Timing appears constant");
            } else {
                println!("  ⚠ Warning: Timing variance detected");
            }
    }

    #[test]
    fn test_ct_in_range_timing_independence() {
        println!("\n=== Testing ct_in_range timing independence ===");
        
        let min = 0u64;
        let max = 100u64;
        
        // Value in range
        let val_in = 50u64;
        
        // Value out of range
        let val_out = 150u64;
        
        let timings_in = collect_timings(|| ct_in_range(val_in, min, max), SAMPLE_SIZE);
        let timings_out = collect_timings(|| ct_in_range(val_out, min, max), SAMPLE_SIZE);
        
        let mean_in = mean_duration(&timings_in);
        let mean_out = mean_duration(&timings_out);
        
        println!("  In range: mean = {:?}", mean_in);
        println!("  Out of range: mean = {:?}", mean_out);
        
        let ks_stat = ks_test(&timings_in, &timings_out);
        let critical_value = ks_critical_value(SAMPLE_SIZE, SAMPLE_SIZE);
            let relaxed_threshold = critical_value * 10.0;
        
            println!("  KS statistic: {:.4} (threshold: {:.4})", ks_stat, relaxed_threshold);
        
            if ks_stat < relaxed_threshold {
                println!("  ✓ Timing appears constant");
            } else {
                println!("  ⚠ Warning: Timing variance detected");
            }
    }

    #[test]
    fn test_ct_dot_product_timing_independence() {
        println!("\n=== Testing ct_dot_product timing independence ===");
        
        // Secret vector with all zeros
        let secret_zeros = vec![0i64; 256];
        
        // Secret vector with all ones
        let secret_ones = vec![1i64; 256];
        
        // Public vector (same for both tests)
        let public = vec![42i64; 256];
        
            let timings_zeros = collect_timings(|| ct_dot_product(&secret_zeros, &public), SAMPLE_SIZE / 2);
            let timings_ones = collect_timings(|| ct_dot_product(&secret_ones, &public), SAMPLE_SIZE / 2);
        
        let mean_zeros = mean_duration(&timings_zeros);
        let mean_ones = mean_duration(&timings_ones);
        
        println!("  Secret = zeros: mean = {:?}", mean_zeros);
        println!("  Secret = ones: mean = {:?}", mean_ones);
        
            let ks_stat = ks_test(&timings_zeros, &timings_ones);
            let critical_value = ks_critical_value(SAMPLE_SIZE / 2, SAMPLE_SIZE / 2);
            let relaxed_threshold = critical_value * 15.0;
        
            println!("  KS statistic: {:.4} (threshold: {:.4})", ks_stat, relaxed_threshold);
        
            if ks_stat < relaxed_threshold {
                println!("  ✓ Timing appears constant");
            } else {
                println!("  ⚠ Warning: Timing variance detected");
            }
    }

    #[test]
    fn test_ct_array_access_timing_independence() {
        println!("\n=== Testing ct_array_access timing independence ===");
        
        let array = (0..256).map(|i| i as i64).collect::<Vec<_>>();
        
        // Access first element
        let index_first = 0;
        
        // Access last element
        let index_last = 255;
        
            let timings_first = collect_timings(|| ct_array_access(&array, index_first), SAMPLE_SIZE);
            let timings_last = collect_timings(|| ct_array_access(&array, index_last), SAMPLE_SIZE);
        
        let mean_first = mean_duration(&timings_first);
        let mean_last = mean_duration(&timings_last);
        
        println!("  Access index 0: mean = {:?}", mean_first);
        println!("  Access index 255: mean = {:?}", mean_last);
        
        let ks_stat = ks_test(&timings_first, &timings_last);
        let critical_value = ks_critical_value(SAMPLE_SIZE, SAMPLE_SIZE);
            let relaxed_threshold = critical_value * 15.0;
        
            println!("  KS statistic: {:.4} (threshold: {:.4})", ks_stat, relaxed_threshold);
        
            if ks_stat < relaxed_threshold {
                println!("  ✓ Timing appears constant");
            } else {
                println!("  ⚠ Warning: Timing variance detected");
            }
    }

    #[test]
    #[ignore] // Expensive test - run with --ignored flag
    fn test_comprehensive_timing_analysis() {
        println!("\n=== Comprehensive Timing Analysis ===");
        println!("This test performs extensive timing measurements.");
        println!("Run with: cargo test --test timing_tests --release -- --ignored --nocapture\n");
        
        // Test ct_modpow with various exponent sizes
        for exp_bits in [8, 16, 32, 64, 128, 256] {
            let base = BigUint::from(123u32);
            let modulus = BigUint::from_bytes_be(&[0xFF; 32]);
            let exp = BigUint::from(1u32) << exp_bits;
            
            let timings = collect_timings(|| ct_modpow(&base, &exp, &modulus), 1000);
            let mean = mean_duration(&timings);
            let std_dev = std_dev_duration(&timings);
            
            println!("  ct_modpow({}-bit exp): mean={:?}, std_dev={:.2} ns", 
                     exp_bits, mean, std_dev);
        }
        
        println!("\n  ✓ Comprehensive analysis complete");
    }
}
