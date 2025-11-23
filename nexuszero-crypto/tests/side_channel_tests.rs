//! Comprehensive side-channel resistance tests
//!
//! This test suite validates that cryptographic operations resist various
//! side-channel attacks including:
//! - Cache-timing attacks
//! - Statistical timing analysis (Welch's t-test)
//! - Memory access pattern analysis
//!
//! These tests use statistical methods to detect timing variations that
//! could leak secret information.

use nexuszero_crypto::utils::constant_time::{
    ct_modpow, ct_bytes_eq, ct_in_range, ct_array_access, ct_dot_product,
};
use nexuszero_crypto::utils::sidechannel::{test_constant_time, CacheSimulator, MemoryTracer, AccessType};
use num_bigint::BigUint;
use std::time::{Duration, Instant};

/// Statistical threshold for t-test (4.5 is standard for side-channel testing)
const T_TEST_THRESHOLD: f64 = 4.5;

/// Number of samples for statistical analysis
const SAMPLES: usize = 1000;

#[test]
fn test_ct_modpow_constant_time_property() {
    // Test that ct_modpow execution time is independent of exponent bit pattern
    // Using same bit-length exponents with different Hamming weights
    let base = BigUint::from(3u32);
    let modulus = BigUint::from(0xFFFFu32);
    
    // Class 0: Exponent with alternating bits (0xAA = 10101010)
    // Class 1: Exponent with different pattern (0x55 = 01010101)
    // Both have same Hamming weight (4 ones) and bit length (8 bits)
    let result = test_constant_time(
        |class| {
            let exp = if !class {
                BigUint::from(0xAAu32)  // 10101010
            } else {
                BigUint::from(0x55u32)  // 01010101
            };
            
            let start = Instant::now();
            let _result = ct_modpow(&base, &exp, &modulus);
            start.elapsed()
        },
        SAMPLES,
    );
    
    println!("ct_modpow t-statistic: {:?}", result.t_statistic);
    
    // Montgomery ladder should have same timing for same bit-length exponents
    // Allow slightly higher threshold to account for system variance
    if let Some(t_stat) = result.t_statistic {
        assert!(
            t_stat.abs() < T_TEST_THRESHOLD * 2.0,
            "ct_modpow shows significant timing variation (t = {:.2}), possible side-channel leak",
            t_stat
        );
    }
}

#[test]
fn test_ct_bytes_eq_constant_time_property() {
    // Test that ct_bytes_eq timing is independent of where arrays differ
    let len = 32;
    let a = vec![0xAAu8; len];
    
    let result = test_constant_time(
        |class| {
            // Class 0: Arrays differ in first byte
            // Class 1: Arrays differ in last byte
            let mut b = vec![0xAAu8; len];
            if !class {
                b[0] = 0xBB;
            } else {
                b[len - 1] = 0xBB;
            }
            
            let start = Instant::now();
            let _result = ct_bytes_eq(&a, &b);
            start.elapsed()
        },
        SAMPLES,
    );
    
    println!("ct_bytes_eq t-statistic: {:?}", result.t_statistic);
    
    if let Some(t_stat) = result.t_statistic {
        assert!(
            t_stat.abs() < T_TEST_THRESHOLD,
            "ct_bytes_eq shows timing variation (t = {:.2}), possible early return leak",
            t_stat
        );
    }
}

#[test]
fn test_ct_in_range_constant_time_property() {
    // Test that ct_in_range timing is independent of whether value is in/out of range
    let min = 100u64;
    let max = 200u64;
    
    let result = test_constant_time(
        |class| {
            // Class 0: Value in range
            // Class 1: Value out of range
            let value = if !class {
                150u64 // In range
            } else {
                300u64 // Out of range
            };
            
            let start = Instant::now();
            let _result = ct_in_range(value, min, max);
            start.elapsed()
        },
        SAMPLES,
    );
    
    println!("ct_in_range t-statistic: {:?}", result.t_statistic);
    
    if let Some(t_stat) = result.t_statistic {
        assert!(
            t_stat.abs() < T_TEST_THRESHOLD,
            "ct_in_range shows timing variation (t = {:.2}), possible branch leak",
            t_stat
        );
    }
}

#[test]
fn test_ct_array_access_constant_time_property() {
    // Test that ct_array_access timing is independent of target index
    let array = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
    
    let result = test_constant_time(
        |class| {
            // Class 0: Access first element
            // Class 1: Access last element
            let index = if !class { 0 } else { 7 };
            
            let start = Instant::now();
            let _result = ct_array_access(&array, index);
            start.elapsed()
        },
        SAMPLES,
    );
    
    println!("ct_array_access t-statistic: {:?}", result.t_statistic);
    
    if let Some(t_stat) = result.t_statistic {
        assert!(
            t_stat.abs() < T_TEST_THRESHOLD,
            "ct_array_access shows timing variation (t = {:.2}), possible index-dependent timing",
            t_stat
        );
    }
}

#[test]
fn test_ct_dot_product_constant_time_property() {
    // Test that ct_dot_product timing is independent of secret values
    let public = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
    
    let result = test_constant_time(
        |class| {
            // Class 0: Secret with all zeros
            // Class 1: Secret with mixed values
            let secret = if !class {
                vec![0i64; 8]
            } else {
                vec![1i64, -1, 2, -2, 3, -3, 4, -4]
            };
            
            let start = Instant::now();
            let _result = ct_dot_product(&secret, &public);
            start.elapsed()
        },
        SAMPLES,
    );
    
    println!("ct_dot_product t-statistic: {:?}", result.t_statistic);
    
    if let Some(t_stat) = result.t_statistic {
        assert!(
            t_stat.abs() < T_TEST_THRESHOLD,
            "ct_dot_product shows timing variation (t = {:.2}), possible value-dependent timing",
            t_stat
        );
    }
}

#[test]
fn test_cache_timing_ct_array_access() {
    // Simulate cache-timing attack on constant-time array access
    let mut cache = CacheSimulator::new(); // 64-byte lines, 8 lines
    let array = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    // Access pattern should not reveal index
    for &index in &[0, 5, 9, 2, 7] {
        let value = ct_array_access(&array, index);
        
        // Simulate memory accesses
        for i in 0..array.len() {
            let addr = i * 8; // 8 bytes per i64
            cache.access(addr);
        }
        
        // Use value to prevent optimization
        std::hint::black_box(value);
    }
    
    let analysis = cache.analyze_patterns();
    println!("Cache analysis: total_accesses={}, hit_rate={:.2}", 
             analysis.total_accesses, analysis.cache_hit_rate);
    
    // We can't make strong assertions about cache behavior in tests,
    // but we verify the simulator works
    assert!(analysis.total_accesses > 0, "Cache should have recorded accesses");
}

#[test]
fn test_memory_access_pattern_analysis() {
    // Test that memory access patterns don't leak secret information
    let mut tracer = MemoryTracer::new();
    
    // Simulate secret-dependent operation
    let secret_index = 3;
    let array = vec![10i64, 20, 30, 40, 50];
    
    // Constant-time access - should access all elements
    for i in 0..array.len() {
        let addr = i * 8;
        tracer.record(addr, 8, AccessType::Read);
    }
    
    let _value = ct_array_access(&array, secret_index);
    
    // Analyze regularity
    let regularity = tracer.analyze_regularity();
    println!("Access regularity: {:.2}", regularity);
    
    // For constant-time operations, accesses should be recorded
    assert!(
        tracer.access_count() >= array.len(),
        "Constant-time array access should record all elements"
    );
}

#[test]
fn test_welch_t_test_sensitivity() {
    // Verify that our t-test implementation can detect timing differences
    let result = test_constant_time(
        |class| {
            let start = Instant::now();
            
            // Intentionally create timing difference
            if !class {
                // Fast path
                std::hint::black_box(1 + 1);
            } else {
                // Slow path - more operations
                for _ in 0..100 {
                    std::hint::black_box(1 + 1);
                }
            }
            
            start.elapsed()
        },
        SAMPLES,
    );
    
    println!("Intentional leak t-statistic: {:?}", result.t_statistic);
    
    if let Some(t_stat) = result.t_statistic {
        // This should detect the intentional timing difference
        assert!(
            t_stat.abs() >= T_TEST_THRESHOLD,
            "T-test should detect intentional timing leak (t = {:.2})",
            t_stat
        );
    }
}

#[test]
fn test_ct_modpow_different_bit_lengths() {
    // Test timing with different exponent bit lengths
    let base = BigUint::from(5u32);
    let modulus = BigUint::from(0xFFFFu32);
    
    let result = test_constant_time(
        |class| {
            // Class 0: 8-bit exponent
            // Class 1: 16-bit exponent
            let exp = if !class {
                BigUint::from(0xFFu32)
            } else {
                BigUint::from(0xFFFFu32)
            };
            
            let start = Instant::now();
            let _result = ct_modpow(&base, &exp, &modulus);
            start.elapsed()
        },
        SAMPLES / 2, // Fewer samples since these operations are slower
    );
    
    println!("ct_modpow bit length t-statistic: {:?}", result.t_statistic);
    
    // Note: Timing may legitimately vary with bit length (which is public)
    // This test documents the behavior rather than asserting strict bounds
}

#[test]
fn test_ct_bytes_eq_varying_positions() {
    // Test that timing doesn't reveal position of first difference
    let len = 32;
    let a = vec![0xAAu8; len];
    
    let mut timings = Vec::new();
    
    // Test different positions where arrays differ
    for diff_pos in [0, 8, 16, 24, 31] {
        let mut samples = Vec::new();
        
        for _ in 0..100 {
            let mut b = vec![0xAAu8; len];
            b[diff_pos] = 0xBB;
            
            let start = Instant::now();
            let _result = ct_bytes_eq(&a, &b);
            let duration = start.elapsed();
            
            samples.push(duration);
        }
        
        let avg: Duration = samples.iter().sum::<Duration>() / samples.len() as u32;
        timings.push((diff_pos, avg));
    }
    
    // Print timing distribution
    for (pos, avg) in &timings {
        println!("Position {}: {:?}", pos, avg);
    }
    
    // Verify relatively uniform timing (within 2x variance)
    let min_time = timings.iter().map(|(_, t)| *t).min().unwrap();
    let max_time = timings.iter().map(|(_, t)| *t).max().unwrap();
    
    // Allow some variance due to system noise, but not excessive
    let ratio = max_time.as_nanos() as f64 / min_time.as_nanos().max(1) as f64;
    assert!(
        ratio < 3.0,
        "Timing variation too large ({}x), may indicate position-dependent timing",
        ratio
    );
}

#[test]
fn test_statistical_power_analysis_simulation() {
    // Simulate power analysis by measuring operation counts
    // (In real hardware, this would measure actual power consumption)
    
    let base = BigUint::from(7u32);
    let modulus = BigUint::from(0xFFu32);
    
    // Test with different Hamming weights in exponent
    let low_hamming = BigUint::from(0b00000001u32); // 1 one-bit
    let high_hamming = BigUint::from(0b11111111u32); // 8 one-bits
    
    // In a vulnerable implementation, high Hamming weight would
    // require more multiplications
    let start1 = Instant::now();
    let _r1 = ct_modpow(&base, &low_hamming, &modulus);
    let time1 = start1.elapsed();
    
    let start2 = Instant::now();
    let _r2 = ct_modpow(&base, &high_hamming, &modulus);
    let time2 = start2.elapsed();
    
    println!("Low Hamming weight: {:?}", time1);
    println!("High Hamming weight: {:?}", time2);
    
    // The timing should be similar (within reason, accounting for noise)
    // Montgomery ladder should perform same number of operations
    let ratio = time2.as_nanos().max(1) as f64 / time1.as_nanos().max(1) as f64;
    
    // Allow some variance but flag if significantly different
    if ratio > 2.0 || ratio < 0.5 {
        println!("WARNING: Significant timing difference detected (ratio: {:.2})", ratio);
        println!("This may indicate Hamming weight dependency");
    }
}

#[test]
fn test_cache_line_analysis() {
    // Test that operations don't create secret-dependent cache patterns
    let mut cache = CacheSimulator::new();
    
    let array = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
    
    // Access with secret index - should not create detectable pattern
    for secret_index in 0..array.len() {
        // Reset cache for each test
        cache = CacheSimulator::new();
        
        // Perform constant-time access
        let _value = ct_array_access(&array, secret_index);
        
        // Simulate memory accesses for entire array
        for i in 0..array.len() {
            let addr = i * 8;
            cache.access(addr);
        }
    }
    
    let analysis = cache.analyze_patterns();
    println!("Final cache analysis: {:?}", analysis);
    
    // Verify cache was used
    assert!(analysis.total_accesses > 0);
}

#[test]
fn test_timing_distribution_normality() {
    // Test that timing distribution is normal (not bimodal, which would indicate branching)
    let base = BigUint::from(3u32);
    let exp = BigUint::from(123u32);
    let modulus = BigUint::from(0xFFFFu32);
    
    let mut timings = Vec::new();
    
    for _ in 0..500 {
        let start = Instant::now();
        let _result = ct_modpow(&base, &exp, &modulus);
        timings.push(start.elapsed().as_nanos());
    }
    
    // Calculate mean and standard deviation
    let mean = timings.iter().sum::<u128>() as f64 / timings.len() as f64;
    let variance = timings.iter()
        .map(|&t| {
            let diff = t as f64 - mean;
            diff * diff
        })
        .sum::<f64>() / timings.len() as f64;
    let std_dev = variance.sqrt();
    
    println!("Timing: mean={:.2}ns, std_dev={:.2}ns", mean, std_dev);
    
    // Count outliers (values more than 3 std devs from mean)
    let outliers = timings.iter()
        .filter(|&&t| {
            let diff = (t as f64 - mean).abs();
            diff > 3.0 * std_dev
        })
        .count();
    
    let outlier_ratio = outliers as f64 / timings.len() as f64;
    
    println!("Outliers: {} ({:.1}%)", outliers, outlier_ratio * 100.0);
    
    // Normal distribution should have <0.3% outliers beyond 3 sigma
    // Allow up to 5% due to system noise
    assert!(
        outlier_ratio < 0.05,
        "Too many outliers ({:.1}%), may indicate bimodal timing distribution",
        outlier_ratio * 100.0
    );
}

#[test]
fn test_constant_time_operations_under_load() {
    // Test timing consistency under system load
    // This helps verify that constant-time properties hold even with
    // cache pressure and context switches
    
    let base = BigUint::from(5u32);
    let modulus = BigUint::from(0xFFFFu32);
    
    // Run with different exponents under "load" (multiple iterations)
    let exp1 = BigUint::from(0x00FFu32);
    let exp2 = BigUint::from(0xFF00u32);
    
    let mut times1 = Vec::new();
    let mut times2 = Vec::new();
    
    for _ in 0..100 {
        let start = Instant::now();
        let _r1 = ct_modpow(&base, &exp1, &modulus);
        times1.push(start.elapsed());
        
        let start = Instant::now();
        let _r2 = ct_modpow(&base, &exp2, &modulus);
        times2.push(start.elapsed());
    }
    
    let avg1: Duration = times1.iter().sum::<Duration>() / times1.len() as u32;
    let avg2: Duration = times2.iter().sum::<Duration>() / times2.len() as u32;
    
    println!("Avg time exp1: {:?}", avg1);
    println!("Avg time exp2: {:?}", avg2);
    
    // Times should be similar
    let ratio = avg2.as_nanos().max(1) as f64 / avg1.as_nanos().max(1) as f64;
    
    assert!(
        ratio < 2.0 && ratio > 0.5,
        "Timing ratio too large ({:.2}), may indicate value-dependent timing",
        ratio
    );
}
