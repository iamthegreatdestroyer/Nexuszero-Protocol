//! Property-based timing tests for constant-time operations
//!
//! These tests use proptest to generate varied inputs and statistically
//! verify that execution time does not significantly depend on secret data
//! patterns. Thresholds are relaxed to avoid flakiness on non-isolated hosts.

use proptest::prelude::*;
use std::time::{Duration, Instant};
use nexuszero_crypto::utils::constant_time::{
    ct_bytes_eq, ct_in_range, ct_dot_product, ct_array_access, ct_modpow
};
use num_bigint::BigUint;

// Number of repetitions per measurement to smooth OS noise
const REPEAT: usize = 50;

// Helper: average duration of running f() REPEAT times
fn avg_duration<F: Fn() -> R, R>(f: F) -> Duration {
    let start = Instant::now();
    for _ in 0..REPEAT { let _ = f(); }
    start.elapsed() / REPEAT as u32
}

// Base threshold for relative timing differences. Can be overridden by setting
// the TIMING_THRESHOLD environment variable (e.g. TIMING_THRESHOLD=0.15).
// Default tightened from 0.35 (35%) to 0.20 (20%).
fn base_threshold() -> f64 {
    std::env::var("TIMING_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v.max(0.05)) // never allow less than 5% to avoid extreme flakiness
        .unwrap_or(0.20)
}

fn rel_diff(a: Duration, b: Duration) -> f64 {
    let (x, y) = if a > b { (a, b) } else { (b, a) };
    (x.as_nanos() - y.as_nanos()) as f64 / y.as_nanos() as f64
}

proptest! {
    #[test]
    fn prop_ct_bytes_eq_timing_independent(a in proptest::collection::vec(any::<u8>(), 1..64)) {
        // Create matching and differing vectors of same length
        let mut b_match = a.clone();
        let mut b_diff = a.clone();
        // Flip last byte (if length >=1) to force mismatch
        let last = b_diff.len()-1;
        b_diff[last] = b_diff[last].wrapping_add(1);

        let t_match = avg_duration(|| ct_bytes_eq(&a, &b_match));
        let t_diff  = avg_duration(|| ct_bytes_eq(&a, &b_diff));

        let rd = rel_diff(t_match, t_diff);
        let thr = base_threshold();
        prop_assert!(rd <= thr,
            "ct_bytes_eq timing differs too much: rel_diff={:.2}% len={} t_match={:?} t_diff={:?}",
            rd*100.0, a.len(), t_match, t_diff);
    }

    #[test]
    fn prop_ct_in_range_timing_independent(val in 0u64..200, min in 0u64..100, max in 101u64..200) {
        // Ensure min < max
        let (lo, hi) = if min < max { (min, max) } else { (max-1, max) };
        let in_val = (lo + hi)/2; // definitely inside
        let out_val = hi + 1;     // definitely outside

        let t_in  = avg_duration(|| ct_in_range(in_val, lo, hi));
        let t_out = avg_duration(|| ct_in_range(out_val, lo, hi));

        let rd = rel_diff(t_in, t_out);
        let thr = base_threshold();
        prop_assert!(rd <= thr,
            "ct_in_range timing differs: rel_diff={:.2}% range=[{},{}] t_in={:?} t_out={:?}",
            rd*100.0, lo, hi, t_in, t_out);
    }

    #[test]
    fn prop_ct_array_access_index_independent(array in proptest::collection::vec(any::<i64>(), 2..128)) {
        let first_idx = 0usize;
        let last_idx  = array.len()-1;
        let t_first = avg_duration(|| ct_array_access(&array, first_idx));
        let t_last  = avg_duration(|| ct_array_access(&array, last_idx));
        let rd = rel_diff(t_first, t_last);
        let thr = base_threshold();
        prop_assert!(rd <= thr,
            "ct_array_access timing differs by index: rel_diff={:.2}% len={} t_first={:?} t_last={:?}",
            rd*100.0, array.len(), t_first, t_last);
    }

    #[test]
    fn prop_ct_dot_product_secret_pattern_independent(len in 2usize..64) {
        // Build two secret vectors of same length with very different patterns
        let secret_all_zeros = vec![0i64; len];
        let secret_all_ones  = vec![1i64; len];
        let public = vec![7i64; len];

        let t_zeros = avg_duration(|| ct_dot_product(&secret_all_zeros, &public));
        let t_ones  = avg_duration(|| ct_dot_product(&secret_all_ones, &public));

        let rd = rel_diff(t_zeros, t_ones);
        let thr = base_threshold();
        prop_assert!(rd <= thr,
            "ct_dot_product timing differs by secret pattern: rel_diff={:.2}% len={} t0={:?} t1={:?}",
            rd*100.0, len, t_zeros, t_ones);
    }

    #[test]
    fn prop_ct_modpow_bitpattern_independent(bits in 8u32..24u32) {
        // Build two exponents of same bit length but inverted alternating patterns
        let base = BigUint::from(17u32);
        let modulus = BigUint::from_bytes_be(&vec![0xFF; 32]);
        let pattern_a = ((1u64 << bits) - 1) & 0xAAAAAAAAAAAAAAAAu64; // mask alt bits
        let pattern_b = ((1u64 << bits) - 1) & 0x5555555555555555u64; // inverse alt
        let exp_a = BigUint::from(pattern_a);
        let exp_b = BigUint::from(pattern_b);

        let t_a = avg_duration(|| ct_modpow(&base, &exp_a, &modulus));
        let t_b = avg_duration(|| ct_modpow(&base, &exp_b, &modulus));
        let rd = rel_diff(t_a, t_b);
        // Slightly looser threshold for modpow due to higher variance (1.5x base)
        let thr = base_threshold() * 1.5;
        prop_assert!(rd <= thr,
            "ct_modpow timing differs by bit pattern: rel_diff={:.2}% bits={} tA={:?} tB={:?}",
            rd*100.0, bits, t_a, t_b);
    }
}
