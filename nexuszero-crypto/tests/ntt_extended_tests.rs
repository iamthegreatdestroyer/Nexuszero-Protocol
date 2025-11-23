use nexuszero_crypto::lattice::ring_lwe::*;
use rand::Rng;

// Verify NTT multiplication matches schoolbook for random polynomials when enabled.
#[test]
fn test_ntt_vs_schoolbook_random() {
    std::env::set_var("NEXUSZERO_USE_NTT", "1");
    let q = 12289u64; // parameter with primitive root
    let n = 512usize;
    let root = find_primitive_root(n, q).unwrap();
    for _ in 0..10 {
        let a = sample_poly_uniform(n, q);
        let b = sample_poly_uniform(n, q);
        let ntt_prod = poly_mult_ntt(&a, &b, q);
        let school_prod = poly_mult_schoolbook(&a, &b, q);
        assert_eq!(ntt_prod.coeffs, school_prod.coeffs, "Mismatch in NTT vs schoolbook product");
    }
}

// Property: intt(ntt(p)) = p for multiple random polynomials.
#[test]
fn test_ntt_round_trip_multiple() {
    let q = 12289u64;
    let n = 512usize;
    let root = find_primitive_root(n, q).unwrap();
    for _ in 0..25 {
        let poly = sample_poly_uniform(n, q);
        let t = ntt(&poly, q, root);
        let back = intt(&t, n, q, root);
        assert_eq!(poly.coeffs, back.coeffs, "Round-trip failed");
    }
}

// Edge case: multiplication with zero polynomial.
#[test]
fn test_ntt_multiplication_with_zero() {
    std::env::set_var("NEXUSZERO_USE_NTT", "1");
    let q = 12289u64;
    let n = 512usize;
    let zero = Polynomial::zero(n, q);
    let rand_poly = sample_poly_uniform(n, q);
    let prod = poly_mult_ntt(&zero, &rand_poly, q);
    assert!(prod.coeffs.iter().all(|&c| c == 0));
}

// Stress test: random sizes fallback path triggers schoolbook for unsupported primitive root combos.
#[test]
fn test_ntt_fallback_path() {
    std::env::set_var("NEXUSZERO_USE_NTT", "1");
    let q = 12289u64;
    // Size without known primitive root mapping (e.g., 128) triggers fallback
    let n = 128usize;
    let a = sample_poly_uniform(n, q);
    let b = sample_poly_uniform(n, q);
    let prod = poly_mult_ntt(&a, &b, q);
    let school = poly_mult_schoolbook(&a, &b, q);
    assert_eq!(prod.coeffs, school.coeffs);
}
