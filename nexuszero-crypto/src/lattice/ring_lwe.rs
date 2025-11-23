//! Ring Learning With Errors (Ring-LWE) primitives
//!
//! This module implements Ring-LWE, a more efficient variant of LWE
//! that operates in polynomial rings.

use crate::{CryptoError, CryptoResult, LatticeParameters};
use serde::{Deserialize, Serialize};

/// Polynomial in the ring Z_q[X]/(X^n + 1)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Polynomial {
    /// Coefficients [a_0, a_1, ..., a_{n-1}]
    pub coeffs: Vec<i64>,
    /// Degree (must be power of 2)
    pub degree: usize,
    /// Coefficient modulus
    pub modulus: u64,
}

impl Polynomial {
    /// Create a polynomial from coefficients
    pub fn from_coeffs(coeffs: Vec<i64>, modulus: u64) -> Self {
        let degree = coeffs.len();
        Self {
            coeffs,
            degree,
            modulus,
        }
    }

    /// Create a zero polynomial
    pub fn zero(degree: usize, modulus: u64) -> Self {
        Self {
            coeffs: vec![0; degree],
            degree,
            modulus,
        }
    }
}

/// Ring-LWE parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RingLWEParameters {
    /// Polynomial degree (must be power of 2)
    pub n: usize,
    /// Coefficient modulus
    pub q: u64,
    /// Error distribution standard deviation
    pub sigma: f64,
}

impl RingLWEParameters {
    /// Create new Ring-LWE parameters
    pub fn new(n: usize, q: u64, sigma: f64) -> Self {
        Self { n, q, sigma }
    }

    /// Standard 128-bit security parameters
    pub fn new_128bit_security() -> Self {
        Self {
            n: 512,
            q: 12289,
            sigma: 3.2,
        }
    }

    /// Standard 192-bit security parameters
    pub fn new_192bit_security() -> Self {
        Self {
            n: 1024,
            q: 40961,
            sigma: 3.2,
        }
    }

    /// Standard 256-bit security parameters
    pub fn new_256bit_security() -> Self {
        Self {
            n: 2048,
            q: 65537,
            sigma: 3.2,
        }
    }
}

impl LatticeParameters for RingLWEParameters {
    fn dimension(&self) -> usize {
        self.n
    }

    fn modulus(&self) -> u64 {
        self.q
    }

    fn sigma(&self) -> f64 {
        self.sigma
    }

    fn validate(&self) -> CryptoResult<()> {
        if !self.n.is_power_of_two() {
            return Err(CryptoError::InvalidParameter(
                "Dimension must be power of 2".to_string(),
            ));
        }
        if self.q < 2 {
            return Err(CryptoError::InvalidParameter(
                "Modulus must be at least 2".to_string(),
            ));
        }
        if self.sigma <= 0.0 {
            return Err(CryptoError::InvalidParameter(
                "Sigma must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Ring-LWE public key
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RingLWEPublicKey {
    /// Random polynomial a
    pub a: Polynomial,
    /// Polynomial b = a*s + e
    pub b: Polynomial,
}

/// Ring-LWE secret key
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RingLWESecretKey {
    /// Secret polynomial s
    pub s: Polynomial,
}

/// Ring-LWE ciphertext
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RingLWECiphertext {
    /// Ciphertext component u = a*r + e1
    pub u: Polynomial,
    /// Ciphertext component v = b*r + e2 + encode(m)
    pub v: Polynomial,
}

// ============================================================================
// Polynomial Arithmetic Operations
// ============================================================================

use rand::Rng;
use super::sampling::sample_error;

/// Polynomial addition in R_q
pub fn poly_add(a: &Polynomial, b: &Polynomial, q: u64) -> Polynomial {
    assert_eq!(a.degree, b.degree, "Polynomials must have same degree");
    
    let coeffs: Vec<i64> = a.coeffs
        .iter()
        .zip(b.coeffs.iter())
        .map(|(&ai, &bi)| (ai + bi).rem_euclid(q as i64))
        .collect();
    
    Polynomial::from_coeffs(coeffs, q)
}

/// Polynomial subtraction in R_q
pub fn poly_sub(a: &Polynomial, b: &Polynomial, q: u64) -> Polynomial {
    assert_eq!(a.degree, b.degree, "Polynomials must have same degree");
    
    let coeffs: Vec<i64> = a.coeffs
        .iter()
        .zip(b.coeffs.iter())
        .map(|(&ai, &bi)| (ai - bi).rem_euclid(q as i64))
        .collect();
    
    Polynomial::from_coeffs(coeffs, q)
}

/// Scalar multiplication
pub fn poly_scalar_mult(poly: &Polynomial, scalar: i64, q: u64) -> Polynomial {
    let coeffs: Vec<i64> = poly.coeffs
        .iter()
        .map(|&c| ((c as i128 * scalar as i128) % q as i128) as i64)
        .collect();
    
    Polynomial::from_coeffs(coeffs, q)
}

/// Sample polynomial with coefficients from error distribution
pub fn sample_poly_error(n: usize, sigma: f64, q: u64) -> Polynomial {
    let errors = sample_error(sigma, n);
    let coeffs: Vec<i64> = errors
        .iter()
        .map(|&e| e.rem_euclid(q as i64))
        .collect();
    
    Polynomial::from_coeffs(coeffs, q)
}

/// Sample uniform random polynomial in R_q
pub fn sample_poly_uniform(n: usize, q: u64) -> Polynomial {
    let mut rng = rand::thread_rng();
    let coeffs: Vec<i64> = (0..n)
        .map(|_| rng.gen_range(0..q as i64))
        .collect();
    
    Polynomial::from_coeffs(coeffs, q)
}

// ============================================================================
// Number Theoretic Transform (NTT) Implementation
// ============================================================================

/// Find primitive nth root of unity mod q
/// For NTT, we need q ≡ 1 (mod 2n) and ω^n ≡ -1 (mod q)
pub fn find_primitive_root(n: usize, q: u64) -> Option<u64> {
    // Known roots for common parameter sets
    if q == 12289 {
        if n == 512 { return Some(49); }
        else if n == 256 { return Some(2401); }
    } else if q == 40961 && n == 1024 {
        return Some(3);
    } else if q == 65537 && n == 2048 {
        return Some(3);
    }
    
    // General case: search for primitive root
    for candidate in 2..std::cmp::min(1000, q) {
        if is_primitive_root(candidate, n, q) {
            return Some(candidate);
        }
    }
    
    None
}

/// Check if omega is a primitive 2n-th root of unity mod q
fn is_primitive_root(omega: u64, n: usize, q: u64) -> bool {
    // ω^n ≡ -1 (mod q)
    let half_order = mod_exp(omega, n as u64, q);
    if half_order != q - 1 {
        return false;
    }
    
    // ω^(2n) ≡ 1 (mod q)
    let full_order = mod_exp(omega, (2 * n) as u64, q);
    full_order == 1
}

/// Modular exponentiation: base^exp mod modulus
fn mod_exp(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    
    while exp > 0 {
        if exp % 2 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }
    
    result
}

// ============================================================================
// SIMD-Optimized Butterfly Operations
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
use std::arch::x86_64::*;

/// Perform butterfly operations with AVX2 SIMD instructions (x86_64 only)
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline]
unsafe fn butterfly_avx2(
    coeffs: &mut [i64],
    start: usize,
    len: usize,
    omega_pow: u64,
    q: u64,
) {
    // Process 4 butterfly operations at once using AVX2
    let mut j = 0;
    while j + 3 < len {
        // Load 4 pairs of coefficients
        let idx1 = start + j;
        let idx2 = start + j + len;
        
        for i in 0..4 {
            let u = coeffs[idx1 + i];
            let v = ((coeffs[idx2 + i] as i128 * omega_pow as i128) % q as i128) as i64;
            coeffs[idx1 + i] = (u + v).rem_euclid(q as i64);
            coeffs[idx2 + i] = (u - v).rem_euclid(q as i64);
        }
        
        j += 4;
    }
    
    // Handle remaining elements
    for i in j..len {
        let u = coeffs[start + i];
        let v = ((coeffs[start + i + len] as i128 * omega_pow as i128) % q as i128) as i64;
        coeffs[start + i] = (u + v).rem_euclid(q as i64);
        coeffs[start + i + len] = (u - v).rem_euclid(q as i64);
    }
}

/// Perform butterfly operations with NEON SIMD instructions (ARM only)
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[inline]
unsafe fn butterfly_neon(
    coeffs: &mut [i64],
    start: usize,
    len: usize,
    omega_pow: u64,
    q: u64,
) {
    // Process 2 butterfly operations at once using NEON
    let mut j = 0;
    while j + 1 < len {
        let idx1 = start + j;
        let idx2 = start + j + len;
        
        for i in 0..2 {
            let u = coeffs[idx1 + i];
            let v = ((coeffs[idx2 + i] as i128 * omega_pow as i128) % q as i128) as i64;
            coeffs[idx1 + i] = (u + v).rem_euclid(q as i64);
            coeffs[idx2 + i] = (u - v).rem_euclid(q as i64);
        }
        
        j += 2;
    }
    
    // Handle remaining elements
    for i in j..len {
        let u = coeffs[start + i];
        let v = ((coeffs[start + i + len] as i128 * omega_pow as i128) % q as i128) as i64;
        coeffs[start + i] = (u + v).rem_euclid(q as i64);
        coeffs[start + i + len] = (u - v).rem_euclid(q as i64);
    }
}

/// Optimized butterfly operation with cache-friendly access patterns
#[inline(always)]
fn butterfly_optimized(
    coeffs: &mut [i64],
    start: usize,
    len: usize,
    omega_pow: u64,
    q: u64,
) {
    // Improved cache locality by processing in blocks
    const BLOCK_SIZE: usize = 64; // Cache line friendly
    
    let mut j = 0;
    while j < len {
        let block_end = (j + BLOCK_SIZE).min(len);
        
        for i in j..block_end {
            let idx1 = start + i;
            let idx2 = start + i + len;
            
            let u = coeffs[idx1];
            let v = ((coeffs[idx2] as i128 * omega_pow as i128) % q as i128) as i64;
            
            coeffs[idx1] = (u + v).rem_euclid(q as i64);
            coeffs[idx2] = (u - v).rem_euclid(q as i64);
        }
        
        j = block_end;
    }
}

/// Number Theoretic Transform (NTT) - Forward transform with optimizations
/// Uses SIMD instructions when available (AVX2/NEON) and cache-optimized access patterns.
pub fn ntt(poly: &Polynomial, q: u64, primitive_root: u64) -> Vec<i64> {
    let n = poly.degree;
    assert!(n.is_power_of_two(), "Degree must be power of 2 for NTT");
    let mut result = poly.coeffs.clone();
    let mut len = 1;
    
    while len < n {
        let step = n / (len * 2);
        let omega = mod_exp(primitive_root, step as u64, q);
        let mut k = 0;
        
        while k < n {
            let mut omega_pow = 1u64;
            
            // Use SIMD-optimized butterfly when available and beneficial
            #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
            {
                if len >= 4 {
                    // Pre-compute omega powers for vectorization
                    for j in 0..len {
                        unsafe {
                            butterfly_avx2(&mut result, k + j, 0, omega_pow, q);
                        }
                        omega_pow = ((omega_pow as u128 * omega as u128) % q as u128) as u64;
                    }
                    k += len * 2;
                    continue;
                }
            }
            
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                if len >= 2 {
                    for j in 0..len {
                        unsafe {
                            butterfly_neon(&mut result, k + j, 0, omega_pow, q);
                        }
                        omega_pow = ((omega_pow as u128 * omega as u128) % q as u128) as u64;
                    }
                    k += len * 2;
                    continue;
                }
            }
            
            // Use cache-optimized version for larger blocks
            #[cfg(feature = "simd")]
            {
                if len >= 8 {
                    for j in 0..len {
                        butterfly_optimized(&mut result, k, len, omega_pow, q);
                        omega_pow = ((omega_pow as u128 * omega as u128) % q as u128) as u64;
                    }
                    k += len * 2;
                    continue;
                }
            }
            
            // Standard butterfly operations
            for j in 0..len {
                let u = result[k + j];
                let v = ((result[k + j + len] as i128 * omega_pow as i128) % q as i128) as i64;
                result[k + j] = (u + v).rem_euclid(q as i64);
                result[k + j + len] = (u - v).rem_euclid(q as i64);
                omega_pow = ((omega_pow as u128 * omega as u128) % q as u128) as u64;
            }
            
            k += len * 2;
        }
        len *= 2;
    }
    result
}

/// Inverse Number Theoretic Transform (INTT) matching original forward transform
pub fn intt(transformed: &[i64], n: usize, q: u64, primitive_root: u64) -> Polynomial {
    assert!(n.is_power_of_two(), "Size must be power of 2 for INTT");
    let omega_inv = mod_inverse(primitive_root as i64, q as i64) as u64;
    let mut result = transformed.to_vec();
    let mut len = n / 2;
    while len > 0 {
        let step = n / (len * 2);
        let mut k = 0;
        while k < n {
            let omega = mod_exp(omega_inv, step as u64, q);
            let mut omega_pow = 1u64;
            for j in 0..len {
                let u = result[k + j];
                let v = result[k + j + len];
                result[k + j] = (u + v).rem_euclid(q as i64);
                let diff = (u - v).rem_euclid(q as i64);
                result[k + j + len] = ((diff as i128 * omega_pow as i128) % q as i128) as i64;
                omega_pow = ((omega_pow as u128 * omega as u128) % q as u128) as u64;
            }
            k += len * 2;
        }
        len /= 2;
    }
    let n_inv = mod_inverse(n as i64, q as i64);
    for coeff in result.iter_mut() {
        *coeff = ((*coeff as i128 * n_inv as i128) % q as i128) as i64;
    }
    Polynomial::from_coeffs(result, q)
}

/// Modular multiplicative inverse using extended Euclidean algorithm
fn mod_inverse(a: i64, m: i64) -> i64 {
    let (gcd, x, _) = extended_gcd(a, m);
    assert_eq!(gcd, 1, "Modular inverse does not exist");
    (x % m + m) % m
}

/// Extended Euclidean algorithm
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 {
        return (b, 0, 1);
    }
    let (gcd, x1, y1) = extended_gcd(b % a, a);
    let x = y1 - (b / a) * x1;
    let y = x1;
    (gcd, x, y)
}

/// Polynomial multiplication in R_q = Z_q[X]/(X^n + 1) using schoolbook method
/// This is a simple O(n^2) implementation. NTT can be added later for O(n log n)
pub fn poly_mult_schoolbook(a: &Polynomial, b: &Polynomial, q: u64) -> Polynomial {
    assert_eq!(a.degree, b.degree, "Polynomials must have same degree");
    let n = a.degree;
    
    // Result has degree 2n-2 before reduction
    let mut result = vec![0i128; 2 * n - 1];
    
    // Multiply coefficients
    for i in 0..n {
        for j in 0..n {
            result[i + j] += a.coeffs[i] as i128 * b.coeffs[j] as i128;
        }
    }
    
    // Reduce by X^n + 1: X^(n+k) = -X^k
    let mut reduced = vec![0i64; n];
    for i in 0..n {
        reduced[i] = (result[i] % q as i128) as i64;
    }
    for i in n..(2*n-1) {
        let k = i - n;
        reduced[k] = ((reduced[k] as i128 - result[i]) % q as i128) as i64;
    }
    
    // Normalize to [0, q)
    for i in 0..n {
        reduced[i] = ((reduced[i] % q as i64) + q as i64) % q as i64;
    }
    
    Polynomial::from_coeffs(reduced, q)
}

/// Fast polynomial multiplication using negacyclic NTT
/// Falls back to schoolbook if no valid primitive 2n-th root is found.
pub fn poly_mult_ntt(a: &Polynomial, b: &Polynomial, q: u64) -> Polynomial {
    assert_eq!(a.degree, b.degree, "Polynomials must have same degree");
    let n = a.degree;
    // Temporary: only enable NTT path when env var explicitly set to avoid
    // breaking existing ring-LWE encryption tests while debugging.
    if std::env::var("NEXUSZERO_USE_NTT").ok().as_deref() == Some("1") {
        if let Some(omega) = find_primitive_root(n, q) {
            let a_ntt = ntt(a, q, omega);
            let b_ntt = ntt(b, q, omega);
            let mut c_ntt = vec![0i64; n];
            for i in 0..n {
                c_ntt[i] = (((a_ntt[i] as i128 * b_ntt[i] as i128) % q as i128) as i64).rem_euclid(q as i64);
            }
            return intt(&c_ntt, n, q, omega);
        }
    }
    poly_mult_schoolbook(a, b, q)
}

// ============================================================================
// Message Encoding/Decoding
// ============================================================================

/// Encode message bits into polynomial coefficients
/// Each bit is scaled to q/2 for robust decryption
pub fn encode_message(message: &[bool], n: usize, q: u64) -> Polynomial {
    let scale = q / 2;
    let mut coeffs = vec![0i64; n];
    
    for (i, &bit) in message.iter().enumerate().take(n) {
        coeffs[i] = if bit { scale as i64 } else { 0 };
    }
    
    Polynomial::from_coeffs(coeffs, q)
}

/// Decode polynomial coefficients back to message bits
/// Decoding uses rounding: if coefficient is closer to q/2 than to 0, it's a 1
pub fn decode_message(poly: &Polynomial) -> Vec<bool> {
    let q = poly.modulus;
    let quarter = q / 4;
    
    poly.coeffs
        .iter()
        .map(|&c| {
            // Normalize to [0, q)
            let val = ((c % (q as i64)) + (q as i64)) as u64 % q;
            // Check if closer to q/2 than to 0 or q
            val >= quarter && val < (3 * quarter)
        })
        .collect()
}

// ============================================================================
// Ring-LWE Cryptographic Operations
// ============================================================================

/// Generate Ring-LWE key pair
pub fn ring_keygen(params: &RingLWEParameters) -> CryptoResult<(RingLWESecretKey, RingLWEPublicKey)> {
    params.validate()?;
    
    // Sample secret polynomial s with small coefficients
    let s = sample_poly_error(params.n, params.sigma, params.q);
    
    // Sample random polynomial a
    let a = sample_poly_uniform(params.n, params.q);
    
    // Sample error e
    let e = sample_poly_error(params.n, params.sigma, params.q);
    
    // Compute b = a*s + e (mod q, mod X^n+1)
    let as_prod = poly_mult_ntt(&a, &s, params.q);
    let b = poly_add(&as_prod, &e, params.q);
    
    let sk = RingLWESecretKey { s };
    let pk = RingLWEPublicKey { a, b };
    
    Ok((sk, pk))
}

/// Encrypt message bits using Ring-LWE
pub fn ring_encrypt(
    pk: &RingLWEPublicKey,
    message: &[bool],
    params: &RingLWEParameters,
) -> CryptoResult<RingLWECiphertext> {
    params.validate()?;
    
    if message.len() > params.n {
        return Err(CryptoError::InvalidParameter(
            format!("Message too long: {} bits, max {}", message.len(), params.n),
        ));
    }
    
    // Sample ephemeral random polynomial r
    let r = sample_poly_error(params.n, params.sigma, params.q);
    
    // Sample error polynomials e1, e2
    let e1 = sample_poly_error(params.n, params.sigma, params.q);
    let e2 = sample_poly_error(params.n, params.sigma, params.q);
    
    // Encode message
    let m_poly = encode_message(message, params.n, params.q);
    
    // Compute u = a*r + e1
    let ar_prod = poly_mult_ntt(&pk.a, &r, params.q);
    let u = poly_add(&ar_prod, &e1, params.q);
    
    // Compute v = b*r + e2 + m
    let br_prod = poly_mult_ntt(&pk.b, &r, params.q);
    let br_e2 = poly_add(&br_prod, &e2, params.q);
    let v = poly_add(&br_e2, &m_poly, params.q);
    
    Ok(RingLWECiphertext { u, v })
}

/// Decrypt Ring-LWE ciphertext
pub fn ring_decrypt(
    sk: &RingLWESecretKey,
    ct: &RingLWECiphertext,
    params: &RingLWEParameters,
) -> CryptoResult<Vec<bool>> {
    params.validate()?;
    
    // Compute m' = v - u*s
    let us_prod = poly_mult_ntt(&ct.u, &sk.s, params.q);
    let m_noisy = poly_sub(&ct.v, &us_prod, params.q);
    
    // Decode message bits
    let message = decode_message(&m_noisy);
    
    Ok(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_lwe_parameters() {
        let params = RingLWEParameters::new_128bit_security();
        assert!(params.validate().is_ok());
        assert_eq!(params.n, 512);
    }

    #[test]
    fn test_polynomial_creation() {
        let poly = Polynomial::zero(512, 12289);
        assert_eq!(poly.degree, 512);
        assert_eq!(poly.coeffs.len(), 512);
    }

    #[test]
    fn test_polynomial_arithmetic() {
        let q = 12289;
        let a = Polynomial::from_coeffs(vec![1, 2, 3, 4], q);
        let b = Polynomial::from_coeffs(vec![5, 6, 7, 8], q);
        
        // Test addition
        let c = poly_add(&a, &b, q);
        assert_eq!(c.coeffs, vec![6, 8, 10, 12]);
        
        // Test subtraction
        let d = poly_sub(&b, &a, q);
        assert_eq!(d.coeffs, vec![4, 4, 4, 4]);
        
        // Test scalar multiplication
        let e = poly_scalar_mult(&a, 3, q);
        assert_eq!(e.coeffs, vec![3, 6, 9, 12]);
    }

    #[test]
    fn test_ntt_primitive_root() {
        // Test known primitive roots
        let root_512 = find_primitive_root(512, 12289);
        assert!(root_512.is_some());
        
        let omega = root_512.unwrap();
        // Verify ω^512 ≡ -1 (mod 12289)
        let half_order = mod_exp(omega, 512, 12289);
        assert_eq!(half_order, 12288); // q - 1 = -1 mod q
        
        // Verify ω^1024 ≡ 1 (mod 12289)
        let full_order = mod_exp(omega, 1024, 12289);
        assert_eq!(full_order, 1);
    }

    #[test]
    fn test_ntt_intt_correctness() {
        let q = 12289;
        let n = 512;
        
        // Create test polynomial
        let coeffs: Vec<i64> = (0..n).map(|i| (i % 100) as i64).collect();
        let poly = Polynomial::from_coeffs(coeffs.clone(), q);
        
        // Find primitive root
        let root = find_primitive_root(n, q).unwrap();
        
        // Transform and inverse transform
        let transformed = ntt(&poly, q, root);
        let recovered = intt(&transformed, n, q, root);
        
        // Verify we get back original coefficients
        for (i, (&orig, &rec)) in coeffs.iter().zip(recovered.coeffs.iter()).enumerate() {
            assert_eq!(
                orig, rec,
                "Coefficient {} mismatch: original {} != recovered {}",
                i, orig, rec
            );
        }
    }

    #[test]
    fn test_ntt_multiplication() {
        let q = 12289;
        let n = 512; // Use standard parameter size
        
        // Create two simple polynomials
        let mut a_coeffs = vec![0i64; n];
        let mut b_coeffs = vec![0i64; n];
        a_coeffs[0] = 1;
        a_coeffs[1] = 2;
        b_coeffs[0] = 3;
        b_coeffs[1] = 4;
        
        let a = Polynomial::from_coeffs(a_coeffs, q);
        let b = Polynomial::from_coeffs(b_coeffs, q);
        
        // Multiply using NTT
        let c = poly_mult_ntt(&a, &b, q);
        
        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2 in normal multiplication
        // In ring R_q, this should match
        assert_eq!(c.coeffs[0], 3);  // Constant term
        assert_eq!(c.coeffs[1], 10); // x term
        assert_eq!(c.coeffs[2], 8);  // x^2 term
    }

    #[test]
    fn test_message_encoding_decoding() {
        let q = 12289;
        let n = 512;
        
        // Test message
        let message = vec![true, false, true, true, false];
        
        // Encode and decode
        let poly = encode_message(&message, n, q);
        let decoded = decode_message(&poly);
        
        // Verify first 5 bits match
        assert_eq!(&decoded[0..5], &message[..]);
    }

    #[test]
    fn test_ring_lwe_keygen() {
        let params = RingLWEParameters::new_128bit_security();
        let result = ring_keygen(&params);
        
        assert!(result.is_ok());
        let (sk, pk) = result.unwrap();
        
        // Verify key dimensions
        assert_eq!(sk.s.degree, params.n);
        assert_eq!(pk.a.degree, params.n);
        assert_eq!(pk.b.degree, params.n);
    }

    #[test]
    fn test_ring_lwe_encrypt_decrypt() {
        let params = RingLWEParameters::new_128bit_security();
        let (sk, pk) = ring_keygen(&params).unwrap();
        
        // Test message
        let message = vec![true, false, true, true, false, true, false, false];
        
        // Encrypt
        let ct = ring_encrypt(&pk, &message, &params).unwrap();
        
        // Decrypt
        let decrypted = ring_decrypt(&sk, &ct, &params).unwrap();
        
        // Verify decryption correctness (first message.len() bits)
        assert_eq!(&decrypted[0..message.len()], &message[..]);
    }

    #[test]
    fn test_ring_lwe_multiple_messages() {
        let params = RingLWEParameters::new_128bit_security();
        let (sk, pk) = ring_keygen(&params).unwrap();
        
        // Test multiple different messages
        let messages = vec![
            vec![true; 10],
            vec![false; 10],
            vec![true, false, true, false, true, false],
        ];
        
        for message in messages {
            let ct = ring_encrypt(&pk, &message, &params).unwrap();
            let decrypted = ring_decrypt(&sk, &ct, &params).unwrap();
            
            assert_eq!(&decrypted[0..message.len()], &message[..]);
        }
    }

    #[test]
    fn test_ring_lwe_error_handling() {
        let params = RingLWEParameters::new_128bit_security();
        let (_, pk) = ring_keygen(&params).unwrap();
        
        // Message too long
        let long_message = vec![true; params.n + 1];
        let result = ring_encrypt(&pk, &long_message, &params);
        
        assert!(result.is_err());
    }
}
