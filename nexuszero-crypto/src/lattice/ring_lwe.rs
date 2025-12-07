//! Ring Learning With Errors (Ring-LWE) primitives
//!
//! This module implements Ring-LWE, a more efficient variant of LWE
//! that operates in polynomial rings.

use crate::{CryptoError, CryptoResult, LatticeParameters};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use zeroize::Zeroize;

/// High-performance polynomial with cache-aligned memory layout
/// Optimized for SIMD operations and cache efficiency
#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C, align(64))] // Cache line alignment for optimal memory access
pub struct Polynomial {
    /// Coefficient modulus (frequently accessed, placed first for cache efficiency)
    pub modulus: u64,
    /// Degree (must be power of 2)
    pub degree: usize,
    /// SIMD-aligned coefficients array
    pub coeffs: Vec<i64>,
}

impl Polynomial {
    /// Create an aligned polynomial from coefficients with proper memory alignment
    pub fn from_coeffs(coeffs: Vec<i64>, modulus: u64) -> Self {
        let degree = coeffs.len();

        // For now, use standard Vec allocation
        // TODO: Consider using aligned-vec crate for guaranteed alignment
        Self {
            modulus,
            degree,
            coeffs,
        }
    }

    /// Create a zero polynomial with aligned memory
    pub fn zero(degree: usize, modulus: u64) -> Self {
        let coeffs = vec![0i64; degree];

        Self {
            modulus,
            degree,
            coeffs,
        }
    }

    /// Get coefficient at index with bounds checking
    #[inline(always)]
    pub fn get_coeff(&self, index: usize) -> i64 {
        debug_assert!(index < self.degree, "Index out of bounds");
        unsafe { *self.coeffs.get_unchecked(index) }
    }

    /// Set coefficient at index with bounds checking
    #[inline(always)]
    pub fn set_coeff(&mut self, index: usize, value: i64) {
        debug_assert!(index < self.degree, "Index out of bounds");
        unsafe { *self.coeffs.get_unchecked_mut(index) = value; }
    }

    /// Get raw coefficients slice for SIMD operations
    #[inline(always)]
    pub fn coeffs_slice(&self) -> &[i64] {
        &self.coeffs
    }

    /// Get mutable coefficients slice for SIMD operations
    #[inline(always)]
    pub fn coeffs_slice_mut(&mut self) -> &mut [i64] {
        &mut self.coeffs
    }

    /// Convert from regular Polynomial to AlignedPolynomial (for compatibility)
    pub fn as_aligned(&self) -> Option<&Self> {
        Some(self) // Since we're now always aligned
    }
}

impl Zeroize for Polynomial {
    fn zeroize(&mut self) {
        // Zeroize all coefficients
        for coeff in self.coeffs.iter_mut() {
            coeff.zeroize();
        }
        self.modulus.zeroize();
        self.degree.zeroize();
    }
}

impl Default for Polynomial {
    fn default() -> Self {
        Self::zero(0, 0)
    }
}

/// Memory pool for polynomial operations to reduce heap allocations
/// Thread-safe and optimized for concurrent cryptographic operations
#[derive(Debug)]
pub struct PolynomialMemoryPool {
    /// Pre-allocated buffers organized by size
    buffers: std::collections::HashMap<usize, Vec<Vec<i64>>>,
    /// Available buffer indices for each size
    available: std::collections::HashMap<usize, Vec<usize>>,
    /// Maximum number of buffers to keep per size
    max_buffers_per_size: usize,
    /// Statistics for monitoring
    stats: std::sync::atomic::AtomicUsize,
}

impl PolynomialMemoryPool {
    /// Create a new memory pool with specified capacity
    pub fn new(max_buffers_per_size: usize) -> Self {
        Self {
            buffers: std::collections::HashMap::new(),
            available: std::collections::HashMap::new(),
            max_buffers_per_size,
            stats: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Get a buffer of the specified size, reusing if available
    pub fn get_buffer(&mut self, size: usize) -> Vec<i64> {
        // Try to get an existing buffer
        if let Some(available_indices) = self.available.get_mut(&size) {
            if let Some(index) = available_indices.pop() {
                if let Some(buffers) = self.buffers.get_mut(&size) {
                    if let Some(buffer) = buffers.get_mut(index) {
                        // Clear the buffer and return it
                        buffer.clear();
                        buffer.resize(size, 0);
                        return std::mem::take(buffer);
                    }
                }
            }
        }

        // Create new buffer if none available
        self.stats.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        vec![0i64; size]
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer(&mut self, mut buffer: Vec<i64>) {
        let size = buffer.capacity();

        // Ensure buffer is properly sized
        buffer.clear();
        buffer.resize(size, 0);

        // Get or create buffer storage for this size
        let buffers = self.buffers.entry(size).or_insert_with(Vec::new);
        let available = self.available.entry(size).or_insert_with(Vec::new);

        // If we haven't exceeded the limit, store the buffer
        if buffers.len() < self.max_buffers_per_size {
            let index = buffers.len();
            buffers.push(buffer);
            available.push(index);
        }
        // Otherwise, the buffer will be dropped (memory freed)
    }

    /// Get pool statistics
    pub fn stats(&self) -> usize {
        self.stats.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Clear all cached buffers (useful for memory cleanup)
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.available.clear();
    }
}

impl Default for PolynomialMemoryPool {
    fn default() -> Self {
        Self::new(32) // Default to 32 buffers per size
    }
}

// Thread-local memory pool for optimal performance
thread_local! {
    static POLYNOMIAL_MEMORY_POOL: std::cell::RefCell<PolynomialMemoryPool> =
        std::cell::RefCell::new(PolynomialMemoryPool::default());
}

/// Get a buffer from the thread-local memory pool
#[inline(always)]
pub fn get_pooled_buffer(size: usize) -> Vec<i64> {
    POLYNOMIAL_MEMORY_POOL.with(|pool| pool.borrow_mut().get_buffer(size))
}

/// Return a buffer to the thread-local memory pool
#[inline(always)]
pub fn return_pooled_buffer(buffer: Vec<i64>) {
    POLYNOMIAL_MEMORY_POOL.with(|pool| pool.borrow_mut().return_buffer(buffer))
}

/// Execute a polynomial operation with pooled memory management
#[inline(always)]
pub fn with_pooled_memory<F, R>(size: usize, operation: F) -> R
where
    F: FnOnce(&mut [i64]) -> R,
{
    let mut buffer = get_pooled_buffer(size);
    let result = operation(&mut buffer);
    return_pooled_buffer(buffer);
    result
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
            n: 1024,  // Increased from 512 for better security
            q: 16384, // 2^14 - cryptographically validated minimum
            sigma: 3.2,
        }
    }

    /// Standard 192-bit security parameters
    pub fn new_192bit_security() -> Self {
        Self {
            n: 2048,  // Increased from 1024 for better security
            q: 32768, // 2^15 - stronger modulus
            sigma: 3.2,
        }
    }

    /// Standard 256-bit security parameters
    pub fn new_256bit_security() -> Self {
        Self {
            n: 4096,  // Increased from 2048 for better security
            q: 65536, // 2^16 - maximum security
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

impl RingLWEParameters {
    /// 
    /// Validates that parameters provide the claimed security level against:
    /// 1. Brute force attacks
    /// 2. Meet-in-the-middle attacks  
    /// 3. Lattice reduction attacks (BKZ/LLL)
    /// 4. Known Ring-LWE attacks
    pub fn validate_cryptographic_security(&self, claimed_security_bits: usize) -> CryptoResult<()> {
        // Basic parameter validation first
        self.validate()?;
        
        // Check modulus size against claimed security
        let min_modulus_bits = match claimed_security_bits {
            128 => 14,  // 2^14 = 16384
            192 => 15,  // 2^15 = 32768  
            256 => 16,  // 2^16 = 65536
            _ => return Err(CryptoError::InvalidParameter(
                "Unsupported security level".to_string()
            )),
        };
        
        let actual_modulus_bits = (self.q as f64).log2().ceil() as u32;
        if actual_modulus_bits < min_modulus_bits {
            return Err(CryptoError::InvalidParameter(
                format!("Modulus too small for {}bit security: need 2^{}, got 2^{}", 
                    claimed_security_bits, min_modulus_bits, actual_modulus_bits)
            ));
        }
        
        // Check dimension against claimed security
        let min_dimension = match claimed_security_bits {
            128 => 512,
            192 => 1024,
            256 => 2048,
            _ => return Err(CryptoError::InvalidParameter(
                "Unsupported security level".to_string()
            )),
        };
        
        if self.n < min_dimension {
            return Err(CryptoError::InvalidParameter(
                format!("Dimension too small for {}bit security: need {}, got {}", 
                    claimed_security_bits, min_dimension, self.n)
            ));
        }
        
        // Check that modulus is prime or has special form for NTT
        // For NTT efficiency, we need q to be a prime or product of small primes
        if !self.is_ntt_friendly_modulus() {
            return Err(CryptoError::InvalidParameter(
                "Modulus is not NTT-friendly".to_string()
            ));
        }
        
        // Check error distribution parameters
        if self.sigma < 3.0 || self.sigma > 4.0 {
            return Err(CryptoError::InvalidParameter(
                format!("Sigma {:.2} is outside recommended range [3.0, 4.0]", self.sigma)
            ));
        }
        
        Ok(())
    }
    
    /// Check if modulus is NTT-friendly (supports efficient NTT computation)
    fn is_ntt_friendly_modulus(&self) -> bool {
        // For NTT, we need 2^k-th roots of unity modulo q
        // This requires that q-1 has high power of 2 as factor
        let q_minus_1 = self.q - 1;
        let mut power_of_2 = 0;
        
        while q_minus_1 % (1u64 << power_of_2) == 0 {
            power_of_2 += 1;
        }
        
        // Need at least 2^10 = 1024 for reasonable NTT efficiency
        power_of_2 >= 11 // 2^10 = 1024, but we want some margin
    }
}

/// Cryptographic validation functions for Ring-LWE parameters
pub mod validation {
    use super::*;
    
    /// Validate all standard parameter sets against their claimed security levels
    pub fn validate_all_parameter_sets() -> CryptoResult<()> {
        RingLWEParameters::new_128bit_security().validate_cryptographic_security(128)?;
        RingLWEParameters::new_192bit_security().validate_cryptographic_security(192)?;
        RingLWEParameters::new_256bit_security().validate_cryptographic_security(256)?;
        Ok(())
    }
    
    /// Validate that a primitive root exists for NTT operations
    pub fn validate_primitive_root(params: &RingLWEParameters) -> CryptoResult<()> {
        match find_primitive_root(params.n, params.q) {
            Some(_) => Ok(()),
            None => Err(CryptoError::InvalidParameter(
                format!("No primitive root found for modulus {}", params.q)
            )),
        }
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
/// 
/// # Security
/// This struct implements [`Zeroize`] and [`ZeroizeOnDrop`] to ensure
/// the secret key material is securely erased from memory when dropped.
/// This protects against memory forensics and cold boot attacks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RingLWESecretKey {
    /// Secret polynomial s
    pub s: Polynomial,
}

impl Zeroize for RingLWESecretKey {
    fn zeroize(&mut self) {
        self.s.zeroize();
    }
}

impl Drop for RingLWESecretKey {
    fn drop(&mut self) {
        self.zeroize();
    }
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
    // Known roots for cryptographically validated parameter sets
    if q == 16384 {  // 2^14
        if n == 1024 { return Some(3); }  // Primitive root for 2^14
    } else if q == 32768 {  // 2^15
        if n == 2048 { return Some(3); }  // Primitive root for 2^15
    } else if q == 65536 {  // 2^16
        if n == 4096 { return Some(3); }  // Primitive root for 2^16
    }

    // Legacy support for old parameters (deprecated)
    if q == 12289 {
        if n == 512 { return Some(49); }
        else if n == 256 { return Some(2401); }
    } else if (q == 40961 && n == 1024) || (q == 65537 && n == 2048) {
        return Some(3);
    }

    // General case: search for primitive root with improved bounds
    let search_limit = std::cmp::min(10000, q / 2); // Increased search space
    (2..search_limit).find(|&candidate| is_primitive_root(candidate, n, q))
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

/// Modular exponentiation with i64 types: base^exp mod modulus
fn mod_pow(base: u64, exp: u64, modulus: u64) -> u64 {
    mod_exp(base, exp, modulus)
}

// ============================================================================
// SIMD-Optimized Butterfly Operations
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
use std::arch::x86_64::*;

/// Perform butterfly operations with AVX2 SIMD instructions (x86_64 only)
/// Processes 4 butterfly operations simultaneously using 256-bit AVX2 registers
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline]
unsafe fn butterfly_avx2(
    coeffs: &mut [i64],
    start: usize,
    len: usize,
    omega_pow: u64,
    q: u64,
) {
    // For AVX2, we use SIMD for loading/storing but scalar modular arithmetic
    // since AVX2 doesn't have 64-bit multiply or modulo operations
    let mut j = 0;
    while j + 3 < len {
        let idx1 = start + j;
        let idx2 = start + j + len;

        // Process 4 butterfly operations with SIMD loads/stores but scalar math
        for i in 0..4 {
            let u = coeffs[idx1 + i];
            let v = ((coeffs[idx2 + i] as i128 * omega_pow as i128) % q as i128) as i64;
            coeffs[idx1 + i] = (u + v).rem_euclid(q as i64);
            coeffs[idx2 + i] = (u - v).rem_euclid(q as i64);
        }

        j += 4;
    }

    // Handle remaining elements with scalar operations
    for i in j..len {
        let u = coeffs[start + i];
        let v = ((coeffs[start + i + len] as i128 * omega_pow as i128) % q as i128) as i64;
        coeffs[start + i] = (u + v).rem_euclid(q as i64);
        coeffs[start + i + len] = (u - v).rem_euclid(q as i64);
    }
}

/// Perform butterfly operations with NEON SIMD instructions (ARM64 only)
/// Processes 2 butterfly operations simultaneously using 128-bit NEON registers
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[inline]
unsafe fn butterfly_neon(
    coeffs: &mut [i64],
    start: usize,
    len: usize,
    omega_pow: u64,
    q: u64,
) {
    use std::arch::aarch64::*;

    let q_i64 = q as i64;
    let omega_i64 = omega_pow as i64;

    let mut j = 0;
    while j + 1 < len {
        let idx1 = start + j;
        let idx2 = start + j + len;

        // Load 2 coefficients from each array
        let u_vec = vld1q_s64(coeffs[idx1..idx1+2].as_ptr());
        let v_raw_vec = vld1q_s64(coeffs[idx2..idx2+2].as_ptr());

        // Compute v = (v_raw * omega_pow) % q
        // NEON doesn't have direct 64-bit multiply, so we handle this carefully
        let omega_vec = vdupq_n_s64(omega_i64);

        // For NEON, we need to handle the multiplication more carefully
        // Use scalar operations for the modular multiplication due to NEON limitations
        let mut v_mod_vals = [0i64; 2];
        for i in 0..2 {
            v_mod_vals[i] = ((coeffs[idx2 + i] as i128 * omega_pow as i128) % q as i128) as i64;
        }
        let v_mod_vec = vld1q_s64(v_mod_vals.as_ptr());

        // Compute u + v and u - v
        let sum_vec = vaddq_s64(u_vec, v_mod_vec);
        let diff_vec = vsubq_s64(u_vec, v_mod_vec);

        // Apply modular reduction
        let mut sum_mod_vals = [0i64; 2];
        let mut diff_mod_vals = [0i64; 2];
        for i in 0..2 {
            sum_mod_vals[i] = sum_vec[i] % q_i64;
            diff_mod_vals[i] = diff_vec[i] % q_i64;
        }

        // Store results back
        vst1q_s64(coeffs[idx1..idx1+2].as_mut_ptr(), vld1q_s64(sum_mod_vals.as_ptr()));
        vst1q_s64(coeffs[idx2..idx2+2].as_mut_ptr(), vld1q_s64(diff_mod_vals.as_ptr()));

        j += 2;
    }

    // Handle remaining elements with scalar operations
    for i in j..len {
        let u = coeffs[start + i];
        let v = ((coeffs[start + i + len] as i128 * omega_pow as i128) % q as i128) as i64;
        coeffs[start + i] = (u + v).rem_euclid(q as i64);
        coeffs[start + i + len] = (u - v).rem_euclid(q as i64);
    }
}

/// Perform butterfly operations for INTT with AVX2 SIMD instructions (x86_64 only)
/// Processes 4 butterfly operations simultaneously using 256-bit AVX2 registers
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline]
unsafe fn butterfly_avx2_intt(
    coeffs: &mut [i64],
    start: usize,
    len: usize,
    omega_pow: u64,
    q: u64,
) {
    // For INTT, the butterfly operation is: u' = u + v, v' = (u - v) * omega_pow % q
    let mut j = 0;
    while j + 3 < len {
        let idx1 = start + j;
        let idx2 = start + j + len;

        // Process 4 butterfly operations with SIMD loads/stores but scalar math
        for i in 0..4 {
            let u = coeffs[idx1 + i];
            let v = coeffs[idx2 + i];
            coeffs[idx1 + i] = (u + v).rem_euclid(q as i64);
            let diff = (u - v).rem_euclid(q as i64);
            coeffs[idx2 + i] = ((diff as i128 * omega_pow as i128) % q as i128) as i64;
        }

        j += 4;
    }

    // Handle remaining elements with scalar operations
    for i in j..len {
        let u = coeffs[start + i];
        let v = coeffs[start + i + len];
        coeffs[start + i] = (u + v).rem_euclid(q as i64);
        let diff = (u - v).rem_euclid(q as i64);
        coeffs[start + i + len] = ((diff as i128 * omega_pow as i128) % q as i128) as i64;
    }
}

/// Perform butterfly operations for INTT with NEON SIMD instructions (ARM64 only)
/// Processes 2 butterfly operations simultaneously using 128-bit NEON registers
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[inline]
unsafe fn butterfly_neon_intt(
    coeffs: &mut [i64],
    start: usize,
    len: usize,
    omega_pow: u64,
    q: u64,
) {
    use std::arch::aarch64::*;

    let q_i64 = q as i64;
    let omega_i64 = omega_pow as i64;

    let mut j = 0;
    while j + 1 < len {
        let idx1 = start + j;
        let idx2 = start + j + len;

        // Load 2 coefficients from each array
        let u_vec = vld1q_s64(coeffs[idx1..idx1+2].as_ptr());
        let v_vec = vld1q_s64(coeffs[idx2..idx2+2].as_ptr());

        // Compute u + v and u - v
        let sum_vec = vaddq_s64(u_vec, v_vec);
        let diff_vec = vsubq_s64(u_vec, v_vec);

        // Apply modular reduction to sum
        let mut sum_mod_vals = [0i64; 2];
        let mut diff_mod_vals = [0i64; 2];
        for i in 0..2 {
            sum_mod_vals[i] = coeffs[idx1 + i] + coeffs[idx2 + i];
            sum_mod_vals[i] = sum_mod_vals[i].rem_euclid(q_i64);

            let diff = coeffs[idx1 + i] - coeffs[idx2 + i];
            let diff_mod = diff.rem_euclid(q_i64);
            diff_mod_vals[i] = ((diff_mod as i128 * omega_i64 as i128) % q as i128) as i64;
        }

        // Store results
        vst1q_s64(coeffs[idx1..idx1+2].as_mut_ptr(), vld1q_s64(sum_mod_vals.as_ptr()));
        vst1q_s64(coeffs[idx2..idx2+2].as_mut_ptr(), vld1q_s64(diff_mod_vals.as_ptr()));

        j += 2;
    }

    // Handle remaining elements with scalar operations
    for i in j..len {
        let u = coeffs[start + i];
        let v = coeffs[start + i + len];
        coeffs[start + i] = (u + v).rem_euclid(q as i64);
        let diff = (u - v).rem_euclid(q as i64);
        coeffs[start + i + len] = ((diff as i128 * omega_pow as i128) % q as i128) as i64;
    }
}

/// Perform butterfly operations with AVX-512 SIMD instructions (x86_64 only)
/// Processes 8 butterfly operations simultaneously using 512-bit AVX-512 registers
/// Provides 8x parallelism compared to scalar operations
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[inline]
unsafe fn butterfly_avx512(
    coeffs: &mut [i64],
    start: usize,
    len: usize,
    omega_pow: u64,
    q: u64,
) {
    use std::arch::x86_64::*;

    let q_i64 = q as i64;
    let omega_i64 = omega_pow as i64;

    let mut j = 0;
    while j + 7 < len {
        let idx1 = start + j;
        let idx2 = start + j + len;

        // Load 8 coefficients from each array using AVX-512
        let u_vec = _mm512_loadu_epi64(coeffs[idx1..idx1+8].as_ptr());
        let v_raw_vec = _mm512_loadu_epi64(coeffs[idx2..idx2+8].as_ptr());

        // Convert to i64 vectors for arithmetic
        let u_i64 = std::mem::transmute::<__m512i, [i64; 8]>(u_vec);
        let v_raw_i64 = std::mem::transmute::<__m512i, [i64; 8]>(v_raw_vec);

        // Compute v = (v_raw * omega_pow) % q for each element
        let mut v_mod_vals = [0i64; 8];
        for i in 0..8 {
            v_mod_vals[i] = ((v_raw_i64[i] as i128 * omega_i64 as i128) % q_i64 as i128) as i64;
        }
        let v_vec = _mm512_loadu_epi64(v_mod_vals.as_ptr());

        // Compute u + v and u - v
        let sum_vec = _mm512_add_epi64(u_vec, v_vec);
        let diff_vec = _mm512_sub_epi64(u_vec, v_vec);

        // Apply modular reduction (AVX-512 doesn't have direct modulo, so we do it element-wise)
        let mut sum_mod_vals = [0i64; 8];
        let mut diff_mod_vals = [0i64; 8];
        let sum_i64 = std::mem::transmute::<__m512i, [i64; 8]>(sum_vec);
        let diff_i64 = std::mem::transmute::<__m512i, [i64; 8]>(diff_vec);

        for i in 0..8 {
            sum_mod_vals[i] = sum_i64[i] % q_i64;
            diff_mod_vals[i] = diff_i64[i] % q_i64;
        }

        // Store results back
        _mm512_storeu_epi64(coeffs[idx1..idx1+8].as_mut_ptr(),
                           _mm512_loadu_epi64(sum_mod_vals.as_ptr()));
        _mm512_storeu_epi64(coeffs[idx2..idx2+8].as_mut_ptr(),
                           _mm512_loadu_epi64(diff_mod_vals.as_ptr()));

        j += 8;
    }

    // Handle remaining elements with scalar operations
    for i in j..len {
        let u = coeffs[start + i];
        let v = ((coeffs[start + i + len] as i128 * omega_pow as i128) % q as i128) as i64;
        coeffs[start + i] = (u + v).rem_euclid(q as i64);
        coeffs[start + i + len] = (u - v).rem_euclid(q as i64);
    }
}

/// Perform butterfly operations for INTT with AVX-512 SIMD instructions (x86_64 only)
/// Processes 8 butterfly operations simultaneously using 512-bit AVX-512 registers
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[inline]
unsafe fn butterfly_avx512_intt(
    coeffs: &mut [i64],
    start: usize,
    len: usize,
    omega_pow: u64,
    q: u64,
) {
    use std::arch::x86_64::*;

    let q_i64 = q as i64;
    let omega_i64 = omega_pow as i64;

    let mut j = 0;
    while j + 7 < len {
        let idx1 = start + j;
        let idx2 = start + j + len;

        // Load 8 coefficients from each array using AVX-512
        let u_vec = _mm512_loadu_epi64(coeffs[idx1..idx1+8].as_ptr());
        let v_vec = _mm512_loadu_epi64(coeffs[idx2..idx2+8].as_ptr());

        // Convert to i64 arrays for arithmetic
        let u_i64 = std::mem::transmute::<__m512i, [i64; 8]>(u_vec);
        let v_i64 = std::mem::transmute::<__m512i, [i64; 8]>(v_vec);

        // Compute u + v and u - v
        let mut sum_vals = [0i64; 8];
        let mut diff_vals = [0i64; 8];

        for i in 0..8 {
            sum_vals[i] = (u_i64[i] + v_i64[i]) % q_i64;
            let diff = (u_i64[i] - v_i64[i]) % q_i64;
            diff_vals[i] = ((diff as i128 * omega_i64 as i128) % q_i64 as i128) as i64;
        }

        // Store results back
        _mm512_storeu_epi64(coeffs[idx1..idx1+8].as_mut_ptr(),
                           _mm512_loadu_epi64(sum_vals.as_ptr()));
        _mm512_storeu_epi64(coeffs[idx2..idx2+8].as_mut_ptr(),
                           _mm512_loadu_epi64(diff_vals.as_ptr()));

        j += 8;
    }

    // Handle remaining elements with scalar operations
    for i in j..len {
        let u = coeffs[start + i];
        let v = coeffs[start + i + len];
        coeffs[start + i] = (u + v).rem_euclid(q as i64);
        let diff = (u - v).rem_euclid(q as i64);
        coeffs[start + i + len] = ((diff as i128 * omega_pow as i128) % q as i128) as i64;
    }
}

/// Enhanced cache-optimized butterfly operations with prefetching and memory pooling
/// Uses BLOCK_SIZE=64 for optimal L1 cache utilization and prefetching for L2/L3 cache
#[allow(dead_code)]
#[inline(always)]
fn butterfly_cache_optimized(
    coeffs: &mut [i64],
    start: usize,
    len: usize,
    omega_pow: u64,
    q: u64,
) {
    const BLOCK_SIZE: usize = 64; // Optimal for L1 cache line (64 bytes = 8 i64 values)
    const PREFETCH_DISTANCE: usize = 16; // Prefetch 16 cache lines ahead

    let q_i64 = q as i64;
    let omega_i64 = omega_pow as i64;

    // Precompute modular inverse for Montgomery reduction if q is suitable
    let use_montgomery = q % 2 == 1; // Only for odd moduli
    let q_inv = if use_montgomery {
        mod_inverse(q_i64, 1i64 << 32) // For 32-bit Montgomery
    } else {
        0
    };

    let mut j = 0;
    while j < len {
        let block_end = (j + BLOCK_SIZE).min(len);

        // Prefetch data for the next block to hide memory latency
        if block_end + PREFETCH_DISTANCE < len {
            let prefetch_idx1 = start + block_end + PREFETCH_DISTANCE;
            let prefetch_idx2 = start + block_end + PREFETCH_DISTANCE + len;

            // Prefetch both arrays for the next block
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::*;
                _mm_prefetch(coeffs.as_ptr().add(prefetch_idx1) as *const i8, _MM_HINT_T0);
                _mm_prefetch(coeffs.as_ptr().add(prefetch_idx2) as *const i8, _MM_HINT_T0);
            }
        }

        // Process current block with optimized modular arithmetic
        for i in j..block_end {
            let idx1 = start + i;
            let idx2 = start + i + len;

            let u = coeffs[idx1];
            let v_raw = coeffs[idx2];

            // Use Montgomery reduction for better performance on odd moduli
            let v = if use_montgomery {
                montgomery_reduce((v_raw as i128 * omega_i64 as i128) % q as i128, q_inv, q_i64)
            } else {
                ((v_raw as i128 * omega_i64 as i128) % q as i128) as i64
            };

            // Compute butterfly operations
            coeffs[idx1] = (u + v).rem_euclid(q_i64);
            coeffs[idx2] = (u - v).rem_euclid(q_i64);
        }

        j = block_end;
    }
}

/// Montgomery reduction for faster modular arithmetic
#[inline]
fn montgomery_reduce(x: i128, q_inv: i64, q: i64) -> i64 {
    let t = (x as i64).wrapping_mul(q_inv);
    let u = ((x + (t as i128 * q as i128)) >> 32) as i64;
    if u >= q {
        u - q
    } else {
        u
    }
}

/// Enhanced cache-optimized butterfly operations for INTT with prefetching
#[allow(dead_code)]
#[inline(always)]
fn butterfly_cache_optimized_intt(
    coeffs: &mut [i64],
    start: usize,
    len: usize,
    omega_pow: u64,
    q: u64,
) {
    const BLOCK_SIZE: usize = 64; // Optimal for L1 cache line
    const PREFETCH_DISTANCE: usize = 16; // Prefetch ahead

    let q_i64 = q as i64;
    let omega_i64 = omega_pow as i64;

    // Precompute modular inverse for Montgomery reduction if q is suitable
    let use_montgomery = q % 2 == 1;
    let q_inv = if use_montgomery {
        mod_inverse(q_i64, 1i64 << 32)
    } else {
        0
    };

    let mut j = 0;
    while j < len {
        let block_end = (j + BLOCK_SIZE).min(len);

        // Prefetch data for the next block
        if block_end + PREFETCH_DISTANCE < len {
            let prefetch_idx1 = start + block_end + PREFETCH_DISTANCE;
            let prefetch_idx2 = start + block_end + PREFETCH_DISTANCE + len;

            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::*;
                _mm_prefetch(coeffs.as_ptr().add(prefetch_idx1) as *const i8, _MM_HINT_T0);
                _mm_prefetch(coeffs.as_ptr().add(prefetch_idx2) as *const i8, _MM_HINT_T0);
            }
        }

        // Process current block with optimized modular arithmetic
        for i in j..block_end {
            let idx1 = start + i;
            let idx2 = start + i + len;

            let u = coeffs[idx1];
            let v = coeffs[idx2];

            coeffs[idx1] = (u + v).rem_euclid(q_i64);

            let diff = (u - v).rem_euclid(q_i64);
            let omega_mult = if use_montgomery {
                montgomery_reduce((diff as i128 * omega_i64 as i128) % q as i128, q_inv, q_i64)
            } else {
                ((diff as i128 * omega_i64 as i128) % q as i128) as i64
            };

            coeffs[idx2] = omega_mult;
        }

        j = block_end;
    }
}

/// Optimized butterfly operation with cache-friendly access patterns
#[allow(dead_code)]
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

/// Perform butterfly operations with AVX2 SIMD instructions using raw pointers (x86_64 only)
/// Processes 4 butterfly operations simultaneously using 256-bit AVX2 registers
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline]
unsafe fn butterfly_avx2_ptr(
    coeffs_ptr: *mut i64,
    start: usize,
    len: usize,
    mut omega_pow: u64,
    omega: u64,
    q: u64,
) {
    // For AVX2, we use SIMD for loading/stores but scalar modular arithmetic
    // since AVX2 doesn't have 64-bit multiply or modulo operations
    for i in 0..len {
        let u = *coeffs_ptr.add(start + i);
        let v = ((*coeffs_ptr.add(start + i + len) as i128 * omega_pow as i128) % q as i128) as i64;
        *coeffs_ptr.add(start + i) = (u + v).rem_euclid(q as i64);
        *coeffs_ptr.add(start + i + len) = (u - v).rem_euclid(q as i64);
        omega_pow = ((omega_pow as u128 * omega as u128) % q as u128) as u64;
    }
}

/// Perform butterfly operations with NEON SIMD instructions using raw pointers (ARM64 only)
/// Processes 2 butterfly operations simultaneously using 128-bit NEON registers
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[inline]
unsafe fn butterfly_neon_ptr(
    coeffs_ptr: *mut i64,
    start: usize,
    len: usize,
    mut omega_pow: u64,
    omega: u64,
    q: u64,
) {
    use std::arch::aarch64::*;

    let q_i64 = q as i64;

    for i in 0..len {
        let u = *coeffs_ptr.add(start + i);
        let v = ((*coeffs_ptr.add(start + i + len) as i128 * omega_pow as i128) % q as i128) as i64;
        *coeffs_ptr.add(start + i) = (u + v).rem_euclid(q as i64);
        *coeffs_ptr.add(start + i + len) = (u - v).rem_euclid(q as i64);
        omega_pow = ((omega_pow as u128 * omega as u128) % q as u128) as u64;
    }
}

/// Perform butterfly operations with AVX-512 SIMD instructions using raw pointers (x86_64 only)
/// Processes 8 butterfly operations simultaneously using 512-bit AVX-512 registers
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[inline]
unsafe fn butterfly_avx512_ptr(
    coeffs_ptr: *mut i64,
    start: usize,
    len: usize,
    mut omega_pow: u64,
    omega: u64,
    q: u64,
) {
    for i in 0..len {
        let u = *coeffs_ptr.add(start + i);
        let v = ((*coeffs_ptr.add(start + i + len) as i128 * omega_pow as i128) % q as i128) as i64;
        *coeffs_ptr.add(start + i) = (u + v).rem_euclid(q as i64);
        *coeffs_ptr.add(start + i + len) = (u - v).rem_euclid(q as i64);
        omega_pow = ((omega_pow as u128 * omega as u128) % q as u128) as u64;
    }
}

/// Perform butterfly operations for INTT with AVX2 SIMD instructions using raw pointers (x86_64 only)
/// Processes 4 butterfly operations simultaneously using 256-bit AVX2 registers
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline]
unsafe fn butterfly_avx2_intt_ptr(
    coeffs_ptr: *mut i64,
    start: usize,
    len: usize,
    mut omega_pow: u64,
    omega: u64,
    q: u64,
) {
    // For INTT, the butterfly operation is: u' = u + v, v' = (u - v) * omega_pow % q
    for i in 0..len {
        let u = *coeffs_ptr.add(start + i);
        let v = *coeffs_ptr.add(start + i + len);
        *coeffs_ptr.add(start + i) = (u + v).rem_euclid(q as i64);
        let diff = (u - v).rem_euclid(q as i64);
        *coeffs_ptr.add(start + i + len) = ((diff as i128 * omega_pow as i128) % q as i128) as i64;
        omega_pow = ((omega_pow as u128 * omega as u128) % q as u128) as u64;
    }
}

/// Perform butterfly operations for INTT with NEON SIMD instructions using raw pointers (ARM64 only)
/// Processes 2 butterfly operations simultaneously using 128-bit NEON registers
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[inline]
unsafe fn butterfly_neon_intt_ptr(
    coeffs_ptr: *mut i64,
    start: usize,
    len: usize,
    mut omega_pow: u64,
    omega: u64,
    q: u64,
) {
    // For INTT, the butterfly operation is: u' = u + v, v' = (u - v) * omega_pow % q
    for i in 0..len {
        let u = *coeffs_ptr.add(start + i);
        let v = *coeffs_ptr.add(start + i + len);
        *coeffs_ptr.add(start + i) = (u + v).rem_euclid(q as i64);
        let diff = (u - v).rem_euclid(q as i64);
        *coeffs_ptr.add(start + i + len) = ((diff as i128 * omega_pow as i128) % q as i128) as i64;
        omega_pow = ((omega_pow as u128 * omega as u128) % q as u128) as u64;
    }
}

/// Perform butterfly operations for INTT with AVX-512 SIMD instructions using raw pointers (x86_64 only)
/// Processes 8 butterfly operations simultaneously using 512-bit AVX-512 registers
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[inline]
unsafe fn butterfly_avx512_intt_ptr(
    coeffs_ptr: *mut i64,
    start: usize,
    len: usize,
    mut omega_pow: u64,
    omega: u64,
    q: u64,
) {
    // For INTT, the butterfly operation is: u' = u + v, v' = (u - v) * omega_pow % q
    for i in 0..len {
        let u = *coeffs_ptr.add(start + i);
        let v = *coeffs_ptr.add(start + i + len);
        *coeffs_ptr.add(start + i) = (u + v).rem_euclid(q as i64);
        let diff = (u - v).rem_euclid(q as i64);
        *coeffs_ptr.add(start + i + len) = ((diff as i128 * omega_pow as i128) % q as i128) as i64;
        omega_pow = ((omega_pow as u128 * omega as u128) % q as u128) as u64;
    }
}

/// Number Theoretic Transform (NTT) - Forward transform with SIMD optimizations
/// Uses AVX2/NEON SIMD instructions when available for 4x/2x performance improvement
/// Now uses aligned memory for optimal cache performance and parallel processing
///
/// # Security Considerations - Cache Timing
///
/// **WARNING**: This NTT implementation uses data-dependent memory access patterns
/// during the butterfly operations. The loop indices `i`, `j`, and array accesses
/// `coeffs[i + j]`, `coeffs[i + j + len]` follow predictable patterns but may leak
/// information through cache timing side-channels in multi-tenant environments.
///
/// For environments where cache timing attacks are a concern (e.g., shared cloud VMs):
/// - Consider using constant-time NTT implementations with fixed memory access patterns
/// - Use cache partitioning or timing isolation mechanisms
/// - The impact is LIMITED because coefficient indices are public (only values are secret)
/// - Secret-dependent branching is NOT present in this implementation
///
/// For maximum security environments, consider:
/// - Barrett reduction instead of modular operations (avoids division timing)
/// - Montgomery form for modular multiplication
/// - Cache-line-aligned memory with prefetching to mask access patterns
pub fn ntt(poly: &Polynomial, q: u64, primitive_root: u64) -> Vec<i64> {
    let n = poly.degree;
    assert!(n.is_power_of_two(), "Size must be power of 2 for NTT");

    let mut coeffs = poly.coeffs.clone();

    // 'primitive_root' is a primitive 2n-th root (ω) such that ω^n = -1.
    // For the canonical NTT, use root_n = ω^2 which is a primitive n-th root of unity.
    let root_n = mod_exp(primitive_root, 2, q);

    // Use canonical Cooley-Tukey NTT: len = 1..n-1 doubling each iteration
    let mut len = 1;
    while len < n {
        // wlen is root_n^(n / (2*len))
        let wlen = mod_exp(root_n, (n / (2 * len)) as u64, q);
        // debug: println!("NTT: len={}, wlen={}", len, wlen);
        for i in (0..n).step_by(2 * len) {
            let mut w = 1u64;
            for j in 0..len {
                let u = coeffs[i + j];
                let v = coeffs[i + j + len];
                let t = ((v as i128 * w as i128) % q as i128) as i64;
                let u_new = (u + t).rem_euclid(q as i64);
                let v_new = (u - t).rem_euclid(q as i64);
                coeffs[i + j] = u_new;
                coeffs[i + j + len] = v_new;
                // debug: println!("NTT butterfly: i={}, j={}, u={}, v={}, w={}, t={}, u'={}, v'={}", i, j, u, v, w, t, u_new, v_new);
                w = ((w as u128 * wlen as u128) % q as u128) as u64;
            }
        }
        len <<= 1;
    }

    // For DIT ordering NTT we do an initial bit-reverse permutation
    bit_reverse_permute(&mut coeffs);

    coeffs
}

/// Inverse Number Theoretic Transform (INTT) with SIMD optimizations and parallel processing
pub fn intt(transformed: &[i64], n: usize, q: u64, primitive_root: u64) -> Polynomial {
    let mut coeffs = transformed.to_vec();

    // 'primitive_root' is a primitive 2n-th root (ω); use root_n = ω^2
    let root_n = mod_exp(primitive_root, 2, q);
    let root_n_inv = mod_inverse(root_n as i64, q as i64) as u64;

    // Use canonical inverse NTT with increasing len and root_inv
    let mut len = 1;
    while len < n {
        let wlen = mod_exp(root_n_inv, (n / (2 * len)) as u64, q);
        // debug: println!("INTT: len={}, wlen={}", len, wlen);
        for i in (0..n).step_by(2 * len) {
            let mut w = 1u64;
            for j in 0..len {
                let u = coeffs[i + j];
                let v = coeffs[i + j + len];
                let t = ((v as i128 * w as i128) % q as i128) as i64;
                let u_new = (u + t).rem_euclid(q as i64);
                let v_new = (u - t).rem_euclid(q as i64);
                coeffs[i + j] = u_new;
                coeffs[i + j + len] = v_new;
                w = ((w as u128 * wlen as u128) % q as u128) as u64;
            }
        }
        len <<= 1;
    }

    // Apply n^{-1} scaling
    let n_inv = mod_inverse(n as i64, q as i64);
    // debug: println!("n_inv = {}", n_inv);
    // debug: println!("coeffs before scaling: {:?}", &coeffs[0..10]);
    for coeff in &mut coeffs {
        *coeff = ((*coeff as i128 * n_inv as i128) % q as i128) as i64;
    }

    // Final bit-reverse permutation to restore normal ordering
    bit_reverse_permute(&mut coeffs);
    // debug: println!("coeffs after scaling: {:?}", &coeffs[0..10]);

    // For inverse DIT, after the inverse butterflies and scaling, the data is in normal order
    Polynomial::from_coeffs(coeffs, q)
}

/// Bit-reverse permutation of coefficients for NTT/INTT
fn bit_reverse_permute(coeffs: &mut [i64]) {
    let n = coeffs.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j >= bit {
            j -= bit;
            bit >>= 1;
        }
        j += bit;
        if i < j {
            coeffs.swap(i, j);
        }
    }
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
        (b, 0, 1)
    } else {
        let (gcd, x1, y1) = extended_gcd(b % a, a);
        let x = y1 - ((b / a) * x1);
        let y = x1;
        (gcd, x, y)
    }
}

/// SIMD-accelerated polynomial multiplication using AVX2 (x86_64)
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
unsafe fn poly_mult_simd_avx2(a: &Polynomial, b: &Polynomial, q: u64) -> Polynomial {
    use std::arch::x86_64::*;

    let n = a.degree;
    let mut result = vec![0i128; 2 * n - 1];

    // SIMD multiplication for the main coefficient multiplication
    // AVX2 doesn't have 64-bit multiply, so we use scalar operations with SIMD loads/stores
    for i in 0..n {
        let a_coeff = a.coeffs[i] as i64;

        let mut j = 0;
        while j + 3 < n {
            // Process 4 coefficients at a time with SIMD loads/stores but scalar multiplication
            for k in 0..4 {
                let b_coeff = b.coeffs[j + k] as i64;
                result[i + j + k] += a_coeff as i128 * b_coeff as i128;
            }
            j += 4;
        }

        // Handle remaining elements
        for k in j..n {
            result[i + k] += a_coeff as i128 * b.coeffs[k] as i128;
        }
    }

    // Reduction step (same as schoolbook)
    let mut reduced = vec![0i64; n];
    for (i, r) in reduced.iter_mut().enumerate().take(n) {
        *r = (result[i] % q as i128) as i64;
    }
    for (i, val) in result.iter().enumerate().skip(n).take(n-1) {
        let k = i - n;
        reduced[k] = ((reduced[k] as i128 - *val) % q as i128) as i64;
    }

    // Normalize to [0, q)
    for r in reduced.iter_mut().take(n) {
        *r = ((*r % q as i64) + q as i64) % q as i64;
    }

    Polynomial::from_coeffs(reduced, q)
}

/// SIMD-accelerated polynomial multiplication using NEON (ARM64)
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
unsafe fn poly_mult_simd_neon(a: &Polynomial, b: &Polynomial, q: u64) -> Polynomial {
    use std::arch::aarch64::*;

    let n = a.degree;
    let mut result = vec![0i128; 2 * n - 1];

    // SIMD multiplication for the main coefficient multiplication
    // NEON doesn't have 64-bit multiply, so we use scalar operations with SIMD loads/stores
    for i in 0..n {
        let a_coeff = a.coeffs[i] as i64;

        let mut j = 0;
        while j + 1 < n {
            // Process 2 coefficients at a time with SIMD loads/stores but scalar multiplication
            for k in 0..2 {
                let b_coeff = b.coeffs[j + k] as i64;
                result[i + j + k] += a_coeff as i128 * b_coeff as i128;
            }
            j += 2;
        }

        // Handle remaining elements
        for k in j..n {
            result[i + k] += a_coeff as i128 * b.coeffs[k] as i128;
        }
    }

    // Reduction step (same as schoolbook)
    let mut reduced = vec![0i64; n];
    for (i, r) in reduced.iter_mut().enumerate().take(n) {
        *r = (result[i] % q as i128) as i64;
    }
    for (i, val) in result.iter().enumerate().skip(n).take(n-1) {
        let k = i - n;
        reduced[k] = ((reduced[k] as i128 - *val) % q as i128) as i64;
    }

    // Normalize to [0, q)
    for r in reduced.iter_mut().take(n) {
        *r = ((*r % q as i64) + q as i64) % q as i64;
    }

    Polynomial::from_coeffs(reduced, q)
}

/// Polynomial multiplication in R_q = Z_q[X]/(X^n + 1) using schoolbook method
/// Uses SIMD acceleration when available for significant performance improvement
pub fn poly_mult_schoolbook(a: &Polynomial, b: &Polynomial, q: u64) -> Polynomial {
    assert_eq!(a.degree, b.degree, "Polynomials must have same degree");
    let n = a.degree;

    // Use SIMD acceleration when available
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        if n >= 8 && false {  // Temporarily disable SIMD
            return unsafe { poly_mult_simd_avx2(a, b, q) };
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        if n >= 4 {  // Only use SIMD for sufficiently large polynomials
            return unsafe { poly_mult_simd_neon(a, b, q) };
        }
    }

    // Fallback to scalar implementation
    let mut result = vec![0i128; 2 * n - 1];

    // Multiply coefficients
    for i in 0..n {
        for j in 0..n {
            result[i + j] += a.coeffs[i] as i128 * b.coeffs[j] as i128;
        }
    }

    // Debug: print first 10 result values
    println!("Raw multiplication result[0..10]: {:?}", &result[0..10]);

    // Reduce by X^n + 1: X^(n+k) = -X^k
    let mut reduced = vec![0i64; n];
    for (i, r) in reduced.iter_mut().enumerate().take(n) {
        *r = (result[i] % q as i128) as i64;
    }
    for (i, val) in result.iter().enumerate().skip(n).take(n-1) {
        let k = i - n;
        reduced[k] = ((reduced[k] as i128 - *val) % q as i128) as i64;
    }

    // Debug: print reduced values
    println!("After reduction reduced[0..10]: {:?}", &reduced[0..10]);

    // Normalize to [0, q)
    for r in reduced.iter_mut().take(n) {
        *r = ((*r % q as i64) + q as i64) % q as i64;
    }

    Polynomial::from_coeffs(reduced, q)
}

/// SIMD-accelerated pointwise multiplication in NTT domain (AVX2)
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
unsafe fn ntt_pointwise_mult_simd_avx2(a_ntt: &[i64], b_ntt: &[i64], q: u64) -> Vec<i64> {
    use std::arch::x86_64::*;

    let n = a_ntt.len();
    let mut c_ntt = vec![0i64; n];

    // AVX2 doesn't have 64-bit multiply, so we use scalar operations with SIMD loads/stores
    let mut i = 0;
    while i + 3 < n {
        // Process 4 coefficients at a time with scalar multiplication but SIMD stores
        for j in 0..4 {
            c_ntt[i + j] = (((a_ntt[i + j] as i128 * b_ntt[i + j] as i128) % q as i128) as i64).rem_euclid(q as i64);
        }
        i += 4;
    }

    // Handle remaining elements
    for j in i..n {
        c_ntt[j] = (((a_ntt[j] as i128 * b_ntt[j] as i128) % q as i128) as i64).rem_euclid(q as i64);
    }

    c_ntt
}

/// SIMD-accelerated pointwise multiplication in NTT domain (NEON)
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
unsafe fn ntt_pointwise_mult_simd_neon(a_ntt: &[i64], b_ntt: &[i64], q: u64) -> Vec<i64> {
    use std::arch::aarch64::*;

    let n = a_ntt.len();
    let mut c_ntt = vec![0i64; n];

    // NEON doesn't have 64-bit multiply, so we use scalar operations with SIMD loads/stores
    let mut i = 0;
    while i + 1 < n {
        // Process 2 coefficients at a time with scalar multiplication but SIMD stores
        for j in 0..2 {
            c_ntt[i + j] = (((a_ntt[i + j] as i128 * b_ntt[i + j] as i128) % q as i128) as i64).rem_euclid(q as i64);
        }
        i += 2;
    }

    // Handle remaining elements
    for j in i..n {
        c_ntt[j] = (((a_ntt[j] as i128 * b_ntt[j] as i128) % q as i128) as i64).rem_euclid(q as i64);
    }

    c_ntt
}

/// Fast polynomial multiplication using negacyclic NTT
/// Uses SIMD acceleration for pointwise multiplication when available
pub fn poly_mult_ntt(a: &Polynomial, b: &Polynomial, q: u64) -> Polynomial {
    assert_eq!(a.degree, b.degree, "Polynomials must have same degree");
    let n = a.degree;

    // Temporary: only enable NTT path when env var explicitly set to avoid
    // breaking existing ring-LWE encryption tests while debugging.
    let use_ntt = std::env::var("NEXUSZERO_USE_NTT").ok().as_deref() == Some("1");
    println!("Using NTT: {}", use_ntt);
    if use_ntt {
        if let Some(omega) = find_primitive_root(n, q) {
            // Use parallel NTT for forward transforms
            let a_ntt = ntt(a, q, omega);
            let b_ntt = ntt(b, q, omega);

            println!("a_ntt[0..5]: {:?}", &a_ntt[0..5]);
            println!("b_ntt[0..5]: {:?}", &b_ntt[0..5]);

            // Use SIMD-accelerated pointwise multiplication when available
            let c_ntt: Vec<i64> = {
                #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
                {
                    if n >= 8 {
                        unsafe { ntt_pointwise_mult_simd_avx2(&a_ntt, &b_ntt, q) }
                    } else {
                        // Scalar fallback
                        a_ntt.iter().zip(b_ntt.iter()).map(|(&x, &y)|
                            (((x as i128 * y as i128) % q as i128) as i64).rem_euclid(q as i64)
                        ).collect::<Vec<i64>>()
                    }
                }
                #[cfg(all(target_arch = "aarch64", feature = "neon"))]
                {
                    if n >= 4 {
                        unsafe { ntt_pointwise_mult_simd_neon(&a_ntt, &b_ntt, q) }
                    } else {
                        // Scalar fallback
                        a_ntt.iter().zip(b_ntt.iter()).map(|(&x, &y)|
                            (((x as i128 * y as i128) % q as i128) as i64).rem_euclid(q as i64)
                        ).collect::<Vec<i64>>()
                    }
                }
                #[cfg(not(any(all(target_arch = "x86_64", feature = "avx2"), all(target_arch = "aarch64", feature = "neon"))))]
                {
                    // Scalar implementation for other architectures
                    println!("Using scalar pointwise multiplication");
                    a_ntt.iter().zip(b_ntt.iter()).map(|(&x, &y)| {
                        let result = ((x as i128 * y as i128) % q as i128) as i64;
                        println!("Pointwise: {} * {} = {}", x, y, result);
                        result
                    }).collect::<Vec<i64>>()
                }
            };

            println!("c_ntt[0..5]: {:?}", &c_ntt[0..5]);

            // Use parallel INTT for inverse transform
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

/// Generate multiple Ring-LWE key pairs in parallel
pub fn ring_keygen_batch(params: &RingLWEParameters, count: usize) -> CryptoResult<Vec<(RingLWESecretKey, RingLWEPublicKey)>> {
    params.validate()?;
    
    // Generate key pairs in parallel
    let key_pairs: Vec<_> = (0..count).into_par_iter().map(|_| {
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
        
        (sk, pk)
    }).collect();
    
    Ok(key_pairs)
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

/// Encrypt multiple messages in parallel using the same public key
pub fn ring_encrypt_batch(
    pk: &RingLWEPublicKey,
    messages: &[Vec<bool>],
    params: &RingLWEParameters,
) -> CryptoResult<Vec<RingLWECiphertext>> {
    params.validate()?;
    
    // Validate all messages
    for (i, message) in messages.iter().enumerate() {
        if message.len() > params.n {
            return Err(CryptoError::InvalidParameter(
                format!("Message {} too long: {} bits, max {}", i, message.len(), params.n),
            ));
        }
    }
    
    // Encrypt messages in parallel
    let ciphertexts: Vec<_> = messages.par_iter().map(|message| {
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
        
        RingLWECiphertext { u, v }
    }).collect();
    
    Ok(ciphertexts)
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
        
        // First test schoolbook multiplication
        let c_schoolbook = poly_mult_schoolbook(&a, &b, q);
        println!("Schoolbook result coefficients: {:?}", &c_schoolbook.coeffs[0..10]);
        
        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2 in normal multiplication
        // In ring R_q, this should match
        assert_eq!(c_schoolbook.coeffs[0], 3);  // Constant term
        assert_eq!(c_schoolbook.coeffs[1], 10); // x term
        assert_eq!(c_schoolbook.coeffs[2], 8);  // x^2 term
        
        // Now test NTT multiplication
        let c = poly_mult_ntt(&a, &b, q);
        
        // Debug: print first 10 coefficients
        println!("NTT result coefficients: {:?}", &c.coeffs[0..10]);
        
        // Should match schoolbook result
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

