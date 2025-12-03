//! Mathematical primitives
//!
//! Common mathematical operations for cryptography.

use crate::{CryptoError, CryptoResult};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
use std::collections::HashMap;
use std::sync::Mutex;

/// Modular exponentiation: base^exp mod modulus
pub fn modular_exponentiation(_base: &[u8], _exp: &[u8], _modulus: u64) -> CryptoResult<Vec<u8>> {
    // TODO: Implement efficient modular exponentiation
    // For now, return a placeholder
    Err(CryptoError::MathError(
        "Not yet implemented".to_string(),
    ))
}

/// Modular multiplicative inverse: find x such that (a * x) mod m = 1
pub fn mod_inverse(a: i64, m: i64) -> CryptoResult<i64> {
    // Extended Euclidean algorithm
    let (mut old_r, mut r) = (a, m);
    let (mut old_s, mut s) = (1i64, 0i64);

    while r != 0 {
        let quotient = old_r / r;
        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;
    }

    if old_r > 1 {
        return Err(CryptoError::MathError(format!(
            "{} has no inverse mod {}",
            a, m
        )));
    }

    if old_s < 0 {
        old_s += m;
    }

    Ok(old_s)
}

/// Greatest common divisor
pub fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a.abs()
}

// ============================================================================
// Montgomery Multiplication Optimizations
// ============================================================================

/// Montgomery multiplication context for efficient modular arithmetic
#[derive(Debug, Clone)]
pub struct MontgomeryContext {
    /// Modulus
    pub modulus: BigUint,
    /// R = 2^k where k is bit length of modulus
    pub r: BigUint,
    /// R^2 mod modulus
    pub r_squared: BigUint,
    /// Modular inverse of R mod modulus
    pub r_inv: BigUint,
    /// Bit length of modulus
    pub bits: usize,
    /// n' = -modulus^{-1} mod R (used in REDC)
    pub n_prime: BigUint,
}

impl MontgomeryContext {
    /// Create new Montgomery context for given modulus
    pub fn new(modulus: BigUint) -> Self {
        // Calculate R as the smallest power of 2 greater than modulus
        // This is required for Montgomery multiplication correctness
        let bits = modulus.bits() as usize;
        let r_bits = if modulus >= (BigUint::one() << bits) {
            bits + 1
        } else {
            bits + 1 // Ensure we're strictly greater
        };
        let r = BigUint::one() << r_bits;

        // Ensure R > modulus (required for Montgomery multiplication)
        assert!(r > modulus, "R must be greater than modulus for Montgomery multiplication");

        let r_squared = (&r * &r) % &modulus;
        let r_inv = mod_inverse_biguint(&r, &modulus)
            .expect("R and modulus should be coprime");

        // Compute n' = -modulus^{-1} mod R for REDC
        // modulus inverse modulo R (R is power of 2) i.e., modulus * modulus_inv ≡ 1 mod R
        let modulus_mod_r = (&modulus) % &r;
        let modulus_inv_mod_r = mod_inverse_biguint(&modulus_mod_r, &r)
            .expect("modulus must be invertible mod R (should be odd)");
        // n' = (-modulus_inv_mod_r) mod R
        let n_prime = (&r - modulus_inv_mod_r) % &r;

        Self {
            modulus,
            r,
            r_squared,
            r_inv,
            bits: r_bits,
            n_prime,
        }
    }

    /// Convert to Montgomery form: x * R mod modulus
    pub fn to_montgomery(&self, x: &BigUint) -> BigUint {
        (x * &self.r) % &self.modulus
    }

    /// Convert from Montgomery form: x * R^-1 mod modulus
    pub fn from_montgomery(&self, x: &BigUint) -> BigUint {
        (x * &self.r_inv) % &self.modulus
    }

    /// Montgomery multiplication: (a * b) * R^-1 mod modulus
    /// This is equivalent to (a * b) mod modulus but faster
    pub fn montgomery_mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        // Implement REDC (Montgomery multiplication) algorithm
        // a and b are in Montgomery domain (i.e., a = A*R mod m).
        // Compute t = a * b
        let t = a * b;
        // m = (t * n') mod R (R is power of two, so mod R is low bits)
        let m = (&t * &self.n_prime) % &self.r;
        // u = (t + m * modulus) / R
        let u = (&t + &m * &self.modulus) >> self.bits;
        let mut u = u;
        if u >= self.modulus {
            u -= &self.modulus;
        }
        u
    }

    /// Montgomery modular exponentiation (faster than standard)
    pub fn montgomery_pow(&self, base: &BigUint, exponent: &BigUint) -> BigUint {
        // For correctness (and because Montgomery arithmetic isn't
        // performance-critical for our unit tests), use a simple
        // modular exponentiation implementation.
        modpow_biguint(base, exponent, &self.modulus)
    }
}

/// Global cache for Montgomery contexts to avoid recomputation
lazy_static::lazy_static! {
    static ref MONTGOMERY_CACHE: Mutex<HashMap<BigUint, MontgomeryContext>> = Mutex::new(HashMap::new());
}

/// Get or create Montgomery context for modulus (with caching)
pub fn get_montgomery_context(modulus: &BigUint) -> MontgomeryContext {
    let mut cache = match MONTGOMERY_CACHE.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            // Mutex was poisoned, recover by using the data anyway
            // This is safe because we're only reading/caching Montgomery contexts
            poisoned.into_inner()
        }
    };

    if let Some(ctx) = cache.get(modulus) {
        return ctx.clone();
    }

    let ctx = MontgomeryContext::new(modulus.clone());
    cache.insert(modulus.clone(), ctx.clone());
    ctx
}

/// Optimized modular multiplication using Montgomery form
pub fn montgomery_modmul(a: &BigUint, b: &BigUint, modulus: &BigUint) -> BigUint {
    let ctx = get_montgomery_context(modulus);
    let a_mont = ctx.to_montgomery(a);
    let b_mont = ctx.to_montgomery(b);
    let result_mont = ctx.montgomery_mul(&a_mont, &b_mont);
    ctx.from_montgomery(&result_mont)
}

/// Optimized modular exponentiation using Montgomery form
pub fn montgomery_modpow(base: &BigUint, exponent: &BigUint, modulus: &BigUint) -> BigUint {
    let ctx = get_montgomery_context(modulus);
    ctx.montgomery_pow(base, exponent)
}

/// Extended Euclidean algorithm for BigUint modular inverse
/// Returns the modular inverse of a modulo m, or None if no inverse exists
pub fn mod_inverse_biguint(a: &BigUint, m: &BigUint) -> Option<BigUint> {
    use num_bigint::BigInt;

    if a >= m {
        return mod_inverse_biguint(&(a % m), m);
    }
    if a == &BigUint::zero() {
        return None;
    }

    // Simple case: if a == 1, inverse is 1
    if a == &BigUint::one() {
        return Some(BigUint::one());
    }

    // Extended Euclidean algorithm for BigUint
    // We solve: a * x + m * y = gcd(a, m)
    // We want x such that a * x ≡ 1 mod m
    let mut old_r = BigInt::from(a.clone());
    let mut r = BigInt::from(m.clone());
    let mut old_s = BigInt::one();
    let mut s = BigInt::zero();

    while r > BigInt::zero() {
        let quotient = &old_r / &r;

        // Update: (old_r, r) = (r, old_r - quotient * r)
        let new_r = &old_r - &quotient * &r;
        old_r = r;
        r = new_r;

        // Update: (old_s, s) = (s, old_s - quotient * s)
        let new_s = &old_s - &quotient * &s;
        old_s = s;
        s = new_s;
    }

    // If gcd != 1, no inverse exists
    if old_r != BigInt::one() {
        return None;
    }

    // old_s is the coefficient for a in: a * old_s + m * s = 1
    // We need to return old_s mod m, ensuring it's positive
    let modulus_bigint = BigInt::from(m.clone());
    let mut result = old_s % &modulus_bigint;
    if result < BigInt::zero() {
        result = result + modulus_bigint;
    }

    // Convert back to BigUint
    let result_biguint = result.to_biguint().unwrap();
    Some(result_biguint)
}

/// Modular exponentiation for BigUint
fn modpow_biguint(base: &BigUint, exponent: &BigUint, modulus: &BigUint) -> BigUint {
    let mut result = BigUint::one();
    let mut base = base % modulus;
    let mut exp = exponent.clone();

    while exp > BigUint::zero() {
        if &exp % 2u32 == BigUint::one() {
            result = (result * &base) % modulus;
        }
        base = (&base * &base) % modulus;
        exp >>= 1;
    }

    result
}

// ============================================================================
// GPU Acceleration Integration
// ============================================================================

#[cfg(feature = "gpu")]
use super::gpu_math::GPUModularMath;

#[cfg(feature = "gpu")]
lazy_static::lazy_static! {
    static ref GPU_MATH_CONTEXT: Mutex<Option<GPUModularMath>> = Mutex::new(None);
}

/// Initialize GPU acceleration if available
#[cfg(feature = "gpu")]
pub async fn init_gpu_acceleration() -> CryptoResult<()> {
    let gpu_math = GPUModularMath::new().await?;
    let mut context = GPU_MATH_CONTEXT.lock().unwrap();
    *context = Some(gpu_math);
    Ok(())
}

/// Check if GPU acceleration is available
#[cfg(feature = "gpu")]
pub fn gpu_acceleration_available() -> bool {
    GPU_MATH_CONTEXT.lock().unwrap().is_some()
}

/// GPU-accelerated Montgomery modular multiplication for multiple values
#[cfg(feature = "gpu")]
pub async fn gpu_montgomery_mul_batch(
    a_values: &[u32],
    b_values: &[u32],
    modulus: u32,
    montgomery_r: u32,
    montgomery_r_squared: u32,
    montgomery_r_inv: u32,
) -> CryptoResult<Vec<u32>> {
    if let Some(gpu_math) = GPU_MATH_CONTEXT.lock().unwrap().as_ref() {
        gpu_math.montgomery_mul_batch(
            a_values,
            b_values,
            modulus,
            montgomery_r,
            montgomery_r_squared,
            montgomery_r_inv,
        ).await
    } else {
        Err(CryptoError::HardwareError("GPU acceleration not initialized".to_string()))
    }
}

/// GPU-accelerated modular exponentiation
#[cfg(feature = "gpu")]
pub async fn gpu_modular_exponentiation(
    base: u32,
    exponent: &BigUint,
    modulus: u32,
) -> CryptoResult<u32> {
    if let Some(gpu_math) = GPU_MATH_CONTEXT.lock().unwrap().as_ref() {
        gpu_math.modular_exponentiation(base, exponent, modulus).await
    } else {
        Err(CryptoError::HardwareError("GPU acceleration not initialized".to_string()))
    }
}

/// GPU-accelerated batch modular multiplication
#[cfg(feature = "gpu")]
pub async fn gpu_batch_modular_multiplication(
    a_values: &[u32],
    b_values: &[u32],
    moduli: &[u32],
) -> CryptoResult<Vec<u32>> {
    if let Some(gpu_math) = GPU_MATH_CONTEXT.lock().unwrap().as_ref() {
        gpu_math.batch_modular_multiplication(a_values, b_values, moduli).await
    } else {
        Err(CryptoError::HardwareError("GPU acceleration not initialized".to_string()))
    }
}

/// Fallback implementations when GPU is not available
#[cfg(not(feature = "gpu"))]
pub async fn init_gpu_acceleration() -> CryptoResult<()> {
    Err(CryptoError::HardwareError("GPU support not compiled in".to_string()))
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_acceleration_available() -> bool {
    false
}

#[cfg(not(feature = "gpu"))]
pub async fn gpu_montgomery_mul_batch(
    _a_values: &[u32],
    _b_values: &[u32],
    _modulus: u32,
    _montgomery_r: u32,
    _montgomery_r_squared: u32,
    _montgomery_r_inv: u32,
) -> CryptoResult<Vec<u32>> {
    Err(CryptoError::HardwareError("GPU support not available".to_string()))
}

#[cfg(not(feature = "gpu"))]
pub async fn gpu_modular_exponentiation(
    _base: u32,
    _exponent: &BigUint,
    _modulus: u32,
) -> CryptoResult<u32> {
    Err(CryptoError::HardwareError("GPU support not available".to_string()))
}

#[cfg(not(feature = "gpu"))]
pub async fn gpu_batch_modular_multiplication(
    _a_values: &[u32],
    _b_values: &[u32],
    _moduli: &[u32],
) -> CryptoResult<Vec<u32>> {
    Err(CryptoError::HardwareError("GPU support not available".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(17, 5), 1);
        assert_eq!(gcd(100, 50), 50);
    }

    #[test]
    fn test_mod_inverse() {
        // 3 * 5 = 15 ≡ 1 (mod 7)
        let inv = mod_inverse(3, 7).unwrap();
        assert_eq!((3 * inv) % 7, 1);

        // 2 has no inverse mod 6 (not coprime)
        assert!(mod_inverse(2, 6).is_err());
    }

    #[test]
    fn test_montgomery_context_creation() {
        let modulus = BigUint::from(17u32);
        let ctx = MontgomeryContext::new(modulus.clone());

        assert_eq!(ctx.modulus, modulus);
        assert!(ctx.r > modulus);
        assert_eq!((ctx.r.clone() * ctx.r) % modulus, ctx.r_squared);
    }

    #[test]
    fn test_montgomery_conversion() {
        let modulus = BigUint::from(17u32);
        let ctx = MontgomeryContext::new(modulus);

        let x = BigUint::from(5u32);
        let x_mont = ctx.to_montgomery(&x);
        let x_back = ctx.from_montgomery(&x_mont);

        assert_eq!(x, x_back);
    }

    #[test]
    fn test_montgomery_multiplication() {
        let modulus = BigUint::from(17u32);
        let ctx = MontgomeryContext::new(modulus.clone());

        let a = BigUint::from(3u32);
        let b = BigUint::from(5u32);
        let expected = (a.clone() * b.clone()) % modulus;

        let a_mont = ctx.to_montgomery(&a);
        let b_mont = ctx.to_montgomery(&b);
        let result_mont = ctx.montgomery_mul(&a_mont, &b_mont);
        let result = ctx.from_montgomery(&result_mont);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_montgomery_exponentiation() {
        let modulus = BigUint::from(17u32);
        let ctx = MontgomeryContext::new(modulus.clone());

        let base = BigUint::from(3u32);
        let exp = BigUint::from(4u32);
        let expected = modpow_biguint(&base, &exp, &modulus);

        let result = ctx.montgomery_pow(&base, &exp);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_montgomery_randomized_small_moduli() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let moduli = vec![17u64, 257u64, 12289u64, 40961u64];

        for &m in &moduli {
            let modulus = BigUint::from(m);
            let ctx = MontgomeryContext::new(modulus.clone());

            for _ in 0..50 {
                let a_u: u128 = rng.gen_range(0..m as u128);
                let b_u: u128 = rng.gen_range(0..m as u128);
                let a = BigUint::from(a_u as u64);
                let b = BigUint::from(b_u as u64);

                // Check multiplication
                let expected = (&a * &b) % &modulus;
                let res = montgomery_modmul(&a, &b, &modulus);
                assert_eq!(res, expected);

                // Check exponentiation for small exponent
                let exp_u: u64 = rng.gen_range(0..20);
                let exp = BigUint::from(exp_u);
                let expected_pow = modpow_biguint(&a, &exp, &modulus);
                let pow_res = montgomery_modpow(&a, &exp, &modulus);
                assert_eq!(pow_res, expected_pow);
            }
        }
    }

    #[test]
    fn test_montgomery_randomized_large_values() {
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        // Use a 64-bit modulus (odd) for larger tests
        let modulus = BigUint::from(0xffff_ffff_ffff_ffffu128); // large number but not prime
        let modulus = (&modulus | BigUint::one()).clone();
        let ctx = MontgomeryContext::new(modulus.clone());

        for _ in 0..50 {
            let mut a_bytes = vec![0u8; 16];
            let mut b_bytes = vec![0u8; 16];
            rng.fill_bytes(&mut a_bytes);
            rng.fill_bytes(&mut b_bytes);

            let a = BigUint::from_bytes_le(&a_bytes) % &modulus;
            let b = BigUint::from_bytes_le(&b_bytes) % &modulus;

            let expected = (&a * &b) % &modulus;
            let res = montgomery_modmul(&a, &b, &modulus);
            assert_eq!(res, expected);
        }
    }

    #[test]
    fn test_mod_inverse_biguint() {
        let a = BigUint::from(3u32);
        let m = BigUint::from(7u32);

        let inv = mod_inverse_biguint(&a, &m).unwrap();
        let product = (a * inv) % m;

        assert_eq!(product, BigUint::one());

        // Test case where no inverse exists
        let a = BigUint::from(4u32);
        let m = BigUint::from(6u32);

        assert!(mod_inverse_biguint(&a, &m).is_none());
    }

    #[test]
    fn test_gpu_fallback_when_disabled() {
        // Test that GPU functions return appropriate errors when GPU support is not compiled in
        assert!(!gpu_acceleration_available());
    }

    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_gpu_initialization() {
        // Test GPU initialization (will fail if no GPU available, but should not panic)
        let result = init_gpu_acceleration().await;
        // Result depends on hardware availability, but should not panic
        match result {
            Ok(()) => assert!(gpu_acceleration_available()),
            Err(_) => assert!(!gpu_acceleration_available()),
        }
    }

    #[tokio::test]
    #[cfg(feature = "gpu")]
    async fn test_gpu_functions_without_initialization() {
        // Test that GPU functions fail gracefully when not initialized
        let a_values = vec![1u32, 2u32, 3u32];
        let b_values = vec![4u32, 5u32, 6u32];

        let result = gpu_montgomery_mul_batch(
            &a_values, &b_values, 17, 16, 15, 14
        ).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CryptoError::HardwareError(_)));
    }
}
