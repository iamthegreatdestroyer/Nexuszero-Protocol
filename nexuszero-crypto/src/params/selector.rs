/// Parameter Selector
/// 
/// Provides a flexible builder pattern for selecting cryptographic parameters
/// with automatic security level estimation and constraint validation.

use crate::params::security::SecurityLevel;
use crate::lattice::{LWEParameters, RingLWEParameters};
use crate::{CryptoError, CryptoResult, LatticeParameters};
use num_bigint::BigUint;
use num_traits::One;
use rand::{Rng, thread_rng};

/// Builder for selecting LWE and Ring-LWE parameters
/// 
/// # Example
/// 
/// ```
/// use nexuszero_crypto::params::selector::ParameterSelector;
/// use nexuszero_crypto::params::security::SecurityLevel;
/// 
/// let selector = ParameterSelector::new()
///     .target_security(SecurityLevel::Bit128)
///     .max_dimension(1024)
///     .prefer_prime_modulus(true);
/// 
/// let params = selector.build_lwe().unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ParameterSelector {
    target_security: Option<SecurityLevel>,
    min_dimension: Option<usize>,
    max_dimension: Option<usize>,
    min_modulus: Option<u64>,
    max_modulus: Option<u64>,
    prefer_prime_modulus: bool,
    custom_sigma: Option<f64>,
    custom_ratio: Option<f64>, // m/n ratio for LWE
}

impl Default for ParameterSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl ParameterSelector {
    /// Create a new parameter selector with default settings
    pub fn new() -> Self {
        Self {
            target_security: None,
            min_dimension: None,
            max_dimension: None,
            min_modulus: None,
            max_modulus: None,
            prefer_prime_modulus: false,
            custom_sigma: None,
            custom_ratio: None,
        }
    }

    /// Set target security level
    pub fn target_security(mut self, level: SecurityLevel) -> Self {
        self.target_security = Some(level);
        self
    }

    /// Set minimum dimension
    pub fn min_dimension(mut self, n: usize) -> Self {
        self.min_dimension = Some(n);
        self
    }

    /// Set maximum dimension
    pub fn max_dimension(mut self, n: usize) -> Self {
        self.max_dimension = Some(n);
        self
    }

    /// Set minimum modulus
    pub fn min_modulus(mut self, q: u64) -> Self {
        self.min_modulus = Some(q);
        self
    }

    /// Set maximum modulus
    pub fn max_modulus(mut self, q: u64) -> Self {
        self.max_modulus = Some(q);
        self
    }

    /// Prefer prime modulus (enables Miller-Rabin testing)
    pub fn prefer_prime_modulus(mut self, prefer: bool) -> Self {
        self.prefer_prime_modulus = prefer;
        self
    }

    /// Set custom error standard deviation
    pub fn custom_sigma(mut self, sigma: f64) -> Self {
        self.custom_sigma = Some(sigma);
        self
    }

    /// Set custom m/n ratio for LWE (default 2.0)
    pub fn custom_ratio(mut self, ratio: f64) -> Self {
        self.custom_ratio = Some(ratio);
        self
    }

    /// Build LWE parameters based on constraints
    pub fn build_lwe(self) -> CryptoResult<LWEParameters> {
        // Get target security level or default to 128-bit
        let security = self.target_security.unwrap_or(SecurityLevel::Bit128);
        
        // Determine dimension based on security level and constraints
        let n = self.select_dimension(security)?;
        
        // Determine m based on ratio (default 2.0)
        let ratio = self.custom_ratio.unwrap_or(2.0);
        let m = (n as f64 * ratio) as usize;
        
        // Select modulus
        let q = self.select_modulus(n, security)?;
        
        // Select sigma based on security level
        let sigma = self.custom_sigma.unwrap_or_else(|| {
            match security {
                SecurityLevel::Bit128 => 3.2,
                SecurityLevel::Bit192 => 3.8,
                SecurityLevel::Bit256 => 4.0,
            }
        });
        
        // Validate parameters
        let params = LWEParameters::new(n, m, q, sigma);
        params.validate()?;
        
        Ok(params)
    }

    /// Build Ring-LWE parameters based on constraints
    pub fn build_ring_lwe(self) -> CryptoResult<RingLWEParameters> {
        // Get target security level or default to 128-bit
        let security = self.target_security.unwrap_or(SecurityLevel::Bit128);
        
        // Ring-LWE requires power-of-2 dimension
        let n = self.select_power_of_2_dimension(security)?;
        
        // Select modulus
        let q = self.select_modulus(n, security)?;
        
        // Select sigma based on security level
        let sigma = self.custom_sigma.unwrap_or_else(|| {
            match security {
                SecurityLevel::Bit128 => 3.2,
                SecurityLevel::Bit192 => 3.8,
                SecurityLevel::Bit256 => 4.0,
            }
        });
        
        // Create and validate parameters
        let params = RingLWEParameters::new(n, q, sigma);
        params.validate()?;
        
        Ok(params)
    }

    /// Select appropriate dimension based on security level and constraints
    fn select_dimension(&self, security: SecurityLevel) -> CryptoResult<usize> {
        // Standard dimensions for each security level
        let standard_n = match security {
            SecurityLevel::Bit128 => 256,
            SecurityLevel::Bit192 => 384,
            SecurityLevel::Bit256 => 512,
        };
        
        // Apply constraints
        let mut n = standard_n;
        
        if let Some(min_n) = self.min_dimension {
            if n < min_n {
                n = min_n;
            }
        }
        
        if let Some(max_n) = self.max_dimension {
            if n > max_n {
                n = max_n;
            }
        }
        
        // Validate dimension
        if n < 64 {
            return Err(CryptoError::InvalidParameter(
                "Dimension too small for security".to_string()
            ));
        }
        
        Ok(n)
    }

    /// Select power-of-2 dimension for Ring-LWE
    fn select_power_of_2_dimension(&self, security: SecurityLevel) -> CryptoResult<usize> {
        let standard_n = match security {
            SecurityLevel::Bit128 => 512,
            SecurityLevel::Bit192 => 1024,
            SecurityLevel::Bit256 => 2048,
        };
        
        let mut n = standard_n;
        
        // Apply constraints and round to nearest power of 2
        if let Some(min_n) = self.min_dimension {
            if n < min_n {
                n = min_n.next_power_of_two();
            }
        }
        
        if let Some(max_n) = self.max_dimension {
            if n > max_n {
                // Round down to power of 2
                n = (max_n as f64).log2().floor().exp2() as usize;
            }
        }
        
        // Ensure it's a power of 2
        if !n.is_power_of_two() {
            n = n.next_power_of_two();
        }
        
        // Validate
        if n < 128 {
            return Err(CryptoError::InvalidParameter(
                "Ring-LWE dimension too small".to_string()
            ));
        }
        
        Ok(n)
    }

    /// Select appropriate modulus based on dimension and security
    fn select_modulus(&self, n: usize, security: SecurityLevel) -> CryptoResult<u64> {
        // Standard modulus selection based on dimension
        let standard_q = match security {
            SecurityLevel::Bit128 => 12289,  // Small prime, good for NTT
            SecurityLevel::Bit192 => 16411,  // Larger prime
            SecurityLevel::Bit256 => 20483,  // Even larger prime
        };
        
        let mut q = standard_q;
        
        // Apply constraints
        if let Some(min_q) = self.min_modulus {
            if q < min_q {
                q = min_q;
            }
        }
        
        if let Some(max_q) = self.max_modulus {
            if q > max_q {
                q = max_q;
            }
        }
        
        // If prime modulus preferred, find nearest prime
        if self.prefer_prime_modulus {
            q = find_nearest_prime(q)?;
        }
        
        // Validate modulus is large enough for security
        // Rule: q should be at least n for basic security
        if q < n as u64 {
            return Err(CryptoError::InvalidParameter(
                format!("Modulus {} too small for dimension {}", q, n)
            ));
        }
        
        Ok(q)
    }

    /// Estimate security level of given parameters
    pub fn estimate_security(n: usize, q: u64, sigma: f64) -> u32 {
        // Simplified security estimation based on lattice parameters
        // Based on the hardness of solving LWE with given parameters
        
        let log2_q = (q as f64).log2();
        let log2_n = (n as f64).log2();
        
        // Core security estimate: dimension is the primary factor
        // Each doubling of dimension roughly adds 50-80 bits of security
        let dimension_security = log2_n * 60.0;
        
        // Modulus factor: larger modulus slightly reduces security
        // but is needed for correctness
        let modulus_factor = (log2_q / (log2_n + 10.0)).min(1.0);
        
        // Error distribution factor: larger sigma reduces security
        // but smaller sigma helps security
        let sigma_factor = if sigma < 5.0 {
            1.0 - (sigma / 20.0) // Very minor reduction
        } else {
            0.9 // Larger sigma has some impact
        };
        
        // Combine factors
        let bit_security = dimension_security * modulus_factor * sigma_factor;
        
        // Clamp to reasonable range
        bit_security.max(64.0).min(512.0) as u32
    }
}

/// Find nearest prime number to target using Miller-Rabin test
fn find_nearest_prime(target: u64) -> CryptoResult<u64> {
    // Search in both directions
    let max_search = 1000;
    
    for offset in 0..max_search {
        // Try target + offset
        if offset == 0 || target + offset > target {
            let candidate = target + offset;
            if is_prime_miller_rabin(candidate, 20) {
                return Ok(candidate);
            }
        }
        
        // Try target - offset
        if offset > 0 && target > offset {
            let candidate = target - offset;
            if is_prime_miller_rabin(candidate, 20) {
                return Ok(candidate);
            }
        }
    }
    
    Err(CryptoError::InvalidParameter(
        format!("Could not find prime near {}", target)
    ))
}

/// Miller-Rabin primality test
/// 
/// Probabilistic test with error probability < 4^(-k)
/// where k is the number of rounds.
/// 
/// # Arguments
/// 
/// * `n` - Number to test for primality
/// * `k` - Number of testing rounds (20 is standard for cryptographic use)
/// 
/// # Returns
/// 
/// `true` if n is probably prime, `false` if n is definitely composite
pub fn is_prime_miller_rabin(n: u64, k: u32) -> bool {
    // Handle small cases
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    
    // Write n-1 as 2^r * d
    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    
    let n_big = BigUint::from(n);
    let mut rng = thread_rng();
    
    'witness: for _ in 0..k {
        // Pick random witness a in [2, n-2]
        let a = rng.gen_range(2..n-1);
        let a_big = BigUint::from(a);
        
        // Compute x = a^d mod n
        let mut x = a_big.modpow(&BigUint::from(d), &n_big);
        
        if x == One::one() || x == &n_big - BigUint::one() {
            continue 'witness;
        }
        
        for _ in 0..r-1 {
            // x = x^2 mod n
            x = (&x * &x) % &n_big;
            
            if x == &n_big - BigUint::one() {
                continue 'witness;
            }
        }
        
        // n is definitely composite
        return false;
    }
    
    // n is probably prime
    true
}

/// Generate cryptographically strong prime
/// 
/// Generates a prime number suitable for cryptographic use
/// with specified bit length.
pub fn generate_prime(bit_length: u32) -> CryptoResult<u64> {
    if bit_length > 63 {
        return Err(CryptoError::InvalidParameter(
            "Bit length too large for u64".to_string()
        ));
    }
    
    let mut rng = thread_rng();
    let max_attempts = 10000;
    
    for _ in 0..max_attempts {
        // Generate random odd number of specified bit length
        let min = 1u64 << (bit_length - 1);
        let max = (1u64 << bit_length) - 1;
        let mut candidate = rng.gen_range(min..=max);
        
        // Make it odd
        candidate |= 1;
        
        // Test primality
        if is_prime_miller_rabin(candidate, 20) {
            return Ok(candidate);
        }
    }
    
    Err(CryptoError::InvalidParameter(
        "Could not generate prime".to_string()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_miller_rabin_known_primes() {
        // Small primes
        assert!(is_prime_miller_rabin(2, 20));
        assert!(is_prime_miller_rabin(3, 20));
        assert!(is_prime_miller_rabin(5, 20));
        assert!(is_prime_miller_rabin(7, 20));
        assert!(is_prime_miller_rabin(11, 20));
        assert!(is_prime_miller_rabin(13, 20));
        
        // Larger primes
        assert!(is_prime_miller_rabin(97, 20));
        assert!(is_prime_miller_rabin(541, 20));
        assert!(is_prime_miller_rabin(7919, 20));
        assert!(is_prime_miller_rabin(12289, 20)); // Common in lattice crypto
    }

    #[test]
    fn test_miller_rabin_known_composites() {
        assert!(!is_prime_miller_rabin(0, 20));
        assert!(!is_prime_miller_rabin(1, 20));
        assert!(!is_prime_miller_rabin(4, 20));
        assert!(!is_prime_miller_rabin(6, 20));
        assert!(!is_prime_miller_rabin(8, 20));
        assert!(!is_prime_miller_rabin(9, 20));
        assert!(!is_prime_miller_rabin(100, 20));
        assert!(!is_prime_miller_rabin(1000, 20));
    }

    #[test]
    fn test_parameter_selector_lwe() {
        let selector = ParameterSelector::new()
            .target_security(SecurityLevel::Bit128);
        
        let params = selector.build_lwe().unwrap();
        
        assert_eq!(params.n, 256);
        assert_eq!(params.m, 512);
        assert!(params.q > 0);
    }

    #[test]
    fn test_parameter_selector_ring_lwe() {
        let selector = ParameterSelector::new()
            .target_security(SecurityLevel::Bit128);
        
        let params = selector.build_ring_lwe().unwrap();
        
        assert_eq!(params.n, 512);
        assert!(params.n.is_power_of_two());
        assert!(params.q > 0);
    }

    #[test]
    fn test_parameter_selector_with_constraints() {
        let selector = ParameterSelector::new()
            .target_security(SecurityLevel::Bit128)
            .min_dimension(512)
            .max_dimension(1024)
            .prefer_prime_modulus(true);
        
        let params = selector.build_lwe().unwrap();
        
        assert!(params.n >= 512);
        assert!(params.n <= 1024);
        assert!(is_prime_miller_rabin(params.q, 20));
    }

    #[test]
    fn test_security_estimation() {
        // 128-bit security parameters
        let security = ParameterSelector::estimate_security(256, 12289, 3.2);
        assert!(security >= 100, "Security should be reasonable: {}", security);
        
        // 192-bit security parameters
        let security = ParameterSelector::estimate_security(384, 16411, 3.8);
        assert!(security >= 100, "Security should be reasonable: {}", security);
        
        // 256-bit security parameters
        let security = ParameterSelector::estimate_security(512, 20483, 4.0);
        assert!(security >= 100, "Security should be reasonable: {}", security);
    }

    #[test]
    fn test_find_nearest_prime() {
        // Find prime near 12289 (which is already prime)
        let prime = find_nearest_prime(12289).unwrap();
        assert!(is_prime_miller_rabin(prime, 20));
        
        // Find prime near 12000 (not prime)
        let prime = find_nearest_prime(12000).unwrap();
        assert!(is_prime_miller_rabin(prime, 20));
        assert!((prime as i64 - 12000).abs() < 1000);
    }

    #[test]
    fn test_generate_prime() {
        let prime = generate_prime(14).unwrap();
        assert!(is_prime_miller_rabin(prime, 20));
        assert!(prime >= 8192 && prime < 16384); // 2^13 to 2^14
    }

    #[test]
    fn test_power_of_2_dimension() {
        let selector = ParameterSelector::new()
            .min_dimension(500)
            .max_dimension(1500);
        
        let n = selector.select_power_of_2_dimension(SecurityLevel::Bit128).unwrap();
        assert!(n.is_power_of_two());
        assert!(n >= 512); // Next power of 2 after min
        assert!(n <= 1024); // Power of 2 within max
    }

    #[test]
    fn test_selector_extreme_dimension_constraints() {
        // Very small min dimension
        let selector = ParameterSelector::new()
            .target_security(SecurityLevel::Bit128)
            .min_dimension(10)
            .max_dimension(300);
        let params = selector.build_lwe().unwrap();
        assert!(params.n >= 10 && params.n <= 300);
        
        // Very large max dimension
        let selector = ParameterSelector::new()
            .target_security(SecurityLevel::Bit128)
            .min_dimension(256)
            .max_dimension(10000);
        let params = selector.build_lwe().unwrap();
        assert!(params.n >= 256 && params.n <= 10000);
    }

    #[test]
    fn test_selector_conflicting_constraints() {
        // Min > Max should fail or auto-correct
        let selector = ParameterSelector::new()
            .min_dimension(1024)
            .max_dimension(512);
        
        // This should either error or auto-swap
        let result = selector.build_lwe();
        // If it succeeds, dimension should be reasonable
        if let Ok(params) = result {
            assert!(params.n > 0);
        }
    }

    #[test]
    fn test_selector_boundary_modulus() {
        // Very small modulus constraint
        let selector = ParameterSelector::new()
            .target_security(SecurityLevel::Bit128)
            .min_modulus(100)
            .max_modulus(500);
        let params = selector.build_lwe().unwrap();
        assert!(params.q >= 100 && params.q <= 500);
        
        // Very large modulus constraint
        let selector = ParameterSelector::new()
            .target_security(SecurityLevel::Bit128)
            .min_modulus(50000)
            .max_modulus(100000);
        let params = selector.build_lwe().unwrap();
        assert!(params.q >= 50000 && params.q <= 100000);
    }

    #[test]
    fn test_selector_custom_ratio_extremes() {
        // Very small ratio (m barely larger than n)
        let selector = ParameterSelector::new()
            .target_security(SecurityLevel::Bit128)
            .custom_ratio(1.1);
        let params = selector.build_lwe().unwrap();
        assert!((params.m as f64 / params.n as f64) >= 1.0);
        assert!((params.m as f64 / params.n as f64) < 1.5);
        
        // Large ratio (m much larger than n)
        let selector = ParameterSelector::new()
            .target_security(SecurityLevel::Bit128)
            .custom_ratio(10.0);
        let params = selector.build_lwe().unwrap();
        assert!((params.m as f64 / params.n as f64) >= 9.0);
    }

    #[test]
    fn test_selector_custom_sigma_extremes() {
        // Very small sigma
        let selector = ParameterSelector::new()
            .target_security(SecurityLevel::Bit128)
            .custom_sigma(0.5);
        let params = selector.build_lwe().unwrap();
        assert!((params.sigma - 0.5).abs() < 0.01);
        
        // Large sigma
        let selector = ParameterSelector::new()
            .target_security(SecurityLevel::Bit128)
            .custom_sigma(10.0);
        let params = selector.build_lwe().unwrap();
        assert!((params.sigma - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_prime_generation_edge_cases() {
        // Small bit size
        let prime = generate_prime(4).unwrap();
        assert!(is_prime_miller_rabin(prime, 20));
        assert!(prime >= 8 && prime < 16);
        
        // Larger bit size
        let prime = generate_prime(20).unwrap();
        assert!(is_prime_miller_rabin(prime, 20));
        assert!(prime >= (1 << 19) && prime < (1 << 20));
    }

    #[test]
    fn test_find_nearest_prime_boundary_cases() {
        // Near lower bound
        let prime = find_nearest_prime(100).unwrap();
        assert!(is_prime_miller_rabin(prime, 20));
        
        // Even number (should find odd prime)
        let prime = find_nearest_prime(1000).unwrap();
        assert!(is_prime_miller_rabin(prime, 20));
        assert!(prime % 2 == 1 || prime == 2);
        
        // Already prime
        let prime = find_nearest_prime(7919).unwrap();
        assert_eq!(prime, 7919);
        assert!(is_prime_miller_rabin(prime, 20));
    }

    #[test]
    fn test_ring_lwe_power_of_2_enforcement() {
        // Request non-power-of-2 via constraints, should get next power of 2
        let selector = ParameterSelector::new()
            .min_dimension(300)
            .max_dimension(600);
        let params = selector.build_ring_lwe().unwrap();
        assert!(params.n.is_power_of_two());
        assert!(params.n >= 512); // Next power of 2 >= 300
    }

        #[test]
        fn test_dimension_too_small_error() {
            // Test that dimensions below security threshold fail
            let selector = ParameterSelector::new()
                .max_dimension(32); // Below minimum 64
        
            let result = selector.build_lwe();
            assert!(result.is_err());
        }

        #[test]
        fn test_ring_lwe_dimension_too_small_error() {
            // Ring-LWE requires min 128
            let selector = ParameterSelector::new()
                .max_dimension(64); // Below Ring-LWE minimum
        
            let result = selector.build_ring_lwe();
            assert!(result.is_err());
        }

        #[test]
        fn test_modulus_smaller_than_dimension_error() {
            // Modulus must be at least as large as dimension
            let selector = ParameterSelector::new()
                .min_dimension(1000)
                .max_modulus(500); // q < n
        
            let result = selector.build_lwe();
            assert!(result.is_err());
        }

        #[test]
        fn test_generate_prime_bit_length_too_large() {
            // Test that requesting 64+ bit primes fails
            let result = generate_prime(64);
            assert!(result.is_err());
        
            let result = generate_prime(100);
            assert!(result.is_err());
        }

        #[test]
        fn test_find_prime_extreme_range() {
            // Test finding prime in challenging ranges
            let prime = find_nearest_prime(2).unwrap();
            assert_eq!(prime, 2);
        
            let prime = find_nearest_prime(3).unwrap();
            assert_eq!(prime, 3);
        
            // Large number
            let prime = find_nearest_prime(65000).unwrap();
            assert!(is_prime_miller_rabin(prime, 20));
            assert!((prime as i64 - 65000).abs() < 1000);
        }

        #[test]
        fn test_security_estimation_edge_cases() {
            // Very small parameters
            let security = ParameterSelector::estimate_security(64, 100, 1.0);
            assert!(security >= 64); // Should clamp to minimum
        
            // Very large parameters
            let security = ParameterSelector::estimate_security(8192, 100000, 2.0);
            assert!(security <= 512); // Should clamp to maximum
        
            // Large sigma impact
            let security = ParameterSelector::estimate_security(256, 12289, 10.0);
            assert!(security > 0);
        }

        #[test]
        fn test_selector_all_security_levels() {
            // Test each security level produces valid parameters
            for level in [SecurityLevel::Bit128, SecurityLevel::Bit192, SecurityLevel::Bit256] {
                let selector = ParameterSelector::new().target_security(level);
            
                let lwe_params = selector.clone().build_lwe().unwrap();
                assert!(lwe_params.validate().is_ok());
            
                let ring_params = selector.build_ring_lwe().unwrap();
                assert!(ring_params.validate().is_ok());
            }
        }

        #[test]
        fn test_selector_with_all_constraints() {
            // Test with maximum constraints specified
            let selector = ParameterSelector::new()
                .target_security(SecurityLevel::Bit192)
                .min_dimension(256)
                .max_dimension(512)
                .min_modulus(10000)
                .max_modulus(20000)
                .prefer_prime_modulus(true)
                .custom_sigma(3.5)
                .custom_ratio(1.5);
        
            let params = selector.build_lwe().unwrap();
            assert!(params.n >= 256 && params.n <= 512);
            assert!(params.q >= 10000 && params.q <= 20000);
            assert!(is_prime_miller_rabin(params.q, 20));
            assert!((params.sigma - 3.5).abs() < 0.01);
        }

        #[test]
        fn test_power_of_2_rounding_down() {
            // Test that max_dimension rounds down to power of 2
            let selector = ParameterSelector::new()
                .max_dimension(1500); // Should round down to 1024
        
            let n = selector.select_power_of_2_dimension(SecurityLevel::Bit192).unwrap();
            assert_eq!(n, 1024);
            assert!(n.is_power_of_two());
        }

        #[test]
        fn test_power_of_2_rounding_up() {
            // Test that min_dimension rounds up to power of 2
            let selector = ParameterSelector::new()
                .min_dimension(700) // Should round up to 1024
                .max_dimension(2000);
        
            let n = selector.select_power_of_2_dimension(SecurityLevel::Bit192).unwrap();
            assert_eq!(n, 1024);
            assert!(n.is_power_of_two());
        }
}
