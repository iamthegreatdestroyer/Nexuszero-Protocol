//! Learning With Errors (LWE) primitives
//!
//! This module implements the LWE cryptographic scheme, which forms
//! the security foundation for quantum-resistant cryptography.

use crate::{CryptoError, CryptoResult, LatticeParameters};
use ndarray::{Array1, Array2};
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};

/// LWE parameters defining security and operational characteristics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LWEParameters {
    /// Dimension (security parameter)
    pub n: usize,
    /// Number of samples
    pub m: usize,
    /// Modulus
    pub q: u64,
    /// Error distribution standard deviation
    pub sigma: f64,
}

impl LWEParameters {
    /// Create new LWE parameters
    pub fn new(n: usize, m: usize, q: u64, sigma: f64) -> Self {
        Self { n, m, q, sigma }
    }
}

impl LatticeParameters for LWEParameters {
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
        if self.n == 0 {
            return Err(CryptoError::InvalidParameter(
                "Dimension must be positive".to_string(),
            ));
        }
        if self.m == 0 {
            return Err(CryptoError::InvalidParameter(
                "Number of samples must be positive".to_string(),
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

/// LWE public key
#[derive(Clone, Debug)]
pub struct LWEPublicKey {
    /// Random matrix A (m × n)
    pub a: Array2<i64>,
    /// Vector b = As + e (m-dimensional)
    pub b: Array1<i64>,
}

/// LWE secret key
#[derive(Clone, Debug)]
pub struct LWESecretKey {
    /// Secret vector s (n-dimensional)
    pub s: Array1<i64>,
}

/// LWE ciphertext
#[derive(Clone, Debug)]
pub struct LWECiphertext {
    /// Ciphertext component u
    pub u: Array1<i64>,
    /// Ciphertext component v
    pub v: i64,
}

/// Generate LWE key pair
pub fn keygen<R: Rng + CryptoRng>(
    params: &LWEParameters,
    rng: &mut R,
) -> CryptoResult<(LWESecretKey, LWEPublicKey)> {
    params.validate()?;

    // Generate secret key s
    let s = sample_secret(params.n, params.q, rng);

    // Generate random matrix A
    let a = sample_matrix(params.m, params.n, params.q, rng);

    // Generate error vector e
    let e = crate::lattice::sampling::sample_error(params.sigma, params.m);

    // Compute b = As + e (mod q)
    let as_product = a.dot(&s);
    let b = as_product
        .iter()
        .zip(e.iter())
        .map(|(as_i, e_i)| (as_i + e_i).rem_euclid(params.q as i64))
        .collect::<Array1<i64>>();

    let sk = LWESecretKey { s };
    let pk = LWEPublicKey { a, b };

    Ok((sk, pk))
}

/// Encrypt a message bit
pub fn encrypt<R: Rng + CryptoRng>(
    pk: &LWEPublicKey,
    message: bool,
    params: &LWEParameters,
    rng: &mut R,
) -> CryptoResult<LWECiphertext> {
    // Sample random vector r
    let r = sample_binary_vector(params.m, rng);

    // Compute u = A^T r (mod q)
    let u = pk
        .a
        .t()
        .dot(&r)
        .mapv(|x| x.rem_euclid(params.q as i64));

    // Encode message (0 -> 0, 1 -> q/2)
    let encoded_msg = if message { (params.q / 2) as i64 } else { 0 };

    // Compute v = b^T r + encoded_msg (mod q)
    let v = (pk.b.dot(&r) + encoded_msg).rem_euclid(params.q as i64);

    Ok(LWECiphertext { u, v })
}

/// Decrypt a ciphertext
pub fn decrypt(
    sk: &LWESecretKey,
    ct: &LWECiphertext,
    params: &LWEParameters,
) -> CryptoResult<bool> {
    // Compute m' = v - s^T u (mod q)
    let m_prime = (ct.v - sk.s.dot(&ct.u)).rem_euclid(params.q as i64);

    // Decode: if m' is closer to q/2 than to 0, message is 1
    let _threshold = (params.q / 4) as i64;
    let distance_to_zero = m_prime.min(params.q as i64 - m_prime);
    let distance_to_half = (m_prime - (params.q / 2) as i64)
        .abs()
        .min((params.q / 2) as i64);

    Ok(distance_to_half < distance_to_zero)
}

// Helper functions

fn sample_secret<R: Rng + CryptoRng>(n: usize, q: u64, rng: &mut R) -> Array1<i64> {
    Array1::from_shape_fn(n, |_| rng.gen_range(0..q) as i64)
}

fn sample_matrix<R: Rng + CryptoRng>(
    m: usize,
    n: usize,
    q: u64,
    rng: &mut R,
) -> Array2<i64> {
    Array2::from_shape_fn((m, n), |_| rng.gen_range(0..q) as i64)
}

fn sample_binary_vector<R: Rng + CryptoRng>(m: usize, rng: &mut R) -> Array1<i64> {
    Array1::from_shape_fn(m, |_| if rng.gen_bool(0.5) { 1 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lwe_parameters_validation() {
        let params = LWEParameters::new(256, 512, 12289, 3.2);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_lwe_encrypt_decrypt() {
        let params = LWEParameters::new(32, 64, 97, 2.0);
        let mut rng = rand::thread_rng();

        let (sk, pk) = keygen(&params, &mut rng).unwrap();

        // Test encrypting false
        let ct_false = encrypt(&pk, false, &params, &mut rng).unwrap();
        let decrypted_false = decrypt(&sk, &ct_false, &params).unwrap();
        assert_eq!(decrypted_false, false);

        // Test encrypting true
        let ct_true = encrypt(&pk, true, &params, &mut rng).unwrap();
        let decrypted_true = decrypt(&sk, &ct_true, &params).unwrap();
        assert_eq!(decrypted_true, true);
    }
    
    #[test]
    fn test_keygen_with_various_parameters() {
        let mut rng = rand::thread_rng();
        
        // Test with minimal parameters
        let params = LWEParameters::new(8, 16, 11, 1.0);
        let result = keygen(&params, &mut rng);
        assert!(result.is_ok());
        
        // Test with large parameters
        let params = LWEParameters::new(512, 1024, 40961, 5.0);
        let result = keygen(&params, &mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decrypt_edge_values() {
        // Test decryption with reliable parameters
        let params = LWEParameters::new(64, 128, 257, 2.0);
        let mut rng = rand::thread_rng();
        let (sk, pk) = keygen(&params, &mut rng).unwrap();
        
        // Test multiple encryptions and decryptions
        for _ in 0..10 {
            let ct_false = encrypt(&pk, false, &params, &mut rng).unwrap();
            let ct_true = encrypt(&pk, true, &params, &mut rng).unwrap();
            
            assert_eq!(decrypt(&sk, &ct_false, &params).unwrap(), false);
            assert_eq!(decrypt(&sk, &ct_true, &params).unwrap(), true);
        }
    }

    #[test]
    fn test_lwe_with_small_modulus() {
        // Test with small but secure modulus
        let params = LWEParameters::new(32, 64, 127, 1.5);
        let mut rng = rand::thread_rng();
        
        let (sk, pk) = keygen(&params, &mut rng).unwrap();
        let ct = encrypt(&pk, true, &params, &mut rng).unwrap();
        let decrypted = decrypt(&sk, &ct, &params).unwrap();
        assert_eq!(decrypted, true);
        
        // Test false as well
        let ct_false = encrypt(&pk, false, &params, &mut rng).unwrap();
        let decrypted_false = decrypt(&sk, &ct_false, &params).unwrap();
        assert_eq!(decrypted_false, false);
    }

    #[test]
    fn test_error_distribution_statistical_properties() {
        use crate::lattice::sampling::sample_error;
        let sigma = 3.2;
        let n = 1000;
        
        // Sample error vector
        let samples = sample_error(sigma, n);
        
        // Check mean is close to 0
        let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        assert!(mean.abs() < 1.0, "Mean should be close to 0: {}", mean);
        
        // Check standard deviation is close to sigma
        let variance: f64 = samples.iter().map(|&x| {
            let x_f = x as f64;
            (x_f - mean).powi(2)
        }).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        assert!((std_dev - sigma).abs() < 1.0, "Std dev should be close to {}: {}", sigma, std_dev);
        
        // Check distribution is roughly bell-shaped (most values within 3 sigma)
        let within_3sigma = samples.iter().filter(|&&x| (x as f64).abs() <= 3.0 * sigma).count();
        let ratio = within_3sigma as f64 / n as f64;
        assert!(ratio > 0.95, "At least 95% of samples should be within 3 sigma: {}", ratio);
    }

    #[test]
    fn test_error_magnitude_bounds() {
        use crate::lattice::sampling::sample_error;
        let sigma = 2.0;
        
        // Sample error vector and check all values are bounded
        for _ in 0..10 {
            let errors = sample_error(sigma, 100);
            for &error in errors.iter() {
                // In practice, errors beyond 6*sigma are extremely rare
                let error_f = (error as f64).abs();
                assert!(error_f < 6.0 * sigma, "Error magnitude too large: {}", error_f);
            }
        }
    }

    #[test]
    fn test_lwe_invalid_parameters() {
        // Zero dimension
        let params = LWEParameters::new(0, 64, 97, 2.0);
        assert!(params.validate().is_err());
        
        // Zero samples
        let params = LWEParameters::new(32, 0, 97, 2.0);
        assert!(params.validate().is_err());
        
        // Invalid modulus
        let params = LWEParameters::new(32, 64, 1, 2.0);
        assert!(params.validate().is_err());
        
        // Negative/zero sigma
        let params = LWEParameters::new(32, 64, 97, 0.0);
        assert!(params.validate().is_err());
        
        let params = LWEParameters::new(32, 64, 97, -1.0);
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_encryption_produces_different_ciphertexts() {
        let params = LWEParameters::new(32, 64, 97, 2.0);
        let mut rng = rand::thread_rng();
        let (sk, pk) = keygen(&params, &mut rng).unwrap();
        
        // Encrypt same message multiple times
        let ct1 = encrypt(&pk, true, &params, &mut rng).unwrap();
        let ct2 = encrypt(&pk, true, &params, &mut rng).unwrap();
        
        // Ciphertexts should be different (probabilistic encryption)
        let same_u = ct1.u.iter().zip(ct2.u.iter()).all(|(a, b)| a == b);
        let same_v = ct1.v == ct2.v;
        assert!(!(same_u && same_v), "Ciphertexts should differ due to randomness");
        
        // But both decrypt to the same plaintext
        assert_eq!(decrypt(&sk, &ct1, &params).unwrap(), true);
        assert_eq!(decrypt(&sk, &ct2, &params).unwrap(), true);
    }
}
