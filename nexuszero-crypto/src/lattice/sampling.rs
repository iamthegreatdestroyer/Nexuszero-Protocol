//! Error sampling from discrete Gaussian distributions
//!
//! This module implements sampling from discrete Gaussian distributions,
//! which is crucial for LWE security.

use ndarray::Array1;
use rand::Rng;

/// Sample error vector from discrete Gaussian distribution
///
/// Uses Box-Muller transform to generate continuous Gaussian,
/// then rounds to nearest integer for discrete Gaussian.
pub fn sample_error(sigma: f64, dimension: usize) -> Array1<i64> {
    let mut rng = rand::thread_rng();
    Array1::from_shape_fn(dimension, |_| sample_gaussian_int(sigma, &mut rng))
}

/// Sample a single integer from discrete Gaussian
fn sample_gaussian_int<R: Rng>(sigma: f64, rng: &mut R) -> i64 {
    // Box-Muller transform for continuous Gaussian
    let u1: f64 = rng.gen();
    let u2: f64 = rng.gen();

    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    let scaled = z * sigma;

    // Round to nearest integer
    scaled.round() as i64
}

/// Sample uniform random values
pub fn sample_uniform(min: i64, max: i64, dimension: usize) -> Array1<i64> {
    let mut rng = rand::thread_rng();
    Array1::from_shape_fn(dimension, |_| rng.gen_range(min..max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_sampling() {
        let errors = sample_error(3.2, 1000);

        // Check dimension
        assert_eq!(errors.len(), 1000);

        // Compute empirical mean (should be close to 0)
        let mean: f64 = errors.iter().map(|&x| x as f64).sum::<f64>() / 1000.0;
        assert!(mean.abs() < 1.0, "Mean {} too far from 0", mean);

        // Compute empirical standard deviation (should be close to sigma)
        let variance: f64 = errors
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / 1000.0;
        let std_dev = variance.sqrt();
        assert!(
            (std_dev - 3.2).abs() < 0.5,
            "Std dev {} too far from 3.2",
            std_dev
        );
    }

    #[test]
    fn test_uniform_sampling() {
        let values = sample_uniform(0, 100, 1000);
        assert_eq!(values.len(), 1000);

        // All values should be in range [0, 100)
        for &val in values.iter() {
            assert!(val >= 0 && val < 100);
        }
    }
}
