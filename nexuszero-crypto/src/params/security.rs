//! Security parameter definitions and selection
//!
//! This module provides standard security levels and parameter sets.

use crate::lattice::RingLWEParameters;
use crate::{CryptoError, CryptoResult};
use serde::{Deserialize, Serialize};

/// Standard security levels (NIST post-quantum standards)
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityLevel {
    /// 128-bit security (NIST Level 1) - Comparable to AES-128
    Bit128,
    /// 192-bit security (NIST Level 3) - Comparable to AES-192
    Bit192,
    /// 256-bit security (NIST Level 5) - Comparable to AES-256
    Bit256,
}

/// Complete parameter set for a security level
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterSet {
    /// Security level
    pub security_level: SecurityLevel,
    /// Ring dimension (power of 2)
    pub n: usize,
    /// Coefficient modulus (prime)
    pub q: u64,
    /// Error distribution standard deviation
    pub sigma: f64,
    /// Estimated proof size (bytes)
    pub proof_size: usize,
    /// Estimated prove time (milliseconds)
    pub prove_time_ms: u64,
    /// Estimated verify time (milliseconds)
    pub verify_time_ms: u64,
}

impl ParameterSet {
    /// Get standard parameter set for security level
    pub fn from_security_level(level: SecurityLevel) -> Self {
        match level {
            SecurityLevel::Bit128 => Self::standard_128bit(),
            SecurityLevel::Bit192 => Self::standard_192bit(),
            SecurityLevel::Bit256 => Self::standard_256bit(),
        }
    }

    /// Standard 128-bit security parameters
    fn standard_128bit() -> Self {
        ParameterSet {
            security_level: SecurityLevel::Bit128,
            n: 512,
            q: 12289,
            sigma: 3.2,
            proof_size: 8_192,
            prove_time_ms: 80,
            verify_time_ms: 40,
        }
    }

    /// Standard 192-bit security parameters
    fn standard_192bit() -> Self {
        ParameterSet {
            security_level: SecurityLevel::Bit192,
            n: 1024,
            q: 40961,
            sigma: 3.2,
            proof_size: 16_384,
            prove_time_ms: 150,
            verify_time_ms: 75,
        }
    }

    /// Standard 256-bit security parameters
    fn standard_256bit() -> Self {
        ParameterSet {
            security_level: SecurityLevel::Bit256,
            n: 2048,
            q: 65537,
            sigma: 3.2,
            proof_size: 32_768,
            prove_time_ms: 300,
            verify_time_ms: 150,
        }
    }

    /// Validate parameter set
    pub fn validate(&self) -> CryptoResult<()> {
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

/// Cryptographic parameters for the proof system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CryptoParameters {
    /// Security level
    pub security_level: SecurityLevel,
    /// Ring-LWE parameters
    pub ring_params: RingLWEParameters,
}

impl CryptoParameters {
    /// Create parameters from security level
    pub fn from_security_level(level: SecurityLevel) -> Self {
        let param_set = ParameterSet::from_security_level(level);
        let ring_params = RingLWEParameters::new(param_set.n, param_set.q, param_set.sigma);

        Self {
            security_level: level,
            ring_params,
        }
    }

    /// Standard 128-bit security
    pub fn new_128bit_security() -> Self {
        Self::from_security_level(SecurityLevel::Bit128)
    }

    /// Standard 192-bit security
    pub fn new_192bit_security() -> Self {
        Self::from_security_level(SecurityLevel::Bit192)
    }

    /// Standard 256-bit security
    pub fn new_256bit_security() -> Self {
        Self::from_security_level(SecurityLevel::Bit256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_sets() {
        let params_128 = ParameterSet::from_security_level(SecurityLevel::Bit128);
        assert!(params_128.validate().is_ok());
        assert_eq!(params_128.n, 512);

        let params_192 = ParameterSet::from_security_level(SecurityLevel::Bit192);
        assert!(params_192.validate().is_ok());
        assert_eq!(params_192.n, 1024);

        let params_256 = ParameterSet::from_security_level(SecurityLevel::Bit256);
        assert!(params_256.validate().is_ok());
        assert_eq!(params_256.n, 2048);
    }

    #[test]
    fn test_crypto_parameters() {
        let params = CryptoParameters::new_128bit_security();
        assert_eq!(params.security_level, SecurityLevel::Bit128);
        assert_eq!(params.ring_params.n, 512);
    }
}
