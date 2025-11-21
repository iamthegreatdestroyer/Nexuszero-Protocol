//! Security parameter selection and management
//!
//! This module provides parameter sets for different security levels
//! and algorithms for selecting optimal parameters.

pub mod security;

/// Parameter selection with Miller-Rabin primality testing and security estimation
pub mod selector;

// Re-export main types
pub use security::{CryptoParameters, ParameterSet, SecurityLevel};
pub use selector::{ParameterSelector, is_prime_miller_rabin, generate_prime};
