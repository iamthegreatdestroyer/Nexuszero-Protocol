//! Lattice-based cryptographic primitives
//!
//! This module provides implementations of:
//! - Learning With Errors (LWE)
//! - Ring Learning With Errors (Ring-LWE)
//! - Error sampling from discrete Gaussian distributions

pub mod lwe;
pub mod ring_lwe;
pub mod sampling;

// Re-export main types
pub use lwe::{LWECiphertext, LWEParameters, LWEPublicKey, LWESecretKey};
pub use ring_lwe::{
    Polynomial, RingLWECiphertext, RingLWEParameters, RingLWEPublicKey, RingLWESecretKey,
};
pub use sampling::{sample_error, sample_uniform};
