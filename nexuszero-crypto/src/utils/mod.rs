//! Utility functions and mathematical primitives
//!
//! This module provides common utility functions used throughout the library.

pub mod math;
pub mod constant_time;
pub mod hardware;
pub mod sidechannel;

// Re-export common functions
pub use math::{mod_inverse, modular_exponentiation};

// Re-export constant-time cryptographic utilities
pub use constant_time::{
    ct_modpow, ct_bytes_eq, ct_in_range, ct_array_access, ct_dot_product,
    ct_less_than, ct_less_or_equal, ct_greater_than, ct_greater_or_equal,
    ct_modpow_blinded, ct_dot_product_blinded, blind_value, unblind_value,
};
