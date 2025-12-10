//! Utility functions and mathematical primitives
//!
//! This module provides common utility functions used throughout the library.

pub mod math;
pub mod gpu_math;
pub mod constant_time;
pub mod hardware;
pub mod sidechannel;
pub mod dual_exponentiation;

// Re-export common functions
pub use math::{mod_inverse, modular_exponentiation, montgomery_modmul, montgomery_modpow};

// Re-export dual and multi-exponentiation functions
pub use dual_exponentiation::{
    MultiExpConfig, ExpTable, ShamirTrick, InterleavedExponentiation,
    VectorExponentiation, WindowedMultiExponentiation,
};

// Re-export GPU acceleration functions
#[cfg(feature = "gpu")]
pub use gpu_math::GPUModularMath;
#[cfg(feature = "gpu")]
pub use math::{
    init_gpu_acceleration, gpu_acceleration_available,
    gpu_montgomery_mul_batch, gpu_modular_exponentiation,
    gpu_batch_modular_multiplication,
};

// Re-export constant-time cryptographic utilities
pub use constant_time::{
    ct_modpow, ct_bytes_eq, ct_in_range, ct_array_access, ct_dot_product,
    ct_less_than, ct_less_or_equal, ct_greater_than, ct_greater_or_equal,
    ct_modpow_blinded, ct_dot_product_blinded, blind_value, unblind_value,
};
