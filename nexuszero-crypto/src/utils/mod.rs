//! Utility functions and mathematical primitives
//!
//! This module provides common utility functions used throughout the library.

pub mod math;

// Re-export common functions
pub use math::{mod_inverse, modular_exponentiation};
