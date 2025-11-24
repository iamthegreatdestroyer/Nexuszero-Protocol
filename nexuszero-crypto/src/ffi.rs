//! Foreign Function Interface (FFI) bindings for Python integration
//!
//! This module provides C-compatible functions that can be called from Python
//! using ctypes or cffi. It enables the Python neural optimizer to interact
//! with the Rust cryptographic primitives.

use std::ffi::c_char;
// intentionally not importing `ptr` to avoid clippy unused import in non-test builds
use crate::params::security::SecurityLevel;
use crate::params::selector::ParameterSelector;

/// C-compatible structure for cryptographic parameters
#[repr(C)]
pub struct CryptoParams {
    /// Lattice dimension (must be power of 2)
    pub n: u32,
    /// Coefficient modulus
    pub q: u32,
    /// Error distribution standard deviation
    pub sigma: f64,
}

/// C-compatible structure for optimization results
#[repr(C)]
pub struct OptimizationResult {
    /// Optimal lattice dimension
    pub optimal_n: u32,
    /// Optimal modulus
    pub optimal_q: u32,
    /// Optimal sigma value
    pub optimal_sigma: f64,
    /// Estimated proof size in bytes
    pub estimated_proof_size: u64,
    /// Estimated proof generation time in milliseconds
    pub estimated_prove_time_ms: u64,
}

/// Error codes for FFI functions
pub const FFI_SUCCESS: i32 = 0;
pub const FFI_ERROR_INVALID_PARAM: i32 = -1;
pub const FFI_ERROR_INTERNAL: i32 = -2;
pub const FFI_ERROR_NULL_POINTER: i32 = -3;

/// Estimate cryptographic parameters based on security level and circuit size
///
/// # Arguments
/// * `security_level` - Target security level in bits (128, 192, or 256)
/// * `circuit_size` - Size of the circuit to prove (affects performance)
/// * `result` - Output parameter to store the optimization results
///
/// # Returns
/// * `FFI_SUCCESS` (0) on success
/// * `FFI_ERROR_INVALID_PARAM` if parameters are invalid
/// * `FFI_ERROR_NULL_POINTER` if result pointer is null
/// * `FFI_ERROR_INTERNAL` for other errors
///
/// # Safety
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that `result` points to valid memory.
#[no_mangle]
pub unsafe extern "C" fn nexuszero_estimate_parameters(
    security_level: u32,
    circuit_size: u32,
    result: *mut OptimizationResult,
) -> i32 {
    // Validate input pointer
    if result.is_null() {
        return FFI_ERROR_NULL_POINTER;
    }

    // Validate security level
    let sec_level = match security_level {
        128 => SecurityLevel::Bit128,
        192 => SecurityLevel::Bit192,
        256 => SecurityLevel::Bit256,
        _ => return FFI_ERROR_INVALID_PARAM,
    };

    // Validate circuit size (must be reasonable)
    if circuit_size == 0 || circuit_size > 1_000_000 {
        return FFI_ERROR_INVALID_PARAM;
    }

    // Build parameter selector based on security level and circuit size
    let selector = ParameterSelector::new()
        .target_security(sec_level)
        .max_dimension(2048)
        .prefer_prime_modulus(true);

    // Build Ring-LWE parameters
    let params = match selector.build_ring_lwe() {
        Ok(p) => p,
        Err(_) => return FFI_ERROR_INTERNAL,
    };

    // Estimate proof size based on parameters
    // Formula: proof_size ≈ n * log2(q) / 8 + overhead
    let bits_per_coefficient = (params.q as f64).log2().ceil() as u64;
    let base_proof_size = (params.n as u64 * bits_per_coefficient) / 8;
    let overhead = 256; // Challenge, metadata, etc.
    let estimated_proof_size = base_proof_size + overhead;

    // Estimate proof generation time based on dimension
    // Formula: time ≈ n^2 / 1000 (simplified model)
    // Adjusted for circuit complexity
    let base_time = (params.n as u64 * params.n as u64) / 1000;
    let circuit_factor = (circuit_size as f64 / 1000.0).sqrt();
    let estimated_prove_time_ms = (base_time as f64 * circuit_factor) as u64;

    // Write results to output parameter
    unsafe {
        *result = OptimizationResult {
            optimal_n: params.n as u32,
            optimal_q: params.q as u32,
            optimal_sigma: params.sigma,
            estimated_proof_size,
            estimated_prove_time_ms,
        };
    }

    FFI_SUCCESS
}

/// Free resources associated with an OptimizationResult
///
/// # Arguments
/// * `result` - Pointer to the OptimizationResult to free
///
/// # Safety
/// This function is safe to call with null pointers (no-op).
/// If the pointer is non-null, the caller must ensure it was allocated
/// by this library and has not been freed already.
#[no_mangle]
pub unsafe extern "C" fn nexuszero_free_result(_result: *mut OptimizationResult) {
    // No-op: nothing to free for OptimizationResult's POD fields
    
    // Since OptimizationResult contains only POD types (no heap allocations),
    // we don't need to do anything here. In the future, if we add heap-allocated
    // fields (like strings), we would need to free them here.
    
    // For now, this is a no-op but provides the API for future extensions
}

/// Get the version of the nexuszero-crypto library
///
/// # Returns
/// A null-terminated C string containing the version.
/// The string is valid for the lifetime of the program.
///
/// # Safety
/// The returned pointer is valid for the lifetime of the program
/// and must not be freed by the caller.
#[no_mangle]
pub extern "C" fn nexuszero_get_version() -> *const c_char {
    static VERSION: &str = "0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Validate cryptographic parameters
///
/// # Arguments
/// * `params` - Pointer to CryptoParams to validate
///
/// # Returns
/// * `FFI_SUCCESS` if parameters are valid
/// * `FFI_ERROR_NULL_POINTER` if params is null
/// * `FFI_ERROR_INVALID_PARAM` if parameters are invalid
///
/// # Safety
/// The caller must ensure params points to valid memory.
#[no_mangle]
pub unsafe extern "C" fn nexuszero_validate_params(params: *const CryptoParams) -> i32 {
    if params.is_null() {
        return FFI_ERROR_NULL_POINTER;
    }

    unsafe {
        let p = &*params;
        
        // Validate n is power of 2
        if p.n == 0 || (p.n & (p.n - 1)) != 0 {
            return FFI_ERROR_INVALID_PARAM;
        }

        // Validate q is at least 2
        if p.q < 2 {
            return FFI_ERROR_INVALID_PARAM;
        }

        // Validate sigma is positive
        if p.sigma <= 0.0 {
            return FFI_ERROR_INVALID_PARAM;
        }
    }

    FFI_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_parameters_128bit() {
        let mut result = OptimizationResult {
            optimal_n: 0,
            optimal_q: 0,
            optimal_sigma: 0.0,
            estimated_proof_size: 0,
            estimated_prove_time_ms: 0,
        };

        let status = unsafe { nexuszero_estimate_parameters(128, 1000, &mut result as *mut _) };
        
        assert_eq!(status, FFI_SUCCESS);
        assert!(result.optimal_n > 0);
        assert!(result.optimal_q > 0);
        assert!(result.optimal_sigma > 0.0);
        assert!(result.estimated_proof_size > 0);
    }

    #[test]
    fn test_estimate_parameters_192bit() {
        let mut result = OptimizationResult {
            optimal_n: 0,
            optimal_q: 0,
            optimal_sigma: 0.0,
            estimated_proof_size: 0,
            estimated_prove_time_ms: 0,
        };

        let status = unsafe { nexuszero_estimate_parameters(192, 5000, &mut result as *mut _) };
        
        assert_eq!(status, FFI_SUCCESS);
        assert!(result.optimal_n >= 512); // Should be at least 512 for 192-bit security
        assert!(result.optimal_q > 0);
    }

    #[test]
    fn test_estimate_parameters_256bit() {
        let mut result = OptimizationResult {
            optimal_n: 0,
            optimal_q: 0,
            optimal_sigma: 0.0,
            estimated_proof_size: 0,
            estimated_prove_time_ms: 0,
        };

        let status = unsafe { nexuszero_estimate_parameters(256, 10000, &mut result as *mut _) };
        
        assert_eq!(status, FFI_SUCCESS);
        assert!(result.optimal_n >= 1024); // Should be at least 1024 for 256-bit security
    }

    #[test]
    fn test_estimate_parameters_invalid_security() {
        let mut result = OptimizationResult {
            optimal_n: 0,
            optimal_q: 0,
            optimal_sigma: 0.0,
            estimated_proof_size: 0,
            estimated_prove_time_ms: 0,
        };

        let status = unsafe { nexuszero_estimate_parameters(99, 1000, &mut result as *mut _) };
        assert_eq!(status, FFI_ERROR_INVALID_PARAM);
    }

    #[test]
    fn test_estimate_parameters_null_pointer() {
        let status = unsafe { nexuszero_estimate_parameters(128, 1000, std::ptr::null_mut()) };
        assert_eq!(status, FFI_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_estimate_parameters_invalid_circuit_size() {
        let mut result = OptimizationResult {
            optimal_n: 0,
            optimal_q: 0,
            optimal_sigma: 0.0,
            estimated_proof_size: 0,
            estimated_prove_time_ms: 0,
        };

        // Circuit size of 0 should fail
        let status = unsafe { nexuszero_estimate_parameters(128, 0, &mut result as *mut _) };
        assert_eq!(status, FFI_ERROR_INVALID_PARAM);

        // Very large circuit size should fail
        let status = unsafe { nexuszero_estimate_parameters(128, 10_000_000, &mut result as *mut _) };
        assert_eq!(status, FFI_ERROR_INVALID_PARAM);
    }

    #[test]
    fn test_free_result_null_pointer() {
        // Should not crash
        unsafe { nexuszero_free_result(std::ptr::null_mut()); }
    }

    #[test]
    fn test_free_result_valid_pointer() {
        let mut result = OptimizationResult {
            optimal_n: 512,
            optimal_q: 12289,
            optimal_sigma: 3.2,
            estimated_proof_size: 8192,
            estimated_prove_time_ms: 100,
        };

        // Should not crash
        unsafe { nexuszero_free_result(&mut result as *mut _); }
    }

    #[test]
    fn test_get_version() {
        let version_ptr = nexuszero_get_version();
        assert!(!version_ptr.is_null());
        
        // Convert to Rust string to verify
        let version = unsafe {
            std::ffi::CStr::from_ptr(version_ptr)
                .to_str()
                .expect("Version should be valid UTF-8")
        };
        
        assert!(version.starts_with("0."));
    }

    #[test]
    fn test_validate_params_valid() {
        let params = CryptoParams {
            n: 512,
            q: 12289,
            sigma: 3.2,
        };

        let status = unsafe { nexuszero_validate_params(&params as *const _) };
        assert_eq!(status, FFI_SUCCESS);
    }

    #[test]
    fn test_validate_params_invalid_n() {
        // n is not power of 2
        let params = CryptoParams {
            n: 500,
            q: 12289,
            sigma: 3.2,
        };

        let status = unsafe { nexuszero_validate_params(&params as *const _) };
        assert_eq!(status, FFI_ERROR_INVALID_PARAM);
    }

    #[test]
    fn test_validate_params_invalid_q() {
        let params = CryptoParams {
            n: 512,
            q: 1, // Too small
            sigma: 3.2,
        };

        let status = unsafe { nexuszero_validate_params(&params as *const _) };
        assert_eq!(status, FFI_ERROR_INVALID_PARAM);
    }

    #[test]
    fn test_validate_params_invalid_sigma() {
        let params = CryptoParams {
            n: 512,
            q: 12289,
            sigma: -1.0, // Negative
        };

        let status = unsafe { nexuszero_validate_params(&params as *const _) };
        assert_eq!(status, FFI_ERROR_INVALID_PARAM);
    }

    #[test]
    fn test_validate_params_null_pointer() {
        let status = unsafe { nexuszero_validate_params(std::ptr::null()) };
        assert_eq!(status, FFI_ERROR_NULL_POINTER);
    }
}
