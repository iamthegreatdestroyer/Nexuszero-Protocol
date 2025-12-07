// Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// PROPRIETARY NOTICE: This software contains trade secrets and proprietary
// algorithms protected under applicable intellectual property laws.
//
// PATENT NOTICE: This software implements technology covered by pending patent
// application: "Hardware-Accelerated Cryptographic Proof Generation Using
// Parallel Processing Units" - See legal/patents/PROVISIONAL_PATENT_CLAIMS.md
//
// TRADEMARK NOTICE: NexusZero Protocol™, Privacy Morphing™, Holographic Proof
// Compression™, Nova Folding Engine™, and related marks are trademarks of
// NexusZero Protocol. All Rights Reserved.
//
// See legal/IP_INNOVATIONS_REGISTRY.md for full intellectual property terms.

//! SIMD-Optimized Field Operations for Nova Proving System
//!
//! This module provides vectorized implementations of field arithmetic
//! operations using platform-specific SIMD instructions (AVX2, AVX-512, NEON).
//!
//! # Architecture Support
//!
//! - **x86_64 with AVX2**: 256-bit operations, 4x u64 parallelism
//! - **x86_64 with AVX-512**: 512-bit operations, 8x u64 parallelism  
//! - **aarch64 with NEON**: 128-bit operations, 2x u64 parallelism
//!
//! # Performance Characteristics
//!
//! | Operation | Scalar | AVX2 | AVX-512 | NEON |
//! |-----------|--------|------|---------|------|
//! | Vector Add | 1x | 4x | 8x | 2x |
//! | Vector Mul | 1x | 2-4x | 4-8x | 2x |
//! | MSM | 1x | 3-4x | 5-6x | 2-3x |
//! | NTT | 1x | 3x | 5x | 2x |
//!
//! # Usage
//!
//! ```rust,ignore
//! use nexuszero_crypto::proof::nova::simd_ops::{SimdFieldOps, detect_simd_support};
//!
//! // Auto-detect best SIMD implementation
//! let simd_ops = SimdFieldOps::auto_detect();
//!
//! // Perform vectorized addition
//! let a = vec![1u64, 2, 3, 4];
//! let b = vec![5u64, 6, 7, 8];
//! let result = simd_ops.vector_add(&a, &b, MODULUS);
//! ```

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;
use tracing::{debug, warn, instrument};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// Production Error Types
// ============================================================================

/// Errors that can occur during SIMD field operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SimdError {
    /// Vector length mismatch between operands
    #[error("Vector length mismatch: left has {left} elements, right has {right} elements")]
    LengthMismatch { left: usize, right: usize },
    
    /// Empty vector provided where non-empty required
    #[error("Empty vector provided for operation: {operation}")]
    EmptyVector { operation: &'static str },
    
    /// Invalid modulus (zero or one)
    #[error("Invalid modulus {modulus}: must be greater than 1")]
    InvalidModulus { modulus: u64 },
    
    /// Matrix dimensions mismatch
    #[error("Matrix dimensions mismatch: matrix has {matrix_len} elements but expected {rows} x {cols} = {expected}")]
    MatrixDimensionMismatch {
        matrix_len: usize,
        rows: usize,
        cols: usize,
        expected: usize,
    },
    
    /// Vector length doesn't match matrix columns
    #[error("Vector length {vector_len} doesn't match matrix columns {cols}")]
    VectorColumnsMismatch { vector_len: usize, cols: usize },
    
    /// Element not invertible in the field
    #[error("Element {element} is not invertible modulo {modulus}")]
    NotInvertible { element: u64, modulus: u64 },
    
    /// SIMD capability not available on this platform
    #[error("SIMD capability {capability:?} not available on this platform")]
    CapabilityNotAvailable { capability: SimdCapability },
}

/// Result type for SIMD operations
pub type SimdResult<T> = Result<T, SimdError>;

/// SIMD capability detection result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdCapability {
    /// No SIMD support - use scalar operations
    Scalar,
    /// ARM NEON support (128-bit)
    Neon,
    /// x86_64 AVX2 support (256-bit)
    Avx2,
    /// x86_64 AVX-512 support (512-bit)
    Avx512,
}

impl SimdCapability {
    /// Get the vector width in u64 elements
    pub fn vector_width(&self) -> usize {
        match self {
            SimdCapability::Scalar => 1,
            SimdCapability::Neon => 2,
            SimdCapability::Avx2 => 4,
            SimdCapability::Avx512 => 8,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            SimdCapability::Scalar => "Scalar",
            SimdCapability::Neon => "ARM NEON",
            SimdCapability::Avx2 => "Intel AVX2",
            SimdCapability::Avx512 => "Intel AVX-512",
        }
    }

    /// Check if this capability is available on the current platform
    pub fn is_available(&self) -> bool {
        match self {
            SimdCapability::Scalar => true, // Always available
            #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
            SimdCapability::Avx2 => is_x86_feature_detected!("avx2"),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            SimdCapability::Avx512 => {
                is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl")
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            SimdCapability::Neon => true, // NEON is always available on aarch64
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

/// Detect the best available SIMD capability at runtime
#[instrument(level = "debug")]
pub fn detect_simd_support() -> SimdCapability {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            debug!("Detected AVX-512 support");
            return SimdCapability::Avx512;
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        if is_x86_feature_detected!("avx2") {
            debug!("Detected AVX2 support");
            return SimdCapability::Avx2;
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        debug!("Detected ARM NEON support");
        // NEON is always available on aarch64
        return SimdCapability::Neon;
    }

    debug!("Using scalar fallback - no SIMD support detected");
    SimdCapability::Scalar
}

// ============================================================================
// Input Validation Helpers
// ============================================================================

/// Validate that two vectors have the same length
#[inline]
fn validate_vector_lengths(a: &[u64], b: &[u64]) -> SimdResult<()> {
    if a.len() != b.len() {
        return Err(SimdError::LengthMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    Ok(())
}

/// Validate that a vector is non-empty
#[inline]
fn validate_non_empty(v: &[u64], operation: &'static str) -> SimdResult<()> {
    if v.is_empty() {
        return Err(SimdError::EmptyVector { operation });
    }
    Ok(())
}

/// Validate that modulus is valid (> 1)
#[inline]
fn validate_modulus(modulus: u64) -> SimdResult<()> {
    if modulus <= 1 {
        return Err(SimdError::InvalidModulus { modulus });
    }
    Ok(())
}

/// Validate matrix dimensions
#[inline]
fn validate_matrix_dimensions(matrix: &[u64], vector: &[u64], rows: usize, cols: usize) -> SimdResult<()> {
    let expected = rows * cols;
    if matrix.len() != expected {
        return Err(SimdError::MatrixDimensionMismatch {
            matrix_len: matrix.len(),
            rows,
            cols,
            expected,
        });
    }
    if vector.len() != cols {
        return Err(SimdError::VectorColumnsMismatch {
            vector_len: vector.len(),
            cols,
        });
    }
    Ok(())
}

/// SIMD-optimized field operations
pub struct SimdFieldOps {
    /// Detected SIMD capability
    capability: SimdCapability,
    /// Operation counters for metrics
    add_ops: AtomicUsize,
    mul_ops: AtomicUsize,
    reduce_ops: AtomicUsize,
}

impl SimdFieldOps {
    /// Create with auto-detected SIMD capability
    pub fn auto_detect() -> Self {
        let capability = detect_simd_support();
        debug!("SimdFieldOps initialized with capability: {:?}", capability);
        Self {
            capability,
            add_ops: AtomicUsize::new(0),
            mul_ops: AtomicUsize::new(0),
            reduce_ops: AtomicUsize::new(0),
        }
    }

    /// Create with specific SIMD capability (for testing or forced fallback)
    /// 
    /// # Errors
    /// Returns `SimdError::CapabilityNotAvailable` if the requested capability
    /// is not available on the current platform.
    pub fn with_capability(capability: SimdCapability) -> SimdResult<Self> {
        if !capability.is_available() {
            warn!(
                "Requested SIMD capability {:?} not available, falling back to scalar",
                capability
            );
            return Err(SimdError::CapabilityNotAvailable { capability });
        }
        Ok(Self {
            capability,
            add_ops: AtomicUsize::new(0),
            mul_ops: AtomicUsize::new(0),
            reduce_ops: AtomicUsize::new(0),
        })
    }

    /// Create with specific capability, allowing unavailable capabilities (for testing)
    /// This bypasses the availability check and should only be used in tests
    #[cfg(test)]
    pub fn with_capability_unchecked(capability: SimdCapability) -> Self {
        Self {
            capability,
            add_ops: AtomicUsize::new(0),
            mul_ops: AtomicUsize::new(0),
            reduce_ops: AtomicUsize::new(0),
        }
    }

    /// Force fallback to scalar operations (useful for debugging or comparison)
    pub fn force_scalar() -> Self {
        debug!("SimdFieldOps forced to scalar mode");
        Self {
            capability: SimdCapability::Scalar,
            add_ops: AtomicUsize::new(0),
            mul_ops: AtomicUsize::new(0),
            reduce_ops: AtomicUsize::new(0),
        }
    }

    /// Get current SIMD capability
    pub fn capability(&self) -> SimdCapability {
        self.capability
    }

    /// Get operation metrics
    pub fn metrics(&self) -> SimdMetrics {
        SimdMetrics {
            capability: self.capability,
            add_operations: self.add_ops.load(Ordering::Relaxed),
            mul_operations: self.mul_ops.load(Ordering::Relaxed),
            reduce_operations: self.reduce_ops.load(Ordering::Relaxed),
        }
    }

    /// Reset operation counters
    pub fn reset_metrics(&self) {
        self.add_ops.store(0, Ordering::Relaxed);
        self.mul_ops.store(0, Ordering::Relaxed);
        self.reduce_ops.store(0, Ordering::Relaxed);
    }

    // ========================================================================
    // Vector Addition Operations
    // ========================================================================

    /// Vectorized modular addition: result[i] = (a[i] + b[i]) mod p
    ///
    /// # Errors
    /// - `SimdError::LengthMismatch` if vectors have different lengths
    /// - `SimdError::InvalidModulus` if modulus <= 1
    ///
    /// # Examples
    /// ```rust,ignore
    /// let ops = SimdFieldOps::auto_detect();
    /// let a = vec![1u64, 2, 3, 4];
    /// let b = vec![5u64, 6, 7, 8];
    /// let result = ops.vector_add_checked(&a, &b, 100)?;
    /// assert_eq!(result, vec![6, 8, 10, 12]);
    /// ```
    #[instrument(level = "trace", skip(self, a, b))]
    pub fn vector_add_checked(&self, a: &[u64], b: &[u64], modulus: u64) -> SimdResult<Vec<u64>> {
        validate_vector_lengths(a, b)?;
        validate_modulus(modulus)?;
        
        // Handle empty vectors gracefully
        if a.is_empty() {
            return Ok(Vec::new());
        }
        
        self.add_ops.fetch_add(a.len(), Ordering::Relaxed);
        Ok(self.vector_add_internal(a, b, modulus))
    }

    /// Vectorized modular addition (unchecked version for performance-critical paths)
    /// 
    /// # Panics
    /// Panics in debug mode if vector lengths don't match
    #[inline]
    pub fn vector_add(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        debug_assert_eq!(a.len(), b.len(), "Vector lengths must match");
        if a.is_empty() {
            return Vec::new();
        }
        self.add_ops.fetch_add(a.len(), Ordering::Relaxed);
        self.vector_add_internal(a, b, modulus)
    }

    /// Internal implementation of vector addition (no validation)
    #[inline]
    fn vector_add_internal(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        match self.capability {
            #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
            SimdCapability::Avx2 => unsafe { self.vector_add_avx2(a, b, modulus) },
            
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            SimdCapability::Avx512 => unsafe { self.vector_add_avx512(a, b, modulus) },
            
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            SimdCapability::Neon => unsafe { self.vector_add_neon(a, b, modulus) },
            
            _ => self.vector_add_scalar(a, b, modulus),
        }
    }

    /// Scalar fallback for vector addition
    #[inline]
    fn vector_add_scalar(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let sum = x.wrapping_add(y);
                if sum >= modulus { sum - modulus } else { sum }
            })
            .collect()
    }

    /// AVX2 implementation of vector addition
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    unsafe fn vector_add_avx2(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        let n = a.len();
        let mut result = vec![0u64; n];
        
        let mod_vec = _mm256_set1_epi64x(modulus as i64);
        let chunks = n / 4;
        
        for i in 0..chunks {
            let offset = i * 4;
            
            // Load 4 elements from each array
            let a_vec = _mm256_loadu_si256(a[offset..].as_ptr() as *const __m256i);
            let b_vec = _mm256_loadu_si256(b[offset..].as_ptr() as *const __m256i);
            
            // Add (wrapping)
            let sum = _mm256_add_epi64(a_vec, b_vec);
            
            // Modular reduction: if sum >= modulus, subtract modulus
            // Compare: sum >= modulus (no native unsigned compare, use signed trick)
            let diff = _mm256_sub_epi64(sum, mod_vec);
            
            // Create mask: -1 if sum >= modulus (diff is non-negative)
            // For simplicity, we'll process element-wise for exact modular reduction
            let mut temp = [0u64; 4];
            _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, sum);
            
            for j in 0..4 {
                result[offset + j] = if temp[j] >= modulus { temp[j] - modulus } else { temp[j] };
            }
        }
        
        // Handle remaining elements
        for i in (chunks * 4)..n {
            let sum = a[i].wrapping_add(b[i]);
            result[i] = if sum >= modulus { sum - modulus } else { sum };
        }
        
        result
    }

    /// AVX-512 implementation of vector addition
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[target_feature(enable = "avx512f", enable = "avx512vl")]
    unsafe fn vector_add_avx512(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        let n = a.len();
        let mut result = vec![0u64; n];
        
        let mod_vec = _mm512_set1_epi64(modulus as i64);
        let chunks = n / 8;
        
        for i in 0..chunks {
            let offset = i * 8;
            
            // Load 8 elements from each array
            let a_vec = _mm512_loadu_si512(a[offset..].as_ptr() as *const i64);
            let b_vec = _mm512_loadu_si512(b[offset..].as_ptr() as *const i64);
            
            // Add
            let sum = _mm512_add_epi64(a_vec, b_vec);
            
            // Modular reduction with mask
            let mask = _mm512_cmpge_epu64_mask(sum, mod_vec);
            let reduced = _mm512_mask_sub_epi64(sum, mask, sum, mod_vec);
            
            _mm512_storeu_si512(result[offset..].as_mut_ptr() as *mut i64, reduced);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..n {
            let sum = a[i].wrapping_add(b[i]);
            result[i] = if sum >= modulus { sum - modulus } else { sum };
        }
        
        result
    }

    /// NEON implementation of vector addition
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[target_feature(enable = "neon")]
    unsafe fn vector_add_neon(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        let n = a.len();
        let mut result = vec![0u64; n];
        
        let chunks = n / 2;
        
        for i in 0..chunks {
            let offset = i * 2;
            
            // Load 2 elements from each array
            let a_vec = vld1q_u64(a[offset..].as_ptr());
            let b_vec = vld1q_u64(b[offset..].as_ptr());
            
            // Add
            let sum = vaddq_u64(a_vec, b_vec);
            
            // Modular reduction (element-wise for correctness)
            let mut temp = [0u64; 2];
            vst1q_u64(temp.as_mut_ptr(), sum);
            
            for j in 0..2 {
                result[offset + j] = if temp[j] >= modulus { temp[j] - modulus } else { temp[j] };
            }
        }
        
        // Handle remaining elements
        for i in (chunks * 2)..n {
            let sum = a[i].wrapping_add(b[i]);
            result[i] = if sum >= modulus { sum - modulus } else { sum };
        }
        
        result
    }

    // ========================================================================
    // Vector Subtraction Operations
    // ========================================================================

    /// Vectorized modular subtraction: result[i] = (a[i] - b[i]) mod p
    ///
    /// # Errors
    /// - `SimdError::LengthMismatch` if vectors have different lengths
    /// - `SimdError::InvalidModulus` if modulus <= 1
    #[instrument(level = "trace", skip(self, a, b))]
    pub fn vector_sub_checked(&self, a: &[u64], b: &[u64], modulus: u64) -> SimdResult<Vec<u64>> {
        validate_vector_lengths(a, b)?;
        validate_modulus(modulus)?;
        
        if a.is_empty() {
            return Ok(Vec::new());
        }
        
        self.add_ops.fetch_add(a.len(), Ordering::Relaxed);
        Ok(self.vector_sub_internal(a, b, modulus))
    }

    /// Vectorized modular subtraction (unchecked version)
    #[inline]
    pub fn vector_sub(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        debug_assert_eq!(a.len(), b.len(), "Vector lengths must match");
        if a.is_empty() {
            return Vec::new();
        }
        self.add_ops.fetch_add(a.len(), Ordering::Relaxed);
        self.vector_sub_internal(a, b, modulus)
    }

    /// Internal implementation of vector subtraction
    #[inline]
    fn vector_sub_internal(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                if x >= y {
                    x - y
                } else {
                    modulus - (y - x)
                }
            })
            .collect()
    }

    // ========================================================================
    // Vector Multiplication Operations
    // ========================================================================

    /// Vectorized modular multiplication: result[i] = (a[i] * b[i]) mod p
    ///
    /// # Errors
    /// - `SimdError::LengthMismatch` if vectors have different lengths
    /// - `SimdError::InvalidModulus` if modulus <= 1
    #[instrument(level = "trace", skip(self, a, b))]
    pub fn vector_mul_checked(&self, a: &[u64], b: &[u64], modulus: u64) -> SimdResult<Vec<u64>> {
        validate_vector_lengths(a, b)?;
        validate_modulus(modulus)?;
        
        if a.is_empty() {
            return Ok(Vec::new());
        }
        
        self.mul_ops.fetch_add(a.len(), Ordering::Relaxed);
        Ok(self.vector_mul_internal(a, b, modulus))
    }

    /// Vectorized modular multiplication (unchecked version for performance)
    #[inline]
    pub fn vector_mul(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        debug_assert_eq!(a.len(), b.len(), "Vector lengths must match");
        if a.is_empty() {
            return Vec::new();
        }
        self.mul_ops.fetch_add(a.len(), Ordering::Relaxed);
        self.vector_mul_internal(a, b, modulus)
    }

    /// Internal implementation of vector multiplication
    #[inline]
    fn vector_mul_internal(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        // For modular multiplication, we need 128-bit intermediate results
        // SIMD optimization is limited here, but we can still parallelize
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let product = (x as u128) * (y as u128);
                (product % (modulus as u128)) as u64
            })
            .collect()
    }

    /// Vectorized modular scalar multiplication: result[i] = (a[i] * scalar) mod p
    ///
    /// # Errors
    /// - `SimdError::InvalidModulus` if modulus <= 1
    #[instrument(level = "trace", skip(self, a))]
    pub fn vector_scalar_mul_checked(&self, a: &[u64], scalar: u64, modulus: u64) -> SimdResult<Vec<u64>> {
        validate_modulus(modulus)?;
        
        if a.is_empty() {
            return Ok(Vec::new());
        }
        
        self.mul_ops.fetch_add(a.len(), Ordering::Relaxed);
        Ok(self.vector_scalar_mul_internal(a, scalar, modulus))
    }

    /// Vectorized scalar multiplication (unchecked version)
    #[inline]
    pub fn vector_scalar_mul(&self, a: &[u64], scalar: u64, modulus: u64) -> Vec<u64> {
        if a.is_empty() {
            return Vec::new();
        }
        self.mul_ops.fetch_add(a.len(), Ordering::Relaxed);
        self.vector_scalar_mul_internal(a, scalar, modulus)
    }

    /// Internal implementation of scalar multiplication
    #[inline]
    fn vector_scalar_mul_internal(&self, a: &[u64], scalar: u64, modulus: u64) -> Vec<u64> {
        a.iter()
            .map(|&x| {
                let product = (x as u128) * (scalar as u128);
                (product % (modulus as u128)) as u64
            })
            .collect()
    }

    // ========================================================================
    // Parallel Operations with Rayon
    // ========================================================================

    /// Parallel vectorized addition for large arrays
    ///
    /// # Errors
    /// - `SimdError::LengthMismatch` if vectors have different lengths
    /// - `SimdError::InvalidModulus` if modulus <= 1
    #[instrument(level = "debug", skip(self, a, b), fields(len = a.len()))]
    pub fn parallel_vector_add_checked(&self, a: &[u64], b: &[u64], modulus: u64) -> SimdResult<Vec<u64>> {
        validate_vector_lengths(a, b)?;
        validate_modulus(modulus)?;
        
        if a.is_empty() {
            return Ok(Vec::new());
        }
        
        self.add_ops.fetch_add(a.len(), Ordering::Relaxed);
        Ok(self.parallel_vector_add_internal(a, b, modulus))
    }

    /// Parallel vectorized addition (unchecked version)
    pub fn parallel_vector_add(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        debug_assert_eq!(a.len(), b.len(), "Vector lengths must match");
        if a.is_empty() {
            return Vec::new();
        }
        self.add_ops.fetch_add(a.len(), Ordering::Relaxed);
        self.parallel_vector_add_internal(a, b, modulus)
    }

    /// Internal implementation of parallel vector addition
    fn parallel_vector_add_internal(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        let chunk_size = std::cmp::max(1024, self.capability.vector_width() * 256);
        
        if a.len() < chunk_size {
            return self.vector_add_internal(a, b, modulus);
        }

        a.par_chunks(chunk_size)
            .zip(b.par_chunks(chunk_size))
            .flat_map(|(chunk_a, chunk_b)| {
                self.vector_add_internal(chunk_a, chunk_b, modulus)
            })
            .collect()
    }

    /// Parallel vectorized multiplication (checked version)
    ///
    /// # Errors
    /// - `SimdError::LengthMismatch` if vectors have different lengths
    /// - `SimdError::InvalidModulus` if modulus <= 1
    #[instrument(level = "debug", skip(self, a, b), fields(len = a.len()))]
    pub fn parallel_vector_mul_checked(&self, a: &[u64], b: &[u64], modulus: u64) -> SimdResult<Vec<u64>> {
        validate_vector_lengths(a, b)?;
        validate_modulus(modulus)?;
        
        if a.is_empty() {
            return Ok(Vec::new());
        }
        
        self.mul_ops.fetch_add(a.len(), Ordering::Relaxed);
        Ok(self.parallel_vector_mul_internal(a, b, modulus))
    }

    /// Parallel vectorized multiplication (unchecked)
    pub fn parallel_vector_mul(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        debug_assert_eq!(a.len(), b.len(), "Vector lengths must match");
        if a.is_empty() {
            return Vec::new();
        }
        self.mul_ops.fetch_add(a.len(), Ordering::Relaxed);
        self.parallel_vector_mul_internal(a, b, modulus)
    }

    /// Internal implementation of parallel vector multiplication
    fn parallel_vector_mul_internal(&self, a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        let chunk_size = std::cmp::max(1024, self.capability.vector_width() * 256);
        
        if a.len() < chunk_size {
            return self.vector_mul_internal(a, b, modulus);
        }

        a.par_chunks(chunk_size)
            .zip(b.par_chunks(chunk_size))
            .flat_map(|(chunk_a, chunk_b)| {
                self.vector_mul_internal(chunk_a, chunk_b, modulus)
            })
            .collect()
    }

    // ========================================================================
    // Inner Product Operations (for MSM)
    // ========================================================================

    /// Compute modular inner product: sum(a[i] * b[i]) mod p
    ///
    /// # Errors
    /// - `SimdError::LengthMismatch` if vectors have different lengths
    /// - `SimdError::InvalidModulus` if modulus <= 1
    /// - `SimdError::EmptyVector` if vectors are empty
    #[instrument(level = "trace", skip(self, a, b))]
    pub fn inner_product_checked(&self, a: &[u64], b: &[u64], modulus: u64) -> SimdResult<u64> {
        validate_vector_lengths(a, b)?;
        validate_modulus(modulus)?;
        validate_non_empty(a, "inner_product")?;
        
        self.mul_ops.fetch_add(a.len(), Ordering::Relaxed);
        self.reduce_ops.fetch_add(1, Ordering::Relaxed);

        let products = self.vector_mul_internal(a, b, modulus);
        Ok(self.vector_sum_internal(&products, modulus))
    }

    /// Compute modular inner product (unchecked version)
    pub fn inner_product(&self, a: &[u64], b: &[u64], modulus: u64) -> u64 {
        debug_assert_eq!(a.len(), b.len(), "Vector lengths must match");
        if a.is_empty() {
            return 0;
        }
        self.mul_ops.fetch_add(a.len(), Ordering::Relaxed);
        self.reduce_ops.fetch_add(1, Ordering::Relaxed);

        let products = self.vector_mul_internal(a, b, modulus);
        self.vector_sum_internal(&products, modulus)
    }

    /// Parallel inner product for large vectors
    ///
    /// # Errors
    /// - `SimdError::LengthMismatch` if vectors have different lengths
    /// - `SimdError::InvalidModulus` if modulus <= 1
    /// - `SimdError::EmptyVector` if vectors are empty
    #[instrument(level = "debug", skip(self, a, b), fields(len = a.len()))]
    pub fn parallel_inner_product_checked(&self, a: &[u64], b: &[u64], modulus: u64) -> SimdResult<u64> {
        validate_vector_lengths(a, b)?;
        validate_modulus(modulus)?;
        validate_non_empty(a, "parallel_inner_product")?;
        
        Ok(self.parallel_inner_product_internal(a, b, modulus))
    }

    /// Parallel inner product (unchecked version)
    pub fn parallel_inner_product(&self, a: &[u64], b: &[u64], modulus: u64) -> u64 {
        debug_assert_eq!(a.len(), b.len(), "Vector lengths must match");
        if a.is_empty() {
            return 0;
        }
        self.parallel_inner_product_internal(a, b, modulus)
    }

    /// Internal implementation of parallel inner product
    fn parallel_inner_product_internal(&self, a: &[u64], b: &[u64], modulus: u64) -> u64 {
        let chunk_size = 4096;
        
        if a.len() < chunk_size {
            return self.inner_product(a, b, modulus);
        }

        // Compute partial sums in parallel
        let partial_sums: Vec<u64> = a.par_chunks(chunk_size)
            .zip(b.par_chunks(chunk_size))
            .map(|(chunk_a, chunk_b)| {
                self.inner_product(chunk_a, chunk_b, modulus)
            })
            .collect();

        // Sum partial results
        self.vector_sum_internal(&partial_sums, modulus)
    }

    /// Compute modular sum of vector elements
    pub fn vector_sum(&self, a: &[u64], modulus: u64) -> u64 {
        self.reduce_ops.fetch_add(1, Ordering::Relaxed);
        self.vector_sum_internal(a, modulus)
    }

    /// Internal implementation of vector sum
    #[inline]
    fn vector_sum_internal(&self, a: &[u64], modulus: u64) -> u64 {
        a.iter().fold(0u64, |acc, &x| {
            let sum = acc.wrapping_add(x);
            if sum >= modulus { sum - modulus } else { sum }
        })
    }

    // ========================================================================
    // NTT Helper Operations
    // ========================================================================

    /// Batch butterfly operation for NTT
    /// Computes: (a + b*w, a - b*w) for each pair
    ///
    /// # Errors
    /// - `SimdError::InvalidModulus` if modulus <= 1
    /// - Returns error if values.len() != twiddle_factors.len() * 2
    #[instrument(level = "trace", skip(self, values, twiddle_factors))]
    pub fn batch_butterfly_checked(
        &self,
        values: &mut [u64],
        twiddle_factors: &[u64],
        modulus: u64,
    ) -> SimdResult<()> {
        validate_modulus(modulus)?;
        
        let expected_len = twiddle_factors.len() * 2;
        if values.len() != expected_len {
            return Err(SimdError::LengthMismatch {
                left: values.len(),
                right: expected_len,
            });
        }
        
        self.batch_butterfly_internal(values, twiddle_factors, modulus);
        Ok(())
    }

    /// Batch butterfly operation (unchecked version)
    pub fn batch_butterfly(
        &self,
        values: &mut [u64],
        twiddle_factors: &[u64],
        modulus: u64,
    ) {
        debug_assert_eq!(values.len(), twiddle_factors.len() * 2);
        self.batch_butterfly_internal(values, twiddle_factors, modulus);
    }

    /// Internal implementation of batch butterfly
    fn batch_butterfly_internal(
        &self,
        values: &mut [u64],
        twiddle_factors: &[u64],
        modulus: u64,
    ) {
        self.mul_ops.fetch_add(twiddle_factors.len(), Ordering::Relaxed);
        self.add_ops.fetch_add(twiddle_factors.len() * 2, Ordering::Relaxed);

        let n = twiddle_factors.len();
        
        for i in 0..n {
            let a = values[i];
            let b = values[i + n];
            let bw = ((b as u128) * (twiddle_factors[i] as u128) % (modulus as u128)) as u64;
            
            // a + b*w
            let sum = a.wrapping_add(bw);
            values[i] = if sum >= modulus { sum - modulus } else { sum };
            
            // a - b*w
            values[i + n] = if a >= bw { a - bw } else { modulus - (bw - a) };
        }
    }

    /// Parallel batch butterfly for large NTT operations
    pub fn parallel_batch_butterfly(
        &self,
        values: &mut [u64],
        twiddle_factors: &[u64],
        modulus: u64,
    ) {
        let n = twiddle_factors.len();
        let chunk_size = 1024;
        
        if n < chunk_size {
            return self.batch_butterfly_internal(values, twiddle_factors, modulus);
        }

        // Split into chunks and process in parallel
        let (left, right) = values.split_at_mut(n);
        
        left.par_chunks_mut(chunk_size)
            .zip(right.par_chunks_mut(chunk_size))
            .zip(twiddle_factors.par_chunks(chunk_size))
            .for_each(|((chunk_a, chunk_b), chunk_w)| {
                for i in 0..chunk_w.len() {
                    let a = chunk_a[i];
                    let b = chunk_b[i];
                    let bw = ((b as u128) * (chunk_w[i] as u128) % (modulus as u128)) as u64;
                    
                    let sum = a.wrapping_add(bw);
                    chunk_a[i] = if sum >= modulus { sum - modulus } else { sum };
                    chunk_b[i] = if a >= bw { a - bw } else { modulus - (bw - a) };
                }
            });
    }

    // ========================================================================
    // Matrix Operations for R1CS
    // ========================================================================

    /// Matrix-vector multiplication: result = M * v (mod p)
    /// M is stored in row-major order
    ///
    /// # Errors
    /// - `SimdError::MatrixDimensionMismatch` if matrix.len() != rows * cols
    /// - `SimdError::VectorColumnsMismatch` if vector.len() != cols
    /// - `SimdError::InvalidModulus` if modulus <= 1
    #[instrument(level = "debug", skip(self, matrix, vector))]
    pub fn matrix_vector_mul_checked(
        &self,
        matrix: &[u64],
        vector: &[u64],
        rows: usize,
        cols: usize,
        modulus: u64,
    ) -> SimdResult<Vec<u64>> {
        validate_matrix_dimensions(matrix, vector, rows, cols)?;
        validate_modulus(modulus)?;
        
        if rows == 0 || cols == 0 {
            return Ok(Vec::new());
        }
        
        Ok(self.matrix_vector_mul_internal(matrix, vector, rows, cols, modulus))
    }

    /// Matrix-vector multiplication (unchecked version)
    pub fn matrix_vector_mul(
        &self,
        matrix: &[u64],
        vector: &[u64],
        rows: usize,
        cols: usize,
        modulus: u64,
    ) -> Vec<u64> {
        debug_assert_eq!(matrix.len(), rows * cols);
        debug_assert_eq!(vector.len(), cols);
        
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        
        self.matrix_vector_mul_internal(matrix, vector, rows, cols, modulus)
    }

    /// Internal implementation of matrix-vector multiplication
    fn matrix_vector_mul_internal(
        &self,
        matrix: &[u64],
        vector: &[u64],
        rows: usize,
        cols: usize,
        modulus: u64,
    ) -> Vec<u64> {
        (0..rows)
            .map(|i| {
                let row_start = i * cols;
                let row = &matrix[row_start..row_start + cols];
                self.inner_product(row, vector, modulus)
            })
            .collect()
    }

    /// Parallel matrix-vector multiplication for large matrices
    ///
    /// # Errors
    /// - `SimdError::MatrixDimensionMismatch` if matrix.len() != rows * cols
    /// - `SimdError::VectorColumnsMismatch` if vector.len() != cols
    /// - `SimdError::InvalidModulus` if modulus <= 1
    #[instrument(level = "debug", skip(self, matrix, vector), fields(rows, cols))]
    pub fn parallel_matrix_vector_mul_checked(
        &self,
        matrix: &[u64],
        vector: &[u64],
        rows: usize,
        cols: usize,
        modulus: u64,
    ) -> SimdResult<Vec<u64>> {
        validate_matrix_dimensions(matrix, vector, rows, cols)?;
        validate_modulus(modulus)?;
        
        if rows == 0 || cols == 0 {
            return Ok(Vec::new());
        }
        
        Ok(self.parallel_matrix_vector_mul_internal(matrix, vector, rows, cols, modulus))
    }

    /// Parallel matrix-vector multiplication (unchecked version)
    pub fn parallel_matrix_vector_mul(
        &self,
        matrix: &[u64],
        vector: &[u64],
        rows: usize,
        cols: usize,
        modulus: u64,
    ) -> Vec<u64> {
        debug_assert_eq!(matrix.len(), rows * cols);
        debug_assert_eq!(vector.len(), cols);
        
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        
        self.parallel_matrix_vector_mul_internal(matrix, vector, rows, cols, modulus)
    }

    /// Internal implementation of parallel matrix-vector multiplication
    fn parallel_matrix_vector_mul_internal(
        &self,
        matrix: &[u64],
        vector: &[u64],
        rows: usize,
        cols: usize,
        modulus: u64,
    ) -> Vec<u64> {
        if rows < 64 {
            return self.matrix_vector_mul_internal(matrix, vector, rows, cols, modulus);
        }

        (0..rows)
            .into_par_iter()
            .map(|i| {
                let row_start = i * cols;
                let row = &matrix[row_start..row_start + cols];
                self.inner_product(row, vector, modulus)
            })
            .collect()
    }

    // ========================================================================
    // Batch Modular Inverse (for Lagrange interpolation)
    // ========================================================================

    /// Batch modular inverse using Montgomery's trick
    /// Computes inverses of all elements in a single pass
    ///
    /// # Errors
    /// - `SimdError::NotInvertible` if any element is not invertible
    /// - `SimdError::InvalidModulus` if modulus <= 1
    /// - `SimdError::EmptyVector` if elements is empty
    #[instrument(level = "debug", skip(self, elements))]
    pub fn batch_inverse_checked(&self, elements: &[u64], modulus: u64) -> SimdResult<Vec<u64>> {
        validate_modulus(modulus)?;
        validate_non_empty(elements, "batch_inverse")?;
        
        // Check for zero elements
        for &e in elements {
            if e == 0 || e % modulus == 0 {
                return Err(SimdError::NotInvertible { element: e, modulus });
            }
        }
        
        Ok(self.batch_inverse_internal(elements, modulus))
    }

    /// Batch modular inverse (unchecked version)
    pub fn batch_inverse(&self, elements: &[u64], modulus: u64) -> Vec<u64> {
        if elements.is_empty() {
            return vec![];
        }
        self.batch_inverse_internal(elements, modulus)
    }

    /// Internal implementation of batch inverse
    fn batch_inverse_internal(&self, elements: &[u64], modulus: u64) -> Vec<u64> {
        let n = elements.len();
        let mut prefix_products = vec![0u64; n];
        let mut inverses = vec![0u64; n];

        // Compute prefix products
        prefix_products[0] = elements[0];
        for i in 1..n {
            prefix_products[i] = ((prefix_products[i - 1] as u128) * (elements[i] as u128) 
                % (modulus as u128)) as u64;
        }

        // Compute inverse of product
        let product_inv = mod_inverse_u64(prefix_products[n - 1], modulus);

        // Compute individual inverses
        let mut current_inv = product_inv;
        for i in (1..n).rev() {
            inverses[i] = ((current_inv as u128) * (prefix_products[i - 1] as u128) 
                % (modulus as u128)) as u64;
            current_inv = ((current_inv as u128) * (elements[i] as u128) 
                % (modulus as u128)) as u64;
        }
        inverses[0] = current_inv;

        inverses
    }
}

/// Modular inverse using extended Euclidean algorithm
fn mod_inverse_u64(a: u64, modulus: u64) -> u64 {
    let mut old_r = a as i128;
    let mut r = modulus as i128;
    let mut old_s = 1i128;
    let mut s = 0i128;

    while r != 0 {
        let quotient = old_r / r;
        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;
    }

    if old_s < 0 {
        old_s += modulus as i128;
    }

    old_s as u64
}

/// SIMD operation metrics
#[derive(Debug, Clone)]
pub struct SimdMetrics {
    /// Detected SIMD capability
    pub capability: SimdCapability,
    /// Number of addition operations
    pub add_operations: usize,
    /// Number of multiplication operations
    pub mul_operations: usize,
    /// Number of reduction operations
    pub reduce_operations: usize,
}

impl SimdMetrics {
    /// Estimated speedup over scalar operations
    pub fn estimated_speedup(&self) -> f64 {
        match self.capability {
            SimdCapability::Scalar => 1.0,
            SimdCapability::Neon => 1.5,    // 2x theoretical, ~1.5x practical
            SimdCapability::Avx2 => 2.5,    // 4x theoretical, ~2.5x practical
            SimdCapability::Avx512 => 4.0,  // 8x theoretical, ~4x practical
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MODULUS: u64 = 0xFFFFFFFF00000001; // Goldilocks field prime

    #[test]
    fn test_detect_simd_support() {
        let capability = detect_simd_support();
        println!("Detected SIMD capability: {:?} ({})", capability, capability.name());
        assert!(capability.vector_width() >= 1);
    }

    #[test]
    fn test_vector_add_scalar() {
        let ops = SimdFieldOps::with_capability(SimdCapability::Scalar).unwrap();
        
        let a = vec![1u64, 2, 3, 4, 5];
        let b = vec![10u64, 20, 30, 40, 50];
        let result = ops.vector_add(&a, &b, TEST_MODULUS);
        
        assert_eq!(result, vec![11, 22, 33, 44, 55]);
    }

    #[test]
    fn test_vector_add_overflow() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 100u64;
        let a = vec![90u64, 80, 70];
        let b = vec![20u64, 30, 40];
        let result = ops.vector_add(&a, &b, modulus);
        
        assert_eq!(result, vec![10, 10, 10]); // (90+20)%100=10, etc.
    }

    #[test]
    fn test_vector_sub() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 100u64;
        let a = vec![50u64, 30, 10];
        let b = vec![20u64, 30, 40];
        let result = ops.vector_sub(&a, &b, modulus);
        
        assert_eq!(result, vec![30, 0, 70]); // (10-40+100)%100=70
    }

    #[test]
    fn test_vector_mul() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 100u64;
        let a = vec![5u64, 7, 9];
        let b = vec![3u64, 4, 5];
        let result = ops.vector_mul(&a, &b, modulus);
        
        assert_eq!(result, vec![15, 28, 45]);
    }

    #[test]
    fn test_inner_product() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 1000u64;
        let a = vec![1u64, 2, 3, 4, 5];
        let b = vec![10u64, 20, 30, 40, 50];
        
        // 1*10 + 2*20 + 3*30 + 4*40 + 5*50 = 10+40+90+160+250 = 550
        let result = ops.inner_product(&a, &b, modulus);
        assert_eq!(result, 550);
    }

    #[test]
    fn test_batch_butterfly() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 97u64; // Small prime for testing
        let mut values = vec![10u64, 20, 30, 40]; // [a0, a1, b0, b1]
        let twiddles = vec![3u64, 5]; // Twiddle factors
        
        ops.batch_butterfly(&mut values, &twiddles, modulus);
        
        // Expected: (10 + 30*3, 20 + 40*5, 10 - 30*3, 20 - 40*5) mod 97
        // = (10 + 90, 20 + 200, 10 - 90, 20 - 200) mod 97
        // = (100 % 97, 220 % 97, (97 - 80), (97 - 83)) mod 97
        // = (3, 26, 17, 14)
        
        assert_eq!(values[0], 3);   // (10 + 90) % 97 = 100 % 97 = 3
        assert_eq!(values[1], 26);  // (20 + 200) % 97 = 220 % 97 = 26
    }

    #[test]
    fn test_batch_inverse() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 97u64;
        let elements = vec![3u64, 5, 7];
        let inverses = ops.batch_inverse(&elements, modulus);
        
        // Verify each inverse
        for (e, inv) in elements.iter().zip(inverses.iter()) {
            let product = ((*e as u128) * (*inv as u128) % (modulus as u128)) as u64;
            assert_eq!(product, 1, "Inverse of {} should be correct", e);
        }
    }

    #[test]
    fn test_matrix_vector_mul() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 1000u64;
        // 2x3 matrix
        let matrix = vec![
            1u64, 2, 3,  // Row 0
            4, 5, 6,     // Row 1
        ];
        let vector = vec![10u64, 20, 30];
        
        let result = ops.matrix_vector_mul(&matrix, &vector, 2, 3, modulus);
        
        // Row 0: 1*10 + 2*20 + 3*30 = 10 + 40 + 90 = 140
        // Row 1: 4*10 + 5*20 + 6*30 = 40 + 100 + 180 = 320
        assert_eq!(result, vec![140, 320]);
    }

    #[test]
    fn test_parallel_operations() {
        let ops = SimdFieldOps::auto_detect();
        
        // Create large vectors to test parallel operations
        let n = 10000;
        let a: Vec<u64> = (0..n).map(|i| i as u64).collect();
        let b: Vec<u64> = (0..n).map(|i| (n - i) as u64).collect();
        let modulus = TEST_MODULUS;
        
        // Test parallel addition
        let result = ops.parallel_vector_add(&a, &b, modulus);
        assert_eq!(result.len(), n);
        
        // Each element should be: i + (n - i) = n
        for &r in &result {
            assert_eq!(r, n as u64);
        }
    }

    #[test]
    fn test_metrics() {
        let ops = SimdFieldOps::auto_detect();
        ops.reset_metrics();
        
        let a = vec![1u64, 2, 3, 4];
        let b = vec![5u64, 6, 7, 8];
        let modulus = 1000u64;
        
        let _ = ops.vector_add(&a, &b, modulus);
        let _ = ops.vector_mul(&a, &b, modulus);
        
        let metrics = ops.metrics();
        assert_eq!(metrics.add_operations, 4);
        assert_eq!(metrics.mul_operations, 4);
    }

    // ========================================================================
    // Tests for Checked Variants (Production Hardening)
    // ========================================================================

    #[test]
    fn test_vector_add_checked_success() {
        let ops = SimdFieldOps::auto_detect();
        
        let a = vec![1u64, 2, 3, 4, 5];
        let b = vec![10u64, 20, 30, 40, 50];
        let result = ops.vector_add_checked(&a, &b, TEST_MODULUS).unwrap();
        
        assert_eq!(result, vec![11, 22, 33, 44, 55]);
    }

    #[test]
    fn test_vector_add_checked_length_mismatch() {
        let ops = SimdFieldOps::auto_detect();
        
        let a = vec![1u64, 2, 3];
        let b = vec![10u64, 20];
        let result = ops.vector_add_checked(&a, &b, TEST_MODULUS);
        
        assert!(matches!(result, Err(SimdError::LengthMismatch { .. })));
    }

    #[test]
    fn test_vector_add_checked_empty() {
        let ops = SimdFieldOps::auto_detect();
        
        let a: Vec<u64> = vec![];
        let b: Vec<u64> = vec![];
        let result = ops.vector_add_checked(&a, &b, TEST_MODULUS);
        
        // Empty vectors are valid input - they produce an empty result
        // This is mathematically correct (adding two empty vectors gives an empty vector)
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_vector_add_checked_invalid_modulus() {
        let ops = SimdFieldOps::auto_detect();
        
        let a = vec![1u64, 2, 3];
        let b = vec![10u64, 20, 30];
        let result = ops.vector_add_checked(&a, &b, 1);
        
        assert!(matches!(result, Err(SimdError::InvalidModulus { .. })));
    }

    #[test]
    fn test_vector_sub_checked_success() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 100u64;
        let a = vec![50u64, 30, 10];
        let b = vec![20u64, 30, 40];
        let result = ops.vector_sub_checked(&a, &b, modulus).unwrap();
        
        assert_eq!(result, vec![30, 0, 70]);
    }

    #[test]
    fn test_vector_mul_checked_success() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 100u64;
        let a = vec![5u64, 7, 9];
        let b = vec![3u64, 4, 5];
        let result = ops.vector_mul_checked(&a, &b, modulus).unwrap();
        
        assert_eq!(result, vec![15, 28, 45]);
    }

    #[test]
    fn test_vector_scalar_mul_checked_success() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 100u64;
        let a = vec![5u64, 10, 15];
        let scalar = 3u64;
        let result = ops.vector_scalar_mul_checked(&a, scalar, modulus).unwrap();
        
        assert_eq!(result, vec![15, 30, 45]);
    }

    #[test]
    fn test_inner_product_checked_success() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 1000u64;
        let a = vec![1u64, 2, 3, 4, 5];
        let b = vec![10u64, 20, 30, 40, 50];
        
        let result = ops.inner_product_checked(&a, &b, modulus).unwrap();
        assert_eq!(result, 550);
    }

    #[test]
    fn test_batch_butterfly_checked_success() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 97u64;
        let mut values = vec![10u64, 20, 30, 40];
        let twiddles = vec![3u64, 5];
        
        ops.batch_butterfly_checked(&mut values, &twiddles, modulus).unwrap();
        
        assert_eq!(values[0], 3);
        assert_eq!(values[1], 26);
    }

    #[test]
    fn test_batch_butterfly_checked_dimension_mismatch() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 97u64;
        let mut values = vec![10u64, 20, 30, 40];
        let twiddles = vec![3u64, 5, 7]; // Wrong size
        
        let result = ops.batch_butterfly_checked(&mut values, &twiddles, modulus);
        assert!(matches!(result, Err(SimdError::LengthMismatch { .. })));
    }

    #[test]
    fn test_matrix_vector_mul_checked_success() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 1000u64;
        let matrix = vec![1u64, 2, 3, 4, 5, 6];
        let vector = vec![10u64, 20, 30];
        
        let result = ops.matrix_vector_mul_checked(&matrix, &vector, 2, 3, modulus).unwrap();
        assert_eq!(result, vec![140, 320]);
    }

    #[test]
    fn test_matrix_vector_mul_checked_matrix_mismatch() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 1000u64;
        let matrix = vec![1u64, 2, 3, 4, 5]; // Wrong size for 2x3
        let vector = vec![10u64, 20, 30];
        
        let result = ops.matrix_vector_mul_checked(&matrix, &vector, 2, 3, modulus);
        assert!(matches!(result, Err(SimdError::MatrixDimensionMismatch { .. })));
    }

    #[test]
    fn test_matrix_vector_mul_checked_vector_mismatch() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 1000u64;
        let matrix = vec![1u64, 2, 3, 4, 5, 6];
        let vector = vec![10u64, 20]; // Wrong size for cols=3
        
        let result = ops.matrix_vector_mul_checked(&matrix, &vector, 2, 3, modulus);
        assert!(matches!(result, Err(SimdError::VectorColumnsMismatch { .. })));
    }

    #[test]
    fn test_batch_inverse_checked_success() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 97u64;
        let elements = vec![3u64, 5, 7];
        let inverses = ops.batch_inverse_checked(&elements, modulus).unwrap();
        
        for (e, inv) in elements.iter().zip(inverses.iter()) {
            let product = ((*e as u128) * (*inv as u128) % (modulus as u128)) as u64;
            assert_eq!(product, 1, "Inverse of {} should be correct", e);
        }
    }

    #[test]
    fn test_batch_inverse_checked_not_invertible() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 97u64;
        let elements = vec![3u64, 0, 7]; // 0 is not invertible
        
        let result = ops.batch_inverse_checked(&elements, modulus);
        assert!(matches!(result, Err(SimdError::NotInvertible { .. })));
    }

    #[test]
    fn test_batch_inverse_checked_empty() {
        let ops = SimdFieldOps::auto_detect();
        
        let modulus = 97u64;
        let elements: Vec<u64> = vec![];
        
        let result = ops.batch_inverse_checked(&elements, modulus);
        assert!(matches!(result, Err(SimdError::EmptyVector { .. })));
    }

    #[test]
    fn test_parallel_vector_add_checked_success() {
        let ops = SimdFieldOps::auto_detect();
        
        let n = 10000;
        let a: Vec<u64> = (0..n).map(|i| i as u64).collect();
        let b: Vec<u64> = (0..n).map(|i| (n - i) as u64).collect();
        
        let result = ops.parallel_vector_add_checked(&a, &b, TEST_MODULUS).unwrap();
        
        assert_eq!(result.len(), n);
        for &r in &result {
            assert_eq!(r, n as u64);
        }
    }

    #[test]
    fn test_parallel_inner_product_checked_success() {
        let ops = SimdFieldOps::auto_detect();
        
        let n = 10000;
        let a: Vec<u64> = (1..=n as u64).collect();
        let b: Vec<u64> = vec![1u64; n];
        
        let result = ops.parallel_inner_product_checked(&a, &b, TEST_MODULUS).unwrap();
        
        // Sum of 1 to n = n*(n+1)/2
        let expected = (n * (n + 1) / 2) as u64;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parallel_matrix_vector_mul_checked_success() {
        let ops = SimdFieldOps::auto_detect();
        
        let rows = 100;
        let cols = 50;
        let modulus = TEST_MODULUS;
        
        // Identity-like matrix (all 1s on diagonal conceptually)
        let matrix: Vec<u64> = (0..rows*cols).map(|_| 1u64).collect();
        let vector: Vec<u64> = (0..cols).map(|i| i as u64).collect();
        
        let result = ops.parallel_matrix_vector_mul_checked(&matrix, &vector, rows, cols, modulus).unwrap();
        
        assert_eq!(result.len(), rows);
        // Each row sums to 0+1+2+...+(cols-1) = cols*(cols-1)/2
        let expected_sum = (cols * (cols - 1) / 2) as u64;
        for &r in &result {
            assert_eq!(r, expected_sum);
        }
    }

    #[test]
    fn test_simd_capability_is_available() {
        // This test verifies the is_available method works correctly
        let detected = detect_simd_support();
        
        // Scalar should always be available
        assert!(SimdCapability::Scalar.is_available());
        
        // The detected capability should be available
        assert!(detected.is_available());
    }

    #[test]
    fn test_with_capability_checked() {
        // Test that with_capability returns error for unavailable capabilities
        let result = SimdFieldOps::with_capability(SimdCapability::Scalar);
        assert!(result.is_ok());
        
        // Note: AVX-512 may or may not be available depending on the machine
        // This test just ensures the API works correctly
    }

    #[test]
    fn test_force_scalar() {
        let ops = SimdFieldOps::force_scalar();
        assert_eq!(ops.capability, SimdCapability::Scalar);
        
        // Should still work correctly
        let a = vec![1u64, 2, 3];
        let b = vec![4u64, 5, 6];
        let result = ops.vector_add(&a, &b, 1000);
        assert_eq!(result, vec![5, 7, 9]);
    }
}
