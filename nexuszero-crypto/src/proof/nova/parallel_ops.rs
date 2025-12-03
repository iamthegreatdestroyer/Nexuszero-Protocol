// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! Parallel Operations for Nova Proving System
//!
//! This module provides high-performance parallel implementations using Rayon
//! for multi-core CPU utilization. It complements the SIMD operations by
//! enabling horizontal scaling across CPU cores.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   Parallel Processing Pipeline                   │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  Input Data                                                     │
//! │      │                                                          │
//! │      ▼                                                          │
//! │  ┌─────────────────────────────────────────────────────┐       │
//! │  │              Adaptive Chunk Splitter                │       │
//! │  │  (Determines optimal chunk size based on data)      │       │
//! │  └─────────────────────────────────────────────────────┘       │
//! │      │                                                          │
//! │      ▼                                                          │
//! │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                 │
//! │  │ W0  │  │ W1  │  │ W2  │  │ W3  │  │ ... │  Workers        │
//! │  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘                 │
//! │     │        │        │        │        │                     │
//! │     ▼        ▼        ▼        ▼        ▼                     │
//! │  ┌─────────────────────────────────────────────────────┐       │
//! │  │              Result Aggregator                      │       │
//! │  │  (Combines partial results, maintains ordering)     │       │
//! │  └─────────────────────────────────────────────────────┘       │
//! │      │                                                          │
//! │      ▼                                                          │
//! │  Final Result                                                   │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Thread Pool Management
//!
//! The module uses Rayon's global thread pool with adaptive work stealing.
//! For compute-intensive tasks, we use `par_iter()` which automatically
//! splits work across available CPU cores.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nexuszero_crypto::proof::nova::parallel_ops::{ParallelProver, ParallelConfig};
//!
//! // Configure parallel operations
//! let config = ParallelConfig::auto();
//!
//! // Create parallel prover
//! let prover = ParallelProver::new(config);
//!
//! // Parallel MSM computation
//! let result = prover.parallel_msm(&bases, &scalars);
//! ```

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Configuration for parallel operations
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use (0 = auto-detect)
    pub num_threads: usize,
    /// Minimum chunk size for parallelization
    pub min_chunk_size: usize,
    /// Maximum chunk size per worker
    pub max_chunk_size: usize,
    /// Enable work stealing for load balancing
    pub work_stealing: bool,
    /// Threshold for parallel vs sequential execution
    pub parallel_threshold: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: 0, // Auto-detect
            min_chunk_size: 256,
            max_chunk_size: 16384,
            work_stealing: true,
            parallel_threshold: 1024,
        }
    }
}

impl ParallelConfig {
    /// Create with auto-detected settings
    pub fn auto() -> Self {
        let num_cpus = rayon::current_num_threads();
        Self {
            num_threads: num_cpus,
            min_chunk_size: 256,
            max_chunk_size: std::cmp::max(4096, 65536 / num_cpus),
            work_stealing: true,
            parallel_threshold: std::cmp::max(512, num_cpus * 64),
        }
    }

    /// Create for maximum throughput (large batch operations)
    pub fn high_throughput() -> Self {
        Self {
            num_threads: 0,
            min_chunk_size: 1024,
            max_chunk_size: 65536,
            work_stealing: true,
            parallel_threshold: 4096,
        }
    }

    /// Create for low latency (small batch operations)
    pub fn low_latency() -> Self {
        Self {
            num_threads: 0,
            min_chunk_size: 64,
            max_chunk_size: 2048,
            work_stealing: false,
            parallel_threshold: 256,
        }
    }

    /// Calculate optimal chunk size for given data size
    pub fn optimal_chunk_size(&self, data_size: usize) -> usize {
        let num_threads = if self.num_threads == 0 {
            rayon::current_num_threads()
        } else {
            self.num_threads
        };

        let ideal_chunks = num_threads * 4; // Allow work stealing
        let chunk_size = data_size / ideal_chunks;
        
        std::cmp::max(
            self.min_chunk_size,
            std::cmp::min(self.max_chunk_size, chunk_size)
        )
    }
}

/// Parallel operation metrics
#[derive(Debug, Clone)]
pub struct ParallelMetrics {
    /// Total operations performed
    pub total_operations: u64,
    /// Operations performed in parallel
    pub parallel_operations: u64,
    /// Operations performed sequentially
    pub sequential_operations: u64,
    /// Total time spent in parallel operations
    pub parallel_time: Duration,
    /// Number of threads used
    pub threads_used: usize,
    /// Average work per thread
    pub avg_work_per_thread: f64,
}

/// Parallel prover for multi-core computation
pub struct ParallelProver {
    /// Configuration
    config: ParallelConfig,
    /// Metrics tracking
    total_ops: AtomicU64,
    parallel_ops: AtomicU64,
    sequential_ops: AtomicU64,
    parallel_time_ns: AtomicU64,
}

impl ParallelProver {
    /// Create new parallel prover with configuration
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            total_ops: AtomicU64::new(0),
            parallel_ops: AtomicU64::new(0),
            sequential_ops: AtomicU64::new(0),
            parallel_time_ns: AtomicU64::new(0),
        }
    }

    /// Create with auto-detected configuration
    pub fn auto() -> Self {
        Self::new(ParallelConfig::auto())
    }

    /// Get configuration
    pub fn config(&self) -> &ParallelConfig {
        &self.config
    }

    /// Get metrics
    pub fn metrics(&self) -> ParallelMetrics {
        let threads = if self.config.num_threads == 0 {
            rayon::current_num_threads()
        } else {
            self.config.num_threads
        };
        
        let parallel = self.parallel_ops.load(Ordering::Relaxed);
        
        ParallelMetrics {
            total_operations: self.total_ops.load(Ordering::Relaxed),
            parallel_operations: parallel,
            sequential_operations: self.sequential_ops.load(Ordering::Relaxed),
            parallel_time: Duration::from_nanos(self.parallel_time_ns.load(Ordering::Relaxed)),
            threads_used: threads,
            avg_work_per_thread: if threads > 0 { parallel as f64 / threads as f64 } else { 0.0 },
        }
    }

    /// Reset metrics
    pub fn reset_metrics(&self) {
        self.total_ops.store(0, Ordering::Relaxed);
        self.parallel_ops.store(0, Ordering::Relaxed);
        self.sequential_ops.store(0, Ordering::Relaxed);
        self.parallel_time_ns.store(0, Ordering::Relaxed);
    }

    // ========================================================================
    // Parallel Map Operations
    // ========================================================================

    /// Parallel map with automatic chunking
    pub fn parallel_map<T, U, F>(&self, data: &[T], f: F) -> Vec<U>
    where
        T: Sync,
        U: Send,
        F: Fn(&T) -> U + Sync + Send,
    {
        let n = data.len();
        self.total_ops.fetch_add(n as u64, Ordering::Relaxed);

        if n < self.config.parallel_threshold {
            self.sequential_ops.fetch_add(n as u64, Ordering::Relaxed);
            return data.iter().map(f).collect();
        }

        let start = Instant::now();
        self.parallel_ops.fetch_add(n as u64, Ordering::Relaxed);

        let result: Vec<U> = data.par_iter().map(f).collect();

        self.parallel_time_ns.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        result
    }

    /// Parallel map with index
    pub fn parallel_map_indexed<T, U, F>(&self, data: &[T], f: F) -> Vec<U>
    where
        T: Sync,
        U: Send,
        F: Fn(usize, &T) -> U + Sync + Send,
    {
        let n = data.len();
        self.total_ops.fetch_add(n as u64, Ordering::Relaxed);

        if n < self.config.parallel_threshold {
            self.sequential_ops.fetch_add(n as u64, Ordering::Relaxed);
            return data.iter().enumerate().map(|(i, x)| f(i, x)).collect();
        }

        let start = Instant::now();
        self.parallel_ops.fetch_add(n as u64, Ordering::Relaxed);

        let result: Vec<U> = data.par_iter()
            .enumerate()
            .map(|(i, x)| f(i, x))
            .collect();

        self.parallel_time_ns.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        result
    }

    // ========================================================================
    // Parallel Reduce Operations
    // ========================================================================

    /// Parallel reduce with custom combiner
    pub fn parallel_reduce<T, U, M, R>(&self, data: &[T], identity: U, map: M, reduce: R) -> U
    where
        T: Sync,
        U: Send + Clone + Sync,
        M: Fn(&T) -> U + Sync + Send,
        R: Fn(U, U) -> U + Sync + Send,
    {
        let n = data.len();
        self.total_ops.fetch_add(n as u64, Ordering::Relaxed);

        if n < self.config.parallel_threshold {
            self.sequential_ops.fetch_add(n as u64, Ordering::Relaxed);
            return data.iter().map(&map).fold(identity, &reduce);
        }

        let start = Instant::now();
        self.parallel_ops.fetch_add(n as u64, Ordering::Relaxed);

        let result = data.par_iter()
            .map(&map)
            .reduce(|| identity.clone(), &reduce);

        self.parallel_time_ns.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        result
    }

    /// Parallel sum with modular arithmetic
    pub fn parallel_modular_sum(&self, data: &[u64], modulus: u64) -> u64 {
        self.parallel_reduce(
            data,
            0u64,
            |&x| x,
            |a, b| {
                let sum = a.wrapping_add(b);
                if sum >= modulus { sum - modulus } else { sum }
            }
        )
    }

    // ========================================================================
    // MSM (Multi-Scalar Multiplication) Operations
    // ========================================================================

    /// Parallel multi-scalar multiplication
    /// Computes: sum(scalars[i] * bases[i])
    /// Uses Pippenger's bucket method with parallel accumulation
    pub fn parallel_msm_u64(
        &self,
        bases: &[u64],
        scalars: &[u64],
        modulus: u64,
    ) -> u64 {
        debug_assert_eq!(bases.len(), scalars.len());
        
        let n = bases.len();
        self.total_ops.fetch_add(n as u64, Ordering::Relaxed);

        if n < self.config.parallel_threshold {
            self.sequential_ops.fetch_add(n as u64, Ordering::Relaxed);
            return self.msm_sequential(bases, scalars, modulus);
        }

        let start = Instant::now();
        self.parallel_ops.fetch_add(n as u64, Ordering::Relaxed);

        // Use Pippenger-like bucket method with parallel accumulation
        let result = self.msm_pippenger_parallel(bases, scalars, modulus);

        self.parallel_time_ns.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        result
    }

    /// Sequential MSM implementation
    fn msm_sequential(&self, bases: &[u64], scalars: &[u64], modulus: u64) -> u64 {
        bases.iter()
            .zip(scalars.iter())
            .fold(0u64, |acc, (&base, &scalar)| {
                let product = ((base as u128) * (scalar as u128) % (modulus as u128)) as u64;
                let sum = acc.wrapping_add(product);
                if sum >= modulus { sum - modulus } else { sum }
            })
    }

    /// Pippenger's MSM algorithm with parallel bucket accumulation
    fn msm_pippenger_parallel(
        &self,
        bases: &[u64],
        scalars: &[u64],
        modulus: u64,
    ) -> u64 {
        let n = bases.len();
        
        // Determine optimal window size based on input size
        let c = optimal_window_size(n);
        let num_windows = (64 + c - 1) / c; // 64-bit scalars
        let num_buckets = 1 << c;

        // Process each window in parallel
        let window_results: Vec<u64> = (0..num_windows)
            .into_par_iter()
            .map(|window_idx| {
                // Create buckets for this window
                let mut buckets = vec![0u64; num_buckets];
                
                let window_start = window_idx * c;
                let window_mask = ((1u64 << c) - 1) as usize;
                
                // Accumulate points into buckets
                for (base, scalar) in bases.iter().zip(scalars.iter()) {
                    let bucket_idx = ((scalar >> window_start) as usize) & window_mask;
                    if bucket_idx != 0 {
                        let sum = buckets[bucket_idx].wrapping_add(*base);
                        buckets[bucket_idx] = if sum >= modulus { sum - modulus } else { sum };
                    }
                }
                
                // Compute bucket sum using running sum method
                let mut running_sum = 0u64;
                let mut window_sum = 0u64;
                
                for i in (1..num_buckets).rev() {
                    let sum = running_sum.wrapping_add(buckets[i]);
                    running_sum = if sum >= modulus { sum - modulus } else { sum };
                    
                    let sum = window_sum.wrapping_add(running_sum);
                    window_sum = if sum >= modulus { sum - modulus } else { sum };
                }
                
                window_sum
            })
            .collect();

        // Combine window results: result = sum(window_result[i] * 2^(i*c))
        let mut result = 0u64;
        let mut multiplier = 1u64;
        
        for window_result in window_results {
            let product = ((window_result as u128) * (multiplier as u128) % (modulus as u128)) as u64;
            let sum = result.wrapping_add(product);
            result = if sum >= modulus { sum - modulus } else { sum };
            
            // multiplier *= 2^c
            for _ in 0..c {
                multiplier = (multiplier << 1) % modulus;
            }
        }
        
        result
    }

    // ========================================================================
    // NTT (Number Theoretic Transform) Operations
    // ========================================================================

    /// Parallel NTT using Cooley-Tukey algorithm
    pub fn parallel_ntt(
        &self,
        values: &mut [u64],
        omega: u64,
        modulus: u64,
    ) {
        let n = values.len();
        debug_assert!(n.is_power_of_two(), "NTT size must be power of 2");
        
        self.total_ops.fetch_add((n * (n.trailing_zeros() as usize)) as u64, Ordering::Relaxed);

        // Bit-reversal permutation
        self.bit_reverse_permutation(values);

        // Cooley-Tukey iterations
        let log_n = n.trailing_zeros() as usize;
        
        for s in 1..=log_n {
            let m = 1 << s;
            let m_half = m / 2;
            
            // Compute twiddle factor for this stage
            let exp = (n / m) as u64;
            let w_m = mod_pow(omega, exp, modulus);
            
            // Parallel processing of butterfly groups
            if n / m >= self.config.parallel_threshold / m_half {
                let start = Instant::now();
                
                // Process groups in parallel
                values.par_chunks_mut(m).for_each(|chunk| {
                    let mut w = 1u64;
                    for j in 0..m_half {
                        let u = chunk[j];
                        let t = ((chunk[j + m_half] as u128) * (w as u128) % (modulus as u128)) as u64;
                        
                        let sum = u.wrapping_add(t);
                        chunk[j] = if sum >= modulus { sum - modulus } else { sum };
                        
                        chunk[j + m_half] = if u >= t { u - t } else { modulus - (t - u) };
                        
                        w = ((w as u128) * (w_m as u128) % (modulus as u128)) as u64;
                    }
                });
                
                self.parallel_time_ns.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                self.parallel_ops.fetch_add((n / m * m_half) as u64, Ordering::Relaxed);
            } else {
                // Sequential processing for small stages
                for k in (0..n).step_by(m) {
                    let mut w = 1u64;
                    for j in 0..m_half {
                        let u = values[k + j];
                        let t = ((values[k + j + m_half] as u128) * (w as u128) % (modulus as u128)) as u64;
                        
                        let sum = u.wrapping_add(t);
                        values[k + j] = if sum >= modulus { sum - modulus } else { sum };
                        
                        values[k + j + m_half] = if u >= t { u - t } else { modulus - (t - u) };
                        
                        w = ((w as u128) * (w_m as u128) % (modulus as u128)) as u64;
                    }
                }
                self.sequential_ops.fetch_add((n / m * m_half) as u64, Ordering::Relaxed);
            }
        }
    }

    /// Parallel inverse NTT
    pub fn parallel_intt(
        &self,
        values: &mut [u64],
        omega_inv: u64,
        n_inv: u64,
        modulus: u64,
    ) {
        // Forward NTT with inverse omega
        self.parallel_ntt(values, omega_inv, modulus);
        
        // Scale by n^-1
        let n = values.len();
        if n >= self.config.parallel_threshold {
            values.par_iter_mut().for_each(|v| {
                *v = ((*v as u128) * (n_inv as u128) % (modulus as u128)) as u64;
            });
        } else {
            for v in values.iter_mut() {
                *v = ((*v as u128) * (n_inv as u128) % (modulus as u128)) as u64;
            }
        }
    }

    /// Bit-reversal permutation
    fn bit_reverse_permutation(&self, values: &mut [u64]) {
        let n = values.len();
        let log_n = n.trailing_zeros() as usize;
        
        for i in 0..n {
            let j = bit_reverse(i, log_n);
            if i < j {
                values.swap(i, j);
            }
        }
    }

    // ========================================================================
    // Polynomial Operations
    // ========================================================================

    /// Parallel polynomial evaluation using Horner's method
    pub fn parallel_poly_eval(
        &self,
        coeffs: &[u64],
        points: &[u64],
        modulus: u64,
    ) -> Vec<u64> {
        self.parallel_map(points, |&x| {
            // Horner's method for evaluation
            coeffs.iter().rev().fold(0u64, |acc, &coeff| {
                let product = ((acc as u128) * (x as u128) % (modulus as u128)) as u64;
                let sum = product.wrapping_add(coeff);
                if sum >= modulus { sum - modulus } else { sum }
            })
        })
    }

    /// Parallel polynomial multiplication using NTT
    pub fn parallel_poly_mul(
        &self,
        a: &[u64],
        b: &[u64],
        omega: u64,
        omega_inv: u64,
        modulus: u64,
    ) -> Vec<u64> {
        let n = (a.len() + b.len() - 1).next_power_of_two();
        let n_inv = mod_inverse_u64(n as u64, modulus);
        
        // Pad and copy inputs
        let mut a_ntt = vec![0u64; n];
        let mut b_ntt = vec![0u64; n];
        a_ntt[..a.len()].copy_from_slice(a);
        b_ntt[..b.len()].copy_from_slice(b);
        
        // Forward NTT
        self.parallel_ntt(&mut a_ntt, omega, modulus);
        self.parallel_ntt(&mut b_ntt, omega, modulus);
        
        // Point-wise multiplication
        let result_ntt: Vec<u64> = if n >= self.config.parallel_threshold {
            a_ntt.par_iter()
                .zip(b_ntt.par_iter())
                .map(|(&x, &y)| {
                    ((x as u128) * (y as u128) % (modulus as u128)) as u64
                })
                .collect()
        } else {
            a_ntt.iter()
                .zip(b_ntt.iter())
                .map(|(&x, &y)| {
                    ((x as u128) * (y as u128) % (modulus as u128)) as u64
                })
                .collect()
        };
        
        // Inverse NTT
        let mut result = result_ntt;
        self.parallel_intt(&mut result, omega_inv, n_inv, modulus);
        
        // Trim to correct size
        result.truncate(a.len() + b.len() - 1);
        result
    }

    // ========================================================================
    // R1CS Operations
    // ========================================================================

    /// Parallel sparse matrix-vector multiplication for R1CS
    /// Computes: result[i] = sum(matrix[i][j] * vector[j]) for sparse matrix
    pub fn parallel_sparse_matvec(
        &self,
        row_indices: &[usize],
        col_indices: &[usize],
        values: &[u64],
        vector: &[u64],
        num_rows: usize,
        modulus: u64,
    ) -> Vec<u64> {
        debug_assert_eq!(row_indices.len(), col_indices.len());
        debug_assert_eq!(row_indices.len(), values.len());
        
        let n = row_indices.len();
        self.total_ops.fetch_add(n as u64, Ordering::Relaxed);

        // Group entries by row for parallel processing
        let mut row_groups: Vec<Vec<(usize, u64)>> = vec![Vec::new(); num_rows];
        for ((&row, &col), &val) in row_indices.iter().zip(col_indices.iter()).zip(values.iter()) {
            row_groups[row].push((col, val));
        }

        // Process rows in parallel
        if num_rows >= self.config.parallel_threshold {
            let start = Instant::now();
            self.parallel_ops.fetch_add(n as u64, Ordering::Relaxed);

            let result: Vec<u64> = row_groups.par_iter()
                .map(|entries| {
                    entries.iter().fold(0u64, |acc, &(col, val)| {
                        let product = ((val as u128) * (vector[col] as u128) % (modulus as u128)) as u64;
                        let sum = acc.wrapping_add(product);
                        if sum >= modulus { sum - modulus } else { sum }
                    })
                })
                .collect();

            self.parallel_time_ns.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            result
        } else {
            self.sequential_ops.fetch_add(n as u64, Ordering::Relaxed);
            row_groups.iter()
                .map(|entries| {
                    entries.iter().fold(0u64, |acc, &(col, val)| {
                        let product = ((val as u128) * (vector[col] as u128) % (modulus as u128)) as u64;
                        let sum = acc.wrapping_add(product);
                        if sum >= modulus { sum - modulus } else { sum }
                    })
                })
                .collect()
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute optimal window size for Pippenger's algorithm
fn optimal_window_size(n: usize) -> usize {
    if n < 32 {
        2
    } else if n < 1024 {
        4
    } else if n < 16384 {
        6
    } else if n < 262144 {
        8
    } else {
        10
    }
}

/// Bit-reverse an index
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Modular exponentiation
fn mod_pow(base: u64, exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    
    let mut result = 1u128;
    let mut base = (base as u128) % (modulus as u128);
    let mut exp = exp;
    
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % (modulus as u128);
        }
        exp >>= 1;
        base = (base * base) % (modulus as u128);
    }
    
    result as u64
}

/// Modular inverse
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MODULUS: u64 = 97; // Small prime for testing

    #[test]
    fn test_parallel_config_auto() {
        let config = ParallelConfig::auto();
        assert!(config.num_threads > 0);
        assert!(config.min_chunk_size > 0);
    }

    #[test]
    fn test_parallel_map() {
        let prover = ParallelProver::auto();
        
        let data: Vec<u64> = (0..1000).collect();
        let result = prover.parallel_map(&data, |&x| x * 2);
        
        assert_eq!(result.len(), data.len());
        for (i, &r) in result.iter().enumerate() {
            assert_eq!(r, i as u64 * 2);
        }
    }

    #[test]
    fn test_parallel_modular_sum() {
        let prover = ParallelProver::auto();
        
        let data: Vec<u64> = vec![10, 20, 30, 40, 50];
        let result = prover.parallel_modular_sum(&data, 100);
        
        // 10 + 20 + 30 + 40 + 50 = 150 % 100 = 50
        assert_eq!(result, 50);
    }

    #[test]
    fn test_msm_sequential() {
        let prover = ParallelProver::auto();
        
        let bases = vec![1u64, 2, 3, 4, 5];
        let scalars = vec![10u64, 20, 30, 40, 50];
        
        // 1*10 + 2*20 + 3*30 + 4*40 + 5*50 = 10 + 40 + 90 + 160 + 250 = 550
        let result = prover.msm_sequential(&bases, &scalars, 1000);
        assert_eq!(result, 550);
    }

    #[test]
    fn test_parallel_msm() {
        let prover = ParallelProver::auto();
        
        // Create larger input for parallel execution
        let n = 2048;
        let bases: Vec<u64> = (1..=n as u64).collect();
        let scalars: Vec<u64> = (1..=n as u64).collect();
        
        // Use large modulus to avoid overflow issues
        let modulus = 0xFFFFFFFF00000001u64; // Goldilocks prime
        
        let result_seq = prover.msm_sequential(&bases, &scalars, modulus);
        let result_par = prover.parallel_msm_u64(&bases, &scalars, modulus);
        
        // Both should give same result
        assert_eq!(result_seq, result_par);
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0b000, 3), 0b000);
        assert_eq!(bit_reverse(0b001, 3), 0b100);
        assert_eq!(bit_reverse(0b010, 3), 0b010);
        assert_eq!(bit_reverse(0b011, 3), 0b110);
        assert_eq!(bit_reverse(0b100, 3), 0b001);
    }

    #[test]
    fn test_mod_pow() {
        // 3^4 mod 97 = 81
        assert_eq!(mod_pow(3, 4, 97), 81);
        
        // 2^10 mod 100 = 1024 mod 100 = 24
        assert_eq!(mod_pow(2, 10, 100), 24);
    }

    #[test]
    fn test_parallel_ntt() {
        let prover = ParallelProver::auto();
        
        // Use a known NTT-friendly prime: p = 65537 = 2^16 + 1
        // This is a Fermat prime with primitive root 3
        let modulus = 65537u64;
        let n = 8usize;
        
        // Find primitive n-th root of unity
        // omega = g^((p-1)/n) where g is primitive root
        let g = 3u64;
        let exp = (modulus - 1) / (n as u64);
        let omega = mod_pow(g, exp, modulus);
        
        // Test values
        let mut values: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let original = values.clone();
        
        // Forward NTT
        prover.parallel_ntt(&mut values, omega, modulus);
        
        // Values should be changed
        assert_ne!(values, original);
        
        // Inverse NTT
        let omega_inv = mod_inverse_u64(omega, modulus);
        let n_inv = mod_inverse_u64(n as u64, modulus);
        prover.parallel_intt(&mut values, omega_inv, n_inv, modulus);
        
        // Should recover original
        assert_eq!(values, original);
    }

    #[test]
    fn test_parallel_sparse_matvec() {
        let prover = ParallelProver::auto();
        
        // 3x3 sparse matrix:
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        let row_indices = vec![0, 0, 1, 2, 2];
        let col_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1u64, 2, 3, 4, 5];
        let vector = vec![10u64, 20, 30];
        
        let result = prover.parallel_sparse_matvec(
            &row_indices, &col_indices, &values, &vector, 3, 1000
        );
        
        // Row 0: 1*10 + 2*30 = 70
        // Row 1: 3*20 = 60
        // Row 2: 4*10 + 5*30 = 190
        assert_eq!(result, vec![70, 60, 190]);
    }

    #[test]
    fn test_parallel_poly_eval() {
        let prover = ParallelProver::auto();
        
        // Polynomial: 1 + 2x + 3x^2
        let coeffs = vec![1u64, 2, 3];
        let points = vec![0u64, 1, 2];
        
        let result = prover.parallel_poly_eval(&coeffs, &points, 1000);
        
        // At x=0: 1 + 0 + 0 = 1
        // At x=1: 1 + 2 + 3 = 6
        // At x=2: 1 + 4 + 12 = 17
        assert_eq!(result, vec![1, 6, 17]);
    }

    #[test]
    fn test_metrics() {
        let prover = ParallelProver::auto();
        prover.reset_metrics();
        
        // Perform some operations
        let data: Vec<u64> = (0..100).collect();
        let _ = prover.parallel_map(&data, |&x| x + 1);
        let _ = prover.parallel_modular_sum(&data, 1000);
        
        let metrics = prover.metrics();
        assert!(metrics.total_operations > 0);
    }
}
