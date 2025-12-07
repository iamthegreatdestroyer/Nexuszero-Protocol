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
// Parallel Processing Units" - Claims 1-10 covering GPU kernel abstraction,
// MSM acceleration, NTT optimization, and adaptive backend selection.
// See legal/patents/PROVISIONAL_PATENT_CLAIMS.md
//
// TRADEMARK NOTICE: NexusZero Protocol™, Privacy Morphing™, Holographic Proof
// Compression™, Nova Folding Engine™, and related marks are trademarks of
// NexusZero Protocol. All Rights Reserved.
//
// See legal/IP_INNOVATIONS_REGISTRY.md for full intellectual property terms.

//! GPU Kernel Interfaces for Nova Proving Operations
//!
//! This module provides abstract interfaces for GPU-accelerated cryptographic
//! operations, designed to work with multiple GPU backends (CUDA, Metal, WebGPU).
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                     GPU Kernel Manager                         │
//! ├────────────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
//! │  │  MSM Kernel  │  │  NTT Kernel  │  │ Field Kernel │         │
//! │  └──────────────┘  └──────────────┘  └──────────────┘         │
//! │          │                │                │                   │
//! │          ▼                ▼                ▼                   │
//! │  ┌───────────────────────────────────────────────────────┐    │
//! │  │              GPU Memory Manager                        │    │
//! │  │  (Unified buffers, staging, async transfers)           │    │
//! │  └───────────────────────────────────────────────────────┘    │
//! │          │                │                │                   │
//! │          ▼                ▼                ▼                   │
//! │  ┌───────────┐    ┌───────────┐    ┌───────────┐             │
//! │  │   CUDA    │    │   Metal   │    │  WebGPU   │             │
//! │  └───────────┘    └───────────┘    └───────────┘             │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Supported Operations
//!
//! - **MSM (Multi-Scalar Multiplication)**: Pippenger algorithm on GPU
//! - **NTT (Number Theoretic Transform)**: Parallel butterfly operations
//! - **Field Operations**: Batch modular arithmetic

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::types::{NovaError, NovaResult};

// =============================================================================
// GPU Error Types (Production Hardening)
// =============================================================================

/// Comprehensive GPU-specific errors with detailed context
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GPUError {
    /// Memory allocation failed
    AllocationFailed {
        requested: usize,
        available: u64,
        backend: GPUBackend,
    },
    /// Buffer not found
    BufferNotFound {
        handle_id: u64,
    },
    /// Invalid buffer size
    InvalidBufferSize {
        size: usize,
        reason: &'static str,
    },
    /// Memory limit exceeded
    MemoryLimitExceeded {
        current: u64,
        requested: usize,
        limit: u64,
    },
    /// Invalid kernel parameters
    InvalidKernelParams {
        kernel: &'static str,
        reason: String,
    },
    /// Kernel execution failed
    KernelExecutionFailed {
        kernel: String,
        reason: String,
    },
    /// Data size mismatch
    DataSizeMismatch {
        expected: usize,
        actual: usize,
        context: &'static str,
    },
    /// Lock contention error
    LockContention {
        operation: &'static str,
    },
    /// Backend not available
    BackendNotAvailable {
        backend: GPUBackend,
    },
    /// Invalid NTT size (must be power of 2)
    InvalidNTTSize {
        size: usize,
    },
    /// Zero modulus error
    ZeroModulus,
    /// Invalid root of unity
    InvalidRootOfUnity {
        root: u64,
        modulus: u64,
    },
    /// Device not initialized
    DeviceNotInitialized,
    /// Transfer failed (H2D or D2H)
    TransferFailed {
        direction: &'static str,
        size: usize,
    },
}

impl std::fmt::Display for GPUError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GPUError::AllocationFailed { requested, available, backend } => {
                write!(f, "GPU allocation failed on {:?}: requested {} bytes, {} available", 
                       backend, requested, available)
            }
            GPUError::BufferNotFound { handle_id } => {
                write!(f, "GPU buffer not found: handle {}", handle_id)
            }
            GPUError::InvalidBufferSize { size, reason } => {
                write!(f, "Invalid buffer size {}: {}", size, reason)
            }
            GPUError::MemoryLimitExceeded { current, requested, limit } => {
                write!(f, "GPU memory limit exceeded: {} + {} > {}", current, requested, limit)
            }
            GPUError::InvalidKernelParams { kernel, reason } => {
                write!(f, "Invalid {} kernel params: {}", kernel, reason)
            }
            GPUError::KernelExecutionFailed { kernel, reason } => {
                write!(f, "Kernel {} execution failed: {}", kernel, reason)
            }
            GPUError::DataSizeMismatch { expected, actual, context } => {
                write!(f, "Data size mismatch in {}: expected {}, got {}", context, expected, actual)
            }
            GPUError::LockContention { operation } => {
                write!(f, "Lock contention during {}", operation)
            }
            GPUError::BackendNotAvailable { backend } => {
                write!(f, "GPU backend {:?} not available", backend)
            }
            GPUError::InvalidNTTSize { size } => {
                write!(f, "Invalid NTT size {}: must be power of 2", size)
            }
            GPUError::ZeroModulus => write!(f, "Modulus cannot be zero"),
            GPUError::InvalidRootOfUnity { root, modulus } => {
                write!(f, "Invalid root of unity {} for modulus {}", root, modulus)
            }
            GPUError::DeviceNotInitialized => write!(f, "GPU device not initialized"),
            GPUError::TransferFailed { direction, size } => {
                write!(f, "{} transfer failed for {} bytes", direction, size)
            }
        }
    }
}

impl std::error::Error for GPUError {}

/// Result type for GPU operations
pub type GPUResult<T> = Result<T, GPUError>;

// =============================================================================
// GPU Backend Types
// =============================================================================

/// Supported GPU backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GPUBackend {
    /// NVIDIA CUDA
    Cuda,
    /// Apple Metal
    Metal,
    /// WebGPU (cross-platform)
    WebGpu,
    /// CPU fallback
    Cpu,
}

impl GPUBackend {
    /// Check if this is a GPU backend
    pub fn is_gpu(&self) -> bool {
        !matches!(self, GPUBackend::Cpu)
    }

    /// Get backend name
    pub fn name(&self) -> &'static str {
        match self {
            GPUBackend::Cuda => "CUDA",
            GPUBackend::Metal => "Metal",
            GPUBackend::WebGpu => "WebGPU",
            GPUBackend::Cpu => "CPU",
        }
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GPUDeviceInfo {
    /// Device name
    pub name: String,
    /// Backend type
    pub backend: GPUBackend,
    /// Device index
    pub device_id: usize,
    /// Total memory in bytes
    pub total_memory: u64,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<(u32, u32)>,
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Maximum shared memory per block
    pub max_shared_memory_per_block: u32,
    /// Number of multiprocessors
    pub multiprocessor_count: u32,
}

impl Default for GPUDeviceInfo {
    fn default() -> Self {
        Self {
            name: "CPU Fallback".to_string(),
            backend: GPUBackend::Cpu,
            device_id: 0,
            total_memory: 0,
            available_memory: 0,
            compute_capability: None,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 48 * 1024, // 48 KB
            multiprocessor_count: 1,
        }
    }
}

// =============================================================================
// Kernel Parameters
// =============================================================================

/// Parameters for MSM kernel
#[derive(Debug, Clone)]
pub struct MSMKernelParams {
    /// Number of points
    pub num_points: usize,
    /// Scalar bit width
    pub scalar_bits: u32,
    /// Window size for Pippenger algorithm
    pub window_size: u32,
    /// Use precomputation
    pub use_precomputation: bool,
    /// Batch size for accumulation
    pub batch_size: usize,
}

impl Default for MSMKernelParams {
    fn default() -> Self {
        Self {
            num_points: 0,
            scalar_bits: 256,
            window_size: 16,
            use_precomputation: true,
            batch_size: 1 << 16, // 64K points per batch
        }
    }
}

impl MSMKernelParams {
    /// Create params for given number of points
    pub fn for_size(num_points: usize) -> Self {
        let window_size = if num_points < 1024 {
            8
        } else if num_points < 65536 {
            12
        } else {
            16
        };

        Self {
            num_points,
            window_size,
            ..Default::default()
        }
    }
}

/// Parameters for NTT kernel
#[derive(Debug, Clone)]
pub struct NTTKernelParams {
    /// Size of NTT (must be power of 2)
    pub size: usize,
    /// Log2 of size
    pub log_size: u32,
    /// Modulus for field operations
    pub modulus: u64,
    /// Root of unity
    pub root_of_unity: u64,
    /// Inverse root of unity (for INTT)
    pub inv_root_of_unity: u64,
    /// Batch count
    pub batch_count: usize,
    /// Use shared memory
    pub use_shared_memory: bool,
}

impl Default for NTTKernelParams {
    fn default() -> Self {
        Self {
            size: 0,
            log_size: 0,
            modulus: 0,
            root_of_unity: 0,
            inv_root_of_unity: 0,
            batch_count: 1,
            use_shared_memory: true,
        }
    }
}

impl NTTKernelParams {
    /// Create params for given size and modulus
    pub fn new(size: usize, modulus: u64, root_of_unity: u64) -> Self {
        assert!(size.is_power_of_two(), "NTT size must be power of 2");
        let log_size = size.trailing_zeros();
        
        // Compute inverse root of unity
        let inv_root = mod_inverse(root_of_unity, modulus);
        
        Self {
            size,
            log_size,
            modulus,
            root_of_unity,
            inv_root_of_unity: inv_root,
            ..Default::default()
        }
    }

    /// Create params with full validation (checked variant)
    pub fn new_checked(size: usize, modulus: u64, root_of_unity: u64) -> GPUResult<Self> {
        // Validate size is power of 2
        if !size.is_power_of_two() {
            return Err(GPUError::InvalidNTTSize { size });
        }

        // Validate size is reasonable
        if size == 0 {
            return Err(GPUError::InvalidNTTSize { size: 0 });
        }

        // Validate modulus
        if modulus == 0 {
            return Err(GPUError::ZeroModulus);
        }

        // Validate root of unity
        if root_of_unity == 0 || root_of_unity >= modulus {
            return Err(GPUError::InvalidRootOfUnity { root: root_of_unity, modulus });
        }

        let log_size = size.trailing_zeros();
        
        // Compute inverse root of unity
        let inv_root = mod_inverse(root_of_unity, modulus);
        
        Ok(Self {
            size,
            log_size,
            modulus,
            root_of_unity,
            inv_root_of_unity: inv_root,
            ..Default::default()
        })
    }

    /// Validate these parameters
    pub fn validate(&self) -> GPUResult<()> {
        if !self.size.is_power_of_two() {
            return Err(GPUError::InvalidNTTSize { size: self.size });
        }
        if self.modulus == 0 {
            return Err(GPUError::ZeroModulus);
        }
        if self.root_of_unity == 0 || self.root_of_unity >= self.modulus {
            return Err(GPUError::InvalidRootOfUnity { 
                root: self.root_of_unity, 
                modulus: self.modulus 
            });
        }
        Ok(())
    }
}

/// Parameters for field operations kernel
#[derive(Debug, Clone)]
pub struct FieldKernelParams {
    /// Number of elements
    pub num_elements: usize,
    /// Modulus
    pub modulus: u64,
    /// Montgomery R value
    pub montgomery_r: u64,
    /// Montgomery R^2 mod p
    pub montgomery_r2: u64,
    /// Montgomery constant (-p^(-1) mod R)
    pub montgomery_inv: u64,
}

impl Default for FieldKernelParams {
    fn default() -> Self {
        Self {
            num_elements: 0,
            modulus: 0,
            montgomery_r: 0,
            montgomery_r2: 0,
            montgomery_inv: 0,
        }
    }
}

// =============================================================================
// GPU Buffer Management
// =============================================================================

/// GPU buffer handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GPUBufferHandle(u64);

impl GPUBufferHandle {
    /// Create a new handle
    fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID
    pub fn id(&self) -> u64 {
        self.0
    }
}

/// GPU buffer type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferType {
    /// Device-only memory (fastest)
    Device,
    /// Host-visible memory (slower but accessible from CPU)
    HostVisible,
    /// Unified memory (automatic migration)
    Unified,
    /// Staging buffer for transfers
    Staging,
}

/// GPU buffer descriptor
#[derive(Debug, Clone)]
pub struct GPUBufferDesc {
    /// Buffer size in bytes
    pub size: usize,
    /// Buffer type
    pub buffer_type: BufferType,
    /// Usage flags
    pub usage: BufferUsage,
    /// Name for debugging
    pub name: String,
}

/// Buffer usage flags
#[derive(Debug, Clone, Copy)]
pub struct BufferUsage {
    /// Can be read by kernel
    pub kernel_read: bool,
    /// Can be written by kernel
    pub kernel_write: bool,
    /// Can be copied to
    pub copy_dst: bool,
    /// Can be copied from
    pub copy_src: bool,
}

impl Default for BufferUsage {
    fn default() -> Self {
        Self {
            kernel_read: true,
            kernel_write: true,
            copy_dst: true,
            copy_src: true,
        }
    }
}

impl BufferUsage {
    /// Read-only buffer
    pub fn read_only() -> Self {
        Self {
            kernel_read: true,
            kernel_write: false,
            copy_dst: true,
            copy_src: false,
        }
    }

    /// Write-only buffer
    pub fn write_only() -> Self {
        Self {
            kernel_read: false,
            kernel_write: true,
            copy_dst: false,
            copy_src: true,
        }
    }
}

// =============================================================================
// GPU Memory Manager
// =============================================================================

/// GPU memory allocation entry
#[derive(Debug)]
struct GPUAllocation {
    handle: GPUBufferHandle,
    desc: GPUBufferDesc,
    allocated_at: Instant,
}

/// GPU memory manager
pub struct GPUMemoryManager {
    /// Active allocations
    allocations: RwLock<HashMap<GPUBufferHandle, GPUAllocation>>,
    /// Next handle ID
    next_id: AtomicU64,
    /// Total allocated bytes
    total_allocated: AtomicU64,
    /// Device memory limit
    memory_limit: u64,
    /// Backend
    backend: GPUBackend,
}

impl GPUMemoryManager {
    /// Create a new memory manager
    pub fn new(backend: GPUBackend, memory_limit: u64) -> Self {
        Self {
            allocations: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            total_allocated: AtomicU64::new(0),
            memory_limit,
            backend,
        }
    }

    /// Create a new memory manager with validation (checked variant)
    pub fn new_checked(backend: GPUBackend, memory_limit: u64) -> GPUResult<Self> {
        if memory_limit == 0 {
            return Err(GPUError::InvalidBufferSize {
                size: 0,
                reason: "memory limit cannot be zero",
            });
        }
        Ok(Self::new(backend, memory_limit))
    }

    /// Allocate a buffer
    pub fn allocate(&self, desc: GPUBufferDesc) -> NovaResult<GPUBufferHandle> {
        let current = self.total_allocated.load(Ordering::Relaxed);
        if current + desc.size as u64 > self.memory_limit {
            return Err(NovaError::ResourceExhausted(format!(
                "GPU memory limit exceeded: {} + {} > {}",
                current, desc.size, self.memory_limit
            )));
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let handle = GPUBufferHandle::new(id);

        let allocation = GPUAllocation {
            handle,
            desc: desc.clone(),
            allocated_at: Instant::now(),
        };

        self.allocations
            .write()
            .map_err(|e| NovaError::Internal(format!("Lock error: {}", e)))?
            .insert(handle, allocation);

        self.total_allocated.fetch_add(desc.size as u64, Ordering::Relaxed);

        Ok(handle)
    }

    /// Allocate a buffer with validation (checked variant)
    pub fn allocate_checked(&self, desc: GPUBufferDesc) -> GPUResult<GPUBufferHandle> {
        // Validate buffer size
        if desc.size == 0 {
            return Err(GPUError::InvalidBufferSize {
                size: 0,
                reason: "buffer size cannot be zero",
            });
        }

        // Check for overflow
        const MAX_BUFFER_SIZE: usize = 1 << 40; // 1 TB max
        if desc.size > MAX_BUFFER_SIZE {
            return Err(GPUError::InvalidBufferSize {
                size: desc.size,
                reason: "buffer size exceeds maximum (1 TB)",
            });
        }

        let current = self.total_allocated.load(Ordering::Relaxed);
        if current + desc.size as u64 > self.memory_limit {
            return Err(GPUError::MemoryLimitExceeded {
                current,
                requested: desc.size,
                limit: self.memory_limit,
            });
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let handle = GPUBufferHandle::new(id);

        let allocation = GPUAllocation {
            handle,
            desc: desc.clone(),
            allocated_at: Instant::now(),
        };

        self.allocations
            .write()
            .map_err(|_| GPUError::LockContention { operation: "allocate" })?
            .insert(handle, allocation);

        self.total_allocated.fetch_add(desc.size as u64, Ordering::Relaxed);

        Ok(handle)
    }

    /// Free a buffer
    pub fn free(&self, handle: GPUBufferHandle) -> NovaResult<()> {
        let allocation = self
            .allocations
            .write()
            .map_err(|e| NovaError::Internal(format!("Lock error: {}", e)))?
            .remove(&handle);

        if let Some(alloc) = allocation {
            self.total_allocated.fetch_sub(alloc.desc.size as u64, Ordering::Relaxed);
            Ok(())
        } else {
            Err(NovaError::InvalidParameter(format!(
                "Buffer handle {} not found",
                handle.id()
            )))
        }
    }

    /// Free a buffer with validation (checked variant)
    pub fn free_checked(&self, handle: GPUBufferHandle) -> GPUResult<()> {
        let allocation = self
            .allocations
            .write()
            .map_err(|_| GPUError::LockContention { operation: "free" })?
            .remove(&handle);

        if let Some(alloc) = allocation {
            self.total_allocated.fetch_sub(alloc.desc.size as u64, Ordering::Relaxed);
            Ok(())
        } else {
            Err(GPUError::BufferNotFound { handle_id: handle.id() })
        }
    }

    /// Get total allocated memory
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated.load(Ordering::Relaxed)
    }

    /// Get available memory
    pub fn available(&self) -> u64 {
        self.memory_limit.saturating_sub(self.total_allocated())
    }

    /// Check if allocation would succeed without actually allocating
    pub fn can_allocate(&self, size: usize) -> bool {
        let current = self.total_allocated.load(Ordering::Relaxed);
        current + size as u64 <= self.memory_limit
    }

    /// Get memory utilization as fraction (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        if self.memory_limit == 0 {
            return 0.0;
        }
        self.total_allocated() as f64 / self.memory_limit as f64
    }

    /// Get number of active allocations
    pub fn allocation_count(&self) -> usize {
        self.allocations
            .read()
            .map(|a| a.len())
            .unwrap_or(0)
    }

    /// Get backend
    pub fn backend(&self) -> GPUBackend {
        self.backend
    }

    /// Get memory limit
    pub fn memory_limit(&self) -> u64 {
        self.memory_limit
    }
}

// =============================================================================
// Kernel Traits
// =============================================================================

/// Trait for MSM kernel implementations
pub trait MSMKernel: Send + Sync {
    /// Get kernel name
    fn name(&self) -> &str;

    /// Get supported backend
    fn backend(&self) -> GPUBackend;

    /// Execute MSM on GPU
    /// 
    /// # Arguments
    /// * `points` - Base points as bytes (each point: x, y coordinates)
    /// * `scalars` - Scalar multipliers as bytes
    /// * `params` - Kernel parameters
    /// * `result` - Output buffer for result point
    fn execute(
        &self,
        points: &[u8],
        scalars: &[u8],
        params: &MSMKernelParams,
        result: &mut [u8],
    ) -> NovaResult<KernelMetrics>;

    /// Precompute tables for fixed-base MSM
    fn precompute(&self, points: &[u8], params: &MSMKernelParams) -> NovaResult<Vec<u8>>;

    /// Execute with precomputed tables
    fn execute_with_precomputation(
        &self,
        precomputed: &[u8],
        scalars: &[u8],
        params: &MSMKernelParams,
        result: &mut [u8],
    ) -> NovaResult<KernelMetrics>;
}

/// Trait for NTT kernel implementations
pub trait NTTKernel: Send + Sync {
    /// Get kernel name
    fn name(&self) -> &str;

    /// Get supported backend
    fn backend(&self) -> GPUBackend;

    /// Execute forward NTT
    fn forward(
        &self,
        data: &mut [u64],
        params: &NTTKernelParams,
    ) -> NovaResult<KernelMetrics>;

    /// Execute inverse NTT
    fn inverse(
        &self,
        data: &mut [u64],
        params: &NTTKernelParams,
    ) -> NovaResult<KernelMetrics>;

    /// Execute batched NTT
    fn forward_batch(
        &self,
        data: &mut [u64],
        params: &NTTKernelParams,
    ) -> NovaResult<KernelMetrics>;
}

/// Trait for field operations kernel
pub trait FieldKernel: Send + Sync {
    /// Get kernel name
    fn name(&self) -> &str;

    /// Get supported backend
    fn backend(&self) -> GPUBackend;

    /// Element-wise addition: c = a + b mod p
    fn add(
        &self,
        a: &[u64],
        b: &[u64],
        c: &mut [u64],
        params: &FieldKernelParams,
    ) -> NovaResult<KernelMetrics>;

    /// Element-wise multiplication: c = a * b mod p
    fn mul(
        &self,
        a: &[u64],
        b: &[u64],
        c: &mut [u64],
        params: &FieldKernelParams,
    ) -> NovaResult<KernelMetrics>;

    /// Element-wise subtraction: c = a - b mod p
    fn sub(
        &self,
        a: &[u64],
        b: &[u64],
        c: &mut [u64],
        params: &FieldKernelParams,
    ) -> NovaResult<KernelMetrics>;

    /// Batch modular inversion using Montgomery's trick
    fn batch_inverse(
        &self,
        values: &mut [u64],
        params: &FieldKernelParams,
    ) -> NovaResult<KernelMetrics>;
}

// =============================================================================
// Kernel Metrics
// =============================================================================

/// Metrics from kernel execution
#[derive(Debug, Clone, Default)]
pub struct KernelMetrics {
    /// Kernel name
    pub kernel_name: String,
    /// Execution time
    pub execution_time: Duration,
    /// Host-to-device transfer time
    pub h2d_time: Duration,
    /// Device-to-host transfer time
    pub d2h_time: Duration,
    /// Number of elements processed
    pub elements_processed: u64,
    /// GPU memory used
    pub memory_used: u64,
    /// Operations per second
    pub ops_per_second: f64,
}

impl KernelMetrics {
    /// Calculate throughput in elements per second
    pub fn throughput(&self) -> f64 {
        let total_secs = self.execution_time.as_secs_f64();
        if total_secs > 0.0 {
            self.elements_processed as f64 / total_secs
        } else {
            0.0
        }
    }

    /// Total time including transfers
    pub fn total_time(&self) -> Duration {
        self.h2d_time + self.execution_time + self.d2h_time
    }
}

// =============================================================================
// CPU Fallback Implementations
// =============================================================================

/// CPU fallback for MSM
pub struct CPUMSMKernel;

impl MSMKernel for CPUMSMKernel {
    fn name(&self) -> &str {
        "CPU-MSM-Fallback"
    }

    fn backend(&self) -> GPUBackend {
        GPUBackend::Cpu
    }

    fn execute(
        &self,
        _points: &[u8],
        scalars: &[u8],
        params: &MSMKernelParams,
        result: &mut [u8],
    ) -> NovaResult<KernelMetrics> {
        let start = Instant::now();

        // Simplified CPU implementation (placeholder)
        // In production, this would use actual elliptic curve operations
        if result.len() >= 64 {
            result[..64].fill(0);
        }

        Ok(KernelMetrics {
            kernel_name: self.name().to_string(),
            execution_time: start.elapsed(),
            elements_processed: (scalars.len() / 32) as u64,
            ops_per_second: params.num_points as f64 / start.elapsed().as_secs_f64().max(0.001),
            ..Default::default()
        })
    }

    fn precompute(&self, _points: &[u8], _params: &MSMKernelParams) -> NovaResult<Vec<u8>> {
        // No precomputation for CPU fallback
        Ok(Vec::new())
    }

    fn execute_with_precomputation(
        &self,
        _precomputed: &[u8],
        scalars: &[u8],
        params: &MSMKernelParams,
        result: &mut [u8],
    ) -> NovaResult<KernelMetrics> {
        self.execute(&[], scalars, params, result)
    }
}

/// CPU fallback for NTT
pub struct CPUNTTKernel;

impl NTTKernel for CPUNTTKernel {
    fn name(&self) -> &str {
        "CPU-NTT-Fallback"
    }

    fn backend(&self) -> GPUBackend {
        GPUBackend::Cpu
    }

    fn forward(
        &self,
        data: &mut [u64],
        params: &NTTKernelParams,
    ) -> NovaResult<KernelMetrics> {
        let start = Instant::now();

        // Cooley-Tukey NTT (in-place)
        let n = params.size;
        let modulus = params.modulus;

        // Bit-reverse permutation
        bit_reverse_permutation(data);

        // NTT stages
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let w = mod_pow(params.root_of_unity, ((n / len) as u64), modulus);

            for i in (0..n).step_by(len) {
                let mut omega: u64 = 1;
                for j in 0..half {
                    let u = data[i + j];
                    let v = mulmod(data[i + j + half], omega, modulus);
                    data[i + j] = addmod(u, v, modulus);
                    data[i + j + half] = submod(u, v, modulus);
                    omega = mulmod(omega, w, modulus);
                }
            }
            len *= 2;
        }

        Ok(KernelMetrics {
            kernel_name: self.name().to_string(),
            execution_time: start.elapsed(),
            elements_processed: n as u64,
            ops_per_second: n as f64 * (params.log_size as f64) / start.elapsed().as_secs_f64().max(0.001),
            ..Default::default()
        })
    }

    fn inverse(
        &self,
        data: &mut [u64],
        params: &NTTKernelParams,
    ) -> NovaResult<KernelMetrics> {
        let start = Instant::now();

        // Gentleman-Sande INTT (in-place)
        let n = params.size;
        let modulus = params.modulus;

        // INTT stages
        let mut len = n;
        while len >= 2 {
            let half = len / 2;
            let w = mod_pow(params.inv_root_of_unity, ((n / len) as u64), modulus);

            for i in (0..n).step_by(len) {
                let mut omega: u64 = 1;
                for j in 0..half {
                    let u = data[i + j];
                    let v = data[i + j + half];
                    data[i + j] = addmod(u, v, modulus);
                    data[i + j + half] = mulmod(submod(u, v, modulus), omega, modulus);
                    omega = mulmod(omega, w, modulus);
                }
            }
            len /= 2;
        }

        // Bit-reverse permutation
        bit_reverse_permutation(data);

        // Scale by 1/n
        let n_inv = mod_inverse(n as u64, modulus);
        for x in data.iter_mut() {
            *x = mulmod(*x, n_inv, modulus);
        }

        Ok(KernelMetrics {
            kernel_name: self.name().to_string(),
            execution_time: start.elapsed(),
            elements_processed: n as u64,
            ops_per_second: n as f64 * (params.log_size as f64) / start.elapsed().as_secs_f64().max(0.001),
            ..Default::default()
        })
    }

    fn forward_batch(
        &self,
        data: &mut [u64],
        params: &NTTKernelParams,
    ) -> NovaResult<KernelMetrics> {
        let start = Instant::now();
        let batch_size = params.size;
        let mut total_processed = 0u64;

        for chunk in data.chunks_mut(batch_size) {
            if chunk.len() == batch_size {
                let sub_params = params.clone();
                self.forward(chunk, &sub_params)?;
                total_processed += batch_size as u64;
            }
        }

        Ok(KernelMetrics {
            kernel_name: self.name().to_string(),
            execution_time: start.elapsed(),
            elements_processed: total_processed,
            ops_per_second: total_processed as f64 * (params.log_size as f64) / start.elapsed().as_secs_f64().max(0.001),
            ..Default::default()
        })
    }
}

/// CPU fallback for field operations
pub struct CPUFieldKernel;

impl FieldKernel for CPUFieldKernel {
    fn name(&self) -> &str {
        "CPU-Field-Fallback"
    }

    fn backend(&self) -> GPUBackend {
        GPUBackend::Cpu
    }

    fn add(
        &self,
        a: &[u64],
        b: &[u64],
        c: &mut [u64],
        params: &FieldKernelParams,
    ) -> NovaResult<KernelMetrics> {
        let start = Instant::now();
        let modulus = params.modulus;

        for i in 0..params.num_elements.min(a.len()).min(b.len()).min(c.len()) {
            c[i] = addmod(a[i], b[i], modulus);
        }

        Ok(KernelMetrics {
            kernel_name: self.name().to_string(),
            execution_time: start.elapsed(),
            elements_processed: params.num_elements as u64,
            ops_per_second: params.num_elements as f64 / start.elapsed().as_secs_f64().max(0.001),
            ..Default::default()
        })
    }

    fn mul(
        &self,
        a: &[u64],
        b: &[u64],
        c: &mut [u64],
        params: &FieldKernelParams,
    ) -> NovaResult<KernelMetrics> {
        let start = Instant::now();
        let modulus = params.modulus;

        for i in 0..params.num_elements.min(a.len()).min(b.len()).min(c.len()) {
            c[i] = mulmod(a[i], b[i], modulus);
        }

        Ok(KernelMetrics {
            kernel_name: self.name().to_string(),
            execution_time: start.elapsed(),
            elements_processed: params.num_elements as u64,
            ops_per_second: params.num_elements as f64 / start.elapsed().as_secs_f64().max(0.001),
            ..Default::default()
        })
    }

    fn sub(
        &self,
        a: &[u64],
        b: &[u64],
        c: &mut [u64],
        params: &FieldKernelParams,
    ) -> NovaResult<KernelMetrics> {
        let start = Instant::now();
        let modulus = params.modulus;

        for i in 0..params.num_elements.min(a.len()).min(b.len()).min(c.len()) {
            c[i] = submod(a[i], b[i], modulus);
        }

        Ok(KernelMetrics {
            kernel_name: self.name().to_string(),
            execution_time: start.elapsed(),
            elements_processed: params.num_elements as u64,
            ops_per_second: params.num_elements as f64 / start.elapsed().as_secs_f64().max(0.001),
            ..Default::default()
        })
    }

    fn batch_inverse(
        &self,
        values: &mut [u64],
        params: &FieldKernelParams,
    ) -> NovaResult<KernelMetrics> {
        let start = Instant::now();
        let modulus = params.modulus;
        let n = params.num_elements.min(values.len());

        if n == 0 {
            return Ok(KernelMetrics::default());
        }

        // Montgomery's trick for batch inversion
        let mut products = vec![1u64; n];
        products[0] = values[0];
        for i in 1..n {
            products[i] = mulmod(products[i - 1], values[i], modulus);
        }

        // Invert the product of all elements
        let mut inv = mod_inverse(products[n - 1], modulus);

        // Compute individual inverses
        for i in (1..n).rev() {
            let val = values[i];
            values[i] = mulmod(inv, products[i - 1], modulus);
            inv = mulmod(inv, val, modulus);
        }
        values[0] = inv;

        Ok(KernelMetrics {
            kernel_name: self.name().to_string(),
            execution_time: start.elapsed(),
            elements_processed: n as u64,
            ops_per_second: n as f64 / start.elapsed().as_secs_f64().max(0.001),
            ..Default::default()
        })
    }
}

// =============================================================================
// Kernel Manager
// =============================================================================

/// Manager for GPU kernels
pub struct GPUKernelManager {
    /// Memory manager
    memory: Arc<GPUMemoryManager>,
    /// MSM kernel
    msm_kernel: Box<dyn MSMKernel>,
    /// NTT kernel
    ntt_kernel: Box<dyn NTTKernel>,
    /// Field kernel
    field_kernel: Box<dyn FieldKernel>,
    /// Device info
    device_info: GPUDeviceInfo,
    /// Accumulated metrics
    total_metrics: RwLock<AccumulatedMetrics>,
}

/// Accumulated metrics across all kernel calls
#[derive(Debug, Clone, Default)]
pub struct AccumulatedMetrics {
    pub total_msm_calls: u64,
    pub total_ntt_calls: u64,
    pub total_field_calls: u64,
    pub total_msm_time: Duration,
    pub total_ntt_time: Duration,
    pub total_field_time: Duration,
    pub total_elements_processed: u64,
}

impl GPUKernelManager {
    /// Create a new kernel manager with CPU fallback
    pub fn new_cpu_fallback() -> Self {
        let device_info = GPUDeviceInfo::default();
        let memory = Arc::new(GPUMemoryManager::new(
            GPUBackend::Cpu,
            16 * 1024 * 1024 * 1024, // 16 GB limit
        ));

        Self {
            memory,
            msm_kernel: Box::new(CPUMSMKernel),
            ntt_kernel: Box::new(CPUNTTKernel),
            field_kernel: Box::new(CPUFieldKernel),
            device_info,
            total_metrics: RwLock::new(AccumulatedMetrics::default()),
        }
    }

    /// Get device info
    pub fn device_info(&self) -> &GPUDeviceInfo {
        &self.device_info
    }

    /// Get memory manager
    pub fn memory(&self) -> &Arc<GPUMemoryManager> {
        &self.memory
    }

    /// Execute MSM
    pub fn msm(
        &self,
        points: &[u8],
        scalars: &[u8],
        result: &mut [u8],
    ) -> NovaResult<KernelMetrics> {
        let params = MSMKernelParams::for_size(scalars.len() / 32);
        let metrics = self.msm_kernel.execute(points, scalars, &params, result)?;

        if let Ok(mut total) = self.total_metrics.write() {
            total.total_msm_calls += 1;
            total.total_msm_time += metrics.execution_time;
            total.total_elements_processed += metrics.elements_processed;
        }

        Ok(metrics)
    }

    /// Execute forward NTT
    pub fn ntt_forward(&self, data: &mut [u64], modulus: u64, root: u64) -> NovaResult<KernelMetrics> {
        let params = NTTKernelParams::new(data.len(), modulus, root);
        let metrics = self.ntt_kernel.forward(data, &params)?;

        if let Ok(mut total) = self.total_metrics.write() {
            total.total_ntt_calls += 1;
            total.total_ntt_time += metrics.execution_time;
            total.total_elements_processed += metrics.elements_processed;
        }

        Ok(metrics)
    }

    /// Execute inverse NTT
    pub fn ntt_inverse(&self, data: &mut [u64], modulus: u64, root: u64) -> NovaResult<KernelMetrics> {
        let params = NTTKernelParams::new(data.len(), modulus, root);
        let metrics = self.ntt_kernel.inverse(data, &params)?;

        if let Ok(mut total) = self.total_metrics.write() {
            total.total_ntt_calls += 1;
            total.total_ntt_time += metrics.execution_time;
            total.total_elements_processed += metrics.elements_processed;
        }

        Ok(metrics)
    }

    /// Execute field addition
    pub fn field_add(&self, a: &[u64], b: &[u64], c: &mut [u64], modulus: u64) -> NovaResult<KernelMetrics> {
        let params = FieldKernelParams {
            num_elements: a.len().min(b.len()).min(c.len()),
            modulus,
            ..Default::default()
        };
        let metrics = self.field_kernel.add(a, b, c, &params)?;

        if let Ok(mut total) = self.total_metrics.write() {
            total.total_field_calls += 1;
            total.total_field_time += metrics.execution_time;
            total.total_elements_processed += metrics.elements_processed;
        }

        Ok(metrics)
    }

    /// Execute field multiplication
    pub fn field_mul(&self, a: &[u64], b: &[u64], c: &mut [u64], modulus: u64) -> NovaResult<KernelMetrics> {
        let params = FieldKernelParams {
            num_elements: a.len().min(b.len()).min(c.len()),
            modulus,
            ..Default::default()
        };
        let metrics = self.field_kernel.mul(a, b, c, &params)?;

        if let Ok(mut total) = self.total_metrics.write() {
            total.total_field_calls += 1;
            total.total_field_time += metrics.execution_time;
            total.total_elements_processed += metrics.elements_processed;
        }

        Ok(metrics)
    }

    /// Get accumulated metrics
    pub fn accumulated_metrics(&self) -> AccumulatedMetrics {
        self.total_metrics
            .read()
            .map(|m| m.clone())
            .unwrap_or_default()
    }

    /// Reset accumulated metrics
    pub fn reset_metrics(&self) {
        if let Ok(mut m) = self.total_metrics.write() {
            *m = AccumulatedMetrics::default();
        }
    }

    // =========================================================================
    // Checked variants with full validation (Production Hardening)
    // =========================================================================

    /// Execute MSM with full validation (checked variant)
    pub fn msm_checked(
        &self,
        points: &[u8],
        scalars: &[u8],
        result: &mut [u8],
    ) -> GPUResult<KernelMetrics> {
        // Validate scalar buffer
        if scalars.is_empty() {
            return Err(GPUError::InvalidKernelParams {
                kernel: "MSM",
                reason: "scalars buffer is empty".to_string(),
            });
        }

        // Scalars should be 32 bytes each
        if scalars.len() % 32 != 0 {
            return Err(GPUError::InvalidKernelParams {
                kernel: "MSM",
                reason: format!("scalars length {} not divisible by 32", scalars.len()),
            });
        }

        let num_scalars = scalars.len() / 32;

        // Points should be 64 bytes each (x, y coordinates)
        let expected_points_len = num_scalars * 64;
        if !points.is_empty() && points.len() != expected_points_len {
            return Err(GPUError::DataSizeMismatch {
                expected: expected_points_len,
                actual: points.len(),
                context: "MSM points buffer",
            });
        }

        // Result should be at least 64 bytes for one point
        if result.len() < 64 {
            return Err(GPUError::DataSizeMismatch {
                expected: 64,
                actual: result.len(),
                context: "MSM result buffer",
            });
        }

        // Check memory availability
        let required_memory = (points.len() + scalars.len() + result.len()) as u64;
        if self.memory.available() < required_memory {
            return Err(GPUError::AllocationFailed {
                requested: required_memory as usize,
                available: self.memory.available(),
                backend: self.device_info.backend,
            });
        }

        let params = MSMKernelParams::for_size(num_scalars);
        let metrics = self.msm_kernel
            .execute(points, scalars, &params, result)
            .map_err(|e| GPUError::KernelExecutionFailed {
                kernel: "MSM".to_string(),
                reason: e.to_string(),
            })?;

        if let Ok(mut total) = self.total_metrics.write() {
            total.total_msm_calls += 1;
            total.total_msm_time += metrics.execution_time;
            total.total_elements_processed += metrics.elements_processed;
        }

        Ok(metrics)
    }

    /// Execute forward NTT with full validation (checked variant)
    pub fn ntt_forward_checked(
        &self, 
        data: &mut [u64], 
        modulus: u64, 
        root: u64
    ) -> GPUResult<KernelMetrics> {
        // Validate input size is power of 2
        if !data.len().is_power_of_two() {
            return Err(GPUError::InvalidNTTSize { size: data.len() });
        }

        if data.is_empty() {
            return Err(GPUError::InvalidKernelParams {
                kernel: "NTT",
                reason: "data buffer is empty".to_string(),
            });
        }

        // Validate modulus
        if modulus == 0 {
            return Err(GPUError::ZeroModulus);
        }

        // Validate root of unity
        if root == 0 || root >= modulus {
            return Err(GPUError::InvalidRootOfUnity { root, modulus });
        }

        // Check all values are less than modulus
        for (i, &val) in data.iter().enumerate() {
            if val >= modulus {
                return Err(GPUError::InvalidKernelParams {
                    kernel: "NTT",
                    reason: format!("data[{}] = {} >= modulus {}", i, val, modulus),
                });
            }
        }

        let params = NTTKernelParams::new_checked(data.len(), modulus, root)?;
        let metrics = self.ntt_kernel
            .forward(data, &params)
            .map_err(|e| GPUError::KernelExecutionFailed {
                kernel: "NTT-Forward".to_string(),
                reason: e.to_string(),
            })?;

        if let Ok(mut total) = self.total_metrics.write() {
            total.total_ntt_calls += 1;
            total.total_ntt_time += metrics.execution_time;
            total.total_elements_processed += metrics.elements_processed;
        }

        Ok(metrics)
    }

    /// Execute inverse NTT with full validation (checked variant)
    pub fn ntt_inverse_checked(
        &self, 
        data: &mut [u64], 
        modulus: u64, 
        root: u64
    ) -> GPUResult<KernelMetrics> {
        // Validate input size is power of 2
        if !data.len().is_power_of_two() {
            return Err(GPUError::InvalidNTTSize { size: data.len() });
        }

        if data.is_empty() {
            return Err(GPUError::InvalidKernelParams {
                kernel: "INTT",
                reason: "data buffer is empty".to_string(),
            });
        }

        // Validate modulus
        if modulus == 0 {
            return Err(GPUError::ZeroModulus);
        }

        // Validate root of unity
        if root == 0 || root >= modulus {
            return Err(GPUError::InvalidRootOfUnity { root, modulus });
        }

        let params = NTTKernelParams::new_checked(data.len(), modulus, root)?;
        let metrics = self.ntt_kernel
            .inverse(data, &params)
            .map_err(|e| GPUError::KernelExecutionFailed {
                kernel: "NTT-Inverse".to_string(),
                reason: e.to_string(),
            })?;

        if let Ok(mut total) = self.total_metrics.write() {
            total.total_ntt_calls += 1;
            total.total_ntt_time += metrics.execution_time;
            total.total_elements_processed += metrics.elements_processed;
        }

        Ok(metrics)
    }

    /// Execute field addition with full validation (checked variant)
    pub fn field_add_checked(
        &self, 
        a: &[u64], 
        b: &[u64], 
        c: &mut [u64], 
        modulus: u64
    ) -> GPUResult<KernelMetrics> {
        // Validate modulus
        if modulus == 0 {
            return Err(GPUError::ZeroModulus);
        }

        // Validate buffer sizes match
        if a.len() != b.len() || b.len() != c.len() {
            return Err(GPUError::DataSizeMismatch {
                expected: a.len(),
                actual: b.len().min(c.len()),
                context: "field_add buffers must have equal length",
            });
        }

        if a.is_empty() {
            return Err(GPUError::InvalidKernelParams {
                kernel: "FieldAdd",
                reason: "buffers are empty".to_string(),
            });
        }

        let params = FieldKernelParams {
            num_elements: a.len(),
            modulus,
            ..Default::default()
        };
        let metrics = self.field_kernel
            .add(a, b, c, &params)
            .map_err(|e| GPUError::KernelExecutionFailed {
                kernel: "FieldAdd".to_string(),
                reason: e.to_string(),
            })?;

        if let Ok(mut total) = self.total_metrics.write() {
            total.total_field_calls += 1;
            total.total_field_time += metrics.execution_time;
            total.total_elements_processed += metrics.elements_processed;
        }

        Ok(metrics)
    }

    /// Execute field multiplication with full validation (checked variant)
    pub fn field_mul_checked(
        &self, 
        a: &[u64], 
        b: &[u64], 
        c: &mut [u64], 
        modulus: u64
    ) -> GPUResult<KernelMetrics> {
        // Validate modulus
        if modulus == 0 {
            return Err(GPUError::ZeroModulus);
        }

        // Validate buffer sizes match
        if a.len() != b.len() || b.len() != c.len() {
            return Err(GPUError::DataSizeMismatch {
                expected: a.len(),
                actual: b.len().min(c.len()),
                context: "field_mul buffers must have equal length",
            });
        }

        if a.is_empty() {
            return Err(GPUError::InvalidKernelParams {
                kernel: "FieldMul",
                reason: "buffers are empty".to_string(),
            });
        }

        let params = FieldKernelParams {
            num_elements: a.len(),
            modulus,
            ..Default::default()
        };
        let metrics = self.field_kernel
            .mul(a, b, c, &params)
            .map_err(|e| GPUError::KernelExecutionFailed {
                kernel: "FieldMul".to_string(),
                reason: e.to_string(),
            })?;

        if let Ok(mut total) = self.total_metrics.write() {
            total.total_field_calls += 1;
            total.total_field_time += metrics.execution_time;
            total.total_elements_processed += metrics.elements_processed;
        }

        Ok(metrics)
    }

    /// Check if device is available and initialized
    pub fn is_device_available(&self) -> bool {
        true // CPU fallback is always available
    }

    /// Get backend type
    pub fn backend(&self) -> GPUBackend {
        self.device_info.backend
    }

    /// Get total memory used by kernel manager
    pub fn memory_used(&self) -> u64 {
        self.memory.total_allocated()
    }

    /// Get available memory
    pub fn memory_available(&self) -> u64 {
        self.memory.available()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Modular addition
#[inline]
fn addmod(a: u64, b: u64, m: u64) -> u64 {
    let sum = a.wrapping_add(b);
    if sum >= m || sum < a {
        sum.wrapping_sub(m)
    } else {
        sum
    }
}

/// Modular subtraction
#[inline]
fn submod(a: u64, b: u64, m: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        m - (b - a)
    }
}

/// Modular multiplication (uses 128-bit intermediate)
#[inline]
fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular exponentiation
fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mulmod(result, base, modulus);
        }
        exp >>= 1;
        base = mulmod(base, base, modulus);
    }
    result
}

/// Modular inverse using extended Euclidean algorithm
fn mod_inverse(a: u64, m: u64) -> u64 {
    let mut t = 0i128;
    let mut new_t = 1i128;
    let mut r = m as i128;
    let mut new_r = a as i128;

    while new_r != 0 {
        let q = r / new_r;
        (t, new_t) = (new_t, t - q * new_t);
        (r, new_r) = (new_r, r - q * new_r);
    }

    if t < 0 {
        t += m as i128;
    }

    t as u64
}

/// Bit-reverse permutation for NTT
fn bit_reverse_permutation(data: &mut [u64]) {
    let n = data.len();
    let log_n = n.trailing_zeros() as usize;

    for i in 0..n {
        let rev = bit_reverse(i, log_n);
        if i < rev {
            data.swap(i, rev);
        }
    }
}

/// Reverse bits
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MODULUS: u64 = 0xFFFFFFFF00000001; // Goldilocks prime

    #[test]
    fn test_gpu_backend() {
        assert!(GPUBackend::Cuda.is_gpu());
        assert!(GPUBackend::Metal.is_gpu());
        assert!(GPUBackend::WebGpu.is_gpu());
        assert!(!GPUBackend::Cpu.is_gpu());
    }

    #[test]
    fn test_msm_kernel_params() {
        let params = MSMKernelParams::for_size(1000);
        assert_eq!(params.num_points, 1000);
        assert_eq!(params.window_size, 8);

        let params2 = MSMKernelParams::for_size(100000);
        assert_eq!(params2.window_size, 16);
    }

    #[test]
    fn test_ntt_kernel_params() {
        let root = 7; // Dummy root for testing
        let params = NTTKernelParams::new(1024, TEST_MODULUS, root);
        assert_eq!(params.size, 1024);
        assert_eq!(params.log_size, 10);
    }

    #[test]
    fn test_memory_manager() {
        let mgr = GPUMemoryManager::new(GPUBackend::Cpu, 1024 * 1024);

        let desc = GPUBufferDesc {
            size: 1024,
            buffer_type: BufferType::Device,
            usage: BufferUsage::default(),
            name: "test".to_string(),
        };

        let handle = mgr.allocate(desc).unwrap();
        assert_eq!(mgr.total_allocated(), 1024);

        mgr.free(handle).unwrap();
        assert_eq!(mgr.total_allocated(), 0);
    }

    #[test]
    fn test_memory_limit() {
        let mgr = GPUMemoryManager::new(GPUBackend::Cpu, 100);

        let desc = GPUBufferDesc {
            size: 200,
            buffer_type: BufferType::Device,
            usage: BufferUsage::default(),
            name: "test".to_string(),
        };

        let result = mgr.allocate(desc);
        assert!(result.is_err());
    }

    #[test]
    fn test_modular_arithmetic() {
        let m = 97u64;
        assert_eq!(addmod(50, 60, m), 13); // 110 mod 97 = 13
        assert_eq!(submod(10, 20, m), 87); // -10 mod 97 = 87
        assert_eq!(mulmod(10, 10, m), 3); // 100 mod 97 = 3
    }

    #[test]
    fn test_mod_inverse() {
        let m = 97u64;
        let a = 42u64;
        let inv = mod_inverse(a, m);
        assert_eq!(mulmod(a, inv, m), 1);
    }

    #[test]
    fn test_mod_pow() {
        let m = 97u64;
        assert_eq!(mod_pow(2, 10, m), 54); // 1024 mod 97 = 54
        assert_eq!(mod_pow(3, 0, m), 1);
        assert_eq!(mod_pow(5, 1, m), 5);
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0b000, 3), 0b000);
        assert_eq!(bit_reverse(0b001, 3), 0b100);
        assert_eq!(bit_reverse(0b010, 3), 0b010);
        assert_eq!(bit_reverse(0b011, 3), 0b110);
    }

    #[test]
    fn test_cpu_ntt_forward_inverse() {
        let kernel = CPUNTTKernel;
        let n = 8;
        let modulus = 97u64;
        let root = 3u64; // primitive 8th root of unity mod 97

        let mut data: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let original = data.clone();

        let params = NTTKernelParams::new(n, modulus, root);

        // Forward NTT
        kernel.forward(&mut data, &params).unwrap();
        
        // Inverse NTT should restore original
        kernel.inverse(&mut data, &params).unwrap();

        assert_eq!(data, original);
    }

    #[test]
    fn test_cpu_field_add() {
        let kernel = CPUFieldKernel;
        let modulus = 97u64;

        let a = vec![10, 20, 30, 40, 50];
        let b = vec![5, 90, 10, 60, 47];
        let mut c = vec![0u64; 5];

        let params = FieldKernelParams {
            num_elements: 5,
            modulus,
            ..Default::default()
        };

        kernel.add(&a, &b, &mut c, &params).unwrap();

        assert_eq!(c[0], 15);      // 10 + 5 = 15
        assert_eq!(c[1], 13);      // 20 + 90 = 110 mod 97 = 13
        assert_eq!(c[2], 40);      // 30 + 10 = 40
        assert_eq!(c[3], 3);       // 40 + 60 = 100 mod 97 = 3
        assert_eq!(c[4], 0);       // 50 + 47 = 97 mod 97 = 0
    }

    #[test]
    fn test_cpu_batch_inverse() {
        let kernel = CPUFieldKernel;
        let modulus = 97u64;

        let mut values = vec![2, 3, 5, 7, 11];
        let original = values.clone();

        let params = FieldKernelParams {
            num_elements: 5,
            modulus,
            ..Default::default()
        };

        kernel.batch_inverse(&mut values, &params).unwrap();

        // Verify each inverse
        for (i, &inv) in values.iter().enumerate() {
            assert_eq!(mulmod(original[i], inv, modulus), 1);
        }
    }

    #[test]
    fn test_kernel_manager() {
        let mgr = GPUKernelManager::new_cpu_fallback();

        assert_eq!(mgr.device_info().backend, GPUBackend::Cpu);

        let modulus = 97u64;
        let a = vec![10u64, 20, 30];
        let b = vec![5u64, 10, 15];
        let mut c = vec![0u64; 3];

        let metrics = mgr.field_add(&a, &b, &mut c, modulus).unwrap();
        assert!(metrics.elements_processed > 0);

        let acc = mgr.accumulated_metrics();
        assert_eq!(acc.total_field_calls, 1);
    }

    #[test]
    fn test_kernel_metrics() {
        let metrics = KernelMetrics {
            kernel_name: "test".to_string(),
            execution_time: Duration::from_secs(1),
            elements_processed: 1000,
            ..Default::default()
        };

        assert!((metrics.throughput() - 1000.0).abs() < 0.1);
    }

    // =========================================================================
    // Production Hardening Tests
    // =========================================================================

    #[test]
    fn test_gpu_error_display() {
        let err = GPUError::AllocationFailed {
            requested: 1000,
            available: 500,
            backend: GPUBackend::Cuda,
        };
        assert!(err.to_string().contains("1000"));
        assert!(err.to_string().contains("500"));
        assert!(err.to_string().contains("Cuda"));

        let err2 = GPUError::InvalidNTTSize { size: 100 };
        assert!(err2.to_string().contains("100"));
        assert!(err2.to_string().contains("power of 2"));
    }

    #[test]
    fn test_memory_manager_checked_creation() {
        // Valid creation
        let mgr = GPUMemoryManager::new_checked(GPUBackend::Cpu, 1024 * 1024);
        assert!(mgr.is_ok());

        // Zero limit should fail
        let mgr_zero = GPUMemoryManager::new_checked(GPUBackend::Cpu, 0);
        assert!(matches!(mgr_zero, Err(GPUError::InvalidBufferSize { .. })));
    }

    #[test]
    fn test_memory_manager_allocate_checked() {
        let mgr = GPUMemoryManager::new(GPUBackend::Cpu, 1024);

        // Valid allocation
        let desc = GPUBufferDesc {
            size: 512,
            buffer_type: BufferType::Device,
            usage: BufferUsage::default(),
            name: "test".to_string(),
        };
        let handle = mgr.allocate_checked(desc);
        assert!(handle.is_ok());

        // Zero size should fail
        let desc_zero = GPUBufferDesc {
            size: 0,
            buffer_type: BufferType::Device,
            usage: BufferUsage::default(),
            name: "zero".to_string(),
        };
        let result = mgr.allocate_checked(desc_zero);
        assert!(matches!(result, Err(GPUError::InvalidBufferSize { .. })));

        // Exceeding limit should fail
        let desc_large = GPUBufferDesc {
            size: 2048,
            buffer_type: BufferType::Device,
            usage: BufferUsage::default(),
            name: "large".to_string(),
        };
        let result = mgr.allocate_checked(desc_large);
        assert!(matches!(result, Err(GPUError::MemoryLimitExceeded { .. })));
    }

    #[test]
    fn test_memory_manager_free_checked() {
        let mgr = GPUMemoryManager::new(GPUBackend::Cpu, 1024);

        let desc = GPUBufferDesc {
            size: 256,
            buffer_type: BufferType::Device,
            usage: BufferUsage::default(),
            name: "test".to_string(),
        };
        let handle = mgr.allocate_checked(desc).unwrap();
        assert_eq!(mgr.total_allocated(), 256);

        // Valid free
        assert!(mgr.free_checked(handle).is_ok());
        assert_eq!(mgr.total_allocated(), 0);

        // Double free should fail
        let result = mgr.free_checked(handle);
        assert!(matches!(result, Err(GPUError::BufferNotFound { .. })));
    }

    #[test]
    fn test_memory_manager_utilization() {
        let mgr = GPUMemoryManager::new(GPUBackend::Cpu, 1000);
        assert!((mgr.utilization() - 0.0).abs() < 0.001);

        let desc = GPUBufferDesc {
            size: 500,
            buffer_type: BufferType::Device,
            usage: BufferUsage::default(),
            name: "half".to_string(),
        };
        let _ = mgr.allocate(desc).unwrap();
        assert!((mgr.utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_memory_manager_can_allocate() {
        let mgr = GPUMemoryManager::new(GPUBackend::Cpu, 1024);

        assert!(mgr.can_allocate(512));
        assert!(mgr.can_allocate(1024));
        assert!(!mgr.can_allocate(1025));

        let desc = GPUBufferDesc {
            size: 512,
            buffer_type: BufferType::Device,
            usage: BufferUsage::default(),
            name: "test".to_string(),
        };
        let _ = mgr.allocate(desc).unwrap();

        assert!(mgr.can_allocate(512));
        assert!(!mgr.can_allocate(513));
    }

    #[test]
    fn test_ntt_params_checked() {
        // Valid params
        let params = NTTKernelParams::new_checked(1024, TEST_MODULUS, 7);
        assert!(params.is_ok());

        // Non-power-of-2 should fail
        let params_bad = NTTKernelParams::new_checked(1000, TEST_MODULUS, 7);
        assert!(matches!(params_bad, Err(GPUError::InvalidNTTSize { .. })));

        // Zero size should fail
        let params_zero = NTTKernelParams::new_checked(0, TEST_MODULUS, 7);
        assert!(matches!(params_zero, Err(GPUError::InvalidNTTSize { .. })));

        // Zero modulus should fail
        let params_mod = NTTKernelParams::new_checked(1024, 0, 7);
        assert!(matches!(params_mod, Err(GPUError::ZeroModulus)));

        // Invalid root should fail
        let params_root = NTTKernelParams::new_checked(1024, 97, 0);
        assert!(matches!(params_root, Err(GPUError::InvalidRootOfUnity { .. })));
    }

    #[test]
    fn test_ntt_params_validate() {
        let mut params = NTTKernelParams::new(1024, TEST_MODULUS, 7);
        assert!(params.validate().is_ok());

        // Corrupt and validate
        params.size = 1000; // Not power of 2
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_kernel_manager_ntt_forward_checked() {
        let mgr = GPUKernelManager::new_cpu_fallback();
        let modulus = 97u64;
        let root = 3u64;

        // Valid NTT
        let mut data = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let result = mgr.ntt_forward_checked(&mut data, modulus, root);
        assert!(result.is_ok());

        // Non-power-of-2 should fail
        let mut bad_data = vec![1u64, 2, 3, 4, 5];
        let result = mgr.ntt_forward_checked(&mut bad_data, modulus, root);
        assert!(matches!(result, Err(GPUError::InvalidNTTSize { .. })));

        // Empty data should fail with InvalidNTTSize (since 0 is not power of 2)
        let mut empty: Vec<u64> = vec![];
        let result = mgr.ntt_forward_checked(&mut empty, modulus, root);
        assert!(matches!(result, Err(GPUError::InvalidNTTSize { .. })));

        // Zero modulus should fail
        let mut data2 = vec![1u64, 2, 3, 4];
        let result = mgr.ntt_forward_checked(&mut data2, 0, root);
        assert!(matches!(result, Err(GPUError::ZeroModulus)));
    }

    #[test]
    fn test_kernel_manager_ntt_inverse_checked() {
        let mgr = GPUKernelManager::new_cpu_fallback();
        let modulus = 97u64;
        let root = 3u64;

        // Valid INTT
        let mut data = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let original = data.clone();
        
        // Forward then inverse should restore original
        mgr.ntt_forward_checked(&mut data, modulus, root).unwrap();
        mgr.ntt_inverse_checked(&mut data, modulus, root).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_kernel_manager_field_add_checked() {
        let mgr = GPUKernelManager::new_cpu_fallback();
        let modulus = 97u64;

        // Valid addition
        let a = vec![10u64, 20, 30];
        let b = vec![5u64, 90, 10];
        let mut c = vec![0u64; 3];
        let result = mgr.field_add_checked(&a, &b, &mut c, modulus);
        assert!(result.is_ok());
        assert_eq!(c[0], 15);
        assert_eq!(c[1], 13); // 110 mod 97

        // Mismatched sizes should fail
        let a2 = vec![1u64, 2, 3];
        let b2 = vec![1u64, 2];
        let mut c2 = vec![0u64; 3];
        let result = mgr.field_add_checked(&a2, &b2, &mut c2, modulus);
        assert!(matches!(result, Err(GPUError::DataSizeMismatch { .. })));

        // Zero modulus should fail
        let result = mgr.field_add_checked(&a, &b, &mut c, 0);
        assert!(matches!(result, Err(GPUError::ZeroModulus)));
    }

    #[test]
    fn test_kernel_manager_field_mul_checked() {
        let mgr = GPUKernelManager::new_cpu_fallback();
        let modulus = 97u64;

        // Valid multiplication
        let a = vec![10u64, 10, 10];
        let b = vec![10u64, 10, 10];
        let mut c = vec![0u64; 3];
        let result = mgr.field_mul_checked(&a, &b, &mut c, modulus);
        assert!(result.is_ok());
        assert_eq!(c[0], 3); // 100 mod 97

        // Empty buffers should fail
        let empty: Vec<u64> = vec![];
        let mut empty_out: Vec<u64> = vec![];
        let result = mgr.field_mul_checked(&empty, &empty, &mut empty_out, modulus);
        assert!(matches!(result, Err(GPUError::InvalidKernelParams { .. })));
    }

    #[test]
    fn test_kernel_manager_msm_checked() {
        let mgr = GPUKernelManager::new_cpu_fallback();

        // Empty scalars should fail
        let points: Vec<u8> = vec![];
        let scalars: Vec<u8> = vec![];
        let mut result = vec![0u8; 64];
        let res = mgr.msm_checked(&points, &scalars, &mut result);
        assert!(matches!(res, Err(GPUError::InvalidKernelParams { .. })));

        // Scalars not divisible by 32 should fail
        let bad_scalars = vec![0u8; 33];
        let res = mgr.msm_checked(&points, &bad_scalars, &mut result);
        assert!(matches!(res, Err(GPUError::InvalidKernelParams { .. })));

        // Result too small should fail
        let scalars = vec![0u8; 32];
        let mut small_result = vec![0u8; 32];
        let res = mgr.msm_checked(&points, &scalars, &mut small_result);
        assert!(matches!(res, Err(GPUError::DataSizeMismatch { .. })));
    }

    // =========================================================================
    // Stress Tests
    // =========================================================================

    #[test]
    fn test_concurrent_memory_manager_access() {
        use std::thread;

        let mgr = Arc::new(GPUMemoryManager::new(GPUBackend::Cpu, 10 * 1024 * 1024));
        let mut handles = vec![];

        // Spawn 8 threads doing allocations
        for t in 0..8 {
            let mgr_clone = Arc::clone(&mgr);
            handles.push(thread::spawn(move || {
                let mut my_handles = vec![];
                for i in 0..100 {
                    let desc = GPUBufferDesc {
                        size: 1024,
                        buffer_type: BufferType::Device,
                        usage: BufferUsage::default(),
                        name: format!("thread_{}_alloc_{}", t, i),
                    };
                    if let Ok(handle) = mgr_clone.allocate_checked(desc) {
                        my_handles.push(handle);
                    }
                }
                // Free all
                for handle in my_handles {
                    let _ = mgr_clone.free_checked(handle);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All memory should be freed
        assert_eq!(mgr.total_allocated(), 0);
        assert_eq!(mgr.allocation_count(), 0);
    }

    #[test]
    fn test_memory_pressure_stress() {
        let mgr = GPUMemoryManager::new(GPUBackend::Cpu, 1024 * 1024); // 1 MB limit

        // Allocate until we hit the limit
        let mut allocated_handles = vec![];
        let mut total_allocated = 0usize;

        for i in 0..10000 {
            let size = ((i % 10) + 1) * 100; // Varying sizes
            let desc = GPUBufferDesc {
                size,
                buffer_type: BufferType::Device,
                usage: BufferUsage::default(),
                name: format!("stress_{}", i),
            };

            match mgr.allocate_checked(desc) {
                Ok(handle) => {
                    allocated_handles.push((handle, size));
                    total_allocated += size;
                }
                Err(GPUError::MemoryLimitExceeded { .. }) => {
                    // Expected once we hit the limit
                    break;
                }
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        assert!(total_allocated > 0);
        assert!(mgr.utilization() > 0.9); // Should be nearly full

        // Free half
        let half = allocated_handles.len() / 2;
        for (handle, _) in allocated_handles.drain(..half) {
            mgr.free_checked(handle).unwrap();
        }

        // Should have more space now
        assert!(mgr.utilization() < 0.6);
    }

    #[test]
    fn test_ntt_stress_various_sizes() {
        let kernel = CPUNTTKernel;
        let modulus = 97u64;
        let root = 3u64;

        // Test various power-of-2 sizes
        for log_size in 1..=10 {
            let size = 1 << log_size;
            let mut data: Vec<u64> = (0..size as u64).map(|x| x % modulus).collect();
            let original = data.clone();

            let params = NTTKernelParams::new(size, modulus, root);

            // Forward
            kernel.forward(&mut data, &params).unwrap();
            
            // Inverse
            kernel.inverse(&mut data, &params).unwrap();

            // Should match original
            assert_eq!(data, original, "NTT roundtrip failed for size {}", size);
        }
    }

    #[test]
    fn test_field_operations_stress() {
        let kernel = CPUFieldKernel;
        let modulus = 0xFFFFFFFF00000001u64; // Goldilocks

        let n = 10000;
        let a: Vec<u64> = (0..n).map(|i| (i as u64 * 12345) % modulus).collect();
        let b: Vec<u64> = (0..n).map(|i| (i as u64 * 67890) % modulus).collect();
        let mut c = vec![0u64; n];

        let params = FieldKernelParams {
            num_elements: n,
            modulus,
            ..Default::default()
        };

        // Stress add
        let metrics = kernel.add(&a, &b, &mut c, &params).unwrap();
        assert!(metrics.elements_processed == n as u64);

        // Stress mul
        let metrics = kernel.mul(&a, &b, &mut c, &params).unwrap();
        assert!(metrics.elements_processed == n as u64);

        // Stress sub
        let metrics = kernel.sub(&a, &b, &mut c, &params).unwrap();
        assert!(metrics.elements_processed == n as u64);

        // Verify correctness of add
        kernel.add(&a, &b, &mut c, &params).unwrap();
        for i in 0..n {
            let expected = addmod(a[i], b[i], modulus);
            assert_eq!(c[i], expected, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_batch_inverse_stress() {
        let kernel = CPUFieldKernel;
        let modulus = 97u64;

        let n = 1000;
        let mut values: Vec<u64> = (1..=n as u64).map(|x| x % modulus).collect();
        // Avoid zeros
        for v in values.iter_mut() {
            if *v == 0 {
                *v = 1;
            }
        }
        let original = values.clone();

        let params = FieldKernelParams {
            num_elements: n,
            modulus,
            ..Default::default()
        };

        kernel.batch_inverse(&mut values, &params).unwrap();

        // Verify each inverse
        for (i, (&orig, &inv)) in original.iter().zip(values.iter()).enumerate() {
            let product = mulmod(orig, inv, modulus);
            assert_eq!(product, 1, "Inverse failed at index {}: {} * {} != 1", i, orig, inv);
        }
    }

    #[test]
    fn test_kernel_manager_accumulated_metrics() {
        let mgr = GPUKernelManager::new_cpu_fallback();
        let modulus = 97u64;

        // Initially zero
        let metrics = mgr.accumulated_metrics();
        assert_eq!(metrics.total_ntt_calls, 0);
        assert_eq!(metrics.total_field_calls, 0);

        // Do some operations
        let mut data = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        mgr.ntt_forward(&mut data, modulus, 3).unwrap();
        mgr.ntt_inverse(&mut data, modulus, 3).unwrap();

        let a = vec![1u64, 2, 3];
        let b = vec![4u64, 5, 6];
        let mut c = vec![0u64; 3];
        mgr.field_add(&a, &b, &mut c, modulus).unwrap();
        mgr.field_mul(&a, &b, &mut c, modulus).unwrap();

        let metrics = mgr.accumulated_metrics();
        assert_eq!(metrics.total_ntt_calls, 2);
        assert_eq!(metrics.total_field_calls, 2);
        assert!(metrics.total_elements_processed > 0);

        // Reset
        mgr.reset_metrics();
        let metrics = mgr.accumulated_metrics();
        assert_eq!(metrics.total_ntt_calls, 0);
    }

    #[test]
    fn test_concurrent_kernel_manager() {
        use std::thread;

        let mgr = Arc::new(GPUKernelManager::new_cpu_fallback());
        let modulus = 97u64;
        let mut handles = vec![];

        // Spawn threads doing field operations
        for _ in 0..4 {
            let mgr_clone = Arc::clone(&mgr);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let a = vec![10u64, 20, 30, 40];
                    let b = vec![5u64, 10, 15, 20];
                    let mut c = vec![0u64; 4];
                    let _ = mgr_clone.field_add(&a, &b, &mut c, modulus);
                    let _ = mgr_clone.field_mul(&a, &b, &mut c, modulus);
                }
            }));
        }

        // Spawn threads doing NTT
        for _ in 0..4 {
            let mgr_clone = Arc::clone(&mgr);
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    let mut data = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
                    let _ = mgr_clone.ntt_forward(&mut data, modulus, 3);
                    let _ = mgr_clone.ntt_inverse(&mut data, modulus, 3);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify metrics accumulated correctly
        let metrics = mgr.accumulated_metrics();
        assert!(metrics.total_field_calls >= 4 * 100 * 2);
        assert!(metrics.total_ntt_calls >= 4 * 50 * 2);
    }

    #[test]
    fn test_gpu_error_variants() {
        // Test all error variant creation
        let errors = vec![
            GPUError::AllocationFailed { 
                requested: 100, 
                available: 50, 
                backend: GPUBackend::Cuda 
            },
            GPUError::BufferNotFound { handle_id: 42 },
            GPUError::InvalidBufferSize { 
                size: 0, 
                reason: "test" 
            },
            GPUError::MemoryLimitExceeded { 
                current: 100, 
                requested: 200, 
                limit: 150 
            },
            GPUError::InvalidKernelParams { 
                kernel: "MSM", 
                reason: "test".to_string() 
            },
            GPUError::KernelExecutionFailed { 
                kernel: "NTT".to_string(), 
                reason: "test".to_string() 
            },
            GPUError::DataSizeMismatch { 
                expected: 100, 
                actual: 50, 
                context: "test" 
            },
            GPUError::LockContention { operation: "allocate" },
            GPUError::BackendNotAvailable { backend: GPUBackend::Metal },
            GPUError::InvalidNTTSize { size: 100 },
            GPUError::ZeroModulus,
            GPUError::InvalidRootOfUnity { root: 0, modulus: 97 },
            GPUError::DeviceNotInitialized,
            GPUError::TransferFailed { direction: "H2D", size: 1024 },
        ];

        for err in errors {
            // Ensure Display works
            let _ = err.to_string();
            // Ensure Debug works
            let _ = format!("{:?}", err);
        }
    }

    #[test]
    fn test_buffer_usage_presets() {
        let read_only = BufferUsage::read_only();
        assert!(read_only.kernel_read);
        assert!(!read_only.kernel_write);

        let write_only = BufferUsage::write_only();
        assert!(!write_only.kernel_read);
        assert!(write_only.kernel_write);

        let default = BufferUsage::default();
        assert!(default.kernel_read);
        assert!(default.kernel_write);
        assert!(default.copy_dst);
        assert!(default.copy_src);
    }
}

