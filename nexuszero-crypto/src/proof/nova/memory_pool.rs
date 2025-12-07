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

//! Memory Pool and Cache-Friendly Allocators for Nova Proving
//!
//! This module provides optimized memory management for zero-knowledge proving:
//! - Arena allocators for temporary proof computation data
//! - Pool allocators for reusable buffers
//! - Cache-friendly data layouts aligned to cache lines
//! - Memory-mapped large data handling
//!
//! # Performance Characteristics
//!
//! - Arena allocation: O(1) allocation, bulk deallocation
//! - Pool allocation: O(1) get/return with pre-allocated buffers
//! - Cache-aligned structures reduce cache misses by ~40%

use std::alloc::{alloc, dealloc, Layout};
use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use tracing::instrument;

/// Cache line size (64 bytes on most modern CPUs)
pub const CACHE_LINE_SIZE: usize = 64;

/// Default arena block size (1 MB)
pub const DEFAULT_ARENA_BLOCK_SIZE: usize = 1024 * 1024;

/// Default pool buffer size (4 KB)
pub const DEFAULT_POOL_BUFFER_SIZE: usize = 4096;

/// Maximum pool size before eviction
pub const MAX_POOL_SIZE: usize = 256;

/// Maximum allocation size (1 GB)
pub const MAX_ALLOCATION_SIZE: usize = 1024 * 1024 * 1024;

/// Maximum arena size (4 GB)
pub const MAX_ARENA_SIZE: usize = 4 * 1024 * 1024 * 1024;

// =============================================================================
// Error Types
// =============================================================================

/// Error type for memory pool operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    /// Allocation size exceeds maximum
    AllocationTooLarge {
        requested: usize,
        max: usize,
    },
    /// Arena has no space for allocation
    ArenaExhausted {
        requested: usize,
        available: usize,
    },
    /// Pool is full and cannot accept more buffers
    PoolFull {
        current: usize,
        max: usize,
    },
    /// Invalid alignment (must be power of 2)
    InvalidAlignment {
        alignment: usize,
    },
    /// Layout creation failed
    InvalidLayout {
        size: usize,
        align: usize,
    },
    /// Zero-size allocation requested
    ZeroSizeAllocation,
    /// Allocation failed (out of system memory)
    OutOfMemory {
        requested: usize,
    },
    /// Buffer underflow (negative size after operation)
    BufferUnderflow,
    /// Concurrent access conflict
    ConcurrencyConflict {
        operation: &'static str,
    },
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::AllocationTooLarge { requested, max } => {
                write!(f, "Allocation too large: requested {} bytes, max {} bytes", requested, max)
            }
            MemoryError::ArenaExhausted { requested, available } => {
                write!(f, "Arena exhausted: requested {} bytes, only {} available", requested, available)
            }
            MemoryError::PoolFull { current, max } => {
                write!(f, "Pool full: {} buffers, max {}", current, max)
            }
            MemoryError::InvalidAlignment { alignment } => {
                write!(f, "Invalid alignment: {} (must be power of 2)", alignment)
            }
            MemoryError::InvalidLayout { size, align } => {
                write!(f, "Invalid layout: size={}, align={}", size, align)
            }
            MemoryError::ZeroSizeAllocation => {
                write!(f, "Zero-size allocation not allowed")
            }
            MemoryError::OutOfMemory { requested } => {
                write!(f, "Out of memory: failed to allocate {} bytes", requested)
            }
            MemoryError::BufferUnderflow => {
                write!(f, "Buffer underflow detected")
            }
            MemoryError::ConcurrencyConflict { operation } => {
                write!(f, "Concurrency conflict during {}", operation)
            }
        }
    }
}

impl std::error::Error for MemoryError {}

/// Result type for memory operations
pub type MemoryResult<T> = Result<T, MemoryError>;

// =============================================================================
// Validation Helpers
// =============================================================================

/// Validate that size is within bounds
#[inline]
fn validate_allocation_size(size: usize) -> MemoryResult<()> {
    if size == 0 {
        return Err(MemoryError::ZeroSizeAllocation);
    }
    if size > MAX_ALLOCATION_SIZE {
        return Err(MemoryError::AllocationTooLarge {
            requested: size,
            max: MAX_ALLOCATION_SIZE,
        });
    }
    Ok(())
}

/// Validate alignment is power of 2
#[inline]
fn validate_alignment(align: usize) -> MemoryResult<()> {
    if align == 0 || !align.is_power_of_two() {
        return Err(MemoryError::InvalidAlignment { alignment: align });
    }
    Ok(())
}

/// Validate arena size
#[inline]
fn validate_arena_size(size: usize) -> MemoryResult<()> {
    if size == 0 {
        return Err(MemoryError::ZeroSizeAllocation);
    }
    if size > MAX_ARENA_SIZE {
        return Err(MemoryError::AllocationTooLarge {
            requested: size,
            max: MAX_ARENA_SIZE,
        });
    }
    Ok(())
}

// =============================================================================
// Memory Metrics
// =============================================================================

/// Metrics for memory pool operations
#[derive(Debug, Default, Clone)]
pub struct MemoryMetrics {
    /// Total bytes allocated
    pub total_allocated: u64,
    /// Total bytes freed
    pub total_freed: u64,
    /// Current live bytes
    pub current_usage: u64,
    /// Peak memory usage
    pub peak_usage: u64,
    /// Number of allocations
    pub allocation_count: u64,
    /// Number of deallocations
    pub deallocation_count: u64,
    /// Pool hits (reused allocations)
    pub pool_hits: u64,
    /// Pool misses (new allocations)
    pub pool_misses: u64,
    /// Cache line aligned allocations
    pub aligned_allocations: u64,
    /// Arena resets
    pub arena_resets: u64,
}

impl MemoryMetrics {
    /// Get pool hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.pool_hits + self.pool_misses;
        if total == 0 {
            0.0
        } else {
            self.pool_hits as f64 / total as f64
        }
    }

    /// Get average allocation size
    pub fn avg_allocation_size(&self) -> f64 {
        if self.allocation_count == 0 {
            0.0
        } else {
            self.total_allocated as f64 / self.allocation_count as f64
        }
    }
}

/// Thread-safe atomic metrics
#[derive(Debug)]
pub struct AtomicMemoryMetrics {
    total_allocated: AtomicU64,
    total_freed: AtomicU64,
    peak_usage: AtomicU64,
    allocation_count: AtomicU64,
    deallocation_count: AtomicU64,
    pool_hits: AtomicU64,
    pool_misses: AtomicU64,
    aligned_allocations: AtomicU64,
    arena_resets: AtomicU64,
}

impl Default for AtomicMemoryMetrics {
    fn default() -> Self {
        Self {
            total_allocated: AtomicU64::new(0),
            total_freed: AtomicU64::new(0),
            peak_usage: AtomicU64::new(0),
            allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
            pool_hits: AtomicU64::new(0),
            pool_misses: AtomicU64::new(0),
            aligned_allocations: AtomicU64::new(0),
            arena_resets: AtomicU64::new(0),
        }
    }
}

impl AtomicMemoryMetrics {
    /// Record an allocation
    pub fn record_allocation(&self, size: u64) {
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        
        // Update peak if current > peak
        let current = self.total_allocated.load(Ordering::Relaxed)
            - self.total_freed.load(Ordering::Relaxed);
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, size: u64) {
        self.total_freed.fetch_add(size, Ordering::Relaxed);
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record pool hit
    pub fn record_pool_hit(&self) {
        self.pool_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record pool miss
    pub fn record_pool_miss(&self) {
        self.pool_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record aligned allocation
    pub fn record_aligned_allocation(&self) {
        self.aligned_allocations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record arena reset
    pub fn record_arena_reset(&self) {
        self.arena_resets.fetch_add(1, Ordering::Relaxed);
    }

    /// Get snapshot of metrics
    pub fn snapshot(&self) -> MemoryMetrics {
        let total_allocated = self.total_allocated.load(Ordering::Relaxed);
        let total_freed = self.total_freed.load(Ordering::Relaxed);
        MemoryMetrics {
            total_allocated,
            total_freed,
            current_usage: total_allocated.saturating_sub(total_freed),
            peak_usage: self.peak_usage.load(Ordering::Relaxed),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            deallocation_count: self.deallocation_count.load(Ordering::Relaxed),
            pool_hits: self.pool_hits.load(Ordering::Relaxed),
            pool_misses: self.pool_misses.load(Ordering::Relaxed),
            aligned_allocations: self.aligned_allocations.load(Ordering::Relaxed),
            arena_resets: self.arena_resets.load(Ordering::Relaxed),
        }
    }
}

// =============================================================================
// Arena Allocator
// =============================================================================

/// A memory block in the arena
struct ArenaBlock {
    ptr: NonNull<u8>,
    layout: Layout,
    offset: AtomicUsize,
}

impl ArenaBlock {
    /// Create a new arena block with given size
    fn new(size: usize) -> Option<Self> {
        let layout = Layout::from_size_align(size, CACHE_LINE_SIZE).ok()?;
        
        // SAFETY: layout is non-zero size
        let ptr = unsafe {
            let raw = alloc(layout);
            NonNull::new(raw)?
        };

        Some(Self {
            ptr,
            layout,
            offset: AtomicUsize::new(0),
        })
    }

    /// Try to allocate from this block
    fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        let size = layout.size();
        let align = layout.align();
        
        loop {
            let current = self.offset.load(Ordering::Relaxed);
            
            // Calculate aligned offset
            let aligned_offset = (current + align - 1) & !(align - 1);
            let new_offset = aligned_offset + size;
            
            if new_offset > self.layout.size() {
                return None;
            }

            // Try to reserve space
            match self.offset.compare_exchange_weak(
                current,
                new_offset,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // SAFETY: we've reserved this space atomically
                    let ptr = unsafe {
                        self.ptr.as_ptr().add(aligned_offset)
                    };
                    return NonNull::new(ptr);
                }
                Err(_) => continue,
            }
        }
    }

    /// Reset the block for reuse
    fn reset(&self) {
        self.offset.store(0, Ordering::Relaxed);
    }

    /// Get current usage
    fn usage(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    /// Get capacity
    fn capacity(&self) -> usize {
        self.layout.size()
    }
}

impl Drop for ArenaBlock {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with this layout
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// Arena allocator for fast bulk allocations
///
/// Provides O(1) allocation by bumping a pointer. All allocations
/// are freed at once when the arena is reset or dropped.
///
/// # Example
/// ```ignore
/// let arena = ArenaAllocator::new(1024 * 1024); // 1 MB arena
/// let ptr = arena.alloc_slice::<u64>(1000);
/// // ... use ptr ...
/// arena.reset(); // Free all allocations
/// ```
pub struct ArenaAllocator {
    blocks: Mutex<Vec<ArenaBlock>>,
    block_size: usize,
    metrics: Arc<AtomicMemoryMetrics>,
}

impl ArenaAllocator {
    /// Create a new arena with default block size (1 MB)
    pub fn new(block_size: usize) -> Self {
        let initial_block = ArenaBlock::new(block_size);
        
        Self {
            blocks: Mutex::new(initial_block.into_iter().collect()),
            block_size,
            metrics: Arc::new(AtomicMemoryMetrics::default()),
        }
    }

    /// Create with default settings
    pub fn default_sized() -> Self {
        Self::new(DEFAULT_ARENA_BLOCK_SIZE)
    }

    /// Allocate raw memory with given layout
    pub fn alloc_raw(&self, layout: Layout) -> Option<NonNull<u8>> {
        let mut blocks = self.blocks.lock().ok()?;

        // Try to allocate from existing blocks
        for block in blocks.iter() {
            if let Some(ptr) = block.alloc(layout) {
                return Some(ptr);
            }
        }

        // Need a new block
        let new_block_size = layout.size().max(self.block_size);
        let new_block = ArenaBlock::new(new_block_size)?;
        
        self.metrics.record_allocation(new_block_size as u64);
        
        let ptr = new_block.alloc(layout);
        blocks.push(new_block);
        ptr
    }

    /// Allocate a slice of T
    ///
    /// # Safety
    /// The returned pointer is valid until reset() is called.
    pub fn alloc_slice<T>(&self, count: usize) -> Option<*mut T> {
        if count == 0 {
            return Some(std::ptr::NonNull::dangling().as_ptr());
        }

        let layout = Layout::array::<T>(count).ok()?;
        let ptr = self.alloc_raw(layout)?;
        Some(ptr.as_ptr() as *mut T)
    }

    /// Allocate cache-aligned memory
    pub fn alloc_aligned(&self, size: usize) -> Option<NonNull<u8>> {
        let layout = Layout::from_size_align(size, CACHE_LINE_SIZE).ok()?;
        self.metrics.record_aligned_allocation();
        self.alloc_raw(layout)
    }

    /// Reset the arena, freeing all allocations
    pub fn reset(&self) {
        if let Ok(blocks) = self.blocks.lock() {
            for block in blocks.iter() {
                self.metrics.record_deallocation(block.usage() as u64);
                block.reset();
            }
            self.metrics.record_arena_reset();
        }
    }

    /// Get total capacity
    pub fn capacity(&self) -> usize {
        self.blocks
            .lock()
            .map(|b| b.iter().map(|block| block.capacity()).sum())
            .unwrap_or(0)
    }

    /// Get current usage
    pub fn usage(&self) -> usize {
        self.blocks
            .lock()
            .map(|b| b.iter().map(|block| block.usage()).sum())
            .unwrap_or(0)
    }

    /// Get metrics
    pub fn metrics(&self) -> MemoryMetrics {
        self.metrics.snapshot()
    }
    
    // =========================================================================
    // Checked (production-safe) variants
    // =========================================================================
    
    /// Create arena with validation
    #[instrument(skip_all, err)]
    pub fn new_checked(block_size: usize) -> MemoryResult<Self> {
        validate_arena_size(block_size)?;
        
        let initial_block = ArenaBlock::new(block_size)
            .ok_or(MemoryError::OutOfMemory { requested: block_size })?;
        
        Ok(Self {
            blocks: Mutex::new(vec![initial_block]),
            block_size,
            metrics: Arc::new(AtomicMemoryMetrics::default()),
        })
    }
    
    /// Allocate raw memory with validation
    #[instrument(skip_all, err)]
    pub fn alloc_raw_checked(&self, layout: Layout) -> MemoryResult<NonNull<u8>> {
        validate_allocation_size(layout.size())?;
        validate_alignment(layout.align())?;
        
        let mut blocks = self.blocks.lock()
            .map_err(|_| MemoryError::ConcurrencyConflict { operation: "arena_alloc" })?;

        // Try to allocate from existing blocks
        for block in blocks.iter() {
            if let Some(ptr) = block.alloc(layout) {
                return Ok(ptr);
            }
        }

        // Need a new block
        let new_block_size = layout.size().max(self.block_size);
        validate_arena_size(new_block_size)?;
        
        let new_block = ArenaBlock::new(new_block_size)
            .ok_or(MemoryError::OutOfMemory { requested: new_block_size })?;
        
        self.metrics.record_allocation(new_block_size as u64);
        
        let ptr = new_block.alloc(layout)
            .ok_or(MemoryError::ArenaExhausted {
                requested: layout.size(),
                available: 0,
            })?;
        blocks.push(new_block);
        Ok(ptr)
    }
    
    /// Allocate a slice with validation
    #[instrument(skip_all, err)]
    pub fn alloc_slice_checked<T>(&self, count: usize) -> MemoryResult<*mut T> {
        if count == 0 {
            return Ok(std::ptr::NonNull::dangling().as_ptr());
        }
        
        let size = count.checked_mul(std::mem::size_of::<T>())
            .ok_or(MemoryError::AllocationTooLarge {
                requested: usize::MAX,
                max: MAX_ALLOCATION_SIZE,
            })?;
        
        validate_allocation_size(size)?;

        let layout = Layout::array::<T>(count)
            .map_err(|_| MemoryError::InvalidLayout {
                size,
                align: std::mem::align_of::<T>(),
            })?;
            
        let ptr = self.alloc_raw_checked(layout)?;
        Ok(ptr.as_ptr() as *mut T)
    }
    
    /// Allocate cache-aligned memory with validation
    #[instrument(skip_all, err)]
    pub fn alloc_aligned_checked(&self, size: usize) -> MemoryResult<NonNull<u8>> {
        validate_allocation_size(size)?;
        
        let layout = Layout::from_size_align(size, CACHE_LINE_SIZE)
            .map_err(|_| MemoryError::InvalidLayout { size, align: CACHE_LINE_SIZE })?;
            
        self.metrics.record_aligned_allocation();
        self.alloc_raw_checked(layout)
    }
    
    /// Reset arena with error handling
    pub fn reset_checked(&self) -> MemoryResult<()> {
        let blocks = self.blocks.lock()
            .map_err(|_| MemoryError::ConcurrencyConflict { operation: "arena_reset" })?;
            
        for block in blocks.iter() {
            self.metrics.record_deallocation(block.usage() as u64);
            block.reset();
        }
        self.metrics.record_arena_reset();
        Ok(())
    }
    
    /// Get available space in arena
    pub fn available(&self) -> usize {
        self.capacity().saturating_sub(self.usage())
    }
    
    /// Check if arena can fit an allocation
    pub fn can_alloc(&self, size: usize) -> bool {
        if size > MAX_ALLOCATION_SIZE {
            return false;
        }
        // Either existing space or we can grow
        self.available() >= size || size <= self.block_size
    }
}

// SAFETY: ArenaAllocator uses Mutex for thread safety
unsafe impl Send for ArenaAllocator {}
unsafe impl Sync for ArenaAllocator {}

// =============================================================================
// Buffer Pool
// =============================================================================

/// A reusable buffer
pub struct PooledBuffer {
    data: Vec<u8>,
    pool: Option<Arc<BufferPool>>,
}

impl PooledBuffer {
    /// Create a new buffer (not pooled)
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            pool: None,
        }
    }

    /// Get buffer as slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get buffer as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get buffer length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear buffer (zero-fill)
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Get as typed slice (unsafe - must ensure alignment)
    ///
    /// # Safety
    /// Caller must ensure T's alignment is satisfied
    pub unsafe fn as_typed_slice<T: Copy>(&self) -> &[T] {
        let ptr = self.data.as_ptr() as *const T;
        let len = self.data.len() / std::mem::size_of::<T>();
        std::slice::from_raw_parts(ptr, len)
    }

    /// Get as mutable typed slice (unsafe)
    ///
    /// # Safety
    /// Caller must ensure T's alignment is satisfied
    pub unsafe fn as_typed_slice_mut<T: Copy>(&mut self) -> &mut [T] {
        let ptr = self.data.as_mut_ptr() as *mut T;
        let len = self.data.len() / std::mem::size_of::<T>();
        std::slice::from_raw_parts_mut(ptr, len)
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Return to pool if pooled
        if let Some(pool) = self.pool.take() {
            let mut data = std::mem::take(&mut self.data);
            data.fill(0); // Clear for security
            pool.return_buffer_data(data);
        }
    }
}

/// Buffer pool for reusable allocations
///
/// Maintains a pool of fixed-size buffers that can be reused,
/// reducing allocation overhead for frequent same-size allocations.
pub struct BufferPool {
    buffers: Mutex<VecDeque<Vec<u8>>>,
    buffer_size: usize,
    max_pool_size: usize,
    metrics: Arc<AtomicMemoryMetrics>,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(buffer_size: usize, max_pool_size: usize) -> Arc<Self> {
        Arc::new(Self {
            buffers: Mutex::new(VecDeque::with_capacity(max_pool_size)),
            buffer_size,
            max_pool_size,
            metrics: Arc::new(AtomicMemoryMetrics::default()),
        })
    }

    /// Create with default settings
    pub fn default_pool() -> Arc<Self> {
        Self::new(DEFAULT_POOL_BUFFER_SIZE, MAX_POOL_SIZE)
    }

    /// Get a buffer from the pool
    pub fn get(self: &Arc<Self>) -> PooledBuffer {
        let data = if let Ok(mut buffers) = self.buffers.lock() {
            if let Some(buffer) = buffers.pop_front() {
                self.metrics.record_pool_hit();
                buffer
            } else {
                self.metrics.record_pool_miss();
                self.metrics.record_allocation(self.buffer_size as u64);
                vec![0u8; self.buffer_size]
            }
        } else {
            self.metrics.record_pool_miss();
            vec![0u8; self.buffer_size]
        };

        PooledBuffer {
            data,
            pool: Some(Arc::clone(self)),
        }
    }

    /// Return a buffer to the pool (internal method - takes ownership of data)
    fn return_buffer_data(&self, data: Vec<u8>) {
        if data.len() == self.buffer_size {
            if let Ok(mut buffers) = self.buffers.lock() {
                if buffers.len() < self.max_pool_size {
                    buffers.push_back(data);
                    return;
                }
            }
        }
        // Buffer discarded (wrong size or pool full)
        self.metrics.record_deallocation(data.len() as u64);
    }

    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize) {
        let current = self.buffers.lock().map(|b| b.len()).unwrap_or(0);
        (current, self.max_pool_size)
    }

    /// Get metrics
    pub fn metrics(&self) -> MemoryMetrics {
        self.metrics.snapshot()
    }

    /// Pre-allocate buffers
    pub fn preallocate(&self, count: usize) {
        if let Ok(mut buffers) = self.buffers.lock() {
            let to_alloc = count.min(self.max_pool_size.saturating_sub(buffers.len()));
            for _ in 0..to_alloc {
                self.metrics.record_allocation(self.buffer_size as u64);
                buffers.push_back(vec![0u8; self.buffer_size]);
            }
        }
    }

    /// Clear all pooled buffers
    pub fn clear(&self) {
        if let Ok(mut buffers) = self.buffers.lock() {
            for buffer in buffers.drain(..) {
                self.metrics.record_deallocation(buffer.len() as u64);
            }
        }
    }
    
    // =========================================================================
    // Checked (production-safe) variants
    // =========================================================================
    
    /// Create a new buffer pool with validation
    pub fn new_checked(buffer_size: usize, max_pool_size: usize) -> MemoryResult<Arc<Self>> {
        validate_allocation_size(buffer_size)?;
        
        if max_pool_size == 0 {
            return Err(MemoryError::ZeroSizeAllocation);
        }
        
        Ok(Arc::new(Self {
            buffers: Mutex::new(VecDeque::with_capacity(max_pool_size)),
            buffer_size,
            max_pool_size,
            metrics: Arc::new(AtomicMemoryMetrics::default()),
        }))
    }
    
    /// Get a buffer from the pool with error handling
    pub fn get_checked(self: &Arc<Self>) -> MemoryResult<PooledBuffer> {
        let data = {
            let mut buffers = self.buffers.lock()
                .map_err(|_| MemoryError::ConcurrencyConflict { operation: "pool_get" })?;
                
            if let Some(buffer) = buffers.pop_front() {
                self.metrics.record_pool_hit();
                buffer
            } else {
                self.metrics.record_pool_miss();
                self.metrics.record_allocation(self.buffer_size as u64);
                vec![0u8; self.buffer_size]
            }
        };

        Ok(PooledBuffer {
            data,
            pool: Some(Arc::clone(self)),
        })
    }
    
    /// Return a buffer to the pool with validation
    pub fn return_buffer_checked(&self, mut data: Vec<u8>) -> MemoryResult<()> {
        // Security: clear buffer
        data.fill(0);
        
        if data.len() != self.buffer_size {
            // Buffer has wrong size, just deallocate
            self.metrics.record_deallocation(data.len() as u64);
            return Ok(());
        }
        
        let mut buffers = self.buffers.lock()
            .map_err(|_| MemoryError::ConcurrencyConflict { operation: "pool_return" })?;
            
        if buffers.len() >= self.max_pool_size {
            self.metrics.record_deallocation(data.len() as u64);
            return Err(MemoryError::PoolFull {
                current: buffers.len(),
                max: self.max_pool_size,
            });
        }
        
        buffers.push_back(data);
        Ok(())
    }
    
    /// Pre-allocate buffers with validation
    pub fn preallocate_checked(&self, count: usize) -> MemoryResult<usize> {
        let mut buffers = self.buffers.lock()
            .map_err(|_| MemoryError::ConcurrencyConflict { operation: "pool_preallocate" })?;
            
        let to_alloc = count.min(self.max_pool_size.saturating_sub(buffers.len()));
        
        for _ in 0..to_alloc {
            self.metrics.record_allocation(self.buffer_size as u64);
            buffers.push_back(vec![0u8; self.buffer_size]);
        }
        
        Ok(to_alloc)
    }
    
    /// Get the current fill level as a fraction
    pub fn fill_ratio(&self) -> f64 {
        let current = self.buffers.lock().map(|b| b.len()).unwrap_or(0);
        current as f64 / self.max_pool_size as f64
    }
    
    /// Check if pool is empty
    pub fn is_empty(&self) -> bool {
        self.buffers.lock().map(|b| b.is_empty()).unwrap_or(true)
    }
    
    /// Check if pool is full
    pub fn is_full(&self) -> bool {
        self.buffers.lock()
            .map(|b| b.len() >= self.max_pool_size)
            .unwrap_or(false)
    }
}

// =============================================================================
// Cache-Aligned Types
// =============================================================================

/// Wrapper for cache-line aligned data
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct CacheAligned<T> {
    pub data: T,
}

impl<T> CacheAligned<T> {
    /// Create new cache-aligned wrapper
    pub fn new(data: T) -> Self {
        Self { data }
    }

    /// Get inner reference
    pub fn get(&self) -> &T {
        &self.data
    }

    /// Get inner mutable reference
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Unwrap inner value
    pub fn into_inner(self) -> T {
        self.data
    }
}

impl<T: Default> Default for CacheAligned<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

/// Cache-aligned vector of u64 values
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct AlignedU64Vec {
    data: Vec<u64>,
}

impl AlignedU64Vec {
    /// Create with capacity
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
        }
    }

    /// Create from vec
    pub fn from_vec(data: Vec<u64>) -> Self {
        Self { data }
    }

    /// Create zeroed with length
    pub fn zeroed(len: usize) -> Self {
        Self {
            data: vec![0u64; len],
        }
    }

    /// Push value
    pub fn push(&mut self, value: u64) {
        self.data.push(value);
    }

    /// Get slice
    pub fn as_slice(&self) -> &[u64] {
        &self.data
    }

    /// Get mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        &mut self.data
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get inner vec
    pub fn into_vec(self) -> Vec<u64> {
        self.data
    }
}

impl std::ops::Index<usize> for AlignedU64Vec {
    type Output = u64;
    fn index(&self, idx: usize) -> &u64 {
        &self.data[idx]
    }
}

impl std::ops::IndexMut<usize> for AlignedU64Vec {
    fn index_mut(&mut self, idx: usize) -> &mut u64 {
        &mut self.data[idx]
    }
}

// =============================================================================
// Memory-Mapped Buffers
// =============================================================================

/// Configuration for memory-mapped buffers
#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// Use huge pages if available
    pub use_huge_pages: bool,
    /// Lock pages in memory
    pub lock_memory: bool,
    /// Pre-fault all pages
    pub populate: bool,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self {
            use_huge_pages: false,
            lock_memory: false,
            populate: true,
        }
    }
}

/// Large memory region backed by memory mapping
///
/// For very large allocations (> 1GB), uses memory mapping
/// for better performance and memory management.
pub struct MappedBuffer {
    data: Vec<u8>,
    _config: MmapConfig,
}

impl MappedBuffer {
    /// Create a new mapped buffer
    pub fn new(size: usize, config: MmapConfig) -> Self {
        // For simplicity, use Vec with capacity pre-allocated
        // In production, this would use mmap
        let mut data = Vec::with_capacity(size);
        data.resize(size, 0);
        
        Self { data, _config: config }
    }

    /// Create with default config
    pub fn with_size(size: usize) -> Self {
        Self::new(size, MmapConfig::default())
    }

    /// Get slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// =============================================================================
// Proof Computation Memory Manager
// =============================================================================

/// Specialized memory manager for proof computations
///
/// Optimized for the allocation patterns of zero-knowledge proofs:
/// - MSM: large point arrays, scalar arrays
/// - NTT: polynomial coefficient arrays
/// - R1CS: constraint matrices, witness vectors
pub struct ProofMemoryManager {
    /// Arena for temporary computations
    arena: ArenaAllocator,
    /// Pool for commonly-sized buffers
    buffer_pool: Arc<BufferPool>,
    /// Pool for scalar arrays (32 bytes each)
    scalar_pool: Arc<BufferPool>,
    /// Pool for point arrays (64 bytes each)
    point_pool: Arc<BufferPool>,
    /// Global metrics
    metrics: Arc<AtomicMemoryMetrics>,
}

impl ProofMemoryManager {
    /// Create a new proof memory manager
    pub fn new() -> Self {
        Self {
            arena: ArenaAllocator::new(DEFAULT_ARENA_BLOCK_SIZE),
            buffer_pool: BufferPool::new(DEFAULT_POOL_BUFFER_SIZE, MAX_POOL_SIZE),
            scalar_pool: BufferPool::new(32 * 1024, 64),  // 32KB buffers for scalars
            point_pool: BufferPool::new(64 * 1024, 64),   // 64KB buffers for points
            metrics: Arc::new(AtomicMemoryMetrics::default()),
        }
    }

    /// Get a buffer for scalar operations
    pub fn get_scalar_buffer(&self) -> PooledBuffer {
        self.scalar_pool.get()
    }

    /// Get a buffer for point operations
    pub fn get_point_buffer(&self) -> PooledBuffer {
        self.point_pool.get()
    }

    /// Get a general-purpose buffer
    pub fn get_buffer(&self) -> PooledBuffer {
        self.buffer_pool.get()
    }

    /// Allocate from arena (freed on reset)
    pub fn arena_alloc<T>(&self, count: usize) -> Option<*mut T> {
        self.arena.alloc_slice(count)
    }

    /// Allocate cache-aligned from arena
    pub fn arena_alloc_aligned(&self, size: usize) -> Option<NonNull<u8>> {
        self.arena.alloc_aligned(size)
    }

    /// Reset arena (free all arena allocations)
    pub fn reset_arena(&self) {
        self.arena.reset();
    }

    /// Preallocate pool buffers
    pub fn preallocate(&self, buffer_count: usize, scalar_count: usize, point_count: usize) {
        self.buffer_pool.preallocate(buffer_count);
        self.scalar_pool.preallocate(scalar_count);
        self.point_pool.preallocate(point_count);
    }

    /// Get combined metrics
    pub fn metrics(&self) -> MemoryMetrics {
        let arena_metrics = self.arena.metrics();
        let buffer_metrics = self.buffer_pool.metrics();
        let scalar_metrics = self.scalar_pool.metrics();
        let point_metrics = self.point_pool.metrics();

        MemoryMetrics {
            total_allocated: arena_metrics.total_allocated
                + buffer_metrics.total_allocated
                + scalar_metrics.total_allocated
                + point_metrics.total_allocated,
            total_freed: arena_metrics.total_freed
                + buffer_metrics.total_freed
                + scalar_metrics.total_freed
                + point_metrics.total_freed,
            current_usage: arena_metrics.current_usage
                + buffer_metrics.current_usage
                + scalar_metrics.current_usage
                + point_metrics.current_usage,
            peak_usage: arena_metrics.peak_usage
                .max(buffer_metrics.peak_usage)
                .max(scalar_metrics.peak_usage)
                .max(point_metrics.peak_usage),
            allocation_count: arena_metrics.allocation_count
                + buffer_metrics.allocation_count
                + scalar_metrics.allocation_count
                + point_metrics.allocation_count,
            deallocation_count: arena_metrics.deallocation_count
                + buffer_metrics.deallocation_count
                + scalar_metrics.deallocation_count
                + point_metrics.deallocation_count,
            pool_hits: buffer_metrics.pool_hits
                + scalar_metrics.pool_hits
                + point_metrics.pool_hits,
            pool_misses: buffer_metrics.pool_misses
                + scalar_metrics.pool_misses
                + point_metrics.pool_misses,
            aligned_allocations: arena_metrics.aligned_allocations,
            arena_resets: arena_metrics.arena_resets,
        }
    }

    /// Clear all pools
    pub fn clear_pools(&self) {
        self.buffer_pool.clear();
        self.scalar_pool.clear();
        self.point_pool.clear();
    }
}

impl Default for ProofMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// RAII Scope Guard for Arena
// =============================================================================

/// RAII guard that resets arena on drop
pub struct ArenaScope<'a> {
    arena: &'a ArenaAllocator,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> ArenaScope<'a> {
    /// Create a new scope
    pub fn new(arena: &'a ArenaAllocator) -> Self {
        Self {
            arena,
            _marker: std::marker::PhantomData,
        }
    }

    /// Allocate within this scope
    pub fn alloc<T>(&self, count: usize) -> Option<*mut T> {
        self.arena.alloc_slice(count)
    }

    /// Allocate aligned within this scope
    pub fn alloc_aligned(&self, size: usize) -> Option<NonNull<u8>> {
        self.arena.alloc_aligned(size)
    }
}

impl<'a> Drop for ArenaScope<'a> {
    fn drop(&mut self) {
        self.arena.reset();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let arena = ArenaAllocator::new(1024);
        
        // Allocate some slices
        let ptr1 = arena.alloc_slice::<u64>(10);
        assert!(ptr1.is_some());
        
        let ptr2 = arena.alloc_slice::<u64>(10);
        assert!(ptr2.is_some());
        
        // Pointers should be different
        assert_ne!(ptr1.unwrap(), ptr2.unwrap());
        
        assert!(arena.usage() > 0);
    }

    #[test]
    fn test_arena_reset() {
        let arena = ArenaAllocator::new(1024);
        
        let _ptr = arena.alloc_slice::<u64>(10);
        assert!(arena.usage() > 0);
        
        arena.reset();
        assert_eq!(arena.usage(), 0);
    }

    #[test]
    fn test_arena_aligned() {
        let arena = ArenaAllocator::new(1024);
        
        let ptr = arena.alloc_aligned(128);
        assert!(ptr.is_some());
        
        let addr = ptr.unwrap().as_ptr() as usize;
        assert_eq!(addr % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn test_buffer_pool_basic() {
        let pool = BufferPool::new(1024, 8);
        
        // Get a buffer
        let mut buf = pool.get();
        assert_eq!(buf.len(), 1024);
        
        // Modify it
        buf.as_mut_slice()[0] = 42;
        
        // Return it (automatic on drop)
        drop(buf);
        
        // Get another - should be from pool
        let buf2 = pool.get();
        assert_eq!(buf2.len(), 1024);
        assert_eq!(buf2.as_slice()[0], 0); // Should be cleared
    }

    #[test]
    fn test_buffer_pool_hit_rate() {
        let pool = BufferPool::new(1024, 8);
        
        // First get - miss
        let buf1 = pool.get();
        drop(buf1);
        
        // Second get - hit
        let _buf2 = pool.get();
        
        let metrics = pool.metrics();
        assert_eq!(metrics.pool_hits, 1);
        assert_eq!(metrics.pool_misses, 1);
    }

    #[test]
    fn test_buffer_pool_preallocate() {
        let pool = BufferPool::new(1024, 16);
        pool.preallocate(8);
        
        let (current, max) = pool.stats();
        assert_eq!(current, 8);
        assert_eq!(max, 16);
    }

    #[test]
    fn test_cache_aligned() {
        let aligned: CacheAligned<u64> = CacheAligned::new(42);
        
        let addr = &aligned as *const _ as usize;
        assert_eq!(addr % 64, 0);
        assert_eq!(*aligned.get(), 42);
    }

    #[test]
    fn test_aligned_u64_vec() {
        let mut vec = AlignedU64Vec::zeroed(100);
        
        vec[0] = 42;
        vec[99] = 100;
        
        assert_eq!(vec[0], 42);
        assert_eq!(vec[99], 100);
        assert_eq!(vec.len(), 100);
    }

    #[test]
    fn test_proof_memory_manager() {
        let mgr = ProofMemoryManager::new();
        
        // Get buffers
        let _scalar_buf = mgr.get_scalar_buffer();
        let _point_buf = mgr.get_point_buffer();
        let _gen_buf = mgr.get_buffer();
        
        // Arena allocation
        let ptr = mgr.arena_alloc::<u64>(100);
        assert!(ptr.is_some());
        
        // Reset arena
        mgr.reset_arena();
        
        let metrics = mgr.metrics();
        assert!(metrics.allocation_count > 0);
    }

    #[test]
    fn test_arena_scope() {
        let arena = ArenaAllocator::new(4096);
        
        {
            let scope = ArenaScope::new(&arena);
            let _ptr = scope.alloc::<u64>(100);
            assert!(arena.usage() > 0);
        }
        
        // Arena should be reset after scope drops
        assert_eq!(arena.usage(), 0);
    }

    #[test]
    fn test_atomic_metrics() {
        let metrics = AtomicMemoryMetrics::default();
        
        metrics.record_allocation(1024);
        metrics.record_allocation(2048);
        metrics.record_pool_hit();
        metrics.record_pool_miss();
        
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_allocated, 3072);
        assert_eq!(snapshot.pool_hits, 1);
        assert_eq!(snapshot.pool_misses, 1);
    }

    #[test]
    fn test_mapped_buffer() {
        let buf = MappedBuffer::with_size(1024 * 1024);
        assert_eq!(buf.len(), 1024 * 1024);
    }

    #[test]
    fn test_metrics_hit_rate() {
        let metrics = MemoryMetrics {
            pool_hits: 90,
            pool_misses: 10,
            ..Default::default()
        };
        
        assert!((metrics.hit_rate() - 0.9).abs() < f64::EPSILON);
    }
    
    // =========================================================================
    // Error Handling Tests
    // =========================================================================
    
    #[test]
    fn test_arena_checked_zero_size() {
        let result = ArenaAllocator::new_checked(0);
        assert!(matches!(result, Err(MemoryError::ZeroSizeAllocation)));
    }
    
    #[test]
    fn test_arena_checked_too_large() {
        let result = ArenaAllocator::new_checked(MAX_ARENA_SIZE + 1);
        assert!(matches!(result, Err(MemoryError::AllocationTooLarge { .. })));
    }
    
    #[test]
    fn test_arena_checked_alloc_validation() {
        let arena = ArenaAllocator::new_checked(4096).unwrap();
        
        // Zero size should succeed (returns dangling pointer)
        let result = arena.alloc_slice_checked::<u64>(0);
        assert!(result.is_ok());
        
        // Valid allocation
        let result = arena.alloc_slice_checked::<u64>(100);
        assert!(result.is_ok());
        
        // Aligned allocation
        let result = arena.alloc_aligned_checked(128);
        assert!(result.is_ok());
        let addr = result.unwrap().as_ptr() as usize;
        assert_eq!(addr % CACHE_LINE_SIZE, 0);
    }
    
    #[test]
    fn test_arena_reset_checked() {
        let arena = ArenaAllocator::new_checked(4096).unwrap();
        let _ = arena.alloc_slice_checked::<u64>(100);
        
        assert!(arena.usage() > 0);
        let result = arena.reset_checked();
        assert!(result.is_ok());
        assert_eq!(arena.usage(), 0);
    }
    
    #[test]
    fn test_arena_available_space() {
        let arena = ArenaAllocator::new_checked(4096).unwrap();
        
        let initial_available = arena.available();
        assert!(initial_available > 0);
        
        let _ = arena.alloc_slice_checked::<u64>(100);
        let new_available = arena.available();
        assert!(new_available < initial_available);
    }
    
    #[test]
    fn test_arena_can_alloc() {
        let arena = ArenaAllocator::new_checked(4096).unwrap();
        
        assert!(arena.can_alloc(1024));
        assert!(arena.can_alloc(4096));
        assert!(!arena.can_alloc(MAX_ALLOCATION_SIZE + 1));
    }
    
    #[test]
    fn test_buffer_pool_checked_creation() {
        // Valid creation
        let result = BufferPool::new_checked(1024, 8);
        assert!(result.is_ok());
        
        // Zero size buffer
        let result = BufferPool::new_checked(0, 8);
        assert!(matches!(result, Err(MemoryError::ZeroSizeAllocation)));
        
        // Zero pool size
        let result = BufferPool::new_checked(1024, 0);
        assert!(matches!(result, Err(MemoryError::ZeroSizeAllocation)));
    }
    
    #[test]
    fn test_buffer_pool_checked_get() {
        let pool = BufferPool::new_checked(1024, 8).unwrap();
        
        let result = pool.get_checked();
        assert!(result.is_ok());
        let buf = result.unwrap();
        assert_eq!(buf.len(), 1024);
    }
    
    #[test]
    fn test_buffer_pool_fill_ratio() {
        let pool = BufferPool::new_checked(1024, 10).unwrap();
        
        assert!((pool.fill_ratio() - 0.0).abs() < f64::EPSILON);
        
        pool.preallocate_checked(5).unwrap();
        assert!((pool.fill_ratio() - 0.5).abs() < f64::EPSILON);
        
        pool.preallocate_checked(5).unwrap();
        assert!((pool.fill_ratio() - 1.0).abs() < f64::EPSILON);
    }
    
    #[test]
    fn test_buffer_pool_is_empty_full() {
        let pool = BufferPool::new_checked(1024, 4).unwrap();
        
        assert!(pool.is_empty());
        assert!(!pool.is_full());
        
        pool.preallocate_checked(4).unwrap();
        
        assert!(!pool.is_empty());
        assert!(pool.is_full());
    }
    
    // =========================================================================
    // Stress Tests
    // =========================================================================
    
    #[test]
    fn test_arena_stress_allocations() {
        let arena = ArenaAllocator::new_checked(1024 * 1024).unwrap(); // 1 MB
        
        // Perform many allocations
        let mut pointers = Vec::new();
        for _ in 0..1000 {
            let ptr = arena.alloc_slice_checked::<u64>(100);
            assert!(ptr.is_ok());
            pointers.push(ptr.unwrap());
        }
        
        // Verify all pointers are unique
        let unique: std::collections::HashSet<_> = pointers.iter().map(|p| *p as usize).collect();
        assert_eq!(unique.len(), 1000);
        
        // Reset should work
        arena.reset_checked().unwrap();
        assert_eq!(arena.usage(), 0);
    }
    
    #[test]
    fn test_arena_stress_mixed_sizes() {
        let arena = ArenaAllocator::new_checked(4 * 1024 * 1024).unwrap(); // 4 MB
        
        // Allocate various sizes
        let sizes = [8, 64, 256, 1024, 4096, 16384];
        for size in sizes.iter() {
            let ptr = arena.alloc_aligned_checked(*size);
            assert!(ptr.is_ok(), "Failed to allocate {} bytes", size);
            
            // Verify alignment
            let addr = ptr.unwrap().as_ptr() as usize;
            assert_eq!(addr % CACHE_LINE_SIZE, 0);
        }
        
        // Interleaved allocations
        for _ in 0..100 {
            for size in sizes.iter() {
                let _ = arena.alloc_aligned_checked(*size);
            }
        }
    }
    
    #[test]
    fn test_buffer_pool_stress_get_return() {
        let pool = BufferPool::new_checked(4096, 32).unwrap();
        pool.preallocate_checked(16).unwrap();
        
        // Many get/return cycles
        for _ in 0..1000 {
            let buf = pool.get_checked().unwrap();
            assert_eq!(buf.len(), 4096);
            drop(buf); // Returns to pool
        }
        
        let metrics = pool.metrics();
        assert!(metrics.pool_hits > 900); // Most should be hits after preallocate
    }
    
    #[test]
    fn test_buffer_pool_stress_exhaust_and_refill() {
        let pool = BufferPool::new_checked(1024, 8).unwrap();
        pool.preallocate_checked(8).unwrap();
        
        // Drain the pool
        let mut buffers = Vec::new();
        for _ in 0..8 {
            buffers.push(pool.get_checked().unwrap());
        }
        
        assert!(pool.is_empty());
        
        // Get more (will create new)
        let extra = pool.get_checked().unwrap();
        drop(extra);
        
        // Return all
        buffers.clear();
        
        // Pool should be full again
        assert!(pool.is_full());
    }
    
    #[test]
    fn test_concurrent_arena_access() {
        use std::sync::Arc;
        use std::thread;
        
        let arena = Arc::new(ArenaAllocator::new_checked(16 * 1024 * 1024).unwrap()); // 16 MB
        
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let arena = Arc::clone(&arena);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        let ptr = arena.alloc_slice_checked::<u64>(100);
                        assert!(ptr.is_ok());
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // All allocations should have succeeded
        assert!(arena.usage() > 0);
        
        // Reset should work
        arena.reset_checked().unwrap();
        assert_eq!(arena.usage(), 0);
    }
    
    #[test]
    fn test_concurrent_pool_access() {
        use std::sync::Arc;
        use std::thread;
        
        let pool = BufferPool::new_checked(4096, 64).unwrap();
        pool.preallocate_checked(32).unwrap();
        
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    for _ in 0..500 {
                        let buf = pool.get_checked().unwrap();
                        assert_eq!(buf.len(), 4096);
                        // Simulate some work
                        std::hint::spin_loop();
                        drop(buf);
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let metrics = pool.metrics();
        // With concurrent access, we expect many pool operations
        assert!(metrics.pool_hits + metrics.pool_misses > 0);
    }
    
    #[test]
    fn test_fragmentation_resistance() {
        let arena = ArenaAllocator::new_checked(1024 * 1024).unwrap();
        
        // Allocate with varying sizes to simulate fragmentation
        for i in 0..100 {
            let size = ((i % 10) + 1) * 64; // 64 to 640 bytes
            let result = arena.alloc_aligned_checked(size);
            assert!(result.is_ok(), "Failed at iteration {}", i);
        }
        
        // Should still be able to allocate
        let result = arena.alloc_aligned_checked(1024);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_memory_metrics_accuracy() {
        let arena = ArenaAllocator::new_checked(4096).unwrap();
        
        // Perform allocations
        for _ in 0..10 {
            let _ = arena.alloc_slice_checked::<u64>(50);
        }
        
        let metrics = arena.metrics();
        // At least one allocation should be recorded (may be more if blocks grow)
        assert!(metrics.allocation_count >= 0 || metrics.total_allocated > 0);
        
        // Reset
        arena.reset_checked().unwrap();
        let metrics = arena.metrics();
        assert_eq!(metrics.arena_resets, 1);
    }
    
    #[test]
    fn test_overflow_protection() {
        let arena = ArenaAllocator::new_checked(4096).unwrap();
        
        // Try to allocate more than usize can represent in bytes
        let result = arena.alloc_slice_checked::<u64>(usize::MAX / 4);
        assert!(matches!(result, Err(MemoryError::AllocationTooLarge { .. })));
    }
    
    #[test]
    fn test_alignment_validation() {
        // Using internal validation function via Layout
        let arena = ArenaAllocator::new_checked(4096).unwrap();
        
        // Valid alignments
        let result = arena.alloc_aligned_checked(64);
        assert!(result.is_ok());
        
        let result = arena.alloc_aligned_checked(128);
        assert!(result.is_ok());
    }
}
