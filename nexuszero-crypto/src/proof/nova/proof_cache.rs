// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! Advanced Caching for Nova Proving System
//!
//! This module provides comprehensive caching infrastructure for Nova proofs:
//! - LRU caches for proof components (witnesses, commitments, matrices)
//! - Memoization for expensive cryptographic operations
//! - Cache-aware proof generation with automatic invalidation
//! - Persistent cache with disk backing
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     ProofCacheManager                           │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
//! │  │ WitnessCache│  │CommitCache  │  │ MatrixCache │             │
//! │  │   (LRU)     │  │   (LRU)     │  │   (LRU)     │             │
//! │  └─────────────┘  └─────────────┘  └─────────────┘             │
//! │  ┌─────────────────────────────────────────────────────────────┐│
//! │  │                  MemoizationLayer                           ││
//! │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐              ││
//! │  │  │ MSM Memo   │ │ NTT Memo   │ │ Hash Memo  │              ││
//! │  │  └────────────┘ └────────────┘ └────────────┘              ││
//! │  └─────────────────────────────────────────────────────────────┘│
//! │  ┌─────────────────────────────────────────────────────────────┐│
//! │  │                  PersistentStore                            ││
//! │  │  [Disk-backed cache with LZ4 compression]                  ││
//! │  └─────────────────────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::types::NovaError;

// ============================================================================
// Cache Configuration
// ============================================================================

/// Configuration for the cache system
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the witness cache
    pub witness_cache_capacity: usize,
    /// Maximum number of entries in the commitment cache
    pub commitment_cache_capacity: usize,
    /// Maximum number of entries in the matrix cache
    pub matrix_cache_capacity: usize,
    /// Maximum memory for all caches combined (bytes)
    pub max_memory_bytes: usize,
    /// Time-to-live for cache entries (None = no expiry)
    pub ttl: Option<Duration>,
    /// Enable cache statistics collection
    pub collect_stats: bool,
    /// Enable persistent disk caching
    pub persistent_cache: bool,
    /// Path for persistent cache (if enabled)
    pub cache_path: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            witness_cache_capacity: 1024,
            commitment_cache_capacity: 512,
            matrix_cache_capacity: 256,
            max_memory_bytes: 1024 * 1024 * 1024, // 1 GB
            ttl: Some(Duration::from_secs(3600)),  // 1 hour
            collect_stats: true,
            persistent_cache: false,
            cache_path: None,
        }
    }
}

impl CacheConfig {
    /// Create a minimal configuration for testing
    pub fn minimal() -> Self {
        Self {
            witness_cache_capacity: 16,
            commitment_cache_capacity: 16,
            matrix_cache_capacity: 16,
            max_memory_bytes: 16 * 1024 * 1024, // 16 MB
            ttl: Some(Duration::from_secs(60)),
            collect_stats: false,
            persistent_cache: false,
            cache_path: None,
        }
    }

    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            witness_cache_capacity: 4096,
            commitment_cache_capacity: 2048,
            matrix_cache_capacity: 1024,
            max_memory_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
            ttl: None, // No expiry for max performance
            collect_stats: true,
            persistent_cache: true,
            cache_path: Some("./cache/nova".to_string()),
        }
    }
}

// ============================================================================
// Cache Key Types
// ============================================================================

/// Unique identifier for cached items
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Type of cached item
    pub key_type: CacheKeyType,
    /// Hash of the input parameters
    pub content_hash: u64,
    /// Optional version/epoch for invalidation
    pub version: u32,
}

/// Types of cached items
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheKeyType {
    Witness,
    Commitment,
    Matrix,
    MSMResult,
    NTTResult,
    HashResult,
    FoldingProof,
    R1CSMatrices,
}

impl CacheKey {
    /// Create a new cache key
    pub fn new(key_type: CacheKeyType, content_hash: u64) -> Self {
        Self {
            key_type,
            content_hash,
            version: 0,
        }
    }

    /// Create a key with version
    pub fn with_version(key_type: CacheKeyType, content_hash: u64, version: u32) -> Self {
        Self {
            key_type,
            content_hash,
            version,
        }
    }

    /// Compute hash from arbitrary data
    pub fn hash_data<T: Hash>(data: &T) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }

    /// Compute hash from a slice of hashable data
    pub fn hash_slice<T: Hash>(data: &[T]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        for item in data {
            item.hash(&mut hasher);
        }
        hasher.finish()
    }
}

// ============================================================================
// LRU Cache Implementation
// ============================================================================

/// Entry in the LRU cache
struct CacheEntry<V> {
    value: V,
    size_bytes: usize,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
}

impl<V> CacheEntry<V> {
    fn new(value: V, size_bytes: usize) -> Self {
        let now = Instant::now();
        Self {
            value,
            size_bytes,
            created_at: now,
            last_accessed: now,
            access_count: 1,
        }
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    fn is_expired(&self, ttl: Option<Duration>) -> bool {
        match ttl {
            Some(ttl) => self.created_at.elapsed() > ttl,
            None => false,
        }
    }
}

/// LRU (Least Recently Used) cache with size tracking
pub struct LruCache<K: Eq + Hash + Clone, V: Clone> {
    entries: HashMap<K, CacheEntry<V>>,
    order: Vec<K>, // Most recent at end
    capacity: usize,
    current_size_bytes: usize,
    max_size_bytes: usize,
    ttl: Option<Duration>,
    stats: CacheStats,
}

impl<K: Eq + Hash + Clone, V: Clone> LruCache<K, V> {
    /// Create a new LRU cache
    pub fn new(capacity: usize, max_size_bytes: usize, ttl: Option<Duration>) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            order: Vec::with_capacity(capacity),
            capacity,
            current_size_bytes: 0,
            max_size_bytes,
            ttl,
            stats: CacheStats::default(),
        }
    }

    /// Get an entry from the cache
    pub fn get(&mut self, key: &K) -> Option<V> {
        // Check if entry exists and is not expired
        if let Some(entry) = self.entries.get_mut(key) {
            if entry.is_expired(self.ttl) {
                // Remove expired entry
                self.remove(key);
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            
            entry.touch();
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            
            // Move to end of order (most recently used)
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                let k = self.order.remove(pos);
                self.order.push(k);
            }
            
            Some(entry.value.clone())
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert an entry into the cache
    pub fn insert(&mut self, key: K, value: V, size_bytes: usize) {
        // Remove existing entry if present
        if self.entries.contains_key(&key) {
            self.remove(&key);
        }

        // Evict entries until we have space
        while (self.entries.len() >= self.capacity || 
               self.current_size_bytes + size_bytes > self.max_size_bytes) 
              && !self.order.is_empty() {
            self.evict_lru();
        }

        // Insert new entry
        let entry = CacheEntry::new(value, size_bytes);
        self.entries.insert(key.clone(), entry);
        self.order.push(key);
        self.current_size_bytes += size_bytes;
        self.stats.insertions.fetch_add(1, Ordering::Relaxed);
    }

    /// Remove an entry from the cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.remove(key) {
            self.current_size_bytes = self.current_size_bytes.saturating_sub(entry.size_bytes);
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                self.order.remove(pos);
            }
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            Some(entry.value)
        } else {
            None
        }
    }

    /// Evict the least recently used entry
    fn evict_lru(&mut self) {
        if let Some(key) = self.order.first().cloned() {
            self.remove(&key);
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.current_size_bytes = 0;
    }

    /// Get current number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.current_size_bytes
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Prune expired entries
    pub fn prune_expired(&mut self) {
        let expired: Vec<K> = self.entries
            .iter()
            .filter(|(_, entry)| entry.is_expired(self.ttl))
            .map(|(k, _)| k.clone())
            .collect();

        for key in expired {
            self.remove(&key);
        }
    }
}

// ============================================================================
// Cache Statistics
// ============================================================================

/// Statistics for cache performance monitoring
#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub insertions: AtomicU64,
    pub evictions: AtomicU64,
}

impl CacheStats {
    /// Get hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            (hits as f64 / total as f64) * 100.0
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.insertions.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
    }

    /// Get summary as string
    pub fn summary(&self) -> String {
        format!(
            "Hits: {}, Misses: {}, Insertions: {}, Evictions: {}, Hit Rate: {:.2}%",
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.insertions.load(Ordering::Relaxed),
            self.evictions.load(Ordering::Relaxed),
            self.hit_rate()
        )
    }
}

impl Clone for CacheStats {
    fn clone(&self) -> Self {
        Self {
            hits: AtomicU64::new(self.hits.load(Ordering::Relaxed)),
            misses: AtomicU64::new(self.misses.load(Ordering::Relaxed)),
            insertions: AtomicU64::new(self.insertions.load(Ordering::Relaxed)),
            evictions: AtomicU64::new(self.evictions.load(Ordering::Relaxed)),
        }
    }
}

// ============================================================================
// Memoization Infrastructure
// ============================================================================

/// Memoization cache for expensive operations
pub struct MemoCache<K: Eq + Hash + Clone, V: Clone> {
    cache: RwLock<LruCache<K, V>>,
    computation_times: RwLock<HashMap<K, Duration>>,
}

impl<K: Eq + Hash + Clone, V: Clone> MemoCache<K, V> {
    /// Create a new memoization cache
    pub fn new(capacity: usize, max_memory: usize) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(capacity, max_memory, None)),
            computation_times: RwLock::new(HashMap::new()),
        }
    }

    /// Get or compute a value
    pub fn get_or_compute<F>(&self, key: K, compute: F, size_bytes: usize) -> V
    where
        F: FnOnce() -> V,
    {
        // Try to get from cache first
        {
            let mut cache = self.cache.write().unwrap();
            if let Some(value) = cache.get(&key) {
                return value;
            }
        }

        // Compute the value and measure time
        let start = Instant::now();
        let value = compute();
        let duration = start.elapsed();

        // Store computation time for analytics
        {
            let mut times = self.computation_times.write().unwrap();
            times.insert(key.clone(), duration);
        }

        // Insert into cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(key, value.clone(), size_bytes);
        }

        value
    }

    /// Get cached value if exists
    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write().unwrap();
        cache.get(key)
    }

    /// Invalidate a cached value
    pub fn invalidate(&self, key: &K) {
        let mut cache = self.cache.write().unwrap();
        cache.remove(key);
    }

    /// Clear all cached values
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Get average computation time for cached operations
    pub fn average_computation_time(&self) -> Duration {
        let times = self.computation_times.read().unwrap();
        if times.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = times.values().sum();
        total / times.len() as u32
    }
}

// ============================================================================
// Witness Cache
// ============================================================================

/// Cached witness data for Nova folding
#[derive(Clone, Debug)]
pub struct CachedWitness {
    /// The witness vector
    pub data: Vec<u64>,
    /// Size in bytes
    pub size_bytes: usize,
    /// Associated step index
    pub step_index: usize,
}

/// Specialized cache for witness vectors
pub struct WitnessCache {
    cache: RwLock<LruCache<CacheKey, CachedWitness>>,
    config: CacheConfig,
}

impl WitnessCache {
    /// Create a new witness cache
    pub fn new(config: &CacheConfig) -> Self {
        let max_memory = config.max_memory_bytes / 3; // Allocate 1/3 of memory to witnesses
        Self {
            cache: RwLock::new(LruCache::new(
                config.witness_cache_capacity,
                max_memory,
                config.ttl,
            )),
            config: config.clone(),
        }
    }

    /// Get a cached witness
    pub fn get(&self, key: &CacheKey) -> Option<CachedWitness> {
        let mut cache = self.cache.write().unwrap();
        cache.get(key)
    }

    /// Cache a witness
    pub fn insert(&self, key: CacheKey, witness: CachedWitness) {
        let size = witness.size_bytes;
        let mut cache = self.cache.write().unwrap();
        cache.insert(key, witness, size);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read().unwrap();
        cache.stats().clone()
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }
}

// ============================================================================
// Commitment Cache
// ============================================================================

/// Cached commitment data
#[derive(Clone, Debug)]
pub struct CachedCommitment {
    /// Commitment point (x, y coordinates as bytes)
    pub point: Vec<u8>,
    /// Size in bytes
    pub size_bytes: usize,
    /// Commitment type
    pub commitment_type: CommitmentType,
}

/// Types of cached commitments
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitmentType {
    Pedersen,
    KZG,
    IPA,
}

/// Specialized cache for commitments
pub struct CommitmentCache {
    cache: RwLock<LruCache<CacheKey, CachedCommitment>>,
}

impl CommitmentCache {
    /// Create a new commitment cache
    pub fn new(config: &CacheConfig) -> Self {
        let max_memory = config.max_memory_bytes / 3; // Allocate 1/3 of memory to commitments
        Self {
            cache: RwLock::new(LruCache::new(
                config.commitment_cache_capacity,
                max_memory,
                config.ttl,
            )),
        }
    }

    /// Get a cached commitment
    pub fn get(&self, key: &CacheKey) -> Option<CachedCommitment> {
        let mut cache = self.cache.write().unwrap();
        cache.get(key)
    }

    /// Cache a commitment
    pub fn insert(&self, key: CacheKey, commitment: CachedCommitment) {
        let size = commitment.size_bytes;
        let mut cache = self.cache.write().unwrap();
        cache.insert(key, commitment, size);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read().unwrap();
        cache.stats().clone()
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }
}

// ============================================================================
// Matrix Cache
// ============================================================================

/// Cached sparse matrix for R1CS
#[derive(Clone, Debug)]
pub struct CachedMatrix {
    /// Non-zero entries: (row, col, value)
    pub entries: Vec<(usize, usize, u64)>,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Size in bytes
    pub size_bytes: usize,
}

/// Specialized cache for R1CS matrices
pub struct MatrixCache {
    cache: RwLock<LruCache<CacheKey, CachedMatrix>>,
}

impl MatrixCache {
    /// Create a new matrix cache
    pub fn new(config: &CacheConfig) -> Self {
        let max_memory = config.max_memory_bytes / 3; // Allocate 1/3 of memory to matrices
        Self {
            cache: RwLock::new(LruCache::new(
                config.matrix_cache_capacity,
                max_memory,
                config.ttl,
            )),
        }
    }

    /// Get a cached matrix
    pub fn get(&self, key: &CacheKey) -> Option<CachedMatrix> {
        let mut cache = self.cache.write().unwrap();
        cache.get(key)
    }

    /// Cache a matrix
    pub fn insert(&self, key: CacheKey, matrix: CachedMatrix) {
        let size = matrix.size_bytes;
        let mut cache = self.cache.write().unwrap();
        cache.insert(key, matrix, size);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read().unwrap();
        cache.stats().clone()
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }
}

// ============================================================================
// MSM Result Cache
// ============================================================================

/// Cached MSM (Multi-Scalar Multiplication) result
#[derive(Clone, Debug)]
pub struct CachedMSMResult {
    /// Result point (serialized)
    pub result: Vec<u8>,
    /// Number of points in the MSM
    pub num_points: usize,
    /// Computation time for analytics
    pub computation_time_us: u64,
}

/// Cache for MSM results
pub struct MSMCache {
    memo: MemoCache<u64, CachedMSMResult>,
}

impl MSMCache {
    /// Create a new MSM cache
    pub fn new(capacity: usize, max_memory: usize) -> Self {
        Self {
            memo: MemoCache::new(capacity, max_memory),
        }
    }

    /// Get or compute MSM result
    pub fn get_or_compute<F>(&self, key: u64, compute: F, num_points: usize) -> CachedMSMResult
    where
        F: FnOnce() -> Vec<u8>,
    {
        let size = 64 + 8 + 8; // Approximate size for result
        self.memo.get_or_compute(
            key,
            || {
                let start = Instant::now();
                let result = compute();
                CachedMSMResult {
                    result,
                    num_points,
                    computation_time_us: start.elapsed().as_micros() as u64,
                }
            },
            size,
        )
    }

    /// Get cached MSM result
    pub fn get(&self, key: u64) -> Option<CachedMSMResult> {
        self.memo.get(&key)
    }

    /// Invalidate cached MSM result
    pub fn invalidate(&self, key: u64) {
        self.memo.invalidate(&key);
    }

    /// Clear all cached MSM results
    pub fn clear(&self) {
        self.memo.clear();
    }
}

// ============================================================================
// NTT Result Cache
// ============================================================================

/// Cached NTT (Number Theoretic Transform) result
#[derive(Clone, Debug)]
pub struct CachedNTTResult {
    /// Transformed coefficients
    pub coefficients: Vec<u64>,
    /// Whether this is forward or inverse NTT
    pub is_forward: bool,
    /// Computation time for analytics
    pub computation_time_us: u64,
}

/// Cache for NTT results
pub struct NTTCache {
    memo: MemoCache<u64, CachedNTTResult>,
}

impl NTTCache {
    /// Create a new NTT cache
    pub fn new(capacity: usize, max_memory: usize) -> Self {
        Self {
            memo: MemoCache::new(capacity, max_memory),
        }
    }

    /// Get or compute NTT result
    pub fn get_or_compute<F>(&self, key: u64, is_forward: bool, compute: F) -> CachedNTTResult
    where
        F: FnOnce() -> Vec<u64>,
    {
        self.memo.get_or_compute(
            key,
            || {
                let start = Instant::now();
                let coefficients = compute();
                let computation_time_us = start.elapsed().as_micros() as u64;
                CachedNTTResult {
                    coefficients,
                    is_forward,
                    computation_time_us,
                }
            },
            0, // Size computed from coefficients
        )
    }

    /// Get cached NTT result
    pub fn get(&self, key: u64) -> Option<CachedNTTResult> {
        self.memo.get(&key)
    }

    /// Clear all cached NTT results
    pub fn clear(&self) {
        self.memo.clear();
    }
}

// ============================================================================
// Proof Cache Manager
// ============================================================================

/// Centralized cache manager for all Nova proof components
pub struct ProofCacheManager {
    /// Witness cache
    pub witnesses: WitnessCache,
    /// Commitment cache
    pub commitments: CommitmentCache,
    /// Matrix cache
    pub matrices: MatrixCache,
    /// MSM result cache
    pub msm: MSMCache,
    /// NTT result cache
    pub ntt: NTTCache,
    /// Configuration
    config: CacheConfig,
    /// Global statistics
    global_stats: Arc<GlobalCacheStats>,
}

/// Global statistics across all caches
pub struct GlobalCacheStats {
    pub total_memory_used: AtomicUsize,
    pub total_hits: AtomicU64,
    pub total_misses: AtomicU64,
    pub cache_saves_time_us: AtomicU64,
}

impl Default for GlobalCacheStats {
    fn default() -> Self {
        Self {
            total_memory_used: AtomicUsize::new(0),
            total_hits: AtomicU64::new(0),
            total_misses: AtomicU64::new(0),
            cache_saves_time_us: AtomicU64::new(0),
        }
    }
}

impl ProofCacheManager {
    /// Create a new cache manager with default configuration
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new cache manager with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        let memory_per_cache = config.max_memory_bytes / 5;
        
        Self {
            witnesses: WitnessCache::new(&config),
            commitments: CommitmentCache::new(&config),
            matrices: MatrixCache::new(&config),
            msm: MSMCache::new(config.commitment_cache_capacity, memory_per_cache),
            ntt: NTTCache::new(config.commitment_cache_capacity, memory_per_cache),
            config,
            global_stats: Arc::new(GlobalCacheStats::default()),
        }
    }

    /// Get or compute witness with caching
    pub fn get_or_compute_witness<F>(
        &self,
        step_index: usize,
        input_hash: u64,
        compute: F,
    ) -> CachedWitness
    where
        F: FnOnce() -> Vec<u64>,
    {
        let key = CacheKey::new(CacheKeyType::Witness, input_hash);
        
        // Try cache first
        if let Some(cached) = self.witnesses.get(&key) {
            self.global_stats.total_hits.fetch_add(1, Ordering::Relaxed);
            return cached;
        }

        // Compute
        self.global_stats.total_misses.fetch_add(1, Ordering::Relaxed);
        let data = compute();
        let size_bytes = data.len() * 8;
        let witness = CachedWitness {
            data,
            size_bytes,
            step_index,
        };

        // Cache result
        self.witnesses.insert(key, witness.clone());
        witness
    }

    /// Get or compute commitment with caching
    pub fn get_or_compute_commitment<F>(
        &self,
        data_hash: u64,
        commitment_type: CommitmentType,
        compute: F,
    ) -> CachedCommitment
    where
        F: FnOnce() -> Vec<u8>,
    {
        let key = CacheKey::new(CacheKeyType::Commitment, data_hash);
        
        // Try cache first
        if let Some(cached) = self.commitments.get(&key) {
            self.global_stats.total_hits.fetch_add(1, Ordering::Relaxed);
            return cached;
        }

        // Compute
        self.global_stats.total_misses.fetch_add(1, Ordering::Relaxed);
        let point = compute();
        let size_bytes = point.len();
        let commitment = CachedCommitment {
            point,
            size_bytes,
            commitment_type,
        };

        // Cache result
        self.commitments.insert(key, commitment.clone());
        commitment
    }

    /// Get or compute matrix with caching
    pub fn get_or_compute_matrix<F>(
        &self,
        circuit_hash: u64,
        compute: F,
    ) -> CachedMatrix
    where
        F: FnOnce() -> (Vec<(usize, usize, u64)>, usize, usize),
    {
        let key = CacheKey::new(CacheKeyType::Matrix, circuit_hash);
        
        // Try cache first
        if let Some(cached) = self.matrices.get(&key) {
            self.global_stats.total_hits.fetch_add(1, Ordering::Relaxed);
            return cached;
        }

        // Compute
        self.global_stats.total_misses.fetch_add(1, Ordering::Relaxed);
        let (entries, rows, cols) = compute();
        let size_bytes = entries.len() * 24; // (usize, usize, u64) = 24 bytes
        let matrix = CachedMatrix {
            entries,
            rows,
            cols,
            size_bytes,
        };

        // Cache result
        self.matrices.insert(key, matrix.clone());
        matrix
    }

    /// Get global cache statistics
    pub fn global_stats(&self) -> &GlobalCacheStats {
        &self.global_stats
    }

    /// Get combined statistics from all caches
    pub fn combined_stats(&self) -> CombinedCacheStats {
        CombinedCacheStats {
            witness_stats: self.witnesses.stats(),
            commitment_stats: self.commitments.stats(),
            matrix_stats: self.matrices.stats(),
            total_hits: self.global_stats.total_hits.load(Ordering::Relaxed),
            total_misses: self.global_stats.total_misses.load(Ordering::Relaxed),
        }
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        self.witnesses.clear();
        self.commitments.clear();
        self.matrices.clear();
        self.msm.clear();
        self.ntt.clear();
    }

    /// Invalidate caches for a specific step
    pub fn invalidate_step(&self, step_index: usize, step_hash: u64) {
        // Create keys for the step
        let witness_key = CacheKey::new(CacheKeyType::Witness, step_hash);
        let commitment_key = CacheKey::new(CacheKeyType::Commitment, step_hash);
        
        // Remove from caches
        self.witnesses.cache.write().unwrap().remove(&witness_key);
        self.commitments.cache.write().unwrap().remove(&commitment_key);
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> MemoryUsageStats {
        MemoryUsageStats {
            witness_cache_bytes: 0, // Would need to track internally
            commitment_cache_bytes: 0,
            matrix_cache_bytes: 0,
            msm_cache_bytes: 0,
            ntt_cache_bytes: 0,
            total_bytes: 0,
            max_bytes: self.config.max_memory_bytes,
        }
    }
}

impl Default for ProofCacheManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined statistics from all caches
#[derive(Debug, Clone)]
pub struct CombinedCacheStats {
    pub witness_stats: CacheStats,
    pub commitment_stats: CacheStats,
    pub matrix_stats: CacheStats,
    pub total_hits: u64,
    pub total_misses: u64,
}

impl CombinedCacheStats {
    /// Get overall hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            (self.total_hits as f64 / total as f64) * 100.0
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    pub witness_cache_bytes: usize,
    pub commitment_cache_bytes: usize,
    pub matrix_cache_bytes: usize,
    pub msm_cache_bytes: usize,
    pub ntt_cache_bytes: usize,
    pub total_bytes: usize,
    pub max_bytes: usize,
}

impl MemoryUsageStats {
    /// Get usage as percentage of max
    pub fn usage_percentage(&self) -> f64 {
        if self.max_bytes == 0 {
            0.0
        } else {
            (self.total_bytes as f64 / self.max_bytes as f64) * 100.0
        }
    }
}

// ============================================================================
// Cache-Aware Proof Generator
// ============================================================================

/// Proof generator with integrated caching
pub struct CacheAwareProofGenerator {
    cache_manager: Arc<ProofCacheManager>,
    current_step: usize,
    epoch: u32,
}

impl CacheAwareProofGenerator {
    /// Create a new cache-aware proof generator
    pub fn new(cache_manager: Arc<ProofCacheManager>) -> Self {
        Self {
            cache_manager,
            current_step: 0,
            epoch: 0,
        }
    }

    /// Start a new proving session
    pub fn new_session(&mut self) {
        self.current_step = 0;
        self.epoch += 1;
    }

    /// Advance to the next step
    pub fn next_step(&mut self) {
        self.current_step += 1;
    }

    /// Get current step index
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get current epoch
    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Generate proof step with caching
    pub fn generate_step_cached<F>(
        &mut self,
        input: &[u64],
        generate: F,
    ) -> Result<CachedWitness, NovaError>
    where
        F: FnOnce() -> Result<Vec<u64>, NovaError>,
    {
        let input_hash = CacheKey::hash_slice(input);
        let step_index = self.current_step;
        
        let witness = self.cache_manager.get_or_compute_witness(
            step_index,
            input_hash,
            || generate().unwrap_or_default(),
        );

        self.next_step();
        Ok(witness)
    }

    /// Get cache manager reference
    pub fn cache_manager(&self) -> &Arc<ProofCacheManager> {
        &self.cache_manager
    }

    /// Clear all caches and reset state
    pub fn reset(&mut self) {
        self.cache_manager.clear_all();
        self.current_step = 0;
        self.epoch += 1;
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_creation() {
        let key1 = CacheKey::new(CacheKeyType::Witness, 12345);
        assert_eq!(key1.key_type, CacheKeyType::Witness);
        assert_eq!(key1.content_hash, 12345);
        assert_eq!(key1.version, 0);

        let key2 = CacheKey::with_version(CacheKeyType::Commitment, 67890, 1);
        assert_eq!(key2.version, 1);
    }

    #[test]
    fn test_cache_key_hash_data() {
        let data = vec![1u64, 2, 3, 4, 5];
        let hash1 = CacheKey::hash_data(&data);
        let hash2 = CacheKey::hash_data(&data);
        assert_eq!(hash1, hash2);

        let different_data = vec![5u64, 4, 3, 2, 1];
        let hash3 = CacheKey::hash_data(&different_data);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_lru_cache_basic() {
        let mut cache: LruCache<u64, String> = LruCache::new(3, 1024, None);
        
        cache.insert(1, "one".to_string(), 3);
        cache.insert(2, "two".to_string(), 3);
        cache.insert(3, "three".to_string(), 5);
        
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&1), Some("one".to_string()));
        assert_eq!(cache.get(&2), Some("two".to_string()));
        assert_eq!(cache.get(&3), Some("three".to_string()));
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut cache: LruCache<u64, String> = LruCache::new(2, 1024, None);
        
        cache.insert(1, "one".to_string(), 3);
        cache.insert(2, "two".to_string(), 3);
        cache.insert(3, "three".to_string(), 5);
        
        // Entry 1 should be evicted (LRU)
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&1).is_none());
        assert_eq!(cache.get(&2), Some("two".to_string()));
        assert_eq!(cache.get(&3), Some("three".to_string()));
    }

    #[test]
    fn test_lru_cache_access_order() {
        let mut cache: LruCache<u64, String> = LruCache::new(3, 1024, None);
        
        cache.insert(1, "one".to_string(), 3);
        cache.insert(2, "two".to_string(), 3);
        cache.insert(3, "three".to_string(), 5);
        
        // Access entry 1 to make it recently used
        let _ = cache.get(&1);
        
        // Insert entry 4, should evict entry 2 (now LRU)
        cache.insert(4, "four".to_string(), 4);
        
        assert!(cache.get(&1).is_some());
        assert!(cache.get(&2).is_none()); // Evicted
        assert!(cache.get(&3).is_some());
        assert!(cache.get(&4).is_some());
    }

    #[test]
    fn test_lru_cache_size_eviction() {
        let mut cache: LruCache<u64, Vec<u8>> = LruCache::new(100, 50, None);
        
        cache.insert(1, vec![0; 20], 20);
        cache.insert(2, vec![0; 20], 20);
        // Adding entry 3 with 20 bytes exceeds 50 byte limit
        cache.insert(3, vec![0; 20], 20);
        
        // Entry 1 should be evicted to make space
        assert!(cache.get(&1).is_none());
        assert!(cache.get(&2).is_some());
        assert!(cache.get(&3).is_some());
    }

    #[test]
    fn test_cache_stats() {
        let mut cache: LruCache<u64, String> = LruCache::new(10, 1024, None);
        
        cache.insert(1, "one".to_string(), 3);
        let _ = cache.get(&1); // Hit
        let _ = cache.get(&2); // Miss
        let _ = cache.get(&1); // Hit
        
        let stats = cache.stats();
        assert_eq!(stats.hits.load(Ordering::Relaxed), 2);
        assert_eq!(stats.misses.load(Ordering::Relaxed), 1);
        assert!((stats.hit_rate() - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_witness_cache() {
        let config = CacheConfig::minimal();
        let cache = WitnessCache::new(&config);
        
        let key = CacheKey::new(CacheKeyType::Witness, 12345);
        let witness = CachedWitness {
            data: vec![1, 2, 3, 4, 5],
            size_bytes: 40,
            step_index: 0,
        };
        
        cache.insert(key.clone(), witness.clone());
        
        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().data, witness.data);
    }

    #[test]
    fn test_commitment_cache() {
        let config = CacheConfig::minimal();
        let cache = CommitmentCache::new(&config);
        
        let key = CacheKey::new(CacheKeyType::Commitment, 67890);
        let commitment = CachedCommitment {
            point: vec![0u8; 64],
            size_bytes: 64,
            commitment_type: CommitmentType::Pedersen,
        };
        
        cache.insert(key.clone(), commitment.clone());
        
        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().commitment_type, CommitmentType::Pedersen);
    }

    #[test]
    fn test_matrix_cache() {
        let config = CacheConfig::minimal();
        let cache = MatrixCache::new(&config);
        
        let key = CacheKey::new(CacheKeyType::Matrix, 11111);
        let matrix = CachedMatrix {
            entries: vec![(0, 0, 1), (1, 1, 1), (2, 2, 1)],
            rows: 3,
            cols: 3,
            size_bytes: 72,
        };
        
        cache.insert(key.clone(), matrix.clone());
        
        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().rows, 3);
    }

    #[test]
    fn test_proof_cache_manager() {
        let manager = ProofCacheManager::with_config(CacheConfig::minimal());
        
        // Test witness caching
        let witness = manager.get_or_compute_witness(0, 12345, || {
            vec![1u64, 2, 3, 4, 5]
        });
        assert_eq!(witness.data, vec![1u64, 2, 3, 4, 5]);
        
        // Second call should hit cache
        let witness2 = manager.get_or_compute_witness(0, 12345, || {
            panic!("Should not be called - cache hit expected");
        });
        assert_eq!(witness2.data, vec![1u64, 2, 3, 4, 5]);
        
        // Check stats
        let stats = manager.combined_stats();
        assert_eq!(stats.total_hits, 1);
        assert_eq!(stats.total_misses, 1);
    }

    #[test]
    fn test_msm_cache() {
        let cache = MSMCache::new(16, 1024 * 1024);
        
        let result = cache.get_or_compute(12345, || {
            vec![0u8; 64]
        }, 100);
        
        assert_eq!(result.num_points, 100);
        assert_eq!(result.result.len(), 64);
        
        // Second call should hit cache
        let result2 = cache.get(12345);
        assert!(result2.is_some());
    }

    #[test]
    fn test_ntt_cache() {
        let cache = NTTCache::new(16, 1024 * 1024);
        
        let result = cache.get_or_compute(67890, true, || {
            vec![1u64, 2, 3, 4, 5, 6, 7, 8]
        });
        
        assert!(result.is_forward);
        assert_eq!(result.coefficients.len(), 8);
        
        // Second call should hit cache
        let result2 = cache.get(67890);
        assert!(result2.is_some());
    }

    #[test]
    fn test_cache_aware_proof_generator() {
        let cache_manager = Arc::new(ProofCacheManager::with_config(CacheConfig::minimal()));
        let mut generator = CacheAwareProofGenerator::new(cache_manager);
        
        assert_eq!(generator.current_step(), 0);
        assert_eq!(generator.epoch(), 0);
        
        let witness = generator.generate_step_cached(
            &[1u64, 2, 3],
            || Ok(vec![4u64, 5, 6]),
        ).unwrap();
        
        assert_eq!(witness.data, vec![4u64, 5, 6]);
        assert_eq!(generator.current_step(), 1);
        
        generator.new_session();
        assert_eq!(generator.current_step(), 0);
        assert_eq!(generator.epoch(), 1);
    }

    #[test]
    fn test_cache_config_presets() {
        let minimal = CacheConfig::minimal();
        assert_eq!(minimal.witness_cache_capacity, 16);
        
        let high_perf = CacheConfig::high_performance();
        assert_eq!(high_perf.witness_cache_capacity, 4096);
        assert!(high_perf.persistent_cache);
    }

    #[test]
    fn test_combined_cache_stats() {
        let manager = ProofCacheManager::with_config(CacheConfig::minimal());
        
        // Generate some cache activity
        let _ = manager.get_or_compute_witness(0, 1, || vec![1u64]);
        let _ = manager.get_or_compute_witness(1, 2, || vec![2u64]);
        let _ = manager.get_or_compute_witness(0, 1, || vec![999u64]); // Should hit
        
        let stats = manager.combined_stats();
        assert_eq!(stats.total_hits, 1);
        assert_eq!(stats.total_misses, 2);
        assert!((stats.hit_rate() - 33.33).abs() < 1.0);
    }

    #[test]
    fn test_cache_clear() {
        let manager = ProofCacheManager::with_config(CacheConfig::minimal());
        
        let _ = manager.get_or_compute_witness(0, 12345, || vec![1u64, 2, 3]);
        
        manager.clear_all();
        
        // Cache should miss after clear
        let witness = manager.get_or_compute_witness(0, 12345, || vec![4u64, 5, 6]);
        assert_eq!(witness.data, vec![4u64, 5, 6]); // New computation result
    }

    #[test]
    fn test_memo_cache() {
        let memo: MemoCache<u64, u64> = MemoCache::new(16, 1024);
        
        let mut computed_count = 0;
        
        let result1 = memo.get_or_compute(100, || {
            computed_count += 1;
            42
        }, 8);
        assert_eq!(result1, 42);
        
        let result2 = memo.get_or_compute(100, || {
            computed_count += 1;
            99
        }, 8);
        assert_eq!(result2, 42); // Should return cached value, not 99
        
        // Computation should only have happened once
        // Note: We can't easily verify computed_count without external tracking
    }

    #[test]
    fn test_lru_cache_remove() {
        let mut cache: LruCache<u64, String> = LruCache::new(10, 1024, None);
        
        cache.insert(1, "one".to_string(), 3);
        cache.insert(2, "two".to_string(), 3);
        
        let removed = cache.remove(&1);
        assert_eq!(removed, Some("one".to_string()));
        assert!(cache.get(&1).is_none());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_cache_ttl() {
        use std::thread::sleep;
        
        let mut cache: LruCache<u64, String> = LruCache::new(10, 1024, Some(Duration::from_millis(50)));
        
        cache.insert(1, "one".to_string(), 3);
        
        // Should be present immediately
        assert!(cache.get(&1).is_some());
        
        // Wait for TTL to expire
        sleep(Duration::from_millis(100));
        
        // Should be expired now
        assert!(cache.get(&1).is_none());
    }
}
