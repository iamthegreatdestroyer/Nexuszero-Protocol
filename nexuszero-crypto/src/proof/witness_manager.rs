//! Witness Manager for advanced witness lifecycle management
//!
//! This module provides a comprehensive WitnessManager trait and implementation
//! for witness creation, validation, caching, transformation, and lifecycle management.
//! Optimized for zero-copy operations and parallel witness generation.

use crate::proof::{Statement, Witness, WitnessType};
use crate::{CryptoError, CryptoResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use lru::LruCache;

/// Witness metadata for caching and management
#[derive(Clone, Debug)]
pub struct WitnessMetadata {
    /// Unique witness identifier
    pub id: String,
    /// Witness type
    pub witness_type: WitnessType,
    /// Creation timestamp
    pub created_at: Instant,
    /// Last access timestamp
    pub last_accessed: Instant,
    /// TTL duration
    pub ttl: Option<Duration>,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Validation status
    pub is_valid: bool,
    /// Associated statement hash for correlation
    pub statement_hash: Option<String>,
}

/// Cached witness entry with metadata
#[derive(Clone)]
pub struct CachedWitness {
    /// The actual witness
    pub witness: Arc<Witness>,
    /// Metadata for cache management
    pub metadata: WitnessMetadata,
}

/// Witness transformation result
#[derive(Clone, Debug)]
pub enum TransformationResult {
    /// Successful transformation
    Success(Arc<Witness>),
    /// Transformation failed
    Failed(String),
    /// No transformation needed
    NoOp,
}

/// Witness validation constraints
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationConstraints {
    /// Maximum value for range proofs
    pub max_range_value: Option<u64>,
    /// Minimum value for range proofs
    pub min_range_value: Option<u64>,
    /// Maximum preimage length
    pub max_preimage_length: Option<usize>,
    /// Required hash function for preimages
    pub required_hash_function: Option<crate::proof::statement::HashFunction>,
    /// Custom validation rules
    pub custom_constraints: HashMap<String, serde_json::Value>,
}

/// Witness generation configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WitnessGenerationConfig {
    /// Use parallel generation when possible
    pub parallel_generation: bool,
    /// Maximum number of parallel tasks
    pub max_parallel_tasks: usize,
    /// Enable zero-copy operations
    pub zero_copy: bool,
    /// Randomness source configuration
    pub randomness_config: RandomnessConfig,
    /// Validation constraints
    pub validation_constraints: ValidationConstraints,
}

/// Randomness configuration for witness generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RandomnessConfig {
    /// Randomness length in bytes
    pub length: usize,
    /// Use cryptographically secure randomness
    pub secure_random: bool,
    /// Custom entropy source
    pub custom_entropy: Option<Vec<u8>>,
}

/// Witness Manager trait for comprehensive witness lifecycle management
#[async_trait]
pub trait WitnessManager: Send + Sync {
    /// Create a witness from statement and secret data
    async fn create_witness(
        &self,
        statement: &Statement,
        secret_data: &[u8],
        config: &WitnessGenerationConfig,
    ) -> CryptoResult<Arc<Witness>>;

    /// Validate a witness against a statement with enhanced constraints
    async fn validate_witness(
        &self,
        witness: &Witness,
        statement: &Statement,
        constraints: &ValidationConstraints,
    ) -> CryptoResult<bool>;

    /// Transform witness between formats
    async fn transform_witness(
        &self,
        witness: Arc<Witness>,
        target_type: WitnessType,
        options: &HashMap<String, serde_json::Value>,
    ) -> CryptoResult<TransformationResult>;

    /// Cache a witness with TTL
    async fn cache_witness(
        &self,
        witness: Arc<Witness>,
        ttl: Option<Duration>,
        statement_hash: Option<String>,
    ) -> CryptoResult<String>;

    /// Retrieve cached witness by ID
    async fn get_cached_witness(&self, id: &str) -> CryptoResult<Option<Arc<Witness>>>;

    /// Remove witness from cache
    async fn remove_cached_witness(&self, id: &str) -> CryptoResult<bool>;

    /// Get cache statistics
    async fn cache_stats(&self) -> CryptoResult<CacheStats>;

    /// Clean expired witnesses from cache
    async fn cleanup_expired(&self) -> CryptoResult<usize>;

    /// Generate multiple witnesses in parallel
    async fn batch_create_witnesses(
        &self,
        statements: &[Statement],
        secret_data: &[Vec<u8>],
        config: &WitnessGenerationConfig,
    ) -> CryptoResult<Vec<Arc<Witness>>>;

    /// Batch validate witnesses
    async fn batch_validate_witnesses(
        &self,
        witnesses: &[Arc<Witness>],
        statements: &[Statement],
        constraints: &ValidationConstraints,
    ) -> CryptoResult<Vec<bool>>;
}

/// Cache statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total number of cached witnesses
    pub total_entries: usize,
    /// Number of valid (non-expired) entries
    pub valid_entries: usize,
    /// Total memory usage in bytes
    pub memory_usage: usize,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Average access time in nanoseconds
    pub avg_access_time_ns: u64,
}

/// Default WitnessManager implementation with LRU caching
pub struct DefaultWitnessManager {
    /// LRU cache for witnesses
    cache: RwLock<LruCache<String, CachedWitness>>,
    /// Cache statistics
    stats: RwLock<CacheStats>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Cache access times for performance tracking
    access_times: RwLock<Vec<u64>>,
}

impl DefaultWitnessManager {
    /// Create new DefaultWitnessManager with specified cache size
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(
                std::num::NonZeroUsize::new(max_cache_size).unwrap()
            )),
            stats: RwLock::new(CacheStats {
                total_entries: 0,
                valid_entries: 0,
                memory_usage: 0,
                hit_rate: 0.0,
                hits: 0,
                misses: 0,
                avg_access_time_ns: 0,
            }),
            max_cache_size,
            access_times: RwLock::new(Vec::new()),
        }
    }

    /// Get the maximum cache size
    pub fn max_cache_size(&self) -> usize {
        self.max_cache_size
    }

    /// Generate unique witness ID
    fn generate_witness_id(&self) -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        format!("witness_{}", rng.gen::<u64>())
    }

    /// Calculate witness memory usage
    fn calculate_memory_usage(witness: &Witness) -> usize {
        // Estimate memory usage based on witness data
        std::mem::size_of::<Witness>() +
        match witness.get_secret_bytes() {
            Ok(bytes) => bytes.len(),
            Err(_) => 0,
        } +
        witness.randomness().len()
    }

    /// Update cache statistics
    async fn update_stats(&self, access_time_ns: u64, was_hit: bool) {
        let mut stats = self.stats.write().await;
        let mut access_times = self.access_times.write().await;

        if was_hit {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }

        access_times.push(access_time_ns);
        if access_times.len() > 1000 {
            access_times.remove(0); // Keep only last 1000 measurements
        }

        stats.avg_access_time_ns = access_times.iter().sum::<u64>() / access_times.len() as u64;
        stats.hit_rate = if stats.hits + stats.misses > 0 {
            stats.hits as f64 / (stats.hits + stats.misses) as f64
        } else {
            0.0
        };
    }

    /// Update entry counts and memory usage
    async fn update_entry_counts(&self) {
        let mut stats = self.stats.write().await;
        let cache = self.cache.read().await;
        stats.total_entries = cache.len();
        stats.valid_entries = cache.iter()
            .filter(|(_, entry)| {
                if let Some(ttl) = entry.metadata.ttl {
                    entry.metadata.created_at.elapsed() < ttl
                } else {
                    true
                }
            })
            .count();
        stats.memory_usage = cache.iter()
            .map(|(_, entry)| entry.metadata.memory_usage)
            .sum();
    }
}

#[async_trait]
impl WitnessManager for DefaultWitnessManager {
    async fn create_witness(
        &self,
        statement: &Statement,
        secret_data: &[u8],
        config: &WitnessGenerationConfig,
    ) -> CryptoResult<Arc<Witness>> {
        // Validate constraints before creation
        self.validate_constraints(secret_data, config)?;

        let witness = match &statement.statement_type {
            crate::proof::statement::StatementType::DiscreteLog { .. } => {
                Witness::discrete_log(secret_data.to_vec())
            }
            crate::proof::statement::StatementType::Preimage { hash_function, .. } => {
                // Validate hash function constraint
                if let Some(required) = &config.validation_constraints.required_hash_function {
                    if hash_function != required {
                        return Err(CryptoError::ProofError(
                            format!("Hash function {:?} does not match required {:?}", hash_function, required)
                        ));
                    }
                }
                Witness::preimage(secret_data.to_vec())
            }
            crate::proof::statement::StatementType::Range {   .. } => {
                // Parse secret data as (value, blinding_factor)
                if secret_data.len() < 8 {
                    return Err(CryptoError::ProofError("Insufficient data for range witness".to_string()));
                }
                let value = u64::from_be_bytes(secret_data[..8].try_into().unwrap());
                let blinding = secret_data[8..].to_vec();

                // Validate range constraints
                if let Some(min_val) = config.validation_constraints.min_range_value {
                    if value < min_val {
                        return Err(CryptoError::ProofError(
                            format!("Value {} below minimum {}", value, min_val)
                        ));
                    }
                }
                if let Some(max_val) = config.validation_constraints.max_range_value {
                    if value > max_val {
                        return Err(CryptoError::ProofError(
                            format!("Value {} above maximum {}", value, max_val)
                        ));
                    }
                }

                Witness::range(value, blinding)
            }
            crate::proof::statement::StatementType::Custom { description: _ } => {
                // For custom types, store data as-is
                Witness::custom(secret_data.to_vec())
            }
        };

        Ok(Arc::new(witness))
    }

    async fn validate_witness(
        &self,
        witness: &Witness,
        statement: &Statement,
        constraints: &ValidationConstraints,
    ) -> CryptoResult<bool> {
        // Basic statement satisfaction check
        let basic_valid = witness.satisfies_statement(statement);

        if !basic_valid {
            return Ok(false);
        }

        // Enhanced constraint validation
        match (&witness.witness_type(), &statement.statement_type) {
            (WitnessType::Preimage, crate::proof::statement::StatementType::Preimage { .. }) => {
                if let Ok(secret_bytes) = witness.get_secret_bytes() {
                    if let Some(max_len) = constraints.max_preimage_length {
                        if secret_bytes.len() > max_len {
                            return Ok(false);
                        }
                    }
                }
            }
            (WitnessType::Range, crate::proof::statement::StatementType::Range { min, max, .. }) => {
                if let Ok(secret_bytes) = witness.get_secret_bytes() {
                    if secret_bytes.len() >= 8 {
                        let value = u64::from_be_bytes(secret_bytes[..8].try_into().unwrap());

                        if let Some(min_val) = constraints.min_range_value {
                            if value < min_val {
                                return Ok(false);
                            }
                        }
                        if let Some(max_val) = constraints.max_range_value {
                            if value > max_val {
                                return Ok(false);
                            }
                        }

                        // Check against statement range
                        if value < *min || value > *max {
                            return Ok(false);
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(true)
    }

    async fn transform_witness(
        &self,
        witness: Arc<Witness>,
        target_type: WitnessType,
        _options: &HashMap<String, serde_json::Value>,
    ) -> CryptoResult<TransformationResult> {
        // For now, only support identity transformations
        // Future: implement actual format conversions
        if witness.witness_type() == &target_type {
            Ok(TransformationResult::NoOp)
        } else {
            Ok(TransformationResult::Failed(
                format!("Transformation from {:?} to {:?} not implemented",
                       witness.witness_type(), target_type)
            ))
        }
    }

    async fn cache_witness(
        &self,
        witness: Arc<Witness>,
        ttl: Option<Duration>,
        statement_hash: Option<String>,
    ) -> CryptoResult<String> {
        let start_time = Instant::now();
        let id = self.generate_witness_id();
        let memory_usage = Self::calculate_memory_usage(&witness);

        let metadata = WitnessMetadata {
            id: id.clone(),
            witness_type: witness.witness_type().clone(),
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            ttl,
            memory_usage,
            is_valid: true,
            statement_hash,
        };

        let cached_witness = CachedWitness {
            witness: witness.clone(),
            metadata,
        };

        let mut cache = self.cache.write().await;
        cache.put(id.clone(), cached_witness);

        let access_time = start_time.elapsed().as_nanos() as u64;
        self.update_stats(access_time, true).await;

        Ok(id)
    }

    async fn get_cached_witness(&self, id: &str) -> CryptoResult<Option<Arc<Witness>>> {
        let start_time = Instant::now();
        let mut cache = self.cache.write().await;

        if let Some(entry) = cache.get_mut(id) {
            // Check if expired
            if let Some(ttl) = entry.metadata.ttl {
                if entry.metadata.created_at.elapsed() >= ttl {
                    cache.pop(id); // Remove expired entry
                    let access_time = start_time.elapsed().as_nanos() as u64;
                    self.update_stats(access_time, false).await;
                    return Ok(None);
                }
            }

            entry.metadata.last_accessed = Instant::now();
            let access_time = start_time.elapsed().as_nanos() as u64;
            self.update_stats(access_time, true).await;
            Ok(Some(entry.witness.clone()))
        } else {
            let access_time = start_time.elapsed().as_nanos() as u64;
            self.update_stats(access_time, false).await;
            Ok(None)
        }
    }

    async fn remove_cached_witness(&self, id: &str) -> CryptoResult<bool> {
        let mut cache = self.cache.write().await;
        Ok(cache.pop(id).is_some())
    }

    async fn cache_stats(&self) -> CryptoResult<CacheStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }

    async fn cleanup_expired(&self) -> CryptoResult<usize> {
        let mut cache = self.cache.write().await;
        let mut removed = 0;
        let mut to_remove = Vec::new();

        // Collect expired entries
        for (id, entry) in cache.iter() {
            if let Some(ttl) = entry.metadata.ttl {
                if entry.metadata.created_at.elapsed() >= ttl {
                    to_remove.push(id.clone());
                }
            }
        }

        // Remove expired entries
        for id in to_remove {
            cache.pop(&id);
            removed += 1;
        }

        Ok(removed)
    }

    async fn batch_create_witnesses(
        &self,
        statements: &[Statement],
        secret_data: &[Vec<u8>],
        config: &WitnessGenerationConfig,
    ) -> CryptoResult<Vec<Arc<Witness>>> {
        if statements.len() != secret_data.len() {
            return Err(CryptoError::ProofError(
                "Mismatched number of statements and secret data".to_string()
            ));
        }

        if config.parallel_generation {
            use rayon::prelude::*;
            let witnesses: Vec<CryptoResult<Witness>> = statements.par_iter()
                .zip(secret_data.par_iter())
                .map(|(stmt, secret)| {
                    // Create witness synchronously for parallel execution
                    match &stmt.statement_type {
                        crate::proof::statement::StatementType::DiscreteLog { .. } => {
                            Ok(Witness::discrete_log(secret.clone()))
                        }
                        crate::proof::statement::StatementType::Preimage { hash_function, .. } => {
                            if let Some(required) = &config.validation_constraints.required_hash_function {
                                if hash_function != required {
                                    return Err(CryptoError::ProofError(
                                        format!("Hash function {:?} does not match required {:?}", hash_function, required)
                                    ));
                                }
                            }
                            Ok(Witness::preimage(secret.clone()))
                        }
                        crate::proof::statement::StatementType::Range {   .. } => {
                            if secret.len() < 8 {
                                return Err(CryptoError::ProofError("Insufficient data for range witness".to_string()));
                            }
                            let value = u64::from_be_bytes(secret[..8].try_into().unwrap());
                            let blinding = secret[8..].to_vec();

                            if let Some(min_val) = config.validation_constraints.min_range_value {
                                if value < min_val {
                                    return Err(CryptoError::ProofError(
                                        format!("Value {} below minimum {}", value, min_val)
                                    ));
                                }
                            }
                            if let Some(max_val) = config.validation_constraints.max_range_value {
                                if value > max_val {
                                    return Err(CryptoError::ProofError(
                                        format!("Value {} above maximum {}", value, max_val)
                                    ));
                                }
                            }

                            Ok(Witness::range(value, blinding))
                        }
                        crate::proof::statement::StatementType::Custom { description: _ } => {
                            Ok(Witness::custom(secret.clone()))
                        }
                    }
                })
                .collect();

            // Convert results to Arc<Witness>
            let arc_witnesses: CryptoResult<Vec<Arc<Witness>>> = witnesses.into_iter()
                .map(|r| r.map(Arc::new))
                .collect();

            arc_witnesses
        } else {
            // Sequential creation
            let mut witnesses = Vec::with_capacity(statements.len());
            for (stmt, secret) in statements.iter().zip(secret_data.iter()) {
                let witness = self.create_witness(stmt, secret, config).await?;
                witnesses.push(witness);
            }
            Ok(witnesses)
        }
    }

    async fn batch_validate_witnesses(
        &self,
        witnesses: &[Arc<Witness>],
        statements: &[Statement],
        constraints: &ValidationConstraints,
    ) -> CryptoResult<Vec<bool>> {
        if witnesses.len() != statements.len() {
            return Err(CryptoError::ProofError(
                "Mismatched number of witnesses and statements".to_string()
            ));
        }

        use rayon::prelude::*;
        let results: Vec<bool> = witnesses.par_iter()
            .zip(statements.par_iter())
            .map(|(witness, statement)| {
                // Perform validation synchronously
                if let Ok(valid) = futures::executor::block_on(
                    self.validate_witness(witness, statement, constraints)
                ) {
                    valid
                } else {
                    false
                }
            })
            .collect();

        Ok(results)
    }
}

impl DefaultWitnessManager {
    /// Validate constraints before witness creation
    fn validate_constraints(&self, secret_data: &[u8], config: &WitnessGenerationConfig) -> CryptoResult<()> {
        // Check preimage length constraint
        if let Some(max_len) = config.validation_constraints.max_preimage_length {
            if secret_data.len() > max_len {
                return Err(CryptoError::ProofError(
                    format!("Secret data length {} exceeds maximum {}", secret_data.len(), max_len)
                ));
            }
        }

        Ok(())
    }
}

/// Witness transformation utilities
pub mod transforms {
    use super::*;

    /// Transform witness to different format (placeholder for future implementation)
    pub async fn transform_to_format(
        _witness: Arc<Witness>,
        target_format: &str,
        _options: &HashMap<String, serde_json::Value>,
    ) -> CryptoResult<TransformationResult> {
        match target_format {
            "json" => {
                // Placeholder: convert witness to JSON representation
                Ok(TransformationResult::Failed("JSON transformation not implemented".to_string()))
            }
            "binary" => {
                // Placeholder: convert witness to binary format
                Ok(TransformationResult::Failed("Binary transformation not implemented".to_string()))
            }
            _ => {
                Ok(TransformationResult::Failed(format!("Unknown format: {}", target_format)))
            }
        }
    }

    /// Validate witness format compatibility
    pub fn validate_format_compatibility(
        source_type: &WitnessType,
        target_format: &str,
    ) -> bool {
        match (source_type, target_format) {
            (WitnessType::DiscreteLog, "binary") => true,
            (WitnessType::Preimage, "binary") => true,
            (WitnessType::Range, "binary") => true,
            (WitnessType::Custom, "binary") => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::statement::StatementBuilder;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_witness_creation() {
        let manager = DefaultWitnessManager::new(1000);
        let statement = StatementBuilder::new().discrete_log(vec![1, 2, 3], vec![4, 5, 6]).build().unwrap();
        let secret_data = b"test_secret_data";
        let config = WitnessGenerationConfig {
            parallel_generation: false,
            max_parallel_tasks: 4,
            zero_copy: true,
            randomness_config: RandomnessConfig {
                length: 32,
                secure_random: true,
                custom_entropy: None,
            },
            validation_constraints: ValidationConstraints {
                max_range_value: Some(1000),
                min_range_value: Some(0),
                max_preimage_length: Some(1024),
                required_hash_function: None,
                custom_constraints: HashMap::new(),
            },
        };

        let result = manager.create_witness(&statement, secret_data, &config).await;
        assert!(result.is_ok(), "Witness creation should succeed");
    }
}

