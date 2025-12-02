//! Performance optimization for zero-knowledge proofs
//!
//! This module provides high-performance optimizations including parallel processing,
//! batch verification, caching, and adaptive algorithms.

use crate::proof::{Statement, Witness, Proof, Prover, Verifier, ProverConfig, VerifierConfig};
use crate::{CryptoError, CryptoResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Performance metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average proving time in milliseconds
    pub avg_proving_time_ms: f64,
    /// Average verification time in milliseconds
    pub avg_verification_time_ms: f64,
    /// Proofs processed per second
    pub proofs_per_second: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// CPU utilization percentage
    pub cpu_utilization_percent: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Parallel batch prover for high-throughput proof generation
pub struct ParallelBatchProver<T: Prover + Send + Sync> {
    /// Base prover implementation
    base_prover: Arc<T>,
    /// Number of parallel workers
    worker_count: usize,
    /// Batch size for processing
    batch_size: usize,
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
}

impl<T: Prover + Send + Sync + 'static> ParallelBatchProver<T> {
    /// Create a new parallel batch prover
    pub fn new(base_prover: T, worker_count: usize, batch_size: usize) -> Self {
        Self {
            base_prover: Arc::new(base_prover),
            worker_count,
            batch_size,
            metrics: Arc::new(RwLock::new(PerformanceMetrics {
                avg_proving_time_ms: 0.0,
                avg_verification_time_ms: 0.0,
                proofs_per_second: 0.0,
                memory_usage_bytes: 0,
                cpu_utilization_percent: 0.0,
                cache_hit_rate: 0.0,
            })),
        }
    }

    /// Process a large batch in parallel chunks
    pub async fn prove_large_batch(
        &self,
        statements: &[Statement],
        witnesses: &[Witness],
        config: &ProverConfig,
    ) -> CryptoResult<Vec<Proof>> {
        if statements.len() != witnesses.len() {
            return Err(CryptoError::InvalidInput("Mismatched statement and witness counts".to_string()));
        }

        let total_proofs = statements.len();
        let mut all_proofs = Vec::with_capacity(total_proofs);

        // Process in chunks
        for chunk_start in (0..total_proofs).step_by(self.batch_size) {
            let chunk_end = std::cmp::min(chunk_start + self.batch_size, total_proofs);
            let chunk_statements = &statements[chunk_start..chunk_end];
            let chunk_witnesses = &witnesses[chunk_start..chunk_end];

            let start_time = std::time::Instant::now();

            // Process chunk in parallel
            let chunk_proofs = self.prove_parallel_chunk(chunk_statements, chunk_witnesses, config).await?;

            let elapsed = start_time.elapsed();
            self.update_metrics(chunk_proofs.len(), elapsed).await;

            all_proofs.extend(chunk_proofs);
        }

        Ok(all_proofs)
    }

    /// Process a single chunk in parallel
    async fn prove_parallel_chunk(
        &self,
        statements: &[Statement],
        witnesses: &[Witness],
        config: &ProverConfig,
    ) -> CryptoResult<Vec<Proof>> {
        let chunk_size = statements.len();
        let proofs_per_worker = (chunk_size + self.worker_count - 1) / self.worker_count;

        // Create tasks for each worker
        let mut tasks = Vec::new();
        for worker_id in 0..self.worker_count {
            let start_idx = worker_id * proofs_per_worker;
            let end_idx = std::cmp::min(start_idx + proofs_per_worker, chunk_size);

            if start_idx >= chunk_size {
                break;
            }

            let worker_statements: Vec<Statement> = statements[start_idx..end_idx].to_vec();
            let worker_witnesses: Vec<Witness> = witnesses[start_idx..end_idx].to_vec();
            let worker_config = config.clone();
            let prover = Arc::clone(&self.base_prover);

            let task = tokio::spawn(async move {
                prover.prove_batch(&worker_statements, &worker_witnesses, &worker_config).await
            });

            tasks.push(task);
        }

        // Wait for all workers to complete
        let mut all_proofs = Vec::new();
        for task in tasks {
            match task.await {
                Ok(Ok(mut proofs)) => all_proofs.append(&mut proofs),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(CryptoError::InternalError(format!("Task join error: {}", e))),
            }
        }

        Ok(all_proofs)
    }

    /// Update performance metrics
    async fn update_metrics(&self, proof_count: usize, elapsed: std::time::Duration) {
        let mut metrics = self.metrics.write().await;
        let elapsed_ms = elapsed.as_millis() as f64;

        // Update rolling averages
        let alpha = 0.1; // Smoothing factor
        metrics.avg_proving_time_ms = metrics.avg_proving_time_ms * (1.0 - alpha) +
            (elapsed_ms / proof_count as f64) * alpha;
        metrics.proofs_per_second = proof_count as f64 / (elapsed_ms / 1000.0);
    }

    /// Get current performance metrics
    pub async fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.read().await.clone()
    }
}

/// Optimized batch verifier with caching and parallel processing
pub struct OptimizedBatchVerifier<T: Verifier + Send + Sync> {
    /// Base verifier implementation
    base_verifier: Arc<T>,
    /// LRU cache for verification results
    cache: Arc<RwLock<lru::LruCache<String, bool>>>,
    /// Number of parallel workers
    worker_count: usize,
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
}

impl<T: Verifier + Send + Sync + 'static> OptimizedBatchVerifier<T> {
    /// Create a new optimized batch verifier
    pub fn new(base_verifier: T, cache_size: usize, worker_count: usize) -> Self {
        Self {
            base_verifier: Arc::new(base_verifier),
            cache: Arc::new(RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(cache_size).unwrap()
            ))),
            worker_count,
            metrics: Arc::new(RwLock::new(PerformanceMetrics {
                avg_proving_time_ms: 0.0,
                avg_verification_time_ms: 0.0,
                proofs_per_second: 0.0,
                memory_usage_bytes: 0,
                cpu_utilization_percent: 0.0,
                cache_hit_rate: 0.0,
            })),
        }
    }

    /// Verify batch with caching and parallel processing
    pub async fn verify_batch_optimized(
        &self,
        statements: &[Statement],
        proofs: &[Proof],
        config: &VerifierConfig,
    ) -> CryptoResult<Vec<bool>> {
        if statements.len() != proofs.len() {
            return Err(CryptoError::InvalidInput("Mismatched statement and proof counts".to_string()));
        }

        let total_proofs = statements.len();
        let mut results = vec![false; total_proofs];
        let mut uncached_indices = Vec::new();

        // Check cache first
        let mut cache_hits = 0;
        {
            let mut cache = self.cache.write().await;
            for (i, (statement, proof)) in statements.iter().zip(proofs.iter()).enumerate() {
                if let Ok(cache_key) = self.create_cache_key(statement, proof) {
                    if let Some(cached_result) = cache.get(&cache_key) {
                        results[i] = *cached_result;
                        cache_hits += 1;
                    } else {
                        uncached_indices.push(i);
                    }
                } else {
                    // If cache key creation fails, treat as uncached
                    uncached_indices.push(i);
                }
            }
        }

        // Update cache hit rate
        {
            let mut metrics = self.metrics.write().await;
            let total_requests = total_proofs as f64;
            let hit_rate = cache_hits as f64 / total_requests;
            metrics.cache_hit_rate = metrics.cache_hit_rate * 0.9 + hit_rate * 0.1;
        }

        if uncached_indices.is_empty() {
            return Ok(results);
        }

        // Verify uncached proofs in parallel
        let start_time = std::time::Instant::now();
        let uncached_results = self.verify_uncached_parallel(
            &uncached_indices.iter().map(|&i| &statements[i]).collect::<Vec<_>>(),
            &uncached_indices.iter().map(|&i| &proofs[i]).collect::<Vec<_>>(),
            config,
        ).await?;

        let elapsed = start_time.elapsed();
        self.update_metrics(uncached_results.len(), elapsed).await;

        // Update results and cache
        {
            let mut cache = self.cache.write().await;
            for (idx, (uncached_idx, result)) in uncached_indices.iter().zip(uncached_results.iter()).enumerate() {
                results[*uncached_idx] = *result;
                if let Ok(cache_key) = self.create_cache_key(&statements[*uncached_idx], &proofs[*uncached_idx]) {
                    cache.put(cache_key, *result);
                }
                // If cache key creation fails, skip caching but don't fail the verification
            }
        }

        Ok(results)
    }

    /// Verify uncached proofs in parallel
    async fn verify_uncached_parallel(
        &self,
        statements: &[&Statement],
        proofs: &[&Proof],
        config: &VerifierConfig,
    ) -> CryptoResult<Vec<bool>> {
        let chunk_size = statements.len();
        let proofs_per_worker = (chunk_size + self.worker_count - 1) / self.worker_count;

        // Create tasks for each worker
        let mut tasks = Vec::new();
        for worker_id in 0..self.worker_count {
            let start_idx = worker_id * proofs_per_worker;
            let end_idx = std::cmp::min(start_idx + proofs_per_worker, chunk_size);

            if start_idx >= chunk_size {
                break;
            }

            let worker_statements: Vec<Statement> = statements[start_idx..end_idx].iter().map(|s| (*s).clone()).collect();
            let worker_proofs: Vec<Proof> = proofs[start_idx..end_idx].iter().map(|p| (*p).clone()).collect();
            let worker_config = config.clone();
            let verifier = Arc::clone(&self.base_verifier);

            let task = tokio::spawn(async move {
                verifier.verify_batch(&worker_statements, &worker_proofs, &worker_config).await
            });

            tasks.push(task);
        }

        // Wait for all workers to complete
        let mut all_results = Vec::new();
        for task in tasks {
            match task.await {
                Ok(Ok(mut results)) => all_results.append(&mut results),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(CryptoError::InternalError(format!("Task join error: {}", e))),
            }
        }

        Ok(all_results)
    }

    /// Create cache key for statement-proof pair
    fn create_cache_key(&self, statement: &Statement, proof: &Proof) -> CryptoResult<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash the statement
        let statement_hash = statement.hash()?;
        hasher.write(&statement_hash);

        // Hash the proof bytes
        let proof_bytes = proof.to_bytes()?;
        hasher.write(&proof_bytes);

        Ok(format!("{:x}", hasher.finish()))
    }

    /// Update performance metrics
    async fn update_metrics(&self, verification_count: usize, elapsed: std::time::Duration) {
        let mut metrics = self.metrics.write().await;
        let elapsed_ms = elapsed.as_millis() as f64;

        // Update rolling averages
        let alpha = 0.1; // Smoothing factor
        metrics.avg_verification_time_ms = metrics.avg_verification_time_ms * (1.0 - alpha) +
            (elapsed_ms / verification_count as f64) * alpha;
    }

    /// Get current performance metrics
    pub async fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.read().await.clone()
    }

    /// Clear cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}

/// Adaptive prover that chooses optimal strategy based on input characteristics
pub struct AdaptiveProver {
    /// Available prover strategies
    strategies: HashMap<String, Box<dyn Prover + Send + Sync>>,
    /// Performance history for strategy selection
    performance_history: Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl AdaptiveProver {
    /// Create a new adaptive prover
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            performance_history: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a prover strategy
    pub fn add_strategy(&mut self, name: String, prover: Box<dyn Prover + Send + Sync>) {
        self.strategies.insert(name, prover);
    }

    /// Adaptively select and use the best prover strategy
    pub async fn prove_adaptive(
        &self,
        statement: &Statement,
        witness: &Witness,
        config: &ProverConfig,
    ) -> CryptoResult<Proof> {
        let strategy_name = self.select_best_strategy(statement, witness).await?;
        let prover = self.strategies.get(&strategy_name)
            .ok_or_else(|| CryptoError::InvalidInput(format!("Unknown strategy: {}", strategy_name)))?;

        let start_time = std::time::Instant::now();
        let result = prover.prove(statement, witness, config).await?;
        let elapsed = start_time.elapsed().as_millis() as f64;

        // Record performance
        self.record_performance(&strategy_name, elapsed).await;

        Ok(result)
    }

    /// Select the best strategy based on input characteristics and history
    async fn select_best_strategy(&self, statement: &Statement, witness: &Witness) -> CryptoResult<String> {
        // Simple strategy selection based on statement type and size
        // In a real implementation, this would use ML to predict optimal strategy

        let history = self.performance_history.read().await;

        // Find strategy with best average performance
        let mut best_strategy = None;
        let mut best_avg_time = f64::INFINITY;

        for (name, times) in history.iter() {
            if times.is_empty() {
                continue;
            }
            let avg_time = times.iter().sum::<f64>() / times.len() as f64;
            if avg_time < best_avg_time {
                best_avg_time = avg_time;
                best_strategy = Some(name.clone());
            }
        }

        // Fallback to default strategy if no history
        Ok(best_strategy.unwrap_or_else(|| "direct".to_string()))
    }

    /// Record performance for a strategy
    async fn record_performance(&self, strategy: &str, time_ms: f64) {
        let mut history = self.performance_history.write().await;
        history.entry(strategy.to_string())
            .or_insert_with(Vec::new)
            .push(time_ms);

        // Keep only last 100 measurements
        if let Some(times) = history.get_mut(strategy) {
            if times.len() > 100 {
                times.remove(0);
            }
        }
    }
}

/// Performance monitoring and alerting
pub struct PerformanceMonitor {
    /// Performance thresholds
    thresholds: PerformanceThresholds,
    /// Alert callback
    alert_callback: Option<Box<dyn Fn(PerformanceAlert) + Send + Sync>>,
}

#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub max_proving_time_ms: f64,
    pub max_verification_time_ms: f64,
    pub min_proofs_per_second: f64,
    pub max_memory_usage_bytes: u64,
    pub max_cpu_utilization_percent: f64,
    pub min_cache_hit_rate: f64,
}

#[derive(Debug, Clone)]
pub enum PerformanceAlert {
    SlowProving { actual_time_ms: f64, threshold_ms: f64 },
    SlowVerification { actual_time_ms: f64, threshold_ms: f64 },
    LowThroughput { actual_pps: f64, threshold_pps: f64 },
    HighMemoryUsage { actual_bytes: u64, threshold_bytes: u64 },
    HighCpuUsage { actual_percent: f64, threshold_percent: f64 },
    LowCacheHitRate { actual_rate: f64, threshold_rate: f64 },
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(thresholds: PerformanceThresholds) -> Self {
        Self {
            thresholds,
            alert_callback: None,
        }
    }

    /// Set alert callback
    pub fn set_alert_callback<F>(&mut self, callback: F)
    where
        F: Fn(PerformanceAlert) + Send + Sync + 'static,
    {
        self.alert_callback = Some(Box::new(callback));
    }

    /// Check metrics against thresholds and trigger alerts
    pub fn check_and_alert(&self, metrics: &PerformanceMetrics) {
        let alerts = self.check_thresholds(metrics);

        if let Some(callback) = &self.alert_callback {
            for alert in alerts {
                callback(alert);
            }
        }
    }

    /// Check metrics against thresholds
    fn check_thresholds(&self, metrics: &PerformanceMetrics) -> Vec<PerformanceAlert> {
        let mut alerts = Vec::new();

        if metrics.avg_proving_time_ms > self.thresholds.max_proving_time_ms {
            alerts.push(PerformanceAlert::SlowProving {
                actual_time_ms: metrics.avg_proving_time_ms,
                threshold_ms: self.thresholds.max_proving_time_ms,
            });
        }

        if metrics.avg_verification_time_ms > self.thresholds.max_verification_time_ms {
            alerts.push(PerformanceAlert::SlowVerification {
                actual_time_ms: metrics.avg_verification_time_ms,
                threshold_ms: self.thresholds.max_verification_time_ms,
            });
        }

        if metrics.proofs_per_second < self.thresholds.min_proofs_per_second {
            alerts.push(PerformanceAlert::LowThroughput {
                actual_pps: metrics.proofs_per_second,
                threshold_pps: self.thresholds.min_proofs_per_second,
            });
        }

        if metrics.memory_usage_bytes > self.thresholds.max_memory_usage_bytes {
            alerts.push(PerformanceAlert::HighMemoryUsage {
                actual_bytes: metrics.memory_usage_bytes,
                threshold_bytes: self.thresholds.max_memory_usage_bytes,
            });
        }

        if metrics.cpu_utilization_percent > self.thresholds.max_cpu_utilization_percent {
            alerts.push(PerformanceAlert::HighCpuUsage {
                actual_percent: metrics.cpu_utilization_percent,
                threshold_percent: self.thresholds.max_cpu_utilization_percent,
            });
        }

        if metrics.cache_hit_rate < self.thresholds.min_cache_hit_rate {
            alerts.push(PerformanceAlert::LowCacheHitRate {
                actual_rate: metrics.cache_hit_rate,
                threshold_rate: self.thresholds.min_cache_hit_rate,
            });
        }

        alerts
    }
}