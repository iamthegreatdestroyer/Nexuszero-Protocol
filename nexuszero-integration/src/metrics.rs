//! Comprehensive Metrics Collection for NexusZero Protocol
//!
//! This module provides detailed performance metrics tracking for:
//! - Proof generation timing and throughput
//! - Compression ratios and efficiency
//! - Memory usage and allocation patterns
//! - Neural optimization impact measurement
//! - Verification performance tracking
//!
//! # Example
//!
//! ```rust,no_run
//! use nexuszero_integration::metrics::{MetricsCollector, ProofStageMetrics};
//!
//! let mut collector = MetricsCollector::new();
//! collector.start_stage("proof_generation");
//! // ... generate proof ...
//! collector.end_stage("proof_generation");
//! let metrics = collector.finalize();
//! let report = metrics.summary();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// PROOF STAGE METRICS
// ============================================================================

/// Metrics for a single stage of proof processing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofStageMetrics {
    /// Name of the stage (e.g., "validation", "generation", "compression")
    pub stage_name: String,
    /// Duration of this stage in milliseconds
    pub duration_ms: f64,
    /// Memory allocated during this stage (bytes)
    pub memory_bytes: usize,
    /// Additional stage-specific metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl ProofStageMetrics {
    /// Create new stage metrics
    pub fn new(stage_name: &str, duration: Duration) -> Self {
        Self {
            stage_name: stage_name.to_string(),
            duration_ms: duration.as_secs_f64() * 1000.0,
            memory_bytes: 0,
            custom_metrics: HashMap::new(),
        }
    }

    /// Add a custom metric value
    pub fn add_metric(&mut self, key: &str, value: f64) {
        self.custom_metrics.insert(key.to_string(), value);
    }
}

// ============================================================================
// COMPREHENSIVE PROOF METRICS
// ============================================================================

/// Comprehensive metrics for an entire proof operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComprehensiveProofMetrics {
    /// Total time from start to finish (ms)
    pub total_time_ms: f64,
    
    /// Time spent in validation phase (ms)
    pub validation_time_ms: f64,
    
    /// Time spent in parameter selection (ms)
    pub parameter_selection_time_ms: f64,
    
    /// Time spent generating the proof (ms)
    pub generation_time_ms: f64,
    
    /// Time spent in compression (ms)
    pub compression_time_ms: f64,
    
    /// Time spent in verification (ms)
    pub verification_time_ms: f64,
    
    /// Original proof size in bytes
    pub original_proof_size_bytes: usize,
    
    /// Compressed proof size in bytes (if compressed)
    pub compressed_proof_size_bytes: Option<usize>,
    
    /// Compression ratio (original/compressed), 1.0 if not compressed
    pub compression_ratio: f64,
    
    /// Whether neural optimization was used
    pub neural_optimization_used: bool,
    
    /// Speedup factor from neural optimization (vs baseline)
    pub neural_speedup_factor: f64,
    
    /// Security level used
    pub security_level: String,
    
    /// Peak memory usage estimate (bytes)
    pub peak_memory_bytes: usize,
    
    /// Number of commitments in proof
    pub commitment_count: usize,
    
    /// Number of responses in proof
    pub response_count: usize,
    
    /// Per-stage breakdown
    pub stages: Vec<ProofStageMetrics>,
    
    /// Timestamp of metric collection
    pub timestamp: u64,
}

impl Default for ComprehensiveProofMetrics {
    fn default() -> Self {
        Self {
            total_time_ms: 0.0,
            validation_time_ms: 0.0,
            parameter_selection_time_ms: 0.0,
            generation_time_ms: 0.0,
            compression_time_ms: 0.0,
            verification_time_ms: 0.0,
            original_proof_size_bytes: 0,
            compressed_proof_size_bytes: None,
            compression_ratio: 1.0,
            neural_optimization_used: false,
            neural_speedup_factor: 1.0,
            security_level: "Bit128".to_string(),
            peak_memory_bytes: 0,
            commitment_count: 0,
            response_count: 0,
            stages: Vec::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

impl ComprehensiveProofMetrics {
    /// Create metrics from basic measurements
    pub fn from_basic(
        generation_time_ms: f64,
        proof_size: usize,
        compression_ratio: f64,
    ) -> Self {
        Self {
            total_time_ms: generation_time_ms,
            generation_time_ms,
            original_proof_size_bytes: proof_size,
            compression_ratio,
            ..Default::default()
        }
    }

    /// Calculate throughput in proofs per second
    pub fn throughput_per_second(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            1000.0 / self.total_time_ms
        } else {
            0.0
        }
    }

    /// Calculate bytes processed per second
    pub fn bytes_per_second(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            (self.original_proof_size_bytes as f64 * 1000.0) / self.total_time_ms
        } else {
            0.0
        }
    }

    /// Get compression savings in bytes
    pub fn compression_savings(&self) -> usize {
        if let Some(compressed) = self.compressed_proof_size_bytes {
            self.original_proof_size_bytes.saturating_sub(compressed)
        } else {
            0
        }
    }

    /// Add a stage to the metrics
    pub fn add_stage(&mut self, stage: ProofStageMetrics) {
        self.stages.push(stage);
    }

    /// Get total stage time
    pub fn total_stage_time_ms(&self) -> f64 {
        self.stages.iter().map(|s| s.duration_ms).sum()
    }

    /// Generate human-readable summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("=== Proof Metrics Summary ===\n"));
        summary.push_str(&format!("Total Time: {:.2}ms\n", self.total_time_ms));
        summary.push_str(&format!("  - Validation: {:.2}ms\n", self.validation_time_ms));
        summary.push_str(&format!("  - Parameters: {:.2}ms\n", self.parameter_selection_time_ms));
        summary.push_str(&format!("  - Generation: {:.2}ms\n", self.generation_time_ms));
        summary.push_str(&format!("  - Compression: {:.2}ms\n", self.compression_time_ms));
        summary.push_str(&format!("  - Verification: {:.2}ms\n", self.verification_time_ms));
        summary.push_str(&format!("Original Size: {} bytes\n", self.original_proof_size_bytes));
        if let Some(compressed) = self.compressed_proof_size_bytes {
            summary.push_str(&format!("Compressed Size: {} bytes\n", compressed));
            summary.push_str(&format!("Compression Ratio: {:.2}x\n", self.compression_ratio));
            summary.push_str(&format!("Space Saved: {} bytes\n", self.compression_savings()));
        }
        summary.push_str(&format!("Neural Optimization: {}\n", 
            if self.neural_optimization_used { "Enabled" } else { "Disabled" }));
        if self.neural_optimization_used {
            summary.push_str(&format!("Neural Speedup: {:.2}x\n", self.neural_speedup_factor));
        }
        summary.push_str(&format!("Throughput: {:.2} proofs/sec\n", self.throughput_per_second()));
        summary
    }
}

// ============================================================================
// METRICS COLLECTOR
// ============================================================================

/// Collector for gathering metrics during proof operations
#[derive(Debug)]
pub struct MetricsCollector {
    /// Start time of collection
    start_time: Option<Instant>,
    /// Active stage timings
    active_stages: HashMap<String, Instant>,
    /// Completed stage metrics
    completed_stages: Vec<ProofStageMetrics>,
    /// Running metrics accumulator
    metrics: ComprehensiveProofMetrics,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            start_time: None,
            active_stages: HashMap::new(),
            completed_stages: Vec::new(),
            metrics: ComprehensiveProofMetrics::default(),
        }
    }

    /// Start overall metrics collection
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Start timing a specific stage
    pub fn start_stage(&mut self, stage_name: &str) {
        self.active_stages.insert(stage_name.to_string(), Instant::now());
    }

    /// End timing a specific stage and record metrics
    pub fn end_stage(&mut self, stage_name: &str) -> Option<f64> {
        if let Some(start) = self.active_stages.remove(stage_name) {
            let duration = start.elapsed();
            let duration_ms = duration.as_secs_f64() * 1000.0;
            let stage = ProofStageMetrics::new(stage_name, duration);
            self.completed_stages.push(stage);
            
            // Update specific timing fields
            match stage_name {
                "validation" => self.metrics.validation_time_ms = duration_ms,
                "parameter_selection" => self.metrics.parameter_selection_time_ms = duration_ms,
                "generation" => self.metrics.generation_time_ms = duration_ms,
                "compression" => self.metrics.compression_time_ms = duration_ms,
                "verification" => self.metrics.verification_time_ms = duration_ms,
                _ => {}
            }
            
            Some(duration_ms)
        } else {
            None
        }
    }

    /// Record proof size metrics
    pub fn record_proof_size(&mut self, original: usize, compressed: Option<usize>) {
        self.metrics.original_proof_size_bytes = original;
        self.metrics.compressed_proof_size_bytes = compressed;
        if let Some(c) = compressed {
            if c > 0 {
                self.metrics.compression_ratio = original as f64 / c as f64;
            }
        }
    }

    /// Record proof structure counts
    pub fn record_proof_structure(&mut self, commitments: usize, responses: usize) {
        self.metrics.commitment_count = commitments;
        self.metrics.response_count = responses;
    }

    /// Record neural optimization usage
    pub fn record_neural_optimization(&mut self, used: bool, speedup: f64) {
        self.metrics.neural_optimization_used = used;
        self.metrics.neural_speedup_factor = speedup;
    }

    /// Record security level
    pub fn record_security_level(&mut self, level: &str) {
        self.metrics.security_level = level.to_string();
    }

    /// Record peak memory usage
    pub fn record_peak_memory(&mut self, bytes: usize) {
        self.metrics.peak_memory_bytes = bytes;
    }

    /// Finalize and generate the complete metrics report
    pub fn finalize(&mut self) -> ComprehensiveProofMetrics {
        // Calculate total time
        if let Some(start) = self.start_time {
            self.metrics.total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        }
        
        // Add all completed stages
        self.metrics.stages = self.completed_stages.clone();
        
        self.metrics.clone()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// BATCH METRICS AGGREGATOR
// ============================================================================

/// Aggregator for collecting metrics across multiple proof operations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchMetricsAggregator {
    /// All collected metrics
    metrics: Vec<ComprehensiveProofMetrics>,
    /// Start time of batch
    batch_start: u64,
}

impl BatchMetricsAggregator {
    /// Create a new batch aggregator
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            batch_start: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Add metrics from a proof operation
    pub fn add(&mut self, metrics: ComprehensiveProofMetrics) {
        self.metrics.push(metrics);
    }

    /// Get count of collected metrics
    pub fn count(&self) -> usize {
        self.metrics.len()
    }

    /// Calculate average generation time
    pub fn avg_generation_time_ms(&self) -> f64 {
        if self.metrics.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.metrics.iter().map(|m| m.generation_time_ms).sum();
        sum / self.metrics.len() as f64
    }

    /// Calculate average total time
    pub fn avg_total_time_ms(&self) -> f64 {
        if self.metrics.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.metrics.iter().map(|m| m.total_time_ms).sum();
        sum / self.metrics.len() as f64
    }

    /// Calculate average compression ratio
    pub fn avg_compression_ratio(&self) -> f64 {
        if self.metrics.is_empty() {
            return 1.0;
        }
        let sum: f64 = self.metrics.iter().map(|m| m.compression_ratio).sum();
        sum / self.metrics.len() as f64
    }

    /// Calculate total bytes processed
    pub fn total_bytes_processed(&self) -> usize {
        self.metrics.iter().map(|m| m.original_proof_size_bytes).sum()
    }

    /// Get min generation time
    pub fn min_generation_time_ms(&self) -> f64 {
        self.metrics.iter()
            .map(|m| m.generation_time_ms)
            .fold(f64::INFINITY, f64::min)
    }

    /// Get max generation time
    pub fn max_generation_time_ms(&self) -> f64 {
        self.metrics.iter()
            .map(|m| m.generation_time_ms)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Calculate standard deviation of generation time
    pub fn std_dev_generation_time_ms(&self) -> f64 {
        if self.metrics.len() < 2 {
            return 0.0;
        }
        let avg = self.avg_generation_time_ms();
        let variance: f64 = self.metrics.iter()
            .map(|m| (m.generation_time_ms - avg).powi(2))
            .sum::<f64>() / (self.metrics.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate percentile of generation time (0-100)
    pub fn percentile_generation_time_ms(&self, percentile: f64) -> f64 {
        if self.metrics.is_empty() {
            return 0.0;
        }
        let mut times: Vec<f64> = self.metrics.iter().map(|m| m.generation_time_ms).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((percentile / 100.0) * (times.len() - 1) as f64).round() as usize;
        times.get(idx).cloned().unwrap_or(0.0)
    }

    /// Generate batch summary report
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("=== Batch Metrics Summary ===\n"));
        summary.push_str(&format!("Total Proofs: {}\n", self.count()));
        summary.push_str(&format!("Avg Generation Time: {:.2}ms\n", self.avg_generation_time_ms()));
        summary.push_str(&format!("Avg Total Time: {:.2}ms\n", self.avg_total_time_ms()));
        summary.push_str(&format!("Min Generation Time: {:.2}ms\n", self.min_generation_time_ms()));
        summary.push_str(&format!("Max Generation Time: {:.2}ms\n", self.max_generation_time_ms()));
        summary.push_str(&format!("Std Dev: {:.2}ms\n", self.std_dev_generation_time_ms()));
        summary.push_str(&format!("P50: {:.2}ms\n", self.percentile_generation_time_ms(50.0)));
        summary.push_str(&format!("P95: {:.2}ms\n", self.percentile_generation_time_ms(95.0)));
        summary.push_str(&format!("P99: {:.2}ms\n", self.percentile_generation_time_ms(99.0)));
        summary.push_str(&format!("Avg Compression Ratio: {:.2}x\n", self.avg_compression_ratio()));
        summary.push_str(&format!("Total Bytes Processed: {}\n", self.total_bytes_processed()));
        summary
    }
}

impl Default for BatchMetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HISTOGRAM TRACKING
// ============================================================================

/// Histogram for tracking latency distributions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatencyHistogram {
    /// Bucket boundaries in milliseconds
    buckets: Vec<f64>,
    /// Count per bucket
    counts: Vec<usize>,
    /// Total observations
    total: usize,
    /// Sum for average calculation
    sum: f64,
    /// Minimum observed value
    min: f64,
    /// Maximum observed value
    max: f64,
}

impl LatencyHistogram {
    /// Create a new histogram with default buckets (exponential)
    pub fn new() -> Self {
        // Default buckets: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000ms
        Self::with_buckets(vec![
            1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, f64::INFINITY,
        ])
    }

    /// Create with custom bucket boundaries
    pub fn with_buckets(mut buckets: Vec<f64>) -> Self {
        buckets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let counts = vec![0; buckets.len()];
        Self {
            buckets,
            counts,
            total: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Observe a latency value
    pub fn observe(&mut self, value: f64) {
        self.total += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        // Find bucket
        for (i, &bound) in self.buckets.iter().enumerate() {
            if value <= bound {
                self.counts[i] += 1;
                return;
            }
        }
    }

    /// Get average latency
    pub fn average(&self) -> f64 {
        if self.total > 0 {
            self.sum / self.total as f64
        } else {
            0.0
        }
    }

    /// Get minimum latency
    pub fn minimum(&self) -> f64 {
        if self.min == f64::INFINITY { 0.0 } else { self.min }
    }

    /// Get maximum latency
    pub fn maximum(&self) -> f64 {
        if self.max == f64::NEG_INFINITY { 0.0 } else { self.max }
    }

    /// Get count at specific bucket
    pub fn bucket_count(&self, index: usize) -> usize {
        self.counts.get(index).cloned().unwrap_or(0)
    }

    /// Get all bucket counts
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// Get bucket boundaries
    pub fn boundaries(&self) -> &[f64] {
        &self.buckets
    }

    /// Calculate percentile from histogram
    pub fn percentile(&self, p: f64) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        
        let target = (p / 100.0 * self.total as f64).ceil() as usize;
        let mut cumulative = 0;
        
        for (i, &count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                // Linear interpolation within bucket
                if i == 0 {
                    return self.buckets[0];
                }
                let lower = if i == 0 { 0.0 } else { self.buckets[i - 1] };
                let upper = self.buckets[i];
                return (lower + upper) / 2.0;
            }
        }
        
        self.max
    }

    /// Generate ASCII histogram representation
    pub fn ascii_histogram(&self, width: usize) -> String {
        let max_count = self.counts.iter().cloned().max().unwrap_or(1);
        let mut output = String::new();
        
        for (i, &count) in self.counts.iter().enumerate() {
            let upper_bound = if i < self.buckets.len() {
                self.buckets[i]
            } else {
                f64::INFINITY
            };
            
            if upper_bound == f64::INFINITY && count == 0 {
                continue; // Skip empty infinity bucket
            }
            
            let bar_len = if max_count > 0 {
                (count * width / max_count).max(if count > 0 { 1 } else { 0 })
            } else {
                0
            };
            
            let bar: String = "█".repeat(bar_len);
            let bound_str = if upper_bound == f64::INFINITY {
                "+inf".to_string()
            } else {
                format!("{:.0}", upper_bound)
            };
            
            output.push_str(&format!(
                "{:>8}ms | {:width$} | {}\n",
                bound_str,
                bar,
                count,
                width = width
            ));
        }
        
        output
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RATE TRACKER
// ============================================================================

/// Tracks rate of events over time (proofs/second, bytes/second, etc.)
#[derive(Clone, Debug)]
pub struct RateTracker {
    /// Window size for rate calculation
    window_seconds: f64,
    /// Timestamps of events (as seconds since start)
    events: Vec<f64>,
    /// Start time
    start: Instant,
    /// Values associated with events (for weighted rates)
    values: Vec<f64>,
}

impl RateTracker {
    /// Create a new rate tracker with specified window
    pub fn new(window_seconds: f64) -> Self {
        Self {
            window_seconds,
            events: Vec::new(),
            start: Instant::now(),
            values: Vec::new(),
        }
    }

    /// Record an event
    pub fn record(&mut self) {
        self.record_with_value(1.0);
    }

    /// Record an event with a value (e.g., bytes)
    pub fn record_with_value(&mut self, value: f64) {
        let elapsed = self.start.elapsed().as_secs_f64();
        self.events.push(elapsed);
        self.values.push(value);
        self.cleanup();
    }

    /// Get current rate (events per second)
    pub fn rate(&self) -> f64 {
        self.cleanup_snapshot();
        let window_events = self.count_in_window();
        if self.window_seconds > 0.0 {
            window_events as f64 / self.window_seconds
        } else {
            0.0
        }
    }

    /// Get current weighted rate (sum of values per second)
    pub fn weighted_rate(&self) -> f64 {
        let elapsed = self.start.elapsed().as_secs_f64();
        let cutoff = elapsed - self.window_seconds;
        
        let sum: f64 = self.events.iter().zip(self.values.iter())
            .filter(|&(&t, _)| t > cutoff)
            .map(|(_, &v)| v)
            .sum();
        
        if self.window_seconds > 0.0 {
            sum / self.window_seconds
        } else {
            0.0
        }
    }

    /// Count events in window
    fn count_in_window(&self) -> usize {
        let elapsed = self.start.elapsed().as_secs_f64();
        let cutoff = elapsed - self.window_seconds;
        self.events.iter().filter(|&&t| t > cutoff).count()
    }

    /// Cleanup old events (modifies self)
    fn cleanup(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let cutoff = elapsed - self.window_seconds * 2.0; // Keep 2x window for safety
        
        // Find first event to keep
        let keep_from = self.events.iter().position(|&t| t > cutoff).unwrap_or(self.events.len());
        
        if keep_from > 0 {
            self.events.drain(0..keep_from);
            self.values.drain(0..keep_from);
        }
    }

    /// Non-mutating snapshot for rate calculation
    fn cleanup_snapshot(&self) {
        // This is a read-only operation, actual cleanup happens on record
    }

    /// Total events recorded
    pub fn total_events(&self) -> usize {
        self.events.len()
    }
}

// ============================================================================
// PERFORMANCE COMPARISON
// ============================================================================

/// Performance comparison between baseline and optimized runs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceComparison {
    /// Baseline metrics
    pub baseline: ComprehensiveProofMetrics,
    /// Optimized metrics
    pub optimized: ComprehensiveProofMetrics,
}

impl PerformanceComparison {
    /// Create a new comparison
    pub fn new(baseline: ComprehensiveProofMetrics, optimized: ComprehensiveProofMetrics) -> Self {
        Self { baseline, optimized }
    }

    /// Calculate speedup factor
    pub fn speedup_factor(&self) -> f64 {
        if self.optimized.total_time_ms > 0.0 {
            self.baseline.total_time_ms / self.optimized.total_time_ms
        } else {
            0.0
        }
    }

    /// Calculate compression improvement
    pub fn compression_improvement(&self) -> f64 {
        self.optimized.compression_ratio / self.baseline.compression_ratio
    }

    /// Generate comparison report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("=== Performance Comparison ===\n"));
        report.push_str(&format!("\nBaseline:\n"));
        report.push_str(&format!("  Total Time: {:.2}ms\n", self.baseline.total_time_ms));
        report.push_str(&format!("  Compression Ratio: {:.2}x\n", self.baseline.compression_ratio));
        report.push_str(&format!("\nOptimized:\n"));
        report.push_str(&format!("  Total Time: {:.2}ms\n", self.optimized.total_time_ms));
        report.push_str(&format!("  Compression Ratio: {:.2}x\n", self.optimized.compression_ratio));
        report.push_str(&format!("\nImprovement:\n"));
        report.push_str(&format!("  Speedup: {:.2}x\n", self.speedup_factor()));
        report.push_str(&format!("  Compression Improvement: {:.2}x\n", self.compression_improvement()));
        report
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_basic() {
        let mut collector = MetricsCollector::new();
        collector.start();
        
        collector.start_stage("generation");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let duration = collector.end_stage("generation");
        
        assert!(duration.is_some());
        assert!(duration.unwrap() >= 10.0);
        
        let metrics = collector.finalize();
        assert!(metrics.generation_time_ms >= 10.0);
    }

    #[test]
    fn test_comprehensive_metrics_summary() {
        let metrics = ComprehensiveProofMetrics {
            total_time_ms: 100.0,
            generation_time_ms: 80.0,
            original_proof_size_bytes: 1000,
            compressed_proof_size_bytes: Some(500),
            compression_ratio: 2.0,
            ..Default::default()
        };
        
        let summary = metrics.summary();
        assert!(summary.contains("100.00ms"));
        assert!(summary.contains("2.00x"));
    }

    #[test]
    fn test_batch_aggregator() {
        let mut aggregator = BatchMetricsAggregator::new();
        
        for i in 0..10 {
            let metrics = ComprehensiveProofMetrics {
                total_time_ms: 100.0 + i as f64 * 10.0,
                generation_time_ms: 80.0 + i as f64 * 10.0,
                ..Default::default()
            };
            aggregator.add(metrics);
        }
        
        assert_eq!(aggregator.count(), 10);
        assert!(aggregator.avg_generation_time_ms() > 0.0);
        assert!(aggregator.std_dev_generation_time_ms() > 0.0);
    }

    #[test]
    fn test_performance_comparison() {
        let baseline = ComprehensiveProofMetrics {
            total_time_ms: 200.0,
            compression_ratio: 1.0,
            ..Default::default()
        };
        let optimized = ComprehensiveProofMetrics {
            total_time_ms: 100.0,
            compression_ratio: 2.0,
            ..Default::default()
        };
        
        let comparison = PerformanceComparison::new(baseline, optimized);
        assert!((comparison.speedup_factor() - 2.0).abs() < 0.01);
        assert!((comparison.compression_improvement() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_throughput_calculation() {
        let metrics = ComprehensiveProofMetrics {
            total_time_ms: 100.0,
            original_proof_size_bytes: 10000,
            ..Default::default()
        };
        
        assert!((metrics.throughput_per_second() - 10.0).abs() < 0.01);
        assert!((metrics.bytes_per_second() - 100000.0).abs() < 1.0);
    }

    #[test]
    fn test_latency_histogram() {
        let mut histogram = LatencyHistogram::new();
        
        // Add various latencies
        histogram.observe(5.0);
        histogram.observe(15.0);
        histogram.observe(50.0);
        histogram.observe(150.0);
        histogram.observe(500.0);
        
        assert_eq!(histogram.total, 5);
        assert!((histogram.average() - 144.0).abs() < 0.1);
        assert!((histogram.minimum() - 5.0).abs() < 0.01);
        assert!((histogram.maximum() - 500.0).abs() < 0.01);
        
        // Check percentiles
        let p50 = histogram.percentile(50.0);
        assert!(p50 > 0.0);
    }

    #[test]
    fn test_latency_histogram_buckets() {
        let mut histogram = LatencyHistogram::with_buckets(vec![10.0, 50.0, 100.0, f64::INFINITY]);
        
        histogram.observe(5.0);  // bucket 0 (<=10)
        histogram.observe(8.0);  // bucket 0 (<=10)
        histogram.observe(25.0); // bucket 1 (<=50)
        histogram.observe(75.0); // bucket 2 (<=100)
        
        assert_eq!(histogram.bucket_count(0), 2);
        assert_eq!(histogram.bucket_count(1), 1);
        assert_eq!(histogram.bucket_count(2), 1);
    }

    #[test]
    fn test_histogram_ascii() {
        let mut histogram = LatencyHistogram::new();
        for _ in 0..10 {
            histogram.observe(5.0);
        }
        for _ in 0..5 {
            histogram.observe(25.0);
        }
        
        let ascii = histogram.ascii_histogram(20);
        assert!(ascii.contains("█"));
        assert!(ascii.len() > 0);
    }

    #[test]
    fn test_rate_tracker() {
        let mut tracker = RateTracker::new(1.0);
        
        // Record some events
        for _ in 0..10 {
            tracker.record();
        }
        
        assert_eq!(tracker.total_events(), 10);
        // Rate should be >= 0 (could vary based on timing)
        assert!(tracker.rate() >= 0.0);
    }

    #[test]
    fn test_rate_tracker_weighted() {
        let mut tracker = RateTracker::new(10.0);
        
        tracker.record_with_value(100.0);
        tracker.record_with_value(200.0);
        tracker.record_with_value(300.0);
        
        // Total value is 600 over 10 second window = 60 per second
        let weighted = tracker.weighted_rate();
        assert!(weighted > 0.0);
    }
}
