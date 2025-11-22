//! Side-channel testing framework
//!
//! Comprehensive toolkit for detecting timing, cache, and power analysis vulnerabilities.
//!
//! # Components
//!
//! 1. **Cache-Timing Attack Simulation**: Flush+Reload, Prime+Probe
//! 2. **Differential Timing Analysis**: Statistical tests (Welch's t-test, dudect)
//! 3. **Memory Access Pattern Analysis**: Track secret-dependent addressing
//!
//! # Integration
//!
//! Can be extended with hardware tools:
//! - ChipWhisperer for power/EM analysis
//! - Intel PT (Processor Trace) for execution flow
//! - Performance counters for micro-architectural leakage

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// Differential Timing Analysis (dudect-inspired)
// ============================================================================

/// Dudect-style constant-time verification using Welch's t-test
///
/// Compares timing distributions of two input classes to detect correlations.
/// A significant t-statistic indicates timing dependency on secret data.
pub struct DudectAnalyzer {
    /// Measurements for class 0 (e.g., secret bit = 0)
    class0_samples: Vec<Duration>,
    /// Measurements for class 1 (e.g., secret bit = 1)
    class1_samples: Vec<Duration>,
    /// Confidence threshold (typically 4.5 for high confidence)
    t_threshold: f64,
}

impl DudectAnalyzer {
    pub fn new() -> Self {
        Self {
            class0_samples: Vec::new(),
            class1_samples: Vec::new(),
            t_threshold: 4.5, // 99.999% confidence
        }
    }

    /// Add timing sample for class 0
    pub fn add_class0(&mut self, duration: Duration) {
        self.class0_samples.push(duration);
    }

    /// Add timing sample for class 1
    pub fn add_class1(&mut self, duration: Duration) {
        self.class1_samples.push(duration);
    }

    /// Compute Welch's t-statistic
    pub fn compute_t_statistic(&self) -> Option<f64> {
        if self.class0_samples.len() < 10 || self.class1_samples.len() < 10 {
            return None;
        }

        let mean0 = self.mean(&self.class0_samples);
        let mean1 = self.mean(&self.class1_samples);

        let var0 = self.variance(&self.class0_samples, mean0);
        let var1 = self.variance(&self.class1_samples, mean1);

        let n0 = self.class0_samples.len() as f64;
        let n1 = self.class1_samples.len() as f64;

        let se = ((var0 / n0) + (var1 / n1)).sqrt();

        if se == 0.0 {
            return None;
        }

        Some((mean0 - mean1) / se)
    }

    /// Check if timing is constant (t-statistic below threshold)
    pub fn is_constant_time(&self) -> bool {
        self.compute_t_statistic()
            .map(|t| t.abs() < self.t_threshold)
            .unwrap_or(false)
    }

    fn mean(&self, samples: &[Duration]) -> f64 {
        let sum: u128 = samples.iter().map(|d| d.as_nanos()).sum();
        sum as f64 / samples.len() as f64
    }

    fn variance(&self, samples: &[Duration], mean: f64) -> f64 {
        let sum_sq: f64 = samples
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean;
                diff * diff
            })
            .sum();
        sum_sq / samples.len() as f64
    }
}

// ============================================================================
// Cache-Timing Attack Simulation
// ============================================================================

/// Cache line size (typically 64 bytes)
const CACHE_LINE_SIZE: usize = 64;

/// Simulated cache state for Flush+Reload attacks
pub struct CacheSimulator {
    /// Tracks which cache lines are "hot" (recently accessed)
    cache_state: HashMap<usize, bool>,
    /// Access pattern log
    access_log: Vec<CacheAccess>,
}

#[derive(Debug, Clone)]
pub struct CacheAccess {
    pub address: usize,
    pub cache_line: usize,
    pub was_hit: bool,
    pub timestamp: Instant,
}

impl CacheSimulator {
    pub fn new() -> Self {
        Self {
            cache_state: HashMap::new(),
            access_log: Vec::new(),
        }
    }

    /// Simulate memory access and record cache behavior
    pub fn access(&mut self, address: usize) -> bool {
        let cache_line = address / CACHE_LINE_SIZE;
        let was_hit = self.cache_state.get(&cache_line).copied().unwrap_or(false);

        self.cache_state.insert(cache_line, true);
        self.access_log.push(CacheAccess {
            address,
            cache_line,
            was_hit,
            timestamp: Instant::now(),
        });

        was_hit
    }

    /// Flush cache line (Flush+Reload attack primitive)
    pub fn flush(&mut self, address: usize) {
        let cache_line = address / CACHE_LINE_SIZE;
        self.cache_state.remove(&cache_line);
    }

    /// Analyze access patterns for secret-dependent addressing
    pub fn analyze_patterns(&self) -> PatternAnalysis {
        let mut hit_rate = 0.0;
        let mut secret_dependent_lines = Vec::new();

        if !self.access_log.is_empty() {
            let hits = self.access_log.iter().filter(|a| a.was_hit).count();
            hit_rate = hits as f64 / self.access_log.len() as f64;
        }

        // Detect lines accessed non-uniformly (potential secret dependency)
        let mut line_counts: HashMap<usize, usize> = HashMap::new();
        for access in &self.access_log {
            *line_counts.entry(access.cache_line).or_insert(0) += 1;
        }

        let mean_count = self.access_log.len() as f64 / line_counts.len() as f64;
        let unique_lines_count = line_counts.len();
        
        for (line, count) in line_counts {
            let deviation = (count as f64 - mean_count).abs() / mean_count;
            if deviation > 0.5 {
                // More than 50% deviation from uniform
                secret_dependent_lines.push(line);
            }
        }

        PatternAnalysis {
            total_accesses: self.access_log.len(),
            cache_hit_rate: hit_rate,
            unique_cache_lines: unique_lines_count,
            suspicious_lines: secret_dependent_lines,
        }
    }

    /// Clear cache state and logs
    pub fn reset(&mut self) {
        self.cache_state.clear();
        self.access_log.clear();
    }
}

#[derive(Debug)]
pub struct PatternAnalysis {
    pub total_accesses: usize,
    pub cache_hit_rate: f64,
    pub unique_cache_lines: usize,
    pub suspicious_lines: Vec<usize>,
}

// ============================================================================
// Memory Access Tracer
// ============================================================================

/// Trace memory access patterns during cryptographic operations
pub struct MemoryTracer {
    /// Log of all memory accesses
    accesses: Vec<MemoryAccess>,
    /// Track if any secret-dependent indexing occurred
    secret_dependent: bool,
}

#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub address: usize,
    pub size: usize,
    pub access_type: AccessType,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
}

impl MemoryTracer {
    pub fn new() -> Self {
        Self {
            accesses: Vec::new(),
            secret_dependent: false,
        }
    }

    /// Record memory access
    pub fn record(&mut self, address: usize, size: usize, access_type: AccessType) {
        self.accesses.push(MemoryAccess {
            address,
            size,
            access_type,
            timestamp: Instant::now(),
        });
    }

    /// Flag that a secret-dependent memory access was detected
    pub fn mark_secret_dependent(&mut self) {
        self.secret_dependent = true;
    }

    /// Check if any secret-dependent accesses occurred
    pub fn has_secret_dependency(&self) -> bool {
        self.secret_dependent
    }

    /// Get total number of accesses
    pub fn access_count(&self) -> usize {
        self.accesses.len()
    }

    /// Analyze access patterns for regularity
    pub fn analyze_regularity(&self) -> f64 {
        if self.accesses.len() < 2 {
            return 1.0;
        }

        // Check if accesses are evenly spaced (constant-time indicator)
        let intervals: Vec<u128> = self
            .accesses
            .windows(2)
            .map(|w| w[1].timestamp.duration_since(w[0].timestamp).as_nanos())
            .collect();

        let mean: f64 = intervals.iter().map(|&i| i as f64).sum::<f64>() / intervals.len() as f64;

        let variance: f64 = intervals
            .iter()
            .map(|&i| {
                let diff = i as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / intervals.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };

        // Lower CoV = more regular timing (closer to constant-time)
        1.0 - coefficient_of_variation.min(1.0)
    }
}

// ============================================================================
// Integration Test Harness
// ============================================================================

/// Test a function for constant-time behavior using dudect
pub fn test_constant_time<F>(
    mut operation: F,
    num_samples: usize,
) -> DudectResult
where
    F: FnMut(bool) -> Duration,
{
    let mut analyzer = DudectAnalyzer::new();

    for _ in 0..num_samples {
        // Alternate between classes
        let class = rand::random::<bool>();
        let duration = operation(class);

        if class {
            analyzer.add_class1(duration);
        } else {
            analyzer.add_class0(duration);
        }
    }

    let t_stat = analyzer.compute_t_statistic();
    let is_ct = analyzer.is_constant_time();

    DudectResult {
        samples_per_class: num_samples / 2,
        t_statistic: t_stat,
        is_constant_time: is_ct,
        threshold: analyzer.t_threshold,
    }
}

#[derive(Debug)]
pub struct DudectResult {
    pub samples_per_class: usize,
    pub t_statistic: Option<f64>,
    pub is_constant_time: bool,
    pub threshold: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dudect_constant_function() {
        let mut analyzer = DudectAnalyzer::new();

        // Simulate constant-time operation (same duration for both classes)
        for _ in 0..100 {
            analyzer.add_class0(Duration::from_nanos(1000));
            analyzer.add_class1(Duration::from_nanos(1000));
        }

        assert!(analyzer.is_constant_time());
        let t = analyzer.compute_t_statistic().unwrap();
        assert!(t.abs() < 1.0);
    }

    #[test]
    fn test_dudect_variable_function() {
        let mut analyzer = DudectAnalyzer::new();

        // Simulate variable-time operation
        for _ in 0..100 {
            analyzer.add_class0(Duration::from_nanos(1000));
            analyzer.add_class1(Duration::from_nanos(2000)); // 2x slower
        }

        assert!(!analyzer.is_constant_time());
        let t = analyzer.compute_t_statistic().unwrap();
        assert!(t.abs() > 10.0); // Very significant difference
    }

    #[test]
    fn test_cache_simulator() {
        let mut cache = CacheSimulator::new();

        // First access is miss
        assert!(!cache.access(0x1000));

        // Second access to same line is hit
        assert!(cache.access(0x1000));

        // Flush and access again is miss
        cache.flush(0x1000);
        assert!(!cache.access(0x1000));
    }

    #[test]
    fn test_memory_tracer() {
        let mut tracer = MemoryTracer::new();

        tracer.record(0x1000, 8, AccessType::Read);
        tracer.record(0x2000, 8, AccessType::Write);

        assert_eq!(tracer.access_count(), 2);
        assert!(!tracer.has_secret_dependency());

        tracer.mark_secret_dependent();
        assert!(tracer.has_secret_dependency());
    }

    #[test]
    fn test_pattern_analysis() {
        let mut cache = CacheSimulator::new();

        // Uniform access pattern
        for i in 0..100 {
            cache.access(i * CACHE_LINE_SIZE);
        }

        let analysis = cache.analyze_patterns();
        assert_eq!(analysis.total_accesses, 100);
        assert_eq!(analysis.unique_cache_lines, 100);
        assert!(analysis.suspicious_lines.is_empty());
    }
}
