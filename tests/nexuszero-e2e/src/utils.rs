// Common utilities for E2E testing
//
// Provides test fixtures, helpers, and mock data generation

use std::time::{Duration, Instant};

/// Test fixture for measuring execution time
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn elapsed_ms(&self) -> u128 {
        self.elapsed().as_millis()
    }

    pub fn elapsed_secs(&self) -> u64 {
        self.elapsed().as_secs()
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate random test data
pub fn generate_random_bytes(size: usize) -> Vec<u8> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

/// Generate deterministic test data (for reproducibility)
pub fn generate_deterministic_bytes(size: usize, seed: u64) -> Vec<u8> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen()).collect()
}

/// Test result aggregator
#[derive(Debug, Clone, Default)]
pub struct TestMetrics {
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub total_duration: Duration,
    pub avg_duration: Duration,
}

impl TestMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_result(&mut self, passed: bool, duration: Duration) {
        self.total_tests += 1;
        if passed {
            self.passed += 1;
        } else {
            self.failed += 1;
        }
        self.total_duration += duration;
        self.avg_duration = self.total_duration / self.total_tests as u32;
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_tests == 0 {
            return 0.0;
        }
        (self.passed as f64 / self.total_tests as f64) * 100.0
    }

    pub fn summary(&self) -> String {
        format!(
            "Total: {} | Passed: {} | Failed: {} | Success Rate: {:.2}% | Avg Duration: {:?}",
            self.total_tests,
            self.passed,
            self.failed,
            self.success_rate(),
            self.avg_duration
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer() {
        let timer = Timer::new();
        std::thread::sleep(Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10);
    }

    #[test]
    fn test_random_bytes_generation() {
        let data1 = generate_random_bytes(100);
        let data2 = generate_random_bytes(100);
        assert_eq!(data1.len(), 100);
        assert_eq!(data2.len(), 100);
        // Random data should be different
        assert_ne!(data1, data2);
    }

    #[test]
    fn test_deterministic_bytes_generation() {
        let data1 = generate_deterministic_bytes(100, 42);
        let data2 = generate_deterministic_bytes(100, 42);
        let data3 = generate_deterministic_bytes(100, 43);
        assert_eq!(data1, data2); // Same seed = same data
        assert_ne!(data1, data3); // Different seed = different data
    }

    #[test]
    fn test_metrics_tracker() {
        let mut metrics = TestMetrics::new();
        
        metrics.add_result(true, Duration::from_millis(10));
        metrics.add_result(true, Duration::from_millis(20));
        metrics.add_result(false, Duration::from_millis(15));
        
        assert_eq!(metrics.total_tests, 3);
        assert_eq!(metrics.passed, 2);
        assert_eq!(metrics.failed, 1);
        assert!((metrics.success_rate() - 66.67).abs() < 0.1);
    }
}
