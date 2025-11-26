//! Reputation System for Proof Marketplace
//!
//! Tracks prover performance metrics including success rates,
//! quality scores, and generation times.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a prover's reputation in the marketplace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverReputation {
    /// Unique identifier for the prover
    pub prover_id: String,
    /// Total number of proofs attempted
    pub total_proofs: u64,
    /// Number of successfully completed proofs
    pub successful_proofs: u64,
    /// Average quality score (0.0 - 1.0)
    pub average_quality_score: f64,
    /// Average time to generate proofs in milliseconds
    pub average_generation_time_ms: u64,
    /// Timestamp of last reputation update
    pub last_updated: DateTime<Utc>,
}

impl ProverReputation {
    /// Creates a new reputation record for a prover
    pub fn new(prover_id: String) -> Self {
        Self {
            prover_id,
            total_proofs: 0,
            successful_proofs: 0,
            average_quality_score: 0.0,
            average_generation_time_ms: 0,
            last_updated: Utc::now(),
        }
    }

    /// Calculates the success rate as a ratio (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_proofs == 0 {
            0.0
        } else {
            self.successful_proofs as f64 / self.total_proofs as f64
        }
    }

    /// Calculates an overall reputation score (0.0 - 1.0)
    /// Weighted: 50% success rate, 30% quality score, 20% time efficiency
    pub fn overall_score(&self) -> f64 {
        let success_weight = 0.5;
        let quality_weight = 0.3;
        let time_weight = 0.2;

        // Time efficiency: faster is better, normalize assuming 10s is baseline
        let time_efficiency = if self.average_generation_time_ms == 0 {
            1.0
        } else {
            (10_000.0 / self.average_generation_time_ms as f64).min(1.0)
        };

        (self.success_rate() * success_weight)
            + (self.average_quality_score * quality_weight)
            + (time_efficiency * time_weight)
    }
}

/// Tracks and manages reputation for all provers in the marketplace
#[derive(Debug, Default)]
pub struct ReputationTracker {
    /// Storage for all prover reputations
    reputations: HashMap<String, ProverReputation>,
}

impl ReputationTracker {
    /// Creates a new empty reputation tracker
    pub fn new() -> Self {
        Self {
            reputations: HashMap::new(),
        }
    }

    /// Updates a prover's reputation based on proof completion
    ///
    /// # Arguments
    /// * `prover_id` - Unique identifier for the prover
    /// * `success` - Whether the proof was successful
    /// * `quality_score` - Quality rating for the proof (0.0 - 1.0)
    /// * `time_ms` - Time taken to generate the proof in milliseconds
    pub fn update_reputation(
        &mut self,
        prover_id: &str,
        success: bool,
        quality_score: f64,
        time_ms: u64,
    ) {
        let reputation = self
            .reputations
            .entry(prover_id.to_string())
            .or_insert_with(|| ProverReputation::new(prover_id.to_string()));

        let old_total = reputation.total_proofs;
        reputation.total_proofs += 1;

        if success {
            reputation.successful_proofs += 1;
        }

        // Update running average for quality score
        if old_total == 0 {
            reputation.average_quality_score = quality_score;
        } else {
            reputation.average_quality_score = (reputation.average_quality_score * old_total as f64
                + quality_score)
                / reputation.total_proofs as f64;
        }

        // Update running average for generation time
        if old_total == 0 {
            reputation.average_generation_time_ms = time_ms;
        } else {
            reputation.average_generation_time_ms = ((reputation.average_generation_time_ms
                * old_total)
                + time_ms)
                / reputation.total_proofs;
        }

        reputation.last_updated = Utc::now();
    }

    /// Retrieves the reputation for a specific prover
    pub fn get_reputation(&self, prover_id: &str) -> Option<&ProverReputation> {
        self.reputations.get(prover_id)
    }

    /// Returns the top provers sorted by overall reputation score
    ///
    /// # Arguments
    /// * `limit` - Maximum number of provers to return
    pub fn get_top_provers(&self, limit: usize) -> Vec<&ProverReputation> {
        let mut provers: Vec<_> = self.reputations.values().collect();
        provers.sort_by(|a, b| {
            b.overall_score()
                .partial_cmp(&a.overall_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        provers.into_iter().take(limit).collect()
    }

    /// Returns the number of tracked provers
    pub fn prover_count(&self) -> usize {
        self.reputations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prover_reputation_new() {
        let rep = ProverReputation::new("prover1".to_string());
        assert_eq!(rep.prover_id, "prover1");
        assert_eq!(rep.total_proofs, 0);
        assert_eq!(rep.successful_proofs, 0);
        assert_eq!(rep.success_rate(), 0.0);
    }

    #[test]
    fn test_success_rate_calculation() {
        let mut rep = ProverReputation::new("prover1".to_string());
        rep.total_proofs = 10;
        rep.successful_proofs = 8;
        assert!((rep.success_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_reputation_tracker_update() {
        let mut tracker = ReputationTracker::new();

        // First update
        tracker.update_reputation("prover1", true, 0.9, 5000);

        let rep = tracker.get_reputation("prover1").unwrap();
        assert_eq!(rep.total_proofs, 1);
        assert_eq!(rep.successful_proofs, 1);
        assert!((rep.average_quality_score - 0.9).abs() < 0.001);
        assert_eq!(rep.average_generation_time_ms, 5000);

        // Second update
        tracker.update_reputation("prover1", true, 0.7, 3000);

        let rep = tracker.get_reputation("prover1").unwrap();
        assert_eq!(rep.total_proofs, 2);
        assert_eq!(rep.successful_proofs, 2);
        assert!((rep.average_quality_score - 0.8).abs() < 0.001);
        assert_eq!(rep.average_generation_time_ms, 4000);
    }

    #[test]
    fn test_reputation_tracker_failure() {
        let mut tracker = ReputationTracker::new();

        tracker.update_reputation("prover1", true, 0.9, 5000);
        tracker.update_reputation("prover1", false, 0.0, 10000);

        let rep = tracker.get_reputation("prover1").unwrap();
        assert_eq!(rep.total_proofs, 2);
        assert_eq!(rep.successful_proofs, 1);
        assert!((rep.success_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_get_top_provers() {
        let mut tracker = ReputationTracker::new();

        // Prover 1: high success, high quality
        tracker.update_reputation("prover1", true, 0.95, 3000);
        tracker.update_reputation("prover1", true, 0.90, 3500);

        // Prover 2: medium success, medium quality
        tracker.update_reputation("prover2", true, 0.7, 5000);
        tracker.update_reputation("prover2", false, 0.3, 8000);

        // Prover 3: low success, low quality
        tracker.update_reputation("prover3", false, 0.2, 15000);

        let top = tracker.get_top_provers(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].prover_id, "prover1");
        assert_eq!(top[1].prover_id, "prover2");
    }

    #[test]
    fn test_overall_score() {
        let mut rep = ProverReputation::new("prover1".to_string());
        rep.total_proofs = 10;
        rep.successful_proofs = 10; // 100% success
        rep.average_quality_score = 1.0; // Perfect quality
        rep.average_generation_time_ms = 5000; // 5s (good time)

        // Success: 1.0 * 0.5 = 0.5
        // Quality: 1.0 * 0.3 = 0.3
        // Time: (10000/5000).min(1.0) * 0.2 = 1.0 * 0.2 = 0.2
        // Total: 1.0
        assert!((rep.overall_score() - 1.0).abs() < 0.001);
    }
}
