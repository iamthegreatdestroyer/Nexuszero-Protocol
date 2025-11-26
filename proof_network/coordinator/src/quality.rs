//! Quality Verification for Proof Results
//!
//! Provides verification of generated proofs and quality scoring.
//! Includes stub implementations for proof validation logic.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Minimum acceptable proof length in bytes
const MIN_PROOF_LENGTH: usize = 32;

/// Maximum acceptable proof length in bytes
const MAX_PROOF_LENGTH: usize = 1024 * 1024; // 1 MB

/// Result of proof verification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerificationResult {
    /// Proof passed verification with a quality score (0.0 - 1.0)
    Pass { quality_score: f64 },
    /// Proof failed verification with a reason
    Fail { reason: String },
}

impl VerificationResult {
    /// Check if verification passed
    pub fn is_pass(&self) -> bool {
        matches!(self, VerificationResult::Pass { .. })
    }

    /// Get quality score if passed, None if failed
    pub fn quality_score(&self) -> Option<f64> {
        match self {
            VerificationResult::Pass { quality_score } => Some(*quality_score),
            VerificationResult::Fail { .. } => None,
        }
    }

    /// Get failure reason if failed, None if passed
    pub fn failure_reason(&self) -> Option<&str> {
        match self {
            VerificationResult::Pass { .. } => None,
            VerificationResult::Fail { reason } => Some(reason),
        }
    }
}

/// Proof result submitted by a prover
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofResult {
    /// Result identifier
    pub result_id: Uuid,
    /// Associated task identifier
    pub task_id: Uuid,
    /// Prover who generated the proof
    pub prover_id: Uuid,
    /// The generated proof bytes
    pub proof: Vec<u8>,
    /// Optional metadata about the proof
    pub metadata: Option<ProofMetadata>,
}

/// Metadata about a generated proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Algorithm used for proof generation
    pub algorithm: String,
    /// Privacy level of the proof
    pub privacy_level: u8,
}

/// Configuration for the quality verifier
#[derive(Debug, Clone)]
pub struct VerifierConfig {
    /// Minimum acceptable proof length
    pub min_proof_length: usize,
    /// Maximum acceptable proof length
    pub max_proof_length: usize,
    /// Whether to perform deep verification (more expensive)
    pub deep_verification: bool,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            min_proof_length: MIN_PROOF_LENGTH,
            max_proof_length: MAX_PROOF_LENGTH,
            deep_verification: false,
        }
    }
}

/// Quality verifier for proof results
#[derive(Debug, Clone)]
pub struct QualityVerifier {
    config: VerifierConfig,
}

impl Default for QualityVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl QualityVerifier {
    /// Create a new quality verifier with default configuration
    pub fn new() -> Self {
        Self {
            config: VerifierConfig::default(),
        }
    }

    /// Create a new quality verifier with custom configuration
    pub fn with_config(config: VerifierConfig) -> Self {
        Self { config }
    }

    /// Verify a proof result and return a verification result
    ///
    /// This is a stub implementation that performs basic checks:
    /// - Proof length within acceptable bounds
    /// - Proof is not all zeros
    /// - Basic structure validation
    ///
    /// # Arguments
    /// * `proof_result` - The proof result to verify
    ///
    /// # Returns
    /// `VerificationResult::Pass` with quality score or `VerificationResult::Fail` with reason
    pub fn verify_proof(&self, proof_result: &ProofResult) -> VerificationResult {
        let proof = &proof_result.proof;

        // Check minimum length
        if proof.len() < self.config.min_proof_length {
            return VerificationResult::Fail {
                reason: format!(
                    "Proof too short: {} bytes (minimum: {})",
                    proof.len(),
                    self.config.min_proof_length
                ),
            };
        }

        // Check maximum length
        if proof.len() > self.config.max_proof_length {
            return VerificationResult::Fail {
                reason: format!(
                    "Proof too long: {} bytes (maximum: {})",
                    proof.len(),
                    self.config.max_proof_length
                ),
            };
        }

        // Check for empty/zero proof
        if proof.iter().all(|&b| b == 0) {
            return VerificationResult::Fail {
                reason: "Proof contains only zeros".to_string(),
            };
        }

        // Calculate quality score based on various factors
        let quality_score = self.calculate_quality_score(proof_result);

        // Check minimum quality threshold
        const MIN_QUALITY_THRESHOLD: f64 = 0.1;
        if quality_score < MIN_QUALITY_THRESHOLD {
            return VerificationResult::Fail {
                reason: format!(
                    "Quality score too low: {:.2} (minimum: {:.2})",
                    quality_score, MIN_QUALITY_THRESHOLD
                ),
            };
        }

        VerificationResult::Pass { quality_score }
    }

    /// Calculate a quality score for the proof
    ///
    /// This is a stub implementation that uses simple heuristics:
    /// - Entropy of the proof data
    /// - Proof length relative to optimal range
    /// - Presence of metadata
    fn calculate_quality_score(&self, proof_result: &ProofResult) -> f64 {
        let proof = &proof_result.proof;
        let mut score = 0.0;

        // Base score for valid proof (0.5)
        score += 0.5;

        // Entropy bonus (up to 0.2)
        let entropy = self.calculate_entropy(proof);
        score += entropy * 0.2;

        // Length score (up to 0.2)
        // Optimal length is considered to be around 256-1024 bytes
        let length_score = if proof.len() >= 256 && proof.len() <= 1024 {
            1.0
        } else if proof.len() >= 128 && proof.len() <= 2048 {
            0.8
        } else if proof.len() >= 64 && proof.len() <= 4096 {
            0.6
        } else {
            0.4
        };
        score += length_score * 0.2;

        // Metadata bonus (0.1 if present)
        if proof_result.metadata.is_some() {
            score += 0.1;
        }

        // Clamp to [0, 1]
        score.clamp(0.0, 1.0)
    }

    /// Calculate normalized entropy of proof data
    fn calculate_entropy(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        // Count byte frequencies
        let mut freq = [0u64; 256];
        for &byte in data {
            freq[byte as usize] += 1;
        }

        // Calculate Shannon entropy
        let len = data.len() as f64;
        let mut entropy = 0.0;

        for &count in &freq {
            if count > 0 {
                let p = count as f64 / len;
                entropy -= p * p.log2();
            }
        }

        // Normalize to [0, 1] (max entropy for bytes is 8 bits)
        (entropy / 8.0).clamp(0.0, 1.0)
    }

    /// Perform batch verification of multiple proofs
    pub fn verify_batch(&self, proofs: &[ProofResult]) -> Vec<(Uuid, VerificationResult)> {
        proofs
            .iter()
            .map(|p| (p.result_id, self.verify_proof(p)))
            .collect()
    }

    /// Get verification statistics for a batch of proofs
    pub fn get_batch_stats(&self, proofs: &[ProofResult]) -> BatchVerificationStats {
        let results = self.verify_batch(proofs);
        let total = results.len();
        let passed = results.iter().filter(|(_, r)| r.is_pass()).count();
        let failed = total - passed;

        let avg_quality = if passed > 0 {
            results
                .iter()
                .filter_map(|(_, r)| r.quality_score())
                .sum::<f64>()
                / passed as f64
        } else {
            0.0
        };

        BatchVerificationStats {
            total,
            passed,
            failed,
            pass_rate: if total > 0 {
                passed as f64 / total as f64
            } else {
                0.0
            },
            average_quality_score: avg_quality,
        }
    }
}

/// Statistics from batch verification
#[derive(Debug, Clone)]
pub struct BatchVerificationStats {
    /// Total number of proofs verified
    pub total: usize,
    /// Number of proofs that passed
    pub passed: usize,
    /// Number of proofs that failed
    pub failed: usize,
    /// Pass rate (0.0 - 1.0)
    pub pass_rate: f64,
    /// Average quality score of passed proofs
    pub average_quality_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_proof(proof_data: Vec<u8>) -> ProofResult {
        ProofResult {
            result_id: Uuid::new_v4(),
            task_id: Uuid::new_v4(),
            prover_id: Uuid::new_v4(),
            proof: proof_data,
            metadata: None,
        }
    }

    fn create_random_proof(size: usize) -> ProofResult {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Generate pseudo-random bytes
        let mut hasher = DefaultHasher::new();
        let mut proof = Vec::with_capacity(size);
        for i in 0..size {
            i.hash(&mut hasher);
            proof.push((hasher.finish() % 256) as u8);
        }

        create_test_proof(proof)
    }

    #[test]
    fn test_verify_proof_pass() {
        let verifier = QualityVerifier::new();
        let proof = create_random_proof(256);

        let result = verifier.verify_proof(&proof);
        assert!(result.is_pass());
        assert!(result.quality_score().unwrap() > 0.5);
    }

    #[test]
    fn test_verify_proof_too_short() {
        let verifier = QualityVerifier::new();
        let proof = create_test_proof(vec![1, 2, 3, 4]); // Only 4 bytes

        let result = verifier.verify_proof(&proof);
        assert!(!result.is_pass());
        assert!(result.failure_reason().unwrap().contains("too short"));
    }

    #[test]
    fn test_verify_proof_too_long() {
        let config = VerifierConfig {
            max_proof_length: 100,
            ..Default::default()
        };
        let verifier = QualityVerifier::with_config(config);
        let proof = create_random_proof(200);

        let result = verifier.verify_proof(&proof);
        assert!(!result.is_pass());
        assert!(result.failure_reason().unwrap().contains("too long"));
    }

    #[test]
    fn test_verify_proof_all_zeros() {
        let verifier = QualityVerifier::new();
        let proof = create_test_proof(vec![0u8; 100]);

        let result = verifier.verify_proof(&proof);
        assert!(!result.is_pass());
        assert!(result.failure_reason().unwrap().contains("only zeros"));
    }

    #[test]
    fn test_verification_result_methods() {
        let pass = VerificationResult::Pass { quality_score: 0.85 };
        assert!(pass.is_pass());
        assert_eq!(pass.quality_score(), Some(0.85));
        assert_eq!(pass.failure_reason(), None);

        let fail = VerificationResult::Fail {
            reason: "test failure".to_string(),
        };
        assert!(!fail.is_pass());
        assert_eq!(fail.quality_score(), None);
        assert_eq!(fail.failure_reason(), Some("test failure"));
    }

    #[test]
    fn test_quality_score_with_metadata() {
        let verifier = QualityVerifier::new();

        let mut proof_with_meta = create_random_proof(256);
        proof_with_meta.metadata = Some(ProofMetadata {
            generation_time_ms: 100,
            algorithm: "groth16".to_string(),
            privacy_level: 3,
        });

        let mut proof_without_meta = create_random_proof(256);
        proof_without_meta.metadata = None;

        let result_with = verifier.verify_proof(&proof_with_meta);
        let result_without = verifier.verify_proof(&proof_without_meta);

        // Proof with metadata should have higher quality score
        assert!(result_with.quality_score().unwrap() >= result_without.quality_score().unwrap());
    }

    #[test]
    fn test_batch_verification() {
        let verifier = QualityVerifier::new();

        let proofs = vec![
            create_random_proof(256),    // Should pass
            create_random_proof(128),    // Should pass
            create_test_proof(vec![0; 50]), // Should fail (all zeros)
            create_test_proof(vec![1, 2, 3]), // Should fail (too short)
        ];

        let results = verifier.verify_batch(&proofs);
        assert_eq!(results.len(), 4);

        let stats = verifier.get_batch_stats(&proofs);
        assert_eq!(stats.total, 4);
        assert_eq!(stats.passed, 2);
        assert_eq!(stats.failed, 2);
        assert!((stats.pass_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_entropy_calculation() {
        let verifier = QualityVerifier::new();

        // All same bytes - low entropy
        let low_entropy_proof = create_test_proof(vec![42u8; 100]);
        let low_result = verifier.verify_proof(&low_entropy_proof);

        // Random-ish bytes - higher entropy
        let high_entropy_proof = create_random_proof(100);
        let high_result = verifier.verify_proof(&high_entropy_proof);

        // Higher entropy should give higher quality score
        // Note: low entropy proof might fail if quality score is too low
        if let (VerificationResult::Pass { quality_score: low_score }, 
                VerificationResult::Pass { quality_score: high_score }) = 
            (&low_result, &high_result) 
        {
            assert!(high_score >= low_score);
        }
    }

    #[test]
    fn test_custom_config() {
        let config = VerifierConfig {
            min_proof_length: 64,
            max_proof_length: 512,
            deep_verification: true,
        };
        let verifier = QualityVerifier::with_config(config);

        // 50 bytes - below custom minimum
        let short_proof = create_random_proof(50);
        assert!(!verifier.verify_proof(&short_proof).is_pass());

        // 100 bytes - within custom range
        let valid_proof = create_random_proof(100);
        assert!(verifier.verify_proof(&valid_proof).is_pass());
    }
}
