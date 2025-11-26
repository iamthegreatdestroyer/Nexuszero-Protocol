//! Risk Calculator Service

use crate::models::*;
use std::collections::HashMap;

/// Service for risk calculation
pub struct RiskCalculator {
    weights: HashMap<String, f64>,
}

impl Default for RiskCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskCalculator {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("geographic".to_string(), 0.20);
        weights.insert("behavioral".to_string(), 0.25);
        weights.insert("identity".to_string(), 0.20);
        weights.insert("relationship".to_string(), 0.15);
        weights.insert("transaction".to_string(), 0.20);
        
        Self { weights }
    }

    /// Calculate overall risk score from factors
    pub fn calculate_score(&self, factors: &[RiskFactor]) -> f64 {
        if factors.is_empty() {
            return 0.0;
        }

        let total_weight: f64 = factors.iter().map(|f| f.weight).sum();
        let weighted_score: f64 = factors.iter()
            .map(|f| f.score * f.weight)
            .sum();

        if total_weight > 0.0 {
            (weighted_score / total_weight).min(1.0)
        } else {
            0.0
        }
    }

    /// Get risk level from score
    pub fn get_risk_level(&self, score: f64) -> RiskLevel {
        RiskLevel::from_score(score)
    }

    /// Update factor weights
    pub fn update_weights(&mut self, category: &str, weight: f64) {
        self.weights.insert(category.to_string(), weight.min(1.0).max(0.0));
    }
}
