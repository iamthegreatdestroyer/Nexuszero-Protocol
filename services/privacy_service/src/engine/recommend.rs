//! Privacy Level Recommendation Engine
//! Analyzes transaction patterns and context to recommend optimal privacy levels

use crate::error::Result;
use crate::models::PrivacyLevel;
use std::collections::HashMap;

/// Context for privacy recommendation
#[derive(Debug, Clone)]
pub struct RecommendationContext {
    /// Transaction amount (in smallest unit)
    pub amount: Option<u64>,
    /// Transaction type
    pub tx_type: Option<String>,
    /// Sender's historical privacy preferences
    pub sender_history: Option<Vec<PrivacyLevel>>,
    /// Recipient requirements
    pub recipient_requirements: Option<PrivacyLevel>,
    /// Regulatory jurisdiction
    pub jurisdiction: Option<String>,
    /// Time sensitivity
    pub time_sensitive: bool,
    /// Custom factors
    pub custom_factors: HashMap<String, String>,
}

impl Default for RecommendationContext {
    fn default() -> Self {
        Self {
            amount: None,
            tx_type: None,
            sender_history: None,
            recipient_requirements: None,
            jurisdiction: None,
            time_sensitive: false,
            custom_factors: HashMap::new(),
        }
    }
}

/// Privacy recommendation result
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// Recommended privacy level
    pub level: PrivacyLevel,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Reasons for recommendation
    pub reasons: Vec<String>,
    /// Alternative levels with scores
    pub alternatives: Vec<(PrivacyLevel, f64)>,
    /// Warnings or notes
    pub warnings: Vec<String>,
}

/// Recommendation engine for optimal privacy level selection
pub struct RecommendEngine {
    /// Amount thresholds for privacy levels
    amount_thresholds: Vec<(u64, PrivacyLevel)>,
    /// Jurisdiction requirements
    jurisdiction_requirements: HashMap<String, PrivacyLevel>,
    /// Transaction type recommendations
    tx_type_recommendations: HashMap<String, PrivacyLevel>,
}

impl Default for RecommendEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RecommendEngine {
    pub fn new() -> Self {
        let mut amount_thresholds = vec![
            (1_000_000_000, PrivacyLevel::Maximum),      // > 1B: Maximum
            (100_000_000, PrivacyLevel::Enhanced),       // > 100M: Enhanced
            (10_000_000, PrivacyLevel::Standard),        // > 10M: Standard
            (1_000_000, PrivacyLevel::Minimal),          // > 1M: Minimal
            (0, PrivacyLevel::Transparent),              // Default: Transparent
        ];
        amount_thresholds.sort_by(|a, b| b.0.cmp(&a.0));

        let mut jurisdiction_requirements = HashMap::new();
        jurisdiction_requirements.insert("EU".to_string(), PrivacyLevel::Standard);
        jurisdiction_requirements.insert("GDPR".to_string(), PrivacyLevel::Enhanced);
        jurisdiction_requirements.insert("FINMA".to_string(), PrivacyLevel::Standard);
        jurisdiction_requirements.insert("SEC".to_string(), PrivacyLevel::Minimal);

        let mut tx_type_recommendations = HashMap::new();
        tx_type_recommendations.insert("payroll".to_string(), PrivacyLevel::Enhanced);
        tx_type_recommendations.insert("donation".to_string(), PrivacyLevel::Maximum);
        tx_type_recommendations.insert("healthcare".to_string(), PrivacyLevel::Maximum);
        tx_type_recommendations.insert("voting".to_string(), PrivacyLevel::Quantum);
        tx_type_recommendations.insert("standard".to_string(), PrivacyLevel::Standard);
        tx_type_recommendations.insert("micro".to_string(), PrivacyLevel::Minimal);

        Self {
            amount_thresholds,
            jurisdiction_requirements,
            tx_type_recommendations,
        }
    }

    /// Generate privacy level recommendation
    pub async fn recommend(&self, context: &RecommendationContext) -> Result<Recommendation> {
        let mut scores: HashMap<PrivacyLevel, f64> = HashMap::new();
        let mut reasons: Vec<String> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();

        // Initialize all levels with base score
        for level in [
            PrivacyLevel::Transparent,
            PrivacyLevel::Minimal,
            PrivacyLevel::Standard,
            PrivacyLevel::Enhanced,
            PrivacyLevel::Maximum,
            PrivacyLevel::Quantum,
        ] {
            scores.insert(level, 0.0);
        }

        // Factor 1: Amount-based recommendation
        if let Some(amount) = context.amount {
            let level = self.recommend_by_amount(amount);
            *scores.entry(level.clone()).or_insert(0.0) += 0.25;
            reasons.push(format!(
                "Amount {} suggests {:?} privacy",
                amount, level
            ));
        }

        // Factor 2: Transaction type
        if let Some(ref tx_type) = context.tx_type {
            if let Some(level) = self.tx_type_recommendations.get(tx_type) {
                *scores.entry(level.clone()).or_insert(0.0) += 0.30;
                reasons.push(format!(
                    "Transaction type '{}' recommends {:?} privacy",
                    tx_type, level
                ));
            }
        }

        // Factor 3: Jurisdiction requirements
        if let Some(ref jurisdiction) = context.jurisdiction {
            if let Some(level) = self.jurisdiction_requirements.get(jurisdiction) {
                *scores.entry(level.clone()).or_insert(0.0) += 0.20;
                reasons.push(format!(
                    "Jurisdiction '{}' requires minimum {:?} privacy",
                    jurisdiction, level
                ));
                
                // Add warning if below minimum
                warnings.push(format!(
                    "Regulatory requirement: minimum {:?} for {}",
                    level, jurisdiction
                ));
            }
        }

        // Factor 4: Recipient requirements
        if let Some(ref required) = context.recipient_requirements {
            *scores.entry(required.clone()).or_insert(0.0) += 0.15;
            reasons.push(format!(
                "Recipient requires {:?} privacy",
                required
            ));
        }

        // Factor 5: Historical preferences
        if let Some(ref history) = context.sender_history {
            if !history.is_empty() {
                let preferred = self.analyze_history(history);
                *scores.entry(preferred.clone()).or_insert(0.0) += 0.10;
                reasons.push(format!(
                    "Historical preference suggests {:?} privacy",
                    preferred
                ));
            }
        }

        // Factor 6: Time sensitivity
        if context.time_sensitive {
            // Reduce score for slower privacy levels
            *scores.entry(PrivacyLevel::Maximum).or_insert(0.0) -= 0.10;
            *scores.entry(PrivacyLevel::Quantum).or_insert(0.0) -= 0.15;
            warnings.push("Time-sensitive transaction may be delayed with higher privacy".to_string());
        }

        // Find best recommendation
        let (recommended_level, max_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, v)| (k.clone(), *v))
            .unwrap_or((PrivacyLevel::Standard, 0.5));

        // Calculate confidence
        let total_score: f64 = scores.values().sum();
        let confidence = if total_score > 0.0 {
            (max_score / total_score).min(1.0)
        } else {
            0.5
        };

        // Generate alternatives
        let mut alternatives: Vec<(PrivacyLevel, f64)> = scores
            .into_iter()
            .filter(|(k, _)| k != &recommended_level)
            .collect();
        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        alternatives.truncate(3);

        // Default reason if none provided
        if reasons.is_empty() {
            reasons.push("Standard privacy level recommended as default".to_string());
        }

        Ok(Recommendation {
            level: recommended_level,
            confidence,
            reasons,
            alternatives,
            warnings,
        })
    }

    /// Recommend based on amount
    fn recommend_by_amount(&self, amount: u64) -> PrivacyLevel {
        for (threshold, level) in &self.amount_thresholds {
            if amount >= *threshold {
                return level.clone();
            }
        }
        PrivacyLevel::Transparent
    }

    /// Analyze historical preferences
    fn analyze_history(&self, history: &[PrivacyLevel]) -> PrivacyLevel {
        let mut counts: HashMap<PrivacyLevel, usize> = HashMap::new();
        
        for level in history {
            *counts.entry(level.clone()).or_insert(0) += 1;
        }
        
        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(level, _)| level)
            .unwrap_or(PrivacyLevel::Standard)
    }

    /// Get minimum required level for jurisdiction
    pub fn get_jurisdiction_minimum(&self, jurisdiction: &str) -> Option<PrivacyLevel> {
        self.jurisdiction_requirements.get(jurisdiction).cloned()
    }

    /// Validate if level meets requirements
    pub fn validate_level(
        &self,
        level: &PrivacyLevel,
        context: &RecommendationContext,
    ) -> Vec<String> {
        let mut violations = Vec::new();

        // Check jurisdiction requirements
        if let Some(ref jurisdiction) = context.jurisdiction {
            if let Some(required) = self.jurisdiction_requirements.get(jurisdiction) {
                if level.value() < required.value() {
                    violations.push(format!(
                        "Level {:?} below jurisdiction {} minimum of {:?}",
                        level, jurisdiction, required
                    ));
                }
            }
        }

        // Check recipient requirements
        if let Some(ref required) = context.recipient_requirements {
            if level.value() < required.value() {
                violations.push(format!(
                    "Level {:?} below recipient requirement of {:?}",
                    level, required
                ));
            }
        }

        violations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_recommend_by_amount() {
        let engine = RecommendEngine::new();
        
        let context = RecommendationContext {
            amount: Some(500_000_000), // 500M
            ..Default::default()
        };
        
        let rec = engine.recommend(&context).await.unwrap();
        assert!(matches!(rec.level, PrivacyLevel::Enhanced | PrivacyLevel::Maximum));
    }

    #[tokio::test]
    async fn test_recommend_by_tx_type() {
        let engine = RecommendEngine::new();
        
        let context = RecommendationContext {
            tx_type: Some("healthcare".to_string()),
            ..Default::default()
        };
        
        let rec = engine.recommend(&context).await.unwrap();
        assert!(matches!(rec.level, PrivacyLevel::Maximum | PrivacyLevel::Quantum));
    }

    #[tokio::test]
    async fn test_jurisdiction_requirement() {
        let engine = RecommendEngine::new();
        
        let context = RecommendationContext {
            jurisdiction: Some("GDPR".to_string()),
            ..Default::default()
        };
        
        let rec = engine.recommend(&context).await.unwrap();
        assert!(!rec.warnings.is_empty());
    }

    #[test]
    fn test_validate_level() {
        let engine = RecommendEngine::new();
        
        let context = RecommendationContext {
            jurisdiction: Some("GDPR".to_string()),
            ..Default::default()
        };
        
        let violations = engine.validate_level(&PrivacyLevel::Transparent, &context);
        assert!(!violations.is_empty());
        
        let no_violations = engine.validate_level(&PrivacyLevel::Enhanced, &context);
        assert!(no_violations.is_empty());
    }
}
