//! Privacy service integration

use crate::config::Config;
use crate::error::{Result, TransactionError};
use crate::models::PrivacyLevel;
use serde::{Deserialize, Serialize};

/// Privacy service client
pub struct PrivacyServiceClient {
    client: reqwest::Client,
    base_url: String,
}

/// Privacy recommendation response
#[derive(Debug, Serialize, Deserialize)]
pub struct PrivacyRecommendation {
    pub recommended_level: i16,
    pub confidence: f64,
    pub factors: Vec<RecommendationFactor>,
}

/// Factor influencing privacy recommendation
#[derive(Debug, Serialize, Deserialize)]
pub struct RecommendationFactor {
    pub name: String,
    pub impact: String,
    pub weight: f64,
}

impl PrivacyServiceClient {
    /// Create new privacy service client
    pub fn new(config: &Config) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: config.privacy_service_url.clone(),
        }
    }

    /// Get privacy level recommendation
    pub async fn get_recommendation(
        &self,
        sender: &str,
        recipient: &str,
        amount: i64,
        chain_id: &str,
    ) -> Result<PrivacyRecommendation> {
        let response = self
            .client
            .post(format!("{}/api/v1/privacy/recommend", self.base_url))
            .json(&serde_json::json!({
                "sender": sender,
                "recipient": recipient,
                "amount": amount,
                "chain_id": chain_id,
            }))
            .send()
            .await
            .map_err(|e| TransactionError::ExternalService(e.to_string()))?;

        if response.status().is_success() {
            response
                .json()
                .await
                .map_err(|e| TransactionError::ExternalService(e.to_string()))
        } else {
            // Fallback recommendation
            Ok(default_recommendation(amount))
        }
    }

    /// Validate privacy level for a transaction
    pub async fn validate_level(
        &self,
        level: i16,
        amount: i64,
        chain_id: &str,
    ) -> Result<bool> {
        // Basic validation
        if level < 0 || level > 5 {
            return Err(TransactionError::InvalidPrivacyLevel(level));
        }

        // Call privacy service for chain-specific validation
        let response = self
            .client
            .post(format!("{}/api/v1/privacy/validate", self.base_url))
            .json(&serde_json::json!({
                "privacy_level": level,
                "amount": amount,
                "chain_id": chain_id,
            }))
            .send()
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => {
                let result: serde_json::Value = resp.json().await.unwrap_or_default();
                Ok(result["valid"].as_bool().unwrap_or(true))
            }
            _ => Ok(true), // Assume valid if service unavailable
        }
    }

    /// Calculate privacy cost (in gas/fees)
    pub async fn calculate_privacy_cost(
        &self,
        level: i16,
        chain_id: &str,
    ) -> Result<PrivacyCost> {
        let response = self
            .client
            .get(format!(
                "{}/api/v1/privacy/cost?level={}&chain_id={}",
                self.base_url, level, chain_id
            ))
            .send()
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => resp
                .json()
                .await
                .map_err(|e| TransactionError::ExternalService(e.to_string())),
            _ => Ok(estimate_privacy_cost(level)),
        }
    }
}

/// Privacy cost estimate
#[derive(Debug, Serialize, Deserialize)]
pub struct PrivacyCost {
    pub level: i16,
    pub base_gas: u64,
    pub proof_gas: u64,
    pub total_gas: u64,
    pub estimated_fee_usd: f64,
}

/// Generate default recommendation based on amount
fn default_recommendation(amount: i64) -> PrivacyRecommendation {
    let level = if amount > 1_000_000_000_000 {
        // > 1 trillion: maximum privacy
        5
    } else if amount > 100_000_000_000 {
        // > 100 billion: full privacy
        4
    } else if amount > 10_000_000_000 {
        // > 10 billion: amount shielded
        3
    } else if amount > 1_000_000_000 {
        // > 1 billion: partial shielding
        2
    } else {
        // Default: full privacy for most transactions
        4
    };

    PrivacyRecommendation {
        recommended_level: level,
        confidence: 0.7,
        factors: vec![
            RecommendationFactor {
                name: "amount".to_string(),
                impact: "high".to_string(),
                weight: 0.6,
            },
            RecommendationFactor {
                name: "default_policy".to_string(),
                impact: "medium".to_string(),
                weight: 0.4,
            },
        ],
    }
}

/// Estimate privacy cost for a level
fn estimate_privacy_cost(level: i16) -> PrivacyCost {
    let base_gas: u64 = 21000;
    let proof_gas: u64 = match level {
        0 => 0,
        1 => 50_000,
        2 => 50_000,
        3 => 75_000,
        4 => 150_000,
        5 => 250_000,
        _ => 100_000,
    };

    PrivacyCost {
        level,
        base_gas,
        proof_gas,
        total_gas: base_gas + proof_gas,
        estimated_fee_usd: ((base_gas + proof_gas) as f64) * 0.000000001 * 2000.0, // Rough estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_recommendation() {
        let rec = default_recommendation(500_000_000_000); // 500 billion
        assert_eq!(rec.recommended_level, 4);

        let rec = default_recommendation(2_000_000_000_000); // 2 trillion
        assert_eq!(rec.recommended_level, 5);
    }

    #[test]
    fn test_estimate_privacy_cost() {
        let cost = estimate_privacy_cost(0);
        assert_eq!(cost.proof_gas, 0);

        let cost = estimate_privacy_cost(5);
        assert_eq!(cost.proof_gas, 250_000);
        assert!(cost.total_gas > cost.base_gas);
    }
}
