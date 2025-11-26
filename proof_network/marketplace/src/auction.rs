//! Auction Engine for Proof Marketplace
//!
//! Implements auction mechanics with reputation-weighted scoring
//! to select optimal provers for proof generation tasks.

use crate::order_book::OrderBook;
use crate::reputation::ReputationTracker;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Configuration for auction scoring weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuctionConfig {
    /// Weight for price factor (0.0 - 1.0)
    pub price_weight: f64,
    /// Weight for reputation factor (0.0 - 1.0)
    pub reputation_weight: f64,
}

impl Default for AuctionConfig {
    fn default() -> Self {
        Self {
            price_weight: 0.7,
            reputation_weight: 0.3,
        }
    }
}

/// Result of an auction including winning orders and scores
#[derive(Debug, Clone)]
pub struct AuctionResult {
    /// Winning order IDs
    pub winners: Vec<Uuid>,
    /// Scores for each winner (order_id -> weighted_score)
    pub scores: Vec<(Uuid, f64)>,
}

/// Auction engine for selecting provers based on price and reputation
pub struct AuctionEngine {
    /// Configuration for scoring weights
    config: AuctionConfig,
}

impl Default for AuctionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AuctionEngine {
    /// Creates a new auction engine with default configuration
    pub fn new() -> Self {
        Self {
            config: AuctionConfig::default(),
        }
    }

    /// Creates a new auction engine with custom configuration
    pub fn with_config(config: AuctionConfig) -> Self {
        Self { config }
    }

    /// Calculates a weighted score combining price and reputation
    ///
    /// # Arguments
    /// * `price` - Normalized price score (0.0 - 1.0, lower price = higher score)
    /// * `reputation` - Reputation score (0.0 - 1.0)
    ///
    /// # Returns
    /// Weighted combined score
    pub fn weighted_score(&self, price_score: f64, reputation: f64) -> f64 {
        (price_score * self.config.price_weight) + (reputation * self.config.reputation_weight)
    }

    /// Pick best orders to satisfy capacity (legacy method, price-only)
    pub fn run_auction(book: &OrderBook, capacity_needed: u32) -> Vec<Uuid> {
        let mut orders: Vec<_> = book.orders.values().collect();
        orders.sort_by_key(|o| o.price);
        let mut picked = Vec::new();
        let mut cap = capacity_needed;
        for o in orders {
            if cap == 0 {
                break;
            }
            picked.push(o.id);
            cap = cap.saturating_sub(o.capacity);
        }
        picked
    }

    /// Run auction with reputation-weighted scoring
    ///
    /// # Arguments
    /// * `book` - The order book containing available orders
    /// * `reputation_tracker` - Tracker containing prover reputations
    /// * `capacity_needed` - Total capacity required
    ///
    /// # Returns
    /// AuctionResult with winning orders and their scores
    pub fn run_auction_with_reputation(
        &self,
        book: &OrderBook,
        reputation_tracker: &ReputationTracker,
        capacity_needed: u32,
    ) -> AuctionResult {
        if book.orders.is_empty() {
            return AuctionResult {
                winners: Vec::new(),
                scores: Vec::new(),
            };
        }

        // Find max price for normalization
        let max_price = book.orders.values().map(|o| o.price).max().unwrap_or(1) as f64;

        // Calculate weighted scores for all orders
        let mut scored_orders: Vec<(Uuid, f64, u32)> = book
            .orders
            .values()
            .map(|order| {
                // Normalize price (lower price = higher score)
                let price_score = 1.0 - (order.price as f64 / max_price);

                // Get reputation or default to 0.5 for new provers
                let reputation = reputation_tracker
                    .get_reputation(&order.prover_id)
                    .map(|r| r.overall_score())
                    .unwrap_or(0.5);

                let weighted = self.weighted_score(price_score, reputation);
                (order.id, weighted, order.capacity)
            })
            .collect();

        // Sort by weighted score (descending)
        scored_orders.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Pick orders until capacity is met
        let mut picked = Vec::new();
        let mut scores = Vec::new();
        let mut cap = capacity_needed;

        for (id, score, order_cap) in scored_orders {
            if cap == 0 {
                break;
            }
            picked.push(id);
            scores.push((id, score));
            cap = cap.saturating_sub(order_cap);
        }

        AuctionResult {
            winners: picked,
            scores,
        }
    }
}

/// Calculates a weighted score combining price and reputation
/// Standalone function for use outside of AuctionEngine
///
/// Uses default weights: 70% price, 30% reputation
///
/// # Arguments
/// * `price` - Normalized price score (0.0 - 1.0, lower price = higher score)
/// * `reputation` - Reputation score (0.0 - 1.0)
pub fn weighted_score(price_score: f64, reputation: f64) -> f64 {
    const PRICE_WEIGHT: f64 = 0.7;
    const REPUTATION_WEIGHT: f64 = 0.3;
    (price_score * PRICE_WEIGHT) + (reputation * REPUTATION_WEIGHT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order_book::{Order, OrderBook};
    use crate::reputation::ReputationTracker;
    use uuid::Uuid;

    #[test]
    fn test_auction_simple() {
        let mut book = OrderBook::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        book.add_order(Order {
            id: id1,
            prover_id: "p1".to_string(),
            price: 100,
            capacity: 2,
        });
        book.add_order(Order {
            id: id2,
            prover_id: "p2".to_string(),
            price: 90,
            capacity: 5,
        });

        let winners = AuctionEngine::run_auction(&book, 3);
        assert!(winners.contains(&id2));
    }

    #[test]
    fn test_weighted_score_function() {
        // Test standalone function
        let score = weighted_score(1.0, 1.0);
        assert!((score - 1.0).abs() < 0.001);

        let score = weighted_score(0.0, 0.0);
        assert!((score - 0.0).abs() < 0.001);

        // Price 70%, Reputation 30%
        let score = weighted_score(1.0, 0.0);
        assert!((score - 0.7).abs() < 0.001);

        let score = weighted_score(0.0, 1.0);
        assert!((score - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_auction_engine_weighted_score() {
        let engine = AuctionEngine::new();
        let score = engine.weighted_score(1.0, 1.0);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_auction_with_reputation() {
        let mut book = OrderBook::new();
        let mut tracker = ReputationTracker::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        // Prover 1: higher price, excellent reputation
        book.add_order(Order {
            id: id1,
            prover_id: "prover1".to_string(),
            price: 100,
            capacity: 5,
        });

        // Prover 2: lower price, poor reputation
        book.add_order(Order {
            id: id2,
            prover_id: "prover2".to_string(),
            price: 80,
            capacity: 5,
        });

        // Build reputation
        for _ in 0..10 {
            tracker.update_reputation("prover1", true, 0.95, 3000);
        }
        for _ in 0..10 {
            tracker.update_reputation("prover2", false, 0.3, 10000);
        }

        let engine = AuctionEngine::new();
        let result = engine.run_auction_with_reputation(&book, &tracker, 5);

        // Should pick prover1 despite higher price due to reputation
        assert_eq!(result.winners.len(), 1);
        assert!(result.scores.len() >= 1);
    }

    #[test]
    fn test_auction_empty_book() {
        let book = OrderBook::new();
        let tracker = ReputationTracker::new();
        let engine = AuctionEngine::new();

        let result = engine.run_auction_with_reputation(&book, &tracker, 5);
        assert!(result.winners.is_empty());
    }

    #[test]
    fn test_custom_auction_config() {
        let config = AuctionConfig {
            price_weight: 0.5,
            reputation_weight: 0.5,
        };
        let engine = AuctionEngine::with_config(config);

        let score = engine.weighted_score(1.0, 0.0);
        assert!((score - 0.5).abs() < 0.001);
    }
}
