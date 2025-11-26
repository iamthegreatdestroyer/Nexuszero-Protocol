//! Order Book for Proof Marketplace
//!
//! Manages orders, bids (from task requesters), and asks (from provers)
//! with matching functionality.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A basic order in the marketplace (legacy support)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order identifier
    pub id: Uuid,
    /// Prover submitting the order
    pub prover_id: String,
    /// Price offered for proof generation
    pub price: u64,
    /// Proof generation capacity
    pub capacity: u32,
}

/// A bid from a task requester offering payment for proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bid {
    /// Unique bid identifier
    pub bid_id: Uuid,
    /// ID of the requester submitting the bid
    pub requester_id: String,
    /// Maximum price willing to pay
    pub max_price: u64,
    /// Priority level for the task (higher = more urgent)
    pub task_priority: u8,
    /// Deadline for proof completion
    pub deadline: DateTime<Utc>,
}

impl Bid {
    /// Creates a new bid
    pub fn new(requester_id: String, max_price: u64, task_priority: u8, deadline: DateTime<Utc>) -> Self {
        Self {
            bid_id: Uuid::new_v4(),
            requester_id,
            max_price,
            task_priority,
            deadline,
        }
    }
}

/// An ask from a prover offering proof generation services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ask {
    /// Unique ask identifier
    pub ask_id: Uuid,
    /// ID of the prover offering services
    pub prover_id: String,
    /// Minimum price accepted for proof generation
    pub min_price: u64,
    /// Available proof generation capacity
    pub capacity: u32,
    /// Prover's reputation score (0.0 - 1.0)
    pub reputation_score: f64,
}

impl Ask {
    /// Creates a new ask
    pub fn new(prover_id: String, min_price: u64, capacity: u32, reputation_score: f64) -> Self {
        Self {
            ask_id: Uuid::new_v4(),
            prover_id,
            min_price,
            capacity,
            reputation_score,
        }
    }
}

/// A matched pair of bid and ask
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Match {
    /// The matched bid
    pub bid: Bid,
    /// The matched ask
    pub ask: Ask,
    /// The agreed-upon price (midpoint between bid max and ask min)
    pub agreed_price: u64,
}

/// Order book for legacy orders
#[derive(Debug, Default)]
pub struct OrderBook {
    /// Stored orders indexed by ID
    pub orders: HashMap<Uuid, Order>,
}

impl OrderBook {
    /// Creates a new empty order book
    pub fn new() -> Self {
        Self {
            orders: HashMap::new(),
        }
    }

    /// Adds an order to the book
    pub fn add_order(&mut self, order: Order) {
        self.orders.insert(order.id, order);
    }

    /// Returns the best (lowest price) order
    pub fn best_order(&self) -> Option<&Order> {
        self.orders.values().min_by_key(|o| o.price)
    }

    /// Removes an order by ID
    pub fn remove_order(&mut self, id: &Uuid) -> Option<Order> {
        self.orders.remove(id)
    }
}

/// Bid-Ask order book for matching task requesters with provers
#[derive(Debug, Default)]
pub struct BidAskBook {
    /// Active bids from task requesters
    pub bids: Vec<Bid>,
    /// Active asks from provers
    pub asks: Vec<Ask>,
}

impl BidAskBook {
    /// Creates a new empty bid-ask book
    pub fn new() -> Self {
        Self {
            bids: Vec::new(),
            asks: Vec::new(),
        }
    }

    /// Adds a bid to the book
    pub fn add_bid(&mut self, bid: Bid) {
        self.bids.push(bid);
    }

    /// Adds an ask to the book
    pub fn add_ask(&mut self, ask: Ask) {
        self.asks.push(ask);
    }

    /// Removes a bid by ID
    pub fn remove_bid(&mut self, bid_id: &Uuid) -> Option<Bid> {
        if let Some(pos) = self.bids.iter().position(|b| b.bid_id == *bid_id) {
            Some(self.bids.remove(pos))
        } else {
            None
        }
    }

    /// Removes an ask by ID
    pub fn remove_ask(&mut self, ask_id: &Uuid) -> Option<Ask> {
        if let Some(pos) = self.asks.iter().position(|a| a.ask_id == *ask_id) {
            Some(self.asks.remove(pos))
        } else {
            None
        }
    }

    /// Matches bids with asks based on price overlap
    ///
    /// A match occurs when a bid's max_price >= an ask's min_price.
    /// Bids are sorted by priority (descending), then by max_price (descending).
    /// Asks are sorted by reputation (descending), then by min_price (ascending).
    ///
    /// Returns a vector of matches and removes matched items from the book.
    pub fn match_bids_asks(&mut self) -> Vec<Match> {
        // Sort bids by priority (desc) then by max_price (desc)
        self.bids.sort_by(|a, b| {
            b.task_priority
                .cmp(&a.task_priority)
                .then_with(|| b.max_price.cmp(&a.max_price))
        });

        // Sort asks by reputation (desc) then by min_price (asc)
        self.asks.sort_by(|a, b| {
            b.reputation_score
                .partial_cmp(&a.reputation_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.min_price.cmp(&b.min_price))
        });

        let mut matches = Vec::new();
        let mut matched_bid_ids = Vec::new();
        let mut matched_ask_ids = Vec::new();

        for bid in &self.bids {
            for ask in &self.asks {
                // Skip if already matched
                if matched_ask_ids.contains(&ask.ask_id) {
                    continue;
                }

                // Check price overlap
                if bid.max_price >= ask.min_price {
                    let agreed_price = (bid.max_price + ask.min_price) / 2;
                    matches.push(Match {
                        bid: bid.clone(),
                        ask: ask.clone(),
                        agreed_price,
                    });
                    matched_bid_ids.push(bid.bid_id);
                    matched_ask_ids.push(ask.ask_id);
                    break; // Move to next bid
                }
            }
        }

        // Remove matched items
        self.bids.retain(|b| !matched_bid_ids.contains(&b.bid_id));
        self.asks.retain(|a| !matched_ask_ids.contains(&a.ask_id));

        matches
    }

    /// Returns the number of active bids
    pub fn bid_count(&self) -> usize {
        self.bids.len()
    }

    /// Returns the number of active asks
    pub fn ask_count(&self) -> usize {
        self.asks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_bid_creation() {
        let deadline = Utc::now() + Duration::hours(1);
        let bid = Bid::new("requester1".to_string(), 100, 5, deadline);
        assert_eq!(bid.requester_id, "requester1");
        assert_eq!(bid.max_price, 100);
        assert_eq!(bid.task_priority, 5);
    }

    #[test]
    fn test_ask_creation() {
        let ask = Ask::new("prover1".to_string(), 50, 10, 0.95);
        assert_eq!(ask.prover_id, "prover1");
        assert_eq!(ask.min_price, 50);
        assert_eq!(ask.capacity, 10);
        assert!((ask.reputation_score - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_bid_ask_matching() {
        let mut book = BidAskBook::new();
        let deadline = Utc::now() + Duration::hours(1);

        // Add bids
        book.add_bid(Bid::new("req1".to_string(), 100, 5, deadline));
        book.add_bid(Bid::new("req2".to_string(), 80, 3, deadline));

        // Add asks
        book.add_ask(Ask::new("prover1".to_string(), 90, 5, 0.9));
        book.add_ask(Ask::new("prover2".to_string(), 70, 3, 0.8));

        let matches = book.match_bids_asks();

        // Both should match since bid max >= ask min
        assert_eq!(matches.len(), 2);
        assert_eq!(book.bid_count(), 0);
        assert_eq!(book.ask_count(), 0);
    }

    #[test]
    fn test_no_matching_prices() {
        let mut book = BidAskBook::new();
        let deadline = Utc::now() + Duration::hours(1);

        book.add_bid(Bid::new("req1".to_string(), 50, 5, deadline)); // Max 50
        book.add_ask(Ask::new("prover1".to_string(), 100, 5, 0.9)); // Min 100

        let matches = book.match_bids_asks();

        // No match: 50 < 100
        assert_eq!(matches.len(), 0);
        assert_eq!(book.bid_count(), 1);
        assert_eq!(book.ask_count(), 1);
    }

    #[test]
    fn test_agreed_price_calculation() {
        let mut book = BidAskBook::new();
        let deadline = Utc::now() + Duration::hours(1);

        book.add_bid(Bid::new("req1".to_string(), 100, 5, deadline));
        book.add_ask(Ask::new("prover1".to_string(), 80, 5, 0.9));

        let matches = book.match_bids_asks();

        assert_eq!(matches.len(), 1);
        // Agreed price should be midpoint: (100 + 80) / 2 = 90
        assert_eq!(matches[0].agreed_price, 90);
    }

    #[test]
    fn test_priority_based_matching() {
        let mut book = BidAskBook::new();
        let deadline = Utc::now() + Duration::hours(1);

        // Low priority bid
        book.add_bid(Bid::new("req_low".to_string(), 100, 1, deadline));
        // High priority bid
        book.add_bid(Bid::new("req_high".to_string(), 100, 10, deadline));

        // Only one ask available
        book.add_ask(Ask::new("prover1".to_string(), 80, 5, 0.9));

        let matches = book.match_bids_asks();

        assert_eq!(matches.len(), 1);
        // High priority should match first
        assert_eq!(matches[0].bid.requester_id, "req_high");
    }
}
