//! # Proof Marketplace
//!
//! A decentralized marketplace for proof generation services in the NexusZero DPGN.
//!
//! ## Overview
//!
//! The marketplace module provides the economic layer of the Distributed Proof
//! Generation Network, matching task requesters with optimal provers through
//! reputation-weighted auctions.
//!
//! ## Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │                        MARKETPLACE                            │
//! ├───────────────────────────────────────────────────────────────┤
//! │                                                               │
//! │  ┌─────────────┐   ┌─────────────────┐   ┌────────────────┐  │
//! │  │ Order Book  │──▶│ Auction Engine  │──▶│   Reputation   │  │
//! │  │             │   │                 │   │    Tracker     │  │
//! │  │ • Bids      │   │ • Price Weight  │   │                │  │
//! │  │ • Asks      │   │ • Rep Weight    │   │ • Success Rate │  │
//! │  │ • Matching  │   │ • Scoring       │   │ • Quality Avg  │  │
//! │  └─────────────┘   └─────────────────┘   │ • Time Perf    │  │
//! │                                          └────────────────┘  │
//! └───────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Components
//!
//! ### Order Book ([`order_book`])
//!
//! Manages bids from task requesters and asks from provers:
//!
//! ```rust,ignore
//! use proof_network_marketplace::{Bid, Ask, AdvancedOrderBook};
//! use chrono::Utc;
//!
//! let mut book = AdvancedOrderBook::new();
//!
//! // Requester submits a bid
//! let bid = Bid::new(
//!     "requester-1".to_string(),
//!     100, // max price
//!     5,   // priority
//!     Utc::now() + chrono::Duration::hours(1),
//! );
//! book.add_bid(bid);
//!
//! // Prover submits an ask
//! let ask = Ask::new(
//!     "prover-1".to_string(),
//!     80,   // min price
//!     10,   // capacity
//!     0.95, // reputation score
//! );
//! book.add_ask(ask);
//! ```
//!
//! ### Auction Engine ([`auction`])
//!
//! Selects optimal provers using configurable weighted scoring:
//!
//! ```rust,ignore
//! use proof_network_marketplace::{AuctionEngine, AuctionConfig};
//!
//! let config = AuctionConfig {
//!     price_weight: 0.7,      // 70% weight on price
//!     reputation_weight: 0.3, // 30% weight on reputation
//! };
//!
//! let engine = AuctionEngine::with_config(config);
//! ```
//!
//! ### Reputation Tracker ([`reputation`])
//!
//! Tracks prover performance with a composite scoring formula:
//!
//! ```text
//! Overall Score = (Success Rate × 0.5) + (Quality × 0.3) + (Time Efficiency × 0.2)
//! ```
//!
//! ## Example: Complete Auction Flow
//!
//! ```rust,ignore
//! use proof_network_marketplace::{
//!     OrderBook, Order, AuctionEngine, ReputationTracker
//! };
//! use uuid::Uuid;
//!
//! // Create components
//! let mut book = OrderBook::new();
//! let mut reputation = ReputationTracker::new();
//!
//! // Add prover orders
//! book.add_order(Order {
//!     id: Uuid::new_v4(),
//!     prover_id: "prover-1".to_string(),
//!     price: 80,
//!     capacity: 5,
//! });
//!
//! // Update reputation
//! reputation.update_reputation("prover-1", true, 0.95, 1500);
//!
//! // Run auction
//! let engine = AuctionEngine::new();
//! let result = engine.run_auction_with_reputation(&book, &reputation, 5);
//! println!("Winners: {:?}", result.winners);
//! ```

pub mod auction;
pub mod order_book;
pub mod reputation;

pub use auction::*;
pub use order_book::*;
pub use reputation::*;

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_order_book() {
        let mut book = OrderBook::new();
        let id = Uuid::new_v4();
        let order = Order { id, prover_id: "prover1".to_string(), price: 100, capacity: 5 };
        book.add_order(order.clone());
        assert_eq!(book.orders.len(), 1);

        let best = book.best_order().unwrap();
        assert_eq!(best.price, 100);
    }
}
