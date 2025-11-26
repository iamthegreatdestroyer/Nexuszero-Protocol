//! # Polygon Blockchain Connector
//!
//! This crate provides Polygon (PoS) blockchain integration for NexusZero Protocol.
//! Polygon is EVM-compatible, so this connector extends Ethereum patterns with
//! Polygon-specific optimizations like faster block times and lower gas costs.
//!
//! ## Features
//!
//! - EVM-compatible transaction handling
//! - Polygon-specific gas estimation (dynamic priority fees)
//! - Checkpoint verification for PoS security
//! - Mumbai testnet and Mainnet support
//! - Fast finality handling
//!
//! ## Example
//!
//! ```rust,ignore
//! use chain_connectors_polygon::{PolygonConnector, PolygonConfig};
//! use chain_connectors_common::ChainConnector;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = PolygonConfig::mainnet();
//!     let connector = PolygonConnector::new(config).await?;
//!     
//!     let balance = connector.get_balance("0x...").await?;
//!     println!("Balance: {} MATIC", balance.amount);
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod connector;

pub use config::PolygonConfig;
pub use connector::PolygonConnector;

// Re-export common types
pub use chain_connectors_common::{
    BlockInfo, ChainConnector, ChainError, ChainEvent, ChainId,
    EventFilter, FeeEstimate, TransactionReceipt,
};
