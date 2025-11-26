//! Solana blockchain connector for NexusZero Protocol
//!
//! This crate provides:
//! - Solana RPC connectivity
//! - NexusZero program integration
//! - Transaction building and signing
//! - Account data parsing
//! - WebSocket subscription support

pub mod connector;
pub mod config;
pub mod error;
pub mod program;
pub mod accounts;

pub use connector::SolanaConnector;
pub use config::SolanaConfig;
pub use error::SolanaError;

/// Re-export commonly used items
pub mod prelude {
    pub use crate::connector::SolanaConnector;
    pub use crate::config::SolanaConfig;
    pub use crate::error::SolanaError;
    pub use chain_connectors_common::prelude::*;
}
