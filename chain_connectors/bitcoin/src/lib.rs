//! Bitcoin blockchain connector for NexusZero Protocol
//!
//! This crate provides:
//! - Bitcoin Core RPC connectivity
//! - Taproot transaction support
//! - PSBT (Partially Signed Bitcoin Transactions) handling
//! - UTXO management
//! - Privacy proof embedding in witness data

pub mod connector;
pub mod config;
pub mod error;
pub mod taproot;
pub mod utxo;
pub mod psbt;

pub use connector::BitcoinConnector;
pub use config::BitcoinConfig;
pub use error::BitcoinError;

/// Re-export commonly used items
pub mod prelude {
    pub use crate::connector::BitcoinConnector;
    pub use crate::config::BitcoinConfig;
    pub use crate::error::BitcoinError;
    pub use chain_connectors_common::prelude::*;
}
