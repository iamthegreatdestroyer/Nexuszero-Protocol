//! Common interface for all NexusZero blockchain connectors
//!
//! This crate provides:
//! - Unified `ChainConnector` trait for chain operations
//! - Common types for transactions, proofs, and events
//! - Standard error handling across all connectors
//! - Event subscription interface
//! - Transaction building abstractions

pub mod error;
pub mod traits;
pub mod types;
pub mod events;

pub use error::*;
pub use traits::*;
pub use types::*;
pub use events::*;

/// Re-export commonly used items
pub mod prelude {
    pub use crate::error::ChainError;
    pub use crate::traits::ChainConnector;
    pub use crate::types::*;
    pub use crate::events::*;
}
