// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! # Cosmos Blockchain Connector
//!
//! This crate provides Cosmos SDK blockchain integration for NexusZero Protocol,
//! including IBC (Inter-Blockchain Communication) support for cross-chain operations.
//!
//! ## Features
//!
//! - Cosmos SDK transaction building and signing
//! - IBC message handling for cross-chain transfers
//! - Tendermint RPC integration
//! - CosmWasm smart contract interaction
//! - Multiple chain support (Cosmos Hub, Osmosis, etc.)
//!
//! ## Example
//!
//! ```rust,ignore
//! use chain_connectors_cosmos::{CosmosConnector, CosmosConfig};
//! use chain_connectors_common::ChainConnector;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = CosmosConfig::cosmoshub();
//!     let connector = CosmosConnector::new(config)?;
//!     
//!     let balance = connector.get_balance(b"cosmos1...").await?;
//!     println!("Balance: {} ATOM", balance);
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod connector;

pub use config::CosmosConfig;
pub use connector::CosmosConnector;

// Re-export common types
pub use chain_connectors_common::{
    BlockInfo, ChainConnector, ChainError, ChainEvent, ChainId,
    EventFilter, FeeEstimate, TransactionReceipt,
};
