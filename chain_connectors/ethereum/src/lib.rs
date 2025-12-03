// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! Ethereum blockchain connector for NexusZero Protocol
//!
//! This crate provides:
//! - Ethereum chain connectivity via alloy
//! - NexusZero verifier contract integration
//! - NexusZero bridge contract integration
//! - EIP-1559 transaction support
//! - Event subscription via WebSocket

pub mod connector;
pub mod contracts;
pub mod config;
pub mod error;

pub use connector::EthereumConnector;
pub use config::EthereumConfig;
pub use error::EthereumError;

/// Re-export commonly used items
pub mod prelude {
    pub use crate::connector::EthereumConnector;
    pub use crate::config::EthereumConfig;
    pub use crate::error::EthereumError;
    pub use chain_connectors_common::prelude::*;
}
