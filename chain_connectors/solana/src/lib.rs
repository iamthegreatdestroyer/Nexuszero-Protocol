// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

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
