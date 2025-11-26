//! NexusZero Protocol SDK
//!
//! Rust SDK with WASM bindings for the NexusZero Protocol.
//! Provides privacy-preserving transaction capabilities with
//! quantum-resistant cryptographic proofs.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

pub mod client;
pub mod error;
pub mod privacy;
pub mod types;

pub use client::NexusZeroClient;
pub use error::{NexusZeroError, Result};
pub use privacy::{PrivacyEngine, PrivacyLevel, PrivacyParameters};
pub use types::{ProofResult, Transaction, TransactionRequest};

/// Initialize the SDK (for WASM environments)
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Version of the SDK
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get SDK version (WASM export)
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sdk_version() -> String {
    VERSION.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_privacy_level_values() {
        assert_eq!(PrivacyLevel::Transparent as u8, 0);
        assert_eq!(PrivacyLevel::Sovereign as u8, 5);
    }
}
