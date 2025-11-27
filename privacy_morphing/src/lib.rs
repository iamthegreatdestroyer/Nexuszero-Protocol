//! NexusZero Privacy Morphing Engine
//!
//! This crate implements the Adaptive Privacy Morphing (APM) system,
//! which dynamically adjusts transaction privacy parameters to maintain
//! optimal privacy while ensuring regulatory compliance.
//!
//! # Core Components
//!
//! - **Privacy Level**: 1-10 scale privacy settings
//! - **Morphing Schedules**: Automated privacy adjustments over time
//! - **Compliance Integration**: Automatic privacy ceiling for regulated entities
//! - **Anonymity Sets**: Dynamic expansion based on network activity
//!
//! # Example
//!
//! ```rust,no_run
//! use nexuszero_privacy_morphing::{PrivacyMorphingEngine, MorphingConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = MorphingConfig::default();
//!     let engine = PrivacyMorphingEngine::new(config);
//!
//!     // Calculate privacy level for a transaction
//!     let context = TransactionContext {
//!         amount: 1000,
//!         recipient_type: RecipientType::Normal,
//!         compliance_required: false,
//!     };
//!
//!     let privacy_level = engine.calculate_privacy_level(&context).await;
//! }
//! ```

pub mod config;
pub mod engine;
pub mod error;
pub mod schedule;
pub mod types;
pub mod anonymity;
pub mod compliance;

pub use config::MorphingConfig;
pub use engine::PrivacyMorphingEngine;
pub use error::{MorphingError, MorphingResult};
pub use schedule::{MorphingSchedule, ScheduleExecutor};
pub use types::*;
pub use anonymity::AnonymitySetManager;
pub use compliance::ComplianceIntegration;

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default privacy level for new transactions
pub const DEFAULT_PRIVACY_LEVEL: u8 = 5;

/// Maximum privacy level
pub const MAX_PRIVACY_LEVEL: u8 = 10;

/// Minimum privacy level (transparent)
pub const MIN_PRIVACY_LEVEL: u8 = 1;
