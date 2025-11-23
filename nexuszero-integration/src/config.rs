use serde::{Deserialize, Serialize};
use nexuszero_crypto::SecurityLevel;

/// Protocol configuration controlling optimization and compression behaviors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtocolConfig {
    /// Enable neural optimizer parameter selection (stubbed)
    pub use_optimizer: bool,
    /// Enable holographic compression of the proof
    pub use_compression: bool,
    /// Target security level for parameter selection (stored for future use)
    pub security_level: SecurityLevel,
    /// Optional maximum proof size in bytes (advisory)
    pub max_proof_size: Option<usize>,
    /// Optional maximum verification time (ms) target
    pub max_verify_time: Option<f64>,
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            use_optimizer: true,
            use_compression: true,
            security_level: SecurityLevel::Bit128,
            max_proof_size: Some(10_000),
            max_verify_time: Some(50.0),
        }
    }
}

