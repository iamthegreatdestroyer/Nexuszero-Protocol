//! Event types and streaming for chain connectors

use serde::{Deserialize, Serialize};
use crate::types::ChainId;

/// Event filter for subscribing to chain events
#[derive(Debug, Clone, Default)]
pub struct EventFilter {
    /// Event types to filter (empty = all)
    pub event_types: Vec<EventType>,
    /// Starting block number (None = latest)
    pub from_block: Option<u64>,
    /// Ending block number (None = infinite)
    pub to_block: Option<u64>,
    /// Contract addresses to filter (empty = all)
    pub addresses: Vec<Vec<u8>>,
    /// Topic filters for indexed parameters
    pub topics: Vec<Option<[u8; 32]>>,
}

impl EventFilter {
    /// Create a new empty filter (matches all events)
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by event types
    pub fn with_event_types(mut self, types: Vec<EventType>) -> Self {
        self.event_types = types;
        self
    }

    /// Filter from a specific block
    pub fn from_block(mut self, block: u64) -> Self {
        self.from_block = Some(block);
        self
    }

    /// Filter up to a specific block
    pub fn to_block(mut self, block: u64) -> Self {
        self.to_block = Some(block);
        self
    }

    /// Filter by contract addresses
    pub fn with_addresses(mut self, addresses: Vec<Vec<u8>>) -> Self {
        self.addresses = addresses;
        self
    }

    /// Add topic filter
    pub fn with_topic(mut self, index: usize, topic: [u8; 32]) -> Self {
        while self.topics.len() <= index {
            self.topics.push(None);
        }
        self.topics[index] = Some(topic);
        self
    }

    /// Create filter for proof submission events
    pub fn proof_submitted() -> Self {
        Self::new().with_event_types(vec![EventType::ProofSubmitted])
    }

    /// Create filter for proof verification events
    pub fn proof_verified() -> Self {
        Self::new().with_event_types(vec![EventType::ProofVerified])
    }

    /// Create filter for bridge events
    pub fn bridge_events() -> Self {
        Self::new().with_event_types(vec![
            EventType::BridgeTransferInitiated,
            EventType::BridgeTransferCompleted,
        ])
    }
}

/// Types of events that can be emitted
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    /// A proof was submitted to the verifier
    ProofSubmitted,
    /// A proof was verified
    ProofVerified,
    /// A bridge transfer was initiated
    BridgeTransferInitiated,
    /// A bridge transfer was completed
    BridgeTransferCompleted,
    /// A bridge transfer failed
    BridgeTransferFailed,
    /// An HTLC was created
    HtlcCreated,
    /// An HTLC was redeemed
    HtlcRedeemed,
    /// An HTLC was refunded
    HtlcRefunded,
    /// Generic token transfer
    Transfer,
    /// Contract deployed
    ContractDeployed,
    /// Unknown event type
    Unknown,
}

impl EventType {
    /// Get the event signature hash for EVM chains
    pub fn evm_signature(&self) -> Option<[u8; 32]> {
        // These are keccak256 hashes of event signatures
        match self {
            EventType::ProofSubmitted => Some(hex_to_bytes32(
                "8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925",
            )),
            EventType::ProofVerified => Some(hex_to_bytes32(
                "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
            )),
            EventType::BridgeTransferInitiated => Some(hex_to_bytes32(
                "d79d3e1c4a11ef0e2c5d2b9c21a6c5b3d0a8e7f6c5b4a3d2e1f0c9b8a7d6e5f4",
            )),
            _ => None,
        }
    }
}

/// Helper to convert hex string to bytes32
fn hex_to_bytes32(hex: &str) -> [u8; 32] {
    let bytes = hex::decode(hex).unwrap_or_else(|_| vec![0u8; 32]);
    let mut result = [0u8; 32];
    result.copy_from_slice(&bytes[..32.min(bytes.len())]);
    result
}

/// Chain event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainEvent {
    /// Chain the event occurred on
    pub chain: ChainId,
    /// Block number
    pub block_number: u64,
    /// Block hash
    pub block_hash: [u8; 32],
    /// Transaction hash
    pub tx_hash: [u8; 32],
    /// Log index within transaction
    pub log_index: u32,
    /// Event type
    pub event_type: EventType,
    /// Event-specific data
    pub data: EventData,
    /// Raw event data
    pub raw: RawEventData,
}

/// Event-specific parsed data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventData {
    /// Proof submitted event data
    ProofSubmitted {
        proof_id: [u8; 32],
        submitter: Vec<u8>,
        privacy_level: u8,
    },
    /// Proof verified event data
    ProofVerified {
        proof_id: [u8; 32],
        success: bool,
    },
    /// Bridge transfer initiated
    BridgeTransferInitiated {
        transfer_id: [u8; 32],
        source_chain: ChainId,
        target_chain: ChainId,
        sender: Vec<u8>,
        amount: u128,
    },
    /// Bridge transfer completed
    BridgeTransferCompleted {
        transfer_id: [u8; 32],
        recipient: Vec<u8>,
    },
    /// HTLC created
    HtlcCreated {
        htlc_id: [u8; 32],
        sender: Vec<u8>,
        recipient: Vec<u8>,
        amount: u128,
        hash_lock: [u8; 32],
        timeout_block: u64,
    },
    /// HTLC redeemed
    HtlcRedeemed {
        htlc_id: [u8; 32],
        preimage: [u8; 32],
    },
    /// HTLC refunded
    HtlcRefunded {
        htlc_id: [u8; 32],
    },
    /// Token transfer
    Transfer {
        from: Vec<u8>,
        to: Vec<u8>,
        amount: u128,
    },
    /// Unknown/unparsed event
    Unknown {
        topics: Vec<[u8; 32]>,
        data: Vec<u8>,
    },
}

/// Raw event data for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEventData {
    /// Contract address
    pub address: Vec<u8>,
    /// Event topics
    pub topics: Vec<[u8; 32]>,
    /// Event data payload
    pub data: Vec<u8>,
}

/// Event stream type
pub type EventStream = tokio::sync::mpsc::Receiver<ChainEvent>;

/// Event sender for internal use
pub type EventSender = tokio::sync::mpsc::Sender<ChainEvent>;

/// Create an event channel with the given buffer size
pub fn event_channel(buffer_size: usize) -> (EventSender, EventStream) {
    tokio::sync::mpsc::channel(buffer_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_filter_builder() {
        let filter = EventFilter::new()
            .from_block(100)
            .to_block(200)
            .with_event_types(vec![EventType::ProofSubmitted]);

        assert_eq!(filter.from_block, Some(100));
        assert_eq!(filter.to_block, Some(200));
        assert_eq!(filter.event_types.len(), 1);
    }

    #[test]
    fn test_proof_submitted_filter() {
        let filter = EventFilter::proof_submitted();
        assert_eq!(filter.event_types, vec![EventType::ProofSubmitted]);
    }

    #[test]
    fn test_bridge_events_filter() {
        let filter = EventFilter::bridge_events();
        assert_eq!(filter.event_types.len(), 2);
    }
}
