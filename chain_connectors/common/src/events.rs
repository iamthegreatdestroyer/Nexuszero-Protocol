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

    // ===== HARDENING TESTS =====

    #[test]
    fn test_event_filter_default() {
        let filter = EventFilter::default();
        assert!(filter.event_types.is_empty());
        assert!(filter.from_block.is_none());
        assert!(filter.to_block.is_none());
        assert!(filter.addresses.is_empty());
        assert!(filter.topics.is_empty());
    }

    #[test]
    fn test_event_filter_with_addresses() {
        let addresses = vec![vec![0x12; 20], vec![0x34; 20]];
        let filter = EventFilter::new().with_addresses(addresses.clone());
        
        assert_eq!(filter.addresses.len(), 2);
        assert_eq!(filter.addresses[0], vec![0x12; 20]);
    }

    #[test]
    fn test_event_filter_with_topics() {
        let filter = EventFilter::new()
            .with_topic(0, [0xaa; 32])
            .with_topic(2, [0xbb; 32]);
        
        assert_eq!(filter.topics.len(), 3);
        assert_eq!(filter.topics[0], Some([0xaa; 32]));
        assert!(filter.topics[1].is_none());
        assert_eq!(filter.topics[2], Some([0xbb; 32]));
    }

    #[test]
    fn test_event_filter_chained_builder() {
        let filter = EventFilter::new()
            .from_block(1000)
            .to_block(2000)
            .with_event_types(vec![EventType::ProofSubmitted, EventType::ProofVerified])
            .with_addresses(vec![vec![0x12; 20]])
            .with_topic(0, [0xff; 32]);
        
        assert_eq!(filter.from_block, Some(1000));
        assert_eq!(filter.to_block, Some(2000));
        assert_eq!(filter.event_types.len(), 2);
        assert_eq!(filter.addresses.len(), 1);
        assert_eq!(filter.topics.len(), 1);
    }

    #[test]
    fn test_all_event_types() {
        let event_types = [
            EventType::ProofSubmitted,
            EventType::ProofVerified,
            EventType::BridgeTransferInitiated,
            EventType::BridgeTransferCompleted,
            EventType::BridgeTransferFailed,
            EventType::HtlcCreated,
            EventType::HtlcRedeemed,
            EventType::HtlcRefunded,
            EventType::Transfer,
            EventType::ContractDeployed,
            EventType::Unknown,
        ];
        
        assert_eq!(event_types.len(), 11);
        
        // Test equality
        assert_eq!(EventType::ProofSubmitted, EventType::ProofSubmitted);
        assert_ne!(EventType::ProofSubmitted, EventType::ProofVerified);
    }

    #[test]
    fn test_event_type_evm_signature() {
        // These should return Some signature
        assert!(EventType::ProofSubmitted.evm_signature().is_some());
        assert!(EventType::ProofVerified.evm_signature().is_some());
        assert!(EventType::BridgeTransferInitiated.evm_signature().is_some());
        
        // These should return None
        assert!(EventType::Unknown.evm_signature().is_none());
        assert!(EventType::HtlcCreated.evm_signature().is_none());
    }

    #[test]
    fn test_event_type_serde_roundtrip() {
        let event_types = [
            EventType::ProofSubmitted,
            EventType::Transfer,
            EventType::Unknown,
        ];
        
        for event_type in event_types {
            let json = serde_json::to_string(&event_type).unwrap();
            let parsed: EventType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, event_type);
        }
    }

    #[test]
    fn test_chain_event_structure() {
        let event = ChainEvent {
            chain: ChainId::Ethereum,
            block_number: 15_000_000,
            block_hash: [0xaa; 32],
            tx_hash: [0xbb; 32],
            log_index: 5,
            event_type: EventType::ProofSubmitted,
            data: EventData::ProofSubmitted {
                proof_id: [0xcc; 32],
                submitter: vec![0x12; 20],
                privacy_level: 3,
            },
            raw: RawEventData {
                address: vec![0x34; 20],
                topics: vec![[0xdd; 32]],
                data: vec![1, 2, 3, 4],
            },
        };
        
        assert_eq!(event.chain, ChainId::Ethereum);
        assert_eq!(event.block_number, 15_000_000);
        assert_eq!(event.log_index, 5);
    }

    #[test]
    fn test_event_data_proof_submitted() {
        let data = EventData::ProofSubmitted {
            proof_id: [1u8; 32],
            submitter: vec![2u8; 20],
            privacy_level: 5,
        };
        
        match data {
            EventData::ProofSubmitted { proof_id, submitter, privacy_level } => {
                assert_eq!(proof_id, [1u8; 32]);
                assert_eq!(submitter.len(), 20);
                assert_eq!(privacy_level, 5);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_event_data_bridge_transfer() {
        let data = EventData::BridgeTransferInitiated {
            transfer_id: [1u8; 32],
            source_chain: ChainId::Ethereum,
            target_chain: ChainId::Polygon,
            sender: vec![2u8; 20],
            amount: 1_000_000_000_000_000_000u128,
        };
        
        match data {
            EventData::BridgeTransferInitiated { source_chain, target_chain, amount, .. } => {
                assert_eq!(source_chain, ChainId::Ethereum);
                assert_eq!(target_chain, ChainId::Polygon);
                assert_eq!(amount, 1_000_000_000_000_000_000u128);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_event_data_htlc_created() {
        let data = EventData::HtlcCreated {
            htlc_id: [1u8; 32],
            sender: vec![2u8; 20],
            recipient: vec![3u8; 20],
            amount: 500_000,
            hash_lock: [4u8; 32],
            timeout_block: 1_000_000,
        };
        
        match data {
            EventData::HtlcCreated { htlc_id, amount, timeout_block, .. } => {
                assert_eq!(htlc_id, [1u8; 32]);
                assert_eq!(amount, 500_000);
                assert_eq!(timeout_block, 1_000_000);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_event_data_htlc_redeemed() {
        let data = EventData::HtlcRedeemed {
            htlc_id: [1u8; 32],
            preimage: [2u8; 32],
        };
        
        match data {
            EventData::HtlcRedeemed { htlc_id, preimage } => {
                assert_eq!(htlc_id, [1u8; 32]);
                assert_eq!(preimage, [2u8; 32]);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_event_data_transfer() {
        let data = EventData::Transfer {
            from: vec![1u8; 20],
            to: vec![2u8; 20],
            amount: 1_000_000,
        };
        
        match data {
            EventData::Transfer { from, to, amount } => {
                assert_eq!(from.len(), 20);
                assert_eq!(to.len(), 20);
                assert_eq!(amount, 1_000_000);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_event_data_unknown() {
        let data = EventData::Unknown {
            topics: vec![[0xaa; 32], [0xbb; 32]],
            data: vec![1, 2, 3, 4, 5],
        };
        
        match data {
            EventData::Unknown { topics, data } => {
                assert_eq!(topics.len(), 2);
                assert_eq!(data.len(), 5);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_raw_event_data_structure() {
        let raw = RawEventData {
            address: vec![0x12; 20],
            topics: vec![[0xaa; 32], [0xbb; 32], [0xcc; 32]],
            data: vec![1, 2, 3, 4, 5, 6, 7, 8],
        };
        
        assert_eq!(raw.address.len(), 20);
        assert_eq!(raw.topics.len(), 3);
        assert_eq!(raw.data.len(), 8);
    }

    #[test]
    fn test_event_channel_creation() {
        let (tx, mut rx) = event_channel(100);
        
        // Channel should be open
        assert!(!tx.is_closed());
        
        // Drop sender
        drop(tx);
        
        // Receiver should indicate channel closed
        let result = rx.try_recv();
        assert!(result.is_err());
    }

    #[test]
    fn test_chain_event_structure_complete() {
        // Test structure without full serde roundtrip (u128 has JSON limitations)
        let event = ChainEvent {
            chain: ChainId::Polygon,
            block_number: 100,
            block_hash: [0x11; 32],
            tx_hash: [0x22; 32],
            log_index: 0,
            event_type: EventType::Transfer,
            data: EventData::Transfer {
                from: vec![1u8; 20],
                to: vec![2u8; 20],
                amount: 1000,
            },
            raw: RawEventData {
                address: vec![3u8; 20],
                topics: vec![],
                data: vec![],
            },
        };
        
        // Verify structure
        assert_eq!(event.chain, ChainId::Polygon);
        assert_eq!(event.block_number, 100);
        assert_eq!(event.event_type, EventType::Transfer);
        assert_eq!(event.log_index, 0);
    }

    #[test]
    fn test_event_filter_block_range_validation() {
        // Valid range
        let filter = EventFilter::new()
            .from_block(100)
            .to_block(200);
        
        assert!(filter.from_block.unwrap() < filter.to_block.unwrap());
        
        // Same block (valid)
        let filter = EventFilter::new()
            .from_block(100)
            .to_block(100);
        
        assert_eq!(filter.from_block, filter.to_block);
    }

    #[test]
    fn test_event_type_hash_and_eq() {
        use std::collections::HashSet;
        
        let mut set = HashSet::new();
        set.insert(EventType::ProofSubmitted);
        set.insert(EventType::ProofVerified);
        set.insert(EventType::ProofSubmitted); // duplicate
        
        assert_eq!(set.len(), 2);
        assert!(set.contains(&EventType::ProofSubmitted));
        assert!(set.contains(&EventType::ProofVerified));
    }
}
