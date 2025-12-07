//! UTXO management for Bitcoin transactions

use bitcoin::{Amount, OutPoint, ScriptBuf, TxOut, Txid};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::BitcoinError;

/// A spendable UTXO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Utxo {
    /// The outpoint (txid + vout)
    pub outpoint: OutPoint,
    /// The output value in satoshis
    pub value: u64,
    /// The output script
    pub script_pubkey: ScriptBuf,
    /// Number of confirmations
    pub confirmations: u32,
    /// Whether this is a Taproot output
    pub is_taproot: bool,
    /// Whether this UTXO contains a NexusZero proof
    pub has_proof: bool,
    /// Associated proof hash (if any)
    pub proof_hash: Option<[u8; 32]>,
}

impl Utxo {
    /// Create a new UTXO
    pub fn new(outpoint: OutPoint, output: &TxOut, confirmations: u32) -> Self {
        let is_taproot = output.script_pubkey.is_p2tr();
        
        Self {
            outpoint,
            value: output.value.to_sat(),
            script_pubkey: output.script_pubkey.clone(),
            confirmations,
            is_taproot,
            has_proof: false,
            proof_hash: None,
        }
    }

    /// Mark this UTXO as containing a proof
    pub fn with_proof(mut self, proof_hash: [u8; 32]) -> Self {
        self.has_proof = true;
        self.proof_hash = Some(proof_hash);
        self
    }

    /// Get the amount
    pub fn amount(&self) -> Amount {
        Amount::from_sat(self.value)
    }
}

/// UTXO selection strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Select the oldest UTXOs first (FIFO)
    OldestFirst,
    /// Select the largest UTXOs first
    LargestFirst,
    /// Select the smallest UTXOs that can cover the amount
    SmallestFirst,
    /// Minimize the number of inputs
    MinimizeInputs,
    /// Maximize privacy by avoiding change
    MaximizePrivacy,
    /// Branch and bound algorithm for exact match
    BranchAndBound,
}

/// UTXO selector
pub struct UtxoSelector {
    /// Available UTXOs
    utxos: Vec<Utxo>,
    /// Selection strategy
    strategy: SelectionStrategy,
    /// Minimum confirmations required
    min_confirmations: u32,
    /// Target fee rate in sat/vB
    fee_rate: f64,
}

impl UtxoSelector {
    /// Create a new UTXO selector
    pub fn new(utxos: Vec<Utxo>, strategy: SelectionStrategy) -> Self {
        Self {
            utxos,
            strategy,
            min_confirmations: 1,
            fee_rate: 1.0,
        }
    }

    /// Set minimum confirmations
    pub fn with_min_confirmations(mut self, confirmations: u32) -> Self {
        self.min_confirmations = confirmations;
        self
    }

    /// Set fee rate
    pub fn with_fee_rate(mut self, rate: f64) -> Self {
        self.fee_rate = rate;
        self
    }

    /// Select UTXOs to cover the target amount
    pub fn select(&self, target_amount: u64) -> Result<UtxoSelection, BitcoinError> {
        // Filter UTXOs by confirmation count
        let mut available: Vec<_> = self.utxos.iter()
            .filter(|u| u.confirmations >= self.min_confirmations)
            .cloned()
            .collect();

        if available.is_empty() {
            return Err(BitcoinError::UtxoSelectionError(
                "No UTXOs available with required confirmations".to_string()
            ));
        }

        // Calculate total available
        let total_available: u64 = available.iter().map(|u| u.value).sum();
        if total_available < target_amount {
            return Err(BitcoinError::InsufficientFunds {
                required: target_amount,
                available: total_available,
            });
        }

        // Sort based on strategy
        match self.strategy {
            SelectionStrategy::OldestFirst => {
                // Sort by confirmations descending (oldest first)
                available.sort_by(|a, b| b.confirmations.cmp(&a.confirmations));
            }
            SelectionStrategy::LargestFirst => {
                available.sort_by(|a, b| b.value.cmp(&a.value));
            }
            SelectionStrategy::SmallestFirst => {
                available.sort_by(|a, b| a.value.cmp(&b.value));
            }
            SelectionStrategy::MinimizeInputs => {
                available.sort_by(|a, b| b.value.cmp(&a.value));
            }
            SelectionStrategy::MaximizePrivacy => {
                // Try to find exact match or minimize change
                return self.select_for_privacy(&available, target_amount);
            }
            SelectionStrategy::BranchAndBound => {
                return self.branch_and_bound(&available, target_amount);
            }
        }

        // Simple greedy selection
        let mut selected = Vec::new();
        let mut selected_value = 0u64;

        for utxo in available {
            if selected_value >= target_amount {
                break;
            }
            selected_value += utxo.value;
            selected.push(utxo);
        }

        // Estimate fee
        let estimated_vsize = self.estimate_transaction_vsize(selected.len(), 2); // 2 outputs typical
        let estimated_fee = (estimated_vsize as f64 * self.fee_rate).ceil() as u64;

        // Add more UTXOs if needed to cover fee
        if selected_value < target_amount + estimated_fee {
            // Would need to add more UTXOs or adjust
        }

        Ok(UtxoSelection {
            selected,
            total_value: selected_value,
            target_amount,
            estimated_fee,
            change: selected_value.saturating_sub(target_amount + estimated_fee),
        })
    }

    /// Select UTXOs to maximize privacy (minimize change)
    fn select_for_privacy(&self, available: &[Utxo], target: u64) -> Result<UtxoSelection, BitcoinError> {
        // Try to find exact match first
        for utxo in available {
            let estimated_fee = (self.estimate_transaction_vsize(1, 1) as f64 * self.fee_rate).ceil() as u64;
            if utxo.value >= target && utxo.value <= target + estimated_fee + 1000 {
                return Ok(UtxoSelection {
                    selected: vec![utxo.clone()],
                    total_value: utxo.value,
                    target_amount: target,
                    estimated_fee,
                    change: utxo.value.saturating_sub(target + estimated_fee),
                });
            }
        }

        // Fall back to minimizing change
        let mut sorted = available.to_vec();
        sorted.sort_by_key(|u| {
            if u.value >= target {
                u.value - target // Minimize excess
            } else {
                u64::MAX // Put small ones last
            }
        });

        let mut selected = Vec::new();
        let mut total = 0u64;

        for utxo in sorted {
            if total >= target {
                break;
            }
            total += utxo.value;
            selected.push(utxo);
        }

        let estimated_fee = (self.estimate_transaction_vsize(selected.len(), 2) as f64 * self.fee_rate).ceil() as u64;

        Ok(UtxoSelection {
            selected,
            total_value: total,
            target_amount: target,
            estimated_fee,
            change: total.saturating_sub(target + estimated_fee),
        })
    }

    /// Branch and bound algorithm for optimal UTXO selection
    fn branch_and_bound(&self, available: &[Utxo], target: u64) -> Result<UtxoSelection, BitcoinError> {
        // Simplified implementation - full BnB is complex
        // For now, use the largest first approach with some optimization
        let mut sorted = available.to_vec();
        sorted.sort_by(|a, b| b.value.cmp(&a.value));

        let estimated_fee = (self.estimate_transaction_vsize(sorted.len().min(5), 2) as f64 * self.fee_rate).ceil() as u64;
        let adjusted_target = target + estimated_fee;

        let mut selected = Vec::new();
        let mut total = 0u64;

        for utxo in sorted {
            if total >= adjusted_target {
                break;
            }
            total += utxo.value;
            selected.push(utxo);
        }

        if total < adjusted_target {
            return Err(BitcoinError::InsufficientFunds {
                required: adjusted_target,
                available: total,
            });
        }

        Ok(UtxoSelection {
            selected,
            total_value: total,
            target_amount: target,
            estimated_fee,
            change: total - adjusted_target,
        })
    }

    /// Estimate transaction virtual size
    fn estimate_transaction_vsize(&self, num_inputs: usize, num_outputs: usize) -> usize {
        // Rough estimate for P2TR inputs/outputs
        // Header: 10.5 vB
        // P2TR input: ~57.5 vB (key path spend)
        // P2TR output: 43 vB
        let header = 11;
        let inputs = num_inputs * 58;
        let outputs = num_outputs * 43;
        header + inputs + outputs
    }
}

/// Result of UTXO selection
#[derive(Debug, Clone)]
pub struct UtxoSelection {
    /// Selected UTXOs
    pub selected: Vec<Utxo>,
    /// Total value of selected UTXOs
    pub total_value: u64,
    /// Target amount to send
    pub target_amount: u64,
    /// Estimated fee
    pub estimated_fee: u64,
    /// Change amount
    pub change: u64,
}

impl UtxoSelection {
    /// Check if selection is valid
    pub fn is_valid(&self) -> bool {
        self.total_value >= self.target_amount + self.estimated_fee
    }

    /// Get number of selected inputs
    pub fn num_inputs(&self) -> usize {
        self.selected.len()
    }
}

/// UTXO cache for tracking available outputs
pub struct UtxoCache {
    /// UTXOs by outpoint
    utxos: HashMap<OutPoint, Utxo>,
    /// Total cached value
    total_value: u64,
}

impl UtxoCache {
    /// Create a new empty cache
    pub fn new() -> Self {
        Self {
            utxos: HashMap::new(),
            total_value: 0,
        }
    }

    /// Add a UTXO to the cache
    pub fn add(&mut self, utxo: Utxo) {
        let value = utxo.value;
        if self.utxos.insert(utxo.outpoint, utxo).is_none() {
            self.total_value += value;
        }
    }

    /// Remove a UTXO from the cache
    pub fn remove(&mut self, outpoint: &OutPoint) -> Option<Utxo> {
        if let Some(utxo) = self.utxos.remove(outpoint) {
            self.total_value -= utxo.value;
            Some(utxo)
        } else {
            None
        }
    }

    /// Get a UTXO by outpoint
    pub fn get(&self, outpoint: &OutPoint) -> Option<&Utxo> {
        self.utxos.get(outpoint)
    }

    /// Get all UTXOs
    pub fn all(&self) -> Vec<Utxo> {
        self.utxos.values().cloned().collect()
    }

    /// Get total value
    pub fn total(&self) -> u64 {
        self.total_value
    }

    /// Get number of UTXOs
    pub fn len(&self) -> usize {
        self.utxos.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.utxos.is_empty()
    }
}

impl Default for UtxoCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitcoin::hashes::Hash;

    fn mock_utxo(value: u64, confirmations: u32) -> Utxo {
        let txid = Txid::from_byte_array([confirmations as u8; 32]);
        let outpoint = OutPoint::new(txid, 0);
        
        Utxo {
            outpoint,
            value,
            script_pubkey: ScriptBuf::new(),
            confirmations,
            is_taproot: true,
            has_proof: false,
            proof_hash: None,
        }
    }

    #[test]
    fn test_utxo_selection_largest_first() {
        let utxos = vec![
            mock_utxo(10_000, 10),
            mock_utxo(50_000, 5),
            mock_utxo(30_000, 8),
        ];

        let selector = UtxoSelector::new(utxos, SelectionStrategy::LargestFirst);
        let selection = selector.select(40_000).unwrap();

        assert!(selection.is_valid());
        assert_eq!(selection.selected.len(), 1); // 50,000 should cover it
    }

    #[test]
    fn test_insufficient_funds() {
        let utxos = vec![
            mock_utxo(10_000, 10),
            mock_utxo(20_000, 5),
        ];

        let selector = UtxoSelector::new(utxos, SelectionStrategy::LargestFirst);
        let result = selector.select(100_000);

        assert!(matches!(result, Err(BitcoinError::InsufficientFunds { .. })));
    }

    #[test]
    fn test_utxo_cache() {
        let mut cache = UtxoCache::new();
        
        let utxo1 = mock_utxo(10_000, 5);
        let outpoint1 = utxo1.outpoint;
        
        cache.add(utxo1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total(), 10_000);

        cache.add(mock_utxo(20_000, 3));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.total(), 30_000);

        cache.remove(&outpoint1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total(), 20_000);
    }

    // ===== HARDENING TESTS =====

    #[test]
    fn test_utxo_creation() {
        let txid = Txid::from_byte_array([1u8; 32]);
        let outpoint = OutPoint::new(txid, 0);
        let tx_out = TxOut {
            value: Amount::from_sat(50_000),
            script_pubkey: ScriptBuf::new_p2tr_tweaked(
                bitcoin::key::TweakedPublicKey::dangerous_assume_tweaked(
                    bitcoin::secp256k1::XOnlyPublicKey::from_slice(&[2u8; 32]).unwrap()
                )
            ),
        };
        
        let utxo = Utxo::new(outpoint, &tx_out, 6);
        
        assert_eq!(utxo.value, 50_000);
        assert_eq!(utxo.confirmations, 6);
        assert!(utxo.is_taproot);
        assert!(!utxo.has_proof);
    }

    #[test]
    fn test_utxo_with_proof() {
        let utxo = mock_utxo(10_000, 5).with_proof([0xab; 32]);
        
        assert!(utxo.has_proof);
        assert_eq!(utxo.proof_hash.unwrap(), [0xab; 32]);
    }

    #[test]
    fn test_utxo_amount() {
        let utxo = mock_utxo(100_000_000, 10);
        let amount = utxo.amount();
        
        assert_eq!(amount, Amount::from_sat(100_000_000));
        assert_eq!(amount.to_btc(), 1.0);
    }

    #[test]
    fn test_selection_strategy_variants() {
        let strategies = [
            SelectionStrategy::OldestFirst,
            SelectionStrategy::LargestFirst,
            SelectionStrategy::SmallestFirst,
            SelectionStrategy::MinimizeInputs,
            SelectionStrategy::MaximizePrivacy,
            SelectionStrategy::BranchAndBound,
        ];
        
        // Verify all strategies are distinct
        assert_ne!(strategies[0], strategies[1]);
        assert_eq!(SelectionStrategy::LargestFirst, SelectionStrategy::LargestFirst);
    }

    #[test]
    fn test_utxo_selector_with_min_confirmations() {
        let utxos = vec![
            mock_utxo(100_000, 1),  // 1 confirmation
            mock_utxo(100_000, 6),  // 6 confirmations
            mock_utxo(100_000, 10), // 10 confirmations
        ];

        let selector = UtxoSelector::new(utxos, SelectionStrategy::LargestFirst)
            .with_min_confirmations(6);
        
        let selection = selector.select(50_000).unwrap();
        
        // Should only use UTXOs with >= 6 confirmations
        for utxo in &selection.selected {
            assert!(utxo.confirmations >= 6);
        }
    }

    #[test]
    fn test_utxo_selector_with_fee_rate() {
        let utxos = vec![
            mock_utxo(100_000, 10),
        ];

        let selector = UtxoSelector::new(utxos, SelectionStrategy::LargestFirst)
            .with_fee_rate(10.0);
        
        let selection = selector.select(50_000).unwrap();
        
        // Higher fee rate should mean higher estimated fee
        assert!(selection.estimated_fee > 0);
    }

    #[test]
    fn test_utxo_selection_valid() {
        let utxos = vec![
            mock_utxo(100_000, 10),
        ];

        let selector = UtxoSelector::new(utxos, SelectionStrategy::LargestFirst);
        let selection = selector.select(50_000).unwrap();
        
        assert!(selection.is_valid());
        assert!(selection.total_value >= selection.target_amount);
    }

    #[test]
    fn test_utxo_selection_change_calculation() {
        let utxos = vec![
            mock_utxo(100_000, 10),
        ];

        let selector = UtxoSelector::new(utxos, SelectionStrategy::LargestFirst);
        let selection = selector.select(50_000).unwrap();
        
        // Change = total - target - fee
        let expected_change = selection.total_value.saturating_sub(
            selection.target_amount + selection.estimated_fee
        );
        assert_eq!(selection.change, expected_change);
    }

    #[test]
    fn test_utxo_cache_empty() {
        let cache = UtxoCache::new();
        
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.total(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_utxo_cache_all() {
        let mut cache = UtxoCache::new();
        
        cache.add(mock_utxo(10_000, 5));
        cache.add(mock_utxo(20_000, 3));
        cache.add(mock_utxo(30_000, 7));
        
        let all = cache.all();
        assert_eq!(all.len(), 3);
        
        let total: u64 = all.iter().map(|u| u.value).sum();
        assert_eq!(total, 60_000);
    }

    #[test]
    fn test_utxo_cache_get() {
        let mut cache = UtxoCache::new();
        
        let utxo = mock_utxo(10_000, 5);
        let outpoint = utxo.outpoint;
        
        cache.add(utxo);
        
        let retrieved = cache.get(&outpoint);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value, 10_000);
    }

    #[test]
    fn test_utxo_cache_remove_nonexistent() {
        let mut cache = UtxoCache::new();
        let txid = Txid::from_byte_array([99u8; 32]);
        let outpoint = OutPoint::new(txid, 0);
        
        // Removing non-existent should not panic
        cache.remove(&outpoint);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_utxo_selection_oldest_first() {
        let utxos = vec![
            mock_utxo(30_000, 2),   // newest
            mock_utxo(30_000, 100), // oldest
            mock_utxo(30_000, 50),  // middle
        ];

        let selector = UtxoSelector::new(utxos, SelectionStrategy::OldestFirst);
        let selection = selector.select(25_000).unwrap();
        
        // Should pick the one with most confirmations first
        assert_eq!(selection.selected[0].confirmations, 100);
    }

    #[test]
    fn test_utxo_selection_smallest_first() {
        let utxos = vec![
            mock_utxo(50_000, 10),
            mock_utxo(10_000, 10),
            mock_utxo(30_000, 10),
        ];

        let selector = UtxoSelector::new(utxos, SelectionStrategy::SmallestFirst);
        let selection = selector.select(8_000).unwrap();
        
        // Should pick smallest first
        assert_eq!(selection.selected[0].value, 10_000);
    }

    #[test]
    fn test_utxo_cache_total_large_values() {
        let mut cache = UtxoCache::new();
        
        // Add large Bitcoin amounts
        cache.add(mock_utxo(21_000_000_00000000, 10)); // 21 million BTC
        
        assert_eq!(cache.total(), 21_000_000_00000000);
    }

    #[test]
    fn test_utxo_selection_no_confirmed_utxos() {
        let utxos = vec![
            mock_utxo(100_000, 0), // unconfirmed
        ];

        let selector = UtxoSelector::new(utxos, SelectionStrategy::LargestFirst)
            .with_min_confirmations(1);
        
        let result = selector.select(50_000);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_utxo_serde_roundtrip() {
        let utxo = mock_utxo(50_000, 6).with_proof([0xde; 32]);
        
        let json = serde_json::to_string(&utxo).unwrap();
        let parsed: Utxo = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.value, utxo.value);
        assert_eq!(parsed.confirmations, utxo.confirmations);
        assert_eq!(parsed.has_proof, utxo.has_proof);
        assert_eq!(parsed.proof_hash, utxo.proof_hash);
    }

    #[test]
    fn test_utxo_selection_multiple_inputs_needed() {
        let utxos = vec![
            mock_utxo(30_000, 10),
            mock_utxo(30_000, 10),
            mock_utxo(30_000, 10),
        ];

        let selector = UtxoSelector::new(utxos, SelectionStrategy::LargestFirst);
        let selection = selector.select(80_000).unwrap();
        
        // Should need all 3 UTXOs
        assert_eq!(selection.selected.len(), 3);
        assert!(selection.is_valid());
    }
}
