//! Anonymity Set Management

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::error::{MorphingError, MorphingResult};

/// An anonymity set for transaction mixing
#[derive(Debug, Clone)]
pub struct AnonymitySet {
    /// Unique set ID
    pub id: Uuid,
    /// Current set size
    pub size: usize,
    /// Chain this set is for
    pub chain: Option<String>,
    /// Minimum size before allowing transactions
    pub min_size: usize,
    /// Set entries (commitments/nullifiers)
    entries: Vec<String>,
}

impl AnonymitySet {
    /// Create a new anonymity set
    pub fn new(min_size: usize, chain: Option<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            size: 0,
            chain,
            min_size,
            entries: Vec::new(),
        }
    }

    /// Add an entry to the set
    pub fn add_entry(&mut self, entry: String) {
        self.entries.push(entry);
        self.size = self.entries.len();
    }

    /// Check if set is ready for use
    pub fn is_ready(&self) -> bool {
        self.size >= self.min_size
    }

    /// Get random entries for ring signature
    pub fn get_ring_members(&self, count: usize, exclude: &str) -> Vec<String> {
        use rand::seq::SliceRandom;
        
        let mut rng = rand::thread_rng();
        let filtered: Vec<&String> = self.entries.iter()
            .filter(|e| *e != exclude)
            .collect();
        
        filtered.choose_multiple(&mut rng, count.min(filtered.len()))
            .cloned()
            .cloned()
            .collect()
    }
}

/// Statistics for anonymity sets
#[derive(Debug, Clone)]
pub struct AnonymityStats {
    pub active_sets: usize,
    pub total_entries: usize,
}

/// Manager for anonymity sets
pub struct AnonymitySetManager {
    /// Minimum set size
    min_size: usize,
    /// Maximum cached sets
    max_cache_size: usize,
    /// Sets by size requirement
    sets_by_size: RwLock<HashMap<usize, Vec<AnonymitySet>>>,
    /// Sets by chain
    sets_by_chain: RwLock<HashMap<String, Vec<Uuid>>>,
    /// Total entries counter
    total_entries: AtomicUsize,
}

impl AnonymitySetManager {
    /// Create a new anonymity set manager
    pub fn new(min_size: usize, max_cache_size: usize) -> Self {
        Self {
            min_size,
            max_cache_size,
            sets_by_size: RwLock::new(HashMap::new()),
            sets_by_chain: RwLock::new(HashMap::new()),
            total_entries: AtomicUsize::new(0),
        }
    }

    /// Get or create an anonymity set for the given requirements
    pub async fn get_or_create_set(
        &self,
        required_size: usize,
        chain: Option<&str>,
    ) -> MorphingResult<AnonymitySet> {
        let effective_size = required_size.max(self.min_size);
        
        // Try to find an existing set
        {
            let sets = self.sets_by_size.read().await;
            if let Some(size_sets) = sets.get(&effective_size) {
                for set in size_sets {
                    if set.chain.as_deref() == chain && set.is_ready() {
                        return Ok(set.clone());
                    }
                }
            }
        }

        // Create a new set
        let set = self.create_set(effective_size, chain.map(String::from)).await?;
        Ok(set)
    }

    /// Create a new anonymity set
    async fn create_set(
        &self,
        required_size: usize,
        chain: Option<String>,
    ) -> MorphingResult<AnonymitySet> {
        let mut set = AnonymitySet::new(required_size, chain.clone());
        
        // Populate with initial entries (in production, these would be real commitments)
        for i in 0..required_size {
            let entry = format!("commitment_{:016x}", rand::random::<u64>() + i as u64);
            set.add_entry(entry);
        }

        self.total_entries.fetch_add(required_size, Ordering::Relaxed);

        // Cache the set
        {
            let mut sets = self.sets_by_size.write().await;
            sets.entry(required_size)
                .or_insert_with(Vec::new)
                .push(set.clone());

            // Evict old sets if cache is too large
            self.maybe_evict_sets(&mut sets).await;
        }

        // Track by chain
        if let Some(ref chain_name) = chain {
            let mut chain_sets = self.sets_by_chain.write().await;
            chain_sets.entry(chain_name.clone())
                .or_insert_with(Vec::new)
                .push(set.id);
        }

        Ok(set)
    }

    /// Add an entry to matching sets
    pub async fn add_entry_to_sets(
        &self,
        entry: String,
        chain: Option<&str>,
    ) -> MorphingResult<usize> {
        let mut updated = 0;
        let mut sets = self.sets_by_size.write().await;

        for (_, size_sets) in sets.iter_mut() {
            for set in size_sets.iter_mut() {
                if set.chain.as_deref() == chain {
                    set.add_entry(entry.clone());
                    updated += 1;
                }
            }
        }

        if updated > 0 {
            self.total_entries.fetch_add(updated, Ordering::Relaxed);
        }

        Ok(updated)
    }

    /// Get statistics about anonymity sets
    pub async fn get_stats(&self) -> AnonymityStats {
        let sets = self.sets_by_size.read().await;
        let active_sets: usize = sets.values().map(|v| v.len()).sum();

        AnonymityStats {
            active_sets,
            total_entries: self.total_entries.load(Ordering::Relaxed),
        }
    }

    async fn maybe_evict_sets(&self, sets: &mut HashMap<usize, Vec<AnonymitySet>>) {
        let total: usize = sets.values().map(|v| v.len()).sum();
        
        if total > self.max_cache_size {
            // Remove oldest sets (simple FIFO eviction)
            let to_remove = total - self.max_cache_size;
            let mut removed = 0;

            for (_, size_sets) in sets.iter_mut() {
                while removed < to_remove && !size_sets.is_empty() {
                    if let Some(set) = size_sets.first() {
                        self.total_entries.fetch_sub(set.size, Ordering::Relaxed);
                    }
                    size_sets.remove(0);
                    removed += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anonymity_set_creation() {
        let set = AnonymitySet::new(10, Some("ethereum".to_string()));
        assert!(!set.is_ready());
        assert_eq!(set.size, 0);
    }

    #[test]
    fn test_anonymity_set_ready() {
        let mut set = AnonymitySet::new(2, None);
        set.add_entry("entry1".to_string());
        assert!(!set.is_ready());
        set.add_entry("entry2".to_string());
        assert!(set.is_ready());
    }

    #[tokio::test]
    async fn test_manager_get_or_create() {
        let manager = AnonymitySetManager::new(5, 100);
        let set = manager.get_or_create_set(10, Some("ethereum")).await.unwrap();
        assert!(set.is_ready());
        assert_eq!(set.size, 10);
    }

    #[tokio::test]
    async fn test_manager_stats() {
        let manager = AnonymitySetManager::new(5, 100);
        manager.get_or_create_set(10, None).await.unwrap();
        manager.get_or_create_set(20, None).await.unwrap();
        
        let stats = manager.get_stats().await;
        assert_eq!(stats.active_sets, 2);
        assert_eq!(stats.total_entries, 30);
    }
}
