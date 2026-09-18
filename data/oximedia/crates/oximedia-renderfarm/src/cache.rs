// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Cache management for render farm.

use crate::error::Result;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Key
    pub key: String,
    /// Data
    pub data: Vec<u8>,
    /// Size in bytes
    pub size: u64,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Last accessed
    pub last_accessed: DateTime<Utc>,
    /// Access count
    pub access_count: u64,
}

/// Cache eviction policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size in bytes
    pub max_size: u64,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// TTL in seconds (0 = no expiration)
    pub ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size: 10 * 1024 * 1024 * 1024, // 10 GB
            eviction_policy: EvictionPolicy::LRU,
            ttl_seconds: 0,
        }
    }
}

/// Cache manager
pub struct CacheManager {
    config: CacheConfig,
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    lru_order: Arc<RwLock<VecDeque<String>>>,
    current_size: Arc<RwLock<u64>>,
}

impl CacheManager {
    /// Create a new cache manager
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: Arc::new(RwLock::new(HashMap::new())),
            lru_order: Arc::new(RwLock::new(VecDeque::new())),
            current_size: Arc::new(RwLock::new(0)),
        }
    }

    /// Get item from cache
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        let mut entries = self.entries.write();

        if let Some(entry) = entries.get_mut(key) {
            // Check TTL
            if self.config.ttl_seconds > 0 {
                let age = (Utc::now() - entry.created_at).num_seconds() as u64;
                if age > self.config.ttl_seconds {
                    // Expired, remove it
                    drop(entries);
                    let _ = self.remove(key);
                    return None;
                }
            }

            // Update access info
            entry.last_accessed = Utc::now();
            entry.access_count += 1;

            // Update LRU order
            if self.config.eviction_policy == EvictionPolicy::LRU {
                let mut lru_order = self.lru_order.write();
                if let Some(pos) = lru_order.iter().position(|k| k == key) {
                    lru_order.remove(pos);
                }
                lru_order.push_back(key.to_string());
            }

            return Some(entry.data.clone());
        }

        None
    }

    /// Put item in cache
    pub fn put(&self, key: String, data: Vec<u8>) -> Result<()> {
        let size = data.len() as u64;

        // Check if we need to evict
        while *self.current_size.read() + size > self.config.max_size {
            self.evict_one()?;
        }

        let entry = CacheEntry {
            key: key.clone(),
            data,
            size,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
        };

        // Insert entry
        let mut entries = self.entries.write();
        if let Some(old_entry) = entries.insert(key.clone(), entry) {
            // Removed old entry, update size
            *self.current_size.write() -= old_entry.size;
        }
        drop(entries);

        *self.current_size.write() += size;

        // Update order
        let mut lru_order = self.lru_order.write();
        lru_order.push_back(key);

        Ok(())
    }

    /// Remove item from cache
    #[must_use]
    pub fn remove(&self, key: &str) -> Option<CacheEntry> {
        let mut entries = self.entries.write();

        if let Some(entry) = entries.remove(key) {
            *self.current_size.write() -= entry.size;

            // Remove from LRU order
            let mut lru_order = self.lru_order.write();
            if let Some(pos) = lru_order.iter().position(|k| k == key) {
                lru_order.remove(pos);
            }

            return Some(entry);
        }

        None
    }

    /// Evict one entry based on policy
    fn evict_one(&self) -> Result<()> {
        let key_to_evict = match self.config.eviction_policy {
            EvictionPolicy::LRU => self.lru_order.write().pop_front(),
            EvictionPolicy::LFU => {
                let entries = self.entries.read();
                entries
                    .iter()
                    .min_by_key(|(_, e)| e.access_count)
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::FIFO => {
                let entries = self.entries.read();
                entries
                    .iter()
                    .min_by_key(|(_, e)| e.created_at)
                    .map(|(k, _)| k.clone())
            }
        };

        if let Some(key) = key_to_evict {
            let _ = self.remove(&key);
        }

        Ok(())
    }

    /// Clear cache
    pub fn clear(&self) {
        self.entries.write().clear();
        self.lru_order.write().clear();
        *self.current_size.write() = 0;
    }

    /// Get current cache size
    #[must_use]
    pub fn size(&self) -> u64 {
        *self.current_size.read()
    }

    /// Get number of entries
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let entries = self.entries.read();
        let total_entries = entries.len();
        let total_size = *self.current_size.read();
        let total_accesses: u64 = entries.values().map(|e| e.access_count).sum();

        CacheStats {
            total_entries,
            total_size,
            max_size: self.config.max_size,
            utilization: total_size as f64 / self.config.max_size as f64,
            total_accesses,
            avg_accesses: if total_entries > 0 {
                total_accesses as f64 / total_entries as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total entries
    pub total_entries: usize,
    /// Total size
    pub total_size: u64,
    /// Maximum size
    pub max_size: u64,
    /// Utilization (0.0 to 1.0)
    pub utilization: f64,
    /// Total accesses
    pub total_accesses: u64,
    /// Average accesses per entry
    pub avg_accesses: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let config = CacheConfig::default();
        let cache = CacheManager::new(config);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_put_get() -> Result<()> {
        let config = CacheConfig::default();
        let cache = CacheManager::new(config);

        let key = "test".to_string();
        let data = vec![1, 2, 3, 4, 5];

        cache.put(key.clone(), data.clone())?;
        let retrieved = cache.get(&key);

        assert_eq!(retrieved, Some(data));
        Ok(())
    }

    #[test]
    fn test_cache_remove() -> Result<()> {
        let config = CacheConfig::default();
        let cache = CacheManager::new(config);

        let key = "test".to_string();
        let data = vec![1, 2, 3];

        cache.put(key.clone(), data)?;
        let removed = cache.remove(&key);

        assert!(removed.is_some());
        assert!(cache.get(&key).is_none());
        Ok(())
    }

    #[test]
    fn test_cache_eviction() -> Result<()> {
        let config = CacheConfig {
            max_size: 100,
            eviction_policy: EvictionPolicy::LRU,
            ttl_seconds: 0,
        };
        let cache = CacheManager::new(config);

        // Fill cache
        cache.put("key1".to_string(), vec![0; 40])?;
        cache.put("key2".to_string(), vec![0; 40])?;

        // This should trigger eviction of key1
        cache.put("key3".to_string(), vec![0; 40])?;

        assert!(cache.get("key1").is_none());
        assert!(cache.get("key2").is_some());
        assert!(cache.get("key3").is_some());

        Ok(())
    }

    #[test]
    fn test_cache_clear() -> Result<()> {
        let config = CacheConfig::default();
        let cache = CacheManager::new(config);

        cache.put("key1".to_string(), vec![1, 2, 3])?;
        cache.put("key2".to_string(), vec![4, 5, 6])?;

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.size(), 0);

        Ok(())
    }

    #[test]
    fn test_cache_stats() -> Result<()> {
        let config = CacheConfig::default();
        let cache = CacheManager::new(config);

        cache.put("key1".to_string(), vec![0; 100])?;
        cache.put("key2".to_string(), vec![0; 200])?;

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.total_size, 300);

        Ok(())
    }

    #[test]
    fn test_lru_order() -> Result<()> {
        let config = CacheConfig {
            max_size: 200,
            eviction_policy: EvictionPolicy::LRU,
            ttl_seconds: 0,
        };
        let cache = CacheManager::new(config);

        cache.put("key1".to_string(), vec![0; 60])?;
        cache.put("key2".to_string(), vec![0; 60])?;

        // Access key1, making it more recently used
        let _ = cache.get("key1");

        // This should evict key2, not key1
        cache.put("key3".to_string(), vec![0; 100])?;

        assert!(cache.get("key1").is_some());
        assert!(cache.get("key2").is_none());
        assert!(cache.get("key3").is_some());

        Ok(())
    }
}
