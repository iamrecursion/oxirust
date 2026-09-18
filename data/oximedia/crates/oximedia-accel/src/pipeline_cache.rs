//! Pipeline caching to speed up shader compilation (VkPipelineCache analog).
//!
//! Provides a pure-Rust pipeline cache keyed by a Blake3-style FNV-1a hash of
//! shader source + specialisation constants.  On Vulkan, you would serialise
//! the cache to disk; here we serialise to a `Vec<u8>` blob for portability.

#![allow(dead_code)]

use crate::error::{AccelError, AccelResult};
use std::collections::HashMap;
use std::time::Instant;

/// FNV-1a 64-bit hash used as a lightweight cache key.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// Descriptor for a compute pipeline that can be cached.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineDescriptor {
    /// Shader source identifier (e.g. file name or embedded string key).
    pub shader_id: String,
    /// Specialisation constants encoded as raw bytes.
    pub spec_constants: Vec<u8>,
    /// Pipeline layout hash (push-constant size, descriptor set layout hash).
    pub layout_hash: u64,
}

impl PipelineDescriptor {
    /// Create a new descriptor.
    #[must_use]
    pub fn new(shader_id: impl Into<String>) -> Self {
        Self {
            shader_id: shader_id.into(),
            spec_constants: Vec::new(),
            layout_hash: 0,
        }
    }

    /// Set raw specialisation constant bytes.
    #[must_use]
    pub fn with_spec_constants(mut self, data: Vec<u8>) -> Self {
        self.spec_constants = data;
        self
    }

    /// Set the layout hash.
    #[must_use]
    pub fn with_layout_hash(mut self, hash: u64) -> Self {
        self.layout_hash = hash;
        self
    }

    /// Compute a stable 64-bit cache key from all descriptor fields.
    #[must_use]
    pub fn cache_key(&self) -> u64 {
        let mut buf = self.shader_id.as_bytes().to_vec();
        buf.extend_from_slice(&self.spec_constants);
        buf.extend_from_slice(&self.layout_hash.to_le_bytes());
        fnv1a_64(&buf)
    }
}

/// A cached pipeline blob.
#[derive(Debug, Clone)]
pub struct CachedPipeline {
    /// The pipeline descriptor that produced this cache entry.
    pub descriptor: PipelineDescriptor,
    /// Serialised pipeline data (in a real Vulkan implementation this would be
    /// the VkPipelineCache blob; here we store a placeholder).
    pub blob: Vec<u8>,
    /// When this entry was inserted.
    pub inserted_at: Instant,
    /// Number of times this entry has been retrieved.
    pub hit_count: u64,
}

impl CachedPipeline {
    fn new(descriptor: PipelineDescriptor, blob: Vec<u8>) -> Self {
        Self {
            descriptor,
            blob,
            inserted_at: Instant::now(),
            hit_count: 0,
        }
    }
}

/// Statistics for the pipeline cache.
#[derive(Debug, Clone, Default)]
pub struct PipelineCacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of entries currently in the cache.
    pub entries: usize,
    /// Total size of all cached blobs in bytes.
    pub total_blob_bytes: usize,
}

impl PipelineCacheStats {
    /// Cache hit ratio (0.0 when no requests have been made).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

/// In-process pipeline cache — analog to `VkPipelineCache`.
///
/// Stores compiled pipeline blobs keyed by a hash of the pipeline descriptor.
/// Supports import/export of the cache blob for disk persistence.
pub struct PipelineCache {
    entries: HashMap<u64, CachedPipeline>,
    max_entries: usize,
    stats: PipelineCacheStats,
}

impl PipelineCache {
    /// Create a cache with a given maximum number of entries.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries: max_entries.max(1),
            stats: PipelineCacheStats::default(),
        }
    }

    /// Create a cache with the default capacity (256 entries).
    #[must_use]
    pub fn default_capacity() -> Self {
        Self::new(256)
    }

    /// Look up a cached pipeline.
    ///
    /// Records a hit/miss in the stats.
    pub fn get(&mut self, descriptor: &PipelineDescriptor) -> Option<&CachedPipeline> {
        let key = descriptor.cache_key();
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.hit_count += 1;
            self.stats.hits += 1;
            Some(entry)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert or update a pipeline cache entry.
    ///
    /// When the cache is full, the oldest entry (by `inserted_at`) is evicted.
    ///
    /// # Errors
    ///
    /// Returns `AccelError::Unsupported` if the blob is empty (indicates a
    /// compilation that produced no output).
    pub fn insert(&mut self, descriptor: PipelineDescriptor, blob: Vec<u8>) -> AccelResult<u64> {
        if blob.is_empty() {
            return Err(AccelError::ShaderCompilation(
                "pipeline cache: refusing to store empty blob".to_string(),
            ));
        }

        let key = descriptor.cache_key();

        // Evict oldest if full.
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&key) {
            let oldest_key = self
                .entries
                .iter()
                .min_by_key(|(_, v)| v.inserted_at)
                .map(|(&k, _)| k);
            if let Some(k) = oldest_key {
                if let Some(entry) = self.entries.remove(&k) {
                    self.stats.total_blob_bytes =
                        self.stats.total_blob_bytes.saturating_sub(entry.blob.len());
                }
            }
        }

        self.stats.total_blob_bytes += blob.len();
        let entry = CachedPipeline::new(descriptor, blob);
        self.entries.insert(key, entry);
        self.stats.entries = self.entries.len();
        Ok(key)
    }

    /// Remove a specific entry.
    pub fn remove(&mut self, descriptor: &PipelineDescriptor) -> bool {
        let key = descriptor.cache_key();
        if let Some(entry) = self.entries.remove(&key) {
            self.stats.total_blob_bytes =
                self.stats.total_blob_bytes.saturating_sub(entry.blob.len());
            self.stats.entries = self.entries.len();
            true
        } else {
            false
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats.entries = 0;
        self.stats.total_blob_bytes = 0;
    }

    /// Number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current statistics.
    #[must_use]
    pub fn stats(&self) -> &PipelineCacheStats {
        &self.stats
    }

    /// Export cache contents as a flat blob for disk persistence.
    ///
    /// Format: `[u32 entry_count] ([u64 key] [u64 blob_len] [blob...])*`
    ///
    /// # Errors
    ///
    /// Currently infallible; kept as `AccelResult` for future compatibility.
    pub fn export(&self) -> AccelResult<Vec<u8>> {
        let mut out = Vec::new();
        let count = self.entries.len() as u32;
        out.extend_from_slice(&count.to_le_bytes());
        for (&key, entry) in &self.entries {
            out.extend_from_slice(&key.to_le_bytes());
            let blob_len = entry.blob.len() as u64;
            out.extend_from_slice(&blob_len.to_le_bytes());
            out.extend_from_slice(&entry.blob);
        }
        Ok(out)
    }

    /// Import a previously exported cache blob.
    ///
    /// Entries are merged into the existing cache (existing entries are
    /// overwritten by imported ones).
    ///
    /// # Errors
    ///
    /// Returns `AccelError::BufferSizeMismatch` if the blob is malformed.
    pub fn import(&mut self, data: &[u8]) -> AccelResult<usize> {
        let mut pos = 0;

        let count_bytes: [u8; 4] = data
            .get(pos..pos + 4)
            .and_then(|s| s.try_into().ok())
            .ok_or(AccelError::BufferSizeMismatch {
                expected: 4,
                actual: data.len(),
            })?;
        let count = u32::from_le_bytes(count_bytes) as usize;
        pos += 4;

        let mut imported = 0;
        for _ in 0..count {
            let key_bytes: [u8; 8] = data
                .get(pos..pos + 8)
                .and_then(|s| s.try_into().ok())
                .ok_or_else(|| AccelError::BufferSizeMismatch {
                    expected: pos + 8,
                    actual: data.len(),
                })?;
            let _key = u64::from_le_bytes(key_bytes);
            pos += 8;

            let len_bytes: [u8; 8] = data
                .get(pos..pos + 8)
                .and_then(|s| s.try_into().ok())
                .ok_or_else(|| AccelError::BufferSizeMismatch {
                    expected: pos + 8,
                    actual: data.len(),
                })?;
            let blob_len = u64::from_le_bytes(len_bytes) as usize;
            pos += 8;

            let blob = data
                .get(pos..pos + blob_len)
                .ok_or_else(|| AccelError::BufferSizeMismatch {
                    expected: pos + blob_len,
                    actual: data.len(),
                })?
                .to_vec();
            pos += blob_len;

            // Re-key by the actual descriptor; since we don't store the
            // descriptor fields in the export format we synthesise a synthetic
            // descriptor keyed by the stored key value.
            let desc = PipelineDescriptor {
                shader_id: format!("imported_{_key}"),
                spec_constants: Vec::new(),
                layout_hash: _key,
            };
            // Only insert if not already present (do not overwrite hot entries).
            if !self.entries.contains_key(&_key) {
                self.stats.total_blob_bytes += blob.len();
                self.entries.insert(_key, CachedPipeline::new(desc, blob));
            }
            imported += 1;
        }
        self.stats.entries = self.entries.len();
        Ok(imported)
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::default_capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_desc(id: &str) -> PipelineDescriptor {
        PipelineDescriptor::new(id)
    }

    fn make_blob(n: u8) -> Vec<u8> {
        vec![n; 64]
    }

    #[test]
    fn test_pipeline_cache_insert_and_get() {
        let mut cache = PipelineCache::new(10);
        let desc = make_desc("scale");
        cache
            .insert(desc.clone(), make_blob(1))
            .expect("insert should succeed");
        let entry = cache.get(&desc).expect("entry should exist");
        assert_eq!(entry.blob[0], 1);
    }

    #[test]
    fn test_pipeline_cache_miss() {
        let mut cache = PipelineCache::new(10);
        let desc = make_desc("missing");
        assert!(cache.get(&desc).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_pipeline_cache_hit_count() {
        let mut cache = PipelineCache::new(10);
        let desc = make_desc("scale");
        cache
            .insert(desc.clone(), make_blob(7))
            .expect("insert should succeed");
        cache.get(&desc);
        cache.get(&desc);
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
    }

    #[test]
    fn test_pipeline_cache_eviction_when_full() {
        let mut cache = PipelineCache::new(2);
        let d1 = make_desc("shader_a");
        let d2 = make_desc("shader_b");
        let d3 = make_desc("shader_c");
        cache.insert(d1.clone(), make_blob(1)).expect("insert d1");
        // Small sleep to ensure different Instant ordering.
        std::thread::sleep(std::time::Duration::from_millis(2));
        cache.insert(d2.clone(), make_blob(2)).expect("insert d2");
        // Inserting d3 should evict the oldest (d1).
        cache.insert(d3.clone(), make_blob(3)).expect("insert d3");
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_pipeline_cache_remove() {
        let mut cache = PipelineCache::new(10);
        let desc = make_desc("scale");
        cache
            .insert(desc.clone(), make_blob(1))
            .expect("insert should succeed");
        assert!(cache.remove(&desc));
        assert!(!cache.remove(&desc)); // double-remove returns false
        assert!(cache.is_empty());
    }

    #[test]
    fn test_pipeline_cache_clear() {
        let mut cache = PipelineCache::new(10);
        cache
            .insert(make_desc("a"), make_blob(1))
            .expect("insert a");
        cache
            .insert(make_desc("b"), make_blob(2))
            .expect("insert b");
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().total_blob_bytes, 0);
    }

    #[test]
    fn test_pipeline_cache_empty_blob_rejected() {
        let mut cache = PipelineCache::new(10);
        let desc = make_desc("empty");
        let result = cache.insert(desc, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_cache_export_import_roundtrip() {
        let mut cache = PipelineCache::new(10);
        cache
            .insert(make_desc("s1"), make_blob(0xAA))
            .expect("insert s1");
        cache
            .insert(make_desc("s2"), make_blob(0xBB))
            .expect("insert s2");

        let blob = cache.export().expect("export should succeed");

        let mut cache2 = PipelineCache::new(10);
        let count = cache2.import(&blob).expect("import should succeed");
        assert_eq!(count, 2);
        assert_eq!(cache2.len(), 2);
    }

    #[test]
    fn test_pipeline_cache_import_malformed() {
        let mut cache = PipelineCache::new(10);
        let bad = vec![0u8; 2]; // too short to have count
        let result = cache.import(&bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_descriptor_cache_key_stable() {
        let d1 = make_desc("scale");
        let d2 = make_desc("scale");
        assert_eq!(d1.cache_key(), d2.cache_key());
    }

    #[test]
    fn test_pipeline_descriptor_cache_key_differs() {
        let d1 = make_desc("scale");
        let d2 = make_desc("color");
        assert_ne!(d1.cache_key(), d2.cache_key());
    }

    #[test]
    fn test_pipeline_stats_hit_ratio() {
        let mut stats = PipelineCacheStats::default();
        assert_eq!(stats.hit_ratio(), 0.0);
        stats.hits = 3;
        stats.misses = 1;
        assert!((stats.hit_ratio() - 0.75).abs() < 1e-9);
    }
}
