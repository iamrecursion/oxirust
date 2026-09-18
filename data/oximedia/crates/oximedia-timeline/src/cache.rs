//! Caching system for optimized playback.

use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use crate::clip::ClipId;
use crate::types::Position;

/// Cache entry with timestamp.
#[derive(Clone, Debug)]
struct CacheEntry<T> {
    data: T,
    timestamp: SystemTime,
    size: usize,
}

/// Frame cache for video playback.
pub struct FrameCache {
    cache: Arc<DashMap<Position, CacheEntry<Vec<u8>>>>,
    max_size: usize,
    current_size: usize,
}

impl FrameCache {
    /// Creates a new frame cache.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            max_size,
            current_size: 0,
        }
    }

    /// Adds a frame to the cache.
    pub fn insert(&mut self, position: Position, frame: Vec<u8>) {
        let size = frame.len();

        // Evict if necessary
        while self.current_size + size > self.max_size && !self.cache.is_empty() {
            self.evict_oldest();
        }

        let entry = CacheEntry {
            data: frame,
            timestamp: SystemTime::now(),
            size,
        };

        self.cache.insert(position, entry);
        self.current_size += size;
    }

    /// Gets a frame from the cache.
    #[must_use]
    pub fn get(&self, position: Position) -> Option<Vec<u8>> {
        self.cache.get(&position).map(|entry| entry.data.clone())
    }

    /// Checks if a frame is cached.
    #[must_use]
    pub fn contains(&self, position: Position) -> bool {
        self.cache.contains_key(&position)
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.current_size = 0;
    }

    /// Evicts the oldest entry.
    fn evict_oldest(&mut self) {
        let mut oldest_pos = None;
        let mut oldest_time = SystemTime::now();

        for entry in self.cache.iter() {
            if entry.value().timestamp < oldest_time {
                oldest_time = entry.value().timestamp;
                oldest_pos = Some(*entry.key());
            }
        }

        if let Some(pos) = oldest_pos {
            if let Some((_, entry)) = self.cache.remove(&pos) {
                self.current_size -= entry.size;
            }
        }
    }

    /// Pre-caches frames in a range.
    pub fn precache_range(
        &mut self,
        start: Position,
        end: Position,
        frame_provider: impl Fn(Position) -> Vec<u8>,
    ) {
        for frame in start.value()..end.value() {
            let position = Position::new(frame);
            if !self.contains(position) {
                let frame_data = frame_provider(position);
                self.insert(position, frame_data);
            }
        }
    }

    /// Returns cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entry_count: self.cache.len(),
            total_size: self.current_size,
            max_size: self.max_size,
        }
    }
}

/// Audio buffer cache.
pub struct AudioCache {
    cache: Arc<DashMap<Position, CacheEntry<Vec<f32>>>>,
    max_size: usize,
    current_size: usize,
}

impl AudioCache {
    /// Creates a new audio cache.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            max_size,
            current_size: 0,
        }
    }

    /// Adds audio samples to the cache.
    pub fn insert(&mut self, position: Position, samples: Vec<f32>) {
        let size = samples.len() * std::mem::size_of::<f32>();

        while self.current_size + size > self.max_size && !self.cache.is_empty() {
            self.evict_oldest();
        }

        let entry = CacheEntry {
            data: samples,
            timestamp: SystemTime::now(),
            size,
        };

        self.cache.insert(position, entry);
        self.current_size += size;
    }

    /// Gets audio samples from the cache.
    #[must_use]
    pub fn get(&self, position: Position) -> Option<Vec<f32>> {
        self.cache.get(&position).map(|entry| entry.data.clone())
    }

    /// Checks if audio is cached.
    #[must_use]
    pub fn contains(&self, position: Position) -> bool {
        self.cache.contains_key(&position)
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.current_size = 0;
    }

    /// Evicts the oldest entry.
    fn evict_oldest(&mut self) {
        let mut oldest_pos = None;
        let mut oldest_time = SystemTime::now();

        for entry in self.cache.iter() {
            if entry.value().timestamp < oldest_time {
                oldest_time = entry.value().timestamp;
                oldest_pos = Some(*entry.key());
            }
        }

        if let Some(pos) = oldest_pos {
            if let Some((_, entry)) = self.cache.remove(&pos) {
                self.current_size -= entry.size;
            }
        }
    }

    /// Returns cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entry_count: self.cache.len(),
            total_size: self.current_size,
            max_size: self.max_size,
        }
    }
}

/// Effect cache for rendered effects.
pub struct EffectCache {
    cache: Arc<DashMap<(ClipId, Position), CacheEntry<Vec<u8>>>>,
    max_age: Duration,
}

impl EffectCache {
    /// Creates a new effect cache.
    #[must_use]
    pub fn new(max_age: Duration) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            max_age,
        }
    }

    /// Adds rendered effect to cache.
    pub fn insert(&self, clip_id: ClipId, position: Position, data: Vec<u8>) {
        let size = data.len();
        let entry = CacheEntry {
            data,
            timestamp: SystemTime::now(),
            size,
        };
        self.cache.insert((clip_id, position), entry);
    }

    /// Gets rendered effect from cache.
    #[must_use]
    pub fn get(&self, clip_id: ClipId, position: Position) -> Option<Vec<u8>> {
        let key = (clip_id, position);
        if let Some(entry) = self.cache.get(&key) {
            // Check if entry is still valid
            if let Ok(age) = SystemTime::now().duration_since(entry.timestamp) {
                if age < self.max_age {
                    return Some(entry.data.clone());
                }
            }
            // Entry too old, remove it
            drop(entry);
            self.cache.remove(&key);
        }
        None
    }

    /// Invalidates cache for a clip.
    pub fn invalidate_clip(&self, clip_id: ClipId) {
        self.cache.retain(|(id, _), _| *id != clip_id);
    }

    /// Clears old entries.
    pub fn cleanup(&self) {
        let now = SystemTime::now();
        self.cache.retain(|_, entry| {
            if let Ok(age) = now.duration_since(entry.timestamp) {
                age < self.max_age
            } else {
                false
            }
        });
    }

    /// Clears the cache.
    pub fn clear(&self) {
        self.cache.clear();
    }
}

/// Cache statistics.
#[derive(Clone, Debug)]
pub struct CacheStats {
    /// Number of entries.
    pub entry_count: usize,
    /// Total size in bytes.
    pub total_size: usize,
    /// Maximum size in bytes.
    pub max_size: usize,
}

impl CacheStats {
    /// Calculates cache usage percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn usage_percentage(&self) -> f64 {
        if self.max_size == 0 {
            return 0.0;
        }
        (self.total_size as f64 / self.max_size as f64) * 100.0
    }

    /// Checks if cache is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.total_size >= self.max_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_cache_insert_get() {
        let mut cache = FrameCache::new(1024);
        let frame = vec![0u8; 256];
        cache.insert(Position::new(0), frame.clone());
        assert_eq!(cache.get(Position::new(0)), Some(frame));
    }

    #[test]
    fn test_frame_cache_contains() {
        let mut cache = FrameCache::new(1024);
        cache.insert(Position::new(0), vec![0u8; 256]);
        assert!(cache.contains(Position::new(0)));
        assert!(!cache.contains(Position::new(1)));
    }

    #[test]
    fn test_frame_cache_eviction() {
        let mut cache = FrameCache::new(512);
        cache.insert(Position::new(0), vec![0u8; 300]);
        cache.insert(Position::new(1), vec![0u8; 300]);
        // Second insert should trigger eviction
        assert!(cache.current_size <= cache.max_size);
    }

    #[test]
    fn test_frame_cache_clear() {
        let mut cache = FrameCache::new(1024);
        cache.insert(Position::new(0), vec![0u8; 256]);
        cache.clear();
        assert!(!cache.contains(Position::new(0)));
        assert_eq!(cache.current_size, 0);
    }

    #[test]
    fn test_audio_cache_insert_get() {
        let mut cache = AudioCache::new(1024);
        let samples = vec![0.0f32; 64];
        cache.insert(Position::new(0), samples.clone());
        assert_eq!(cache.get(Position::new(0)), Some(samples));
    }

    #[test]
    fn test_audio_cache_contains() {
        let mut cache = AudioCache::new(1024);
        cache.insert(Position::new(0), vec![0.0f32; 64]);
        assert!(cache.contains(Position::new(0)));
        assert!(!cache.contains(Position::new(1)));
    }

    #[test]
    fn test_effect_cache() {
        let cache = EffectCache::new(Duration::from_secs(60));
        let clip_id = ClipId::new();
        let data = vec![0u8; 256];

        cache.insert(clip_id, Position::new(0), data.clone());
        assert_eq!(cache.get(clip_id, Position::new(0)), Some(data));
    }

    #[test]
    fn test_effect_cache_invalidate() {
        let cache = EffectCache::new(Duration::from_secs(60));
        let clip_id = ClipId::new();
        cache.insert(clip_id, Position::new(0), vec![0u8; 256]);
        cache.invalidate_clip(clip_id);
        assert!(cache.get(clip_id, Position::new(0)).is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = FrameCache::new(1024);
        let stats = cache.stats();
        assert_eq!(stats.max_size, 1024);
        assert_eq!(stats.entry_count, 0);
        assert!((stats.usage_percentage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_stats_full() {
        let mut cache = FrameCache::new(512);
        cache.insert(Position::new(0), vec![0u8; 512]);
        let stats = cache.stats();
        assert!(stats.is_full());
    }
}
