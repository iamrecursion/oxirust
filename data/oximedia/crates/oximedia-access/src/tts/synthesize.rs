//! Text-to-speech synthesis with LRU-style result caching.

use crate::error::{AccessError, AccessResult};
use crate::tts::TtsConfig;
use bytes::Bytes;
use oximedia_audio::frame::AudioBuffer;
use std::collections::{HashMap, VecDeque};

/// Maximum number of entries in the TTS cache.
const TTS_CACHE_MAX_SIZE: usize = 256;

/// A cached TTS synthesis result.
#[derive(Debug, Clone)]
pub struct TtsCacheEntry {
    /// The synthesized audio buffer.
    pub audio: AudioBuffer,
    /// Number of times this entry has been accessed.
    pub hit_count: u64,
    /// Original text that was synthesized.
    pub text: String,
}

impl TtsCacheEntry {
    fn new(text: String, audio: AudioBuffer) -> Self {
        Self {
            audio,
            hit_count: 0,
            text,
        }
    }
}

/// Statistics about the TTS cache.
#[derive(Debug, Clone, Default)]
pub struct TtsCacheStats {
    /// Total number of cache hits.
    pub hits: u64,
    /// Total number of cache misses.
    pub misses: u64,
    /// Current number of entries in the cache.
    pub size: usize,
    /// Total number of evictions.
    pub evictions: u64,
}

impl TtsCacheStats {
    /// Hit ratio (0.0 to 1.0).
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Text-to-speech synthesizer with result caching.
pub struct TextToSpeech {
    config: TtsConfig,
    /// Cache mapping synthesis key -> cached entry.
    cache: HashMap<String, TtsCacheEntry>,
    /// Cache statistics.
    stats: TtsCacheStats,
}

impl TextToSpeech {
    /// Create a new TTS synthesizer.
    #[must_use]
    pub fn new(config: TtsConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            stats: TtsCacheStats::default(),
        }
    }

    /// Build the cache key from configuration and text.
    fn cache_key(&self, text: &str) -> String {
        format!(
            "{}|{}|{:.3}|{:.3}",
            self.config.voice, text, self.config.rate, self.config.pitch
        )
    }

    /// Evict the entry with the lowest hit count to make room.
    fn evict_one(&mut self) {
        if self.cache.is_empty() {
            return;
        }
        let evict_key = self
            .cache
            .iter()
            .min_by_key(|(_, v)| v.hit_count)
            .map(|(k, _)| k.clone());

        if let Some(key) = evict_key {
            self.cache.remove(&key);
            self.stats.evictions += 1;
        }
    }

    /// Synthesize text to speech, using the cache when available.
    ///
    /// Integration point for TTS services:
    /// - Amazon Polly
    /// - Google Cloud Text-to-Speech
    /// - Microsoft Azure Speech
    /// - IBM Watson Text to Speech
    /// - Local engines (eSpeak, Festival, Piper, etc.)
    pub fn synthesize(&mut self, text: &str) -> AccessResult<AudioBuffer> {
        if text.is_empty() {
            return Err(AccessError::TtsFailed("Empty text".to_string()));
        }

        let key = self.cache_key(text);

        if let Some(entry) = self.cache.get_mut(&key) {
            entry.hit_count += 1;
            self.stats.hits += 1;
            return Ok(entry.audio.clone());
        }

        self.stats.misses += 1;
        let audio = self.synthesize_raw(text)?;

        if self.cache.len() >= TTS_CACHE_MAX_SIZE {
            self.evict_one();
        }

        self.cache
            .insert(key, TtsCacheEntry::new(text.to_string(), audio.clone()));
        self.stats.size = self.cache.len();

        Ok(audio)
    }

    /// Internal synthesis without caching (raw synthesis logic).
    fn synthesize_raw(&self, text: &str) -> AccessResult<AudioBuffer> {
        let duration_samples = text.len() * 100;
        let samples = vec![0.0f32; duration_samples * 2];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        Ok(AudioBuffer::Interleaved(Bytes::from(bytes)))
    }

    /// Synthesize with SSML markup.
    pub fn synthesize_ssml(&mut self, ssml: &str) -> AccessResult<AudioBuffer> {
        self.synthesize(ssml)
    }

    /// Pre-synthesize a list of text segments and cache the results.
    pub fn prefetch(&mut self, texts: &[&str]) -> AccessResult<()> {
        for text in texts {
            self.synthesize(text)?;
        }
        Ok(())
    }

    /// Invalidate a specific entry in the cache.
    pub fn invalidate(&mut self, text: &str) {
        let key = self.cache_key(text);
        if self.cache.remove(&key).is_some() {
            self.stats.size = self.cache.len();
        }
    }

    /// Clear the entire cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.stats.size = 0;
    }

    /// Get a snapshot of cache statistics.
    #[must_use]
    pub fn cache_stats(&self) -> TtsCacheStats {
        TtsCacheStats {
            size: self.cache.len(),
            ..self.stats.clone()
        }
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &TtsConfig {
        &self.config
    }

    /// Number of entries currently in the cache.
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

impl Default for TextToSpeech {
    fn default() -> Self {
        Self::new(TtsConfig::default())
    }
}

// ─── Standalone TTS Synthesis Cache ─────────────────────────────────────────

/// A simple LRU-style cache for raw TTS synthesis results.
///
/// Keys on `(text, voice_id)` pairs and stores the synthesized audio as raw
/// bytes (`Vec<u8>`).  When the cache is full the oldest-inserted entry is
/// evicted (FIFO order), keeping memory consumption bounded.
///
/// # Example
///
/// ```rust
/// use oximedia_access::tts::synthesize::TtsSynthesisCache;
///
/// let mut cache = TtsSynthesisCache::new(128);
/// cache.insert("Hello".to_string(), "en-US-Neural".to_string(), vec![0u8; 64]);
/// assert!(cache.get("Hello", "en-US-Neural").is_some());
/// ```
pub struct TtsSynthesisCache {
    /// Key → synthesized audio bytes.
    map: HashMap<(String, String), Vec<u8>>,
    /// Insertion-order queue for eviction (FIFO LRU approximation).
    order: VecDeque<(String, String)>,
    /// Maximum number of entries before eviction.
    capacity: usize,
}

impl TtsSynthesisCache {
    /// Create a new cache with the given capacity.
    ///
    /// A capacity of 0 is silently promoted to 1 to avoid degenerate behaviour.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    /// Look up a cached synthesis result by text and voice identifier.
    ///
    /// Returns `Some(&audio_bytes)` on a hit, `None` on a miss.
    #[must_use]
    pub fn get(&self, text: &str, voice_id: &str) -> Option<&Vec<u8>> {
        self.map.get(&(text.to_string(), voice_id.to_string()))
    }

    /// Insert a synthesis result into the cache.
    ///
    /// If the cache is at capacity the oldest entry is evicted before the new
    /// entry is stored.  If `(text, voice_id)` already exists its value is
    /// overwritten in place (no duplicate in the eviction queue).
    pub fn insert(&mut self, text: String, voice_id: String, audio: Vec<u8>) {
        use std::collections::hash_map::Entry;
        let key = (text, voice_id);
        if let Entry::Occupied(mut e) = self.map.entry(key.clone()) {
            // Update existing — just overwrite the bytes, keep queue order.
            e.insert(audio);
            return;
        }
        if self.map.len() >= self.capacity {
            if let Some(evict) = self.order.pop_front() {
                self.map.remove(&evict);
            }
        }
        self.map.insert(key.clone(), audio);
        self.order.push_back(key);
    }

    /// Number of entries currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    /// The maximum number of entries this cache will hold before eviction.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_creation() {
        let tts = TextToSpeech::default();
        assert_eq!(tts.config().sample_rate, 24000);
    }

    #[test]
    fn test_synthesize() {
        let mut tts = TextToSpeech::default();
        let result = tts.synthesize("Hello world");
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_text() {
        let mut tts = TextToSpeech::default();
        let result = tts.synthesize("");
        assert!(result.is_err());
    }

    #[test]
    fn test_cache_miss_then_hit() {
        let mut tts = TextToSpeech::default();
        tts.synthesize("Hello cache").expect("synthesis ok");
        let stats = tts.cache_stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
        tts.synthesize("Hello cache").expect("synthesis ok");
        let stats = tts.cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_different_texts() {
        let mut tts = TextToSpeech::default();
        tts.synthesize("First sentence").expect("ok");
        tts.synthesize("Second sentence").expect("ok");
        assert_eq!(tts.cache_size(), 2);
    }

    #[test]
    fn test_cache_invalidate() {
        let mut tts = TextToSpeech::default();
        tts.synthesize("Test text").expect("ok");
        assert_eq!(tts.cache_size(), 1);
        tts.invalidate("Test text");
        assert_eq!(tts.cache_size(), 0);
    }

    #[test]
    fn test_cache_clear() {
        let mut tts = TextToSpeech::default();
        tts.synthesize("A").expect("ok");
        tts.synthesize("B").expect("ok");
        tts.synthesize("C").expect("ok");
        assert_eq!(tts.cache_size(), 3);
        tts.clear_cache();
        assert_eq!(tts.cache_size(), 0);
    }

    #[test]
    fn test_prefetch() {
        let mut tts = TextToSpeech::default();
        let texts = ["Hello", "World", "Goodbye"];
        tts.prefetch(&texts).expect("prefetch ok");
        assert_eq!(tts.cache_size(), 3);
        tts.synthesize("Hello").expect("ok");
        let stats = tts.cache_stats();
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn test_cache_eviction_on_overflow() {
        let mut tts = TextToSpeech::default();
        for i in 0..TTS_CACHE_MAX_SIZE {
            tts.synthesize(&format!("Entry {i}")).expect("ok");
        }
        assert_eq!(tts.cache_size(), TTS_CACHE_MAX_SIZE);
        tts.synthesize("Overflow entry").expect("ok");
        assert_eq!(tts.cache_size(), TTS_CACHE_MAX_SIZE);
        let stats = tts.cache_stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_hit_ratio() {
        let mut tts = TextToSpeech::default();
        tts.synthesize("X").expect("ok");
        tts.synthesize("X").expect("ok");
        tts.synthesize("X").expect("ok");
        let stats = tts.cache_stats();
        let ratio = stats.hit_ratio();
        assert!((ratio - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_hit_ratio_no_requests() {
        let tts = TextToSpeech::default();
        assert!((tts.cache_stats().hit_ratio()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ssml_synthesis() {
        let mut tts = TextToSpeech::default();
        let ssml = "<speak>Hello <break time=\"500ms\"/> world.</speak>";
        let result = tts.synthesize_ssml(ssml);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache_size_after_invalidate_missing() {
        let mut tts = TextToSpeech::default();
        tts.synthesize("Present").expect("ok");
        tts.invalidate("Non-existent text segment");
        assert_eq!(tts.cache_size(), 1);
    }

    #[test]
    fn test_repeated_prefetch_uses_cache() {
        let mut tts = TextToSpeech::default();
        tts.prefetch(&["Line 1", "Line 2"]).expect("ok");
        tts.prefetch(&["Line 1", "Line 2"]).expect("ok");
        let stats = tts.cache_stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 2);
    }

    // ─── TtsSynthesisCache tests ───────────────────────────────────────────

    #[test]
    fn test_tts_cache_hit() {
        let mut cache = TtsSynthesisCache::new(16);
        let audio = vec![1u8, 2u8, 3u8, 4u8];
        cache.insert(
            "Hello".to_string(),
            "en-US-Neural".to_string(),
            audio.clone(),
        );
        let result = cache.get("Hello", "en-US-Neural");
        assert!(result.is_some(), "expected a cache hit");
        assert_eq!(result.expect("should hit"), &audio);
    }

    #[test]
    fn test_tts_cache_miss() {
        let cache = TtsSynthesisCache::new(16);
        let result = cache.get("Unseen text", "en-US-Neural");
        assert!(result.is_none(), "expected a cache miss for unseen key");
    }

    #[test]
    fn test_tts_cache_eviction() {
        let capacity = 4usize;
        let mut cache = TtsSynthesisCache::new(capacity);
        // Fill to capacity; the earliest entry has key ("entry-0", "voice")
        for i in 0..capacity {
            cache.insert(format!("entry-{i}"), "voice".to_string(), vec![i as u8]);
        }
        assert_eq!(cache.len(), capacity);
        // Insert one more — should evict "entry-0"
        cache.insert(
            "entry-overflow".to_string(),
            "voice".to_string(),
            vec![99u8],
        );
        assert_eq!(
            cache.len(),
            capacity,
            "cache size must remain at capacity after eviction"
        );
        assert!(
            cache.get("entry-0", "voice").is_none(),
            "earliest entry must have been evicted"
        );
        assert!(
            cache.get("entry-overflow", "voice").is_some(),
            "newly inserted entry must be present"
        );
    }
}
