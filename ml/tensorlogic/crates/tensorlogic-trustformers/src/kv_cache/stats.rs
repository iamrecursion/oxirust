use super::simple_cache::KvCache;

/// Tracks statistics for autoregressive token generation with KV-cache.
#[derive(Debug, Clone)]
pub struct InferenceStats {
    /// Total tokens generated so far.
    pub tokens_generated: usize,
    /// Number of cache reads that avoided recomputation.
    pub cache_hits: usize,
    /// Total number of attention operations executed.
    pub total_attention_ops: usize,
    /// Running average of cache sequence length across recorded steps.
    pub avg_cache_len: f64,
    /// Peak memory usage in bytes observed across all steps.
    pub peak_memory_bytes: usize,
}

impl InferenceStats {
    /// Create a zeroed `InferenceStats`.
    pub fn new() -> Self {
        Self {
            tokens_generated: 0,
            cache_hits: 0,
            total_attention_ops: 0,
            avg_cache_len: 0.0,
            peak_memory_bytes: 0,
        }
    }

    /// Record a single generation step given the current cache state.
    pub fn record_step(&mut self, cache: &KvCache) {
        self.tokens_generated += 1;
        let cache_len = cache.current_len();
        if cache_len > 0 {
            self.cache_hits += 1;
        }
        self.total_attention_ops += 1;

        // Exponential moving average of cache length.
        let n = self.total_attention_ops as f64;
        self.avg_cache_len = ((n - 1.0) * self.avg_cache_len + cache_len as f64) / n;

        let mem = cache.memory_usage_bytes();
        if mem > self.peak_memory_bytes {
            self.peak_memory_bytes = mem;
        }
    }

    /// Return a human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "InferenceStats {{ tokens_generated: {}, cache_hits: {}, total_attention_ops: {}, avg_cache_len: {:.1}, peak_memory_bytes: {} }}",
            self.tokens_generated,
            self.cache_hits,
            self.total_attention_ops,
            self.avg_cache_len,
            self.peak_memory_bytes
        )
    }
}

impl Default for InferenceStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_stats_record_step() {
        let mut stats = InferenceStats::new();
        let cache = KvCache::new(1, 2, 4, 16);
        stats.record_step(&cache);
        assert_eq!(
            stats.tokens_generated, 1,
            "tokens_generated should increment"
        );
    }

    #[test]
    fn test_inference_stats_summary_non_empty() {
        let stats = InferenceStats::new();
        let s = stats.summary();
        assert!(!s.is_empty(), "summary must return non-empty string");
        assert!(
            s.contains("tokens_generated"),
            "summary should contain field names"
        );
    }
}
