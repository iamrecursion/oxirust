//! Cache eviction strategies.

/// Cache eviction strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Least Recently Used - evict least recently accessed items.
    Lru,

    /// Least Frequently Used - evict least frequently accessed items.
    Lfu,

    /// First In First Out - evict oldest items.
    Fifo,
}

impl CacheStrategy {
    /// Get the name of this strategy.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Lru => "LRU",
            Self::Lfu => "LFU",
            Self::Fifo => "FIFO",
        }
    }
}

impl Default for CacheStrategy {
    fn default() -> Self {
        Self::Lru
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_strategy() {
        assert_eq!(CacheStrategy::Lru.name(), "LRU");
        assert_eq!(CacheStrategy::Lfu.name(), "LFU");
        assert_eq!(CacheStrategy::Fifo.name(), "FIFO");
    }

    #[test]
    fn test_default_strategy() {
        assert_eq!(CacheStrategy::default(), CacheStrategy::Lru);
    }
}
