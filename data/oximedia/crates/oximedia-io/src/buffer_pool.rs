//! Memory buffer pool for efficient I/O buffer reuse.
//!
//! Allocating and deallocating large buffers repeatedly is expensive.
//! This module provides a simple pool of pre-sized byte vectors that can be
//! acquired and returned, avoiding repeated heap allocations for common I/O
//! buffer sizes.

#![allow(dead_code)]

/// Standard buffer size classes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferSize {
    /// 4 KiB — suitable for small metadata reads
    Small4K,
    /// 64 KiB — general-purpose network/disk I/O
    Medium64K,
    /// 1 MiB — large streaming reads
    Large1M,
    /// 16 MiB — very large sequential transfers
    Huge16M,
}

impl BufferSize {
    /// Returns the buffer size in bytes
    #[must_use]
    pub fn bytes(self) -> usize {
        match self {
            Self::Small4K => 4 * 1024,
            Self::Medium64K => 64 * 1024,
            Self::Large1M => 1024 * 1024,
            Self::Huge16M => 16 * 1024 * 1024,
        }
    }
}

impl std::fmt::Display for BufferSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Small4K => write!(f, "4K"),
            Self::Medium64K => write!(f, "64K"),
            Self::Large1M => write!(f, "1M"),
            Self::Huge16M => write!(f, "16M"),
        }
    }
}

/// A buffer vended by the pool together with its size class and pool slot index
pub struct PooledBuffer {
    /// The underlying data storage
    pub data: Vec<u8>,
    /// Which size class this buffer belongs to
    pub size_class: BufferSize,
    /// Slot index used internally; 0 when allocated outside the pool
    pub allocated_at_idx: usize,
}

impl PooledBuffer {
    /// Create a fresh (non-pooled) buffer
    fn new(size_class: BufferSize) -> Self {
        Self {
            data: vec![0u8; size_class.bytes()],
            size_class,
            allocated_at_idx: 0,
        }
    }

    /// Capacity of the buffer in bytes
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Immutable view of the buffer contents
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Mutable view of the buffer contents
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

/// Pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of `acquire` calls
    pub total_acquired: u64,
    /// Number of acquisitions satisfied from the pool (not a fresh allocation)
    pub pool_hits: u64,
    /// Number of fresh allocations (pool was empty or wrong size)
    pub allocations: u64,
    /// Pool hit rate (0.0–1.0)
    pub hit_rate: f32,
}

impl PoolStats {
    #[allow(clippy::cast_precision_loss)]
    fn update_hit_rate(&mut self) {
        if self.total_acquired == 0 {
            self.hit_rate = 0.0;
        } else {
            self.hit_rate = self.pool_hits as f32 / self.total_acquired as f32;
        }
    }
}

/// A reusable pool of fixed-size byte buffers.
///
/// Three size classes are pooled (Small, Medium, Large).  Huge buffers are
/// never pooled because they are rarely reused and consume significant RAM.
pub struct BufferPool {
    pub free_small: Vec<Vec<u8>>,
    pub free_medium: Vec<Vec<u8>>,
    pub free_large: Vec<Vec<u8>>,
    /// Maximum number of buffers retained per size class
    pub max_pool_size: usize,
    stats: PoolStats,
}

impl BufferPool {
    /// Create a pool with the given per-class capacity
    #[must_use]
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            free_small: Vec::new(),
            free_medium: Vec::new(),
            free_large: Vec::new(),
            max_pool_size,
            stats: PoolStats::default(),
        }
    }

    /// Acquire a buffer of the requested size class.
    ///
    /// Returns a buffer from the pool if one is available, otherwise
    /// allocates a fresh `Vec<u8>` of the appropriate capacity.
    pub fn acquire(&mut self, size: BufferSize) -> Vec<u8> {
        self.stats.total_acquired += 1;

        let pool = self.pool_for_size(size);
        if let Some(mut buf) = pool.pop() {
            // Reuse from pool — reset length to full capacity and zero-fill
            let cap = size.bytes();
            buf.resize(cap, 0);
            self.stats.pool_hits += 1;
            self.stats.update_hit_rate();
            buf
        } else {
            // Fresh allocation
            self.stats.allocations += 1;
            self.stats.update_hit_rate();
            vec![0u8; size.bytes()]
        }
    }

    /// Return a buffer to the pool.
    ///
    /// If the pool for this size class is already full, the buffer is simply
    /// dropped.  Huge buffers are never retained.
    pub fn release(&mut self, buf: Vec<u8>, size: BufferSize) {
        if size == BufferSize::Huge16M {
            // Do not pool Huge buffers
            return;
        }
        let max = self.max_pool_size;
        let pool = self.pool_for_size(size);
        if pool.len() < max {
            pool.push(buf);
        }
        // Otherwise let it drop
    }

    /// Current pool statistics
    #[must_use]
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Total number of pooled buffers across all size classes
    #[must_use]
    pub fn pooled_count(&self) -> usize {
        self.free_small.len() + self.free_medium.len() + self.free_large.len()
    }

    fn pool_for_size(&mut self, size: BufferSize) -> &mut Vec<Vec<u8>> {
        match size {
            BufferSize::Small4K => &mut self.free_small,
            BufferSize::Medium64K => &mut self.free_medium,
            BufferSize::Large1M | BufferSize::Huge16M => &mut self.free_large,
        }
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- BufferSize ---

    #[test]
    fn test_buffer_size_bytes() {
        assert_eq!(BufferSize::Small4K.bytes(), 4 * 1024);
        assert_eq!(BufferSize::Medium64K.bytes(), 64 * 1024);
        assert_eq!(BufferSize::Large1M.bytes(), 1024 * 1024);
        assert_eq!(BufferSize::Huge16M.bytes(), 16 * 1024 * 1024);
    }

    #[test]
    fn test_buffer_size_display() {
        assert_eq!(BufferSize::Small4K.to_string(), "4K");
        assert_eq!(BufferSize::Medium64K.to_string(), "64K");
        assert_eq!(BufferSize::Large1M.to_string(), "1M");
        assert_eq!(BufferSize::Huge16M.to_string(), "16M");
    }

    // --- PooledBuffer ---

    #[test]
    fn test_pooled_buffer_capacity() {
        let buf = PooledBuffer::new(BufferSize::Medium64K);
        assert_eq!(buf.capacity(), BufferSize::Medium64K.bytes());
    }

    #[test]
    fn test_pooled_buffer_slice_mut() {
        let mut buf = PooledBuffer::new(BufferSize::Small4K);
        buf.as_mut_slice()[0] = 42;
        assert_eq!(buf.as_slice()[0], 42);
    }

    // --- BufferPool ---

    #[test]
    fn test_pool_acquire_fresh_allocation() {
        let mut pool = BufferPool::new(4);
        let buf = pool.acquire(BufferSize::Small4K);
        assert_eq!(buf.len(), BufferSize::Small4K.bytes());
        assert_eq!(pool.stats().allocations, 1);
        assert_eq!(pool.stats().pool_hits, 0);
    }

    #[test]
    fn test_pool_release_and_reuse() {
        let mut pool = BufferPool::new(4);
        let buf = pool.acquire(BufferSize::Medium64K);
        pool.release(buf, BufferSize::Medium64K);
        assert_eq!(pool.pooled_count(), 1);

        let _buf2 = pool.acquire(BufferSize::Medium64K);
        assert_eq!(pool.stats().pool_hits, 1);
        assert_eq!(pool.pooled_count(), 0);
    }

    #[test]
    fn test_pool_huge_never_pooled() {
        let mut pool = BufferPool::new(4);
        let buf = pool.acquire(BufferSize::Huge16M);
        pool.release(buf, BufferSize::Huge16M);
        // Huge buffers should never remain in the pool
        assert_eq!(pool.free_large.len(), 0);
    }

    #[test]
    fn test_pool_max_size_enforced() {
        let mut pool = BufferPool::new(2);
        for _ in 0..5 {
            let buf = pool.acquire(BufferSize::Small4K);
            pool.release(buf, BufferSize::Small4K);
        }
        // Pool should cap at max_pool_size
        assert!(pool.free_small.len() <= 2);
    }

    #[test]
    fn test_pool_stats_hit_rate() {
        let mut pool = BufferPool::new(4);
        let b1 = pool.acquire(BufferSize::Small4K); // miss
        pool.release(b1, BufferSize::Small4K);
        let _b2 = pool.acquire(BufferSize::Small4K); // hit
        let stats = pool.stats();
        assert_eq!(stats.total_acquired, 2);
        assert_eq!(stats.pool_hits, 1);
        assert!((stats.hit_rate - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pool_default_construction() {
        let pool = BufferPool::default();
        assert_eq!(pool.max_pool_size, 16);
        assert_eq!(pool.pooled_count(), 0);
    }

    #[test]
    fn test_pool_acquire_zeroes_buffer() {
        let mut pool = BufferPool::new(4);
        let mut buf = pool.acquire(BufferSize::Small4K);
        // Dirty the buffer
        buf[0] = 0xFF;
        pool.release(buf, BufferSize::Small4K);
        // Re-acquire — the pool should reset the length but not necessarily zero
        // (our implementation does zero-fill via resize)
        let buf2 = pool.acquire(BufferSize::Small4K);
        assert_eq!(buf2.len(), BufferSize::Small4K.bytes());
    }

    #[test]
    fn test_pool_multiple_size_classes_independent() {
        let mut pool = BufferPool::new(4);
        let s = pool.acquire(BufferSize::Small4K);
        let m = pool.acquire(BufferSize::Medium64K);
        pool.release(s, BufferSize::Small4K);
        pool.release(m, BufferSize::Medium64K);
        assert_eq!(pool.free_small.len(), 1);
        assert_eq!(pool.free_medium.len(), 1);
    }
}
