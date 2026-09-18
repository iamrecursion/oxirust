//! Frame buffer pool for zero-copy operations.
//!
//! This module provides an ID-based buffer pool (`BufferPool`) for efficient
//! frame buffer reuse with explicit ownership tracking. Features include:
//!
//! - Memory pressure level monitoring (Low / Medium / High / Critical)
//! - Pressure callbacks fired on level transitions
//! - Automatic pool shrinking to reclaim idle memory
//!
//! # Example
//!
//! ```
//! use oximedia_core::buffer_pool::{BufferPool, PressureThresholds, MemoryPressureLevel};
//!
//! let mut pool = BufferPool::with_pressure(4, 1024, PressureThresholds::default());
//! pool.add_pressure_callback(Box::new(|level| {
//!     // react to memory pressure level changes
//!     let _ = level;
//! }));
//! // Acquire 3/4 buffers (75%) → High pressure level
//! let id0 = pool.acquire().expect("buffer available");
//! let id1 = pool.acquire().expect("buffer available");
//! let id2 = pool.acquire().expect("buffer available");
//! assert_eq!(pool.current_pressure_level(), MemoryPressureLevel::High);
//! pool.release(id0);
//! pool.release(id1);
//! pool.release(id2);
//! assert_eq!(pool.current_pressure_level(), MemoryPressureLevel::Low);
//! ```

#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────────────────────
// MemoryPressureLevel
// ─────────────────────────────────────────────────────────────────────────────

/// Severity level of memory pressure in a buffer pool.
///
/// Level is determined by the fraction of buffers that are currently in use:
///
/// | Level    | In-use fraction |
/// |----------|----------------|
/// | Low      | < medium watermark |
/// | Medium   | medium ≤ f < high  |
/// | High     | high ≤ f < critical |
/// | Critical | ≥ critical watermark |
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryPressureLevel {
    /// Less than the medium watermark is in use — pool is healthy.
    Low,
    /// Between medium and high watermarks.
    Medium,
    /// Between high and critical watermarks.
    High,
    /// At or above the critical watermark — pool nearly exhausted.
    Critical,
}

impl Default for MemoryPressureLevel {
    fn default() -> Self {
        Self::Low
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PressureThresholds
// ─────────────────────────────────────────────────────────────────────────────

/// Fractional watermarks that define memory pressure level transitions.
///
/// All values are in the range `[0.0, 1.0]` and represent the fraction of
/// pool buffers that must be in-use to reach that level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureThresholds {
    /// In-use fraction at or above which the level becomes `Medium`. Default 0.5.
    pub medium_watermark: f64,
    /// In-use fraction at or above which the level becomes `High`. Default 0.75.
    pub high_watermark: f64,
    /// In-use fraction at or above which the level becomes `Critical`. Default 0.9.
    pub critical_watermark: f64,
}

impl Default for PressureThresholds {
    fn default() -> Self {
        Self {
            medium_watermark: 0.5,
            high_watermark: 0.75,
            critical_watermark: 0.9,
        }
    }
}

impl PressureThresholds {
    /// Creates custom thresholds.  All values must be in `[0.0, 1.0]` and
    /// `medium ≤ high ≤ critical`.
    ///
    /// # Panics
    ///
    /// Panics if values are out of order or outside `[0.0, 1.0]`.
    #[must_use]
    pub fn new(medium: f64, high: f64, critical: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&medium)
                && (0.0..=1.0).contains(&high)
                && (0.0..=1.0).contains(&critical),
            "Watermarks must be in [0.0, 1.0]"
        );
        assert!(
            medium <= high && high <= critical,
            "Watermarks must be in ascending order"
        );
        Self {
            medium_watermark: medium,
            high_watermark: high,
            critical_watermark: critical,
        }
    }

    /// Maps an in-use fraction to the corresponding [`MemoryPressureLevel`].
    #[must_use]
    pub fn level_for_fraction(&self, in_use_fraction: f64) -> MemoryPressureLevel {
        if in_use_fraction >= self.critical_watermark {
            MemoryPressureLevel::Critical
        } else if in_use_fraction >= self.high_watermark {
            MemoryPressureLevel::High
        } else if in_use_fraction >= self.medium_watermark {
            MemoryPressureLevel::Medium
        } else {
            MemoryPressureLevel::Low
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Callback type alias
// ─────────────────────────────────────────────────────────────────────────────

/// A callable invoked whenever the pressure level changes.
///
/// Receives the new [`MemoryPressureLevel`] as its argument.
pub type MemoryPressureCallback = Box<dyn Fn(MemoryPressureLevel) + Send + Sync>;

// ─────────────────────────────────────────────────────────────────────────────
// BufferDesc / PooledBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// Descriptor for a pooled buffer slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferDesc {
    /// Size of the buffer in bytes.
    pub size_bytes: usize,
    /// Required memory alignment in bytes.
    pub alignment: usize,
    /// Pool identifier this descriptor belongs to.
    pub pool_id: u32,
}

impl BufferDesc {
    /// Creates a new `BufferDesc`.
    #[must_use]
    pub fn new(size_bytes: usize, alignment: usize, pool_id: u32) -> Self {
        Self {
            size_bytes,
            alignment,
            pool_id,
        }
    }

    /// Returns `true` if the alignment equals 4096 (one memory page).
    #[must_use]
    pub fn is_page_aligned(&self) -> bool {
        self.alignment == 4096
    }

    /// Returns how many slots of `slot_size` bytes are needed to hold this buffer.
    ///
    /// # Panics
    ///
    /// Panics if `slot_size` is zero.
    #[must_use]
    pub fn slots_needed(&self, slot_size: usize) -> usize {
        assert!(slot_size > 0, "slot_size must be non-zero");
        self.size_bytes.div_ceil(slot_size)
    }
}

/// A buffer managed by the pool with an associated unique ID.
#[derive(Debug)]
pub struct PooledBuffer {
    /// Unique identifier for this buffer within the pool.
    pub id: u64,
    /// Raw buffer data.
    pub data: Vec<u8>,
    /// Descriptor for this buffer.
    pub desc: BufferDesc,
    /// Whether this buffer is currently in use.
    pub in_use: bool,
}

impl PooledBuffer {
    /// Creates a new `PooledBuffer`.
    #[must_use]
    pub fn new(id: u64, desc: BufferDesc) -> Self {
        let data = vec![0u8; desc.size_bytes];
        Self {
            id,
            data,
            desc,
            in_use: false,
        }
    }

    /// Resets the buffer: zeroes the data and marks it as not in use.
    pub fn reset(&mut self) {
        self.data.fill(0);
        self.in_use = false;
    }

    /// Returns the number of bytes available (equal to the buffer size).
    #[must_use]
    pub fn available_size(&self) -> usize {
        self.data.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BufferPool
// ─────────────────────────────────────────────────────────────────────────────

/// A pool of frame buffers identified by integer IDs.
///
/// Buffers are acquired by ID and released back to the pool by ID.
/// When constructed with [`BufferPool::with_pressure`], the pool monitors
/// its fill ratio and fires registered callbacks on level transitions.
pub struct BufferPool {
    /// Managed buffers.
    pub buffers: Vec<PooledBuffer>,
    /// Counter for assigning unique IDs to new buffers.
    pub next_id: u64,
    /// Pressure thresholds configuration (if any).
    thresholds: Option<PressureThresholds>,
    /// Last known pressure level (used to detect transitions).
    last_pressure: MemoryPressureLevel,
    /// Registered pressure callbacks.
    pressure_callbacks: Vec<MemoryPressureCallback>,
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("total", &self.buffers.len())
            .field("available", &self.available_count())
            .field("last_pressure", &self.last_pressure)
            .finish()
    }
}

impl BufferPool {
    /// Creates a new `BufferPool` with `count` buffers each of `buf_size` bytes.
    ///
    /// All buffers share `pool_id = 0` and default alignment of 64.
    /// No pressure thresholds or callbacks are configured.
    #[must_use]
    pub fn new(count: usize, buf_size: usize) -> Self {
        let mut buffers = Vec::with_capacity(count);
        for id in 0..count as u64 {
            let desc = BufferDesc::new(buf_size, 64, 0);
            buffers.push(PooledBuffer::new(id, desc));
        }
        Self {
            buffers,
            next_id: count as u64,
            thresholds: None,
            last_pressure: MemoryPressureLevel::Low,
            pressure_callbacks: Vec::new(),
        }
    }

    /// Creates a `BufferPool` with memory pressure monitoring enabled.
    ///
    /// See [`PressureThresholds`] for details on watermark configuration.
    #[must_use]
    pub fn with_pressure(count: usize, buf_size: usize, thresholds: PressureThresholds) -> Self {
        let mut pool = Self::new(count, buf_size);
        pool.thresholds = Some(thresholds);
        pool
    }

    /// Registers a callback to be invoked whenever the pressure level transitions.
    ///
    /// The callback receives the new [`MemoryPressureLevel`] as its argument.
    /// Multiple callbacks may be registered and are called in registration order.
    pub fn add_pressure_callback(&mut self, cb: MemoryPressureCallback) {
        self.pressure_callbacks.push(cb);
    }

    // ── Pressure helpers ─────────────────────────────────────────────────────

    /// Computes the current in-use fraction `[0.0, 1.0]`.
    #[must_use]
    fn in_use_fraction(&self) -> f64 {
        let total = self.buffers.len();
        if total == 0 {
            return 0.0;
        }
        let in_use = self.buffers.iter().filter(|b| b.in_use).count();
        in_use as f64 / total as f64
    }

    /// Returns the current memory pressure level.
    ///
    /// If no thresholds are configured, always returns [`MemoryPressureLevel::Low`].
    #[must_use]
    pub fn current_pressure_level(&self) -> MemoryPressureLevel {
        match &self.thresholds {
            None => MemoryPressureLevel::Low,
            Some(t) => t.level_for_fraction(self.in_use_fraction()),
        }
    }

    /// Fires registered callbacks if the pressure level has changed since the
    /// last call.  Updates `last_pressure` to the current level.
    fn notify_pressure(&mut self) {
        let current = self.current_pressure_level();
        if current != self.last_pressure {
            self.last_pressure = current;
            for cb in &self.pressure_callbacks {
                cb(current);
            }
        }
    }

    // ── Core operations ───────────────────────────────────────────────────────

    /// Acquires an available buffer and returns its ID.
    ///
    /// Returns `None` if no buffer is free.
    /// Triggers pressure notification after the acquisition.
    #[must_use]
    pub fn acquire(&mut self) -> Option<u64> {
        let acquired = self.buffers.iter_mut().find(|b| !b.in_use).map(|buf| {
            buf.in_use = true;
            buf.id
        });
        if acquired.is_some() {
            self.notify_pressure();
        }
        acquired
    }

    /// Releases the buffer with the given `id` back to the pool.
    ///
    /// If the ID is not found this is a no-op.
    /// Triggers pressure notification after the release.
    pub fn release(&mut self, id: u64) {
        if let Some(buf) = self.buffers.iter_mut().find(|b| b.id == id) {
            buf.reset();
        }
        self.notify_pressure();
    }

    // ── Pool shrinking ────────────────────────────────────────────────────────

    /// Removes free (not in-use) buffers until the pool has at most `target_count`
    /// total buffers.
    ///
    /// Only idle buffers are removed; buffers currently in use are never evicted.
    /// Returns the number of buffers that were removed.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_core::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::new(8, 64);
    /// let removed = pool.shrink_to(4);
    /// assert_eq!(removed, 4);
    /// assert_eq!(pool.total_count(), 4);
    /// ```
    pub fn shrink_to(&mut self, target_count: usize) -> usize {
        let mut removed = 0usize;
        // Drain from the back to keep ID ordering stable.
        let mut i = self.buffers.len();
        while i > 0 && self.buffers.len() > target_count {
            i -= 1;
            if !self.buffers[i].in_use {
                self.buffers.remove(i);
                removed += 1;
            }
        }
        if removed > 0 {
            self.notify_pressure();
        }
        removed
    }

    /// Automatically shrinks the pool when pressure is `Low` and more than half
    /// of the buffers are idle.
    ///
    /// Shrinks down to half of the current total count, rounding up so at least
    /// one buffer always remains. Returns the number of buffers removed.
    pub fn auto_shrink(&mut self) -> usize {
        let current_level = self.current_pressure_level();
        if current_level != MemoryPressureLevel::Low {
            return 0;
        }
        let total = self.buffers.len();
        let available = self.available_count();
        if total == 0 || available <= total / 2 {
            return 0;
        }
        let target = (total / 2).max(1);
        self.shrink_to(target)
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Returns the number of buffers not currently in use.
    #[must_use]
    pub fn available_count(&self) -> usize {
        self.buffers.iter().filter(|b| !b.in_use).count()
    }

    /// Returns the total number of buffers managed by this pool.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.buffers.len()
    }

    /// Returns the number of buffers currently in use.
    #[must_use]
    pub fn in_use_count(&self) -> usize {
        self.buffers.iter().filter(|b| b.in_use).count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // --- BufferDesc tests ---

    #[test]
    fn test_buffer_desc_new() {
        let desc = BufferDesc::new(1024, 64, 1);
        assert_eq!(desc.size_bytes, 1024);
        assert_eq!(desc.alignment, 64);
        assert_eq!(desc.pool_id, 1);
    }

    #[test]
    fn test_buffer_desc_is_page_aligned_true() {
        let desc = BufferDesc::new(8192, 4096, 0);
        assert!(desc.is_page_aligned());
    }

    #[test]
    fn test_buffer_desc_is_page_aligned_false() {
        let desc = BufferDesc::new(8192, 64, 0);
        assert!(!desc.is_page_aligned());
    }

    #[test]
    fn test_buffer_desc_slots_needed_exact() {
        let desc = BufferDesc::new(1024, 64, 0);
        assert_eq!(desc.slots_needed(512), 2);
    }

    #[test]
    fn test_buffer_desc_slots_needed_round_up() {
        let desc = BufferDesc::new(1025, 64, 0);
        assert_eq!(desc.slots_needed(512), 3);
    }

    #[test]
    fn test_buffer_desc_slots_needed_single_slot() {
        let desc = BufferDesc::new(100, 64, 0);
        assert_eq!(desc.slots_needed(200), 1);
    }

    // --- PooledBuffer tests ---

    #[test]
    fn test_pooled_buffer_initial_state() {
        let desc = BufferDesc::new(256, 64, 0);
        let buf = PooledBuffer::new(42, desc);
        assert_eq!(buf.id, 42);
        assert!(!buf.in_use);
        assert_eq!(buf.available_size(), 256);
        assert!(buf.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_pooled_buffer_reset() {
        let desc = BufferDesc::new(4, 64, 0);
        let mut buf = PooledBuffer::new(1, desc);
        buf.in_use = true;
        buf.data[0] = 0xFF;
        buf.reset();
        assert!(!buf.in_use);
        assert!(buf.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_pooled_buffer_available_size() {
        let desc = BufferDesc::new(512, 64, 0);
        let buf = PooledBuffer::new(0, desc);
        assert_eq!(buf.available_size(), 512);
    }

    // --- BufferPool basic tests ---

    #[test]
    fn test_pool_new() {
        let pool = BufferPool::new(4, 1024);
        assert_eq!(pool.total_count(), 4);
        assert_eq!(pool.available_count(), 4);
    }

    #[test]
    fn test_pool_acquire_returns_id() {
        let mut pool = BufferPool::new(2, 256);
        let id = pool.acquire();
        assert!(id.is_some());
    }

    #[test]
    fn test_pool_acquire_exhausts_buffers() {
        let mut pool = BufferPool::new(2, 256);
        let _id1 = pool.acquire().expect("acquire should succeed");
        let _id2 = pool.acquire().expect("acquire should succeed");
        assert!(pool.acquire().is_none());
    }

    #[test]
    fn test_pool_available_count_decrements_on_acquire() {
        let mut pool = BufferPool::new(3, 64);
        assert_eq!(pool.available_count(), 3);
        let _ = pool.acquire();
        assert_eq!(pool.available_count(), 2);
        let _ = pool.acquire();
        assert_eq!(pool.available_count(), 1);
    }

    #[test]
    fn test_pool_release_makes_buffer_available() {
        let mut pool = BufferPool::new(1, 64);
        let id = pool.acquire().expect("acquire should succeed");
        assert_eq!(pool.available_count(), 0);
        pool.release(id);
        assert_eq!(pool.available_count(), 1);
    }

    #[test]
    fn test_pool_release_unknown_id_is_noop() {
        let mut pool = BufferPool::new(2, 64);
        let before = pool.available_count();
        pool.release(999);
        assert_eq!(pool.available_count(), before);
    }

    #[test]
    fn test_pool_total_count_unchanged_after_ops() {
        let mut pool = BufferPool::new(5, 128);
        let ids: Vec<u64> = (0..5).filter_map(|_| pool.acquire()).collect();
        assert_eq!(pool.total_count(), 5);
        for id in ids {
            pool.release(id);
        }
        assert_eq!(pool.total_count(), 5);
    }

    // --- PressureThresholds tests ---

    #[test]
    fn test_pressure_thresholds_default() {
        let t = PressureThresholds::default();
        assert_eq!(t.level_for_fraction(0.0), MemoryPressureLevel::Low);
        assert_eq!(t.level_for_fraction(0.5), MemoryPressureLevel::Medium);
        assert_eq!(t.level_for_fraction(0.75), MemoryPressureLevel::High);
        assert_eq!(t.level_for_fraction(0.9), MemoryPressureLevel::Critical);
        assert_eq!(t.level_for_fraction(1.0), MemoryPressureLevel::Critical);
    }

    #[test]
    fn test_pressure_thresholds_custom() {
        let t = PressureThresholds::new(0.4, 0.6, 0.8);
        assert_eq!(t.level_for_fraction(0.3), MemoryPressureLevel::Low);
        assert_eq!(t.level_for_fraction(0.5), MemoryPressureLevel::Medium);
        assert_eq!(t.level_for_fraction(0.7), MemoryPressureLevel::High);
        assert_eq!(t.level_for_fraction(0.85), MemoryPressureLevel::Critical);
    }

    #[test]
    #[should_panic(expected = "Watermarks must be in ascending order")]
    fn test_pressure_thresholds_out_of_order_panics() {
        let _ = PressureThresholds::new(0.8, 0.5, 0.9);
    }

    // --- Memory pressure level tests ---

    #[test]
    fn test_pool_initial_pressure_level_low() {
        let pool = BufferPool::with_pressure(4, 64, PressureThresholds::default());
        assert_eq!(pool.current_pressure_level(), MemoryPressureLevel::Low);
    }

    #[test]
    fn test_pool_pressure_level_increases_with_usage() {
        let mut pool = BufferPool::with_pressure(4, 64, PressureThresholds::default());
        // acquire 2 / 4 = 0.5 → Medium
        let _id0 = pool.acquire();
        let _id1 = pool.acquire();
        assert_eq!(pool.current_pressure_level(), MemoryPressureLevel::Medium);
        // acquire 3 / 4 = 0.75 → High
        let _id2 = pool.acquire();
        assert_eq!(pool.current_pressure_level(), MemoryPressureLevel::High);
        // acquire 4 / 4 = 1.0 → Critical
        let _id3 = pool.acquire();
        assert_eq!(pool.current_pressure_level(), MemoryPressureLevel::Critical);
    }

    #[test]
    fn test_pool_pressure_level_decreases_on_release() {
        let mut pool = BufferPool::with_pressure(4, 64, PressureThresholds::default());
        let id0 = pool.acquire().expect("should acquire");
        let id1 = pool.acquire().expect("should acquire");
        let id2 = pool.acquire().expect("should acquire");
        let id3 = pool.acquire().expect("should acquire");
        assert_eq!(pool.current_pressure_level(), MemoryPressureLevel::Critical);
        pool.release(id3);
        assert_eq!(pool.current_pressure_level(), MemoryPressureLevel::High);
        pool.release(id2);
        assert_eq!(pool.current_pressure_level(), MemoryPressureLevel::Medium);
        pool.release(id1);
        pool.release(id0);
        assert_eq!(pool.current_pressure_level(), MemoryPressureLevel::Low);
    }

    // --- Pressure callback tests ---

    #[test]
    fn test_pressure_callback_fired_on_transition() {
        let events: Arc<Mutex<Vec<MemoryPressureLevel>>> = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);

        let mut pool = BufferPool::with_pressure(4, 64, PressureThresholds::default());
        pool.add_pressure_callback(Box::new(move |level| {
            events_clone.lock().expect("lock").push(level);
        }));

        // 2/4 = 0.5 → Medium transition
        let _id0 = pool.acquire();
        let _id1 = pool.acquire();
        // 3/4 = 0.75 → High transition
        let _id2 = pool.acquire();
        // 4/4 = 1.0 → Critical transition
        let _id3 = pool.acquire();

        let recorded = events.lock().expect("lock").clone();
        assert_eq!(
            recorded,
            vec![
                MemoryPressureLevel::Medium,
                MemoryPressureLevel::High,
                MemoryPressureLevel::Critical,
            ]
        );
    }

    #[test]
    fn test_pressure_callback_not_fired_on_same_level() {
        let events: Arc<Mutex<Vec<MemoryPressureLevel>>> = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);

        // 10-buffer pool; thresholds at 0.5/0.75/0.9
        let mut pool = BufferPool::with_pressure(10, 64, PressureThresholds::default());
        pool.add_pressure_callback(Box::new(move |level| {
            events_clone.lock().expect("lock").push(level);
        }));

        // acquire 2 → Low (1/10, 2/10 both < 0.5)
        let _a = pool.acquire(); // 0.1 → Low (no transition from initial Low)
        let _b = pool.acquire(); // 0.2 → Low (no transition)

        let recorded = events.lock().expect("lock").clone();
        // No transitions: started at Low and stayed Low
        assert!(recorded.is_empty());
    }

    // --- Pool shrinking tests ---

    #[test]
    fn test_shrink_to_removes_free_buffers() {
        let mut pool = BufferPool::new(8, 64);
        let removed = pool.shrink_to(4);
        assert_eq!(removed, 4);
        assert_eq!(pool.total_count(), 4);
        assert_eq!(pool.available_count(), 4);
    }

    #[test]
    fn test_shrink_to_does_not_remove_in_use_buffers() {
        let mut pool = BufferPool::new(4, 64);
        let id0 = pool.acquire().expect("should acquire");
        let id1 = pool.acquire().expect("should acquire");
        // 2 in use, 2 free; shrink to 1 — can only remove 2 free ones, leaving 2 (the in-use)
        let removed = pool.shrink_to(1);
        assert_eq!(removed, 2);
        assert_eq!(pool.total_count(), 2);
        assert_eq!(pool.in_use_count(), 2);
        pool.release(id0);
        pool.release(id1);
    }

    #[test]
    fn test_shrink_to_noop_when_already_at_or_below_target() {
        let mut pool = BufferPool::new(4, 64);
        let removed = pool.shrink_to(4);
        assert_eq!(removed, 0);
        assert_eq!(pool.total_count(), 4);

        let removed2 = pool.shrink_to(10);
        assert_eq!(removed2, 0);
        assert_eq!(pool.total_count(), 4);
    }

    #[test]
    fn test_auto_shrink_when_low_pressure() {
        let mut pool = BufferPool::with_pressure(8, 64, PressureThresholds::default());
        // All free → Low pressure; available (8) > half (4) → shrink
        let removed = pool.auto_shrink();
        assert!(removed > 0);
        assert!(pool.total_count() < 8);
    }

    #[test]
    fn test_auto_shrink_does_not_shrink_under_pressure() {
        let mut pool = BufferPool::with_pressure(4, 64, PressureThresholds::default());
        // Acquire 2/4 = Medium pressure
        let _id0 = pool.acquire();
        let _id1 = pool.acquire();
        let removed = pool.auto_shrink();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_in_use_count() {
        let mut pool = BufferPool::new(4, 64);
        assert_eq!(pool.in_use_count(), 0);
        let _ = pool.acquire();
        let _ = pool.acquire();
        assert_eq!(pool.in_use_count(), 2);
    }

    // --- Pool without pressure thresholds always reports Low ---

    #[test]
    fn test_no_thresholds_always_low() {
        let mut pool = BufferPool::new(2, 64);
        let _ = pool.acquire();
        let _ = pool.acquire();
        assert_eq!(pool.current_pressure_level(), MemoryPressureLevel::Low);
    }
}
