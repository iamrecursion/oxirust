//! Double/triple buffering strategy and staging buffer ring for streaming
//! frame processing and efficient CPU-to-GPU transfers.
//!
//! # Buffering strategies
//!
//! * **Double buffering** — one buffer is being processed while the CPU fills
//!   the other; eliminates pipeline stalls at the cost of 2× memory.
//! * **Triple buffering** — adds a third "in-flight" slot so the GPU is never
//!   idle waiting for the CPU and vice-versa.
//!
//! # Staging buffer ring
//!
//! The [`StagingRing`] maintains a circular array of fixed-size CPU-visible
//! staging slots.  Each slot is checked out, filled with data, and released
//! back to the ring after the GPU transfer completes.  This avoids repeated
//! allocation/deallocation of staging memory.

#![allow(dead_code)]

use crate::error::{AccelError, AccelResult};
use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

// ──────────────────────────────────────────────────────────────────────────────
// Buffering strategy
// ──────────────────────────────────────────────────────────────────────────────

/// Number of frames held simultaneously in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferingMode {
    /// Single buffer — no pipelining (simplest, lowest latency for small ops).
    Single,
    /// Two buffers — fill one while the other is in-flight.
    Double,
    /// Three buffers — fills one, submits one, presents one concurrently.
    Triple,
    /// Custom count (≥1).
    Custom(usize),
}

impl BufferingMode {
    /// Return the number of buffer slots this mode requires.
    #[must_use]
    pub fn slot_count(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Double => 2,
            Self::Triple => 3,
            Self::Custom(n) => n.max(1),
        }
    }
}

/// A ring of fixed-size `Vec<u8>` buffers for streaming frame data.
///
/// Mimics the CPU-side management of a VkBuffer ring used for staging.
pub struct FrameBufferRing {
    /// The actual backing storage for each slot.
    slots: Vec<Vec<u8>>,
    /// Which slot is currently being written by the CPU.
    write_cursor: usize,
    /// Which slot is currently being read / consumed.
    read_cursor: usize,
    /// Number of slots that are filled and waiting to be consumed.
    pending: usize,
    /// Capacity of each slot in bytes.
    slot_capacity: usize,
    mode: BufferingMode,
}

impl FrameBufferRing {
    /// Create a new ring with the given buffering mode and per-slot byte capacity.
    #[must_use]
    pub fn new(mode: BufferingMode, slot_capacity: usize) -> Self {
        let count = mode.slot_count();
        let slots = (0..count)
            .map(|_| Vec::with_capacity(slot_capacity))
            .collect();
        Self {
            slots,
            write_cursor: 0,
            read_cursor: 0,
            pending: 0,
            slot_capacity,
            mode,
        }
    }

    /// Number of slots in the ring.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Per-slot capacity in bytes.
    #[must_use]
    pub fn slot_capacity(&self) -> usize {
        self.slot_capacity
    }

    /// Returns `true` if all slots are filled (no write slot available).
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.pending == self.slots.len()
    }

    /// Returns `true` if no slots are pending consumption.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending == 0
    }

    /// Number of frames waiting to be consumed.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending
    }

    /// Write a frame into the next available slot.
    ///
    /// # Errors
    ///
    /// Returns `AccelError::OutOfMemory` if the ring is full.
    /// Returns `AccelError::BufferSizeMismatch` if `data` is larger than
    /// `slot_capacity`.
    pub fn write_frame(&mut self, data: &[u8]) -> AccelResult<usize> {
        if self.is_full() {
            return Err(AccelError::OutOfMemory);
        }
        if data.len() > self.slot_capacity {
            return Err(AccelError::BufferSizeMismatch {
                expected: self.slot_capacity,
                actual: data.len(),
            });
        }
        let idx = self.write_cursor;
        self.slots[idx].clear();
        self.slots[idx].extend_from_slice(data);
        self.write_cursor = (self.write_cursor + 1) % self.slots.len();
        self.pending += 1;
        Ok(idx)
    }

    /// Read the next pending frame (returns a reference to the slot data).
    ///
    /// Returns `None` if no frames are pending.
    pub fn read_frame(&mut self) -> Option<&[u8]> {
        if self.is_empty() {
            return None;
        }
        let idx = self.read_cursor;
        self.read_cursor = (self.read_cursor + 1) % self.slots.len();
        self.pending -= 1;
        Some(&self.slots[idx])
    }

    /// Buffering mode.
    #[must_use]
    pub fn mode(&self) -> BufferingMode {
        self.mode
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Staging buffer ring (for CPU→GPU uploads)
// ──────────────────────────────────────────────────────────────────────────────

/// A single slot in the staging ring.
#[derive(Debug)]
pub struct StagingSlot {
    /// Slot index in the ring.
    pub index: usize,
    /// The staging data (CPU-visible, filled before GPU upload).
    pub data: Vec<u8>,
    /// Maximum capacity of this slot.
    pub capacity: usize,
}

impl StagingSlot {
    /// Returns `true` if the slot has been populated (non-empty).
    #[must_use]
    pub fn is_populated(&self) -> bool {
        !self.data.is_empty()
    }
}

/// Statistics for the staging ring.
#[derive(Debug, Clone, Default)]
pub struct StagingRingStats {
    /// Total number of slots checked out.
    pub checkouts: u64,
    /// Total number of slots returned.
    pub returns: u64,
    /// Maximum simultaneous checkouts observed.
    pub peak_in_flight: usize,
    /// Number of times a checkout had to wait (ring exhausted).
    pub wait_count: u64,
}

struct StagingRingInner {
    free: VecDeque<StagingSlot>,
    in_flight: usize,
    stats: StagingRingStats,
}

/// A thread-safe ring of staging buffer slots.
///
/// Callers check out a slot, fill it, submit the GPU upload, then return the
/// slot when the fence signals completion.
pub struct StagingRing {
    inner: Mutex<StagingRingInner>,
    returned: Condvar,
}

impl StagingRing {
    /// Create a ring with `count` slots of `slot_capacity` bytes each.
    #[must_use]
    pub fn new(count: usize, slot_capacity: usize) -> Self {
        let mut free = VecDeque::with_capacity(count);
        for i in 0..count.max(1) {
            free.push_back(StagingSlot {
                index: i,
                data: Vec::with_capacity(slot_capacity),
                capacity: slot_capacity,
            });
        }
        Self {
            inner: Mutex::new(StagingRingInner {
                free,
                in_flight: 0,
                stats: StagingRingStats::default(),
            }),
            returned: Condvar::new(),
        }
    }

    /// Check out an available staging slot (blocks if none is free).
    ///
    /// # Errors
    ///
    /// Returns `AccelError::Synchronization` if the internal mutex is
    /// poisoned.
    pub fn checkout(&self) -> AccelResult<StagingSlot> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| AccelError::Synchronization(format!("staging ring lock: {e}")))?;

        // Wait until a free slot is available.
        let mut waited = false;
        while guard.free.is_empty() {
            waited = true;
            guard = self
                .returned
                .wait(guard)
                .map_err(|e| AccelError::Synchronization(format!("staging ring cvar: {e}")))?;
        }

        let mut slot = guard.free.pop_front().ok_or(AccelError::OutOfMemory)?; // should not reach here due to wait loop

        slot.data.clear();
        guard.in_flight += 1;
        guard.stats.checkouts += 1;
        if waited {
            guard.stats.wait_count += 1;
        }
        if guard.in_flight > guard.stats.peak_in_flight {
            guard.stats.peak_in_flight = guard.in_flight;
        }

        Ok(slot)
    }

    /// Return a slot back to the ring after GPU transfer completion.
    ///
    /// # Errors
    ///
    /// Returns `AccelError::Synchronization` if the internal mutex is
    /// poisoned.
    pub fn return_slot(&self, mut slot: StagingSlot) -> AccelResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| AccelError::Synchronization(format!("staging ring lock: {e}")))?;
        slot.data.clear();
        guard.in_flight = guard.in_flight.saturating_sub(1);
        guard.stats.returns += 1;
        guard.free.push_back(slot);
        self.returned.notify_one();
        Ok(())
    }

    /// Number of free (available) slots.
    ///
    /// # Errors
    ///
    /// Returns `AccelError::Synchronization` if the mutex is poisoned.
    pub fn free_count(&self) -> AccelResult<usize> {
        self.inner
            .lock()
            .map(|g| g.free.len())
            .map_err(|e| AccelError::Synchronization(format!("staging ring lock: {e}")))
    }

    /// Number of slots currently checked out.
    ///
    /// # Errors
    ///
    /// Returns `AccelError::Synchronization` if the mutex is poisoned.
    pub fn in_flight_count(&self) -> AccelResult<usize> {
        self.inner
            .lock()
            .map(|g| g.in_flight)
            .map_err(|e| AccelError::Synchronization(format!("staging ring lock: {e}")))
    }

    /// Returns a snapshot of ring statistics.
    ///
    /// # Errors
    ///
    /// Returns `AccelError::Synchronization` if the mutex is poisoned.
    pub fn stats(&self) -> AccelResult<StagingRingStats> {
        self.inner
            .lock()
            .map(|g| g.stats.clone())
            .map_err(|e| AccelError::Synchronization(format!("staging ring lock: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FrameBufferRing tests ──────────────────────────────────────────────────

    #[test]
    fn test_frame_ring_double_buffering_basic() {
        let ring = FrameBufferRing::new(BufferingMode::Double, 1024);
        assert_eq!(ring.slot_count(), 2);
        assert!(!ring.is_full());
        assert!(ring.is_empty());
    }

    #[test]
    fn test_frame_ring_triple_buffering() {
        let ring = FrameBufferRing::new(BufferingMode::Triple, 512);
        assert_eq!(ring.slot_count(), 3);
    }

    #[test]
    fn test_frame_ring_write_and_read() {
        let mut ring = FrameBufferRing::new(BufferingMode::Double, 256);
        ring.write_frame(&[1, 2, 3]).expect("write should succeed");
        ring.write_frame(&[4, 5, 6]).expect("write should succeed");
        assert!(ring.is_full());

        let f1 = ring.read_frame().expect("read frame 1");
        assert_eq!(&f1[..3], &[1, 2, 3]);
        let f2 = ring.read_frame().expect("read frame 2");
        assert_eq!(&f2[..3], &[4, 5, 6]);
        assert!(ring.is_empty());
    }

    #[test]
    fn test_frame_ring_full_returns_error() {
        let mut ring = FrameBufferRing::new(BufferingMode::Single, 64);
        ring.write_frame(&[0u8; 64]).expect("write should succeed");
        let result = ring.write_frame(&[0u8; 1]);
        assert!(matches!(result, Err(AccelError::OutOfMemory)));
    }

    #[test]
    fn test_frame_ring_oversized_data() {
        let mut ring = FrameBufferRing::new(BufferingMode::Double, 8);
        let result = ring.write_frame(&[0u8; 16]);
        assert!(matches!(result, Err(AccelError::BufferSizeMismatch { .. })));
    }

    #[test]
    fn test_frame_ring_read_empty_returns_none() {
        let mut ring = FrameBufferRing::new(BufferingMode::Triple, 64);
        assert!(ring.read_frame().is_none());
    }

    #[test]
    fn test_frame_ring_pending_count() {
        let mut ring = FrameBufferRing::new(BufferingMode::Triple, 64);
        ring.write_frame(b"a").expect("write a");
        ring.write_frame(b"b").expect("write b");
        assert_eq!(ring.pending_count(), 2);
    }

    #[test]
    fn test_buffering_mode_slot_counts() {
        assert_eq!(BufferingMode::Single.slot_count(), 1);
        assert_eq!(BufferingMode::Double.slot_count(), 2);
        assert_eq!(BufferingMode::Triple.slot_count(), 3);
        assert_eq!(BufferingMode::Custom(5).slot_count(), 5);
        assert_eq!(BufferingMode::Custom(0).slot_count(), 1); // clamped to 1
    }

    // ── StagingRing tests ─────────────────────────────────────────────────────

    #[test]
    fn test_staging_ring_checkout_and_return() {
        let ring = StagingRing::new(3, 4096);
        assert_eq!(ring.free_count().expect("free_count"), 3);

        let mut slot = ring.checkout().expect("checkout");
        slot.data.extend_from_slice(&[0xAB; 128]);
        assert!(slot.is_populated());

        assert_eq!(ring.in_flight_count().expect("in_flight"), 1);

        ring.return_slot(slot).expect("return_slot");
        assert_eq!(ring.free_count().expect("free_count after return"), 3);
        assert_eq!(ring.in_flight_count().expect("in_flight after return"), 0);
    }

    #[test]
    fn test_staging_ring_stats() {
        let ring = StagingRing::new(2, 512);
        let s1 = ring.checkout().expect("checkout s1");
        let s2 = ring.checkout().expect("checkout s2");
        ring.return_slot(s1).expect("return s1");
        ring.return_slot(s2).expect("return s2");

        let stats = ring.stats().expect("stats");
        assert_eq!(stats.checkouts, 2);
        assert_eq!(stats.returns, 2);
        assert!(stats.peak_in_flight >= 1);
    }

    #[test]
    fn test_staging_ring_slot_cleared_on_return() {
        let ring = StagingRing::new(1, 1024);
        let mut slot = ring.checkout().expect("checkout");
        slot.data.push(0xFF);
        ring.return_slot(slot).expect("return");
        let slot2 = ring.checkout().expect("checkout again");
        assert!(
            slot2.data.is_empty(),
            "slot data should be cleared on return"
        );
        ring.return_slot(slot2).expect("return again");
    }

    #[test]
    fn test_staging_ring_slot_capacity() {
        let ring = StagingRing::new(2, 8192);
        let slot = ring.checkout().expect("checkout");
        assert_eq!(slot.capacity, 8192);
        ring.return_slot(slot).expect("return");
    }

    #[test]
    fn test_staging_ring_concurrent_checkout() {
        use std::sync::Arc;
        use std::thread;

        let ring = Arc::new(StagingRing::new(4, 256));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let r = Arc::clone(&ring);
            handles.push(thread::spawn(move || {
                let slot = r.checkout().expect("concurrent checkout");
                // Simulate some work.
                std::thread::sleep(std::time::Duration::from_millis(2));
                r.return_slot(slot).expect("concurrent return");
            }));
        }

        for h in handles {
            h.join().expect("thread join");
        }

        assert_eq!(ring.free_count().expect("free_count"), 4);
        assert_eq!(ring.in_flight_count().expect("in_flight"), 0);
    }
}
