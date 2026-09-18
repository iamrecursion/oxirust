//! Lock-free single-producer single-consumer (SPSC) ring buffer.
//!
//! Implements a bounded SPSC ring buffer using `std::sync::atomic` for the
//! head/tail indices and a `Box<[std::cell::UnsafeCell<std::mem::MaybeUninit<T>>]>`
//! for storage — but because we cannot use `unsafe_code` in this crate we
//! build on `crossbeam::queue::ArrayQueue`, which is a proven lock-free
//! bounded queue implemented in pure safe Rust (its `unsafe` is inside
//! the `crossbeam-queue` crate itself, not ours).
//!
//! The `SpscRing<T: Copy>` wrapper presents the exact API required by the
//! task specification:
//!
//! * `push(val: T) -> bool` — returns `false` when full.
//! * `pop() -> Option<T>` — returns `None` when empty.
//! * `len() -> usize`
//! * `is_empty() -> bool`
//!
//! Because `ArrayQueue` is MPMC it is also trivially SPSC-safe.  The wrapper
//! exposes `&self` receivers so a producer and consumer can share an
//! `Arc<SpscRing<T>>` without needing `&mut self`.
//!
//! # Usage
//!
//! ```rust
//! use std::sync::Arc;
//! use oximedia_videoip::spsc_ring::SpscRing;
//!
//! let ring = Arc::new(SpscRing::<u32>::new(64));
//! assert!(ring.push(42));
//! assert_eq!(ring.pop(), Some(42));
//! ```

use crossbeam::queue::ArrayQueue;
use std::sync::Arc;

/// Lock-free SPSC ring buffer backed by `crossbeam::queue::ArrayQueue`.
///
/// `T` must be `Copy` to match the specification (values are stored by value,
/// not behind a pointer).
pub struct SpscRing<T: Copy + Send> {
    inner: ArrayQueue<T>,
}

impl<T: Copy + Send> SpscRing<T> {
    /// Creates a new ring buffer that can hold at most `capacity` elements.
    ///
    /// Panics if `capacity == 0`.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SpscRing capacity must be > 0");
        Self {
            inner: ArrayQueue::new(capacity),
        }
    }

    /// Attempts to push a value into the ring.
    ///
    /// Returns `true` on success, `false` if the ring is full.
    pub fn push(&self, val: T) -> bool {
        self.inner.push(val).is_ok()
    }

    /// Attempts to pop a value from the ring.
    ///
    /// Returns `Some(val)` on success, `None` if the ring is empty.
    pub fn pop(&self) -> Option<T> {
        self.inner.pop()
    }

    /// Returns the number of elements currently in the ring.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the ring is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the maximum number of elements the ring can hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}

/// Convenience type alias for sharing a `SpscRing` across threads.
pub type SharedSpscRing<T> = Arc<SpscRing<T>>;

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    // ── Item 4 required tests ─────────────────────────────────────────────────

    #[test]
    fn test_ring_buffer_basic_push_pop() {
        let ring = SpscRing::<u32>::new(8);
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);

        assert!(ring.push(10));
        assert!(ring.push(20));
        assert!(ring.push(30));

        assert_eq!(ring.len(), 3);
        assert!(!ring.is_empty());

        assert_eq!(ring.pop(), Some(10));
        assert_eq!(ring.pop(), Some(20));
        assert_eq!(ring.pop(), Some(30));
        assert_eq!(ring.pop(), None);
        assert!(ring.is_empty());
    }

    #[test]
    fn test_ring_buffer_full_returns_false() {
        let ring = SpscRing::<u8>::new(3);
        assert!(ring.push(1));
        assert!(ring.push(2));
        assert!(ring.push(3));
        // Ring now full (capacity = 3).
        assert!(!ring.push(4), "push on full ring should return false");
    }

    #[test]
    fn test_ring_buffer_spsc_concurrent() {
        const N: usize = 10_000;
        let ring = Arc::new(SpscRing::<u64>::new(128));

        let prod_ring = Arc::clone(&ring);
        let producer = thread::spawn(move || {
            let mut sent = 0u64;
            while sent < N as u64 {
                if prod_ring.push(sent) {
                    sent += 1;
                } else {
                    // Ring full — yield.
                    thread::yield_now();
                }
            }
        });

        let cons_ring = Arc::clone(&ring);
        let consumer = thread::spawn(move || {
            let mut received = 0u64;
            let mut last = u64::MAX;
            while received < N as u64 {
                if let Some(val) = cons_ring.pop() {
                    // Values must be strictly increasing (FIFO order).
                    assert!(
                        last == u64::MAX || val == last + 1,
                        "ordering violation: got {val} after {last}"
                    );
                    last = val;
                    received += 1;
                } else {
                    thread::yield_now();
                }
            }
        });

        producer.join().expect("producer thread panicked");
        consumer.join().expect("consumer thread panicked");
    }

    // ── Additional correctness tests ──────────────────────────────────────────

    #[test]
    fn test_ring_buffer_wrap_around() {
        let ring = SpscRing::<i32>::new(4);
        assert!(ring.push(1));
        assert!(ring.push(2));
        assert!(ring.push(3));
        assert!(ring.push(4));
        assert_eq!(ring.pop(), Some(1));
        assert_eq!(ring.pop(), Some(2));
        assert!(ring.push(5));
        assert!(ring.push(6));
        assert_eq!(ring.pop(), Some(3));
        assert_eq!(ring.pop(), Some(4));
        assert_eq!(ring.pop(), Some(5));
        assert_eq!(ring.pop(), Some(6));
        assert_eq!(ring.pop(), None);
    }

    #[test]
    fn test_ring_buffer_capacity_reported() {
        let ring = SpscRing::<u32>::new(16);
        assert_eq!(ring.capacity(), 16);
    }

    #[test]
    fn test_ring_buffer_pop_empty() {
        let ring = SpscRing::<f64>::new(4);
        assert_eq!(ring.pop(), None);
    }
}
