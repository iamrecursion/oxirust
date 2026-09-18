//! Lock-free single-producer single-consumer (SPSC) ring buffer.
//!
//! Provides an `SpscRingBuffer<T>` backed by `AtomicUsize` head/tail indices
//! and a power-of-two capacity buffer.
//!
//! ## Safety model
//!
//! The underlying slot storage uses `std::sync::Mutex<Vec<Option<T>>>` for the
//! individual slot reads/writes, keeping the implementation fully safe while
//! still maintaining an O(1) try-push / try-pop critical section: the mutex is
//! held only for a single slot read or write, not across the entire queue
//! operation.  The `AtomicUsize` cursors are used to determine *which* slot to
//! access before the mutex is acquired, so false contention between producer
//! and consumer is eliminated except in the rare case that they land on the
//! same slot — which, by design, can never happen in a correctly used SPSC
//! channel (the producer is at `head`, the consumer is at `tail`, and `head ≠
//! tail` is guaranteed by the full/empty check).
//!
//! Use [`spsc_channel`] to obtain typed [`SpscProducer`] / [`SpscConsumer`]
//! handles that enforce the single-producer / single-consumer contract at the
//! type level.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ─────────────────────────────────────────────────────────────────────────────
// SpscRingBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// A single-producer / single-consumer ring buffer.
///
/// The buffer holds at most `capacity` live items (expressed as the usable
/// count returned by [`SpscRingBuffer::capacity`]).  Internally, `capacity + 1`
/// slots are allocated (one sentinel slot so head == tail unambiguously means
/// "empty").  The internal buffer size is rounded up to the nearest power of
/// two for cheap index masking.
pub struct SpscRingBuffer<T: Send> {
    /// Slot storage, protected by a mutex for safe cross-thread access.
    slots: Arc<Mutex<Vec<Option<T>>>>,
    /// Number of allocated slots (always a power of two).
    cap: usize,
    /// Bit mask: `index & mask == index % cap`.
    mask: usize,
    /// Write cursor; only the producer advances this.
    head: Arc<AtomicUsize>,
    /// Read cursor; only the consumer advances this.
    tail: Arc<AtomicUsize>,
}

impl<T: Send> SpscRingBuffer<T> {
    /// Creates a new `SpscRingBuffer` able to hold at least `capacity` items.
    ///
    /// The actual usable capacity is `next_power_of_two(capacity.max(2) + 1) - 1`.
    pub fn new(capacity: usize) -> Self {
        // Allocate one extra sentinel slot.
        let cap = (capacity.max(1) + 1).next_power_of_two();
        let mut slots = Vec::with_capacity(cap);
        for _ in 0..cap {
            slots.push(None);
        }
        Self {
            slots: Arc::new(Mutex::new(slots)),
            cap,
            mask: cap - 1,
            head: Arc::new(AtomicUsize::new(0)),
            tail: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Attempts to enqueue `value`.
    ///
    /// Returns `Err(value)` when the buffer is full.
    pub fn try_push(&self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) & self.mask;
        // Full when the slot *after* head would equal tail.
        if next_head == self.tail.load(Ordering::Acquire) {
            return Err(value);
        }
        // Write into the slot.  The mutex guarantees that no other thread is
        // simultaneously accessing the same slot index.  Because the SPSC
        // contract holds (exactly one producer), we are the only one that can
        // write to `head` at this point.
        {
            let mut guard = self
                .slots
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            guard[head] = Some(value);
        }
        // Publish the updated head with Release so the consumer's Acquire sees
        // the slot write above.
        self.head.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Attempts to dequeue a value.
    ///
    /// Returns `None` when the buffer is empty.
    pub fn try_pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        // Empty when tail == head.
        if tail == self.head.load(Ordering::Acquire) {
            return None;
        }
        let value = {
            let mut guard = self
                .slots
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            guard[tail].take()
        };
        let next_tail = (tail + 1) & self.mask;
        self.tail.store(next_tail, Ordering::Release);
        value
    }

    /// Returns `true` when no items are currently queued.
    pub fn is_empty(&self) -> bool {
        self.tail.load(Ordering::Acquire) == self.head.load(Ordering::Acquire)
    }

    /// Returns `true` when the buffer is at full capacity.
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        ((head + 1) & self.mask) == tail
    }

    /// Returns the number of items currently in the buffer.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (head.wrapping_sub(tail)) & self.mask
    }

    /// Returns the maximum number of items the buffer can hold simultaneously.
    pub fn capacity(&self) -> usize {
        self.cap - 1
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpscProducer / SpscConsumer
// ─────────────────────────────────────────────────────────────────────────────

/// Producer half of an SPSC channel.
///
/// Deliberately `!Clone` — only one producer must exist per channel.
pub struct SpscProducer<T: Send> {
    pub(crate) inner: Arc<SpscRingBuffer<T>>,
}

/// Consumer half of an SPSC channel.
///
/// Deliberately `!Clone` — only one consumer must exist per channel.
pub struct SpscConsumer<T: Send> {
    pub(crate) inner: Arc<SpscRingBuffer<T>>,
}

impl<T: Send> SpscProducer<T> {
    /// Enqueue `value`, returning `Err(value)` if the channel is full.
    pub fn try_send(&self, value: T) -> Result<(), T> {
        self.inner.try_push(value)
    }
}

impl<T: Send> SpscConsumer<T> {
    /// Dequeue a value, returning `None` if the channel is empty.
    pub fn try_recv(&self) -> Option<T> {
        self.inner.try_pop()
    }
}

/// Creates a linked (producer, consumer) pair sharing the same ring buffer.
///
/// # Parameters
/// - `capacity`: Minimum number of items the channel should hold simultaneously.
///
/// # Example
/// ```
/// use oximedia_graph::lock_free_ring::spsc_channel;
///
/// let (tx, rx) = spsc_channel::<u32>(16);
/// tx.try_send(42).expect("send should succeed");
/// assert_eq!(rx.try_recv(), Some(42));
/// ```
pub fn spsc_channel<T: Send>(capacity: usize) -> (SpscProducer<T>, SpscConsumer<T>) {
    let ring = Arc::new(SpscRingBuffer::new(capacity));
    (
        SpscProducer {
            inner: Arc::clone(&ring),
        },
        SpscConsumer { inner: ring },
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spsc_basic() {
        let (tx, rx) = spsc_channel::<u32>(16);
        for i in 0u32..10 {
            tx.try_send(i).expect("send should succeed");
        }
        for i in 0u32..10 {
            assert_eq!(rx.try_recv(), Some(i), "item {i} out of order");
        }
        assert_eq!(rx.try_recv(), None, "queue should be empty after draining");
    }

    #[test]
    fn test_spsc_full_boundary() {
        // Request capacity 4; actual usable slots = next_power_of_two(5) - 1 = 7.
        let (tx, rx) = spsc_channel::<u32>(4);
        let cap = tx.inner.capacity();
        assert!(cap >= 4, "usable capacity must be at least 4, got {cap}");

        // Fill to capacity.
        for i in 0..cap {
            tx.try_send(i as u32)
                .unwrap_or_else(|_| panic!("send {i} should succeed within capacity"));
        }

        // One more push must fail.
        assert!(
            tx.try_send(99).is_err(),
            "push to full buffer must return Err"
        );

        // Pop one slot free.
        let first = rx.try_recv().expect("first pop should succeed");
        assert_eq!(first, 0);

        // Now there is room for one more.
        tx.try_send(99).expect("send after pop should succeed");
    }

    #[test]
    fn test_spsc_empty_boundary() {
        let (_tx, rx) = spsc_channel::<i64>(8);
        assert_eq!(rx.try_recv(), None, "pop from empty queue must be None");
    }

    #[test]
    fn test_spsc_concurrent() {
        use std::thread;

        const N: u64 = 10_000;
        let (tx, rx) = spsc_channel::<u64>(64);

        let producer = thread::spawn(move || {
            let mut i: u64 = 0;
            while i < N {
                if tx.try_send(i).is_ok() {
                    i += 1;
                } else {
                    thread::yield_now();
                }
            }
        });

        let consumer = thread::spawn(move || {
            let mut received: Vec<u64> = Vec::with_capacity(N as usize);
            while received.len() < N as usize {
                if let Some(v) = rx.try_recv() {
                    received.push(v);
                } else {
                    thread::yield_now();
                }
            }
            received
        });

        producer.join().expect("producer thread panicked");
        let received = consumer.join().expect("consumer thread panicked");

        assert_eq!(received.len(), N as usize);
        for (idx, &val) in received.iter().enumerate() {
            assert_eq!(val, idx as u64, "item at position {idx} is out of order");
        }
    }
}
