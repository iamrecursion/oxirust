//! Lock-free style ring buffer for streaming I/O.
//!
//! Provides a generic circular buffer and a byte-specialised variant
//! with slice-oriented push/pop helpers for use in streaming pipelines.

#![allow(dead_code)]

/// A circular (ring) buffer with fixed capacity.
///
/// Items are stored in a pre-allocated `Vec<Option<T>>`.  `push` fails when
/// the buffer is full; `pop` removes and returns the oldest item.
pub struct RingBuffer<T> {
    data: Vec<Option<T>>,
    head: usize,
    tail: usize,
    capacity: usize,
    len: usize,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        let mut data = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(None);
        }
        Self {
            data,
            head: 0,
            tail: 0,
            capacity,
            len: 0,
        }
    }

    /// Push an item onto the back of the buffer.
    ///
    /// Returns `false` if the buffer is full and the item was not inserted.
    pub fn push(&mut self, item: T) -> bool {
        if self.is_full() {
            return false;
        }
        self.data[self.tail] = Some(item);
        self.tail = (self.tail + 1) % self.capacity;
        self.len += 1;
        true
    }

    /// Remove and return the oldest item from the front of the buffer.
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let item = self.data[self.head].take();
        self.head = (self.head + 1) % self.capacity;
        self.len -= 1;
        item
    }

    /// Peek at the next item without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        self.data[self.head].as_ref()
    }

    /// Number of items currently in the buffer
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains no items
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if the buffer is at capacity
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Maximum number of items the buffer can hold
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Remove all items from the buffer
    pub fn clear(&mut self) {
        for slot in &mut self.data {
            *slot = None;
        }
        self.head = 0;
        self.tail = 0;
        self.len = 0;
    }
}

impl<T: Clone> RingBuffer<T> {
    /// Collect all items into a `Vec` without removing them (front-to-back order)
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.len);
        let mut idx = self.head;
        for _ in 0..self.len {
            if let Some(ref item) = self.data[idx] {
                result.push(item.clone());
            }
            idx = (idx + 1) % self.capacity;
        }
        result
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Byte-specialised ring buffer
// ──────────────────────────────────────────────────────────────────────────────

/// A ring buffer specialised for `u8` data with slice-oriented helpers.
pub struct ByteRingBuffer {
    inner: RingBuffer<u8>,
}

impl ByteRingBuffer {
    /// Create a new byte ring buffer with the given capacity in bytes
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: RingBuffer::new(capacity),
        }
    }

    /// Push as many bytes from `data` as will fit.
    ///
    /// Returns the number of bytes actually pushed.
    pub fn push_slice(&mut self, data: &[u8]) -> usize {
        let mut pushed = 0;
        for &byte in data {
            if !self.inner.push(byte) {
                break;
            }
            pushed += 1;
        }
        pushed
    }

    /// Pop exactly `n` bytes, returning `None` if fewer than `n` are available.
    pub fn pop_exact(&mut self, n: usize) -> Option<Vec<u8>> {
        if self.inner.len() < n {
            return None;
        }
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(b) = self.inner.pop() {
                result.push(b);
            }
        }
        Some(result)
    }

    /// Push a single byte; returns `false` if the buffer is full
    pub fn push(&mut self, byte: u8) -> bool {
        self.inner.push(byte)
    }

    /// Pop a single byte
    pub fn pop(&mut self) -> Option<u8> {
        self.inner.pop()
    }

    /// Peek at the next byte without removing it
    #[must_use]
    pub fn peek(&self) -> Option<&u8> {
        self.inner.peek()
    }

    /// Number of bytes currently stored
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if no bytes are stored
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns `true` if the buffer is at capacity
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    /// Maximum byte capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Wait-free SPSC Ring Buffer (safe implementation using AtomicU8)
// ──────────────────────────────────────────────────────────────────────────────

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

/// Shared state for the SPSC ring buffer, using `AtomicU8` for data to avoid
/// `UnsafeCell` and satisfy `unsafe_code = "deny"`.
struct SpscInner {
    /// Backing storage using atomic bytes; length is `capacity + 1`.
    data: Vec<AtomicU8>,
    /// One more than the usable capacity (the allocated slot count).
    slot_count: usize,
    /// Write index (owned by the producer, read by the consumer).
    head: AtomicUsize,
    /// Read index (owned by the consumer, read by the producer).
    tail: AtomicUsize,
}

/// The producer half of a wait-free SPSC ring buffer.
///
/// Only one thread should hold this handle. Pushing bytes never blocks;
/// it returns the number of bytes successfully enqueued.
pub struct SpscProducer {
    inner: Arc<SpscInner>,
}

/// The consumer half of a wait-free SPSC ring buffer.
///
/// Only one thread should hold this handle. Popping bytes never blocks;
/// it returns the number of bytes successfully dequeued.
pub struct SpscConsumer {
    inner: Arc<SpscInner>,
}

/// Create a new SPSC ring buffer pair with the given byte capacity.
///
/// Returns `(producer, consumer)`.
///
/// # Errors
///
/// Returns `Err` if `capacity` is zero.
pub fn spsc_ring_buffer(capacity: usize) -> Result<(SpscProducer, SpscConsumer), &'static str> {
    if capacity == 0 {
        return Err("SPSC ring buffer capacity must be > 0");
    }
    let slot_count = capacity + 1; // one extra slot to distinguish full from empty
    let mut data = Vec::with_capacity(slot_count);
    for _ in 0..slot_count {
        data.push(AtomicU8::new(0));
    }
    let inner = Arc::new(SpscInner {
        data,
        slot_count,
        head: AtomicUsize::new(0),
        tail: AtomicUsize::new(0),
    });
    Ok((
        SpscProducer {
            inner: Arc::clone(&inner),
        },
        SpscConsumer { inner },
    ))
}

impl SpscProducer {
    /// Push bytes into the ring buffer without blocking.
    ///
    /// Returns the number of bytes successfully enqueued (may be less than
    /// `data.len()` if the buffer is full).
    pub fn push(&self, data: &[u8]) -> usize {
        let head = self.inner.head.load(Ordering::Relaxed);
        let tail = self.inner.tail.load(Ordering::Acquire);

        let available = if head >= tail {
            self.inner.slot_count - 1 - (head - tail)
        } else {
            tail - head - 1
        };

        let to_write = data.len().min(available);
        for i in 0..to_write {
            let idx = (head + i) % self.inner.slot_count;
            self.inner.data[idx].store(data[i], Ordering::Relaxed);
        }
        // Release so consumer sees the written data
        self.inner
            .head
            .store((head + to_write) % self.inner.slot_count, Ordering::Release);
        to_write
    }

    /// Returns the number of bytes currently stored in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        let head = self.inner.head.load(Ordering::Relaxed);
        let tail = self.inner.tail.load(Ordering::Acquire);
        if head >= tail {
            head - tail
        } else {
            self.inner.slot_count - (tail - head)
        }
    }

    /// Returns `true` if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the usable capacity of the buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.slot_count - 1
    }
}

impl SpscConsumer {
    /// Pop up to `buf.len()` bytes from the ring buffer without blocking.
    ///
    /// Returns the number of bytes actually read into `buf`.
    pub fn pop(&self, buf: &mut [u8]) -> usize {
        let tail = self.inner.tail.load(Ordering::Relaxed);
        let head = self.inner.head.load(Ordering::Acquire);

        let available = if head >= tail {
            head - tail
        } else {
            self.inner.slot_count - (tail - head)
        };

        let to_read = buf.len().min(available);
        for i in 0..to_read {
            let idx = (tail + i) % self.inner.slot_count;
            buf[i] = self.inner.data[idx].load(Ordering::Relaxed);
        }
        // Release so producer sees the freed slots
        self.inner
            .tail
            .store((tail + to_read) % self.inner.slot_count, Ordering::Release);
        to_read
    }

    /// Pop exactly `count` bytes, returning `None` if fewer are available.
    pub fn pop_exact(&self, count: usize) -> Option<Vec<u8>> {
        let tail = self.inner.tail.load(Ordering::Relaxed);
        let head = self.inner.head.load(Ordering::Acquire);
        let available = if head >= tail {
            head - tail
        } else {
            self.inner.slot_count - (tail - head)
        };
        if available < count {
            return None;
        }
        let mut result = vec![0u8; count];
        let read = self.pop(&mut result);
        debug_assert_eq!(read, count);
        Some(result)
    }

    /// Returns the number of bytes currently available to read.
    #[must_use]
    pub fn len(&self) -> usize {
        let tail = self.inner.tail.load(Ordering::Relaxed);
        let head = self.inner.head.load(Ordering::Acquire);
        if head >= tail {
            head - tail
        } else {
            self.inner.slot_count - (tail - head)
        }
    }

    /// Returns `true` if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the usable capacity of the buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.slot_count - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── RingBuffer<u32> ───────────────────────────────────────────────────────

    #[test]
    fn test_ring_new_empty() {
        let rb: RingBuffer<u32> = RingBuffer::new(4);
        assert!(rb.is_empty());
        assert!(!rb.is_full());
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.capacity(), 4);
    }

    #[test]
    fn test_ring_push_and_pop_fifo() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(4);
        assert!(rb.push(1));
        assert!(rb.push(2));
        assert!(rb.push(3));
        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), None);
    }

    #[test]
    fn test_ring_full_returns_false() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(2);
        assert!(rb.push(10));
        assert!(rb.push(20));
        assert!(rb.is_full());
        assert!(!rb.push(30)); // must fail
    }

    #[test]
    fn test_ring_peek_does_not_remove() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(4);
        rb.push(42);
        assert_eq!(rb.peek(), Some(&42));
        assert_eq!(rb.len(), 1);
        assert_eq!(rb.pop(), Some(42));
    }

    #[test]
    fn test_ring_wrap_around() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.push(3);
        rb.pop(); // remove 1
        rb.push(4); // wrap
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), Some(4));
    }

    #[test]
    fn test_ring_clear() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(4);
        rb.push(1);
        rb.push(2);
        rb.clear();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn test_ring_to_vec() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(4);
        rb.push(10);
        rb.push(20);
        rb.push(30);
        assert_eq!(rb.to_vec(), vec![10, 20, 30]);
    }

    // ── ByteRingBuffer ────────────────────────────────────────────────────────

    #[test]
    fn test_byte_ring_push_slice_full() {
        let mut brb = ByteRingBuffer::new(4);
        let pushed = brb.push_slice(&[1, 2, 3, 4, 5]);
        assert_eq!(pushed, 4); // only 4 fit
        assert!(brb.is_full());
    }

    #[test]
    fn test_byte_ring_pop_exact_success() {
        let mut brb = ByteRingBuffer::new(8);
        brb.push_slice(&[10, 20, 30, 40]);
        let out = brb.pop_exact(3).expect("pop_exact should succeed");
        assert_eq!(out, vec![10, 20, 30]);
        assert_eq!(brb.len(), 1);
    }

    #[test]
    fn test_byte_ring_pop_exact_insufficient() {
        let mut brb = ByteRingBuffer::new(8);
        brb.push_slice(&[1, 2]);
        assert!(brb.pop_exact(5).is_none());
        // Data should still be there
        assert_eq!(brb.len(), 2);
    }

    #[test]
    fn test_byte_ring_peek() {
        let mut brb = ByteRingBuffer::new(8);
        brb.push(0xAB);
        assert_eq!(brb.peek(), Some(&0xAB));
        assert_eq!(brb.len(), 1);
    }

    #[test]
    fn test_byte_ring_wrap_around() {
        let mut brb = ByteRingBuffer::new(4);
        brb.push_slice(&[1, 2, 3, 4]);
        brb.pop();
        brb.pop();
        let pushed = brb.push_slice(&[5, 6]);
        assert_eq!(pushed, 2);
        assert_eq!(brb.pop(), Some(3));
        assert_eq!(brb.pop(), Some(4));
        assert_eq!(brb.pop(), Some(5));
        assert_eq!(brb.pop(), Some(6));
    }

    // ── SPSC Ring Buffer ────────────────────────────────────────────────────────

    #[test]
    fn test_spsc_basic_push_pop() {
        let (prod, cons) = spsc_ring_buffer(16).expect("should create");
        let written = prod.push(b"hello");
        assert_eq!(written, 5);
        assert_eq!(prod.len(), 5);
        assert_eq!(cons.len(), 5);

        let mut buf = [0u8; 16];
        let read = cons.pop(&mut buf);
        assert_eq!(read, 5);
        assert_eq!(&buf[..5], b"hello");
        assert!(cons.is_empty());
    }

    #[test]
    fn test_spsc_capacity_enforcement() {
        let (prod, _cons) = spsc_ring_buffer(4).expect("should create");
        let written = prod.push(b"abcdef");
        assert_eq!(written, 4); // only 4 fit
        assert_eq!(prod.len(), 4);
    }

    #[test]
    fn test_spsc_empty_pop() {
        let (_prod, cons) = spsc_ring_buffer(8).expect("should create");
        let mut buf = [0u8; 8];
        let read = cons.pop(&mut buf);
        assert_eq!(read, 0);
        assert!(cons.is_empty());
    }

    #[test]
    fn test_spsc_wrap_around() {
        let (prod, cons) = spsc_ring_buffer(4).expect("should create");

        // Fill buffer
        prod.push(b"abcd");
        // Drain 2 bytes
        let mut buf = [0u8; 2];
        cons.pop(&mut buf);
        assert_eq!(&buf, b"ab");

        // Push 2 more (wraps around)
        let written = prod.push(b"ef");
        assert_eq!(written, 2);

        // Read remaining 4 bytes
        let mut buf2 = [0u8; 4];
        let read = cons.pop(&mut buf2);
        assert_eq!(read, 4);
        assert_eq!(&buf2, b"cdef");
    }

    #[test]
    fn test_spsc_pop_exact() {
        let (prod, cons) = spsc_ring_buffer(16).expect("should create");
        prod.push(b"hello world");

        let result = cons.pop_exact(5);
        assert_eq!(result, Some(b"hello".to_vec()));

        // Not enough for 20 bytes
        assert!(cons.pop_exact(20).is_none());
        // But the remaining 6 bytes are still there
        assert_eq!(cons.len(), 6);
    }

    #[test]
    fn test_spsc_zero_capacity_error() {
        assert!(spsc_ring_buffer(0).is_err());
    }

    #[test]
    fn test_spsc_capacity_accessor() {
        let (prod, cons) = spsc_ring_buffer(32).expect("should create");
        assert_eq!(prod.capacity(), 32);
        assert_eq!(cons.capacity(), 32);
    }

    #[test]
    fn test_spsc_interleaved_operations() {
        let (prod, cons) = spsc_ring_buffer(8).expect("should create");

        for i in 0..100u8 {
            let data = [i];
            let written = prod.push(&data);
            assert_eq!(written, 1);

            let mut buf = [0u8; 1];
            let read = cons.pop(&mut buf);
            assert_eq!(read, 1);
            assert_eq!(buf[0], i);
        }
    }

    #[test]
    fn test_spsc_large_transfer() {
        let (prod, cons) = spsc_ring_buffer(1024).expect("should create");
        let data: Vec<u8> = (0..255).cycle().take(1024).collect();

        let written = prod.push(&data);
        assert_eq!(written, 1024);

        let mut buf = vec![0u8; 1024];
        let read = cons.pop(&mut buf);
        assert_eq!(read, 1024);
        assert_eq!(buf, data);
    }

    #[test]
    fn test_spsc_thread_safety() {
        // Verify that producer and consumer can be sent to different threads
        let (prod, cons) = spsc_ring_buffer(256).expect("should create");

        let producer = std::thread::spawn(move || {
            let mut total = 0usize;
            for i in 0..100u8 {
                total += prod.push(&[i]);
            }
            total
        });

        let consumer = std::thread::spawn(move || {
            let mut total = 0usize;
            let mut buf = [0u8; 1];
            // Spin until we've read 100 bytes
            while total < 100 {
                let n = cons.pop(&mut buf);
                total += n;
                if n == 0 {
                    std::thread::yield_now();
                }
            }
            total
        });

        let prod_total = producer.join().expect("producer panicked");
        let cons_total = consumer.join().expect("consumer panicked");
        assert_eq!(prod_total, 100);
        assert_eq!(cons_total, 100);
    }
}
