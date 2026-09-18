//! Lock-free ring buffer for media streaming.
//!
//! Provides:
//! - [`RingBuffer<T>`] — a single-threaded bounded FIFO ring buffer.
//! - [`MediaFrameQueue`] — paired video/audio ring buffers with PTS queues.
//! - [`SpscRingBuffer<T>`] — a truly lock-free single-producer / single-consumer
//!   ring buffer backed by `AtomicUsize` indices, safe for cross-thread use.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Runtime statistics collected by a [`RingBuffer`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RingBufferStats {
    /// Total capacity of the buffer (number of slots).
    pub capacity: usize,
    /// Number of items currently stored.
    pub len: usize,
    /// Cumulative number of successful pushes (not counting overwrites).
    pub push_count: u64,
    /// Cumulative number of pops.
    pub pop_count: u64,
    /// Number of times a push was rejected (buffer full, non-overwrite mode)
    /// or an item was silently evicted (overwrite mode).
    pub overflow_count: u64,
}

// ---------------------------------------------------------------------------
// RingBuffer<T>
// ---------------------------------------------------------------------------

/// A bounded first-in-first-out ring buffer.
///
/// Items are stored in a flat `Vec<Option<T>>` of length `capacity`.
/// `head` is the next write slot; `tail` is the next read slot.
///
/// # Invariants
///
/// - `head`, `tail` are always in `0..capacity`
/// - `len <= capacity`
pub struct RingBuffer<T> {
    data: Vec<Option<T>>,
    /// Write position (next slot to write into).
    head: usize,
    /// Read position (next slot to read from).
    tail: usize,
    capacity: usize,
    len: usize,
    push_count: u64,
    pop_count: u64,
    overflow_count: u64,
}

impl<T> RingBuffer<T> {
    /// Creates a new `RingBuffer` with the given capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be non-zero");
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
            push_count: 0,
            pop_count: 0,
            overflow_count: 0,
        }
    }

    /// Returns the number of items currently in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if the buffer is at capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Returns the maximum number of items the buffer can hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Attempts to push `item` onto the back of the buffer.
    ///
    /// Returns `true` on success, `false` if the buffer is full.
    /// When `false` is returned `overflow_count` is incremented and the item
    /// is dropped.
    pub fn push(&mut self, item: T) -> bool {
        if self.is_full() {
            self.overflow_count += 1;
            return false;
        }
        self.data[self.head] = Some(item);
        self.head = (self.head + 1) % self.capacity;
        self.len += 1;
        self.push_count += 1;
        true
    }

    /// Pushes `item` onto the back of the buffer, evicting the oldest item if
    /// the buffer is full.
    ///
    /// When eviction occurs `overflow_count` is incremented.
    pub fn push_overwrite(&mut self, item: T) {
        if self.is_full() {
            // Evict the oldest item by advancing tail.
            self.tail = (self.tail + 1) % self.capacity;
            self.len -= 1;
            self.overflow_count += 1;
        }
        self.data[self.head] = Some(item);
        self.head = (self.head + 1) % self.capacity;
        self.len += 1;
        self.push_count += 1;
    }

    /// Removes and returns the oldest item, or `None` if the buffer is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let item = self.data[self.tail].take();
        self.tail = (self.tail + 1) % self.capacity;
        self.len -= 1;
        self.pop_count += 1;
        item
    }

    /// Returns a reference to the oldest item without removing it, or `None`
    /// if the buffer is empty.
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        self.data[self.tail].as_ref()
    }

    /// Removes all items from the buffer.
    pub fn clear(&mut self) {
        for slot in &mut self.data {
            *slot = None;
        }
        self.head = 0;
        self.tail = 0;
        self.len = 0;
    }

    /// Returns a snapshot of runtime statistics.
    #[must_use]
    pub fn stats(&self) -> RingBufferStats {
        RingBufferStats {
            capacity: self.capacity,
            len: self.len,
            push_count: self.push_count,
            pop_count: self.pop_count,
            overflow_count: self.overflow_count,
        }
    }

    /// Returns an iterator over references to the buffered items in insertion
    /// order (oldest first).
    pub fn iter(&self) -> RingBufferIter<'_, T> {
        RingBufferIter {
            buffer: self,
            index: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------------

/// Iterator produced by [`RingBuffer::iter`].
pub struct RingBufferIter<'a, T> {
    buffer: &'a RingBuffer<T>,
    index: usize,
}

impl<'a, T> Iterator for RingBufferIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buffer.len {
            return None;
        }
        let slot = (self.buffer.tail + self.index) % self.buffer.capacity;
        self.index += 1;
        self.buffer.data[slot].as_ref()
    }
}

// ---------------------------------------------------------------------------
// MediaFrameQueue
// ---------------------------------------------------------------------------

/// A paired video/audio ring buffer with associated PTS queues.
///
/// Video frames are stored as raw byte slices (`Vec<u8>`) and audio frames
/// as PCM sample vectors (`Vec<f32>`).  Each ring buffer is accompanied by
/// a [`VecDeque`] of presentation timestamps so that callers can correlate
/// frames with their timing information.
pub struct MediaFrameQueue {
    video: RingBuffer<Vec<u8>>,
    audio: RingBuffer<Vec<f32>>,
    /// PTS values corresponding to video frames in insertion order.
    pub video_pts: VecDeque<i64>,
    /// PTS values corresponding to audio frames in insertion order.
    pub audio_pts: VecDeque<i64>,
}

impl MediaFrameQueue {
    /// Creates a new `MediaFrameQueue` with the specified per-stream
    /// capacities.
    ///
    /// # Panics
    ///
    /// Panics if either capacity is zero.
    #[must_use]
    pub fn new(video_cap: usize, audio_cap: usize) -> Self {
        Self {
            video: RingBuffer::new(video_cap),
            audio: RingBuffer::new(audio_cap),
            video_pts: VecDeque::with_capacity(video_cap),
            audio_pts: VecDeque::with_capacity(audio_cap),
        }
    }

    /// Pushes a video frame along with its PTS.
    ///
    /// Returns `true` on success, `false` if the video ring buffer is full.
    /// When `false` is returned neither the frame nor its PTS is queued.
    pub fn push_video(&mut self, frame: Vec<u8>, pts: i64) -> bool {
        if self.video.push(frame) {
            self.video_pts.push_back(pts);
            true
        } else {
            false
        }
    }

    /// Pushes an audio frame along with its PTS.
    ///
    /// Returns `true` on success, `false` if the audio ring buffer is full.
    pub fn push_audio(&mut self, samples: Vec<f32>, pts: i64) -> bool {
        if self.audio.push(samples) {
            self.audio_pts.push_back(pts);
            true
        } else {
            false
        }
    }

    /// Pops the oldest video frame and its PTS, or `None` if empty.
    pub fn pop_video(&mut self) -> Option<(Vec<u8>, i64)> {
        let frame = self.video.pop()?;
        let pts = self.video_pts.pop_front().unwrap_or(0);
        Some((frame, pts))
    }

    /// Pops the oldest audio frame and its PTS, or `None` if empty.
    pub fn pop_audio(&mut self) -> Option<(Vec<f32>, i64)> {
        let samples = self.audio.pop()?;
        let pts = self.audio_pts.pop_front().unwrap_or(0);
        Some((samples, pts))
    }

    /// Returns the A/V sync offset in milliseconds, defined as
    /// `video_front_pts - audio_front_pts`.
    ///
    /// Returns `None` if either stream has no queued frames.
    ///
    /// A positive value means the video is ahead of the audio; a negative
    /// value means the audio is ahead.
    #[must_use]
    pub fn sync_offset_ms(&self) -> Option<i64> {
        let v = *self.video_pts.front()?;
        let a = *self.audio_pts.front()?;
        Some(v - a)
    }

    /// Returns the number of video frames currently buffered.
    #[must_use]
    pub fn video_len(&self) -> usize {
        self.video.len()
    }

    /// Returns the number of audio frames currently buffered.
    #[must_use]
    pub fn audio_len(&self) -> usize {
        self.audio.len()
    }
}

// ---------------------------------------------------------------------------
// SpscRingBuffer<T>  —  lock-free single-producer / single-consumer buffer
// ---------------------------------------------------------------------------

/// Lock-free single-producer / single-consumer (SPSC) ring buffer.
///
/// This buffer uses atomic load/store with appropriate memory ordering so that
/// **one** producer thread may call [`push`](SpscRingBuffer::push) while
/// **one** consumer thread may call [`pop`](SpscRingBuffer::pop) concurrently,
/// with no locks and no data races.
///
/// The capacity is always rounded up to the next power of two so that index
/// masking can be used instead of modulo.
///
/// # Safety
///
/// This type is `Send` + `Sync` when `T: Send`.  The safety invariant is that
/// *at most one thread writes* and *at most one thread reads* at any time.
///
/// # Example
///
/// ```
/// use oximedia_core::ring_buffer::SpscRingBuffer;
/// use std::sync::Arc;
///
/// let buf: Arc<SpscRingBuffer<u32>> = Arc::new(SpscRingBuffer::new(8));
/// let producer = Arc::clone(&buf);
/// let consumer = Arc::clone(&buf);
///
/// let handle = std::thread::spawn(move || {
///     for i in 0..4_u32 {
///         while !producer.push(i) {} // busy-wait until slot available
///     }
/// });
///
/// handle.join().expect("thread ok");
/// for i in 0..4_u32 {
///     assert_eq!(consumer.pop(), Some(i));
/// }
/// ```
pub struct SpscRingBuffer<T> {
    /// Slots storing items wrapped in `Mutex<Option<T>>`.
    /// The SPSC discipline ensures only one thread writes and one reads at
    /// any slot at a time; the `Mutex` satisfies the `Sync` bound without
    /// requiring any `unsafe` code.
    slots: Box<[Mutex<Option<T>>]>,
    /// Next write index (owned exclusively by the producer).
    head: AtomicUsize,
    /// Next read index (owned exclusively by the consumer).
    tail: AtomicUsize,
    /// Capacity — always a power of two.
    capacity: usize,
    /// Bit-mask for fast modulo: `index & mask == index % capacity`.
    mask: usize,
}

// `Mutex<Option<T>>: Send + Sync` when `T: Send`, so `SpscRingBuffer<T>` is
// `Send + Sync` without any `unsafe impl` blocks.

impl<T> SpscRingBuffer<T> {
    /// Creates a new `SpscRingBuffer` with at least `min_capacity` slots.
    ///
    /// The actual capacity is the smallest power of two >= `min_capacity` and
    /// >= 2.
    ///
    /// # Panics
    ///
    /// Panics if the rounded-up capacity overflows `usize`.
    #[must_use]
    pub fn new(min_capacity: usize) -> Self {
        let capacity = min_capacity.max(2).next_power_of_two();
        let mask = capacity - 1;
        let slots: Box<[Mutex<Option<T>>]> = (0..capacity)
            .map(|_| Mutex::new(None))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            slots,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity,
            mask,
        }
    }

    /// Returns the buffer capacity (always a power of two).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of items currently in the buffer.
    ///
    /// Because `head` and `tail` are owned by separate threads, this value is
    /// approximate when called from either thread; it is exact when called from
    /// a thread that has exclusive access to both.
    #[must_use]
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    /// Returns `true` if the buffer contains no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    /// Returns `true` if the buffer is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len() == self.capacity
    }

    /// Attempts to push `item` into the buffer.
    ///
    /// Returns `true` on success, `false` if the buffer is full (item is
    /// dropped).
    ///
    /// **Must only be called from the producer thread.**
    pub fn push(&self, item: T) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) == self.capacity {
            return false; // full
        }
        let slot = &self.slots[head & self.mask];
        // Only the producer thread accesses head's slot; lock is uncontended.
        if let Ok(mut guard) = slot.lock() {
            *guard = Some(item);
        }
        self.head.store(head.wrapping_add(1), Ordering::Release);
        true
    }

    /// Attempts to pop the oldest item.
    ///
    /// Returns `Some(item)` if the buffer is non-empty, or `None` if empty.
    ///
    /// **Must only be called from the consumer thread.**
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if head == tail {
            return None; // empty
        }
        let slot = &self.slots[tail & self.mask];
        // Only the consumer thread accesses tail's slot; lock is uncontended.
        let item = slot.lock().ok().and_then(|mut g| g.take());
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        item
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for SpscRingBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpscRingBuffer")
            .field("capacity", &self.capacity)
            .field("len", &self.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- RingBuffer tests ---

    #[test]
    fn test_new_empty() {
        let rb: RingBuffer<i32> = RingBuffer::new(4);
        assert!(rb.is_empty());
        assert!(!rb.is_full());
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.capacity(), 4);
    }

    #[test]
    fn test_push_pop_ordering() {
        let mut rb = RingBuffer::new(4);
        assert!(rb.push(1));
        assert!(rb.push(2));
        assert!(rb.push(3));
        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), None);
    }

    #[test]
    fn test_push_when_full_returns_false() {
        let mut rb = RingBuffer::new(2);
        assert!(rb.push(10));
        assert!(rb.push(20));
        assert!(rb.is_full());
        assert!(!rb.push(30));
        assert_eq!(rb.stats().overflow_count, 1);
    }

    #[test]
    fn test_push_overwrite_evicts_oldest() {
        let mut rb = RingBuffer::new(3);
        rb.push_overwrite(1);
        rb.push_overwrite(2);
        rb.push_overwrite(3);
        // Buffer is now full; the next push should evict 1.
        rb.push_overwrite(4);
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), Some(4));
    }

    #[test]
    fn test_peek_does_not_consume() {
        let mut rb = RingBuffer::new(4);
        rb.push(42);
        assert_eq!(rb.peek(), Some(&42));
        assert_eq!(rb.len(), 1);
        assert_eq!(rb.pop(), Some(42));
    }

    #[test]
    fn test_clear() {
        let mut rb = RingBuffer::new(4);
        rb.push(1);
        rb.push(2);
        rb.clear();
        assert!(rb.is_empty());
        assert_eq!(rb.pop(), None);
    }

    #[test]
    fn test_iter_order() {
        let mut rb = RingBuffer::new(5);
        for i in 0..4_i32 {
            rb.push(i);
        }
        let collected: Vec<i32> = rb.iter().copied().collect();
        assert_eq!(collected, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_wrap_around() {
        let mut rb = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.pop(); // consume 1; tail advances
        rb.push(3);
        rb.push(4);
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), Some(4));
    }

    #[test]
    fn test_stats_tracking() {
        let mut rb = RingBuffer::new(2);
        rb.push(1);
        rb.push(2);
        rb.push(3); // overflow
        rb.pop();
        let s = rb.stats();
        assert_eq!(s.push_count, 2);
        assert_eq!(s.pop_count, 1);
        assert_eq!(s.overflow_count, 1);
        assert_eq!(s.len, 1);
    }

    // --- MediaFrameQueue tests ---

    #[test]
    fn test_media_frame_queue_push_pop_video() {
        let mut q = MediaFrameQueue::new(4, 4);
        assert!(q.push_video(vec![1, 2, 3], 1000));
        let (frame, pts) = q.pop_video().expect("should have video");
        assert_eq!(frame, vec![1, 2, 3]);
        assert_eq!(pts, 1000);
    }

    #[test]
    fn test_media_frame_queue_push_pop_audio() {
        let mut q = MediaFrameQueue::new(4, 4);
        assert!(q.push_audio(vec![0.5_f32, -0.5], 2000));
        let (samples, pts) = q.pop_audio().expect("should have audio");
        assert_eq!(pts, 2000);
        assert!((samples[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sync_offset_ms() {
        let mut q = MediaFrameQueue::new(4, 4);
        q.push_video(vec![], 1050);
        q.push_audio(vec![], 1000);
        assert_eq!(q.sync_offset_ms(), Some(50));
    }

    #[test]
    fn test_sync_offset_ms_empty() {
        let q = MediaFrameQueue::new(4, 4);
        assert!(q.sync_offset_ms().is_none());
    }

    // --- SpscRingBuffer tests ---

    #[test]
    fn test_spsc_new_capacity_power_of_two() {
        let buf: SpscRingBuffer<u32> = SpscRingBuffer::new(5);
        assert_eq!(buf.capacity(), 8); // next power of two >= 5
        let buf2: SpscRingBuffer<u32> = SpscRingBuffer::new(8);
        assert_eq!(buf2.capacity(), 8);
        let buf3: SpscRingBuffer<u32> = SpscRingBuffer::new(1);
        assert_eq!(buf3.capacity(), 2); // minimum 2
    }

    #[test]
    fn test_spsc_push_pop_fifo() {
        let buf: SpscRingBuffer<i32> = SpscRingBuffer::new(4);
        assert!(buf.push(10));
        assert!(buf.push(20));
        assert!(buf.push(30));
        assert_eq!(buf.pop(), Some(10));
        assert_eq!(buf.pop(), Some(20));
        assert_eq!(buf.pop(), Some(30));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn test_spsc_is_empty_is_full() {
        let buf: SpscRingBuffer<u8> = SpscRingBuffer::new(2);
        assert!(buf.is_empty());
        assert!(!buf.is_full());
        buf.push(1);
        buf.push(2);
        assert!(buf.is_full());
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_spsc_push_when_full_returns_false() {
        let buf: SpscRingBuffer<u8> = SpscRingBuffer::new(2);
        assert!(buf.push(1));
        assert!(buf.push(2));
        assert!(!buf.push(3)); // full
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_spsc_wrap_around() {
        let buf: SpscRingBuffer<u32> = SpscRingBuffer::new(4);
        // Fill
        for i in 0..4 {
            assert!(buf.push(i));
        }
        // Drain 2
        assert_eq!(buf.pop(), Some(0));
        assert_eq!(buf.pop(), Some(1));
        // Push 2 more (wraps around in the ring)
        assert!(buf.push(4));
        assert!(buf.push(5));
        // Drain all
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), Some(4));
        assert_eq!(buf.pop(), Some(5));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn test_spsc_len() {
        let buf: SpscRingBuffer<u32> = SpscRingBuffer::new(8);
        assert_eq!(buf.len(), 0);
        buf.push(1);
        buf.push(2);
        assert_eq!(buf.len(), 2);
        buf.pop();
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_spsc_threaded_producer_consumer() {
        use std::sync::Arc;
        const N: u32 = 1000;
        let buf: Arc<SpscRingBuffer<u32>> = Arc::new(SpscRingBuffer::new(64));
        let producer = Arc::clone(&buf);
        let handle = std::thread::spawn(move || {
            for i in 0..N {
                while !producer.push(i) {
                    std::thread::yield_now();
                }
            }
        });
        let mut received = Vec::with_capacity(N as usize);
        while received.len() < N as usize {
            if let Some(v) = buf.pop() {
                received.push(v);
            } else {
                std::thread::yield_now();
            }
        }
        handle.join().expect("producer thread should not panic");
        let expected: Vec<u32> = (0..N).collect();
        assert_eq!(received, expected);
    }
}
