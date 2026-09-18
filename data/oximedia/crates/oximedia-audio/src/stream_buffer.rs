//! Lock-free ring-queue stream buffer for audio frame pipelines.
//!
//! This module provides two buffer implementations:
//!
//! - [`StreamBuffer`]: A simple FIFO queue suitable for single-threaded use.
//! - [`LockFreeRingBuffer`]: A single-producer single-consumer (SPSC) lock-free
//!   ring buffer designed for real-time audio threading.  Uses `AtomicUsize`
//!   sequence numbers to coordinate access without a mutex.
//!
//! # Lock-free design
//!
//! [`LockFreeRingBuffer`] can safely be shared between exactly **one writer
//! thread** (audio callback or capture thread) and **one reader thread**
//! (processing or playback thread) without any locking.  The implementation
//! follows the classic SPSC ring-buffer pattern:
//!
//! 1. `head` is updated only by the **producer** (writer).
//! 2. `tail` is updated only by the **consumer** (reader).
//! 3. Both indices are `AtomicUsize` accessed with `Acquire`/`Release`
//!    ordering to ensure the data written by the producer is visible to the
//!    consumer.
//!
//! The effective capacity is `capacity - 1` samples to distinguish between
//! full and empty states without an extra flag.
//!
//! # Example
//!
//! ```
//! use oximedia_audio::stream_buffer::LockFreeRingBuffer;
//! use std::sync::Arc;
//!
//! let buf = Arc::new(LockFreeRingBuffer::new(1024));
//!
//! // Producer side (audio callback thread)
//! let samples = vec![0.0_f32; 256];
//! buf.write_samples(&samples);
//!
//! // Consumer side (processing thread)
//! let mut out = vec![0.0_f32; 256];
//! let n = buf.read_samples(&mut out);
//! assert_eq!(n, 256);
//! ```
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Configuration for a [`StreamBuffer`].
#[derive(Debug, Clone, Copy)]
pub struct StreamBufferConfig {
    /// Maximum number of frames the buffer may hold.
    pub max_frames: usize,
    /// Sample rate in Hz (used for duration calculations).
    pub sample_rate: u32,
    /// Number of channels per frame.
    pub channels: u16,
}

impl StreamBufferConfig {
    /// Create a new configuration.
    #[must_use]
    pub fn new(max_frames: usize, sample_rate: u32, channels: u16) -> Self {
        Self {
            max_frames,
            sample_rate,
            channels,
        }
    }

    /// Maximum queue depth expressed in frames.
    #[must_use]
    pub fn max_frames(&self) -> usize {
        self.max_frames
    }
}

impl Default for StreamBufferConfig {
    fn default() -> Self {
        Self {
            max_frames: 64,
            sample_rate: 48_000,
            channels: 2,
        }
    }
}

/// A single audio frame inside the stream buffer.
#[derive(Debug, Clone)]
pub struct StreamFrame {
    /// Interleaved PCM samples (f32).
    pub samples: Vec<f32>,
    /// Presentation timestamp in samples since stream start.
    pub pts_samples: u64,
    /// Number of channels in this frame.
    pub channels: u16,
    /// Sample rate of the frame (Hz).
    pub sample_rate: u32,
}

impl StreamFrame {
    /// Create a new frame.
    #[must_use]
    pub fn new(samples: Vec<f32>, pts_samples: u64, channels: u16, sample_rate: u32) -> Self {
        Self {
            samples,
            pts_samples,
            channels,
            sample_rate,
        }
    }

    /// Number of multi-channel audio samples (frames) in this buffer.
    ///
    /// That is, the length of `samples` divided by the channel count.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    /// Duration of this frame in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn duration_ms(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.sample_count() as f64 / self.sample_rate as f64 * 1_000.0
    }
}

/// FIFO queue of [`StreamFrame`]s with a configurable capacity.
#[derive(Debug)]
pub struct StreamBuffer {
    queue: VecDeque<StreamFrame>,
    config: StreamBufferConfig,
    /// Total number of frames ever pushed (monotonically increasing).
    total_pushed: u64,
    /// Total number of frames ever popped.
    total_popped: u64,
}

impl StreamBuffer {
    /// Create a new stream buffer with the given configuration.
    #[must_use]
    pub fn new(config: StreamBufferConfig) -> Self {
        Self {
            queue: VecDeque::with_capacity(config.max_frames),
            config,
            total_pushed: 0,
            total_popped: 0,
        }
    }

    /// Push a frame into the buffer.
    ///
    /// Returns `false` (and discards the frame) when the buffer is full.
    pub fn push_frame(&mut self, frame: StreamFrame) -> bool {
        if self.queue.len() >= self.config.max_frames {
            return false;
        }
        self.queue.push_back(frame);
        self.total_pushed += 1;
        true
    }

    /// Pop the oldest frame from the buffer, or `None` if empty.
    pub fn pop_frame(&mut self) -> Option<StreamFrame> {
        let frame = self.queue.pop_front();
        if frame.is_some() {
            self.total_popped += 1;
        }
        frame
    }

    /// Approximate total buffered audio in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn duration_ms(&self) -> f64 {
        self.queue.iter().map(|f| f.duration_ms()).sum()
    }

    /// Number of frames currently queued.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` when no frames are queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns `true` when the queue has reached its configured maximum.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.queue.len() >= self.config.max_frames
    }

    /// Total frames pushed since creation.
    #[must_use]
    pub fn total_pushed(&self) -> u64 {
        self.total_pushed
    }

    /// Total frames popped since creation.
    #[must_use]
    pub fn total_popped(&self) -> u64 {
        self.total_popped
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lock-free SPSC ring buffer
// ─────────────────────────────────────────────────────────────────────────────

/// Single-producer single-consumer lock-free ring buffer for `f32` samples.
///
/// Designed for real-time audio pipelines where one thread writes audio data
/// (the audio callback or capture thread) and another thread reads it (the
/// processing or playback thread).
///
/// Samples are stored as their `u32` bit patterns in `AtomicU32` cells so
/// that the entire structure is `Send + Sync` without any `unsafe` code.
/// The SPSC ring-buffer protocol using `AtomicUsize` head/tail indices
/// ensures correct ordering between producer and consumer.
///
/// ## Capacity
///
/// The actual number of samples that can be buffered is `capacity - 1`.
/// Choose a power-of-two capacity (e.g., 2048, 4096) for best performance.
///
/// ## Thread safety
///
/// Only one writer and one reader are supported.  Using more than one writer
/// or more than one reader concurrently results in incorrect data ordering.
pub struct LockFreeRingBuffer {
    /// Internal sample storage as atomic u32 (f32 bit patterns).
    data: Vec<AtomicU32>,
    /// Write index (producer-owned).
    head: AtomicUsize,
    /// Read index (consumer-owned).
    tail: AtomicUsize,
    /// Capacity (length of `data`).
    cap: usize,
}

// `Vec<AtomicU32>` is already `Send + Sync`, so `LockFreeRingBuffer` is too.
// No `unsafe impl` blocks are required.

impl LockFreeRingBuffer {
    /// Create a new ring buffer that can hold up to `capacity - 1` samples.
    ///
    /// `capacity` should be a power of two for best performance.  A minimum
    /// capacity of 2 is enforced.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(2);
        let data = (0..cap).map(|_| AtomicU32::new(0)).collect();
        Self {
            data,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            cap,
        }
    }

    /// Returns the total capacity of the buffer (number of samples that can
    /// ever be stored).  The usable capacity is `capacity() - 1`.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Number of samples currently available for reading.
    #[must_use]
    pub fn available(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        if head >= tail {
            head - tail
        } else {
            self.cap - tail + head
        }
    }

    /// Number of free slots available for writing.
    #[must_use]
    pub fn free(&self) -> usize {
        self.cap - 1 - self.available()
    }

    /// Returns `true` when the buffer contains no readable samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    /// Returns `true` when the buffer is full (cannot accept more writes).
    #[must_use]
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (head + 1) % self.cap == tail
    }

    /// Write a single sample.
    ///
    /// Returns `true` when successful, `false` when the buffer is full.
    ///
    /// **Must only be called from the producer thread.**
    pub fn write(&self, sample: f32) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) % self.cap;
        if next_head == self.tail.load(Ordering::Acquire) {
            return false; // full
        }
        // Store the f32 bit pattern atomically.  Only the producer writes to
        // data[head], and head < cap, so the index is always in bounds.
        self.data[head].store(sample.to_bits(), Ordering::Relaxed);
        self.head.store(next_head, Ordering::Release);
        true
    }

    /// Read a single sample.
    ///
    /// Returns `Some(sample)` when data is available, `None` when empty.
    ///
    /// **Must only be called from the consumer thread.**
    pub fn read(&self) -> Option<f32> {
        let tail = self.tail.load(Ordering::Relaxed);
        if tail == self.head.load(Ordering::Acquire) {
            return None; // empty
        }
        let bits = self.data[tail].load(Ordering::Relaxed);
        self.tail.store((tail + 1) % self.cap, Ordering::Release);
        Some(f32::from_bits(bits))
    }

    /// Write a block of samples.
    ///
    /// Returns the number of samples actually written (may be less than
    /// `samples.len()` when the buffer does not have enough free space).
    ///
    /// **Must only be called from the producer thread.**
    pub fn write_samples(&self, samples: &[f32]) -> usize {
        let mut written = 0;
        for &s in samples {
            if !self.write(s) {
                break;
            }
            written += 1;
        }
        written
    }

    /// Read samples into `dst`.
    ///
    /// Returns the number of samples actually read (may be less than
    /// `dst.len()` when the buffer does not have enough data).
    ///
    /// **Must only be called from the consumer thread.**
    pub fn read_samples(&self, dst: &mut [f32]) -> usize {
        let mut read = 0;
        for slot in dst.iter_mut() {
            match self.read() {
                Some(s) => {
                    *slot = s;
                    read += 1;
                }
                None => break,
            }
        }
        read
    }

    /// Clear all samples from the buffer.
    ///
    /// **Must only be called when no concurrent reads/writes are in progress.**
    pub fn clear(&self) {
        let head = self.head.load(Ordering::Relaxed);
        self.tail.store(head, Ordering::Release);
    }
}

impl std::fmt::Debug for LockFreeRingBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LockFreeRingBuffer")
            .field("cap", &self.cap)
            .field("available", &self.available())
            .field("free", &self.free())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn make_frame(n_samples: usize, pts: u64) -> StreamFrame {
        StreamFrame::new(
            vec![0.0_f32; n_samples * 2], // stereo
            pts,
            2,
            48_000,
        )
    }

    #[test]
    fn test_config_max_frames() {
        let cfg = StreamBufferConfig::new(32, 48_000, 2);
        assert_eq!(cfg.max_frames(), 32);
    }

    #[test]
    fn test_config_default() {
        let cfg = StreamBufferConfig::default();
        assert_eq!(cfg.max_frames, 64);
        assert_eq!(cfg.sample_rate, 48_000);
    }

    #[test]
    fn test_frame_sample_count() {
        let frame = make_frame(480, 0);
        assert_eq!(frame.sample_count(), 480);
    }

    #[test]
    fn test_frame_duration_ms() {
        let frame = make_frame(480, 0); // 480/48000 = 10ms
        let dur = frame.duration_ms();
        assert!((dur - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_frame_duration_zero_rate() {
        let frame = StreamFrame::new(vec![0.0; 4], 0, 2, 0);
        assert_eq!(frame.duration_ms(), 0.0);
    }

    #[test]
    fn test_buffer_push_and_pop() {
        let cfg = StreamBufferConfig::default();
        let mut buf = StreamBuffer::new(cfg);
        let f = make_frame(480, 0);
        assert!(buf.push_frame(f));
        assert_eq!(buf.len(), 1);
        let popped = buf.pop_frame();
        assert!(popped.is_some());
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_fifo_order() {
        let cfg = StreamBufferConfig::default();
        let mut buf = StreamBuffer::new(cfg);
        buf.push_frame(make_frame(480, 0));
        buf.push_frame(make_frame(480, 480));
        let first = buf.pop_frame().expect("should succeed");
        assert_eq!(first.pts_samples, 0);
        let second = buf.pop_frame().expect("should succeed");
        assert_eq!(second.pts_samples, 480);
    }

    #[test]
    fn test_buffer_full_rejects_push() {
        let cfg = StreamBufferConfig::new(2, 48_000, 2);
        let mut buf = StreamBuffer::new(cfg);
        assert!(buf.push_frame(make_frame(480, 0)));
        assert!(buf.push_frame(make_frame(480, 480)));
        assert!(buf.is_full());
        assert!(!buf.push_frame(make_frame(480, 960)));
    }

    #[test]
    fn test_buffer_is_empty_initially() {
        let buf = StreamBuffer::new(StreamBufferConfig::default());
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_pop_empty_returns_none() {
        let mut buf = StreamBuffer::new(StreamBufferConfig::default());
        assert!(buf.pop_frame().is_none());
    }

    #[test]
    fn test_buffer_duration_ms() {
        let cfg = StreamBufferConfig::default();
        let mut buf = StreamBuffer::new(cfg);
        buf.push_frame(make_frame(480, 0)); // 10ms
        buf.push_frame(make_frame(480, 480)); // 10ms
        let dur = buf.duration_ms();
        assert!((dur - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_buffer_total_counters() {
        let cfg = StreamBufferConfig::default();
        let mut buf = StreamBuffer::new(cfg);
        buf.push_frame(make_frame(480, 0));
        buf.push_frame(make_frame(480, 480));
        buf.pop_frame();
        assert_eq!(buf.total_pushed(), 2);
        assert_eq!(buf.total_popped(), 1);
    }

    #[test]
    fn test_frame_zero_channels() {
        let frame = StreamFrame::new(vec![0.0; 10], 0, 0, 48_000);
        assert_eq!(frame.sample_count(), 0);
    }

    // ── LockFreeRingBuffer tests ───────────────────────────────────────────────

    #[test]
    fn test_ringbuf_initially_empty() {
        let rb = LockFreeRingBuffer::new(16);
        assert!(rb.is_empty());
        assert_eq!(rb.available(), 0);
    }

    #[test]
    fn test_ringbuf_write_and_read_single() {
        let rb = LockFreeRingBuffer::new(16);
        assert!(rb.write(0.5));
        assert!(!rb.is_empty());
        let s = rb.read().expect("should have data");
        assert!((s - 0.5).abs() < 1e-7);
        assert!(rb.is_empty());
    }

    #[test]
    fn test_ringbuf_write_block_read_block() {
        let rb = LockFreeRingBuffer::new(64);
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let written = rb.write_samples(&data);
        assert_eq!(written, 32);
        let mut dst = vec![0.0_f32; 32];
        let read = rb.read_samples(&mut dst);
        assert_eq!(read, 32);
        for (a, b) in data.iter().zip(dst.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ringbuf_full_rejects_write() {
        let rb = LockFreeRingBuffer::new(4); // usable cap = 3
        assert!(rb.write(1.0));
        assert!(rb.write(2.0));
        assert!(rb.write(3.0));
        assert!(rb.is_full());
        assert!(!rb.write(4.0)); // should fail
    }

    #[test]
    fn test_ringbuf_read_empty_returns_none() {
        let rb = LockFreeRingBuffer::new(16);
        assert!(rb.read().is_none());
    }

    #[test]
    fn test_ringbuf_wrap_around() {
        let rb = LockFreeRingBuffer::new(8); // usable cap = 7
                                             // Fill then partially drain, then fill again — exercises wrap-around
        for i in 0..7 {
            rb.write(i as f32);
        }
        for _ in 0..4 {
            rb.read();
        }
        for i in 0..4 {
            assert!(rb.write(i as f32 + 10.0));
        }
        // Remaining: 4 from original + 4 new = 7; but usable cap = 7, so last write should fail
        assert_eq!(rb.available(), 7);
    }

    #[test]
    fn test_ringbuf_clear() {
        let rb = LockFreeRingBuffer::new(16);
        rb.write_samples(&[1.0, 2.0, 3.0]);
        assert_eq!(rb.available(), 3);
        rb.clear();
        assert!(rb.is_empty());
    }

    #[test]
    fn test_ringbuf_capacity() {
        let rb = LockFreeRingBuffer::new(32);
        assert_eq!(rb.capacity(), 32);
    }

    #[test]
    fn test_ringbuf_free() {
        let rb = LockFreeRingBuffer::new(16); // usable = 15
        rb.write(0.5);
        assert_eq!(rb.free(), 14);
    }

    #[test]
    fn test_ringbuf_fifo_order() {
        let rb = LockFreeRingBuffer::new(16);
        rb.write(1.0);
        rb.write(2.0);
        rb.write(3.0);
        assert_eq!(rb.read().expect("1"), 1.0);
        assert_eq!(rb.read().expect("2"), 2.0);
        assert_eq!(rb.read().expect("3"), 3.0);
    }

    #[test]
    fn test_ringbuf_arc_shared() {
        let rb = Arc::new(LockFreeRingBuffer::new(64));
        let rb2 = Arc::clone(&rb);
        // Simulate producer/consumer in the same thread for determinism
        rb.write_samples(&[0.1, 0.2, 0.3]);
        let mut out = vec![0.0_f32; 3];
        rb2.read_samples(&mut out);
        assert!((out[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_ringbuf_minimum_capacity_enforced() {
        let rb = LockFreeRingBuffer::new(0); // should be clamped to 2
        assert_eq!(rb.capacity(), 2);
    }

    #[test]
    fn test_ringbuf_debug_format() {
        let rb = LockFreeRingBuffer::new(16);
        let s = format!("{rb:?}");
        assert!(s.contains("LockFreeRingBuffer"));
    }
}
