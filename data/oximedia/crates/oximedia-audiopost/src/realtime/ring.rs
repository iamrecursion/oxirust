//! Single-producer single-consumer wait-free ring buffer for f32 audio samples.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Single-producer single-consumer wait-free ring buffer for f32 audio samples.
///
/// Capacity is always rounded up to the next power of two for efficient
/// modular indexing via bitmasking.  Samples are stored as their bit
/// representation in [`AtomicU32`] slots, which makes all memory operations
/// atomic and avoids `unsafe` code.
///
/// # Concurrency model
///
/// Exactly one thread may call [`push_slice`] (the producer) and exactly one
/// thread may call [`pop_slice`] (the consumer) at any given time.  Using
/// multiple producers or multiple consumers simultaneously is **unsound** and
/// will produce incorrect results.
///
/// [`push_slice`]: AudioRingBuffer::push_slice
/// [`pop_slice`]: AudioRingBuffer::pop_slice
pub struct AudioRingBuffer {
    data: Vec<AtomicU32>,
    capacity: usize,
    mask: usize,
    /// Monotonically increasing producer write index.
    head: AtomicUsize,
    /// Monotonically increasing consumer read index.
    tail: AtomicUsize,
}

// AtomicU32 is Sync+Send; AudioRingBuffer is therefore also Sync+Send.
// The SPSC contract is enforced by the caller (documented above).
// No unsafe impls are needed because AtomicU32 already is Sync.

impl AudioRingBuffer {
    /// Create a new ring buffer.
    ///
    /// `capacity` is rounded up to the next power of two (minimum 16).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two().max(16);
        let data = (0..cap).map(|_| AtomicU32::new(0)).collect();
        Self {
            data,
            capacity: cap,
            mask: cap - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Returns the number of samples available to read.
    #[inline]
    pub fn available_read(&self) -> usize {
        let h = self.head.load(Ordering::Acquire);
        let t = self.tail.load(Ordering::Acquire);
        h.wrapping_sub(t)
    }

    /// Returns the number of samples available to write.
    #[inline]
    pub fn available_write(&self) -> usize {
        self.capacity - self.available_read()
    }

    /// Push samples from `src` into the ring buffer.
    ///
    /// Returns the number of samples actually written (may be less than
    /// `src.len()` if the buffer does not have enough free space).
    pub fn push_slice(&self, src: &[f32]) -> usize {
        let avail = self.available_write();
        let n = src.len().min(avail);
        if n == 0 {
            return 0;
        }
        let h = self.head.load(Ordering::Relaxed);
        for i in 0..n {
            let slot = &self.data[h.wrapping_add(i) & self.mask];
            slot.store(src[i].to_bits(), Ordering::Relaxed);
        }
        self.head.store(h.wrapping_add(n), Ordering::Release);
        n
    }

    /// Pop samples from the ring buffer into `dst`.
    ///
    /// Returns the number of samples actually read (may be less than
    /// `dst.len()` if fewer samples are available).
    pub fn pop_slice(&self, dst: &mut [f32]) -> usize {
        let avail = self.available_read();
        let n = dst.len().min(avail);
        if n == 0 {
            return 0;
        }
        let t = self.tail.load(Ordering::Relaxed);
        for i in 0..n {
            let slot = &self.data[t.wrapping_add(i) & self.mask];
            dst[i] = f32::from_bits(slot.load(Ordering::Relaxed));
        }
        self.tail.store(t.wrapping_add(n), Ordering::Release);
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_push_pop_correctness() {
        let ring = AudioRingBuffer::new(1024);
        let data: Vec<f32> = (0..512).map(|i| i as f32 / 512.0).collect();
        let n = ring.push_slice(&data);
        assert_eq!(n, 512);
        let mut out = vec![0.0_f32; 512];
        let m = ring.pop_slice(&mut out);
        assert_eq!(m, 512);
        assert_eq!(out, data);
    }

    #[test]
    fn test_ring_buffer_capacity_rounding() {
        let ring = AudioRingBuffer::new(100);
        // Should round up to 128.
        assert_eq!(ring.capacity, 128);
    }

    #[test]
    fn test_ring_buffer_full_then_empty() {
        let ring = AudioRingBuffer::new(16);
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(ring.push_slice(&data), 16);
        // Buffer is now full — no more writes.
        assert_eq!(ring.push_slice(&[1.0]), 0);
        // Drain it.
        let mut out = vec![0.0_f32; 16];
        assert_eq!(ring.pop_slice(&mut out), 16);
        assert_eq!(out, data);
        // Buffer is now empty.
        assert_eq!(ring.pop_slice(&mut out), 0);
    }

    #[test]
    fn test_ring_buffer_wraparound() {
        let ring = AudioRingBuffer::new(16);
        // Fill completely.
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(ring.push_slice(&data), 16);
        // Read half — free up 8 slots.
        let mut out = vec![0.0_f32; 8];
        assert_eq!(ring.pop_slice(&mut out), 8);
        // Write 8 more (wraps around the ring).
        let data2: Vec<f32> = (16..24).map(|i| i as f32).collect();
        assert_eq!(ring.push_slice(&data2), 8);
        // Read remaining 16.
        let mut out2 = vec![0.0_f32; 16];
        let n = ring.pop_slice(&mut out2);
        assert_eq!(n, 16);
        assert_eq!(&out2[..8], &data[8..16]);
        assert_eq!(&out2[8..16], data2.as_slice());
    }

    #[test]
    fn test_ring_buffer_available_counters() {
        let ring = AudioRingBuffer::new(32);
        assert_eq!(ring.available_read(), 0);
        assert_eq!(ring.available_write(), 32);
        let chunk: Vec<f32> = vec![1.0; 10];
        ring.push_slice(&chunk);
        assert_eq!(ring.available_read(), 10);
        assert_eq!(ring.available_write(), 22);
    }

    #[test]
    fn test_ring_buffer_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let ring = Arc::new(AudioRingBuffer::new(4096));
        let ring_clone = ring.clone();
        let n_total = 1_000_000_usize;

        let producer = thread::spawn(move || {
            let chunk: Vec<f32> = (0..64).map(|i| i as f32).collect();
            let mut total_sent = 0_usize;
            while total_sent < n_total {
                let n = ring_clone.push_slice(&chunk);
                total_sent += n;
                if n == 0 {
                    std::hint::spin_loop();
                }
            }
        });

        let mut total_received = 0_usize;
        let mut checksum = 0.0_f64;
        let mut buf = vec![0.0_f32; 64];
        while total_received < n_total {
            let n = ring.pop_slice(&mut buf);
            for &s in &buf[..n] {
                checksum += s as f64;
            }
            total_received += n;
            if n == 0 {
                std::hint::spin_loop();
            }
        }
        producer.join().expect("producer thread panicked");

        // Each chunk is [0, 1, ..., 63]; n_total=1_000_000 exactly 15625 chunks.
        // Sum = 15625 * (0+1+…+63) = 15625 * 2016 = 31_500_000.
        let expected = (n_total / 64) as f64 * (0..64_i64).sum::<i64>() as f64;
        assert!(
            (checksum - expected).abs() < 1.0,
            "checksum mismatch: {} vs {}",
            checksum,
            expected
        );
    }
}
