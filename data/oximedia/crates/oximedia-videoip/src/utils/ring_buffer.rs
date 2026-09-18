//! Lock-free ring buffer for packet buffering.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Lock-free single-producer single-consumer ring buffer.
pub struct RingBuffer<T> {
    /// Buffer storage.
    buffer: Vec<Option<T>>,
    /// Read position.
    read_pos: Arc<AtomicUsize>,
    /// Write position.
    write_pos: Arc<AtomicUsize>,
    /// Buffer capacity.
    capacity: usize,
}

impl<T> RingBuffer<T> {
    /// Creates a new ring buffer with the specified capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(None);
        }

        Self {
            buffer,
            read_pos: Arc::new(AtomicUsize::new(0)),
            write_pos: Arc::new(AtomicUsize::new(0)),
            capacity,
        }
    }

    /// Attempts to push an item into the buffer.
    ///
    /// Returns `Ok(())` if successful, `Err(item)` if the buffer is full.
    pub fn push(&mut self, item: T) -> Result<(), T> {
        let write = self.write_pos.load(Ordering::Acquire);
        let read = self.read_pos.load(Ordering::Acquire);

        let next_write = (write + 1) % self.capacity;
        if next_write == read {
            // Buffer is full
            return Err(item);
        }

        self.buffer[write] = Some(item);
        self.write_pos.store(next_write, Ordering::Release);

        Ok(())
    }

    /// Attempts to pop an item from the buffer.
    ///
    /// Returns `Some(item)` if successful, `None` if the buffer is empty.
    pub fn pop(&mut self) -> Option<T> {
        let read = self.read_pos.load(Ordering::Acquire);
        let write = self.write_pos.load(Ordering::Acquire);

        if read == write {
            // Buffer is empty
            return None;
        }

        let item = self.buffer[read].take();
        let next_read = (read + 1) % self.capacity;
        self.read_pos.store(next_read, Ordering::Release);

        item
    }

    /// Returns the number of items currently in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        let write = self.write_pos.load(Ordering::Acquire);
        let read = self.read_pos.load(Ordering::Acquire);

        if write >= read {
            write - read
        } else {
            self.capacity - read + write
        }
    }

    /// Returns true if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the buffer is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        let write = self.write_pos.load(Ordering::Acquire);
        let read = self.read_pos.load(Ordering::Acquire);
        let next_write = (write + 1) % self.capacity;

        next_write == read
    }

    /// Returns the buffer capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clears all items from the buffer.
    pub fn clear(&mut self) {
        while self.pop().is_some() {}
    }
}

/// Packet buffer using ring buffer for efficient packet queuing.
pub struct PacketBuffer {
    /// Internal ring buffer.
    buffer: RingBuffer<crate::packet::Packet>,
    /// Buffer high watermark (triggers overflow warning).
    high_watermark: usize,
    /// Buffer low watermark (triggers underflow warning).
    low_watermark: usize,
    /// Overflow count.
    overflow_count: u64,
    /// Underflow count.
    underflow_count: u64,
}

impl PacketBuffer {
    /// Creates a new packet buffer.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: RingBuffer::new(capacity),
            high_watermark: (capacity * 3) / 4,
            low_watermark: capacity / 4,
            overflow_count: 0,
            underflow_count: 0,
        }
    }

    /// Adds a packet to the buffer.
    ///
    /// Returns `true` if successful, `false` if the buffer is full.
    pub fn add(&mut self, packet: crate::packet::Packet) -> bool {
        if let Ok(()) = self.buffer.push(packet) {
            if self.buffer.len() >= self.high_watermark {
                tracing::warn!("Packet buffer high watermark reached");
            }
            true
        } else {
            self.overflow_count += 1;
            tracing::error!("Packet buffer overflow");
            false
        }
    }

    /// Retrieves a packet from the buffer.
    pub fn get(&mut self) -> Option<crate::packet::Packet> {
        match self.buffer.pop() {
            Some(packet) => {
                if self.buffer.len() <= self.low_watermark {
                    self.underflow_count += 1;
                }
                Some(packet)
            }
            None => None,
        }
    }

    /// Returns the number of packets in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns true if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns the overflow count.
    #[must_use]
    pub const fn overflow_count(&self) -> u64 {
        self.overflow_count
    }

    /// Returns the underflow count.
    #[must_use]
    pub const fn underflow_count(&self) -> u64 {
        self.underflow_count
    }

    /// Returns the buffer occupancy as a percentage (0.0-1.0).
    #[must_use]
    pub fn occupancy(&self) -> f64 {
        self.buffer.len() as f64 / self.buffer.capacity() as f64
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::PacketBuilder;
    use bytes::Bytes;

    #[test]
    fn test_ring_buffer_new() {
        let buffer: RingBuffer<u32> = RingBuffer::new(10);
        assert_eq!(buffer.capacity(), 10);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_ring_buffer_push_pop() {
        let mut buffer = RingBuffer::new(5);

        assert!(buffer.push(1).is_ok());
        assert!(buffer.push(2).is_ok());
        assert!(buffer.push(3).is_ok());

        assert_eq!(buffer.len(), 3);

        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(3));
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn test_ring_buffer_full() {
        let mut buffer = RingBuffer::new(3);

        assert!(buffer.push(1).is_ok());
        assert!(buffer.push(2).is_ok());

        assert!(buffer.is_full());

        // Should fail when full
        assert!(buffer.push(3).is_err());
    }

    #[test]
    fn test_ring_buffer_wrap_around() {
        let mut buffer = RingBuffer::new(3);

        // Fill buffer
        buffer.push(1).expect("should succeed in test");
        buffer.push(2).expect("should succeed in test");

        // Pop one
        assert_eq!(buffer.pop(), Some(1));

        // Push again (should wrap around)
        buffer.push(3).expect("should succeed in test");

        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(3));
    }

    #[test]
    fn test_ring_buffer_clear() {
        let mut buffer = RingBuffer::new(5);

        buffer.push(1).expect("should succeed in test");
        buffer.push(2).expect("should succeed in test");
        buffer.push(3).expect("should succeed in test");

        buffer.clear();

        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_packet_buffer() {
        let mut buffer = PacketBuffer::new(10);

        let packet = PacketBuilder::new(0)
            .video()
            .build(Bytes::from_static(b"test"))
            .expect("should succeed in test");

        assert!(buffer.add(packet));
        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());

        let retrieved = buffer.get();
        assert!(retrieved.is_some());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_packet_buffer_overflow() {
        let mut buffer = PacketBuffer::new(3);

        for i in 0..5 {
            let packet = PacketBuilder::new(i)
                .video()
                .build(Bytes::from_static(b"test"))
                .expect("should succeed in test");

            buffer.add(packet);
        }

        assert!(buffer.overflow_count() > 0);
    }

    #[test]
    fn test_packet_buffer_occupancy() {
        let mut buffer = PacketBuffer::new(10);

        for i in 0..5 {
            let packet = PacketBuilder::new(i)
                .video()
                .build(Bytes::from_static(b"test"))
                .expect("should succeed in test");

            buffer.add(packet);
        }

        assert!((buffer.occupancy() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_packet_buffer_clear() {
        let mut buffer = PacketBuffer::new(10);

        for i in 0..5 {
            let packet = PacketBuilder::new(i)
                .video()
                .build(Bytes::from_static(b"test"))
                .expect("should succeed in test");

            buffer.add(packet);
        }

        buffer.clear();
        assert_eq!(buffer.len(), 0);
    }
}
