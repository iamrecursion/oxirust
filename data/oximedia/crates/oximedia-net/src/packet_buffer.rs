#![allow(dead_code)]
//! Network packet buffer with priority queuing and expiry.
//!
//! Provides a bounded, priority-aware packet buffer that automatically
//! discards expired or lowest-priority packets when capacity is exceeded.

use std::time::{Duration, Instant};

/// Priority level for network packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PacketPriority {
    /// Background bulk data, lowest priority.
    Background = 0,
    /// Best-effort data.
    BestEffort = 1,
    /// Interactive real-time data.
    Interactive = 2,
    /// Critical control / signalling data.
    Critical = 3,
}

impl PacketPriority {
    /// Returns a numeric value for the priority (higher = more important).
    #[must_use]
    pub fn numeric_value(self) -> u8 {
        self as u8
    }

    /// Returns `true` when this priority level is higher than `other`.
    #[must_use]
    pub fn is_higher_than(self, other: Self) -> bool {
        self.numeric_value() > other.numeric_value()
    }
}

/// A single network packet with metadata.
#[derive(Debug, Clone)]
pub struct NetworkPacket {
    /// Packet payload bytes.
    pub data: Vec<u8>,
    /// Sequence number.
    pub sequence: u64,
    /// Priority of this packet.
    pub priority: PacketPriority,
    /// Absolute timestamp when the packet was enqueued.
    pub enqueued_at: Instant,
    /// Maximum lifetime of the packet before it is considered expired.
    pub ttl: Duration,
}

impl NetworkPacket {
    /// Creates a new packet.
    #[must_use]
    pub fn new(data: Vec<u8>, sequence: u64, priority: PacketPriority, ttl: Duration) -> Self {
        Self {
            data,
            sequence,
            priority,
            enqueued_at: Instant::now(),
            ttl,
        }
    }

    /// Returns `true` when the packet's TTL has elapsed.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.enqueued_at.elapsed() >= self.ttl
    }

    /// Returns the number of payload bytes.
    #[must_use]
    pub fn payload_len(&self) -> usize {
        self.data.len()
    }
}

/// A bounded network packet buffer with priority support.
///
/// When the buffer is full and a new packet arrives, the oldest
/// lowest-priority packet is dropped to make room.
#[derive(Debug)]
pub struct PacketBuffer {
    packets: Vec<NetworkPacket>,
    capacity: usize,
    dropped: u64,
}

impl PacketBuffer {
    /// Creates a new `PacketBuffer` with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            packets: Vec::with_capacity(capacity),
            capacity,
            dropped: 0,
        }
    }

    /// Returns the maximum number of packets the buffer can hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the current number of packets in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.packets.len()
    }

    /// Returns `true` when the buffer contains no packets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.packets.is_empty()
    }

    /// Returns the total number of packets dropped due to capacity overflow.
    #[must_use]
    pub fn dropped_count(&self) -> u64 {
        self.dropped
    }

    /// Removes all expired packets from the buffer.
    pub fn purge_expired(&mut self) {
        let before = self.packets.len();
        self.packets.retain(|p| !p.is_expired());
        let removed = before - self.packets.len();
        self.dropped += removed as u64;
    }

    /// Enqueues a packet.
    ///
    /// If the buffer is at capacity, the oldest packet with the lowest
    /// priority is dropped first.  If no lower-priority packet exists,
    /// the incoming packet is dropped.
    ///
    /// Returns `true` if the packet was accepted.
    pub fn enqueue(&mut self, packet: NetworkPacket) -> bool {
        if self.packets.len() < self.capacity {
            self.packets.push(packet);
            return true;
        }
        // Find the oldest lowest-priority packet
        if let Some(idx) = self.find_drop_candidate(packet.priority) {
            self.packets.remove(idx);
            self.dropped += 1;
            self.packets.push(packet);
            true
        } else {
            self.dropped += 1;
            false
        }
    }

    /// Dequeues the highest-priority, oldest packet.
    ///
    /// Returns `None` when the buffer is empty.
    pub fn dequeue(&mut self) -> Option<NetworkPacket> {
        if self.packets.is_empty() {
            return None;
        }
        // Find the highest-priority packet (latest priority wins; earliest
        // sequence among ties).
        let idx = self
            .packets
            .iter()
            .enumerate()
            .max_by_key(|(_, p)| (p.priority.numeric_value(), u64::MAX - p.sequence))
            .map(|(i, _)| i)?;
        Some(self.packets.remove(idx))
    }

    /// Drops the single oldest packet with the lowest priority.
    ///
    /// Returns `true` if a packet was dropped.
    pub fn drop_oldest(&mut self) -> bool {
        if self.packets.is_empty() {
            return false;
        }
        if let Some(idx) = self.find_drop_candidate(PacketPriority::Critical) {
            self.packets.remove(idx);
            self.dropped += 1;
            true
        } else {
            false
        }
    }

    /// Finds the index of the oldest, lowest-priority packet whose priority
    /// is strictly less than `incoming_priority`.
    fn find_drop_candidate(&self, incoming_priority: PacketPriority) -> Option<usize> {
        self.packets
            .iter()
            .enumerate()
            .filter(|(_, p)| p.priority < incoming_priority)
            .min_by_key(|(_, p)| (p.priority.numeric_value(), p.sequence))
            .map(|(i, _)| i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pkt(seq: u64, prio: PacketPriority) -> NetworkPacket {
        NetworkPacket::new(vec![0u8; 4], seq, prio, Duration::from_secs(60))
    }

    #[test]
    fn test_priority_numeric_value() {
        assert_eq!(PacketPriority::Background.numeric_value(), 0);
        assert_eq!(PacketPriority::Critical.numeric_value(), 3);
    }

    #[test]
    fn test_priority_is_higher_than() {
        assert!(PacketPriority::Critical.is_higher_than(PacketPriority::BestEffort));
        assert!(!PacketPriority::Background.is_higher_than(PacketPriority::Interactive));
    }

    #[test]
    fn test_packet_payload_len() {
        let p = pkt(1, PacketPriority::BestEffort);
        assert_eq!(p.payload_len(), 4);
    }

    #[test]
    fn test_packet_not_expired_fresh() {
        let p = pkt(1, PacketPriority::BestEffort);
        assert!(!p.is_expired());
    }

    #[test]
    fn test_packet_expired_zero_ttl() {
        let p = NetworkPacket::new(vec![], 1, PacketPriority::Background, Duration::ZERO);
        // Duration::ZERO means already expired on next elapsed check
        assert!(p.is_expired());
    }

    #[test]
    fn test_buffer_enqueue_within_capacity() {
        let mut buf = PacketBuffer::new(4);
        assert!(buf.enqueue(pkt(1, PacketPriority::BestEffort)));
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_buffer_dequeue_empty() {
        let mut buf = PacketBuffer::new(4);
        assert!(buf.dequeue().is_none());
    }

    #[test]
    fn test_buffer_dequeue_highest_priority() {
        let mut buf = PacketBuffer::new(4);
        buf.enqueue(pkt(1, PacketPriority::Background));
        buf.enqueue(pkt(2, PacketPriority::Critical));
        let p = buf.dequeue().expect("should succeed in test");
        assert_eq!(p.priority, PacketPriority::Critical);
    }

    #[test]
    fn test_buffer_drop_oldest_low_priority() {
        let mut buf = PacketBuffer::new(2);
        buf.enqueue(pkt(1, PacketPriority::Background));
        buf.enqueue(pkt(2, PacketPriority::BestEffort));
        assert!(buf.drop_oldest());
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_buffer_drop_on_overflow() {
        let mut buf = PacketBuffer::new(1);
        buf.enqueue(pkt(1, PacketPriority::Background));
        // Higher priority packet should evict the low-priority one
        let accepted = buf.enqueue(pkt(2, PacketPriority::Critical));
        assert!(accepted);
        assert_eq!(buf.dropped_count(), 1);
    }

    #[test]
    fn test_buffer_overflow_same_priority_dropped() {
        let mut buf = PacketBuffer::new(1);
        buf.enqueue(pkt(1, PacketPriority::Critical));
        // Cannot evict — incoming has same priority, not higher
        let accepted = buf.enqueue(pkt(2, PacketPriority::Critical));
        assert!(!accepted);
        assert_eq!(buf.dropped_count(), 1);
    }

    #[test]
    fn test_buffer_capacity() {
        let buf = PacketBuffer::new(8);
        assert_eq!(buf.capacity(), 8);
    }

    #[test]
    fn test_buffer_is_empty() {
        let buf = PacketBuffer::new(4);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_purge_expired() {
        let mut buf = PacketBuffer::new(4);
        // Add a packet with zero TTL (immediately expired)
        let p = NetworkPacket::new(vec![], 1, PacketPriority::BestEffort, Duration::ZERO);
        buf.packets.push(p);
        buf.enqueue(pkt(2, PacketPriority::Interactive));
        buf.purge_expired();
        assert_eq!(buf.len(), 1);
    }
}
