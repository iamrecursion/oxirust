//! SRT loss list management.
//!
//! Tracks lost packets and manages retransmission requests.

use std::collections::{BTreeSet, VecDeque};

/// Loss list for tracking lost packets.
#[derive(Debug, Default)]
pub struct LossList {
    /// Set of lost sequence numbers.
    lost_packets: BTreeSet<u32>,
    /// Maximum number of entries to track.
    max_entries: usize,
}

impl LossList {
    /// Creates a new loss list.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            lost_packets: BTreeSet::new(),
            max_entries,
        }
    }

    /// Adds a lost packet sequence number.
    pub fn add(&mut self, seq: u32) {
        if self.lost_packets.len() < self.max_entries {
            self.lost_packets.insert(seq);
        }
    }

    /// Adds a range of lost packets.
    pub fn add_range(&mut self, start: u32, end: u32) {
        let mut seq = start;
        while seq != end && self.lost_packets.len() < self.max_entries {
            self.lost_packets.insert(seq);
            seq = seq.wrapping_add(1);
        }
    }

    /// Removes a packet from the loss list (when received or timed out).
    pub fn remove(&mut self, seq: u32) -> bool {
        self.lost_packets.remove(&seq)
    }

    /// Returns true if the packet is in the loss list.
    #[must_use]
    pub fn contains(&self, seq: u32) -> bool {
        self.lost_packets.contains(&seq)
    }

    /// Returns the number of lost packets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lost_packets.len()
    }

    /// Returns true if the loss list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lost_packets.is_empty()
    }

    /// Gets all lost sequence numbers.
    #[must_use]
    pub fn lost_sequences(&self) -> Vec<u32> {
        self.lost_packets.iter().copied().collect()
    }

    /// Gets lost sequences as compressed ranges for NAK.
    #[must_use]
    pub fn compressed_ranges(&self) -> Vec<LossRange> {
        let mut ranges = Vec::new();
        let mut current_start: Option<u32> = None;
        let mut current_end: Option<u32> = None;

        for &seq in &self.lost_packets {
            match (current_start, current_end) {
                (None, None) => {
                    current_start = Some(seq);
                    current_end = Some(seq);
                }
                (Some(start), Some(end)) => {
                    if seq == end.wrapping_add(1) {
                        current_end = Some(seq);
                    } else {
                        ranges.push(LossRange { start, end });
                        current_start = Some(seq);
                        current_end = Some(seq);
                    }
                }
                _ => unreachable!(),
            }
        }

        if let (Some(start), Some(end)) = (current_start, current_end) {
            ranges.push(LossRange { start, end });
        }

        ranges
    }

    /// Clears the loss list.
    pub fn clear(&mut self) {
        self.lost_packets.clear();
    }

    /// Returns the oldest lost sequence number.
    #[must_use]
    pub fn oldest(&self) -> Option<u32> {
        self.lost_packets.iter().next().copied()
    }
}

/// Represents a range of lost packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LossRange {
    /// Start sequence number (inclusive).
    pub start: u32,
    /// End sequence number (inclusive).
    pub end: u32,
}

impl LossRange {
    /// Creates a single-packet range.
    #[must_use]
    pub const fn single(seq: u32) -> Self {
        Self {
            start: seq,
            end: seq,
        }
    }

    /// Returns the number of packets in this range.
    #[must_use]
    pub fn count(&self) -> u32 {
        if self.end >= self.start {
            self.end - self.start + 1
        } else {
            // Handle wraparound
            (u32::MAX - self.start) + self.end + 2
        }
    }

    /// Returns true if this is a single packet.
    #[must_use]
    pub const fn is_single(&self) -> bool {
        self.start == self.end
    }
}

/// Packet receive buffer for handling out-of-order packets.
#[derive(Debug)]
pub struct ReceiveBuffer {
    /// Expected next sequence number.
    expected_seq: u32,
    /// Buffered out-of-order packets.
    buffer: VecDeque<BufferedPacket>,
    /// Maximum buffer size.
    max_size: usize,
}

/// A buffered packet.
#[derive(Debug, Clone)]
pub struct BufferedPacket {
    /// Sequence number.
    pub seq: u32,
    /// Packet data.
    pub data: bytes::Bytes,
}

impl ReceiveBuffer {
    /// Creates a new receive buffer.
    #[must_use]
    pub const fn new(initial_seq: u32, max_size: usize) -> Self {
        Self {
            expected_seq: initial_seq,
            buffer: VecDeque::new(),
            max_size,
        }
    }

    /// Returns the next expected sequence number.
    #[must_use]
    pub const fn expected_seq(&self) -> u32 {
        self.expected_seq
    }

    /// Processes a received packet.
    ///
    /// Returns `Some(seq)` if this packet can be delivered immediately,
    /// or `None` if it's buffered for later.
    pub fn process(&mut self, seq: u32, data: bytes::Bytes) -> Option<u32> {
        if seq == self.expected_seq {
            self.expected_seq = self.expected_seq.wrapping_add(1);
            Some(seq)
        } else if seq_after(seq, self.expected_seq) {
            // Out of order - buffer it
            if self.buffer.len() < self.max_size {
                self.buffer.push_back(BufferedPacket { seq, data });
                self.buffer.make_contiguous().sort_by_key(|p| p.seq);
            }
            None
        } else {
            // Duplicate or old packet
            None
        }
    }

    /// Tries to deliver buffered packets in order.
    ///
    /// Returns sequence numbers that can now be delivered.
    pub fn try_deliver(&mut self) -> Vec<u32> {
        let mut delivered = Vec::new();

        while let Some(front) = self.buffer.front() {
            if front.seq == self.expected_seq {
                delivered.push(front.seq);
                self.expected_seq = self.expected_seq.wrapping_add(1);
                self.buffer.pop_front();
            } else {
                break;
            }
        }

        delivered
    }

    /// Detects gaps in the sequence space.
    ///
    /// Returns sequence numbers that appear to be lost.
    #[must_use]
    pub fn detect_gaps(&self) -> Vec<u32> {
        let mut gaps = Vec::new();

        if self.buffer.is_empty() {
            return gaps;
        }

        let first_buffered = self.buffer.front().map(|p| p.seq).unwrap_or(0);

        // Check gap between expected and first buffered
        let mut seq = self.expected_seq;
        while seq != first_buffered && gaps.len() < 1000 {
            gaps.push(seq);
            seq = seq.wrapping_add(1);
        }

        // Check gaps between buffered packets
        for window in self.buffer.iter().collect::<Vec<_>>().windows(2) {
            let curr = window[0].seq;
            let next = window[1].seq;

            let mut s = curr.wrapping_add(1);
            while s != next && gaps.len() < 1000 {
                gaps.push(s);
                s = s.wrapping_add(1);
            }
        }

        gaps
    }

    /// Returns the number of buffered packets.
    #[must_use]
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    /// Clears all buffered packets.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Checks if sequence a is after sequence b (with wraparound).
const fn seq_after(a: u32, b: u32) -> bool {
    let diff = a.wrapping_sub(b);
    diff > 0 && diff < 0x8000_0000
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[test]
    fn test_loss_list_add_remove() {
        let mut list = LossList::new(100);
        list.add(10);
        list.add(20);
        list.add(30);

        assert_eq!(list.len(), 3);
        assert!(list.contains(10));
        assert!(list.contains(20));
        assert!(!list.contains(15));

        assert!(list.remove(20));
        assert_eq!(list.len(), 2);
        assert!(!list.contains(20));
    }

    #[test]
    fn test_loss_list_range() {
        let mut list = LossList::new(100);
        list.add_range(10, 15);

        assert_eq!(list.len(), 5);
        assert!(list.contains(10));
        assert!(list.contains(14));
        assert!(!list.contains(15));
    }

    #[test]
    fn test_loss_list_compressed_ranges() {
        let mut list = LossList::new(100);
        list.add(10);
        list.add(11);
        list.add(12);
        list.add(20);
        list.add(21);

        let ranges = list.compressed_ranges();
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], LossRange { start: 10, end: 12 });
        assert_eq!(ranges[1], LossRange { start: 20, end: 21 });
    }

    #[test]
    fn test_loss_range_count() {
        let range = LossRange { start: 10, end: 14 };
        assert_eq!(range.count(), 5);

        let single = LossRange::single(42);
        assert_eq!(single.count(), 1);
        assert!(single.is_single());
    }

    #[test]
    fn test_receive_buffer_in_order() {
        let mut buf = ReceiveBuffer::new(100, 50);

        assert_eq!(buf.process(100, Bytes::from(vec![1])), Some(100));
        assert_eq!(buf.expected_seq(), 101);
        assert_eq!(buf.process(101, Bytes::from(vec![2])), Some(101));
        assert_eq!(buf.expected_seq(), 102);
    }

    #[test]
    fn test_receive_buffer_out_of_order() {
        let mut buf = ReceiveBuffer::new(100, 50);

        // Receive packet 102 before 100
        assert_eq!(buf.process(102, Bytes::from(vec![3])), None);
        assert_eq!(buf.buffered_count(), 1);

        // Receive packet 100
        assert_eq!(buf.process(100, Bytes::from(vec![1])), Some(100));

        // Now 101 is expected, 102 is still buffered
        assert_eq!(buf.process(101, Bytes::from(vec![2])), Some(101));

        // Should deliver buffered 102
        let delivered = buf.try_deliver();
        assert_eq!(delivered, vec![102]);
        assert_eq!(buf.expected_seq(), 103);
    }

    #[test]
    fn test_receive_buffer_detect_gaps() {
        let mut buf = ReceiveBuffer::new(100, 50);

        buf.process(105, Bytes::from(vec![1]));
        buf.process(106, Bytes::from(vec![2]));

        let gaps = buf.detect_gaps();
        assert_eq!(gaps, vec![100, 101, 102, 103, 104]);
    }

    #[test]
    fn test_seq_after() {
        assert!(seq_after(10, 5));
        assert!(!seq_after(5, 10));
        assert!(!seq_after(5, 5));

        // Wraparound
        assert!(seq_after(0, 0xFFFF_FFFF));
        assert!(seq_after(10, 0xFFFF_FFF0));
    }

    #[test]
    fn test_loss_list_oldest() {
        let mut list = LossList::new(100);
        list.add(30);
        list.add(10);
        list.add(20);

        assert_eq!(list.oldest(), Some(10));
    }

    // ── New NAK range-compression tests ─────────────────────────────────────

    /// `add_range(10, 20)` uses an exclusive end; adds seqs 10–19.
    /// `compressed_ranges()` should yield exactly one range with start=10, end=19.
    #[test]
    fn test_loss_contiguous_run_is_one_range() {
        let mut list = LossList::new(1000);
        list.add_range(10, 20); // exclusive end → inserts 10..=19
        let ranges = list.compressed_ranges();
        assert_eq!(
            ranges.len(),
            1,
            "contiguous run must compress to a single range"
        );
        assert_eq!(ranges[0].start, 10);
        assert_eq!(ranges[0].end, 19);
    }

    /// Non-adjacent singletons [1, 3, 5, 7] each form their own range.
    #[test]
    fn test_loss_scattered_singletons_are_n_ranges() {
        let mut list = LossList::new(1000);
        for seq in [1u32, 3, 5, 7] {
            list.add(seq);
        }
        let ranges = list.compressed_ranges();
        assert_eq!(
            ranges.len(),
            4,
            "four non-adjacent singletons must produce 4 ranges"
        );
    }

    /// `add_range(1, 6)` inserts 1–5; `add_range(6, 11)` inserts 6–10.
    /// In the BTreeSet, 6 == 5.wrapping_add(1) so they merge into [1, 10].
    #[test]
    fn test_loss_adjacent_ranges_merge() {
        let mut list = LossList::new(1000);
        list.add_range(1, 6); // inserts 1..=5
        list.add_range(6, 11); // inserts 6..=10; 6 is adjacent to 5
        let ranges = list.compressed_ranges();
        assert_eq!(
            ranges.len(),
            1,
            "adjacent ranges must merge into one; got: {:?}",
            ranges
        );
        assert_eq!(ranges[0].start, 1);
        assert_eq!(ranges[0].end, 10);
    }

    /// After `remove(12)`, `contains(12)` must be false.
    #[test]
    fn test_loss_remove_shrinks_range() {
        let mut list = LossList::new(1000);
        list.add_range(10, 16); // inserts 10..=15
        assert!(list.contains(12), "12 should be present before remove");
        let removed = list.remove(12);
        assert!(removed, "remove should return true for an existing entry");
        assert!(!list.contains(12), "12 must not be present after remove");
        // Surrounding entries must still be present.
        assert!(list.contains(11));
        assert!(list.contains(13));
    }

    /// With max_entries=3, adding 5 entries only retains the first 3.
    #[test]
    fn test_loss_max_entries_cap() {
        let mut list = LossList::new(3);
        for seq in [1u32, 2, 3, 4, 5] {
            list.add(seq);
        }
        assert!(
            list.len() <= 3,
            "len() must not exceed max_entries; got {}",
            list.len()
        );
    }

    /// `oldest()` must return the minimum (not insertion-order oldest) seq number.
    #[test]
    fn test_loss_oldest_correct() {
        let mut list = LossList::new(1000);
        list.add(100);
        list.add(50);
        list.add(200);
        assert_eq!(
            list.oldest(),
            Some(50),
            "oldest() must return the minimum seq number"
        );
    }
}
