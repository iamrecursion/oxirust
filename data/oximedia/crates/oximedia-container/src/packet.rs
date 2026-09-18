//! Packet types for compressed media data.

use bitflags::bitflags;
use bytes::Bytes;
use oximedia_core::Timestamp;

bitflags! {
    /// Flags indicating packet properties.
    ///
    /// These flags provide information about the packet's role in the stream
    /// and whether it can be decoded independently.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PacketFlags: u32 {
        /// Packet contains a keyframe (can be decoded independently).
        ///
        /// For video, this typically indicates an I-frame.
        /// For audio, most packets are effectively keyframes.
        const KEYFRAME = 0x0001;

        /// Packet data may be corrupt.
        ///
        /// Set when the demuxer detects potential corruption but
        /// still provides the data for attempted recovery.
        const CORRUPT = 0x0002;

        /// Packet should be discarded.
        ///
        /// Used for packets that are part of the stream but should
        /// not be decoded (e.g., during seeking).
        const DISCARD = 0x0004;
    }
}

impl Default for PacketFlags {
    fn default() -> Self {
        Self::empty()
    }
}

/// A compressed media packet from a container.
///
/// Packets are the fundamental unit of compressed data in a container.
/// Each packet typically contains one or more compressed frames from
/// a single stream.
///
/// # Examples
///
/// ```
/// use oximedia_container::{Packet, PacketFlags};
/// use oximedia_core::{Timestamp, Rational};
/// use bytes::Bytes;
///
/// let packet = Packet::new(
///     0,
///     Bytes::from_static(&[0, 1, 2, 3]),
///     Timestamp::new(1000, Rational::new(1, 1000)),
///     PacketFlags::KEYFRAME,
/// );
///
/// assert!(packet.is_keyframe());
/// assert_eq!(packet.size(), 4);
/// ```
#[derive(Clone, Debug)]
pub struct Packet {
    /// Index of the stream this packet belongs to.
    pub stream_index: usize,

    /// Compressed packet data.
    pub data: Bytes,

    /// Presentation and decode timestamps.
    pub timestamp: Timestamp,

    /// Packet flags.
    pub flags: PacketFlags,
}

impl Packet {
    /// Creates a new packet.
    ///
    /// # Arguments
    ///
    /// * `stream_index` - Index of the stream this packet belongs to
    /// * `data` - Compressed packet data
    /// * `timestamp` - Presentation/decode timestamps
    /// * `flags` - Packet flags (keyframe, corrupt, etc.)
    #[must_use]
    pub const fn new(
        stream_index: usize,
        data: Bytes,
        timestamp: Timestamp,
        flags: PacketFlags,
    ) -> Self {
        Self {
            stream_index,
            data,
            timestamp,
            flags,
        }
    }

    /// Returns true if this packet is a keyframe.
    ///
    /// Keyframes can be decoded independently without reference to
    /// other frames in the stream.
    #[must_use]
    pub const fn is_keyframe(&self) -> bool {
        self.flags.contains(PacketFlags::KEYFRAME)
    }

    /// Returns true if this packet may be corrupt.
    #[must_use]
    pub const fn is_corrupt(&self) -> bool {
        self.flags.contains(PacketFlags::CORRUPT)
    }

    /// Returns true if this packet should be discarded.
    #[must_use]
    pub const fn should_discard(&self) -> bool {
        self.flags.contains(PacketFlags::DISCARD)
    }

    /// Returns the size of the packet data in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the packet data is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the presentation timestamp in the stream's timebase.
    #[must_use]
    pub const fn pts(&self) -> i64 {
        self.timestamp.pts
    }

    /// Returns the decode timestamp if available.
    #[must_use]
    pub const fn dts(&self) -> Option<i64> {
        self.timestamp.dts
    }

    /// Returns the packet duration if available.
    #[must_use]
    pub const fn duration(&self) -> Option<i64> {
        self.timestamp.duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    #[test]
    fn test_packet_new() {
        let data = Bytes::from_static(&[1, 2, 3, 4, 5]);
        let timestamp = Timestamp::new(1000, Rational::new(1, 1000));
        let packet = Packet::new(0, data, timestamp, PacketFlags::KEYFRAME);

        assert_eq!(packet.stream_index, 0);
        assert_eq!(packet.size(), 5);
        assert!(packet.is_keyframe());
        assert!(!packet.is_corrupt());
    }

    #[test]
    fn test_packet_flags() {
        let data = Bytes::new();
        let timestamp = Timestamp::new(0, Rational::new(1, 1000));

        let keyframe = Packet::new(0, data.clone(), timestamp, PacketFlags::KEYFRAME);
        assert!(keyframe.is_keyframe());
        assert!(!keyframe.is_corrupt());

        let corrupt = Packet::new(
            0,
            data.clone(),
            timestamp,
            PacketFlags::CORRUPT | PacketFlags::KEYFRAME,
        );
        assert!(corrupt.is_keyframe());
        assert!(corrupt.is_corrupt());

        let discard = Packet::new(0, data, timestamp, PacketFlags::DISCARD);
        assert!(discard.should_discard());
    }

    #[test]
    fn test_packet_timestamps() {
        let data = Bytes::new();
        let mut timestamp = Timestamp::new(1000, Rational::new(1, 48000));
        timestamp.dts = Some(999);
        timestamp.duration = Some(1024);

        let packet = Packet::new(0, data, timestamp, PacketFlags::empty());

        assert_eq!(packet.pts(), 1000);
        assert_eq!(packet.dts(), Some(999));
        assert_eq!(packet.duration(), Some(1024));
    }

    #[test]
    fn test_packet_is_empty() {
        let timestamp = Timestamp::new(0, Rational::new(1, 1000));

        let empty = Packet::new(0, Bytes::new(), timestamp, PacketFlags::empty());
        assert!(empty.is_empty());
        assert_eq!(empty.size(), 0);

        let non_empty = Packet::new(0, Bytes::from_static(&[1]), timestamp, PacketFlags::empty());
        assert!(!non_empty.is_empty());
        assert_eq!(non_empty.size(), 1);
    }
}
