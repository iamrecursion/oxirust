//! RTP packet parsing (RFC 3550 §5.1).
//!
//! ```text
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |V=2|P|X|  CC   |M|     PT      |       sequence number         |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                           timestamp                           |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |           synchronization source (SSRC) identifier            |
//! +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
//! |            contributing source (CSRC) identifiers             |
//! |                             ....                              |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```
//!
//! Only the parts needed to demultiplex an RTSP-delivered RTP stream are
//! parsed here. Reassembly of codec-specific NAL units / AU frames is the
//! caller's job — that logic lives in the codec depacketizers.

use crate::error::NetError;

/// Minimum RTP header size (no CSRCs, no extension).
pub const RTP_HEADER_MIN: usize = 12;

/// Parsed RTP packet header plus a borrowed payload slice.
#[derive(Debug, Clone, Copy)]
pub struct RtpPacket<'a> {
    /// Protocol version. Always 2 for RFC 3550.
    pub version: u8,
    /// Padding flag — when set, the last payload byte gives the padding length.
    pub padding: bool,
    /// Extension header present.
    pub extension: bool,
    /// Marker bit — codec-specific (end of frame, start of talk-spurt, etc.).
    pub marker: bool,
    /// Payload type (0–127).
    pub payload_type: u8,
    /// 16-bit packet sequence number, big-endian on the wire.
    pub sequence: u16,
    /// 32-bit RTP timestamp.
    pub timestamp: u32,
    /// Synchronization source identifier.
    pub ssrc: u32,
    /// Slice of payload bytes (after CSRCs, extension, and trailing padding).
    pub payload: &'a [u8],
}

impl<'a> RtpPacket<'a> {
    /// Parse an RTP packet from `buf`.
    ///
    /// The returned `payload` slice borrows from `buf`.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Protocol`] if the buffer is too short, the
    /// version is not 2, or the declared CSRC/extension/padding lengths
    /// don't fit.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::RtpPacket;
    ///
    /// // Minimal RTP packet: V=2, M=1, PT=96, seq=1, ts=0, ssrc=0, payload=b"X"
    /// let mut buf = vec![0x80, 0x60 | 0x80, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
    /// buf.extend_from_slice(b"X");
    /// let pkt = RtpPacket::parse(&buf).unwrap();
    /// assert_eq!(pkt.payload_type, 96);
    /// assert!(pkt.marker);
    /// assert_eq!(pkt.payload, b"X");
    /// ```
    pub fn parse(buf: &'a [u8]) -> Result<Self, NetError> {
        if buf.len() < RTP_HEADER_MIN {
            return Err(NetError::Protocol(format!(
                "RTP packet too small: {} < {}",
                buf.len(),
                RTP_HEADER_MIN
            )));
        }

        let b0 = buf[0];
        let version = b0 >> 6;
        if version != 2 {
            return Err(NetError::Protocol(format!(
                "unsupported RTP version: {version}"
            )));
        }
        let padding = (b0 & 0x20) != 0;
        let extension = (b0 & 0x10) != 0;
        let csrc_count = (b0 & 0x0F) as usize;

        let b1 = buf[1];
        let marker = (b1 & 0x80) != 0;
        let payload_type = b1 & 0x7F;

        let sequence = u16::from_be_bytes([buf[2], buf[3]]);
        let timestamp = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let ssrc = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);

        let csrc_bytes = csrc_count * 4;
        let mut offset = RTP_HEADER_MIN + csrc_bytes;
        if buf.len() < offset {
            return Err(NetError::Protocol(format!(
                "RTP CSRC list truncated (need {offset}, have {})",
                buf.len()
            )));
        }

        if extension {
            // Extension is: 16-bit profile + 16-bit length in 32-bit words.
            if buf.len() < offset + 4 {
                return Err(NetError::Protocol("RTP extension header truncated".into()));
            }
            let ext_len_words = u16::from_be_bytes([buf[offset + 2], buf[offset + 3]]) as usize;
            offset += 4 + ext_len_words * 4;
            if buf.len() < offset {
                return Err(NetError::Protocol("RTP extension body truncated".into()));
            }
        }

        let mut end = buf.len();
        if padding {
            if end <= offset {
                return Err(NetError::Protocol(
                    "RTP padding with no payload room".into(),
                ));
            }
            let pad_len = buf[end - 1] as usize;
            if pad_len == 0 || end < offset + pad_len {
                return Err(NetError::Protocol(format!(
                    "RTP padding length {pad_len} invalid"
                )));
            }
            end -= pad_len;
        }

        Ok(Self {
            version,
            padding,
            extension,
            marker,
            payload_type,
            sequence,
            timestamp,
            ssrc,
            payload: &buf[offset..end],
        })
    }
}

/// Builds outbound RTP packets with automatic sequence number management.
///
/// Produces minimal RTP headers: `V=2, P=0, X=0, CC=0`, with a 12-byte fixed header.
///
/// # Example
///
/// ```
/// use oximedia_net::rtsp::{RtpPacket, rtp::RtpPacketBuilder};
/// let mut builder = RtpPacketBuilder::new(0xDEAD_BEEF, 96);
/// let raw = builder.build(b"hello");
/// let pkt = RtpPacket::parse(&raw).unwrap();
/// assert_eq!(pkt.ssrc, 0xDEAD_BEEF);
/// assert_eq!(pkt.payload_type, 96);
/// assert_eq!(pkt.payload, b"hello");
/// assert_eq!(pkt.sequence, 0); // first packet uses seq=0
/// ```
#[derive(Debug, Clone)]
pub struct RtpPacketBuilder {
    /// Synchronization source identifier — uniquely identifies this stream source.
    pub ssrc: u32,
    /// Payload type (0–127).
    pub payload_type: u8,
    /// Current sequence number; incremented on each call to [`build`](Self::build).
    /// Exposed for testing/state inspection; prefer [`next_sequence`](Self::next_sequence).
    pub sequence: u16,
    /// Current RTP timestamp; callers must advance this per-codec.
    pub timestamp: u32,
    /// Marker bit value used by [`build`](Self::build).
    pub marker: bool,
}

impl RtpPacketBuilder {
    /// Create a new builder with `ssrc` and `payload_type`.
    /// Sequence starts at 0 and increments before each packet is emitted.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::rtp::RtpPacketBuilder;
    /// let b = RtpPacketBuilder::new(1, 96);
    /// assert_eq!(b.ssrc, 1);
    /// assert_eq!(b.payload_type, 96);
    /// ```
    #[must_use]
    pub fn new(ssrc: u32, payload_type: u8) -> Self {
        Self {
            ssrc,
            payload_type,
            sequence: u16::MAX, // first next_sequence() call wraps to 0
            timestamp: 0,
            marker: false,
        }
    }

    /// Override the timestamp before building.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::rtp::RtpPacketBuilder;
    /// let b = RtpPacketBuilder::new(1, 96).with_timestamp(90000);
    /// assert_eq!(b.timestamp, 90000);
    /// ```
    #[must_use]
    pub fn with_timestamp(mut self, ts: u32) -> Self {
        self.timestamp = ts;
        self
    }

    /// Increment and return the sequence number (wraps at 2^16).
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::rtp::RtpPacketBuilder;
    /// let mut b = RtpPacketBuilder::new(1, 96);
    /// assert_eq!(b.next_sequence(), 0); // 0xFFFF + 1 wraps to 0
    /// assert_eq!(b.next_sequence(), 1);
    /// ```
    pub fn next_sequence(&mut self) -> u16 {
        self.sequence = self.sequence.wrapping_add(1);
        self.sequence
    }

    /// Build one RTP packet from `payload` bytes.
    ///
    /// Increments the sequence number. Uses `self.marker` and `self.timestamp`.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::{RtpPacket, rtp::RtpPacketBuilder};
    /// let mut b = RtpPacketBuilder::new(42, 96);
    /// let raw = b.build(b"frame");
    /// let pkt = RtpPacket::parse(&raw).unwrap();
    /// assert_eq!(pkt.payload, b"frame");
    /// assert_eq!(pkt.ssrc, 42);
    /// ```
    #[must_use]
    pub fn build(&mut self, payload: &[u8]) -> Vec<u8> {
        self.build_with_marker(payload, self.marker)
    }

    /// Build one RTP packet, overriding the marker bit.
    ///
    /// Increments the sequence number. Does NOT update `self.marker`.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::{RtpPacket, rtp::RtpPacketBuilder};
    /// let mut b = RtpPacketBuilder::new(1, 96);
    /// let raw = b.build_with_marker(b"data", true);
    /// let pkt = RtpPacket::parse(&raw).unwrap();
    /// assert!(pkt.marker);
    /// ```
    #[must_use]
    pub fn build_with_marker(&mut self, payload: &[u8], marker: bool) -> Vec<u8> {
        let seq = self.next_sequence();
        let mut out = Vec::with_capacity(RTP_HEADER_MIN + payload.len());
        // Byte 0: V=2(10), P=0, X=0, CC=0 → 0x80
        out.push(0x80);
        // Byte 1: M | PT
        let pt_byte = if marker {
            0x80 | (self.payload_type & 0x7F)
        } else {
            self.payload_type & 0x7F
        };
        out.push(pt_byte);
        // Bytes 2–3: sequence number (big-endian)
        out.extend_from_slice(&seq.to_be_bytes());
        // Bytes 4–7: timestamp (big-endian)
        out.extend_from_slice(&self.timestamp.to_be_bytes());
        // Bytes 8–11: SSRC (big-endian)
        out.extend_from_slice(&self.ssrc.to_be_bytes());
        // Payload
        out.extend_from_slice(payload);
        out
    }
}

/// Tracks sequence numbers and detects loss / reordering on a single RTP stream.
///
/// 16-bit sequence numbers wrap roughly every minute on busy video flows, so
/// the comparison is done modulo 2^16 with a signed delta. A delta > 0 means
/// the new packet is later; a delta < 0 means reorder; delta == 0 is duplicate.
#[derive(Debug, Default)]
pub struct SequenceTracker {
    last_seq: Option<u16>,
    /// Total packets observed (including dups and reorders).
    pub received: u64,
    /// Packets that arrived earlier than the highest seq we've seen (reorder).
    pub reordered: u64,
    /// Packets with the same seq as the previous packet (duplicate).
    pub duplicates: u64,
    /// Inferred losses (gaps between consecutive in-order packets).
    pub lost: u64,
}

impl SequenceTracker {
    /// Empty tracker.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::SequenceTracker;
    /// let t = SequenceTracker::new();
    /// assert_eq!(t.received, 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a sequence number; returns the signed delta relative to the
    /// previously highest-seen value (None on the first packet).
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::SequenceTracker;
    /// let mut t = SequenceTracker::new();
    /// assert_eq!(t.observe(100), None);     // first packet, no delta
    /// assert_eq!(t.observe(101), Some(1));  // next packet, in order
    /// assert_eq!(t.observe(105), Some(4));  // 3-packet gap
    /// assert_eq!(t.lost, 3);
    /// ```
    pub fn observe(&mut self, seq: u16) -> Option<i32> {
        self.received += 1;
        match self.last_seq {
            None => {
                self.last_seq = Some(seq);
                None
            }
            Some(prev) => {
                let delta = signed_seq_delta(prev, seq);
                if delta == 0 {
                    self.duplicates += 1;
                } else if delta < 0 {
                    self.reordered += 1;
                } else {
                    if delta > 1 {
                        self.lost += (delta as u64) - 1;
                    }
                    self.last_seq = Some(seq);
                }
                Some(delta)
            }
        }
    }
}

/// Returns `new - prev` interpreted as a signed delta modulo 2^16.
///
/// Sequence numbers wrap, so a raw subtraction would be wrong across the
/// boundary. This treats deltas > 32_768 as negative (i.e. the new value
/// represents an earlier sequence).
fn signed_seq_delta(prev: u16, new: u16) -> i32 {
    let diff = new.wrapping_sub(prev) as i32;
    if diff > 32_768 {
        diff - 65_536
    } else {
        diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_packet(seq: u16, ts: u32, pt: u8, marker: bool, payload: &[u8]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + payload.len());
        buf.push(0x80); // V=2, P=0, X=0, CC=0
        buf.push((u8::from(marker) << 7) | (pt & 0x7F));
        buf.extend_from_slice(&seq.to_be_bytes());
        buf.extend_from_slice(&ts.to_be_bytes());
        buf.extend_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    #[test]
    fn parses_minimal_packet() {
        let raw = minimal_packet(1234, 90_000, 96, true, b"abcd");
        let pkt = RtpPacket::parse(&raw).unwrap();
        assert_eq!(pkt.version, 2);
        assert!(pkt.marker);
        assert_eq!(pkt.payload_type, 96);
        assert_eq!(pkt.sequence, 1234);
        assert_eq!(pkt.timestamp, 90_000);
        assert_eq!(pkt.ssrc, 0xDEAD_BEEF);
        assert_eq!(pkt.payload, b"abcd");
    }

    #[test]
    fn rejects_short_buffer() {
        assert!(RtpPacket::parse(&[0u8; 4]).is_err());
    }

    #[test]
    fn rejects_wrong_version() {
        let mut raw = minimal_packet(1, 0, 0, false, b"");
        raw[0] = 0x40; // V=1
        assert!(RtpPacket::parse(&raw).is_err());
    }

    #[test]
    fn parses_with_csrcs() {
        let mut raw = vec![0x82, 0x60, 0x00, 0x01, 0, 0, 0, 0, 0, 0, 0, 0]; // CC=2
        raw.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 2]);
        raw.extend_from_slice(b"data");
        let pkt = RtpPacket::parse(&raw).unwrap();
        assert_eq!(pkt.payload, b"data");
    }

    #[test]
    fn parses_with_extension() {
        // V=2, X=1, CC=0; PT=96; seq=1; ts=0; ssrc=0; ext_profile=0xBEDE; ext_len=1; ext word
        let mut raw = vec![0x90, 0x60, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        raw.extend_from_slice(&[0xBE, 0xDE, 0x00, 0x01]);
        raw.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        raw.extend_from_slice(b"payload");
        let pkt = RtpPacket::parse(&raw).unwrap();
        assert!(pkt.extension);
        assert_eq!(pkt.payload, b"payload");
    }

    #[test]
    fn parses_with_padding() {
        // 4-byte payload + 3 bytes padding + length byte = pad_len=4
        let mut raw = vec![0xA0, 0x60, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        raw.extend_from_slice(b"DATA"); // real payload
        raw.extend_from_slice(&[0, 0, 0, 4]); // 3 padding bytes + length=4
        let pkt = RtpPacket::parse(&raw).unwrap();
        assert!(pkt.padding);
        assert_eq!(pkt.payload, b"DATA");
    }

    #[test]
    fn sequence_tracker_detects_gap() {
        let mut t = SequenceTracker::new();
        assert_eq!(t.observe(100), None);
        assert_eq!(t.observe(101), Some(1));
        assert_eq!(t.observe(105), Some(4));
        assert_eq!(t.lost, 3);
    }

    #[test]
    fn sequence_tracker_detects_reorder() {
        let mut t = SequenceTracker::new();
        t.observe(100);
        t.observe(102);
        assert_eq!(t.observe(101), Some(-1));
        assert_eq!(t.reordered, 1);
    }

    #[test]
    fn sequence_tracker_detects_duplicate() {
        let mut t = SequenceTracker::new();
        t.observe(100);
        assert_eq!(t.observe(100), Some(0));
        assert_eq!(t.duplicates, 1);
    }

    #[test]
    fn sequence_tracker_handles_wrap() {
        let mut t = SequenceTracker::new();
        t.observe(65_534);
        // 65_535 → +1
        assert_eq!(t.observe(65_535), Some(1));
        // 0 wraps from 65_535 → +1
        assert_eq!(t.observe(0), Some(1));
        // 1 → +1
        assert_eq!(t.observe(1), Some(1));
        assert_eq!(t.lost, 0);
    }
}
