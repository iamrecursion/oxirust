//! SMPTE ST 2110-20 uncompressed video transport over RTP/UDP.
//!
//! Implements the ST 2110-20 standard for transporting uncompressed video
//! (YCbCr 4:2:2 10-bit, etc.) as RTP packets over IP networks.
//!
//! # Packet Structure
//!
//! Each RTP packet carries one or more rows of video data. A row extension
//! header identifies which video line(s) are contained, the pixel offset,
//! and whether continuation packets follow for the same lines.
//!
//! # Reference
//! SMPTE ST 2110-20:2022 — Uncompressed Active Video

#![allow(dead_code)]

use std::collections::HashMap;

use crate::error::{VideoIpError, VideoIpResult};

/// Maximum payload size per ST 2110-20 packet (bytes).
/// Leaves room for IP (20) + UDP (8) + RTP (12) + row extension (6 per line) headers
/// in a standard 1500-byte Ethernet MTU.
pub const MAX_PAYLOAD_BYTES: usize = 1400;

/// Bytes per 2 pixels for YCbCr 4:2:2 8-bit.
pub const YUV422_8BIT_BYTES_PER_2PIX: usize = 4;

// ─── RTP-like packet ────────────────────────────────────────────────────────

/// A single ST 2110-20 transport packet.
///
/// Carries a slice of one video line (or a portion thereof). Multiple packets
/// are typically needed to transport a complete video frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct St2110_20Packet {
    /// RTP sequence number (wraps at 65535).
    pub seq_num: u16,
    /// RTP timestamp (90 kHz clock or frame-rate derived).
    pub timestamp: u32,
    /// Video line number (0-based).
    pub line_num: u16,
    /// Pixel offset within the line where this payload begins.
    pub offset: u16,
    /// When `true`, more packets for the same line/timestamp follow.
    pub continuation: bool,
    /// Raw YCbCr payload bytes.
    pub payload: Vec<u8>,
}

impl St2110_20Packet {
    /// Serialises the packet to bytes.
    ///
    /// Layout (big-endian):
    /// ```text
    /// [0..2]  seq_num
    /// [2..6]  timestamp
    /// [6..8]  line_num
    /// [8..10] offset
    /// [10]    flags: bit7 = continuation
    /// [11]    reserved
    /// [12..]  payload
    /// ```
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + self.payload.len());
        out.extend_from_slice(&self.seq_num.to_be_bytes());
        out.extend_from_slice(&self.timestamp.to_be_bytes());
        out.extend_from_slice(&self.line_num.to_be_bytes());
        out.extend_from_slice(&self.offset.to_be_bytes());
        let flags: u8 = if self.continuation { 0x80 } else { 0x00 };
        out.push(flags);
        out.push(0x00); // reserved
        out.extend_from_slice(&self.payload);
        out
    }

    /// Deserialises a packet from bytes.
    ///
    /// Returns `None` if the buffer is too short (< 12 bytes).
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }
        let seq_num = u16::from_be_bytes([data[0], data[1]]);
        let timestamp = u32::from_be_bytes([data[2], data[3], data[4], data[5]]);
        let line_num = u16::from_be_bytes([data[6], data[7]]);
        let offset = u16::from_be_bytes([data[8], data[9]]);
        let continuation = (data[10] & 0x80) != 0;
        let payload = data[12..].to_vec();
        Some(Self {
            seq_num,
            timestamp,
            line_num,
            offset,
            continuation,
            payload,
        })
    }
}

// ─── Sender ─────────────────────────────────────────────────────────────────

/// ST 2110-20 frame sender — packetises a raw YCbCr 4:2:2 frame.
#[derive(Debug, Clone)]
pub struct St2110_20Sender {
    /// Timestamp increment per frame (depends on clock rate and frame rate).
    pub timestamp_increment: u32,
    /// SSRC / stream identifier (stored in the `timestamp` field for simulation).
    next_seq: u16,
    next_timestamp: u32,
}

impl St2110_20Sender {
    /// Creates a new sender.
    ///
    /// `timestamp_increment` is typically `90_000 / fps` for a 90 kHz clock.
    #[must_use]
    pub fn new(timestamp_increment: u32) -> Self {
        Self {
            timestamp_increment,
            next_seq: 0,
            next_timestamp: 0,
        }
    }

    /// Creates a sender pre-configured for a common frame rate.
    ///
    /// Uses a 90 kHz RTP clock. For 29.97 fps the increment is 3003 (rounded
    /// from 90 000 / 29.97).
    #[must_use]
    pub fn for_fps(fps_num: u32, fps_den: u32) -> Self {
        let increment = (90_000u64 * u64::from(fps_den)) / u64::from(fps_num);
        Self::new(increment as u32)
    }

    /// Packetises one raw YCbCr 4:2:2 frame into ST 2110-20 packets.
    ///
    /// `yuv422` must be exactly `width * height * 2` bytes (UYVY packing,
    /// 2 bytes per pixel). Returns an error if the buffer length is wrong.
    pub fn send_frame(
        &mut self,
        yuv422: &[u8],
        width: u32,
        height: u32,
    ) -> VideoIpResult<Vec<St2110_20Packet>> {
        let expected = (width as usize) * (height as usize) * 2;
        if yuv422.len() != expected {
            return Err(VideoIpError::InvalidPacket(format!(
                "expected {expected} bytes for {width}×{height} YUV422, got {}",
                yuv422.len()
            )));
        }

        let bytes_per_line = (width as usize) * 2; // UYVY: 2 bytes/pixel
        let ts = self.next_timestamp;
        self.next_timestamp = self.next_timestamp.wrapping_add(self.timestamp_increment);

        let mut packets: Vec<St2110_20Packet> = Vec::new();

        for line in 0..height {
            let line_start = (line as usize) * bytes_per_line;
            let line_data = &yuv422[line_start..line_start + bytes_per_line];

            // Split a single line into MTU-sized chunks
            let chunks: Vec<&[u8]> = line_data.chunks(MAX_PAYLOAD_BYTES).collect();
            let num_chunks = chunks.len();

            for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
                // pixel offset = byte_offset / 2 (each pixel = 2 bytes in UYVY)
                let byte_offset = chunk_idx * MAX_PAYLOAD_BYTES;
                let pixel_offset = (byte_offset / 2) as u16;

                // continuation = there are more chunks for this line, OR more lines follow
                let more_chunks = chunk_idx + 1 < num_chunks;
                let more_lines = line + 1 < height;
                let continuation = more_chunks || more_lines;

                let pkt = St2110_20Packet {
                    seq_num: self.next_seq,
                    timestamp: ts,
                    line_num: line as u16,
                    offset: pixel_offset,
                    continuation,
                    payload: chunk.to_vec(),
                };

                self.next_seq = self.next_seq.wrapping_add(1);
                packets.push(pkt);
            }
        }

        Ok(packets)
    }
}

// ─── Receiver ───────────────────────────────────────────────────────────────

/// Accumulation buffer for one frame being reassembled.
#[derive(Debug, Default)]
struct FrameBuffer {
    /// Collected line data: line_num → reassembled bytes for that line.
    lines: HashMap<u16, Vec<u8>>,
    /// Total number of lines expected (set on first packet).
    expected_lines: Option<u16>,
    /// Width in pixels (derived from first line).
    width_pixels: Option<u16>,
}

impl FrameBuffer {
    fn insert_chunk(&mut self, line_num: u16, offset: u16, payload: &[u8]) {
        let byte_offset = (offset as usize) * 2;
        let entry = self.lines.entry(line_num).or_default();

        let needed = byte_offset + payload.len();
        if entry.len() < needed {
            entry.resize(needed, 0u8);
        }
        entry[byte_offset..byte_offset + payload.len()].copy_from_slice(payload);
    }

    /// Returns `true` when all expected lines have been received.
    fn is_complete(&self) -> bool {
        match (self.expected_lines, self.width_pixels) {
            (Some(h), Some(w)) => {
                let bytes_per_line = (w as usize) * 2;
                (0..h).all(|l| {
                    self.lines
                        .get(&l)
                        .map_or(false, |v| v.len() == bytes_per_line)
                })
            }
            _ => false,
        }
    }

    /// Assembles the complete frame in scan-line order.
    fn assemble(&self) -> Option<Vec<u8>> {
        let height = self.expected_lines?;
        let width = self.width_pixels?;
        let bytes_per_line = (width as usize) * 2;
        let mut out = Vec::with_capacity((height as usize) * bytes_per_line);
        for l in 0..height {
            let line = self.lines.get(&l)?;
            if line.len() != bytes_per_line {
                return None;
            }
            out.extend_from_slice(line);
        }
        Some(out)
    }
}

/// ST 2110-20 frame receiver — reassembles packets into complete frames.
#[derive(Debug, Default)]
pub struct St2110_20Receiver {
    /// Per-timestamp accumulation buffers.
    buffers: HashMap<u32, FrameBuffer>,
    /// Frame dimensions provided at construction time.
    width: u16,
    height: u16,
}

impl St2110_20Receiver {
    /// Creates a new receiver for frames of the given dimensions.
    #[must_use]
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            buffers: HashMap::new(),
            width,
            height,
        }
    }

    /// Feeds one packet into the receiver.
    ///
    /// Returns `Some(frame_bytes)` when a complete frame has been assembled,
    /// `None` otherwise.
    pub fn receive_packet(&mut self, pkt: &St2110_20Packet) -> Option<Vec<u8>> {
        let buf = self.buffers.entry(pkt.timestamp).or_default();

        // Set dimensions on first packet for this timestamp
        if buf.expected_lines.is_none() {
            buf.expected_lines = Some(self.height);
            buf.width_pixels = Some(self.width);
        }

        buf.insert_chunk(pkt.line_num, pkt.offset, &pkt.payload);

        if buf.is_complete() {
            let frame = buf.assemble();
            self.buffers.remove(&pkt.timestamp);
            frame
        } else {
            None
        }
    }

    /// Returns the number of incomplete frame buffers currently held.
    #[must_use]
    pub fn pending_frames(&self) -> usize {
        self.buffers.len()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: u32, height: u32) -> Vec<u8> {
        // Simple ramp pattern: byte value = (pixel index) % 256
        let len = (width * height * 2) as usize;
        (0..len).map(|i| (i % 256) as u8).collect()
    }

    // ── Packet serialisation ──────────────────────────────────────────────

    #[test]
    fn test_packet_roundtrip_serialisation() {
        let pkt = St2110_20Packet {
            seq_num: 0xABCD,
            timestamp: 0x1234_5678,
            line_num: 42,
            offset: 100,
            continuation: true,
            payload: vec![1, 2, 3, 4, 5],
        };
        let bytes = pkt.to_bytes();
        let decoded = St2110_20Packet::from_bytes(&bytes).expect("decode must succeed");
        assert_eq!(decoded, pkt);
    }

    #[test]
    fn test_packet_continuation_flag_false() {
        let pkt = St2110_20Packet {
            seq_num: 1,
            timestamp: 2,
            line_num: 0,
            offset: 0,
            continuation: false,
            payload: vec![0xFF],
        };
        let bytes = pkt.to_bytes();
        let decoded = St2110_20Packet::from_bytes(&bytes).expect("decode");
        assert!(!decoded.continuation);
    }

    #[test]
    fn test_packet_from_bytes_too_short() {
        let data = [0u8; 5];
        assert!(St2110_20Packet::from_bytes(&data).is_none());
    }

    #[test]
    fn test_packet_serialised_length() {
        let pkt = St2110_20Packet {
            seq_num: 0,
            timestamp: 0,
            line_num: 0,
            offset: 0,
            continuation: false,
            payload: vec![0u8; 100],
        };
        assert_eq!(pkt.to_bytes().len(), 12 + 100);
    }

    #[test]
    fn test_packet_empty_payload() {
        let pkt = St2110_20Packet {
            seq_num: 7,
            timestamp: 99,
            line_num: 3,
            offset: 0,
            continuation: false,
            payload: vec![],
        };
        let bytes = pkt.to_bytes();
        assert_eq!(bytes.len(), 12);
        let decoded = St2110_20Packet::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.seq_num, 7);
        assert!(decoded.payload.is_empty());
    }

    // ── Sender ────────────────────────────────────────────────────────────

    #[test]
    fn test_sender_small_frame_packet_count() {
        // 4×2 frame → 2 lines × 8 bytes/line → each line fits in one packet
        let mut sender = St2110_20Sender::new(3000);
        let frame = make_frame(4, 2);
        let pkts = sender.send_frame(&frame, 4, 2).expect("send");
        assert_eq!(pkts.len(), 2, "one packet per line");
    }

    #[test]
    fn test_sender_wrong_buffer_size_error() {
        let mut sender = St2110_20Sender::new(3000);
        let frame = vec![0u8; 100]; // wrong size for 4×2
        assert!(sender.send_frame(&frame, 4, 2).is_err());
    }

    #[test]
    fn test_sender_sequence_numbers_increment() {
        let mut sender = St2110_20Sender::new(3000);
        let frame = make_frame(4, 4);
        let pkts = sender.send_frame(&frame, 4, 4).expect("send");
        for (i, pkt) in pkts.iter().enumerate() {
            assert_eq!(pkt.seq_num, i as u16);
        }
    }

    #[test]
    fn test_sender_timestamp_increments_each_frame() {
        let mut sender = St2110_20Sender::new(3000);
        let frame = make_frame(4, 2);
        let pkts1 = sender.send_frame(&frame, 4, 2).expect("frame 1");
        let pkts2 = sender.send_frame(&frame, 4, 2).expect("frame 2");
        assert_eq!(pkts1[0].timestamp + 3000, pkts2[0].timestamp);
    }

    #[test]
    fn test_sender_line_numbers_correct() {
        let mut sender = St2110_20Sender::new(3000);
        let frame = make_frame(4, 3);
        let pkts = sender.send_frame(&frame, 4, 3).expect("send");
        let line_nums: Vec<u16> = pkts.iter().map(|p| p.line_num).collect();
        assert_eq!(line_nums, vec![0, 1, 2]);
    }

    #[test]
    fn test_sender_last_packet_continuation_false() {
        let mut sender = St2110_20Sender::new(3000);
        let frame = make_frame(4, 2);
        let pkts = sender.send_frame(&frame, 4, 2).expect("send");
        assert!(!pkts.last().expect("last pkt").continuation);
    }

    #[test]
    fn test_sender_large_line_splits_into_multiple_packets() {
        // 1920 wide × 2 bytes = 3840 bytes/line → ceil(3840/1400) = 3 packets/line
        let mut sender = St2110_20Sender::new(3000);
        let frame = make_frame(1920, 1);
        let pkts = sender.send_frame(&frame, 1920, 1).expect("send");
        assert_eq!(pkts.len(), 3);
    }

    #[test]
    fn test_sender_pixel_offsets_in_split_packets() {
        let mut sender = St2110_20Sender::new(3000);
        let frame = make_frame(1920, 1);
        let pkts = sender.send_frame(&frame, 1920, 1).expect("send");
        assert_eq!(pkts[0].offset, 0);
        // chunk 1: byte offset 1400 → pixel offset 700
        assert_eq!(pkts[1].offset, 700);
        // chunk 2: byte offset 2800 → pixel offset 1400
        assert_eq!(pkts[2].offset, 1400);
    }

    // ── Receiver ──────────────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_small_frame() {
        let width: u32 = 4;
        let height: u32 = 2;
        let original = make_frame(width, height);

        let mut sender = St2110_20Sender::new(3000);
        let mut receiver = St2110_20Receiver::new(width as u16, height as u16);

        let pkts = sender.send_frame(&original, width, height).expect("send");

        let mut result: Option<Vec<u8>> = None;
        for pkt in &pkts {
            result = receiver.receive_packet(pkt);
        }
        let frame = result.expect("complete frame");
        assert_eq!(frame, original);
    }

    #[test]
    fn test_roundtrip_hd_frame() {
        let width: u32 = 1920;
        let height: u32 = 1080;
        let original = make_frame(width, height);

        let mut sender = St2110_20Sender::new(3000);
        let mut receiver = St2110_20Receiver::new(width as u16, height as u16);

        let pkts = sender.send_frame(&original, width, height).expect("send");
        let mut result: Option<Vec<u8>> = None;
        for pkt in &pkts {
            result = receiver.receive_packet(pkt);
        }
        assert_eq!(result.expect("complete frame"), original);
    }

    #[test]
    fn test_receiver_incomplete_frame_returns_none() {
        let mut receiver = St2110_20Receiver::new(4, 4);
        let pkt = St2110_20Packet {
            seq_num: 0,
            timestamp: 1000,
            line_num: 0,
            offset: 0,
            continuation: true,
            payload: vec![0u8; 8], // only one line of 4
        };
        assert!(receiver.receive_packet(&pkt).is_none());
        assert_eq!(receiver.pending_frames(), 1);
    }

    #[test]
    fn test_receiver_multiple_timestamps_independent() {
        let width: u32 = 4;
        let height: u32 = 2;
        let frame_a = make_frame(width, height);
        let frame_b: Vec<u8> = (0..(width * height * 2) as usize)
            .map(|i| ((i + 10) % 256) as u8)
            .collect();

        let mut sender_a = St2110_20Sender::new(3000);
        let mut sender_b = St2110_20Sender::new(3000);
        // Give sender_b a different starting timestamp to avoid collision
        sender_b.next_timestamp = 9000;

        let pkts_a = sender_a.send_frame(&frame_a, width, height).expect("a");
        let pkts_b = sender_b.send_frame(&frame_b, width, height).expect("b");

        let mut receiver = St2110_20Receiver::new(width as u16, height as u16);

        // Interleave delivery
        for pkt in pkts_a.iter().zip(pkts_b.iter()).flat_map(|(a, b)| [a, b]) {
            receiver.receive_packet(pkt);
        }

        // Feed remaining packets (if uneven)
        // (here lengths are equal, so nothing extra needed)

        // Both frames should eventually complete; re-run sequentially to verify isolation
        let mut receiver2 = St2110_20Receiver::new(width as u16, height as u16);
        let mut got_a = None;
        let mut got_b = None;
        for pkt in &pkts_a {
            got_a = receiver2.receive_packet(pkt).or(got_a);
        }
        for pkt in &pkts_b {
            got_b = receiver2.receive_packet(pkt).or(got_b);
        }
        assert_eq!(got_a.expect("frame A"), frame_a);
        assert_eq!(got_b.expect("frame B"), frame_b);
    }

    #[test]
    fn test_for_fps_30() {
        let sender = St2110_20Sender::for_fps(30, 1);
        assert_eq!(sender.timestamp_increment, 3000);
    }

    #[test]
    fn test_for_fps_25() {
        let sender = St2110_20Sender::for_fps(25, 1);
        assert_eq!(sender.timestamp_increment, 3600);
    }
}
