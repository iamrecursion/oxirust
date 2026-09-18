//! RTMP Extended Protocol (RTMP+) — spec-aligned types for AV1/VP9/HEVC support.
//!
//! Implements the [Enhanced RTMP v2 specification](https://veovera.org/docs/enhanced/enhanced-rtmp-v2-spec)
//! using the canonical type names from that specification:
//!
//! - [`RtmpExtendedCodec`] — the three modern codecs RTMP+ supports beyond H.264
//! - [`ExVideoPacketType`] — packet classification (sequence start, coded frames, etc.)
//! - [`ExVideoHeader`] — the 5-byte extended video tag header (1 byte flag/type + 4-byte FourCC)
//! - [`RtmpExtendedPacket`] — a complete extended video packet (header + payload)
//!
//! # Wire format (ExVideoHeader)
//!
//! ```text
//! byte 0: [frame_type:4 | is_ex_header:1 | packet_type:3]
//!           is_ex_header is always 1 for extended packets (bit 3 set)
//! bytes 1-4: FourCC codec identifier
//! ```
//!
//! For `CodedFrames` packets there are an additional 3 bytes for the
//! composition-time offset *before* the bitstream payload.
//!
//! # Example
//!
//! ```rust
//! use oximedia_net::rtmp::rtmp_ext::{RtmpExtendedCodec, RtmpExtendedPacket};
//!
//! let pkt = RtmpExtendedPacket::new_av1_sequence(vec![0x81, 0x00, 0x0C]);
//! let bytes = pkt.serialize();
//! let decoded = RtmpExtendedPacket::parse(&bytes).expect("roundtrip");
//! assert!(matches!(decoded.header.codec, RtmpExtendedCodec::Av1));
//! ```

use crate::error::NetError;

// ─── Codec ────────────────────────────────────────────────────────────────────

/// Modern video codecs supported by the Enhanced RTMP (RTMP+) specification.
///
/// These correspond to the FourCC identifiers defined in
/// <https://veovera.org/docs/enhanced/enhanced-rtmp-v2-spec>.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RtmpExtendedCodec {
    /// AV1 — AOMedia Video 1, patent-free next-generation codec (FourCC `av01`).
    Av1,
    /// VP9 — Google's royalty-free codec (FourCC `vp09`).
    Vp9,
    /// HEVC — High Efficiency Video Coding / H.265 (FourCC `hvc1`).
    Hevc,
}

impl RtmpExtendedCodec {
    /// Returns the 4-byte FourCC for this codec.
    #[must_use]
    pub const fn fourcc(&self) -> [u8; 4] {
        match self {
            Self::Av1 => *b"av01",
            Self::Vp9 => *b"vp09",
            Self::Hevc => *b"hvc1",
        }
    }

    /// Returns the human-readable codec name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
            Self::Hevc => "HEVC",
        }
    }

    /// Returns `true` for codecs that are patent-free / royalty-free.
    #[must_use]
    pub const fn is_patent_free(&self) -> bool {
        matches!(self, Self::Av1 | Self::Vp9)
    }

    /// Parses a FourCC byte slice into a codec variant.
    ///
    /// Returns `None` for unrecognised FourCC values.
    #[must_use]
    pub fn from_fourcc(bytes: &[u8; 4]) -> Option<Self> {
        match bytes {
            b"av01" => Some(Self::Av1),
            b"vp09" => Some(Self::Vp9),
            b"hvc1" => Some(Self::Hevc),
            _ => None,
        }
    }
}

// ─── ExVideoPacketType ────────────────────────────────────────────────────────

/// Packet classification for Extended RTMP video packets.
///
/// Encoded in the lower 3 bits of the first header byte when the `is_ex_header`
/// flag (bit 3) is set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExVideoPacketType {
    /// Decoder configuration record / sequence header (`0`).
    SequenceStart = 0,
    /// Encoded video bitstream — composition time is present (`1`).
    CodedFrames = 1,
    /// End of sequence — signals the decoder to flush (`2`).
    SequenceEnd = 2,
    /// Encoded video bitstream — composition time offset omitted (`3`).
    CodedFramesX = 3,
    /// Out-of-band metadata (HDR10+, colour volume, etc.) (`4`).
    Metadata = 4,
    /// MPEG-2 TS sequence start for backward compatibility (`5`).
    Mpeg2SequenceStart = 5,
}

impl ExVideoPacketType {
    /// Converts the lower 3 bits of a byte to a packet type.
    ///
    /// Returns `None` for undefined values.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Option<Self> {
        match bits & 0x07 {
            0 => Some(Self::SequenceStart),
            1 => Some(Self::CodedFrames),
            2 => Some(Self::SequenceEnd),
            3 => Some(Self::CodedFramesX),
            4 => Some(Self::Metadata),
            5 => Some(Self::Mpeg2SequenceStart),
            _ => None,
        }
    }

    /// Returns the 3-bit wire encoding of this packet type.
    #[must_use]
    pub const fn as_bits(self) -> u8 {
        self as u8
    }
}

// ─── ExVideoHeader ────────────────────────────────────────────────────────────

/// Enhanced RTMP video tag header.
///
/// Carries:
/// - The `is_ex_header` flag (always `true` for extended packets).
/// - A 4-bit `frame_type` (1 = keyframe, 2 = inter-frame, …).
/// - The [`ExVideoPacketType`] in the lower 3 bits of the first byte.
/// - The 4-byte FourCC codec identifier.
///
/// # Wire layout
///
/// ```text
/// byte 0: [frame_type:4][is_ex_header=1:1][packet_type:3]
/// bytes 1-4: FourCC
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExVideoHeader {
    /// Always `true` for enhanced headers; distinguishes from legacy RTMP.
    pub is_ex_header: bool,
    /// Frame type nibble: `1` = keyframe, `2` = inter-frame, `5` = command.
    pub frame_type: u8,
    /// Which codec produced this packet.
    pub codec: RtmpExtendedCodec,
    /// Packet classification (sequence start, coded frames, …).
    pub packet_type: ExVideoPacketType,
}

impl ExVideoHeader {
    /// Constructs a new enhanced video header.
    ///
    /// `frame_type` follows the RTMP convention: `1` = keyframe, `2` = inter.
    #[must_use]
    pub fn new(codec: RtmpExtendedCodec, frame_type: u8, packet_type: ExVideoPacketType) -> Self {
        Self {
            is_ex_header: true,
            frame_type,
            codec,
            packet_type,
        }
    }

    /// Serialises the header to its 5-byte wire representation.
    ///
    /// Layout: `[frame_type:4 | ex_bit:1 | pkt_type:3][fourcc:4]`
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(5);
        let ft = (self.frame_type & 0x0F) << 4;
        let ex = if self.is_ex_header { 0x08 } else { 0x00 };
        let pt = self.packet_type.as_bits() & 0x07;
        out.push(ft | ex | pt);
        out.extend_from_slice(&self.codec.fourcc());
        out
    }

    /// Parses a header from at least 5 bytes of wire data.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Parse`] if:
    /// - `data` is shorter than 5 bytes.
    /// - The `is_ex_header` bit is not set.
    /// - The packet type nibble is undefined.
    /// - The FourCC is not one of the recognised codecs.
    pub fn parse(data: &[u8]) -> Result<Self, NetError> {
        if data.len() < 5 {
            return Err(NetError::parse(
                0,
                format!(
                    "ExVideoHeader requires at least 5 bytes, got {}",
                    data.len()
                ),
            ));
        }

        let first = data[0];
        let is_ex_header = (first & 0x08) != 0;
        if !is_ex_header {
            return Err(NetError::parse(
                0,
                "ExVideoHeader: is_ex_header bit not set — this is a legacy RTMP packet",
            ));
        }

        let frame_type = (first >> 4) & 0x0F;
        let pkt_bits = first & 0x07;
        let packet_type = ExVideoPacketType::from_bits(pkt_bits).ok_or_else(|| {
            NetError::parse(
                0,
                format!("ExVideoHeader: unknown packet type bits {pkt_bits}"),
            )
        })?;

        let fourcc: [u8; 4] = [data[1], data[2], data[3], data[4]];
        let codec = RtmpExtendedCodec::from_fourcc(&fourcc).ok_or_else(|| {
            NetError::parse(
                1,
                format!(
                    "ExVideoHeader: unrecognised FourCC {:?}",
                    std::str::from_utf8(&fourcc).unwrap_or("????")
                ),
            )
        })?;

        Ok(Self {
            is_ex_header,
            frame_type,
            codec,
            packet_type,
        })
    }

    /// Returns `true` if this header describes a keyframe.
    #[must_use]
    pub const fn is_keyframe(&self) -> bool {
        self.frame_type == 1
    }
}

// ─── RtmpExtendedPacket ───────────────────────────────────────────────────────

/// A complete Enhanced RTMP video packet (header + payload).
///
/// The payload interpretation depends on [`ExVideoPacketType`]:
/// - `SequenceStart` — codec-specific decoder configuration record.
/// - `CodedFrames` — raw bitstream with a 3-byte composition-time prefix.
/// - `CodedFramesX` — raw bitstream without composition-time (offset always 0).
/// - `SequenceEnd` — empty payload.
/// - `Metadata` — codec-specific metadata (HDR10+, colour info, …).
/// - `Mpeg2SequenceStart` — MPEG-2 TS initialisation data.
#[derive(Debug, Clone)]
pub struct RtmpExtendedPacket {
    /// 5-byte extended video header.
    pub header: ExVideoHeader,
    /// Codec-specific payload bytes.
    pub payload: Vec<u8>,
}

impl RtmpExtendedPacket {
    // ── Convenience constructors ─────────────────────────────────────────────

    /// Creates an AV1 sequence-start packet containing the AV1 `Sequence Header OBU`.
    ///
    /// This corresponds to `ExVideoPacketType::SequenceStart` for the `av01` FourCC.
    #[must_use]
    pub fn new_av1_sequence(config_obu: Vec<u8>) -> Self {
        Self {
            header: ExVideoHeader::new(
                RtmpExtendedCodec::Av1,
                1, // keyframe convention for sequence headers
                ExVideoPacketType::SequenceStart,
            ),
            payload: config_obu,
        }
    }

    /// Creates an AV1 coded-frames packet.
    ///
    /// `is_keyframe` maps to `frame_type = 1` (key) or `2` (inter).
    #[must_use]
    pub fn new_av1_frame(data: Vec<u8>, is_keyframe: bool) -> Self {
        let frame_type = if is_keyframe { 1 } else { 2 };
        Self {
            header: ExVideoHeader::new(
                RtmpExtendedCodec::Av1,
                frame_type,
                ExVideoPacketType::CodedFramesX, // AV1 uses CodedFramesX (no composition time)
            ),
            payload: data,
        }
    }

    /// Creates a VP9 sequence-start packet containing the VP9 `CodecPrivate` blob.
    #[must_use]
    pub fn new_vp9_sequence(vp9_config: Vec<u8>) -> Self {
        Self {
            header: ExVideoHeader::new(RtmpExtendedCodec::Vp9, 1, ExVideoPacketType::SequenceStart),
            payload: vp9_config,
        }
    }

    // ── Serialisation ─────────────────────────────────────────────────────────

    /// Serialises the packet to its complete wire representation.
    ///
    /// Layout:
    /// ```text
    /// [ExVideoHeader: 5 bytes]
    /// [composition_time: 3 bytes, only for CodedFrames]
    /// [payload]
    /// ```
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = self.header.serialize(); // 5 bytes

        // CodedFrames includes a 3-byte composition-time offset (big-endian signed).
        // We emit zero here; callers that need precise PTS delta should set it via
        // the raw constructor path.
        if self.header.packet_type == ExVideoPacketType::CodedFrames {
            out.push(0x00);
            out.push(0x00);
            out.push(0x00);
        }

        out.extend_from_slice(&self.payload);
        out
    }

    /// Parses a complete extended packet from raw bytes.
    ///
    /// # Errors
    ///
    /// Propagates [`ExVideoHeader::parse`] errors, or returns [`NetError::Parse`]
    /// if the buffer is too short to contain the declared payload.
    pub fn parse(data: &[u8]) -> Result<Self, NetError> {
        let header = ExVideoHeader::parse(data)?;

        // For CodedFrames there are 3 additional composition-time bytes.
        let payload_offset = if header.packet_type == ExVideoPacketType::CodedFrames {
            if data.len() < 8 {
                return Err(NetError::parse(
                    5,
                    format!(
                        "RtmpExtendedPacket CodedFrames requires 8+ bytes, got {}",
                        data.len()
                    ),
                ));
            }
            8 // 5 (header) + 3 (composition time)
        } else {
            5 // 5 (header only)
        };

        let payload = data[payload_offset..].to_vec();
        Ok(Self { header, payload })
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Returns `true` if this packet is a keyframe.
    #[must_use]
    pub fn is_keyframe(&self) -> bool {
        self.header.is_keyframe()
    }

    /// Returns `true` if this packet contains codec decoder configuration.
    #[must_use]
    pub fn is_sequence_start(&self) -> bool {
        self.header.packet_type == ExVideoPacketType::SequenceStart
    }

    /// Returns the codec for this packet.
    #[must_use]
    pub fn codec(&self) -> RtmpExtendedCodec {
        self.header.codec
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. RtmpExtendedCodec FourCC values
    #[test]
    fn test_codec_fourcc() {
        assert_eq!(&RtmpExtendedCodec::Av1.fourcc(), b"av01");
        assert_eq!(&RtmpExtendedCodec::Vp9.fourcc(), b"vp09");
        assert_eq!(&RtmpExtendedCodec::Hevc.fourcc(), b"hvc1");
    }

    // 2. Codec names
    #[test]
    fn test_codec_names() {
        assert_eq!(RtmpExtendedCodec::Av1.name(), "AV1");
        assert_eq!(RtmpExtendedCodec::Vp9.name(), "VP9");
        assert_eq!(RtmpExtendedCodec::Hevc.name(), "HEVC");
    }

    // 3. Patent-free check
    #[test]
    fn test_codec_patent_free() {
        assert!(RtmpExtendedCodec::Av1.is_patent_free());
        assert!(RtmpExtendedCodec::Vp9.is_patent_free());
        assert!(!RtmpExtendedCodec::Hevc.is_patent_free());
    }

    // 4. Codec from FourCC
    #[test]
    fn test_codec_from_fourcc() {
        assert_eq!(
            RtmpExtendedCodec::from_fourcc(b"av01"),
            Some(RtmpExtendedCodec::Av1)
        );
        assert_eq!(
            RtmpExtendedCodec::from_fourcc(b"vp09"),
            Some(RtmpExtendedCodec::Vp9)
        );
        assert_eq!(RtmpExtendedCodec::from_fourcc(b"avc1"), None);
    }

    // 5. ExVideoPacketType round-trip via bits
    #[test]
    fn test_packet_type_bits_roundtrip() {
        for (bits, expected) in [
            (0u8, ExVideoPacketType::SequenceStart),
            (1, ExVideoPacketType::CodedFrames),
            (2, ExVideoPacketType::SequenceEnd),
            (3, ExVideoPacketType::CodedFramesX),
            (4, ExVideoPacketType::Metadata),
            (5, ExVideoPacketType::Mpeg2SequenceStart),
        ] {
            let decoded = ExVideoPacketType::from_bits(bits).expect("should decode");
            assert_eq!(decoded, expected);
            assert_eq!(decoded.as_bits(), bits);
        }
    }

    // 6. ExVideoPacketType unknown bits → None
    #[test]
    fn test_packet_type_unknown() {
        // Values 6 and 7 are undefined in the spec
        assert!(ExVideoPacketType::from_bits(6).is_none());
        assert!(ExVideoPacketType::from_bits(7).is_none());
    }

    // 7. ExVideoHeader::new sets is_ex_header = true
    #[test]
    fn test_header_new_sets_ex_bit() {
        let h = ExVideoHeader::new(RtmpExtendedCodec::Av1, 1, ExVideoPacketType::SequenceStart);
        assert!(h.is_ex_header);
        assert!(h.is_keyframe());
    }

    // 8. ExVideoHeader serialize: correct first byte and FourCC
    #[test]
    fn test_header_serialize_av1_sequence() {
        let h = ExVideoHeader::new(
            RtmpExtendedCodec::Av1,
            1, // keyframe
            ExVideoPacketType::SequenceStart,
        );
        let bytes = h.serialize();
        assert_eq!(bytes.len(), 5);
        // byte 0: frame_type=1 (0x1x), ex_bit=1 (0x08), pkt_type=0 (0x00)  → 0x18
        assert_eq!(bytes[0], 0x18, "first byte should be 0x18");
        assert_eq!(&bytes[1..5], b"av01");
    }

    // 9. ExVideoHeader serialize/parse roundtrip — VP9 inter
    #[test]
    fn test_header_roundtrip_vp9_inter() {
        let original = ExVideoHeader::new(
            RtmpExtendedCodec::Vp9,
            2, // inter-frame
            ExVideoPacketType::CodedFramesX,
        );
        let bytes = original.serialize();
        let decoded = ExVideoHeader::parse(&bytes).expect("should parse");
        assert_eq!(decoded.codec, RtmpExtendedCodec::Vp9);
        assert_eq!(decoded.frame_type, 2);
        assert_eq!(decoded.packet_type, ExVideoPacketType::CodedFramesX);
        assert!(decoded.is_ex_header);
    }

    // 10. ExVideoHeader parse rejects non-extended marker
    #[test]
    fn test_header_parse_rejects_legacy() {
        // bit 3 of byte 0 is 0 → legacy RTMP packet
        let data = [0x10u8, b'a', b'v', b'0', b'1'];
        let result = ExVideoHeader::parse(&data);
        assert!(result.is_err());
    }

    // 11. ExVideoHeader parse rejects too-short buffer
    #[test]
    fn test_header_parse_too_short() {
        let data = [0x18u8, b'a', b'v']; // only 3 bytes
        let result = ExVideoHeader::parse(&data);
        assert!(result.is_err());
    }

    // 12. ExVideoHeader parse rejects unknown FourCC
    #[test]
    fn test_header_parse_unknown_fourcc() {
        let data = [0x18u8, b'x', b'y', b'z', b'w']; // unrecognised FourCC
        let result = ExVideoHeader::parse(&data);
        assert!(result.is_err());
    }

    // 13. RtmpExtendedPacket::new_av1_sequence
    #[test]
    fn test_packet_av1_sequence() {
        let obu = vec![0x81, 0x00, 0x0C, 0x00];
        let pkt = RtmpExtendedPacket::new_av1_sequence(obu.clone());
        assert!(pkt.is_sequence_start());
        assert!(pkt.is_keyframe());
        assert_eq!(pkt.codec(), RtmpExtendedCodec::Av1);
        assert_eq!(pkt.payload, obu);
    }

    // 14. RtmpExtendedPacket::new_av1_frame keyframe
    #[test]
    fn test_packet_av1_keyframe() {
        let data = vec![0x12, 0x34, 0x56];
        let pkt = RtmpExtendedPacket::new_av1_frame(data.clone(), true);
        assert!(pkt.is_keyframe());
        assert!(!pkt.is_sequence_start());
        assert_eq!(pkt.header.packet_type, ExVideoPacketType::CodedFramesX);
    }

    // 15. RtmpExtendedPacket::new_av1_frame inter-frame
    #[test]
    fn test_packet_av1_interframe() {
        let pkt = RtmpExtendedPacket::new_av1_frame(vec![0xAB], false);
        assert!(!pkt.is_keyframe());
        assert_eq!(pkt.header.frame_type, 2);
    }

    // 16. RtmpExtendedPacket::new_vp9_sequence
    #[test]
    fn test_packet_vp9_sequence() {
        let cfg = vec![0x00, 0x63, 0x00, 0x01];
        let pkt = RtmpExtendedPacket::new_vp9_sequence(cfg.clone());
        assert!(pkt.is_sequence_start());
        assert_eq!(pkt.codec(), RtmpExtendedCodec::Vp9);
        assert_eq!(pkt.payload, cfg);
    }

    // 17. RtmpExtendedPacket serialize/parse roundtrip — AV1 SequenceStart
    #[test]
    fn test_packet_roundtrip_av1_sequence() {
        let obu = vec![0x81, 0x00, 0x0C];
        let original = RtmpExtendedPacket::new_av1_sequence(obu.clone());
        let bytes = original.serialize();
        let decoded = RtmpExtendedPacket::parse(&bytes).expect("should parse");
        assert_eq!(decoded.header.codec, RtmpExtendedCodec::Av1);
        assert_eq!(decoded.header.packet_type, ExVideoPacketType::SequenceStart);
        assert_eq!(decoded.payload, obu);
    }

    // 18. RtmpExtendedPacket serialize/parse roundtrip — CodedFramesX (no comp time)
    #[test]
    fn test_packet_roundtrip_av1_coded_frames_x() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let original = RtmpExtendedPacket::new_av1_frame(data.clone(), true);
        let bytes = original.serialize();
        let decoded = RtmpExtendedPacket::parse(&bytes).expect("should parse");
        assert_eq!(decoded.header.packet_type, ExVideoPacketType::CodedFramesX);
        assert_eq!(decoded.payload, data);
    }

    // 19. CodedFrames packet serializes 3-byte composition time
    #[test]
    fn test_packet_coded_frames_composition_time_bytes() {
        let pkt = RtmpExtendedPacket {
            header: ExVideoHeader::new(RtmpExtendedCodec::Vp9, 1, ExVideoPacketType::CodedFrames),
            payload: vec![0xAA, 0xBB],
        };
        let bytes = pkt.serialize();
        // 5 (header) + 3 (comp time) + 2 (payload) = 10
        assert_eq!(bytes.len(), 10);
        // Composition time bytes are at positions 5-7
        assert_eq!(bytes[5], 0x00);
        assert_eq!(bytes[6], 0x00);
        assert_eq!(bytes[7], 0x00);
    }

    // 20. CodedFrames parse consumes composition time bytes
    #[test]
    fn test_packet_coded_frames_parse_skips_comp_time() {
        let pkt = RtmpExtendedPacket {
            header: ExVideoHeader::new(RtmpExtendedCodec::Av1, 2, ExVideoPacketType::CodedFrames),
            payload: vec![0x11, 0x22, 0x33],
        };
        let bytes = pkt.serialize();
        let decoded = RtmpExtendedPacket::parse(&bytes).expect("should parse");
        assert_eq!(decoded.payload, vec![0x11, 0x22, 0x33]);
    }

    // 21. RtmpExtendedPacket::parse rejects too-short CodedFrames
    #[test]
    fn test_packet_parse_coded_frames_too_short() {
        // CodedFrames with only 6 bytes (needs 8+)
        let data = [0x19u8, b'a', b'v', b'0', b'1', 0x00, 0x01]; // 7 bytes
        let result = RtmpExtendedPacket::parse(&data);
        assert!(result.is_err());
    }

    // 22. HEVC sequence-start packet roundtrip
    #[test]
    fn test_packet_hevc_sequence_roundtrip() {
        let cfg = vec![0x01, 0x64, 0x00, 0x1F];
        let pkt = RtmpExtendedPacket {
            header: ExVideoHeader::new(
                RtmpExtendedCodec::Hevc,
                1,
                ExVideoPacketType::SequenceStart,
            ),
            payload: cfg.clone(),
        };
        let bytes = pkt.serialize();
        let decoded = RtmpExtendedPacket::parse(&bytes).expect("should parse");
        assert_eq!(decoded.codec(), RtmpExtendedCodec::Hevc);
        assert_eq!(decoded.payload, cfg);
    }
}
