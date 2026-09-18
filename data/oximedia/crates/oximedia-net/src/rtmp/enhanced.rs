//! RTMP Enhanced (RTMP+) with AV1/VP9 codec support.
//!
//! Implements the Enhanced RTMP specification (originally from Vewd/YouTube)
//! which extends RTMP to support modern codecs beyond H.264/AAC:
//!
//! - **AV1** (AOMedia Video 1) — patent-free, next-generation codec
//! - **VP9** — Google's royalty-free codec
//! - **HEVC** (H.265) — high-efficiency video coding
//! - **Opus** — versatile audio codec
//! - **FLAC** — lossless audio
//!
//! Enhanced RTMP uses a new FourCC-based codec identification scheme
//! in the video/audio tag headers instead of the legacy codec IDs.

use bytes::{BufMut, Bytes, BytesMut};
use std::fmt;

// ─── Enhanced Video Codec ─────────────────────────────────────────────────────

/// FourCC code for codec identification in Enhanced RTMP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC([u8; 4]);

impl FourCC {
    /// AV1 codec.
    pub const AV1: Self = Self(*b"av01");
    /// VP9 codec.
    pub const VP9: Self = Self(*b"vp09");
    /// HEVC (H.265) codec.
    pub const HEVC: Self = Self(*b"hvc1");
    /// Opus audio codec.
    pub const OPUS: Self = Self(*b"Opus");
    /// FLAC audio codec.
    pub const FLAC: Self = Self(*b"fLaC");
    /// AAC audio codec (legacy, for reference).
    pub const AAC: Self = Self(*b"mp4a");
    /// AVC (H.264) video codec (legacy, for reference).
    pub const AVC: Self = Self(*b"avc1");

    /// Creates a FourCC from raw bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(bytes)
    }

    /// Creates a FourCC from a string slice.
    ///
    /// Returns `None` if the string is not exactly 4 bytes.
    #[must_use]
    pub fn from_str_code(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() == 4 {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(bytes);
            Some(Self(arr))
        } else {
            None
        }
    }

    /// Returns the raw FourCC bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 4] {
        &self.0
    }

    /// Returns the FourCC as a string (if valid UTF-8).
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.0).ok()
    }

    /// Returns the codec name for display.
    #[must_use]
    pub fn codec_name(&self) -> &'static str {
        match *self {
            Self::AV1 => "AV1",
            Self::VP9 => "VP9",
            Self::HEVC => "HEVC",
            Self::OPUS => "Opus",
            Self::FLAC => "FLAC",
            Self::AAC => "AAC",
            Self::AVC => "AVC/H.264",
            _ => "Unknown",
        }
    }

    /// Returns whether this FourCC represents a patent-free codec.
    #[must_use]
    pub fn is_patent_free(&self) -> bool {
        matches!(*self, Self::AV1 | Self::VP9 | Self::OPUS | Self::FLAC)
    }
}

impl fmt::Display for FourCC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_str() {
            Some(s) => f.write_str(s),
            None => write!(
                f,
                "0x{:02X}{:02X}{:02X}{:02X}",
                self.0[0], self.0[1], self.0[2], self.0[3]
            ),
        }
    }
}

// ─── Enhanced Video Packet Type ───────────────────────────────────────────────

/// Enhanced RTMP video packet types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EnhancedVideoPacketType {
    /// Sequence start (codec configuration / decoder config record).
    SequenceStart = 0,
    /// Coded frames (video data).
    CodedFrames = 1,
    /// Sequence end.
    SequenceEnd = 2,
    /// Coded frames without composition time offset.
    CodedFramesX = 3,
    /// Metadata (e.g., HDR metadata, color info).
    Metadata = 4,
    /// MPEG-2 TS sequence start (for backwards compat).
    Mpeg2TsSequenceStart = 5,
}

impl EnhancedVideoPacketType {
    /// Creates from raw byte value.
    #[must_use]
    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::SequenceStart),
            1 => Some(Self::CodedFrames),
            2 => Some(Self::SequenceEnd),
            3 => Some(Self::CodedFramesX),
            4 => Some(Self::Metadata),
            5 => Some(Self::Mpeg2TsSequenceStart),
            _ => None,
        }
    }
}

// ─── Enhanced Frame Type ──────────────────────────────────────────────────────

/// Enhanced RTMP video frame type (upper 4 bits of first byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EnhancedFrameType {
    /// Key frame (IDR for AV1/VP9).
    Key = 1,
    /// Inter frame (predicted).
    Inter = 2,
    /// Disposable inter frame.
    DisposableInter = 3,
    /// Command frame.
    Command = 5,
}

impl EnhancedFrameType {
    /// Creates from the upper 4 bits of the tag byte.
    #[must_use]
    pub const fn from_nibble(nibble: u8) -> Option<Self> {
        match nibble {
            1 => Some(Self::Key),
            2 => Some(Self::Inter),
            3 => Some(Self::DisposableInter),
            5 => Some(Self::Command),
            _ => None,
        }
    }

    /// Returns true for keyframes.
    #[must_use]
    pub const fn is_keyframe(&self) -> bool {
        matches!(self, Self::Key)
    }
}

// ─── Enhanced Audio Packet Type ───────────────────────────────────────────────

/// Enhanced RTMP audio packet type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EnhancedAudioPacketType {
    /// Sequence start (audio specific config).
    SequenceStart = 0,
    /// Coded audio frames.
    CodedFrames = 1,
    /// Sequence end.
    SequenceEnd = 2,
    /// Multichannel configuration.
    MultichannelConfig = 4,
    /// Multitrack audio.
    Multitrack = 5,
}

impl EnhancedAudioPacketType {
    /// Creates from raw byte value.
    #[must_use]
    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::SequenceStart),
            1 => Some(Self::CodedFrames),
            2 => Some(Self::SequenceEnd),
            4 => Some(Self::MultichannelConfig),
            5 => Some(Self::Multitrack),
            _ => None,
        }
    }
}

// ─── Enhanced Video Tag ───────────────────────────────────────────────────────

/// An enhanced RTMP video tag header.
///
/// Enhanced RTMP uses a different tag structure:
/// - Byte 0: `[frame_type:4][is_ex_header:1][packet_type:3]` (when `is_ex_header` = 1)
/// - Bytes 1-4: FourCC codec identifier
/// - Remaining: codec-specific data
#[derive(Debug, Clone)]
pub struct EnhancedVideoTag {
    /// Frame type (key, inter, etc.).
    pub frame_type: EnhancedFrameType,
    /// Whether this is an enhanced (ex) header.
    pub is_ex_header: bool,
    /// Packet type (sequence start, coded frames, etc.).
    pub packet_type: EnhancedVideoPacketType,
    /// Codec FourCC.
    pub codec: FourCC,
    /// Composition time offset in milliseconds (for CodedFrames).
    pub composition_time_ms: i32,
    /// Raw payload data.
    pub data: Bytes,
}

impl EnhancedVideoTag {
    /// Creates a new enhanced video tag for AV1.
    #[must_use]
    pub fn av1_keyframe(data: Bytes) -> Self {
        Self {
            frame_type: EnhancedFrameType::Key,
            is_ex_header: true,
            packet_type: EnhancedVideoPacketType::CodedFrames,
            codec: FourCC::AV1,
            composition_time_ms: 0,
            data,
        }
    }

    /// Creates a new enhanced video tag for VP9.
    #[must_use]
    pub fn vp9_keyframe(data: Bytes) -> Self {
        Self {
            frame_type: EnhancedFrameType::Key,
            is_ex_header: true,
            packet_type: EnhancedVideoPacketType::CodedFrames,
            codec: FourCC::VP9,
            composition_time_ms: 0,
            data,
        }
    }

    /// Creates a sequence start tag for codec initialization.
    #[must_use]
    pub fn sequence_start(codec: FourCC, config_data: Bytes) -> Self {
        Self {
            frame_type: EnhancedFrameType::Key,
            is_ex_header: true,
            packet_type: EnhancedVideoPacketType::SequenceStart,
            codec,
            composition_time_ms: 0,
            data: config_data,
        }
    }

    /// Creates a sequence end tag.
    #[must_use]
    pub fn sequence_end(codec: FourCC) -> Self {
        Self {
            frame_type: EnhancedFrameType::Key,
            is_ex_header: true,
            packet_type: EnhancedVideoPacketType::SequenceEnd,
            codec,
            composition_time_ms: 0,
            data: Bytes::new(),
        }
    }

    /// Encodes the enhanced video tag to bytes.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(5 + self.data.len());

        // Byte 0: [frame_type:4][is_ex_header:1][packet_type:3]
        let frame_nibble = (self.frame_type as u8) << 4;
        let ex_bit = if self.is_ex_header { 0x08 } else { 0x00 };
        let pkt_type = self.packet_type as u8;
        buf.put_u8(frame_nibble | ex_bit | pkt_type);

        // Bytes 1-4: FourCC
        buf.put_slice(self.codec.as_bytes());

        // For CodedFrames, add composition time offset (3 bytes, big-endian signed)
        if self.packet_type == EnhancedVideoPacketType::CodedFrames {
            let ct = self.composition_time_ms;
            buf.put_u8(((ct >> 16) & 0xFF) as u8);
            buf.put_u8(((ct >> 8) & 0xFF) as u8);
            buf.put_u8((ct & 0xFF) as u8);
        }

        // Payload
        buf.put_slice(&self.data);

        buf.freeze()
    }

    /// Decodes an enhanced video tag from bytes.
    ///
    /// Returns `None` if the data is too short or not an enhanced header.
    #[must_use]
    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 5 {
            return None;
        }

        let first = data[0];
        let frame_type_nibble = (first >> 4) & 0x0F;
        let is_ex_header = (first & 0x08) != 0;
        let packet_type_bits = first & 0x07;

        if !is_ex_header {
            return None;
        }

        let frame_type = EnhancedFrameType::from_nibble(frame_type_nibble)?;
        let packet_type = EnhancedVideoPacketType::from_byte(packet_type_bits)?;

        let mut fourcc = [0u8; 4];
        fourcc.copy_from_slice(&data[1..5]);
        let codec = FourCC::from_bytes(fourcc);

        let (composition_time_ms, payload_start) =
            if packet_type == EnhancedVideoPacketType::CodedFrames && data.len() >= 8 {
                let ct = ((data[5] as i32) << 16) | ((data[6] as i32) << 8) | (data[7] as i32);
                // Sign-extend 24-bit value
                let ct = if ct & 0x80_0000 != 0 {
                    ct | !0xFF_FFFF
                } else {
                    ct
                };
                (ct, 8)
            } else {
                (0, 5)
            };

        let payload = if data.len() > payload_start {
            Bytes::copy_from_slice(&data[payload_start..])
        } else {
            Bytes::new()
        };

        Some(Self {
            frame_type,
            is_ex_header,
            packet_type,
            codec,
            composition_time_ms,
            data: payload,
        })
    }

    /// Returns the codec name.
    #[must_use]
    pub fn codec_name(&self) -> &'static str {
        self.codec.codec_name()
    }

    /// Returns whether this tag uses a patent-free codec.
    #[must_use]
    pub fn is_patent_free(&self) -> bool {
        self.codec.is_patent_free()
    }
}

// ─── Enhanced Audio Tag ───────────────────────────────────────────────────────

/// An enhanced RTMP audio tag header.
///
/// Similar to the video tag, uses FourCC for codec identification.
#[derive(Debug, Clone)]
pub struct EnhancedAudioTag {
    /// Packet type.
    pub packet_type: EnhancedAudioPacketType,
    /// Codec FourCC.
    pub codec: FourCC,
    /// Raw payload data.
    pub data: Bytes,
}

impl EnhancedAudioTag {
    /// Creates an Opus audio tag.
    #[must_use]
    pub fn opus(data: Bytes) -> Self {
        Self {
            packet_type: EnhancedAudioPacketType::CodedFrames,
            codec: FourCC::OPUS,
            data,
        }
    }

    /// Creates a FLAC audio tag.
    #[must_use]
    pub fn flac(data: Bytes) -> Self {
        Self {
            packet_type: EnhancedAudioPacketType::CodedFrames,
            codec: FourCC::FLAC,
            data,
        }
    }

    /// Creates an audio sequence start tag.
    #[must_use]
    pub fn sequence_start(codec: FourCC, config: Bytes) -> Self {
        Self {
            packet_type: EnhancedAudioPacketType::SequenceStart,
            codec,
            data: config,
        }
    }

    /// Encodes the enhanced audio tag.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(5 + self.data.len());
        // Byte 0: [sound_format=0x09:4][sound_rate=3:2][sound_size=1:1][sound_type=1:1]
        // 0x9F = enhanced audio marker (sound_format=9, rest=0xF)
        buf.put_u8(0x9F);
        // Byte 1: packet type
        buf.put_u8(self.packet_type as u8);
        // Bytes 2-5: FourCC
        buf.put_slice(self.codec.as_bytes());
        // Payload
        buf.put_slice(&self.data);
        buf.freeze()
    }

    /// Decodes an enhanced audio tag.
    #[must_use]
    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 6 {
            return None;
        }

        // Check enhanced audio marker
        if data[0] != 0x9F {
            return None;
        }

        let packet_type = EnhancedAudioPacketType::from_byte(data[1])?;

        let mut fourcc = [0u8; 4];
        fourcc.copy_from_slice(&data[2..6]);
        let codec = FourCC::from_bytes(fourcc);

        let payload = if data.len() > 6 {
            Bytes::copy_from_slice(&data[6..])
        } else {
            Bytes::new()
        };

        Some(Self {
            packet_type,
            codec,
            data: payload,
        })
    }

    /// Returns the codec name.
    #[must_use]
    pub fn codec_name(&self) -> &'static str {
        self.codec.codec_name()
    }
}

// ─── Enhanced RTMP Capability ─────────────────────────────────────────────────

/// Describes the enhanced RTMP capabilities of a connection.
#[derive(Debug, Clone)]
pub struct EnhancedRtmpCapabilities {
    /// Supported video codecs.
    pub video_codecs: Vec<FourCC>,
    /// Supported audio codecs.
    pub audio_codecs: Vec<FourCC>,
    /// Whether the peer supports enhanced RTMP.
    pub enhanced_supported: bool,
    /// Protocol version string.
    pub version: String,
}

impl Default for EnhancedRtmpCapabilities {
    fn default() -> Self {
        Self {
            video_codecs: vec![FourCC::AV1, FourCC::VP9, FourCC::AVC],
            audio_codecs: vec![FourCC::OPUS, FourCC::FLAC, FourCC::AAC],
            enhanced_supported: true,
            version: "1.0".to_owned(),
        }
    }
}

impl EnhancedRtmpCapabilities {
    /// Creates capabilities with only patent-free codecs.
    #[must_use]
    pub fn patent_free_only() -> Self {
        Self {
            video_codecs: vec![FourCC::AV1, FourCC::VP9],
            audio_codecs: vec![FourCC::OPUS, FourCC::FLAC],
            enhanced_supported: true,
            version: "1.0".to_owned(),
        }
    }

    /// Returns whether a video codec is supported.
    #[must_use]
    pub fn supports_video_codec(&self, codec: &FourCC) -> bool {
        self.video_codecs.contains(codec)
    }

    /// Returns whether an audio codec is supported.
    #[must_use]
    pub fn supports_audio_codec(&self, codec: &FourCC) -> bool {
        self.audio_codecs.contains(codec)
    }

    /// Negotiates capabilities with a peer, returning the intersection.
    #[must_use]
    pub fn negotiate(&self, peer: &Self) -> Self {
        let video = self
            .video_codecs
            .iter()
            .filter(|c| peer.video_codecs.contains(c))
            .copied()
            .collect();
        let audio = self
            .audio_codecs
            .iter()
            .filter(|c| peer.audio_codecs.contains(c))
            .copied()
            .collect();

        Self {
            video_codecs: video,
            audio_codecs: audio,
            enhanced_supported: self.enhanced_supported && peer.enhanced_supported,
            version: self.version.clone(),
        }
    }

    /// Returns the list of supported codecs as a summary string.
    #[must_use]
    pub fn codec_summary(&self) -> String {
        let video: Vec<&str> = self.video_codecs.iter().map(|c| c.codec_name()).collect();
        let audio: Vec<&str> = self.audio_codecs.iter().map(|c| c.codec_name()).collect();
        format!("video=[{}], audio=[{}]", video.join(", "), audio.join(", "))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. FourCC constants
    #[test]
    fn test_fourcc_constants() {
        assert_eq!(FourCC::AV1.as_bytes(), b"av01");
        assert_eq!(FourCC::VP9.as_bytes(), b"vp09");
        assert_eq!(FourCC::OPUS.as_bytes(), b"Opus");
        assert_eq!(FourCC::FLAC.as_bytes(), b"fLaC");
    }

    // 2. FourCC from string
    #[test]
    fn test_fourcc_from_str() {
        let cc = FourCC::from_str_code("av01");
        assert!(cc.is_some());
        assert_eq!(cc.expect("should succeed"), FourCC::AV1);
        assert!(FourCC::from_str_code("too_long").is_none());
    }

    // 3. FourCC codec name
    #[test]
    fn test_fourcc_codec_name() {
        assert_eq!(FourCC::AV1.codec_name(), "AV1");
        assert_eq!(FourCC::VP9.codec_name(), "VP9");
        assert_eq!(FourCC::HEVC.codec_name(), "HEVC");
    }

    // 4. FourCC patent-free check
    #[test]
    fn test_fourcc_patent_free() {
        assert!(FourCC::AV1.is_patent_free());
        assert!(FourCC::VP9.is_patent_free());
        assert!(FourCC::OPUS.is_patent_free());
        assert!(FourCC::FLAC.is_patent_free());
        assert!(!FourCC::HEVC.is_patent_free());
        assert!(!FourCC::AAC.is_patent_free());
    }

    // 5. FourCC display
    #[test]
    fn test_fourcc_display() {
        assert_eq!(format!("{}", FourCC::AV1), "av01");
        assert_eq!(format!("{}", FourCC::OPUS), "Opus");
    }

    // 6. Enhanced video packet type from byte
    #[test]
    fn test_video_packet_type() {
        assert_eq!(
            EnhancedVideoPacketType::from_byte(0),
            Some(EnhancedVideoPacketType::SequenceStart)
        );
        assert_eq!(
            EnhancedVideoPacketType::from_byte(1),
            Some(EnhancedVideoPacketType::CodedFrames)
        );
        assert!(EnhancedVideoPacketType::from_byte(99).is_none());
    }

    // 7. Enhanced frame type
    #[test]
    fn test_frame_type() {
        assert!(EnhancedFrameType::Key.is_keyframe());
        assert!(!EnhancedFrameType::Inter.is_keyframe());
        assert_eq!(
            EnhancedFrameType::from_nibble(1),
            Some(EnhancedFrameType::Key)
        );
        assert!(EnhancedFrameType::from_nibble(0).is_none());
    }

    // 8. Enhanced audio packet type
    #[test]
    fn test_audio_packet_type() {
        assert_eq!(
            EnhancedAudioPacketType::from_byte(0),
            Some(EnhancedAudioPacketType::SequenceStart)
        );
        assert_eq!(
            EnhancedAudioPacketType::from_byte(5),
            Some(EnhancedAudioPacketType::Multitrack)
        );
        assert!(EnhancedAudioPacketType::from_byte(3).is_none());
    }

    // 9. AV1 keyframe tag creation
    #[test]
    fn test_av1_keyframe() {
        let tag = EnhancedVideoTag::av1_keyframe(Bytes::from(vec![1, 2, 3]));
        assert_eq!(tag.codec, FourCC::AV1);
        assert!(tag.frame_type.is_keyframe());
        assert!(tag.is_ex_header);
        assert!(tag.is_patent_free());
    }

    // 10. VP9 keyframe tag creation
    #[test]
    fn test_vp9_keyframe() {
        let tag = EnhancedVideoTag::vp9_keyframe(Bytes::from(vec![4, 5, 6]));
        assert_eq!(tag.codec, FourCC::VP9);
        assert_eq!(tag.codec_name(), "VP9");
    }

    // 11. Sequence start tag
    #[test]
    fn test_sequence_start() {
        let tag = EnhancedVideoTag::sequence_start(FourCC::AV1, Bytes::from(vec![0xAA]));
        assert_eq!(tag.packet_type, EnhancedVideoPacketType::SequenceStart);
    }

    // 12. Sequence end tag
    #[test]
    fn test_sequence_end() {
        let tag = EnhancedVideoTag::sequence_end(FourCC::VP9);
        assert_eq!(tag.packet_type, EnhancedVideoPacketType::SequenceEnd);
        assert!(tag.data.is_empty());
    }

    // 13. Video tag encode/decode roundtrip for SequenceStart
    #[test]
    fn test_video_tag_encode_decode_sequence_start() {
        let original = EnhancedVideoTag::sequence_start(FourCC::AV1, Bytes::from(vec![0x01, 0x02]));
        let encoded = original.encode();
        let decoded = EnhancedVideoTag::decode(&encoded);
        assert!(decoded.is_some());
        let decoded = decoded.expect("should decode");
        assert_eq!(decoded.codec, FourCC::AV1);
        assert_eq!(decoded.packet_type, EnhancedVideoPacketType::SequenceStart);
        assert_eq!(decoded.data, Bytes::from(vec![0x01, 0x02]));
    }

    // 14. Video tag encode/decode roundtrip for CodedFrames
    #[test]
    fn test_video_tag_encode_decode_coded_frames() {
        let original = EnhancedVideoTag::av1_keyframe(Bytes::from(vec![0xDE, 0xAD]));
        let encoded = original.encode();
        let decoded = EnhancedVideoTag::decode(&encoded);
        assert!(decoded.is_some());
        let decoded = decoded.expect("should decode");
        assert_eq!(decoded.codec, FourCC::AV1);
        assert_eq!(decoded.frame_type, EnhancedFrameType::Key);
        assert_eq!(decoded.data, Bytes::from(vec![0xDE, 0xAD]));
    }

    // 15. Video tag decode rejects non-enhanced
    #[test]
    fn test_video_tag_decode_rejects_non_enhanced() {
        // Create a byte sequence without the ex_header bit
        let data = vec![0x10, 0x00, 0x00, 0x00, 0x00]; // frame_type=1, no ex bit
        let decoded = EnhancedVideoTag::decode(&data);
        assert!(decoded.is_none());
    }

    // 16. Video tag decode rejects too short
    #[test]
    fn test_video_tag_decode_too_short() {
        let data = vec![0x18, 0x61]; // Only 2 bytes
        assert!(EnhancedVideoTag::decode(&data).is_none());
    }

    // 17. Opus audio tag
    #[test]
    fn test_opus_audio_tag() {
        let tag = EnhancedAudioTag::opus(Bytes::from(vec![0xBB]));
        assert_eq!(tag.codec, FourCC::OPUS);
        assert_eq!(tag.codec_name(), "Opus");
    }

    // 18. FLAC audio tag
    #[test]
    fn test_flac_audio_tag() {
        let tag = EnhancedAudioTag::flac(Bytes::from(vec![0xCC]));
        assert_eq!(tag.codec, FourCC::FLAC);
    }

    // 19. Audio tag encode/decode roundtrip
    #[test]
    fn test_audio_tag_encode_decode() {
        let original = EnhancedAudioTag::opus(Bytes::from(vec![0x11, 0x22]));
        let encoded = original.encode();
        let decoded = EnhancedAudioTag::decode(&encoded);
        assert!(decoded.is_some());
        let decoded = decoded.expect("should decode");
        assert_eq!(decoded.codec, FourCC::OPUS);
        assert_eq!(decoded.data, Bytes::from(vec![0x11, 0x22]));
    }

    // 20. Audio tag decode rejects non-enhanced marker
    #[test]
    fn test_audio_tag_decode_rejects_non_enhanced() {
        let data = vec![0xAF, 0x01, b'O', b'p', b'u', b's', 0x00];
        assert!(EnhancedAudioTag::decode(&data).is_none());
    }

    // 21. Audio sequence start
    #[test]
    fn test_audio_sequence_start() {
        let tag = EnhancedAudioTag::sequence_start(FourCC::OPUS, Bytes::from(vec![0x00]));
        assert_eq!(tag.packet_type, EnhancedAudioPacketType::SequenceStart);
    }

    // 22. Enhanced capabilities default
    #[test]
    fn test_capabilities_default() {
        let caps = EnhancedRtmpCapabilities::default();
        assert!(caps.enhanced_supported);
        assert!(caps.supports_video_codec(&FourCC::AV1));
        assert!(caps.supports_video_codec(&FourCC::VP9));
        assert!(caps.supports_audio_codec(&FourCC::OPUS));
    }

    // 23. Patent-free only capabilities
    #[test]
    fn test_patent_free_capabilities() {
        let caps = EnhancedRtmpCapabilities::patent_free_only();
        assert!(caps.supports_video_codec(&FourCC::AV1));
        assert!(caps.supports_video_codec(&FourCC::VP9));
        assert!(!caps.supports_video_codec(&FourCC::AVC));
        assert!(!caps.supports_audio_codec(&FourCC::AAC));
    }

    // 24. Capabilities negotiation
    #[test]
    fn test_capabilities_negotiation() {
        let local = EnhancedRtmpCapabilities::default();
        let peer = EnhancedRtmpCapabilities::patent_free_only();
        let negotiated = local.negotiate(&peer);
        assert!(negotiated.supports_video_codec(&FourCC::AV1));
        assert!(negotiated.supports_video_codec(&FourCC::VP9));
        assert!(!negotiated.supports_video_codec(&FourCC::AVC));
    }

    // 25. Codec summary
    #[test]
    fn test_codec_summary() {
        let caps = EnhancedRtmpCapabilities::patent_free_only();
        let summary = caps.codec_summary();
        assert!(summary.contains("AV1"));
        assert!(summary.contains("VP9"));
        assert!(summary.contains("Opus"));
        assert!(summary.contains("FLAC"));
    }

    // 26. Composition time encoding
    #[test]
    fn test_composition_time() {
        let mut tag = EnhancedVideoTag::av1_keyframe(Bytes::new());
        tag.composition_time_ms = 33; // ~30fps offset
        let encoded = tag.encode();
        let decoded = EnhancedVideoTag::decode(&encoded).expect("should decode");
        assert_eq!(decoded.composition_time_ms, 33);
    }

    // 27. FourCC equality
    #[test]
    fn test_fourcc_equality() {
        assert_eq!(FourCC::AV1, FourCC::from_bytes(*b"av01"));
        assert_ne!(FourCC::AV1, FourCC::VP9);
    }
}
