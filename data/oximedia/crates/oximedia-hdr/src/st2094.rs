//! SMPTE ST 2094 dynamic metadata stubs.
//!
//! Provides serialisable structs for:
//! - ST 2094-10 (Samsung HDR10+ variant, application_identifier = 4)
//! - ST 2094-40 (Dolby Vision / HDR10+ application_identifier = 7 variant)
//!
//! The byte-level serialisation uses a simplified little-endian layout
//! suitable for embedding in HEVC SEI NAL units.

use crate::{HdrError, Result};

// ── ST 2094-10 ────────────────────────────────────────────────────────────────

/// SMPTE ST 2094-10 dynamic metadata (HDR10+ scene-level data).
///
/// This structure carries the per-frame or per-scene brightness metadata
/// defined in SMPTE ST 2094-10, targeted at consumer HDR displays.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct St2094_10Metadata {
    /// Application identifier (4 for HDR10+/Samsung variant).
    pub application_identifier: u8,
    /// Application version (typically 1).
    pub application_version: u8,
    /// Number of analysis windows (1 or 3).
    pub num_windows: u8,
    /// Target system display maximum luminance (nits, integer).
    pub targeted_system_display_maximum_luminance: u32,
    /// Maximum scene colour light level per R/G/B channel.
    pub maxscl: [u32; 3],
    /// Average maximum RGB value across the frame.
    pub average_maxrgb: u32,
    /// Distribution of MaxRGB values (Bezier curve control points).
    pub distribution_maxrgb: Vec<u32>,
    /// Fraction of pixels above the knee point (0–255 scaled integer).
    pub fraction_bright_pixels: u32,
    /// Whether tone-mapping metadata is present.
    pub tone_mapping_flag: bool,
    /// Tone-mapping knee point X coordinate (0–4095).
    pub knee_point_x: u16,
    /// Tone-mapping knee point Y coordinate (0–4095).
    pub knee_point_y: u16,
    /// Bezier curve anchor values (up to 9; 0–1023 range each).
    pub bezier_curve_anchors: Vec<u16>,
}

impl St2094_10Metadata {
    /// Create a default single-window metadata block targeting 1000 nits.
    pub fn new_default() -> Self {
        Self {
            application_identifier: 4,
            application_version: 1,
            num_windows: 1,
            targeted_system_display_maximum_luminance: 1000,
            maxscl: [1_000_000u32; 3],
            average_maxrgb: 500_000,
            distribution_maxrgb: vec![0, 16, 32, 64, 128, 256, 512, 1024, 2048],
            fraction_bright_pixels: 0,
            tone_mapping_flag: true,
            knee_point_x: 512,
            knee_point_y: 512,
            bezier_curve_anchors: vec![0, 128, 256, 384, 512, 640, 768, 896, 1023],
        }
    }

    /// Serialise this metadata to a simplified HEVC SEI payload (little-endian).
    ///
    /// # Layout
    /// ```text
    /// [0]    u8   application_identifier
    /// [1]    u8   application_version
    /// [2]    u8   num_windows
    /// [3..6] u32  targeted_system_display_maximum_luminance
    /// [7..18] 3×u32 maxscl[0..3]
    /// [19..22] u32 average_maxrgb
    /// [23]   u8   distribution_maxrgb count
    /// [24..] n×u32 distribution_maxrgb values
    /// ...    u32  fraction_bright_pixels
    /// ...    u8   tone_mapping_flag (0/1)
    /// ...    u16  knee_point_x
    /// ...    u16  knee_point_y
    /// ...    u8   bezier_curve_anchors count
    /// ...    n×u16 bezier_curve_anchors
    /// ```
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        buf.push(self.application_identifier);
        buf.push(self.application_version);
        buf.push(self.num_windows);
        buf.extend_from_slice(&self.targeted_system_display_maximum_luminance.to_le_bytes());
        for &ch in &self.maxscl {
            buf.extend_from_slice(&ch.to_le_bytes());
        }
        buf.extend_from_slice(&self.average_maxrgb.to_le_bytes());

        // distribution_maxrgb: length-prefixed
        let dist_len = self.distribution_maxrgb.len().min(255) as u8;
        buf.push(dist_len);
        for &v in self.distribution_maxrgb.iter().take(usize::from(dist_len)) {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        buf.extend_from_slice(&self.fraction_bright_pixels.to_le_bytes());
        buf.push(u8::from(self.tone_mapping_flag));
        buf.extend_from_slice(&self.knee_point_x.to_le_bytes());
        buf.extend_from_slice(&self.knee_point_y.to_le_bytes());

        // bezier_curve_anchors: length-prefixed (max 9)
        let anchor_len = self.bezier_curve_anchors.len().min(9).min(255) as u8;
        buf.push(anchor_len);
        for &a in self
            .bezier_curve_anchors
            .iter()
            .take(usize::from(anchor_len))
        {
            buf.extend_from_slice(&a.to_le_bytes());
        }

        buf
    }

    /// Parse a payload previously produced by `serialize`.
    ///
    /// # Errors
    /// Returns `HdrError::MetadataParseError` if the payload is too short
    /// or contains an inconsistency.
    pub fn parse(data: &[u8]) -> Result<Self> {
        // Minimum fixed-size header: 3 + 4 + 12 + 4 = 23 bytes, then at least 1 for dist_len
        const FIXED_HEADER: usize = 3 + 4 + 12 + 4;
        if data.len() < FIXED_HEADER + 1 {
            return Err(HdrError::MetadataParseError(format!(
                "ST 2094-10 payload too short: {} bytes (need at least {})",
                data.len(),
                FIXED_HEADER + 1
            )));
        }

        let application_identifier = data[0];
        let application_version = data[1];
        let num_windows = data[2];

        let mut off = 3usize;
        let targeted_system_display_maximum_luminance = read_u32(data, &mut off)?;

        let maxscl = [
            read_u32(data, &mut off)?,
            read_u32(data, &mut off)?,
            read_u32(data, &mut off)?,
        ];
        let average_maxrgb = read_u32(data, &mut off)?;

        let dist_len = read_u8(data, &mut off)? as usize;
        let mut distribution_maxrgb = Vec::with_capacity(dist_len);
        for _ in 0..dist_len {
            distribution_maxrgb.push(read_u32(data, &mut off)?);
        }

        let fraction_bright_pixels = read_u32(data, &mut off)?;
        let tone_mapping_flag = read_u8(data, &mut off)? != 0;
        let knee_point_x = read_u16(data, &mut off)?;
        let knee_point_y = read_u16(data, &mut off)?;

        let anchor_len = read_u8(data, &mut off)? as usize;
        let mut bezier_curve_anchors = Vec::with_capacity(anchor_len);
        for _ in 0..anchor_len {
            bezier_curve_anchors.push(read_u16(data, &mut off)?);
        }

        Ok(Self {
            application_identifier,
            application_version,
            num_windows,
            targeted_system_display_maximum_luminance,
            maxscl,
            average_maxrgb,
            distribution_maxrgb,
            fraction_bright_pixels,
            tone_mapping_flag,
            knee_point_x,
            knee_point_y,
            bezier_curve_anchors,
        })
    }
}

// ── ST 2094-40 (Dolby Vision / HDR10+ application variant) ───────────────────

/// Extension block type tags for ST 2094-40 payloads.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExtBlockType {
    /// Level 1 — per-frame statistics.
    Level1,
    /// Level 2 — per-shot trim controls.
    Level2,
    /// Level 3 — display-specific processing hint.
    Level3,
    /// Level 4 — per-display trim.
    Level4,
    /// Level 5 — aspect ratio.
    Level5,
    /// Level 6 — video endpoint compatibility metadata.
    Level6,
    /// Unknown/vendor extension.
    Unknown(u8),
}

impl ExtBlockType {
    fn from_tag(tag: u8) -> Self {
        match tag {
            1 => Self::Level1,
            2 => Self::Level2,
            3 => Self::Level3,
            4 => Self::Level4,
            5 => Self::Level5,
            6 => Self::Level6,
            other => Self::Unknown(other),
        }
    }

    fn to_tag(&self) -> u8 {
        match self {
            Self::Level1 => 1,
            Self::Level2 => 2,
            Self::Level3 => 3,
            Self::Level4 => 4,
            Self::Level5 => 5,
            Self::Level6 => 6,
            Self::Unknown(t) => *t,
        }
    }
}

/// A single extension block within an ST 2094-40 payload.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExtBlock {
    /// Block type identifier.
    pub block_type: ExtBlockType,
    /// Raw payload bytes for this block.
    pub payload: Vec<u8>,
}

impl ExtBlock {
    /// Create an extension block with an arbitrary payload.
    pub fn new(block_type: ExtBlockType, payload: Vec<u8>) -> Self {
        Self {
            block_type,
            payload,
        }
    }

    fn encode(&self) -> Vec<u8> {
        // Layout: 1 byte type tag, 2 bytes payload length (LE), then payload bytes.
        let mut out = Vec::with_capacity(3 + self.payload.len());
        out.push(self.block_type.to_tag());
        let len = self.payload.len().min(u16::MAX as usize) as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&self.payload[..len as usize]);
        out
    }

    fn decode(data: &[u8], off: &mut usize) -> Result<Self> {
        let tag = read_u8(data, off)?;
        let len = read_u16(data, off)? as usize;
        if *off + len > data.len() {
            return Err(HdrError::MetadataParseError(format!(
                "ExtBlock payload truncated: need {len} bytes at offset {}",
                *off
            )));
        }
        let payload = data[*off..*off + len].to_vec();
        *off += len;
        Ok(Self {
            block_type: ExtBlockType::from_tag(tag),
            payload,
        })
    }
}

/// SMPTE ST 2094-40 metadata (application_identifier = 7, HDR10+ Dolby variant).
///
/// Carries a variable number of typed extension blocks that encode per-scene
/// or per-frame display processing hints for Dolby Vision-compatible decoders.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct St2094_40Metadata {
    /// Application identifier (7 for Dolby Vision / HDR10+ variant).
    pub application_identifier: u8,
    /// Application version.
    pub application_version: u8,
    /// Number of extension blocks present.
    pub num_ext_blocks: u8,
    /// Extension blocks.
    pub ext_blocks: Vec<ExtBlock>,
}

impl St2094_40Metadata {
    /// Create a default empty ST 2094-40 metadata block.
    pub fn new_default() -> Self {
        Self {
            application_identifier: 7,
            application_version: 0,
            num_ext_blocks: 0,
            ext_blocks: Vec::new(),
        }
    }

    /// Add an extension block, updating `num_ext_blocks`.
    pub fn add_ext_block(&mut self, block: ExtBlock) {
        self.ext_blocks.push(block);
        self.num_ext_blocks = self.ext_blocks.len().min(255) as u8;
    }

    /// Serialise to bytes (little-endian).
    ///
    /// # Layout
    /// ```text
    /// [0] u8  application_identifier
    /// [1] u8  application_version
    /// [2] u8  num_ext_blocks
    /// [3..] N × ExtBlock (1 byte type + 2 byte len LE + payload)
    /// ```
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        buf.push(self.application_identifier);
        buf.push(self.application_version);
        let actual_len = self.ext_blocks.len().min(255) as u8;
        buf.push(actual_len);
        for block in self.ext_blocks.iter().take(usize::from(actual_len)) {
            buf.extend_from_slice(&block.encode());
        }
        buf
    }

    /// Parse a payload produced by `serialize`.
    ///
    /// # Errors
    /// Returns `HdrError::MetadataParseError` if the payload is malformed.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 3 {
            return Err(HdrError::MetadataParseError(format!(
                "ST 2094-40 payload too short: {} bytes",
                data.len()
            )));
        }
        let application_identifier = data[0];
        let application_version = data[1];
        let num_ext_blocks = data[2];

        let mut off = 3usize;
        let mut ext_blocks = Vec::with_capacity(usize::from(num_ext_blocks));
        for _ in 0..num_ext_blocks {
            ext_blocks.push(ExtBlock::decode(data, &mut off)?);
        }

        Ok(Self {
            application_identifier,
            application_version,
            num_ext_blocks,
            ext_blocks,
        })
    }
}

// ── Byte-reading helpers ──────────────────────────────────────────────────────

fn read_u8(data: &[u8], off: &mut usize) -> Result<u8> {
    if *off >= data.len() {
        return Err(HdrError::MetadataParseError(format!(
            "read_u8: offset {} out of range (len {})",
            *off,
            data.len()
        )));
    }
    let v = data[*off];
    *off += 1;
    Ok(v)
}

fn read_u16(data: &[u8], off: &mut usize) -> Result<u16> {
    if *off + 2 > data.len() {
        return Err(HdrError::MetadataParseError(format!(
            "read_u16: offset {} out of range (len {})",
            *off,
            data.len()
        )));
    }
    let v = u16::from_le_bytes([data[*off], data[*off + 1]]);
    *off += 2;
    Ok(v)
}

fn read_u32(data: &[u8], off: &mut usize) -> Result<u32> {
    if *off + 4 > data.len() {
        return Err(HdrError::MetadataParseError(format!(
            "read_u32: offset {} out of range (len {})",
            *off,
            data.len()
        )));
    }
    let v = u32::from_le_bytes([data[*off], data[*off + 1], data[*off + 2], data[*off + 3]]);
    *off += 4;
    Ok(v)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ST 2094-10 tests

    // 1. Default constructor sets expected application_identifier
    #[test]
    fn test_st2094_10_default_app_id() {
        let m = St2094_10Metadata::new_default();
        assert_eq!(m.application_identifier, 4);
    }

    // 2. Default constructor sets tone_mapping_flag
    #[test]
    fn test_st2094_10_default_tone_mapping_flag() {
        let m = St2094_10Metadata::new_default();
        assert!(m.tone_mapping_flag);
    }

    // 3. Round-trip: serialize then parse produces identical struct
    #[test]
    fn test_st2094_10_round_trip() {
        let orig = St2094_10Metadata::new_default();
        let bytes = orig.serialize();
        let parsed = St2094_10Metadata::parse(&bytes).expect("parse");
        assert_eq!(parsed, orig);
    }

    // 4. Parse with too-short buffer returns error
    #[test]
    fn test_st2094_10_parse_too_short() {
        assert!(St2094_10Metadata::parse(&[0u8; 5]).is_err());
    }

    // 5. Modified knee point survives round-trip
    #[test]
    fn test_st2094_10_knee_point_round_trip() {
        let mut m = St2094_10Metadata::new_default();
        m.knee_point_x = 1024;
        m.knee_point_y = 800;
        let bytes = m.serialize();
        let parsed = St2094_10Metadata::parse(&bytes).expect("knee round-trip");
        assert_eq!(parsed.knee_point_x, 1024);
        assert_eq!(parsed.knee_point_y, 800);
    }

    // 6. Custom distribution survives round-trip
    #[test]
    fn test_st2094_10_distribution_round_trip() {
        let mut m = St2094_10Metadata::new_default();
        m.distribution_maxrgb = vec![10, 20, 30];
        let bytes = m.serialize();
        let parsed = St2094_10Metadata::parse(&bytes).expect("dist round-trip");
        assert_eq!(parsed.distribution_maxrgb, vec![10u32, 20, 30]);
    }

    // 7. Empty distribution round-trips correctly
    #[test]
    fn test_st2094_10_empty_distribution_round_trip() {
        let mut m = St2094_10Metadata::new_default();
        m.distribution_maxrgb = Vec::new();
        let bytes = m.serialize();
        let parsed = St2094_10Metadata::parse(&bytes).expect("empty dist");
        assert!(parsed.distribution_maxrgb.is_empty());
    }

    // 8. tone_mapping_flag = false round-trips
    #[test]
    fn test_st2094_10_no_tone_mapping_flag() {
        let mut m = St2094_10Metadata::new_default();
        m.tone_mapping_flag = false;
        let bytes = m.serialize();
        let parsed = St2094_10Metadata::parse(&bytes).expect("no tm flag");
        assert!(!parsed.tone_mapping_flag);
    }

    // ST 2094-40 tests

    // 9. Default constructor sets application_identifier = 7
    #[test]
    fn test_st2094_40_default_app_id() {
        let m = St2094_40Metadata::new_default();
        assert_eq!(m.application_identifier, 7);
        assert_eq!(m.num_ext_blocks, 0);
        assert!(m.ext_blocks.is_empty());
    }

    // 10. Round-trip with no ext_blocks
    #[test]
    fn test_st2094_40_round_trip_empty() {
        let orig = St2094_40Metadata::new_default();
        let bytes = orig.serialize();
        let parsed = St2094_40Metadata::parse(&bytes).expect("empty 40");
        assert_eq!(parsed, orig);
    }

    // 11. Round-trip with one Level1 ext_block
    #[test]
    fn test_st2094_40_round_trip_one_block() {
        let mut m = St2094_40Metadata::new_default();
        m.add_ext_block(ExtBlock::new(ExtBlockType::Level1, vec![1, 2, 3, 4]));
        let bytes = m.serialize();
        let parsed = St2094_40Metadata::parse(&bytes).expect("one block");
        assert_eq!(parsed.num_ext_blocks, 1);
        assert_eq!(parsed.ext_blocks.len(), 1);
        assert_eq!(parsed.ext_blocks[0].block_type, ExtBlockType::Level1);
        assert_eq!(parsed.ext_blocks[0].payload, vec![1u8, 2, 3, 4]);
    }

    // 12. Round-trip with multiple ext_blocks of different levels
    #[test]
    fn test_st2094_40_round_trip_multi_block() {
        let mut m = St2094_40Metadata::new_default();
        m.add_ext_block(ExtBlock::new(ExtBlockType::Level2, vec![0xAA, 0xBB]));
        m.add_ext_block(ExtBlock::new(ExtBlockType::Level6, vec![0xFF]));
        let bytes = m.serialize();
        let parsed = St2094_40Metadata::parse(&bytes).expect("multi block");
        assert_eq!(parsed.num_ext_blocks, 2);
        assert_eq!(parsed.ext_blocks[0].block_type, ExtBlockType::Level2);
        assert_eq!(parsed.ext_blocks[1].block_type, ExtBlockType::Level6);
    }

    // 13. Unknown block type round-trips correctly
    #[test]
    fn test_st2094_40_unknown_block_round_trip() {
        let mut m = St2094_40Metadata::new_default();
        m.add_ext_block(ExtBlock::new(ExtBlockType::Unknown(99), vec![7, 8, 9]));
        let bytes = m.serialize();
        let parsed = St2094_40Metadata::parse(&bytes).expect("unknown block");
        assert_eq!(parsed.ext_blocks[0].block_type, ExtBlockType::Unknown(99));
    }

    // 14. Parse with too-short buffer returns error
    #[test]
    fn test_st2094_40_parse_too_short() {
        assert!(St2094_40Metadata::parse(&[0u8; 2]).is_err());
    }

    // 15. Parse with truncated ext_block payload returns error
    #[test]
    fn test_st2094_40_parse_truncated_block() {
        // Header says 1 ext_block, but the block claims 100 bytes of payload that aren't there
        let mut data = vec![7u8, 0u8, 1u8]; // app_id, ver, num_blocks=1
        data.push(1u8); // block type Level1
        data.extend_from_slice(&100u16.to_le_bytes()); // payload len = 100
        data.extend_from_slice(&[0u8; 10]); // only 10 bytes present
        assert!(St2094_40Metadata::parse(&data).is_err());
    }

    // Helper round-trip helpers

    // 16. ExtBlockType tag round-trip for all known levels
    #[test]
    fn test_ext_block_type_tags() {
        let types = [
            ExtBlockType::Level1,
            ExtBlockType::Level2,
            ExtBlockType::Level3,
            ExtBlockType::Level4,
            ExtBlockType::Level5,
            ExtBlockType::Level6,
            ExtBlockType::Unknown(42),
        ];
        for t in &types {
            let tag = t.to_tag();
            let roundtrip = ExtBlockType::from_tag(tag);
            assert_eq!(roundtrip, *t, "tag round-trip failed for {t:?}");
        }
    }
}
