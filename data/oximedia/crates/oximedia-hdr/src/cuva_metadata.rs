//! CUVA (China Ultra-High Definition Video Association) HDR metadata.
//!
//! Implements the metadata structures defined in T/UHD 004, China's national
//! HDR standard.  The binary SEI layout used here follows the bitstream
//! semantics of the standard while keeping the on-wire format self-contained
//! and unambiguous.

use crate::{HdrError, Result};

// ── Picture type ──────────────────────────────────────────────────────────────

/// Frame classification used by the CUVA tone-mapping engine.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CuvaPictureType {
    /// Ordinary picture — no special signalling.
    Normal,
    /// Scene-change frame — the tone-mapping state should be reset.
    SceneChange,
    /// Cross-fade / dissolve transition.
    Fade,
}

impl CuvaPictureType {
    fn as_u8(&self) -> u8 {
        match self {
            CuvaPictureType::Normal => 0,
            CuvaPictureType::SceneChange => 1,
            CuvaPictureType::Fade => 2,
        }
    }

    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(CuvaPictureType::Normal),
            1 => Ok(CuvaPictureType::SceneChange),
            2 => Ok(CuvaPictureType::Fade),
            other => Err(HdrError::MetadataParseError(format!(
                "unknown CUVA picture_type: {other}"
            ))),
        }
    }
}

// ── Extended white point ───────────────────────────────────────────────────────

/// Optional extended white-point signalling carried in CUVA SEI.
///
/// Coordinates are CIE 1931 xy, scaled by 50 000 (same convention as HDR10
/// mastering display metadata).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CuvaWhitepoint {
    /// CIE x chromaticity × 50 000.
    pub white_x: u16,
    /// CIE y chromaticity × 50 000.
    pub white_y: u16,
}

// ── Tone-mapping parameters ───────────────────────────────────────────────────

/// Bezier-curve knee-point and anchor parameters for CUVA tone mapping.
///
/// The curve is defined in a normalized [0, 65535] → [0, 65535] space where
/// 0 maps to the minimum display luminance and 65535 maps to the peak.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CuvaToneMapParams {
    /// Input (scene) signal level at the knee point, normalized 0–65535.
    pub knee_point_x: u16,
    /// Output (display) signal level at the knee point, normalized 0–65535.
    pub knee_point_y: u16,
    /// Number of additional Bezier curve anchor points (0–9).
    pub bezier_curve_num: u8,
    /// Anchor y-values in normalized [0, 65535] space.  Length must equal
    /// `bezier_curve_num`.
    pub bezier_curve_anchors: Vec<u16>,
}

impl CuvaToneMapParams {
    /// Identity (pass-through) tone-map params — knee point at midpoint,
    /// no additional anchors.
    pub fn identity() -> Self {
        Self {
            knee_point_x: 32_768,
            knee_point_y: 32_768,
            bezier_curve_num: 0,
            bezier_curve_anchors: Vec::new(),
        }
    }
}

// ── Main metadata struct ──────────────────────────────────────────────────────

/// CUVA HDR frame metadata (Chinese Ultra-HD standard T/UHD 004).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CuvaMetadata {
    /// System start code; shall be `0xC0` for CUVA streams.
    pub system_start_code: u8,
    /// Metadata version; typically `1`.
    pub version: u8,
    /// Per-frame picture classification.
    pub picture_type: CuvaPictureType,
    /// Peak luminance of the mastering display in units of 0.0001 cd/m².
    ///
    /// Example: `10_000_000` represents 1 000 nits.
    pub max_luminance: u32,
    /// Minimum luminance of the mastering display in units of 0.0001 cd/m².
    pub min_luminance: u32,
    /// Optional extended white-point coordinates.
    pub extended_whitepoint: Option<CuvaWhitepoint>,
    /// Tone-mapping curve parameters.
    pub tone_mapping_params: CuvaToneMapParams,
}

// ── SEI binary layout ─────────────────────────────────────────────────────────
//
// Fixed header (12 bytes):
//   [0]      system_start_code   u8
//   [1]      version             u8
//   [2]      picture_type        u8   (0/1/2)
//   [3..6]   max_luminance       u32 LE
//   [7..10]  min_luminance       u32 LE
//   [11]     has_whitepoint      u8   (0 or 1)
//
// Conditional whitepoint (4 bytes if has_whitepoint == 1):
//   [12..13] white_x             u16 LE
//   [14..15] white_y             u16 LE
//
// Tone-map params (dynamic, starts at offset 12 or 16):
//   [+0..1]  knee_point_x        u16 LE
//   [+2..3]  knee_point_y        u16 LE
//   [+4]     bezier_curve_num    u8
//   [+5 .. +5+N*2-1]  anchors   N × u16 LE
//
// Minimum size (no whitepoint, 0 anchors): 12 + 5 = 17 bytes.

const SYSTEM_START_CODE: u8 = 0xC0;
const MAX_BEZIER_ANCHORS: u8 = 9;

impl CuvaMetadata {
    /// Construct a default CUVA metadata instance suitable for 1 000-nit HDR
    /// content with an identity tone-map curve and no extended white point.
    pub fn new_default() -> Self {
        Self {
            system_start_code: SYSTEM_START_CODE,
            version: 1,
            picture_type: CuvaPictureType::Normal,
            max_luminance: 10_000_000, // 1 000 nits in 0.0001-nit units
            min_luminance: 5,          // 0.0005 nits
            extended_whitepoint: None,
            tone_mapping_params: CuvaToneMapParams::identity(),
        }
    }

    /// Serialize this metadata into a HEVC unregistered-user-data SEI payload.
    ///
    /// The payload begins with `system_start_code` and is self-describing;
    /// it can be round-tripped through [`CuvaMetadata::parse_sei`].
    pub fn serialize_sei(&self) -> Vec<u8> {
        let anchor_bytes = self.tone_mapping_params.bezier_curve_anchors.len() * 2;
        let has_wp = self.extended_whitepoint.is_some();
        let capacity = 12 + if has_wp { 4 } else { 0 } + 5 + anchor_bytes;
        let mut buf = Vec::with_capacity(capacity);

        // Fixed header
        buf.push(self.system_start_code);
        buf.push(self.version);
        buf.push(self.picture_type.as_u8());
        buf.extend_from_slice(&self.max_luminance.to_le_bytes());
        buf.extend_from_slice(&self.min_luminance.to_le_bytes());
        buf.push(if has_wp { 1u8 } else { 0u8 });

        // Optional extended whitepoint
        if let Some(ref wp) = self.extended_whitepoint {
            buf.extend_from_slice(&wp.white_x.to_le_bytes());
            buf.extend_from_slice(&wp.white_y.to_le_bytes());
        }

        // Tone-map params
        buf.extend_from_slice(&self.tone_mapping_params.knee_point_x.to_le_bytes());
        buf.extend_from_slice(&self.tone_mapping_params.knee_point_y.to_le_bytes());
        buf.push(self.tone_mapping_params.bezier_curve_num);
        for anchor in &self.tone_mapping_params.bezier_curve_anchors {
            buf.extend_from_slice(&anchor.to_le_bytes());
        }

        buf
    }

    /// Parse a CUVA SEI payload produced by [`CuvaMetadata::serialize_sei`].
    ///
    /// # Errors
    /// Returns [`HdrError::MetadataParseError`] if:
    /// - The buffer is shorter than the minimum required length (17 bytes).
    /// - `system_start_code` is not `0xC0`.
    /// - `picture_type` is not one of 0, 1, or 2.
    /// - `bezier_curve_num` exceeds 9.
    /// - The buffer is too short to contain all declared anchors.
    pub fn parse_sei(data: &[u8]) -> Result<Self> {
        // Minimum: 12-byte fixed header + 5-byte tone-map (no wp, 0 anchors)
        const MIN_LEN: usize = 17;
        if data.len() < MIN_LEN {
            return Err(HdrError::MetadataParseError(format!(
                "CUVA SEI too short: {} bytes (need at least {MIN_LEN})",
                data.len()
            )));
        }

        let system_start_code = data[0];
        if system_start_code != SYSTEM_START_CODE {
            return Err(HdrError::MetadataParseError(format!(
                "invalid CUVA system_start_code: 0x{system_start_code:02X} (expected 0xC0)"
            )));
        }

        let version = data[1];
        let picture_type = CuvaPictureType::from_u8(data[2])?;
        let max_luminance = u32::from_le_bytes([data[3], data[4], data[5], data[6]]);
        let min_luminance = u32::from_le_bytes([data[7], data[8], data[9], data[10]]);
        let has_wp = data[11] != 0;

        let mut offset = 12usize;

        let extended_whitepoint = if has_wp {
            if data.len() < offset + 4 {
                return Err(HdrError::MetadataParseError(
                    "CUVA SEI truncated before extended whitepoint".to_string(),
                ));
            }
            let white_x = u16::from_le_bytes([data[offset], data[offset + 1]]);
            let white_y = u16::from_le_bytes([data[offset + 2], data[offset + 3]]);
            offset += 4;
            Some(CuvaWhitepoint { white_x, white_y })
        } else {
            None
        };

        // Tone-map params
        if data.len() < offset + 5 {
            return Err(HdrError::MetadataParseError(
                "CUVA SEI truncated before tone-map params".to_string(),
            ));
        }
        let knee_point_x = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let knee_point_y = u16::from_le_bytes([data[offset + 2], data[offset + 3]]);
        let bezier_curve_num = data[offset + 4];
        offset += 5;

        if bezier_curve_num > MAX_BEZIER_ANCHORS {
            return Err(HdrError::MetadataParseError(format!(
                "CUVA bezier_curve_num {bezier_curve_num} exceeds maximum of {MAX_BEZIER_ANCHORS}"
            )));
        }

        let anchor_count = bezier_curve_num as usize;
        if data.len() < offset + anchor_count * 2 {
            return Err(HdrError::MetadataParseError(format!(
                "CUVA SEI truncated: need {} bytes for {anchor_count} anchors, have {}",
                offset + anchor_count * 2,
                data.len()
            )));
        }

        let mut bezier_curve_anchors = Vec::with_capacity(anchor_count);
        for i in 0..anchor_count {
            let base = offset + i * 2;
            bezier_curve_anchors.push(u16::from_le_bytes([data[base], data[base + 1]]));
        }

        Ok(CuvaMetadata {
            system_start_code,
            version,
            picture_type,
            max_luminance,
            min_luminance,
            extended_whitepoint,
            tone_mapping_params: CuvaToneMapParams {
                knee_point_x,
                knee_point_y,
                bezier_curve_num,
                bezier_curve_anchors,
            },
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── new_default field values ──────────────────────────────────────────────

    #[test]
    fn test_new_default_system_start_code() {
        let m = CuvaMetadata::new_default();
        assert_eq!(m.system_start_code, 0xC0);
    }

    #[test]
    fn test_new_default_version() {
        let m = CuvaMetadata::new_default();
        assert_eq!(m.version, 1);
    }

    #[test]
    fn test_new_default_picture_type() {
        let m = CuvaMetadata::new_default();
        assert_eq!(m.picture_type, CuvaPictureType::Normal);
    }

    #[test]
    fn test_new_default_max_luminance() {
        let m = CuvaMetadata::new_default();
        assert_eq!(m.max_luminance, 10_000_000);
    }

    #[test]
    fn test_new_default_min_luminance() {
        let m = CuvaMetadata::new_default();
        assert_eq!(m.min_luminance, 5);
    }

    #[test]
    fn test_new_default_no_whitepoint() {
        let m = CuvaMetadata::new_default();
        assert!(m.extended_whitepoint.is_none());
    }

    #[test]
    fn test_new_default_tone_map_identity() {
        let m = CuvaMetadata::new_default();
        assert_eq!(m.tone_mapping_params.knee_point_x, 32_768);
        assert_eq!(m.tone_mapping_params.knee_point_y, 32_768);
        assert_eq!(m.tone_mapping_params.bezier_curve_num, 0);
        assert!(m.tone_mapping_params.bezier_curve_anchors.is_empty());
    }

    // ── Round-trip: no whitepoint, no anchors ─────────────────────────────────

    #[test]
    fn test_round_trip_no_whitepoint() {
        let orig = CuvaMetadata::new_default();
        let sei = orig.serialize_sei();
        let parsed = CuvaMetadata::parse_sei(&sei).expect("parse no-wp");
        assert_eq!(parsed.system_start_code, 0xC0);
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.picture_type, CuvaPictureType::Normal);
        assert_eq!(parsed.max_luminance, 10_000_000);
        assert_eq!(parsed.min_luminance, 5);
        assert!(parsed.extended_whitepoint.is_none());
        assert_eq!(parsed.tone_mapping_params.knee_point_x, 32_768);
        assert_eq!(parsed.tone_mapping_params.bezier_curve_num, 0);
    }

    // ── Round-trip: with extended whitepoint ──────────────────────────────────

    #[test]
    fn test_round_trip_with_whitepoint() {
        let orig = CuvaMetadata {
            system_start_code: 0xC0,
            version: 1,
            picture_type: CuvaPictureType::SceneChange,
            max_luminance: 4_000_000,
            min_luminance: 10,
            extended_whitepoint: Some(CuvaWhitepoint {
                white_x: 15_635, // D65 x ≈ 0.3127 × 50000
                white_y: 16_450, // D65 y ≈ 0.3290 × 50000
            }),
            tone_mapping_params: CuvaToneMapParams::identity(),
        };
        let sei = orig.serialize_sei();
        let parsed = CuvaMetadata::parse_sei(&sei).expect("parse with wp");
        assert_eq!(parsed.picture_type, CuvaPictureType::SceneChange);
        let wp = parsed.extended_whitepoint.expect("should have whitepoint");
        assert_eq!(wp.white_x, 15_635);
        assert_eq!(wp.white_y, 16_450);
    }

    // ── Round-trip: with Bezier anchors ───────────────────────────────────────

    #[test]
    fn test_round_trip_with_bezier_anchors() {
        let anchors: Vec<u16> = vec![8_000, 24_000, 49_000];
        let orig = CuvaMetadata {
            system_start_code: 0xC0,
            version: 1,
            picture_type: CuvaPictureType::Fade,
            max_luminance: 6_000_000,
            min_luminance: 3,
            extended_whitepoint: None,
            tone_mapping_params: CuvaToneMapParams {
                knee_point_x: 40_000,
                knee_point_y: 30_000,
                bezier_curve_num: 3,
                bezier_curve_anchors: anchors.clone(),
            },
        };
        let sei = orig.serialize_sei();
        let parsed = CuvaMetadata::parse_sei(&sei).expect("parse with anchors");
        assert_eq!(parsed.picture_type, CuvaPictureType::Fade);
        assert_eq!(parsed.tone_mapping_params.knee_point_x, 40_000);
        assert_eq!(parsed.tone_mapping_params.knee_point_y, 30_000);
        assert_eq!(parsed.tone_mapping_params.bezier_curve_num, 3);
        assert_eq!(parsed.tone_mapping_params.bezier_curve_anchors, anchors);
    }

    // ── Error: too-short buffer ───────────────────────────────────────────────

    #[test]
    fn test_parse_sei_too_short() {
        let result = CuvaMetadata::parse_sei(&[0xC0u8; 10]);
        assert!(result.is_err(), "should reject buffer shorter than minimum");
    }

    #[test]
    fn test_parse_sei_empty() {
        let result = CuvaMetadata::parse_sei(&[]);
        assert!(result.is_err());
    }

    // ── Error: wrong system_start_code ────────────────────────────────────────

    #[test]
    fn test_parse_sei_wrong_start_code() {
        let mut buf = CuvaMetadata::new_default().serialize_sei();
        buf[0] = 0xAB; // not 0xC0
        let result = CuvaMetadata::parse_sei(&buf);
        assert!(result.is_err(), "should reject wrong system_start_code");
    }

    // ── Error: unknown picture_type ───────────────────────────────────────────

    #[test]
    fn test_parse_sei_unknown_picture_type() {
        let mut buf = CuvaMetadata::new_default().serialize_sei();
        buf[2] = 0xFF; // no such picture type
        let result = CuvaMetadata::parse_sei(&buf);
        assert!(result.is_err(), "should reject unknown picture_type");
    }

    // ── Error: bezier_curve_num > 9 ───────────────────────────────────────────

    #[test]
    fn test_parse_sei_too_many_anchors() {
        // Build a valid payload then forcibly set bezier_curve_num = 10
        let orig = CuvaMetadata::new_default();
        let mut buf = orig.serialize_sei();
        // bezier_curve_num is at offset 12 (no whitepoint) + 4 (knee xy) + 0 = 16
        let anchor_offset = 12 + 4; // offset 16
        buf[anchor_offset] = 10; // bezier_curve_num = 10 (exceeds max of 9)
        let result = CuvaMetadata::parse_sei(&buf);
        assert!(result.is_err(), "should reject bezier_curve_num > 9");
    }

    // ── Picture type encoding ─────────────────────────────────────────────────

    #[test]
    fn test_picture_type_round_trip_all_variants() {
        for pt in [
            CuvaPictureType::Normal,
            CuvaPictureType::SceneChange,
            CuvaPictureType::Fade,
        ] {
            let m = CuvaMetadata {
                system_start_code: 0xC0,
                version: 1,
                picture_type: pt.clone(),
                max_luminance: 10_000_000,
                min_luminance: 5,
                extended_whitepoint: None,
                tone_mapping_params: CuvaToneMapParams::identity(),
            };
            let sei = m.serialize_sei();
            let parsed = CuvaMetadata::parse_sei(&sei).expect("parse picture type variant");
            assert_eq!(parsed.picture_type, pt);
        }
    }

    // ── Luminance boundary values ─────────────────────────────────────────────

    #[test]
    fn test_luminance_max_u32_round_trip() {
        let orig = CuvaMetadata {
            system_start_code: 0xC0,
            version: 1,
            picture_type: CuvaPictureType::Normal,
            max_luminance: u32::MAX,
            min_luminance: 0,
            extended_whitepoint: None,
            tone_mapping_params: CuvaToneMapParams::identity(),
        };
        let sei = orig.serialize_sei();
        let parsed = CuvaMetadata::parse_sei(&sei).expect("parse u32::MAX luminance");
        assert_eq!(parsed.max_luminance, u32::MAX);
        assert_eq!(parsed.min_luminance, 0);
    }

    // ── Nine anchors (maximum allowed) ────────────────────────────────────────

    #[test]
    fn test_round_trip_nine_anchors() {
        let anchors: Vec<u16> = (0..9).map(|i| i * 7_281).collect();
        let orig = CuvaMetadata {
            system_start_code: 0xC0,
            version: 1,
            picture_type: CuvaPictureType::Normal,
            max_luminance: 10_000_000,
            min_luminance: 5,
            extended_whitepoint: None,
            tone_mapping_params: CuvaToneMapParams {
                knee_point_x: 32_768,
                knee_point_y: 32_768,
                bezier_curve_num: 9,
                bezier_curve_anchors: anchors.clone(),
            },
        };
        let sei = orig.serialize_sei();
        let parsed = CuvaMetadata::parse_sei(&sei).expect("parse 9 anchors");
        assert_eq!(parsed.tone_mapping_params.bezier_curve_num, 9);
        assert_eq!(parsed.tone_mapping_params.bezier_curve_anchors, anchors);
    }
}
