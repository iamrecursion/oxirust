//! HDR mastering display metadata: HDR10 static metadata (SMPTE ST 2086),
//! content light level (CTA-861), and SEI NALU encoding/decoding helpers.

use crate::{HdrError, Result};

/// Identifies the HDR container/signalling format.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum HdrFormat {
    /// HDR10 static metadata (SMPTE ST 2086 + CTA-861).
    Hdr10,
    /// HDR10+ dynamic metadata (SMPTE ST 2094-40).
    Hdr10Plus,
    /// HLG broadcast profile (ARIB STD-B67).
    HlgBroadcast,
    /// Dolby Vision Profile 5 (IPTPQc2 single-layer).
    DolbyVisionProfile5,
    /// Unknown or proprietary format.
    Unknown,
}

/// Mastering display colour volume (SMPTE ST 2086).
///
/// All chromaticity coordinates are CIE 1931 xy.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HdrMasteringMetadata {
    /// Signalling format.
    pub format: HdrFormat,
    /// Red primary xy chromaticity.  Rec. 2020: (0.708, 0.292).
    pub primary_r: (f32, f32),
    /// Green primary xy chromaticity.  Rec. 2020: (0.170, 0.797).
    pub primary_g: (f32, f32),
    /// Blue primary xy chromaticity.  Rec. 2020: (0.131, 0.046).
    pub primary_b: (f32, f32),
    /// White point xy chromaticity.  D65: (0.3127, 0.3290).
    pub white_point: (f32, f32),
    /// Mastering display peak luminance in cd/m².
    pub max_luminance_nits: f32,
    /// Mastering display minimum luminance in cd/m².
    pub min_luminance_nits: f32,
}

impl HdrMasteringMetadata {
    /// Factory: Rec. 2020 primaries, D65 white point, HDR10 format.
    pub fn rec2020_hdr10(max_nits: f32) -> Self {
        Self {
            format: HdrFormat::Hdr10,
            primary_r: (0.708, 0.292),
            primary_g: (0.170, 0.797),
            primary_b: (0.131, 0.046),
            white_point: (0.3127, 0.3290),
            max_luminance_nits: max_nits,
            min_luminance_nits: 0.005,
        }
    }

    /// Factory: Rec. 2020 primaries, D65 white point, HLG broadcast format.
    pub fn hlg_broadcast() -> Self {
        Self {
            format: HdrFormat::HlgBroadcast,
            primary_r: (0.708, 0.292),
            primary_g: (0.170, 0.797),
            primary_b: (0.131, 0.046),
            white_point: (0.3127, 0.3290),
            max_luminance_nits: 1000.0,
            min_luminance_nits: 0.005,
        }
    }

    /// Encode mastering display metadata into a simplified SEI NALU payload.
    ///
    /// Layout (little-endian):
    /// - 1 byte  : format tag (0=Hdr10, 1=Hdr10Plus, 2=HlgBroadcast, 3=DolbyVision, 0xFF=Unknown)
    /// - 6 × u16 : primary_r x,y  primary_g x,y  primary_b x,y  (×50 000, rounded)
    /// - 2 × u16 : white_point x,y
    /// - 2 × u32 : max_luminance (×10 000), min_luminance (×10 000) as integers
    ///
    /// Total: 1 + 12 + 4 + 8 = 25 bytes.
    pub fn encode_sei(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(25);

        let format_tag: u8 = match self.format {
            HdrFormat::Hdr10 => 0,
            HdrFormat::Hdr10Plus => 1,
            HdrFormat::HlgBroadcast => 2,
            HdrFormat::DolbyVisionProfile5 => 3,
            HdrFormat::Unknown => 0xFF,
        };
        buf.push(format_tag);

        let encode_xy = |v: f32| -> u16 { (v * 50_000.0).round() as u16 };
        let encode_lum = |v: f32| -> u32 { (v * 10_000.0).round() as u32 };

        for (x, y) in [
            self.primary_r,
            self.primary_g,
            self.primary_b,
            self.white_point,
        ] {
            buf.extend_from_slice(&encode_xy(x).to_le_bytes());
            buf.extend_from_slice(&encode_xy(y).to_le_bytes());
        }

        buf.extend_from_slice(&encode_lum(self.max_luminance_nits).to_le_bytes());
        buf.extend_from_slice(&encode_lum(self.min_luminance_nits).to_le_bytes());

        buf
    }

    /// Decode a SEI payload produced by `encode_sei`.
    ///
    /// # Errors
    /// Returns `HdrError::MetadataParseError` if the buffer is too short or the format tag
    /// is unrecognised.
    pub fn decode_sei(data: &[u8]) -> Result<Self> {
        if data.len() < 25 {
            return Err(HdrError::MetadataParseError(format!(
                "SEI payload too short: {} bytes (need 25)",
                data.len()
            )));
        }

        let format = match data[0] {
            0 => HdrFormat::Hdr10,
            1 => HdrFormat::Hdr10Plus,
            2 => HdrFormat::HlgBroadcast,
            3 => HdrFormat::DolbyVisionProfile5,
            0xFF => HdrFormat::Unknown,
            other => {
                return Err(HdrError::MetadataParseError(format!(
                    "unknown format tag: 0x{other:02X}"
                )))
            }
        };

        let decode_xy = |bytes: &[u8]| -> f32 {
            let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
            raw as f32 / 50_000.0
        };
        let decode_lum = |bytes: &[u8]| -> f32 {
            let raw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            raw as f32 / 10_000.0
        };

        let primary_r = (decode_xy(&data[1..3]), decode_xy(&data[3..5]));
        let primary_g = (decode_xy(&data[5..7]), decode_xy(&data[7..9]));
        let primary_b = (decode_xy(&data[9..11]), decode_xy(&data[11..13]));
        let white_point = (decode_xy(&data[13..15]), decode_xy(&data[15..17]));
        let max_luminance_nits = decode_lum(&data[17..21]);
        let min_luminance_nits = decode_lum(&data[21..25]);

        Ok(Self {
            format,
            primary_r,
            primary_g,
            primary_b,
            white_point,
            max_luminance_nits,
            min_luminance_nits,
        })
    }
}

/// Content Light Level (CTA-861.3).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContentLightLevel {
    /// MaxCLL: maximum content light level across the entire content, in nits.
    pub max_cll_nits: u16,
    /// MaxFALL: maximum frame-average light level, in nits.
    pub max_fall_nits: u16,
}

impl ContentLightLevel {
    /// Construct a `ContentLightLevel` with explicit values.
    pub fn new(max_cll: u16, max_fall: u16) -> Self {
        Self {
            max_cll_nits: max_cll,
            max_fall_nits: max_fall,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_f32(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_rec2020_hdr10_format() {
        let m = HdrMasteringMetadata::rec2020_hdr10(4000.0);
        assert_eq!(m.format, HdrFormat::Hdr10);
        assert!(approx_f32(m.max_luminance_nits, 4000.0, 0.01));
    }

    #[test]
    fn test_rec2020_hdr10_primaries() {
        let m = HdrMasteringMetadata::rec2020_hdr10(1000.0);
        assert!(approx_f32(m.primary_r.0, 0.708, 0.001));
        assert!(approx_f32(m.primary_g.1, 0.797, 0.001));
        assert!(approx_f32(m.white_point.0, 0.3127, 0.0001));
    }

    #[test]
    fn test_hlg_broadcast_format() {
        let m = HdrMasteringMetadata::hlg_broadcast();
        assert_eq!(m.format, HdrFormat::HlgBroadcast);
        assert!(approx_f32(m.max_luminance_nits, 1000.0, 0.01));
    }

    #[test]
    fn test_sei_encode_decode_round_trip_hdr10() {
        let orig = HdrMasteringMetadata::rec2020_hdr10(4000.0);
        let sei = orig.encode_sei();
        assert_eq!(sei.len(), 25, "SEI payload length must be 25 bytes");
        let decoded = HdrMasteringMetadata::decode_sei(&sei).expect("decode_sei");
        assert_eq!(decoded.format, HdrFormat::Hdr10);
        assert!(approx_f32(decoded.max_luminance_nits, 4000.0, 0.01));
        assert!(approx_f32(decoded.primary_r.0, 0.708, 0.001));
    }

    #[test]
    fn test_sei_encode_decode_round_trip_hlg() {
        let orig = HdrMasteringMetadata::hlg_broadcast();
        let sei = orig.encode_sei();
        let decoded = HdrMasteringMetadata::decode_sei(&sei).expect("decode_sei hlg");
        assert_eq!(decoded.format, HdrFormat::HlgBroadcast);
        assert!(approx_f32(decoded.max_luminance_nits, 1000.0, 0.01));
    }

    #[test]
    fn test_sei_decode_too_short_error() {
        let result = HdrMasteringMetadata::decode_sei(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sei_decode_unknown_format_tag_error() {
        let mut buf = vec![0u8; 25];
        buf[0] = 0xAB; // unrecognised tag
        let result = HdrMasteringMetadata::decode_sei(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_sei_decode_unknown_format_tag_ff() {
        let orig = HdrMasteringMetadata {
            format: HdrFormat::Unknown,
            primary_r: (0.64, 0.33),
            primary_g: (0.30, 0.60),
            primary_b: (0.15, 0.06),
            white_point: (0.3127, 0.3290),
            max_luminance_nits: 100.0,
            min_luminance_nits: 0.1,
        };
        let sei = orig.encode_sei();
        let decoded = HdrMasteringMetadata::decode_sei(&sei).expect("decode unknown");
        assert_eq!(decoded.format, HdrFormat::Unknown);
    }

    #[test]
    fn test_content_light_level_new() {
        let cll = ContentLightLevel::new(1000, 400);
        assert_eq!(cll.max_cll_nits, 1000);
        assert_eq!(cll.max_fall_nits, 400);
    }

    #[test]
    fn test_min_luminance_round_trip() {
        let orig = HdrMasteringMetadata::rec2020_hdr10(4000.0);
        let sei = orig.encode_sei();
        let decoded = HdrMasteringMetadata::decode_sei(&sei).expect("decode_sei min");
        assert!(approx_f32(decoded.min_luminance_nits, 0.005, 0.001));
    }
}
