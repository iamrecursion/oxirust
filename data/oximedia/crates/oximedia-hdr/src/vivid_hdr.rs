//! Samsung SL-HDR1 / VividHDR metadata.
//!
//! SL-HDR1 is the scene-linear HDR system used in Samsung displays and
//! encoded as an unregistered-user-data SEI NAL unit in H.265/HEVC streams.
//! This module provides the metadata structures, a 34-byte binary SEI
//! serializer, and a matching parser.

use crate::{HdrError, Result};

// ── EOTF enum ─────────────────────────────────────────────────────────────────

/// Electro-optical transfer function signalled in the SL-HDR1 SEI.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum VividEotf {
    /// Standard dynamic range (BT.1886 or sRGB).
    Sdr,
    /// Hybrid log-gamma (BT.2100 HLG).
    Hlg,
    /// Perceptual quantizer (SMPTE ST 2084 / BT.2100 PQ).
    Pq,
    /// Reserved for future EOTF definitions.
    Future,
}

impl VividEotf {
    fn as_u8(&self) -> u8 {
        match self {
            VividEotf::Sdr => 0,
            VividEotf::Hlg => 1,
            VividEotf::Pq => 2,
            VividEotf::Future => 3,
        }
    }

    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(VividEotf::Sdr),
            1 => Ok(VividEotf::Hlg),
            2 => Ok(VividEotf::Pq),
            3 => Ok(VividEotf::Future),
            other => Err(HdrError::MetadataParseError(format!(
                "unknown VividEotf byte: {other}"
            ))),
        }
    }
}

// ── Tone-map parameters ───────────────────────────────────────────────────────

/// Display capabilities and tone-mapping parameters for SL-HDR1.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VividToneMapParams {
    /// Maximum display peak luminance in nits (cd/m²).
    pub max_display_luminance: f32,
    /// Minimum display luminance in nits (cd/m²).
    pub min_display_luminance: f32,
    /// Multiplicative tone-mapping gain factor applied to the scene signal.
    pub tm_gain: f32,
    /// Additive black-level lift applied after tone mapping.
    pub tm_black_level: f32,
}

// ── Main metadata struct ──────────────────────────────────────────────────────

/// SL-HDR1 (Samsung VividHDR) frame metadata.
///
/// Carried as a 34-byte HEVC unregistered-user-data SEI payload.  When
/// `metadata_present` is `false` the tone-map parameters are still serialized
/// but a decoder may ignore them.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VividHdrMetadata {
    /// Transfer function of the encoded video signal.
    pub eotf: VividEotf,
    /// Indicates that the optional luminance metadata fields are meaningful.
    pub metadata_present: bool,
    /// Scene maximum luminance in nits.
    pub scene_max_luminance: f32,
    /// Scene average (mean) luminance in nits.
    pub scene_average_luminance: f32,
    /// Scene black level in nits.
    pub scene_black_level: f32,
    /// Tone-mapping curve parameters for the target display.
    pub tone_map_params: VividToneMapParams,
}

// ── SEI binary layout ─────────────────────────────────────────────────────────
//
// Fixed 34-byte payload (all multi-byte values little-endian):
//   [0]      eotf                   u8
//   [1]      metadata_present       u8  (1 = true, 0 = false)
//   [2..5]   scene_max_luminance    f32 LE (IEEE 754)
//   [6..9]   scene_average_lum      f32 LE
//   [10..13] scene_black_level      f32 LE
//   [14..17] max_display_luminance  f32 LE
//   [18..21] min_display_luminance  f32 LE
//   [22..25] tm_gain                f32 LE
//   [26..29] tm_black_level         f32 LE
//   [30..33] reserved               u32 LE (write 0, ignore on read)

const SEI_LEN: usize = 34;

impl VividHdrMetadata {
    /// Construct SL-HDR1 metadata pre-configured for 10-bit PQ HDR10 content.
    ///
    /// - EOTF: PQ (SMPTE ST 2084)
    /// - Target display: 1 000 nits peak / 0.005 nits black
    /// - Tone-mapping gain: 1.0 (unity), black lift: 0.0
    pub fn new_for_hdr10(max_nits: f32, avg_nits: f32) -> Self {
        Self {
            eotf: VividEotf::Pq,
            metadata_present: true,
            scene_max_luminance: max_nits,
            scene_average_luminance: avg_nits,
            scene_black_level: 0.005,
            tone_map_params: VividToneMapParams {
                max_display_luminance: 1_000.0,
                min_display_luminance: 0.005,
                tm_gain: 1.0,
                tm_black_level: 0.0,
            },
        }
    }

    /// Serialize to a 34-byte HEVC SEI payload.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(SEI_LEN);

        buf.push(self.eotf.as_u8());
        buf.push(if self.metadata_present { 1u8 } else { 0u8 });
        buf.extend_from_slice(&self.scene_max_luminance.to_le_bytes());
        buf.extend_from_slice(&self.scene_average_luminance.to_le_bytes());
        buf.extend_from_slice(&self.scene_black_level.to_le_bytes());
        buf.extend_from_slice(&self.tone_map_params.max_display_luminance.to_le_bytes());
        buf.extend_from_slice(&self.tone_map_params.min_display_luminance.to_le_bytes());
        buf.extend_from_slice(&self.tone_map_params.tm_gain.to_le_bytes());
        buf.extend_from_slice(&self.tone_map_params.tm_black_level.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved

        debug_assert_eq!(buf.len(), SEI_LEN);
        buf
    }

    /// Parse a 34-byte SL-HDR1 SEI payload.
    ///
    /// # Errors
    /// Returns [`HdrError::MetadataParseError`] if:
    /// - `data.len() < 34`
    /// - The EOTF byte is not in the range 0–3.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < SEI_LEN {
            return Err(HdrError::MetadataParseError(format!(
                "VividHDR SEI too short: {} bytes (need {SEI_LEN})",
                data.len()
            )));
        }

        let eotf = VividEotf::from_u8(data[0])?;
        let metadata_present = data[1] != 0;

        let read_f32 = |offset: usize| -> f32 {
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        };

        let scene_max_luminance = read_f32(2);
        let scene_average_luminance = read_f32(6);
        let scene_black_level = read_f32(10);
        let max_display_luminance = read_f32(14);
        let min_display_luminance = read_f32(18);
        let tm_gain = read_f32(22);
        let tm_black_level = read_f32(26);
        // bytes 30-33 are reserved; intentionally ignored

        Ok(VividHdrMetadata {
            eotf,
            metadata_present,
            scene_max_luminance,
            scene_average_luminance,
            scene_black_level,
            tone_map_params: VividToneMapParams {
                max_display_luminance,
                min_display_luminance,
                tm_gain,
                tm_black_level,
            },
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    // ── new_for_hdr10 field values ────────────────────────────────────────────

    #[test]
    fn test_new_for_hdr10_eotf() {
        let m = VividHdrMetadata::new_for_hdr10(4000.0, 300.0);
        assert_eq!(m.eotf, VividEotf::Pq);
    }

    #[test]
    fn test_new_for_hdr10_metadata_present() {
        let m = VividHdrMetadata::new_for_hdr10(4000.0, 300.0);
        assert!(m.metadata_present);
    }

    #[test]
    fn test_new_for_hdr10_luminances() {
        let m = VividHdrMetadata::new_for_hdr10(4000.0, 300.0);
        assert!(approx(m.scene_max_luminance, 4000.0));
        assert!(approx(m.scene_average_luminance, 300.0));
        assert!(approx(m.scene_black_level, 0.005));
    }

    #[test]
    fn test_new_for_hdr10_tone_map_defaults() {
        let m = VividHdrMetadata::new_for_hdr10(1000.0, 100.0);
        assert!(approx(m.tone_map_params.max_display_luminance, 1000.0));
        assert!(approx(m.tone_map_params.min_display_luminance, 0.005));
        assert!(approx(m.tone_map_params.tm_gain, 1.0));
        assert!(approx(m.tone_map_params.tm_black_level, 0.0));
    }

    // ── serialize length ──────────────────────────────────────────────────────

    #[test]
    fn test_serialize_length_is_34() {
        let m = VividHdrMetadata::new_for_hdr10(1000.0, 200.0);
        let bytes = m.serialize();
        assert_eq!(
            bytes.len(),
            34,
            "SL-HDR1 SEI payload must be exactly 34 bytes"
        );
    }

    // ── Round-trip: PQ ────────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_pq() {
        let orig = VividHdrMetadata::new_for_hdr10(4000.0, 350.0);
        let bytes = orig.serialize();
        let parsed = VividHdrMetadata::parse(&bytes).expect("parse PQ round-trip");
        assert_eq!(parsed.eotf, VividEotf::Pq);
        assert!(parsed.metadata_present);
        assert!(approx(parsed.scene_max_luminance, 4000.0));
        assert!(approx(parsed.scene_average_luminance, 350.0));
        assert!(approx(parsed.scene_black_level, 0.005));
        assert!(approx(parsed.tone_map_params.max_display_luminance, 1000.0));
        assert!(approx(parsed.tone_map_params.tm_gain, 1.0));
    }

    // ── Round-trip: HLG ───────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_hlg() {
        let orig = VividHdrMetadata {
            eotf: VividEotf::Hlg,
            metadata_present: true,
            scene_max_luminance: 1000.0,
            scene_average_luminance: 200.0,
            scene_black_level: 0.01,
            tone_map_params: VividToneMapParams {
                max_display_luminance: 1000.0,
                min_display_luminance: 0.01,
                tm_gain: 0.9,
                tm_black_level: 0.005,
            },
        };
        let bytes = orig.serialize();
        let parsed = VividHdrMetadata::parse(&bytes).expect("parse HLG round-trip");
        assert_eq!(parsed.eotf, VividEotf::Hlg);
        assert!(approx(parsed.scene_max_luminance, 1000.0));
        assert!(approx(parsed.tone_map_params.tm_gain, 0.9));
    }

    // ── Round-trip: metadata_present = false ──────────────────────────────────

    #[test]
    fn test_round_trip_metadata_not_present() {
        let orig = VividHdrMetadata {
            eotf: VividEotf::Sdr,
            metadata_present: false,
            scene_max_luminance: 100.0,
            scene_average_luminance: 80.0,
            scene_black_level: 0.1,
            tone_map_params: VividToneMapParams {
                max_display_luminance: 300.0,
                min_display_luminance: 0.1,
                tm_gain: 1.0,
                tm_black_level: 0.0,
            },
        };
        let bytes = orig.serialize();
        let parsed = VividHdrMetadata::parse(&bytes).expect("parse metadata_present=false");
        assert_eq!(parsed.eotf, VividEotf::Sdr);
        assert!(!parsed.metadata_present);
        assert!(approx(parsed.scene_max_luminance, 100.0));
    }

    // ── Round-trip: Future EOTF ───────────────────────────────────────────────

    #[test]
    fn test_round_trip_future_eotf() {
        let orig = VividHdrMetadata {
            eotf: VividEotf::Future,
            metadata_present: false,
            scene_max_luminance: 0.0,
            scene_average_luminance: 0.0,
            scene_black_level: 0.0,
            tone_map_params: VividToneMapParams {
                max_display_luminance: 0.0,
                min_display_luminance: 0.0,
                tm_gain: 1.0,
                tm_black_level: 0.0,
            },
        };
        let bytes = orig.serialize();
        let parsed = VividHdrMetadata::parse(&bytes).expect("parse Future eotf");
        assert_eq!(parsed.eotf, VividEotf::Future);
    }

    // ── Error: too-short buffer ───────────────────────────────────────────────

    #[test]
    fn test_parse_too_short() {
        let result = VividHdrMetadata::parse(&[0u8; 20]);
        assert!(
            result.is_err(),
            "should reject buffer shorter than 34 bytes"
        );
    }

    #[test]
    fn test_parse_empty() {
        let result = VividHdrMetadata::parse(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_exactly_33_bytes_rejected() {
        let result = VividHdrMetadata::parse(&[0u8; 33]);
        assert!(result.is_err());
    }

    // ── Error: unknown EOTF byte ──────────────────────────────────────────────

    #[test]
    fn test_parse_unknown_eotf() {
        let mut buf = VividHdrMetadata::new_for_hdr10(1000.0, 100.0).serialize();
        buf[0] = 0xFF; // no such EOTF
        let result = VividHdrMetadata::parse(&buf);
        assert!(result.is_err(), "should reject unknown EOTF byte");
    }

    #[test]
    fn test_parse_eotf_byte_4_unknown() {
        let mut buf = VividHdrMetadata::new_for_hdr10(1000.0, 100.0).serialize();
        buf[0] = 4; // one past Future
        let result = VividHdrMetadata::parse(&buf);
        assert!(result.is_err());
    }

    // ── Payload byte 30-33 reserved field is zeroed ───────────────────────────

    #[test]
    fn test_reserved_bytes_are_zero() {
        let m = VividHdrMetadata::new_for_hdr10(1000.0, 200.0);
        let bytes = m.serialize();
        assert_eq!(&bytes[30..34], &[0u8; 4], "reserved bytes must be zero");
    }

    // ── All four EOTF variants encode/decode correctly ────────────────────────

    #[test]
    fn test_all_eotf_variants_round_trip() {
        for eotf in [
            VividEotf::Sdr,
            VividEotf::Hlg,
            VividEotf::Pq,
            VividEotf::Future,
        ] {
            let m = VividHdrMetadata {
                eotf: eotf.clone(),
                metadata_present: true,
                scene_max_luminance: 500.0,
                scene_average_luminance: 100.0,
                scene_black_level: 0.01,
                tone_map_params: VividToneMapParams {
                    max_display_luminance: 500.0,
                    min_display_luminance: 0.01,
                    tm_gain: 1.0,
                    tm_black_level: 0.0,
                },
            };
            let bytes = m.serialize();
            let parsed = VividHdrMetadata::parse(&bytes)
                .unwrap_or_else(|e| panic!("parse failed for {eotf:?}: {e}"));
            assert_eq!(parsed.eotf, eotf);
        }
    }
}
