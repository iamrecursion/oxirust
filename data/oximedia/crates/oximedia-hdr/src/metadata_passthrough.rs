//! HDR metadata passthrough in container.
//!
//! Provides types and helpers for extracting, injecting, and carrying HDR SEI
//! NALU payloads (HDR10 MDCV/CLL, HDR10+, HLG, Dolby Vision) through a media
//! container without re-encoding the video.

use crate::color_volume::{ContentLightLevel, Hdr10PlusMetadata, MasteringDisplayColorVolume};

// ─── HdrSeiPayload ────────────────────────────────────────────────────────────

/// The HDR SEI payload carried in a NAL unit.
#[derive(Debug, Clone)]
pub enum HdrSeiPayload {
    /// HDR10 static metadata: Mastering Display Colour Volume + Content Light Level.
    Hdr10 {
        /// SMPTE ST 2086 mastering display colour volume.
        mdcv: MasteringDisplayColorVolume,
        /// CEA-861.3 content light level.
        cll: ContentLightLevel,
    },
    /// HDR10+ dynamic metadata (SMPTE ST 2094-40).
    Hdr10Plus {
        /// Parsed HDR10+ dynamic metadata.
        metadata: Hdr10PlusMetadata,
    },
    /// HLG broadcast profile (ARIB STD-B67).
    Hlg,
    /// Dolby Vision RPU profile/level signal.
    DolbyVision {
        /// Dolby Vision profile number.
        profile: u8,
        /// Dolby Vision level number.
        level: u8,
    },
}

// ─── HdrMetadataPassthrough ───────────────────────────────────────────────────

/// Carries an optional [`HdrSeiPayload`] through a processing pipeline.
///
/// Attach to a frame or packet before muxing; the muxer reads `get_payload`
/// and injects the SEI before writing the NAL stream.
#[derive(Debug, Clone)]
pub struct HdrMetadataPassthrough {
    payload: Option<HdrSeiPayload>,
}

impl HdrMetadataPassthrough {
    /// Create a new, empty passthrough container.
    pub fn new() -> Self {
        Self { payload: None }
    }

    /// Store an HDR SEI payload.
    pub fn set_payload(&mut self, payload: HdrSeiPayload) {
        self.payload = Some(payload);
    }

    /// Retrieve the stored payload, if any.
    pub fn get_payload(&self) -> Option<&HdrSeiPayload> {
        self.payload.as_ref()
    }

    /// Clear the stored payload.
    pub fn clear(&mut self) {
        self.payload = None;
    }

    /// Return a human-readable name for the active HDR format.
    pub fn format_name(&self) -> &str {
        match &self.payload {
            None => "SDR",
            Some(HdrSeiPayload::Hdr10 { .. }) => "HDR10",
            Some(HdrSeiPayload::Hdr10Plus { .. }) => "HDR10+",
            Some(HdrSeiPayload::Hlg) => "HLG",
            Some(HdrSeiPayload::DolbyVision { .. }) => "DolbyVision",
        }
    }
}

impl Default for HdrMetadataPassthrough {
    fn default() -> Self {
        Self::new()
    }
}

// ─── SeiNaluExtractor ─────────────────────────────────────────────────────────

/// Extracts HDR SEI payloads from raw NAL unit bytes.
///
/// Understands both H.264 SEI (NAL type 6) and H.265 prefix/suffix SEI
/// (NAL types 39 and 40).
#[derive(Debug, Clone, Default)]
pub struct SeiNaluExtractor;

impl SeiNaluExtractor {
    /// Create a new extractor.
    pub fn new() -> Self {
        Self
    }

    /// Attempt to extract an HDR SEI payload from a raw NAL unit.
    ///
    /// Returns `None` if the NAL unit is not an SEI type, is too short, or
    /// contains no recognised HDR SEI message.
    pub fn extract_hdr_sei(&self, nalu: &[u8]) -> Option<HdrSeiPayload> {
        if nalu.len() < 2 {
            return None;
        }

        let nal_type_h264 = nalu[0] & 0x1f;
        let nal_type_h265 = (nalu[0] >> 1) & 0x3f;

        // Determine where the SEI messages begin.
        let payload_offset = if nal_type_h264 == 6 {
            // H.264 SEI: 1-byte NAL header.
            1usize
        } else if nal_type_h265 == 39 || nal_type_h265 == 40 {
            // H.265 SEI: 2-byte NAL header.
            2usize
        } else {
            return None;
        };

        if nalu.len() <= payload_offset + 1 {
            return None;
        }

        let sei_type = nalu[payload_offset];
        let sei_size = nalu.get(payload_offset + 1).copied()? as usize;
        let data_start = payload_offset + 2;
        let data_end = data_start.checked_add(sei_size)?;

        if nalu.len() < data_end {
            return None;
        }

        let payload = &nalu[data_start..data_end];

        match sei_type {
            // Mastering Display Colour Volume (SMPTE ST 2086)
            137 => {
                if payload.len() < 24 {
                    return None;
                }
                let mdcv = parse_mdcv_bytes(payload)?;
                Some(HdrSeiPayload::Hdr10 {
                    mdcv,
                    cll: ContentLightLevel {
                        max_cll: 0,
                        max_fall: 0,
                    },
                })
            }
            // Content Light Level (CEA-861.3)
            144 => {
                if payload.len() < 4 {
                    return None;
                }
                let max_cll = u16::from_be_bytes([payload[0], payload[1]]);
                let max_fall = u16::from_be_bytes([payload[2], payload[3]]);
                Some(HdrSeiPayload::Hdr10 {
                    mdcv: MasteringDisplayColorVolume::rec2020_reference(),
                    cll: ContentLightLevel { max_cll, max_fall },
                })
            }
            // User Data Registered (ITU-T T.35) — may be HDR10+
            4 => {
                if payload.first().copied() == Some(0xB5) {
                    Some(HdrSeiPayload::Hdr10Plus {
                        metadata: Hdr10PlusMetadata {
                            system_start_code: 0xB5,
                            application_version: 1,
                            num_windows: 1,
                            target_system_display_max_luminance: 1000,
                            maxscl: [1000, 1000, 1000],
                            average_maxrgb: 500,
                        },
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Parse a 24-byte MDCV payload (big-endian, HEVC SEI layout).
fn parse_mdcv_bytes(data: &[u8]) -> Option<MasteringDisplayColorVolume> {
    if data.len() < 24 {
        return None;
    }
    // 6 × u16 primaries: R_x, R_y, G_x, G_y, B_x, B_y
    let rx = u16::from_be_bytes([data[0], data[1]]);
    let ry = u16::from_be_bytes([data[2], data[3]]);
    let gx = u16::from_be_bytes([data[4], data[5]]);
    let gy = u16::from_be_bytes([data[6], data[7]]);
    let bx = u16::from_be_bytes([data[8], data[9]]);
    let by_ = u16::from_be_bytes([data[10], data[11]]);
    let wx = u16::from_be_bytes([data[12], data[13]]);
    let wy = u16::from_be_bytes([data[14], data[15]]);
    let max_lum = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
    let min_lum = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
    Some(MasteringDisplayColorVolume {
        primaries: [[rx, ry], [gx, gy], [bx, by_]],
        white_point: [wx, wy],
        max_luminance: max_lum,
        min_luminance: min_lum,
    })
}

// ─── HdrSeiInjector ───────────────────────────────────────────────────────────

/// Injects synthetic HDR SEI NAL units into a frame byte stream.
///
/// The injected bytes are prepended to the existing frame data using a 4-byte
/// Annex-B start code (`00 00 00 01`).
#[derive(Debug, Clone, Default)]
pub struct HdrSeiInjector;

impl HdrSeiInjector {
    /// Create a new injector.
    pub fn new() -> Self {
        Self
    }

    /// Prepend a synthetic SEI NAL unit for `sei` to `frame`.
    ///
    /// The frame data is shifted right; the injected bytes occupy the front.
    pub fn inject_into_frame(&self, frame: &mut Vec<u8>, sei: &HdrSeiPayload) {
        let prefix = Self::build_sei_prefix(sei);
        // Prepend: allocate new vec, write prefix, extend with original frame.
        let orig_len = frame.len();
        let new_len = prefix.len() + orig_len;
        let mut new_frame = Vec::with_capacity(new_len);
        new_frame.extend_from_slice(&prefix);
        new_frame.extend_from_slice(frame);
        *frame = new_frame;
    }

    fn build_sei_prefix(sei: &HdrSeiPayload) -> Vec<u8> {
        // Annex-B start code + H.264 SEI NAL header.
        const START_CODE: [u8; 4] = [0x00, 0x00, 0x00, 0x01];
        const NAL_SEI: u8 = 0x06;

        let mut buf = Vec::new();
        buf.extend_from_slice(&START_CODE);
        buf.push(NAL_SEI);

        match sei {
            HdrSeiPayload::Hdr10 { .. } => {
                // MDCV SEI: type=137, size=24, 24 placeholder bytes.
                buf.push(137);
                buf.push(24);
                buf.extend_from_slice(&[0u8; 24]);
                // CLL SEI: type=144, size=4, 4 placeholder bytes.
                buf.push(144);
                buf.push(4);
                buf.extend_from_slice(&[0u8; 4]);
            }
            HdrSeiPayload::Hdr10Plus { .. } => {
                // User data registered: type=4, size=4, T.35 marker + 3 bytes.
                buf.push(4);
                buf.push(4);
                buf.extend_from_slice(&[0xB5, 0x00, 0x3C, 0x01]);
            }
            HdrSeiPayload::Hlg => {
                // Minimal pic_timing marker: type=1, size=2.
                buf.push(1);
                buf.push(2);
                buf.extend_from_slice(&[0x00, 0x00]);
            }
            HdrSeiPayload::DolbyVision { profile, level } => {
                // Unregistered SEI stub: type=5, size=2, profile+level bytes.
                buf.push(5);
                buf.push(2);
                buf.push(*profile);
                buf.push(*level);
            }
        }

        buf
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hdr10_payload() -> HdrSeiPayload {
        HdrSeiPayload::Hdr10 {
            mdcv: MasteringDisplayColorVolume::rec2020_reference(),
            cll: ContentLightLevel {
                max_cll: 1000,
                max_fall: 400,
            },
        }
    }

    // ── HdrMetadataPassthrough ────────────────────────────────────────────────

    #[test]
    fn test_passthrough_default_is_sdr() {
        let pt = HdrMetadataPassthrough::new();
        assert_eq!(pt.format_name(), "SDR");
        assert!(pt.get_payload().is_none());
    }

    #[test]
    fn test_passthrough_set_and_get_hdr10() {
        let mut pt = HdrMetadataPassthrough::new();
        pt.set_payload(make_hdr10_payload());
        assert!(pt.get_payload().is_some());
        assert_eq!(pt.format_name(), "HDR10");
    }

    #[test]
    fn test_passthrough_set_hdr10plus() {
        let mut pt = HdrMetadataPassthrough::new();
        pt.set_payload(HdrSeiPayload::Hdr10Plus {
            metadata: Hdr10PlusMetadata {
                system_start_code: 0xB5,
                application_version: 1,
                num_windows: 1,
                target_system_display_max_luminance: 1000,
                maxscl: [100, 200, 150],
                average_maxrgb: 120,
            },
        });
        assert_eq!(pt.format_name(), "HDR10+");
    }

    #[test]
    fn test_passthrough_set_hlg() {
        let mut pt = HdrMetadataPassthrough::new();
        pt.set_payload(HdrSeiPayload::Hlg);
        assert_eq!(pt.format_name(), "HLG");
    }

    #[test]
    fn test_passthrough_set_dolby_vision() {
        let mut pt = HdrMetadataPassthrough::new();
        pt.set_payload(HdrSeiPayload::DolbyVision {
            profile: 5,
            level: 6,
        });
        assert_eq!(pt.format_name(), "DolbyVision");
    }

    #[test]
    fn test_passthrough_clear() {
        let mut pt = HdrMetadataPassthrough::new();
        pt.set_payload(HdrSeiPayload::Hlg);
        pt.clear();
        assert!(pt.get_payload().is_none());
        assert_eq!(pt.format_name(), "SDR");
    }

    // ── SeiNaluExtractor ─────────────────────────────────────────────────────

    #[test]
    fn test_extractor_empty_returns_none() {
        let ex = SeiNaluExtractor::new();
        assert!(ex.extract_hdr_sei(&[]).is_none());
    }

    #[test]
    fn test_extractor_single_byte_returns_none() {
        let ex = SeiNaluExtractor::new();
        assert!(ex.extract_hdr_sei(&[0x06]).is_none());
    }

    #[test]
    fn test_extractor_truncated_payload_returns_none() {
        let ex = SeiNaluExtractor::new();
        // H.264 SEI, type=137 (MDCV), size=24 but only 3 payload bytes.
        let nalu = vec![0x06u8, 137, 24, 0x00, 0x01, 0x02];
        assert!(ex.extract_hdr_sei(&nalu).is_none());
    }

    #[test]
    fn test_extractor_non_sei_nal_returns_none() {
        let ex = SeiNaluExtractor::new();
        // H.264 IDR slice (nal_type=5).
        let nalu = vec![0x65u8, 0x88, 0x84, 0x00];
        assert!(ex.extract_hdr_sei(&nalu).is_none());
    }

    #[test]
    fn test_extractor_cll_sei_returns_hdr10() {
        let ex = SeiNaluExtractor::new();
        // H.264 SEI, type=144 (CLL), size=4, max_cll=1000, max_fall=400.
        let mut nalu = vec![0x06u8, 144, 4];
        nalu.extend_from_slice(&1000u16.to_be_bytes());
        nalu.extend_from_slice(&400u16.to_be_bytes());
        let result = ex.extract_hdr_sei(&nalu);
        assert!(result.is_some(), "expected Hdr10 payload");
        if let Some(HdrSeiPayload::Hdr10 { cll, .. }) = result {
            assert_eq!(cll.max_cll, 1000);
            assert_eq!(cll.max_fall, 400);
        } else {
            panic!("expected Hdr10 variant");
        }
    }

    #[test]
    fn test_extractor_hdr10plus_user_data() {
        let ex = SeiNaluExtractor::new();
        // H.264 SEI, type=4 (user_data_registered), country_code=0xB5.
        let nalu = vec![0x06u8, 4, 4, 0xB5, 0x00, 0x3C, 0x01];
        let result = ex.extract_hdr_sei(&nalu);
        assert!(result.is_some());
        assert!(matches!(result, Some(HdrSeiPayload::Hdr10Plus { .. })));
    }

    #[test]
    fn test_extractor_user_data_non_hdr10plus_returns_none() {
        let ex = SeiNaluExtractor::new();
        // H.264 SEI, type=4 but country_code != 0xB5.
        let nalu = vec![0x06u8, 4, 4, 0x26, 0x00, 0x00, 0x00];
        assert!(ex.extract_hdr_sei(&nalu).is_none());
    }

    #[test]
    fn test_extractor_mdcv_sei_returns_hdr10() {
        let ex = SeiNaluExtractor::new();
        // H.264 SEI, type=137 (MDCV), size=24, all-zero payload.
        let mut nalu = vec![0x06u8, 137, 24];
        nalu.extend_from_slice(&[0u8; 24]);
        let result = ex.extract_hdr_sei(&nalu);
        assert!(result.is_some());
        assert!(matches!(result, Some(HdrSeiPayload::Hdr10 { .. })));
    }

    // ── HdrSeiInjector ───────────────────────────────────────────────────────

    #[test]
    fn test_injector_hdr10_prepends_start_code() {
        let injector = HdrSeiInjector::new();
        let mut frame = vec![0xABu8, 0xCDu8];
        injector.inject_into_frame(&mut frame, &make_hdr10_payload());
        // Frame should start with Annex-B start code.
        assert_eq!(&frame[..4], &[0x00, 0x00, 0x00, 0x01], "frame={frame:?}");
        // Original frame data should still be present.
        assert!(frame.ends_with(&[0xABu8, 0xCDu8]));
    }

    #[test]
    fn test_injector_hdr10_increases_size() {
        let injector = HdrSeiInjector::new();
        let mut frame = vec![0x00u8; 10];
        let orig_len = frame.len();
        injector.inject_into_frame(&mut frame, &make_hdr10_payload());
        assert!(frame.len() > orig_len, "frame should grow after inject");
    }

    #[test]
    fn test_injector_hdr10plus_marker_present() {
        let injector = HdrSeiInjector::new();
        let mut frame = vec![0x01u8];
        injector.inject_into_frame(
            &mut frame,
            &HdrSeiPayload::Hdr10Plus {
                metadata: Hdr10PlusMetadata {
                    system_start_code: 0xB5,
                    application_version: 1,
                    num_windows: 1,
                    target_system_display_max_luminance: 400,
                    maxscl: [100, 200, 150],
                    average_maxrgb: 120,
                },
            },
        );
        // T.35 country code 0xB5 must appear in the injected prefix.
        assert!(frame.contains(&0xB5), "frame={frame:?}");
    }

    #[test]
    fn test_injector_hlg_prepends_start_code() {
        let injector = HdrSeiInjector::new();
        let mut frame = vec![0xFF; 4];
        injector.inject_into_frame(&mut frame, &HdrSeiPayload::Hlg);
        assert_eq!(&frame[..4], &[0x00, 0x00, 0x00, 0x01]);
    }

    #[test]
    fn test_injector_dolby_vision_profile_level() {
        let injector = HdrSeiInjector::new();
        let mut frame = vec![0x01u8];
        injector.inject_into_frame(
            &mut frame,
            &HdrSeiPayload::DolbyVision {
                profile: 8,
                level: 9,
            },
        );
        // Profile and level bytes must be in the injected data.
        assert!(frame.contains(&8u8), "profile byte missing");
        assert!(frame.contains(&9u8), "level byte missing");
    }

    #[test]
    fn test_injector_multiple_injects_accumulate() {
        let injector = HdrSeiInjector::new();
        let mut frame = vec![0xDEu8, 0xADu8];
        injector.inject_into_frame(&mut frame, &HdrSeiPayload::Hlg);
        let len_after_first = frame.len();
        injector.inject_into_frame(&mut frame, &HdrSeiPayload::Hlg);
        assert!(
            frame.len() > len_after_first,
            "second inject should grow frame further"
        );
    }
}
