//! Dolby Vision profile definitions and metadata detection.
//!
//! Supports common broadcast and streaming Dolby Vision profiles (4, 5, 7, 8, 9)
//! including base-layer signal compatibility detection and cross-version
//! backward-compatibility querying.
//!
//! # Profile Overview
//!
//! | Profile | Description                              |
//! |---------|------------------------------------------|
//! | P4      | HDR10 base-layer + Dolby Vision EL        |
//! | P5      | Dolby Vision single-layer                |
//! | P7      | Dual-layer BL+EL (cinema)                |
//! | P8      | HDR10 base + Dolby Vision metadata        |
//! | P9      | SDR base + Dolby Vision metadata          |

// ── DolbyVisionProfile ────────────────────────────────────────────────────────

/// Dolby Vision profile variant.
///
/// Each profile describes the base-layer signal type and whether an
/// enhancement layer (EL) is present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DolbyVisionProfile {
    /// Profile 4 — HDR10 base-layer with Dolby Vision EL.
    /// BL: PQ/BT.2020 (HDR10), EL: Dolby Vision RPU.
    Profile4,

    /// Profile 5 — Dolby Vision single-layer (no separate EL).
    /// BL: Dolby Vision native PQ, no EL.
    Profile5,

    /// Profile 7 — Dual-layer BL+EL (cinema distribution).
    /// BL: SDR/HLG, EL: full Dolby Vision enhancement.
    Profile7,

    /// Profile 8 — HDR10 base-layer with Dolby Vision RPU metadata only.
    /// BL: PQ/BT.2020 (HDR10 compatible), no EL.
    Profile8,

    /// Profile 9 — SDR BT.709 base-layer with Dolby Vision RPU metadata.
    /// Broadcast-friendly; backward-compatible with legacy SDR displays.
    Profile9,
}

impl DolbyVisionProfile {
    /// Return the numeric profile number.
    pub fn number(&self) -> u8 {
        match self {
            DolbyVisionProfile::Profile4 => 4,
            DolbyVisionProfile::Profile5 => 5,
            DolbyVisionProfile::Profile7 => 7,
            DolbyVisionProfile::Profile8 => 8,
            DolbyVisionProfile::Profile9 => 9,
        }
    }

    /// Whether this profile includes a separate enhancement layer (EL).
    pub fn has_enhancement_layer(&self) -> bool {
        matches!(
            self,
            DolbyVisionProfile::Profile4 | DolbyVisionProfile::Profile7
        )
    }

    /// Whether this profile is a single-layer (no EL) profile.
    pub fn is_single_layer(&self) -> bool {
        !self.has_enhancement_layer()
    }

    /// Parse from the numeric profile number.
    ///
    /// Returns `None` for unsupported profile numbers.
    pub fn from_number(n: u8) -> Option<Self> {
        match n {
            4 => Some(DolbyVisionProfile::Profile4),
            5 => Some(DolbyVisionProfile::Profile5),
            7 => Some(DolbyVisionProfile::Profile7),
            8 => Some(DolbyVisionProfile::Profile8),
            9 => Some(DolbyVisionProfile::Profile9),
            _ => None,
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            DolbyVisionProfile::Profile4 => "HDR10 BL + Dolby Vision EL (Profile 4)",
            DolbyVisionProfile::Profile5 => "Dolby Vision single-layer (Profile 5)",
            DolbyVisionProfile::Profile7 => "Dual-layer BL+EL cinema (Profile 7)",
            DolbyVisionProfile::Profile8 => "HDR10 base + DV RPU metadata (Profile 8)",
            DolbyVisionProfile::Profile9 => "SDR base + DV RPU metadata (Profile 9)",
        }
    }
}

impl std::fmt::Display for DolbyVisionProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dolby Vision Profile {}", self.number())
    }
}

// ── DvRpuData ─────────────────────────────────────────────────────────────────

/// Raw Dolby Vision RPU (Reference Processing Unit) header data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DvRpuData {
    /// RPU type (0 = regular, others reserved).
    pub rpu_type: u8,
    /// RPU format bitfield.
    pub rpu_format: u16,
    /// VDR RPU profile number.
    pub vdr_rpu_profile: u8,
    /// VDR RPU level.
    pub vdr_rpu_level: u8,
}

impl DvRpuData {
    /// Construct a default RPU for Profile 8 (common streaming).
    pub fn profile8_default() -> Self {
        Self {
            rpu_type: 0,
            rpu_format: 0x0180,
            vdr_rpu_profile: 8,
            vdr_rpu_level: 6,
        }
    }
}

// ── BlSignalCompatibility ─────────────────────────────────────────────────────

/// Base-layer signal compatibility for Dolby Vision content.
///
/// Describes what the base-layer signal looks like on a non-Dolby-Vision
/// display or processing path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlSignalCompatibility {
    /// BL is a native Dolby Vision signal.
    DolbyVision,
    /// BL is HDR10 (PQ / BT.2020).
    Hdr10,
    /// BL is SDR (BT.709 or similar).
    Sdr,
    /// BL is HLG.
    Hlg,
}

impl BlSignalCompatibility {
    /// Return the MPEG-4 / HEVC compatibility identifier (hevc_dvbl_signal_compatibility_id).
    pub fn compat_id(&self) -> u8 {
        match self {
            BlSignalCompatibility::DolbyVision => 0,
            BlSignalCompatibility::Hdr10 => 1,
            BlSignalCompatibility::Sdr => 2,
            BlSignalCompatibility::Hlg => 4,
        }
    }

    /// Parse from the numeric compatibility identifier.
    pub fn from_compat_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(BlSignalCompatibility::DolbyVision),
            1 => Some(BlSignalCompatibility::Hdr10),
            2 => Some(BlSignalCompatibility::Sdr),
            4 => Some(BlSignalCompatibility::Hlg),
            _ => None,
        }
    }
}

impl std::fmt::Display for BlSignalCompatibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BlSignalCompatibility::DolbyVision => "Dolby Vision",
            BlSignalCompatibility::Hdr10 => "HDR10",
            BlSignalCompatibility::Sdr => "SDR",
            BlSignalCompatibility::Hlg => "HLG",
        };
        write!(f, "{s}")
    }
}

// ── DvMetadata ────────────────────────────────────────────────────────────────

/// Top-level Dolby Vision stream metadata descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DvMetadata {
    /// Active Dolby Vision profile.
    pub profile: DolbyVisionProfile,
    /// Active level (1–13; relates to resolution / frame-rate tier).
    pub level: u8,
    /// Base-layer signal type visible to non-DV decoders.
    pub bl_signal_compatibility: BlSignalCompatibility,
    /// Whether an RPU NAL is present in the stream.
    pub rpu_present: bool,
    /// Whether an enhancement layer (EL) is present.
    pub el_present: bool,
    /// Whether a base layer (BL) is present.
    pub bl_present: bool,
}

// ── detect_profile ────────────────────────────────────────────────────────────

/// Derive the [`DolbyVisionProfile`] from stream metadata using rule-based
/// detection consistent with the Dolby Vision UHD Blu-ray and broadcast
/// specifications.
///
/// Detection rules:
/// - EL present + BL HDR10 → Profile 4
/// - EL present + BL DolbyVision → Profile 7
/// - No EL + BL DolbyVision → Profile 5
/// - No EL + BL HDR10 → Profile 8
/// - No EL + BL SDR → Profile 9
/// - No EL + BL HLG → Profile 8 (closest broadcast-compatible profile)
pub fn detect_profile(data: &DvMetadata) -> DolbyVisionProfile {
    if data.el_present {
        match data.bl_signal_compatibility {
            BlSignalCompatibility::Hdr10 => DolbyVisionProfile::Profile4,
            BlSignalCompatibility::DolbyVision => DolbyVisionProfile::Profile7,
            // Fallback: dual-layer with non-HDR10 BL treated as P7.
            BlSignalCompatibility::Sdr | BlSignalCompatibility::Hlg => DolbyVisionProfile::Profile7,
        }
    } else {
        match data.bl_signal_compatibility {
            BlSignalCompatibility::DolbyVision => DolbyVisionProfile::Profile5,
            BlSignalCompatibility::Hdr10 => DolbyVisionProfile::Profile8,
            BlSignalCompatibility::Sdr => DolbyVisionProfile::Profile9,
            BlSignalCompatibility::Hlg => DolbyVisionProfile::Profile8,
        }
    }
}

// ── CrossVersionDvMeta ────────────────────────────────────────────────────────

/// Dolby Vision specification version tracker for backward-compatibility checks.
///
/// `version_major` follows the Dolby Vision application specification series
/// (e.g. 2 for DVS2.x, 3 for DVS3.x), while `version_minor` tracks the
/// minor release within that series.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossVersionDvMeta {
    /// Major specification version (e.g. 2, 3, 4).
    pub version_major: u8,
    /// Minor specification version.
    pub version_minor: u8,
}

impl CrossVersionDvMeta {
    /// Create a new version descriptor.
    pub fn new(major: u8, minor: u8) -> Self {
        Self {
            version_major: major,
            version_minor: minor,
        }
    }

    /// Return whether this version is backward-compatible with the installed
    /// base of Dolby Vision decoders.
    ///
    /// The current backward-compatibility boundary is major version ≤ 4.
    /// Content authored with major > 4 may use features unavailable in
    /// older decoder firmware.
    pub fn is_backward_compatible(&self) -> bool {
        self.version_major <= 4
    }

    /// Return `true` if this version is at least `(major, minor)`.
    pub fn is_at_least(&self, major: u8, minor: u8) -> bool {
        (self.version_major, self.version_minor) >= (major, minor)
    }
}

impl std::fmt::Display for CrossVersionDvMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DVS{}.{}", self.version_major, self.version_minor)
    }
}

// ── RPU NAL Unit Generation ─────────────────────────────────────────────────

/// Configuration for generating Dolby Vision RPU NAL units.
///
/// An RPU (Reference Processing Unit) NAL unit carries the metadata that a
/// Dolby Vision decoder uses to reconstruct the HDR image from the base layer.
/// This includes VDR (Visual Dynamic Range) polynomial mapping coefficients,
/// NLQ (Non-Linear Quantization) parameters, and colour remapping info.
#[derive(Debug, Clone)]
pub struct RpuGenerationConfig {
    /// Target Dolby Vision profile.
    pub profile: DolbyVisionProfile,
    /// Base-layer signal type.
    pub bl_signal_compat: BlSignalCompatibility,
    /// VDR RPU level (commonly 6).
    pub vdr_rpu_level: u8,
    /// Maximum display mastering luminance in nits (e.g. 1000, 4000).
    pub max_display_mastering_luminance: u32,
    /// Minimum display mastering luminance in 0.0001-nit units (e.g. 1 = 0.0001 nits).
    pub min_display_mastering_luminance: u32,
    /// Number of polynomial pivots for the VDR mapping (1–8, typically 3).
    pub num_pivots: u8,
    /// Polynomial pivot values in PQ signal space (normalised 0–4095).
    /// Length should be `num_pivots + 1`.
    pub pivots: Vec<u16>,
    /// Polynomial mapping order per piece (1=linear, 2=quadratic).
    /// Length should be `num_pivots`.
    pub mapping_order: Vec<u8>,
}

impl RpuGenerationConfig {
    /// Create a default configuration for Profile 8 (HDR10 + DV RPU).
    ///
    /// Uses 3 pivots with linear mapping suitable for 1000-nit content.
    pub fn profile8_default() -> Self {
        Self {
            profile: DolbyVisionProfile::Profile8,
            bl_signal_compat: BlSignalCompatibility::Hdr10,
            vdr_rpu_level: 6,
            max_display_mastering_luminance: 1000,
            min_display_mastering_luminance: 1,
            num_pivots: 3,
            pivots: vec![0, 1023, 2048, 4095],
            mapping_order: vec![1, 1, 1],
        }
    }

    /// Create a default configuration for Profile 5 (DV single-layer).
    pub fn profile5_default() -> Self {
        Self {
            profile: DolbyVisionProfile::Profile5,
            bl_signal_compat: BlSignalCompatibility::DolbyVision,
            vdr_rpu_level: 6,
            max_display_mastering_luminance: 4000,
            min_display_mastering_luminance: 1,
            num_pivots: 3,
            pivots: vec![0, 1023, 2048, 4095],
            mapping_order: vec![2, 2, 1],
        }
    }
}

/// A generated Dolby Vision RPU NAL unit.
#[derive(Debug, Clone)]
pub struct RpuNalUnit {
    /// The raw byte payload (excluding start code prefix).
    pub payload: Vec<u8>,
    /// Profile used to generate this RPU.
    pub profile: DolbyVisionProfile,
    /// CRC-32 of the payload for integrity verification.
    pub crc32: u32,
}

/// Generate a Dolby Vision RPU NAL unit from the given configuration.
///
/// The generated RPU follows the ETSI TS 103 572 / Dolby specification
/// for RPU bitstream syntax. The output is a complete RPU payload
/// (without the NAL start code prefix `0x00 0x00 0x01 0x7E`).
///
/// # Errors
/// Returns `HdrError::MetadataParseError` if:
/// - `num_pivots` is 0 or > 8
/// - `pivots.len()` != `num_pivots + 1`
/// - `mapping_order.len()` != `num_pivots`
/// - Profile number is unsupported
pub fn generate_rpu_nal(config: &RpuGenerationConfig) -> crate::Result<RpuNalUnit> {
    // Validate configuration
    if config.num_pivots == 0 || config.num_pivots > 8 {
        return Err(crate::HdrError::MetadataParseError(format!(
            "num_pivots must be 1–8, got {}",
            config.num_pivots
        )));
    }
    let expected_pivots = usize::from(config.num_pivots) + 1;
    if config.pivots.len() != expected_pivots {
        return Err(crate::HdrError::MetadataParseError(format!(
            "pivots length {} does not match num_pivots + 1 = {}",
            config.pivots.len(),
            expected_pivots
        )));
    }
    if config.mapping_order.len() != usize::from(config.num_pivots) {
        return Err(crate::HdrError::MetadataParseError(format!(
            "mapping_order length {} does not match num_pivots = {}",
            config.mapping_order.len(),
            config.num_pivots
        )));
    }

    let mut payload = Vec::with_capacity(256);

    // ── RPU header ──
    // rpu_nal_prefix (1 byte): NAL unit header byte for RPU (0x19 = unspec62)
    payload.push(0x19);

    // rpu_type (1 byte): 2 = Dolby Vision RPU
    payload.push(0x02);

    // rpu_format (2 bytes): 0x0180 for standard VDR RPU
    payload.extend_from_slice(&0x0180_u16.to_be_bytes());

    // vdr_rpu_profile (1 byte)
    payload.push(config.profile.number());

    // vdr_rpu_level (1 byte)
    payload.push(config.vdr_rpu_level);

    // bl_signal_comp_id (1 byte)
    payload.push(config.bl_signal_compat.compat_id());

    // el_type (1 byte): 0 = no EL for single-layer, 1 = EL present
    let el_type: u8 = if config.profile.has_enhancement_layer() {
        1
    } else {
        0
    };
    payload.push(el_type);

    // ── VDR DM data: mastering display ──
    // max_display_mastering_luminance (4 bytes, big-endian)
    payload.extend_from_slice(&config.max_display_mastering_luminance.to_be_bytes());
    // min_display_mastering_luminance (4 bytes, big-endian)
    payload.extend_from_slice(&config.min_display_mastering_luminance.to_be_bytes());

    // ── Polynomial mapping section ──
    // num_pivots (1 byte)
    payload.push(config.num_pivots);

    // pivot values (each 2 bytes, big-endian)
    for &pivot in &config.pivots {
        payload.extend_from_slice(&pivot.to_be_bytes());
    }

    // mapping_order per piece (1 byte each)
    for &order in &config.mapping_order {
        payload.push(order);
    }

    // ── Polynomial coefficients (identity mapping as default) ──
    // For each piece, write the polynomial coefficients.
    // Order 1 (linear): coeff_0 (intercept) + coeff_1 (slope)
    // Order 2 (quadratic): coeff_0 + coeff_1 + coeff_2
    // We generate identity-like coefficients by default.
    for (i, &order) in config.mapping_order.iter().enumerate() {
        // For the identity mapping, each piece maps input to output linearly.
        let start = config.pivots.get(i).copied().unwrap_or(0);
        let end = config.pivots.get(i + 1).copied().unwrap_or(4095);
        let range = (end as f32 - start as f32).max(1.0);

        // Intercept: normalised start/4095
        let c0 = start as f32 / 4095.0;
        // Slope: piece range / total range
        let c1 = range / 4095.0;

        // Encode as fixed-point Q15 (signed 16-bit, scale = 1/32768)
        let c0_fp = (c0 * 32768.0) as i16;
        let c1_fp = (c1 * 32768.0) as i16;
        payload.extend_from_slice(&c0_fp.to_be_bytes());
        payload.extend_from_slice(&c1_fp.to_be_bytes());

        if order >= 2 {
            // Quadratic coefficient (0 for near-identity)
            let c2_fp: i16 = 0;
            payload.extend_from_slice(&c2_fp.to_be_bytes());
        }
    }

    // ── NLQ section (if applicable) ──
    // For profiles with EL, write a minimal NLQ header.
    if config.profile.has_enhancement_layer() {
        // nlq_method_idc (1 byte): 0 = none
        payload.push(0x00);
        // nlq_num_pivots_minus1 (1 byte): 0
        payload.push(0x00);
        // nlq_pred_pivot_value (2 bytes): 2048
        payload.extend_from_slice(&2048_u16.to_be_bytes());
    }

    // ── Trailing alignment + CRC ──
    // Pad to byte boundary (ensure even length for CRC)
    if payload.len() % 2 != 0 {
        payload.push(0x00);
    }

    // CRC-32 of the payload
    let crc = crc32_compute(&payload);

    // Append CRC (4 bytes, big-endian)
    payload.extend_from_slice(&crc.to_be_bytes());

    Ok(RpuNalUnit {
        payload,
        profile: config.profile,
        crc32: crc,
    })
}

/// Parse the profile and level from a raw RPU NAL payload.
///
/// Expects a payload generated by [`generate_rpu_nal`] (without start code prefix).
///
/// # Errors
/// Returns `HdrError::MetadataParseError` if the payload is too short or the
/// profile number is unrecognised.
pub fn parse_rpu_nal_header(payload: &[u8]) -> crate::Result<(DolbyVisionProfile, u8)> {
    if payload.len() < 6 {
        return Err(crate::HdrError::MetadataParseError(format!(
            "RPU payload too short: {} bytes (need >= 6)",
            payload.len()
        )));
    }

    let profile_num = payload[4];
    let level = payload[5];

    let profile = DolbyVisionProfile::from_number(profile_num).ok_or_else(|| {
        crate::HdrError::MetadataParseError(format!(
            "unsupported profile number in RPU: {profile_num}"
        ))
    })?;

    Ok((profile, level))
}

/// Verify the CRC-32 of an RPU NAL payload.
///
/// The last 4 bytes of the payload should be the CRC-32 of the preceding bytes.
///
/// # Errors
/// Returns `HdrError::MetadataParseError` if the payload is too short or
/// the CRC does not match.
pub fn verify_rpu_crc(payload: &[u8]) -> crate::Result<()> {
    if payload.len() < 5 {
        return Err(crate::HdrError::MetadataParseError(
            "RPU payload too short for CRC verification".to_string(),
        ));
    }

    let data_len = payload.len() - 4;
    let stored_crc = u32::from_be_bytes([
        payload[data_len],
        payload[data_len + 1],
        payload[data_len + 2],
        payload[data_len + 3],
    ]);

    let computed_crc = crc32_compute(&payload[..data_len]);

    if stored_crc != computed_crc {
        return Err(crate::HdrError::MetadataParseError(format!(
            "RPU CRC mismatch: stored=0x{stored_crc:08X}, computed=0x{computed_crc:08X}"
        )));
    }

    Ok(())
}

/// Simple CRC-32 computation (IEEE 802.3 polynomial, no lookup table).
fn crc32_compute(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_meta(
        profile: DolbyVisionProfile,
        compat: BlSignalCompatibility,
        el: bool,
    ) -> DvMetadata {
        DvMetadata {
            profile,
            level: 6,
            bl_signal_compatibility: compat,
            rpu_present: true,
            el_present: el,
            bl_present: true,
        }
    }

    // 1. Profile numbers
    #[test]
    fn test_profile_numbers() {
        assert_eq!(DolbyVisionProfile::Profile4.number(), 4);
        assert_eq!(DolbyVisionProfile::Profile5.number(), 5);
        assert_eq!(DolbyVisionProfile::Profile7.number(), 7);
        assert_eq!(DolbyVisionProfile::Profile8.number(), 8);
        assert_eq!(DolbyVisionProfile::Profile9.number(), 9);
    }

    // 2. Profile from_number round-trip
    #[test]
    fn test_profile_from_number_roundtrip() {
        for p in [
            DolbyVisionProfile::Profile4,
            DolbyVisionProfile::Profile5,
            DolbyVisionProfile::Profile7,
            DolbyVisionProfile::Profile8,
            DolbyVisionProfile::Profile9,
        ] {
            let n = p.number();
            let back = DolbyVisionProfile::from_number(n).expect("roundtrip");
            assert_eq!(back, p);
        }
    }

    // 3. from_number for unknown value
    #[test]
    fn test_profile_from_number_unknown() {
        assert!(DolbyVisionProfile::from_number(99).is_none());
        assert!(DolbyVisionProfile::from_number(0).is_none());
    }

    // 4. Enhancement layer presence
    #[test]
    fn test_has_el() {
        assert!(DolbyVisionProfile::Profile4.has_enhancement_layer());
        assert!(DolbyVisionProfile::Profile7.has_enhancement_layer());
        assert!(!DolbyVisionProfile::Profile5.has_enhancement_layer());
        assert!(!DolbyVisionProfile::Profile8.has_enhancement_layer());
        assert!(!DolbyVisionProfile::Profile9.has_enhancement_layer());
    }

    // 5. Single-layer inverse
    #[test]
    fn test_single_layer() {
        assert!(!DolbyVisionProfile::Profile4.is_single_layer());
        assert!(DolbyVisionProfile::Profile5.is_single_layer());
    }

    // 6. Profile descriptions are non-empty
    #[test]
    fn test_profile_description_non_empty() {
        for p in [
            DolbyVisionProfile::Profile4,
            DolbyVisionProfile::Profile5,
            DolbyVisionProfile::Profile7,
            DolbyVisionProfile::Profile8,
            DolbyVisionProfile::Profile9,
        ] {
            assert!(!p.description().is_empty());
        }
    }

    // 7. detect_profile: HDR10 + EL → P4
    #[test]
    fn test_detect_profile_p4() {
        let meta = make_meta(
            DolbyVisionProfile::Profile4,
            BlSignalCompatibility::Hdr10,
            true,
        );
        assert_eq!(detect_profile(&meta), DolbyVisionProfile::Profile4);
    }

    // 8. detect_profile: DV BL + EL → P7
    #[test]
    fn test_detect_profile_p7() {
        let meta = make_meta(
            DolbyVisionProfile::Profile7,
            BlSignalCompatibility::DolbyVision,
            true,
        );
        assert_eq!(detect_profile(&meta), DolbyVisionProfile::Profile7);
    }

    // 9. detect_profile: DV BL, no EL → P5
    #[test]
    fn test_detect_profile_p5() {
        let meta = make_meta(
            DolbyVisionProfile::Profile5,
            BlSignalCompatibility::DolbyVision,
            false,
        );
        assert_eq!(detect_profile(&meta), DolbyVisionProfile::Profile5);
    }

    // 10. detect_profile: HDR10 BL, no EL → P8
    #[test]
    fn test_detect_profile_p8() {
        let meta = make_meta(
            DolbyVisionProfile::Profile8,
            BlSignalCompatibility::Hdr10,
            false,
        );
        assert_eq!(detect_profile(&meta), DolbyVisionProfile::Profile8);
    }

    // 11. detect_profile: SDR BL, no EL → P9
    #[test]
    fn test_detect_profile_p9() {
        let meta = make_meta(
            DolbyVisionProfile::Profile9,
            BlSignalCompatibility::Sdr,
            false,
        );
        assert_eq!(detect_profile(&meta), DolbyVisionProfile::Profile9);
    }

    // 12. detect_profile: HLG BL, no EL → P8 (broadcast compat)
    #[test]
    fn test_detect_profile_hlg_no_el() {
        let meta = make_meta(
            DolbyVisionProfile::Profile8,
            BlSignalCompatibility::Hlg,
            false,
        );
        assert_eq!(detect_profile(&meta), DolbyVisionProfile::Profile8);
    }

    // 13. BlSignalCompatibility compat_id round-trip
    #[test]
    fn test_bl_compat_id_roundtrip() {
        for bc in [
            BlSignalCompatibility::DolbyVision,
            BlSignalCompatibility::Hdr10,
            BlSignalCompatibility::Sdr,
            BlSignalCompatibility::Hlg,
        ] {
            let id = bc.compat_id();
            let back = BlSignalCompatibility::from_compat_id(id).expect("roundtrip");
            assert_eq!(back, bc);
        }
    }

    // 14. BlSignalCompatibility from_compat_id unknown
    #[test]
    fn test_bl_compat_id_unknown() {
        assert!(BlSignalCompatibility::from_compat_id(99).is_none());
    }

    // 15. CrossVersionDvMeta backward compat
    #[test]
    fn test_cross_version_backward_compat() {
        let v1 = CrossVersionDvMeta::new(3, 1);
        assert!(v1.is_backward_compatible());

        let v2 = CrossVersionDvMeta::new(4, 0);
        assert!(v2.is_backward_compatible());

        let v3 = CrossVersionDvMeta::new(5, 0);
        assert!(!v3.is_backward_compatible());
    }

    // 16. CrossVersionDvMeta is_at_least
    #[test]
    fn test_cross_version_at_least() {
        let v = CrossVersionDvMeta::new(3, 2);
        assert!(v.is_at_least(3, 2));
        assert!(v.is_at_least(3, 0));
        assert!(!v.is_at_least(4, 0));
        assert!(!v.is_at_least(3, 3));
    }

    // 17. CrossVersionDvMeta Display
    #[test]
    fn test_cross_version_display() {
        let v = CrossVersionDvMeta::new(2, 9);
        assert_eq!(v.to_string(), "DVS2.9");
    }

    // 18. DvRpuData default profile
    #[test]
    fn test_rpu_profile8_default() {
        let rpu = DvRpuData::profile8_default();
        assert_eq!(rpu.vdr_rpu_profile, 8);
        assert_eq!(rpu.vdr_rpu_level, 6);
    }

    // 19. Profile Display trait
    #[test]
    fn test_profile_display() {
        let p = DolbyVisionProfile::Profile8;
        assert_eq!(p.to_string(), "Dolby Vision Profile 8");
    }

    // 20. BlSignalCompatibility Display
    #[test]
    fn test_bl_compat_display() {
        assert_eq!(BlSignalCompatibility::Hdr10.to_string(), "HDR10");
        assert_eq!(BlSignalCompatibility::Sdr.to_string(), "SDR");
    }

    // ── Synthetic-bitstream RPU tests ─────────────────────────────────────────
    //
    // The tests below construct RPU NAL bytes per the layout emitted by
    // `generate_rpu_nal` and consumed by `parse_rpu_nal_header` /
    // `verify_rpu_crc`.
    //
    // IMPORTANT: These are SYNTHETIC SPEC-STRUCTURED test vectors hand-built
    // from the documented byte layout.  They are NOT captured from real Dolby
    // Vision streams or proprietary content.  Their purpose is to verify that
    // the encode ↔ parse ↔ verify paths are internally self-consistent and
    // that the parser handles both well-formed and malformed inputs correctly.
    //
    // Byte layout of the RPU preamble (matching `generate_rpu_nal` output):
    //   [0]    0x19           – NAL prefix / NALU type for RPU
    //   [1]    0x02           – rpu_type = 2 ("regular" RPU)
    //   [2-3]  0x01 0x80      – rpu_format = 0x0180 (standard VDR RPU)
    //   [4]    profile_num    – vdr_rpu_profile
    //   [5]    vdr_rpu_level  – VDR RPU level (commonly 6)
    //   [6]    compat_id      – BL signal compatibility
    //   [7]    el_type        – 0 = no EL, 1 = EL present
    //   [8-11] max_mastering  – max display mastering luminance (BE u32)
    //   [12-15] min_mastering – min display mastering luminance (BE u32)
    //   ...                  – polynomial pivot + coefficient section
    //   (last 4 bytes)       – CRC-32 (IEEE 802.3 polynomial, big-endian)

    /// Build the minimal 8-byte spec-shaped RPU preamble used by
    /// `parse_rpu_nal_header`.  NOT generated by `generate_rpu_nal` — these
    /// bytes are hand-constructed to represent the minimal valid NAL header.
    fn make_synthetic_rpu_preamble(profile_num: u8, level: u8) -> Vec<u8> {
        vec![
            0x19, // NAL prefix
            0x02, // rpu_type = 2
            0x01,
            0x80, // rpu_format = 0x0180
            profile_num,
            level,
            0x01, // bl_signal_compat: HDR10 (compat_id = 1)
            0x00, // el_type: no EL
        ]
    }

    // 21. generate_rpu_nal: Profile8 default – CRC passes and header parses correctly
    #[test]
    fn test_generate_rpu_nal_profile8_roundtrip() {
        let cfg = RpuGenerationConfig::profile8_default();
        let nal = generate_rpu_nal(&cfg).expect("generate P8 RPU NAL");

        verify_rpu_crc(&nal.payload).expect("CRC must pass on freshly generated P8 NAL");

        let (profile, level) = parse_rpu_nal_header(&nal.payload).expect("parse P8 NAL header");
        assert_eq!(profile, DolbyVisionProfile::Profile8);
        assert_eq!(level, 6);
        assert_eq!(nal.profile, DolbyVisionProfile::Profile8);
    }

    // 22. generate_rpu_nal: Profile5 default – CRC passes and header parses correctly
    #[test]
    fn test_generate_rpu_nal_profile5_roundtrip() {
        let cfg = RpuGenerationConfig::profile5_default();
        let nal = generate_rpu_nal(&cfg).expect("generate P5 RPU NAL");

        verify_rpu_crc(&nal.payload).expect("CRC must pass on freshly generated P5 NAL");

        let (profile, level) = parse_rpu_nal_header(&nal.payload).expect("parse P5 NAL header");
        assert_eq!(profile, DolbyVisionProfile::Profile5);
        assert_eq!(level, 6);
    }

    // 23. generate_rpu_nal: Profile4 with EL – NLQ section written; CRC + parse agree
    #[test]
    fn test_generate_rpu_nal_profile4_with_el() {
        let cfg = RpuGenerationConfig {
            profile: DolbyVisionProfile::Profile4,
            bl_signal_compat: BlSignalCompatibility::Hdr10,
            vdr_rpu_level: 6,
            max_display_mastering_luminance: 4000,
            min_display_mastering_luminance: 1,
            num_pivots: 3,
            pivots: vec![0, 1023, 2048, 4095],
            mapping_order: vec![1, 1, 1],
        };
        let nal = generate_rpu_nal(&cfg).expect("generate P4 RPU NAL");
        assert_eq!(nal.profile, DolbyVisionProfile::Profile4);
        verify_rpu_crc(&nal.payload).expect("CRC must pass on freshly generated P4 NAL");

        let (profile, _) = parse_rpu_nal_header(&nal.payload).expect("parse P4 NAL header");
        assert_eq!(profile, DolbyVisionProfile::Profile4);
    }

    // 24. generate_rpu_nal: Profile7 cinema with EL and quadratic mapping
    #[test]
    fn test_generate_rpu_nal_profile7_quadratic() {
        let cfg = RpuGenerationConfig {
            profile: DolbyVisionProfile::Profile7,
            bl_signal_compat: BlSignalCompatibility::DolbyVision,
            vdr_rpu_level: 6,
            max_display_mastering_luminance: 1000,
            min_display_mastering_luminance: 1,
            num_pivots: 2,
            pivots: vec![0, 2048, 4095],
            mapping_order: vec![2, 2],
        };
        let nal = generate_rpu_nal(&cfg).expect("generate P7 quadratic RPU NAL");
        verify_rpu_crc(&nal.payload).expect("CRC must pass on P7 quadratic NAL");

        let (profile, _) = parse_rpu_nal_header(&nal.payload).expect("parse P7 NAL header");
        assert_eq!(profile, DolbyVisionProfile::Profile7);
    }

    // 25. generate_rpu_nal: Profile9 SDR base – lower level, single-piece linear mapping
    #[test]
    fn test_generate_rpu_nal_profile9_sdr_base() {
        let cfg = RpuGenerationConfig {
            profile: DolbyVisionProfile::Profile9,
            bl_signal_compat: BlSignalCompatibility::Sdr,
            vdr_rpu_level: 3,
            max_display_mastering_luminance: 100,
            min_display_mastering_luminance: 0,
            num_pivots: 1,
            pivots: vec![0, 4095],
            mapping_order: vec![1],
        };
        let nal = generate_rpu_nal(&cfg).expect("generate P9 RPU NAL");
        verify_rpu_crc(&nal.payload).expect("CRC must pass on P9 NAL");

        let (profile, level) = parse_rpu_nal_header(&nal.payload).expect("parse P9 NAL header");
        assert_eq!(profile, DolbyVisionProfile::Profile9);
        assert_eq!(level, 3);
    }

    // 26. generate_rpu_nal: fixed protocol bytes at positions 0-3 are always spec values
    #[test]
    fn test_generate_rpu_nal_fixed_protocol_bytes() {
        let cfg = RpuGenerationConfig::profile8_default();
        let nal = generate_rpu_nal(&cfg).expect("generate P8 for protocol byte check");
        assert_eq!(nal.payload[0], 0x19, "byte[0]: NAL prefix must be 0x19");
        assert_eq!(nal.payload[1], 0x02, "byte[1]: rpu_type must be 2");
        assert_eq!(
            nal.payload[2], 0x01,
            "byte[2]: rpu_format high byte must be 0x01"
        );
        assert_eq!(
            nal.payload[3], 0x80,
            "byte[3]: rpu_format low byte must be 0x80"
        );
    }

    // 27. generate_rpu_nal: mastering luminance sentinel values survive round-trip in bytes
    #[test]
    fn test_generate_rpu_nal_mastering_luminance_bytes() {
        let cfg = RpuGenerationConfig {
            profile: DolbyVisionProfile::Profile8,
            bl_signal_compat: BlSignalCompatibility::Hdr10,
            vdr_rpu_level: 6,
            max_display_mastering_luminance: 0xDEAD_BEEF,
            min_display_mastering_luminance: 0x0000_0001,
            num_pivots: 1,
            pivots: vec![0, 4095],
            mapping_order: vec![1],
        };
        let nal = generate_rpu_nal(&cfg).expect("generate RPU with sentinel luminance values");
        // max_display_mastering_luminance is stored at payload bytes 8-11
        let max_read = u32::from_be_bytes([
            nal.payload[8],
            nal.payload[9],
            nal.payload[10],
            nal.payload[11],
        ]);
        // min_display_mastering_luminance is stored at payload bytes 12-15
        let min_read = u32::from_be_bytes([
            nal.payload[12],
            nal.payload[13],
            nal.payload[14],
            nal.payload[15],
        ]);
        assert_eq!(max_read, 0xDEAD_BEEF, "max mastering luminance round-trip");
        assert_eq!(min_read, 0x0000_0001, "min mastering luminance round-trip");
    }

    // 28. generate_rpu_nal: error on num_pivots = 0
    #[test]
    fn test_generate_rpu_nal_zero_pivots_error() {
        let cfg = RpuGenerationConfig {
            profile: DolbyVisionProfile::Profile8,
            bl_signal_compat: BlSignalCompatibility::Hdr10,
            vdr_rpu_level: 6,
            max_display_mastering_luminance: 1000,
            min_display_mastering_luminance: 1,
            num_pivots: 0,
            pivots: vec![0],
            mapping_order: vec![],
        };
        assert!(
            generate_rpu_nal(&cfg).is_err(),
            "num_pivots = 0 must return an error"
        );
    }

    // 29. generate_rpu_nal: error on num_pivots > 8 (limit is 8)
    #[test]
    fn test_generate_rpu_nal_too_many_pivots_error() {
        let cfg = RpuGenerationConfig {
            profile: DolbyVisionProfile::Profile8,
            bl_signal_compat: BlSignalCompatibility::Hdr10,
            vdr_rpu_level: 6,
            max_display_mastering_luminance: 1000,
            min_display_mastering_luminance: 1,
            num_pivots: 9,
            pivots: vec![0, 455, 910, 1365, 1820, 2275, 2730, 3185, 3640, 4095],
            mapping_order: vec![1; 9],
        };
        assert!(
            generate_rpu_nal(&cfg).is_err(),
            "num_pivots = 9 must return an error"
        );
    }

    // 30. generate_rpu_nal: error when pivot vector length doesn't match num_pivots + 1
    #[test]
    fn test_generate_rpu_nal_pivot_length_mismatch_error() {
        let cfg = RpuGenerationConfig {
            profile: DolbyVisionProfile::Profile8,
            bl_signal_compat: BlSignalCompatibility::Hdr10,
            vdr_rpu_level: 6,
            max_display_mastering_luminance: 1000,
            min_display_mastering_luminance: 1,
            num_pivots: 3,
            pivots: vec![0, 1023, 4095], // length 3, expected 4 (num_pivots + 1)
            mapping_order: vec![1, 1, 1],
        };
        assert!(
            generate_rpu_nal(&cfg).is_err(),
            "pivot vector length mismatch must return an error"
        );
    }

    // 31. generate_rpu_nal: error when mapping_order length doesn't match num_pivots
    #[test]
    fn test_generate_rpu_nal_mapping_order_mismatch_error() {
        let cfg = RpuGenerationConfig {
            profile: DolbyVisionProfile::Profile8,
            bl_signal_compat: BlSignalCompatibility::Hdr10,
            vdr_rpu_level: 6,
            max_display_mastering_luminance: 1000,
            min_display_mastering_luminance: 1,
            num_pivots: 3,
            pivots: vec![0, 1023, 2048, 4095],
            mapping_order: vec![1, 1], // length 2, expected 3 (== num_pivots)
        };
        assert!(
            generate_rpu_nal(&cfg).is_err(),
            "mapping_order length mismatch must return an error"
        );
    }

    // ── parse_rpu_nal_header synthetic-preamble tests ────────────────────────

    // 32. parse_rpu_nal_header: correctly extracts Profile 4 from synthetic preamble
    #[test]
    fn test_parse_rpu_nal_header_synthetic_p4() {
        let payload = make_synthetic_rpu_preamble(4, 6);
        let (profile, level) = parse_rpu_nal_header(&payload).expect("parse P4 synthetic preamble");
        assert_eq!(profile, DolbyVisionProfile::Profile4);
        assert_eq!(level, 6);
    }

    // 33. parse_rpu_nal_header: correctly extracts Profile 5 from synthetic preamble
    #[test]
    fn test_parse_rpu_nal_header_synthetic_p5() {
        let payload = make_synthetic_rpu_preamble(5, 1);
        let (profile, level) = parse_rpu_nal_header(&payload).expect("parse P5 synthetic preamble");
        assert_eq!(profile, DolbyVisionProfile::Profile5);
        assert_eq!(level, 1);
    }

    // 34. parse_rpu_nal_header: correctly extracts Profile 7 from synthetic preamble
    #[test]
    fn test_parse_rpu_nal_header_synthetic_p7() {
        let payload = make_synthetic_rpu_preamble(7, 6);
        let (profile, level) = parse_rpu_nal_header(&payload).expect("parse P7 synthetic preamble");
        assert_eq!(profile, DolbyVisionProfile::Profile7);
        assert_eq!(level, 6);
    }

    // 35. parse_rpu_nal_header: correctly extracts Profile 8 from synthetic preamble
    #[test]
    fn test_parse_rpu_nal_header_synthetic_p8() {
        let payload = make_synthetic_rpu_preamble(8, 6);
        let (profile, level) = parse_rpu_nal_header(&payload).expect("parse P8 synthetic preamble");
        assert_eq!(profile, DolbyVisionProfile::Profile8);
        assert_eq!(level, 6);
    }

    // 36. parse_rpu_nal_header: correctly extracts Profile 9 from synthetic preamble
    #[test]
    fn test_parse_rpu_nal_header_synthetic_p9() {
        let payload = make_synthetic_rpu_preamble(9, 3);
        let (profile, level) = parse_rpu_nal_header(&payload).expect("parse P9 synthetic preamble");
        assert_eq!(profile, DolbyVisionProfile::Profile9);
        assert_eq!(level, 3);
    }

    // 37. parse_rpu_nal_header: errors on payload shorter than 6 bytes
    #[test]
    fn test_parse_rpu_nal_header_too_short() {
        // Exactly 5 bytes — the function requires >= 6
        let payload: Vec<u8> = vec![0x19, 0x02, 0x01, 0x80, 0x08];
        assert!(
            parse_rpu_nal_header(&payload).is_err(),
            "5-byte payload must fail (minimum is 6)"
        );
    }

    // 38. parse_rpu_nal_header: errors on empty payload
    #[test]
    fn test_parse_rpu_nal_header_empty_payload() {
        assert!(
            parse_rpu_nal_header(&[]).is_err(),
            "empty payload must fail"
        );
    }

    // 39. parse_rpu_nal_header: errors on unsupported profile number 0
    #[test]
    fn test_parse_rpu_nal_header_unknown_profile_zero() {
        let payload = make_synthetic_rpu_preamble(0, 6);
        assert!(
            parse_rpu_nal_header(&payload).is_err(),
            "profile 0 is not a supported DV profile"
        );
    }

    // 40. parse_rpu_nal_header: errors on unsupported profile number 99
    #[test]
    fn test_parse_rpu_nal_header_unknown_profile_99() {
        let payload = make_synthetic_rpu_preamble(99, 6);
        assert!(
            parse_rpu_nal_header(&payload).is_err(),
            "profile 99 is not a supported DV profile"
        );
    }

    // ── verify_rpu_crc tests ─────────────────────────────────────────────────

    // 41. verify_rpu_crc: passes on freshly generated Profile8 NAL
    #[test]
    fn test_verify_rpu_crc_valid_p8() {
        let cfg = RpuGenerationConfig::profile8_default();
        let nal = generate_rpu_nal(&cfg).expect("generate P8 for CRC verify");
        verify_rpu_crc(&nal.payload).expect("CRC must be valid on a freshly generated P8 NAL");
    }

    // 42. verify_rpu_crc: passes on freshly generated Profile5 NAL
    #[test]
    fn test_verify_rpu_crc_valid_p5() {
        let cfg = RpuGenerationConfig::profile5_default();
        let nal = generate_rpu_nal(&cfg).expect("generate P5 for CRC verify");
        verify_rpu_crc(&nal.payload).expect("CRC must be valid on a freshly generated P5 NAL");
    }

    // 43. verify_rpu_crc: detects single-byte corruption in the middle of the payload
    #[test]
    fn test_verify_rpu_crc_corrupted_middle_byte() {
        let cfg = RpuGenerationConfig::profile8_default();
        let nal = generate_rpu_nal(&cfg).expect("generate P8 for corruption test");
        let mut corrupted = nal.payload.clone();
        // Flip all bits in a byte midway through the data section (well before the CRC)
        let mid = corrupted.len() / 2;
        corrupted[mid] ^= 0xFF;
        assert!(
            verify_rpu_crc(&corrupted).is_err(),
            "single-byte mid-payload corruption must fail CRC"
        );
    }

    // 44. verify_rpu_crc: detects corruption in the mastering luminance field (bytes 8-11)
    #[test]
    fn test_verify_rpu_crc_corrupted_luminance_field() {
        let cfg = RpuGenerationConfig::profile8_default();
        let nal = generate_rpu_nal(&cfg).expect("generate P8 for luminance corruption test");
        let mut corrupted = nal.payload.clone();
        // max_display_mastering_luminance high byte is at index 8
        corrupted[8] ^= 0x01;
        assert!(
            verify_rpu_crc(&corrupted).is_err(),
            "luminance-field corruption must fail CRC"
        );
    }

    // 45. verify_rpu_crc: detects corruption of the profile byte (index 4)
    #[test]
    fn test_verify_rpu_crc_corrupted_profile_byte() {
        let cfg = RpuGenerationConfig::profile8_default();
        let nal = generate_rpu_nal(&cfg).expect("generate P8 for profile-byte corruption test");
        let mut corrupted = nal.payload.clone();
        // Profile number is at index 4
        corrupted[4] ^= 0xFF;
        assert!(
            verify_rpu_crc(&corrupted).is_err(),
            "profile-byte corruption must fail CRC"
        );
    }

    // 46. verify_rpu_crc: detects corruption of the stored CRC bytes themselves
    #[test]
    fn test_verify_rpu_crc_corrupted_crc_bytes() {
        let cfg = RpuGenerationConfig::profile8_default();
        let nal = generate_rpu_nal(&cfg).expect("generate P8 for CRC-byte corruption test");
        let mut corrupted = nal.payload.clone();
        let len = corrupted.len();
        // Flip the last CRC byte (LSB of the big-endian CRC word)
        corrupted[len - 1] ^= 0xFF;
        assert!(
            verify_rpu_crc(&corrupted).is_err(),
            "corrupted stored CRC must fail verification"
        );
    }

    // 47. verify_rpu_crc: errors on payload shorter than 5 bytes
    #[test]
    fn test_verify_rpu_crc_too_short() {
        let payload: Vec<u8> = vec![0x19, 0x02, 0x01]; // 3 bytes, minimum is 5
        assert!(
            verify_rpu_crc(&payload).is_err(),
            "too-short payload must fail CRC verification"
        );
    }

    // ── Full-pipeline round-trip tests ───────────────────────────────────────

    // 48. Round-trip: generate → CRC verify → header parse for all five DV profiles
    #[test]
    fn test_rpu_roundtrip_all_profiles() {
        let test_cases: &[(DolbyVisionProfile, BlSignalCompatibility, u8)] = &[
            (
                DolbyVisionProfile::Profile4,
                BlSignalCompatibility::Hdr10,
                6,
            ),
            (
                DolbyVisionProfile::Profile5,
                BlSignalCompatibility::DolbyVision,
                6,
            ),
            (
                DolbyVisionProfile::Profile7,
                BlSignalCompatibility::DolbyVision,
                6,
            ),
            (
                DolbyVisionProfile::Profile8,
                BlSignalCompatibility::Hdr10,
                6,
            ),
            (DolbyVisionProfile::Profile9, BlSignalCompatibility::Sdr, 3),
        ];

        for &(expected_profile, compat, level) in test_cases {
            let cfg = RpuGenerationConfig {
                profile: expected_profile,
                bl_signal_compat: compat,
                vdr_rpu_level: level,
                max_display_mastering_luminance: 1000,
                min_display_mastering_luminance: 1,
                num_pivots: 3,
                pivots: vec![0, 1023, 2048, 4095],
                mapping_order: vec![1, 1, 1],
            };
            let nal = generate_rpu_nal(&cfg).unwrap_or_else(|e| {
                panic!("generate_rpu_nal failed for {expected_profile:?}: {e}")
            });

            verify_rpu_crc(&nal.payload)
                .unwrap_or_else(|e| panic!("CRC failed for {expected_profile:?}: {e}"));

            let (parsed_profile, parsed_level) =
                parse_rpu_nal_header(&nal.payload).unwrap_or_else(|e| {
                    panic!("parse_rpu_nal_header failed for {expected_profile:?}: {e}")
                });

            assert_eq!(
                parsed_profile, expected_profile,
                "profile mismatch in round-trip"
            );
            assert_eq!(parsed_level, level, "level mismatch in round-trip");
            assert_eq!(nal.profile, expected_profile, "nal.profile field mismatch");
        }
    }

    // 49. Edge: maximum allowed pivot count (8 pivots, 9 pivot values)
    #[test]
    fn test_generate_rpu_nal_max_pivots() {
        let cfg = RpuGenerationConfig {
            profile: DolbyVisionProfile::Profile8,
            bl_signal_compat: BlSignalCompatibility::Hdr10,
            vdr_rpu_level: 6,
            max_display_mastering_luminance: 10_000,
            min_display_mastering_luminance: 1,
            num_pivots: 8,
            pivots: vec![0, 455, 910, 1365, 1820, 2275, 2730, 3640, 4095],
            mapping_order: vec![1, 1, 1, 1, 1, 1, 1, 1],
        };
        let nal = generate_rpu_nal(&cfg).expect("8-pivot P8 NAL must succeed");
        verify_rpu_crc(&nal.payload).expect("CRC on 8-pivot P8 NAL must pass");
        let (profile, _) = parse_rpu_nal_header(&nal.payload).expect("parse 8-pivot P8 header");
        assert_eq!(profile, DolbyVisionProfile::Profile8);
    }

    // 50. crc32 field in RpuNalUnit agrees with the last 4 bytes of the payload
    #[test]
    fn test_generate_rpu_nal_crc32_field_matches_payload() {
        let cfg = RpuGenerationConfig::profile8_default();
        let nal = generate_rpu_nal(&cfg).expect("generate P8 for CRC field check");
        let len = nal.payload.len();
        let crc_from_payload = u32::from_be_bytes([
            nal.payload[len - 4],
            nal.payload[len - 3],
            nal.payload[len - 2],
            nal.payload[len - 1],
        ]);
        assert_eq!(
            nal.crc32, crc_from_payload,
            "RpuNalUnit::crc32 must equal the last 4 payload bytes"
        );
    }
}
