//! Dolby Vision Profile 8 encoder.
//!
//! Profile 8 is the "base-layer only" profile, using a single video track with
//! an attached RPU NAL. Two sub-variants are defined:
//!
//! - **8.1**: HDR10 compatible base layer (PQ EOTF, BT.2020 primaries)
//! - **8.4**: HLG compatible base layer (HLG OETF, BT.2020 primaries)
//!
//! Because Profile 8 carries no enhancement layer, the base-layer signal is
//! fully decodable by non-Dolby displays.  The RPU provides per-frame trim
//! metadata that allows Dolby Vision capable displays to apply DM processing.

use crate::metadata::Level1Metadata;
use thiserror::Error;

// ─── Error ───────────────────────────────────────────────────────────────────

/// Errors specific to Profile 8 encoding.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Dv8Error {
    /// The supplied [`Dv8Config`] is semantically invalid.
    #[error("Invalid Dolby Vision 8 config: {0}")]
    InvalidConfig(String),

    /// The frame byte slice was empty or has an unexpected length.
    #[error("Invalid frame data: {0}")]
    InvalidFrameData(String),

    /// The requested profile variant is not supported by this encoder.
    #[error("Unsupported profile variant: {0}")]
    UnsupportedVariant(String),

    /// A low-level encoding step failed.
    #[error("Encoding failed: {0}")]
    EncodingFailed(String),
}

// ─── Sub-types ───────────────────────────────────────────────────────────────

/// The transfer function / primaries of the HDR base layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrBaseLayer {
    /// Perceptual Quantizer (SMPTE ST 2084) – HDR10 compatible.
    Hdr10,
    /// Hybrid Log-Gamma (ITU-R BT.2100) – HLG compatible.
    Hlg,
    /// Standard Dynamic Range (BT.1886 / gamma 2.4).
    Sdr,
}

impl HdrBaseLayer {
    /// Returns `true` when the base layer uses PQ transfer function.
    #[must_use]
    pub fn is_pq(self) -> bool {
        matches!(self, Self::Hdr10)
    }

    /// Returns `true` when the base layer uses HLG transfer function.
    #[must_use]
    pub fn is_hlg(self) -> bool {
        matches!(self, Self::Hlg)
    }

    /// Human-readable name of the base layer.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Hdr10 => "HDR10 (PQ)",
            Self::Hlg => "HLG",
            Self::Sdr => "SDR (BT.1886)",
        }
    }
}

/// Color space mapping applied inside the RPU enhancement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpaceMapping {
    /// No color-space conversion – passthrough.
    Identity,
    /// BT.709 → BT.2020 gamut expansion.
    BT709ToBT2020,
    /// HLG → PQ signal re-mapping.
    HlgToPq,
}

impl ColorSpaceMapping {
    /// Returns `true` when no transform is needed.
    #[must_use]
    pub fn is_identity(self) -> bool {
        matches!(self, Self::Identity)
    }
}

/// Optional RPU enhancement layer parameters.
#[derive(Debug, Clone)]
pub struct RpuEnhancement {
    /// Enhancement strength in [0.0, 1.0].  0.0 = passthrough.
    pub enhancement_factor: f32,
    /// Color-space mapping applied during enhancement processing.
    pub color_space_mapping: ColorSpaceMapping,
}

impl RpuEnhancement {
    /// Create a new enhancement descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`Dv8Error::InvalidConfig`] when `enhancement_factor` is outside
    /// the valid range `[0.0, 1.0]`.
    pub fn new(
        enhancement_factor: f32,
        color_space_mapping: ColorSpaceMapping,
    ) -> Result<Self, Dv8Error> {
        if !(0.0..=1.0).contains(&enhancement_factor) {
            return Err(Dv8Error::InvalidConfig(format!(
                "enhancement_factor {enhancement_factor} is outside [0.0, 1.0]"
            )));
        }
        Ok(Self {
            enhancement_factor,
            color_space_mapping,
        })
    }

    /// Returns `true` when the enhancement layer is active (factor > 0.0).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.enhancement_factor > 0.0
    }
}

/// Profile 8 sub-variant selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile8Variant {
    /// Profile 8.1 — HDR10 compatible (PQ EOTF).
    V8_1,
    /// Profile 8.4 — HLG compatible (HLG OETF).
    V8_4,
}

impl Profile8Variant {
    /// Returns `true` for Profile 8.1 (HDR10 compatible).
    #[must_use]
    pub fn is_hdr10(self) -> bool {
        matches!(self, Self::V8_1)
    }

    /// Returns `true` for Profile 8.4 (HLG compatible).
    #[must_use]
    pub fn is_hlg(self) -> bool {
        matches!(self, Self::V8_4)
    }

    /// Expected base-layer type for this variant.
    #[must_use]
    pub fn expected_base_layer(self) -> HdrBaseLayer {
        match self {
            Self::V8_1 => HdrBaseLayer::Hdr10,
            Self::V8_4 => HdrBaseLayer::Hlg,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::V8_1 => "Profile 8.1 (HDR10)",
            Self::V8_4 => "Profile 8.4 (HLG)",
        }
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a Profile 8 encode session.
#[derive(Debug, Clone)]
pub struct Dv8Config {
    /// Transfer function of the base-layer signal.
    pub base_layer: HdrBaseLayer,
    /// Optional RPU enhancement parameters.
    pub enhancement_layer: Option<RpuEnhancement>,
    /// Exact Profile 8 sub-variant (8.1 or 8.4).
    pub profile_variant: Profile8Variant,
}

impl Dv8Config {
    /// Create a Profile 8.1 (HDR10 compatible) configuration.
    #[must_use]
    pub fn profile_8_1() -> Self {
        Self {
            base_layer: HdrBaseLayer::Hdr10,
            enhancement_layer: None,
            profile_variant: Profile8Variant::V8_1,
        }
    }

    /// Create a Profile 8.4 (HLG compatible) configuration.
    #[must_use]
    pub fn profile_8_4() -> Self {
        Self {
            base_layer: HdrBaseLayer::Hlg,
            enhancement_layer: None,
            profile_variant: Profile8Variant::V8_4,
        }
    }

    /// Validate that the config is internally consistent.
    ///
    /// # Errors
    ///
    /// Returns [`Dv8Error::InvalidConfig`] when the base layer type is
    /// inconsistent with the declared profile variant.
    pub fn validate(&self) -> Result<(), Dv8Error> {
        let expected = self.profile_variant.expected_base_layer();
        if self.base_layer != expected {
            return Err(Dv8Error::InvalidConfig(format!(
                "{} requires {:?} base layer, got {:?}",
                self.profile_variant.name(),
                expected,
                self.base_layer,
            )));
        }
        Ok(())
    }
}

// ─── Output ──────────────────────────────────────────────────────────────────

/// Encoded Profile 8 output for a single video frame.
#[derive(Debug, Clone)]
pub struct Dv8Packet {
    /// Raw base-layer bytes (PQ or HLG video signal).
    pub base_layer: Vec<u8>,
    /// Serialised RPU NAL unit, if present.
    pub rpu: Option<Vec<u8>>,
    /// Profile variant that produced this packet.
    pub profile_variant: Profile8Variant,
    /// Zero-based frame ordinal within the encode session.
    pub frame_index: u64,
}

impl Dv8Packet {
    /// Returns `true` when an RPU is attached to this packet.
    #[must_use]
    pub fn has_rpu(&self) -> bool {
        self.rpu.is_some()
    }

    /// Total byte size of the packet (base layer + optional RPU).
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        let rpu_len = self.rpu.as_ref().map_or(0, |r| r.len());
        self.base_layer.len() + rpu_len
    }
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

/// Stateful Profile 8 encoder.
///
/// The encoder maintains a monotonically increasing frame counter and derives
/// Level-1 PQ statistics from each frame when an RPU is required.
pub struct Dv8Encoder {
    /// Default configuration used when no per-frame config override is provided.
    pub config: Dv8Config,
    frame_counter: u64,
}

impl Dv8Encoder {
    /// Construct a new encoder from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Dv8Error::InvalidConfig`] when `config.validate()` fails.
    pub fn new(config: Dv8Config) -> Result<Self, Dv8Error> {
        config.validate()?;
        Ok(Self {
            config,
            frame_counter: 0,
        })
    }

    /// Number of frames encoded so far.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_counter
    }

    /// Encode a single frame using auto-computed Level-1 statistics.
    ///
    /// # Errors
    ///
    /// Returns [`Dv8Error::InvalidFrameData`] when `frame` is empty.
    pub fn encode_frame(
        &mut self,
        frame: &[u8],
        config: &Dv8Config,
    ) -> Result<Dv8Packet, Dv8Error> {
        if frame.is_empty() {
            return Err(Dv8Error::InvalidFrameData(
                "frame data must not be empty".to_string(),
            ));
        }
        config.validate()?;

        let level1 = compute_level1_from_bytes(frame);
        self.encode_inner(frame, config, Some(level1))
    }

    /// Encode a frame with explicit Level-1 metadata supplied by the caller.
    ///
    /// # Errors
    ///
    /// Returns [`Dv8Error::InvalidFrameData`] when `frame` is empty.
    pub fn encode_frame_with_metadata(
        &mut self,
        frame: &[u8],
        config: &Dv8Config,
        level1: Level1Metadata,
    ) -> Result<Dv8Packet, Dv8Error> {
        if frame.is_empty() {
            return Err(Dv8Error::InvalidFrameData(
                "frame data must not be empty".to_string(),
            ));
        }
        config.validate()?;
        self.encode_inner(frame, config, Some(level1))
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn encode_inner(
        &mut self,
        frame: &[u8],
        config: &Dv8Config,
        level1: Option<Level1Metadata>,
    ) -> Result<Dv8Packet, Dv8Error> {
        let idx = self.frame_counter;
        self.frame_counter += 1;

        let base_layer = frame.to_vec();

        // Profile 8.4 (HLG): no RPU needed – HLG signal is self-describing.
        // Profile 8.1 (HDR10): attach a minimal RPU with L1 trim metadata.
        let rpu = match config.profile_variant {
            Profile8Variant::V8_4 => None,
            Profile8Variant::V8_1 => {
                let l1 = level1.unwrap_or_default();
                Some(build_minimal_rpu(&l1, idx))
            }
        };

        Ok(Dv8Packet {
            base_layer,
            rpu,
            profile_variant: config.profile_variant,
            frame_index: idx,
        })
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Derive Level-1 PQ statistics from raw frame bytes.
///
/// Each pair of consecutive bytes is interpreted as a 16-bit big-endian value
/// that is mapped to the 12-bit PQ range (0–4095) by right-shifting 4 bits.
fn compute_level1_from_bytes(frame: &[u8]) -> Level1Metadata {
    if frame.is_empty() {
        return Level1Metadata::default();
    }

    // Sample every other byte pair to keep O(n) but fast.
    let mut min_pq: u16 = 4095;
    let mut max_pq: u16 = 0;
    let mut sum: u64 = 0;
    let mut count: u64 = 0;

    let pairs = frame.chunks(2);
    for chunk in pairs {
        let raw: u16 = if chunk.len() == 2 {
            (u16::from(chunk[0]) << 8) | u16::from(chunk[1])
        } else {
            u16::from(chunk[0]) << 8
        };
        // Map 16-bit → 12-bit PQ
        let pq = raw >> 4;
        if pq < min_pq {
            min_pq = pq;
        }
        if pq > max_pq {
            max_pq = pq;
        }
        sum += u64::from(pq);
        count += 1;
    }

    let avg_pq = sum.checked_div(count).unwrap_or(0) as u16;

    Level1Metadata {
        min_pq,
        max_pq,
        avg_pq,
    }
}

/// Build a minimal RPU byte payload carrying Level-1 metadata.
///
/// The format is intentionally simple: a 1-byte marker, 2-byte frame index,
/// and the three 2-byte PQ values from Level-1 metadata.  This is sufficient
/// for unit-test verification without duplicating the full RPU bitstream writer.
fn build_minimal_rpu(level1: &Level1Metadata, frame_index: u64) -> Vec<u8> {
    // Marker byte | frame_index (u64, 8 bytes) | min_pq | avg_pq | max_pq
    let mut rpu = Vec::with_capacity(15);
    rpu.push(0x7C); // RPU NAL type marker (simplified)
    let idx_bytes = frame_index.to_be_bytes();
    rpu.extend_from_slice(&idx_bytes);
    rpu.extend_from_slice(&level1.min_pq.to_be_bytes());
    rpu.extend_from_slice(&level1.avg_pq.to_be_bytes());
    rpu.extend_from_slice(&level1.max_pq.to_be_bytes());
    rpu
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── HdrBaseLayer ─────────────────────────────────────────────────────────

    #[test]
    fn test_hdr_base_layer_is_pq() {
        assert!(HdrBaseLayer::Hdr10.is_pq());
        assert!(!HdrBaseLayer::Hlg.is_pq());
        assert!(!HdrBaseLayer::Sdr.is_pq());
    }

    #[test]
    fn test_hdr_base_layer_is_hlg() {
        assert!(HdrBaseLayer::Hlg.is_hlg());
        assert!(!HdrBaseLayer::Hdr10.is_hlg());
    }

    #[test]
    fn test_hdr_base_layer_name() {
        assert_eq!(HdrBaseLayer::Hdr10.name(), "HDR10 (PQ)");
        assert_eq!(HdrBaseLayer::Hlg.name(), "HLG");
        assert_eq!(HdrBaseLayer::Sdr.name(), "SDR (BT.1886)");
    }

    // ── ColorSpaceMapping ────────────────────────────────────────────────────

    #[test]
    fn test_color_space_mapping_identity() {
        assert!(ColorSpaceMapping::Identity.is_identity());
        assert!(!ColorSpaceMapping::BT709ToBT2020.is_identity());
        assert!(!ColorSpaceMapping::HlgToPq.is_identity());
    }

    // ── RpuEnhancement ───────────────────────────────────────────────────────

    #[test]
    fn test_rpu_enhancement_valid() {
        let enh = RpuEnhancement::new(0.5, ColorSpaceMapping::Identity);
        assert!(enh.is_ok());
        let enh = enh.expect("valid RpuEnhancement");
        assert!(enh.is_active());
        assert_eq!(enh.color_space_mapping, ColorSpaceMapping::Identity);
    }

    #[test]
    fn test_rpu_enhancement_passthrough() {
        let enh = RpuEnhancement::new(0.0, ColorSpaceMapping::Identity)
            .expect("passthrough enhancement should be valid");
        assert!(!enh.is_active());
    }

    #[test]
    fn test_rpu_enhancement_out_of_range() {
        let result = RpuEnhancement::new(1.5, ColorSpaceMapping::Identity);
        assert!(result.is_err());
        let result2 = RpuEnhancement::new(-0.1, ColorSpaceMapping::Identity);
        assert!(result2.is_err());
    }

    // ── Profile8Variant ──────────────────────────────────────────────────────

    #[test]
    fn test_profile8_variant_properties() {
        assert!(Profile8Variant::V8_1.is_hdr10());
        assert!(!Profile8Variant::V8_1.is_hlg());
        assert!(Profile8Variant::V8_4.is_hlg());
        assert!(!Profile8Variant::V8_4.is_hdr10());
    }

    #[test]
    fn test_profile8_variant_expected_base_layer() {
        assert_eq!(
            Profile8Variant::V8_1.expected_base_layer(),
            HdrBaseLayer::Hdr10
        );
        assert_eq!(
            Profile8Variant::V8_4.expected_base_layer(),
            HdrBaseLayer::Hlg
        );
    }

    #[test]
    fn test_profile8_variant_name() {
        assert!(Profile8Variant::V8_1.name().contains("8.1"));
        assert!(Profile8Variant::V8_4.name().contains("8.4"));
    }

    // ── Dv8Config ────────────────────────────────────────────────────────────

    #[test]
    fn test_config_profile_8_1_valid() {
        let cfg = Dv8Config::profile_8_1();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.profile_variant, Profile8Variant::V8_1);
        assert_eq!(cfg.base_layer, HdrBaseLayer::Hdr10);
    }

    #[test]
    fn test_config_profile_8_4_valid() {
        let cfg = Dv8Config::profile_8_4();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.profile_variant, Profile8Variant::V8_4);
        assert_eq!(cfg.base_layer, HdrBaseLayer::Hlg);
    }

    #[test]
    fn test_config_mismatch_invalid() {
        // 8.4 config but using HDR10 base layer – invalid
        let cfg = Dv8Config {
            base_layer: HdrBaseLayer::Hdr10,
            enhancement_layer: None,
            profile_variant: Profile8Variant::V8_4,
        };
        assert!(cfg.validate().is_err());
    }

    // ── Dv8Encoder construction ───────────────────────────────────────────────

    #[test]
    fn test_encoder_new_valid() {
        let cfg = Dv8Config::profile_8_1();
        let enc = Dv8Encoder::new(cfg);
        assert!(enc.is_ok());
        assert_eq!(
            enc.expect("encoder should construct from valid config")
                .frame_count(),
            0
        );
    }

    #[test]
    fn test_encoder_new_invalid_config() {
        let cfg = Dv8Config {
            base_layer: HdrBaseLayer::Hlg,
            enhancement_layer: None,
            profile_variant: Profile8Variant::V8_1,
        };
        assert!(Dv8Encoder::new(cfg).is_err());
    }

    // ── encode_frame (8.1) ────────────────────────────────────────────────────

    #[test]
    fn test_encode_frame_8_1_basic() {
        let cfg = Dv8Config::profile_8_1();
        let mut enc = Dv8Encoder::new(cfg.clone()).expect("create 8.1 encoder");
        let frame: Vec<u8> = (0u8..=255).collect();
        let pkt = enc.encode_frame(&frame, &cfg).expect("encode 8.1 frame");
        assert_eq!(pkt.base_layer, frame);
        assert!(pkt.has_rpu(), "Profile 8.1 must include RPU");
        assert_eq!(pkt.frame_index, 0);
        assert_eq!(pkt.profile_variant, Profile8Variant::V8_1);
    }

    #[test]
    fn test_encode_frame_8_1_rpu_contains_marker() {
        let cfg = Dv8Config::profile_8_1();
        let mut enc = Dv8Encoder::new(cfg.clone()).expect("create 8.1 encoder for RPU test");
        let frame = vec![0xAA_u8; 64];
        let pkt = enc
            .encode_frame(&frame, &cfg)
            .expect("encode frame for RPU marker test");
        let rpu = pkt.rpu.expect("profile 8.1 packet must contain RPU");
        assert_eq!(rpu[0], 0x7C, "RPU must start with NAL marker byte");
    }

    #[test]
    fn test_encode_frame_8_4_no_rpu() {
        let cfg = Dv8Config::profile_8_4();
        let mut enc = Dv8Encoder::new(cfg.clone()).expect("create 8.4 encoder");
        let frame = vec![0x55_u8; 32];
        let pkt = enc.encode_frame(&frame, &cfg).expect("encode 8.4 frame");
        assert!(!pkt.has_rpu(), "Profile 8.4 must NOT include RPU");
        assert_eq!(pkt.base_layer, frame);
    }

    #[test]
    fn test_encode_frame_empty_error() {
        let cfg = Dv8Config::profile_8_1();
        let mut enc = Dv8Encoder::new(cfg.clone()).expect("create encoder for empty-frame test");
        let result = enc.encode_frame(&[], &cfg);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Dv8Error::InvalidFrameData(_)));
    }

    #[test]
    fn test_encode_frame_count_increments() {
        let cfg = Dv8Config::profile_8_1();
        let mut enc = Dv8Encoder::new(cfg.clone()).expect("create encoder for frame-count test");
        let frame = vec![0xFF_u8; 16];
        for i in 0..5u64 {
            let pkt = enc
                .encode_frame(&frame, &cfg)
                .expect("encode frame in count test");
            assert_eq!(pkt.frame_index, i);
        }
        assert_eq!(enc.frame_count(), 5);
    }

    #[test]
    fn test_encode_multiple_frames_sequential_indices() {
        let cfg = Dv8Config::profile_8_4();
        let mut enc =
            Dv8Encoder::new(cfg.clone()).expect("create encoder for sequential-indices test");
        let frame = vec![0x10_u8; 128];
        let p0 = enc.encode_frame(&frame, &cfg).expect("encode frame 0");
        let p1 = enc.encode_frame(&frame, &cfg).expect("encode frame 1");
        let p2 = enc.encode_frame(&frame, &cfg).expect("encode frame 2");
        assert_eq!(p0.frame_index, 0);
        assert_eq!(p1.frame_index, 1);
        assert_eq!(p2.frame_index, 2);
    }

    // ── encode_frame_with_metadata ────────────────────────────────────────────

    #[test]
    fn test_encode_frame_with_metadata() {
        let cfg = Dv8Config::profile_8_1();
        let mut enc = Dv8Encoder::new(cfg.clone()).expect("create encoder for metadata test");
        let frame = vec![0x80_u8; 64];
        let l1 = Level1Metadata {
            min_pq: 100,
            avg_pq: 2000,
            max_pq: 3500,
        };
        let pkt = enc
            .encode_frame_with_metadata(&frame, &cfg, l1)
            .expect("encode frame with L1 metadata");
        assert!(pkt.has_rpu());
        // RPU encodes L1 values at known byte offsets
        let rpu = pkt.rpu.expect("RPU must be present for metadata test");
        let min_pq = u16::from_be_bytes([rpu[9], rpu[10]]);
        let avg_pq = u16::from_be_bytes([rpu[11], rpu[12]]);
        let max_pq = u16::from_be_bytes([rpu[13], rpu[14]]);
        assert_eq!(min_pq, 100);
        assert_eq!(avg_pq, 2000);
        assert_eq!(max_pq, 3500);
    }

    #[test]
    fn test_encode_frame_with_metadata_empty_error() {
        let cfg = Dv8Config::profile_8_1();
        let mut enc =
            Dv8Encoder::new(cfg.clone()).expect("create encoder for empty metadata error test");
        let l1 = Level1Metadata::default();
        let result = enc.encode_frame_with_metadata(&[], &cfg, l1);
        assert!(result.is_err());
    }

    // ── Dv8Packet ─────────────────────────────────────────────────────────────

    #[test]
    fn test_dv8_packet_total_bytes() {
        let cfg = Dv8Config::profile_8_1();
        let mut enc = Dv8Encoder::new(cfg.clone()).expect("create encoder for total-bytes test");
        let frame = vec![0xAB_u8; 100];
        let pkt = enc
            .encode_frame(&frame, &cfg)
            .expect("encode frame for total-bytes test");
        assert!(pkt.total_bytes() >= 100);
    }

    #[test]
    fn test_dv8_packet_base_layer_preserved() {
        let cfg = Dv8Config::profile_8_4();
        let mut enc = Dv8Encoder::new(cfg.clone()).expect("create encoder for base-layer test");
        let frame: Vec<u8> = (0..200u8).collect();
        let pkt = enc
            .encode_frame(&frame, &cfg)
            .expect("encode frame for base-layer preservation test");
        assert_eq!(
            pkt.base_layer, frame,
            "base layer must be identical to input"
        );
    }

    // ── Dv8Error display ─────────────────────────────────────────────────────

    #[test]
    fn test_dv8_error_display() {
        let e = Dv8Error::InvalidConfig("test".to_string());
        assert!(e.to_string().contains("test"));
        let e2 = Dv8Error::InvalidFrameData("empty".to_string());
        assert!(e2.to_string().contains("empty"));
        let e3 = Dv8Error::UnsupportedVariant("v9".to_string());
        assert!(e3.to_string().contains("v9"));
        let e4 = Dv8Error::EncodingFailed("overflow".to_string());
        assert!(e4.to_string().contains("overflow"));
    }

    // ── Enhancement layer round-trip ─────────────────────────────────────────

    #[test]
    fn test_config_with_enhancement_layer() {
        let enh = RpuEnhancement::new(0.75, ColorSpaceMapping::BT709ToBT2020)
            .expect("create enhancement layer");
        let cfg = Dv8Config {
            base_layer: HdrBaseLayer::Hdr10,
            enhancement_layer: Some(enh),
            profile_variant: Profile8Variant::V8_1,
        };
        assert!(cfg.validate().is_ok());
        assert!(cfg.enhancement_layer.is_some());
        let e = cfg
            .enhancement_layer
            .expect("enhancement layer should be Some");
        assert!((e.enhancement_factor - 0.75).abs() < 1e-6);
        assert_eq!(e.color_space_mapping, ColorSpaceMapping::BT709ToBT2020);
    }
}
