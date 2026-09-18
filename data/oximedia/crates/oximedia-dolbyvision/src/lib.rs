//! Dolby Vision RPU (Reference Processing Unit) metadata parser and writer.
//!
//! This crate provides **metadata-only** Dolby Vision support, respecting Dolby's intellectual property.
//! It can parse and generate RPU metadata structures but does not implement proprietary encoding algorithms.
//!
//! # Supported Profiles
//!
//! - **Profile 5**: IPT-PQ, backward compatible with HDR10
//! - **Profile 7**: MEL (Metadata Enhancement Layer) + BL (Base Layer), single track
//! - **Profile 8**: BL only, backward compatible with HDR10
//! - **Profile 8.1**: Low-latency variant of Profile 8
//! - **Profile 8.4**: HLG-based, backward compatible with HLG
//!
//! # Examples
//!
//! ```rust
//! use oximedia_dolbyvision::{DolbyVisionRpu, Profile};
//!
//! // Create new RPU for Profile 8.4
//! let rpu = DolbyVisionRpu::new(Profile::Profile8_4);
//! assert_eq!(rpu.profile, Profile::Profile8_4);
//! ```

// SIMD intrinsics in ipt_pq_simd require unsafe blocks; all other modules are safe.
#![warn(unsafe_code)]
#![warn(missing_docs)]

pub mod ambient_metadata;
pub mod cm_analysis;
pub mod compat;
pub mod delivery_spec;
pub mod display_config;
pub mod dm_metadata;
pub mod dv_xml_metadata;
pub mod enhancement;
pub mod frame_analysis;
pub mod ipt_pq;
pub mod ipt_pq_simd;
pub mod level_analysis;
pub mod level_mapping;
pub mod mapping_curve;
pub mod mastering;
mod metadata;
pub mod metadata_block;
pub mod metadata_validator;
#[allow(dead_code)]
mod parser;
pub mod profile8;
pub mod profile_convert;
pub mod profiles;
mod rpu;
pub mod scene_trim;
pub mod shot_boundary;
pub mod shot_metadata;
pub mod shot_metadata_ext;
pub mod target_display;
pub mod tone_mapping;
mod tonemap;
pub mod trim_passes;
pub mod validation;
mod writer;
pub mod xml_metadata;

// ── Wave 6: orphan modules ────────────────────────────────────────────────────
// Wire dv_xml_export first; dv_compliance, rpu_statistics, rpu_validator depend on it.
pub mod dv_xml_export;

/// Dolby Vision compliance checker (profile constraints, L1/L2/L6 consistency).
pub mod dv_compliance;
/// RPU sequence statistics with PQ histogram and percentile computation (full).
pub mod rpu_statistics;
/// RPU metadata sequence validator (Basic / Strict / Broadcast levels, comprehensive).
pub mod rpu_validator;

// Remaining orphans (alphabetical):
pub mod auto_profile_detect;
pub mod batch_rpu;
pub mod conformance;
pub mod display_mapping;
/// HDR10+/DolbyVision extended bridge.
pub mod dv_hdr10plus_bridge;
pub mod gamut_mapping;
/// HDR10+ bridge (basic).
pub mod hdr10plus_bridge;
pub mod metadata_report;
pub mod profile10;
pub mod rpu_merge;
/// RPU statistics (lightweight).
pub mod rpu_stats;
pub mod rpu_timeline;
/// RPU validation (basic).
pub mod rpu_validate;
pub mod rpu_visualize;
pub mod scene_stats;
pub mod scene_trim_enhanced;
pub mod streaming_parser;
pub mod tone_map_preview;
pub mod trim_meta;
pub mod xml_manifest;

// Re-export types, but avoid ambiguous glob re-exports
pub use metadata::{
    ColorVolumeTransform, ContentMetadataDescriptor, ContentType, HueVector, Level10Metadata,
    Level11Metadata, Level1Metadata, Level2Metadata, Level3Metadata, Level4Metadata,
    Level5Metadata, Level6Metadata, Level7Metadata, Level8Metadata, Level9Metadata, MetadataBlock,
    SaturationVector, TrimPass,
};
pub use rpu::*;
pub use tonemap::{
    apply_dolbyvision_tonemap, apply_eotf, apply_inverse_eotf, bt1886_to_linear, hlg_constants,
    hlg_to_linear, linear_to_bt1886, linear_to_hlg, linear_to_pq, pq_constants, pq_to_linear,
    BilateralGrid, ColorVolumeLut, ReshapingLut, TonemapParams,
};

use oximedia_core::error::OxiError;
use thiserror::Error;

/// Errors that can occur during Dolby Vision RPU processing.
#[derive(Debug, Error)]
pub enum DolbyVisionError {
    /// Invalid RPU header
    #[error("Invalid RPU header: {0}")]
    InvalidHeader(String),

    /// Invalid RPU payload
    #[error("Invalid RPU payload: {0}")]
    InvalidPayload(String),

    /// Unsupported Dolby Vision profile
    #[error("Unsupported profile: {0}")]
    UnsupportedProfile(u8),

    /// Invalid NAL unit
    #[error("Invalid NAL unit: {0}")]
    InvalidNalUnit(String),

    /// CRC mismatch
    #[error("CRC mismatch: expected {expected:#x}, got {actual:#x}")]
    CrcMismatch {
        /// Expected CRC value
        expected: u32,
        /// Actual CRC value
        actual: u32,
    },

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic error
    #[error("{0}")]
    Generic(String),
}

impl From<DolbyVisionError> for OxiError {
    fn from(err: DolbyVisionError) -> Self {
        OxiError::Codec(err.to_string())
    }
}

/// Result type for Dolby Vision operations.
pub type Result<T> = std::result::Result<T, DolbyVisionError>;

/// Dolby Vision profile identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Profile {
    /// Profile 5: IPT-PQ, backward compatible with HDR10
    Profile5 = 5,
    /// Profile 7: MEL + BL, single track, full enhancement
    Profile7 = 7,
    /// Profile 8: BL only, backward compatible with HDR10
    Profile8 = 8,
    /// Profile 8.1: Low-latency variant
    Profile8_1 = 81,
    /// Profile 8.4: HLG-based, backward compatible with HLG
    Profile8_4 = 84,
}

impl Profile {
    /// Create profile from numeric value.
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            5 => Some(Self::Profile5),
            7 => Some(Self::Profile7),
            8 => Some(Self::Profile8),
            81 => Some(Self::Profile8_1),
            84 => Some(Self::Profile8_4),
            _ => None,
        }
    }

    /// Check if profile requires backward compatibility.
    #[must_use]
    pub const fn is_backward_compatible(self) -> bool {
        matches!(self, Self::Profile5 | Self::Profile8 | Self::Profile8_4)
    }

    /// Check if profile supports MEL (Metadata Enhancement Layer).
    #[must_use]
    pub const fn has_mel(self) -> bool {
        matches!(self, Self::Profile7)
    }

    /// Check if profile uses HLG transfer function.
    #[must_use]
    pub const fn is_hlg(self) -> bool {
        matches!(self, Self::Profile8_4)
    }

    /// Check if profile is low-latency variant.
    #[must_use]
    pub const fn is_low_latency(self) -> bool {
        matches!(self, Self::Profile8_1)
    }
}

/// Main Dolby Vision RPU (Reference Processing Unit) structure.
///
/// Contains all metadata required for Dolby Vision HDR display mapping.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DolbyVisionRpu {
    /// Dolby Vision profile
    pub profile: Profile,

    /// RPU header
    pub header: RpuHeader,

    /// VDR (Vizio Display Management) metadata
    pub vdr_dm_data: Option<VdrDmData>,

    /// Level 1 metadata (frame-level)
    pub level1: Option<Level1Metadata>,

    /// Level 2 metadata (trim passes)
    pub level2: Option<Level2Metadata>,

    /// Level 4 metadata (global dimming)
    pub level4: Option<Level4Metadata>,

    /// Level 5 metadata (active area)
    pub level5: Option<Level5Metadata>,

    /// Level 6 metadata (fallback)
    pub level6: Option<Level6Metadata>,

    /// Level 7 metadata (source display color volume)
    pub level7: Option<Level7Metadata>,

    /// Level 8 metadata (target display)
    pub level8: Option<Level8Metadata>,

    /// Level 9 metadata (source display)
    pub level9: Option<Level9Metadata>,

    /// Level 11 metadata (content type)
    pub level11: Option<Level11Metadata>,
}

impl DolbyVisionRpu {
    /// Create a new RPU with default values for the given profile.
    #[must_use]
    pub fn new(profile: Profile) -> Self {
        Self {
            profile,
            header: RpuHeader::default_for_profile(profile),
            vdr_dm_data: None,
            level1: None,
            level2: None,
            level4: None,
            level5: None,
            level6: None,
            level7: None,
            level8: None,
            level9: None,
            level11: None,
        }
    }

    /// Parse RPU from NAL unit bytes (including NAL header).
    ///
    /// # Errors
    ///
    /// Returns error if NAL unit is invalid or RPU parsing fails.
    pub fn parse_from_nal(data: &[u8]) -> Result<Self> {
        parser::parse_nal_unit(data)
    }

    /// Parse RPU from raw bitstream (without NAL wrapper).
    ///
    /// # Errors
    ///
    /// Returns error if RPU parsing fails.
    pub fn parse_from_bitstream(data: &[u8]) -> Result<Self> {
        parser::parse_rpu_bitstream(data)
    }

    /// Write RPU to NAL unit bytes (including NAL header).
    ///
    /// # Errors
    ///
    /// Returns error if writing fails.
    pub fn write_to_nal(&self) -> Result<Vec<u8>> {
        writer::write_nal_unit(self)
    }

    /// Write RPU to raw bitstream (without NAL wrapper).
    ///
    /// # Errors
    ///
    /// Returns error if writing fails.
    pub fn write_to_bitstream(&self) -> Result<Vec<u8>> {
        writer::write_rpu_bitstream(self)
    }

    /// Apply tone mapping to convert from source to target display.
    ///
    /// # Errors
    ///
    /// Returns error if tone mapping fails or required metadata is missing.
    pub fn apply_tonemap(&self, pixel_data: &mut [f32]) -> Result<()> {
        tonemap::apply_dolbyvision_tonemap(self, pixel_data)
    }

    /// Validate RPU structure for consistency, including all metadata levels.
    ///
    /// # Errors
    ///
    /// Returns error if validation fails.
    pub fn validate(&self) -> Result<()> {
        validation::validate_rpu(self)
    }
}

impl Default for DolbyVisionRpu {
    fn default() -> Self {
        Self::new(Profile::Profile8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_creation() {
        assert_eq!(Profile::from_u8(5), Some(Profile::Profile5));
        assert_eq!(Profile::from_u8(7), Some(Profile::Profile7));
        assert_eq!(Profile::from_u8(8), Some(Profile::Profile8));
        assert_eq!(Profile::from_u8(81), Some(Profile::Profile8_1));
        assert_eq!(Profile::from_u8(84), Some(Profile::Profile8_4));
        assert_eq!(Profile::from_u8(99), None);
    }

    #[test]
    fn test_profile_properties() {
        assert!(Profile::Profile5.is_backward_compatible());
        assert!(Profile::Profile8.is_backward_compatible());
        assert!(Profile::Profile8_4.is_backward_compatible());
        assert!(!Profile::Profile7.is_backward_compatible());

        assert!(Profile::Profile7.has_mel());
        assert!(!Profile::Profile8.has_mel());

        assert!(Profile::Profile8_4.is_hlg());
        assert!(!Profile::Profile8.is_hlg());

        assert!(Profile::Profile8_1.is_low_latency());
        assert!(!Profile::Profile8.is_low_latency());
    }

    #[test]
    fn test_rpu_creation() {
        let rpu = DolbyVisionRpu::new(Profile::Profile8);
        assert_eq!(rpu.profile, Profile::Profile8);
        assert!(rpu.validate().is_ok());
    }

    #[test]
    fn test_default_rpu() {
        let rpu = DolbyVisionRpu::default();
        assert_eq!(rpu.profile, Profile::Profile8);
    }
}
