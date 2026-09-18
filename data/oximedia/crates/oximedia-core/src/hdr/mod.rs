//! HDR (High Dynamic Range) metadata and conversion support.
//!
//! This module provides types and utilities for handling HDR video metadata,
//! including mastering display color volume (MDCV), content light level (CLL),
//! transfer characteristics, and color primaries.
//!
//! # Overview
//!
//! HDR video uses wider color gamuts and higher dynamic range than standard SDR
//! (Standard Dynamic Range) video. This module supports:
//!
//! - **Metadata Types:**
//!   - Mastering Display Color Volume (MDCV/SMPTE 2086)
//!   - Content Light Level (CLL) - `MaxCLL` and `MaxFALL`
//!   - HDR10+ dynamic metadata (structure only)
#![allow(clippy::match_same_arms)]
//!   - Dolby Vision metadata (read-only)
//!   - HLG (Hybrid Log-Gamma) parameters
//!
//! - **Transfer Functions:**
//!   - ST.2084 (PQ - Perceptual Quantizer) for HDR10
//!   - HLG (ARIB STD-B67) for broadcast HDR
//!   - BT.709 for SDR
//!   - BT.2020 (HDR container)
//!   - sRGB
//!
//! - **Color Primaries:**
//!   - BT.709 (Rec.709) - HD/SDR standard
//!   - BT.2020 (Rec.2020) - UHD/HDR standard
//!   - DCI-P3 - Digital cinema
//!   - Display P3 - Apple displays
//!   - Custom primaries
//!
//! # Example
//!
//! ```
//! use oximedia_core::hdr::{HdrMetadata, MasteringDisplayColorVolume, ContentLightLevel};
//! use oximedia_core::hdr::{TransferCharacteristic, ColorPrimaries};
//!
//! // Create HDR10 metadata
//! let mdcv = MasteringDisplayColorVolume {
//!     display_primaries: ColorPrimaries::BT2020.primaries(),
//!     white_point: ColorPrimaries::BT2020.white_point(),
//!     max_luminance: 1000.0,
//!     min_luminance: 0.005,
//! };
//!
//! let cll = ContentLightLevel {
//!     max_cll: 1000,
//!     max_fall: 400,
//! };
//!
//! let metadata = HdrMetadata {
//!     mdcv: Some(mdcv),
//!     cll: Some(cll),
//!     transfer: TransferCharacteristic::Pq,
//!     hdr10_plus: None,
//!     dolby_vision: None,
//!     hlg: None,
//!     primaries: ColorPrimaries::BT2020,
//! };
//!
//! // Check if content is HDR
//! assert!(metadata.is_hdr());
//! ```

#![forbid(unsafe_code)]

pub mod convert;
pub mod metadata;
pub mod parser;
pub mod primaries;
pub mod transfer;

pub use convert::{
    ColorGamutMapper, GamutMappingMode, HdrToSdrConverter, PqToHlgConverter, ToneMappingMode,
};
pub use metadata::{
    ContentLightLevel, DolbyVisionMetadata, Hdr10PlusMetadata, HlgParameters,
    MasteringDisplayColorVolume,
};
pub use parser::{Av1ColorConfig, HevcSeiParser, MatroskaColorElements, Vp9ColorConfig};
pub use primaries::{ColorPrimaries, Primaries, WhitePoint};
pub use transfer::TransferCharacteristic;

/// HDR metadata container.
///
/// This structure aggregates all HDR-related metadata for a video stream,
/// including mastering display information, content light levels, and
/// color space parameters.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::{HdrMetadata, TransferCharacteristic, ColorPrimaries};
///
/// let metadata = HdrMetadata::default();
/// assert!(!metadata.is_hdr());
///
/// let hdr10 = HdrMetadata::hdr10(1000.0, 0.005, 1000, 400);
/// assert!(hdr10.is_hdr());
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HdrMetadata {
    /// Mastering display color volume (SMPTE 2086).
    pub mdcv: Option<MasteringDisplayColorVolume>,
    /// Content light level information.
    pub cll: Option<ContentLightLevel>,
    /// HDR10+ dynamic metadata.
    pub hdr10_plus: Option<Hdr10PlusMetadata>,
    /// Dolby Vision metadata (read-only).
    pub dolby_vision: Option<DolbyVisionMetadata>,
    /// HLG parameters.
    pub hlg: Option<HlgParameters>,
    /// Transfer characteristic (EOTF).
    pub transfer: TransferCharacteristic,
    /// Color primaries.
    pub primaries: ColorPrimaries,
}

impl HdrMetadata {
    /// Creates a new HDR metadata container with default values.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::HdrMetadata;
    ///
    /// let metadata = HdrMetadata::new();
    /// assert!(!metadata.is_hdr());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates HDR10 metadata with mastering display and content light level.
    ///
    /// # Arguments
    ///
    /// * `max_luminance` - Maximum display luminance in nits (cd/m²)
    /// * `min_luminance` - Minimum display luminance in nits (cd/m²)
    /// * `max_cll` - Maximum content light level in nits
    /// * `max_fall` - Maximum frame-average light level in nits
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::HdrMetadata;
    ///
    /// // Typical HDR10 content mastered at 1000 nits
    /// let metadata = HdrMetadata::hdr10(1000.0, 0.005, 1000, 400);
    /// assert!(metadata.is_hdr());
    /// assert!(metadata.is_hdr10());
    /// ```
    #[must_use]
    pub fn hdr10(max_luminance: f64, min_luminance: f64, max_cll: u16, max_fall: u16) -> Self {
        Self {
            mdcv: Some(MasteringDisplayColorVolume {
                display_primaries: ColorPrimaries::BT2020.primaries(),
                white_point: ColorPrimaries::BT2020.white_point(),
                max_luminance,
                min_luminance,
            }),
            cll: Some(ContentLightLevel { max_cll, max_fall }),
            hdr10_plus: None,
            dolby_vision: None,
            hlg: None,
            transfer: TransferCharacteristic::Pq,
            primaries: ColorPrimaries::BT2020,
        }
    }

    /// Creates HLG metadata.
    ///
    /// # Arguments
    ///
    /// * `application_version` - HLG application version
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::HdrMetadata;
    ///
    /// let metadata = HdrMetadata::hlg(0);
    /// assert!(metadata.is_hdr());
    /// assert!(metadata.is_hlg());
    /// ```
    #[must_use]
    pub fn hlg(application_version: u8) -> Self {
        Self {
            mdcv: None,
            cll: None,
            hdr10_plus: None,
            dolby_vision: None,
            hlg: Some(HlgParameters {
                application_version,
            }),
            transfer: TransferCharacteristic::Hlg,
            primaries: ColorPrimaries::BT2020,
        }
    }

    /// Returns true if this metadata represents HDR content.
    ///
    /// Content is considered HDR if it uses PQ or HLG transfer characteristics,
    /// or if it has mastering display metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::{HdrMetadata, TransferCharacteristic};
    ///
    /// let mut metadata = HdrMetadata::new();
    /// assert!(!metadata.is_hdr());
    ///
    /// metadata.transfer = TransferCharacteristic::Pq;
    /// assert!(metadata.is_hdr());
    /// ```
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        matches!(
            self.transfer,
            TransferCharacteristic::Pq | TransferCharacteristic::Hlg
        ) || self.mdcv.is_some()
    }

    /// Returns true if this is HDR10 content.
    ///
    /// HDR10 uses PQ transfer function and BT.2020 primaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::HdrMetadata;
    ///
    /// let metadata = HdrMetadata::hdr10(1000.0, 0.005, 1000, 400);
    /// assert!(metadata.is_hdr10());
    /// ```
    #[must_use]
    pub fn is_hdr10(&self) -> bool {
        self.transfer == TransferCharacteristic::Pq
            && self.primaries == ColorPrimaries::BT2020
            && self.mdcv.is_some()
    }

    /// Returns true if this is HDR10+ content.
    ///
    /// HDR10+ extends HDR10 with dynamic metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::HdrMetadata;
    ///
    /// let mut metadata = HdrMetadata::hdr10(1000.0, 0.005, 1000, 400);
    /// assert!(!metadata.is_hdr10_plus());
    /// ```
    #[must_use]
    pub fn is_hdr10_plus(&self) -> bool {
        self.is_hdr10() && self.hdr10_plus.is_some()
    }

    /// Returns true if this is HLG content.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::HdrMetadata;
    ///
    /// let metadata = HdrMetadata::hlg(0);
    /// assert!(metadata.is_hlg());
    /// ```
    #[must_use]
    pub fn is_hlg(&self) -> bool {
        self.transfer == TransferCharacteristic::Hlg
    }

    /// Returns true if this is Dolby Vision content.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::HdrMetadata;
    ///
    /// let metadata = HdrMetadata::new();
    /// assert!(!metadata.is_dolby_vision());
    /// ```
    #[must_use]
    pub fn is_dolby_vision(&self) -> bool {
        self.dolby_vision.is_some()
    }

    /// Estimates the peak luminance of the content in nits.
    ///
    /// Returns the most appropriate luminance value from available metadata,
    /// falling back to standard defaults if no metadata is present.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::HdrMetadata;
    ///
    /// let metadata = HdrMetadata::hdr10(1000.0, 0.005, 1200, 400);
    /// assert!((metadata.estimate_peak_luminance() - 1200.0).abs() < 0.01);
    /// ```
    #[must_use]
    pub fn estimate_peak_luminance(&self) -> f64 {
        // Prefer CLL (actual content) over mastering display
        if let Some(cll) = &self.cll {
            f64::from(cll.max_cll)
        } else if let Some(mdcv) = &self.mdcv {
            mdcv.max_luminance
        } else {
            match self.transfer {
                TransferCharacteristic::Pq => 1000.0,  // HDR10 nominal
                TransferCharacteristic::Hlg => 1000.0, // HLG nominal
                _ => 100.0,                            // SDR nominal
            }
        }
    }

    /// Estimates the minimum luminance of the content in nits.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::HdrMetadata;
    ///
    /// let metadata = HdrMetadata::hdr10(1000.0, 0.005, 1000, 400);
    /// assert!((metadata.estimate_min_luminance() - 0.005).abs() < 0.0001);
    /// ```
    #[must_use]
    pub fn estimate_min_luminance(&self) -> f64 {
        if let Some(mdcv) = &self.mdcv {
            mdcv.min_luminance
        } else {
            match self.transfer {
                TransferCharacteristic::Pq | TransferCharacteristic::Hlg => 0.005,
                _ => 0.1,
            }
        }
    }

    /// Returns a human-readable description of the HDR format.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::HdrMetadata;
    ///
    /// let metadata = HdrMetadata::hdr10(1000.0, 0.005, 1000, 400);
    /// assert_eq!(metadata.format_name(), "HDR10");
    ///
    /// let hlg = HdrMetadata::hlg(0);
    /// assert_eq!(hlg.format_name(), "HLG");
    /// ```
    #[must_use]
    pub fn format_name(&self) -> &str {
        if self.is_dolby_vision() {
            "Dolby Vision"
        } else if self.is_hdr10_plus() {
            "HDR10+"
        } else if self.is_hdr10() {
            "HDR10"
        } else if self.is_hlg() {
            "HLG"
        } else if self.transfer == TransferCharacteristic::Pq {
            "PQ (ST.2084)"
        } else {
            "SDR"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_metadata() {
        let metadata = HdrMetadata::default();
        assert!(!metadata.is_hdr());
        assert!(!metadata.is_hdr10());
        assert!(!metadata.is_hlg());
    }

    #[test]
    fn test_hdr10_metadata() {
        let metadata = HdrMetadata::hdr10(1000.0, 0.005, 1000, 400);
        assert!(metadata.is_hdr());
        assert!(metadata.is_hdr10());
        assert!(!metadata.is_hdr10_plus());
        assert!(!metadata.is_hlg());
        assert_eq!(metadata.format_name(), "HDR10");
    }

    #[test]
    fn test_hlg_metadata() {
        let metadata = HdrMetadata::hlg(0);
        assert!(metadata.is_hdr());
        assert!(metadata.is_hlg());
        assert!(!metadata.is_hdr10());
        assert_eq!(metadata.format_name(), "HLG");
    }

    #[test]
    fn test_peak_luminance_estimation() {
        let metadata = HdrMetadata::hdr10(1000.0, 0.005, 1200, 400);
        assert!((metadata.estimate_peak_luminance() - 1200.0).abs() < 0.01);

        let metadata = HdrMetadata::new();
        assert!((metadata.estimate_peak_luminance() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_min_luminance_estimation() {
        let metadata = HdrMetadata::hdr10(1000.0, 0.005, 1000, 400);
        assert!((metadata.estimate_min_luminance() - 0.005).abs() < 0.0001);
    }
}
