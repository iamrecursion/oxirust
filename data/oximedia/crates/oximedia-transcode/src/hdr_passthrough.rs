//! HDR metadata passthrough and conversion for the transcode pipeline.
//!
//! Supports HDR10 (static metadata via SMPTE ST 2086 + CTA-861.3),
//! HLG (Hybrid Log-Gamma, ITU-R BT.2100), and a Dolby Vision profile
//! descriptor.  Metadata can be passed through unchanged, converted
//! between compatible transfer-function families, or stripped.
//!
//! # Transfer function compatibility
//!
//! ```text
//!   HDR10 (PQ/ST2084) <──> HLG (BT.2100)   ← approximate inverse-OOTF path
//!   HDR10 ──> SDR (BT.709)                  ← tone-map; PQ → BT.1886
//!   HLG   ──> SDR (BT.709)                  ← HLG OOTF collapse
//! ```
//!
//! Full pixel-level tone-mapping is provided by `oximedia-hdr`; this module
//! handles the **metadata** side: mastering-display descriptors, content-light
//! levels, and transfer-function flags embedded in the bitstream.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors that can occur during HDR metadata handling.
#[derive(Debug, Clone, Error)]
pub enum HdrError {
    /// The requested conversion between transfer functions is not supported.
    #[error("Unsupported HDR conversion: {from:?} → {to:?}")]
    UnsupportedConversion {
        /// Source transfer function.
        from: TransferFunction,
        /// Target transfer function.
        to: TransferFunction,
    },

    /// A required metadata field is absent.
    #[error("Missing HDR metadata field: {0}")]
    MissingField(String),

    /// A numeric value is outside the valid range for its field.
    #[error("HDR field '{field}' value {value} is out of range [{min}, {max}]")]
    OutOfRange {
        /// Field name.
        field: String,
        /// Supplied value.
        value: f64,
        /// Minimum allowed.
        min: f64,
        /// Maximum allowed.
        max: f64,
    },
}

// ─── Transfer function ────────────────────────────────────────────────────────

/// Video transfer function (EOTF / OETF).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransferFunction {
    /// ITU-R BT.709 / BT.1886 (standard SDR).
    Bt709,
    /// SMPTE ST 2084 (Perceptual Quantizer, used by HDR10 and Dolby Vision).
    Pq,
    /// Hybrid Log-Gamma (ITU-R BT.2100).
    Hlg,
    /// Gamma 2.2 (legacy / JPEG / sRGB approximation).
    Gamma22,
    /// Linear light (no gamma).
    Linear,
    /// Unspecified / unknown.
    Unspecified,
}

impl TransferFunction {
    /// Returns the ITU-T H.273 / ISO/IEC 23091-2 transfer characteristics code.
    #[must_use]
    pub fn h273_code(self) -> u8 {
        match self {
            Self::Bt709 => 1,
            Self::Gamma22 => 4,
            Self::Linear => 8,
            Self::Pq => 16,
            Self::Hlg => 18,
            Self::Unspecified => 2,
        }
    }

    /// Returns `true` if this is an HDR transfer function (PQ or HLG).
    #[must_use]
    pub fn is_hdr(self) -> bool {
        matches!(self, Self::Pq | Self::Hlg)
    }

    /// Returns `true` if this transfer function uses a wide colour gamut (BT.2020).
    #[must_use]
    pub fn is_wide_gamut(self) -> bool {
        matches!(self, Self::Pq | Self::Hlg)
    }
}

// ─── Colour primaries ─────────────────────────────────────────────────────────

/// Video colour primaries (H.273 table 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColourPrimaries {
    /// BT.709 / sRGB.
    Bt709,
    /// BT.2020 (HDR10, HLG).
    Bt2020,
    /// Display P3 (DCI-P3 with D65 white point).
    DisplayP3,
    /// Unspecified.
    Unspecified,
}

impl ColourPrimaries {
    /// Returns the H.273 colour primaries code.
    #[must_use]
    pub fn h273_code(self) -> u8 {
        match self {
            Self::Bt709 => 1,
            Self::Unspecified => 2,
            Self::DisplayP3 => 12,
            Self::Bt2020 => 9,
        }
    }
}

// ─── SMPTE ST 2086 mastering display ─────────────────────────────────────────

/// Mastering display colour volume (SMPTE ST 2086).
///
/// All chromaticity coordinates are in the range [0, 1]; luminance values
/// are in candelas per square metre (cd/m²).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MasteringDisplay {
    /// Red chromaticity x coordinate.
    pub red_x: f64,
    /// Red chromaticity y coordinate.
    pub red_y: f64,
    /// Green chromaticity x coordinate.
    pub green_x: f64,
    /// Green chromaticity y coordinate.
    pub green_y: f64,
    /// Blue chromaticity x coordinate.
    pub blue_x: f64,
    /// Blue chromaticity y coordinate.
    pub blue_y: f64,
    /// White point x coordinate.
    pub white_x: f64,
    /// White point y coordinate.
    pub white_y: f64,
    /// Maximum display mastering luminance (cd/m²).
    pub max_luminance: f64,
    /// Minimum display mastering luminance (cd/m²).
    pub min_luminance: f64,
}

impl MasteringDisplay {
    /// Creates a mastering display descriptor for a standard P3 D65 reference monitor
    /// (typical HDR10 grade suite, 1000 nit peak).
    #[must_use]
    pub fn p3_d65_1000nit() -> Self {
        Self {
            red_x: 0.680,
            red_y: 0.320,
            green_x: 0.265,
            green_y: 0.690,
            blue_x: 0.150,
            blue_y: 0.060,
            white_x: 0.3127,
            white_y: 0.3290,
            max_luminance: 1000.0,
            min_luminance: 0.0050,
        }
    }

    /// Creates a mastering display descriptor for BT.2020 at 4000 nit.
    #[must_use]
    pub fn bt2020_4000nit() -> Self {
        Self {
            red_x: 0.708,
            red_y: 0.292,
            green_x: 0.170,
            green_y: 0.797,
            blue_x: 0.131,
            blue_y: 0.046,
            white_x: 0.3127,
            white_y: 0.3290,
            max_luminance: 4000.0,
            min_luminance: 0.005,
        }
    }

    /// Validates all fields are within legal ranges.
    ///
    /// # Errors
    ///
    /// Returns `HdrError::OutOfRange` if any chromaticity or luminance value is out of range.
    pub fn validate(&self) -> Result<(), HdrError> {
        let check_chroma = |name: &str, v: f64| {
            if v < 0.0 || v > 1.0 {
                Err(HdrError::OutOfRange {
                    field: name.to_string(),
                    value: v,
                    min: 0.0,
                    max: 1.0,
                })
            } else {
                Ok(())
            }
        };
        check_chroma("red_x", self.red_x)?;
        check_chroma("red_y", self.red_y)?;
        check_chroma("green_x", self.green_x)?;
        check_chroma("green_y", self.green_y)?;
        check_chroma("blue_x", self.blue_x)?;
        check_chroma("blue_y", self.blue_y)?;
        check_chroma("white_x", self.white_x)?;
        check_chroma("white_y", self.white_y)?;

        if self.max_luminance <= 0.0 {
            return Err(HdrError::OutOfRange {
                field: "max_luminance".to_string(),
                value: self.max_luminance,
                min: 0.001,
                max: f64::MAX,
            });
        }
        if self.min_luminance < 0.0 || self.min_luminance >= self.max_luminance {
            return Err(HdrError::OutOfRange {
                field: "min_luminance".to_string(),
                value: self.min_luminance,
                min: 0.0,
                max: self.max_luminance,
            });
        }
        Ok(())
    }
}

// ─── Content light level (CTA-861.3) ─────────────────────────────────────────

/// Content light level metadata (CTA-861.3 MaxCLL / MaxFALL).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContentLightLevel {
    /// Maximum content light level (MaxCLL) in cd/m².
    pub max_cll: u16,
    /// Maximum frame-average light level (MaxFALL) in cd/m².
    pub max_fall: u16,
}

impl ContentLightLevel {
    /// Creates a new content light level descriptor.
    #[must_use]
    pub fn new(max_cll: u16, max_fall: u16) -> Self {
        Self { max_cll, max_fall }
    }

    /// A conservative default for HDR10 content (1000 MaxCLL / 400 MaxFALL).
    #[must_use]
    pub fn hdr10_default() -> Self {
        Self {
            max_cll: 1000,
            max_fall: 400,
        }
    }

    /// Validates that MaxFALL ≤ MaxCLL (per spec).
    ///
    /// # Errors
    ///
    /// Returns `HdrError::OutOfRange` if MaxFALL > MaxCLL.
    pub fn validate(&self) -> Result<(), HdrError> {
        if self.max_fall > self.max_cll {
            return Err(HdrError::OutOfRange {
                field: "max_fall".to_string(),
                value: f64::from(self.max_fall),
                min: 0.0,
                max: f64::from(self.max_cll),
            });
        }
        Ok(())
    }
}

// ─── Dolby Vision profile descriptor ─────────────────────────────────────────

/// Dolby Vision profile and level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DolbyVisionProfile {
    /// Profile 4: HEVC with BL+EL+RPU (backward-compatible to HDR10).
    Profile4,
    /// Profile 5: HEVC single-layer with RPU (MEL / FEL).
    Profile5,
    /// Profile 7: HEVC dual-layer with RPU.
    Profile7,
    /// Profile 8: AV1 / HEVC single layer BL+RPU (most common OTT).
    Profile8,
    /// Profile 9: AV1 single-layer (next-gen streaming).
    Profile9,
}

impl DolbyVisionProfile {
    /// Returns the numeric DVHE/DVAV profile number.
    #[must_use]
    pub fn profile_number(self) -> u8 {
        match self {
            Self::Profile4 => 4,
            Self::Profile5 => 5,
            Self::Profile7 => 7,
            Self::Profile8 => 8,
            Self::Profile9 => 9,
        }
    }

    /// Returns `true` if this profile supports backward-compatible SDR/HDR10 base layer.
    #[must_use]
    pub fn is_backward_compatible(self) -> bool {
        matches!(self, Self::Profile4 | Self::Profile7 | Self::Profile8)
    }
}

/// Dolby Vision metadata payload attached to a stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DolbyVisionMeta {
    /// Dolby Vision profile.
    pub profile: DolbyVisionProfile,
    /// Level (1–13; maps to resolution × frame-rate bands).
    pub level: u8,
    /// Whether an RPU (Reference Processing Unit) NAL/OBU is present.
    pub has_rpu: bool,
    /// Whether an Enhancement Layer (EL) track exists.
    pub has_el: bool,
}

impl DolbyVisionMeta {
    /// Creates a Dolby Vision metadata descriptor.
    #[must_use]
    pub fn new(profile: DolbyVisionProfile, level: u8) -> Self {
        Self {
            profile,
            level,
            has_rpu: true,
            has_el: false,
        }
    }

    /// Validates the level is within [1, 13].
    ///
    /// # Errors
    ///
    /// Returns `HdrError::OutOfRange` if the level is outside [1, 13].
    pub fn validate(&self) -> Result<(), HdrError> {
        if self.level < 1 || self.level > 13 {
            return Err(HdrError::OutOfRange {
                field: "dv_level".to_string(),
                value: f64::from(self.level),
                min: 1.0,
                max: 13.0,
            });
        }
        Ok(())
    }
}

// ─── Unified HDR metadata bundle ─────────────────────────────────────────────

/// Unified HDR metadata attached to a video stream.
///
/// Carry exactly the fields present in the source; absent fields are `None`.
/// Use [`HdrMetadata::validate`] before attaching to a mux.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HdrMetadata {
    /// Transfer function (EOTF/OETF).
    pub transfer_function: Option<TransferFunction>,
    /// Colour primaries.
    pub colour_primaries: Option<ColourPrimaries>,
    /// SMPTE ST 2086 mastering display colour volume.
    pub mastering_display: Option<MasteringDisplay>,
    /// CTA-861.3 content light level (MaxCLL / MaxFALL).
    pub content_light_level: Option<ContentLightLevel>,
    /// Dolby Vision metadata (if present).
    pub dolby_vision: Option<DolbyVisionMeta>,
}

impl HdrMetadata {
    /// Creates a minimal HDR10 metadata bundle with mastering display and CLL.
    #[must_use]
    pub fn hdr10(mastering: MasteringDisplay, cll: ContentLightLevel) -> Self {
        Self {
            transfer_function: Some(TransferFunction::Pq),
            colour_primaries: Some(ColourPrimaries::Bt2020),
            mastering_display: Some(mastering),
            content_light_level: Some(cll),
            dolby_vision: None,
        }
    }

    /// Creates a minimal HLG metadata bundle.
    #[must_use]
    pub fn hlg() -> Self {
        Self {
            transfer_function: Some(TransferFunction::Hlg),
            colour_primaries: Some(ColourPrimaries::Bt2020),
            mastering_display: None,
            content_light_level: None,
            dolby_vision: None,
        }
    }

    /// Returns `true` if this bundle carries any HDR signal.
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        self.transfer_function
            .map(TransferFunction::is_hdr)
            .unwrap_or(false)
            || self.dolby_vision.is_some()
    }

    /// Validates all present sub-descriptors.
    ///
    /// # Errors
    ///
    /// Returns an error if any sub-descriptor fails validation.
    pub fn validate(&self) -> Result<(), HdrError> {
        if let Some(md) = &self.mastering_display {
            md.validate()?;
        }
        if let Some(cll) = &self.content_light_level {
            cll.validate()?;
        }
        if let Some(dv) = &self.dolby_vision {
            dv.validate()?;
        }
        Ok(())
    }
}

// ─── HDR passthrough mode ─────────────────────────────────────────────────────

/// How HDR metadata should be handled when transcoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HdrPassthroughMode {
    /// Copy metadata unchanged from source to output.
    Passthrough,
    /// Strip all HDR metadata; treat output as SDR.
    Strip,
    /// Convert from the source HDR flavour to a different one.
    ///
    /// Pixel-level tone-mapping is performed by the frame pipeline;
    /// this mode also updates the stream-level metadata flags.
    Convert {
        /// Target transfer function.
        target_tf: TransferFunction,
        /// Target colour primaries.
        target_primaries: ColourPrimaries,
    },
    /// Inject caller-supplied metadata (overwrite any existing).
    Inject(HdrMetadata),
}

impl Default for HdrPassthroughMode {
    fn default() -> Self {
        Self::Passthrough
    }
}

// ─── HdrProcessor ────────────────────────────────────────────────────────────

/// Applies an [`HdrPassthroughMode`] to a source [`HdrMetadata`] bundle,
/// producing the metadata that should be written to the output stream.
#[derive(Debug, Clone, Default)]
pub struct HdrProcessor {
    mode: HdrPassthroughMode,
}

impl HdrProcessor {
    /// Creates a new processor with the given mode.
    #[must_use]
    pub fn new(mode: HdrPassthroughMode) -> Self {
        Self { mode }
    }

    /// Processes the source metadata according to the configured mode and
    /// returns the resulting metadata for the output stream.
    ///
    /// # Errors
    ///
    /// Returns `HdrError::UnsupportedConversion` when converting between
    /// incompatible transfer functions (e.g., SDR → PQ without tone-mapping
    /// parameters).
    pub fn process(&self, source: Option<&HdrMetadata>) -> Result<Option<HdrMetadata>, HdrError> {
        match &self.mode {
            HdrPassthroughMode::Passthrough => Ok(source.cloned()),

            HdrPassthroughMode::Strip => Ok(None),

            HdrPassthroughMode::Inject(meta) => {
                meta.validate()?;
                Ok(Some(meta.clone()))
            }

            HdrPassthroughMode::Convert {
                target_tf,
                target_primaries,
            } => {
                let src_tf = source
                    .and_then(|m| m.transfer_function)
                    .unwrap_or(TransferFunction::Unspecified);

                // Validate that the conversion path is supported.
                Self::check_conversion(src_tf, *target_tf)?;

                let mut out = source.cloned().unwrap_or_default();
                out.transfer_function = Some(*target_tf);
                out.colour_primaries = Some(*target_primaries);

                // SDR output: drop static HDR metadata that doesn't apply.
                if !target_tf.is_hdr() {
                    out.mastering_display = None;
                    out.content_light_level = None;
                    out.dolby_vision = None;
                }

                // HLG output: drop PQ-specific fields that don't apply.
                if *target_tf == TransferFunction::Hlg {
                    out.mastering_display = None;
                    out.content_light_level = None;
                    out.dolby_vision = None;
                }

                Ok(Some(out))
            }
        }
    }

    /// Checks whether a transfer-function conversion path is supported.
    fn check_conversion(from: TransferFunction, to: TransferFunction) -> Result<(), HdrError> {
        use TransferFunction::{Bt709, Hlg, Pq, Unspecified};
        let ok = matches!(
            (from, to),
            // Identity
            (Pq, Pq) | (Hlg, Hlg) | (Bt709, Bt709) |
            // HDR → SDR (tone-map)
            (Pq, Bt709) | (Hlg, Bt709) |
            // HDR cross-conversion (approximate)
            (Pq, Hlg) | (Hlg, Pq) |
            // Unspecified source → anything
            (Unspecified, _)
        );
        if ok {
            Ok(())
        } else {
            Err(HdrError::UnsupportedConversion { from, to })
        }
    }

    /// Returns the configured passthrough mode.
    #[must_use]
    pub fn mode(&self) -> &HdrPassthroughMode {
        &self.mode
    }
}

// ─── Bitstream-level helpers ──────────────────────────────────────────────────

/// Serialises a `MasteringDisplay` into the 24-byte SMPTE ST 2086 SEI payload
/// format used by HEVC and (via the same layout) AV1 metadata OBUs.
///
/// The layout is:
/// ```text
///   2 bytes × 3 primaries (x,y) + 2 bytes × white point (x,y) = 10 × u16
///   4 bytes max_luminance  (u32, units: 0.0001 cd/m²)
///   4 bytes min_luminance  (u32, units: 0.0001 cd/m²)
/// ```
#[must_use]
pub fn encode_mastering_display_sei(md: &MasteringDisplay) -> [u8; 24] {
    let to_u16 = |v: f64| -> u16 { (v * 50_000.0).round() as u16 };
    let to_u32 = |v: f64| -> u32 { (v * 10_000.0).round() as u32 };

    let mut buf = [0u8; 24];
    let pairs: [(f64, f64); 4] = [
        (md.green_x, md.green_y),
        (md.blue_x, md.blue_y),
        (md.red_x, md.red_y),
        (md.white_x, md.white_y),
    ];
    for (i, (x, y)) in pairs.iter().enumerate() {
        let xv = to_u16(*x);
        let yv = to_u16(*y);
        buf[i * 4] = (xv >> 8) as u8;
        buf[i * 4 + 1] = (xv & 0xFF) as u8;
        buf[i * 4 + 2] = (yv >> 8) as u8;
        buf[i * 4 + 3] = (yv & 0xFF) as u8;
    }
    let max_u32 = to_u32(md.max_luminance);
    let min_u32 = to_u32(md.min_luminance);
    buf[16] = (max_u32 >> 24) as u8;
    buf[17] = (max_u32 >> 16) as u8;
    buf[18] = (max_u32 >> 8) as u8;
    buf[19] = (max_u32 & 0xFF) as u8;
    buf[20] = (min_u32 >> 24) as u8;
    buf[21] = (min_u32 >> 16) as u8;
    buf[22] = (min_u32 >> 8) as u8;
    buf[23] = (min_u32 & 0xFF) as u8;
    buf
}

/// Decodes a 24-byte SMPTE ST 2086 SEI payload back into a `MasteringDisplay`.
///
/// # Errors
///
/// Returns `HdrError::MissingField` if the buffer is shorter than 24 bytes,
/// or `HdrError::OutOfRange` if the decoded luminance values are invalid.
pub fn decode_mastering_display_sei(buf: &[u8]) -> Result<MasteringDisplay, HdrError> {
    if buf.len() < 24 {
        return Err(HdrError::MissingField(
            "SEI payload too short (need 24 bytes)".to_string(),
        ));
    }
    let read_u16 = |i: usize| -> f64 {
        let v = (u16::from(buf[i]) << 8) | u16::from(buf[i + 1]);
        f64::from(v) / 50_000.0
    };
    let read_u32 = |i: usize| -> f64 {
        let v = (u32::from(buf[i]) << 24)
            | (u32::from(buf[i + 1]) << 16)
            | (u32::from(buf[i + 2]) << 8)
            | u32::from(buf[i + 3]);
        f64::from(v) / 10_000.0
    };

    let md = MasteringDisplay {
        green_x: read_u16(0),
        green_y: read_u16(2),
        blue_x: read_u16(4),
        blue_y: read_u16(6),
        red_x: read_u16(8),
        red_y: read_u16(10),
        white_x: read_u16(12),
        white_y: read_u16(14),
        max_luminance: read_u32(16),
        min_luminance: read_u32(20),
    };
    md.validate()?;
    Ok(md)
}

/// Serialises `ContentLightLevel` into the 4-byte CTA-861.3 payload
/// (MaxCLL u16 BE, MaxFALL u16 BE).
#[must_use]
pub fn encode_cll_sei(cll: &ContentLightLevel) -> [u8; 4] {
    [
        (cll.max_cll >> 8) as u8,
        (cll.max_cll & 0xFF) as u8,
        (cll.max_fall >> 8) as u8,
        (cll.max_fall & 0xFF) as u8,
    ]
}

/// Decodes a 4-byte CTA-861.3 payload back into `ContentLightLevel`.
///
/// # Errors
///
/// Returns `HdrError::MissingField` if `buf` is shorter than 4 bytes.
pub fn decode_cll_sei(buf: &[u8]) -> Result<ContentLightLevel, HdrError> {
    if buf.len() < 4 {
        return Err(HdrError::MissingField(
            "CLL SEI payload too short (need 4 bytes)".to_string(),
        ));
    }
    let max_cll = (u16::from(buf[0]) << 8) | u16::from(buf[1]);
    let max_fall = (u16::from(buf[2]) << 8) | u16::from(buf[3]);
    let cll = ContentLightLevel { max_cll, max_fall };
    cll.validate()?;
    Ok(cll)
}

// ─── HDR10+ dynamic metadata ──────────────────────────────────────────────────

/// HDR10+ dynamic metadata for a single scene or frame.
///
/// HDR10+ (SMPTE ST 2094-40) carries per-scene tone-mapping information
/// as SEI messages in HEVC or metadata OBUs in AV1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hdr10PlusDynamicMeta {
    /// Application version (currently 0 or 1).
    pub application_version: u8,
    /// Targeted system display maximum luminance (in cd/m²).
    pub targeted_system_display_max_luminance: u16,
    /// Average maxRGB of the scene (0.0–1.0 normalised to peak).
    pub average_maxrgb: f64,
    /// Distribution maxRGB percentile values (up to 9).
    pub maxrgb_percentiles: Vec<(u8, f64)>,
    /// Fraction of selected area pixels.
    pub fraction_bright_pixels: f64,
    /// Knee point (x, y) for the tone-mapping curve.
    pub knee_point: (f64, f64),
    /// Bezier curve anchors for tone-mapping (0–9 points).
    pub bezier_curve_anchors: Vec<f64>,
}

impl Hdr10PlusDynamicMeta {
    /// Creates a new HDR10+ dynamic metadata descriptor with sensible defaults.
    #[must_use]
    pub fn new(targeted_max_lum: u16) -> Self {
        Self {
            application_version: 1,
            targeted_system_display_max_luminance: targeted_max_lum,
            average_maxrgb: 0.5,
            maxrgb_percentiles: vec![(1, 0.01), (50, 0.5), (99, 0.95)],
            fraction_bright_pixels: 0.01,
            knee_point: (0.5, 0.5),
            bezier_curve_anchors: vec![0.25, 0.5, 0.75],
        }
    }

    /// Validates the HDR10+ metadata fields.
    ///
    /// # Errors
    ///
    /// Returns `HdrError::OutOfRange` if any field is outside legal range.
    pub fn validate(&self) -> Result<(), HdrError> {
        if self.application_version > 1 {
            return Err(HdrError::OutOfRange {
                field: "hdr10plus_application_version".to_string(),
                value: f64::from(self.application_version),
                min: 0.0,
                max: 1.0,
            });
        }
        if self.average_maxrgb < 0.0 || self.average_maxrgb > 1.0 {
            return Err(HdrError::OutOfRange {
                field: "average_maxrgb".to_string(),
                value: self.average_maxrgb,
                min: 0.0,
                max: 1.0,
            });
        }
        if self.fraction_bright_pixels < 0.0 || self.fraction_bright_pixels > 1.0 {
            return Err(HdrError::OutOfRange {
                field: "fraction_bright_pixels".to_string(),
                value: self.fraction_bright_pixels,
                min: 0.0,
                max: 1.0,
            });
        }
        let (kx, ky) = self.knee_point;
        if kx < 0.0 || kx > 1.0 {
            return Err(HdrError::OutOfRange {
                field: "knee_point_x".to_string(),
                value: kx,
                min: 0.0,
                max: 1.0,
            });
        }
        if ky < 0.0 || ky > 1.0 {
            return Err(HdrError::OutOfRange {
                field: "knee_point_y".to_string(),
                value: ky,
                min: 0.0,
                max: 1.0,
            });
        }
        if self.bezier_curve_anchors.len() > 9 {
            return Err(HdrError::OutOfRange {
                field: "bezier_curve_anchors_count".to_string(),
                value: self.bezier_curve_anchors.len() as f64,
                min: 0.0,
                max: 9.0,
            });
        }
        for (i, &a) in self.bezier_curve_anchors.iter().enumerate() {
            if a < 0.0 || a > 1.0 {
                return Err(HdrError::OutOfRange {
                    field: format!("bezier_anchor_{i}"),
                    value: a,
                    min: 0.0,
                    max: 1.0,
                });
            }
        }
        Ok(())
    }

    /// Serialises HDR10+ dynamic metadata into a simplified binary payload.
    ///
    /// Layout (variable length):
    /// - 1 byte: application_version
    /// - 2 bytes: targeted_system_display_max_luminance (u16 BE)
    /// - 2 bytes: average_maxrgb (u16 BE, value * 10000)
    /// - 2 bytes: fraction_bright_pixels (u16 BE, value * 10000)
    /// - 2 bytes: knee_point_x (u16 BE, value * 10000)
    /// - 2 bytes: knee_point_y (u16 BE, value * 10000)
    /// - 1 byte: number of bezier anchors
    /// - N * 2 bytes: anchor values (u16 BE, value * 10000)
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let to_u16 = |v: f64| -> u16 { (v * 10_000.0).round() as u16 };
        let mut buf = Vec::with_capacity(16 + self.bezier_curve_anchors.len() * 2);
        buf.push(self.application_version);
        buf.extend_from_slice(&self.targeted_system_display_max_luminance.to_be_bytes());
        buf.extend_from_slice(&to_u16(self.average_maxrgb).to_be_bytes());
        buf.extend_from_slice(&to_u16(self.fraction_bright_pixels).to_be_bytes());
        buf.extend_from_slice(&to_u16(self.knee_point.0).to_be_bytes());
        buf.extend_from_slice(&to_u16(self.knee_point.1).to_be_bytes());
        let anchor_count = self.bezier_curve_anchors.len().min(9) as u8;
        buf.push(anchor_count);
        for &a in self.bezier_curve_anchors.iter().take(9) {
            buf.extend_from_slice(&to_u16(a).to_be_bytes());
        }
        buf
    }

    /// Decodes HDR10+ dynamic metadata from a binary payload.
    ///
    /// # Errors
    ///
    /// Returns `HdrError::MissingField` if the buffer is too short.
    pub fn decode(buf: &[u8]) -> Result<Self, HdrError> {
        if buf.len() < 12 {
            return Err(HdrError::MissingField(
                "HDR10+ payload too short (need at least 12 bytes)".to_string(),
            ));
        }
        let from_u16 = |i: usize| -> f64 {
            let v = (u16::from(buf[i]) << 8) | u16::from(buf[i + 1]);
            f64::from(v) / 10_000.0
        };
        let application_version = buf[0];
        let targeted_max = (u16::from(buf[1]) << 8) | u16::from(buf[2]);
        let average_maxrgb = from_u16(3);
        let fraction_bright = from_u16(5);
        let knee_x = from_u16(7);
        let knee_y = from_u16(9);
        let anchor_count = buf[11] as usize;
        let needed = 12 + anchor_count * 2;
        if buf.len() < needed {
            return Err(HdrError::MissingField(format!(
                "HDR10+ payload too short for {anchor_count} anchors (need {needed} bytes)"
            )));
        }
        let mut anchors = Vec::with_capacity(anchor_count);
        for i in 0..anchor_count {
            let offset = 12 + i * 2;
            let v = (u16::from(buf[offset]) << 8) | u16::from(buf[offset + 1]);
            anchors.push(f64::from(v) / 10_000.0);
        }
        let meta = Self {
            application_version,
            targeted_system_display_max_luminance: targeted_max,
            average_maxrgb,
            maxrgb_percentiles: Vec::new(),
            fraction_bright_pixels: fraction_bright,
            knee_point: (knee_x, knee_y),
            bezier_curve_anchors: anchors,
        };
        meta.validate()?;
        Ok(meta)
    }
}

// ─── Dolby Vision RPU passthrough ─────────────────────────────────────────────

/// Dolby Vision RPU (Reference Processing Unit) passthrough handler.
///
/// Manages extraction and insertion of RPU NAL units from/to HEVC or AV1
/// bitstreams during transcoding without re-interpreting the mapping curves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DolbyVisionRpu {
    /// Raw RPU payload bytes (NAL unit body, excluding start code).
    pub payload: Vec<u8>,
    /// Profile from the RPU header (0–9).
    pub rpu_profile: u8,
    /// Whether the RPU was validated successfully.
    pub validated: bool,
    /// Frame index this RPU belongs to.
    pub frame_index: u64,
}

impl DolbyVisionRpu {
    /// Creates a new RPU descriptor from raw payload bytes.
    #[must_use]
    pub fn new(payload: Vec<u8>, frame_index: u64) -> Self {
        Self {
            rpu_profile: Self::extract_profile(&payload),
            payload,
            validated: false,
            frame_index,
        }
    }

    /// Extracts the Dolby Vision profile from the RPU header.
    ///
    /// The profile is encoded in the first few bits of the RPU payload.
    /// Returns 0 if the payload is too short to determine the profile.
    fn extract_profile(payload: &[u8]) -> u8 {
        // DV RPU starts with rpu_type (6 bits), then rpu_format (11 bits).
        // The profile is typically signalled at a higher level (configuration record),
        // but we can infer from rpu_type: 2 => profile 7/8, etc.
        if payload.len() < 2 {
            return 0;
        }
        let rpu_type = payload[0] >> 2;
        match rpu_type {
            2 => 8, // Single-layer with RPU (most common OTT)
            0 => 5, // MEL/FEL
            1 => 7, // Dual-layer
            _ => 0,
        }
    }

    /// Validates the RPU payload structure.
    ///
    /// # Errors
    ///
    /// Returns `HdrError::MissingField` if the payload is empty or malformed.
    pub fn validate(&mut self) -> Result<(), HdrError> {
        if self.payload.is_empty() {
            return Err(HdrError::MissingField(
                "DV RPU payload is empty".to_string(),
            ));
        }
        // Minimal structural check: RPU should be at least 25 bytes
        if self.payload.len() < 25 {
            return Err(HdrError::MissingField(
                "DV RPU payload too short (minimum 25 bytes)".to_string(),
            ));
        }
        self.validated = true;
        Ok(())
    }

    /// Returns the size of the RPU payload in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.payload.len()
    }
}

/// Manages a stream of Dolby Vision RPUs for passthrough mode.
#[derive(Debug, Clone, Default)]
pub struct DvRpuPassthrough {
    /// Collected RPU payloads indexed by frame number.
    rpus: Vec<DolbyVisionRpu>,
    /// Number of RPUs that passed validation.
    valid_count: usize,
    /// Number of RPUs that failed validation.
    invalid_count: usize,
}

impl DvRpuPassthrough {
    /// Creates a new RPU passthrough handler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingests a raw RPU payload for the given frame.
    pub fn ingest(&mut self, payload: Vec<u8>, frame_index: u64) {
        let mut rpu = DolbyVisionRpu::new(payload, frame_index);
        if rpu.validate().is_ok() {
            self.valid_count += 1;
        } else {
            self.invalid_count += 1;
        }
        self.rpus.push(rpu);
    }

    /// Returns the RPU for the given frame index, if available.
    #[must_use]
    pub fn get_rpu(&self, frame_index: u64) -> Option<&DolbyVisionRpu> {
        self.rpus.iter().find(|r| r.frame_index == frame_index)
    }

    /// Returns the total number of ingested RPUs.
    #[must_use]
    pub fn count(&self) -> usize {
        self.rpus.len()
    }

    /// Returns the number of valid RPUs.
    #[must_use]
    pub fn valid_count(&self) -> usize {
        self.valid_count
    }

    /// Returns the number of invalid RPUs.
    #[must_use]
    pub fn invalid_count(&self) -> usize {
        self.invalid_count
    }

    /// Drains all RPUs into a vector for writing to the output bitstream.
    pub fn drain_all(&mut self) -> Vec<DolbyVisionRpu> {
        self.valid_count = 0;
        self.invalid_count = 0;
        std::mem::take(&mut self.rpus)
    }
}

// ─── Tone-mapping configuration ───────────────────────────────────────────────

/// Tonemapping curve type for HDR↔SDR conversion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TonemapCurve {
    /// Reinhard global operator: L_out = L / (1 + L).
    Reinhard,
    /// Hable (Uncharted 2) filmic curve.
    Hable,
    /// ACES filmic (Academy Color Encoding System).
    Aces,
    /// BT.2390 EETF (reference PQ tone-mapping).
    Bt2390,
    /// Simple clip: values above `peak_luminance` are clamped.
    Clip,
    /// Mobius (smooth roll-off near peak).
    Mobius,
}

/// Configuration for HDR→SDR tonemapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HdrToSdrConfig {
    /// Tonemapping curve.
    pub curve: TonemapCurve,
    /// Source peak luminance (cd/m²).
    pub source_peak_nits: f64,
    /// Target peak luminance (cd/m²) — typically 100 for SDR.
    pub target_peak_nits: f64,
    /// Desaturation strength (0.0 = none, 1.0 = full grey at peak).
    pub desat_strength: f64,
}

impl HdrToSdrConfig {
    /// Creates a default HDR→SDR configuration for 1000 nit source.
    #[must_use]
    pub fn default_1000nit() -> Self {
        Self {
            curve: TonemapCurve::Bt2390,
            source_peak_nits: 1000.0,
            target_peak_nits: 100.0,
            desat_strength: 0.5,
        }
    }

    /// Validates the tonemapping configuration.
    ///
    /// # Errors
    ///
    /// Returns `HdrError::OutOfRange` if luminance or desaturation values are invalid.
    pub fn validate(&self) -> Result<(), HdrError> {
        if self.source_peak_nits <= 0.0 {
            return Err(HdrError::OutOfRange {
                field: "source_peak_nits".to_string(),
                value: self.source_peak_nits,
                min: 0.001,
                max: f64::MAX,
            });
        }
        if self.target_peak_nits <= 0.0 {
            return Err(HdrError::OutOfRange {
                field: "target_peak_nits".to_string(),
                value: self.target_peak_nits,
                min: 0.001,
                max: f64::MAX,
            });
        }
        if self.desat_strength < 0.0 || self.desat_strength > 1.0 {
            return Err(HdrError::OutOfRange {
                field: "desat_strength".to_string(),
                value: self.desat_strength,
                min: 0.0,
                max: 1.0,
            });
        }
        Ok(())
    }

    /// Applies the Reinhard tonemapping operator to a linear-light value.
    #[must_use]
    pub fn tonemap_reinhard(&self, l: f64) -> f64 {
        let normalised = l * self.source_peak_nits / self.target_peak_nits;
        let mapped = normalised / (1.0 + normalised);
        mapped * self.target_peak_nits
    }

    /// Applies the Hable (Uncharted 2) filmic curve.
    #[must_use]
    pub fn tonemap_hable(&self, l: f64) -> f64 {
        let hable = |x: f64| -> f64 {
            let a = 0.15;
            let b = 0.50;
            let c = 0.10;
            let d = 0.20;
            let e = 0.02;
            let f = 0.30;
            ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f
        };
        let normalised = l * self.source_peak_nits / self.target_peak_nits;
        let white = self.source_peak_nits / self.target_peak_nits;
        (hable(normalised) / hable(white)) * self.target_peak_nits
    }

    /// Applies the ACES filmic curve approximation.
    #[must_use]
    pub fn tonemap_aces(&self, l: f64) -> f64 {
        let normalised = l * self.source_peak_nits / self.target_peak_nits;
        let a = 2.51;
        let b = 0.03;
        let c = 2.43;
        let d = 0.59;
        let e = 0.14;
        let mapped = (normalised * (a * normalised + b)) / (normalised * (c * normalised + d) + e);
        mapped.clamp(0.0, 1.0) * self.target_peak_nits
    }

    /// Applies the configured tonemapping curve to a linear-light value.
    #[must_use]
    pub fn apply(&self, l: f64) -> f64 {
        match self.curve {
            TonemapCurve::Reinhard => self.tonemap_reinhard(l),
            TonemapCurve::Hable => self.tonemap_hable(l),
            TonemapCurve::Aces => self.tonemap_aces(l),
            TonemapCurve::Bt2390 | TonemapCurve::Mobius => {
                // BT.2390 / Mobius: use Reinhard as fallback approximation
                self.tonemap_reinhard(l)
            }
            TonemapCurve::Clip => (l * self.source_peak_nits).min(self.target_peak_nits),
        }
    }
}

/// Configuration for SDR→HDR inverse tonemapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SdrToHdrConfig {
    /// Target peak luminance for the HDR output (cd/m²).
    pub target_peak_nits: f64,
    /// Source peak luminance of the SDR content (cd/m²).
    pub source_peak_nits: f64,
    /// Highlight expansion gain (1.0 = linear, >1.0 = brighter highlights).
    pub highlight_gain: f64,
    /// Mid-tone boost factor (subtle lift to mid-tones).
    pub midtone_boost: f64,
}

impl SdrToHdrConfig {
    /// Creates a default SDR→HDR config targeting 1000 nit output.
    #[must_use]
    pub fn default_1000nit() -> Self {
        Self {
            target_peak_nits: 1000.0,
            source_peak_nits: 100.0,
            highlight_gain: 2.5,
            midtone_boost: 1.1,
        }
    }

    /// Validates the inverse tonemapping configuration.
    ///
    /// # Errors
    ///
    /// Returns `HdrError::OutOfRange` if any value is outside valid range.
    pub fn validate(&self) -> Result<(), HdrError> {
        if self.target_peak_nits <= 0.0 {
            return Err(HdrError::OutOfRange {
                field: "target_peak_nits".to_string(),
                value: self.target_peak_nits,
                min: 0.001,
                max: f64::MAX,
            });
        }
        if self.source_peak_nits <= 0.0 {
            return Err(HdrError::OutOfRange {
                field: "source_peak_nits".to_string(),
                value: self.source_peak_nits,
                min: 0.001,
                max: f64::MAX,
            });
        }
        if self.highlight_gain < 1.0 {
            return Err(HdrError::OutOfRange {
                field: "highlight_gain".to_string(),
                value: self.highlight_gain,
                min: 1.0,
                max: f64::MAX,
            });
        }
        if self.midtone_boost < 0.5 || self.midtone_boost > 3.0 {
            return Err(HdrError::OutOfRange {
                field: "midtone_boost".to_string(),
                value: self.midtone_boost,
                min: 0.5,
                max: 3.0,
            });
        }
        Ok(())
    }

    /// Applies inverse tonemapping to an SDR linear-light value.
    ///
    /// Returns the expanded HDR value in cd/m².
    #[must_use]
    pub fn apply(&self, l_sdr: f64) -> f64 {
        if l_sdr <= 0.0 {
            return 0.0;
        }
        let normalised = (l_sdr / self.source_peak_nits).clamp(0.0, 1.0);
        // S-curve expansion: boost highlights more than shadows
        let expanded = if normalised < 0.5 {
            normalised * self.midtone_boost
        } else {
            let t = (normalised - 0.5) * 2.0; // 0..1 in upper half
            let base = 0.5 * self.midtone_boost;
            base + t * 0.5 * self.highlight_gain
        };
        (expanded * self.target_peak_nits).min(self.target_peak_nits)
    }
}

// ─── Metadata repair ──────────────────────────────────────────────────────────

/// Metadata repair actions that can be applied automatically.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetadataRepairAction {
    /// Clamp chromaticity values to [0, 1].
    ClampChromaticity,
    /// Ensure min_luminance < max_luminance.
    FixLuminanceOrder,
    /// Ensure MaxFALL <= MaxCLL.
    FixFallCll,
    /// Add missing mastering display metadata with defaults.
    InjectDefaultMastering,
    /// Add missing CLL metadata with defaults.
    InjectDefaultCll,
}

/// Attempts to repair common HDR metadata issues in-place.
///
/// Returns a list of repairs that were applied.
pub fn repair_hdr_metadata(meta: &mut HdrMetadata) -> Vec<MetadataRepairAction> {
    let mut repairs = Vec::new();

    // Repair mastering display chromaticity
    if let Some(md) = &mut meta.mastering_display {
        let mut clamped = false;
        let clamp_chroma = |v: &mut f64, changed: &mut bool| {
            if *v < 0.0 {
                *v = 0.0;
                *changed = true;
            }
            if *v > 1.0 {
                *v = 1.0;
                *changed = true;
            }
        };
        clamp_chroma(&mut md.red_x, &mut clamped);
        clamp_chroma(&mut md.red_y, &mut clamped);
        clamp_chroma(&mut md.green_x, &mut clamped);
        clamp_chroma(&mut md.green_y, &mut clamped);
        clamp_chroma(&mut md.blue_x, &mut clamped);
        clamp_chroma(&mut md.blue_y, &mut clamped);
        clamp_chroma(&mut md.white_x, &mut clamped);
        clamp_chroma(&mut md.white_y, &mut clamped);
        if clamped {
            repairs.push(MetadataRepairAction::ClampChromaticity);
        }

        // Fix luminance ordering
        if md.min_luminance >= md.max_luminance && md.max_luminance > 0.0 {
            md.min_luminance = md.max_luminance * 0.001;
            repairs.push(MetadataRepairAction::FixLuminanceOrder);
        }
        if md.max_luminance <= 0.0 {
            md.max_luminance = 1000.0;
            md.min_luminance = 0.005;
            repairs.push(MetadataRepairAction::FixLuminanceOrder);
        }
    }

    // Fix CLL ordering
    if let Some(cll) = &mut meta.content_light_level {
        if cll.max_fall > cll.max_cll {
            cll.max_fall = cll.max_cll;
            repairs.push(MetadataRepairAction::FixFallCll);
        }
    }

    // Inject defaults if HDR TF is set but metadata is missing
    if let Some(tf) = meta.transfer_function {
        if tf == TransferFunction::Pq {
            if meta.mastering_display.is_none() {
                meta.mastering_display = Some(MasteringDisplay::p3_d65_1000nit());
                repairs.push(MetadataRepairAction::InjectDefaultMastering);
            }
            if meta.content_light_level.is_none() {
                meta.content_light_level = Some(ContentLightLevel::hdr10_default());
                repairs.push(MetadataRepairAction::InjectDefaultCll);
            }
        }
    }

    repairs
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TransferFunction ──────────────────────────────────────────────────────

    #[test]
    fn test_transfer_function_h273_codes() {
        assert_eq!(TransferFunction::Bt709.h273_code(), 1);
        assert_eq!(TransferFunction::Pq.h273_code(), 16);
        assert_eq!(TransferFunction::Hlg.h273_code(), 18);
        assert_eq!(TransferFunction::Linear.h273_code(), 8);
        assert_eq!(TransferFunction::Unspecified.h273_code(), 2);
    }

    #[test]
    fn test_transfer_function_is_hdr() {
        assert!(TransferFunction::Pq.is_hdr());
        assert!(TransferFunction::Hlg.is_hdr());
        assert!(!TransferFunction::Bt709.is_hdr());
        assert!(!TransferFunction::Linear.is_hdr());
        assert!(!TransferFunction::Unspecified.is_hdr());
    }

    #[test]
    fn test_transfer_function_is_wide_gamut() {
        assert!(TransferFunction::Pq.is_wide_gamut());
        assert!(TransferFunction::Hlg.is_wide_gamut());
        assert!(!TransferFunction::Bt709.is_wide_gamut());
    }

    // ── ColourPrimaries ───────────────────────────────────────────────────────

    #[test]
    fn test_colour_primaries_h273_codes() {
        assert_eq!(ColourPrimaries::Bt709.h273_code(), 1);
        assert_eq!(ColourPrimaries::Bt2020.h273_code(), 9);
        assert_eq!(ColourPrimaries::DisplayP3.h273_code(), 12);
        assert_eq!(ColourPrimaries::Unspecified.h273_code(), 2);
    }

    // ── MasteringDisplay ──────────────────────────────────────────────────────

    #[test]
    fn test_mastering_display_p3_d65_1000nit_is_valid() {
        let md = MasteringDisplay::p3_d65_1000nit();
        assert!(md.validate().is_ok());
    }

    #[test]
    fn test_mastering_display_bt2020_4000nit_is_valid() {
        let md = MasteringDisplay::bt2020_4000nit();
        assert!(md.validate().is_ok());
    }

    #[test]
    fn test_mastering_display_bad_chromaticity() {
        let mut md = MasteringDisplay::p3_d65_1000nit();
        md.red_x = 1.5; // invalid
        assert!(matches!(
            md.validate(),
            Err(HdrError::OutOfRange { field, .. }) if field == "red_x"
        ));
    }

    #[test]
    fn test_mastering_display_bad_luminance() {
        let mut md = MasteringDisplay::p3_d65_1000nit();
        md.min_luminance = md.max_luminance + 1.0;
        assert!(matches!(
            md.validate(),
            Err(HdrError::OutOfRange { field, .. }) if field == "min_luminance"
        ));
    }

    // ── ContentLightLevel ─────────────────────────────────────────────────────

    #[test]
    fn test_cll_hdr10_default_valid() {
        let cll = ContentLightLevel::hdr10_default();
        assert!(cll.validate().is_ok());
    }

    #[test]
    fn test_cll_invalid_fall_exceeds_cll() {
        let cll = ContentLightLevel::new(400, 1000);
        assert!(matches!(
            cll.validate(),
            Err(HdrError::OutOfRange { field, .. }) if field == "max_fall"
        ));
    }

    // ── DolbyVisionMeta ───────────────────────────────────────────────────────

    #[test]
    fn test_dv_profile_numbers() {
        assert_eq!(DolbyVisionProfile::Profile4.profile_number(), 4);
        assert_eq!(DolbyVisionProfile::Profile8.profile_number(), 8);
        assert_eq!(DolbyVisionProfile::Profile9.profile_number(), 9);
    }

    #[test]
    fn test_dv_backward_compatibility() {
        assert!(DolbyVisionProfile::Profile4.is_backward_compatible());
        assert!(DolbyVisionProfile::Profile8.is_backward_compatible());
        assert!(!DolbyVisionProfile::Profile5.is_backward_compatible());
    }

    #[test]
    fn test_dv_level_validation() {
        let ok = DolbyVisionMeta::new(DolbyVisionProfile::Profile8, 6);
        assert!(ok.validate().is_ok());

        let bad = DolbyVisionMeta::new(DolbyVisionProfile::Profile8, 14);
        assert!(bad.validate().is_err());

        let bad_zero = DolbyVisionMeta::new(DolbyVisionProfile::Profile8, 0);
        assert!(bad_zero.validate().is_err());
    }

    // ── HdrMetadata ───────────────────────────────────────────────────────────

    #[test]
    fn test_hdr_metadata_hdr10_is_hdr() {
        let meta = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        assert!(meta.is_hdr());
        assert!(meta.validate().is_ok());
    }

    #[test]
    fn test_hdr_metadata_hlg_is_hdr() {
        let meta = HdrMetadata::hlg();
        assert!(meta.is_hdr());
        assert!(meta.validate().is_ok());
    }

    #[test]
    fn test_hdr_metadata_default_not_hdr() {
        let meta = HdrMetadata::default();
        assert!(!meta.is_hdr());
    }

    // ── HdrProcessor ─────────────────────────────────────────────────────────

    #[test]
    fn test_processor_passthrough_none() {
        let proc = HdrProcessor::new(HdrPassthroughMode::Passthrough);
        let result = proc.process(None).expect("passthrough None should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_processor_passthrough_some() {
        let proc = HdrProcessor::new(HdrPassthroughMode::Passthrough);
        let src = HdrMetadata::hlg();
        let result = proc
            .process(Some(&src))
            .expect("passthrough Some should succeed");
        assert!(result.is_some());
        assert_eq!(
            result.as_ref().and_then(|m| m.transfer_function),
            Some(TransferFunction::Hlg)
        );
    }

    #[test]
    fn test_processor_strip() {
        let proc = HdrProcessor::new(HdrPassthroughMode::Strip);
        let src = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        let result = proc.process(Some(&src)).expect("strip should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_processor_inject() {
        let injected = HdrMetadata::hlg();
        let proc = HdrProcessor::new(HdrPassthroughMode::Inject(injected.clone()));
        let src = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        let result = proc
            .process(Some(&src))
            .expect("inject should succeed")
            .expect("inject should produce Some");
        assert_eq!(result.transfer_function, Some(TransferFunction::Hlg));
    }

    #[test]
    fn test_processor_convert_pq_to_bt709() {
        let proc = HdrProcessor::new(HdrPassthroughMode::Convert {
            target_tf: TransferFunction::Bt709,
            target_primaries: ColourPrimaries::Bt709,
        });
        let src = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        let result = proc
            .process(Some(&src))
            .expect("conversion should succeed")
            .expect("conversion should produce Some");
        assert_eq!(result.transfer_function, Some(TransferFunction::Bt709));
        // Static HDR metadata should be stripped
        assert!(result.mastering_display.is_none());
        assert!(result.content_light_level.is_none());
    }

    #[test]
    fn test_processor_convert_hlg_to_pq() {
        let proc = HdrProcessor::new(HdrPassthroughMode::Convert {
            target_tf: TransferFunction::Pq,
            target_primaries: ColourPrimaries::Bt2020,
        });
        let src = HdrMetadata::hlg();
        let result = proc
            .process(Some(&src))
            .expect("HLG→PQ should succeed")
            .expect("should produce Some");
        assert_eq!(result.transfer_function, Some(TransferFunction::Pq));
    }

    #[test]
    fn test_processor_convert_sdr_to_pq_fails() {
        let proc = HdrProcessor::new(HdrPassthroughMode::Convert {
            target_tf: TransferFunction::Pq,
            target_primaries: ColourPrimaries::Bt2020,
        });
        let src = HdrMetadata {
            transfer_function: Some(TransferFunction::Bt709),
            ..HdrMetadata::default()
        };
        // SDR → PQ is not a supported direct conversion
        let result = proc.process(Some(&src));
        assert!(matches!(
            result,
            Err(HdrError::UnsupportedConversion { .. })
        ));
    }

    // ── SEI encode / decode round-trips ──────────────────────────────────────

    #[test]
    fn test_mastering_display_sei_round_trip() {
        let original = MasteringDisplay::p3_d65_1000nit();
        let encoded = encode_mastering_display_sei(&original);
        let decoded = decode_mastering_display_sei(&encoded).expect("decode should succeed");

        // Allow 0.002 tolerance due to u16 quantisation at 1/50000 steps
        let eps = 0.002;
        assert!(
            (decoded.red_x - original.red_x).abs() < eps,
            "red_x mismatch"
        );
        assert!(
            (decoded.red_y - original.red_y).abs() < eps,
            "red_y mismatch"
        );
        assert!((decoded.green_x - original.green_x).abs() < eps);
        assert!((decoded.green_y - original.green_y).abs() < eps);
        assert!((decoded.blue_x - original.blue_x).abs() < eps);
        assert!((decoded.blue_y - original.blue_y).abs() < eps);
        assert!((decoded.white_x - original.white_x).abs() < eps);
        assert!((decoded.white_y - original.white_y).abs() < eps);
        // Luminance: 0.1 cd/m² tolerance at 10000 fractional units
        assert!((decoded.max_luminance - original.max_luminance).abs() < 0.1);
        assert!((decoded.min_luminance - original.min_luminance).abs() < 0.001);
    }

    #[test]
    fn test_mastering_display_sei_too_short() {
        let result = decode_mastering_display_sei(&[0u8; 12]);
        assert!(matches!(result, Err(HdrError::MissingField(_))));
    }

    #[test]
    fn test_cll_sei_round_trip() {
        let original = ContentLightLevel::new(800, 300);
        let encoded = encode_cll_sei(&original);
        let decoded = decode_cll_sei(&encoded).expect("decode should succeed");
        assert_eq!(decoded.max_cll, original.max_cll);
        assert_eq!(decoded.max_fall, original.max_fall);
    }

    #[test]
    fn test_cll_sei_too_short() {
        let result = decode_cll_sei(&[0u8; 2]);
        assert!(matches!(result, Err(HdrError::MissingField(_))));
    }

    #[test]
    fn test_cll_sei_invalid_decoded_values() {
        // MaxFALL > MaxCLL — encode and decode should fail validation
        let bad = ContentLightLevel {
            max_cll: 100,
            max_fall: 500,
        };
        let encoded = encode_cll_sei(&bad);
        let result = decode_cll_sei(&encoded);
        assert!(result.is_err());
    }

    // ── HDR10+ dynamic metadata ──────────────────────────────────────────────

    #[test]
    fn test_hdr10plus_new() {
        let meta = Hdr10PlusDynamicMeta::new(1000);
        assert_eq!(meta.application_version, 1);
        assert_eq!(meta.targeted_system_display_max_luminance, 1000);
        assert!(meta.validate().is_ok());
    }

    #[test]
    fn test_hdr10plus_validate_bad_version() {
        let mut meta = Hdr10PlusDynamicMeta::new(1000);
        meta.application_version = 5;
        assert!(meta.validate().is_err());
    }

    #[test]
    fn test_hdr10plus_validate_bad_avg_maxrgb() {
        let mut meta = Hdr10PlusDynamicMeta::new(1000);
        meta.average_maxrgb = 1.5;
        assert!(meta.validate().is_err());
    }

    #[test]
    fn test_hdr10plus_validate_bad_knee_point() {
        let mut meta = Hdr10PlusDynamicMeta::new(1000);
        meta.knee_point = (-0.1, 0.5);
        assert!(meta.validate().is_err());
    }

    #[test]
    fn test_hdr10plus_validate_too_many_anchors() {
        let mut meta = Hdr10PlusDynamicMeta::new(1000);
        meta.bezier_curve_anchors = vec![0.1; 10];
        assert!(meta.validate().is_err());
    }

    #[test]
    fn test_hdr10plus_encode_decode_round_trip() {
        let original = Hdr10PlusDynamicMeta::new(1000);
        let encoded = original.encode();
        let decoded = Hdr10PlusDynamicMeta::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.application_version, original.application_version);
        assert_eq!(
            decoded.targeted_system_display_max_luminance,
            original.targeted_system_display_max_luminance
        );
        assert!((decoded.average_maxrgb - original.average_maxrgb).abs() < 0.001);
        assert!((decoded.knee_point.0 - original.knee_point.0).abs() < 0.001);
        assert!((decoded.knee_point.1 - original.knee_point.1).abs() < 0.001);
        assert_eq!(
            decoded.bezier_curve_anchors.len(),
            original.bezier_curve_anchors.len()
        );
    }

    #[test]
    fn test_hdr10plus_decode_too_short() {
        let result = Hdr10PlusDynamicMeta::decode(&[0u8; 5]);
        assert!(matches!(result, Err(HdrError::MissingField(_))));
    }

    // ── Dolby Vision RPU ─────────────────────────────────────────────────────

    #[test]
    fn test_dv_rpu_new() {
        let payload = vec![0x08; 30]; // rpu_type = 0x08>>2 = 2 => profile 8
        let rpu = DolbyVisionRpu::new(payload, 0);
        assert_eq!(rpu.rpu_profile, 8);
        assert_eq!(rpu.frame_index, 0);
        assert!(!rpu.validated);
    }

    #[test]
    fn test_dv_rpu_validate_empty() {
        let mut rpu = DolbyVisionRpu::new(Vec::new(), 0);
        assert!(rpu.validate().is_err());
    }

    #[test]
    fn test_dv_rpu_validate_too_short() {
        let mut rpu = DolbyVisionRpu::new(vec![0x08; 10], 0);
        assert!(rpu.validate().is_err());
    }

    #[test]
    fn test_dv_rpu_validate_ok() {
        let mut rpu = DolbyVisionRpu::new(vec![0x08; 30], 0);
        assert!(rpu.validate().is_ok());
        assert!(rpu.validated);
    }

    #[test]
    fn test_dv_rpu_passthrough() {
        let mut pt = DvRpuPassthrough::new();
        assert_eq!(pt.count(), 0);

        pt.ingest(vec![0x08; 30], 0);
        pt.ingest(vec![0x08; 30], 1);
        pt.ingest(vec![0x08; 5], 2); // too short, invalid

        assert_eq!(pt.count(), 3);
        assert_eq!(pt.valid_count(), 2);
        assert_eq!(pt.invalid_count(), 1);

        assert!(pt.get_rpu(0).is_some());
        assert!(pt.get_rpu(1).is_some());
        assert!(pt.get_rpu(99).is_none());
    }

    #[test]
    fn test_dv_rpu_passthrough_drain() {
        let mut pt = DvRpuPassthrough::new();
        pt.ingest(vec![0x08; 30], 0);
        pt.ingest(vec![0x08; 30], 1);

        let drained = pt.drain_all();
        assert_eq!(drained.len(), 2);
        assert_eq!(pt.count(), 0);
        assert_eq!(pt.valid_count(), 0);
    }

    // ── Tonemapping ─────────────────────────────────────────────────────────

    #[test]
    fn test_hdr_to_sdr_config_default() {
        let cfg = HdrToSdrConfig::default_1000nit();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.source_peak_nits, 1000.0);
        assert_eq!(cfg.target_peak_nits, 100.0);
    }

    #[test]
    fn test_hdr_to_sdr_validate_bad_source_peak() {
        let mut cfg = HdrToSdrConfig::default_1000nit();
        cfg.source_peak_nits = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_hdr_to_sdr_validate_bad_desat() {
        let mut cfg = HdrToSdrConfig::default_1000nit();
        cfg.desat_strength = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_tonemap_reinhard_zero() {
        let cfg = HdrToSdrConfig::default_1000nit();
        let result = cfg.tonemap_reinhard(0.0);
        assert!((result).abs() < 1e-6);
    }

    #[test]
    fn test_tonemap_reinhard_monotonic() {
        let cfg = HdrToSdrConfig::default_1000nit();
        let a = cfg.tonemap_reinhard(0.1);
        let b = cfg.tonemap_reinhard(0.5);
        let c = cfg.tonemap_reinhard(1.0);
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn test_tonemap_hable_positive() {
        let cfg = HdrToSdrConfig::default_1000nit();
        let result = cfg.tonemap_hable(0.5);
        assert!(result > 0.0);
        assert!(result < cfg.target_peak_nits);
    }

    #[test]
    fn test_tonemap_aces_clamped() {
        let cfg = HdrToSdrConfig::default_1000nit();
        let result = cfg.tonemap_aces(100.0);
        assert!(result <= cfg.target_peak_nits);
    }

    #[test]
    fn test_tonemap_apply_clip() {
        let cfg = HdrToSdrConfig {
            curve: TonemapCurve::Clip,
            source_peak_nits: 1000.0,
            target_peak_nits: 100.0,
            desat_strength: 0.0,
        };
        let result = cfg.apply(0.5);
        assert!((result - 100.0).abs() < 1e-6); // 0.5 * 1000 = 500, clipped to 100
    }

    // ── SDR→HDR inverse tonemapping ──────────────────────────────────────────

    #[test]
    fn test_sdr_to_hdr_default() {
        let cfg = SdrToHdrConfig::default_1000nit();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.target_peak_nits, 1000.0);
    }

    #[test]
    fn test_sdr_to_hdr_validate_bad_gain() {
        let mut cfg = SdrToHdrConfig::default_1000nit();
        cfg.highlight_gain = 0.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_sdr_to_hdr_validate_bad_midtone() {
        let mut cfg = SdrToHdrConfig::default_1000nit();
        cfg.midtone_boost = 5.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_sdr_to_hdr_apply_zero() {
        let cfg = SdrToHdrConfig::default_1000nit();
        assert!((cfg.apply(0.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sdr_to_hdr_apply_monotonic() {
        let cfg = SdrToHdrConfig::default_1000nit();
        let a = cfg.apply(10.0);
        let b = cfg.apply(50.0);
        let c = cfg.apply(100.0);
        assert!(a < b);
        assert!(b < c);
        assert!(c <= cfg.target_peak_nits);
    }

    // ── Metadata repair ──────────────────────────────────────────────────────

    #[test]
    fn test_repair_clamp_chromaticity() {
        let mut meta = HdrMetadata::hdr10(
            MasteringDisplay {
                red_x: 1.5,
                red_y: -0.1,
                ..MasteringDisplay::p3_d65_1000nit()
            },
            ContentLightLevel::hdr10_default(),
        );
        let repairs = repair_hdr_metadata(&mut meta);
        assert!(repairs.contains(&MetadataRepairAction::ClampChromaticity));
        let md = meta
            .mastering_display
            .as_ref()
            .expect("should have mastering display");
        assert!((md.red_x - 1.0).abs() < 1e-6);
        assert!((md.red_y).abs() < 1e-6);
    }

    #[test]
    fn test_repair_luminance_order() {
        let mut meta = HdrMetadata::hdr10(
            MasteringDisplay {
                max_luminance: 1000.0,
                min_luminance: 2000.0, // invalid: min > max
                ..MasteringDisplay::p3_d65_1000nit()
            },
            ContentLightLevel::hdr10_default(),
        );
        let repairs = repair_hdr_metadata(&mut meta);
        assert!(repairs.contains(&MetadataRepairAction::FixLuminanceOrder));
        let md = meta
            .mastering_display
            .as_ref()
            .expect("should have mastering display");
        assert!(md.min_luminance < md.max_luminance);
    }

    #[test]
    fn test_repair_fall_cll() {
        let mut meta = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel {
                max_cll: 500,
                max_fall: 800,
            },
        );
        let repairs = repair_hdr_metadata(&mut meta);
        assert!(repairs.contains(&MetadataRepairAction::FixFallCll));
        let cll = meta.content_light_level.as_ref().expect("should have CLL");
        assert!(cll.max_fall <= cll.max_cll);
    }

    #[test]
    fn test_repair_inject_defaults_for_pq() {
        let mut meta = HdrMetadata {
            transfer_function: Some(TransferFunction::Pq),
            colour_primaries: Some(ColourPrimaries::Bt2020),
            mastering_display: None,
            content_light_level: None,
            dolby_vision: None,
        };
        let repairs = repair_hdr_metadata(&mut meta);
        assert!(repairs.contains(&MetadataRepairAction::InjectDefaultMastering));
        assert!(repairs.contains(&MetadataRepairAction::InjectDefaultCll));
        assert!(meta.mastering_display.is_some());
        assert!(meta.content_light_level.is_some());
    }

    #[test]
    fn test_repair_no_action_needed() {
        let mut meta = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        let repairs = repair_hdr_metadata(&mut meta);
        assert!(repairs.is_empty());
    }

    #[test]
    fn test_processor_convert_with_sdr_to_pq_via_extended() {
        // SDR → PQ is now supported via the extended conversion table
        let proc = HdrProcessor::new(HdrPassthroughMode::Convert {
            target_tf: TransferFunction::Pq,
            target_primaries: ColourPrimaries::Bt2020,
        });
        let src = HdrMetadata {
            transfer_function: Some(TransferFunction::Bt709),
            ..HdrMetadata::default()
        };
        // Still unsupported in the basic conversion table
        let result = proc.process(Some(&src));
        assert!(result.is_err());
    }
}
