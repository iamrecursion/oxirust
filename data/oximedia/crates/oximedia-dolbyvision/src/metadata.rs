//! Dolby Vision metadata levels.
//!
//! This module defines metadata structures for different Dolby Vision levels.
//!
//! # Metadata level reference table
//!
//! Dolby Vision RPUs carry their dynamic and static grading data as a set of
//! numbered "levels", each with its own purpose and field layout. The table
//! below summarizes every level modeled by this crate; see each struct's own
//! docs for exact field semantics and unit scaling.
//!
//! | Level | Struct | Scope | Purpose | Key fields |
//! |---|---|---|---|---|
//! | L1 | [`Level1Metadata`] | Per-frame | Dynamic frame statistics used to drive the reshaping/tone curve for the current picture. | `min_pq`, `max_pq`, `avg_pq` |
//! | L2 | [`Level2Metadata`] | Per-frame, per-target-display | Trim pass: manual color-grading adjustments (from a colorist trim session) for one target display. Multiple L2 blocks may exist per frame, one per `target_display_index`. | `target_display_index`, `trim_slope`/`trim_offset`/`trim_power`, `trim_chroma_weight`, `trim_saturation_gain`, `ms_weight`, `saturation_vector_field`, `hue_vector_field` |
//! | L3 | [`Level3Metadata`] | Reserved | Reserved for future use by the Dolby Vision spec; carried through as an opaque byte buffer. | `reserved` |
//! | L4 | [`Level4Metadata`] | Per-frame or sequence | Global dimming / backward-compatible (HDR10) anchor: the reference luminance the base layer was graded at, so a non-DV display can approximate global dimming. | `anchor_pq`, `anchor_power`, `active_area_flag` |
//! | L5 | [`Level5Metadata`] | Per-frame | Active area: letterbox/pillarbox offsets describing the image region within the coded frame (e.g. to exclude black bars from tone mapping). | `active_area_left_offset`/`right`/`top`/`bottom_offset` |
//! | L6 | [`Level6Metadata`] | Sequence (static) | HDR10 fallback/compatibility metadata: `MaxCLL`/`MaxFALL` and mastering-display color volume (SMPTE ST 2086), for non-DV HDR10 playback. | `max_cll`, `max_fall`, `min_display_mastering_luminance`, `max_display_mastering_luminance`, `master_display_primaries`, `master_display_white_point` |
//! | L7 | [`Level7Metadata`] | Sequence (static) | Source display color volume: gamut and luminance range of the mastering display used for grading, for gamut-aware display management. | `source_min_pq`/`source_max_pq`, `source_primary_index`, `red_primary`/`green_primary`/`blue_primary`, `white_point` |
//! | L8 | [`Level8Metadata`] | Per-frame, per-target-display | Target display characteristics: describes one specific target display (peak/black luminance, primaries, EOTF, physical size, viewing environment) that L2 trims are computed for. | `target_display_index`, `target_max_pq`/`target_min_pq`, `target_primary_index`, `target_eotf`, `diagonal_size`, `peak_luminance`, `diffuse_white_luminance`, `ambient_luminance`, `surround_reflection` |
//! | L9 | [`Level9Metadata`] | Sequence (static) | Source display characteristics: mastering-display primaries and luminance range used as the reference for display management (complements L7). | `source_primary_index`, `source_max_pq`/`source_min_pq`, `source_diagonal` |
//! | L10 | [`Level10Metadata`] | Reserved | Reserved for future use; carried through as an opaque byte buffer. | `reserved` |
//! | L11 | [`Level11Metadata`] | Per-frame or sequence | Content type and creative-intent description: content classification plus display-management hints (sharpness, noise reduction, temporal filtering) applied downstream of tone mapping. | `content_type`, `whitepoint`, `reference_mode_flag`, `sharpness`, `noise_reduction`, `mpeg_noise_reduction`, `frame_rate`, `temporal_filter_strength` |
//!
//! [`parser::parse_rpu_bitstream`](crate::parser::parse_rpu_bitstream) parses
//! L1, L2, L5, L6, L8, L9, and L11 unconditionally for every RPU (L3, L4, L7,
//! and L10 are represented in this crate's data model but not emitted by the
//! current bitstream parser/writer pair); absent levels simply decode to
//! their `Default` value.

/// Level 1 metadata: Frame-level metadata.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level1Metadata {
    /// Minimum PQ value in frame
    pub min_pq: u16,

    /// Maximum PQ value in frame
    pub max_pq: u16,

    /// Average PQ value in frame
    pub avg_pq: u16,
}

/// Level 2 metadata: Trim passes for target display adaptation.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level2Metadata {
    /// Target display index
    pub target_display_index: u8,

    /// Trim slope
    pub trim_slope: i16,

    /// Trim offset
    pub trim_offset: i16,

    /// Trim power
    pub trim_power: i16,

    /// Trim chroma weight
    pub trim_chroma_weight: i16,

    /// Trim saturation gain
    pub trim_saturation_gain: i16,

    /// MS weight (mastering display weight)
    pub ms_weight: i16,

    /// Target mid contrast
    pub target_mid_contrast: u16,

    /// Clip trim
    pub clip_trim: u16,

    /// Saturation vector field
    pub saturation_vector_field: Vec<SaturationVector>,

    /// Hue vector field
    pub hue_vector_field: Vec<HueVector>,
}

/// Saturation adjustment vector.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SaturationVector {
    /// Saturation gain
    pub saturation_gain: i16,
}

/// Hue adjustment vector.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HueVector {
    /// Hue angle shift
    pub hue_angle: i16,
}

/// Level 3 metadata: Reserved for future use.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level3Metadata {
    /// Reserved data
    pub reserved: Vec<u8>,
}

/// Level 5 metadata: Active area (image area within frame).
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level5Metadata {
    /// Active area left offset
    pub active_area_left_offset: u16,

    /// Active area right offset
    pub active_area_right_offset: u16,

    /// Active area top offset
    pub active_area_top_offset: u16,

    /// Active area bottom offset
    pub active_area_bottom_offset: u16,
}

/// Level 6 metadata: Fallback metadata for non-Dolby Vision displays.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level6Metadata {
    /// Maximum content light level (nits)
    pub max_cll: u16,

    /// Maximum frame-average light level (nits)
    pub max_fall: u16,

    /// Master display minimum luminance (0.0001 nits units)
    pub min_display_mastering_luminance: u32,

    /// Master display maximum luminance (nits)
    pub max_display_mastering_luminance: u32,

    /// Master display primaries (x, y for R, G, B in 0.00002 units)
    pub master_display_primaries: [[u16; 2]; 3],

    /// Master display white point (x, y in 0.00002 units)
    pub master_display_white_point: [u16; 2],
}

impl Level6Metadata {
    /// Create Level 6 metadata for BT.2020 primaries.
    #[must_use]
    pub fn bt2020() -> Self {
        Self {
            max_cll: 1000,
            max_fall: 400,
            min_display_mastering_luminance: 50, // 0.005 nits
            max_display_mastering_luminance: 1000,
            master_display_primaries: [
                [34000, 16000], // Red: (0.680, 0.320)
                [13250, 34500], // Green: (0.265, 0.690)
                [7500, 3000],   // Blue: (0.150, 0.060)
            ],
            master_display_white_point: [15635, 16450], // D65: (0.3127, 0.3290)
        }
    }

    /// Create Level 6 metadata for DCI-P3 primaries.
    #[must_use]
    pub fn dci_p3() -> Self {
        Self {
            max_cll: 1000,
            max_fall: 400,
            min_display_mastering_luminance: 50,
            max_display_mastering_luminance: 1000,
            master_display_primaries: [
                [34000, 15500], // Red: (0.680, 0.310)
                [16500, 35000], // Green: (0.330, 0.700)
                [7500, 3000],   // Blue: (0.150, 0.060)
            ],
            master_display_white_point: [15700, 17850], // DCI white: (0.314, 0.357)
        }
    }
}

/// Level 8 metadata: Target display characteristics.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level8Metadata {
    /// Target display index
    pub target_display_index: u8,

    /// Target maximum luminance (nits)
    pub target_max_pq: u16,

    /// Target minimum luminance (PQ code)
    pub target_min_pq: u16,

    /// Target primary index (0 = BT.2020, 1 = DCI-P3, 2 = BT.709)
    pub target_primary_index: u8,

    /// Target EOTF (0 = BT.1886, 1 = PQ, 2 = HLG)
    pub target_eotf: u8,

    /// Diagonal size in inches
    pub diagonal_size: u16,

    /// Peak luminance (nits)
    pub peak_luminance: u16,

    /// Diffuse white luminance (nits)
    pub diffuse_white_luminance: u16,

    /// Ambient luminance (nits)
    pub ambient_luminance: u16,

    /// Surround reflection
    pub surround_reflection: u16,
}

impl Level8Metadata {
    /// Create Level 8 metadata for a standard HDR display (1000 nits).
    #[must_use]
    pub fn hdr_1000() -> Self {
        Self {
            target_display_index: 0,
            target_max_pq: 3696, // 1000 nits in PQ
            target_min_pq: 62,   // 0.005 nits in PQ
            target_primary_index: 0,
            target_eotf: 1, // PQ
            diagonal_size: 65,
            peak_luminance: 1000,
            diffuse_white_luminance: 200,
            ambient_luminance: 5,
            surround_reflection: 10,
        }
    }

    /// Create Level 8 metadata for a high-end HDR display (4000 nits).
    #[must_use]
    pub fn hdr_4000() -> Self {
        Self {
            target_display_index: 1,
            target_max_pq: 4079, // 4000 nits in PQ
            target_min_pq: 62,
            target_primary_index: 0,
            target_eotf: 1, // PQ
            diagonal_size: 65,
            peak_luminance: 4000,
            diffuse_white_luminance: 200,
            ambient_luminance: 5,
            surround_reflection: 10,
        }
    }

    /// Create Level 8 metadata for an HLG display.
    #[must_use]
    pub fn hlg() -> Self {
        Self {
            target_display_index: 2,
            target_max_pq: 2081, // ~100 nits nominal
            target_min_pq: 0,
            target_primary_index: 0,
            target_eotf: 2, // HLG
            diagonal_size: 55,
            peak_luminance: 1000,
            diffuse_white_luminance: 100,
            ambient_luminance: 5,
            surround_reflection: 10,
        }
    }
}

/// Level 9 metadata: Source display characteristics.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level9Metadata {
    /// Source primary index (0 = BT.2020, 1 = DCI-P3, 2 = BT.709)
    pub source_primary_index: u8,

    /// Source maximum PQ
    pub source_max_pq: u16,

    /// Source minimum PQ
    pub source_min_pq: u16,

    /// Source diagonal size in inches
    pub source_diagonal: u16,
}

impl Level9Metadata {
    /// Create Level 9 metadata for BT.2020 mastering.
    #[must_use]
    pub fn bt2020_mastering() -> Self {
        Self {
            source_primary_index: 0,
            source_max_pq: 3696, // 1000 nits
            source_min_pq: 62,   // 0.005 nits
            source_diagonal: 65,
        }
    }

    /// Create Level 9 metadata for DCI-P3 mastering.
    #[must_use]
    pub fn dci_p3_mastering() -> Self {
        Self {
            source_primary_index: 1,
            source_max_pq: 3696,
            source_min_pq: 62,
            source_diagonal: 65,
        }
    }
}

/// Level 4 metadata: Global dimming data.
///
/// Describes the anchor luminance for backward-compatible (HDR10) rendering
/// when a Dolby Vision display management pipeline is not available.
/// The anchor PQ code represents the reference level at which the base layer
/// was graded, enabling global dimming compensation.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level4Metadata {
    /// Anchor PQ code value (0-4095) representing the dimming anchor luminance.
    pub anchor_pq: u16,
    /// Anchor power gain (fixed-point, scaled by 2^12, default 4096 = 1.0).
    pub anchor_power: u16,
    /// Active area flag: when true, dimming only applies to the active area.
    pub active_area_flag: bool,
}

impl Level4Metadata {
    /// Create Level 4 metadata with a typical SDR-like anchor (100 nits).
    #[must_use]
    pub fn sdr_anchor() -> Self {
        Self {
            anchor_pq: nits_to_pq(100),
            anchor_power: 1 << 12, // 1.0 in fixed-point
            active_area_flag: false,
        }
    }

    /// Create Level 4 metadata for a high-brightness mastering anchor (1000 nits).
    #[must_use]
    pub fn hdr_anchor() -> Self {
        Self {
            anchor_pq: nits_to_pq(1000),
            anchor_power: 1 << 12,
            active_area_flag: false,
        }
    }

    /// Create Level 4 metadata for a custom anchor luminance in nits.
    #[must_use]
    pub fn custom_anchor(nits: u16, power_gain: f32) -> Self {
        let anchor_power = (power_gain * (1 << 12) as f32).clamp(0.0, 65535.0) as u16;
        Self {
            anchor_pq: nits_to_pq(nits),
            anchor_power,
            active_area_flag: false,
        }
    }

    /// Return the anchor power as a floating-point multiplier.
    #[must_use]
    pub fn anchor_power_f32(&self) -> f32 {
        f32::from(self.anchor_power) / (1 << 12) as f32
    }

    /// Return the anchor luminance in nits.
    #[must_use]
    pub fn anchor_nits(&self) -> u16 {
        pq_to_nits(self.anchor_pq)
    }

    /// Validate that the metadata fields are within specification ranges.
    ///
    /// Returns a list of error descriptions; empty means valid.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.anchor_pq > 4095 {
            errors.push(format!("anchor_pq {} exceeds maximum 4095", self.anchor_pq));
        }
        if self.anchor_power == 0 {
            errors.push("anchor_power must be > 0".to_string());
        }
        errors
    }
}

/// Level 7 metadata: Source display color volume.
///
/// Describes the color volume (gamut and luminance range) of the mastering display
/// used for content creation. This allows receivers to understand the intended
/// color volume and perform appropriate gamut mapping.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level7Metadata {
    /// Source display minimum PQ code value (0-4095).
    pub source_min_pq: u16,
    /// Source display maximum PQ code value (0-4095).
    pub source_max_pq: u16,
    /// Source display color primaries index (0 = BT.2020, 1 = DCI-P3, 2 = BT.709).
    pub source_primary_index: u8,
    /// Source display primaries: red (x, y) in 0.00002 units.
    pub red_primary: [u16; 2],
    /// Source display primaries: green (x, y) in 0.00002 units.
    pub green_primary: [u16; 2],
    /// Source display primaries: blue (x, y) in 0.00002 units.
    pub blue_primary: [u16; 2],
    /// Source display white point (x, y) in 0.00002 units.
    pub white_point: [u16; 2],
}

impl Level7Metadata {
    /// Create Level 7 metadata for BT.2020 primaries with 1000-nit peak.
    #[must_use]
    pub fn bt2020_1000nits() -> Self {
        Self {
            source_min_pq: 62, // ~0.005 nits
            source_max_pq: nits_to_pq(1000),
            source_primary_index: 0,
            red_primary: [34000, 16000],   // (0.680, 0.320)
            green_primary: [13250, 34500], // (0.265, 0.690)
            blue_primary: [7500, 3000],    // (0.150, 0.060)
            white_point: [15635, 16450],   // D65
        }
    }

    /// Create Level 7 metadata for BT.2020 primaries with 4000-nit peak.
    #[must_use]
    pub fn bt2020_4000nits() -> Self {
        Self {
            source_min_pq: 62,
            source_max_pq: nits_to_pq(4000),
            source_primary_index: 0,
            red_primary: [34000, 16000],
            green_primary: [13250, 34500],
            blue_primary: [7500, 3000],
            white_point: [15635, 16450],
        }
    }

    /// Create Level 7 metadata for DCI-P3 primaries with 1000-nit peak.
    #[must_use]
    pub fn dci_p3_1000nits() -> Self {
        Self {
            source_min_pq: 62,
            source_max_pq: nits_to_pq(1000),
            source_primary_index: 1,
            red_primary: [34000, 15500],   // (0.680, 0.310)
            green_primary: [16500, 35000], // (0.330, 0.700)
            blue_primary: [7500, 3000],    // (0.150, 0.060)
            white_point: [15635, 16450],   // D65
        }
    }

    /// Create Level 7 metadata for BT.709 primaries with 100-nit peak (SDR mastering).
    #[must_use]
    pub fn bt709_100nits() -> Self {
        Self {
            source_min_pq: 62,
            source_max_pq: nits_to_pq(100),
            source_primary_index: 2,
            red_primary: [32000, 16500],   // (0.640, 0.330)
            green_primary: [15000, 30000], // (0.300, 0.600)
            blue_primary: [7500, 3000],    // (0.150, 0.060)
            white_point: [15635, 16450],   // D65
        }
    }

    /// Return the source display minimum luminance in nits.
    #[must_use]
    pub fn source_min_nits(&self) -> u16 {
        pq_to_nits(self.source_min_pq)
    }

    /// Return the source display maximum luminance in nits.
    #[must_use]
    pub fn source_max_nits(&self) -> u16 {
        pq_to_nits(self.source_max_pq)
    }

    /// Return the primary name as a string.
    #[must_use]
    pub fn primary_name(&self) -> &str {
        match self.source_primary_index {
            0 => "BT.2020",
            1 => "DCI-P3",
            2 => "BT.709",
            _ => "Unknown",
        }
    }

    /// Validate that the metadata fields are within specification ranges.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.source_min_pq > 4095 {
            errors.push(format!(
                "source_min_pq {} exceeds maximum 4095",
                self.source_min_pq
            ));
        }
        if self.source_max_pq > 4095 {
            errors.push(format!(
                "source_max_pq {} exceeds maximum 4095",
                self.source_max_pq
            ));
        }
        if self.source_min_pq >= self.source_max_pq {
            errors.push(format!(
                "source_min_pq ({}) >= source_max_pq ({})",
                self.source_min_pq, self.source_max_pq
            ));
        }
        if self.source_primary_index > 2 {
            errors.push(format!(
                "source_primary_index {} is out of range (0-2)",
                self.source_primary_index
            ));
        }
        errors
    }
}

/// Level 10 metadata: Reserved for future use.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level10Metadata {
    /// Reserved data
    pub reserved: Vec<u8>,
}

/// Level 11 metadata: Content type and description.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level11Metadata {
    /// Content type
    pub content_type: ContentType,

    /// White point
    pub whitepoint: u8,

    /// Reference mode flag
    pub reference_mode_flag: bool,

    /// Sharpness
    pub sharpness: u8,

    /// Noise reduction
    pub noise_reduction: u8,

    /// MPEG noise reduction
    pub mpeg_noise_reduction: u8,

    /// Frame rate
    pub frame_rate: u8,

    /// Temporal filter strength
    pub temporal_filter_strength: u8,
}

/// Content type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ContentType {
    /// Unknown content type
    #[default]
    Unknown = 0,

    /// Movie content
    Movie = 1,

    /// TV content
    Tv = 2,

    /// Sports content
    Sports = 3,

    /// Gaming content
    Gaming = 4,

    /// Animation content
    Animation = 5,
}

impl ContentType {
    /// Create from numeric value.
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Movie,
            2 => Self::Tv,
            3 => Self::Sports,
            4 => Self::Gaming,
            5 => Self::Animation,
            _ => Self::Unknown,
        }
    }
}

/// CMD (Content Metadata Descriptor) for extended content information.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContentMetadataDescriptor {
    /// Content title
    pub title: Option<String>,

    /// Content description
    pub description: Option<String>,

    /// Content language (ISO 639-2)
    pub language: Option<String>,

    /// Content creation date (ISO 8601)
    pub creation_date: Option<String>,

    /// Content creator
    pub creator: Option<String>,

    /// Content copyright
    pub copyright: Option<String>,

    /// Additional metadata key-value pairs
    pub additional_metadata: Vec<(String, String)>,
}

impl ContentMetadataDescriptor {
    /// Create a new empty CMD.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            title: None,
            description: None,
            language: None,
            creation_date: None,
            creator: None,
            copyright: None,
            additional_metadata: Vec::new(),
        }
    }

    /// Add a custom metadata field.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.additional_metadata.push((key, value));
    }
}

/// Trim pass for display adaptation.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrimPass {
    /// Target max PQ
    pub target_max_pq: u16,

    /// Target min PQ
    pub target_min_pq: u16,

    /// Trim slope
    pub trim_slope: i16,

    /// Trim offset
    pub trim_offset: i16,

    /// Trim power
    pub trim_power: i16,

    /// Trim chroma weight
    pub trim_chroma_weight: i16,

    /// Trim saturation gain
    pub trim_saturation_gain: i16,

    /// MS weight
    pub ms_weight: i16,
}

impl TrimPass {
    /// Create identity trim pass (no modification).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            target_max_pq: 4095,
            target_min_pq: 0,
            trim_slope: 1 << 12,
            trim_offset: 0,
            trim_power: 1 << 12,
            trim_chroma_weight: 1 << 12,
            trim_saturation_gain: 1 << 12,
            ms_weight: 1 << 12,
        }
    }

    /// Create trim pass for specific target peak brightness.
    #[must_use]
    pub fn for_peak_brightness(target_nits: u16) -> Self {
        // Convert nits to PQ code value (simplified)
        let target_max_pq = nits_to_pq(target_nits);

        Self {
            target_max_pq,
            target_min_pq: 62, // 0.005 nits
            trim_slope: 1 << 12,
            trim_offset: 0,
            trim_power: 1 << 12,
            trim_chroma_weight: 1 << 12,
            trim_saturation_gain: 1 << 12,
            ms_weight: 1 << 12,
        }
    }
}

/// Convert nits to PQ code value (0-4095).
#[must_use]
pub fn nits_to_pq(nits: u16) -> u16 {
    const M1: f64 = 0.159_301_758_113_479_8;
    const M2: f64 = 78.843_750;
    const C1: f64 = 0.835_937_5;
    const C2: f64 = 18.851_562_5;
    const C3: f64 = 18.6875;

    let y = f64::from(nits) / 10_000.0;
    let y_m1 = y.powf(M1);
    let pq = ((C1 + C2 * y_m1) / (1.0 + C3 * y_m1)).powf(M2);

    (pq * 4095.0).min(4095.0) as u16
}

/// Convert PQ code value (0-4095) to nits.
#[must_use]
#[allow(dead_code)]
pub fn pq_to_nits(pq: u16) -> u16 {
    const M1_INV: f64 = 1.0 / 0.159_301_758_113_479_8;
    const M2_INV: f64 = 1.0 / 78.843_750;
    const C1: f64 = 0.835_937_5;
    const C2: f64 = 18.851_562_5;
    const C3: f64 = 18.6875;

    let pq_norm = f64::from(pq) / 4095.0;
    let v = pq_norm.powf(M2_INV);
    let y = ((v - C1).max(0.0) / (C2 - C3 * v)).powf(M1_INV);

    (y * 10_000.0).min(10_000.0) as u16
}

/// Metadata block for generic extension.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetadataBlock {
    /// Block ID
    pub block_id: u8,

    /// Block length
    pub block_length: u16,

    /// Block data
    pub block_data: Vec<u8>,
}

/// Color volume transform parameters.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorVolumeTransform {
    /// 3D LUT size per dimension
    pub lut_size: u8,

    /// 3D LUT data (flattened RGB cube)
    pub lut_data: Vec<[u16; 3]>,
}

impl ColorVolumeTransform {
    /// Create identity transform.
    #[must_use]
    pub fn identity(size: u8) -> Self {
        let total_points = usize::from(size) * usize::from(size) * usize::from(size);
        let mut lut_data = Vec::with_capacity(total_points);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let scale = 4095_u16 / u16::from(size - 1);
                    lut_data.push([
                        u16::from(r) * scale,
                        u16::from(g) * scale,
                        u16::from(b) * scale,
                    ]);
                }
            }
        }

        Self {
            lut_size: size,
            lut_data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_type() {
        assert_eq!(ContentType::from_u8(0), ContentType::Unknown);
        assert_eq!(ContentType::from_u8(1), ContentType::Movie);
        assert_eq!(ContentType::from_u8(2), ContentType::Tv);
        assert_eq!(ContentType::from_u8(3), ContentType::Sports);
        assert_eq!(ContentType::from_u8(99), ContentType::Unknown);
    }

    #[test]
    fn test_level6_presets() {
        let bt2020 = Level6Metadata::bt2020();
        assert_eq!(bt2020.max_cll, 1000);
        assert_eq!(bt2020.master_display_primaries[0][0], 34000);

        let dci_p3 = Level6Metadata::dci_p3();
        assert_eq!(dci_p3.max_cll, 1000);
    }

    #[test]
    fn test_level8_presets() {
        let hdr1000 = Level8Metadata::hdr_1000();
        assert_eq!(hdr1000.peak_luminance, 1000);
        assert_eq!(hdr1000.target_eotf, 1);

        let hdr4000 = Level8Metadata::hdr_4000();
        assert_eq!(hdr4000.peak_luminance, 4000);

        let hlg = Level8Metadata::hlg();
        assert_eq!(hlg.target_eotf, 2);
    }

    #[test]
    fn test_level9_presets() {
        let bt2020 = Level9Metadata::bt2020_mastering();
        assert_eq!(bt2020.source_primary_index, 0);

        let p3 = Level9Metadata::dci_p3_mastering();
        assert_eq!(p3.source_primary_index, 1);
    }

    #[test]
    fn test_nits_pq_conversion() {
        let pq_100 = nits_to_pq(100);
        let nits_100 = pq_to_nits(pq_100);
        assert!((nits_100 as i32 - 100).abs() <= 5);

        let pq_1000 = nits_to_pq(1000);
        let nits_1000 = pq_to_nits(pq_1000);
        assert!((nits_1000 as i32 - 1000).abs() <= 50);

        let pq_10000 = nits_to_pq(10000);
        assert_eq!(pq_10000, 4095);
    }

    #[test]
    fn test_trim_pass() {
        let identity = TrimPass::identity();
        assert_eq!(identity.target_max_pq, 4095);
        assert_eq!(identity.trim_slope, 1 << 12);

        let trim_1000 = TrimPass::for_peak_brightness(1000);
        assert!(trim_1000.target_max_pq > 0);
        assert!(trim_1000.target_max_pq < 4095);
    }

    #[test]
    fn test_cmd() {
        let mut cmd = ContentMetadataDescriptor::new();
        cmd.add_metadata("key1".to_string(), "value1".to_string());
        assert_eq!(cmd.additional_metadata.len(), 1);
        assert_eq!(cmd.additional_metadata[0].0, "key1");
    }

    #[test]
    fn test_color_volume_transform() {
        let cvt = ColorVolumeTransform::identity(5);
        assert_eq!(cvt.lut_size, 5);
        assert_eq!(cvt.lut_data.len(), 125);
    }
}
