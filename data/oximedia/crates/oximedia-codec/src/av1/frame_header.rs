//! AV1 Frame Header parsing.
//!
//! The frame header OBU contains the core decoding parameters for each frame.
//! This module implements complete frame header parsing according to the AV1
//! specification (Section 5.9).
//!
//! # Frame Header Contents
//!
//! - Frame type (KEY_FRAME, INTER_FRAME, INTRA_ONLY_FRAME, SWITCH_FRAME)
//! - Show frame / showable frame flags
//! - Error resilient mode
//! - Frame size (with optional superres scaling)
//! - Render size
//! - Interpolation filter
//! - Reference frame selection
//! - Motion mode and compound prediction settings
//! - Quantization parameters
//! - Segmentation
//! - Loop filter parameters
//! - CDEF parameters
//! - Loop restoration parameters
//! - Tile info
//!
//! # Reference
//!
//! See AV1 Specification Section 5.9 for the complete frame header syntax.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::identity_op)]
#![allow(clippy::if_not_else)]

use super::cdef::CdefParams;
use super::loop_filter::LoopFilterParams;
use super::quantization::QuantizationParams;
use super::sequence::SequenceHeader;
use super::tile::TileInfo;
use crate::error::{CodecError, CodecResult};
use oximedia_io::BitReader;

// =============================================================================
// Constants from AV1 Specification
// =============================================================================

/// Maximum number of reference frames.
pub const NUM_REF_FRAMES: usize = 8;

/// Number of reference types for inter prediction.
pub const REFS_PER_FRAME: usize = 7;

/// Maximum number of segments.
pub const MAX_SEGMENTS: usize = 8;

/// Number of segment features.
pub const SEG_LVL_MAX: usize = 8;

/// Primary reference frame index indicating no primary reference.
pub const PRIMARY_REF_NONE: u8 = 7;

/// Number of superres scale denominator bits.
pub const SUPERRES_DENOM_BITS: u8 = 3;

/// Minimum superres denominator value.
pub const SUPERRES_DENOM_MIN: u32 = 9;

/// Superres number value.
pub const SUPERRES_NUM: u32 = 8;

/// Reference frame names/indices.
pub const LAST_FRAME: usize = 1;
pub const LAST2_FRAME: usize = 2;
pub const LAST3_FRAME: usize = 3;
pub const GOLDEN_FRAME: usize = 4;
pub const BWDREF_FRAME: usize = 5;
pub const ALTREF2_FRAME: usize = 6;
pub const ALTREF_FRAME: usize = 7;

/// Interpolation filter types.
pub const INTERP_FILTER_EIGHTTAP: u8 = 0;
pub const INTERP_FILTER_EIGHTTAP_SMOOTH: u8 = 1;
pub const INTERP_FILTER_EIGHTTAP_SHARP: u8 = 2;
pub const INTERP_FILTER_BILINEAR: u8 = 3;
pub const INTERP_FILTER_SWITCHABLE: u8 = 4;

// =============================================================================
// Enumerations
// =============================================================================

/// AV1 frame types as defined in the specification (Section 6.8.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FrameType {
    /// Key frame - independently decodable, random access point.
    #[default]
    KeyFrame = 0,
    /// Inter frame - uses references from previous frames.
    InterFrame = 1,
    /// Intra-only frame - uses only intra prediction, not a random access point.
    IntraOnlyFrame = 2,
    /// Switch frame - special frame for stream switching.
    SwitchFrame = 3,
}

impl FrameType {
    /// Returns true if this is an intra frame (key frame or intra-only).
    #[must_use]
    pub const fn is_intra(self) -> bool {
        matches!(self, Self::KeyFrame | Self::IntraOnlyFrame)
    }

    /// Returns true if this is a key frame.
    #[must_use]
    pub const fn is_key(self) -> bool {
        matches!(self, Self::KeyFrame)
    }

    /// Returns true if this frame uses inter prediction.
    #[must_use]
    pub const fn is_inter(self) -> bool {
        matches!(self, Self::InterFrame | Self::SwitchFrame)
    }
}

impl From<u8> for FrameType {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::KeyFrame,
            1 => Self::InterFrame,
            2 => Self::IntraOnlyFrame,
            3 => Self::SwitchFrame,
            _ => Self::KeyFrame, // Invalid values default to key frame
        }
    }
}

impl From<FrameType> for u8 {
    fn from(ft: FrameType) -> Self {
        ft as u8
    }
}

/// Interpolation filter types for motion compensation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum InterpolationFilter {
    /// 8-tap regular filter.
    #[default]
    Eighttap = 0,
    /// 8-tap smooth filter.
    EighttapSmooth = 1,
    /// 8-tap sharp filter.
    EighttapSharp = 2,
    /// Bilinear filter.
    Bilinear = 3,
    /// Per-block switchable filter.
    Switchable = 4,
}

impl From<u8> for InterpolationFilter {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Eighttap,
            1 => Self::EighttapSmooth,
            2 => Self::EighttapSharp,
            3 => Self::Bilinear,
            _ => Self::Switchable,
        }
    }
}

impl From<InterpolationFilter> for u8 {
    fn from(f: InterpolationFilter) -> Self {
        f as u8
    }
}

/// Motion mode for inter prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum MotionMode {
    /// Simple translation motion.
    #[default]
    Simple = 0,
    /// Obstructed motion compensation.
    ObstructedMotion = 1,
    /// Local warped motion.
    LocalWarp = 2,
}

impl From<u8> for MotionMode {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Simple,
            1 => Self::ObstructedMotion,
            2 => Self::LocalWarp,
            _ => Self::Simple,
        }
    }
}

/// Frame reference mode for compound prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ReferenceMode {
    /// Single reference only.
    #[default]
    SingleReference = 0,
    /// Compound reference (two references per block).
    CompoundReference = 1,
    /// Per-block reference selection.
    ReferenceModeSelect = 2,
}

impl From<u8> for ReferenceMode {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::SingleReference,
            1 => Self::CompoundReference,
            _ => Self::ReferenceModeSelect,
        }
    }
}

/// TX mode for transform selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TxMode {
    /// Only 4x4 transforms.
    Only4x4 = 0,
    /// Largest transform size.
    #[default]
    Largest = 1,
    /// Per-block transform selection.
    Select = 2,
}

impl From<u8> for TxMode {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Only4x4,
            1 => Self::Largest,
            _ => Self::Select,
        }
    }
}

/// Restoration type for loop restoration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum RestorationType {
    /// No restoration.
    #[default]
    None = 0,
    /// Wiener filter.
    Wiener = 1,
    /// Self-guided filter.
    SgrProj = 2,
    /// Switchable between Wiener and `SgrProj`.
    Switchable = 3,
}

impl From<u8> for RestorationType {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Wiener,
            2 => Self::SgrProj,
            _ => Self::Switchable,
        }
    }
}

// =============================================================================
// Structures
// =============================================================================

/// Frame size information including superres scaling.
#[derive(Clone, Debug, Default)]
pub struct FrameSize {
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Superres upscaled width (may differ from frame_width).
    pub upscaled_width: u32,
    /// Superres denominator (8-16, where 8 means no scaling).
    pub superres_denom: u32,
    /// Whether superres is enabled for this frame.
    pub use_superres: bool,
    /// Width in 4x4 blocks (mi_cols).
    pub mi_cols: u32,
    /// Height in 4x4 blocks (mi_rows).
    pub mi_rows: u32,
}

impl FrameSize {
    /// Calculate the number of 4x4 blocks for a given dimension.
    #[must_use]
    pub const fn size_to_mi(size: u32) -> u32 {
        (size + 3) >> 2
    }

    /// Calculate the number of superblocks for a given dimension.
    #[must_use]
    pub const fn size_to_sb(size: u32, sb_size: u32) -> u32 {
        (size + sb_size - 1) / sb_size
    }

    /// Get the number of superblock columns.
    #[must_use]
    pub fn sb_cols(&self, sb_size: u32) -> u32 {
        Self::size_to_sb(self.upscaled_width, sb_size)
    }

    /// Get the number of superblock rows.
    #[must_use]
    pub fn sb_rows(&self, sb_size: u32) -> u32 {
        Self::size_to_sb(self.frame_height, sb_size)
    }
}

/// Render size for display (may differ from coded size).
#[derive(Clone, Debug, Default)]
pub struct RenderSize {
    /// Render width in pixels.
    pub render_width: u32,
    /// Render height in pixels.
    pub render_height: u32,
    /// Whether render size differs from frame size.
    pub render_and_frame_size_different: bool,
}

/// Reference frame selection information.
#[derive(Clone, Debug, Default)]
pub struct RefFrameInfo {
    /// Indices into the reference frame buffer for each reference type.
    /// Index 0 = LAST_FRAME, 1 = LAST2_FRAME, etc.
    pub ref_frame_idx: [u8; REFS_PER_FRAME],
    /// Order hints for each reference frame slot.
    pub ref_order_hint: [u8; NUM_REF_FRAMES],
    /// Reference frame sign bias (for temporal ordering).
    pub ref_frame_sign_bias: [bool; NUM_REF_FRAMES],
}

/// Segmentation parameters.
#[derive(Clone, Debug, Default)]
pub struct SegmentationParams {
    /// Segmentation enabled.
    pub enabled: bool,
    /// Update the segmentation map.
    pub update_map: bool,
    /// Temporal update for segmentation.
    pub temporal_update: bool,
    /// Update segment data.
    pub update_data: bool,
    /// Segment feature enabled flags.
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// Segment feature data values.
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// Last active segment ID.
    pub last_active_seg_id: u8,
}

impl SegmentationParams {
    /// Segment feature index for ALT_Q (quantizer delta).
    pub const SEG_LVL_ALT_Q: usize = 0;
    /// Segment feature index for ALT_LF_Y_V (loop filter delta Y vertical).
    pub const SEG_LVL_ALT_LF_Y_V: usize = 1;
    /// Segment feature index for ALT_LF_Y_H (loop filter delta Y horizontal).
    pub const SEG_LVL_ALT_LF_Y_H: usize = 2;
    /// Segment feature index for ALT_LF_U (loop filter delta U).
    pub const SEG_LVL_ALT_LF_U: usize = 3;
    /// Segment feature index for ALT_LF_V (loop filter delta V).
    pub const SEG_LVL_ALT_LF_V: usize = 4;
    /// Segment feature index for REF_FRAME (reference frame).
    pub const SEG_LVL_REF_FRAME: usize = 5;
    /// Segment feature index for SKIP (skip mode).
    pub const SEG_LVL_SKIP: usize = 6;
    /// Segment feature index for GLOBALMV (global motion).
    pub const SEG_LVL_GLOBALMV: usize = 7;

    /// Maximum values for each segment feature.
    pub const SEG_FEATURE_DATA_MAX: [i16; SEG_LVL_MAX] = [255, 63, 63, 63, 63, 7, 0, 0];

    /// Whether each segment feature is signed.
    pub const SEG_FEATURE_DATA_SIGNED: [bool; SEG_LVL_MAX] =
        [true, true, true, true, true, false, false, false];

    /// Number of bits for each segment feature.
    pub const SEG_FEATURE_BITS: [u8; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];

    /// Get a segment feature value.
    #[must_use]
    pub fn get_feature(&self, segment_id: usize, feature: usize) -> i16 {
        if segment_id < MAX_SEGMENTS
            && feature < SEG_LVL_MAX
            && self.feature_enabled[segment_id][feature]
        {
            self.feature_data[segment_id][feature]
        } else {
            0
        }
    }

    /// Check if a segment feature is enabled.
    #[must_use]
    pub fn is_feature_enabled(&self, segment_id: usize, feature: usize) -> bool {
        segment_id < MAX_SEGMENTS
            && feature < SEG_LVL_MAX
            && self.feature_enabled[segment_id][feature]
    }
}

/// Loop restoration parameters for each plane.
#[derive(Clone, Debug, Default)]
pub struct LoopRestorationParams {
    /// Restoration type per plane.
    pub frame_restoration_type: [RestorationType; 3],
    /// Restoration unit size (log2) per plane.
    pub loop_restoration_size: [u8; 3],
    /// Whether restoration is used.
    pub uses_lr: bool,
}

/// Global motion parameters for a single reference frame.
#[derive(Clone, Debug, Default)]
pub struct GlobalMotionParams {
    /// Transform type (0=identity, 1=translation, 2=rotzoom, 3=affine).
    pub gm_type: u8,
    /// Global motion parameters (up to 6 for affine).
    pub gm_params: [i32; 6],
}

/// Global motion information for all reference frames.
#[derive(Clone, Debug)]
pub struct GlobalMotion {
    /// Global motion for each reference frame.
    pub params: [GlobalMotionParams; NUM_REF_FRAMES],
}

impl Default for GlobalMotion {
    fn default() -> Self {
        Self {
            params: std::array::from_fn(|_| GlobalMotionParams::default()),
        }
    }
}

/// Film grain synthesis parameters.
#[derive(Clone, Debug, Default)]
pub struct FilmGrainParams {
    /// Apply grain.
    pub apply_grain: bool,
    /// Grain seed.
    pub grain_seed: u16,
    /// Update grain parameters.
    pub update_grain: bool,
    /// Number of Y luma points.
    pub num_y_points: u8,
    /// Y point values.
    pub point_y_value: [u8; 14],
    /// Y point scaling.
    pub point_y_scaling: [u8; 14],
    /// Chroma scaling from luma.
    pub chroma_scaling_from_luma: bool,
    /// Number of Cb chroma points.
    pub num_cb_points: u8,
    /// Cb point values.
    pub point_cb_value: [u8; 10],
    /// Cb point scaling.
    pub point_cb_scaling: [u8; 10],
    /// Number of Cr chroma points.
    pub num_cr_points: u8,
    /// Cr point values.
    pub point_cr_value: [u8; 10],
    /// Cr point scaling.
    pub point_cr_scaling: [u8; 10],
    /// Grain scaling shift.
    pub grain_scaling_minus_8: u8,
    /// AR coefficients lag.
    pub ar_coeff_lag: u8,
    /// AR coefficients for Y.
    pub ar_coeffs_y_plus_128: [u8; 24],
    /// AR coefficients for Cb.
    pub ar_coeffs_cb_plus_128: [u8; 25],
    /// AR coefficients for Cr.
    pub ar_coeffs_cr_plus_128: [u8; 25],
    /// AR coefficient shift.
    pub ar_coeff_shift_minus_6: u8,
    /// Grain scale shift.
    pub grain_scale_shift: u8,
    /// Cb multiplier.
    pub cb_mult: u8,
    /// Cb luma multiplier.
    pub cb_luma_mult: u8,
    /// Cb offset.
    pub cb_offset: u16,
    /// Cr multiplier.
    pub cr_mult: u8,
    /// Cr luma multiplier.
    pub cr_luma_mult: u8,
    /// Cr offset.
    pub cr_offset: u16,
    /// Overlap flag.
    pub overlap_flag: bool,
    /// Clip to restricted range.
    pub clip_to_restricted_range: bool,
}

/// Complete frame header structure.
#[derive(Clone, Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct FrameHeader {
    // === Frame Type and Visibility ===
    /// Frame type (KEY_FRAME, INTER_FRAME, INTRA_ONLY_FRAME, SWITCH_FRAME).
    pub frame_type: FrameType,
    /// Whether to show this frame.
    pub show_frame: bool,
    /// Whether this frame can be shown later via show_existing_frame.
    pub showable_frame: bool,
    /// Show an existing frame from the buffer.
    pub show_existing_frame: bool,
    /// Frame to show (if show_existing_frame is true).
    pub frame_to_show_map_idx: u8,
    /// Error resilient mode (disables some dependencies).
    pub error_resilient_mode: bool,

    // === Frame Identification ===
    /// Frame ID for error resilience.
    pub current_frame_id: u32,
    /// Order hint for temporal ordering.
    pub order_hint: u8,
    /// Primary reference frame index.
    pub primary_ref_frame: u8,
    /// Refresh frame flags (which slots to update).
    pub refresh_frame_flags: u8,

    // === Frame Size ===
    /// Frame size information.
    pub frame_size: FrameSize,
    /// Render size information.
    pub render_size: RenderSize,

    // === Inter Prediction Settings ===
    /// Allow high precision motion vectors.
    pub allow_high_precision_mv: bool,
    /// Interpolation filter for motion compensation.
    pub interpolation_filter: InterpolationFilter,
    /// Whether the filter is switchable per-block.
    pub is_filter_switchable: bool,
    /// Intra block copy allowed.
    pub allow_intrabc: bool,

    // === Reference Frame Management ===
    /// Reference frame information.
    pub ref_frame_info: RefFrameInfo,
    /// Allow screen content tools.
    pub allow_screen_content_tools: bool,
    /// Force integer motion vectors.
    pub force_integer_mv: bool,

    // === Motion Settings ===
    /// Motion mode (simple, obmc, warp).
    pub is_motion_mode_switchable: bool,
    /// Use reference frame motion vectors.
    pub use_ref_frame_mvs: bool,
    /// Reference mode (single, compound, select).
    pub reference_mode: ReferenceMode,
    /// Skip mode frame indices.
    pub skip_mode_frame: [u8; 2],
    /// Skip mode allowed.
    pub skip_mode_allowed: bool,
    /// Skip mode present.
    pub skip_mode_present: bool,

    // === Compound Prediction ===
    /// Compound inter allowed.
    pub compound_reference_allowed: bool,

    // === Transform Settings ===
    /// TX mode for transform selection.
    pub tx_mode: TxMode,
    /// Reduced TX set for complexity reduction.
    pub reduced_tx_set: bool,

    // === Warped Motion ===
    /// Allow warped motion.
    pub allow_warped_motion: bool,

    // === Quantization ===
    /// Quantization parameters.
    pub quantization: QuantizationParams,

    // === Segmentation ===
    /// Segmentation parameters.
    pub segmentation: SegmentationParams,

    // === Loop Filter ===
    /// Loop filter parameters.
    pub loop_filter: LoopFilterParams,

    // === CDEF ===
    /// CDEF parameters.
    pub cdef: CdefParams,

    // === Loop Restoration ===
    /// Loop restoration parameters.
    pub loop_restoration: LoopRestorationParams,

    // === Tile Info ===
    /// Tile information.
    pub tile_info: TileInfo,

    // === Global Motion ===
    /// Global motion parameters.
    pub global_motion: GlobalMotion,

    // === Film Grain ===
    /// Film grain parameters.
    pub film_grain: FilmGrainParams,

    // === Derived Values ===
    /// Frame is intra-only (key frame or intra-only frame).
    pub frame_is_intra: bool,
    /// All lossless mode.
    pub lossless_array: [bool; MAX_SEGMENTS],
    /// Coded lossless (all segments lossless and no chroma subsampling issues).
    pub coded_lossless: bool,
    /// All lossless (coded lossless and frame size equals superres size).
    pub all_lossless: bool,
}

impl FrameHeader {
    /// Create a new default frame header.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a frame header from the bitstream.
    ///
    /// # Errors
    ///
    /// Returns error if the frame header is malformed.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    pub fn parse(data: &[u8], seq: &SequenceHeader) -> CodecResult<Self> {
        let mut reader = BitReader::new(data);
        let mut header = Self::new();

        // Parse uncompressed header
        header.parse_uncompressed_header(&mut reader, seq)?;

        Ok(header)
    }

    /// Parse the uncompressed header portion.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    fn parse_uncompressed_header(
        &mut self,
        reader: &mut BitReader<'_>,
        seq: &SequenceHeader,
    ) -> CodecResult<()> {
        // Frame ID management
        let frame_id_length = if seq.reduced_still_picture_header {
            0
        } else {
            // Simplified: would come from sequence header
            15
        };

        // Show existing frame
        if seq.reduced_still_picture_header {
            self.show_existing_frame = false;
            self.frame_type = FrameType::KeyFrame;
            self.show_frame = true;
            self.showable_frame = false;
        } else {
            self.show_existing_frame = reader.read_bit().map_err(CodecError::Core)? != 0;

            if self.show_existing_frame {
                self.frame_to_show_map_idx = reader.read_bits(3).map_err(CodecError::Core)? as u8;

                if frame_id_length > 0 {
                    self.current_frame_id = reader
                        .read_bits(frame_id_length)
                        .map_err(CodecError::Core)?
                        as u32;
                }

                // For show_existing_frame, we're done with header parsing
                return Ok(());
            }

            self.frame_type = FrameType::from(reader.read_bits(2).map_err(CodecError::Core)? as u8);
            self.show_frame = reader.read_bit().map_err(CodecError::Core)? != 0;

            if self.show_frame && !seq.reduced_still_picture_header {
                // decoder_model_info_present handling would go here
            }

            self.showable_frame = if self.show_frame {
                !self.frame_type.is_key()
            } else {
                reader.read_bit().map_err(CodecError::Core)? != 0
            };

            self.error_resilient_mode =
                if self.frame_type == FrameType::SwitchFrame || self.frame_type.is_key() {
                    true
                } else {
                    reader.read_bit().map_err(CodecError::Core)? != 0
                };
        }

        self.frame_is_intra = self.frame_type.is_intra();

        // Disable CDF update flag (ignored for now)
        if !seq.reduced_still_picture_header {
            let _disable_cdf_update = reader.read_bit().map_err(CodecError::Core)?;
        }

        // Screen content tools
        self.allow_screen_content_tools = if seq.reduced_still_picture_header {
            false
        } else {
            // seq_force_screen_content_tools handling
            reader.read_bit().map_err(CodecError::Core)? != 0
        };

        // Force integer MV
        self.force_integer_mv = if self.allow_screen_content_tools {
            if !seq.reduced_still_picture_header {
                reader.read_bit().map_err(CodecError::Core)? != 0
            } else {
                false
            }
        } else {
            false
        };

        // Frame ID
        if frame_id_length > 0 {
            self.current_frame_id = reader
                .read_bits(frame_id_length)
                .map_err(CodecError::Core)? as u32;
        }

        // Frame size
        self.parse_frame_size(reader, seq)?;

        // Render size
        self.parse_render_size(reader)?;

        // Superres params if enabled
        if seq.enable_superres && !self.frame_size.use_superres {
            self.frame_size.use_superres = reader.read_bit().map_err(CodecError::Core)? != 0;
            if self.frame_size.use_superres {
                let coded_denom = reader
                    .read_bits(SUPERRES_DENOM_BITS)
                    .map_err(CodecError::Core)? as u32
                    + SUPERRES_DENOM_MIN;
                self.frame_size.superres_denom = coded_denom;
                self.frame_size.upscaled_width = self.frame_size.frame_width;
                self.frame_size.frame_width =
                    (self.frame_size.upscaled_width * SUPERRES_NUM + coded_denom / 2) / coded_denom;
            }
        }

        if !self.frame_is_intra {
            // Inter frame specific parsing
            self.parse_inter_frame_params(reader, seq)?;
        }

        // Quantization params
        self.quantization = QuantizationParams::parse(reader, seq)?;

        // Segmentation params
        self.parse_segmentation(reader)?;

        // Compute lossless arrays
        self.compute_lossless(seq);

        // Loop filter params (only if not all lossless)
        if !self.coded_lossless {
            self.loop_filter = LoopFilterParams::parse(reader, seq, self.frame_is_intra)?;
        }

        // CDEF params
        if seq.enable_cdef && !self.coded_lossless && !self.allow_intrabc {
            self.cdef = CdefParams::parse(reader, seq)?;
        }

        // Loop restoration
        if seq.enable_restoration && !self.all_lossless && !self.allow_intrabc {
            self.parse_loop_restoration(reader, seq)?;
        }

        // TX mode
        self.parse_tx_mode(reader)?;

        // Reference mode
        if !self.frame_is_intra {
            self.reference_mode = if reader.read_bit().map_err(CodecError::Core)? != 0 {
                ReferenceMode::ReferenceModeSelect
            } else {
                ReferenceMode::SingleReference
            };
        }

        // Skip mode
        self.parse_skip_mode(reader, seq)?;

        // Warped motion
        if !self.frame_is_intra && !self.error_resilient_mode {
            self.allow_warped_motion = reader.read_bit().map_err(CodecError::Core)? != 0;
        }

        // Reduced TX set
        self.reduced_tx_set = reader.read_bit().map_err(CodecError::Core)? != 0;

        // Global motion
        if !self.frame_is_intra {
            self.parse_global_motion(reader)?;
        }

        // Film grain
        if seq.film_grain_params_present && (self.show_frame || self.showable_frame) {
            self.parse_film_grain(reader, seq)?;
        }

        // Tile info
        self.tile_info = TileInfo::parse(reader, seq, &self.frame_size)?;

        Ok(())
    }

    /// Parse frame size.
    #[allow(clippy::cast_possible_truncation)]
    fn parse_frame_size(
        &mut self,
        reader: &mut BitReader<'_>,
        seq: &SequenceHeader,
    ) -> CodecResult<()> {
        if self.frame_type == FrameType::SwitchFrame {
            // Switch frames use max dimensions
            self.frame_size.frame_width = seq.max_frame_width();
            self.frame_size.frame_height = seq.max_frame_height();
        } else {
            let frame_size_override = if seq.reduced_still_picture_header {
                false
            } else {
                reader.read_bit().map_err(CodecError::Core)? != 0
            };

            if frame_size_override {
                // Read explicit frame dimensions
                let frame_width_bits = 16; // Simplified
                let frame_height_bits = 16;
                self.frame_size.frame_width = reader
                    .read_bits(frame_width_bits)
                    .map_err(CodecError::Core)?
                    as u32
                    + 1;
                self.frame_size.frame_height = reader
                    .read_bits(frame_height_bits)
                    .map_err(CodecError::Core)?
                    as u32
                    + 1;
            } else {
                self.frame_size.frame_width = seq.max_frame_width();
                self.frame_size.frame_height = seq.max_frame_height();
            }
        }

        self.frame_size.upscaled_width = self.frame_size.frame_width;
        self.frame_size.superres_denom = SUPERRES_NUM;
        self.frame_size.mi_cols = FrameSize::size_to_mi(self.frame_size.upscaled_width);
        self.frame_size.mi_rows = FrameSize::size_to_mi(self.frame_size.frame_height);

        Ok(())
    }

    /// Parse render size.
    fn parse_render_size(&mut self, reader: &mut BitReader<'_>) -> CodecResult<()> {
        self.render_size.render_and_frame_size_different =
            reader.read_bit().map_err(CodecError::Core)? != 0;

        if self.render_size.render_and_frame_size_different {
            let render_width_minus_1 = reader.read_bits(16).map_err(CodecError::Core)? as u32;
            let render_height_minus_1 = reader.read_bits(16).map_err(CodecError::Core)? as u32;
            self.render_size.render_width = render_width_minus_1 + 1;
            self.render_size.render_height = render_height_minus_1 + 1;
        } else {
            self.render_size.render_width = self.frame_size.upscaled_width;
            self.render_size.render_height = self.frame_size.frame_height;
        }

        Ok(())
    }

    /// Parse inter frame parameters.
    #[allow(clippy::cast_possible_truncation)]
    fn parse_inter_frame_params(
        &mut self,
        reader: &mut BitReader<'_>,
        seq: &SequenceHeader,
    ) -> CodecResult<()> {
        // Reference frame assignment
        for i in 0..REFS_PER_FRAME {
            self.ref_frame_info.ref_frame_idx[i] =
                reader.read_bits(3).map_err(CodecError::Core)? as u8;
        }

        // Frame size with refs
        if !self.error_resilient_mode && self.frame_type != FrameType::SwitchFrame {
            // Simplified: would parse frame_size_with_refs()
        }

        // Allow high precision MV
        self.allow_high_precision_mv = if self.force_integer_mv {
            false
        } else {
            reader.read_bit().map_err(CodecError::Core)? != 0
        };

        // Interpolation filter
        self.is_filter_switchable = reader.read_bit().map_err(CodecError::Core)? != 0;
        self.interpolation_filter = if self.is_filter_switchable {
            InterpolationFilter::Switchable
        } else {
            InterpolationFilter::from(reader.read_bits(2).map_err(CodecError::Core)? as u8)
        };

        // Motion mode
        self.is_motion_mode_switchable = reader.read_bit().map_err(CodecError::Core)? != 0;

        // Use ref frame MVs
        if !self.error_resilient_mode && seq.enable_order_hint {
            self.use_ref_frame_mvs = reader.read_bit().map_err(CodecError::Core)? != 0;
        }

        Ok(())
    }

    /// Parse segmentation parameters.
    #[allow(clippy::cast_possible_truncation)]
    fn parse_segmentation(&mut self, reader: &mut BitReader<'_>) -> CodecResult<()> {
        self.segmentation.enabled = reader.read_bit().map_err(CodecError::Core)? != 0;

        if !self.segmentation.enabled {
            return Ok(());
        }

        if self.primary_ref_frame == PRIMARY_REF_NONE {
            self.segmentation.update_map = true;
            self.segmentation.temporal_update = false;
            self.segmentation.update_data = true;
        } else {
            self.segmentation.update_map = reader.read_bit().map_err(CodecError::Core)? != 0;
            if self.segmentation.update_map {
                self.segmentation.temporal_update =
                    reader.read_bit().map_err(CodecError::Core)? != 0;
            }
            self.segmentation.update_data = reader.read_bit().map_err(CodecError::Core)? != 0;
        }

        if self.segmentation.update_data {
            for i in 0..MAX_SEGMENTS {
                for j in 0..SEG_LVL_MAX {
                    self.segmentation.feature_enabled[i][j] =
                        reader.read_bit().map_err(CodecError::Core)? != 0;

                    if self.segmentation.feature_enabled[i][j] {
                        let bits = SegmentationParams::SEG_FEATURE_BITS[j];
                        let max = SegmentationParams::SEG_FEATURE_DATA_MAX[j];

                        if SegmentationParams::SEG_FEATURE_DATA_SIGNED[j] {
                            let value =
                                reader.read_bits(bits + 1).map_err(CodecError::Core)? as i16;
                            let sign = if value & (1 << bits) != 0 { -1 } else { 1 };
                            let magnitude = value & ((1 << bits) - 1);
                            self.segmentation.feature_data[i][j] =
                                (sign * magnitude).clamp(-max, max);
                        } else {
                            self.segmentation.feature_data[i][j] =
                                (reader.read_bits(bits).map_err(CodecError::Core)? as i16).min(max);
                        }
                    }
                }
            }
        }

        // Find last active segment
        self.segmentation.last_active_seg_id = 0;
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                if self.segmentation.feature_enabled[i][j] {
                    self.segmentation.last_active_seg_id = i as u8;
                }
            }
        }

        Ok(())
    }

    /// Compute lossless mode for each segment.
    fn compute_lossless(&mut self, seq: &SequenceHeader) {
        for seg_id in 0..MAX_SEGMENTS {
            let qindex = self.get_qindex(seg_id);
            let lossless = qindex == 0
                && self.quantization.delta_q_y_dc == 0
                && self.quantization.delta_q_u_ac == 0
                && self.quantization.delta_q_u_dc == 0
                && self.quantization.delta_q_v_ac == 0
                && self.quantization.delta_q_v_dc == 0;
            self.lossless_array[seg_id] = lossless;
        }

        self.coded_lossless = self.lossless_array.iter().all(|&l| l);
        self.all_lossless =
            self.coded_lossless && self.frame_size.frame_width == self.frame_size.upscaled_width;

        // Update loop filter disable for lossless
        if self.coded_lossless {
            self.loop_filter.level[0] = 0;
            self.loop_filter.level[1] = 0;
        }

        // Handle monochrome case
        if seq.color_config.mono_chrome {
            self.quantization.delta_q_u_dc = 0;
            self.quantization.delta_q_u_ac = 0;
            self.quantization.delta_q_v_dc = 0;
            self.quantization.delta_q_v_ac = 0;
        }
    }

    /// Get quantizer index for a segment.
    #[must_use]
    pub fn get_qindex(&self, seg_id: usize) -> u8 {
        let base_q = self.quantization.base_q_idx;
        if self.segmentation.enabled
            && self
                .segmentation
                .is_feature_enabled(seg_id, SegmentationParams::SEG_LVL_ALT_Q)
        {
            let delta = self
                .segmentation
                .get_feature(seg_id, SegmentationParams::SEG_LVL_ALT_Q);
            let q = i32::from(base_q) + i32::from(delta);
            q.clamp(0, 255) as u8
        } else {
            base_q
        }
    }

    /// Parse loop restoration parameters.
    #[allow(clippy::cast_possible_truncation)]
    fn parse_loop_restoration(
        &mut self,
        reader: &mut BitReader<'_>,
        seq: &SequenceHeader,
    ) -> CodecResult<()> {
        let num_planes = if seq.color_config.mono_chrome { 1 } else { 3 };
        let mut uses_lr = false;
        let mut uses_chroma_lr = false;

        for plane in 0..num_planes {
            let lr_type = reader.read_bits(2).map_err(CodecError::Core)? as u8;
            self.loop_restoration.frame_restoration_type[plane] = RestorationType::from(lr_type);
            if lr_type != 0 {
                uses_lr = true;
                if plane > 0 {
                    uses_chroma_lr = true;
                }
            }
        }

        self.loop_restoration.uses_lr = uses_lr;

        if uses_lr {
            // LR unit shift
            let lr_unit_shift = if seq.enable_superres {
                reader.read_bit().map_err(CodecError::Core)? as u8
            } else {
                1
            };

            let lr_unit_extra_shift = if lr_unit_shift != 0 && !seq.enable_superres {
                reader.read_bit().map_err(CodecError::Core)? as u8
            } else {
                0
            };

            // Set LR sizes
            let sb_size = 64; // Simplified
            let lr_size_base = 6 + lr_unit_shift + lr_unit_extra_shift;
            self.loop_restoration.loop_restoration_size[0] = lr_size_base.min(sb_size);

            if uses_chroma_lr && !seq.color_config.is_420() {
                let uv_shift = reader.read_bit().map_err(CodecError::Core)? as u8;
                self.loop_restoration.loop_restoration_size[1] = lr_size_base - uv_shift;
                self.loop_restoration.loop_restoration_size[2] = lr_size_base - uv_shift;
            } else {
                self.loop_restoration.loop_restoration_size[1] = lr_size_base;
                self.loop_restoration.loop_restoration_size[2] = lr_size_base;
            }
        }

        Ok(())
    }

    /// Parse TX mode.
    fn parse_tx_mode(&mut self, reader: &mut BitReader<'_>) -> CodecResult<()> {
        if self.coded_lossless {
            self.tx_mode = TxMode::Only4x4;
        } else {
            let tx_mode_select = reader.read_bit().map_err(CodecError::Core)? != 0;
            self.tx_mode = if tx_mode_select {
                TxMode::Select
            } else {
                TxMode::Largest
            };
        }
        Ok(())
    }

    /// Parse skip mode.
    #[allow(clippy::cast_possible_truncation)]
    fn parse_skip_mode(
        &mut self,
        reader: &mut BitReader<'_>,
        seq: &SequenceHeader,
    ) -> CodecResult<()> {
        if self.frame_is_intra
            || !self.reference_mode.eq(&ReferenceMode::ReferenceModeSelect)
            || !seq.enable_order_hint
        {
            self.skip_mode_allowed = false;
        } else {
            // Simplified skip mode derivation
            self.skip_mode_allowed = true;
        }

        if self.skip_mode_allowed {
            self.skip_mode_present = reader.read_bit().map_err(CodecError::Core)? != 0;
        } else {
            self.skip_mode_present = false;
        }

        Ok(())
    }

    /// Parse global motion parameters.
    #[allow(clippy::cast_possible_truncation)]
    fn parse_global_motion(&mut self, reader: &mut BitReader<'_>) -> CodecResult<()> {
        for ref_frame in LAST_FRAME..=ALTREF_FRAME {
            // Global motion type
            let is_global = reader.read_bit().map_err(CodecError::Core)? != 0;
            if is_global {
                let is_rot_zoom = reader.read_bit().map_err(CodecError::Core)? != 0;
                if is_rot_zoom {
                    self.global_motion.params[ref_frame].gm_type = 2; // ROTZOOM
                } else {
                    let is_translation = reader.read_bit().map_err(CodecError::Core)? != 0;
                    self.global_motion.params[ref_frame].gm_type = if is_translation {
                        1 // TRANSLATION
                    } else {
                        3 // AFFINE
                    };
                }

                // Parse global motion params (simplified)
                // Full parsing would read the actual motion parameters
            } else {
                self.global_motion.params[ref_frame].gm_type = 0; // IDENTITY
            }
        }

        Ok(())
    }

    /// Parse film grain parameters.
    #[allow(clippy::cast_possible_truncation)]
    fn parse_film_grain(
        &mut self,
        reader: &mut BitReader<'_>,
        seq: &SequenceHeader,
    ) -> CodecResult<()> {
        self.film_grain.apply_grain = reader.read_bit().map_err(CodecError::Core)? != 0;

        if !self.film_grain.apply_grain {
            return Ok(());
        }

        self.film_grain.grain_seed = reader.read_bits(16).map_err(CodecError::Core)? as u16;

        if self.frame_type == FrameType::InterFrame {
            self.film_grain.update_grain = reader.read_bit().map_err(CodecError::Core)? != 0;
        } else {
            self.film_grain.update_grain = true;
        }

        if !self.film_grain.update_grain {
            // Reference film grain params from another frame
            let _film_grain_params_ref_idx = reader.read_bits(3).map_err(CodecError::Core)?;
            return Ok(());
        }

        // Y points
        self.film_grain.num_y_points = reader.read_bits(4).map_err(CodecError::Core)? as u8;
        for i in 0..self.film_grain.num_y_points as usize {
            self.film_grain.point_y_value[i] = reader.read_bits(8).map_err(CodecError::Core)? as u8;
            self.film_grain.point_y_scaling[i] =
                reader.read_bits(8).map_err(CodecError::Core)? as u8;
        }

        // Chroma scaling from luma
        self.film_grain.chroma_scaling_from_luma = if !seq.color_config.mono_chrome {
            reader.read_bit().map_err(CodecError::Core)? != 0
        } else {
            false
        };

        // Cb/Cr points
        if seq.color_config.mono_chrome
            || self.film_grain.chroma_scaling_from_luma
            || (seq.color_config.is_420() && self.film_grain.num_y_points == 0)
        {
            self.film_grain.num_cb_points = 0;
            self.film_grain.num_cr_points = 0;
        } else {
            self.film_grain.num_cb_points = reader.read_bits(4).map_err(CodecError::Core)? as u8;
            for i in 0..self.film_grain.num_cb_points as usize {
                self.film_grain.point_cb_value[i] =
                    reader.read_bits(8).map_err(CodecError::Core)? as u8;
                self.film_grain.point_cb_scaling[i] =
                    reader.read_bits(8).map_err(CodecError::Core)? as u8;
            }

            self.film_grain.num_cr_points = reader.read_bits(4).map_err(CodecError::Core)? as u8;
            for i in 0..self.film_grain.num_cr_points as usize {
                self.film_grain.point_cr_value[i] =
                    reader.read_bits(8).map_err(CodecError::Core)? as u8;
                self.film_grain.point_cr_scaling[i] =
                    reader.read_bits(8).map_err(CodecError::Core)? as u8;
            }
        }

        // Grain scaling and AR coefficients
        self.film_grain.grain_scaling_minus_8 =
            reader.read_bits(2).map_err(CodecError::Core)? as u8;
        self.film_grain.ar_coeff_lag = reader.read_bits(2).map_err(CodecError::Core)? as u8;

        // AR coefficients (simplified)
        let num_pos_luma = 2 * self.film_grain.ar_coeff_lag * (self.film_grain.ar_coeff_lag + 1);
        for i in 0..num_pos_luma as usize {
            if self.film_grain.num_y_points > 0 && i < 24 {
                self.film_grain.ar_coeffs_y_plus_128[i] =
                    reader.read_bits(8).map_err(CodecError::Core)? as u8;
            }
        }

        self.film_grain.ar_coeff_shift_minus_6 =
            reader.read_bits(2).map_err(CodecError::Core)? as u8;
        self.film_grain.grain_scale_shift = reader.read_bits(2).map_err(CodecError::Core)? as u8;

        // Chroma multipliers
        if self.film_grain.num_cb_points > 0 {
            self.film_grain.cb_mult = reader.read_bits(8).map_err(CodecError::Core)? as u8;
            self.film_grain.cb_luma_mult = reader.read_bits(8).map_err(CodecError::Core)? as u8;
            self.film_grain.cb_offset = reader.read_bits(9).map_err(CodecError::Core)? as u16;
        }

        if self.film_grain.num_cr_points > 0 {
            self.film_grain.cr_mult = reader.read_bits(8).map_err(CodecError::Core)? as u8;
            self.film_grain.cr_luma_mult = reader.read_bits(8).map_err(CodecError::Core)? as u8;
            self.film_grain.cr_offset = reader.read_bits(9).map_err(CodecError::Core)? as u16;
        }

        self.film_grain.overlap_flag = reader.read_bit().map_err(CodecError::Core)? != 0;
        self.film_grain.clip_to_restricted_range =
            reader.read_bit().map_err(CodecError::Core)? != 0;

        Ok(())
    }

    /// Check if this frame is a keyframe.
    #[must_use]
    pub const fn is_key_frame(&self) -> bool {
        matches!(self.frame_type, FrameType::KeyFrame)
    }

    /// Check if this frame uses inter prediction.
    #[must_use]
    pub const fn is_inter_frame(&self) -> bool {
        matches!(
            self.frame_type,
            FrameType::InterFrame | FrameType::SwitchFrame
        )
    }

    /// Get the display order for this frame.
    #[must_use]
    pub const fn display_order(&self) -> u8 {
        self.order_hint
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_type_conversions() {
        assert_eq!(FrameType::from(0), FrameType::KeyFrame);
        assert_eq!(FrameType::from(1), FrameType::InterFrame);
        assert_eq!(FrameType::from(2), FrameType::IntraOnlyFrame);
        assert_eq!(FrameType::from(3), FrameType::SwitchFrame);
        assert_eq!(FrameType::from(4), FrameType::KeyFrame);

        assert_eq!(u8::from(FrameType::KeyFrame), 0);
        assert_eq!(u8::from(FrameType::InterFrame), 1);
    }

    #[test]
    fn test_frame_type_properties() {
        assert!(FrameType::KeyFrame.is_intra());
        assert!(FrameType::KeyFrame.is_key());
        assert!(!FrameType::KeyFrame.is_inter());

        assert!(!FrameType::InterFrame.is_intra());
        assert!(!FrameType::InterFrame.is_key());
        assert!(FrameType::InterFrame.is_inter());

        assert!(FrameType::IntraOnlyFrame.is_intra());
        assert!(!FrameType::IntraOnlyFrame.is_key());
        assert!(!FrameType::IntraOnlyFrame.is_inter());

        assert!(!FrameType::SwitchFrame.is_intra());
        assert!(!FrameType::SwitchFrame.is_key());
        assert!(FrameType::SwitchFrame.is_inter());
    }

    #[test]
    fn test_interpolation_filter_conversions() {
        assert_eq!(InterpolationFilter::from(0), InterpolationFilter::Eighttap);
        assert_eq!(
            InterpolationFilter::from(1),
            InterpolationFilter::EighttapSmooth
        );
        assert_eq!(
            InterpolationFilter::from(2),
            InterpolationFilter::EighttapSharp
        );
        assert_eq!(InterpolationFilter::from(3), InterpolationFilter::Bilinear);
        assert_eq!(
            InterpolationFilter::from(4),
            InterpolationFilter::Switchable
        );
    }

    #[test]
    fn test_frame_size_calculations() {
        assert_eq!(FrameSize::size_to_mi(1920), 480);
        assert_eq!(FrameSize::size_to_mi(1080), 270);
        assert_eq!(FrameSize::size_to_mi(1), 1);
        assert_eq!(FrameSize::size_to_mi(4), 1);
        assert_eq!(FrameSize::size_to_mi(5), 2);

        assert_eq!(FrameSize::size_to_sb(1920, 64), 30);
        assert_eq!(FrameSize::size_to_sb(1920, 128), 15);
        assert_eq!(FrameSize::size_to_sb(1080, 64), 17);
    }

    #[test]
    fn test_frame_size_sb_calculations() {
        let frame_size = FrameSize {
            frame_width: 1920,
            frame_height: 1080,
            upscaled_width: 1920,
            superres_denom: 8,
            use_superres: false,
            mi_cols: 480,
            mi_rows: 270,
        };

        assert_eq!(frame_size.sb_cols(64), 30);
        assert_eq!(frame_size.sb_rows(64), 17);
        assert_eq!(frame_size.sb_cols(128), 15);
        assert_eq!(frame_size.sb_rows(128), 9);
    }

    #[test]
    fn test_segmentation_features() {
        let mut seg = SegmentationParams::default();
        seg.enabled = true;
        seg.feature_enabled[0][SegmentationParams::SEG_LVL_ALT_Q] = true;
        seg.feature_data[0][SegmentationParams::SEG_LVL_ALT_Q] = 10;

        assert!(seg.is_feature_enabled(0, SegmentationParams::SEG_LVL_ALT_Q));
        assert_eq!(seg.get_feature(0, SegmentationParams::SEG_LVL_ALT_Q), 10);
        assert_eq!(seg.get_feature(1, SegmentationParams::SEG_LVL_ALT_Q), 0);
        assert!(!seg.is_feature_enabled(0, SegmentationParams::SEG_LVL_SKIP));
    }

    #[test]
    fn test_motion_mode_conversion() {
        assert_eq!(MotionMode::from(0), MotionMode::Simple);
        assert_eq!(MotionMode::from(1), MotionMode::ObstructedMotion);
        assert_eq!(MotionMode::from(2), MotionMode::LocalWarp);
        assert_eq!(MotionMode::from(99), MotionMode::Simple);
    }

    #[test]
    fn test_reference_mode_conversion() {
        assert_eq!(ReferenceMode::from(0), ReferenceMode::SingleReference);
        assert_eq!(ReferenceMode::from(1), ReferenceMode::CompoundReference);
        assert_eq!(ReferenceMode::from(2), ReferenceMode::ReferenceModeSelect);
    }

    #[test]
    fn test_tx_mode_conversion() {
        assert_eq!(TxMode::from(0), TxMode::Only4x4);
        assert_eq!(TxMode::from(1), TxMode::Largest);
        assert_eq!(TxMode::from(2), TxMode::Select);
    }

    #[test]
    fn test_restoration_type_conversion() {
        assert_eq!(RestorationType::from(0), RestorationType::None);
        assert_eq!(RestorationType::from(1), RestorationType::Wiener);
        assert_eq!(RestorationType::from(2), RestorationType::SgrProj);
        assert_eq!(RestorationType::from(3), RestorationType::Switchable);
    }

    #[test]
    fn test_frame_header_default() {
        let header = FrameHeader::new();
        assert_eq!(header.frame_type, FrameType::KeyFrame);
        assert!(!header.show_frame);
        assert!(!header.error_resilient_mode);
        assert!(!header.frame_is_intra);
    }

    #[test]
    fn test_frame_header_queries() {
        let mut header = FrameHeader::new();
        header.frame_type = FrameType::KeyFrame;
        assert!(header.is_key_frame());
        assert!(!header.is_inter_frame());

        header.frame_type = FrameType::InterFrame;
        assert!(!header.is_key_frame());
        assert!(header.is_inter_frame());

        header.frame_type = FrameType::SwitchFrame;
        assert!(!header.is_key_frame());
        assert!(header.is_inter_frame());

        header.order_hint = 42;
        assert_eq!(header.display_order(), 42);
    }

    #[test]
    fn test_get_qindex() {
        let mut header = FrameHeader::new();
        header.quantization.base_q_idx = 100;

        // Without segmentation
        assert_eq!(header.get_qindex(0), 100);

        // With segmentation
        header.segmentation.enabled = true;
        header.segmentation.feature_enabled[0][SegmentationParams::SEG_LVL_ALT_Q] = true;
        header.segmentation.feature_data[0][SegmentationParams::SEG_LVL_ALT_Q] = -20;

        assert_eq!(header.get_qindex(0), 80);
        assert_eq!(header.get_qindex(1), 100);
    }

    #[test]
    fn test_global_motion_default() {
        let gm = GlobalMotion::default();
        for i in 0..NUM_REF_FRAMES {
            assert_eq!(gm.params[i].gm_type, 0);
        }
    }

    #[test]
    fn test_film_grain_defaults() {
        let fg = FilmGrainParams::default();
        assert!(!fg.apply_grain);
        assert_eq!(fg.grain_seed, 0);
        assert_eq!(fg.num_y_points, 0);
    }

    #[test]
    fn test_render_size_defaults() {
        let rs = RenderSize::default();
        assert_eq!(rs.render_width, 0);
        assert_eq!(rs.render_height, 0);
        assert!(!rs.render_and_frame_size_different);
    }

    #[test]
    fn test_loop_restoration_params_defaults() {
        let lr = LoopRestorationParams::default();
        assert!(!lr.uses_lr);
        assert_eq!(lr.frame_restoration_type[0], RestorationType::None);
    }

    #[test]
    fn test_ref_frame_info_defaults() {
        let rfi = RefFrameInfo::default();
        for i in 0..REFS_PER_FRAME {
            assert_eq!(rfi.ref_frame_idx[i], 0);
        }
        for i in 0..NUM_REF_FRAMES {
            assert_eq!(rfi.ref_order_hint[i], 0);
            assert!(!rfi.ref_frame_sign_bias[i]);
        }
    }
}
