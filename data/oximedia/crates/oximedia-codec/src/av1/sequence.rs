//! AV1 Sequence Header parsing and SVC temporal scalability.
//!
//! The Sequence Header OBU contains codec-level configuration.
//! This module also provides temporal scalability (SVC) layer support
//! as defined in AV1 Annex A for scalable encoding.

use crate::error::{CodecError, CodecResult};
use oximedia_io::BitReader;

/// AV1 Sequence Header OBU.
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct SequenceHeader {
    /// Profile (0=Main, 1=High, 2=Professional).
    pub profile: u8,
    /// Still picture mode.
    pub still_picture: bool,
    /// Reduced still picture header.
    pub reduced_still_picture_header: bool,
    /// Maximum frame width minus 1.
    pub max_frame_width_minus_1: u32,
    /// Maximum frame height minus 1.
    pub max_frame_height_minus_1: u32,
    /// Enable order hint.
    pub enable_order_hint: bool,
    /// Order hint bits minus 1.
    pub order_hint_bits: u8,
    /// Enable superres.
    pub enable_superres: bool,
    /// Enable CDEF.
    pub enable_cdef: bool,
    /// Enable restoration.
    pub enable_restoration: bool,
    /// Color configuration.
    pub color_config: ColorConfig,
    /// Film grain params present.
    pub film_grain_params_present: bool,
}

impl SequenceHeader {
    /// Get maximum frame width.
    #[must_use]
    pub const fn max_frame_width(&self) -> u32 {
        self.max_frame_width_minus_1 + 1
    }

    /// Get maximum frame height.
    #[must_use]
    pub const fn max_frame_height(&self) -> u32 {
        self.max_frame_height_minus_1 + 1
    }

    /// Parse sequence header from OBU payload.
    ///
    /// # Errors
    ///
    /// Returns error if the header is malformed.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    pub fn parse(data: &[u8]) -> CodecResult<Self> {
        let mut reader = BitReader::new(data);

        let profile = reader.read_bits(3).map_err(CodecError::Core)? as u8;
        if profile > 2 {
            return Err(CodecError::InvalidBitstream(format!(
                "Invalid profile: {profile}"
            )));
        }

        let still_picture = reader.read_bit().map_err(CodecError::Core)? != 0;
        let reduced_still_picture_header = reader.read_bit().map_err(CodecError::Core)? != 0;

        if reduced_still_picture_header && !still_picture {
            return Err(CodecError::InvalidBitstream(
                "reduced_still_picture_header requires still_picture".to_string(),
            ));
        }

        // Skip timing info and operating points for simplified parsing
        if reduced_still_picture_header {
            reader.read_bits(5).map_err(CodecError::Core)?; // seq_level_idx[0]
        } else {
            // timing_info_present_flag
            let timing_info_present = reader.read_bit().map_err(CodecError::Core)? != 0;
            if timing_info_present {
                reader.skip_bits(64); // Simplified: skip timing info
                let decoder_model_info_present = reader.read_bit().map_err(CodecError::Core)? != 0;
                if decoder_model_info_present {
                    reader.skip_bits(47); // Simplified: skip decoder model info
                }
            }
            // initial_display_delay_present_flag
            reader.read_bit().map_err(CodecError::Core)?;
            // operating_points_cnt_minus_1
            let op_count = reader.read_bits(5).map_err(CodecError::Core)? as usize + 1;
            for _ in 0..op_count {
                reader.skip_bits(12); // operating_point_idc
                let level = reader.read_bits(5).map_err(CodecError::Core)? as u8;
                if level > 7 {
                    reader.skip_bits(1); // seq_tier
                }
            }
        }

        let frame_width_bits = reader.read_bits(4).map_err(CodecError::Core)? as u8 + 1;
        let frame_height_bits = reader.read_bits(4).map_err(CodecError::Core)? as u8 + 1;
        let max_frame_width_minus_1 = reader
            .read_bits(frame_width_bits)
            .map_err(CodecError::Core)? as u32;
        let max_frame_height_minus_1 = reader
            .read_bits(frame_height_bits)
            .map_err(CodecError::Core)? as u32;

        let enable_order_hint;
        let order_hint_bits;
        let enable_superres;
        let enable_cdef;
        let enable_restoration;

        if reduced_still_picture_header {
            enable_order_hint = false;
            order_hint_bits = 0;
            enable_superres = false;
            enable_cdef = false;
            enable_restoration = false;
        } else {
            // frame_id_numbers_present_flag
            let frame_id_present = reader.read_bit().map_err(CodecError::Core)? != 0;
            if frame_id_present {
                reader.skip_bits(7); // delta_frame_id_length_minus_2 + additional_frame_id_length_minus_1
            }

            // Tool enables
            reader.skip_bits(7); // Various tool flags

            enable_order_hint = reader.read_bit().map_err(CodecError::Core)? != 0;

            if enable_order_hint {
                reader.skip_bits(2); // enable_jnt_comp + enable_ref_frame_mvs
            }

            // seq_choose_screen_content_tools
            let seq_choose_screen_content_tools = reader.read_bit().map_err(CodecError::Core)? != 0;
            if !seq_choose_screen_content_tools {
                reader.skip_bits(1);
            }

            // seq_choose_integer_mv
            let seq_choose_integer_mv = reader.read_bit().map_err(CodecError::Core)? != 0;
            if !seq_choose_integer_mv {
                reader.skip_bits(1);
            }

            order_hint_bits = if enable_order_hint {
                reader.read_bits(3).map_err(CodecError::Core)? as u8 + 1
            } else {
                0
            };

            enable_superres = reader.read_bit().map_err(CodecError::Core)? != 0;
            enable_cdef = reader.read_bit().map_err(CodecError::Core)? != 0;
            enable_restoration = reader.read_bit().map_err(CodecError::Core)? != 0;
        }

        let color_config = ColorConfig::parse(&mut reader, profile)?;
        let film_grain_params_present = reader.read_bit().map_err(CodecError::Core)? != 0;

        Ok(Self {
            profile,
            still_picture,
            reduced_still_picture_header,
            max_frame_width_minus_1,
            max_frame_height_minus_1,
            enable_order_hint,
            order_hint_bits,
            enable_superres,
            enable_cdef,
            enable_restoration,
            color_config,
            film_grain_params_present,
        })
    }
}

/// Color configuration.
#[derive(Clone, Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct ColorConfig {
    /// Bit depth (8, 10, or 12).
    pub bit_depth: u8,
    /// Monochrome mode.
    pub mono_chrome: bool,
    /// Number of planes.
    pub num_planes: u8,
    /// Color primaries.
    pub color_primaries: u8,
    /// Transfer characteristics.
    pub transfer_characteristics: u8,
    /// Matrix coefficients.
    pub matrix_coefficients: u8,
    /// Full color range.
    pub color_range: bool,
    /// Subsampling X.
    pub subsampling_x: bool,
    /// Subsampling Y.
    pub subsampling_y: bool,
    /// Separate UV delta Q.
    pub separate_uv_delta_q: bool,
}

impl ColorConfig {
    /// Check if this is 4:2:0 subsampling.
    #[must_use]
    pub const fn is_420(&self) -> bool {
        self.subsampling_x && self.subsampling_y
    }

    /// Parse color config from bitstream.
    #[allow(clippy::cast_possible_truncation)]
    fn parse(reader: &mut BitReader<'_>, profile: u8) -> CodecResult<Self> {
        let high_bitdepth = reader.read_bit().map_err(CodecError::Core)? != 0;

        let twelve_bit = if profile == 2 && high_bitdepth {
            reader.read_bit().map_err(CodecError::Core)? != 0
        } else {
            false
        };

        let bit_depth = if profile == 2 && twelve_bit {
            12
        } else if high_bitdepth {
            10
        } else {
            8
        };

        let mono_chrome = if profile == 1 {
            false
        } else {
            reader.read_bit().map_err(CodecError::Core)? != 0
        };

        let num_planes = if mono_chrome { 1 } else { 3 };

        let color_description_present = reader.read_bit().map_err(CodecError::Core)? != 0;

        let (color_primaries, transfer_characteristics, matrix_coefficients) =
            if color_description_present {
                let cp = reader.read_bits(8).map_err(CodecError::Core)? as u8;
                let tc = reader.read_bits(8).map_err(CodecError::Core)? as u8;
                let mc = reader.read_bits(8).map_err(CodecError::Core)? as u8;
                (cp, tc, mc)
            } else {
                (2, 2, 2)
            };

        let color_range;
        let subsampling_x;
        let subsampling_y;

        if mono_chrome {
            color_range = reader.read_bit().map_err(CodecError::Core)? != 0;
            subsampling_x = true;
            subsampling_y = true;
        } else if color_primaries == 1 && transfer_characteristics == 13 && matrix_coefficients == 0
        {
            color_range = true;
            subsampling_x = false;
            subsampling_y = false;
        } else {
            color_range = reader.read_bit().map_err(CodecError::Core)? != 0;

            if profile == 0 {
                subsampling_x = true;
                subsampling_y = true;
            } else if profile == 1 {
                subsampling_x = false;
                subsampling_y = false;
            } else if bit_depth == 12 {
                subsampling_x = reader.read_bit().map_err(CodecError::Core)? != 0;
                subsampling_y = if subsampling_x {
                    reader.read_bit().map_err(CodecError::Core)? != 0
                } else {
                    false
                };
            } else {
                subsampling_x = true;
                subsampling_y = false;
            }

            if subsampling_x && subsampling_y {
                reader.skip_bits(2); // chroma_sample_position
            }
        }

        let separate_uv_delta_q = if mono_chrome {
            false
        } else {
            reader.read_bit().map_err(CodecError::Core)? != 0
        };

        Ok(Self {
            bit_depth,
            mono_chrome,
            num_planes,
            color_primaries,
            transfer_characteristics,
            matrix_coefficients,
            color_range,
            subsampling_x,
            subsampling_y,
            separate_uv_delta_q,
        })
    }
}

// =============================================================================
// Temporal Scalability (SVC) Support
// =============================================================================

/// Maximum number of temporal layers in AV1 SVC.
pub const MAX_TEMPORAL_LAYERS: usize = 8;

/// Maximum number of spatial layers in AV1 SVC.
pub const MAX_SPATIAL_LAYERS: usize = 4;

/// Maximum total operating points (temporal x spatial).
pub const MAX_OPERATING_POINTS: usize = MAX_TEMPORAL_LAYERS * MAX_SPATIAL_LAYERS;

/// SVC (Scalable Video Coding) configuration for AV1.
///
/// AV1 supports temporal scalability through operating points defined
/// in the sequence header. Each operating point specifies which temporal
/// and spatial layers are included, enabling adaptive streaming where
/// decoders can drop higher layers for lower latency or bandwidth.
///
/// # Temporal Layer Structure
///
/// ```text
/// T0: I----P---------P---------P    (base layer, always decodable)
/// T1:      |    P         P         (enhancement, depends on T0)
/// T2:      |  P   P     P   P      (highest, depends on T0+T1)
/// ```
///
/// # Example
///
/// ```ignore
/// use oximedia_codec::av1::sequence::{SvcConfig, TemporalLayerConfig};
///
/// let mut svc = SvcConfig::new(3, 1); // 3 temporal, 1 spatial
/// svc.set_temporal_layer(0, TemporalLayerConfig {
///     layer_id: 0,
///     framerate_fraction: 0.25,
///     bitrate_fraction: 0.5,
///     qp_offset: 0,
///     reference_mode: SvcReferenceMode::KeyOnly,
/// });
/// ```
#[derive(Clone, Debug)]
pub struct SvcConfig {
    /// Number of temporal layers (1-8).
    pub num_temporal_layers: u8,
    /// Number of spatial layers (1-4).
    pub num_spatial_layers: u8,
    /// Per-layer temporal configuration.
    pub temporal_layers: Vec<TemporalLayerConfig>,
    /// Per-layer spatial configuration.
    pub spatial_layers: Vec<SpatialLayerConfig>,
    /// Operating points derived from layer configuration.
    pub operating_points: Vec<OperatingPoint>,
    /// Enable inter-layer prediction.
    pub inter_layer_prediction: bool,
}

impl SvcConfig {
    /// Create a new SVC configuration.
    ///
    /// # Arguments
    ///
    /// * `temporal_layers` - Number of temporal layers (clamped to 1-8)
    /// * `spatial_layers` - Number of spatial layers (clamped to 1-4)
    #[must_use]
    pub fn new(temporal_layers: u8, spatial_layers: u8) -> Self {
        let num_t = temporal_layers.clamp(1, MAX_TEMPORAL_LAYERS as u8);
        let num_s = spatial_layers.clamp(1, MAX_SPATIAL_LAYERS as u8);

        let mut config = Self {
            num_temporal_layers: num_t,
            num_spatial_layers: num_s,
            temporal_layers: Vec::with_capacity(num_t as usize),
            spatial_layers: Vec::with_capacity(num_s as usize),
            operating_points: Vec::new(),
            inter_layer_prediction: true,
        };

        // Initialize default temporal layers with dyadic framerate distribution
        for t in 0..num_t {
            let fraction = 1.0 / (1 << (num_t - 1 - t)) as f32;
            let bitrate_frac = Self::default_bitrate_fraction(t, num_t);
            config.temporal_layers.push(TemporalLayerConfig {
                layer_id: t,
                framerate_fraction: fraction,
                bitrate_fraction: bitrate_frac,
                qp_offset: t as i8 * 2,
                reference_mode: if t == 0 {
                    SvcReferenceMode::KeyAndPrevious
                } else {
                    SvcReferenceMode::PreviousLayer
                },
            });
        }

        // Initialize default spatial layers (full resolution for single layer)
        for s in 0..num_s {
            let scale = 1.0 / (1 << (num_s - 1 - s)) as f32;
            config.spatial_layers.push(SpatialLayerConfig {
                layer_id: s,
                width_scale: scale,
                height_scale: scale,
                bitrate_fraction: 1.0 / num_s as f32,
            });
        }

        config.build_operating_points();
        config
    }

    /// Compute default bitrate fraction for temporal layer using dyadic distribution.
    fn default_bitrate_fraction(layer: u8, total: u8) -> f32 {
        if total <= 1 {
            return 1.0;
        }
        // Base layer gets largest share; each enhancement gets progressively less
        // For 3 layers: T0=0.5, T1=0.3, T2=0.2
        let weight = (1 << (total - 1 - layer)) as f32;
        let total_weight: f32 = (0..total).map(|t| (1 << (total - 1 - t)) as f32).sum();
        weight / total_weight
    }

    /// Set configuration for a specific temporal layer.
    ///
    /// # Arguments
    ///
    /// * `layer_id` - Temporal layer index (0-based)
    /// * `config` - Layer configuration
    pub fn set_temporal_layer(&mut self, layer_id: u8, config: TemporalLayerConfig) {
        let idx = layer_id as usize;
        if idx < self.temporal_layers.len() {
            self.temporal_layers[idx] = config;
            self.build_operating_points();
        }
    }

    /// Set configuration for a specific spatial layer.
    pub fn set_spatial_layer(&mut self, layer_id: u8, config: SpatialLayerConfig) {
        let idx = layer_id as usize;
        if idx < self.spatial_layers.len() {
            self.spatial_layers[idx] = config;
            self.build_operating_points();
        }
    }

    /// Build operating points from layer configurations.
    ///
    /// Each operating point is identified by an `operating_point_idc` bitmask
    /// where bits 0-7 indicate temporal layers and bits 8-11 indicate spatial layers.
    fn build_operating_points(&mut self) {
        self.operating_points.clear();

        // Generate operating points for each combination of cumulative layers
        for s in 0..self.num_spatial_layers {
            for t in 0..self.num_temporal_layers {
                let temporal_mask: u16 = (1 << (t + 1)) - 1; // Include layers 0..=t
                let spatial_mask: u16 = ((1u16 << (s + 1)) - 1) << 8;
                let idc = temporal_mask | spatial_mask;

                let cumulative_bitrate: f32 = self
                    .temporal_layers
                    .iter()
                    .take((t + 1) as usize)
                    .map(|l| l.bitrate_fraction)
                    .sum();

                let framerate: f32 = self
                    .temporal_layers
                    .get(t as usize)
                    .map_or(1.0, |l| l.framerate_fraction);

                self.operating_points.push(OperatingPoint {
                    idc,
                    temporal_id: t,
                    spatial_id: s,
                    level: Self::estimate_level(t, s),
                    tier: 0, // Main tier
                    cumulative_bitrate_fraction: cumulative_bitrate,
                    cumulative_framerate_fraction: framerate,
                });
            }
        }
    }

    /// Estimate AV1 level for a given layer combination.
    fn estimate_level(temporal_id: u8, spatial_id: u8) -> u8 {
        // Simplified level estimation:
        // Base layer starts at level 2.0 (idx=0), each enhancement bumps it
        let base = 0u8; // Level 2.0
        base.saturating_add(temporal_id)
            .saturating_add(spatial_id * 2)
    }

    /// Get operating point for given temporal and spatial IDs.
    #[must_use]
    pub fn get_operating_point(&self, temporal_id: u8, spatial_id: u8) -> Option<&OperatingPoint> {
        self.operating_points
            .iter()
            .find(|op| op.temporal_id == temporal_id && op.spatial_id == spatial_id)
    }

    /// Get total number of operating points.
    #[must_use]
    pub fn num_operating_points(&self) -> usize {
        self.operating_points.len()
    }

    /// Determine which temporal layer a frame belongs to based on its index.
    ///
    /// Uses a dyadic temporal structure:
    /// - Layer 0 (base): frames 0, 4, 8, ...  (for 3 layers)
    /// - Layer 1:        frames 2, 6, 10, ...
    /// - Layer 2:        frames 1, 3, 5, 7, ...
    #[must_use]
    pub fn frame_temporal_layer(&self, frame_index: u64) -> u8 {
        if self.num_temporal_layers <= 1 {
            return 0;
        }

        let n = self.num_temporal_layers;
        let period = 1u64 << (n - 1);

        if frame_index % period == 0 {
            return 0; // Base layer
        }

        // Find highest power of 2 that divides frame_index
        for layer in 1..n {
            let step = period >> layer;
            if step > 0 && frame_index % step == 0 {
                return layer;
            }
        }

        n - 1 // Highest layer
    }

    /// Get QP offset for a frame based on its temporal layer.
    #[must_use]
    pub fn frame_qp_offset(&self, frame_index: u64) -> i8 {
        let layer = self.frame_temporal_layer(frame_index);
        self.temporal_layers
            .get(layer as usize)
            .map_or(0, |l| l.qp_offset)
    }

    /// Check if a frame at given index can be dropped without affecting
    /// lower temporal layers.
    #[must_use]
    pub fn is_droppable(&self, frame_index: u64) -> bool {
        self.frame_temporal_layer(frame_index) > 0
    }

    /// Get the reference mode for a frame based on its temporal layer.
    #[must_use]
    pub fn frame_reference_mode(&self, frame_index: u64) -> SvcReferenceMode {
        let layer = self.frame_temporal_layer(frame_index);
        self.temporal_layers
            .get(layer as usize)
            .map_or(SvcReferenceMode::KeyAndPrevious, |l| l.reference_mode)
    }

    /// Validate the SVC configuration.
    ///
    /// # Errors
    ///
    /// Returns error if the configuration is inconsistent.
    pub fn validate(&self) -> CodecResult<()> {
        if self.num_temporal_layers == 0 || self.num_temporal_layers > MAX_TEMPORAL_LAYERS as u8 {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid temporal layer count: {}",
                self.num_temporal_layers
            )));
        }

        if self.num_spatial_layers == 0 || self.num_spatial_layers > MAX_SPATIAL_LAYERS as u8 {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid spatial layer count: {}",
                self.num_spatial_layers
            )));
        }

        // Verify bitrate fractions sum approximately to 1.0
        let total_bitrate: f32 = self
            .temporal_layers
            .iter()
            .map(|l| l.bitrate_fraction)
            .sum();
        if (total_bitrate - 1.0).abs() > 0.01 {
            return Err(CodecError::InvalidParameter(format!(
                "Temporal bitrate fractions sum to {total_bitrate}, expected ~1.0"
            )));
        }

        // Verify framerate fractions are monotonically increasing
        for i in 1..self.temporal_layers.len() {
            if self.temporal_layers[i].framerate_fraction
                < self.temporal_layers[i - 1].framerate_fraction
            {
                return Err(CodecError::InvalidParameter(
                    "Temporal framerate fractions must be non-decreasing".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Configuration for a single temporal layer.
#[derive(Clone, Debug)]
pub struct TemporalLayerConfig {
    /// Layer identifier (0 = base, higher = enhancement).
    pub layer_id: u8,
    /// Fraction of full framerate this layer represents (0.0-1.0).
    pub framerate_fraction: f32,
    /// Fraction of total bitrate allocated to this layer (0.0-1.0).
    pub bitrate_fraction: f32,
    /// QP offset relative to base layer (positive = lower quality).
    pub qp_offset: i8,
    /// Reference frame mode for this layer.
    pub reference_mode: SvcReferenceMode,
}

/// Configuration for a single spatial layer.
#[derive(Clone, Debug)]
pub struct SpatialLayerConfig {
    /// Layer identifier (0 = lowest resolution).
    pub layer_id: u8,
    /// Width scale relative to full resolution (0.0-1.0).
    pub width_scale: f32,
    /// Height scale relative to full resolution (0.0-1.0).
    pub height_scale: f32,
    /// Fraction of total bitrate allocated to this layer.
    pub bitrate_fraction: f32,
}

/// Reference frame mode for SVC temporal layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SvcReferenceMode {
    /// Only reference key frames (most restrictive).
    KeyOnly,
    /// Reference key frames and previous same-layer frame.
    KeyAndPrevious,
    /// Reference the previous layer's frame.
    PreviousLayer,
    /// Reference any available frame (least restrictive).
    Any,
}

/// An AV1 operating point defined by temporal+spatial layer combination.
#[derive(Clone, Debug)]
pub struct OperatingPoint {
    /// Operating point IDC bitmask (bits 0-7: temporal, bits 8-11: spatial).
    pub idc: u16,
    /// Maximum temporal ID included.
    pub temporal_id: u8,
    /// Maximum spatial ID included.
    pub spatial_id: u8,
    /// AV1 level index for this operating point.
    pub level: u8,
    /// Tier (0 = Main, 1 = High).
    pub tier: u8,
    /// Cumulative bitrate fraction up to this temporal layer.
    pub cumulative_bitrate_fraction: f32,
    /// Cumulative framerate fraction at this temporal layer.
    pub cumulative_framerate_fraction: f32,
}

impl OperatingPoint {
    /// Get the operating point IDC for a given temporal and spatial layer set.
    #[must_use]
    pub fn compute_idc(max_temporal_id: u8, max_spatial_id: u8) -> u16 {
        let temporal_mask: u16 = (1 << (max_temporal_id + 1)) - 1;
        let spatial_mask: u16 = ((1u16 << (max_spatial_id + 1)) - 1) << 8;
        temporal_mask | spatial_mask
    }

    /// Check if this operating point includes a given temporal layer.
    #[must_use]
    pub fn includes_temporal(&self, temporal_id: u8) -> bool {
        (self.idc & (1 << temporal_id)) != 0
    }

    /// Check if this operating point includes a given spatial layer.
    #[must_use]
    pub fn includes_spatial(&self, spatial_id: u8) -> bool {
        (self.idc & (1 << (spatial_id + 8))) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_header_dimensions() {
        let header = SequenceHeader {
            profile: 0,
            still_picture: false,
            reduced_still_picture_header: false,
            max_frame_width_minus_1: 1919,
            max_frame_height_minus_1: 1079,
            enable_order_hint: false,
            order_hint_bits: 0,
            enable_superres: false,
            enable_cdef: false,
            enable_restoration: false,
            color_config: ColorConfig::default(),
            film_grain_params_present: false,
        };

        assert_eq!(header.max_frame_width(), 1920);
        assert_eq!(header.max_frame_height(), 1080);
    }

    #[test]
    fn test_color_config_subsampling() {
        let config_420 = ColorConfig {
            subsampling_x: true,
            subsampling_y: true,
            ..Default::default()
        };
        assert!(config_420.is_420());
    }

    // =====================================================================
    // SVC Tests
    // =====================================================================

    #[test]
    fn test_svc_config_creation() {
        let svc = SvcConfig::new(3, 1);
        assert_eq!(svc.num_temporal_layers, 3);
        assert_eq!(svc.num_spatial_layers, 1);
        assert_eq!(svc.temporal_layers.len(), 3);
        assert_eq!(svc.spatial_layers.len(), 1);
        assert!(svc.inter_layer_prediction);
    }

    #[test]
    fn test_svc_config_clamping() {
        let svc = SvcConfig::new(0, 10);
        assert_eq!(svc.num_temporal_layers, 1);
        assert_eq!(svc.num_spatial_layers, 4);
    }

    #[test]
    fn test_svc_single_layer() {
        let svc = SvcConfig::new(1, 1);
        assert_eq!(svc.num_operating_points(), 1);
        assert_eq!(svc.frame_temporal_layer(0), 0);
        assert_eq!(svc.frame_temporal_layer(1), 0);
        assert!(!svc.is_droppable(0));
    }

    #[test]
    fn test_svc_two_temporal_layers() {
        let svc = SvcConfig::new(2, 1);
        assert_eq!(svc.num_temporal_layers, 2);

        // Dyadic pattern: T0 at even frames, T1 at odd frames
        assert_eq!(svc.frame_temporal_layer(0), 0);
        assert_eq!(svc.frame_temporal_layer(1), 1);
        assert_eq!(svc.frame_temporal_layer(2), 0);
        assert_eq!(svc.frame_temporal_layer(3), 1);

        assert!(!svc.is_droppable(0));
        assert!(svc.is_droppable(1));
        assert!(!svc.is_droppable(2));
    }

    #[test]
    fn test_svc_three_temporal_layers() {
        let svc = SvcConfig::new(3, 1);

        // Dyadic pattern for 3 layers:
        // T0: frames 0, 4, 8, ...
        // T1: frames 2, 6, 10, ...
        // T2: frames 1, 3, 5, 7, ...
        assert_eq!(svc.frame_temporal_layer(0), 0);
        assert_eq!(svc.frame_temporal_layer(1), 2);
        assert_eq!(svc.frame_temporal_layer(2), 1);
        assert_eq!(svc.frame_temporal_layer(3), 2);
        assert_eq!(svc.frame_temporal_layer(4), 0);
    }

    #[test]
    fn test_svc_operating_points() {
        let svc = SvcConfig::new(3, 2);
        // 3 temporal x 2 spatial = 6 operating points
        assert_eq!(svc.num_operating_points(), 6);

        // Base operating point (T0, S0)
        let base_op = svc.get_operating_point(0, 0);
        assert!(base_op.is_some());
        let base = base_op.expect("base operating point exists");
        assert_eq!(base.temporal_id, 0);
        assert_eq!(base.spatial_id, 0);
        assert!(base.includes_temporal(0));
        assert!(!base.includes_temporal(1));
    }

    #[test]
    fn test_svc_operating_point_idc() {
        // T0+T1, S0 => temporal bits 0b11, spatial bits 0b01 << 8
        let idc = OperatingPoint::compute_idc(1, 0);
        assert_eq!(idc, 0x0103); // 0b0000_0001_0000_0011
    }

    #[test]
    fn test_svc_qp_offset() {
        let svc = SvcConfig::new(3, 1);

        // Base layer has lowest QP offset (highest quality)
        let qp0 = svc.frame_qp_offset(0); // T0
        let qp1 = svc.frame_qp_offset(2); // T1
        let qp2 = svc.frame_qp_offset(1); // T2

        // Higher layers have higher QP offset (lower quality)
        assert!(qp0 <= qp1);
        assert!(qp1 <= qp2);
    }

    #[test]
    fn test_svc_bitrate_fractions() {
        let svc = SvcConfig::new(3, 1);

        let total: f32 = svc.temporal_layers.iter().map(|l| l.bitrate_fraction).sum();
        assert!((total - 1.0).abs() < 0.01);

        // Base layer should get largest share
        assert!(svc.temporal_layers[0].bitrate_fraction > svc.temporal_layers[1].bitrate_fraction);
        assert!(svc.temporal_layers[1].bitrate_fraction > svc.temporal_layers[2].bitrate_fraction);
    }

    #[test]
    fn test_svc_framerate_fractions() {
        let svc = SvcConfig::new(3, 1);

        // Framerate should be monotonically non-decreasing
        for i in 1..svc.temporal_layers.len() {
            assert!(
                svc.temporal_layers[i].framerate_fraction
                    >= svc.temporal_layers[i - 1].framerate_fraction
            );
        }
    }

    #[test]
    fn test_svc_validation_valid() {
        let svc = SvcConfig::new(3, 1);
        assert!(svc.validate().is_ok());
    }

    #[test]
    fn test_svc_validation_bad_bitrate() {
        let mut svc = SvcConfig::new(2, 1);
        svc.temporal_layers[0].bitrate_fraction = 0.1;
        svc.temporal_layers[1].bitrate_fraction = 0.1;
        // Sum = 0.2, far from 1.0
        assert!(svc.validate().is_err());
    }

    #[test]
    fn test_svc_set_temporal_layer() {
        let mut svc = SvcConfig::new(3, 1);
        svc.set_temporal_layer(
            1,
            TemporalLayerConfig {
                layer_id: 1,
                framerate_fraction: 0.5,
                bitrate_fraction: 0.3,
                qp_offset: 4,
                reference_mode: SvcReferenceMode::Any,
            },
        );

        assert_eq!(svc.temporal_layers[1].qp_offset, 4);
        assert_eq!(svc.temporal_layers[1].reference_mode, SvcReferenceMode::Any);
    }

    #[test]
    fn test_svc_spatial_layer_defaults() {
        let svc = SvcConfig::new(2, 2);

        assert_eq!(svc.spatial_layers.len(), 2);
        // Lower spatial layer has smaller scale
        assert!(svc.spatial_layers[0].width_scale < svc.spatial_layers[1].width_scale);
    }

    #[test]
    fn test_svc_reference_mode() {
        let svc = SvcConfig::new(3, 1);

        let ref0 = svc.frame_reference_mode(0); // T0
        let ref2 = svc.frame_reference_mode(1); // T2

        // Base layer references key+previous; enhancement references previous layer
        assert_eq!(ref0, SvcReferenceMode::KeyAndPrevious);
        assert_eq!(ref2, SvcReferenceMode::PreviousLayer);
    }

    #[test]
    fn test_operating_point_includes() {
        let op = OperatingPoint {
            idc: 0x0107, // T0+T1+T2, S0
            temporal_id: 2,
            spatial_id: 0,
            level: 2,
            tier: 0,
            cumulative_bitrate_fraction: 1.0,
            cumulative_framerate_fraction: 1.0,
        };

        assert!(op.includes_temporal(0));
        assert!(op.includes_temporal(1));
        assert!(op.includes_temporal(2));
        assert!(!op.includes_temporal(3));
        assert!(op.includes_spatial(0));
        assert!(!op.includes_spatial(1));
    }

    #[test]
    fn test_svc_droppable_frames() {
        let svc = SvcConfig::new(3, 1);

        // Collect droppable status for first 8 frames
        let droppable: Vec<bool> = (0..8).map(|i| svc.is_droppable(i)).collect();

        // Only base layer frames are not droppable
        assert!(!droppable[0]); // T0
        assert!(droppable[1]); // T2
        assert!(droppable[2]); // T1
        assert!(droppable[3]); // T2
        assert!(!droppable[4]); // T0
    }
}
