//! AV1 CDEF (Constrained Directional Enhancement Filter) parameters.
//!
//! CDEF is a directional filter applied after the loop filter to reduce
//! ringing and mosquito noise artifacts. It uses directional filtering
//! along edges to preserve sharpness while reducing noise.
//!
//! # CDEF Algorithm
//!
//! 1. For each 8x8 block, detect the edge direction
//! 2. Apply filtering along and perpendicular to the edge
//! 3. Use different strengths for primary (along edge) and secondary (perpendicular)
//!
//! # Parameters
//!
//! - Damping: Controls how much filtering is applied (higher = less filtering)
//! - Primary strength: Filter strength along the detected edge direction
//! - Secondary strength: Filter strength perpendicular to the edge
//! - CDEF bits: Number of bits to signal CDEF preset index
//!
//! # Reference
//!
//! See AV1 Specification Section 5.9.19 for CDEF syntax and
//! Section 7.15 for CDEF semantics.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::manual_div_ceil)]

use super::sequence::SequenceHeader;
use crate::error::{CodecError, CodecResult};
use oximedia_io::BitReader;

// =============================================================================
// Constants
// =============================================================================

/// Maximum CDEF bits.
pub const CDEF_MAX_BITS: u8 = 3;

/// Maximum number of CDEF presets (2^CDEF_MAX_BITS).
pub const CDEF_MAX_PRESETS: usize = 8;

/// Maximum primary strength.
pub const CDEF_MAX_PRIMARY_STRENGTH: u8 = 15;

/// Maximum secondary strength.
pub const CDEF_MAX_SECONDARY_STRENGTH: u8 = 4;

/// CDEF block size (8x8).
pub const CDEF_BLOCK_SIZE: usize = 8;

/// Number of directions for CDEF.
pub const CDEF_NUM_DIRECTIONS: usize = 8;

/// Minimum damping value.
pub const CDEF_DAMPING_MIN: u8 = 3;

/// Maximum damping value.
pub const CDEF_DAMPING_MAX: u8 = 6;

/// CDEF secondary strength table.
pub const CDEF_SEC_STRENGTHS: [u8; 4] = [0, 1, 2, 4];

// =============================================================================
// Structures
// =============================================================================

/// CDEF strength parameters for a single preset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CdefStrength {
    /// Primary strength (0-15).
    pub primary: u8,
    /// Secondary strength index (0-3, maps to 0, 1, 2, 4).
    pub secondary: u8,
}

impl CdefStrength {
    /// Create a new CDEF strength.
    #[must_use]
    pub const fn new(primary: u8, secondary: u8) -> Self {
        Self { primary, secondary }
    }

    /// Get the actual secondary strength value.
    #[must_use]
    pub fn secondary_value(&self) -> u8 {
        CDEF_SEC_STRENGTHS
            .get(self.secondary as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Check if this preset has any filtering.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.primary > 0 || self.secondary > 0
    }

    /// Parse a CDEF strength from bitstream.
    #[allow(clippy::cast_possible_truncation)]
    pub fn parse(reader: &mut BitReader<'_>) -> CodecResult<Self> {
        let primary = reader.read_bits(4).map_err(CodecError::Core)? as u8;
        let secondary = reader.read_bits(2).map_err(CodecError::Core)? as u8;
        Ok(Self { primary, secondary })
    }
}

/// CDEF parameters as parsed from the frame header.
#[derive(Clone, Debug)]
pub struct CdefParams {
    /// Damping value for Y plane.
    pub damping_y: u8,
    /// Damping value for UV planes.
    pub damping_uv: u8,
    /// Number of bits to signal CDEF preset index.
    pub bits: u8,
    /// Y plane strength presets.
    pub y_strengths: [CdefStrength; CDEF_MAX_PRESETS],
    /// UV plane strength presets.
    pub uv_strengths: [CdefStrength; CDEF_MAX_PRESETS],
}

impl Default for CdefParams {
    fn default() -> Self {
        Self {
            damping_y: CDEF_DAMPING_MIN,
            damping_uv: CDEF_DAMPING_MIN,
            bits: 0,
            y_strengths: [CdefStrength::default(); CDEF_MAX_PRESETS],
            uv_strengths: [CdefStrength::default(); CDEF_MAX_PRESETS],
        }
    }
}

impl CdefParams {
    /// Create new CDEF parameters with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse CDEF parameters from the bitstream.
    ///
    /// # Errors
    ///
    /// Returns error if the bitstream is malformed.
    #[allow(clippy::cast_possible_truncation)]
    pub fn parse(reader: &mut BitReader<'_>, seq: &SequenceHeader) -> CodecResult<Self> {
        let mut cdef = Self::new();
        let num_planes = if seq.color_config.mono_chrome { 1 } else { 3 };

        // Damping
        let damping_minus_3 = reader.read_bits(2).map_err(CodecError::Core)? as u8;
        cdef.damping_y = damping_minus_3 + CDEF_DAMPING_MIN;
        cdef.damping_uv = cdef.damping_y;

        // Number of CDEF bits
        cdef.bits = reader.read_bits(2).map_err(CodecError::Core)? as u8;

        // Parse Y strengths
        let num_presets = 1usize << cdef.bits;
        for i in 0..num_presets {
            cdef.y_strengths[i] = CdefStrength::parse(reader)?;
        }

        // Parse UV strengths (if not monochrome)
        if num_planes > 1 {
            for i in 0..num_presets {
                cdef.uv_strengths[i] = CdefStrength::parse(reader)?;
            }
        }

        Ok(cdef)
    }

    /// Get the number of CDEF presets.
    #[must_use]
    pub fn num_presets(&self) -> usize {
        1usize << self.bits
    }

    /// Check if CDEF is enabled (at least one preset has non-zero strength).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        let num = self.num_presets();
        for i in 0..num {
            if self.y_strengths[i].is_enabled() || self.uv_strengths[i].is_enabled() {
                return true;
            }
        }
        false
    }

    /// Get the Y strength for a preset index.
    #[must_use]
    pub fn get_y_strength(&self, idx: usize) -> CdefStrength {
        self.y_strengths.get(idx).copied().unwrap_or_default()
    }

    /// Get the UV strength for a preset index.
    #[must_use]
    pub fn get_uv_strength(&self, idx: usize) -> CdefStrength {
        self.uv_strengths.get(idx).copied().unwrap_or_default()
    }

    /// Get damping for a specific plane.
    #[must_use]
    pub const fn get_damping(&self, plane: usize) -> u8 {
        if plane == 0 {
            self.damping_y
        } else {
            self.damping_uv
        }
    }

    /// Get the adjusted damping value for a bit depth.
    #[must_use]
    pub fn adjusted_damping(&self, plane: usize, bit_depth: u8) -> u8 {
        let base = self.get_damping(plane);
        // Damping is adjusted based on bit depth
        base + (bit_depth - 8).min(4)
    }

    /// Check if a specific preset index is valid.
    #[must_use]
    pub fn is_valid_preset(&self, idx: usize) -> bool {
        idx < self.num_presets()
    }
}

/// CDEF direction and variance information for a block.
#[derive(Clone, Copy, Debug, Default)]
pub struct CdefDirection {
    /// Detected direction (0-7).
    pub direction: u8,
    /// Variance of the direction.
    pub variance: u32,
}

impl CdefDirection {
    /// Create a new CDEF direction.
    #[must_use]
    pub const fn new(direction: u8, variance: u32) -> Self {
        Self {
            direction,
            variance,
        }
    }

    /// Get the opposite direction.
    #[must_use]
    pub const fn opposite(&self) -> u8 {
        (self.direction + 4) % CDEF_NUM_DIRECTIONS as u8
    }

    /// Get the perpendicular direction (clockwise).
    #[must_use]
    pub const fn perpendicular_cw(&self) -> u8 {
        (self.direction + 2) % CDEF_NUM_DIRECTIONS as u8
    }

    /// Get the perpendicular direction (counter-clockwise).
    #[must_use]
    pub const fn perpendicular_ccw(&self) -> u8 {
        (self.direction + 6) % CDEF_NUM_DIRECTIONS as u8
    }
}

/// Direction offsets for CDEF filtering.
///
/// Each direction has offsets for the primary and secondary taps.
#[derive(Clone, Copy, Debug)]
pub struct CdefDirectionOffsets {
    /// Primary direction tap offsets (2 taps).
    pub primary: [(i8, i8); 2],
    /// Secondary direction tap offsets (4 taps).
    pub secondary: [(i8, i8); 4],
}

/// Direction kernel offsets for all 8 directions.
pub const CDEF_DIRECTION_OFFSETS: [CdefDirectionOffsets; CDEF_NUM_DIRECTIONS] = [
    // Direction 0: Horizontal
    CdefDirectionOffsets {
        primary: [(-1, 0), (1, 0)],
        secondary: [(-2, 0), (2, 0), (-1, -1), (1, 1)],
    },
    // Direction 1: 22.5 degrees
    CdefDirectionOffsets {
        primary: [(-1, -1), (1, 1)],
        secondary: [(-2, -1), (2, 1), (-1, -2), (1, 2)],
    },
    // Direction 2: 45 degrees
    CdefDirectionOffsets {
        primary: [(0, -1), (0, 1)],
        secondary: [(-1, -1), (1, 1), (-1, -2), (1, 2)],
    },
    // Direction 3: 67.5 degrees
    CdefDirectionOffsets {
        primary: [(1, -1), (-1, 1)],
        secondary: [(2, -1), (-2, 1), (1, -2), (-1, 2)],
    },
    // Direction 4: Vertical
    CdefDirectionOffsets {
        primary: [(0, -1), (0, 1)],
        secondary: [(0, -2), (0, 2), (-1, -1), (1, 1)],
    },
    // Direction 5: 112.5 degrees
    CdefDirectionOffsets {
        primary: [(-1, -1), (1, 1)],
        secondary: [(-2, -1), (2, 1), (-1, -2), (1, 2)],
    },
    // Direction 6: 135 degrees
    CdefDirectionOffsets {
        primary: [(-1, 0), (1, 0)],
        secondary: [(-1, -1), (1, 1), (-2, 0), (2, 0)],
    },
    // Direction 7: 157.5 degrees
    CdefDirectionOffsets {
        primary: [(1, -1), (-1, 1)],
        secondary: [(2, -1), (-2, 1), (1, -2), (-1, 2)],
    },
];

/// CDEF filter tap weights.
#[derive(Clone, Copy, Debug)]
pub struct CdefTapWeights {
    /// Primary tap weight.
    pub primary: i16,
    /// Secondary tap weight.
    pub secondary: i16,
}

impl CdefTapWeights {
    /// Create tap weights from strengths and damping.
    #[must_use]
    pub fn from_strengths(strength: &CdefStrength, _damping: u8) -> Self {
        // Weights depend on strength and damping
        let primary = i16::from(strength.primary);
        let secondary = i16::from(strength.secondary_value());

        Self { primary, secondary }
    }
}

/// CDEF block filter state.
#[derive(Clone, Debug, Default)]
pub struct CdefBlockState {
    /// CDEF preset index for this block.
    pub preset_idx: u8,
    /// Skip CDEF for this block.
    pub skip: bool,
    /// Detected direction.
    pub direction: CdefDirection,
}

impl CdefBlockState {
    /// Create a new CDEF block state.
    #[must_use]
    pub const fn new(preset_idx: u8) -> Self {
        Self {
            preset_idx,
            skip: false,
            direction: CdefDirection {
                direction: 0,
                variance: 0,
            },
        }
    }

    /// Check if filtering should be applied.
    #[must_use]
    pub const fn should_filter(&self) -> bool {
        !self.skip
    }
}

/// CDEF superblock information.
#[derive(Clone, Debug)]
pub struct CdefSuperblock {
    /// CDEF indices for each 8x8 block in the superblock.
    /// For 64x64 SB: 64 blocks (8x8), for 128x128 SB: 256 blocks.
    pub block_indices: Vec<u8>,
    /// Superblock size (64 or 128).
    pub sb_size: usize,
}

impl CdefSuperblock {
    /// Create a new CDEF superblock.
    #[must_use]
    pub fn new(sb_size: usize) -> Self {
        let num_blocks = (sb_size / CDEF_BLOCK_SIZE) * (sb_size / CDEF_BLOCK_SIZE);
        Self {
            block_indices: vec![0; num_blocks],
            sb_size,
        }
    }

    /// Get the CDEF index for a block at the given position.
    #[must_use]
    pub fn get_index(&self, row: usize, col: usize) -> u8 {
        let blocks_per_row = self.sb_size / CDEF_BLOCK_SIZE;
        let idx = row * blocks_per_row + col;
        self.block_indices.get(idx).copied().unwrap_or(0)
    }

    /// Set the CDEF index for a block.
    pub fn set_index(&mut self, row: usize, col: usize, value: u8) {
        let blocks_per_row = self.sb_size / CDEF_BLOCK_SIZE;
        let idx = row * blocks_per_row + col;
        if idx < self.block_indices.len() {
            self.block_indices[idx] = value;
        }
    }

    /// Get the number of 8x8 blocks in the superblock.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.block_indices.len()
    }

    /// Get the number of blocks per row/column.
    #[must_use]
    pub fn blocks_per_side(&self) -> usize {
        self.sb_size / CDEF_BLOCK_SIZE
    }
}

/// CDEF configuration for a frame.
#[derive(Clone, Debug)]
pub struct CdefFrameConfig {
    /// CDEF parameters from frame header.
    pub params: CdefParams,
    /// Bit depth.
    pub bit_depth: u8,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl CdefFrameConfig {
    /// Create a new CDEF frame configuration.
    #[must_use]
    pub fn new(params: CdefParams, bit_depth: u8, width: u32, height: u32) -> Self {
        Self {
            params,
            bit_depth,
            width,
            height,
        }
    }

    /// Get the number of 8x8 blocks in the frame (width).
    #[must_use]
    pub fn blocks_wide(&self) -> u32 {
        (self.width + (CDEF_BLOCK_SIZE as u32) - 1) / (CDEF_BLOCK_SIZE as u32)
    }

    /// Get the number of 8x8 blocks in the frame (height).
    #[must_use]
    pub fn blocks_high(&self) -> u32 {
        (self.height + (CDEF_BLOCK_SIZE as u32) - 1) / (CDEF_BLOCK_SIZE as u32)
    }

    /// Get total number of 8x8 blocks.
    #[must_use]
    pub fn total_blocks(&self) -> u32 {
        self.blocks_wide() * self.blocks_high()
    }

    /// Get the adjusted damping for a plane.
    #[must_use]
    pub fn get_damping(&self, plane: usize) -> u8 {
        self.params.adjusted_damping(plane, self.bit_depth)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute the CDEF filter value with clamping.
#[must_use]
pub fn constrain(diff: i16, threshold: i16, damping: u8) -> i16 {
    if threshold == 0 {
        return 0;
    }

    #[allow(clippy::cast_possible_truncation)]
    let shift = damping.saturating_sub(threshold.unsigned_abs() as u8);
    let magnitude = diff.abs().min(threshold.abs());

    if shift >= 15 {
        0
    } else {
        let clamped = magnitude - (magnitude >> shift);
        if diff < 0 {
            -clamped
        } else {
            clamped
        }
    }
}

/// Calculate the clipping value for CDEF.
#[must_use]
pub const fn cdef_clip(value: i16, max: i16) -> i16 {
    if value < 0 {
        0
    } else if value > max {
        max
    } else {
        value
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdef_strength_default() {
        let s = CdefStrength::default();
        assert_eq!(s.primary, 0);
        assert_eq!(s.secondary, 0);
        assert!(!s.is_enabled());
    }

    #[test]
    fn test_cdef_strength_new() {
        let s = CdefStrength::new(10, 2);
        assert_eq!(s.primary, 10);
        assert_eq!(s.secondary, 2);
        assert!(s.is_enabled());
    }

    #[test]
    fn test_cdef_strength_secondary_value() {
        assert_eq!(CdefStrength::new(0, 0).secondary_value(), 0);
        assert_eq!(CdefStrength::new(0, 1).secondary_value(), 1);
        assert_eq!(CdefStrength::new(0, 2).secondary_value(), 2);
        assert_eq!(CdefStrength::new(0, 3).secondary_value(), 4);
    }

    #[test]
    fn test_cdef_params_default() {
        let params = CdefParams::default();
        assert_eq!(params.damping_y, CDEF_DAMPING_MIN);
        assert_eq!(params.damping_uv, CDEF_DAMPING_MIN);
        assert_eq!(params.bits, 0);
        assert_eq!(params.num_presets(), 1);
    }

    #[test]
    fn test_cdef_params_num_presets() {
        let mut params = CdefParams::default();
        assert_eq!(params.num_presets(), 1);

        params.bits = 1;
        assert_eq!(params.num_presets(), 2);

        params.bits = 2;
        assert_eq!(params.num_presets(), 4);

        params.bits = 3;
        assert_eq!(params.num_presets(), 8);
    }

    #[test]
    fn test_cdef_params_is_enabled() {
        let mut params = CdefParams::default();
        assert!(!params.is_enabled());

        params.y_strengths[0].primary = 5;
        assert!(params.is_enabled());

        params.y_strengths[0].primary = 0;
        params.uv_strengths[0].secondary = 2;
        assert!(params.is_enabled());
    }

    #[test]
    fn test_cdef_params_get_damping() {
        let params = CdefParams {
            damping_y: 5,
            damping_uv: 4,
            ..Default::default()
        };

        assert_eq!(params.get_damping(0), 5);
        assert_eq!(params.get_damping(1), 4);
        assert_eq!(params.get_damping(2), 4);
    }

    #[test]
    fn test_cdef_params_adjusted_damping() {
        let params = CdefParams {
            damping_y: 3,
            damping_uv: 3,
            ..Default::default()
        };

        assert_eq!(params.adjusted_damping(0, 8), 3);
        assert_eq!(params.adjusted_damping(0, 10), 5);
        assert_eq!(params.adjusted_damping(0, 12), 7);
    }

    #[test]
    fn test_cdef_params_valid_preset() {
        let mut params = CdefParams::default();
        params.bits = 2;

        assert!(params.is_valid_preset(0));
        assert!(params.is_valid_preset(3));
        assert!(!params.is_valid_preset(4));
    }

    #[test]
    fn test_cdef_direction() {
        let dir = CdefDirection::new(2, 100);
        assert_eq!(dir.direction, 2);
        assert_eq!(dir.variance, 100);
        assert_eq!(dir.opposite(), 6);
        assert_eq!(dir.perpendicular_cw(), 4);
        assert_eq!(dir.perpendicular_ccw(), 0);
    }

    #[test]
    fn test_cdef_direction_wrap() {
        let dir = CdefDirection::new(6, 0);
        assert_eq!(dir.opposite(), 2);
        assert_eq!(dir.perpendicular_cw(), 0);
        assert_eq!(dir.perpendicular_ccw(), 4);
    }

    #[test]
    fn test_cdef_block_state() {
        let state = CdefBlockState::new(3);
        assert_eq!(state.preset_idx, 3);
        assert!(!state.skip);
        assert!(state.should_filter());

        let mut state2 = CdefBlockState::new(0);
        state2.skip = true;
        assert!(!state2.should_filter());
    }

    #[test]
    fn test_cdef_superblock() {
        let mut sb = CdefSuperblock::new(64);
        assert_eq!(sb.num_blocks(), 64);
        assert_eq!(sb.blocks_per_side(), 8);

        sb.set_index(2, 3, 5);
        assert_eq!(sb.get_index(2, 3), 5);
        assert_eq!(sb.get_index(0, 0), 0);
    }

    #[test]
    fn test_cdef_superblock_128() {
        let sb = CdefSuperblock::new(128);
        assert_eq!(sb.num_blocks(), 256);
        assert_eq!(sb.blocks_per_side(), 16);
    }

    #[test]
    fn test_cdef_frame_config() {
        let params = CdefParams::default();
        let config = CdefFrameConfig::new(params, 8, 1920, 1080);

        assert_eq!(config.blocks_wide(), 240);
        assert_eq!(config.blocks_high(), 135);
        assert_eq!(config.total_blocks(), 240 * 135);
    }

    #[test]
    fn test_constrain() {
        // Zero threshold returns zero
        assert_eq!(constrain(10, 0, 3), 0);

        // Positive diff
        let result = constrain(5, 10, 3);
        assert!(result >= 0 && result <= 5);

        // Negative diff
        let result = constrain(-5, 10, 3);
        assert!(result <= 0 && result >= -5);
    }

    #[test]
    fn test_cdef_clip() {
        assert_eq!(cdef_clip(-5, 255), 0);
        assert_eq!(cdef_clip(100, 255), 100);
        assert_eq!(cdef_clip(300, 255), 255);
    }

    #[test]
    fn test_cdef_direction_offsets() {
        // Horizontal direction
        let offsets = &CDEF_DIRECTION_OFFSETS[0];
        assert_eq!(offsets.primary[0], (-1, 0));
        assert_eq!(offsets.primary[1], (1, 0));

        // Vertical direction (direction 4)
        let offsets = &CDEF_DIRECTION_OFFSETS[4];
        assert_eq!(offsets.primary[0], (0, -1));
        assert_eq!(offsets.primary[1], (0, 1));
    }

    #[test]
    fn test_cdef_tap_weights() {
        let strength = CdefStrength::new(8, 2);
        let weights = CdefTapWeights::from_strengths(&strength, 3);
        assert_eq!(weights.primary, 8);
        assert_eq!(weights.secondary, 2);
    }

    #[test]
    fn test_constants() {
        assert_eq!(CDEF_MAX_BITS, 3);
        assert_eq!(CDEF_MAX_PRESETS, 8);
        assert_eq!(CDEF_BLOCK_SIZE, 8);
        assert_eq!(CDEF_NUM_DIRECTIONS, 8);
        assert_eq!(CDEF_DAMPING_MIN, 3);
        assert_eq!(CDEF_DAMPING_MAX, 6);
    }

    #[test]
    fn test_sec_strengths_table() {
        assert_eq!(CDEF_SEC_STRENGTHS, [0, 1, 2, 4]);
    }
}
