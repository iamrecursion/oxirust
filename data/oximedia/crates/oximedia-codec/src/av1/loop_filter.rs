//! AV1 Loop Filter parameters.
//!
//! The loop filter is applied after transform reconstruction to reduce
//! blocking artifacts at block boundaries. AV1 uses a direction-adaptive
//! loop filter with separate parameters for each edge type.
//!
//! # Loop Filter Parameters
//!
//! - Filter level (0-63) per plane and direction
//! - Sharpness (0-7) affects filter threshold
//! - Delta values for mode and reference frame adjustments
//!
//! # Reference
//!
//! See AV1 Specification Section 5.9.11 for loop filter syntax and
//! Section 7.14 for loop filter semantics.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::identity_op)]
#![allow(clippy::if_not_else)]
#![allow(clippy::missing_errors_doc)]

use super::sequence::SequenceHeader;
use crate::error::{CodecError, CodecResult};
use oximedia_io::BitReader;

// =============================================================================
// Constants
// =============================================================================

/// Maximum loop filter level.
pub const MAX_LOOP_FILTER_LEVEL: u8 = 63;

/// Maximum sharpness level.
pub const MAX_SHARPNESS_LEVEL: u8 = 7;

/// Number of loop filter mode deltas.
pub const MAX_MODE_LF_DELTAS: usize = 2;

/// Number of reference frame deltas (including intra).
pub const TOTAL_REFS_PER_FRAME: usize = 8;

/// Loop filter level bits in bitstream.
pub const LF_LEVEL_BITS: u8 = 6;

/// Loop filter delta bits in bitstream.
pub const LF_DELTA_BITS: u8 = 6;

/// Default reference deltas.
pub const DEFAULT_REF_DELTAS: [i8; TOTAL_REFS_PER_FRAME] = [1, 0, 0, 0, 0, -1, -1, -1];

/// Default mode deltas.
pub const DEFAULT_MODE_DELTAS: [i8; MAX_MODE_LF_DELTAS] = [0, 0];

// =============================================================================
// Structures
// =============================================================================

/// Loop filter parameters as parsed from the frame header.
#[derive(Clone, Debug)]
pub struct LoopFilterParams {
    /// Loop filter level for Y vertical edges.
    pub level: [u8; 4],
    /// Sharpness level (0-7).
    pub sharpness: u8,
    /// Delta coding enabled.
    pub delta_enabled: bool,
    /// Update delta values.
    pub delta_update: bool,
    /// Reference frame deltas.
    pub ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    /// Mode deltas (for ZERO_MV and MV modes).
    pub mode_deltas: [i8; MAX_MODE_LF_DELTAS],
}

impl Default for LoopFilterParams {
    fn default() -> Self {
        Self {
            level: [0; 4],
            sharpness: 0,
            delta_enabled: true,
            delta_update: true,
            ref_deltas: DEFAULT_REF_DELTAS,
            mode_deltas: DEFAULT_MODE_DELTAS,
        }
    }
}

impl LoopFilterParams {
    /// Create a new loop filter params with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse loop filter parameters from the bitstream.
    ///
    /// # Errors
    ///
    /// Returns error if the bitstream is malformed.
    #[allow(clippy::cast_possible_truncation)]
    pub fn parse(
        reader: &mut BitReader<'_>,
        seq: &SequenceHeader,
        frame_is_intra: bool,
    ) -> CodecResult<Self> {
        let mut lf = Self::new();

        // Read filter levels
        lf.level[0] = reader.read_bits(LF_LEVEL_BITS).map_err(CodecError::Core)? as u8;
        lf.level[1] = reader.read_bits(LF_LEVEL_BITS).map_err(CodecError::Core)? as u8;

        // Chroma levels only if there are chroma planes and Y levels > 0
        if !seq.color_config.mono_chrome && (lf.level[0] > 0 || lf.level[1] > 0) {
            lf.level[2] = reader.read_bits(LF_LEVEL_BITS).map_err(CodecError::Core)? as u8;
            lf.level[3] = reader.read_bits(LF_LEVEL_BITS).map_err(CodecError::Core)? as u8;
        }

        // Sharpness
        lf.sharpness = reader.read_bits(3).map_err(CodecError::Core)? as u8;

        // Delta coding
        lf.delta_enabled = reader.read_bit().map_err(CodecError::Core)? != 0;

        if lf.delta_enabled {
            lf.delta_update = reader.read_bit().map_err(CodecError::Core)? != 0;

            if lf.delta_update {
                // Reference deltas
                for i in 0..TOTAL_REFS_PER_FRAME {
                    let update = reader.read_bit().map_err(CodecError::Core)? != 0;
                    if update {
                        lf.ref_deltas[i] = Self::read_delta(reader)?;
                    }
                }

                // Mode deltas (only for inter frames)
                if !frame_is_intra {
                    for i in 0..MAX_MODE_LF_DELTAS {
                        let update = reader.read_bit().map_err(CodecError::Core)? != 0;
                        if update {
                            lf.mode_deltas[i] = Self::read_delta(reader)?;
                        }
                    }
                }
            }
        }

        Ok(lf)
    }

    /// Read a signed delta value using su(1+6) format.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn read_delta(reader: &mut BitReader<'_>) -> CodecResult<i8> {
        let abs_value = reader.read_bits(LF_DELTA_BITS).map_err(CodecError::Core)? as i8;
        if abs_value != 0 {
            let sign = reader.read_bit().map_err(CodecError::Core)?;
            if sign != 0 {
                Ok(-abs_value)
            } else {
                Ok(abs_value)
            }
        } else {
            Ok(0)
        }
    }

    /// Get the Y vertical filter level.
    #[must_use]
    pub const fn level_y_v(&self) -> u8 {
        self.level[0]
    }

    /// Get the Y horizontal filter level.
    #[must_use]
    pub const fn level_y_h(&self) -> u8 {
        self.level[1]
    }

    /// Get the U filter level.
    #[must_use]
    pub const fn level_u(&self) -> u8 {
        self.level[2]
    }

    /// Get the V filter level.
    #[must_use]
    pub const fn level_v(&self) -> u8 {
        self.level[3]
    }

    /// Check if loop filter is enabled for any plane.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.level.iter().any(|&l| l > 0)
    }

    /// Get the filter level for a specific plane and direction.
    ///
    /// # Arguments
    ///
    /// * `plane` - Plane index (0=Y, 1=U, 2=V)
    /// * `direction` - 0 for vertical, 1 for horizontal
    #[must_use]
    pub fn get_level(&self, plane: usize, direction: usize) -> u8 {
        match (plane, direction) {
            (0, 0) => self.level[0],
            (0, 1) => self.level[1],
            (1, _) => self.level[2],
            (2, _) => self.level[3],
            _ => 0,
        }
    }

    /// Compute the filter level for a block.
    ///
    /// This applies delta adjustments based on reference frame and mode.
    ///
    /// # Arguments
    ///
    /// * `base_level` - Base filter level from frame header
    /// * `ref_frame` - Reference frame index (0 for intra)
    /// * `mode` - Mode index (0 for ZEROMV, 1 for other MV modes)
    /// * `segment_delta` - Delta from segmentation
    #[must_use]
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    pub fn compute_level(
        &self,
        base_level: u8,
        ref_frame: usize,
        mode: usize,
        segment_delta: i16,
    ) -> u8 {
        if base_level == 0 || !self.delta_enabled {
            return base_level;
        }

        let mut level = i32::from(base_level);

        // Apply segmentation delta
        level += i32::from(segment_delta);

        // Apply reference frame delta
        if ref_frame < TOTAL_REFS_PER_FRAME {
            level += i32::from(self.ref_deltas[ref_frame]);
        }

        // Apply mode delta (only for inter blocks)
        if ref_frame > 0 && mode < MAX_MODE_LF_DELTAS {
            level += i32::from(self.mode_deltas[mode]);
        }

        level.clamp(0, i32::from(MAX_LOOP_FILTER_LEVEL)) as u8
    }

    /// Get the limit value for the filter based on sharpness.
    ///
    /// The limit determines the maximum difference for the filter to be applied.
    #[must_use]
    pub fn get_limit(&self, level: u8) -> u8 {
        if self.sharpness > 0 {
            let block_limit = (9 - self.sharpness).max(1);
            let shift = (self.sharpness + 3) >> 2;
            ((level >> shift) as u8).min(block_limit)
        } else {
            ((level >> 0) as u8).max(1)
        }
    }

    /// Get the threshold value for the filter.
    #[must_use]
    pub fn get_threshold(&self, level: u8) -> u8 {
        // thresh = 0.5 * limit
        self.get_limit(level) >> 1
    }

    /// Get the high edge variance threshold.
    #[must_use]
    pub const fn get_hev_threshold(&self, level: u8) -> u8 {
        if level >= 40 {
            2
        } else if level >= 20 {
            1
        } else {
            0
        }
    }

    /// Check if delta updates are present.
    #[must_use]
    pub const fn has_delta_updates(&self) -> bool {
        self.delta_enabled && self.delta_update
    }

    /// Reset deltas to default values.
    pub fn reset_deltas(&mut self) {
        self.ref_deltas = DEFAULT_REF_DELTAS;
        self.mode_deltas = DEFAULT_MODE_DELTAS;
    }

    /// Set filter level for all planes.
    pub fn set_level_all(&mut self, level: u8) {
        self.level = [level; 4];
    }

    /// Serialize loop filter parameters to bitstream.
    #[must_use]
    pub fn to_bytes(&self, seq: &SequenceHeader, frame_is_intra: bool) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();

        // This is a simplified serialization
        // In practice, you would write bits to a BitWriter

        bits.push(self.level[0]);
        bits.push(self.level[1]);

        if !seq.color_config.mono_chrome && (self.level[0] > 0 || self.level[1] > 0) {
            bits.push(self.level[2]);
            bits.push(self.level[3]);
        }

        bits.push(self.sharpness);
        bits.push(u8::from(self.delta_enabled));

        if self.delta_enabled {
            bits.push(u8::from(self.delta_update));

            if self.delta_update {
                for &delta in &self.ref_deltas {
                    #[allow(clippy::cast_sign_loss)]
                    bits.push(delta.unsigned_abs());
                }
                if !frame_is_intra {
                    for &delta in &self.mode_deltas {
                        #[allow(clippy::cast_sign_loss)]
                        bits.push(delta.unsigned_abs());
                    }
                }
            }
        }

        bits
    }
}

/// Loop filter edge information.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoopFilterEdge {
    /// Edge direction (0=vertical, 1=horizontal).
    pub direction: u8,
    /// Filter level for this edge.
    pub level: u8,
    /// Limit value.
    pub limit: u8,
    /// Threshold value.
    pub threshold: u8,
    /// High edge variance threshold.
    pub hev_threshold: u8,
}

impl LoopFilterEdge {
    /// Create a new loop filter edge with the given parameters.
    #[must_use]
    pub fn new(params: &LoopFilterParams, level: u8, direction: u8) -> Self {
        Self {
            direction,
            level,
            limit: params.get_limit(level),
            threshold: params.get_threshold(level),
            hev_threshold: params.get_hev_threshold(level),
        }
    }

    /// Check if filtering should be applied.
    #[must_use]
    pub const fn should_filter(&self) -> bool {
        self.level > 0
    }
}

/// Loop filter mask for a superblock.
///
/// Contains bitmasks indicating which edges need filtering.
#[derive(Clone, Debug, Default)]
pub struct LoopFilterMask {
    /// Vertical edge masks for each transform size.
    pub left_y: [u64; 4],
    /// Horizontal edge masks for each transform size.
    pub above_y: [u64; 4],
    /// Vertical edge masks for U plane.
    pub left_u: [u16; 4],
    /// Horizontal edge masks for U plane.
    pub above_u: [u16; 4],
    /// Vertical edge masks for V plane.
    pub left_v: [u16; 4],
    /// Horizontal edge masks for V plane.
    pub above_v: [u16; 4],
}

impl LoopFilterMask {
    /// Create a new empty loop filter mask.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a bit in the Y vertical mask.
    pub fn set_left_y(&mut self, tx_size: usize, row: usize, col: usize, sb_size: usize) {
        if tx_size < 4 && row < sb_size && col < sb_size {
            let bit = row * sb_size + col;
            if bit < 64 {
                self.left_y[tx_size] |= 1u64 << bit;
            }
        }
    }

    /// Set a bit in the Y horizontal mask.
    pub fn set_above_y(&mut self, tx_size: usize, row: usize, col: usize, sb_size: usize) {
        if tx_size < 4 && row < sb_size && col < sb_size {
            let bit = row * sb_size + col;
            if bit < 64 {
                self.above_y[tx_size] |= 1u64 << bit;
            }
        }
    }

    /// Clear all masks.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Check if any edges need filtering.
    #[must_use]
    pub fn has_edges(&self) -> bool {
        self.left_y.iter().any(|&m| m != 0)
            || self.above_y.iter().any(|&m| m != 0)
            || self.left_u.iter().any(|&m| m != 0)
            || self.above_u.iter().any(|&m| m != 0)
            || self.left_v.iter().any(|&m| m != 0)
            || self.above_v.iter().any(|&m| m != 0)
    }
}

/// Loop filter context for block-level decisions.
#[derive(Clone, Debug, Default)]
pub struct LoopFilterContext {
    /// Level lookup table indexed by reference frame and mode.
    level_lookup: [[u8; MAX_MODE_LF_DELTAS + 1]; TOTAL_REFS_PER_FRAME],
}

impl LoopFilterContext {
    /// Create a new loop filter context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize the level lookup table.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    pub fn init(&mut self, params: &LoopFilterParams, base_level: u8) {
        if base_level == 0 || !params.delta_enabled {
            for ref_frame in 0..TOTAL_REFS_PER_FRAME {
                for mode in 0..=MAX_MODE_LF_DELTAS {
                    self.level_lookup[ref_frame][mode] = base_level;
                }
            }
            return;
        }

        for ref_frame in 0..TOTAL_REFS_PER_FRAME {
            for mode in 0..=MAX_MODE_LF_DELTAS {
                let mut level = i32::from(base_level);
                level += i32::from(params.ref_deltas[ref_frame]);
                if ref_frame > 0 && mode < MAX_MODE_LF_DELTAS {
                    level += i32::from(params.mode_deltas[mode]);
                }
                self.level_lookup[ref_frame][mode] =
                    level.clamp(0, i32::from(MAX_LOOP_FILTER_LEVEL)) as u8;
            }
        }
    }

    /// Get the filter level for a block.
    #[must_use]
    pub fn get_level(&self, ref_frame: usize, mode: usize) -> u8 {
        if ref_frame < TOTAL_REFS_PER_FRAME && mode <= MAX_MODE_LF_DELTAS {
            self.level_lookup[ref_frame][mode]
        } else {
            0
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_filter_default() {
        let lf = LoopFilterParams::default();
        assert_eq!(lf.level[0], 0);
        assert_eq!(lf.sharpness, 0);
        assert!(lf.delta_enabled);
        assert!(!lf.is_enabled());
    }

    #[test]
    fn test_loop_filter_is_enabled() {
        let mut lf = LoopFilterParams::default();
        assert!(!lf.is_enabled());

        lf.level[0] = 10;
        assert!(lf.is_enabled());

        lf.level[0] = 0;
        lf.level[2] = 5;
        assert!(lf.is_enabled());
    }

    #[test]
    fn test_loop_filter_get_level() {
        let mut lf = LoopFilterParams::default();
        lf.level = [10, 20, 30, 40];

        assert_eq!(lf.get_level(0, 0), 10); // Y vertical
        assert_eq!(lf.get_level(0, 1), 20); // Y horizontal
        assert_eq!(lf.get_level(1, 0), 30); // U
        assert_eq!(lf.get_level(1, 1), 30); // U (both directions same)
        assert_eq!(lf.get_level(2, 0), 40); // V
        assert_eq!(lf.get_level(3, 0), 0); // Invalid plane
    }

    #[test]
    fn test_loop_filter_accessors() {
        let mut lf = LoopFilterParams::default();
        lf.level = [10, 20, 30, 40];

        assert_eq!(lf.level_y_v(), 10);
        assert_eq!(lf.level_y_h(), 20);
        assert_eq!(lf.level_u(), 30);
        assert_eq!(lf.level_v(), 40);
    }

    #[test]
    fn test_compute_level() {
        let mut lf = LoopFilterParams::default();
        lf.delta_enabled = true;
        lf.ref_deltas = [1, -1, 2, 0, 0, 0, 0, 0];
        lf.mode_deltas = [0, -2];

        // Base level only
        let level = lf.compute_level(30, 0, 0, 0);
        assert_eq!(level, 31); // 30 + ref_delta[0]=1

        // With mode delta
        let level = lf.compute_level(30, 1, 1, 0);
        assert_eq!(level, 27); // 30 + ref_delta[1]=-1 + mode_delta[1]=-2

        // With segmentation delta
        let level = lf.compute_level(30, 0, 0, 10);
        assert_eq!(level, 41); // 30 + 10 + ref_delta[0]=1

        // Clamping to max
        let level = lf.compute_level(60, 0, 0, 10);
        assert_eq!(level, 63); // Clamped to MAX_LOOP_FILTER_LEVEL
    }

    #[test]
    fn test_compute_level_disabled() {
        let mut lf = LoopFilterParams::default();
        lf.delta_enabled = false;
        lf.ref_deltas[0] = 10;

        // Delta disabled, should return base level
        let level = lf.compute_level(30, 0, 0, 0);
        assert_eq!(level, 30);

        // Base level 0 returns 0
        let level = lf.compute_level(0, 0, 0, 0);
        assert_eq!(level, 0);
    }

    #[test]
    fn test_get_limit() {
        let mut lf = LoopFilterParams::default();

        // Sharpness 0
        lf.sharpness = 0;
        assert_eq!(lf.get_limit(30), 30);

        // Sharpness > 0 reduces limit
        lf.sharpness = 4;
        let limit = lf.get_limit(32);
        assert!(limit <= 32);
        assert!(limit <= 9 - 4);
    }

    #[test]
    fn test_get_threshold() {
        let lf = LoopFilterParams::default();
        let limit = lf.get_limit(20);
        let threshold = lf.get_threshold(20);
        assert_eq!(threshold, limit >> 1);
    }

    #[test]
    fn test_get_hev_threshold() {
        let lf = LoopFilterParams::default();

        assert_eq!(lf.get_hev_threshold(10), 0);
        assert_eq!(lf.get_hev_threshold(25), 1);
        assert_eq!(lf.get_hev_threshold(45), 2);
    }

    #[test]
    fn test_has_delta_updates() {
        let mut lf = LoopFilterParams::default();
        lf.delta_enabled = true;
        lf.delta_update = true;
        assert!(lf.has_delta_updates());

        lf.delta_enabled = false;
        assert!(!lf.has_delta_updates());

        lf.delta_enabled = true;
        lf.delta_update = false;
        assert!(!lf.has_delta_updates());
    }

    #[test]
    fn test_reset_deltas() {
        let mut lf = LoopFilterParams::default();
        lf.ref_deltas = [10; TOTAL_REFS_PER_FRAME];
        lf.mode_deltas = [5; MAX_MODE_LF_DELTAS];

        lf.reset_deltas();

        assert_eq!(lf.ref_deltas, DEFAULT_REF_DELTAS);
        assert_eq!(lf.mode_deltas, DEFAULT_MODE_DELTAS);
    }

    #[test]
    fn test_set_level_all() {
        let mut lf = LoopFilterParams::default();
        lf.set_level_all(25);

        assert_eq!(lf.level, [25, 25, 25, 25]);
    }

    #[test]
    fn test_loop_filter_edge() {
        let params = LoopFilterParams::default();
        let edge = LoopFilterEdge::new(&params, 30, 0);

        assert_eq!(edge.level, 30);
        assert_eq!(edge.direction, 0);
        assert!(edge.should_filter());

        let edge_zero = LoopFilterEdge::new(&params, 0, 1);
        assert!(!edge_zero.should_filter());
    }

    #[test]
    fn test_loop_filter_mask() {
        let mut mask = LoopFilterMask::new();
        assert!(!mask.has_edges());

        mask.set_left_y(0, 0, 0, 8);
        assert!(mask.has_edges());
        assert_eq!(mask.left_y[0], 1);

        mask.set_above_y(1, 2, 3, 8);
        assert_eq!(mask.above_y[1], 1u64 << (2 * 8 + 3));

        mask.clear();
        assert!(!mask.has_edges());
    }

    #[test]
    fn test_loop_filter_context() {
        let mut params = LoopFilterParams::default();
        params.delta_enabled = true;
        params.ref_deltas = [2, -1, 0, 0, 0, 0, 0, 0];
        params.mode_deltas = [0, -3];

        let mut ctx = LoopFilterContext::new();
        ctx.init(&params, 30);

        assert_eq!(ctx.get_level(0, 0), 32); // 30 + 2
        assert_eq!(ctx.get_level(1, 0), 29); // 30 - 1 + 0 (mode delta only for inter)
        assert_eq!(ctx.get_level(1, 1), 26); // 30 - 1 - 3
    }

    #[test]
    fn test_loop_filter_context_disabled() {
        let mut params = LoopFilterParams::default();
        params.delta_enabled = false;

        let mut ctx = LoopFilterContext::new();
        ctx.init(&params, 30);

        // All levels should be base level when delta is disabled
        assert_eq!(ctx.get_level(0, 0), 30);
        assert_eq!(ctx.get_level(1, 1), 30);
    }

    #[test]
    fn test_default_deltas() {
        assert_eq!(DEFAULT_REF_DELTAS, [1, 0, 0, 0, 0, -1, -1, -1]);
        assert_eq!(DEFAULT_MODE_DELTAS, [0, 0]);
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_LOOP_FILTER_LEVEL, 63);
        assert_eq!(MAX_SHARPNESS_LEVEL, 7);
        assert_eq!(LF_LEVEL_BITS, 6);
        assert_eq!(LF_DELTA_BITS, 6);
    }
}
