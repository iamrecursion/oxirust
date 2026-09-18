//! VP9 Loop filter information.
//!
//! This module provides structures for loop filter parameters used to
//! reduce blocking artifacts at block boundaries after decoding.

#![forbid(unsafe_code)]
#![allow(dead_code)]

use super::mv::MvRefType;

/// Maximum loop filter level.
pub const MAX_LOOP_FILTER: i32 = 63;

/// Maximum sharpness level.
pub const MAX_SHARPNESS: u8 = 7;

/// Number of reference frames for delta.
pub const REF_FRAMES: usize = 4;

/// Number of modes for delta.
pub const MODE_DELTAS: usize = 2;

/// Loop filter information parsed from frame header.
#[derive(Clone, Debug, Default)]
pub struct LoopFilterInfo {
    /// Base loop filter level (0-63).
    pub level: u8,
    /// Sharpness level (0-7).
    pub sharpness: u8,
    /// Delta values enabled.
    pub delta_enabled: bool,
    /// Delta values are updated in this frame.
    pub delta_update: bool,
    /// Reference frame deltas.
    pub ref_deltas: [i8; REF_FRAMES],
    /// Mode deltas (zero and non-zero motion).
    pub mode_deltas: [i8; MODE_DELTAS],
}

impl LoopFilterInfo {
    /// Default reference deltas.
    const DEFAULT_REF_DELTAS: [i8; REF_FRAMES] = [1, 0, -1, -1];

    /// Default mode deltas.
    const DEFAULT_MODE_DELTAS: [i8; MODE_DELTAS] = [0, 0];

    /// Creates a new loop filter info with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            level: 0,
            sharpness: 0,
            delta_enabled: true,
            delta_update: false,
            ref_deltas: Self::DEFAULT_REF_DELTAS,
            mode_deltas: Self::DEFAULT_MODE_DELTAS,
        }
    }

    /// Resets to default values.
    pub fn reset(&mut self) {
        self.level = 0;
        self.sharpness = 0;
        self.delta_enabled = true;
        self.delta_update = false;
        self.ref_deltas = Self::DEFAULT_REF_DELTAS;
        self.mode_deltas = Self::DEFAULT_MODE_DELTAS;
    }

    /// Returns true if the loop filter is enabled.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.level > 0
    }

    /// Returns the reference delta for a given reference type.
    #[must_use]
    pub fn ref_delta(&self, ref_type: MvRefType) -> i8 {
        self.ref_deltas[ref_type.index()]
    }

    /// Returns the mode delta for a given mode index.
    #[must_use]
    pub fn mode_delta(&self, mode_index: usize) -> i8 {
        if mode_index < MODE_DELTAS {
            self.mode_deltas[mode_index]
        } else {
            0
        }
    }

    /// Calculates the effective filter level for a block.
    ///
    /// The effective level is the base level adjusted by reference and mode deltas.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn effective_level(&self, ref_type: MvRefType, is_zero_mv: bool) -> u8 {
        if !self.delta_enabled {
            return self.level;
        }

        let ref_delta = self.ref_delta(ref_type);
        let mode_delta = if is_zero_mv {
            self.mode_deltas[0]
        } else {
            self.mode_deltas[1]
        };

        let level = i32::from(self.level) + i32::from(ref_delta) + i32::from(mode_delta);
        level.clamp(0, MAX_LOOP_FILTER) as u8
    }

    /// Sets the reference delta for a given index.
    pub fn set_ref_delta(&mut self, index: usize, delta: i8) {
        if index < REF_FRAMES {
            self.ref_deltas[index] = delta;
        }
    }

    /// Sets the mode delta for a given index.
    pub fn set_mode_delta(&mut self, index: usize, delta: i8) {
        if index < MODE_DELTAS {
            self.mode_deltas[index] = delta;
        }
    }
}

/// Loop filter mask for block boundaries.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoopFilterMask {
    /// Left edge mask (vertical filter).
    pub left: u64,
    /// Above edge mask (horizontal filter).
    pub above: u64,
    /// Interior mask (4x4 boundaries).
    pub interior: u64,
}

impl LoopFilterMask {
    /// Creates an empty mask.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            left: 0,
            above: 0,
            interior: 0,
        }
    }

    /// Creates a full mask for a superblock.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            left: u64::MAX,
            above: u64::MAX,
            interior: u64::MAX,
        }
    }

    /// Returns true if any edges need filtering.
    #[must_use]
    pub const fn has_edges(&self) -> bool {
        self.left != 0 || self.above != 0 || self.interior != 0
    }

    /// Clears all masks.
    pub fn clear(&mut self) {
        self.left = 0;
        self.above = 0;
        self.interior = 0;
    }
}

/// Loop filter parameters per edge.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoopFilterParams {
    /// Filter level for this edge.
    pub level: u8,
    /// Edge limit based on level and sharpness.
    pub limit: u8,
    /// Block limit (blimit).
    pub blimit: u8,
    /// Threshold for flat regions.
    pub thresh: u8,
}

impl LoopFilterParams {
    /// Limit lookup table based on level.
    const LVL_TO_LIM: [u8; 64] = [
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4,
    ];

    /// Block limit lookup table based on level.
    const LVL_TO_BLIM: [u8; 64] = [
        0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7,
        7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
        11, 11, 12, 12, 12, 12, 12,
    ];

    /// Threshold lookup table based on level.
    const LVL_TO_THRESH: [u8; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3,
    ];

    /// Creates loop filter parameters for a given level and sharpness.
    #[must_use]
    pub fn from_level(level: u8, sharpness: u8) -> Self {
        if level == 0 {
            return Self::default();
        }

        let level_idx = level.min(63) as usize;
        let mut limit = Self::LVL_TO_LIM[level_idx];

        // Adjust limit based on sharpness
        if sharpness > 0 {
            limit >>= (sharpness + 3) >> 2;
            limit = limit.min(9 - sharpness);
        }
        limit = limit.max(1);

        Self {
            level,
            limit,
            blimit: Self::LVL_TO_BLIM[level_idx],
            thresh: Self::LVL_TO_THRESH[level_idx],
        }
    }

    /// Returns true if filtering should be applied.
    #[must_use]
    pub const fn should_filter(&self) -> bool {
        self.level > 0
    }
}

/// Loop filter state for a frame.
#[derive(Clone, Debug)]
pub struct LoopFilterState {
    /// Loop filter info from frame header.
    pub info: LoopFilterInfo,
    /// Pre-computed parameters for each level.
    params_cache: Vec<LoopFilterParams>,
    /// Cache valid flag.
    cache_valid: bool,
}

impl Default for LoopFilterState {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopFilterState {
    /// Creates a new loop filter state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            info: LoopFilterInfo::new(),
            params_cache: vec![LoopFilterParams::default(); 64],
            cache_valid: false,
        }
    }

    /// Updates the loop filter info and invalidates cache.
    pub fn update(&mut self, info: LoopFilterInfo) {
        if self.info.sharpness != info.sharpness {
            self.cache_valid = false;
        }
        self.info = info;
    }

    /// Returns the loop filter parameters for a given level.
    #[must_use]
    pub fn params(&mut self, level: u8) -> LoopFilterParams {
        if !self.cache_valid {
            self.rebuild_cache();
        }
        self.params_cache[level.min(63) as usize]
    }

    /// Rebuilds the parameter cache.
    #[allow(clippy::cast_possible_truncation)]
    fn rebuild_cache(&mut self) {
        for level in 0..64 {
            self.params_cache[level] =
                LoopFilterParams::from_level(level as u8, self.info.sharpness);
        }
        self.cache_valid = true;
    }

    /// Returns the effective level for a block.
    #[must_use]
    pub fn block_level(&self, ref_type: MvRefType, is_zero_mv: bool) -> u8 {
        self.info.effective_level(ref_type, is_zero_mv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_filter_info_new() {
        let info = LoopFilterInfo::new();
        assert_eq!(info.level, 0);
        assert_eq!(info.sharpness, 0);
        assert!(info.delta_enabled);
        assert!(!info.delta_update);
    }

    #[test]
    fn test_loop_filter_info_enabled() {
        let mut info = LoopFilterInfo::new();
        assert!(!info.is_enabled());
        info.level = 32;
        assert!(info.is_enabled());
    }

    #[test]
    fn test_loop_filter_info_effective_level() {
        let mut info = LoopFilterInfo::new();
        info.level = 32;
        info.delta_enabled = true;
        info.ref_deltas = [2, -1, 0, 1];
        info.mode_deltas = [1, -1];

        // INTRA with zero MV
        assert_eq!(info.effective_level(MvRefType::Intra, true), 35); // 32 + 2 + 1

        // LAST with non-zero MV
        assert_eq!(info.effective_level(MvRefType::Last, false), 30); // 32 + (-1) + (-1)
    }

    #[test]
    fn test_loop_filter_info_clamped() {
        let mut info = LoopFilterInfo::new();
        info.level = 60;
        info.delta_enabled = true;
        info.ref_deltas = [10, 0, 0, 0];
        info.mode_deltas = [5, 0];

        // Should clamp to MAX_LOOP_FILTER
        assert_eq!(info.effective_level(MvRefType::Intra, true), 63);
    }

    #[test]
    fn test_loop_filter_info_delta_disabled() {
        let mut info = LoopFilterInfo::new();
        info.level = 32;
        info.delta_enabled = false;
        info.ref_deltas = [10, 10, 10, 10];

        // Should ignore deltas
        assert_eq!(info.effective_level(MvRefType::Intra, true), 32);
    }

    #[test]
    fn test_loop_filter_mask() {
        let mask = LoopFilterMask::empty();
        assert!(!mask.has_edges());

        let full_mask = LoopFilterMask::full();
        assert!(full_mask.has_edges());
    }

    #[test]
    fn test_loop_filter_params() {
        let params = LoopFilterParams::from_level(0, 0);
        assert!(!params.should_filter());

        let params32 = LoopFilterParams::from_level(32, 0);
        assert!(params32.should_filter());
        assert!(params32.limit > 0);
        assert!(params32.blimit > 0);
    }

    #[test]
    fn test_loop_filter_params_sharpness() {
        let params_s0 = LoopFilterParams::from_level(32, 0);
        let params_s4 = LoopFilterParams::from_level(32, 4);

        // Higher sharpness should reduce limit
        assert!(params_s4.limit <= params_s0.limit);
    }

    #[test]
    fn test_loop_filter_state() {
        let mut state = LoopFilterState::new();
        state.info.level = 32;
        state.info.sharpness = 2;

        let params = state.params(32);
        assert!(params.should_filter());
    }

    #[test]
    fn test_loop_filter_set_deltas() {
        let mut info = LoopFilterInfo::new();
        info.set_ref_delta(0, 5);
        info.set_mode_delta(1, -3);

        assert_eq!(info.ref_deltas[0], 5);
        assert_eq!(info.mode_deltas[1], -3);
    }

    #[test]
    fn test_loop_filter_reset() {
        let mut info = LoopFilterInfo::new();
        info.level = 50;
        info.sharpness = 5;
        info.ref_deltas = [10, 10, 10, 10];

        info.reset();

        assert_eq!(info.level, 0);
        assert_eq!(info.sharpness, 0);
        assert_eq!(info.ref_deltas, LoopFilterInfo::DEFAULT_REF_DELTAS);
    }
}
