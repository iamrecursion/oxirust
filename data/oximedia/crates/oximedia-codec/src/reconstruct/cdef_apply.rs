//! CDEF (Constrained Directional Enhancement Filter) application.
//!
//! CDEF is a directional filter applied after the loop filter to reduce
//! ringing and mosquito noise artifacts. It uses directional filtering
//! along edges to preserve sharpness while reducing noise.

#![forbid(unsafe_code)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::identity_op)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::single_match_else)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use super::pipeline::FrameContext;
use super::{FrameBuffer, PlaneBuffer, PlaneType, ReconstructResult};

// =============================================================================
// Constants
// =============================================================================

/// CDEF block size (8x8).
pub const CDEF_BLOCK_SIZE: usize = 8;

/// Number of CDEF directions.
pub const CDEF_NUM_DIRECTIONS: usize = 8;

/// Maximum primary strength.
pub const CDEF_MAX_PRIMARY: u8 = 15;

/// Maximum secondary strength.
pub const CDEF_MAX_SECONDARY: u8 = 4;

/// Minimum damping value.
pub const CDEF_DAMPING_MIN: u8 = 3;

/// Maximum damping value.
pub const CDEF_DAMPING_MAX: u8 = 6;

/// Secondary strength values.
pub const CDEF_SEC_STRENGTHS: [u8; 4] = [0, 1, 2, 4];

// =============================================================================
// Direction Offsets
// =============================================================================

/// Direction tap offsets for primary filtering (2 taps per direction).
const CDEF_PRIMARY_OFFSETS: [[(i8, i8); 2]; CDEF_NUM_DIRECTIONS] = [
    [(-1, 0), (1, 0)],  // 0: Horizontal
    [(-1, -1), (1, 1)], // 1: 22.5 degrees
    [(0, -1), (0, 1)],  // 2: Vertical
    [(1, -1), (-1, 1)], // 3: 67.5 degrees
    [(0, -1), (0, 1)],  // 4: Vertical (same as 2)
    [(-1, -1), (1, 1)], // 5: 112.5 degrees
    [(-1, 0), (1, 0)],  // 6: Horizontal (same as 0)
    [(1, -1), (-1, 1)], // 7: 157.5 degrees
];

/// Direction tap offsets for secondary filtering (4 taps per direction).
const CDEF_SECONDARY_OFFSETS: [[(i8, i8); 4]; CDEF_NUM_DIRECTIONS] = [
    [(-2, 0), (2, 0), (-1, -1), (1, 1)],
    [(-2, -1), (2, 1), (-1, -2), (1, 2)],
    [(-1, -1), (1, 1), (-1, -2), (1, 2)],
    [(2, -1), (-2, 1), (1, -2), (-1, 2)],
    [(0, -2), (0, 2), (-1, -1), (1, 1)],
    [(-2, -1), (2, 1), (-1, -2), (1, 2)],
    [(-1, -1), (1, 1), (-2, 0), (2, 0)],
    [(2, -1), (-2, 1), (1, -2), (-1, 2)],
];

// =============================================================================
// CDEF Block Configuration
// =============================================================================

/// Configuration for CDEF on a single block.
#[derive(Clone, Copy, Debug, Default)]
pub struct CdefBlockConfig {
    /// Primary strength (0-15).
    pub primary_strength: u8,
    /// Secondary strength (0-3, maps to 0, 1, 2, 4).
    pub secondary_strength: u8,
    /// Damping value.
    pub damping: u8,
    /// Detected direction (0-7).
    pub direction: u8,
    /// Skip CDEF for this block.
    pub skip: bool,
}

impl CdefBlockConfig {
    /// Create a new block configuration.
    #[must_use]
    pub const fn new(primary: u8, secondary: u8, damping: u8) -> Self {
        Self {
            primary_strength: primary,
            secondary_strength: secondary,
            damping,
            direction: 0,
            skip: false,
        }
    }

    /// Get the actual secondary strength value.
    #[must_use]
    pub fn secondary_value(&self) -> u8 {
        CDEF_SEC_STRENGTHS
            .get(self.secondary_strength as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Check if filtering is enabled.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        !self.skip && (self.primary_strength > 0 || self.secondary_strength > 0)
    }
}

// =============================================================================
// CDEF Filter Result
// =============================================================================

/// Result of CDEF filtering on a block.
#[derive(Clone, Debug, Default)]
pub struct CdefFilterResult {
    /// Direction detected for this block.
    pub direction: u8,
    /// Variance measure.
    pub variance: u32,
    /// Number of pixels modified.
    pub pixels_modified: u32,
}

// =============================================================================
// Direction Detection
// =============================================================================

/// Detect the dominant edge direction in an 8x8 block.
fn detect_direction(block: &[i16], stride: usize) -> (u8, u32) {
    let mut best_dir = 0u8;
    let mut best_var = u32::MAX;

    // Direction detection kernels (simplified)
    const DIR_KERNELS: [[i8; 8]; 8] = [
        [1, 1, 1, 1, -1, -1, -1, -1], // Horizontal
        [1, 1, 1, 0, 0, -1, -1, -1],  // 22.5 deg
        [1, 1, 0, -1, -1, 0, 1, 1],   // Vertical
        [-1, -1, -1, 0, 0, 1, 1, 1],  // 67.5 deg
        [-1, -1, -1, -1, 1, 1, 1, 1], // Horizontal flip
        [-1, -1, 0, 1, 1, 0, -1, -1], // 112.5 deg
        [-1, -1, -1, -1, 1, 1, 1, 1], // Vertical flip
        [1, 1, 1, 0, 0, -1, -1, -1],  // 157.5 deg
    ];

    for (dir, kernel) in DIR_KERNELS.iter().enumerate() {
        let mut sum: i32 = 0;
        let mut sum_sq: i32 = 0;

        for row in 0..8 {
            for col in 0..8 {
                let pixel = i32::from(block[row * stride + col]);
                let weight = i32::from(kernel[(row + col) % 8]);
                let val = pixel * weight;
                sum += val;
                sum_sq += val * val;
            }
        }

        // Compute variance-like measure
        let variance = (sum_sq - (sum * sum / 64)).unsigned_abs();

        if variance < best_var {
            best_var = variance;
            best_dir = dir as u8;
        }
    }

    (best_dir, best_var)
}

// =============================================================================
// CDEF Filtering
// =============================================================================

/// Compute the constrained difference.
fn constrain(diff: i16, threshold: i16, damping: u8) -> i16 {
    if threshold == 0 {
        return 0;
    }

    let sign = if diff < 0 { -1i16 } else { 1i16 };
    let abs_diff = diff.abs();
    let abs_thresh = threshold.abs();

    let damping_shift = damping.saturating_sub(abs_thresh as u8);

    if damping_shift >= 15 {
        return 0;
    }

    let clamped = abs_diff.min(abs_thresh);
    let adjusted = clamped - (clamped >> damping_shift);

    sign * adjusted
}

/// Apply CDEF filter to a single 8x8 block.
fn filter_block(
    src: &[i16],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    config: &CdefBlockConfig,
    bd: u8,
) {
    let max_val = (1i16 << bd) - 1;
    let primary = i16::from(config.primary_strength);
    let secondary = i16::from(config.secondary_value());
    let damping = config.damping;
    let dir = config.direction as usize;

    for row in 0..CDEF_BLOCK_SIZE {
        for col in 0..CDEF_BLOCK_SIZE {
            let src_idx = row * src_stride + col;
            let dst_idx = row * dst_stride + col;

            let center = src[src_idx];
            let mut sum: i32 = 0;

            // Primary filter (along edge)
            if primary > 0 {
                for &(dx, dy) in &CDEF_PRIMARY_OFFSETS[dir] {
                    let nx = col as i32 + i32::from(dx);
                    let ny = row as i32 + i32::from(dy);

                    if nx >= 0
                        && nx < CDEF_BLOCK_SIZE as i32
                        && ny >= 0
                        && ny < CDEF_BLOCK_SIZE as i32
                    {
                        let neighbor_idx = ny as usize * src_stride + nx as usize;
                        let neighbor = src[neighbor_idx];
                        let diff = neighbor - center;
                        sum += i32::from(constrain(diff, primary, damping)) * 2;
                    }
                }
            }

            // Secondary filter (perpendicular)
            if secondary > 0 {
                for &(dx, dy) in &CDEF_SECONDARY_OFFSETS[dir] {
                    let nx = col as i32 + i32::from(dx);
                    let ny = row as i32 + i32::from(dy);

                    if nx >= 0
                        && nx < CDEF_BLOCK_SIZE as i32
                        && ny >= 0
                        && ny < CDEF_BLOCK_SIZE as i32
                    {
                        let neighbor_idx = ny as usize * src_stride + nx as usize;
                        let neighbor = src[neighbor_idx];
                        let diff = neighbor - center;
                        sum += i32::from(constrain(diff, secondary, damping));
                    }
                }
            }

            // Apply filter result
            let filtered = i32::from(center) + ((sum + 8) >> 4);
            dst[dst_idx] = (filtered as i16).clamp(0, max_val);
        }
    }
}

// =============================================================================
// CDEF Applicator
// =============================================================================

/// CDEF applicator for applying CDEF to frames.
#[derive(Debug)]
pub struct CdefApplicator {
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Bit depth.
    bit_depth: u8,
    /// Y damping.
    damping_y: u8,
    /// UV damping.
    damping_uv: u8,
    /// Y strength presets.
    y_presets: Vec<(u8, u8)>,
    /// UV strength presets.
    uv_presets: Vec<(u8, u8)>,
    /// Temporary buffer for filtering.
    temp_buffer: Vec<i16>,
}

impl CdefApplicator {
    /// Create a new CDEF applicator.
    #[must_use]
    pub fn new(width: u32, height: u32, bit_depth: u8) -> Self {
        let buffer_size = CDEF_BLOCK_SIZE * CDEF_BLOCK_SIZE;

        Self {
            width,
            height,
            bit_depth,
            damping_y: CDEF_DAMPING_MIN,
            damping_uv: CDEF_DAMPING_MIN,
            y_presets: vec![(0, 0); 8],
            uv_presets: vec![(0, 0); 8],
            temp_buffer: vec![0i16; buffer_size],
        }
    }

    /// Set damping values.
    pub fn set_damping(&mut self, y_damping: u8, uv_damping: u8) {
        self.damping_y = y_damping.clamp(CDEF_DAMPING_MIN, CDEF_DAMPING_MAX);
        self.damping_uv = uv_damping.clamp(CDEF_DAMPING_MIN, CDEF_DAMPING_MAX);
    }

    /// Set Y strength preset.
    pub fn set_y_preset(&mut self, index: usize, primary: u8, secondary: u8) {
        if index < self.y_presets.len() {
            self.y_presets[index] = (primary.min(CDEF_MAX_PRIMARY), secondary.min(3));
        }
    }

    /// Set UV strength preset.
    pub fn set_uv_preset(&mut self, index: usize, primary: u8, secondary: u8) {
        if index < self.uv_presets.len() {
            self.uv_presets[index] = (primary.min(CDEF_MAX_PRIMARY), secondary.min(3));
        }
    }

    /// Get damping for a plane.
    #[must_use]
    pub fn damping(&self, plane: PlaneType) -> u8 {
        match plane {
            PlaneType::Y => self.damping_y,
            PlaneType::U | PlaneType::V => self.damping_uv,
        }
    }

    /// Get adjusted damping for bit depth.
    #[must_use]
    pub fn adjusted_damping(&self, plane: PlaneType) -> u8 {
        let base = self.damping(plane);
        base + self.bit_depth.saturating_sub(8).min(4)
    }

    /// Apply CDEF to a frame.
    ///
    /// # Errors
    ///
    /// Returns error if CDEF application fails.
    pub fn apply(
        &mut self,
        frame: &mut FrameBuffer,
        _context: &FrameContext,
    ) -> ReconstructResult<()> {
        let bd = frame.bit_depth();

        // Apply to Y plane
        self.apply_to_plane(frame.y_plane_mut(), PlaneType::Y, bd)?;

        // Apply to chroma planes
        if let Some(u) = frame.u_plane_mut() {
            self.apply_to_plane(u, PlaneType::U, bd)?;
        }
        if let Some(v) = frame.v_plane_mut() {
            self.apply_to_plane(v, PlaneType::V, bd)?;
        }

        Ok(())
    }

    /// Apply CDEF to a single plane.
    fn apply_to_plane(
        &mut self,
        plane: &mut PlaneBuffer,
        plane_type: PlaneType,
        bd: u8,
    ) -> ReconstructResult<()> {
        let width = plane.width() as usize;
        let height = plane.height() as usize;
        let stride = plane.stride();

        let damping = self.adjusted_damping(plane_type);

        // Process each 8x8 block
        let blocks_x = width / CDEF_BLOCK_SIZE;
        let blocks_y = height / CDEF_BLOCK_SIZE;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x = bx * CDEF_BLOCK_SIZE;
                let y = by * CDEF_BLOCK_SIZE;

                // Get block data
                let block_start = y * stride + x;
                let block_data = &plane.data()[block_start..];

                // Detect direction
                let (direction, _variance) = detect_direction(block_data, stride);

                // Get strength from preset (use preset 0 for simplicity)
                let (primary, secondary) = match plane_type {
                    PlaneType::Y => self.y_presets.first().copied().unwrap_or((0, 0)),
                    _ => self.uv_presets.first().copied().unwrap_or((0, 0)),
                };

                // Skip if no filtering needed
                if primary == 0 && secondary == 0 {
                    continue;
                }

                // Create block config
                let config = CdefBlockConfig {
                    primary_strength: primary,
                    secondary_strength: secondary,
                    damping,
                    direction,
                    skip: false,
                };

                // Filter the block
                filter_block(
                    block_data,
                    stride,
                    &mut self.temp_buffer,
                    CDEF_BLOCK_SIZE,
                    &config,
                    bd,
                );

                // Write back filtered data
                let plane_data = plane.data_mut();
                for row in 0..CDEF_BLOCK_SIZE {
                    for col in 0..CDEF_BLOCK_SIZE {
                        let src_idx = row * CDEF_BLOCK_SIZE + col;
                        let dst_idx = (y + row) * stride + (x + col);
                        plane_data[dst_idx] = self.temp_buffer[src_idx];
                    }
                }
            }
        }

        Ok(())
    }

    /// Filter a single block with given parameters.
    pub fn filter_single_block(
        &mut self,
        plane: &mut PlaneBuffer,
        x: u32,
        y: u32,
        config: &CdefBlockConfig,
    ) -> CdefFilterResult {
        let bd = plane.bit_depth();
        let stride = plane.stride();

        // Get block data
        let block_start = y as usize * stride + x as usize;
        let block_data = &plane.data()[block_start..];

        // Detect direction if not specified
        let (direction, variance) = if config.direction == 0 {
            detect_direction(block_data, stride)
        } else {
            (config.direction, 0)
        };

        let mut actual_config = *config;
        actual_config.direction = direction;

        if !actual_config.is_enabled() {
            return CdefFilterResult {
                direction,
                variance,
                pixels_modified: 0,
            };
        }

        // Filter the block
        filter_block(
            block_data,
            stride,
            &mut self.temp_buffer,
            CDEF_BLOCK_SIZE,
            &actual_config,
            bd,
        );

        // Count modified pixels and write back
        let mut pixels_modified = 0u32;
        let plane_data = plane.data_mut();

        for row in 0..CDEF_BLOCK_SIZE {
            for col in 0..CDEF_BLOCK_SIZE {
                let src_idx = row * CDEF_BLOCK_SIZE + col;
                let dst_idx = (y as usize + row) * stride + (x as usize + col);

                if plane_data[dst_idx] != self.temp_buffer[src_idx] {
                    pixels_modified += 1;
                }
                plane_data[dst_idx] = self.temp_buffer[src_idx];
            }
        }

        CdefFilterResult {
            direction,
            variance,
            pixels_modified,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruct::ChromaSubsampling;

    #[test]
    fn test_cdef_block_config() {
        let config = CdefBlockConfig::new(8, 2, 4);
        assert_eq!(config.primary_strength, 8);
        assert_eq!(config.secondary_strength, 2);
        assert_eq!(config.secondary_value(), 2);
        assert!(config.is_enabled());

        let config_disabled = CdefBlockConfig::new(0, 0, 4);
        assert!(!config_disabled.is_enabled());
    }

    #[test]
    fn test_cdef_secondary_values() {
        let mut config = CdefBlockConfig::new(0, 0, 4);
        assert_eq!(config.secondary_value(), 0);

        config.secondary_strength = 1;
        assert_eq!(config.secondary_value(), 1);

        config.secondary_strength = 2;
        assert_eq!(config.secondary_value(), 2);

        config.secondary_strength = 3;
        assert_eq!(config.secondary_value(), 4);
    }

    #[test]
    fn test_constrain() {
        // Zero threshold returns zero
        assert_eq!(constrain(10, 0, 3), 0);

        // Positive difference
        let result = constrain(5, 10, 3);
        assert!(result >= 0 && result <= 5);

        // Negative difference
        let result = constrain(-5, 10, 3);
        assert!(result <= 0 && result >= -5);
    }

    #[test]
    fn test_detect_direction() {
        // Test direction detection returns valid direction
        let block = vec![128i16; 64];
        let (dir, var) = detect_direction(&block, 8);
        assert!(dir < 8); // Direction should be 0-7
        let _ = var; // Variance depends on block content
    }

    #[test]
    fn test_cdef_applicator_creation() {
        let applicator = CdefApplicator::new(1920, 1080, 8);
        assert_eq!(applicator.width, 1920);
        assert_eq!(applicator.height, 1080);
        assert_eq!(applicator.bit_depth, 8);
    }

    #[test]
    fn test_cdef_applicator_damping() {
        let mut applicator = CdefApplicator::new(64, 64, 8);
        applicator.set_damping(4, 5);

        assert_eq!(applicator.damping(PlaneType::Y), 4);
        assert_eq!(applicator.damping(PlaneType::U), 5);
        assert_eq!(applicator.damping(PlaneType::V), 5);
    }

    #[test]
    fn test_cdef_applicator_adjusted_damping() {
        let applicator = CdefApplicator::new(64, 64, 10);
        let adj_damping = applicator.adjusted_damping(PlaneType::Y);

        // 10-bit adds 2 to base damping
        assert_eq!(adj_damping, CDEF_DAMPING_MIN + 2);
    }

    #[test]
    fn test_cdef_applicator_presets() {
        let mut applicator = CdefApplicator::new(64, 64, 8);
        applicator.set_y_preset(0, 8, 2);
        applicator.set_uv_preset(0, 4, 1);

        assert_eq!(applicator.y_presets[0], (8, 2));
        assert_eq!(applicator.uv_presets[0], (4, 1));
    }

    #[test]
    fn test_cdef_applicator_apply() {
        let mut frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);
        let context = FrameContext::new(64, 64);

        let mut applicator = CdefApplicator::new(64, 64, 8);
        applicator.set_y_preset(0, 4, 1);

        let result = applicator.apply(&mut frame, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_filter_block() {
        let src = vec![128i16; 64];
        let mut dst = vec![0i16; 64];

        let config = CdefBlockConfig::new(4, 1, 4);

        filter_block(&src, 8, &mut dst, 8, &config, 8);

        // Uniform input should produce similar output
        for &val in &dst {
            assert!((val - 128).abs() < 10);
        }
    }

    #[test]
    fn test_filter_single_block() {
        let mut frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);

        // Set some test values
        for y in 0..8 {
            for x in 0..8 {
                frame.y_plane_mut().set(x, y, 128);
            }
        }

        let mut applicator = CdefApplicator::new(64, 64, 8);
        let config = CdefBlockConfig::new(4, 1, 4);

        let result = applicator.filter_single_block(frame.y_plane_mut(), 0, 0, &config);

        assert!(result.direction < CDEF_NUM_DIRECTIONS as u8);
    }

    #[test]
    fn test_constants() {
        assert_eq!(CDEF_BLOCK_SIZE, 8);
        assert_eq!(CDEF_NUM_DIRECTIONS, 8);
        assert_eq!(CDEF_MAX_PRIMARY, 15);
        assert_eq!(CDEF_MAX_SECONDARY, 4);
        assert_eq!(CDEF_DAMPING_MIN, 3);
        assert_eq!(CDEF_DAMPING_MAX, 6);
    }
}
