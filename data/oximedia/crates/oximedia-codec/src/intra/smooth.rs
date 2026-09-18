//! Smooth prediction implementations (AV1).
//!
//! Smooth prediction uses weighted interpolation between neighboring samples
//! to create gradual transitions. Three variants are provided:
//!
//! - **SMOOTH**: Bilinear interpolation between all neighbors
//! - **SMOOTH_V**: Vertical interpolation (top to bottom-left)
//! - **SMOOTH_H**: Horizontal interpolation (left to top-right)
//!
//! Weight tables define the blending factors at each position.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_range_loop)]

use super::{BlockDimensions, IntraPredContext, IntraPredictor};

/// Smooth weight tables for different block sizes.
/// Weights are in 1/256ths and decrease from edge to center.
pub mod weights {
    /// Weights for 4-sample blocks.
    pub const SMOOTH_WEIGHTS_4: [u16; 4] = [255, 149, 85, 64];

    /// Weights for 8-sample blocks.
    pub const SMOOTH_WEIGHTS_8: [u16; 8] = [255, 197, 146, 105, 73, 50, 37, 32];

    /// Weights for 16-sample blocks.
    pub const SMOOTH_WEIGHTS_16: [u16; 16] = [
        255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
    ];

    /// Weights for 32-sample blocks.
    pub const SMOOTH_WEIGHTS_32: [u16; 32] = [
        255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74, 66, 59, 52,
        45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
    ];

    /// Weights for 64-sample blocks.
    pub const SMOOTH_WEIGHTS_64: [u16; 64] = [
        255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156, 150, 144, 138,
        133, 127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69, 65, 61, 57, 54, 50, 47, 44,
        41, 38, 35, 32, 29, 27, 25, 22, 20, 18, 16, 15, 13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
    ];

    /// Get weight table for a given size.
    #[must_use]
    pub fn get_weights(size: usize) -> &'static [u16] {
        match size {
            4 => &SMOOTH_WEIGHTS_4,
            8 => &SMOOTH_WEIGHTS_8,
            16 => &SMOOTH_WEIGHTS_16,
            32 => &SMOOTH_WEIGHTS_32,
            64 => &SMOOTH_WEIGHTS_64,
            _ => {
                // For sizes > 64, use the 64 table
                if size > 64 {
                    &SMOOTH_WEIGHTS_64
                } else {
                    // Fallback to nearest smaller
                    if size > 32 {
                        &SMOOTH_WEIGHTS_32
                    } else if size > 16 {
                        &SMOOTH_WEIGHTS_16
                    } else if size > 8 {
                        &SMOOTH_WEIGHTS_8
                    } else {
                        &SMOOTH_WEIGHTS_4
                    }
                }
            }
        }
    }

    /// Interpolate weight for sizes not in the table.
    #[must_use]
    pub fn interpolate_weight(size: usize, idx: usize) -> u16 {
        let weights = get_weights(size);
        let table_size = weights.len();

        if size == table_size {
            return weights[idx];
        }

        // Scale index to table size
        let scaled_idx = (idx * table_size) / size;
        let frac = ((idx * table_size) % size) * 256 / size;

        let w0 = weights[scaled_idx.min(table_size - 1)];
        let w1 = weights[(scaled_idx + 1).min(table_size - 1)];

        // Linear interpolation
        let w0_32 = u32::from(w0);
        let w1_32 = u32::from(w1);
        let result = (w0_32 * (256 - frac as u32) + w1_32 * frac as u32 + 128) / 256;
        result as u16
    }
}

/// Smooth predictor (bilinear interpolation).
#[derive(Clone, Copy, Debug, Default)]
pub struct SmoothPredictor;

impl SmoothPredictor {
    /// Create a new smooth predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Perform smooth prediction.
    pub fn predict_smooth(
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let top = ctx.top_samples();
        let left = ctx.left_samples();

        // Get bottom-left and top-right samples for interpolation
        let bottom_left = left[dims.height.saturating_sub(1)];
        let top_right = top[dims.width.saturating_sub(1)];

        let weights_x = weights::get_weights(dims.width);
        let weights_y = weights::get_weights(dims.height);

        for y in 0..dims.height {
            let row_start = y * stride;
            let weight_y = if y < weights_y.len() {
                weights_y[y]
            } else {
                weights::interpolate_weight(dims.height, y)
            };

            for x in 0..dims.width {
                let weight_x = if x < weights_x.len() {
                    weights_x[x]
                } else {
                    weights::interpolate_weight(dims.width, x)
                };

                // Bilinear interpolation
                // pred = (weight_y * top[x] + (256 - weight_y) * bottom_left
                //       + weight_x * left[y] + (256 - weight_x) * top_right + 256) / 512
                let top_sample = u32::from(top[x]);
                let left_sample = u32::from(left[y]);
                let bl = u32::from(bottom_left);
                let tr = u32::from(top_right);

                let wy = u32::from(weight_y);
                let wx = u32::from(weight_x);

                let vertical = wy * top_sample + (256 - wy) * bl;
                let horizontal = wx * left_sample + (256 - wx) * tr;

                let pred = (vertical + horizontal + 256) / 512;
                output[row_start + x] = pred as u16;
            }
        }
    }
}

impl IntraPredictor for SmoothPredictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        Self::predict_smooth(ctx, output, stride, dims);
    }
}

/// Smooth-V predictor (vertical smooth).
#[derive(Clone, Copy, Debug, Default)]
pub struct SmoothVPredictor;

impl SmoothVPredictor {
    /// Create a new smooth-V predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Perform smooth-V prediction.
    pub fn predict_smooth_v(
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let top = ctx.top_samples();
        let left = ctx.left_samples();

        // Bottom-left sample for vertical interpolation
        let bottom_left = left[dims.height.saturating_sub(1)];

        let weights_y = weights::get_weights(dims.height);

        for y in 0..dims.height {
            let row_start = y * stride;
            let weight_y = if y < weights_y.len() {
                weights_y[y]
            } else {
                weights::interpolate_weight(dims.height, y)
            };

            for x in 0..dims.width {
                // Vertical interpolation only
                // pred = (weight_y * top[x] + (256 - weight_y) * bottom_left + 128) / 256
                let top_sample = u32::from(top[x]);
                let bl = u32::from(bottom_left);
                let wy = u32::from(weight_y);

                let pred = (wy * top_sample + (256 - wy) * bl + 128) / 256;
                output[row_start + x] = pred as u16;
            }
        }
    }
}

impl IntraPredictor for SmoothVPredictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        Self::predict_smooth_v(ctx, output, stride, dims);
    }
}

/// Smooth-H predictor (horizontal smooth).
#[derive(Clone, Copy, Debug, Default)]
pub struct SmoothHPredictor;

impl SmoothHPredictor {
    /// Create a new smooth-H predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Perform smooth-H prediction.
    pub fn predict_smooth_h(
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let top = ctx.top_samples();
        let left = ctx.left_samples();

        // Top-right sample for horizontal interpolation
        let top_right = top[dims.width.saturating_sub(1)];

        let weights_x = weights::get_weights(dims.width);

        for y in 0..dims.height {
            let row_start = y * stride;
            let left_sample = u32::from(left[y]);

            for x in 0..dims.width {
                let weight_x = if x < weights_x.len() {
                    weights_x[x]
                } else {
                    weights::interpolate_weight(dims.width, x)
                };

                // Horizontal interpolation only
                // pred = (weight_x * left[y] + (256 - weight_x) * top_right + 128) / 256
                let tr = u32::from(top_right);
                let wx = u32::from(weight_x);

                let pred = (wx * left_sample + (256 - wx) * tr + 128) / 256;
                output[row_start + x] = pred as u16;
            }
        }
    }
}

impl IntraPredictor for SmoothHPredictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        Self::predict_smooth_h(ctx, output, stride, dims);
    }
}

/// Bilinear interpolation helper for smooth modes.
#[inline]
pub fn bilinear_interpolate(
    top: u16,
    left: u16,
    bottom_left: u16,
    top_right: u16,
    weight_x: u16,
    weight_y: u16,
) -> u16 {
    let t = u32::from(top);
    let l = u32::from(left);
    let bl = u32::from(bottom_left);
    let tr = u32::from(top_right);
    let wx = u32::from(weight_x);
    let wy = u32::from(weight_y);

    // Bilinear blend
    let vertical = wy * t + (256 - wy) * bl;
    let horizontal = wx * l + (256 - wx) * tr;

    let result = (vertical + horizontal + 256) / 512;
    result as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra::context::IntraPredContext;
    use crate::intra::BitDepth;

    fn create_test_context() -> IntraPredContext {
        let mut ctx = IntraPredContext::new(8, 8, BitDepth::Bits8);

        // Set uniform top samples: all 200
        for i in 0..16 {
            ctx.set_top_sample(i, 200);
        }

        // Set uniform left samples: all 100
        for i in 0..16 {
            ctx.set_left_sample(i, 100);
        }

        ctx.set_top_left_sample(150);
        ctx.set_availability(true, true);

        ctx
    }

    #[test]
    fn test_smooth_weights() {
        // Check weight tables exist and are decreasing
        let w4 = weights::get_weights(4);
        assert_eq!(w4.len(), 4);
        assert!(w4[0] > w4[3]);

        let w8 = weights::get_weights(8);
        assert_eq!(w8.len(), 8);
        assert!(w8[0] > w8[7]);

        let w16 = weights::get_weights(16);
        assert_eq!(w16.len(), 16);
        assert_eq!(w16[0], 255);
    }

    #[test]
    fn test_smooth_prediction() {
        let ctx = create_test_context();
        let predictor = SmoothPredictor::new();
        let dims = BlockDimensions::new(4, 4);
        let mut output = vec![0u16; 16];

        predictor.predict(&ctx, &mut output, 4, dims);

        // All outputs should be between 100 and 200 (the left and top values)
        for &val in &output {
            assert!(val >= 100 && val <= 200, "Value {} out of range", val);
        }

        // Top-left corner should be closer to average
        // Bottom-right corner should blend more
        assert!(output[0] >= output[15] - 50);
    }

    #[test]
    fn test_smooth_v_prediction() {
        let ctx = create_test_context();
        let predictor = SmoothVPredictor::new();
        let dims = BlockDimensions::new(4, 4);
        let mut output = vec![0u16; 16];

        predictor.predict(&ctx, &mut output, 4, dims);

        // Each row should have the same value (vertical interpolation)
        for y in 0..4 {
            let row_start = y * 4;
            let first = output[row_start];
            for x in 1..4 {
                assert_eq!(output[row_start + x], first, "Row {} not uniform", y);
            }
        }

        // Values should decrease from top to bottom (top=200, bottom_left=100)
        assert!(output[0] > output[12]);
    }

    #[test]
    fn test_smooth_h_prediction() {
        let ctx = create_test_context();
        let predictor = SmoothHPredictor::new();
        let dims = BlockDimensions::new(4, 4);
        let mut output = vec![0u16; 16];

        predictor.predict(&ctx, &mut output, 4, dims);

        // Each column should have the same value (horizontal interpolation)
        for x in 0..4 {
            let first = output[x];
            for y in 1..4 {
                assert_eq!(output[y * 4 + x], first, "Column {} not uniform", x);
            }
        }

        // Values should increase from left to right (left=100, top_right=200)
        assert!(output[0] < output[3]);
    }

    #[test]
    fn test_bilinear_interpolate() {
        // Equal weights should give average
        let result = bilinear_interpolate(100, 100, 100, 100, 128, 128);
        assert_eq!(result, 100);

        // Different samples
        let result = bilinear_interpolate(200, 100, 100, 200, 128, 128);
        assert!(result >= 140 && result <= 160);
    }

    #[test]
    fn test_weight_interpolation() {
        // Test interpolation for non-standard sizes
        let w = weights::interpolate_weight(6, 0);
        assert!(w > 200); // Should be high at edge

        let w = weights::interpolate_weight(6, 5);
        assert!(w < 100); // Should be lower at center
    }

    #[test]
    fn test_smooth_rectangular_block() {
        let ctx = create_test_context();
        let predictor = SmoothPredictor::new();
        let dims = BlockDimensions::new(8, 4);
        let mut output = vec![0u16; 32];

        predictor.predict(&ctx, &mut output, 8, dims);

        // All values should be in valid range
        for &val in &output {
            assert!(val >= 100 && val <= 200);
        }
    }
}
