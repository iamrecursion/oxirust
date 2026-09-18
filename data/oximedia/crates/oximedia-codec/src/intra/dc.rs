//! DC prediction implementations.
//!
//! DC prediction uses the average of neighboring samples. Several variants
//! exist depending on which neighbors are available:
//!
//! - **Both neighbors**: Average of top and left samples
//! - **Top only**: Average of top samples
//! - **Left only**: Average of left samples
//! - **No neighbors**: Use midpoint value (128 for 8-bit)
//!
//! DC prediction is the simplest and most common intra mode.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::similar_names)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::match_same_arms)]

use super::{BitDepth, BlockDimensions, IntraPredContext, IntraPredictor};

/// DC prediction mode variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DcMode {
    /// Average of both top and left neighbors.
    #[default]
    Both,
    /// Average of top neighbors only.
    TopOnly,
    /// Average of left neighbors only.
    LeftOnly,
    /// No neighbors available, use midpoint.
    NoNeighbors,
    /// DC with gradient (adds gradient based on position).
    WithGradient,
}

/// DC predictor implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct DcPredictor {
    /// Bit depth for sample values.
    bit_depth: BitDepth,
}

impl DcPredictor {
    /// Create a new DC predictor.
    #[must_use]
    pub const fn new(bit_depth: BitDepth) -> Self {
        Self { bit_depth }
    }

    /// Calculate DC value from top samples only.
    fn dc_top_only(top: &[u16], width: usize) -> u16 {
        if width == 0 {
            return 128;
        }

        let sum: u32 = top.iter().take(width).map(|&s| u32::from(s)).sum();
        let avg = (sum + (width as u32 / 2)) / width as u32;
        avg as u16
    }

    /// Calculate DC value from left samples only.
    fn dc_left_only(left: &[u16], height: usize) -> u16 {
        if height == 0 {
            return 128;
        }

        let sum: u32 = left.iter().take(height).map(|&s| u32::from(s)).sum();
        let avg = (sum + (height as u32 / 2)) / height as u32;
        avg as u16
    }

    /// Calculate DC value from both neighbors.
    fn dc_both(top: &[u16], left: &[u16], width: usize, height: usize) -> u16 {
        if width == 0 && height == 0 {
            return 128;
        }

        let top_sum: u32 = top.iter().take(width).map(|&s| u32::from(s)).sum();
        let left_sum: u32 = left.iter().take(height).map(|&s| u32::from(s)).sum();

        let total = width + height;
        let sum = top_sum + left_sum;
        let avg = (sum + (total as u32 / 2)) / total as u32;
        avg as u16
    }

    /// Predict with DC mode.
    pub fn predict_dc(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let mode = self.determine_mode(ctx);
        let dc_value = self.calculate_dc(ctx, dims, mode);

        // Fill block with DC value
        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                output[row_start + x] = dc_value;
            }
        }
    }

    /// Predict with DC and gradient adjustment.
    pub fn predict_dc_gradient(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let base_dc = self.calculate_dc(ctx, dims, DcMode::Both);
        let top = ctx.top_samples();
        let left = ctx.left_samples();
        let max_val = self.bit_depth.max_value();

        // Calculate gradients
        let top_left = ctx.top_left_sample();

        for y in 0..dims.height {
            let row_start = y * stride;
            let left_diff = i32::from(left[y]) - i32::from(top_left);

            for x in 0..dims.width {
                let top_diff = i32::from(top[x]) - i32::from(top_left);

                // Add gradient to base DC
                let gradient = (top_diff + left_diff) / 2;
                let pred = i32::from(base_dc) + gradient;

                // Clamp to valid range
                let clamped = pred.clamp(0, i32::from(max_val));
                output[row_start + x] = clamped as u16;
            }
        }
    }

    /// Determine which DC mode to use based on neighbor availability.
    fn determine_mode(&self, ctx: &IntraPredContext) -> DcMode {
        let has_top = ctx.has_top();
        let has_left = ctx.has_left();

        match (has_top, has_left) {
            (true, true) => DcMode::Both,
            (true, false) => DcMode::TopOnly,
            (false, true) => DcMode::LeftOnly,
            (false, false) => DcMode::NoNeighbors,
        }
    }

    /// Calculate DC value based on mode.
    fn calculate_dc(&self, ctx: &IntraPredContext, dims: BlockDimensions, mode: DcMode) -> u16 {
        match mode {
            DcMode::Both => Self::dc_both(
                ctx.top_samples(),
                ctx.left_samples(),
                dims.width,
                dims.height,
            ),
            DcMode::TopOnly => Self::dc_top_only(ctx.top_samples(), dims.width),
            DcMode::LeftOnly => Self::dc_left_only(ctx.left_samples(), dims.height),
            DcMode::NoNeighbors => self.bit_depth.midpoint(),
            DcMode::WithGradient => Self::dc_both(
                ctx.top_samples(),
                ctx.left_samples(),
                dims.width,
                dims.height,
            ),
        }
    }
}

impl IntraPredictor for DcPredictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        self.predict_dc(ctx, output, stride, dims);
    }
}

/// Top-only DC predictor.
#[derive(Clone, Copy, Debug, Default)]
pub struct DcTopPredictor {
    bit_depth: BitDepth,
}

impl DcTopPredictor {
    /// Create a new top-only DC predictor.
    #[must_use]
    pub const fn new(bit_depth: BitDepth) -> Self {
        Self { bit_depth }
    }
}

impl IntraPredictor for DcTopPredictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let dc_value = if ctx.has_top() {
            DcPredictor::dc_top_only(ctx.top_samples(), dims.width)
        } else {
            self.bit_depth.midpoint()
        };

        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                output[row_start + x] = dc_value;
            }
        }
    }
}

/// Left-only DC predictor.
#[derive(Clone, Copy, Debug, Default)]
pub struct DcLeftPredictor {
    bit_depth: BitDepth,
}

impl DcLeftPredictor {
    /// Create a new left-only DC predictor.
    #[must_use]
    pub const fn new(bit_depth: BitDepth) -> Self {
        Self { bit_depth }
    }
}

impl IntraPredictor for DcLeftPredictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let dc_value = if ctx.has_left() {
            DcPredictor::dc_left_only(ctx.left_samples(), dims.height)
        } else {
            self.bit_depth.midpoint()
        };

        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                output[row_start + x] = dc_value;
            }
        }
    }
}

/// No-neighbors DC predictor (uses midpoint value).
#[derive(Clone, Copy, Debug, Default)]
pub struct Dc128Predictor {
    bit_depth: BitDepth,
}

impl Dc128Predictor {
    /// Create a new 128 (midpoint) DC predictor.
    #[must_use]
    pub const fn new(bit_depth: BitDepth) -> Self {
        Self { bit_depth }
    }
}

impl IntraPredictor for Dc128Predictor {
    fn predict(
        &self,
        _ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let dc_value = self.bit_depth.midpoint();

        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                output[row_start + x] = dc_value;
            }
        }
    }
}

/// DC predictor with gradient adjustment.
#[derive(Clone, Copy, Debug, Default)]
pub struct DcGradientPredictor {
    bit_depth: BitDepth,
}

impl DcGradientPredictor {
    /// Create a new gradient DC predictor.
    #[must_use]
    pub const fn new(bit_depth: BitDepth) -> Self {
        Self { bit_depth }
    }
}

impl IntraPredictor for DcGradientPredictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let predictor = DcPredictor::new(self.bit_depth);
        predictor.predict_dc_gradient(ctx, output, stride, dims);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra::context::IntraPredContext;

    fn create_test_context() -> IntraPredContext {
        let mut ctx = IntraPredContext::new(8, 8, BitDepth::Bits8);

        // Set top samples: [100, 110, 120, 130, 140, 150, 160, 170]
        for i in 0..8 {
            ctx.set_top_sample(i, 100 + (i as u16 * 10));
        }

        // Set left samples: [80, 90, 100, 110, 120, 130, 140, 150]
        for i in 0..8 {
            ctx.set_left_sample(i, 80 + (i as u16 * 10));
        }

        ctx.set_top_left_sample(90);
        ctx.set_availability(true, true);

        ctx
    }

    #[test]
    fn test_dc_top_only() {
        let top = [100u16, 110, 120, 130];
        let dc = DcPredictor::dc_top_only(&top, 4);
        // Average: (100 + 110 + 120 + 130) / 4 = 460 / 4 = 115
        assert_eq!(dc, 115);
    }

    #[test]
    fn test_dc_left_only() {
        let left = [80u16, 90, 100, 110];
        let dc = DcPredictor::dc_left_only(&left, 4);
        // Average: (80 + 90 + 100 + 110) / 4 = 380 / 4 = 95
        assert_eq!(dc, 95);
    }

    #[test]
    fn test_dc_both() {
        let top = [100u16, 110, 120, 130];
        let left = [80u16, 90, 100, 110];
        let dc = DcPredictor::dc_both(&top, &left, 4, 4);
        // Average: (460 + 380) / 8 = 840 / 8 = 105
        assert_eq!(dc, 105);
    }

    #[test]
    fn test_dc_predictor_both() {
        let ctx = create_test_context();
        let predictor = DcPredictor::new(BitDepth::Bits8);
        let dims = BlockDimensions::new(8, 8);
        let mut output = vec![0u16; 64];

        predictor.predict(&ctx, &mut output, 8, dims);

        // All values should be the same DC value
        let dc_value = output[0];
        assert!(output.iter().all(|&v| v == dc_value));

        // Top sum: 100+110+120+130+140+150+160+170 = 1080
        // Left sum: 80+90+100+110+120+130+140+150 = 920
        // Total: (1080 + 920) / 16 = 125
        assert_eq!(dc_value, 125);
    }

    #[test]
    fn test_dc_128_predictor() {
        let mut ctx = IntraPredContext::new(4, 4, BitDepth::Bits8);
        ctx.set_availability(false, false);

        let predictor = Dc128Predictor::new(BitDepth::Bits8);
        let dims = BlockDimensions::new(4, 4);
        let mut output = vec![0u16; 16];

        predictor.predict(&ctx, &mut output, 4, dims);

        // All values should be 128
        assert!(output.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_dc_top_predictor() {
        let ctx = create_test_context();
        let predictor = DcTopPredictor::new(BitDepth::Bits8);
        let dims = BlockDimensions::new(8, 8);
        let mut output = vec![0u16; 64];

        predictor.predict(&ctx, &mut output, 8, dims);

        // Top sum: 1080 / 8 = 135
        assert!(output.iter().all(|&v| v == 135));
    }

    #[test]
    fn test_dc_left_predictor() {
        let ctx = create_test_context();
        let predictor = DcLeftPredictor::new(BitDepth::Bits8);
        let dims = BlockDimensions::new(8, 8);
        let mut output = vec![0u16; 64];

        predictor.predict(&ctx, &mut output, 8, dims);

        // Left sum: 920 / 8 = 115
        assert!(output.iter().all(|&v| v == 115));
    }

    #[test]
    fn test_dc_mode_determination() {
        let predictor = DcPredictor::new(BitDepth::Bits8);

        let mut ctx = IntraPredContext::new(4, 4, BitDepth::Bits8);

        ctx.set_availability(true, true);
        assert_eq!(predictor.determine_mode(&ctx), DcMode::Both);

        ctx.set_availability(true, false);
        assert_eq!(predictor.determine_mode(&ctx), DcMode::TopOnly);

        ctx.set_availability(false, true);
        assert_eq!(predictor.determine_mode(&ctx), DcMode::LeftOnly);

        ctx.set_availability(false, false);
        assert_eq!(predictor.determine_mode(&ctx), DcMode::NoNeighbors);
    }

    #[test]
    fn test_bit_depth_10() {
        let predictor = Dc128Predictor::new(BitDepth::Bits10);
        let ctx = IntraPredContext::new(4, 4, BitDepth::Bits10);
        let dims = BlockDimensions::new(4, 4);
        let mut output = vec![0u16; 16];

        predictor.predict(&ctx, &mut output, 4, dims);

        // Midpoint for 10-bit is 512
        assert!(output.iter().all(|&v| v == 512));
    }
}
