//! Directional intra prediction implementations.
//!
//! Directional prediction projects samples from neighboring blocks along
//! specified angles. Supported angles range from 0 to 270 degrees with
//! various resolutions depending on the codec.
//!
//! # Prediction Modes
//!
//! - **Vertical (90 degrees)**: Copy top samples down
//! - **Horizontal (180 degrees)**: Copy left samples across
//! - **D45 (45 degrees)**: Diagonal down-right
//! - **D135 (135 degrees)**: Diagonal up-left
//! - **D113/D117 (113/117 degrees)**: Near-vertical
//! - **D157/D153 (157/153 degrees)**: Near-horizontal
//! - **D203/D207 (203/207 degrees)**: Bottom-left diagonal
//! - **D67/D63 (67/63 degrees)**: Top-right diagonal
//!
//! # Sub-pixel Interpolation
//!
//! For angles that don't align with sample positions, linear interpolation
//! is used to generate fractional samples.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::manual_rem_euclid)]
#![allow(clippy::cast_lossless)]

use super::{
    AngleDelta, BitDepth, BlockDimensions, DirectionalMode, IntraPredContext, IntraPredictor,
};

/// Directional predictor that handles all angular modes.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalPredictor {
    /// Base angle in degrees.
    angle: u16,
    /// Angle delta adjustment.
    delta: AngleDelta,
    /// Bit depth.
    bit_depth: BitDepth,
}

impl DirectionalPredictor {
    /// Create a new directional predictor.
    #[must_use]
    pub const fn new(angle: u16, bit_depth: BitDepth) -> Self {
        Self {
            angle,
            delta: AngleDelta::Zero,
            bit_depth,
        }
    }

    /// Create with angle delta (AV1).
    #[must_use]
    pub const fn with_delta(angle: u16, delta: AngleDelta, bit_depth: BitDepth) -> Self {
        Self {
            angle,
            delta,
            bit_depth,
        }
    }

    /// Get effective angle including delta.
    #[must_use]
    pub const fn effective_angle(&self) -> i16 {
        self.angle as i16 + self.delta.degrees()
    }

    /// Predict using the configured angle.
    pub fn predict_angle(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let angle = self.effective_angle();

        // Route to specialized implementations
        match angle {
            90 => self.predict_vertical(ctx, output, stride, dims),
            180 => self.predict_horizontal(ctx, output, stride, dims),
            45 => self.predict_d45(ctx, output, stride, dims),
            135 => self.predict_d135(ctx, output, stride, dims),
            _ => self.predict_generic(ctx, output, stride, dims, angle),
        }
    }

    /// Vertical prediction (90 degrees).
    fn predict_vertical(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let top = ctx.top_samples();

        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                output[row_start + x] = top[x];
            }
        }
    }

    /// Horizontal prediction (180 degrees).
    fn predict_horizontal(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let left = ctx.left_samples();

        for y in 0..dims.height {
            let row_start = y * stride;
            let left_val = left[y];
            for x in 0..dims.width {
                output[row_start + x] = left_val;
            }
        }
    }

    /// D45 prediction (45 degrees, down-right diagonal).
    fn predict_d45(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let top = ctx.top_samples();

        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                // Sample from top-right diagonal
                let idx = x + y + 1;
                let sample = if idx < ctx.top_samples().len() {
                    top[idx]
                } else {
                    // Use last available sample
                    top[ctx.top_samples().len().saturating_sub(1)]
                };
                output[row_start + x] = sample;
            }
        }
    }

    /// D135 prediction (135 degrees, up-left diagonal).
    fn predict_d135(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let top = ctx.top_samples();
        let left = ctx.left_samples();
        let top_left = ctx.top_left_sample();

        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                let sample = if y > x {
                    // Use left neighbor
                    left[y - x - 1]
                } else if y < x {
                    // Use top neighbor
                    top[x - y - 1]
                } else {
                    // Use top-left
                    top_left
                };
                output[row_start + x] = sample;
            }
        }
    }

    /// Generic angular prediction with interpolation.
    fn predict_generic(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
        angle: i16,
    ) {
        // Calculate dx and dy for the angle (in 1/256ths of a pixel)
        let (dx, dy) = get_direction_deltas(angle);

        if angle < 90 {
            // Angles 0-89: primarily vertical, from top
            self.predict_from_top(ctx, output, stride, dims, dx, dy);
        } else if angle < 180 {
            // Angles 90-179: blend of top and left
            self.predict_from_top_left(ctx, output, stride, dims, dx, dy, angle);
        } else if angle < 270 {
            // Angles 180-269: primarily horizontal, from left
            self.predict_from_left(ctx, output, stride, dims, dx, dy);
        } else {
            // Angles 270+: from left going up
            self.predict_from_left_up(ctx, output, stride, dims, dx, dy);
        }
    }

    /// Predict from top samples (angles 0-89).
    fn predict_from_top(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
        dx: i32,
        _dy: i32,
    ) {
        let top = ctx.top_samples();
        let max_idx = top.len().saturating_sub(1);

        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                // Calculate source position
                let src_x = ((x as i32) * 256 + (y as i32 + 1) * dx) / 256;
                let frac = (((x as i32) * 256 + (y as i32 + 1) * dx) % 256) as u16;

                let idx = src_x.clamp(0, max_idx as i32) as usize;
                let idx_next = (idx + 1).min(max_idx);

                // Linear interpolation
                let sample = interpolate(top[idx], top[idx_next], frac);
                output[row_start + x] = sample;
            }
        }
    }

    /// Predict from top-left (angles around 90-179).
    fn predict_from_top_left(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
        dx: i32,
        dy: i32,
        angle: i16,
    ) {
        let top = ctx.top_samples();
        let left = ctx.left_samples();
        let top_left = ctx.top_left_sample();

        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                let sample = if angle <= 90 {
                    // Use top samples
                    let src_x = (x as i32) + ((y as i32 + 1) * dx) / 256;
                    let frac = ((y as i32 + 1) * dx) % 256;
                    get_sample_from_neighbors(
                        top,
                        left,
                        top_left,
                        src_x,
                        -1, // indicating top row
                        frac.unsigned_abs() as u16,
                        true,
                    )
                } else {
                    // Use left samples
                    let src_y = (y as i32) + ((x as i32 + 1) * dy) / 256;
                    let frac = ((x as i32 + 1) * dy) % 256;
                    get_sample_from_neighbors(
                        top,
                        left,
                        top_left,
                        -1, // indicating left column
                        src_y,
                        frac.unsigned_abs() as u16,
                        false,
                    )
                };
                output[row_start + x] = sample;
            }
        }
    }

    /// Predict from left samples (angles 180-269).
    fn predict_from_left(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
        _dx: i32,
        dy: i32,
    ) {
        let left = ctx.left_samples();
        let max_idx = left.len().saturating_sub(1);

        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                // Calculate source position
                let src_y = ((y as i32) * 256 + (x as i32 + 1) * dy) / 256;
                let frac = (((y as i32) * 256 + (x as i32 + 1) * dy) % 256) as u16;

                let idx = src_y.clamp(0, max_idx as i32) as usize;
                let idx_next = (idx + 1).min(max_idx);

                // Linear interpolation
                let sample = interpolate(left[idx], left[idx_next], frac);
                output[row_start + x] = sample;
            }
        }
    }

    /// Predict from left going upward (angles 270+).
    fn predict_from_left_up(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
        dx: i32,
        _dy: i32,
    ) {
        let left = ctx.left_samples();
        let top = ctx.top_samples();
        let top_left = ctx.top_left_sample();

        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                let src_idx = (y as i32) - ((x as i32 + 1) * dx.abs()) / 256;
                let frac = (((x as i32 + 1) * dx.abs()) % 256) as u16;

                let sample = if src_idx >= 0 {
                    let idx = src_idx as usize;
                    let idx_next = (idx + 1).min(left.len().saturating_sub(1));
                    interpolate(left[idx], left[idx_next], frac)
                } else {
                    // Sample from top
                    let top_idx = (-(src_idx + 1)) as usize;
                    if top_idx == 0 {
                        top_left
                    } else if top_idx <= top.len() {
                        top[top_idx - 1]
                    } else {
                        top[top.len().saturating_sub(1)]
                    }
                };
                output[row_start + x] = sample;
            }
        }
    }
}

impl IntraPredictor for DirectionalPredictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        self.predict_angle(ctx, output, stride, dims);
    }
}

/// Vertical predictor (90 degrees).
#[derive(Clone, Copy, Debug, Default)]
pub struct VerticalPredictor;

impl VerticalPredictor {
    /// Create a new vertical predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl IntraPredictor for VerticalPredictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let top = ctx.top_samples();

        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                output[row_start + x] = top[x];
            }
        }
    }
}

/// Horizontal predictor (180 degrees).
#[derive(Clone, Copy, Debug, Default)]
pub struct HorizontalPredictor;

impl HorizontalPredictor {
    /// Create a new horizontal predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl IntraPredictor for HorizontalPredictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let left = ctx.left_samples();

        for y in 0..dims.height {
            let row_start = y * stride;
            let left_val = left[y];
            for x in 0..dims.width {
                output[row_start + x] = left_val;
            }
        }
    }
}

/// D45 predictor (45 degrees).
#[derive(Clone, Copy, Debug, Default)]
pub struct D45Predictor;

impl D45Predictor {
    /// Create a new D45 predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl IntraPredictor for D45Predictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let predictor = DirectionalPredictor::new(45, BitDepth::Bits8);
        predictor.predict_d45(ctx, output, stride, dims);
    }
}

/// D63 predictor (63 degrees, VP9 naming).
#[derive(Clone, Copy, Debug, Default)]
pub struct D63Predictor;

impl D63Predictor {
    /// Create a new D63 predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl IntraPredictor for D63Predictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let predictor = DirectionalPredictor::new(63, BitDepth::Bits8);
        predictor.predict_angle(ctx, output, stride, dims);
    }
}

/// D67 predictor (67 degrees, AV1 naming).
#[derive(Clone, Copy, Debug, Default)]
pub struct D67Predictor;

impl D67Predictor {
    /// Create a new D67 predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl IntraPredictor for D67Predictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let predictor = DirectionalPredictor::new(67, BitDepth::Bits8);
        predictor.predict_angle(ctx, output, stride, dims);
    }
}

/// D113 predictor (113 degrees, AV1 naming).
#[derive(Clone, Copy, Debug, Default)]
pub struct D113Predictor;

impl D113Predictor {
    /// Create a new D113 predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl IntraPredictor for D113Predictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let predictor = DirectionalPredictor::new(113, BitDepth::Bits8);
        predictor.predict_angle(ctx, output, stride, dims);
    }
}

/// D117 predictor (117 degrees, VP9 naming).
#[derive(Clone, Copy, Debug, Default)]
pub struct D117Predictor;

impl D117Predictor {
    /// Create a new D117 predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl IntraPredictor for D117Predictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let predictor = DirectionalPredictor::new(117, BitDepth::Bits8);
        predictor.predict_angle(ctx, output, stride, dims);
    }
}

/// D135 predictor (135 degrees).
#[derive(Clone, Copy, Debug, Default)]
pub struct D135Predictor;

impl D135Predictor {
    /// Create a new D135 predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl IntraPredictor for D135Predictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let predictor = DirectionalPredictor::new(135, BitDepth::Bits8);
        predictor.predict_d135(ctx, output, stride, dims);
    }
}

/// D153 predictor (153 degrees, VP9 naming).
#[derive(Clone, Copy, Debug, Default)]
pub struct D153Predictor;

impl D153Predictor {
    /// Create a new D153 predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl IntraPredictor for D153Predictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let predictor = DirectionalPredictor::new(153, BitDepth::Bits8);
        predictor.predict_angle(ctx, output, stride, dims);
    }
}

/// D157 predictor (157 degrees, AV1 naming).
#[derive(Clone, Copy, Debug, Default)]
pub struct D157Predictor;

impl D157Predictor {
    /// Create a new D157 predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl IntraPredictor for D157Predictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let predictor = DirectionalPredictor::new(157, BitDepth::Bits8);
        predictor.predict_angle(ctx, output, stride, dims);
    }
}

/// D203 predictor (203 degrees, AV1 naming).
#[derive(Clone, Copy, Debug, Default)]
pub struct D203Predictor;

impl D203Predictor {
    /// Create a new D203 predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl IntraPredictor for D203Predictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let predictor = DirectionalPredictor::new(203, BitDepth::Bits8);
        predictor.predict_angle(ctx, output, stride, dims);
    }
}

/// D207 predictor (207 degrees, VP9 naming).
#[derive(Clone, Copy, Debug, Default)]
pub struct D207Predictor;

impl D207Predictor {
    /// Create a new D207 predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl IntraPredictor for D207Predictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let predictor = DirectionalPredictor::new(207, BitDepth::Bits8);
        predictor.predict_angle(ctx, output, stride, dims);
    }
}

/// Get direction deltas (dx, dy) for an angle.
/// Returns values in 1/256ths of a pixel.
#[must_use]
fn get_direction_deltas(angle: i16) -> (i32, i32) {
    // Normalize angle to 0-360
    let angle = ((angle % 360) + 360) % 360;

    // Calculate dx and dy based on angle
    // Using lookup for common angles, interpolation for others
    match angle {
        0 => (0, 256),
        45 => (181, 181),
        90 => (256, 0),
        135 => (181, -181),
        180 => (0, -256),
        225 => (-181, -181),
        270 => (-256, 0),
        315 => (-181, 181),
        _ => {
            // Approximate using trigonometry
            let radians = (angle as f64) * std::f64::consts::PI / 180.0;
            let dx = (radians.sin() * 256.0).round() as i32;
            let dy = (radians.cos() * 256.0).round() as i32;
            (dx, dy)
        }
    }
}

/// Linear interpolation between two samples.
#[inline]
fn interpolate(a: u16, b: u16, frac: u16) -> u16 {
    // frac is in 1/256ths (0-255)
    let a32 = u32::from(a);
    let b32 = u32::from(b);
    let frac32 = u32::from(frac);

    let result = (a32 * (256 - frac32) + b32 * frac32 + 128) / 256;
    result as u16
}

/// Get sample from combined top/left neighbors.
fn get_sample_from_neighbors(
    top: &[u16],
    left: &[u16],
    top_left: u16,
    x: i32,
    y: i32,
    frac: u16,
    use_top: bool,
) -> u16 {
    if use_top {
        // Sample from top row
        let idx = x.clamp(0, (top.len() - 1) as i32) as usize;
        let idx_next = (idx + 1).min(top.len() - 1);
        interpolate(top[idx], top[idx_next], frac)
    } else {
        // Sample from left column
        if y < 0 {
            top_left
        } else {
            let idx = y.clamp(0, (left.len() - 1) as i32) as usize;
            let idx_next = (idx + 1).min(left.len() - 1);
            interpolate(left[idx], left[idx_next], frac)
        }
    }
}

/// Get direction samples from neighbors based on angle.
pub fn get_direction_samples(
    ctx: &IntraPredContext,
    angle: i16,
    width: usize,
    height: usize,
) -> Vec<u16> {
    let mode = DirectionalMode::new(angle as u16);
    let mut samples = Vec::with_capacity(width * height);

    let top = ctx.top_samples();
    let left = ctx.left_samples();

    if mode.is_vertical_ish() {
        // Primarily from top
        for y in 0..height {
            for x in 0..width {
                let idx = (x + y).min(top.len() - 1);
                samples.push(top[idx]);
            }
        }
    } else {
        // Primarily from left
        for y in 0..height {
            for x in 0..width {
                let idx = (x + y).min(left.len() - 1);
                samples.push(left[idx]);
            }
        }
    }

    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra::context::IntraPredContext;

    fn create_test_context() -> IntraPredContext {
        let mut ctx = IntraPredContext::new(8, 8, BitDepth::Bits8);

        // Set top samples: [10, 20, 30, 40, 50, 60, 70, 80, ...]
        for i in 0..16 {
            ctx.set_top_sample(i, ((i + 1) * 10) as u16);
        }

        // Set left samples: [15, 25, 35, 45, 55, 65, 75, 85, ...]
        for i in 0..16 {
            ctx.set_left_sample(i, (15 + i * 10) as u16);
        }

        ctx.set_top_left_sample(5);
        ctx.set_availability(true, true);

        ctx
    }

    #[test]
    fn test_vertical_prediction() {
        let ctx = create_test_context();
        let predictor = VerticalPredictor::new();
        let dims = BlockDimensions::new(4, 4);
        let mut output = vec![0u16; 16];

        predictor.predict(&ctx, &mut output, 4, dims);

        // Each column should have the same value (from top)
        assert_eq!(output[0], 10); // x=0
        assert_eq!(output[1], 20); // x=1
        assert_eq!(output[2], 30); // x=2
        assert_eq!(output[3], 40); // x=3

        // Row 1 should be the same
        assert_eq!(output[4], 10);
        assert_eq!(output[5], 20);
    }

    #[test]
    fn test_horizontal_prediction() {
        let ctx = create_test_context();
        let predictor = HorizontalPredictor::new();
        let dims = BlockDimensions::new(4, 4);
        let mut output = vec![0u16; 16];

        predictor.predict(&ctx, &mut output, 4, dims);

        // Each row should have the same value (from left)
        // Row 0: all 15
        assert_eq!(output[0], 15);
        assert_eq!(output[1], 15);
        assert_eq!(output[2], 15);
        assert_eq!(output[3], 15);

        // Row 1: all 25
        assert_eq!(output[4], 25);
        assert_eq!(output[5], 25);
    }

    #[test]
    fn test_d45_prediction() {
        let ctx = create_test_context();
        let predictor = D45Predictor::new();
        let dims = BlockDimensions::new(4, 4);
        let mut output = vec![0u16; 16];

        predictor.predict(&ctx, &mut output, 4, dims);

        // D45 samples from top diagonal
        // (0,0) -> top[1] = 20
        // (1,0) -> top[2] = 30
        // (0,1) -> top[2] = 30
        assert_eq!(output[0], 20);
        assert_eq!(output[1], 30);
        assert_eq!(output[4], 30); // row 1, col 0
    }

    #[test]
    fn test_d135_prediction() {
        let ctx = create_test_context();
        let predictor = D135Predictor::new();
        let dims = BlockDimensions::new(4, 4);
        let mut output = vec![0u16; 16];

        predictor.predict(&ctx, &mut output, 4, dims);

        // D135 samples along up-left diagonal
        // (0,0) -> top_left = 5
        // (1,0) -> top[0] = 10
        // (0,1) -> left[0] = 15
        assert_eq!(output[0], 5);
        assert_eq!(output[1], 10);
        assert_eq!(output[4], 15); // row 1, col 0
    }

    #[test]
    fn test_interpolation() {
        // No interpolation (frac = 0)
        assert_eq!(interpolate(100, 200, 0), 100);

        // Full interpolation (frac = 255)
        let result = interpolate(100, 200, 255);
        assert!(result >= 199 && result <= 200);

        // Half interpolation (frac = 128)
        let result = interpolate(100, 200, 128);
        assert!(result >= 149 && result <= 151);
    }

    #[test]
    fn test_direction_deltas() {
        let (dx, dy) = get_direction_deltas(0);
        assert_eq!((dx, dy), (0, 256));

        let (dx, dy) = get_direction_deltas(90);
        assert_eq!((dx, dy), (256, 0));

        let (dx, dy) = get_direction_deltas(180);
        assert_eq!((dx, dy), (0, -256));

        let (dx, dy) = get_direction_deltas(45);
        assert_eq!((dx, dy), (181, 181));
    }

    #[test]
    fn test_directional_predictor() {
        let ctx = create_test_context();
        let predictor = DirectionalPredictor::new(90, BitDepth::Bits8);
        let dims = BlockDimensions::new(4, 4);
        let mut output = vec![0u16; 16];

        predictor.predict(&ctx, &mut output, 4, dims);

        // Should be same as vertical
        assert_eq!(output[0], 10);
        assert_eq!(output[1], 20);
    }

    #[test]
    fn test_directional_with_delta() {
        let predictor = DirectionalPredictor::with_delta(90, AngleDelta::Plus3, BitDepth::Bits8);
        assert_eq!(predictor.effective_angle(), 99);
    }
}
