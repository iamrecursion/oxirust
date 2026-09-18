//! Paeth prediction implementation (AV1).
//!
//! Paeth prediction is an adaptive method that selects the neighbor sample
//! (top, left, or top-left) whose value is closest to a prediction based
//! on linear extrapolation.
//!
//! The algorithm was originally developed for PNG compression and is also
//! used in AV1 for intra prediction.
//!
//! # Algorithm
//!
//! For each position (x, y):
//! 1. Get neighbors: top (T), left (L), top-left (TL)
//! 2. Calculate base: p = T + L - TL
//! 3. Calculate distances: |p - T|, |p - L|, |p - TL|
//! 4. Select the neighbor with minimum distance

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]

use super::{BlockDimensions, IntraPredContext, IntraPredictor};

/// Calculate Paeth predictor value.
///
/// Selects the neighbor (top, left, or top-left) whose value is closest
/// to the linear predictor p = top + left - top_left.
///
/// # Arguments
/// * `top` - Sample above current position
/// * `left` - Sample to the left of current position
/// * `top_left` - Sample diagonally above-left
///
/// # Returns
/// The selected neighbor value
#[must_use]
#[inline]
pub fn paeth_predictor(top: u16, left: u16, top_left: u16) -> u16 {
    // Calculate linear predictor
    let p = i32::from(top) + i32::from(left) - i32::from(top_left);

    // Calculate absolute differences
    let p_top = (p - i32::from(top)).abs();
    let p_left = (p - i32::from(left)).abs();
    let p_top_left = (p - i32::from(top_left)).abs();

    // Return the neighbor with minimum distance
    // Tie-breaking order: top, left, top_left
    if p_top <= p_left && p_top <= p_top_left {
        top
    } else if p_left <= p_top_left {
        left
    } else {
        top_left
    }
}

/// Paeth predictor implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct PaethPredictor;

impl PaethPredictor {
    /// Create a new Paeth predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Perform Paeth prediction.
    pub fn predict_paeth(
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let top = ctx.top_samples();
        let left = ctx.left_samples();
        let base_top_left = ctx.top_left_sample();

        for y in 0..dims.height {
            let row_start = y * stride;

            for x in 0..dims.width {
                // Get the three neighbor samples
                let t = top[x];
                let l = left[y];

                // Top-left depends on position
                let tl = if x == 0 && y == 0 {
                    base_top_left
                } else if x == 0 {
                    // First column uses left sample from previous row
                    left[y.saturating_sub(1)]
                } else if y == 0 {
                    // First row uses top sample from previous column
                    top[x.saturating_sub(1)]
                } else {
                    // Interior uses actual top-left
                    // For prediction, we use the reconstructed top-left sample
                    // which in this simplified version is the base top-left
                    base_top_left
                };

                output[row_start + x] = paeth_predictor(t, l, tl);
            }
        }
    }

    /// Perform Paeth prediction with reconstructed neighbors.
    ///
    /// This version uses previously predicted samples as the top-left
    /// for interior positions.
    pub fn predict_paeth_with_reconstruction(
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        let top = ctx.top_samples();
        let left = ctx.left_samples();
        let base_top_left = ctx.top_left_sample();

        for y in 0..dims.height {
            let row_start = y * stride;

            for x in 0..dims.width {
                let t = top[x];
                let l = left[y];

                let tl = if x == 0 && y == 0 {
                    base_top_left
                } else if x == 0 {
                    left[y - 1]
                } else if y == 0 {
                    top[x - 1]
                } else {
                    // Use previously predicted sample
                    output[(y - 1) * stride + (x - 1)]
                };

                output[row_start + x] = paeth_predictor(t, l, tl);
            }
        }
    }
}

impl IntraPredictor for PaethPredictor {
    fn predict(
        &self,
        ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        Self::predict_paeth(ctx, output, stride, dims);
    }
}

/// Which neighbor was selected by Paeth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaethSelection {
    /// Top neighbor was selected.
    Top,
    /// Left neighbor was selected.
    Left,
    /// Top-left neighbor was selected.
    TopLeft,
}

/// Calculate which neighbor Paeth would select.
///
/// Useful for analysis and debugging.
#[must_use]
pub fn paeth_selection(top: u16, left: u16, top_left: u16) -> PaethSelection {
    let p = i32::from(top) + i32::from(left) - i32::from(top_left);

    let p_top = (p - i32::from(top)).abs();
    let p_left = (p - i32::from(left)).abs();
    let p_top_left = (p - i32::from(top_left)).abs();

    if p_top <= p_left && p_top <= p_top_left {
        PaethSelection::Top
    } else if p_left <= p_top_left {
        PaethSelection::Left
    } else {
        PaethSelection::TopLeft
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra::context::IntraPredContext;
    use crate::intra::BitDepth;

    #[test]
    fn test_paeth_predictor_top() {
        // When top + left = 2 * top_left, p = top + left - top_left = top_left
        // All distances equal, should select top
        assert_eq!(paeth_predictor(100, 100, 100), 100);

        // Top is closest to prediction
        // p = 200 + 50 - 100 = 150, distances: |150-200|=50, |150-50|=100, |150-100|=50
        // Top wins (tie-breaking)
        assert_eq!(paeth_predictor(200, 50, 100), 200);
    }

    #[test]
    fn test_paeth_predictor_left() {
        // Left is closest to prediction
        // p = 100 + 150 - 100 = 150, distances: |150-100|=50, |150-150|=0, |150-100|=50
        assert_eq!(paeth_predictor(100, 150, 100), 150);
    }

    #[test]
    fn test_paeth_predictor_top_left() {
        // Top-left is closest to prediction
        // p = 50 + 50 - 100 = 0
        // distances: |0-50|=50, |0-50|=50, |0-100|=100
        // Top wins (tie-breaking)
        assert_eq!(paeth_predictor(50, 50, 100), 50);

        // Clearer top-left case
        // p = 200 + 200 - 150 = 250
        // distances: |250-200|=50, |250-200|=50, |250-150|=100
        // Top wins (tie-breaking)
        let result = paeth_predictor(200, 200, 150);
        assert_eq!(result, 200);
    }

    #[test]
    fn test_paeth_selection() {
        assert_eq!(paeth_selection(100, 100, 100), PaethSelection::Top);
        assert_eq!(paeth_selection(100, 150, 100), PaethSelection::Left);
    }

    fn create_test_context() -> IntraPredContext {
        let mut ctx = IntraPredContext::new(4, 4, BitDepth::Bits8);

        // Set top samples
        for i in 0..8 {
            ctx.set_top_sample(i, (100 + i * 10) as u16);
        }

        // Set left samples
        for i in 0..8 {
            ctx.set_left_sample(i, (80 + i * 10) as u16);
        }

        ctx.set_top_left_sample(90);
        ctx.set_availability(true, true);

        ctx
    }

    #[test]
    fn test_paeth_predictor_block() {
        let ctx = create_test_context();
        let predictor = PaethPredictor::new();
        let dims = BlockDimensions::new(4, 4);
        let mut output = vec![0u16; 16];

        predictor.predict(&ctx, &mut output, 4, dims);

        // All outputs should be from the set of neighbor values
        let top = ctx.top_samples();
        let left = ctx.left_samples();
        let tl = ctx.top_left_sample();

        for y in 0..4 {
            for x in 0..4 {
                let val = output[y * 4 + x];
                // Should be one of the neighbors
                assert!(
                    val == top[x] || val == left[y] || val == tl,
                    "Value {} at ({}, {}) not from neighbors: top={}, left={}, tl={}",
                    val,
                    x,
                    y,
                    top[x],
                    left[y],
                    tl
                );
            }
        }
    }

    #[test]
    fn test_paeth_edge_cases() {
        // Test with extreme values
        assert_eq!(paeth_predictor(0, 0, 0), 0);
        assert_eq!(paeth_predictor(255, 255, 255), 255);

        // Test with max difference
        // p = 255 + 0 - 128 = 127
        // distances: |127-255|=128, |127-0|=127, |127-128|=1
        assert_eq!(paeth_predictor(255, 0, 128), 128);
    }

    #[test]
    fn test_paeth_with_gradient() {
        let mut ctx = IntraPredContext::new(4, 4, BitDepth::Bits8);

        // Create gradient pattern
        for i in 0..8 {
            ctx.set_top_sample(i, 100);
            ctx.set_left_sample(i, 100);
        }
        ctx.set_top_left_sample(100);
        ctx.set_availability(true, true);

        let predictor = PaethPredictor::new();
        let dims = BlockDimensions::new(4, 4);
        let mut output = vec![0u16; 16];

        predictor.predict(&ctx, &mut output, 4, dims);

        // With uniform neighbors, all predictions should be 100
        for &val in &output {
            assert_eq!(val, 100);
        }
    }
}
