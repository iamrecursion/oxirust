// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Intra prediction for Theora.
//!
//! Implements various intra prediction modes for encoding keyframes.
//! Intra prediction exploits spatial redundancy within a frame by
//! predicting block content from neighboring reconstructed blocks.

/// Intra prediction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraPredMode {
    /// DC prediction (average of neighbors).
    Dc,
    /// Vertical prediction (copy from top).
    Vertical,
    /// Horizontal prediction (copy from left).
    Horizontal,
    /// True motion prediction (diagonal).
    TrueMotion,
    /// Diagonal down-left prediction.
    DiagonalDownLeft,
    /// Diagonal down-right prediction.
    DiagonalDownRight,
    /// Vertical-right prediction.
    VerticalRight,
    /// Horizontal-down prediction.
    HorizontalDown,
    /// Vertical-left prediction.
    VerticalLeft,
    /// Horizontal-up prediction.
    HorizontalUp,
}

impl IntraPredMode {
    /// Get mode index for bitstream encoding.
    #[must_use]
    pub const fn index(self) -> u8 {
        match self {
            Self::Dc => 0,
            Self::Vertical => 1,
            Self::Horizontal => 2,
            Self::TrueMotion => 3,
            Self::DiagonalDownLeft => 4,
            Self::DiagonalDownRight => 5,
            Self::VerticalRight => 6,
            Self::HorizontalDown => 7,
            Self::VerticalLeft => 8,
            Self::HorizontalUp => 9,
        }
    }

    /// Create mode from index.
    #[must_use]
    pub const fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::Dc),
            1 => Some(Self::Vertical),
            2 => Some(Self::Horizontal),
            3 => Some(Self::TrueMotion),
            4 => Some(Self::DiagonalDownLeft),
            5 => Some(Self::DiagonalDownRight),
            6 => Some(Self::VerticalRight),
            7 => Some(Self::HorizontalDown),
            8 => Some(Self::VerticalLeft),
            9 => Some(Self::HorizontalUp),
            _ => None,
        }
    }

    /// Check if this mode requires top neighbors.
    #[must_use]
    pub const fn requires_top(self) -> bool {
        matches!(
            self,
            Self::Vertical
                | Self::DiagonalDownLeft
                | Self::DiagonalDownRight
                | Self::VerticalRight
                | Self::VerticalLeft
        )
    }

    /// Check if this mode requires left neighbors.
    #[must_use]
    pub const fn requires_left(self) -> bool {
        matches!(
            self,
            Self::Horizontal | Self::DiagonalDownRight | Self::HorizontalDown | Self::HorizontalUp
        )
    }
}

/// Context for intra prediction.
pub struct IntraPredContext {
    /// Top neighbors (16 pixels).
    pub top: [u8; 16],
    /// Left neighbors (16 pixels).
    pub left: [u8; 16],
    /// Top-left corner pixel.
    pub top_left: u8,
    /// Top-right corner pixel.
    pub top_right: u8,
    /// Whether top neighbors are available.
    pub has_top: bool,
    /// Whether left neighbors are available.
    pub has_left: bool,
}

impl Default for IntraPredContext {
    fn default() -> Self {
        Self {
            top: [128; 16],
            left: [128; 16],
            top_left: 128,
            top_right: 128,
            has_top: false,
            has_left: false,
        }
    }
}

impl IntraPredContext {
    /// Create context from plane data.
    pub fn from_plane(plane: &[u8], stride: usize, x: usize, y: usize, block_size: usize) -> Self {
        let mut ctx = Self::default();

        // Extract top neighbors
        if y > 0 {
            ctx.has_top = true;
            for i in 0..block_size {
                if x + i < stride {
                    let offset = (y - 1) * stride + x + i;
                    if offset < plane.len() {
                        ctx.top[i] = plane[offset];
                    }
                }
            }

            // Top-left corner
            if x > 0 {
                let offset = (y - 1) * stride + x - 1;
                if offset < plane.len() {
                    ctx.top_left = plane[offset];
                }
            }

            // Top-right corner
            if x + block_size < stride {
                let offset = (y - 1) * stride + x + block_size;
                if offset < plane.len() {
                    ctx.top_right = plane[offset];
                }
            }
        }

        // Extract left neighbors
        if x > 0 {
            ctx.has_left = true;
            for i in 0..block_size {
                if y + i < plane.len() / stride {
                    let offset = (y + i) * stride + x - 1;
                    if offset < plane.len() {
                        ctx.left[i] = plane[offset];
                    }
                }
            }
        }

        ctx
    }
}

/// Perform intra prediction for an 8x8 block.
///
/// # Arguments
///
/// * `ctx` - Prediction context with neighbor pixels
/// * `mode` - Prediction mode to use
/// * `output` - Output 8x8 block
pub fn predict_intra_8x8(ctx: &IntraPredContext, mode: IntraPredMode, output: &mut [u8; 64]) {
    match mode {
        IntraPredMode::Dc => predict_dc_8x8(ctx, output),
        IntraPredMode::Vertical => predict_vertical_8x8(ctx, output),
        IntraPredMode::Horizontal => predict_horizontal_8x8(ctx, output),
        IntraPredMode::TrueMotion => predict_tm_8x8(ctx, output),
        IntraPredMode::DiagonalDownLeft => predict_ddl_8x8(ctx, output),
        IntraPredMode::DiagonalDownRight => predict_ddr_8x8(ctx, output),
        IntraPredMode::VerticalRight => predict_vr_8x8(ctx, output),
        IntraPredMode::HorizontalDown => predict_hd_8x8(ctx, output),
        IntraPredMode::VerticalLeft => predict_vl_8x8(ctx, output),
        IntraPredMode::HorizontalUp => predict_hu_8x8(ctx, output),
    }
}

/// DC prediction: fill with average of neighbors.
fn predict_dc_8x8(ctx: &IntraPredContext, output: &mut [u8; 64]) {
    let mut sum = 0u32;
    let mut count = 0u32;

    if ctx.has_top {
        for &pixel in &ctx.top[..8] {
            sum += u32::from(pixel);
            count += 1;
        }
    }

    if ctx.has_left {
        for &pixel in &ctx.left[..8] {
            sum += u32::from(pixel);
            count += 1;
        }
    }

    let dc_value = count
        .checked_div(2)
        .and_then(|half| (sum + half).checked_div(count))
        .unwrap_or(128);

    output.fill(dc_value as u8);
}

/// Vertical prediction: copy from top.
fn predict_vertical_8x8(ctx: &IntraPredContext, output: &mut [u8; 64]) {
    for y in 0..8 {
        for x in 0..8 {
            output[y * 8 + x] = ctx.top[x];
        }
    }
}

/// Horizontal prediction: copy from left.
fn predict_horizontal_8x8(ctx: &IntraPredContext, output: &mut [u8; 64]) {
    for y in 0..8 {
        for x in 0..8 {
            output[y * 8 + x] = ctx.left[y];
        }
    }
}

/// True motion prediction: gradient-based prediction.
fn predict_tm_8x8(ctx: &IntraPredContext, output: &mut [u8; 64]) {
    let tl = i16::from(ctx.top_left);

    for y in 0..8 {
        for x in 0..8 {
            let top = i16::from(ctx.top[x]);
            let left = i16::from(ctx.left[y]);
            let pred = left + top - tl;
            output[y * 8 + x] = pred.clamp(0, 255) as u8;
        }
    }
}

/// Diagonal down-left prediction.
fn predict_ddl_8x8(ctx: &IntraPredContext, output: &mut [u8; 64]) {
    for y in 0..8 {
        for x in 0..8 {
            let idx = x + y;
            if idx < 7 {
                let p0 = u16::from(ctx.top[idx]);
                let p1 = u16::from(ctx.top[idx + 1]);
                output[y * 8 + x] = ((p0 + p1 + 1) / 2) as u8;
            } else if idx < 14 {
                let val = ctx.top[7];
                output[y * 8 + x] = val;
            } else {
                output[y * 8 + x] = ctx.top_right;
            }
        }
    }
}

/// Diagonal down-right prediction.
fn predict_ddr_8x8(ctx: &IntraPredContext, output: &mut [u8; 64]) {
    for y in 0..8 {
        for x in 0..8 {
            let idx = x as i16 - y as i16;
            let pred = if idx >= 0 {
                if idx < 7 {
                    let p0 = u16::from(ctx.top[idx as usize]);
                    let p1 = u16::from(if idx == 0 {
                        ctx.top_left
                    } else {
                        ctx.top[(idx - 1) as usize]
                    });
                    ((p0 + p1 + 1) / 2) as u8
                } else {
                    ctx.top[7]
                }
            } else {
                let abs_idx = (-idx - 1) as usize;
                if abs_idx < 8 {
                    let p0 = u16::from(ctx.left[abs_idx]);
                    let p1 = u16::from(if abs_idx == 0 {
                        ctx.top_left
                    } else {
                        ctx.left[abs_idx - 1]
                    });
                    ((p0 + p1 + 1) / 2) as u8
                } else {
                    ctx.left[7]
                }
            };
            output[y * 8 + x] = pred;
        }
    }
}

/// Vertical-right prediction.
fn predict_vr_8x8(ctx: &IntraPredContext, output: &mut [u8; 64]) {
    for y in 0..8 {
        for x in 0..8 {
            let idx = (x * 2) as i16 - y as i16;
            let pred = if idx >= 0 {
                let i = (idx / 2) as usize;
                if i < 8 {
                    ctx.top[i]
                } else {
                    ctx.top[7]
                }
            } else {
                let i = ((-idx - 1) / 2) as usize;
                if i < 8 {
                    ctx.left[i]
                } else {
                    ctx.left[7]
                }
            };
            output[y * 8 + x] = pred;
        }
    }
}

/// Horizontal-down prediction.
fn predict_hd_8x8(ctx: &IntraPredContext, output: &mut [u8; 64]) {
    for y in 0..8 {
        for x in 0..8 {
            let idx = y as i16 - (x * 2) as i16;
            let pred = if idx >= 0 {
                let i = (idx / 2) as usize;
                if i < 8 {
                    ctx.left[i]
                } else {
                    ctx.left[7]
                }
            } else {
                let i = ((-idx - 1) / 2) as usize;
                if i < 8 {
                    ctx.top[i]
                } else {
                    ctx.top[7]
                }
            };
            output[y * 8 + x] = pred;
        }
    }
}

/// Vertical-left prediction.
fn predict_vl_8x8(ctx: &IntraPredContext, output: &mut [u8; 64]) {
    for y in 0..8 {
        for x in 0..8 {
            let idx = x + (y / 2);
            let pred = if idx < 8 {
                ctx.top[idx]
            } else if idx < 15 {
                let blend_idx = idx - 8;
                if blend_idx < 7 {
                    let p0 = u16::from(ctx.top[7]);
                    let p1 = u16::from(ctx.top_right);
                    ((p0 + p1 + 1) / 2) as u8
                } else {
                    ctx.top_right
                }
            } else {
                ctx.top_right
            };
            output[y * 8 + x] = pred;
        }
    }
}

/// Horizontal-up prediction.
fn predict_hu_8x8(ctx: &IntraPredContext, output: &mut [u8; 64]) {
    for y in 0..8 {
        for x in 0..8 {
            let idx = y + (x / 2);
            let pred = if idx < 8 { ctx.left[idx] } else { ctx.left[7] };
            output[y * 8 + x] = pred;
        }
    }
}

/// Select best intra prediction mode using Sum of Absolute Differences (SAD).
///
/// # Arguments
///
/// * `original` - Original block to encode
/// * `ctx` - Prediction context
///
/// # Returns
///
/// Best mode and its SAD cost.
pub fn select_best_mode(original: &[u8; 64], ctx: &IntraPredContext) -> (IntraPredMode, u32) {
    let modes = [
        IntraPredMode::Dc,
        IntraPredMode::Vertical,
        IntraPredMode::Horizontal,
        IntraPredMode::TrueMotion,
        IntraPredMode::DiagonalDownLeft,
        IntraPredMode::DiagonalDownRight,
        IntraPredMode::VerticalRight,
        IntraPredMode::HorizontalDown,
        IntraPredMode::VerticalLeft,
        IntraPredMode::HorizontalUp,
    ];

    let mut best_mode = IntraPredMode::Dc;
    let mut best_sad = u32::MAX;

    for &mode in &modes {
        // Skip modes that require unavailable neighbors
        if mode.requires_top() && !ctx.has_top {
            continue;
        }
        if mode.requires_left() && !ctx.has_left {
            continue;
        }

        let mut predicted = [0u8; 64];
        predict_intra_8x8(ctx, mode, &mut predicted);

        let sad = calculate_sad(original, &predicted);
        if sad < best_sad {
            best_sad = sad;
            best_mode = mode;
        }
    }

    (best_mode, best_sad)
}

/// Calculate Sum of Absolute Differences between two blocks.
fn calculate_sad(block1: &[u8; 64], block2: &[u8; 64]) -> u32 {
    let mut sad = 0u32;
    for i in 0..64 {
        sad += (i32::from(block1[i]) - i32::from(block2[i])).unsigned_abs();
    }
    sad
}

/// Calculate Sum of Squared Errors between two blocks.
pub fn calculate_sse(block1: &[u8; 64], block2: &[u8; 64]) -> u32 {
    let mut sse = 0u32;
    for i in 0..64 {
        let diff = i32::from(block1[i]) - i32::from(block2[i]);
        sse += (diff * diff) as u32;
    }
    sse
}

/// Rate-distortion optimized mode selection.
///
/// Balances prediction quality with mode signaling cost.
pub fn select_mode_rdo(
    original: &[u8; 64],
    ctx: &IntraPredContext,
    lambda: f32,
) -> (IntraPredMode, f32) {
    let modes = [
        IntraPredMode::Dc,
        IntraPredMode::Vertical,
        IntraPredMode::Horizontal,
        IntraPredMode::TrueMotion,
        IntraPredMode::DiagonalDownLeft,
        IntraPredMode::DiagonalDownRight,
        IntraPredMode::VerticalRight,
        IntraPredMode::HorizontalDown,
        IntraPredMode::VerticalLeft,
        IntraPredMode::HorizontalUp,
    ];

    // Mode encoding costs (in bits)
    let mode_costs = [1.0, 3.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];

    let mut best_mode = IntraPredMode::Dc;
    let mut best_cost = f32::MAX;

    for (i, &mode) in modes.iter().enumerate() {
        if mode.requires_top() && !ctx.has_top {
            continue;
        }
        if mode.requires_left() && !ctx.has_left {
            continue;
        }

        let mut predicted = [0u8; 64];
        predict_intra_8x8(ctx, mode, &mut predicted);

        let sse = calculate_sse(original, &predicted);
        let distortion = sse as f32;
        let rate = mode_costs[i];
        let cost = distortion + lambda * rate;

        if cost < best_cost {
            best_cost = cost;
            best_mode = mode;
        }
    }

    (best_mode, best_cost)
}

/// Perform 4x4 intra prediction (for more detailed prediction).
pub fn predict_intra_4x4(
    ctx: &IntraPredContext,
    mode: IntraPredMode,
    output: &mut [u8; 16],
    offset_x: usize,
    offset_y: usize,
) {
    match mode {
        IntraPredMode::Dc => {
            let mut sum = 0u32;
            let mut count = 0u32;

            if ctx.has_top && offset_y == 0 {
                for i in 0..4 {
                    sum += u32::from(ctx.top[offset_x + i]);
                    count += 1;
                }
            }

            if ctx.has_left && offset_x == 0 {
                for i in 0..4 {
                    sum += u32::from(ctx.left[offset_y + i]);
                    count += 1;
                }
            }

            let dc = count
                .checked_div(2)
                .and_then(|half| (sum + half).checked_div(count))
                .unwrap_or(128);

            output.fill(dc as u8);
        }
        IntraPredMode::Vertical => {
            for y in 0..4 {
                for x in 0..4 {
                    output[y * 4 + x] = ctx.top[offset_x + x];
                }
            }
        }
        IntraPredMode::Horizontal => {
            for y in 0..4 {
                for x in 0..4 {
                    output[y * 4 + x] = ctx.left[offset_y + y];
                }
            }
        }
        _ => {
            // For other modes, use 8x8 prediction and extract 4x4
            let mut temp = [0u8; 64];
            predict_intra_8x8(ctx, mode, &mut temp);
            for y in 0..4 {
                for x in 0..4 {
                    output[y * 4 + x] = temp[(offset_y + y) * 8 + offset_x + x];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intra_pred_mode() {
        assert_eq!(IntraPredMode::Dc.index(), 0);
        assert_eq!(IntraPredMode::from_index(0), Some(IntraPredMode::Dc));
        assert_eq!(IntraPredMode::from_index(10), None);
    }

    #[test]
    fn test_dc_prediction() {
        let mut ctx = IntraPredContext::default();
        ctx.has_top = true;
        ctx.has_left = true;
        ctx.top.fill(100);
        ctx.left.fill(200);

        let mut output = [0u8; 64];
        predict_dc_8x8(&ctx, &mut output);

        let expected = (100 + 200) / 2;
        assert_eq!(output[0], expected as u8);
    }

    #[test]
    fn test_vertical_prediction() {
        let mut ctx = IntraPredContext::default();
        ctx.has_top = true;
        for i in 0..8 {
            ctx.top[i] = (i * 10) as u8;
        }

        let mut output = [0u8; 64];
        predict_vertical_8x8(&ctx, &mut output);

        // Check that first column matches top
        for x in 0..8 {
            assert_eq!(output[x], ctx.top[x]);
        }
    }

    #[test]
    fn test_horizontal_prediction() {
        let mut ctx = IntraPredContext::default();
        ctx.has_left = true;
        for i in 0..8 {
            ctx.left[i] = (i * 10) as u8;
        }

        let mut output = [0u8; 64];
        predict_horizontal_8x8(&ctx, &mut output);

        // Check that first row matches left
        for y in 0..8 {
            assert_eq!(output[y * 8], ctx.left[y]);
        }
    }

    #[test]
    fn test_mode_selection() {
        let original = [128u8; 64];
        let ctx = IntraPredContext::default();

        let (mode, sad) = select_best_mode(&original, &ctx);
        assert_eq!(mode, IntraPredMode::Dc); // DC should be best for uniform block
        assert_eq!(sad, 0); // Perfect match
    }

    #[test]
    fn test_sse_calculation() {
        let block1 = [100u8; 64];
        let block2 = [110u8; 64];
        let sse = calculate_sse(&block1, &block2);
        assert_eq!(sse, 64 * 10 * 10); // 64 pixels, each diff = 10
    }
}
