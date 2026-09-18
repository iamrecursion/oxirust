//! AV1 intra prediction SIMD operations.
//!
//! Implements various intra prediction modes used in AV1 encoding/decoding.

use crate::simd::traits::SimdOps;
use crate::simd::types::U8x16;

/// AV1 intra prediction modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntraMode {
    /// DC prediction (average of neighbors).
    Dc,
    /// Horizontal prediction.
    Horizontal,
    /// Vertical prediction.
    Vertical,
    /// Diagonal down-left prediction.
    DiagonalDownLeft,
    /// Diagonal down-right prediction.
    DiagonalDownRight,
    /// Vertical right prediction.
    VerticalRight,
    /// Horizontal down prediction.
    HorizontalDown,
    /// Vertical left prediction.
    VerticalLeft,
    /// Horizontal up prediction.
    HorizontalUp,
    /// True motion (paeth) prediction.
    Paeth,
    /// Smooth prediction.
    Smooth,
    /// Smooth vertical prediction.
    SmoothV,
    /// Smooth horizontal prediction.
    SmoothH,
}

/// AV1 intra prediction SIMD operations.
pub struct IntraPredSimd<S> {
    #[allow(dead_code)]
    simd: S,
}

impl<S: SimdOps> IntraPredSimd<S> {
    /// Create a new intra prediction SIMD instance.
    #[inline]
    pub const fn new(simd: S) -> Self {
        Self { simd }
    }

    /// Perform intra prediction for a 4x4 block.
    ///
    /// # Arguments
    /// * `mode` - Prediction mode
    /// * `top` - Top reference pixels (4 pixels)
    /// * `left` - Left reference pixels (4 pixels)
    /// * `top_left` - Top-left corner pixel
    /// * `dst` - Destination buffer
    /// * `stride` - Destination stride
    pub fn predict_4x4(
        &self,
        mode: IntraMode,
        top: &[u8],
        left: &[u8],
        top_left: u8,
        dst: &mut [u8],
        stride: usize,
    ) {
        match mode {
            IntraMode::Dc => self.predict_dc_4x4(top, left, dst, stride),
            IntraMode::Horizontal => self.predict_h_4x4(left, dst, stride),
            IntraMode::Vertical => self.predict_v_4x4(top, dst, stride),
            IntraMode::Paeth => self.predict_paeth_4x4(top, left, top_left, dst, stride),
            IntraMode::Smooth => self.predict_smooth_4x4(top, left, top_left, dst, stride),
            IntraMode::SmoothV => self.predict_smooth_v_4x4(top, left, dst, stride),
            IntraMode::SmoothH => self.predict_smooth_h_4x4(top, left, dst, stride),
            _ => self.predict_dc_4x4(top, left, dst, stride), // Default to DC
        }
    }

    /// Perform intra prediction for an 8x8 block.
    pub fn predict_8x8(
        &self,
        mode: IntraMode,
        top: &[u8],
        left: &[u8],
        top_left: u8,
        dst: &mut [u8],
        stride: usize,
    ) {
        match mode {
            IntraMode::Dc => self.predict_dc_8x8(top, left, dst, stride),
            IntraMode::Horizontal => self.predict_h_8x8(left, dst, stride),
            IntraMode::Vertical => self.predict_v_8x8(top, dst, stride),
            IntraMode::Paeth => self.predict_paeth_8x8(top, left, top_left, dst, stride),
            IntraMode::Smooth => self.predict_smooth_8x8(top, left, top_left, dst, stride),
            _ => self.predict_dc_8x8(top, left, dst, stride),
        }
    }

    // ========================================================================
    // 4x4 Prediction Modes
    // ========================================================================

    /// DC prediction for 4x4 block.
    fn predict_dc_4x4(&self, top: &[u8], left: &[u8], dst: &mut [u8], stride: usize) {
        // Calculate DC value as average of top and left
        let mut sum = 0u32;
        for i in 0..4 {
            if i < top.len() {
                sum += u32::from(top[i]);
            }
            if i < left.len() {
                sum += u32::from(left[i]);
            }
        }
        let dc = ((sum + 4) / 8) as u8;

        // Fill block with DC value
        for y in 0..4 {
            let offset = y * stride;
            if dst.len() >= offset + 4 {
                for x in 0..4 {
                    dst[offset + x] = dc;
                }
            }
        }
    }

    /// Horizontal prediction for 4x4 block.
    fn predict_h_4x4(&self, left: &[u8], dst: &mut [u8], stride: usize) {
        for y in 0..4 {
            let offset = y * stride;
            if dst.len() >= offset + 4 && y < left.len() {
                let pixel = left[y];
                for x in 0..4 {
                    dst[offset + x] = pixel;
                }
            }
        }
    }

    /// Vertical prediction for 4x4 block.
    fn predict_v_4x4(&self, top: &[u8], dst: &mut [u8], stride: usize) {
        if top.len() < 4 {
            return;
        }

        for y in 0..4 {
            let offset = y * stride;
            if dst.len() >= offset + 4 {
                dst[offset..offset + 4].copy_from_slice(&top[..4]);
            }
        }
    }

    /// Paeth (gradient) prediction for 4x4 block.
    fn predict_paeth_4x4(
        &self,
        top: &[u8],
        left: &[u8],
        top_left: u8,
        dst: &mut [u8],
        stride: usize,
    ) {
        for y in 0..4 {
            for x in 0..4 {
                let offset = y * stride + x;
                if offset >= dst.len() || y >= left.len() || x >= top.len() {
                    continue;
                }

                let t = top[x];
                let l = left[y];
                let tl = top_left;

                dst[offset] = self.paeth_predictor(l, t, tl);
            }
        }
    }

    /// Smooth prediction for 4x4 block.
    fn predict_smooth_4x4(
        &self,
        top: &[u8],
        left: &[u8],
        _top_left: u8,
        dst: &mut [u8],
        stride: usize,
    ) {
        // Smooth prediction blends horizontal and vertical predictions
        for y in 0..4 {
            for x in 0..4 {
                let offset = y * stride + x;
                if offset >= dst.len() || y >= left.len() || x >= top.len() {
                    continue;
                }

                let h_weight = ((4 - x) * 64 / 4) as u32;
                let v_weight = ((4 - y) * 64 / 4) as u32;
                let h_pred = u32::from(left[y]) * h_weight;
                let v_pred = u32::from(top[x]) * v_weight;

                let pred = (h_pred + v_pred + 64) / 128;
                dst[offset] = pred as u8;
            }
        }
    }

    /// Smooth vertical prediction for 4x4 block.
    fn predict_smooth_v_4x4(&self, top: &[u8], left: &[u8], dst: &mut [u8], stride: usize) {
        if top.len() < 4 || left.len() < 4 {
            return;
        }

        let bottom = left[3]; // Bottom-most left pixel

        for y in 0..4 {
            let weight = ((4 - y) * 64 / 4) as u32;
            for x in 0..4 {
                let offset = y * stride + x;
                if offset >= dst.len() || x >= top.len() {
                    continue;
                }

                let pred =
                    (u32::from(top[x]) * weight + u32::from(bottom) * (64 - weight) + 32) / 64;
                dst[offset] = pred as u8;
            }
        }
    }

    /// Smooth horizontal prediction for 4x4 block.
    fn predict_smooth_h_4x4(&self, top: &[u8], left: &[u8], dst: &mut [u8], stride: usize) {
        if top.len() < 4 || left.len() < 4 {
            return;
        }

        let right = top[3]; // Right-most top pixel

        for y in 0..4 {
            for x in 0..4 {
                let offset = y * stride + x;
                if offset >= dst.len() || y >= left.len() {
                    continue;
                }

                let weight = ((4 - x) * 64 / 4) as u32;
                let pred =
                    (u32::from(left[y]) * weight + u32::from(right) * (64 - weight) + 32) / 64;
                dst[offset] = pred as u8;
            }
        }
    }

    // ========================================================================
    // 8x8 Prediction Modes
    // ========================================================================

    /// DC prediction for 8x8 block (SIMD accelerated).
    fn predict_dc_8x8(&self, top: &[u8], left: &[u8], dst: &mut [u8], stride: usize) {
        // Calculate DC value
        let mut sum = 0u32;
        for i in 0..8 {
            if i < top.len() {
                sum += u32::from(top[i]);
            }
            if i < left.len() {
                sum += u32::from(left[i]);
            }
        }
        let dc = ((sum + 8) / 16) as u8;

        // Fill block using SIMD
        let dc_vec = U8x16::splat(dc);
        let dc_array = dc_vec.to_array();
        for y in 0..8 {
            let offset = y * stride;
            if dst.len() >= offset + 8 {
                dst[offset..offset + 8].copy_from_slice(&dc_array[..8]);
            }
        }
    }

    /// Horizontal prediction for 8x8 block.
    fn predict_h_8x8(&self, left: &[u8], dst: &mut [u8], stride: usize) {
        for y in 0..8 {
            let offset = y * stride;
            if dst.len() >= offset + 8 && y < left.len() {
                let pixel_vec = U8x16::splat(left[y]);
                let pixel_array = pixel_vec.to_array();
                dst[offset..offset + 8].copy_from_slice(&pixel_array[..8]);
            }
        }
    }

    /// Vertical prediction for 8x8 block.
    fn predict_v_8x8(&self, top: &[u8], dst: &mut [u8], stride: usize) {
        if top.len() < 8 {
            return;
        }

        for y in 0..8 {
            let offset = y * stride;
            if dst.len() >= offset + 8 {
                dst[offset..offset + 8].copy_from_slice(&top[..8]);
            }
        }
    }

    /// Paeth prediction for 8x8 block.
    fn predict_paeth_8x8(
        &self,
        top: &[u8],
        left: &[u8],
        top_left: u8,
        dst: &mut [u8],
        stride: usize,
    ) {
        for y in 0..8 {
            for x in 0..8 {
                let offset = y * stride + x;
                if offset >= dst.len() || y >= left.len() || x >= top.len() {
                    continue;
                }

                let t = top[x];
                let l = left[y];
                let tl = top_left;

                dst[offset] = self.paeth_predictor(l, t, tl);
            }
        }
    }

    /// Smooth prediction for 8x8 block.
    fn predict_smooth_8x8(
        &self,
        top: &[u8],
        left: &[u8],
        _top_left: u8,
        dst: &mut [u8],
        stride: usize,
    ) {
        for y in 0..8 {
            for x in 0..8 {
                let offset = y * stride + x;
                if offset >= dst.len() || y >= left.len() || x >= top.len() {
                    continue;
                }

                let h_weight = ((8 - x) * 64 / 8) as u32;
                let v_weight = ((8 - y) * 64 / 8) as u32;
                let h_pred = u32::from(left[y]) * h_weight;
                let v_pred = u32::from(top[x]) * v_weight;

                let pred = (h_pred + v_pred + 64) / 128;
                dst[offset] = pred as u8;
            }
        }
    }

    // ========================================================================
    // Helper Functions
    // ========================================================================

    /// Paeth predictor (gradient prediction).
    ///
    /// Selects the neighbor (left, top, or top-left) that is closest
    /// to the gradient prediction.
    fn paeth_predictor(&self, left: u8, top: u8, top_left: u8) -> u8 {
        let l = i32::from(left);
        let t = i32::from(top);
        let tl = i32::from(top_left);

        let base = l + t - tl;
        let dist_l = (base - l).abs();
        let dist_t = (base - t).abs();
        let dist_tl = (base - tl).abs();

        if dist_l <= dist_t && dist_l <= dist_tl {
            left
        } else if dist_t <= dist_tl {
            top
        } else {
            top_left
        }
    }
}
