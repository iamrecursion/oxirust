//! VP9 intra prediction SIMD operations.
//!
//! Implements intra prediction modes for VP9.

use crate::simd::traits::SimdOps;
use crate::simd::types::U8x16;

/// VP9 intra prediction modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vp9IntraMode {
    /// DC prediction.
    Dc,
    /// Vertical prediction.
    V,
    /// Horizontal prediction.
    H,
    /// True motion (gradient) prediction.
    Tm,
    /// Diagonal down-left.
    D45,
    /// Diagonal down-right.
    D135,
    /// Diagonal down-right (117 degrees).
    D117,
    /// Diagonal down-left (153 degrees).
    D153,
    /// Diagonal down-left (207 degrees).
    D207,
    /// Diagonal down-right (63 degrees).
    D63,
}

/// VP9 intra prediction SIMD operations.
pub struct Vp9IntraPredSimd<S> {
    #[allow(dead_code)]
    simd: S,
}

impl<S: SimdOps> Vp9IntraPredSimd<S> {
    /// Create a new VP9 intra prediction SIMD instance.
    #[inline]
    pub const fn new(simd: S) -> Self {
        Self { simd }
    }

    /// Perform intra prediction for a 4x4 block.
    pub fn predict_4x4(
        &self,
        mode: Vp9IntraMode,
        above: &[u8],
        left: &[u8],
        above_left: u8,
        dst: &mut [u8],
        stride: usize,
    ) {
        match mode {
            Vp9IntraMode::Dc => self.dc_4x4(above, left, dst, stride),
            Vp9IntraMode::V => self.v_4x4(above, dst, stride),
            Vp9IntraMode::H => self.h_4x4(left, dst, stride),
            Vp9IntraMode::Tm => self.tm_4x4(above, left, above_left, dst, stride),
            _ => self.dc_4x4(above, left, dst, stride),
        }
    }

    /// Perform intra prediction for an 8x8 block.
    pub fn predict_8x8(
        &self,
        mode: Vp9IntraMode,
        above: &[u8],
        left: &[u8],
        above_left: u8,
        dst: &mut [u8],
        stride: usize,
    ) {
        match mode {
            Vp9IntraMode::Dc => self.dc_8x8(above, left, dst, stride),
            Vp9IntraMode::V => self.v_8x8(above, dst, stride),
            Vp9IntraMode::H => self.h_8x8(left, dst, stride),
            Vp9IntraMode::Tm => self.tm_8x8(above, left, above_left, dst, stride),
            _ => self.dc_8x8(above, left, dst, stride),
        }
    }

    /// Perform intra prediction for a 16x16 block.
    pub fn predict_16x16(
        &self,
        mode: Vp9IntraMode,
        above: &[u8],
        left: &[u8],
        above_left: u8,
        dst: &mut [u8],
        stride: usize,
    ) {
        match mode {
            Vp9IntraMode::Dc => self.dc_16x16(above, left, dst, stride),
            Vp9IntraMode::V => self.v_16x16(above, dst, stride),
            Vp9IntraMode::H => self.h_16x16(left, dst, stride),
            Vp9IntraMode::Tm => self.tm_16x16(above, left, above_left, dst, stride),
            _ => self.dc_16x16(above, left, dst, stride),
        }
    }

    // ========================================================================
    // 4x4 Prediction Modes
    // ========================================================================

    /// DC prediction for 4x4.
    fn dc_4x4(&self, above: &[u8], left: &[u8], dst: &mut [u8], stride: usize) {
        let mut sum = 0u32;
        for i in 0..4 {
            if i < above.len() {
                sum += u32::from(above[i]);
            }
            if i < left.len() {
                sum += u32::from(left[i]);
            }
        }
        let dc = ((sum + 4) >> 3) as u8;

        for y in 0..4 {
            let offset = y * stride;
            if dst.len() >= offset + 4 {
                for pixel in &mut dst[offset..offset + 4] {
                    *pixel = dc;
                }
            }
        }
    }

    /// Vertical prediction for 4x4.
    fn v_4x4(&self, above: &[u8], dst: &mut [u8], stride: usize) {
        if above.len() < 4 {
            return;
        }

        for y in 0..4 {
            let offset = y * stride;
            if dst.len() >= offset + 4 {
                dst[offset..offset + 4].copy_from_slice(&above[..4]);
            }
        }
    }

    /// Horizontal prediction for 4x4.
    fn h_4x4(&self, left: &[u8], dst: &mut [u8], stride: usize) {
        for y in 0..4 {
            let offset = y * stride;
            if dst.len() >= offset + 4 && y < left.len() {
                let pixel = left[y];
                for i in 0..4 {
                    dst[offset + i] = pixel;
                }
            }
        }
    }

    /// True motion (gradient) prediction for 4x4.
    fn tm_4x4(&self, above: &[u8], left: &[u8], above_left: u8, dst: &mut [u8], stride: usize) {
        for y in 0..4 {
            for x in 0..4 {
                let offset = y * stride + x;
                if offset >= dst.len() || y >= left.len() || x >= above.len() {
                    continue;
                }

                let pred = i32::from(left[y]) + i32::from(above[x]) - i32::from(above_left);
                dst[offset] = pred.clamp(0, 255) as u8;
            }
        }
    }

    // ========================================================================
    // 8x8 Prediction Modes
    // ========================================================================

    /// DC prediction for 8x8.
    fn dc_8x8(&self, above: &[u8], left: &[u8], dst: &mut [u8], stride: usize) {
        let mut sum = 0u32;
        for i in 0..8 {
            if i < above.len() {
                sum += u32::from(above[i]);
            }
            if i < left.len() {
                sum += u32::from(left[i]);
            }
        }
        let dc = ((sum + 8) >> 4) as u8;

        let dc_vec = U8x16::splat(dc);
        let dc_array = dc_vec.to_array();
        for y in 0..8 {
            let offset = y * stride;
            if dst.len() >= offset + 8 {
                dst[offset..offset + 8].copy_from_slice(&dc_array[..8]);
            }
        }
    }

    /// Vertical prediction for 8x8.
    fn v_8x8(&self, above: &[u8], dst: &mut [u8], stride: usize) {
        if above.len() < 8 {
            return;
        }

        for y in 0..8 {
            let offset = y * stride;
            if dst.len() >= offset + 8 {
                dst[offset..offset + 8].copy_from_slice(&above[..8]);
            }
        }
    }

    /// Horizontal prediction for 8x8.
    fn h_8x8(&self, left: &[u8], dst: &mut [u8], stride: usize) {
        for y in 0..8 {
            let offset = y * stride;
            if dst.len() >= offset + 8 && y < left.len() {
                let pixel_vec = U8x16::splat(left[y]);
                let pixel_array = pixel_vec.to_array();
                dst[offset..offset + 8].copy_from_slice(&pixel_array[..8]);
            }
        }
    }

    /// True motion prediction for 8x8.
    fn tm_8x8(&self, above: &[u8], left: &[u8], above_left: u8, dst: &mut [u8], stride: usize) {
        for y in 0..8 {
            for x in 0..8 {
                let offset = y * stride + x;
                if offset >= dst.len() || y >= left.len() || x >= above.len() {
                    continue;
                }

                let pred = i32::from(left[y]) + i32::from(above[x]) - i32::from(above_left);
                dst[offset] = pred.clamp(0, 255) as u8;
            }
        }
    }

    // ========================================================================
    // 16x16 Prediction Modes
    // ========================================================================

    /// DC prediction for 16x16.
    fn dc_16x16(&self, above: &[u8], left: &[u8], dst: &mut [u8], stride: usize) {
        let mut sum = 0u32;
        for i in 0..16 {
            if i < above.len() {
                sum += u32::from(above[i]);
            }
            if i < left.len() {
                sum += u32::from(left[i]);
            }
        }
        let dc = ((sum + 16) >> 5) as u8;

        let dc_vec = U8x16::splat(dc);
        for y in 0..16 {
            let offset = y * stride;
            if dst.len() >= offset + 16 {
                dst[offset..offset + 16].copy_from_slice(&dc_vec.to_array());
            }
        }
    }

    /// Vertical prediction for 16x16.
    fn v_16x16(&self, above: &[u8], dst: &mut [u8], stride: usize) {
        if above.len() < 16 {
            return;
        }

        for y in 0..16 {
            let offset = y * stride;
            if dst.len() >= offset + 16 {
                dst[offset..offset + 16].copy_from_slice(&above[..16]);
            }
        }
    }

    /// Horizontal prediction for 16x16.
    fn h_16x16(&self, left: &[u8], dst: &mut [u8], stride: usize) {
        for y in 0..16 {
            let offset = y * stride;
            if dst.len() >= offset + 16 && y < left.len() {
                let pixel_vec = U8x16::splat(left[y]);
                dst[offset..offset + 16].copy_from_slice(&pixel_vec.to_array());
            }
        }
    }

    /// True motion prediction for 16x16.
    fn tm_16x16(&self, above: &[u8], left: &[u8], above_left: u8, dst: &mut [u8], stride: usize) {
        for y in 0..16 {
            for x in 0..16 {
                let offset = y * stride + x;
                if offset >= dst.len() || y >= left.len() || x >= above.len() {
                    continue;
                }

                let pred = i32::from(left[y]) + i32::from(above[x]) - i32::from(above_left);
                dst[offset] = pred.clamp(0, 255) as u8;
            }
        }
    }
}
