//! AV1 motion compensation SIMD operations.
//!
//! Implements interpolation filters and motion compensation for AV1.

use crate::simd::traits::SimdOps;
use crate::simd::types::{I16x8, I32x4, U8x16};

/// AV1 motion compensation SIMD operations.
pub struct MotionCompSimd<S> {
    simd: S,
}

impl<S: SimdOps> MotionCompSimd<S> {
    /// Create a new motion compensation SIMD instance.
    #[inline]
    pub const fn new(simd: S) -> Self {
        Self { simd }
    }

    /// Copy a block without interpolation (integer-pel motion).
    pub fn copy_block(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        width: usize,
        height: usize,
    ) {
        for y in 0..height {
            let src_offset = y * src_stride;
            let dst_offset = y * dst_stride;

            if src.len() >= src_offset + width && dst.len() >= dst_offset + width {
                dst[dst_offset..dst_offset + width]
                    .copy_from_slice(&src[src_offset..src_offset + width]);
            }
        }
    }

    /// Horizontal 8-tap interpolation filter.
    ///
    /// Applies an 8-tap filter horizontally for sub-pixel motion compensation.
    pub fn filter_h_8tap(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        coeffs: &[i16; 8],
        width: usize,
        height: usize,
    ) {
        for y in 0..height {
            for x in 0..width {
                let src_offset = y * src_stride + x;
                let dst_offset = y * dst_stride + x;

                if src.len() < src_offset + 8 || dst_offset >= dst.len() {
                    continue;
                }

                // Load 8 source pixels
                let mut pixels = I16x8::zero();
                for i in 0..8 {
                    if src_offset + i < src.len() {
                        pixels[i] = i16::from(src[src_offset + i]);
                    }
                }

                // Load filter coefficients
                let filter = I16x8::from_array(*coeffs);

                // Multiply and accumulate
                let products = self.simd.mul_i16x8(pixels, filter);
                let sum = self.simd.horizontal_sum_i16x8(products);

                // Round and shift
                let result = (sum + 64) >> 7;
                dst[dst_offset] = result.clamp(0, 255) as u8;
            }
        }
    }

    /// Vertical 8-tap interpolation filter.
    pub fn filter_v_8tap(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        coeffs: &[i16; 8],
        width: usize,
        height: usize,
    ) {
        for y in 0..height {
            for x in 0..width {
                let dst_offset = y * dst_stride + x;

                if dst_offset >= dst.len() {
                    continue;
                }

                // Load 8 vertical pixels
                let mut pixels = I16x8::zero();
                for i in 0..8 {
                    let src_offset = (y + i) * src_stride + x;
                    if src_offset < src.len() {
                        pixels[i] = i16::from(src[src_offset]);
                    }
                }

                // Load filter coefficients
                let filter = I16x8::from_array(*coeffs);

                // Multiply and accumulate
                let products = self.simd.mul_i16x8(pixels, filter);
                let sum = self.simd.horizontal_sum_i16x8(products);

                // Round and shift
                let result = (sum + 64) >> 7;
                dst[dst_offset] = result.clamp(0, 255) as u8;
            }
        }
    }

    /// 2D 8-tap interpolation (both horizontal and vertical).
    #[allow(clippy::too_many_arguments)]
    pub fn filter_2d_8tap(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        h_coeffs: &[i16; 8],
        v_coeffs: &[i16; 8],
        width: usize,
        height: usize,
    ) {
        // Allocate temporary buffer for horizontal filtering
        let temp_size = (height + 7) * width;
        let mut temp = vec![0i16; temp_size];

        // Horizontal filtering to temp buffer
        for y in 0..height + 7 {
            for x in 0..width {
                let src_offset = y * src_stride + x;
                let temp_offset = y * width + x;

                if temp_offset >= temp.len() {
                    continue;
                }

                // Load 8 horizontal pixels
                let mut pixels = I16x8::zero();
                for i in 0..8 {
                    if src_offset + i < src.len() {
                        pixels[i] = i16::from(src[src_offset + i]);
                    }
                }

                // Apply horizontal filter
                let filter = I16x8::from_array(*h_coeffs);
                let products = self.simd.mul_i16x8(pixels, filter);
                let sum = self.simd.horizontal_sum_i16x8(products);

                temp[temp_offset] = ((sum + 64) >> 7) as i16;
            }
        }

        // Vertical filtering from temp to dst
        for y in 0..height {
            for x in 0..width {
                let dst_offset = y * dst_stride + x;

                if dst_offset >= dst.len() {
                    continue;
                }

                // Load 8 vertical pixels from temp
                let mut pixels = I16x8::zero();
                for i in 0..8 {
                    let temp_offset = (y + i) * width + x;
                    if temp_offset < temp.len() {
                        pixels[i] = temp[temp_offset];
                    }
                }

                // Apply vertical filter
                let filter = I16x8::from_array(*v_coeffs);
                let products = self.simd.mul_i16x8(pixels, filter);
                let sum = self.simd.horizontal_sum_i16x8(products);

                // Round and shift
                let result = (sum + 64) >> 7;
                dst[dst_offset] = result.clamp(0, 255) as u8;
            }
        }
    }

    /// Bilinear interpolation (simple 2-tap filter).
    pub fn bilinear_h(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        fraction: u8,
        width: usize,
        height: usize,
    ) {
        let w1 = fraction;
        let w0 = 64 - w1;

        for y in 0..height {
            for x in 0..width {
                let src_offset = y * src_stride + x;
                let dst_offset = y * dst_stride + x;

                if src_offset + 1 >= src.len() || dst_offset >= dst.len() {
                    continue;
                }

                let p0 = u32::from(src[src_offset]);
                let p1 = u32::from(src[src_offset + 1]);

                let result = (p0 * u32::from(w0) + p1 * u32::from(w1) + 32) / 64;
                dst[dst_offset] = result as u8;
            }
        }
    }

    /// Bilinear vertical interpolation.
    pub fn bilinear_v(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        fraction: u8,
        width: usize,
        height: usize,
    ) {
        let w1 = fraction;
        let w0 = 64 - w1;

        for y in 0..height {
            for x in 0..width {
                let src_offset = y * src_stride + x;
                let dst_offset = y * dst_stride + x;

                if src_offset + src_stride >= src.len() || dst_offset >= dst.len() {
                    continue;
                }

                let p0 = u32::from(src[src_offset]);
                let p1 = u32::from(src[src_offset + src_stride]);

                let result = (p0 * u32::from(w0) + p1 * u32::from(w1) + 32) / 64;
                dst[dst_offset] = result as u8;
            }
        }
    }

    /// Average two blocks for bi-directional prediction.
    pub fn average_blocks(
        &self,
        src1: &[u8],
        src2: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        stride: usize,
    ) {
        for y in 0..height {
            let offset = y * stride;

            // Process 16 pixels at a time using SIMD
            let chunks = width / 16;
            for i in 0..chunks {
                let pos = offset + i * 16;

                if src1.len() < pos + 16 || src2.len() < pos + 16 || dst.len() < pos + 16 {
                    continue;
                }

                let mut v1 = U8x16::zero();
                let mut v2 = U8x16::zero();
                v1.copy_from_slice(&src1[pos..pos + 16]);
                v2.copy_from_slice(&src2[pos..pos + 16]);

                let avg = self.simd.avg_u8x16(v1, v2);
                let avg_array = avg.to_array();
                dst[pos..pos + 16].copy_from_slice(&avg_array);
            }

            // Handle remaining pixels
            for x in (chunks * 16)..width {
                let pos = offset + x;
                if src1.len() > pos && src2.len() > pos && dst.len() > pos {
                    dst[pos] = ((u16::from(src1[pos]) + u16::from(src2[pos]) + 1) / 2) as u8;
                }
            }
        }
    }

    /// Weighted prediction (combine two blocks with weights).
    #[allow(clippy::too_many_arguments)]
    pub fn weighted_pred(
        &self,
        src1: &[u8],
        src2: &[u8],
        dst: &mut [u8],
        weight1: u8,
        weight2: u8,
        width: usize,
        height: usize,
        stride: usize,
    ) {
        let total_weight = u32::from(weight1) + u32::from(weight2);

        for y in 0..height {
            for x in 0..width {
                let offset = y * stride + x;

                if src1.len() <= offset || src2.len() <= offset || dst.len() <= offset {
                    continue;
                }

                let p1 = u32::from(src1[offset]) * u32::from(weight1);
                let p2 = u32::from(src2[offset]) * u32::from(weight2);

                let result = (p1 + p2 + total_weight / 2) / total_weight;
                dst[offset] = result.clamp(0, 255) as u8;
            }
        }
    }

    /// OBMC (Overlapped Block Motion Compensation) blending.
    #[allow(clippy::too_many_arguments)]
    pub fn obmc_blend(
        &self,
        pred: &[u8],
        obmc: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        stride: usize,
        weights: &[u8],
    ) {
        for y in 0..height {
            for x in 0..width {
                let offset = y * stride + x;
                let weight_idx = (y * width + x).min(weights.len().saturating_sub(1));

                if pred.len() <= offset || obmc.len() <= offset || dst.len() <= offset {
                    continue;
                }

                let w = u32::from(weights[weight_idx]);
                let p1 = u32::from(pred[offset]) * w;
                let p2 = u32::from(obmc[offset]) * (64 - w);

                let result = (p1 + p2 + 32) / 64;
                dst[offset] = result as u8;
            }
        }
    }

    /// SIMD-optimized horizontal filtering for 4-pixel wide blocks.
    #[allow(dead_code)]
    fn filter_h_4_simd(&self, src: &[u8], coeffs: &[i16; 8]) -> [u8; 4] {
        let mut pixels = I16x8::zero();
        for i in 0..8.min(src.len()) {
            pixels[i] = i16::from(src[i]);
        }

        let filter = I16x8::from_array(*coeffs);
        let result = self.simd.pmaddwd(pixels, filter);

        let sum = self.simd.horizontal_sum_i32x4(result);
        let final_val = (sum + 64) >> 7;

        [
            final_val.clamp(0, 255) as u8,
            final_val.clamp(0, 255) as u8,
            final_val.clamp(0, 255) as u8,
            final_val.clamp(0, 255) as u8,
        ]
    }
}

/// Standard AV1 8-tap interpolation filter coefficients.
pub mod filter_coeffs {
    /// Regular filter (smooth).
    pub const REGULAR: [i16; 8] = [-1, 3, -7, 127, 8, -3, 1, 0];

    /// Sharp filter (preserves edges).
    pub const SHARP: [i16; 8] = [-1, 3, -8, 127, 8, -2, 1, 0];

    /// Smooth filter (reduces high frequencies).
    pub const SMOOTH: [i16; 8] = [-2, 6, -13, 120, 13, -6, 2, 0];

    /// Bilinear filter.
    pub const BILINEAR: [i16; 8] = [0, 0, 0, 128, 0, 0, 0, 0];
}
