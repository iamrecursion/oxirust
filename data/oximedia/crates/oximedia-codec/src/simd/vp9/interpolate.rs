//! VP9 8-tap interpolation filter SIMD operations.
//!
//! Implements sub-pixel interpolation for VP9 motion compensation.

use crate::simd::traits::SimdOps;
use crate::simd::types::I16x8;

/// VP9 interpolation SIMD operations.
pub struct Vp9InterpolateSimd<S> {
    simd: S,
}

impl<S: SimdOps> Vp9InterpolateSimd<S> {
    /// Create a new VP9 interpolation SIMD instance.
    #[inline]
    pub const fn new(simd: S) -> Self {
        Self { simd }
    }

    /// Horizontal 8-tap filter for VP9.
    ///
    /// Applies 8-tap filter horizontally for sub-pixel motion compensation.
    #[allow(clippy::too_many_arguments)]
    pub fn filter_h_8tap(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        filter_idx: usize,
        width: usize,
        height: usize,
    ) {
        let filter = &VP9_FILTERS[filter_idx % VP9_FILTERS.len()];

        for y in 0..height {
            for x in 0..width {
                let src_offset = y * src_stride + x.saturating_sub(3);
                let dst_offset = y * dst_stride + x;

                if dst_offset >= dst.len() || src_offset + 8 > src.len() {
                    continue;
                }

                let result = self.convolve_8tap_h(&src[src_offset..], filter);
                dst[dst_offset] = result;
            }
        }
    }

    /// Vertical 8-tap filter for VP9.
    #[allow(clippy::too_many_arguments)]
    pub fn filter_v_8tap(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        filter_idx: usize,
        width: usize,
        height: usize,
    ) {
        let filter = &VP9_FILTERS[filter_idx % VP9_FILTERS.len()];

        for y in 0..height {
            for x in 0..width {
                let dst_offset = y * dst_stride + x;

                if dst_offset >= dst.len() {
                    continue;
                }

                let result = self.convolve_8tap_v(src, src_stride, x, y.saturating_sub(3), filter);
                dst[dst_offset] = result;
            }
        }
    }

    /// 2D 8-tap filter (separable: horizontal then vertical).
    #[allow(clippy::too_many_arguments)]
    pub fn filter_2d_8tap(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        h_filter_idx: usize,
        v_filter_idx: usize,
        width: usize,
        height: usize,
    ) {
        // Temporary buffer for horizontal filtering
        let temp_height = height + 7;
        let mut temp = vec![0u8; temp_height * width];

        // Horizontal filtering
        self.filter_h_8tap(
            src,
            src_stride,
            &mut temp,
            width,
            h_filter_idx,
            width,
            temp_height,
        );

        // Vertical filtering
        self.filter_v_8tap(&temp, width, dst, dst_stride, v_filter_idx, width, height);
    }

    /// Bilinear interpolation (optimized path).
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
        let inv_fraction = 8 - fraction;

        for y in 0..height {
            for x in 0..width {
                let src_offset = y * src_stride + x;
                let dst_offset = y * dst_stride + x;

                if src_offset + 1 >= src.len() || dst_offset >= dst.len() {
                    continue;
                }

                let p0 = u16::from(src[src_offset]);
                let p1 = u16::from(src[src_offset + 1]);

                let result = (p0 * u16::from(inv_fraction) + p1 * u16::from(fraction) + 4) >> 3;
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
        let inv_fraction = 8 - fraction;

        for y in 0..height {
            for x in 0..width {
                let src_offset = y * src_stride + x;
                let dst_offset = y * dst_stride + x;

                if src_offset + src_stride >= src.len() || dst_offset >= dst.len() {
                    continue;
                }

                let p0 = u16::from(src[src_offset]);
                let p1 = u16::from(src[src_offset + src_stride]);

                let result = (p0 * u16::from(inv_fraction) + p1 * u16::from(fraction) + 4) >> 3;
                dst[dst_offset] = result as u8;
            }
        }
    }

    // ========================================================================
    // Internal Convolution Operations
    // ========================================================================

    /// Convolve 8 horizontal pixels using 8-tap filter.
    fn convolve_8tap_h(&self, src: &[u8], filter: &[i16; 8]) -> u8 {
        if src.len() < 8 {
            return 0;
        }

        // Load 8 source pixels
        let mut pixels = I16x8::zero();
        for i in 0..8 {
            pixels[i] = i16::from(src[i]);
        }

        // Load filter coefficients
        let coeffs = I16x8::from_array(*filter);

        // Multiply and accumulate using SIMD
        let products = self.simd.mul_i16x8(pixels, coeffs);
        let sum = self.simd.horizontal_sum_i16x8(products);

        // Round and shift: VP9 uses 7-bit fractional precision
        let result = (sum + 64) >> 7;
        result.clamp(0, 255) as u8
    }

    /// Convolve 8 vertical pixels using 8-tap filter.
    fn convolve_8tap_v(
        &self,
        src: &[u8],
        stride: usize,
        x: usize,
        y: usize,
        filter: &[i16; 8],
    ) -> u8 {
        // Load 8 vertical pixels
        let mut pixels = I16x8::zero();
        for i in 0..8 {
            let offset = (y + i) * stride + x;
            if offset < src.len() {
                pixels[i] = i16::from(src[offset]);
            }
        }

        // Load filter coefficients
        let coeffs = I16x8::from_array(*filter);

        // Multiply and accumulate
        let products = self.simd.mul_i16x8(pixels, coeffs);
        let sum = self.simd.horizontal_sum_i16x8(products);

        // Round and shift
        let result = (sum + 64) >> 7;
        result.clamp(0, 255) as u8
    }

    /// Process 4 pixels in parallel using SIMD.
    #[allow(dead_code)]
    fn convolve_8tap_h_4(&self, src: &[u8], stride: usize, filter: &[i16; 8]) -> [u8; 4] {
        let mut result = [0u8; 4];

        for i in 0..4 {
            let offset = i * stride;
            if offset + 8 <= src.len() {
                result[i] = self.convolve_8tap_h(&src[offset..], filter);
            }
        }

        result
    }

    /// SIMD-optimized 8-tap convolution for 8 pixels.
    #[allow(dead_code)]
    fn convolve_8tap_h_8_simd(&self, src: &[u8], filter: &[i16; 8]) -> [u8; 8] {
        let mut result = [0u8; 8];

        // Process each output pixel
        for i in 0..8 {
            if i + 8 <= src.len() {
                result[i] = self.convolve_8tap_h(&src[i..], filter);
            }
        }

        result
    }
}

/// VP9 8-tap interpolation filter coefficients.
///
/// VP9 defines 4 filter types for different sub-pixel positions:
/// - EIGHTTAP_REGULAR: Standard smooth filter
/// - EIGHTTAP_SMOOTH: Extra smooth filter
/// - EIGHTTAP_SHARP: Sharp filter preserving edges
/// - BILINEAR: Simple 2-tap filter
pub const VP9_FILTERS: [[i16; 8]; 16] = [
    // Bilinear (for reference)
    [0, 0, 0, 128, 0, 0, 0, 0],
    // Regular filters (1/8 to 7/8 sub-pixel positions)
    [0, 1, -5, 126, 8, -3, 1, 0],
    [0, 2, -10, 123, 18, -6, 1, 0],
    [0, 3, -15, 116, 30, -10, 2, 0],
    [0, 3, -18, 106, 44, -14, 3, 0],
    [0, 4, -20, 94, 58, -17, 3, 0],
    [0, 4, -21, 80, 74, -21, 4, 0],
    [0, 3, -17, 58, 94, -20, 4, 0],
    // Smooth filters
    [0, -3, -1, 128, 8, -3, 0, 0],
    [0, -2, 2, 126, 8, -2, 0, 0],
    [0, -2, 6, 120, 12, -4, 0, 0],
    [0, -1, 8, 112, 18, -6, 1, 0],
    // Sharp filters
    [0, 1, -3, 127, 4, -1, 0, 0],
    [0, 2, -7, 123, 12, -4, 2, 0],
    [0, 3, -11, 114, 26, -8, 4, 0],
    [0, 4, -16, 102, 44, -13, 5, 0],
];

/// Get filter index for a sub-pixel position (0-7).
///
/// # Arguments
/// * `subpel` - Sub-pixel position (0-7)
/// * `filter_type` - Filter type (0=regular, 1=smooth, 2=sharp)
///
/// # Returns
/// Index into VP9_FILTERS array
#[inline]
pub fn get_filter_index(subpel: u8, filter_type: u8) -> usize {
    if subpel == 0 {
        return 0; // No filtering needed
    }

    let base = match filter_type {
        0 => 1,  // Regular: indices 1-7
        1 => 8,  // Smooth: indices 8-11
        2 => 12, // Sharp: indices 12-15
        _ => 1,
    };

    base + (subpel as usize - 1).min(7)
}
