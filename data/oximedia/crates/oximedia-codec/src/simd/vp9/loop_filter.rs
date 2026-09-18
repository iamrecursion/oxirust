//! VP9 loop filter SIMD operations.
//!
//! Implements the VP9 deblocking loop filter with SIMD acceleration.

use crate::simd::traits::SimdOps;
use crate::simd::types::{I16x8, U8x16};

/// VP9 loop filter SIMD operations.
pub struct Vp9LoopFilterSimd<S> {
    simd: S,
}

impl<S: SimdOps> Vp9LoopFilterSimd<S> {
    /// Create a new VP9 loop filter SIMD instance.
    #[inline]
    pub const fn new(simd: S) -> Self {
        Self { simd }
    }

    /// Horizontal edge filter (4-pixel filter).
    pub fn filter_h_edge(
        &self,
        pixels: &mut [u8],
        stride: usize,
        loop_filter_level: u8,
        limit: u8,
        blimit: u8,
        thresh: u8,
    ) {
        if pixels.len() < stride * 8 {
            return;
        }

        // Load 8 rows of pixels (4 on each side of edge)
        let p3 = self.load_row(&pixels[0..], stride, 4);
        let p2 = self.load_row(&pixels[stride..], stride, 4);
        let p1 = self.load_row(&pixels[stride * 2..], stride, 4);
        let p0 = self.load_row(&pixels[stride * 3..], stride, 4);
        let q0 = self.load_row(&pixels[stride * 4..], stride, 4);
        let q1 = self.load_row(&pixels[stride * 5..], stride, 4);
        let q2 = self.load_row(&pixels[stride * 6..], stride, 4);
        let q3 = self.load_row(&pixels[stride * 7..], stride, 4);

        // Apply filter
        let (p0_out, q0_out) = self.filter_4(p3, p2, p1, p0, q0, q1, q2, q3, limit, blimit, thresh);

        // Store filtered pixels
        self.store_row(&mut pixels[stride * 3..], &p0_out, stride, 4);
        self.store_row(&mut pixels[stride * 4..], &q0_out, stride, 4);

        // Suppress unused parameter warning
        let _ = loop_filter_level;
    }

    /// Vertical edge filter (4-pixel filter).
    pub fn filter_v_edge(
        &self,
        pixels: &mut [u8],
        stride: usize,
        loop_filter_level: u8,
        limit: u8,
        blimit: u8,
        thresh: u8,
    ) {
        // For vertical filtering, we need to work with transposed data
        // This is a simplified version that processes column by column

        for y in 0..4 {
            let offset = y * stride;
            if pixels.len() < offset + 8 {
                continue;
            }

            // Load 8 pixels across the vertical edge
            let mut row = [0u8; 8];
            row.copy_from_slice(&pixels[offset..offset + 8]);

            // Apply filter logic (similar to horizontal but on vertical data)
            let p3 = row[0];
            let p2 = row[1];
            let p1 = row[2];
            let p0 = row[3];
            let q0 = row[4];
            let q1 = row[5];
            let q2 = row[6];
            let q3 = row[7];

            if self.needs_filter_scalar(p1, p0, q0, q1, blimit, limit, thresh) {
                let (p0_new, q0_new) = self.filter_scalar(p1, p0, q0, q1);
                pixels[offset + 3] = p0_new;
                pixels[offset + 4] = q0_new;
            }

            // Suppress unused parameter warnings
            let _ = (loop_filter_level, p3, p2, q2, q3);
        }
    }

    /// Wide horizontal filter (8-pixel filter).
    pub fn filter_h_wide(
        &self,
        pixels: &mut [u8],
        stride: usize,
        loop_filter_level: u8,
        limit: u8,
        blimit: u8,
        thresh: u8,
    ) {
        if pixels.len() < stride * 8 {
            return;
        }

        // Load pixels
        let p3 = self.load_row(&pixels[0..], stride, 8);
        let p2 = self.load_row(&pixels[stride..], stride, 8);
        let p1 = self.load_row(&pixels[stride * 2..], stride, 8);
        let p0 = self.load_row(&pixels[stride * 3..], stride, 8);
        let q0 = self.load_row(&pixels[stride * 4..], stride, 8);
        let q1 = self.load_row(&pixels[stride * 5..], stride, 8);
        let q2 = self.load_row(&pixels[stride * 6..], stride, 8);
        let q3 = self.load_row(&pixels[stride * 7..], stride, 8);

        // Apply wide filter
        let (p1_out, p0_out, q0_out, q1_out) =
            self.filter_8(p3, p2, p1, p0, q0, q1, q2, q3, limit, blimit, thresh);

        // Store filtered pixels
        self.store_row(&mut pixels[stride * 2..], &p1_out, stride, 8);
        self.store_row(&mut pixels[stride * 3..], &p0_out, stride, 8);
        self.store_row(&mut pixels[stride * 4..], &q0_out, stride, 8);
        self.store_row(&mut pixels[stride * 5..], &q1_out, stride, 8);

        // Suppress unused parameter warning
        let _ = loop_filter_level;
    }

    // ========================================================================
    // Internal Filter Operations
    // ========================================================================

    /// 4-pixel loop filter.
    #[allow(clippy::too_many_arguments)]
    fn filter_4(
        &self,
        p3: U8x16,
        p2: U8x16,
        p1: U8x16,
        p0: U8x16,
        q0: U8x16,
        q1: U8x16,
        q2: U8x16,
        q3: U8x16,
        limit: u8,
        blimit: u8,
        thresh: u8,
    ) -> (U8x16, U8x16) {
        // Check if filtering is needed
        if !self.needs_filter(&p1, &p0, &q0, &q1, blimit, limit, thresh) {
            return (p0, q0);
        }

        // Convert to i16 for signed arithmetic
        let p0_i16 = self.simd.widen_low_u8_to_i16(p0);
        let q0_i16 = self.simd.widen_low_u8_to_i16(q0);

        // Compute filter: delta = (q0 - p0) / 2
        let diff = self.simd.sub_i16x8(q0_i16, p0_i16);
        let delta = self.simd.shr_i16x8(diff, 1);

        // Clamp delta
        let delta_clamped = self
            .simd
            .clamp_i16x8(delta, -i16::from(limit), i16::from(limit));

        // Apply filter
        let p0_new = self.simd.sub_i16x8(p0_i16, delta_clamped);
        let q0_new = self.simd.add_i16x8(q0_i16, delta_clamped);

        // Convert back to u8
        let p0_out = self.i16_to_u8(&p0_new);
        let q0_out = self.i16_to_u8(&q0_new);

        // Suppress unused parameter warnings
        let _ = (p3, p2, q2, q3);

        (p0_out, q0_out)
    }

    /// 8-pixel wide filter.
    #[allow(clippy::too_many_arguments)]
    fn filter_8(
        &self,
        p3: U8x16,
        p2: U8x16,
        p1: U8x16,
        p0: U8x16,
        q0: U8x16,
        q1: U8x16,
        q2: U8x16,
        q3: U8x16,
        limit: u8,
        blimit: u8,
        thresh: u8,
    ) -> (U8x16, U8x16, U8x16, U8x16) {
        // Check if filtering is needed
        if !self.needs_filter(&p1, &p0, &q0, &q1, blimit, limit, thresh) {
            return (p1, p0, q0, q1);
        }

        // Convert to i16
        let p1_i16 = self.simd.widen_low_u8_to_i16(p1);
        let p0_i16 = self.simd.widen_low_u8_to_i16(p0);
        let q0_i16 = self.simd.widen_low_u8_to_i16(q0);
        let q1_i16 = self.simd.widen_low_u8_to_i16(q1);

        // Compute delta
        let diff = self.simd.sub_i16x8(q0_i16, p0_i16);
        let delta = self.simd.shr_i16x8(diff, 1);
        let delta_clamped = self
            .simd
            .clamp_i16x8(delta, -i16::from(limit), i16::from(limit));

        // Apply weaker filtering to outer pixels
        let delta_outer = self.simd.shr_i16x8(delta_clamped, 1);

        let p1_new = self.simd.sub_i16x8(p1_i16, delta_outer);
        let p0_new = self.simd.sub_i16x8(p0_i16, delta_clamped);
        let q0_new = self.simd.add_i16x8(q0_i16, delta_clamped);
        let q1_new = self.simd.add_i16x8(q1_i16, delta_outer);

        // Convert back to u8
        let p1_out = self.i16_to_u8(&p1_new);
        let p0_out = self.i16_to_u8(&p0_new);
        let q0_out = self.i16_to_u8(&q0_new);
        let q1_out = self.i16_to_u8(&q1_new);

        // Suppress unused parameter warnings
        let _ = (p3, p2, q2, q3);

        (p1_out, p0_out, q0_out, q1_out)
    }

    /// Scalar filter for single pixels.
    fn filter_scalar(&self, p1: u8, p0: u8, q0: u8, q1: u8) -> (u8, u8) {
        let p1_i = i32::from(p1);
        let p0_i = i32::from(p0);
        let q0_i = i32::from(q0);
        let q1_i = i32::from(q1);

        // Compute filter value
        let delta = ((q0_i - p0_i) * 3 + (p1_i - q1_i)) / 8;
        let delta_clamped = delta.clamp(-16, 15);

        let p0_new = (p0_i - delta_clamped).clamp(0, 255) as u8;
        let q0_new = (q0_i + delta_clamped).clamp(0, 255) as u8;

        (p0_new, q0_new)
    }

    /// Check if filtering is needed.
    fn needs_filter(
        &self,
        p1: &U8x16,
        p0: &U8x16,
        q0: &U8x16,
        q1: &U8x16,
        blimit: u8,
        limit: u8,
        thresh: u8,
    ) -> bool {
        // Simplified check: average absolute difference
        let mut sum = 0u32;
        for i in 0..4 {
            sum += u32::from(p0[i].abs_diff(q0[i]));
            sum += u32::from(p1[i].abs_diff(p0[i]));
            sum += u32::from(q1[i].abs_diff(q0[i]));
        }
        let avg = sum / 12;

        avg >= u32::from(thresh) && avg < u32::from(blimit) && avg < u32::from(limit)
    }

    /// Scalar version of needs_filter check.
    fn needs_filter_scalar(
        &self,
        p1: u8,
        p0: u8,
        q0: u8,
        q1: u8,
        blimit: u8,
        limit: u8,
        thresh: u8,
    ) -> bool {
        let diff = p0.abs_diff(q0);
        let diff_p = p1.abs_diff(p0);
        let diff_q = q1.abs_diff(q0);

        diff >= thresh && diff < blimit && diff < limit && diff_p < limit && diff_q < limit
    }

    /// Convert I16x8 to U8x16 with saturation.
    fn i16_to_u8(&self, v: &I16x8) -> U8x16 {
        let mut result = U8x16::zero();
        for i in 0..8 {
            result[i] = v[i].clamp(0, 255) as u8;
        }
        result
    }

    /// Load a row of pixels into U8x16.
    fn load_row(&self, pixels: &[u8], _stride: usize, count: usize) -> U8x16 {
        let mut result = U8x16::zero();
        let actual_count = count.min(pixels.len()).min(16);
        for i in 0..actual_count {
            result[i] = pixels[i];
        }
        result
    }

    /// Store U8x16 to a row of pixels.
    fn store_row(&self, pixels: &mut [u8], data: &U8x16, _stride: usize, count: usize) {
        let actual_count = count.min(pixels.len()).min(16);
        let data_array = data.to_array();
        pixels[..actual_count].copy_from_slice(&data_array[..actual_count]);
    }
}
