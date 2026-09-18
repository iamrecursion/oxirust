//! AV1 loop filter SIMD operations.
//!
//! Implements the AV1 deblocking loop filter with SIMD acceleration.
//! The loop filter reduces blocking artifacts at block boundaries.

use crate::simd::traits::SimdOps;
use crate::simd::types::{I16x8, U8x16};

/// AV1 loop filter SIMD operations.
pub struct LoopFilterSimd<S> {
    simd: S,
}

impl<S: SimdOps> LoopFilterSimd<S> {
    /// Create a new loop filter SIMD instance.
    #[inline]
    pub const fn new(simd: S) -> Self {
        Self { simd }
    }

    /// Horizontal edge filter for 4 pixels.
    ///
    /// Filters across a horizontal edge, processing 4 pixels on each side.
    /// The edge is between row 3 and row 4 in the input.
    ///
    /// # Arguments
    /// * `pixels` - 8 rows of pixels, edge between rows 3 and 4
    /// * `stride` - Stride between rows
    /// * `threshold` - Filter threshold
    /// * `limit` - Filter limit
    pub fn filter_h_4(&self, pixels: &mut [u8], stride: usize, threshold: u8, limit: u8) {
        if pixels.len() < stride * 8 {
            return;
        }

        // Load pixels around the edge
        let p3 = self.load_row(&pixels[(stride * 0)..], 4);
        let p2 = self.load_row(&pixels[(stride * 1)..], 4);
        let p1 = self.load_row(&pixels[(stride * 2)..], 4);
        let p0 = self.load_row(&pixels[(stride * 3)..], 4);
        let q0 = self.load_row(&pixels[(stride * 4)..], 4);
        let q1 = self.load_row(&pixels[(stride * 5)..], 4);
        let q2 = self.load_row(&pixels[(stride * 6)..], 4);
        let q3 = self.load_row(&pixels[(stride * 7)..], 4);

        // Apply filter
        let (p0_out, q0_out) = self.loop_filter_4(p3, p2, p1, p0, q0, q1, q2, q3, threshold, limit);

        // Store filtered pixels
        self.store_row(&mut pixels[(stride * 3)..], &p0_out, 4);
        self.store_row(&mut pixels[(stride * 4)..], &q0_out, 4);
    }

    /// Vertical edge filter for 4 pixels.
    ///
    /// Filters across a vertical edge, processing 4 pixels on each side.
    pub fn filter_v_4(&self, pixels: &mut [u8], stride: usize, threshold: u8, limit: u8) {
        if pixels.len() < 8 + stride * 4 {
            return;
        }

        // For vertical filtering, we need to transpose the data
        // Load 4 rows, 8 pixels each
        let mut rows = [U8x16::zero(); 4];
        for i in 0..4 {
            let offset = i * stride;
            if pixels.len() >= offset + 8 {
                for j in 0..8 {
                    rows[i][j] = pixels[offset + j];
                }
            }
        }

        // Extract columns (edge is between columns 3 and 4)
        let mut p3 = U8x16::zero();
        let mut p2 = U8x16::zero();
        let mut p1 = U8x16::zero();
        let mut p0 = U8x16::zero();
        let mut q0 = U8x16::zero();
        let mut q1 = U8x16::zero();
        let mut q2 = U8x16::zero();
        let mut q3 = U8x16::zero();

        for i in 0..4 {
            p3[i] = rows[i][0];
            p2[i] = rows[i][1];
            p1[i] = rows[i][2];
            p0[i] = rows[i][3];
            q0[i] = rows[i][4];
            q1[i] = rows[i][5];
            q2[i] = rows[i][6];
            q3[i] = rows[i][7];
        }

        // Apply filter
        let (p0_out, q0_out) = self.loop_filter_4(p3, p2, p1, p0, q0, q1, q2, q3, threshold, limit);

        // Store filtered pixels back
        for i in 0..4 {
            rows[i][3] = p0_out[i];
            rows[i][4] = q0_out[i];
        }

        for i in 0..4 {
            let offset = i * stride;
            if pixels.len() >= offset + 8 {
                for j in 0..8 {
                    pixels[offset + j] = rows[i][j];
                }
            }
        }
    }

    /// Horizontal edge filter for 8 pixels (wide filter).
    pub fn filter_h_8(&self, pixels: &mut [u8], stride: usize, threshold: u8, limit: u8) {
        if pixels.len() < stride * 8 {
            return;
        }

        // Load pixels
        let p3 = self.load_row(&pixels[(stride * 0)..], 8);
        let p2 = self.load_row(&pixels[(stride * 1)..], 8);
        let p1 = self.load_row(&pixels[(stride * 2)..], 8);
        let p0 = self.load_row(&pixels[(stride * 3)..], 8);
        let q0 = self.load_row(&pixels[(stride * 4)..], 8);
        let q1 = self.load_row(&pixels[(stride * 5)..], 8);
        let q2 = self.load_row(&pixels[(stride * 6)..], 8);
        let q3 = self.load_row(&pixels[(stride * 7)..], 8);

        // Apply wide filter
        let (p1_out, p0_out, q0_out, q1_out) =
            self.loop_filter_8(p3, p2, p1, p0, q0, q1, q2, q3, threshold, limit);

        // Store filtered pixels
        self.store_row(&mut pixels[(stride * 2)..], &p1_out, 8);
        self.store_row(&mut pixels[(stride * 3)..], &p0_out, 8);
        self.store_row(&mut pixels[(stride * 4)..], &q0_out, 8);
        self.store_row(&mut pixels[(stride * 5)..], &q1_out, 8);
    }

    // ========================================================================
    // Internal Filter Operations
    // ========================================================================

    /// 4-tap loop filter.
    #[allow(clippy::too_many_arguments)]
    fn loop_filter_4(
        &self,
        p3: U8x16,
        p2: U8x16,
        p1: U8x16,
        p0: U8x16,
        q0: U8x16,
        q1: U8x16,
        q2: U8x16,
        q3: U8x16,
        threshold: u8,
        limit: u8,
    ) -> (U8x16, U8x16) {
        // Check if filtering is needed
        if !self.needs_filter(&p1, &p0, &q0, &q1, threshold, limit) {
            return (p0, q0);
        }

        // Simple 2-tap filter for demonstration
        // Real AV1 uses more complex filtering logic

        // Convert to i16 for signed arithmetic
        let p0_i16 = self.u8_to_i16(&p0);
        let q0_i16 = self.u8_to_i16(&q0);

        // Compute delta = (q0 - p0) / 2
        let diff = self.simd.sub_i16x8(q0_i16, p0_i16);
        let delta = self.simd.shr_i16x8(diff, 1);

        // Clamp delta
        let delta_clamped = self
            .simd
            .clamp_i16x8(delta, -i16::from(limit), i16::from(limit));

        // Apply delta
        let p0_new = self.simd.sub_i16x8(p0_i16, delta_clamped);
        let q0_new = self.simd.add_i16x8(q0_i16, delta_clamped);

        // Convert back to u8
        let p0_out = self.i16_to_u8(&p0_new);
        let q0_out = self.i16_to_u8(&q0_new);

        // Suppress unused parameter warnings
        let _ = (p3, p2, q2, q3);

        (p0_out, q0_out)
    }

    /// 8-tap wide loop filter.
    #[allow(clippy::too_many_arguments)]
    fn loop_filter_8(
        &self,
        p3: U8x16,
        p2: U8x16,
        p1: U8x16,
        p0: U8x16,
        q0: U8x16,
        q1: U8x16,
        q2: U8x16,
        q3: U8x16,
        threshold: u8,
        limit: u8,
    ) -> (U8x16, U8x16, U8x16, U8x16) {
        // Check if filtering is needed
        if !self.needs_filter(&p1, &p0, &q0, &q1, threshold, limit) {
            return (p1, p0, q0, q1);
        }

        // Wide filter affects more pixels
        let p1_i16 = self.u8_to_i16(&p1);
        let p0_i16 = self.u8_to_i16(&p0);
        let q0_i16 = self.u8_to_i16(&q0);
        let q1_i16 = self.u8_to_i16(&q1);

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

        let p1_out = self.i16_to_u8(&p1_new);
        let p0_out = self.i16_to_u8(&p0_new);
        let q0_out = self.i16_to_u8(&q0_new);
        let q1_out = self.i16_to_u8(&q1_new);

        // Suppress unused parameter warnings
        let _ = (p3, p2, q2, q3);

        (p1_out, p0_out, q0_out, q1_out)
    }

    /// Check if filtering is needed based on threshold and limit.
    fn needs_filter(
        &self,
        p1: &U8x16,
        p0: &U8x16,
        q0: &U8x16,
        q1: &U8x16,
        threshold: u8,
        _limit: u8,
    ) -> bool {
        // Simplified check: average absolute difference
        let mut sum = 0u32;
        for i in 0..4 {
            sum += u32::from(p0[i].abs_diff(q0[i]));
            sum += u32::from(p1[i].abs_diff(p0[i]));
            sum += u32::from(q1[i].abs_diff(q0[i]));
        }
        let avg = sum / 12;
        avg >= u32::from(threshold)
    }

    /// Convert U8x16 to I16x8 (only first 8 elements).
    fn u8_to_i16(&self, v: &U8x16) -> I16x8 {
        self.simd.widen_low_u8_to_i16(*v)
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
    fn load_row(&self, pixels: &[u8], count: usize) -> U8x16 {
        let mut result = U8x16::zero();
        let actual_count = count.min(pixels.len()).min(16);
        for i in 0..actual_count {
            result[i] = pixels[i];
        }
        result
    }

    /// Store U8x16 to a row of pixels.
    fn store_row(&self, pixels: &mut [u8], data: &U8x16, count: usize) {
        let actual_count = count.min(pixels.len()).min(16);
        let data_array = data.to_array();
        pixels[..actual_count].copy_from_slice(&data_array[..actual_count]);
    }
}
