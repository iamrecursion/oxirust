//! VP9 DCT/IDCT SIMD operations.
//!
//! Implements forward and inverse DCT transforms for VP9.

use crate::simd::traits::SimdOpsExt;
use crate::simd::types::I16x8;

/// VP9 DCT SIMD operations.
pub struct Vp9DctSimd<S> {
    simd: S,
}

impl<S: SimdOpsExt> Vp9DctSimd<S> {
    /// Create a new VP9 DCT SIMD instance.
    #[inline]
    pub const fn new(simd: S) -> Self {
        Self { simd }
    }

    /// Forward DCT 4x4 transform.
    pub fn fdct_4x4(&self, input: &[i16; 16], output: &mut [i16; 16]) {
        let mut rows = [I16x8::zero(); 4];

        // Load input
        for i in 0..4 {
            for j in 0..4 {
                rows[i][j] = input[i * 4 + j];
            }
        }

        // 1D DCT on rows
        self.fdct_4_1d(&mut rows);

        // Transpose
        let rows = self.simd.transpose_4x4_i16(&rows);

        // 1D DCT on columns
        let mut cols = rows;
        self.fdct_4_1d(&mut cols);

        // Transpose back and store
        let result = self.simd.transpose_4x4_i16(&cols);
        for i in 0..4 {
            for j in 0..4 {
                output[i * 4 + j] = result[i][j];
            }
        }
    }

    /// Inverse DCT 4x4 transform.
    pub fn idct_4x4(&self, input: &[i16; 16], output: &mut [i16; 16]) {
        let mut rows = [I16x8::zero(); 4];

        // Load input
        for i in 0..4 {
            for j in 0..4 {
                rows[i][j] = input[i * 4 + j];
            }
        }

        // 1D IDCT on rows
        self.idct_4_1d(&mut rows);

        // Transpose
        let rows = self.simd.transpose_4x4_i16(&rows);

        // 1D IDCT on columns
        let mut cols = rows;
        self.idct_4_1d(&mut cols);

        // Transpose back and store
        let result = self.simd.transpose_4x4_i16(&cols);
        for i in 0..4 {
            for j in 0..4 {
                output[i * 4 + j] = result[i][j];
            }
        }
    }

    /// Forward DCT 8x8 transform.
    pub fn fdct_8x8(&self, input: &[i16; 64], output: &mut [i16; 64]) {
        let mut rows = [I16x8::zero(); 8];

        // Load input
        for i in 0..8 {
            for j in 0..8 {
                rows[i][j] = input[i * 8 + j];
            }
        }

        // 1D DCT on rows
        self.fdct_8_1d(&mut rows);

        // Transpose
        let rows = self.simd.transpose_8x8_i16(&rows);

        // 1D DCT on columns
        let mut cols = rows;
        self.fdct_8_1d(&mut cols);

        // Transpose back and store
        let result = self.simd.transpose_8x8_i16(&cols);
        for i in 0..8 {
            for j in 0..8 {
                output[i * 8 + j] = result[i][j];
            }
        }
    }

    /// Inverse DCT 8x8 transform.
    pub fn idct_8x8(&self, input: &[i16; 64], output: &mut [i16; 64]) {
        let mut rows = [I16x8::zero(); 8];

        // Load input
        for i in 0..8 {
            for j in 0..8 {
                rows[i][j] = input[i * 8 + j];
            }
        }

        // 1D IDCT on rows
        self.idct_8_1d(&mut rows);

        // Transpose
        let rows = self.simd.transpose_8x8_i16(&rows);

        // 1D IDCT on columns
        let mut cols = rows;
        self.idct_8_1d(&mut cols);

        // Transpose back and store
        let result = self.simd.transpose_8x8_i16(&cols);
        for i in 0..8 {
            for j in 0..8 {
                output[i * 8 + j] = result[i][j];
            }
        }
    }

    /// Forward DCT 16x16 transform.
    pub fn fdct_16x16(&self, input: &[i16; 256], output: &mut [i16; 256]) {
        // Process as four 8x8 blocks for simplicity
        for block_y in 0..2 {
            for block_x in 0..2 {
                let mut block_input = [0i16; 64];
                let mut block_output = [0i16; 64];

                // Extract 8x8 block
                for y in 0..8 {
                    for x in 0..8 {
                        let src_idx = (block_y * 8 + y) * 16 + (block_x * 8 + x);
                        let dst_idx = y * 8 + x;
                        block_input[dst_idx] = input[src_idx];
                    }
                }

                // Transform 8x8 block
                self.fdct_8x8(&block_input, &mut block_output);

                // Store result
                for y in 0..8 {
                    for x in 0..8 {
                        let src_idx = y * 8 + x;
                        let dst_idx = (block_y * 8 + y) * 16 + (block_x * 8 + x);
                        output[dst_idx] = block_output[src_idx];
                    }
                }
            }
        }
    }

    /// Inverse DCT 16x16 transform.
    pub fn idct_16x16(&self, input: &[i16; 256], output: &mut [i16; 256]) {
        // Process as four 8x8 blocks
        for block_y in 0..2 {
            for block_x in 0..2 {
                let mut block_input = [0i16; 64];
                let mut block_output = [0i16; 64];

                // Extract 8x8 block
                for y in 0..8 {
                    for x in 0..8 {
                        let src_idx = (block_y * 8 + y) * 16 + (block_x * 8 + x);
                        let dst_idx = y * 8 + x;
                        block_input[dst_idx] = input[src_idx];
                    }
                }

                // Transform 8x8 block
                self.idct_8x8(&block_input, &mut block_output);

                // Store result
                for y in 0..8 {
                    for x in 0..8 {
                        let src_idx = y * 8 + x;
                        let dst_idx = (block_y * 8 + y) * 16 + (block_x * 8 + x);
                        output[dst_idx] = block_output[src_idx];
                    }
                }
            }
        }
    }

    /// Forward DCT 32x32 transform (high precision).
    pub fn fdct_32x32(&self, input: &[i16], output: &mut [i16]) {
        if input.len() < 1024 || output.len() < 1024 {
            return;
        }

        // Process as sixteen 8x8 blocks
        for block_y in 0..4 {
            for block_x in 0..4 {
                let mut block_input = [0i16; 64];
                let mut block_output = [0i16; 64];

                // Extract 8x8 block
                for y in 0..8 {
                    for x in 0..8 {
                        let src_idx = (block_y * 8 + y) * 32 + (block_x * 8 + x);
                        let dst_idx = y * 8 + x;
                        if src_idx < input.len() {
                            block_input[dst_idx] = input[src_idx];
                        }
                    }
                }

                // Transform 8x8 block
                self.fdct_8x8(&block_input, &mut block_output);

                // Store result
                for y in 0..8 {
                    for x in 0..8 {
                        let src_idx = y * 8 + x;
                        let dst_idx = (block_y * 8 + y) * 32 + (block_x * 8 + x);
                        if dst_idx < output.len() {
                            output[dst_idx] = block_output[src_idx];
                        }
                    }
                }
            }
        }
    }

    /// Inverse DCT 32x32 transform.
    pub fn idct_32x32(&self, input: &[i16], output: &mut [i16]) {
        if input.len() < 1024 || output.len() < 1024 {
            return;
        }

        // Process as sixteen 8x8 blocks
        for block_y in 0..4 {
            for block_x in 0..4 {
                let mut block_input = [0i16; 64];
                let mut block_output = [0i16; 64];

                // Extract 8x8 block
                for y in 0..8 {
                    for x in 0..8 {
                        let src_idx = (block_y * 8 + y) * 32 + (block_x * 8 + x);
                        let dst_idx = y * 8 + x;
                        if src_idx < input.len() {
                            block_input[dst_idx] = input[src_idx];
                        }
                    }
                }

                // Transform 8x8 block
                self.idct_8x8(&block_input, &mut block_output);

                // Store result
                for y in 0..8 {
                    for x in 0..8 {
                        let src_idx = y * 8 + x;
                        let dst_idx = (block_y * 8 + y) * 32 + (block_x * 8 + x);
                        if dst_idx < output.len() {
                            output[dst_idx] = block_output[src_idx];
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Internal 1D Transform Helpers
    // ========================================================================

    /// 1D forward DCT-4.
    fn fdct_4_1d(&self, rows: &mut [I16x8; 4]) {
        // Stage 1: butterflies
        let (s0, s3) = self.simd.butterfly_i16x8(rows[0], rows[3]);
        let (s1, s2) = self.simd.butterfly_i16x8(rows[1], rows[2]);

        // Stage 2: butterflies and rotations
        let (x0, x1) = self.simd.butterfly_i16x8(s0, s1);
        let (x3, x2) = self.simd.butterfly_i16x8(s3, s2);

        rows[0] = x0;
        rows[1] = x2;
        rows[2] = x1;
        rows[3] = x3;
    }

    /// 1D inverse DCT-4.
    fn idct_4_1d(&self, rows: &mut [I16x8; 4]) {
        let t0 = rows[0];
        let t1 = rows[2];
        let t2 = rows[1];
        let t3 = rows[3];

        // Stage 1
        let (s0, s1) = self.simd.butterfly_i16x8(t0, t2);
        let (s3, s2) = self.simd.butterfly_i16x8(t3, t1);

        // Stage 2
        let (x0, x3) = self.simd.butterfly_i16x8(s0, s3);
        let (x1, x2) = self.simd.butterfly_i16x8(s1, s2);

        rows[0] = x0;
        rows[1] = x1;
        rows[2] = x2;
        rows[3] = x3;
    }

    /// 1D forward DCT-8.
    fn fdct_8_1d(&self, rows: &mut [I16x8; 8]) {
        // Stage 1: butterflies
        let (s0, s7) = self.simd.butterfly_i16x8(rows[0], rows[7]);
        let (s1, s6) = self.simd.butterfly_i16x8(rows[1], rows[6]);
        let (s2, s5) = self.simd.butterfly_i16x8(rows[2], rows[5]);
        let (s3, s4) = self.simd.butterfly_i16x8(rows[3], rows[4]);

        // Stage 2
        let (t0, t3) = self.simd.butterfly_i16x8(s0, s3);
        let (t1, t2) = self.simd.butterfly_i16x8(s1, s2);
        let (t4, t7) = self.simd.butterfly_i16x8(s4, s7);
        let (t5, t6) = self.simd.butterfly_i16x8(s5, s6);

        // Stage 3
        let (u0, u1) = self.simd.butterfly_i16x8(t0, t1);
        let (u2, u3) = self.simd.butterfly_i16x8(t2, t3);
        let (u4, u5) = self.simd.butterfly_i16x8(t4, t5);
        let (u6, u7) = self.simd.butterfly_i16x8(t6, t7);

        rows[0] = u0;
        rows[1] = u4;
        rows[2] = u2;
        rows[3] = u6;
        rows[4] = u1;
        rows[5] = u5;
        rows[6] = u3;
        rows[7] = u7;
    }

    /// 1D inverse DCT-8.
    fn idct_8_1d(&self, rows: &mut [I16x8; 8]) {
        let t0 = rows[0];
        let t4 = rows[1];
        let t2 = rows[2];
        let t6 = rows[3];
        let t1 = rows[4];
        let t5 = rows[5];
        let t3 = rows[6];
        let t7 = rows[7];

        // Stage 1
        let (s0, s1) = self.simd.butterfly_i16x8(t0, t1);
        let (s2, s3) = self.simd.butterfly_i16x8(t2, t3);
        let (s4, s5) = self.simd.butterfly_i16x8(t4, t5);
        let (s6, s7) = self.simd.butterfly_i16x8(t6, t7);

        // Stage 2
        let (u0, u3) = self.simd.butterfly_i16x8(s0, s3);
        let (u1, u2) = self.simd.butterfly_i16x8(s1, s2);
        let (u4, u7) = self.simd.butterfly_i16x8(s4, s7);
        let (u5, u6) = self.simd.butterfly_i16x8(s5, s6);

        // Stage 3
        let (x0, x7) = self.simd.butterfly_i16x8(u0, u7);
        let (x1, x6) = self.simd.butterfly_i16x8(u1, u6);
        let (x2, x5) = self.simd.butterfly_i16x8(u2, u5);
        let (x3, x4) = self.simd.butterfly_i16x8(u3, u4);

        rows[0] = x0;
        rows[1] = x1;
        rows[2] = x2;
        rows[3] = x3;
        rows[4] = x4;
        rows[5] = x5;
        rows[6] = x6;
        rows[7] = x7;
    }
}
