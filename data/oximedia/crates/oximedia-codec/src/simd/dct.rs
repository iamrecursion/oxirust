//! Discrete Cosine Transform (DCT) operations.
//!
//! DCT is fundamental to video compression, converting spatial domain data
//! to frequency domain. This module provides:
//!
//! - Forward DCT for encoding
//! - Inverse DCT for decoding
//! - Support for 4x4, 8x8, 16x16, and 32x32 block sizes
//!
//! The implementations use integer arithmetic for deterministic results
//! across platforms.
//!
//! # DCT Types
//!
//! Video codecs typically use DCT-II for forward transform and DCT-III
//! (the inverse of DCT-II) for inverse transform. Modern codecs like
//! AV1 also use asymmetric DST for certain blocks.

#![forbid(unsafe_code)]
// Allow loop indexing for matrix operations
#![allow(clippy::needless_range_loop)]
// Allow truncation casts for DCT coefficient handling
#![allow(clippy::cast_possible_truncation)]

use super::scalar::ScalarFallback;
use super::traits::{SimdOps, SimdOpsExt};
use super::types::{I16x8, I32x4};

/// DCT operations using SIMD.
pub struct DctOps<S: SimdOps> {
    simd: S,
}

impl<S: SimdOps + Default> Default for DctOps<S> {
    fn default() -> Self {
        Self::new(S::default())
    }
}

impl<S: SimdOps> DctOps<S> {
    /// Create a new DCT operations instance.
    #[inline]
    #[must_use]
    pub const fn new(simd: S) -> Self {
        Self { simd }
    }

    /// Get the underlying SIMD implementation.
    #[inline]
    #[must_use]
    pub const fn simd(&self) -> &S {
        &self.simd
    }
}

/// DCT coefficients for 4x4 transform (scaled by 64).
///
/// Based on the 4-point DCT-II matrix:
/// ```text
/// [ a  a  a  a ]   a = cos(0) = 1
/// [ b  c -c -b ]   b = cos(pi/8), c = cos(3pi/8)
/// [ a -a -a  a ]
/// [ c -b  b -c ]
/// ```
#[allow(dead_code)]
pub const DCT4_COEFFS: [[i16; 4]; 4] = [
    [64, 64, 64, 64],   // row 0: all positive
    [83, 36, -36, -83], // row 1: b, c, -c, -b (scaled)
    [64, -64, -64, 64], // row 2: a, -a, -a, a
    [36, -83, 83, -36], // row 3: c, -b, b, -c (scaled)
];

/// DCT coefficients for 8x8 transform (scaled by 64).
#[allow(dead_code)]
pub const DCT8_COEFFS: [[i16; 8]; 8] = [
    [64, 64, 64, 64, 64, 64, 64, 64],
    [89, 75, 50, 18, -18, -50, -75, -89],
    [83, 36, -36, -83, -83, -36, 36, 83],
    [75, -18, -89, -50, 50, 89, 18, -75],
    [64, -64, -64, 64, 64, -64, -64, 64],
    [50, -89, 18, 75, -75, -18, 89, -50],
    [36, -83, 83, -36, -36, 83, -83, 36],
    [18, -50, 75, -89, 89, -75, 50, -18],
];

impl<S: SimdOps + SimdOpsExt> DctOps<S> {
    /// Forward 4x4 DCT.
    ///
    /// Transforms a 4x4 block of residuals to frequency coefficients.
    ///
    /// # Arguments
    /// * `input` - 4x4 input block (row-major)
    /// * `output` - 4x4 output coefficients (row-major)
    #[allow(dead_code)]
    pub fn forward_dct_4x4(&self, input: &[i16; 16], output: &mut [i16; 16]) {
        // Load input rows into vectors
        let rows = [
            I16x8::from_array([input[0], input[1], input[2], input[3], 0, 0, 0, 0]),
            I16x8::from_array([input[4], input[5], input[6], input[7], 0, 0, 0, 0]),
            I16x8::from_array([input[8], input[9], input[10], input[11], 0, 0, 0, 0]),
            I16x8::from_array([input[12], input[13], input[14], input[15], 0, 0, 0, 0]),
        ];

        // First pass: transform rows
        let mut temp = [[0i16; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0i32;
                for k in 0..4 {
                    sum += i32::from(rows[i].0[k]) * i32::from(DCT4_COEFFS[j][k]);
                }
                // Round and scale
                temp[i][j] = ((sum + 32) >> 6) as i16;
            }
        }

        // Second pass: transform columns (transpose and transform)
        for j in 0..4 {
            for i in 0..4 {
                let mut sum = 0i32;
                for k in 0..4 {
                    sum += i32::from(temp[k][j]) * i32::from(DCT4_COEFFS[i][k]);
                }
                // Round and scale
                output[i * 4 + j] = ((sum + 32) >> 6) as i16;
            }
        }
    }

    /// Inverse 4x4 DCT.
    ///
    /// Transforms 4x4 frequency coefficients back to spatial domain.
    ///
    /// # Arguments
    /// * `input` - 4x4 input coefficients (row-major)
    /// * `output` - 4x4 output block (row-major)
    #[allow(dead_code)]
    pub fn inverse_dct_4x4(&self, input: &[i16; 16], output: &mut [i16; 16]) {
        // First pass: transform columns
        let mut temp = [[0i64; 4]; 4];
        for j in 0..4 {
            for i in 0..4 {
                let mut sum = 0i64;
                for k in 0..4 {
                    sum += i64::from(input[k * 4 + j]) * i64::from(DCT4_COEFFS[k][i]);
                }
                temp[i][j] = sum;
            }
        }

        // Second pass: transform rows
        // Total normalization: 64*64*N*N = 64*64*16 = 65536 = 2^16
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0i64;
                for k in 0..4 {
                    sum += temp[i][k] * i64::from(DCT4_COEFFS[k][j]);
                }
                // Round and scale (divide by 65536 = 64*64*4*4)
                output[i * 4 + j] = ((sum + 32768) >> 16) as i16;
            }
        }
    }

    /// Forward 8x8 DCT.
    #[allow(dead_code)]
    pub fn forward_dct_8x8(&self, input: &[i16; 64], output: &mut [i16; 64]) {
        // First pass: transform rows
        let mut temp = [[0i32; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                let mut sum = 0i32;
                for k in 0..8 {
                    sum += i32::from(input[i * 8 + k]) * i32::from(DCT8_COEFFS[j][k]);
                }
                temp[i][j] = (sum + 32) >> 6;
            }
        }

        // Second pass: transform columns
        for j in 0..8 {
            for i in 0..8 {
                let mut sum = 0i32;
                for k in 0..8 {
                    sum += temp[k][j] * i32::from(DCT8_COEFFS[i][k]);
                }
                output[i * 8 + j] = ((sum + 32) >> 6) as i16;
            }
        }
    }

    /// Inverse 8x8 DCT.
    #[allow(dead_code)]
    pub fn inverse_dct_8x8(&self, input: &[i16; 64], output: &mut [i16; 64]) {
        // First pass: transform columns
        let mut temp = [[0i64; 8]; 8];
        for j in 0..8 {
            for i in 0..8 {
                let mut sum = 0i64;
                for k in 0..8 {
                    sum += i64::from(input[k * 8 + j]) * i64::from(DCT8_COEFFS[k][i]);
                }
                temp[i][j] = sum;
            }
        }

        // Second pass: transform rows
        // Total normalization: 64*64*N*N = 64*64*64 = 262144 = 2^18
        for i in 0..8 {
            for j in 0..8 {
                let mut sum = 0i64;
                for k in 0..8 {
                    sum += temp[i][k] * i64::from(DCT8_COEFFS[k][j]);
                }
                // Round and scale (divide by 262144 = 64*64*8*8)
                output[i * 8 + j] = ((sum + 131_072) >> 18) as i16;
            }
        }
    }

    /// Forward 16x16 DCT using recursive decomposition.
    ///
    /// Decomposes into 4 8x8 DCTs for efficiency.
    #[allow(dead_code)]
    pub fn forward_dct_16x16(&self, input: &[i16; 256], output: &mut [i16; 256]) {
        // For now, use direct computation
        // A real implementation would use recursive decomposition
        self.forward_dct_nxn::<16>(input, output);
    }

    /// Inverse 16x16 DCT.
    #[allow(dead_code)]
    pub fn inverse_dct_16x16(&self, input: &[i16; 256], output: &mut [i16; 256]) {
        self.inverse_dct_nxn::<16>(input, output);
    }

    /// Forward 32x32 DCT.
    #[allow(dead_code)]
    pub fn forward_dct_32x32(&self, input: &[i16; 1024], output: &mut [i16; 1024]) {
        self.forward_dct_nxn::<32>(input, output);
    }

    /// Inverse 32x32 DCT.
    #[allow(dead_code)]
    pub fn inverse_dct_32x32(&self, input: &[i16; 1024], output: &mut [i16; 1024]) {
        self.inverse_dct_nxn::<32>(input, output);
    }

    /// Generic forward DCT for `NxN` block.
    #[allow(dead_code, clippy::unused_self)]
    fn forward_dct_nxn<const N: usize>(&self, input: &[i16], output: &mut [i16]) {
        let coeffs = generate_dct_coeffs::<N>();

        // First pass: rows
        let mut temp = vec![0i32; N * N];
        for i in 0..N {
            for j in 0..N {
                let mut sum = 0i32;
                for k in 0..N {
                    sum += i32::from(input[i * N + k]) * coeffs[j][k];
                }
                temp[i * N + j] = (sum + 32) >> 6;
            }
        }

        // Second pass: columns
        for j in 0..N {
            for i in 0..N {
                let mut sum = 0i32;
                for k in 0..N {
                    sum += temp[k * N + j] * coeffs[i][k];
                }
                output[i * N + j] = ((sum + 32) >> 6) as i16;
            }
        }
    }

    /// Generic inverse DCT for `NxN` block.
    #[allow(dead_code, clippy::unused_self)]
    fn inverse_dct_nxn<const N: usize>(&self, input: &[i16], output: &mut [i16]) {
        let coeffs = generate_dct_coeffs::<N>();

        // Calculate shift: 12 + 2*log2(N)
        // N=4: shift=16, N=8: shift=18, N=16: shift=20, N=32: shift=22
        let n_shift = (N as u32).trailing_zeros();
        let total_shift = 12 + 2 * n_shift;
        let round = 1i64 << (total_shift - 1);

        // First pass: columns
        let mut temp = vec![0i64; N * N];
        for j in 0..N {
            for i in 0..N {
                let mut sum = 0i64;
                for k in 0..N {
                    sum += i64::from(input[k * N + j]) * i64::from(coeffs[k][i]);
                }
                temp[i * N + j] = sum;
            }
        }

        // Second pass: rows
        for i in 0..N {
            for j in 0..N {
                let mut sum = 0i64;
                for k in 0..N {
                    sum += temp[i * N + k] * i64::from(coeffs[k][j]);
                }
                output[i * N + j] = ((sum + round) >> total_shift) as i16;
            }
        }
    }

    /// Butterfly operation for DCT.
    #[inline]
    #[allow(dead_code)]
    pub fn butterfly_add(&self, a: I16x8, b: I16x8) -> I16x8 {
        self.simd.add_i16x8(a, b)
    }

    /// Butterfly operation for DCT (subtraction).
    #[inline]
    #[allow(dead_code)]
    pub fn butterfly_sub(&self, a: I16x8, b: I16x8) -> I16x8 {
        self.simd.sub_i16x8(a, b)
    }

    /// Multiply-add for DCT coefficients.
    #[inline]
    #[allow(dead_code)]
    pub fn dct_madd(&self, a: I16x8, coeff: I16x8) -> I32x4 {
        self.simd.pmaddwd(a, coeff)
    }
}

/// Generate DCT coefficients for `NxN` transform.
///
/// Uses the DCT-II formula: C[k][n] = cos(pi * k * (2n + 1) / (2N))
#[allow(clippy::cast_precision_loss)]
fn generate_dct_coeffs<const N: usize>() -> Vec<Vec<i32>> {
    let mut coeffs = vec![vec![0i32; N]; N];
    let pi = std::f64::consts::PI;
    let n_f64 = N as f64;

    for k in 0..N {
        for n in 0..N {
            let angle = pi * (k as f64) * (2.0 * (n as f64) + 1.0) / (2.0 * n_f64);
            coeffs[k][n] = (angle.cos() * 64.0).round() as i32;
        }
    }

    coeffs
}

/// Create a DCT operations instance with scalar fallback.
#[inline]
#[must_use]
pub fn dct_ops() -> DctOps<ScalarFallback> {
    DctOps::new(ScalarFallback::new())
}

/// Quantize DCT coefficients.
///
/// # Arguments
/// * `coeffs` - DCT coefficients
/// * `qp` - Quantization parameter (0-51 for H.264/AV1)
/// * `output` - Quantized coefficients
#[allow(dead_code)]
pub fn quantize_4x4(coeffs: &[i16; 16], qp: u8, output: &mut [i16; 16]) {
    // Simplified quantization (real implementation uses tables)
    let scale: i32 = 1 << (15 - (qp / 6));

    for (i, &c) in coeffs.iter().enumerate() {
        let val = i32::from(c);
        let sign = if val < 0 { -1i32 } else { 1i32 };
        output[i] = (sign * ((val.abs() * scale + (1 << 14)) >> 15)) as i16;
    }
}

/// Dequantize DCT coefficients.
#[allow(dead_code)]
pub fn dequantize_4x4(coeffs: &[i16; 16], qp: u8, output: &mut [i16; 16]) {
    let scale = 1 << (qp / 6);

    for (i, &c) in coeffs.iter().enumerate() {
        output[i] = (i32::from(c) * scale) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct4_coeffs_orthogonality() {
        // Verify that DCT matrix rows are approximately orthogonal
        for i in 0..4 {
            for j in i + 1..4 {
                let dot: i32 = (0..4)
                    .map(|k| i32::from(DCT4_COEFFS[i][k]) * i32::from(DCT4_COEFFS[j][k]))
                    .sum();
                // Dot product of different rows should be near zero
                assert!(
                    dot.abs() < 100,
                    "Rows {} and {} not orthogonal: {}",
                    i,
                    j,
                    dot
                );
            }
        }
    }

    #[test]
    fn test_forward_inverse_4x4_identity() {
        let ops = dct_ops();

        // Test with a simple block
        let input = [
            100, 102, 104, 106, 110, 112, 114, 116, 120, 122, 124, 126, 130, 132, 134, 136,
        ];

        let mut dct_output = [0i16; 16];
        let mut reconstructed = [0i16; 16];

        ops.forward_dct_4x4(&input, &mut dct_output);
        ops.inverse_dct_4x4(&dct_output, &mut reconstructed);

        // Reconstructed should be close to original
        for i in 0..16 {
            let diff = (i32::from(input[i]) - i32::from(reconstructed[i])).abs();
            assert!(
                diff <= 2,
                "Mismatch at {}: {} vs {}",
                i,
                input[i],
                reconstructed[i]
            );
        }
    }

    #[test]
    fn test_forward_inverse_8x8_identity() {
        let ops = dct_ops();

        // Test with constant block
        let input = [128i16; 64];
        let mut dct_output = [0i16; 64];
        let mut reconstructed = [0i16; 64];

        ops.forward_dct_8x8(&input, &mut dct_output);

        // DC coefficient should be large, others near zero
        assert!(dct_output[0].abs() > 100);
        for i in 1..64 {
            assert!(
                dct_output[i].abs() < 10,
                "Non-DC coeff {} too large: {}",
                i,
                dct_output[i]
            );
        }

        ops.inverse_dct_8x8(&dct_output, &mut reconstructed);

        // Reconstructed should be close to original
        for i in 0..64 {
            let diff = (i32::from(input[i]) - i32::from(reconstructed[i])).abs();
            assert!(
                diff <= 2,
                "Mismatch at {}: {} vs {}",
                i,
                input[i],
                reconstructed[i]
            );
        }
    }

    #[test]
    fn test_dct_zero_input() {
        let ops = dct_ops();

        let input = [0i16; 16];
        let mut output = [1i16; 16]; // Initialize with non-zero

        ops.forward_dct_4x4(&input, &mut output);

        // All outputs should be zero
        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, 0, "Non-zero output at {}: {}", i, v);
        }
    }

    #[test]
    fn test_dct_dc_only() {
        let ops = dct_ops();

        // Constant input should produce only DC coefficient
        let input = [64i16; 16];
        let mut output = [0i16; 16];

        ops.forward_dct_4x4(&input, &mut output);

        // DC coefficient should be non-zero
        assert!(output[0] != 0);

        // AC coefficients should be near zero
        for (i, &v) in output.iter().enumerate().skip(1) {
            assert!(v.abs() < 5, "AC coeff {} too large: {}", i, v);
        }
    }

    #[test]
    fn test_quantize_dequantize() {
        let coeffs = [100i16, -50, 25, -12, 6, -3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut quantized = [0i16; 16];
        let mut dequantized = [0i16; 16];

        // Quantize with moderate QP
        quantize_4x4(&coeffs, 20, &mut quantized);

        // Large coefficients should survive quantization
        assert!(quantized[0] != 0);

        // Dequantize
        dequantize_4x4(&quantized, 20, &mut dequantized);

        // Should be approximately the same
        let dc_diff = (i32::from(coeffs[0]) - i32::from(dequantized[0])).abs();
        assert!(
            dc_diff < i32::from(coeffs[0]) / 2,
            "DC diff too large: {}",
            dc_diff
        );
    }

    #[test]
    fn test_generate_dct_coeffs() {
        let coeffs = generate_dct_coeffs::<4>();

        assert_eq!(coeffs.len(), 4);
        assert_eq!(coeffs[0].len(), 4);

        // First row should be all positive (cos(0) = 1)
        for &c in &coeffs[0] {
            assert!(c > 0);
        }
    }

    #[test]
    fn test_dct8_coeffs() {
        // Verify DCT8 coefficient properties
        // First row should be constant (all 64)
        assert_eq!(DCT8_COEFFS[0], [64, 64, 64, 64, 64, 64, 64, 64]);

        // Row 4 should alternate: +64, -64, -64, +64, +64, -64, -64, +64
        assert_eq!(DCT8_COEFFS[4], [64, -64, -64, 64, 64, -64, -64, 64]);
    }
}
