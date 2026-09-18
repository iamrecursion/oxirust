//! VP8 DCT/IDCT transforms.
//!
//! This module implements the 4x4 Discrete Cosine Transform and inverse
//! transform used in VP8. VP8 uses only 4x4 blocks (unlike VP9 which
//! supports multiple sizes).
//!
//! The VP8 DCT is based on the WHT (Walsh-Hadamard Transform) for DC
//! coefficients of I16 macroblocks, and standard 4x4 DCT for other blocks.

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// DCT coefficient precision (14 bits).
const DCT_PRECISION: i32 = 14;

/// Rounding constant for DCT.
const DCT_ROUND: i32 = 1 << (DCT_PRECISION - 1);

/// Cosine constants scaled by 2^14.
const COS_PI_8: i32 = 11585; // cos(pi/8) * 2^14
const COS_3PI_8: i32 = 6270; // cos(3*pi/8) * 2^14
const SIN_PI_8: i32 = 15137; // sin(pi/8) * 2^14
const SIN_3PI_8: i32 = 11585; // sin(3*pi/8) * 2^14

/// Sqrt(2) scaled by 2^13.
const SQRT2: i32 = 11585;

/// 4x4 coefficient block.
pub type Block4x4 = [[i16; 4]; 4];

/// 4x4 pixel block (for reconstruction).
pub type PixelBlock4x4 = [[u8; 4]; 4];

/// Performs 1D inverse DCT on 4 coefficients.
///
/// # Arguments
///
/// * `input` - Input coefficients (4 values)
/// * `output` - Output samples (4 values)
#[allow(clippy::similar_names)]
fn idct4_1d(input: &[i32; 4], output: &mut [i32; 4]) {
    let a1 = input[0] + input[2];
    let b1 = input[0] - input[2];

    let c1 = (input[1] * COS_3PI_8 - input[3] * SIN_PI_8 + DCT_ROUND) >> DCT_PRECISION;
    let d1 = (input[1] * SIN_PI_8 + input[3] * COS_3PI_8 + DCT_ROUND) >> DCT_PRECISION;

    output[0] = a1 + d1;
    output[1] = b1 + c1;
    output[2] = b1 - c1;
    output[3] = a1 - d1;
}

/// Performs 4x4 inverse DCT on a block of coefficients.
///
/// This implements the 2D separable inverse DCT:
/// 1. Apply 1D IDCT to each row
/// 2. Apply 1D IDCT to each column
///
/// # Arguments
///
/// * `coeffs` - Input coefficient block (4x4)
/// * `output` - Output residual block (4x4), added to prediction
/// * `stride` - Stride of the output buffer
/// * `pred` - Prediction buffer to add residuals to
#[allow(clippy::cast_sign_loss)]
pub fn idct4x4(coeffs: &Block4x4, output: &mut [u8], stride: usize, pred: &[u8]) {
    let mut temp = [[0i32; 4]; 4];
    let mut residual = [[0i32; 4]; 4];

    // Convert to i32 and apply row transform
    for row in 0..4 {
        let input = [
            i32::from(coeffs[row][0]),
            i32::from(coeffs[row][1]),
            i32::from(coeffs[row][2]),
            i32::from(coeffs[row][3]),
        ];
        let mut row_out = [0i32; 4];
        idct4_1d(&input, &mut row_out);
        temp[row] = row_out;
    }

    // Transpose and apply column transform
    for col in 0..4 {
        let input = [temp[0][col], temp[1][col], temp[2][col], temp[3][col]];
        let mut col_out = [0i32; 4];
        idct4_1d(&input, &mut col_out);

        for row in 0..4 {
            residual[row][col] = col_out[row];
        }
    }

    // Add to prediction and clamp
    for row in 0..4 {
        for col in 0..4 {
            let res = (residual[row][col] + 8) >> 4; // Rounding
            let pred_val = i32::from(pred[row * stride + col]);
            let pixel = (pred_val + res).clamp(0, 255) as u8;
            output[row * stride + col] = pixel;
        }
    }
}

/// Performs 4x4 Walsh-Hadamard Transform (WHT) for DC coefficients.
///
/// VP8 uses WHT for the DC coefficients of I16 macroblocks.
///
/// # Arguments
///
/// * `input` - Input DC coefficients (4 values)
/// * `output` - Output DC values (4 values)
fn iwht4_1d(input: &[i32; 4], output: &mut [i32; 4]) {
    let a = input[0] + input[2];
    let b = input[1] + input[3];
    let c = input[1] - input[3];
    let d = input[0] - input[2];

    output[0] = a + b;
    output[1] = d + c;
    output[2] = a - b;
    output[3] = d - c;
}

/// Performs 4x4 inverse Walsh-Hadamard Transform on DC block.
///
/// # Arguments
///
/// * `coeffs` - Input DC coefficient block (4x4)
/// * `dc_out` - Output DC values (16 values, row-major)
#[allow(clippy::cast_possible_truncation)]
pub fn iwht4x4(coeffs: &Block4x4, dc_out: &mut [i16; 16]) {
    let mut temp = [[0i32; 4]; 4];
    let mut output = [[0i32; 4]; 4];

    // Apply row transform
    for row in 0..4 {
        let input = [
            i32::from(coeffs[row][0]),
            i32::from(coeffs[row][1]),
            i32::from(coeffs[row][2]),
            i32::from(coeffs[row][3]),
        ];
        let mut row_out = [0i32; 4];
        iwht4_1d(&input, &mut row_out);
        temp[row] = row_out;
    }

    // Transpose and apply column transform
    for col in 0..4 {
        let input = [temp[0][col], temp[1][col], temp[2][col], temp[3][col]];
        let mut col_out = [0i32; 4];
        iwht4_1d(&input, &mut col_out);

        for row in 0..4 {
            output[row][col] = col_out[row];
        }
    }

    // Scale and output
    for row in 0..4 {
        for col in 0..4 {
            let val = (output[row][col] + 1) >> 1; // WHT scaling
            dc_out[row * 4 + col] = val.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
    }
}

/// Dequantizes a single coefficient.
///
/// # Arguments
///
/// * `coeff` - Quantized coefficient
/// * `quant` - Quantization value
#[must_use]
#[inline]
pub fn dequantize_coeff(coeff: i16, quant: i32) -> i16 {
    let dequant = i32::from(coeff) * quant;
    dequant.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// Dequantizes a 4x4 block of coefficients.
///
/// # Arguments
///
/// * `quantized` - Quantized coefficient block
/// * `dc_quant` - DC quantization value
/// * `ac_quant` - AC quantization value
/// * `output` - Dequantized output block
pub fn dequantize_block(quantized: &Block4x4, dc_quant: i32, ac_quant: i32, output: &mut Block4x4) {
    for row in 0..4 {
        for col in 0..4 {
            let quant = if row == 0 && col == 0 {
                dc_quant
            } else {
                ac_quant
            };
            output[row][col] = dequantize_coeff(quantized[row][col], quant);
        }
    }
}

/// VP8 zigzag scan order for 4x4 block.
///
/// This defines the order in which coefficients are decoded from the bitstream.
#[allow(dead_code)]
pub static ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Converts from zigzag order to raster order.
///
/// # Arguments
///
/// * `zigzag` - Coefficients in zigzag order (16 values)
/// * `raster` - Output block in raster order (4x4)
#[allow(dead_code)]
pub fn zigzag_to_raster(zigzag: &[i16; 16], raster: &mut Block4x4) {
    for (i, &pos) in ZIGZAG_4X4.iter().enumerate() {
        let row = pos / 4;
        let col = pos % 4;
        raster[row][col] = zigzag[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idct4_1d() {
        let input = [1000i32, 0, 0, 0]; // DC only
        let mut output = [0i32; 4];
        idct4_1d(&input, &mut output);

        // DC only input should give roughly equal outputs
        for val in &output {
            assert!(*val > 0);
        }
    }

    #[test]
    fn test_iwht4_1d() {
        let input = [16i32, 0, 0, 0]; // DC only
        let mut output = [0i32; 4];
        iwht4_1d(&input, &mut output);

        // WHT of DC should distribute evenly
        assert_eq!(output[0], 16);
        assert_eq!(output[1], 16);
        assert_eq!(output[2], 16);
        assert_eq!(output[3], 16);
    }

    #[test]
    fn test_idct4x4() {
        let mut coeffs = [[0i16; 4]; 4];
        coeffs[0][0] = 100; // DC coefficient

        let pred = [128u8; 16];
        let mut output = [0u8; 16];

        idct4x4(&coeffs, &mut output, 4, &pred);

        // Output should differ from prediction
        assert!(output.iter().any(|&v| v != 128));
    }

    #[test]
    fn test_iwht4x4() {
        let mut coeffs = [[0i16; 4]; 4];
        coeffs[0][0] = 32;

        let mut dc_out = [0i16; 16];
        iwht4x4(&coeffs, &mut dc_out);

        // All DC values should be the same for DC-only input
        for val in &dc_out {
            assert_eq!(*val, 16); // 32 / 2 = 16
        }
    }

    #[test]
    fn test_dequantize_coeff() {
        assert_eq!(dequantize_coeff(10, 5), 50);
        assert_eq!(dequantize_coeff(-10, 5), -50);
        assert_eq!(dequantize_coeff(0, 100), 0);
    }

    #[test]
    fn test_dequantize_block() {
        let mut quantized = [[0i16; 4]; 4];
        quantized[0][0] = 10; // DC
        quantized[0][1] = 5; // AC
        quantized[1][0] = 3; // AC

        let mut output = [[0i16; 4]; 4];
        dequantize_block(&quantized, 4, 2, &mut output);

        assert_eq!(output[0][0], 40); // DC: 10 * 4
        assert_eq!(output[0][1], 10); // AC: 5 * 2
        assert_eq!(output[1][0], 6); // AC: 3 * 2
    }

    #[test]
    fn test_zigzag_order() {
        // Verify zigzag order starts correctly
        assert_eq!(ZIGZAG_4X4[0], 0); // (0,0)
        assert_eq!(ZIGZAG_4X4[1], 1); // (0,1)
        assert_eq!(ZIGZAG_4X4[2], 4); // (1,0)
        assert_eq!(ZIGZAG_4X4[3], 8); // (2,0)
    }

    #[test]
    fn test_zigzag_to_raster() {
        let zigzag = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut raster = [[0i16; 4]; 4];
        zigzag_to_raster(&zigzag, &mut raster);

        assert_eq!(raster[0][0], 1); // First in zigzag
        assert_eq!(raster[0][1], 2); // Second in zigzag
        assert_eq!(raster[1][0], 3); // Third in zigzag
    }
}
