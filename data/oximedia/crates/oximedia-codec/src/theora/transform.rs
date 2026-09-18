// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! DCT/IDCT transforms for Theora.
//!
//! Implements the VP3-compatible 8x8 DCT and IDCT transforms used in Theora.

use crate::theora::tables::ZIGZAG_SCAN;

/// 8x8 block of coefficients.
pub type Block8x8 = [i16; 64];

/// Forward DCT transform (8x8).
///
/// Converts spatial domain coefficients to frequency domain.
///
/// # Arguments
///
/// * `input` - Input block in spatial domain (row-major order)
/// * `output` - Output block in frequency domain (zigzag order)
pub fn fdct8x8(input: &[i16; 64], output: &mut Block8x8) {
    let mut temp = [0.0f64; 64];

    // 1D DCT-II on rows
    for row in 0..8 {
        let base = row * 8;
        for k in 0..8 {
            let mut sum = 0.0;
            for n in 0..8 {
                sum += f64::from(input[base + n])
                    * ((2 * n + 1) as f64 * k as f64 * core::f64::consts::PI / 16.0).cos();
            }
            let ck = if k == 0 { 1.0 / (2.0f64).sqrt() } else { 1.0 };
            temp[base + k] = sum * ck * 0.5;
        }
    }

    // 1D DCT-II on columns
    let mut temp2 = [0.0f64; 64];
    for col in 0..8 {
        for k in 0..8 {
            let mut sum = 0.0;
            for n in 0..8 {
                sum += temp[n * 8 + col]
                    * ((2 * n + 1) as f64 * k as f64 * core::f64::consts::PI / 16.0).cos();
            }
            let ck = if k == 0 { 1.0 / (2.0f64).sqrt() } else { 1.0 };
            temp2[k * 8 + col] = sum * ck * 0.5;
        }
    }

    // Convert to zigzag order
    for i in 0..64 {
        output[i] = temp2[ZIGZAG_SCAN[i]].round() as i16;
    }
}

/// Inverse DCT transform (8x8).
///
/// Converts frequency domain coefficients to spatial domain.
///
/// # Arguments
///
/// * `input` - Input block in frequency domain (zigzag order)
/// * `output` - Output block in spatial domain (row-major order)
pub fn idct8x8(input: &Block8x8, output: &mut [i16; 64]) {
    // Convert from zigzag order
    let mut freq = [0.0f64; 64];
    for i in 0..64 {
        freq[ZIGZAG_SCAN[i]] = f64::from(input[i]);
    }

    // 1D IDCT (DCT-III) on rows
    let mut temp = [0.0f64; 64];
    for row in 0..8 {
        let base = row * 8;
        for n in 0..8 {
            let mut sum = 0.0;
            for k in 0..8 {
                let ck = if k == 0 { 1.0 / (2.0f64).sqrt() } else { 1.0 };
                sum += ck
                    * freq[base + k]
                    * ((2 * n + 1) as f64 * k as f64 * core::f64::consts::PI / 16.0).cos();
            }
            temp[base + n] = sum * 0.5;
        }
    }

    // 1D IDCT (DCT-III) on columns
    for col in 0..8 {
        for n in 0..8 {
            let mut sum = 0.0;
            for k in 0..8 {
                let ck = if k == 0 { 1.0 / (2.0f64).sqrt() } else { 1.0 };
                sum += ck
                    * temp[k * 8 + col]
                    * ((2 * n + 1) as f64 * k as f64 * core::f64::consts::PI / 16.0).cos();
            }
            output[n * 8 + col] = (sum * 0.5).round().clamp(-32768.0, 32767.0) as i16;
        }
    }
}

/// Quantize a DCT block.
///
/// # Arguments
///
/// * `input` - Input DCT coefficients
/// * `output` - Output quantized coefficients
/// * `quant_matrix` - Quantization matrix (64 values)
pub fn quantize_block(input: &Block8x8, output: &mut Block8x8, quant_matrix: &[u16; 64]) {
    for i in 0..64 {
        let coeff = i32::from(input[i]);
        let quant = i32::from(quant_matrix[i]);

        if quant == 0 {
            output[i] = 0;
            continue;
        }

        let quantized = if coeff >= 0 {
            (coeff + quant / 2) / quant
        } else {
            (coeff - quant / 2) / quant
        };

        output[i] = quantized as i16;
    }
}

/// Dequantize a DCT block.
///
/// # Arguments
///
/// * `input` - Input quantized coefficients
/// * `output` - Output dequantized coefficients
/// * `quant_matrix` - Quantization matrix (64 values)
pub fn dequantize_block(input: &Block8x8, output: &mut Block8x8, quant_matrix: &[u16; 64]) {
    for i in 0..64 {
        let coeff = i32::from(input[i]);
        let quant = i32::from(quant_matrix[i]);
        output[i] = (coeff * quant) as i16;
    }
}

/// Build quantization matrix from base matrix and quality.
///
/// # Arguments
///
/// * `base` - Base quantization matrix
/// * `quality` - Quality index (0-63, higher = better quality)
/// * `output` - Output quantization matrix
pub fn build_quant_matrix(base: &[u16; 64], quality: u8, output: &mut [u16; 64]) {
    let q = quality.min(63);
    let scale = if q < 50 {
        5000 / (u32::from(q) + 50)
    } else {
        200 - u32::from(q) * 2
    };

    for i in 0..64 {
        let val = (u32::from(base[i]) * scale + 50) / 100;
        output[i] = val.clamp(1, 255) as u16;
    }
}

/// Add residual to prediction block.
///
/// # Arguments
///
/// * `prediction` - Prediction block (8x8)
/// * `residual` - Residual coefficients (8x8, in spatial domain)
/// * `output` - Output reconstructed block (8x8)
pub fn add_residual(prediction: &[u8; 64], residual: &[i16; 64], output: &mut [u8; 64]) {
    for i in 0..64 {
        let val = i32::from(prediction[i]) + i32::from(residual[i]);
        output[i] = val.clamp(0, 255) as u8;
    }
}

/// Subtract prediction from block to get residual.
///
/// # Arguments
///
/// * `block` - Original block (8x8)
/// * `prediction` - Prediction block (8x8)
/// * `residual` - Output residual coefficients (8x8)
pub fn subtract_prediction(block: &[u8; 64], prediction: &[u8; 64], residual: &mut [i16; 64]) {
    for i in 0..64 {
        residual[i] = i16::from(block[i]) - i16::from(prediction[i]);
    }
}

/// Copy an 8x8 block from a plane.
///
/// # Arguments
///
/// * `plane` - Source plane
/// * `stride` - Stride of the plane
/// * `x` - X coordinate (in pixels)
/// * `y` - Y coordinate (in pixels)
/// * `output` - Output 8x8 block
pub fn copy_block(plane: &[u8], stride: usize, x: usize, y: usize, output: &mut [u8; 64]) {
    for i in 0..8 {
        let src_offset = (y + i) * stride + x;
        let dst_offset = i * 8;
        if src_offset + 8 <= plane.len() {
            output[dst_offset..dst_offset + 8].copy_from_slice(&plane[src_offset..src_offset + 8]);
        }
    }
}

/// Copy an 8x8 block to a plane.
///
/// # Arguments
///
/// * `block` - Source 8x8 block
/// * `plane` - Destination plane
/// * `stride` - Stride of the plane
/// * `x` - X coordinate (in pixels)
/// * `y` - Y coordinate (in pixels)
pub fn paste_block(block: &[u8; 64], plane: &mut [u8], stride: usize, x: usize, y: usize) {
    for i in 0..8 {
        let src_offset = i * 8;
        let dst_offset = (y + i) * stride + x;
        if dst_offset + 8 <= plane.len() {
            plane[dst_offset..dst_offset + 8].copy_from_slice(&block[src_offset..src_offset + 8]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_idct_roundtrip() {
        let mut input = [0i16; 64];
        for i in 0..64 {
            input[i] = ((i * 13) % 256) as i16 - 128;
        }

        let mut freq = [0i16; 64];
        fdct8x8(&input, &mut freq);

        let mut output = [0i16; 64];
        idct8x8(&freq, &mut output);

        // Check that values are close (within rounding error)
        for i in 0..64 {
            let diff = (input[i] - output[i]).abs();
            assert!(
                diff <= 2,
                "Position {i}: input={}, output={}",
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_quantization() {
        let input = [
            100i16, 50, 25, 12, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        let quant_matrix = [16u16; 64];

        let mut quantized = [0i16; 64];
        quantize_block(&input, &mut quantized, &quant_matrix);

        let mut dequantized = [0i16; 64];
        dequantize_block(&quantized, &mut dequantized, &quant_matrix);

        // Check that DC coefficient is preserved
        assert!((input[0] - dequantized[0]).abs() <= 16);
    }

    #[test]
    fn test_residual_operations() {
        let block = [128u8; 64];
        let prediction = [120u8; 64];
        let mut residual = [0i16; 64];

        subtract_prediction(&block, &prediction, &mut residual);
        assert_eq!(residual[0], 8);

        let mut reconstructed = [0u8; 64];
        add_residual(&prediction, &residual, &mut reconstructed);
        assert_eq!(reconstructed[0], 128);
    }
}
