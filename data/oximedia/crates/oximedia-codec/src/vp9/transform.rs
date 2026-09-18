//! VP9 Transform types and inverse transform functions.
//!
//! This module provides transform types and inverse transform implementations
//! for VP9 decoding. VP9 uses separable 2D transforms including DCT and ADST.
//!
//! Transform sizes: 4x4, 8x8, 16x16, 32x32
//! Transform types: DCT_DCT, ADST_DCT, DCT_ADST, ADST_ADST

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::wildcard_imports)]

use super::partition::TxSize;

/// Number of transform types.
pub const TX_TYPES: usize = 4;

/// Number of WHT (Walsh-Hadamard) transform sizes.
pub const WHT_SIZES: usize = 1;

/// DCT/ADST coefficient precision.
pub const TRANSFORM_PRECISION: usize = 14;

/// Rounding value for transform.
pub const TRANSFORM_ROUND: i32 = 1 << (TRANSFORM_PRECISION - 1);

/// Transform type.
///
/// VP9 uses combinations of DCT and ADST for transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum TxType {
    /// DCT in both row and column.
    #[default]
    DctDct = 0,
    /// ADST in row, DCT in column.
    AdstDct = 1,
    /// DCT in row, ADST in column.
    DctAdst = 2,
    /// ADST in both row and column.
    AdstAdst = 3,
}

impl TxType {
    /// All transform types.
    pub const ALL: [TxType; TX_TYPES] = [
        TxType::DctDct,
        TxType::AdstDct,
        TxType::DctAdst,
        TxType::AdstAdst,
    ];

    /// Converts from u8 value to `TxType`.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::DctDct),
            1 => Some(Self::AdstDct),
            2 => Some(Self::DctAdst),
            3 => Some(Self::AdstAdst),
            _ => None,
        }
    }

    /// Returns the index of this transform type.
    #[must_use]
    pub const fn index(&self) -> usize {
        *self as usize
    }

    /// Returns true if the row transform is DCT.
    #[must_use]
    pub const fn row_is_dct(&self) -> bool {
        matches!(self, Self::DctDct | Self::DctAdst)
    }

    /// Returns true if the column transform is DCT.
    #[must_use]
    pub const fn col_is_dct(&self) -> bool {
        matches!(self, Self::DctDct | Self::AdstDct)
    }

    /// Returns true if the row transform is ADST.
    #[must_use]
    pub const fn row_is_adst(&self) -> bool {
        matches!(self, Self::AdstDct | Self::AdstAdst)
    }

    /// Returns true if the column transform is ADST.
    #[must_use]
    pub const fn col_is_adst(&self) -> bool {
        matches!(self, Self::DctAdst | Self::AdstAdst)
    }

    /// Returns true if this is the identity (DCT_DCT) transform.
    #[must_use]
    pub const fn is_identity(&self) -> bool {
        matches!(self, Self::DctDct)
    }
}

impl From<TxType> for u8 {
    fn from(value: TxType) -> Self {
        value as u8
    }
}

/// DCT constants for various sizes.
mod dct_constants {
    // Cosine values scaled by 2^14 for fixed-point arithmetic

    // 4-point DCT constants
    pub const COS_4_1: i32 = 11585; // cos(1*pi/8) * 2^14
    pub const COS_4_2: i32 = 6270; // cos(3*pi/8) * 2^14
    pub const SIN_4_1: i32 = 15137; // sin(1*pi/8) * 2^14
    pub const SIN_4_2: i32 = 11585; // sin(3*pi/8) * 2^14

    // Commonly used values
    pub const SQRT2: i32 = 11585; // sqrt(2) * 2^13
}

/// Coefficient buffer for transform operations.
#[derive(Clone, Debug)]
pub struct CoeffBuffer {
    /// Coefficient data.
    pub data: Vec<i16>,
    /// Width of the buffer.
    pub width: usize,
    /// Height of the buffer.
    pub height: usize,
}

impl Default for CoeffBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl CoeffBuffer {
    /// Creates a new empty coefficient buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    /// Creates a coefficient buffer for a transform size.
    #[must_use]
    pub fn for_size(tx_size: TxSize) -> Self {
        let size = tx_size.size();
        Self {
            data: vec![0; size * size],
            width: size,
            height: size,
        }
    }

    /// Resizes the buffer.
    pub fn resize(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
        self.data.resize(width * height, 0);
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Returns the coefficient at the given position.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> i16 {
        self.data.get(row * self.width + col).copied().unwrap_or(0)
    }

    /// Sets the coefficient at the given position.
    pub fn set(&mut self, row: usize, col: usize, value: i16) {
        if row < self.height && col < self.width {
            self.data[row * self.width + col] = value;
        }
    }

    /// Returns a reference to a row.
    #[must_use]
    pub fn row(&self, row: usize) -> &[i16] {
        let start = row * self.width;
        let end = start + self.width;
        &self.data[start..end.min(self.data.len())]
    }

    /// Returns a mutable reference to a row.
    pub fn row_mut(&mut self, row: usize) -> &mut [i16] {
        let start = row * self.width;
        let end = start + self.width;
        let len = self.data.len();
        &mut self.data[start..end.min(len)]
    }
}

/// Dequantization context.
#[derive(Clone, Debug, Default)]
pub struct DequantContext {
    /// DC quantization multiplier.
    pub dc_quant: i32,
    /// AC quantization multiplier.
    pub ac_quant: i32,
    /// Segment ID for per-segment quantization.
    pub segment_id: u8,
}

impl DequantContext {
    /// Creates a new dequantization context.
    #[must_use]
    pub const fn new(dc_quant: i32, ac_quant: i32) -> Self {
        Self {
            dc_quant,
            ac_quant,
            segment_id: 0,
        }
    }

    /// Returns the quantization multiplier for a coefficient index.
    #[must_use]
    pub const fn get_quant(&self, index: usize) -> i32 {
        if index == 0 {
            self.dc_quant
        } else {
            self.ac_quant
        }
    }
}

/// VP9 zigzag scan order for 4x4 block.
pub static ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// VP9 zigzag scan order for 8x8 block.
pub static ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// VP9 zigzag scan order for 16x16 block (first 16 entries shown).
pub static ZIGZAG_16X16_PARTIAL: [usize; 16] =
    [0, 1, 16, 32, 17, 2, 3, 18, 33, 48, 64, 49, 34, 19, 4, 5];

/// VP9 zigzag scan order for 32x32 block (first 16 entries shown).
pub static ZIGZAG_32X32_PARTIAL: [usize; 16] =
    [0, 1, 32, 64, 33, 2, 3, 34, 65, 96, 128, 97, 66, 35, 4, 5];

/// Returns the zigzag scan order for a transform size.
#[must_use]
pub fn get_zigzag(tx_size: TxSize) -> &'static [usize] {
    match tx_size {
        TxSize::Tx4x4 => &ZIGZAG_4X4,
        TxSize::Tx8x8 => &ZIGZAG_8X8,
        TxSize::Tx16x16 => &ZIGZAG_16X16_PARTIAL,
        TxSize::Tx32x32 => &ZIGZAG_32X32_PARTIAL,
    }
}

/// Dequantizes coefficients in zigzag order.
///
/// # Arguments
///
/// * `quantized` - Quantized coefficients in zigzag order
/// * `output` - Output buffer for dequantized coefficients
/// * `dequant` - Dequantization context
/// * `tx_size` - Transform size
pub fn dequantize(
    quantized: &[i16],
    output: &mut CoeffBuffer,
    dequant: &DequantContext,
    tx_size: TxSize,
) {
    let size = tx_size.size();
    output.resize(size, size);
    output.clear();

    let zigzag = get_zigzag(tx_size);

    for (i, &scan_pos) in zigzag.iter().enumerate() {
        if i >= quantized.len() {
            break;
        }

        let row = scan_pos / size;
        let col = scan_pos % size;

        let quant = dequant.get_quant(scan_pos);
        let dequant_val = i32::from(quantized[i]) * quant;

        // Clamp to i16 range
        let clamped = dequant_val.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        output.set(row, col, clamped);
    }
}

/// Performs 4-point inverse DCT.
///
/// # Arguments
///
/// * `input` - Input coefficients (4 values)
/// * `output` - Output samples (4 values)
#[allow(clippy::cast_possible_truncation)]
pub fn idct4(input: &[i32; 4], output: &mut [i32; 4]) {
    use dct_constants::*;

    // Stage 1
    let s0 = input[0] + input[2];
    let s1 = input[0] - input[2];
    let s2 = (input[1] * COS_4_1 + input[3] * SIN_4_1 + TRANSFORM_ROUND) >> TRANSFORM_PRECISION;
    let s3 = (input[1] * SIN_4_1 - input[3] * COS_4_1 + TRANSFORM_ROUND) >> TRANSFORM_PRECISION;

    // Stage 2
    output[0] = s0 + s2;
    output[1] = s1 + s3;
    output[2] = s1 - s3;
    output[3] = s0 - s2;
}

/// Performs 4-point inverse ADST.
///
/// # Arguments
///
/// * `input` - Input coefficients (4 values)
/// * `output` - Output samples (4 values)
#[allow(clippy::cast_possible_truncation)]
pub fn iadst4(input: &[i32; 4], output: &mut [i32; 4]) {
    // ADST constants (scaled by 2^14)
    const S0: i32 = 5283; // sin(pi/9)
    const S1: i32 = 9929; // sin(2*pi/9)
    const S2: i32 = 13377; // sin(4*pi/9)
    const S3: i32 = 15212; // sin(5*pi/9)

    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];
    let x3 = input[3];

    let s0 = S0 * x0 + S1 * x1 + S2 * x2 + S3 * x3;
    let s1 = S1 * x0 + S3 * x1 - S0 * x2 - S2 * x3;
    let s2 = S2 * x0 - S0 * x1 - S3 * x2 + S1 * x3;
    let s3 = S3 * x0 - S2 * x1 + S1 * x2 - S0 * x3;

    output[0] = (s0 + TRANSFORM_ROUND) >> TRANSFORM_PRECISION;
    output[1] = (s1 + TRANSFORM_ROUND) >> TRANSFORM_PRECISION;
    output[2] = (s2 + TRANSFORM_ROUND) >> TRANSFORM_PRECISION;
    output[3] = (s3 + TRANSFORM_ROUND) >> TRANSFORM_PRECISION;
}

/// Performs 8-point inverse DCT.
///
/// # Arguments
///
/// * `input` - Input coefficients (8 values)
/// * `output` - Output samples (8 values)
#[allow(clippy::cast_possible_truncation)]
pub fn idct8(input: &[i32; 8], output: &mut [i32; 8]) {
    // Simplified 8-point DCT using 4-point as building block
    // First stage: butterflies
    let mut even = [0i32; 4];
    let mut odd = [0i32; 4];

    even[0] = input[0];
    even[1] = input[2];
    even[2] = input[4];
    even[3] = input[6];

    odd[0] = input[1];
    odd[1] = input[3];
    odd[2] = input[5];
    odd[3] = input[7];

    let mut even_out = [0i32; 4];
    let mut odd_out = [0i32; 4];

    idct4(&even, &mut even_out);
    idct4(&odd, &mut odd_out);

    // Combine
    for i in 0..4 {
        output[i] = even_out[i] + odd_out[i];
        output[7 - i] = even_out[i] - odd_out[i];
    }
}

/// Performs 8-point inverse ADST (skeleton).
///
/// # Arguments
///
/// * `input` - Input coefficients (8 values)
/// * `output` - Output samples (8 values)
pub fn iadst8(input: &[i32; 8], output: &mut [i32; 8]) {
    // Simplified ADST8 - in a full implementation this would use
    // the proper ADST matrix multiplication
    // For now, use a simplified approximation
    let mut temp = [0i32; 4];
    let input4 = [input[0], input[2], input[4], input[6]];
    iadst4(&input4, &mut temp);

    for i in 0..4 {
        output[i] = temp[i];
        output[i + 4] = temp[3 - i];
    }
}

/// Performs 16-point inverse DCT (skeleton).
///
/// # Arguments
///
/// * `input` - Input coefficients (16 values)
/// * `output` - Output samples (16 values)
pub fn idct16(input: &[i32; 16], output: &mut [i32; 16]) {
    // Simplified 16-point DCT using 8-point as building block
    let mut even = [0i32; 8];
    let mut odd = [0i32; 8];

    for i in 0..8 {
        even[i] = input[i * 2];
        odd[i] = input[i * 2 + 1];
    }

    let mut even_out = [0i32; 8];
    let mut odd_out = [0i32; 8];

    idct8(&even, &mut even_out);
    idct8(&odd, &mut odd_out);

    for i in 0..8 {
        output[i] = even_out[i] + odd_out[i];
        output[15 - i] = even_out[i] - odd_out[i];
    }
}

/// Performs 16-point inverse ADST (skeleton).
///
/// # Arguments
///
/// * `input` - Input coefficients (16 values)
/// * `output` - Output samples (16 values)
pub fn iadst16(input: &[i32; 16], output: &mut [i32; 16]) {
    // Simplified ADST16 - full implementation would use proper matrix
    let mut temp = [0i32; 8];
    let input8: [i32; 8] = [
        input[0], input[2], input[4], input[6], input[8], input[10], input[12], input[14],
    ];
    iadst8(&input8, &mut temp);

    for i in 0..8 {
        output[i] = temp[i];
        output[i + 8] = temp[7 - i];
    }
}

/// Performs 32-point inverse DCT (skeleton).
///
/// # Arguments
///
/// * `input` - Input coefficients (32 values)
/// * `output` - Output samples (32 values)
pub fn idct32(input: &[i32; 32], output: &mut [i32; 32]) {
    // Simplified 32-point DCT using 16-point as building block
    let mut even = [0i32; 16];
    let mut odd = [0i32; 16];

    for i in 0..16 {
        even[i] = input[i * 2];
        odd[i] = input[i * 2 + 1];
    }

    let mut even_out = [0i32; 16];
    let mut odd_out = [0i32; 16];

    idct16(&even, &mut even_out);
    idct16(&odd, &mut odd_out);

    for i in 0..16 {
        output[i] = even_out[i] + odd_out[i];
        output[31 - i] = even_out[i] - odd_out[i];
    }
}

/// Performs 4x4 Walsh-Hadamard transform (for lossless mode).
///
/// # Arguments
///
/// * `input` - Input coefficients (4 values)
/// * `output` - Output samples (4 values)
pub fn iwht4(input: &[i32; 4], output: &mut [i32; 4]) {
    let a = input[0] + input[1];
    let b = input[0] - input[1];
    let c = input[2] + input[3];
    let d = input[2] - input[3];

    output[0] = a + c;
    output[1] = b + d;
    output[2] = a - c;
    output[3] = b - d;
}

/// Performs 2D inverse transform on a 4x4 block.
///
/// # Arguments
///
/// * `coeffs` - Input coefficient buffer
/// * `output` - Output pixel buffer
/// * `output_stride` - Output buffer stride
/// * `tx_type` - Transform type (DCT/ADST combination)
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn inverse_transform_4x4(
    coeffs: &CoeffBuffer,
    output: &mut [u8],
    output_stride: usize,
    tx_type: TxType,
) {
    let mut temp = [[0i32; 4]; 4];
    let mut out = [[0i32; 4]; 4];

    // Convert coefficients to i32
    for row in 0..4 {
        for col in 0..4 {
            temp[row][col] = i32::from(coeffs.get(row, col));
        }
    }

    // Row transform
    for row in 0..4 {
        let input = temp[row];
        let mut row_out = [0i32; 4];

        if tx_type.row_is_dct() {
            idct4(&input, &mut row_out);
        } else {
            iadst4(&input, &mut row_out);
        }

        temp[row] = row_out;
    }

    // Transpose
    for row in 0..4 {
        for col in 0..4 {
            out[col][row] = temp[row][col];
        }
    }

    // Column transform
    for col in 0..4 {
        let input = out[col];
        let mut col_out = [0i32; 4];

        if tx_type.col_is_dct() {
            idct4(&input, &mut col_out);
        } else {
            iadst4(&input, &mut col_out);
        }

        out[col] = col_out;
    }

    // Final transpose and output
    for row in 0..4 {
        for col in 0..4 {
            // Add prediction and clamp to 8-bit
            let val = (out[col][row] + 8) >> 4; // Rounding
            let pixel_idx = row * output_stride + col;
            let pred = i32::from(output[pixel_idx]);
            let result = (pred + val).clamp(0, 255) as u8;
            output[pixel_idx] = result;
        }
    }
}

/// Performs 2D inverse transform on an 8x8 block.
///
/// # Arguments
///
/// * `coeffs` - Input coefficient buffer
/// * `output` - Output pixel buffer
/// * `output_stride` - Output buffer stride
/// * `tx_type` - Transform type (DCT/ADST combination)
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn inverse_transform_8x8(
    coeffs: &CoeffBuffer,
    output: &mut [u8],
    output_stride: usize,
    tx_type: TxType,
) {
    let mut temp = [[0i32; 8]; 8];
    let mut out = [[0i32; 8]; 8];

    // Convert coefficients to i32
    for row in 0..8 {
        for col in 0..8 {
            temp[row][col] = i32::from(coeffs.get(row, col));
        }
    }

    // Row transform
    for row in 0..8 {
        let input: [i32; 8] = temp[row];
        let mut row_out = [0i32; 8];

        if tx_type.row_is_dct() {
            idct8(&input, &mut row_out);
        } else {
            iadst8(&input, &mut row_out);
        }

        temp[row] = row_out;
    }

    // Transpose
    for row in 0..8 {
        for col in 0..8 {
            out[col][row] = temp[row][col];
        }
    }

    // Column transform
    for col in 0..8 {
        let input: [i32; 8] = out[col];
        let mut col_out = [0i32; 8];

        if tx_type.col_is_dct() {
            idct8(&input, &mut col_out);
        } else {
            iadst8(&input, &mut col_out);
        }

        out[col] = col_out;
    }

    // Final transpose and output
    for row in 0..8 {
        for col in 0..8 {
            let val = (out[col][row] + 16) >> 5;
            let pixel_idx = row * output_stride + col;
            let pred = i32::from(output[pixel_idx]);
            let result = (pred + val).clamp(0, 255) as u8;
            output[pixel_idx] = result;
        }
    }
}

/// Performs 2D inverse transform on a 16x16 block (skeleton).
///
/// # Arguments
///
/// * `coeffs` - Input coefficient buffer
/// * `output` - Output pixel buffer
/// * `output_stride` - Output buffer stride
/// * `tx_type` - Transform type
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn inverse_transform_16x16(
    coeffs: &CoeffBuffer,
    output: &mut [u8],
    output_stride: usize,
    _tx_type: TxType,
) {
    // Simplified 16x16 transform - only DCT_DCT for now
    let mut temp = [[0i32; 16]; 16];
    let mut out = [[0i32; 16]; 16];

    // Convert coefficients
    for row in 0..16 {
        for col in 0..16 {
            temp[row][col] = i32::from(coeffs.get(row, col));
        }
    }

    // Row transform
    for row in 0..16 {
        let input: [i32; 16] = temp[row];
        let mut row_out = [0i32; 16];
        idct16(&input, &mut row_out);
        temp[row] = row_out;
    }

    // Transpose
    for row in 0..16 {
        for col in 0..16 {
            out[col][row] = temp[row][col];
        }
    }

    // Column transform
    for col in 0..16 {
        let input: [i32; 16] = out[col];
        let mut col_out = [0i32; 16];
        idct16(&input, &mut col_out);
        out[col] = col_out;
    }

    // Output
    for row in 0..16 {
        for col in 0..16 {
            let val = (out[col][row] + 32) >> 6;
            let pixel_idx = row * output_stride + col;
            if pixel_idx < output.len() {
                let pred = i32::from(output[pixel_idx]);
                let result = (pred + val).clamp(0, 255) as u8;
                output[pixel_idx] = result;
            }
        }
    }
}

/// Performs 2D inverse transform on a 32x32 block (skeleton).
///
/// # Arguments
///
/// * `coeffs` - Input coefficient buffer
/// * `output` - Output pixel buffer
/// * `output_stride` - Output buffer stride
/// * `tx_type` - Transform type (only DCT_DCT supported for 32x32)
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn inverse_transform_32x32(
    coeffs: &CoeffBuffer,
    output: &mut [u8],
    output_stride: usize,
    _tx_type: TxType,
) {
    // 32x32 only supports DCT_DCT
    let mut temp = [[0i32; 32]; 32];
    let mut out = [[0i32; 32]; 32];

    // Convert coefficients
    for row in 0..32 {
        for col in 0..32 {
            temp[row][col] = i32::from(coeffs.get(row, col));
        }
    }

    // Row transform
    for row in 0..32 {
        let input: [i32; 32] = temp[row];
        let mut row_out = [0i32; 32];
        idct32(&input, &mut row_out);
        temp[row] = row_out;
    }

    // Transpose
    for row in 0..32 {
        for col in 0..32 {
            out[col][row] = temp[row][col];
        }
    }

    // Column transform
    for col in 0..32 {
        let input: [i32; 32] = out[col];
        let mut col_out = [0i32; 32];
        idct32(&input, &mut col_out);
        out[col] = col_out;
    }

    // Output
    for row in 0..32 {
        for col in 0..32 {
            let val = (out[col][row] + 64) >> 7;
            let pixel_idx = row * output_stride + col;
            if pixel_idx < output.len() {
                let pred = i32::from(output[pixel_idx]);
                let result = (pred + val).clamp(0, 255) as u8;
                output[pixel_idx] = result;
            }
        }
    }
}

/// Applies inverse transform for any supported size.
///
/// # Arguments
///
/// * `coeffs` - Input coefficient buffer
/// * `output` - Output pixel buffer
/// * `output_stride` - Output buffer stride
/// * `tx_size` - Transform size
/// * `tx_type` - Transform type
pub fn apply_inverse_transform(
    coeffs: &CoeffBuffer,
    output: &mut [u8],
    output_stride: usize,
    tx_size: TxSize,
    tx_type: TxType,
) {
    match tx_size {
        TxSize::Tx4x4 => inverse_transform_4x4(coeffs, output, output_stride, tx_type),
        TxSize::Tx8x8 => inverse_transform_8x8(coeffs, output, output_stride, tx_type),
        TxSize::Tx16x16 => inverse_transform_16x16(coeffs, output, output_stride, tx_type),
        TxSize::Tx32x32 => inverse_transform_32x32(coeffs, output, output_stride, tx_type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tx_type() {
        assert_eq!(TxType::DctDct.index(), 0);
        assert_eq!(TxType::AdstAdst.index(), 3);

        assert!(TxType::DctDct.row_is_dct());
        assert!(TxType::DctDct.col_is_dct());

        assert!(TxType::AdstDct.row_is_adst());
        assert!(TxType::AdstDct.col_is_dct());

        assert!(TxType::DctAdst.row_is_dct());
        assert!(TxType::DctAdst.col_is_adst());

        assert!(TxType::AdstAdst.row_is_adst());
        assert!(TxType::AdstAdst.col_is_adst());
    }

    #[test]
    fn test_tx_type_from_u8() {
        assert_eq!(TxType::from_u8(0), Some(TxType::DctDct));
        assert_eq!(TxType::from_u8(3), Some(TxType::AdstAdst));
        assert_eq!(TxType::from_u8(4), None);
    }

    #[test]
    fn test_coeff_buffer() {
        let mut buf = CoeffBuffer::for_size(TxSize::Tx4x4);
        assert_eq!(buf.width, 4);
        assert_eq!(buf.height, 4);

        buf.set(1, 2, 100);
        assert_eq!(buf.get(1, 2), 100);
        assert_eq!(buf.get(0, 0), 0);
    }

    #[test]
    fn test_coeff_buffer_row() {
        let mut buf = CoeffBuffer::for_size(TxSize::Tx4x4);
        buf.row_mut(0).copy_from_slice(&[1, 2, 3, 4]);

        assert_eq!(buf.row(0), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_dequant_context() {
        let ctx = DequantContext::new(10, 20);
        assert_eq!(ctx.get_quant(0), 10);
        assert_eq!(ctx.get_quant(1), 20);
        assert_eq!(ctx.get_quant(15), 20);
    }

    #[test]
    fn test_zigzag() {
        let zigzag = get_zigzag(TxSize::Tx4x4);
        assert_eq!(zigzag.len(), 16);
        assert_eq!(zigzag[0], 0);
        assert_eq!(zigzag[1], 1);
        assert_eq!(zigzag[2], 4);
    }

    #[test]
    fn test_dequantize() {
        let quantized = [10i16, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut output = CoeffBuffer::new();
        let dequant = DequantContext::new(4, 2);

        dequantize(&quantized, &mut output, &dequant, TxSize::Tx4x4);

        assert_eq!(output.get(0, 0), 40); // 10 * 4
        assert_eq!(output.get(0, 1), 10); // 5 * 2
        assert_eq!(output.get(1, 0), 6); // 3 * 2
    }

    #[test]
    fn test_idct4() {
        let input = [1000i32, 0, 0, 0]; // DC only
        let mut output = [0i32; 4];

        idct4(&input, &mut output);

        // All outputs should be similar for DC-only input
        for val in &output {
            assert!(*val > 0);
        }
    }

    #[test]
    fn test_iwht4() {
        let input = [4i32, 0, 0, 0];
        let mut output = [0i32; 4];

        iwht4(&input, &mut output);

        // WHT of DC should give equal distribution
        assert_eq!(output[0], 4);
        assert_eq!(output[1], 4);
        assert_eq!(output[2], 4);
        assert_eq!(output[3], 4);
    }

    #[test]
    fn test_inverse_transform_4x4() {
        let mut coeffs = CoeffBuffer::for_size(TxSize::Tx4x4);
        coeffs.set(0, 0, 100); // DC coefficient only

        let mut output = vec![128u8; 16];

        inverse_transform_4x4(&coeffs, &mut output, 4, TxType::DctDct);

        // After adding DC, all pixels should change from 128
        for val in &output {
            assert_ne!(*val, 128);
        }
    }

    #[test]
    fn test_inverse_transform_8x8() {
        let mut coeffs = CoeffBuffer::for_size(TxSize::Tx8x8);
        coeffs.set(0, 0, 100);

        let mut output = vec![128u8; 64];

        inverse_transform_8x8(&coeffs, &mut output, 8, TxType::DctDct);

        // Check that output changed
        assert!(output.iter().any(|&v| v != 128));
    }

    #[test]
    fn test_apply_inverse_transform() {
        let mut coeffs = CoeffBuffer::for_size(TxSize::Tx4x4);
        coeffs.set(0, 0, 50);

        let mut output = vec![100u8; 16];

        apply_inverse_transform(&coeffs, &mut output, 4, TxSize::Tx4x4, TxType::DctDct);

        // Output should have changed
        assert!(output.iter().any(|&v| v != 100));
    }
}
