//! Quantization tables and functions for H.263.
//!
//! This module provides quantization and dequantization for DCT coefficients
//! according to ITU-T H.263 specification.

/// Default quantization parameter (QP) value.
pub const DEFAULT_QP: u8 = 10;

/// Minimum QP value.
pub const MIN_QP: u8 = 1;

/// Maximum QP value.
pub const MAX_QP: u8 = 31;

/// Quantization step sizes for each QP value (1-31).
///
/// The quantization step doubles approximately every 6 QP units.
pub const QUANT_STEP: &[i32] = &[
    0,  // QP=0 (invalid)
    2,  // QP=1
    4,  // QP=2
    6,  // QP=3
    8,  // QP=4
    10, // QP=5
    12, // QP=6
    14, // QP=7
    16, // QP=8
    18, // QP=9
    20, // QP=10
    22, // QP=11
    24, // QP=12
    26, // QP=13
    28, // QP=14
    30, // QP=15
    32, // QP=16
    34, // QP=17
    36, // QP=18
    38, // QP=19
    40, // QP=20
    42, // QP=21
    44, // QP=22
    46, // QP=23
    48, // QP=24
    50, // QP=25
    52, // QP=26
    54, // QP=27
    56, // QP=28
    58, // QP=29
    60, // QP=30
    62, // QP=31
];

/// Zigzag scan order for 8x8 blocks.
///
/// Maps zigzag index to (row, col) position.
pub const ZIGZAG_8X8: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (4, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 4),
    (0, 5),
    (1, 4),
    (2, 3),
    (3, 2),
    (4, 1),
    (5, 0),
    (6, 0),
    (5, 1),
    (4, 2),
    (3, 3),
    (2, 4),
    (1, 5),
    (0, 6),
    (0, 7),
    (1, 6),
    (2, 5),
    (3, 4),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (7, 1),
    (6, 2),
    (5, 3),
    (4, 4),
    (3, 5),
    (2, 6),
    (1, 7),
    (2, 7),
    (3, 6),
    (4, 5),
    (5, 4),
    (6, 3),
    (7, 2),
    (7, 3),
    (6, 4),
    (5, 5),
    (4, 6),
    (3, 7),
    (4, 7),
    (5, 6),
    (6, 5),
    (7, 4),
    (7, 5),
    (6, 6),
    (5, 7),
    (6, 7),
    (7, 6),
    (7, 7),
];

/// Inverse zigzag scan order (position to index).
pub const INVERSE_ZIGZAG_8X8: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (4, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 4),
    (0, 5),
    (1, 4),
    (2, 3),
    (3, 2),
    (4, 1),
    (5, 0),
    (6, 0),
    (5, 1),
    (4, 2),
    (3, 3),
    (2, 4),
    (1, 5),
    (0, 6),
    (0, 7),
    (1, 6),
    (2, 5),
    (3, 4),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (7, 1),
    (6, 2),
    (5, 3),
    (4, 4),
    (3, 5),
    (2, 6),
    (1, 7),
    (2, 7),
    (3, 6),
    (4, 5),
    (5, 4),
    (6, 3),
    (7, 2),
    (7, 3),
    (6, 4),
    (5, 5),
    (4, 6),
    (3, 7),
    (4, 7),
    (5, 6),
    (6, 5),
    (7, 4),
    (7, 5),
    (6, 6),
    (5, 7),
    (6, 7),
    (7, 6),
    (7, 7),
];

/// Get quantization step size for a given QP.
///
/// # Arguments
///
/// * `qp` - Quantization parameter (1-31)
///
/// # Returns
///
/// Quantization step size, or 0 if QP is out of range.
#[must_use]
pub fn get_quant_step(qp: u8) -> i32 {
    if qp > MAX_QP {
        return 0;
    }
    QUANT_STEP[qp as usize]
}

/// Quantize a DCT coefficient.
///
/// # Arguments
///
/// * `coeff` - The DCT coefficient
/// * `qp` - Quantization parameter (1-31)
/// * `is_dc` - True if this is the DC coefficient
///
/// # Returns
///
/// Quantized coefficient.
#[must_use]
pub fn quantize_coeff(coeff: i16, qp: u8, is_dc: bool) -> i16 {
    if coeff == 0 {
        return 0;
    }

    let step = get_quant_step(qp);
    if step == 0 {
        return 0;
    }

    // DC coefficient uses different quantization
    if is_dc {
        // DC coefficient is quantized with step size of 8
        let dc_step = 8;
        let abs_coeff = coeff.abs() as i32;
        let quantized = (abs_coeff + dc_step / 2) / dc_step;
        let result = quantized.clamp(0, 255) as i16;
        if coeff < 0 {
            -result
        } else {
            result
        }
    } else {
        // AC coefficients
        let abs_coeff = coeff.abs() as i32;
        let quantized = abs_coeff / step;
        let result = quantized.clamp(0, 127) as i16;
        if coeff < 0 {
            -result
        } else {
            result
        }
    }
}

/// Dequantize a coefficient.
///
/// # Arguments
///
/// * `level` - Quantized coefficient level
/// * `qp` - Quantization parameter (1-31)
/// * `is_dc` - True if this is the DC coefficient
///
/// # Returns
///
/// Dequantized coefficient.
#[must_use]
pub fn dequantize_coeff(level: i16, qp: u8, is_dc: bool) -> i16 {
    if level == 0 {
        return 0;
    }

    let step = get_quant_step(qp);
    if step == 0 {
        return 0;
    }

    if is_dc {
        // DC coefficient uses step size of 8
        let dc_step = 8;
        let dequant = level as i32 * dc_step;
        dequant.clamp(-2048, 2047) as i16
    } else {
        // AC coefficients: level * (2 * step)
        // H.263 uses odd quantization: dequant = (2 * |level| + 1) * step
        let abs_level = level.abs() as i32;
        let dequant = (2 * abs_level + 1) * step;
        let result = dequant.clamp(0, 2047) as i16;
        if level < 0 {
            -result
        } else {
            result
        }
    }
}

/// Quantize an 8x8 block of DCT coefficients.
///
/// # Arguments
///
/// * `block` - 8x8 array of DCT coefficients
/// * `qp` - Quantization parameter (1-31)
///
/// # Returns
///
/// 8x8 array of quantized coefficients.
#[must_use]
pub fn quantize_block(block: &[[i16; 8]; 8], qp: u8) -> [[i16; 8]; 8] {
    let mut quantized = [[0i16; 8]; 8];

    for row in 0..8 {
        for col in 0..8 {
            let is_dc = row == 0 && col == 0;
            quantized[row][col] = quantize_coeff(block[row][col], qp, is_dc);
        }
    }

    quantized
}

/// Dequantize an 8x8 block of coefficients.
///
/// # Arguments
///
/// * `block` - 8x8 array of quantized coefficients
/// * `qp` - Quantization parameter (1-31)
///
/// # Returns
///
/// 8x8 array of dequantized coefficients.
#[must_use]
pub fn dequantize_block(block: &[[i16; 8]; 8], qp: u8) -> [[i16; 8]; 8] {
    let mut dequantized = [[0i16; 8]; 8];

    for row in 0..8 {
        for col in 0..8 {
            let is_dc = row == 0 && col == 0;
            dequantized[row][col] = dequantize_coeff(block[row][col], qp, is_dc);
        }
    }

    dequantized
}

/// Convert 8x8 block to zigzag order.
///
/// # Arguments
///
/// * `block` - 8x8 array of coefficients
///
/// # Returns
///
/// 64-element array in zigzag order.
#[must_use]
pub fn block_to_zigzag(block: &[[i16; 8]; 8]) -> [i16; 64] {
    let mut zigzag = [0i16; 64];

    for (idx, &(row, col)) in ZIGZAG_8X8.iter().enumerate() {
        zigzag[idx] = block[row][col];
    }

    zigzag
}

/// Convert zigzag order to 8x8 block.
///
/// # Arguments
///
/// * `zigzag` - 64-element array in zigzag order
///
/// # Returns
///
/// 8x8 array of coefficients.
#[must_use]
pub fn zigzag_to_block(zigzag: &[i16; 64]) -> [[i16; 8]; 8] {
    let mut block = [[0i16; 8]; 8];

    for (idx, &(row, col)) in ZIGZAG_8X8.iter().enumerate() {
        block[row][col] = zigzag[idx];
    }

    block
}

/// Calculate QP from target bitrate and complexity.
///
/// This is a simplified rate control function.
///
/// # Arguments
///
/// * `target_bits` - Target bits for the frame
/// * `actual_bits` - Actual bits used so far
/// * `current_qp` - Current quantization parameter
///
/// # Returns
///
/// Adjusted QP value.
#[must_use]
pub fn adjust_qp(target_bits: usize, actual_bits: usize, current_qp: u8) -> u8 {
    if actual_bits > target_bits {
        // Using too many bits, increase QP
        (current_qp + 1).min(MAX_QP)
    } else if actual_bits < target_bits / 2 {
        // Using too few bits, decrease QP
        (current_qp.saturating_sub(1)).max(MIN_QP)
    } else {
        current_qp
    }
}

/// Calculate adaptive QP based on block variance.
///
/// # Arguments
///
/// * `block` - 8x8 block of pixels
/// * `base_qp` - Base quantization parameter
///
/// # Returns
///
/// Adjusted QP for this block.
#[must_use]
pub fn adaptive_qp(block: &[[u8; 8]; 8], base_qp: u8) -> u8 {
    // Calculate variance
    let mut sum = 0i32;
    let mut sum_sq = 0i32;

    for row in 0..8 {
        for col in 0..8 {
            let val = block[row][col] as i32;
            sum += val;
            sum_sq += val * val;
        }
    }

    let mean = sum / 64;
    let variance = (sum_sq / 64) - (mean * mean);

    // Adjust QP based on variance
    // Higher variance (complex blocks) -> lower QP (better quality)
    // Lower variance (smooth blocks) -> higher QP (more compression)
    if variance > 1000 {
        base_qp.saturating_sub(2).max(MIN_QP)
    } else if variance < 100 {
        (base_qp + 2).min(MAX_QP)
    } else {
        base_qp
    }
}

/// Deadzone quantization for better rate-distortion.
///
/// # Arguments
///
/// * `coeff` - DCT coefficient
/// * `qp` - Quantization parameter
/// * `deadzone_factor` - Deadzone size (0.0-1.0)
///
/// # Returns
///
/// Quantized coefficient with deadzone.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn quantize_with_deadzone(coeff: i16, qp: u8, deadzone_factor: f32) -> i16 {
    if coeff == 0 {
        return 0;
    }

    let step = get_quant_step(qp);
    if step == 0 {
        return 0;
    }

    let abs_coeff = coeff.abs() as i32;
    let deadzone = ((step as f32) * deadzone_factor) as i32;

    if abs_coeff < deadzone {
        return 0;
    }

    let quantized = (abs_coeff - deadzone + step / 2) / step;
    let result = quantized.clamp(0, 127) as i16;

    if coeff < 0 {
        -result
    } else {
        result
    }
}

/// Perceptual quantization matrix for luminance.
///
/// Uses human visual system characteristics to allocate bits.
const PERCEPTUAL_QUANT_LUMA: [[f32; 8]; 8] = [
    [1.00, 1.00, 1.00, 1.00, 1.10, 1.20, 1.30, 1.40],
    [1.00, 1.00, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50],
    [1.00, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60],
    [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70],
    [1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80],
    [1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90],
    [1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00],
    [1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10],
];

/// Perceptual quantization for luminance blocks.
///
/// # Arguments
///
/// * `block` - 8x8 DCT coefficient block
/// * `qp` - Base quantization parameter
///
/// # Returns
///
/// Quantized block with perceptual weighting.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn quantize_perceptual_luma(block: &[[i16; 8]; 8], qp: u8) -> [[i16; 8]; 8] {
    let mut quantized = [[0i16; 8]; 8];
    let base_step = get_quant_step(qp) as f32;

    for row in 0..8 {
        for col in 0..8 {
            let coeff = block[row][col];
            if coeff == 0 {
                continue;
            }

            let weighted_step = base_step * PERCEPTUAL_QUANT_LUMA[row][col];
            let abs_coeff = coeff.abs() as f32;
            let quantized_val = (abs_coeff / weighted_step) as i16;

            quantized[row][col] = if coeff < 0 {
                -quantized_val
            } else {
                quantized_val
            };
        }
    }

    quantized
}

/// Perceptual quantization matrix for chrominance.
const PERCEPTUAL_QUANT_CHROMA: [[f32; 8]; 8] = [
    [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70],
    [1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80],
    [1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90],
    [1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00],
    [1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10],
    [1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20],
    [1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30],
    [1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40],
];

/// Perceptual quantization for chrominance blocks.
///
/// # Arguments
///
/// * `block` - 8x8 DCT coefficient block
/// * `qp` - Base quantization parameter
///
/// # Returns
///
/// Quantized block with perceptual weighting.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn quantize_perceptual_chroma(block: &[[i16; 8]; 8], qp: u8) -> [[i16; 8]; 8] {
    let mut quantized = [[0i16; 8]; 8];
    let base_step = get_quant_step(qp) as f32;

    for row in 0..8 {
        for col in 0..8 {
            let coeff = block[row][col];
            if coeff == 0 {
                continue;
            }

            let weighted_step = base_step * PERCEPTUAL_QUANT_CHROMA[row][col];
            let abs_coeff = coeff.abs() as f32;
            let quantized_val = (abs_coeff / weighted_step) as i16;

            quantized[row][col] = if coeff < 0 {
                -quantized_val
            } else {
                quantized_val
            };
        }
    }

    quantized
}
