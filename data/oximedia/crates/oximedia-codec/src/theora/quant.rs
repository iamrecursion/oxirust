// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Advanced quantization strategies for Theora encoding.
//!
//! Implements perceptual quantization, adaptive dead-zone, and
//! psychovisual optimization for better compression efficiency.

use crate::theora::tables::{BASE_MATRIX_INTER, BASE_MATRIX_INTRA_C, BASE_MATRIX_INTRA_Y};
use crate::theora::transform::Block8x8;

/// Quantization matrix set for a frame.
#[derive(Debug, Clone)]
pub struct QuantMatrixSet {
    /// Intra Y quantization matrix.
    pub intra_y: [u16; 64],
    /// Intra C quantization matrix.
    pub intra_c: [u16; 64],
    /// Inter quantization matrix.
    pub inter: [u16; 64],
}

impl QuantMatrixSet {
    /// Create a new quantization matrix set from quality parameter.
    ///
    /// # Arguments
    ///
    /// * `quality` - Quality value (0-63, higher = better quality)
    #[must_use]
    pub fn from_quality(quality: u8) -> Self {
        let mut intra_y = [0u16; 64];
        let mut intra_c = [0u16; 64];
        let mut inter = [0u16; 64];

        build_quant_matrix(&BASE_MATRIX_INTRA_Y, quality, &mut intra_y);
        build_quant_matrix(&BASE_MATRIX_INTRA_C, quality, &mut intra_c);
        build_quant_matrix(&BASE_MATRIX_INTER, quality, &mut inter);

        Self {
            intra_y,
            intra_c,
            inter,
        }
    }

    /// Create a custom quantization matrix set.
    #[must_use]
    pub const fn custom(intra_y: [u16; 64], intra_c: [u16; 64], inter: [u16; 64]) -> Self {
        Self {
            intra_y,
            intra_c,
            inter,
        }
    }

    /// Apply perceptual weighting to the matrices.
    pub fn apply_perceptual_weighting(&mut self, strength: f32) {
        apply_perceptual_weights(&mut self.intra_y, strength);
        apply_perceptual_weights(&mut self.intra_c, strength);
        apply_perceptual_weights(&mut self.inter, strength);
    }

    /// Get quantization matrix for a block type.
    #[must_use]
    pub const fn get_matrix(&self, is_intra: bool, is_luma: bool) -> &[u16; 64] {
        if is_intra {
            if is_luma {
                &self.intra_y
            } else {
                &self.intra_c
            }
        } else {
            &self.inter
        }
    }
}

/// Build quantization matrix from base matrix and quality.
fn build_quant_matrix(base: &[u16; 64], quality: u8, output: &mut [u16; 64]) {
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

/// Apply perceptual weighting to quantization matrix.
///
/// Weights coefficients based on human visual system sensitivity.
fn apply_perceptual_weights(matrix: &mut [u16; 64], strength: f32) {
    // Perceptual weight table (higher frequency = higher weight = more quantization)
    const PERCEPTUAL_WEIGHTS: [f32; 64] = [
        1.0, 1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.0, 1.1,
        1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.2, 1.3, 1.4, 1.5,
        1.6, 1.7, 1.8, 1.9, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
        2.0, 2.1, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2,
    ];

    for i in 0..64 {
        let weight = 1.0 + (PERCEPTUAL_WEIGHTS[i] - 1.0) * strength;
        let weighted = (f32::from(matrix[i]) * weight) as u32;
        matrix[i] = weighted.clamp(1, 255) as u16;
    }
}

/// Advanced quantizer with dead-zone and rounding control.
pub struct AdvancedQuantizer {
    /// Dead-zone size (0.0 to 1.0).
    dead_zone: f32,
    /// Rounding bias (-0.5 to 0.5).
    bias: f32,
    /// Adaptive dead-zone based on AC energy.
    adaptive: bool,
}

impl AdvancedQuantizer {
    /// Create a new advanced quantizer.
    #[must_use]
    pub const fn new(dead_zone: f32, bias: f32, adaptive: bool) -> Self {
        Self {
            dead_zone,
            bias,
            adaptive,
        }
    }

    /// Quantize a DCT block with advanced settings.
    ///
    /// # Arguments
    ///
    /// * `input` - Input DCT coefficients
    /// * `output` - Output quantized coefficients
    /// * `quant_matrix` - Quantization matrix
    pub fn quantize(&self, input: &Block8x8, output: &mut Block8x8, quant_matrix: &[u16; 64]) {
        // Calculate AC energy for adaptive dead-zone
        let ac_energy = if self.adaptive {
            calculate_ac_energy(input)
        } else {
            0.0
        };

        for i in 0..64 {
            let coeff = i32::from(input[i]);
            let quant = i32::from(quant_matrix[i]);

            if quant == 0 {
                output[i] = 0;
                continue;
            }

            // Apply dead-zone
            let dead_zone_threshold = if i == 0 {
                // DC coefficient: no dead-zone
                0
            } else {
                // AC coefficients: apply dead-zone
                let base_dz = (quant as f32 * self.dead_zone) as i32;
                if self.adaptive {
                    // Reduce dead-zone for high-energy blocks
                    let energy_factor = (1.0 - (ac_energy / 10000.0).min(1.0)) as i32;
                    base_dz * energy_factor / 100
                } else {
                    base_dz
                }
            };

            let abs_coeff = coeff.abs();
            if abs_coeff < dead_zone_threshold {
                output[i] = 0;
                continue;
            }

            // Quantize with bias
            let bias_amount = (quant as f32 * self.bias) as i32;
            let quantized = if coeff >= 0 {
                (coeff + quant / 2 + bias_amount) / quant
            } else {
                (coeff - quant / 2 + bias_amount) / quant
            };

            output[i] = quantized as i16;
        }
    }

    /// Dequantize a block.
    pub fn dequantize(&self, input: &Block8x8, output: &mut Block8x8, quant_matrix: &[u16; 64]) {
        for i in 0..64 {
            let coeff = i32::from(input[i]);
            let quant = i32::from(quant_matrix[i]);
            output[i] = (coeff * quant) as i16;
        }
    }
}

impl Default for AdvancedQuantizer {
    fn default() -> Self {
        Self::new(0.3, 0.0, true)
    }
}

/// Calculate AC energy of a DCT block (excluding DC).
fn calculate_ac_energy(block: &Block8x8) -> f32 {
    let mut energy = 0u32;
    for i in 1..64 {
        let coeff = i32::from(block[i]);
        energy += (coeff * coeff) as u32;
    }
    energy as f32
}

/// Trellis quantization for optimal coefficient selection.
///
/// Uses dynamic programming to find the best quantization decisions
/// considering rate-distortion tradeoffs.
pub struct TrellisQuantizer {
    /// Lambda (rate-distortion tradeoff parameter).
    lambda: f32,
}

impl TrellisQuantizer {
    /// Create a new trellis quantizer.
    #[must_use]
    pub const fn new(lambda: f32) -> Self {
        Self { lambda }
    }

    /// Quantize using trellis optimization.
    ///
    /// # Arguments
    ///
    /// * `input` - Input DCT coefficients
    /// * `output` - Output quantized coefficients
    /// * `quant_matrix` - Quantization matrix
    pub fn quantize(&self, input: &Block8x8, output: &mut Block8x8, quant_matrix: &[u16; 64]) {
        // For each coefficient, try quantization levels and pick best
        for i in 0..64 {
            let coeff = i32::from(input[i]);
            let quant = i32::from(quant_matrix[i]);

            if quant == 0 {
                output[i] = 0;
                continue;
            }

            // Try multiple quantization levels
            let base_level = coeff / quant;
            let mut best_level = base_level;
            let mut best_cost = f32::MAX;

            for delta in -1..=1 {
                let level = base_level + delta;
                let dequant = level * quant;
                let distortion = (coeff - dequant) * (coeff - dequant);
                let rate = self.estimate_level_rate(level);
                let cost = distortion as f32 + self.lambda * rate;

                if cost < best_cost {
                    best_cost = cost;
                    best_level = level;
                }
            }

            output[i] = best_level as i16;
        }
    }

    /// Estimate bitrate cost of a quantization level.
    fn estimate_level_rate(&self, level: i32) -> f32 {
        if level == 0 {
            1.0 // End-of-block or zero run
        } else {
            let abs_level = level.abs();
            let bits = 32 - abs_level.leading_zeros();
            (bits * 2 + 1) as f32 // Sign + magnitude
        }
    }
}

/// Perceptual quantization optimizer.
///
/// Adjusts quantization based on local image characteristics and
/// human visual system properties.
pub struct PerceptualQuantizer {
    /// Base quality.
    quality: u8,
    /// Contrast sensitivity function strength.
    csf_strength: f32,
}

impl PerceptualQuantizer {
    /// Create a new perceptual quantizer.
    #[must_use]
    pub const fn new(quality: u8, csf_strength: f32) -> Self {
        Self {
            quality,
            csf_strength,
        }
    }

    /// Generate perceptually optimized quantization matrix.
    ///
    /// # Arguments
    ///
    /// * `spatial_activity` - Measure of local spatial activity
    /// * `is_intra` - Whether this is an intra block
    /// * `is_luma` - Whether this is a luma block
    #[must_use]
    pub fn generate_matrix(
        &self,
        spatial_activity: f32,
        is_intra: bool,
        is_luma: bool,
    ) -> [u16; 64] {
        let base = if is_intra {
            if is_luma {
                BASE_MATRIX_INTRA_Y
            } else {
                BASE_MATRIX_INTRA_C
            }
        } else {
            BASE_MATRIX_INTER
        };

        let mut matrix = [0u16; 64];
        build_quant_matrix(&base, self.quality, &mut matrix);

        // Apply contrast sensitivity function
        apply_csf_weighting(&mut matrix, self.csf_strength);

        // Adjust based on spatial activity
        let activity_factor = if spatial_activity < 100.0 {
            0.8 // Smooth regions: finer quantization
        } else if spatial_activity > 1000.0 {
            1.2 // Textured regions: coarser quantization
        } else {
            1.0
        };

        for val in &mut matrix {
            *val = (f32::from(*val) * activity_factor).round() as u16;
            *val = (*val).max(1).min(255);
        }

        matrix
    }
}

/// Apply contrast sensitivity function weighting.
fn apply_csf_weighting(matrix: &mut [u16; 64], strength: f32) {
    // CSF weights based on spatial frequency sensitivity
    const CSF_WEIGHTS: [f32; 64] = [
        1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86,
        0.84, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84,
        0.82, 0.80, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.90, 0.88, 0.86, 0.84, 0.82,
        0.80, 0.78, 0.76, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74, 0.86, 0.84, 0.82, 0.80,
        0.78, 0.76, 0.74, 0.72,
    ];

    for i in 0..64 {
        let weight = 1.0 / (1.0 + (1.0 - CSF_WEIGHTS[i]) * strength);
        matrix[i] = (f32::from(matrix[i]) * weight).round() as u16;
        matrix[i] = matrix[i].max(1);
    }
}

/// Quantization parameter selection for macroblock.
///
/// Determines optimal QP based on macroblock characteristics.
pub struct MacroblockQP {
    /// Base QP.
    base_qp: u8,
}

impl MacroblockQP {
    /// Create a new macroblock QP selector.
    #[must_use]
    pub const fn new(base_qp: u8) -> Self {
        Self { base_qp }
    }

    /// Get QP for a macroblock.
    ///
    /// # Arguments
    ///
    /// * `variance` - Macroblock variance
    /// * `edge_strength` - Edge strength measure
    /// * `is_intra` - Whether this is an intra macroblock
    #[must_use]
    pub fn get_mb_qp(&self, variance: f32, edge_strength: f32, is_intra: bool) -> u8 {
        let mut qp = self.base_qp;

        // Adjust based on variance
        if variance < 100.0 {
            qp = qp.saturating_sub(3); // Smooth: better quality
        } else if variance > 1000.0 {
            qp = qp.saturating_add(3); // Textured: can use more compression
        }

        // Adjust based on edge strength
        if edge_strength > 50.0 {
            qp = qp.saturating_sub(2); // Strong edges: preserve detail
        }

        // Intra blocks get slightly better quality
        if is_intra {
            qp = qp.saturating_sub(1);
        }

        qp.min(63)
    }

    /// Calculate edge strength for a block.
    #[must_use]
    pub fn calculate_edge_strength(block: &[u8; 64]) -> f32 {
        let mut strength = 0f32;

        // Horizontal edges
        for y in 0..7 {
            for x in 0..8 {
                let diff = (i16::from(block[(y + 1) * 8 + x]) - i16::from(block[y * 8 + x])).abs();
                strength += diff as f32;
            }
        }

        // Vertical edges
        for y in 0..8 {
            for x in 0..7 {
                let diff = (i16::from(block[y * 8 + x + 1]) - i16::from(block[y * 8 + x])).abs();
                strength += diff as f32;
            }
        }

        strength / 112.0 // Normalize by number of edges
    }
}

/// Coefficient optimization for better compression.
pub struct CoefficientOptimizer {
    /// Threshold for small coefficient elimination.
    threshold: i16,
}

impl CoefficientOptimizer {
    /// Create a new coefficient optimizer.
    #[must_use]
    pub const fn new(threshold: i16) -> Self {
        Self { threshold }
    }

    /// Optimize quantized coefficients.
    ///
    /// Eliminates small coefficients that don't contribute significantly
    /// to quality but increase bitrate.
    pub fn optimize(&self, coeffs: &mut Block8x8) {
        for i in 1..64 {
            // Don't touch DC coefficient
            if coeffs[i].abs() < self.threshold {
                coeffs[i] = 0;
            }
        }
    }

    /// Run-length optimize coefficients.
    ///
    /// Reorganizes zeros for better run-length encoding.
    pub fn rle_optimize(&self, coeffs: &mut Block8x8) {
        // Find last non-zero coefficient
        let mut last_nonzero = 0;
        for i in (0..64).rev() {
            if coeffs[i] != 0 {
                last_nonzero = i;
                break;
            }
        }

        // Zero out trailing coefficients
        for coeff in coeffs.iter_mut().skip(last_nonzero + 1) {
            *coeff = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_matrix_creation() {
        let matrices = QuantMatrixSet::from_quality(30);
        assert!(matrices.intra_y[0] > 0);
        assert!(matrices.intra_c[0] > 0);
        assert!(matrices.inter[0] > 0);
    }

    #[test]
    fn test_perceptual_weighting() {
        let mut matrices = QuantMatrixSet::from_quality(30);
        let before = matrices.intra_y[32];
        matrices.apply_perceptual_weighting(0.5);
        assert_ne!(before, matrices.intra_y[32]);
    }

    #[test]
    fn test_advanced_quantizer() {
        let quantizer = AdvancedQuantizer::new(0.3, 0.0, true);
        let input = [
            100i16, 50, 25, 12, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let quant_matrix = [16u16; 64];
        let mut output = [0i16; 64];

        quantizer.quantize(&input, &mut output, &quant_matrix);
        assert!(output[0] != 0); // DC should be quantized
    }

    #[test]
    fn test_ac_energy_calculation() {
        let block = [100i16; 64];
        let energy = calculate_ac_energy(&block);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_trellis_quantizer() {
        let quantizer = TrellisQuantizer::new(1.0);
        let input = [100i16; 64];
        let quant_matrix = [16u16; 64];
        let mut output = [0i16; 64];

        quantizer.quantize(&input, &mut output, &quant_matrix);
        assert!(output[0] != 0);
    }

    #[test]
    fn test_perceptual_quantizer() {
        let pq = PerceptualQuantizer::new(30, 0.5);
        let matrix = pq.generate_matrix(100.0, true, true);
        assert!(matrix[0] > 0);
        assert!(matrix[63] > 0);
    }

    #[test]
    fn test_macroblock_qp() {
        let mb_qp = MacroblockQP::new(30);
        let qp = mb_qp.get_mb_qp(50.0, 30.0, true);
        assert!(qp <= 63);
    }

    #[test]
    fn test_edge_strength() {
        let block = [128u8; 64];
        let strength = MacroblockQP::calculate_edge_strength(&block);
        assert_eq!(strength, 0.0); // Uniform block has no edges
    }

    #[test]
    fn test_coefficient_optimizer() {
        let optimizer = CoefficientOptimizer::new(2);
        let mut coeffs = [1i16; 64];
        coeffs[0] = 100; // DC
        optimizer.optimize(&mut coeffs);

        // Small coefficients should be zeroed
        assert_eq!(coeffs[1], 0);
        assert_eq!(coeffs[0], 100); // DC preserved
    }
}
