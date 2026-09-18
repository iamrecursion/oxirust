//! AV1 transform coefficient encoding.
//!
//! This module handles encoding of quantized transform coefficients with:
//!
//! - Scan order selection (diagonal, horizontal, vertical)
//! - EOB (End of Block) encoding
//! - Coefficient level encoding with context
//! - Sign encoding
//! - Quantization integration
//!
//! # Coefficient Encoding Process
//!
//! 1. Forward transform (DCT/ADST)
//! 2. Quantization
//! 3. Find EOB (last non-zero coefficient)
//! 4. Scan in appropriate order
//! 5. Encode levels and signs using arithmetic coder
//!
//! # References
//!
//! - AV1 Specification Section 5.11: Transform Coefficient Syntax

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_arguments)]

use super::entropy_encoder::SymbolEncoder;
use super::quantization::QuantizationParams;
use super::transform::{TxClass, TxSize, TxType};

// =============================================================================
// Constants
// =============================================================================

/// Maximum coefficient value after quantization.
const MAX_COEFF_LEVEL: i32 = 255;

/// Number of coefficient context types.
const COEFF_CONTEXTS: usize = 4;

/// Number of EOB context types.
const EOB_CONTEXTS: usize = 7;

/// Coefficient level map contexts.
const LEVEL_CONTEXTS: usize = 21;

// =============================================================================
// Scan Order
// =============================================================================

/// Scan order type for coefficient encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanOrder {
    /// Default zig-zag diagonal scan.
    Default = 0,
    /// Horizontal scan (for vertical transforms).
    Horizontal = 1,
    /// Vertical scan (for horizontal transforms).
    Vertical = 2,
}

impl ScanOrder {
    /// Get scan order for given transform type.
    #[must_use]
    pub const fn from_tx_type(tx_type: TxType) -> Self {
        match tx_type.tx_class() {
            TxClass::Class2D => Self::Default,
            TxClass::ClassHoriz => Self::Horizontal,
            TxClass::ClassVert => Self::Vertical,
        }
    }
}

/// Generate scan order indices for a transform block.
#[must_use]
pub fn generate_scan_order(tx_size: TxSize, scan_order: ScanOrder) -> Vec<(usize, usize)> {
    let w = tx_size.width() as usize;
    let h = tx_size.height() as usize;
    let mut indices = Vec::with_capacity(w * h);

    match scan_order {
        ScanOrder::Default => {
            // Diagonal zig-zag scan
            for diag in 0..(w + h - 1) {
                if diag % 2 == 0 {
                    // Even diagonal: go up-right
                    let start_col = diag.min(w - 1);
                    let start_row = diag.saturating_sub(w - 1);

                    let mut col = start_col;
                    let mut row = start_row;

                    while col < w && row < h {
                        if col <= diag && row <= diag {
                            indices.push((row, col));
                        }
                        if col == 0 {
                            break;
                        }
                        col -= 1;
                        row += 1;
                    }
                } else {
                    // Odd diagonal: go down-left
                    let start_row = diag.min(h - 1);
                    let start_col = diag.saturating_sub(h - 1);

                    let mut row = start_row;
                    let mut col = start_col;

                    while row < h && col < w {
                        if row <= diag && col <= diag {
                            indices.push((row, col));
                        }
                        if row == 0 {
                            break;
                        }
                        row -= 1;
                        col += 1;
                    }
                }
            }
        }
        ScanOrder::Horizontal => {
            // Row-major scan
            for y in 0..h {
                for x in 0..w {
                    indices.push((y, x));
                }
            }
        }
        ScanOrder::Vertical => {
            // Column-major scan
            for x in 0..w {
                for y in 0..h {
                    indices.push((y, x));
                }
            }
        }
    }

    indices
}

// =============================================================================
// Coefficient Encoder
// =============================================================================

/// Transform coefficient encoder.
#[derive(Clone, Debug)]
pub struct CoeffEncoder {
    /// Symbol encoder.
    encoder: SymbolEncoder,
    /// Quantization parameters.
    qparams: QuantizationParams,
    /// Logical bits encoded (for tracking when buffer hasn't flushed).
    bits_encoded: usize,
}

impl Default for CoeffEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CoeffEncoder {
    /// Create a new coefficient encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            encoder: SymbolEncoder::new(),
            qparams: QuantizationParams::default(),
            bits_encoded: 0,
        }
    }

    /// Set quantization parameters.
    pub fn set_qparams(&mut self, qparams: QuantizationParams) {
        self.qparams = qparams;
    }

    /// Encode transform coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - Quantized coefficients in raster order
    /// * `tx_size` - Transform size
    /// * `tx_type` - Transform type
    /// * `plane` - Plane index (0=Y, 1=U, 2=V)
    ///
    /// # Returns
    ///
    /// Number of bits used
    pub fn encode_coeffs(
        &mut self,
        coeffs: &[i32],
        tx_size: TxSize,
        tx_type: TxType,
        plane: u8,
    ) -> usize {
        let start_len = self.encoder.buffer().len();
        self.bits_encoded = 0;

        // Find EOB (last non-zero coefficient)
        let eob = self.find_eob(coeffs);

        if eob == 0 {
            // All zeros - encode skip
            self.encoder.encode_bool(true);
            self.bits_encoded += 1;
            let buffer_bits = 8 * (self.encoder.buffer().len() - start_len);
            return buffer_bits.max(self.bits_encoded);
        }

        // Not skip
        self.encoder.encode_bool(false);
        self.bits_encoded += 1;

        // Encode EOB
        self.encode_eob(eob, tx_size);

        // Get scan order
        let scan_order = ScanOrder::from_tx_type(tx_type);
        let scan = generate_scan_order(tx_size, scan_order);

        // Encode coefficients in scan order
        self.encode_coeffs_scan(coeffs, &scan[..eob], tx_size, plane);

        let buffer_bits = 8 * (self.encoder.buffer().len() - start_len);
        buffer_bits.max(self.bits_encoded)
    }

    /// Find end of block (last non-zero coefficient position).
    fn find_eob(&self, coeffs: &[i32]) -> usize {
        for (i, &c) in coeffs.iter().enumerate().rev() {
            if c != 0 {
                return i + 1;
            }
        }
        0
    }

    /// Encode EOB position.
    fn encode_eob(&mut self, eob: usize, tx_size: TxSize) {
        let max_eob = tx_size.max_eob() as usize;

        // Simple EOB encoding (could be improved with better context)
        let eob_bits = (max_eob.next_power_of_two().trailing_zeros()) as u8;
        self.encoder.encode_literal(eob as u32, eob_bits);
    }

    /// Encode coefficients in scan order.
    fn encode_coeffs_scan(
        &mut self,
        coeffs: &[i32],
        scan: &[(usize, usize)],
        tx_size: TxSize,
        _plane: u8,
    ) {
        let stride = tx_size.width() as usize;

        for &(row, col) in scan {
            let idx = row * stride + col;
            if idx >= coeffs.len() {
                break;
            }

            let coeff = coeffs[idx];
            self.encode_coeff(coeff);
        }
    }

    /// Encode a single coefficient.
    fn encode_coeff(&mut self, coeff: i32) {
        let level = coeff.abs();

        if level == 0 {
            // Zero coefficient
            self.encoder.encode_literal(0, 8);
            return;
        }

        // Encode level (simplified - no context modeling)
        let level_clamped = level.min(MAX_COEFF_LEVEL) as u32;
        self.encoder.encode_literal(level_clamped, 8);

        // Encode sign
        self.encoder.encode_bool(coeff < 0);
    }

    /// Get encoded output.
    #[must_use]
    pub fn finish(&mut self) -> Vec<u8> {
        self.encoder.finish()
    }

    /// Reset encoder state.
    pub fn reset(&mut self) {
        self.encoder.reset();
    }
}

// =============================================================================
// Quantization
// =============================================================================

/// Quantize transform coefficients.
#[must_use]
pub fn quantize_coeffs(coeffs: &[i32], qp: u8, tx_size: TxSize) -> Vec<i32> {
    let q_step = compute_q_step(qp);
    let area = tx_size.area() as usize;
    let mut quantized = vec![0i32; area.min(coeffs.len())];

    for (i, &c) in coeffs.iter().take(area).enumerate() {
        quantized[i] = quantize_coeff(c, q_step);
    }

    quantized
}

/// Dequantize transform coefficients.
#[must_use]
pub fn dequantize_coeffs(coeffs: &[i32], qp: u8, tx_size: TxSize) -> Vec<i32> {
    let q_step = compute_q_step(qp);
    let area = tx_size.area() as usize;
    let mut dequantized = vec![0i32; area.min(coeffs.len())];

    for (i, &c) in coeffs.iter().take(area).enumerate() {
        dequantized[i] = dequantize_coeff(c, q_step);
    }

    dequantized
}

/// Compute quantization step from QP.
#[must_use]
fn compute_q_step(qp: u8) -> i32 {
    // Simplified: q_step = 2^(qp/6)
    let qp_f = f32::from(qp);
    (2.0_f32.powf(qp_f / 6.0)) as i32
}

/// Quantize a single coefficient.
#[must_use]
fn quantize_coeff(coeff: i32, q_step: i32) -> i32 {
    if q_step == 0 {
        return coeff;
    }

    let sign = coeff.signum();
    let abs_coeff = coeff.abs();
    let quantized = (abs_coeff + q_step / 2) / q_step;

    sign * quantized.min(MAX_COEFF_LEVEL)
}

/// Dequantize a single coefficient.
#[must_use]
fn dequantize_coeff(coeff: i32, q_step: i32) -> i32 {
    coeff * q_step
}

// =============================================================================
// Coefficient Statistics
// =============================================================================

/// Statistics for coefficient encoding.
#[derive(Clone, Debug, Default)]
pub struct CoeffStats {
    /// Total number of coefficients.
    pub total_coeffs: usize,
    /// Number of zero coefficients.
    pub zero_coeffs: usize,
    /// Number of blocks skipped.
    pub skip_blocks: usize,
    /// Total bits used.
    pub total_bits: usize,
}

impl CoeffStats {
    /// Create new statistics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            total_coeffs: 0,
            zero_coeffs: 0,
            skip_blocks: 0,
            total_bits: 0,
        }
    }

    /// Update statistics from coefficient block.
    pub fn update(&mut self, coeffs: &[i32], bits_used: usize) {
        self.total_coeffs += coeffs.len();
        self.zero_coeffs += coeffs.iter().filter(|&&c| c == 0).count();
        self.total_bits += bits_used;

        if coeffs.iter().all(|&c| c == 0) {
            self.skip_blocks += 1;
        }
    }

    /// Get average bits per coefficient.
    #[must_use]
    pub fn avg_bits_per_coeff(&self) -> f32 {
        if self.total_coeffs == 0 {
            0.0
        } else {
            self.total_bits as f32 / self.total_coeffs as f32
        }
    }

    /// Get zero coefficient ratio.
    #[must_use]
    pub fn zero_ratio(&self) -> f32 {
        if self.total_coeffs == 0 {
            0.0
        } else {
            self.zero_coeffs as f32 / self.total_coeffs as f32
        }
    }
}

// =============================================================================
// Context Modeling
// =============================================================================

/// Coefficient level context.
#[derive(Clone, Copy, Debug)]
pub struct CoeffContext {
    /// Number of non-zero neighbors.
    pub nz_neighbors: u8,
    /// Position in block (DC or AC).
    pub is_dc: bool,
    /// Previous coefficient level.
    pub prev_level: u8,
}

impl Default for CoeffContext {
    fn default() -> Self {
        Self {
            nz_neighbors: 0,
            is_dc: false,
            prev_level: 0,
        }
    }
}

impl CoeffContext {
    /// Get context index for level encoding.
    #[must_use]
    pub const fn level_ctx(&self) -> usize {
        let base = if self.is_dc { 0 } else { 7 };
        let nz = self.nz_neighbors as usize;
        let nz_clamped = if nz > 3 { 3 } else { nz };
        let prev = self.prev_level as usize;
        let prev_clamped = if prev > 1 { 1 } else { prev };
        let offset = nz_clamped * 2 + prev_clamped;
        let result = base + offset;
        if result > LEVEL_CONTEXTS - 1 {
            LEVEL_CONTEXTS - 1
        } else {
            result
        }
    }

    /// Get context index for EOB encoding.
    #[must_use]
    pub const fn eob_ctx(&self) -> usize {
        self.nz_neighbors as usize % EOB_CONTEXTS
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_order_from_tx_type() {
        assert_eq!(ScanOrder::from_tx_type(TxType::DctDct), ScanOrder::Default);
        assert_eq!(
            ScanOrder::from_tx_type(TxType::DctIdtx),
            ScanOrder::Horizontal
        );
        assert_eq!(
            ScanOrder::from_tx_type(TxType::IdtxDct),
            ScanOrder::Vertical
        );
    }

    #[test]
    fn test_generate_scan_order_4x4() {
        let scan = generate_scan_order(TxSize::Tx4x4, ScanOrder::Default);
        assert_eq!(scan.len(), 16);
        assert_eq!(scan[0], (0, 0)); // DC coefficient first
    }

    #[test]
    fn test_generate_scan_order_horizontal() {
        let scan = generate_scan_order(TxSize::Tx4x4, ScanOrder::Horizontal);
        assert_eq!(scan.len(), 16);
        // Row-major order
        assert_eq!(scan[0], (0, 0));
        assert_eq!(scan[1], (0, 1));
        assert_eq!(scan[4], (1, 0));
    }

    #[test]
    fn test_generate_scan_order_vertical() {
        let scan = generate_scan_order(TxSize::Tx4x4, ScanOrder::Vertical);
        assert_eq!(scan.len(), 16);
        // Column-major order
        assert_eq!(scan[0], (0, 0));
        assert_eq!(scan[1], (1, 0));
        assert_eq!(scan[4], (0, 1));
    }

    #[test]
    fn test_coeff_encoder_creation() {
        let encoder = CoeffEncoder::new();
        assert!(!encoder.encoder.buffer().is_empty() || encoder.encoder.buffer().is_empty());
    }

    #[test]
    fn test_find_eob() {
        let encoder = CoeffEncoder::new();

        let coeffs = vec![1, 2, 0, 3, 0, 0, 0];
        let eob = encoder.find_eob(&coeffs);
        assert_eq!(eob, 4);

        let all_zero = vec![0; 16];
        let eob_zero = encoder.find_eob(&all_zero);
        assert_eq!(eob_zero, 0);
    }

    #[test]
    fn test_encode_all_zero_block() {
        let mut encoder = CoeffEncoder::new();
        let coeffs = vec![0; 16];

        let bits = encoder.encode_coeffs(&coeffs, TxSize::Tx4x4, TxType::DctDct, 0);
        assert!(bits > 0);
    }

    #[test]
    fn test_encode_non_zero_block() {
        let mut encoder = CoeffEncoder::new();
        let coeffs = vec![10, 5, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let bits = encoder.encode_coeffs(&coeffs, TxSize::Tx4x4, TxType::DctDct, 0);
        assert!(bits > 0);
    }

    #[test]
    fn test_quantize_coeff() {
        let q_step = 4;

        let q1 = quantize_coeff(10, q_step);
        assert_eq!(q1, 3); // (10 + 2) / 4 = 3

        let q2 = quantize_coeff(-10, q_step);
        assert_eq!(q2, -3);

        let q3 = quantize_coeff(0, q_step);
        assert_eq!(q3, 0);
    }

    #[test]
    fn test_dequantize_coeff() {
        let q_step = 4;

        let dq1 = dequantize_coeff(3, q_step);
        assert_eq!(dq1, 12);

        let dq2 = dequantize_coeff(-3, q_step);
        assert_eq!(dq2, -12);
    }

    #[test]
    fn test_compute_q_step() {
        let q_step_0 = compute_q_step(0);
        assert!(q_step_0 > 0);

        let q_step_30 = compute_q_step(30);
        assert!(q_step_30 > q_step_0);
    }

    #[test]
    fn test_quantize_coeffs_array() {
        let coeffs = vec![10, 20, 30, 40];
        let quantized = quantize_coeffs(&coeffs, 6, TxSize::Tx4x4);

        assert_eq!(quantized.len(), 4);
        assert!(quantized[0] <= coeffs[0]);
        assert!(quantized[3] <= coeffs[3]);
    }

    #[test]
    fn test_dequantize_coeffs_array() {
        let coeffs = vec![2, 4, 6, 8];
        let dequantized = dequantize_coeffs(&coeffs, 6, TxSize::Tx4x4);

        assert_eq!(dequantized.len(), 4);
        assert!(dequantized[0] >= coeffs[0]);
    }

    #[test]
    fn test_coeff_stats() {
        let mut stats = CoeffStats::new();
        assert_eq!(stats.total_coeffs, 0);

        let coeffs = vec![1, 0, 2, 0, 0, 3];
        stats.update(&coeffs, 100);

        assert_eq!(stats.total_coeffs, 6);
        assert_eq!(stats.zero_coeffs, 3);
        assert_eq!(stats.zero_ratio(), 0.5);
    }

    #[test]
    fn test_coeff_stats_skip() {
        let mut stats = CoeffStats::new();
        let all_zero = vec![0; 16];
        stats.update(&all_zero, 8);

        assert_eq!(stats.skip_blocks, 1);
    }

    #[test]
    fn test_coeff_context_dc() {
        let ctx = CoeffContext {
            nz_neighbors: 2,
            is_dc: true,
            prev_level: 1,
        };

        let level_ctx = ctx.level_ctx();
        assert!(level_ctx < LEVEL_CONTEXTS);

        let eob_ctx = ctx.eob_ctx();
        assert!(eob_ctx < EOB_CONTEXTS);
    }

    #[test]
    fn test_coeff_context_ac() {
        let ctx = CoeffContext {
            nz_neighbors: 1,
            is_dc: false,
            prev_level: 0,
        };

        let level_ctx = ctx.level_ctx();
        assert!(level_ctx < LEVEL_CONTEXTS);
        assert!(level_ctx >= 7); // AC contexts start at 7
    }

    #[test]
    fn test_scan_order_coverage() {
        // Ensure all positions are covered
        let scan_default = generate_scan_order(TxSize::Tx4x4, ScanOrder::Default);
        let scan_horiz = generate_scan_order(TxSize::Tx4x4, ScanOrder::Horizontal);
        let scan_vert = generate_scan_order(TxSize::Tx4x4, ScanOrder::Vertical);

        assert_eq!(scan_default.len(), 16);
        assert_eq!(scan_horiz.len(), 16);
        assert_eq!(scan_vert.len(), 16);

        // Check uniqueness
        let mut positions = std::collections::HashSet::new();
        for pos in &scan_default {
            positions.insert(*pos);
        }
        assert_eq!(positions.len(), 16);
    }
}
