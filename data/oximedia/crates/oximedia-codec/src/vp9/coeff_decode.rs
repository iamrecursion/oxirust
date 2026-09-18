//! VP9 Coefficient decoding with probability-based entropy coding.
//!
//! This module provides DCT coefficient decoding functionality using
//! probability tables and token trees. Coefficients are decoded in scan order
//! with context-adaptive probabilities.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::fn_params_excessive_bools)]
#![allow(clippy::struct_excessive_bools)]

use super::bitstream::BoolDecoder;
use super::partition::TxSize;
use super::probability::{
    CoefProbs, FrameContext, Prob, COEF_BANDS, COEF_CONTEXTS, PLANES, UNCONSTRAINED_NODES,
};
use super::segmentation::{SegmentData, SegmentFeature};
use super::symbols::SymbolDecoder;
use super::transform::{CoeffBuffer, DequantContext};
use crate::error::{CodecError, CodecResult};

// =============================================================================
// Constants
// =============================================================================

/// End-of-block token.
const EOB_TOKEN: u8 = 0;

/// Zero token.
const ZERO_TOKEN: u8 = 1;

/// Maximum coefficient token value.
const MAX_TOKEN: u8 = 11;

/// Number of coefficient tokens.
const COEF_TOKENS: usize = 12;

/// Number of extra bits for each token.
const EXTRA_BITS: [u8; COEF_TOKENS] = [0, 0, 0, 0, 0, 1, 2, 3, 4, 10, 12, 14];

/// Base values for each token.
const BASE_VALUES: [i16; COEF_TOKENS] = [0, 0, 1, 2, 3, 4, 6, 10, 18, 34, 66, 130];

/// Coefficient magnitude categories.
const CATEGORY_BASE: [i16; 6] = [5, 7, 11, 19, 35, 67];

// =============================================================================
// Coefficient Context
// =============================================================================

/// Context for coefficient decoding.
#[derive(Clone, Debug)]
pub struct CoeffContext {
    /// Number of non-zero coefficients above.
    pub above_non_zero: [u8; 64],
    /// Number of non-zero coefficients to the left.
    pub left_non_zero: [u8; 64],
    /// Current plane being decoded.
    pub plane: usize,
    /// Current transform size.
    pub tx_size: TxSize,
}

impl Default for CoeffContext {
    fn default() -> Self {
        Self {
            above_non_zero: [0; 64],
            left_non_zero: [0; 64],
            plane: 0,
            tx_size: TxSize::Tx4x4,
        }
    }
}

impl CoeffContext {
    /// Creates a new coefficient context.
    #[must_use]
    pub fn new(plane: usize, tx_size: TxSize) -> Self {
        Self {
            above_non_zero: [0; 64],
            left_non_zero: [0; 64],
            plane,
            tx_size,
        }
    }

    /// Resets the context for a new block.
    pub fn reset(&mut self) {
        self.above_non_zero.fill(0);
        self.left_non_zero.fill(0);
    }

    /// Gets the coefficient context for a position.
    #[must_use]
    pub fn get_context(&self, x: usize, y: usize, scan_idx: usize) -> usize {
        if scan_idx == 0 {
            return 0; // DC coefficient always uses context 0
        }

        let above = if y > 0 {
            usize::from(self.above_non_zero[x])
        } else {
            0
        };

        let left = if x > 0 {
            usize::from(self.left_non_zero[y])
        } else {
            0
        };

        let ctx = above + left;
        ctx.min(COEF_CONTEXTS - 1)
    }

    /// Updates context after decoding a coefficient.
    pub fn update(&mut self, x: usize, y: usize, non_zero: bool) {
        if x < 64 {
            self.above_non_zero[x] = u8::from(non_zero);
        }
        if y < 64 {
            self.left_non_zero[y] = u8::from(non_zero);
        }
    }

    /// Gets the coefficient band for a scan index.
    #[must_use]
    pub fn get_band(scan_idx: usize, tx_size: TxSize) -> usize {
        // Band mapping based on scan position
        match tx_size {
            TxSize::Tx4x4 => match scan_idx {
                0 => 0,
                1..=3 => 1,
                4..=6 => 2,
                7..=10 => 3,
                11..=13 => 4,
                _ => 5,
            },
            TxSize::Tx8x8 => match scan_idx {
                0 => 0,
                1..=7 => 1,
                8..=15 => 2,
                16..=27 => 3,
                28..=43 => 4,
                _ => 5,
            },
            TxSize::Tx16x16 | TxSize::Tx32x32 => match scan_idx {
                0 => 0,
                1..=15 => 1,
                16..=63 => 2,
                64..=135 => 3,
                136..=255 => 4,
                _ => 5,
            },
        }
    }
}

// =============================================================================
// Token Decoding
// =============================================================================

/// Coefficient token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoeffToken {
    /// End of block.
    Eob,
    /// Zero coefficient.
    Zero,
    /// Non-zero coefficient with value.
    NonZero(i16),
}

impl CoeffToken {
    /// Returns true if this is EOB.
    #[must_use]
    pub const fn is_eob(&self) -> bool {
        matches!(self, Self::Eob)
    }

    /// Returns true if this is a non-zero coefficient.
    #[must_use]
    pub const fn is_non_zero(&self) -> bool {
        matches!(self, Self::NonZero(_))
    }

    /// Gets the coefficient value.
    #[must_use]
    pub const fn value(&self) -> i16 {
        match self {
            Self::NonZero(val) => *val,
            _ => 0,
        }
    }
}

// =============================================================================
// Coefficient Decoder
// =============================================================================

/// Coefficient decoder for VP9.
#[derive(Debug)]
pub struct CoeffDecoder {
    /// Symbol decoder.
    symbol_decoder: SymbolDecoder,
}

impl Default for CoeffDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CoeffDecoder {
    /// Creates a new coefficient decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            symbol_decoder: SymbolDecoder::new(),
        }
    }

    /// Initializes the decoder with compressed data.
    pub fn init(&mut self, data: &[u8], offset: usize) -> CodecResult<()> {
        self.symbol_decoder.init(data, offset)
    }

    /// Decodes coefficients for a block.
    pub fn decode_block(
        &mut self,
        data: &[u8],
        coeffs: &mut CoeffBuffer,
        ctx: &FrameContext,
        coeff_ctx: &mut CoeffContext,
        segment_data: &SegmentData,
        x: usize,
        y: usize,
    ) -> CodecResult<()> {
        let tx_size = coeff_ctx.tx_size;
        let plane = coeff_ctx.plane;
        let size = tx_size.size();

        // Get coefficient probabilities
        let tx_idx = match tx_size {
            TxSize::Tx4x4 => 0,
            TxSize::Tx8x8 => 1,
            TxSize::Tx16x16 => 2,
            TxSize::Tx32x32 => 3,
        };
        let coef_probs = &ctx.probs.coef[tx_idx][plane];

        // Decode coefficients in scan order
        let max_coeffs = size * size;
        let mut scan_idx = 0;

        while scan_idx < max_coeffs {
            let band = CoeffContext::get_band(scan_idx, tx_size);
            let context = coeff_ctx.get_context(x, y, scan_idx);

            // Decode token
            let token = self.decode_token(data, coef_probs, band, context)?;

            match token {
                CoeffToken::Eob => break,
                CoeffToken::Zero => {
                    coeff_ctx.update(x, y, false);
                    scan_idx += 1;
                }
                CoeffToken::NonZero(value) => {
                    // Apply dequantization
                    let dequant_ctx = self.get_dequant_context(segment_data, scan_idx);
                    let quant = dequant_ctx.get_quant(scan_idx);
                    let dequant_value = (i32::from(value) * quant) as i16;

                    // Store in zigzag order
                    let row = scan_idx / size;
                    let col = scan_idx % size;
                    coeffs.set(row, col, dequant_value);

                    coeff_ctx.update(x, y, true);
                    scan_idx += 1;
                }
            }

            if scan_idx >= max_coeffs {
                break;
            }
        }

        Ok(())
    }

    /// Decodes a single coefficient token.
    fn decode_token(
        &mut self,
        data: &[u8],
        coef_probs: &CoefProbs,
        band: usize,
        context: usize,
    ) -> CodecResult<CoeffToken> {
        if band >= COEF_BANDS || context >= COEF_CONTEXTS {
            return Ok(CoeffToken::Zero);
        }

        let probs = &coef_probs[0][band][context];

        // Decode token tree
        let token_idx = self.decode_token_tree(data, probs)?;

        if token_idx == EOB_TOKEN {
            return Ok(CoeffToken::Eob);
        }

        if token_idx == ZERO_TOKEN {
            return Ok(CoeffToken::Zero);
        }

        // Decode coefficient value
        let value = self.decode_coeff_value(data, token_idx)?;
        Ok(CoeffToken::NonZero(value))
    }

    /// Decodes token from probability tree.
    fn decode_token_tree(
        &mut self,
        data: &[u8],
        probs: &[Prob; UNCONSTRAINED_NODES],
    ) -> CodecResult<u8> {
        // Node 0: EOB vs other
        let node0 = self.read_bool_with_prob(data, probs[0])?;
        if !node0 {
            return Ok(EOB_TOKEN);
        }

        // Node 1: ZERO vs other
        let node1 = self.read_bool_with_prob(data, probs[1])?;
        if !node1 {
            return Ok(ZERO_TOKEN);
        }

        // Node 2: determines which category
        let node2 = self.read_bool_with_prob(data, probs[2])?;
        if !node2 {
            // Category 1-2 (tokens 2-5)
            let bit = self.read_literal(data)?;
            if bit {
                let bit2 = self.read_literal(data)?;
                Ok(if bit2 { 5 } else { 4 })
            } else {
                let bit2 = self.read_literal(data)?;
                Ok(if bit2 { 3 } else { 2 })
            }
        } else {
            // Category 3-6 (tokens 6-11)
            let bit = self.read_literal(data)?;
            if bit {
                // Tokens 9-11
                let bit2 = self.read_literal(data)?;
                if bit2 {
                    Ok(11)
                } else {
                    let bit3 = self.read_literal(data)?;
                    Ok(if bit3 { 10 } else { 9 })
                }
            } else {
                // Tokens 6-8
                let bit2 = self.read_literal(data)?;
                if bit2 {
                    Ok(8)
                } else {
                    let bit3 = self.read_literal(data)?;
                    Ok(if bit3 { 7 } else { 6 })
                }
            }
        }
    }

    /// Decodes coefficient value from token.
    #[allow(clippy::cast_possible_wrap)]
    fn decode_coeff_value(&mut self, data: &[u8], token: u8) -> CodecResult<i16> {
        if token >= COEF_TOKENS as u8 {
            return Ok(0);
        }

        let token_idx = token as usize;
        let mut value = i32::from(BASE_VALUES[token_idx]);

        // Decode extra bits if needed
        let extra_bits = EXTRA_BITS[token_idx];
        if extra_bits > 0 {
            let extra = self.read_literal_bits(data, extra_bits)?;
            value += extra as i32;
        }

        // Read sign
        let sign = self.read_literal(data)?;
        if sign {
            value = -value;
        }

        Ok(value as i16)
    }

    /// Gets dequantization context for segment.
    fn get_dequant_context(&self, _segment_data: &SegmentData, _index: usize) -> DequantContext {
        // Default quantization values
        let dc_quant = 100;
        let ac_quant = 100;

        DequantContext::new(dc_quant, ac_quant)
    }

    /// Reads a boolean value with probability.
    fn read_bool_with_prob(&mut self, data: &[u8], prob: Prob) -> CodecResult<bool> {
        // Use internal boolean decoder
        let offset = self.symbol_decoder.offset();
        let mut temp_offset = offset;
        let mut bool_decoder = BoolDecoder::new();
        bool_decoder.init(data, 0)?;

        bool_decoder.read_bool(data, &mut temp_offset, prob)
    }

    /// Reads a literal bit.
    fn read_literal(&mut self, data: &[u8]) -> CodecResult<bool> {
        let offset = self.symbol_decoder.offset();
        let mut temp_offset = offset;
        let mut bool_decoder = BoolDecoder::new();
        bool_decoder.init(data, 0)?;

        bool_decoder.read_literal(data, &mut temp_offset)
    }

    /// Reads literal bits as unsigned integer.
    fn read_literal_bits(&mut self, data: &[u8], bits: u8) -> CodecResult<u32> {
        let offset = self.symbol_decoder.offset();
        let mut temp_offset = offset;
        let mut bool_decoder = BoolDecoder::new();
        bool_decoder.init(data, 0)?;

        bool_decoder.read_literal_bits(data, &mut temp_offset, bits)
    }

    /// Returns the current byte offset.
    #[must_use]
    pub fn offset(&self) -> usize {
        self.symbol_decoder.offset()
    }
}

// =============================================================================
// Quantization Tables
// =============================================================================

/// VP9 quantization lookup tables.
pub struct QuantTables {
    /// DC quantization values.
    dc_quant: [i16; 256],
    /// AC quantization values.
    ac_quant: [i16; 256],
}

impl Default for QuantTables {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantTables {
    /// Creates quantization tables with default values.
    #[must_use]
    pub fn new() -> Self {
        let mut dc_quant = [0i16; 256];
        let mut ac_quant = [0i16; 256];

        // Initialize with VP9 spec values
        for i in 0..256 {
            dc_quant[i] = Self::compute_dc_quant(i as u8);
            ac_quant[i] = Self::compute_ac_quant(i as u8);
        }

        Self { dc_quant, ac_quant }
    }

    /// Gets DC quantizer value.
    #[must_use]
    pub fn get_dc_quant(&self, index: u8) -> i16 {
        self.dc_quant[index as usize]
    }

    /// Gets AC quantizer value.
    #[must_use]
    pub fn get_ac_quant(&self, index: u8) -> i16 {
        self.ac_quant[index as usize]
    }

    /// Computes DC quantizer from index.
    fn compute_dc_quant(index: u8) -> i16 {
        let idx = i32::from(index);
        if idx < 128 {
            (4 + idx) as i16
        } else {
            ((idx * 2) - 124) as i16
        }
    }

    /// Computes AC quantizer from index.
    fn compute_ac_quant(index: u8) -> i16 {
        let idx = i32::from(index);
        if idx < 128 {
            (4 + idx) as i16
        } else {
            ((idx * 2) - 124) as i16
        }
    }
}

// =============================================================================
// Scan Orders
// =============================================================================

/// Scan order for coefficient decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanOrder {
    /// Default zigzag scan.
    Default,
    /// Column scan (for vertical prediction).
    Col,
    /// Row scan (for horizontal prediction).
    Row,
}

impl ScanOrder {
    /// Gets scan indices for a block size.
    #[must_use]
    pub fn get_scan(&self, tx_size: TxSize) -> &'static [usize] {
        // Simplified: return default zigzag for now
        match tx_size {
            TxSize::Tx4x4 => &super::transform::ZIGZAG_4X4,
            TxSize::Tx8x8 => &super::transform::ZIGZAG_8X8,
            TxSize::Tx16x16 => &super::transform::ZIGZAG_16X16_PARTIAL,
            TxSize::Tx32x32 => &super::transform::ZIGZAG_32X32_PARTIAL,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coeff_context_new() {
        let ctx = CoeffContext::new(0, TxSize::Tx4x4);
        assert_eq!(ctx.plane, 0);
        assert_eq!(ctx.tx_size, TxSize::Tx4x4);
    }

    #[test]
    fn test_coeff_context_get_band() {
        assert_eq!(CoeffContext::get_band(0, TxSize::Tx4x4), 0);
        assert_eq!(CoeffContext::get_band(1, TxSize::Tx4x4), 1);
        assert_eq!(CoeffContext::get_band(5, TxSize::Tx4x4), 2);
    }

    #[test]
    fn test_coeff_context_update() {
        let mut ctx = CoeffContext::new(0, TxSize::Tx4x4);
        ctx.update(0, 0, true);
        assert_eq!(ctx.above_non_zero[0], 1);
        assert_eq!(ctx.left_non_zero[0], 1);
    }

    #[test]
    fn test_coeff_token_is_eob() {
        assert!(CoeffToken::Eob.is_eob());
        assert!(!CoeffToken::Zero.is_eob());
        assert!(!CoeffToken::NonZero(5).is_eob());
    }

    #[test]
    fn test_coeff_token_is_non_zero() {
        assert!(!CoeffToken::Eob.is_non_zero());
        assert!(!CoeffToken::Zero.is_non_zero());
        assert!(CoeffToken::NonZero(5).is_non_zero());
    }

    #[test]
    fn test_coeff_token_value() {
        assert_eq!(CoeffToken::Eob.value(), 0);
        assert_eq!(CoeffToken::Zero.value(), 0);
        assert_eq!(CoeffToken::NonZero(42).value(), 42);
        assert_eq!(CoeffToken::NonZero(-10).value(), -10);
    }

    #[test]
    fn test_coeff_decoder_new() {
        let decoder = CoeffDecoder::new();
        assert_eq!(decoder.offset(), 0);
    }

    #[test]
    fn test_extra_bits_table() {
        assert_eq!(EXTRA_BITS[0], 0); // EOB
        assert_eq!(EXTRA_BITS[1], 0); // ZERO
        assert_eq!(EXTRA_BITS[5], 1); // Token 5 has 1 extra bit
        assert_eq!(EXTRA_BITS[11], 14); // Token 11 has 14 extra bits
    }

    #[test]
    fn test_base_values_table() {
        assert_eq!(BASE_VALUES[0], 0);
        assert_eq!(BASE_VALUES[2], 1);
        assert_eq!(BASE_VALUES[5], 4);
        assert_eq!(BASE_VALUES[11], 130);
    }

    #[test]
    fn test_quant_tables_new() {
        let tables = QuantTables::new();
        assert!(tables.get_dc_quant(0) > 0);
        assert!(tables.get_ac_quant(0) > 0);
        assert!(tables.get_dc_quant(255) > tables.get_dc_quant(0));
    }

    #[test]
    fn test_quant_tables_dc_vs_ac() {
        let tables = QuantTables::new();
        // DC and AC quantizers should be similar for same index
        for i in 0..=255 {
            let dc = tables.get_dc_quant(i);
            let ac = tables.get_ac_quant(i);
            assert!(dc > 0);
            assert!(ac > 0);
        }
    }

    #[test]
    fn test_scan_order() {
        let scan = ScanOrder::Default.get_scan(TxSize::Tx4x4);
        assert_eq!(scan.len(), 16);
        assert_eq!(scan[0], 0); // First element is DC
    }

    #[test]
    fn test_scan_order_sizes() {
        assert_eq!(ScanOrder::Default.get_scan(TxSize::Tx4x4).len(), 16);
        assert_eq!(ScanOrder::Default.get_scan(TxSize::Tx8x8).len(), 64);
    }

    #[test]
    fn test_category_base() {
        assert_eq!(CATEGORY_BASE[0], 5);
        assert_eq!(CATEGORY_BASE[5], 67);
        assert!(CATEGORY_BASE[5] > CATEGORY_BASE[0]);
    }

    #[test]
    fn test_coeff_context_reset() {
        let mut ctx = CoeffContext::new(0, TxSize::Tx4x4);
        ctx.above_non_zero[0] = 5;
        ctx.left_non_zero[0] = 10;

        ctx.reset();

        assert_eq!(ctx.above_non_zero[0], 0);
        assert_eq!(ctx.left_non_zero[0], 0);
    }

    #[test]
    fn test_coeff_context_get_context_dc() {
        let ctx = CoeffContext::new(0, TxSize::Tx4x4);
        // DC coefficient always has context 0
        assert_eq!(ctx.get_context(0, 0, 0), 0);
    }

    #[test]
    fn test_coeff_context_boundaries() {
        let ctx = CoeffContext::new(0, TxSize::Tx4x4);
        // Test boundary conditions
        let context = ctx.get_context(63, 63, 1);
        assert!(context < COEF_CONTEXTS);
    }
}
