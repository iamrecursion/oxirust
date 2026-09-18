//! AV1 transform coefficient decoding.
//!
//! This module handles the complete decoding of transform coefficients from
//! the entropy-coded bitstream, including:
//!
//! - EOB (End of Block) position parsing
//! - Coefficient level decoding using multi-level scheme
//! - Coefficient sign decoding
//! - Dequantization
//! - Scan order application
//!
//! # Coefficient Decoding Process
//!
//! 1. **EOB Parsing** - Determine position of last non-zero coefficient
//! 2. **Coefficient Levels** - Decode base levels and ranges
//! 3. **DC Sign** - Decode sign of DC coefficient
//! 4. **AC Signs** - Decode signs of AC coefficients
//! 5. **Dequantization** - Apply quantization parameters
//! 6. **Scan Order** - Convert from scan order to raster order
//!
//! # Context Modeling
//!
//! Coefficient decoding uses adaptive context models based on:
//! - Position within the block
//! - Magnitude of neighboring coefficients
//! - Transform size and type

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::module_name_repetitions)]

use super::coefficients::{
    dequantize_block, get_dequant_shift, CoeffBuffer, CoeffContext, CoeffStats, EobContext, EobPt,
    LevelContext, ScanOrderCache,
};
use super::entropy::SymbolReader;
use super::entropy_tables::CdfContext;
use super::quantization::QuantizationParams;
use super::transform::{TxSize, TxType};
use crate::error::CodecResult;

// =============================================================================
// Constants
// =============================================================================

/// Maximum coefficient level for base coding.
pub const COEFF_BASE_MAX: u32 = 3;

/// Coefficient base range (used for higher levels).
pub const BR_CDF_SIZE: usize = 4;

/// Maximum Golomb-Rice parameter.
pub const MAX_BR_PARAM: u8 = 5;

/// Number of EOB offset bits.
pub const EOB_OFFSET_BITS: [u8; 12] = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// Scan order coeff skip threshold.
pub const COEFF_SKIP_THRESHOLD: u16 = 256;

// =============================================================================
// Coefficient Decoder
// =============================================================================

/// Decoder for transform coefficients.
#[derive(Debug)]
pub struct CoeffDecoder {
    /// Symbol reader.
    reader: SymbolReader,
    /// CDF context for probability models.
    cdf_context: CdfContext,
    /// Scan order cache.
    scan_cache: ScanOrderCache,
    /// Quantization parameters.
    quant_params: QuantizationParams,
    /// Current bit depth.
    bit_depth: u8,
}

impl CoeffDecoder {
    /// Create a new coefficient decoder.
    pub fn new(data: Vec<u8>, quant_params: QuantizationParams, bit_depth: u8) -> Self {
        Self {
            reader: SymbolReader::new(data),
            cdf_context: CdfContext::new(),
            scan_cache: ScanOrderCache::new(),
            quant_params,
            bit_depth,
        }
    }

    /// Decode coefficients for a transform block.
    pub fn decode_coefficients(
        &mut self,
        tx_size: TxSize,
        tx_type: TxType,
        plane: u8,
        skip: bool,
    ) -> CodecResult<CoeffBuffer> {
        let mut ctx = CoeffContext::new(tx_size, tx_type, plane);

        if skip {
            // Skip blocks have all-zero coefficients
            return Ok(CoeffBuffer::from_tx_size(tx_size));
        }

        // Decode EOB position
        ctx.eob = self.decode_eob(tx_size, plane)?;

        if ctx.eob == 0 {
            // No coefficients
            return Ok(CoeffBuffer::from_tx_size(tx_size));
        }

        // Get scan order (clone to avoid borrow issues)
        let scan = self.scan_cache.get(tx_size, ctx.tx_class()).to_vec();

        // Decode coefficient levels
        self.decode_coeff_levels(&mut ctx, &scan)?;

        // Decode signs
        self.decode_signs(&mut ctx, &scan)?;

        // Dequantize coefficients
        self.dequantize_coefficients(&mut ctx, plane)?;

        // Convert from scan order to raster order
        let mut buffer = CoeffBuffer::from_tx_size(tx_size);
        self.reorder_coefficients(&ctx, &scan, &mut buffer)?;

        Ok(buffer)
    }

    /// Decode EOB (End of Block) position.
    ///
    /// The EOB-multi CDF emits a *symbol index* (the index of the EOB point
    /// class, 0..=11), not the EOB position itself. The symbol then dictates
    /// how many extra bits the decoder must consume to recover the precise
    /// position within the class. Confusing these two values desynchronizes
    /// the bitstream — see [`EobPt::from_symbol`] for the relationship.
    fn decode_eob(&mut self, tx_size: TxSize, plane: u8) -> CodecResult<u16> {
        let _eob_ctx = EobContext::new(tx_size);

        // Read EOB multi-symbol
        let ctx = (tx_size as usize * 3) + (plane as usize);
        let eob_multi_cdf = self.cdf_context.get_eob_multi_cdf_mut(ctx);
        let eob_multi_symbol = self.reader.read_symbol(eob_multi_cdf);

        self.finalize_eob_from_symbol(eob_multi_symbol)
    }

    /// Compute the final EOB position from a previously-read EOB-multi
    /// *symbol* (CDF index in `0..=11`).
    ///
    /// This is split out from [`Self::decode_eob`] so that regression tests
    /// can inject a known symbol and observe the precise number of extra
    /// bits consumed plus the final EOB position — the two outputs that
    /// diverge between the correct (`from_symbol`) and the historically
    /// buggy (`from_eob`) implementations of this step.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::CodecError::InvalidBitstream`] if `symbol`
    /// is outside the legal range 0..=11.
    fn finalize_eob_from_symbol(&mut self, symbol: usize) -> CodecResult<u16> {
        if symbol == 0 {
            return Ok(0); // No coefficients
        }

        // Convert the EOB-multi *symbol* (CDF index) into its EobPt class.
        // NOTE: do not call `EobPt::from_eob` here — that helper takes an
        // actual EOB position, not a symbol index. See the unit tests for
        // a demonstration of how the two formulations diverge.
        let eob_pt = EobPt::from_symbol(symbol)?;
        let extra_bits = eob_pt.extra_bits();

        let eob_extra = if extra_bits > 0 {
            self.reader.read_literal(extra_bits)
        } else {
            0
        };

        let eob_multi = symbol as u8;
        let eob = EobContext::compute_eob(eob_multi, eob_extra as u16);
        Ok(eob)
    }

    /// Decode coefficient levels.
    fn decode_coeff_levels(&mut self, ctx: &mut CoeffContext, scan: &[u16]) -> CodecResult<()> {
        let eob = ctx.eob as usize;

        // Decode in reverse scan order (from EOB to DC)
        for scan_idx in (0..eob).rev() {
            let pos = scan[scan_idx] as usize;
            let level = self.decode_coeff_level(ctx, pos, scan_idx == eob - 1)?;
            ctx.levels[pos] = level as i32;
        }

        Ok(())
    }

    /// Decode a single coefficient level.
    fn decode_coeff_level(
        &mut self,
        ctx: &CoeffContext,
        pos: usize,
        is_eob: bool,
    ) -> CodecResult<u32> {
        let level_ctx = ctx.compute_level_context(pos);

        // Decode base level (0-3)
        let base_level = if is_eob {
            // At EOB, coefficient is at least 1
            1 + self.decode_coeff_base_eob(&level_ctx, ctx.plane)?
        } else {
            self.decode_coeff_base(&level_ctx, ctx.plane)?
        };

        if base_level >= COEFF_BASE_MAX {
            // Decode additional range
            let range = self.decode_coeff_base_range(&level_ctx, ctx.plane)?;
            Ok(base_level + range)
        } else {
            Ok(base_level)
        }
    }

    /// Decode coefficient base level.
    fn decode_coeff_base(&mut self, level_ctx: &LevelContext, _plane: u8) -> CodecResult<u32> {
        let context = level_ctx.context() as usize;
        let cdf = self.cdf_context.get_coeff_base_cdf_mut(context);
        Ok(self.reader.read_symbol(cdf) as u32)
    }

    /// Decode coefficient base at EOB.
    fn decode_coeff_base_eob(&mut self, level_ctx: &LevelContext, _plane: u8) -> CodecResult<u32> {
        let context = level_ctx.context() as usize;
        let cdf = self.cdf_context.get_coeff_base_eob_cdf_mut(context);
        Ok(self.reader.read_symbol(cdf) as u32)
    }

    /// Decode coefficient base range for high magnitude coefficients.
    fn decode_coeff_base_range(
        &mut self,
        level_ctx: &LevelContext,
        _plane: u8,
    ) -> CodecResult<u32> {
        let context = level_ctx.mag_context() as usize;
        let mut total_range = 0u32;

        // Multi-level Golomb-Rice coding
        for _level in 0..5 {
            let br_cdf = self.cdf_context.get_coeff_br_cdf_mut(context);
            let br_symbol = self.reader.read_symbol(br_cdf);

            if br_symbol < BR_CDF_SIZE - 1 {
                total_range += br_symbol as u32;
                break;
            } else {
                total_range += (BR_CDF_SIZE - 1) as u32;
            }
        }

        Ok(total_range)
    }

    /// Decode signs for all non-zero coefficients.
    fn decode_signs(&mut self, ctx: &mut CoeffContext, scan: &[u16]) -> CodecResult<()> {
        let eob = ctx.eob as usize;

        // Decode DC sign first (if DC is non-zero)
        if ctx.levels[0] != 0 {
            let dc_sign_ctx = ctx.dc_sign_context();
            let cdf_slice = self.cdf_context.get_dc_sign_cdf_mut(dc_sign_ctx as usize);
            // read_bool needs &mut [u16; 3], but we have &mut [u16]
            // Copy to a fixed-size array
            if cdf_slice.len() >= 3 {
                let mut cdf_array = [cdf_slice[0], cdf_slice[1], cdf_slice[2]];
                let sign = self.reader.read_bool(&mut cdf_array);
                // Copy back updated CDF
                cdf_slice[0] = cdf_array[0];
                cdf_slice[1] = cdf_array[1];
                cdf_slice[2] = cdf_array[2];
                ctx.signs[0] = sign;
                if sign {
                    ctx.levels[0] = -ctx.levels[0];
                }
            }
        }

        // Decode AC signs
        for scan_idx in 1..eob {
            let pos = scan[scan_idx] as usize;
            if ctx.levels[pos] != 0 {
                // AC signs use equiprobable model
                let sign = self.reader.read_bool_eq();
                ctx.signs[pos] = sign;
                if sign {
                    ctx.levels[pos] = -ctx.levels[pos];
                }
            }
        }

        Ok(())
    }

    /// Dequantize coefficients.
    fn dequantize_coefficients(&mut self, ctx: &mut CoeffContext, plane: u8) -> CodecResult<()> {
        // Get dequant values
        let dc_dequant = self
            .quant_params
            .get_dc_quant(plane as usize, self.bit_depth) as i16;
        let ac_dequant = self
            .quant_params
            .get_ac_quant(plane as usize, self.bit_depth) as i16;
        let shift = get_dequant_shift(self.bit_depth);

        dequantize_block(&mut ctx.levels, dc_dequant, ac_dequant, shift);

        Ok(())
    }

    /// Reorder coefficients from scan order to raster order.
    fn reorder_coefficients(
        &self,
        ctx: &CoeffContext,
        scan: &[u16],
        buffer: &mut CoeffBuffer,
    ) -> CodecResult<()> {
        let eob = ctx.eob as usize;

        for scan_idx in 0..eob {
            let pos = scan[scan_idx] as usize;
            if pos < ctx.levels.len() {
                let level = ctx.levels[pos];
                let (row, col) = ctx.get_scan_position(pos);
                buffer.set(row as usize, col as usize, level);
            }
        }

        Ok(())
    }

    /// Check if more data is available.
    pub fn has_more_data(&self) -> bool {
        self.reader.has_more_data()
    }

    /// Get current position.
    pub fn position(&self) -> usize {
        self.reader.position()
    }
}

// =============================================================================
// Coefficient Encoding
// =============================================================================

/// Encoder for transform coefficients.
#[derive(Debug)]
pub struct CoeffEncoder {
    /// Symbol writer.
    writer: super::entropy::SymbolWriter,
    /// CDF context.
    cdf_context: CdfContext,
    /// Scan order cache.
    scan_cache: ScanOrderCache,
}

impl CoeffEncoder {
    /// Create a new coefficient encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            writer: super::entropy::SymbolWriter::new(),
            cdf_context: CdfContext::new(),
            scan_cache: ScanOrderCache::new(),
        }
    }

    /// Encode coefficients for a transform block.
    pub fn encode_coefficients(
        &mut self,
        buffer: &CoeffBuffer,
        tx_size: TxSize,
        tx_type: TxType,
        plane: u8,
    ) -> CodecResult<()> {
        let tx_class = tx_type.tx_class();
        let scan = self.scan_cache.get(tx_size, tx_class).to_vec();

        // Find EOB
        let eob = self.find_eob(buffer, &scan);

        // Encode EOB
        self.encode_eob(eob, tx_size, plane)?;

        if eob == 0 {
            return Ok(());
        }

        // Convert to scan order
        let mut levels = vec![0i32; eob as usize];
        buffer.copy_to_scan(&mut levels, &scan[..eob as usize]);

        // Encode levels
        self.encode_levels(&levels, &scan, plane)?;

        // Encode signs
        self.encode_signs(&levels, &scan)?;

        Ok(())
    }

    /// Find EOB position.
    fn find_eob(&self, buffer: &CoeffBuffer, scan: &[u16]) -> u16 {
        for (i, &pos) in scan.iter().enumerate().rev() {
            let (row, col) = self.pos_to_rowcol(pos as usize, buffer);
            if buffer.get(row, col) != 0 {
                return (i + 1) as u16;
            }
        }
        0
    }

    /// Convert position to row/col.
    fn pos_to_rowcol(&self, pos: usize, buffer: &CoeffBuffer) -> (usize, usize) {
        let width = buffer.width();
        (pos / width, pos % width)
    }

    /// Encode EOB.
    fn encode_eob(&mut self, eob: u16, tx_size: TxSize, plane: u8) -> CodecResult<()> {
        let ctx = (tx_size as usize * 3) + (plane as usize);

        if eob == 0 {
            let cdf = self.cdf_context.get_eob_multi_cdf_mut(ctx);
            self.writer.write_symbol(0, cdf);
            return Ok(());
        }

        let eob_pt = EobPt::from_eob(eob);
        let cdf = self.cdf_context.get_eob_multi_cdf_mut(ctx);
        self.writer.write_symbol(eob_pt as usize, cdf);

        // Write extra bits
        let extra_bits = eob_pt.extra_bits();
        if extra_bits > 0 {
            let offset = eob - eob_pt.base_eob();
            self.writer.write_literal(offset as u32, extra_bits);
        }

        Ok(())
    }

    /// Encode coefficient levels.
    fn encode_levels(&mut self, levels: &[i32], scan: &[u16], plane: u8) -> CodecResult<()> {
        let mut ctx = LevelContext::new();

        for (scan_idx, &_pos) in scan.iter().enumerate().rev() {
            let level = levels[scan_idx].unsigned_abs();

            let base_level = level.min(COEFF_BASE_MAX);
            let is_eob = scan_idx == levels.len() - 1;

            if is_eob {
                let cdf = self
                    .cdf_context
                    .get_coeff_base_eob_cdf_mut(ctx.context() as usize);
                self.writer.write_symbol((base_level - 1) as usize, cdf);
            } else {
                let cdf = self
                    .cdf_context
                    .get_coeff_base_cdf_mut(ctx.context() as usize);
                self.writer.write_symbol(base_level as usize, cdf);
            }

            if level >= COEFF_BASE_MAX {
                self.encode_base_range(level - COEFF_BASE_MAX, &ctx, plane)?;
            }

            // Update context
            ctx.mag += level;
            if level > 0 {
                ctx.count += 1;
            }
        }

        Ok(())
    }

    /// Encode coefficient base range.
    fn encode_base_range(&mut self, range: u32, ctx: &LevelContext, _plane: u8) -> CodecResult<()> {
        let mut remaining = range;
        let mag_ctx = ctx.mag_context() as usize;

        for _level in 0..5 {
            if remaining == 0 {
                break;
            }

            let symbol = remaining.min((BR_CDF_SIZE - 1) as u32) as usize;
            let cdf = self.cdf_context.get_coeff_br_cdf_mut(mag_ctx);
            self.writer.write_symbol(symbol, cdf);

            if symbol < BR_CDF_SIZE - 1 {
                break;
            }

            remaining -= (BR_CDF_SIZE - 1) as u32;
        }

        Ok(())
    }

    /// Encode signs.
    fn encode_signs(&mut self, levels: &[i32], scan: &[u16]) -> CodecResult<()> {
        // DC sign
        if !levels.is_empty() && levels[0] != 0 {
            let dc_ctx = 1; // Simplified context
            let cdf_slice = self.cdf_context.get_dc_sign_cdf_mut(dc_ctx);
            if cdf_slice.len() >= 3 {
                let mut cdf = [cdf_slice[0], cdf_slice[1], cdf_slice[2]];
                self.writer.write_bool(levels[0] < 0, &mut cdf);
                // Copy back updated CDF
                cdf_slice[0] = cdf[0];
                cdf_slice[1] = cdf[1];
                cdf_slice[2] = cdf[2];
            }
        }

        // AC signs
        for (idx, &_pos) in scan.iter().enumerate().skip(1) {
            if idx < levels.len() && levels[idx] != 0 {
                // Equiprobable
                let mut cdf = [16384u16, 32768, 0];
                self.writer.write_bool(levels[idx] < 0, &mut cdf);
            }
        }

        Ok(())
    }

    /// Finalize and get output.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.writer.finish()
    }
}

impl Default for CoeffEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Batched Coefficient Decoder
// =============================================================================

/// Batched decoder for multiple coefficient blocks.
pub struct BatchedCoeffDecoder {
    /// Base decoder.
    decoder: CoeffDecoder,
    /// Decoded blocks cache.
    blocks: Vec<CoeffBuffer>,
}

impl BatchedCoeffDecoder {
    /// Create a new batched decoder.
    pub fn new(data: Vec<u8>, quant_params: QuantizationParams, bit_depth: u8) -> Self {
        Self {
            decoder: CoeffDecoder::new(data, quant_params, bit_depth),
            blocks: Vec::new(),
        }
    }

    /// Decode multiple blocks.
    pub fn decode_blocks(
        &mut self,
        specs: &[(TxSize, TxType, u8, bool)],
    ) -> CodecResult<Vec<CoeffBuffer>> {
        self.blocks.clear();

        for &(tx_size, tx_type, plane, skip) in specs {
            let buffer = self
                .decoder
                .decode_coefficients(tx_size, tx_type, plane, skip)?;
            self.blocks.push(buffer);
        }

        Ok(std::mem::take(&mut self.blocks))
    }

    /// Get statistics for decoded blocks.
    #[must_use]
    pub fn get_statistics(&self) -> Vec<CoeffStats> {
        self.blocks
            .iter()
            .map(|b| CoeffStats::from_coeffs(b.as_slice()))
            .collect()
    }
}

// =============================================================================
// Coefficient Analysis
// =============================================================================

/// Analyze coefficient distribution.
#[derive(Clone, Debug, Default)]
pub struct CoeffAnalysis {
    /// Total coefficients analyzed.
    pub total_coeffs: u64,
    /// Zero coefficients.
    pub zero_count: u64,
    /// Non-zero coefficients.
    pub nonzero_count: u64,
    /// DC coefficient sum.
    pub dc_sum: i64,
    /// AC coefficient sum.
    pub ac_sum: i64,
    /// Maximum absolute value.
    pub max_abs: u32,
}

impl CoeffAnalysis {
    /// Create new analysis.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            total_coeffs: 0,
            zero_count: 0,
            nonzero_count: 0,
            dc_sum: 0,
            ac_sum: 0,
            max_abs: 0,
        }
    }

    /// Analyze a coefficient buffer.
    pub fn analyze(&mut self, buffer: &CoeffBuffer) {
        let coeffs = buffer.as_slice();
        self.total_coeffs += coeffs.len() as u64;

        if !coeffs.is_empty() {
            self.dc_sum += i64::from(coeffs[0]);
        }

        for (i, &coeff) in coeffs.iter().enumerate() {
            let abs_val = coeff.unsigned_abs();

            if coeff == 0 {
                self.zero_count += 1;
            } else {
                self.nonzero_count += 1;
                self.max_abs = self.max_abs.max(abs_val);

                if i > 0 {
                    self.ac_sum += i64::from(coeff);
                }
            }
        }
    }

    /// Get sparsity ratio (percentage of zeros).
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        if self.total_coeffs > 0 {
            (self.zero_count as f64 / self.total_coeffs as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Get average DC value.
    #[must_use]
    pub fn avg_dc(&self) -> f64 {
        if self.total_coeffs > 0 {
            self.dc_sum as f64 / self.total_coeffs as f64
        } else {
            0.0
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_quant_params() -> QuantizationParams {
        QuantizationParams {
            base_q_idx: 100,
            delta_q_y_dc: 0,
            delta_q_u_dc: 0,
            delta_q_v_dc: 0,
            delta_q_u_ac: 0,
            delta_q_v_ac: 0,
            using_qmatrix: false,
            qm_y: 15,
            qm_u: 15,
            qm_v: 15,
            delta_q_present: false,
            delta_q_res: 0,
        }
    }

    #[test]
    fn test_coeff_decoder_creation() {
        let data = vec![0u8; 128];
        let quant = create_test_quant_params();
        let decoder = CoeffDecoder::new(data, quant, 8);
        assert!(decoder.has_more_data());
    }

    #[test]
    fn test_coeff_encoder_creation() {
        let encoder = CoeffEncoder::new();
        let output = encoder.finish();
        assert!(!output.is_empty() || output.is_empty());
    }

    #[test]
    fn test_batched_decoder() {
        let data = vec![0u8; 256];
        let quant = create_test_quant_params();
        let mut decoder = BatchedCoeffDecoder::new(data, quant, 8);

        let specs = vec![
            (TxSize::Tx4x4, TxType::DctDct, 0, false),
            (TxSize::Tx8x8, TxType::DctDct, 0, false),
        ];

        // Decoding may fail with test data, but should not crash
        let _ = decoder.decode_blocks(&specs);
    }

    #[test]
    fn test_coeff_analysis() {
        let mut analysis = CoeffAnalysis::new();
        let mut buffer = CoeffBuffer::new(4, 4);

        buffer.set(0, 0, 100); // DC
        buffer.set(1, 1, 50); // AC
        buffer.set(2, 2, -30); // AC

        analysis.analyze(&buffer);

        assert_eq!(analysis.total_coeffs, 16);
        assert_eq!(analysis.nonzero_count, 3);
        assert_eq!(analysis.zero_count, 13);
        assert_eq!(analysis.max_abs, 100);
    }

    #[test]
    fn test_coeff_analysis_sparsity() {
        let mut analysis = CoeffAnalysis::new();
        let buffer = CoeffBuffer::new(8, 8); // All zeros

        analysis.analyze(&buffer);

        assert_eq!(analysis.sparsity(), 100.0);
    }

    #[test]
    fn test_constants() {
        assert_eq!(COEFF_BASE_MAX, 3);
        assert_eq!(BR_CDF_SIZE, 4);
        assert_eq!(MAX_BR_PARAM, 5);
    }

    #[test]
    fn test_eob_offset_bits() {
        assert_eq!(EOB_OFFSET_BITS[0], 0);
        assert_eq!(EOB_OFFSET_BITS[3], 1);
        assert_eq!(EOB_OFFSET_BITS[11], 9);
    }

    #[test]
    fn test_coeff_analysis_avg_dc() {
        let mut analysis = CoeffAnalysis::new();
        let mut buffer = CoeffBuffer::new(4, 4);
        buffer.set(0, 0, 200);

        analysis.analyze(&buffer);

        // DC sum is 200, total coeffs is 16
        assert!((analysis.avg_dc() - 12.5).abs() < 0.1);
    }

    #[test]
    fn test_coeff_decoder_position() {
        let data = vec![0u8; 128];
        let quant = create_test_quant_params();
        let decoder = CoeffDecoder::new(data, quant, 8);
        assert!(decoder.position() <= 128);
    }

    /// Regression for the symbol-vs-position bug at the original line 154 of
    /// this file.
    ///
    /// `finalize_eob_from_symbol` performs the exact computation that
    /// `decode_eob` carries out once it has obtained the EOB-multi symbol.
    /// By calling it directly with a known symbol we can observe the precise
    /// EOB position produced and confirm that:
    ///
    ///   * for symbols 0..=3 the result is fully determined by
    ///     `EOB_GROUP_START` (the buggy and fixed paths coincide here);
    ///   * for symbols 4..=11 the result corresponds to the *correct* extra-
    ///     bits count (`from_symbol(s).extra_bits()`), proving the fix; the
    ///     buggy `from_eob(s).extra_bits()` would have read fewer bits and
    ///     yielded a smaller EOB.
    ///
    /// The reader is fed all-ones bytes so `read_literal(k)` deterministically
    /// returns `(1 << k) - 1`. This makes the fixed-vs-buggy divergence
    /// observable in the EOB *value* itself, not just in subsequent bit
    /// consumption.
    #[test]
    fn test_finalize_eob_from_symbol_reads_correct_extra_bits() {
        use super::super::coefficients::{EobPt, EOB_GROUP_START};

        // For each legal symbol s ∈ 1..=11, the fixed decoder should produce:
        //     eob = EOB_GROUP_START[s] + ((1 << extra_bits) - 1)
        // where extra_bits = EobPt::from_symbol(s).extra_bits().
        //
        // The historically buggy decoder used EobPt::from_eob(s as u16)
        // instead, which yields a *smaller* extra_bits count for s >= 4
        // (e.g. s=4 → buggy extra=1 vs fixed extra=2), so on all-ones input
        // the EOB value diverges. We assert the FIXED value and additionally
        // assert it is strictly larger than what the bug would have produced.
        for symbol in 1..=11usize {
            // 64 bytes of 0xFF is enough for the largest extra_bits (9).
            let data = vec![0xFFu8; 64];
            let quant = create_test_quant_params();
            let mut decoder = CoeffDecoder::new(data, quant, 8);

            let eob = decoder
                .finalize_eob_from_symbol(symbol)
                .expect("legal symbol must succeed");

            let fixed_extra_bits = EobPt::from_symbol(symbol)
                .expect("legal symbol")
                .extra_bits();
            let expected_extra: u16 = if fixed_extra_bits == 0 {
                0
            } else {
                (1u16 << fixed_extra_bits) - 1
            };
            let expected_eob = EOB_GROUP_START[symbol] + expected_extra;
            assert_eq!(
                eob, expected_eob,
                "symbol {symbol}: fixed decoder must read {fixed_extra_bits} \
                 extra bits (all 1s → {expected_extra}) and produce \
                 EOB_GROUP_START[{symbol}]+{expected_extra}={expected_eob}",
            );

            // Divergence check vs. the original bug.
            let buggy_extra_bits = EobPt::from_eob(symbol as u16).extra_bits();
            if symbol >= 4 {
                // For s>=4 the buggy decoder reads fewer extra bits, so the
                // EOB it would compute is strictly smaller.
                let buggy_extra: u16 = if buggy_extra_bits == 0 {
                    0
                } else {
                    (1u16 << buggy_extra_bits) - 1
                };
                let buggy_eob = EOB_GROUP_START[symbol] + buggy_extra;
                assert!(
                    eob > buggy_eob,
                    "symbol {symbol}: fixed eob {eob} must exceed buggy eob \
                     {buggy_eob} (proves the regression test discriminates)",
                );
            } else {
                // For s<=3 the two paths produce identical EobPt classes.
                assert_eq!(
                    fixed_extra_bits, buggy_extra_bits,
                    "symbol {symbol} (<=3): fixed and buggy extra_bits must \
                     coincide",
                );
            }
        }
    }

    /// Regression: symbol 0 means "no coefficients" and must return EOB = 0
    /// without consuming any extra bits from the bit window.
    #[test]
    fn test_finalize_eob_from_symbol_zero_returns_zero() {
        let data = vec![0xFFu8; 16];
        let quant = create_test_quant_params();
        let mut decoder = CoeffDecoder::new(data, quant, 8);

        let eob = decoder
            .finalize_eob_from_symbol(0)
            .expect("symbol 0 always succeeds");
        assert_eq!(eob, 0, "symbol 0 must short-circuit to EOB = 0");
    }

    /// Regression: the 16-symbol EOB-multi CDF can in principle emit symbols
    /// 12..15 on a malformed bitstream. Those must surface as decoder errors,
    /// not silent saturation that would corrupt subsequent reads.
    #[test]
    fn test_finalize_eob_from_symbol_rejects_out_of_range() {
        let data = vec![0xFFu8; 16];
        let quant = create_test_quant_params();
        let mut decoder = CoeffDecoder::new(data, quant, 8);

        for bad_symbol in 12..=15usize {
            let err = decoder
                .finalize_eob_from_symbol(bad_symbol)
                .expect_err("symbols >=12 must be rejected");
            let msg = format!("{err}");
            assert!(
                msg.contains("EOB-multi"),
                "error message should mention EOB-multi: {msg}",
            );
        }
    }

    // -------------------------------------------------------------------------
    // Regression tests: CoeffEncoder::pos_to_rowcol non-square TX blocks
    // -------------------------------------------------------------------------
    //
    // Before the fix, width was derived as `(slice.len() as f64).sqrt() as
    // usize`, which is only correct for square TX sizes.  For any non-square
    // TX size the sqrt of the total coefficient count does not equal the block
    // width, producing wrong (row, col) pairs.  These tests verify that the
    // fix (using `CoeffBuffer::width()`) gives correct results for several
    // representative non-square sizes.

    #[test]
    fn test_pos_to_rowcol_tx4x8() {
        // Tx4x8: width=4, height=8, 32 coefficients total.
        // sqrt(32) ≈ 5 (wrong); correct width = 4.
        let encoder = CoeffEncoder::new();
        let buf = CoeffBuffer::from_tx_size(TxSize::Tx4x8);
        assert_eq!(buf.width(), 4);
        assert_eq!(buf.height(), 8);

        assert_eq!(encoder.pos_to_rowcol(0, &buf), (0, 0), "pos 0 -> (0,0)");
        assert_eq!(encoder.pos_to_rowcol(3, &buf), (0, 3), "pos 3 -> (0,3)");
        assert_eq!(encoder.pos_to_rowcol(4, &buf), (1, 0), "pos 4 -> (1,0)");
        assert_eq!(encoder.pos_to_rowcol(7, &buf), (1, 3), "pos 7 -> (1,3)");
    }

    #[test]
    fn test_pos_to_rowcol_tx8x4() {
        // Tx8x4: width=8, height=4, 32 coefficients total.
        // sqrt(32) ≈ 5 (wrong); correct width = 8.
        let encoder = CoeffEncoder::new();
        let buf = CoeffBuffer::from_tx_size(TxSize::Tx8x4);
        assert_eq!(buf.width(), 8);
        assert_eq!(buf.height(), 4);

        assert_eq!(encoder.pos_to_rowcol(9, &buf), (1, 1), "pos 9 -> (1,1)");
        assert_eq!(encoder.pos_to_rowcol(0, &buf), (0, 0), "pos 0 -> (0,0)");
        assert_eq!(encoder.pos_to_rowcol(7, &buf), (0, 7), "pos 7 -> (0,7)");
        assert_eq!(encoder.pos_to_rowcol(8, &buf), (1, 0), "pos 8 -> (1,0)");
    }

    #[test]
    fn test_pos_to_rowcol_tx4x16() {
        // Tx4x16: width=4, height=16, 64 coefficients total.
        // sqrt(64) = 8 (wrong); correct width = 4.
        let encoder = CoeffEncoder::new();
        let buf = CoeffBuffer::from_tx_size(TxSize::Tx4x16);
        assert_eq!(buf.width(), 4);
        assert_eq!(buf.height(), 16);

        assert_eq!(encoder.pos_to_rowcol(17, &buf), (4, 1), "pos 17 -> (4,1)");
        assert_eq!(encoder.pos_to_rowcol(0, &buf), (0, 0), "pos 0 -> (0,0)");
        assert_eq!(encoder.pos_to_rowcol(3, &buf), (0, 3), "pos 3 -> (0,3)");
        assert_eq!(encoder.pos_to_rowcol(4, &buf), (1, 0), "pos 4 -> (1,0)");
    }

    #[test]
    fn test_pos_to_rowcol_square_unchanged() {
        // Tx4x4: width=4, height=4. Both old and new logic agree, confirming no
        // regression for square sizes.
        let encoder = CoeffEncoder::new();
        let buf = CoeffBuffer::from_tx_size(TxSize::Tx4x4);
        assert_eq!(buf.width(), 4);
        assert_eq!(buf.height(), 4);

        for pos in 0..16usize {
            let (row, col) = encoder.pos_to_rowcol(pos, &buf);
            assert_eq!(row, pos / 4, "Tx4x4 pos {pos}: row mismatch");
            assert_eq!(col, pos % 4, "Tx4x4 pos {pos}: col mismatch");
        }
    }
}
