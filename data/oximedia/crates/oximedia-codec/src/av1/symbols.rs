//! AV1 symbol decoding from entropy-coded bitstream.
//!
//! This module provides high-level symbol decoding for AV1 syntax elements
//! using the entropy coding engine. It handles:
//!
//! - Partition decoding
//! - Mode decoding (intra/inter)
//! - Motion vector decoding
//! - Transform type and size selection
//! - Skip and skip_mode flags
//! - Segmentation and quantization selection
//!
//! # Symbol Decoding Flow
//!
//! 1. **Block-level symbols** - Partition, skip, segment_id
//! 2. **Mode symbols** - Intra/inter mode selection
//! 3. **Reference frame symbols** - For inter blocks
//! 4. **Motion vector symbols** - MV decoding
//! 5. **Transform symbols** - TX size and type
//! 6. **Coefficient symbols** - Via coeff_decode module
//!
//! # Context Modeling
//!
//! AV1 uses adaptive context-dependent probability models. Contexts
//! are computed from neighboring blocks and frame-level state.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::module_name_repetitions)]

use super::block::{BlockModeInfo, BlockSize, InterMode, IntraMode, PartitionType};
use super::entropy::{uniform_cdf, SymbolReader};
use super::entropy_tables::CdfContext;
use super::transform::{TxSize, TxType};
use crate::error::{CodecError, CodecResult};

// =============================================================================
// Constants
// =============================================================================

/// Maximum partition contexts.
pub const PARTITION_CONTEXTS: usize = 4;

/// Maximum skip contexts.
pub const SKIP_CONTEXTS: usize = 3;

/// Maximum intra mode contexts.
pub const INTRA_MODE_CONTEXTS: usize = 5;

/// Maximum inter mode contexts.
pub const INTER_MODE_CONTEXTS: usize = 7;

/// Maximum reference frame contexts.
pub const REF_CONTEXTS: usize = 3;

/// Maximum MV contexts.
pub const MV_CONTEXTS: usize = 2;

/// Maximum TX size contexts.
pub const TX_SIZE_CONTEXTS: usize = 4;

/// Maximum TX type contexts.
pub const TX_TYPE_CONTEXTS: usize = 4;

/// Number of reference frames.
pub const NUM_REF_FRAMES: usize = 7;

/// Maximum motion vector component.
pub const MAX_MV_COMPONENT: i16 = 1023;

// =============================================================================
// Symbol Decoder State
// =============================================================================

/// Symbol decoder for AV1 syntax elements.
#[derive(Debug)]
pub struct SymbolDecoder {
    /// Underlying symbol reader.
    reader: SymbolReader,
    /// CDF context for probability models.
    cdf_context: CdfContext,
    /// Current frame is intra-only.
    frame_is_intra: bool,
    /// Allow intraBC.
    allow_intrabc: bool,
    /// Current segment ID.
    segment_id: u8,
}

impl SymbolDecoder {
    /// Create a new symbol decoder.
    pub fn new(data: Vec<u8>, frame_is_intra: bool) -> Self {
        Self {
            reader: SymbolReader::new(data),
            cdf_context: CdfContext::new(),
            frame_is_intra,
            allow_intrabc: false,
            segment_id: 0,
        }
    }

    /// Read partition symbol.
    pub fn read_partition(&mut self, bsize: BlockSize, _ctx: u8) -> CodecResult<PartitionType> {
        if bsize == BlockSize::Block4x4 {
            // 4x4 blocks cannot be partitioned
            return Ok(PartitionType::None);
        }

        let mut cdf = uniform_cdf(10);
        let symbol = self.reader.read_symbol(&mut cdf);
        PartitionType::from_u8(symbol as u8)
            .ok_or_else(|| CodecError::InvalidBitstream("Invalid partition type".to_string()))
    }

    /// Read skip flag.
    pub fn read_skip(&mut self, _ctx: u8) -> bool {
        // Use uniform CDF for now
        let mut cdf = [16384u16, 32768, 0];
        self.reader.read_bool(&mut cdf)
    }

    /// Read skip_mode flag.
    pub fn read_skip_mode(&mut self, _ctx: u8) -> bool {
        if self.frame_is_intra {
            return false;
        }

        // Use local CDF array
        let mut cdf = [16384u16, 32768, 0];
        self.reader.read_bool(&mut cdf)
    }

    /// Read segment ID.
    pub fn read_segment_id(&mut self, _ctx: u8, max_segments: u8) -> u8 {
        if max_segments == 1 {
            return 0;
        }

        let mut cdf = uniform_cdf(max_segments as usize);
        let segment_id = self.reader.read_symbol(&mut cdf) as u8;
        self.segment_id = segment_id;
        segment_id
    }

    /// Read is_inter flag.
    pub fn read_is_inter(&mut self, _ctx: u8) -> bool {
        if self.frame_is_intra {
            return false;
        }

        let mut cdf = [16384u16, 32768, 0];
        self.reader.read_bool(&mut cdf)
    }

    /// Read intra mode for luma.
    pub fn read_intra_mode(&mut self, _ctx: u8, _bsize: BlockSize) -> CodecResult<IntraMode> {
        let mut cdf = uniform_cdf(13);
        let symbol = self.reader.read_symbol(&mut cdf);
        IntraMode::from_u8(symbol as u8)
            .ok_or_else(|| CodecError::InvalidBitstream("Invalid intra mode".to_string()))
    }

    /// Read intra mode for chroma (UV).
    pub fn read_uv_mode(&mut self, _y_mode: IntraMode, _ctx: u8) -> CodecResult<IntraMode> {
        let mut cdf = uniform_cdf(13);
        let symbol = self.reader.read_symbol(&mut cdf);
        IntraMode::from_u8(symbol as u8)
            .ok_or_else(|| CodecError::InvalidBitstream("Invalid UV mode".to_string()))
    }

    /// Read angle delta for directional intra modes.
    pub fn read_angle_delta(&mut self, mode: IntraMode) -> i8 {
        if !mode.is_directional() {
            return 0;
        }

        let mut cdf = uniform_cdf(7);
        let symbol = self.reader.read_symbol(&mut cdf);

        // Map symbol to delta: 0->-3, 1->-2, 2->-1, 3->0, 4->1, 5->2, 6->3
        (symbol as i8) - 3
    }

    /// Read palette mode flag.
    pub fn read_use_palette(&mut self, bsize: BlockSize, _ctx: u8) -> bool {
        if bsize == BlockSize::Block4x4
            || bsize == BlockSize::Block4x8
            || bsize == BlockSize::Block8x4
        {
            return false;
        }

        let mut cdf = [16384u16, 32768, 0];
        self.reader.read_bool(&mut cdf)
    }

    /// Read filter intra mode.
    pub fn read_filter_intra_mode(&mut self) -> u8 {
        let mut cdf = uniform_cdf(5);
        self.reader.read_symbol(&mut cdf) as u8
    }

    /// Read inter mode.
    pub fn read_inter_mode(&mut self, _ctx: u8) -> CodecResult<InterMode> {
        let mut cdf = uniform_cdf(4);
        let symbol = self.reader.read_symbol(&mut cdf);
        InterMode::from_u8(symbol as u8)
            .ok_or_else(|| CodecError::InvalidBitstream("Invalid inter mode".to_string()))
    }

    /// Read reference frame indices.
    pub fn read_ref_frames(&mut self, _ctx: u8) -> [i8; 2] {
        if self.frame_is_intra {
            return [-1, -1];
        }

        // Read compound mode flag
        let mut compound_cdf = [16384u16, 32768, 0];
        let is_compound = self.reader.read_bool(&mut compound_cdf);

        if is_compound {
            // Read two reference frames
            let ref0 = self.read_single_ref_frame(0);
            let ref1 = self.read_single_ref_frame(1);
            [ref0, ref1]
        } else {
            // Single reference
            let ref0 = self.read_single_ref_frame(0);
            [ref0, -1]
        }
    }

    /// Read a single reference frame index.
    fn read_single_ref_frame(&mut self, _idx: usize) -> i8 {
        let mut cdf = uniform_cdf(7);
        self.reader.read_symbol(&mut cdf) as i8
    }

    /// Read motion vector.
    pub fn read_mv(&mut self, ctx: u8) -> [i16; 2] {
        let row = self.read_mv_component(ctx, true);
        let col = self.read_mv_component(ctx, false);
        [row, col]
    }

    /// Read a single MV component.
    fn read_mv_component(&mut self, _ctx: u8, _is_row: bool) -> i16 {
        // Read sign
        let mut sign_cdf = [16384u16, 32768, 0];
        let sign = self.reader.read_bool(&mut sign_cdf);

        // Read class (magnitude range) - simplified to uniform
        let mut class_cdf = uniform_cdf(11);
        let class = self.reader.read_symbol(&mut class_cdf) as u8;

        // Read bits based on class
        let mag = self.read_mv_magnitude(class);

        if sign {
            -(mag as i16)
        } else {
            mag as i16
        }
    }

    /// Read MV magnitude bits.
    fn read_mv_magnitude(&mut self, class: u8) -> u16 {
        match class {
            0 => 0, // Class 0: magnitude 0
            1 => 1, // Class 1: magnitude 1
            _ => {
                // Classes 2-10: read additional bits
                let offset_bits = class - 2;
                let mut mag = 1u16 << (offset_bits + 1);

                for _ in 0..offset_bits {
                    let mut bit_cdf = [16384u16, 32768, 0];
                    let bit = self.reader.read_bool(&mut bit_cdf);
                    mag |= u16::from(bit);
                    mag <<= 1;
                }

                mag >> 1
            }
        }
    }

    /// Read transform size.
    pub fn read_tx_size(&mut self, bsize: BlockSize, _ctx: u8) -> TxSize {
        let max_tx_size = bsize.max_tx_size();

        // Use uniform CDF
        let mut cdf = uniform_cdf(5);
        let symbol = self.reader.read_symbol(&mut cdf);

        // Map symbol to TX size
        self.map_tx_size_symbol(symbol, max_tx_size)
    }

    /// Map TX size symbol to actual TX size.
    fn map_tx_size_symbol(&self, symbol: usize, max_tx_size: TxSize) -> TxSize {
        match symbol {
            0 => TxSize::Tx4x4,
            1 => TxSize::Tx8x8.min(max_tx_size),
            2 => TxSize::Tx16x16.min(max_tx_size),
            3 => TxSize::Tx32x32.min(max_tx_size),
            _ => max_tx_size,
        }
    }

    /// Read transform type.
    pub fn read_tx_type(&mut self, _tx_size: TxSize, _is_inter: bool, _ctx: u8) -> TxType {
        // Use uniform CDF
        let mut cdf = uniform_cdf(16);
        let symbol = self.reader.read_symbol(&mut cdf);
        TxType::from_u8(symbol as u8).unwrap_or(TxType::DctDct)
    }

    /// Read compound type for compound prediction.
    pub fn read_compound_type(&mut self, _ctx: u8) -> u8 {
        let mut cdf = uniform_cdf(3);
        self.reader.read_symbol(&mut cdf) as u8
    }

    /// Read interpolation filter.
    pub fn read_interp_filter(&mut self, _ctx: u8) -> u8 {
        let mut cdf = uniform_cdf(4);
        self.reader.read_symbol(&mut cdf) as u8
    }

    /// Read motion mode (simple, OBMC, warped).
    pub fn read_motion_mode(&mut self, bsize: BlockSize, _ctx: u8) -> u8 {
        if bsize == BlockSize::Block4x4
            || bsize == BlockSize::Block4x8
            || bsize == BlockSize::Block8x4
        {
            return 0; // Simple motion only for small blocks
        }

        let mut cdf = uniform_cdf(3);
        self.reader.read_symbol(&mut cdf) as u8
    }

    /// Decode complete block mode info.
    pub fn decode_block_mode(
        &mut self,
        bsize: BlockSize,
        ctx_skip: u8,
        ctx_mode: u8,
    ) -> CodecResult<BlockModeInfo> {
        let mut mode_info = BlockModeInfo::new();
        mode_info.block_size = bsize;

        // Read skip flag
        mode_info.skip = self.read_skip(ctx_skip);

        // Read segment ID (if segmentation is enabled)
        mode_info.segment_id = self.segment_id;

        // Read is_inter
        mode_info.is_inter = self.read_is_inter(ctx_mode);

        if mode_info.is_inter {
            // Inter block
            mode_info.inter_mode = self.read_inter_mode(ctx_mode)?;
            mode_info.ref_frames = self.read_ref_frames(ctx_mode);

            // Read motion vectors if needed
            if mode_info.inter_mode.has_newmv() {
                let mv = self.read_mv(ctx_mode);
                mode_info.mv[0] = mv;
            }

            // Read interpolation filter
            mode_info.interp_filter = [
                self.read_interp_filter(ctx_mode),
                self.read_interp_filter(ctx_mode),
            ];

            // Read motion mode
            mode_info.motion_mode = self.read_motion_mode(bsize, ctx_mode);

            // Compound prediction
            if mode_info.is_compound() {
                mode_info.compound_type = self.read_compound_type(ctx_mode);
            }
        } else {
            // Intra block
            mode_info.intra_mode = self.read_intra_mode(ctx_mode, bsize)?;
            mode_info.uv_mode = self.read_uv_mode(mode_info.intra_mode, ctx_mode)?;
            mode_info.angle_delta = [
                self.read_angle_delta(mode_info.intra_mode),
                self.read_angle_delta(mode_info.uv_mode),
            ];

            // Palette mode
            mode_info.use_palette = self.read_use_palette(bsize, ctx_mode);

            // Filter intra
            if bsize.width() <= 32 && bsize.height() <= 32 {
                mode_info.filter_intra_mode = self.read_filter_intra_mode();
            }
        }

        // Read transform size
        mode_info.tx_size = self.read_tx_size(bsize, ctx_mode);
        // TX type would be read separately based on TX size

        Ok(mode_info)
    }

    /// Check if more data is available.
    pub fn has_more_data(&self) -> bool {
        self.reader.has_more_data()
    }

    /// Get current position in bytes.
    pub fn position(&self) -> usize {
        self.reader.position()
    }

    /// Get remaining bytes.
    pub fn remaining(&self) -> usize {
        self.reader.remaining()
    }
}

// =============================================================================
// TxSize min/max helpers
// =============================================================================

impl TxSize {
    /// Get the minimum of two TX sizes.
    #[must_use]
    pub const fn min(self, other: Self) -> Self {
        let self_area = self.area();
        let other_area = other.area();
        if self_area <= other_area {
            self
        } else {
            other
        }
    }
}

// =============================================================================
// Context Computation Helpers
// =============================================================================

/// Compute partition context from neighbors.
#[must_use]
pub fn compute_partition_context(above: u8, left: u8, bsize: BlockSize) -> u8 {
    let bs = bsize.width_log2();
    let above_split = above < bs;
    let left_split = left < bs;

    match (above_split, left_split) {
        (false, false) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (true, true) => 3,
    }
}

/// Compute skip context from neighbors.
#[must_use]
pub fn compute_skip_context(above_skip: bool, left_skip: bool) -> u8 {
    match (above_skip, left_skip) {
        (false, false) => 0,
        (false, true) | (true, false) => 1,
        (true, true) => 2,
    }
}

/// Compute is_inter context from neighbors.
#[must_use]
pub fn compute_is_inter_context(above_inter: bool, left_inter: bool) -> u8 {
    match (above_inter, left_inter) {
        (false, false) => 0,
        (false, true) | (true, false) => 1,
        (true, true) => 2,
    }
}

/// Compute TX size context from neighbors.
#[must_use]
pub fn compute_tx_size_context(above_tx: TxSize, left_tx: TxSize, max_tx: TxSize) -> u8 {
    let above_cat = tx_size_category(above_tx, max_tx);
    let left_cat = tx_size_category(left_tx, max_tx);

    (above_cat + left_cat).min(3)
}

/// Categorize TX size relative to max.
fn tx_size_category(tx: TxSize, max_tx: TxSize) -> u8 {
    if tx == max_tx {
        0
    } else if tx.width() * 2 >= max_tx.width() && tx.height() * 2 >= max_tx.height() {
        1
    } else {
        2
    }
}

// =============================================================================
// MV Prediction and Context
// =============================================================================

/// Motion vector predictor.
#[derive(Clone, Copy, Debug, Default)]
pub struct MvPredictor {
    /// Candidate motion vectors.
    pub candidates: [[i16; 2]; 3],
    /// Number of valid candidates.
    pub count: usize,
}

impl MvPredictor {
    /// Create a new MV predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            candidates: [[0, 0]; 3],
            count: 0,
        }
    }

    /// Add a candidate MV.
    pub fn add_candidate(&mut self, mv: [i16; 2]) {
        if self.count < 3 {
            self.candidates[self.count] = mv;
            self.count += 1;
        }
    }

    /// Get the nearest MV.
    #[must_use]
    pub fn nearest(&self) -> [i16; 2] {
        if self.count > 0 {
            self.candidates[0]
        } else {
            [0, 0]
        }
    }

    /// Get the near MV.
    #[must_use]
    pub fn near(&self) -> [i16; 2] {
        if self.count > 1 {
            self.candidates[1]
        } else {
            self.nearest()
        }
    }

    /// Compute MV context.
    #[must_use]
    pub fn compute_context(&self) -> u8 {
        if self.count == 0 {
            return 0;
        }

        let mv0_mag = self.mv_magnitude(self.candidates[0]);
        let mv1_mag = if self.count > 1 {
            self.mv_magnitude(self.candidates[1])
        } else {
            0
        };

        if mv0_mag < 16 && mv1_mag < 16 {
            0
        } else {
            1
        }
    }

    /// Compute MV magnitude.
    fn mv_magnitude(&self, mv: [i16; 2]) -> u16 {
        (mv[0].abs() + mv[1].abs()) as u16
    }
}

// =============================================================================
// Symbol Encoding (for completeness)
// =============================================================================

/// Symbol encoder for AV1 syntax elements.
#[derive(Debug)]
pub struct SymbolEncoder {
    /// Underlying symbol writer.
    writer: super::entropy::SymbolWriter,
    /// CDF context.
    cdf_context: CdfContext,
}

impl SymbolEncoder {
    /// Create a new symbol encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            writer: super::entropy::SymbolWriter::new(),
            cdf_context: CdfContext::new(),
        }
    }

    /// Write partition symbol.
    pub fn write_partition(&mut self, partition: PartitionType, _ctx: u8) {
        let mut cdf = uniform_cdf(10);
        self.writer.write_symbol(partition as usize, &mut cdf);
    }

    /// Write skip flag.
    pub fn write_skip(&mut self, skip: bool, _ctx: u8) {
        // Use local CDF array
        let mut cdf = [16384u16, 32768, 0];
        self.writer.write_bool(skip, &mut cdf);
    }

    /// Finalize and get output.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.writer.finish()
    }
}

impl Default for SymbolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_decoder_creation() {
        let data = vec![0u8; 128];
        let decoder = SymbolDecoder::new(data, false);
        assert!(decoder.has_more_data());
    }

    #[test]
    fn test_partition_context() {
        assert_eq!(compute_partition_context(0, 0, BlockSize::Block16x16), 3);
        assert_eq!(compute_partition_context(4, 4, BlockSize::Block16x16), 0);
        assert_eq!(compute_partition_context(4, 3, BlockSize::Block16x16), 2);
    }

    #[test]
    fn test_skip_context() {
        assert_eq!(compute_skip_context(false, false), 0);
        assert_eq!(compute_skip_context(true, false), 1);
        assert_eq!(compute_skip_context(false, true), 1);
        assert_eq!(compute_skip_context(true, true), 2);
    }

    #[test]
    fn test_is_inter_context() {
        assert_eq!(compute_is_inter_context(false, false), 0);
        assert_eq!(compute_is_inter_context(true, false), 1);
        assert_eq!(compute_is_inter_context(false, true), 1);
        assert_eq!(compute_is_inter_context(true, true), 2);
    }

    #[test]
    fn test_tx_size_context() {
        let max_tx = TxSize::Tx16x16;
        let ctx = compute_tx_size_context(TxSize::Tx8x8, TxSize::Tx8x8, max_tx);
        assert!(ctx <= 3);
    }

    #[test]
    fn test_mv_predictor() {
        let mut pred = MvPredictor::new();
        assert_eq!(pred.count, 0);

        pred.add_candidate([10, 20]);
        assert_eq!(pred.count, 1);
        assert_eq!(pred.nearest(), [10, 20]);

        pred.add_candidate([5, 15]);
        assert_eq!(pred.count, 2);
        assert_eq!(pred.near(), [5, 15]);
    }

    #[test]
    fn test_mv_predictor_context() {
        let mut pred = MvPredictor::new();
        pred.add_candidate([5, 10]);
        pred.add_candidate([8, 12]);

        let ctx = pred.compute_context();
        assert!(ctx <= 1);
    }

    #[test]
    fn test_tx_size_min() {
        assert_eq!(TxSize::Tx8x8.min(TxSize::Tx16x16), TxSize::Tx8x8);
        assert_eq!(TxSize::Tx16x16.min(TxSize::Tx8x8), TxSize::Tx8x8);
        assert_eq!(TxSize::Tx4x4.min(TxSize::Tx4x4), TxSize::Tx4x4);
    }

    #[test]
    fn test_symbol_encoder() {
        let mut encoder = SymbolEncoder::new();
        encoder.write_skip(true, 0);
        encoder.write_partition(PartitionType::None, 0);
        let output = encoder.finish();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_mv_magnitude() {
        let pred = MvPredictor::new();
        assert_eq!(pred.mv_magnitude([0, 0]), 0);
        assert_eq!(pred.mv_magnitude([10, 20]), 30);
        assert_eq!(pred.mv_magnitude([-10, 20]), 30);
    }

    #[test]
    fn test_constants() {
        assert_eq!(PARTITION_CONTEXTS, 4);
        assert_eq!(SKIP_CONTEXTS, 3);
        assert_eq!(INTRA_MODE_CONTEXTS, 5);
        assert_eq!(INTER_MODE_CONTEXTS, 7);
        assert_eq!(NUM_REF_FRAMES, 7);
    }

    #[test]
    fn test_symbol_decoder_position() {
        let data = vec![0u8; 128];
        let decoder = SymbolDecoder::new(data, false);
        // Decoder init reads 15 bits (2 bytes) for arithmetic decoder state
        assert_eq!(decoder.remaining(), 126);
    }

    #[test]
    fn test_tx_size_category() {
        let max_tx = TxSize::Tx32x32;
        assert_eq!(tx_size_category(TxSize::Tx32x32, max_tx), 0);
        assert_eq!(tx_size_category(TxSize::Tx16x16, max_tx), 1);
        assert_eq!(tx_size_category(TxSize::Tx4x4, max_tx), 2);
    }
}
