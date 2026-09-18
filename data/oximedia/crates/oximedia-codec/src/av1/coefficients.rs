//! AV1 coefficient parsing and dequantization.
//!
//! This module handles the parsing of transform coefficients from the
//! entropy-coded bitstream, including:
//!
//! - End of block (EOB) position parsing
//! - Coefficient level and sign decoding
//! - Scan order for coefficient serialization
//! - Dequantization helpers
//!
//! # Coefficient Coding Structure
//!
//! AV1 uses a sophisticated multi-level coding scheme:
//!
//! 1. **EOB (End of Block)** - Position of last non-zero coefficient
//! 2. **Coefficient Base** - Base level (0-2) using multi-symbol coding
//! 3. **Coefficient Base Range** - Extended range using Golomb-Rice codes
//! 4. **DC Sign** - Sign of DC coefficient
//! 5. **AC Signs** - Signs of AC coefficients
//!
//! # Scan Orders
//!
//! Coefficients are scanned in a specific order based on:
//! - Transform class (2D, horizontal, vertical)
//! - Transform size
//!
//! # Reference
//!
//! See AV1 Specification Section 5.11.39 for coefficient syntax and
//! Section 7.12 for coefficient semantics.

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::needless_bool_assign)]
#![allow(clippy::if_not_else)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::cast_lossless)]

use super::transform::{TxClass, TxSize, TxType};
use crate::error::{CodecError, CodecResult};

// =============================================================================
// Constants
// =============================================================================

/// Maximum EOB position for any transform size.
pub const MAX_EOB: usize = 4096;

/// Number of EOB position contexts.
pub const EOB_COEF_CONTEXTS: usize = 9;

/// Number of coefficient base contexts.
pub const COEFF_BASE_CONTEXTS: usize = 42;

/// Number of coefficient base EOB contexts.
pub const COEFF_BASE_EOB_CONTEXTS: usize = 3;

/// Number of DC sign contexts.
pub const DC_SIGN_CONTEXTS: usize = 3;

/// Number of coefficient base range contexts.
pub const COEFF_BR_CONTEXTS: usize = 21;

/// Maximum coefficient base level.
pub const COEFF_BASE_RANGE_MAX: u32 = 3;

/// Golomb-Rice parameter for coefficient coding.
pub const COEFF_BR_RICE_PARAM: u8 = 1;

/// Base level cutoffs for coefficient coding.
pub const BASE_LEVEL_CUTOFFS: [u32; 5] = [0, 1, 2, 3, 4];

/// Number of TX classes.
pub const TX_CLASSES: usize = 3;

/// Coefficient context position limit.
pub const COEFF_CONTEXT_MASK: usize = 63;

/// Maximum neighbors for context computation.
pub const MAX_NEIGHBORS: usize = 2;

// =============================================================================
// EOB Position Tables
// =============================================================================

/// EOB offset for each transform size.
pub const EOB_OFFSET: [u16; 19] = [
    0,    // TX_4X4
    16,   // TX_8X8
    80,   // TX_16X16
    336,  // TX_32X32
    1360, // TX_64X64
    16,   // TX_4X8
    16,   // TX_8X4
    80,   // TX_8X16
    80,   // TX_16X8
    336,  // TX_16X32
    336,  // TX_32X16
    1360, // TX_32X64
    1360, // TX_64X32
    48,   // TX_4X16
    48,   // TX_16X4
    176,  // TX_8X32
    176,  // TX_32X8
    592,  // TX_16X64
    592,  // TX_64X16
];

/// EOB extra bits for each transform size.
pub const EOB_EXTRA_BITS: [u8; 19] = [
    0, // TX_4X4
    1, // TX_8X8
    2, // TX_16X16
    3, // TX_32X32
    4, // TX_64X64
    1, // TX_4X8
    1, // TX_8X4
    2, // TX_8X16
    2, // TX_16X8
    3, // TX_16X32
    3, // TX_32X16
    4, // TX_32X64
    4, // TX_64X32
    2, // TX_4X16
    2, // TX_16X4
    3, // TX_8X32
    3, // TX_32X8
    4, // TX_16X64
    4, // TX_64X16
];

/// EOB group start positions.
pub const EOB_GROUP_START: [u16; 12] = [0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513];

/// EOB symbol to position mapping.
pub const EOB_TO_POS: [u16; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

// =============================================================================
// Level Context
// =============================================================================

/// Context for coefficient level coding.
#[derive(Clone, Debug, Default)]
pub struct LevelContext {
    /// Accumulated magnitude of neighbors.
    pub mag: u32,
    /// Number of non-zero neighbors.
    pub count: u8,
    /// Position-based context.
    pub pos_ctx: u8,
}

impl LevelContext {
    /// Create a new level context.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mag: 0,
            count: 0,
            pos_ctx: 0,
        }
    }

    /// Compute context from magnitude.
    #[must_use]
    pub fn mag_context(&self) -> u8 {
        let mag = self.mag;
        if mag > 512 {
            4
        } else if mag > 256 {
            3
        } else if mag > 128 {
            2
        } else if mag > 64 {
            1
        } else {
            0
        }
    }

    /// Compute the combined context.
    #[must_use]
    pub fn context(&self) -> u8 {
        self.mag_context() * 3 + self.count.min(2)
    }
}

// =============================================================================
// Coefficient Context
// =============================================================================

/// Context for coefficient parsing state.
#[derive(Clone, Debug)]
pub struct CoeffContext {
    /// Transform size.
    pub tx_size: TxSize,
    /// Transform type.
    pub tx_type: TxType,
    /// Plane index (0=Y, 1=U, 2=V).
    pub plane: u8,
    /// Current scan position.
    pub scan_pos: u16,
    /// End of block position.
    pub eob: u16,
    /// Coefficient levels (dequantized).
    pub levels: Vec<i32>,
    /// Sign bits.
    pub signs: Vec<bool>,
    /// Accumulated left context.
    pub left_ctx: Vec<u8>,
    /// Accumulated above context.
    pub above_ctx: Vec<u8>,
    /// Block width in 4x4 units.
    pub block_width: u8,
    /// Block height in 4x4 units.
    pub block_height: u8,
}

impl CoeffContext {
    /// Create a new coefficient context.
    #[must_use]
    pub fn new(tx_size: TxSize, tx_type: TxType, plane: u8) -> Self {
        let area = tx_size.area() as usize;
        let width = (tx_size.width() / 4) as u8;
        let height = (tx_size.height() / 4) as u8;

        Self {
            tx_size,
            tx_type,
            plane,
            scan_pos: 0,
            eob: 0,
            levels: vec![0; area],
            signs: vec![false; area],
            left_ctx: vec![0; height as usize * 4],
            above_ctx: vec![0; width as usize * 4],
            block_width: width,
            block_height: height,
        }
    }

    /// Reset the context for a new block.
    pub fn reset(&mut self) {
        self.scan_pos = 0;
        self.eob = 0;
        self.levels.fill(0);
        self.signs.fill(false);
        self.left_ctx.fill(0);
        self.above_ctx.fill(0);
    }

    /// Get the transform class.
    #[must_use]
    pub fn tx_class(&self) -> TxClass {
        self.tx_type.tx_class()
    }

    /// Get scan position for a coefficient index.
    #[must_use]
    pub fn get_scan_position(&self, idx: usize) -> (u32, u32) {
        let width = self.tx_size.width();
        let row = (idx as u32) / width;
        let col = (idx as u32) % width;
        (row, col)
    }

    /// Get coefficient index from row and column.
    #[must_use]
    pub fn get_coeff_index(&self, row: u32, col: u32) -> usize {
        (row * self.tx_size.width() + col) as usize
    }

    /// Compute level context for a position.
    #[must_use]
    pub fn compute_level_context(&self, pos: usize) -> LevelContext {
        let width = self.tx_size.width() as usize;
        let _height = self.tx_size.height() as usize;
        let row = pos / width;
        let col = pos % width;

        let mut ctx = LevelContext::new();

        // Get neighbors (left and above)
        if col > 0 {
            let left = self.levels[row * width + col - 1].unsigned_abs();
            ctx.mag += left;
            if left > 0 {
                ctx.count += 1;
            }
        }

        if row > 0 {
            let above = self.levels[(row - 1) * width + col].unsigned_abs();
            ctx.mag += above;
            if above > 0 {
                ctx.count += 1;
            }
        }

        // Diagonal neighbor
        if row > 0 && col > 0 {
            let diag = self.levels[(row - 1) * width + col - 1].unsigned_abs();
            ctx.mag += diag;
        }

        // Position context
        ctx.pos_ctx = if row + col == 0 {
            0
        } else if row + col < 2 {
            1
        } else if row + col < 4 {
            2
        } else {
            3
        };

        ctx
    }

    /// Get DC sign context.
    #[must_use]
    pub fn dc_sign_context(&self) -> u8 {
        let left_sign = if !self.left_ctx.is_empty() {
            (self.left_ctx[0] as i8 - 1).signum()
        } else {
            0
        };

        let above_sign = if !self.above_ctx.is_empty() {
            (self.above_ctx[0] as i8 - 1).signum()
        } else {
            0
        };

        let sign_sum = left_sign + above_sign;

        if sign_sum < 0 {
            0
        } else if sign_sum > 0 {
            2
        } else {
            1
        }
    }

    /// Set coefficient value at position.
    pub fn set_coeff(&mut self, pos: usize, level: i32, sign: bool) {
        if pos < self.levels.len() {
            self.levels[pos] = if sign { -level } else { level };
            self.signs[pos] = sign;
        }
    }

    /// Get coefficient value at position.
    #[must_use]
    pub fn get_coeff(&self, pos: usize) -> i32 {
        self.levels.get(pos).copied().unwrap_or(0)
    }

    /// Check if block has any non-zero coefficients.
    #[must_use]
    pub fn has_nonzero(&self) -> bool {
        self.eob > 0
    }

    /// Get the number of non-zero coefficients.
    #[must_use]
    pub fn count_nonzero(&self) -> u16 {
        self.levels.iter().filter(|&&l| l != 0).count() as u16
    }
}

impl Default for CoeffContext {
    fn default() -> Self {
        Self::new(TxSize::Tx4x4, TxType::DctDct, 0)
    }
}

// =============================================================================
// Scan Order Generation
// =============================================================================

/// Generate diagonal scan order for a given size.
#[must_use]
pub fn generate_diagonal_scan(width: usize, height: usize) -> Vec<u16> {
    let mut scan = Vec::with_capacity(width * height);

    // Traverse diagonals
    for diag in 0..(width + height - 1) {
        // Start position for this diagonal
        let col_start = if diag < width { 0 } else { diag - width + 1 };
        let col_end = diag.min(height - 1);

        for offset in 0..=(col_end - col_start) {
            let row = col_start + offset;
            let col = diag - row;

            if col < width && row < height {
                scan.push((row * width + col) as u16);
            }
        }
    }

    scan
}

/// Generate horizontal scan order.
#[must_use]
pub fn generate_horizontal_scan(width: usize, height: usize) -> Vec<u16> {
    let mut scan = Vec::with_capacity(width * height);

    for row in 0..height {
        for col in 0..width {
            scan.push((row * width + col) as u16);
        }
    }

    scan
}

/// Generate vertical scan order.
#[must_use]
pub fn generate_vertical_scan(width: usize, height: usize) -> Vec<u16> {
    let mut scan = Vec::with_capacity(width * height);

    for col in 0..width {
        for row in 0..height {
            scan.push((row * width + col) as u16);
        }
    }

    scan
}

/// Get the scan order for a given transform.
#[must_use]
pub fn get_scan_order(tx_size: TxSize, tx_class: TxClass) -> Vec<u16> {
    let width = tx_size.width() as usize;
    let height = tx_size.height() as usize;

    match tx_class {
        TxClass::Class2D => generate_diagonal_scan(width, height),
        TxClass::ClassHoriz => generate_horizontal_scan(width, height),
        TxClass::ClassVert => generate_vertical_scan(width, height),
    }
}

/// Scan order cache for common transform sizes.
#[derive(Clone, Debug)]
pub struct ScanOrderCache {
    /// Cached scan orders indexed by [tx_size][tx_class].
    cache: Vec<Vec<Vec<u16>>>,
}

impl ScanOrderCache {
    /// Create a new scan order cache.
    #[must_use]
    pub fn new() -> Self {
        let mut cache = Vec::with_capacity(19);

        for tx_size_idx in 0..19 {
            let tx_size = TxSize::from_u8(tx_size_idx as u8).unwrap_or_default();
            let mut class_scans = Vec::with_capacity(3);

            for tx_class_idx in 0..3 {
                let tx_class = TxClass::from_u8(tx_class_idx as u8).unwrap_or_default();
                class_scans.push(get_scan_order(tx_size, tx_class));
            }

            cache.push(class_scans);
        }

        Self { cache }
    }

    /// Get scan order from cache.
    #[must_use]
    pub fn get(&self, tx_size: TxSize, tx_class: TxClass) -> &[u16] {
        let size_idx = tx_size as usize;
        let class_idx = tx_class as usize;

        if size_idx < self.cache.len() && class_idx < self.cache[size_idx].len() {
            &self.cache[size_idx][class_idx]
        } else {
            &[]
        }
    }
}

impl Default for ScanOrderCache {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// EOB Parsing Helpers
// =============================================================================

/// EOB (End of Block) position context.
#[derive(Clone, Debug, Default)]
pub struct EobContext {
    /// EOB multi-context.
    pub eob_multi: u8,
    /// EOB extra bits.
    pub eob_extra: u8,
    /// Base context.
    pub base_ctx: u8,
}

impl EobContext {
    /// Create EOB context for a transform size.
    #[must_use]
    pub fn new(tx_size: TxSize) -> Self {
        let size_idx = tx_size as usize;
        let extra_bits = if size_idx < EOB_EXTRA_BITS.len() {
            EOB_EXTRA_BITS[size_idx]
        } else {
            0
        };

        Self {
            eob_multi: 0,
            eob_extra: extra_bits,
            base_ctx: 0,
        }
    }

    /// Get the EOB context from position.
    #[must_use]
    pub fn get_eob_context(eob: u16) -> u8 {
        if eob <= 1 {
            0
        } else if eob <= 2 {
            1
        } else if eob <= 4 {
            2
        } else if eob <= 8 {
            3
        } else if eob <= 16 {
            4
        } else if eob <= 32 {
            5
        } else if eob <= 64 {
            6
        } else if eob <= 128 {
            7
        } else {
            8
        }
    }

    /// Compute EOB from multi-symbol and extra bits.
    #[must_use]
    pub fn compute_eob(eob_multi: u8, eob_extra: u16) -> u16 {
        let group_idx = eob_multi as usize;
        if group_idx >= EOB_GROUP_START.len() {
            return 0;
        }

        let base = EOB_GROUP_START[group_idx];
        base + eob_extra
    }
}

/// EOB point parsing state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EobPt {
    /// No coefficients.
    EobPt0 = 0,
    /// 1 coefficient.
    EobPt1 = 1,
    /// 2 coefficients.
    EobPt2 = 2,
    /// 3-4 coefficients.
    EobPt3To4 = 3,
    /// 5-8 coefficients.
    EobPt5To8 = 4,
    /// 9-16 coefficients.
    EobPt9To16 = 5,
    /// 17-32 coefficients.
    EobPt17To32 = 6,
    /// 33-64 coefficients.
    EobPt33To64 = 7,
    /// 65-128 coefficients.
    EobPt65To128 = 8,
    /// 129-256 coefficients.
    EobPt129To256 = 9,
    /// 257-512 coefficients.
    EobPt257To512 = 10,
    /// 513-1024 coefficients.
    EobPt513To1024 = 11,
}

impl EobPt {
    /// Get the EOB point from an EOB *position* (the actual index of the last
    /// non-zero coefficient in scan order, i.e. the count of decoded
    /// coefficients).
    ///
    /// Use this when you already have the final EOB position — for example in
    /// the encoder, after locating the last non-zero entry in a transform
    /// block.
    ///
    /// For the decoder path, where the bitstream carries the EOB-multi *symbol*
    /// (the index into the EOB-multi CDF, range 0..11), use
    /// [`EobPt::from_symbol`] instead. The two values are not interchangeable:
    /// symbol `s` corresponds to base position
    /// `EOB_GROUP_START[s]`, which is `(1 << (s - 1)) + 1` for `s ≥ 2`.
    #[must_use]
    pub fn from_eob(eob: u16) -> Self {
        match eob {
            0 => Self::EobPt0,
            1 => Self::EobPt1,
            2 => Self::EobPt2,
            3..=4 => Self::EobPt3To4,
            5..=8 => Self::EobPt5To8,
            9..=16 => Self::EobPt9To16,
            17..=32 => Self::EobPt17To32,
            33..=64 => Self::EobPt33To64,
            65..=128 => Self::EobPt65To128,
            129..=256 => Self::EobPt129To256,
            257..=512 => Self::EobPt257To512,
            _ => Self::EobPt513To1024,
        }
    }

    /// Build an [`EobPt`] from the EOB-multi *symbol index* read from the
    /// bitstream (range `0..=11`).
    ///
    /// The EOB-multi CDF in AV1 emits a symbol that directly identifies one of
    /// the 12 EOB point classes. Each enum variant's discriminant equals the
    /// symbol value it represents, so this is essentially the inverse of
    /// [`EobPt::base_eob`] / [`EobPt::extra_bits`].
    ///
    /// Returns [`CodecError::InvalidBitstream`] if `symbol > 11`. The largest
    /// EOB-multi CDF in the AV1 spec carries 16 symbols, so a malformed
    /// bitstream can in principle place symbols 12..15 here — those values are
    /// rejected rather than silently saturated, because they desynchronize the
    /// EOB extra-bits read that follows.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidBitstream`] if `symbol >= 12`.
    pub fn from_symbol(symbol: usize) -> CodecResult<Self> {
        match symbol {
            0 => Ok(Self::EobPt0),
            1 => Ok(Self::EobPt1),
            2 => Ok(Self::EobPt2),
            3 => Ok(Self::EobPt3To4),
            4 => Ok(Self::EobPt5To8),
            5 => Ok(Self::EobPt9To16),
            6 => Ok(Self::EobPt17To32),
            7 => Ok(Self::EobPt33To64),
            8 => Ok(Self::EobPt65To128),
            9 => Ok(Self::EobPt129To256),
            10 => Ok(Self::EobPt257To512),
            11 => Ok(Self::EobPt513To1024),
            other => Err(CodecError::InvalidBitstream(format!(
                "EOB-multi symbol {other} out of range 0..=11"
            ))),
        }
    }

    /// Get the base EOB for this point.
    #[must_use]
    pub const fn base_eob(self) -> u16 {
        match self {
            Self::EobPt0 => 0,
            Self::EobPt1 => 1,
            Self::EobPt2 => 2,
            Self::EobPt3To4 => 3,
            Self::EobPt5To8 => 5,
            Self::EobPt9To16 => 9,
            Self::EobPt17To32 => 17,
            Self::EobPt33To64 => 33,
            Self::EobPt65To128 => 65,
            Self::EobPt129To256 => 129,
            Self::EobPt257To512 => 257,
            Self::EobPt513To1024 => 513,
        }
    }

    /// Get the number of extra bits for this point.
    #[must_use]
    pub const fn extra_bits(self) -> u8 {
        match self {
            Self::EobPt0 | Self::EobPt1 | Self::EobPt2 => 0,
            Self::EobPt3To4 => 1,
            Self::EobPt5To8 => 2,
            Self::EobPt9To16 => 3,
            Self::EobPt17To32 => 4,
            Self::EobPt33To64 => 5,
            Self::EobPt65To128 => 6,
            Self::EobPt129To256 => 7,
            Self::EobPt257To512 => 8,
            Self::EobPt513To1024 => 9,
        }
    }
}

// =============================================================================
// Coefficient Base Range
// =============================================================================

/// Coefficient base range context.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoeffBaseRange {
    /// Base level (0-4).
    pub base_level: u8,
    /// Range context.
    pub range_ctx: u8,
}

impl CoeffBaseRange {
    /// Get context for coefficient base range coding.
    #[must_use]
    pub fn get_br_context(level_ctx: &LevelContext, pos: usize, width: usize) -> u8 {
        let row = pos / width;
        let col = pos % width;

        // Base context from position
        let pos_ctx = if row + col == 0 {
            0
        } else if row + col < 2 {
            7
        } else {
            14
        };

        // Combine with magnitude context
        pos_ctx + level_ctx.mag_context().min(6)
    }

    /// Compute level from base and range.
    #[must_use]
    pub fn compute_level(base: u8, range: u16) -> u32 {
        u32::from(base) + u32::from(range)
    }
}

// =============================================================================
// Dequantization Helpers
// =============================================================================

/// Dequantize a single coefficient.
#[must_use]
pub fn dequantize_coeff(level: i32, dequant: i16, shift: u8) -> i32 {
    let abs_level = level.abs();
    let dq_level = (abs_level * i32::from(dequant)) >> shift;

    if level < 0 {
        -dq_level
    } else {
        dq_level
    }
}

/// Dequantize all coefficients in a block.
pub fn dequantize_block(coeffs: &mut [i32], dc_dequant: i16, ac_dequant: i16, shift: u8) {
    if coeffs.is_empty() {
        return;
    }

    // DC coefficient
    coeffs[0] = dequantize_coeff(coeffs[0], dc_dequant, shift);

    // AC coefficients
    for coeff in coeffs.iter_mut().skip(1) {
        *coeff = dequantize_coeff(*coeff, ac_dequant, shift);
    }
}

/// Compute dequantization shift for a given bit depth.
#[must_use]
pub const fn get_dequant_shift(bit_depth: u8) -> u8 {
    match bit_depth {
        8 => 0,
        10 => 2,
        12 => 4,
        _ => 0,
    }
}

// =============================================================================
// Coefficient Buffer
// =============================================================================

/// Buffer for storing and manipulating coefficient data.
#[derive(Clone, Debug)]
pub struct CoeffBuffer {
    /// Coefficient storage.
    coeffs: Vec<i32>,
    /// Width of the buffer.
    width: usize,
    /// Height of the buffer.
    height: usize,
}

impl CoeffBuffer {
    /// Create a new coefficient buffer.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            coeffs: vec![0; width * height],
            width,
            height,
        }
    }

    /// Create from transform size.
    #[must_use]
    pub fn from_tx_size(tx_size: TxSize) -> Self {
        Self::new(tx_size.width() as usize, tx_size.height() as usize)
    }

    /// Get coefficient at position.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> i32 {
        if row < self.height && col < self.width {
            self.coeffs[row * self.width + col]
        } else {
            0
        }
    }

    /// Set coefficient at position.
    pub fn set(&mut self, row: usize, col: usize, value: i32) {
        if row < self.height && col < self.width {
            self.coeffs[row * self.width + col] = value;
        }
    }

    /// Clear all coefficients.
    pub fn clear(&mut self) {
        self.coeffs.fill(0);
    }

    /// Get the width (number of columns) of the buffer.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Get the height (number of rows) of the buffer.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Get mutable slice of coefficients.
    pub fn as_mut_slice(&mut self) -> &mut [i32] {
        &mut self.coeffs
    }

    /// Get immutable slice of coefficients.
    #[must_use]
    pub fn as_slice(&self) -> &[i32] {
        &self.coeffs
    }

    /// Copy from scan order.
    pub fn copy_from_scan(&mut self, src: &[i32], scan: &[u16]) {
        for (i, &pos) in scan.iter().enumerate() {
            if i < src.len() && (pos as usize) < self.coeffs.len() {
                self.coeffs[pos as usize] = src[i];
            }
        }
    }

    /// Copy to scan order.
    pub fn copy_to_scan(&self, dst: &mut [i32], scan: &[u16]) {
        for (i, &pos) in scan.iter().enumerate() {
            if i < dst.len() && (pos as usize) < self.coeffs.len() {
                dst[i] = self.coeffs[pos as usize];
            }
        }
    }
}

impl Default for CoeffBuffer {
    fn default() -> Self {
        Self::new(4, 4)
    }
}

// =============================================================================
// Neighbor Context Computation
// =============================================================================

/// Get neighbor positions for context computation.
#[must_use]
pub fn get_neighbor_positions(pos: usize, width: usize, _height: usize) -> [(usize, bool); 5] {
    let row = pos / width;
    let col = pos % width;

    let mut neighbors = [(0usize, false); 5];

    // Left neighbor
    if col > 0 {
        neighbors[0] = (row * width + col - 1, true);
    }

    // Above neighbor
    if row > 0 {
        neighbors[1] = ((row - 1) * width + col, true);
    }

    // Top-left diagonal
    if row > 0 && col > 0 {
        neighbors[2] = ((row - 1) * width + col - 1, true);
    }

    // Top-right diagonal
    if row > 0 && col + 1 < width {
        neighbors[3] = ((row - 1) * width + col + 1, true);
    }

    // Two positions left
    if col > 1 {
        neighbors[4] = (row * width + col - 2, true);
    }

    neighbors
}

/// Compute context from neighbor levels.
#[must_use]
pub fn compute_context_from_neighbors(levels: &[i32], neighbors: &[(usize, bool); 5]) -> u8 {
    let mut mag = 0u32;
    let mut count = 0u8;

    for &(pos, valid) in neighbors.iter() {
        if valid && pos < levels.len() {
            let level = levels[pos].unsigned_abs();
            mag += level;
            if level > 0 {
                count += 1;
            }
        }
    }

    // Context based on magnitude and count
    let mag_ctx = if mag > 512 {
        4
    } else if mag > 256 {
        3
    } else if mag > 128 {
        2
    } else if mag > 64 {
        1
    } else {
        0
    };

    mag_ctx * 3 + count.min(2)
}

// =============================================================================
// Sign Coding Helpers
// =============================================================================

/// DC sign context computation.
#[must_use]
pub fn compute_dc_sign_context(left_dc: i32, above_dc: i32) -> u8 {
    let left_sign = left_dc.signum();
    let above_sign = above_dc.signum();

    let sum = left_sign + above_sign;

    if sum < 0 {
        0
    } else if sum > 0 {
        2
    } else {
        1
    }
}

/// Update context after coefficient is decoded.
pub fn update_level_context(
    left_ctx: &mut [u8],
    above_ctx: &mut [u8],
    level: i32,
    row: usize,
    col: usize,
) {
    let level_ctx = (level.unsigned_abs().min(63) as u8) + 1;

    if row < left_ctx.len() {
        left_ctx[row] = level_ctx;
    }

    if col < above_ctx.len() {
        above_ctx[col] = level_ctx;
    }
}

// =============================================================================
// Coefficient Statistics
// =============================================================================

/// Statistics about coefficients in a block.
#[derive(Clone, Debug, Default)]
pub struct CoeffStats {
    /// Number of zero coefficients.
    pub zero_count: u32,
    /// Number of coefficients with level 1.
    pub level1_count: u32,
    /// Number of coefficients with level 2.
    pub level2_count: u32,
    /// Number of coefficients with level > 2.
    pub high_level_count: u32,
    /// Sum of absolute levels.
    pub level_sum: u64,
    /// Maximum absolute level.
    pub max_level: u32,
}

impl CoeffStats {
    /// Compute statistics from coefficient buffer.
    #[must_use]
    pub fn from_coeffs(coeffs: &[i32]) -> Self {
        let mut stats = Self::default();

        for &coeff in coeffs {
            let level = coeff.unsigned_abs();

            match level {
                0 => stats.zero_count += 1,
                1 => stats.level1_count += 1,
                2 => stats.level2_count += 1,
                _ => stats.high_level_count += 1,
            }

            stats.level_sum += u64::from(level);
            stats.max_level = stats.max_level.max(level);
        }

        stats
    }

    /// Get total non-zero count.
    #[must_use]
    pub fn nonzero_count(&self) -> u32 {
        self.level1_count + self.level2_count + self.high_level_count
    }

    /// Get average level (for non-zero coefficients).
    #[must_use]
    pub fn average_level(&self) -> f64 {
        let count = self.nonzero_count();
        if count > 0 {
            self.level_sum as f64 / count as f64
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

    #[test]
    fn test_level_context() {
        let mut ctx = LevelContext::new();
        assert_eq!(ctx.mag, 0);
        assert_eq!(ctx.count, 0);

        ctx.mag = 100;
        ctx.count = 2;
        assert_eq!(ctx.mag_context(), 1);
        assert_eq!(ctx.context(), 1 * 3 + 2);
    }

    #[test]
    fn test_coeff_context_new() {
        let ctx = CoeffContext::new(TxSize::Tx8x8, TxType::DctDct, 0);
        assert_eq!(ctx.levels.len(), 64);
        assert_eq!(ctx.tx_class(), TxClass::Class2D);
    }

    #[test]
    fn test_coeff_context_set_get() {
        let mut ctx = CoeffContext::new(TxSize::Tx4x4, TxType::DctDct, 0);
        ctx.set_coeff(5, 100, false);
        assert_eq!(ctx.get_coeff(5), 100);

        ctx.set_coeff(10, 50, true);
        assert_eq!(ctx.get_coeff(10), -50);
    }

    #[test]
    fn test_diagonal_scan_4x4() {
        let scan = generate_diagonal_scan(4, 4);
        assert_eq!(scan.len(), 16);
        // First few elements should be diagonal
        assert_eq!(scan[0], 0); // (0,0)
        assert_eq!(scan[1], 1); // (0,1) - in row-major for 4x4
    }

    #[test]
    fn test_horizontal_scan() {
        let scan = generate_horizontal_scan(4, 4);
        assert_eq!(scan.len(), 16);
        for i in 0..16 {
            assert_eq!(scan[i], i as u16);
        }
    }

    #[test]
    fn test_vertical_scan() {
        let scan = generate_vertical_scan(4, 4);
        assert_eq!(scan.len(), 16);
        assert_eq!(scan[0], 0);
        assert_eq!(scan[1], 4);
        assert_eq!(scan[2], 8);
        assert_eq!(scan[3], 12);
    }

    #[test]
    fn test_scan_order_cache() {
        let cache = ScanOrderCache::new();
        let scan = cache.get(TxSize::Tx4x4, TxClass::Class2D);
        assert_eq!(scan.len(), 16);
    }

    #[test]
    fn test_eob_context() {
        let ctx = EobContext::new(TxSize::Tx8x8);
        assert!(ctx.eob_extra > 0);
    }

    #[test]
    fn test_eob_pt() {
        assert_eq!(EobPt::from_eob(0), EobPt::EobPt0);
        assert_eq!(EobPt::from_eob(1), EobPt::EobPt1);
        assert_eq!(EobPt::from_eob(5), EobPt::EobPt5To8);
        assert_eq!(EobPt::from_eob(100), EobPt::EobPt65To128);

        assert_eq!(EobPt::EobPt5To8.extra_bits(), 2);
        assert_eq!(EobPt::EobPt5To8.base_eob(), 5);
    }

    #[test]
    fn test_eob_pt_from_symbol_basic() {
        // Regression: the EOB-multi CDF emits a *symbol* (0..=11) that
        // identifies the EobPt class directly. Each variant's discriminant
        // equals its symbol value.
        let expected = [
            EobPt::EobPt0,
            EobPt::EobPt1,
            EobPt::EobPt2,
            EobPt::EobPt3To4,
            EobPt::EobPt5To8,
            EobPt::EobPt9To16,
            EobPt::EobPt17To32,
            EobPt::EobPt33To64,
            EobPt::EobPt65To128,
            EobPt::EobPt129To256,
            EobPt::EobPt257To512,
            EobPt::EobPt513To1024,
        ];
        for (symbol, &want) in expected.iter().enumerate() {
            let got = EobPt::from_symbol(symbol).expect("symbol in 0..=11 must succeed");
            assert_eq!(got, want, "symbol {symbol} should map to {want:?}");
            assert_eq!(
                got as usize, symbol,
                "discriminant of {got:?} should equal symbol {symbol}"
            );
        }
    }

    #[test]
    fn test_eob_pt_from_symbol_rejects_out_of_range() {
        // The 16-symbol EOB-multi CDF can in principle yield symbols 12..15
        // from a malformed bitstream. Those must produce a decoding error
        // rather than silently saturate.
        for bad in 12..=15 {
            let err = EobPt::from_symbol(bad).expect_err("symbols 12..=15 must be rejected");
            let msg = format!("{err}");
            assert!(
                msg.contains("EOB-multi"),
                "error message should mention EOB-multi: {msg}"
            );
        }
        assert!(EobPt::from_symbol(usize::MAX).is_err());
    }

    #[test]
    fn test_eob_pt_from_symbol_differs_from_from_eob() {
        // This is the heart of the bug: the old decoder confused these two.
        // For symbols >= 4 the two functions return different EobPt classes,
        // and crucially different `extra_bits` counts, so a decoder that
        // misuses `from_eob` on a symbol desynchronizes the bitstream.
        //
        // Symbols 0..=3 happen to coincide (EobPt0, EobPt1, EobPt2,
        // EobPt3To4) because for those values the position table and the
        // symbol enumeration agree.
        for symbol in 0..=3usize {
            let from_sym = EobPt::from_symbol(symbol).expect("low symbol always succeeds");
            let from_pos = EobPt::from_eob(symbol as u16);
            assert_eq!(from_sym, from_pos, "symbols 0..=3 coincidentally match");
        }
        for symbol in 4..=11usize {
            let from_sym = EobPt::from_symbol(symbol).expect("symbols 4..=11 must succeed");
            let from_pos = EobPt::from_eob(symbol as u16);
            assert_ne!(
                from_sym, from_pos,
                "symbol {symbol} must NOT coincide with from_eob (the bug)",
            );
            // And the extra_bits read from the bitstream would have been
            // wrong, which is the actual harm.
            assert_ne!(
                from_sym.extra_bits(),
                from_pos.extra_bits(),
                "symbol {symbol} extra_bits mismatch is the desync source",
            );
        }
    }

    #[test]
    fn test_eob_pt_symbol_chain_covers_group_start() {
        // For every legal symbol, the decoder consumes `extra_bits` literal
        // bits and combines them with EOB_GROUP_START[symbol] to obtain the
        // final EOB position. Verify that this combination spans precisely
        // the range advertised by the EobPt variant.
        for symbol in 0..=11usize {
            let pt = EobPt::from_symbol(symbol).expect("legal symbol");
            assert_eq!(pt.base_eob(), EOB_GROUP_START[symbol]);

            let extra = pt.extra_bits();
            // Range size = 2^extra; symbol 0..=2 are degenerate (size 1).
            let range_size: u16 = 1u16 << extra;
            if symbol <= 2 {
                assert_eq!(range_size, 1);
            } else {
                // Final position fits in [base, base + range_size - 1].
                let max_offset = range_size - 1;
                let max_pos = EobPt::from_eob(pt.base_eob() + max_offset);
                assert_eq!(
                    max_pos, pt,
                    "EOB position at top of group must classify as same EobPt",
                );
            }
        }
    }

    #[test]
    fn test_dequantize_coeff() {
        let level = 10;
        let dequant = 16;
        let result = dequantize_coeff(level, dequant, 0);
        assert_eq!(result, 160);

        let neg_result = dequantize_coeff(-level, dequant, 0);
        assert_eq!(neg_result, -160);
    }

    #[test]
    fn test_dequantize_block() {
        let mut coeffs = vec![10, 5, 5, 5, 5, 5, 5, 5];
        dequantize_block(&mut coeffs, 20, 10, 0);

        assert_eq!(coeffs[0], 200); // DC: 10 * 20
        assert_eq!(coeffs[1], 50); // AC: 5 * 10
    }

    #[test]
    fn test_get_dequant_shift() {
        assert_eq!(get_dequant_shift(8), 0);
        assert_eq!(get_dequant_shift(10), 2);
        assert_eq!(get_dequant_shift(12), 4);
    }

    #[test]
    fn test_coeff_buffer() {
        let mut buf = CoeffBuffer::new(4, 4);
        buf.set(1, 2, 100);
        assert_eq!(buf.get(1, 2), 100);
        assert_eq!(buf.get(0, 0), 0);

        buf.clear();
        assert_eq!(buf.get(1, 2), 0);
    }

    #[test]
    fn test_coeff_buffer_from_tx_size() {
        let buf = CoeffBuffer::from_tx_size(TxSize::Tx8x8);
        assert_eq!(buf.as_slice().len(), 64);
    }

    #[test]
    fn test_neighbor_positions() {
        let neighbors = get_neighbor_positions(5, 4, 4);

        // Position 5 is row=1, col=1 in 4x4
        // Left neighbor should be valid at position 4
        assert!(neighbors[0].1);
        assert_eq!(neighbors[0].0, 4);

        // Above neighbor should be valid at position 1
        assert!(neighbors[1].1);
        assert_eq!(neighbors[1].0, 1);
    }

    #[test]
    fn test_compute_dc_sign_context() {
        assert_eq!(compute_dc_sign_context(-5, -3), 0); // Both negative
        assert_eq!(compute_dc_sign_context(5, 3), 2); // Both positive
        assert_eq!(compute_dc_sign_context(-5, 3), 1); // Mixed
        assert_eq!(compute_dc_sign_context(0, 0), 1); // Zero
    }

    #[test]
    fn test_coeff_stats() {
        let coeffs = vec![0, 1, 2, 3, 0, 1, 5, 0];
        let stats = CoeffStats::from_coeffs(&coeffs);

        assert_eq!(stats.zero_count, 3);
        assert_eq!(stats.level1_count, 2);
        assert_eq!(stats.level2_count, 1);
        assert_eq!(stats.high_level_count, 2);
        assert_eq!(stats.max_level, 5);
        assert_eq!(stats.nonzero_count(), 5);
    }

    #[test]
    fn test_coeff_context_dc_sign() {
        let ctx = CoeffContext::new(TxSize::Tx4x4, TxType::DctDct, 0);
        // Default context with empty context arrays
        // Results in neutral sign context (0 or 1 depending on implementation)
        let dc_ctx = ctx.dc_sign_context();
        // Context should be valid (0, 1, or 2)
        assert!(dc_ctx <= 2);
    }

    #[test]
    fn test_coeff_context_level_context() {
        let mut ctx = CoeffContext::new(TxSize::Tx4x4, TxType::DctDct, 0);
        ctx.levels[0] = 5;
        ctx.levels[1] = 3;
        ctx.levels[4] = 2;

        let level_ctx = ctx.compute_level_context(5);
        // Position 5 has neighbors at 4 (left) and 1 (above)
        assert!(level_ctx.mag > 0);
    }

    #[test]
    fn test_eob_compute() {
        // Test EOB computation from multi-symbol and extra bits
        assert_eq!(EobContext::compute_eob(0, 0), 0);
        assert_eq!(EobContext::compute_eob(1, 0), 1);
        assert_eq!(EobContext::compute_eob(2, 0), 2);
    }

    #[test]
    fn test_coeff_base_range() {
        let level_ctx = LevelContext {
            mag: 100,
            count: 2,
            pos_ctx: 1,
        };

        let br_ctx = CoeffBaseRange::get_br_context(&level_ctx, 5, 4);
        assert!(br_ctx > 0);

        let level = CoeffBaseRange::compute_level(2, 5);
        assert_eq!(level, 7);
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_EOB, 4096);
        assert_eq!(EOB_COEF_CONTEXTS, 9);
        assert_eq!(TX_CLASSES, 3);
    }

    #[test]
    fn test_coeff_context_reset() {
        let mut ctx = CoeffContext::new(TxSize::Tx4x4, TxType::DctDct, 0);
        ctx.eob = 10;
        ctx.levels[5] = 100;

        ctx.reset();
        assert_eq!(ctx.eob, 0);
        assert_eq!(ctx.levels[5], 0);
    }

    #[test]
    fn test_coeff_context_count_nonzero() {
        let mut ctx = CoeffContext::new(TxSize::Tx4x4, TxType::DctDct, 0);
        ctx.levels[0] = 5;
        ctx.levels[5] = 3;
        ctx.levels[10] = -2;

        assert_eq!(ctx.count_nonzero(), 3);
    }

    #[test]
    fn test_scan_order_all_sizes() {
        // Test that scan order generation works for all sizes
        for size_idx in 0..19 {
            if let Some(tx_size) = TxSize::from_u8(size_idx) {
                for class_idx in 0..3 {
                    if let Some(tx_class) = TxClass::from_u8(class_idx) {
                        let scan = get_scan_order(tx_size, tx_class);
                        assert_eq!(scan.len(), tx_size.area() as usize);
                    }
                }
            }
        }
    }
}
