//! AV1 block-level data structures and parsing.
//!
//! This module contains the data structures for representing block-level
//! information in AV1, including:
//!
//! - Block sizes from 4x4 to 128x128
//! - Block partition types
//! - Block mode information (intra/inter modes)
//! - Per-plane block context
//!
//! # Block Sizes
//!
//! AV1 supports a wide range of block sizes from 4x4 to 128x128, with
//! various rectangular shapes. The superblock size can be either 64x64
//! or 128x128.
//!
//! # Partitions
//!
//! Blocks can be partitioned recursively into smaller blocks using
//! 10 different partition types including splits, horizontal/vertical
//! splits, and various T-shaped partitions.
//!
//! # Reference
//!
//! See AV1 Specification Section 5.9 for block syntax and Section 6.4
//! for block semantics.

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::similar_names)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::match_same_arms)]

use super::transform::TxSize;

/// Const-compatible min function for u32.
const fn const_min_u32(a: u32, b: u32) -> u32 {
    if a < b {
        a
    } else {
        b
    }
}

// =============================================================================
// Constants
// =============================================================================

/// Maximum number of planes (Y, U, V).
pub const MAX_PLANES: usize = 3;

/// Maximum number of segments.
pub const MAX_SEGMENTS: usize = 8;

/// Maximum superblock size (128x128).
pub const MAX_SB_SIZE: usize = 128;

/// Maximum superblock size in 4x4 units.
pub const MAX_SB_SQUARE: usize = MAX_SB_SIZE / 4;

/// Minimum block size (4x4).
pub const MIN_BLOCK_SIZE: usize = 4;

/// Number of block sizes.
pub const BLOCK_SIZES: usize = 22;

/// Number of partition types.
pub const PARTITION_TYPES: usize = 10;

/// Number of intra modes.
pub const INTRA_MODES: usize = 13;

/// Number of inter modes.
pub const INTER_MODES: usize = 4;

/// Number of reference frames.
pub const REF_FRAMES: usize = 8;

// =============================================================================
// Block Size Enum
// =============================================================================

/// Block size enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlockSize {
    /// 4x4 block.
    #[default]
    Block4x4 = 0,
    /// 4x8 block.
    Block4x8 = 1,
    /// 8x4 block.
    Block8x4 = 2,
    /// 8x8 block.
    Block8x8 = 3,
    /// 8x16 block.
    Block8x16 = 4,
    /// 16x8 block.
    Block16x8 = 5,
    /// 16x16 block.
    Block16x16 = 6,
    /// 16x32 block.
    Block16x32 = 7,
    /// 32x16 block.
    Block32x16 = 8,
    /// 32x32 block.
    Block32x32 = 9,
    /// 32x64 block.
    Block32x64 = 10,
    /// 64x32 block.
    Block64x32 = 11,
    /// 64x64 block.
    Block64x64 = 12,
    /// 64x128 block.
    Block64x128 = 13,
    /// 128x64 block.
    Block128x64 = 14,
    /// 128x128 block.
    Block128x128 = 15,
    /// 4x16 block.
    Block4x16 = 16,
    /// 16x4 block.
    Block16x4 = 17,
    /// 8x32 block.
    Block8x32 = 18,
    /// 32x8 block.
    Block32x8 = 19,
    /// 16x64 block.
    Block16x64 = 20,
    /// 64x16 block.
    Block64x16 = 21,
}

impl BlockSize {
    /// Get width in samples.
    #[must_use]
    pub const fn width(self) -> u32 {
        match self {
            Self::Block4x4 | Self::Block4x8 | Self::Block4x16 => 4,
            Self::Block8x4 | Self::Block8x8 | Self::Block8x16 | Self::Block8x32 => 8,
            Self::Block16x4
            | Self::Block16x8
            | Self::Block16x16
            | Self::Block16x32
            | Self::Block16x64 => 16,
            Self::Block32x8 | Self::Block32x16 | Self::Block32x32 | Self::Block32x64 => 32,
            Self::Block64x16 | Self::Block64x32 | Self::Block64x64 | Self::Block64x128 => 64,
            Self::Block128x64 | Self::Block128x128 => 128,
        }
    }

    /// Get height in samples.
    #[must_use]
    pub const fn height(self) -> u32 {
        match self {
            Self::Block4x4 | Self::Block8x4 | Self::Block16x4 => 4,
            Self::Block4x8 | Self::Block8x8 | Self::Block16x8 | Self::Block32x8 => 8,
            Self::Block4x16
            | Self::Block8x16
            | Self::Block16x16
            | Self::Block32x16
            | Self::Block64x16 => 16,
            Self::Block8x32 | Self::Block16x32 | Self::Block32x32 | Self::Block64x32 => 32,
            Self::Block16x64 | Self::Block32x64 | Self::Block64x64 | Self::Block128x64 => 64,
            Self::Block64x128 | Self::Block128x128 => 128,
        }
    }

    /// Get width log2.
    #[must_use]
    pub const fn width_log2(self) -> u8 {
        match self.width() {
            4 => 2,
            8 => 3,
            16 => 4,
            32 => 5,
            64 => 6,
            128 => 7,
            _ => 0,
        }
    }

    /// Get height log2.
    #[must_use]
    pub const fn height_log2(self) -> u8 {
        match self.height() {
            4 => 2,
            8 => 3,
            16 => 4,
            32 => 5,
            64 => 6,
            128 => 7,
            _ => 0,
        }
    }

    /// Get width in 4x4 units.
    #[must_use]
    pub const fn width_mi(self) -> u32 {
        self.width() / 4
    }

    /// Get height in 4x4 units.
    #[must_use]
    pub const fn height_mi(self) -> u32 {
        self.height() / 4
    }

    /// Check if this is a square block.
    #[must_use]
    pub const fn is_square(self) -> bool {
        self.width() == self.height()
    }

    /// Check if this block size is valid for a superblock.
    #[must_use]
    pub const fn is_superblock(self) -> bool {
        matches!(self, Self::Block64x64 | Self::Block128x128)
    }

    /// Get the area in samples.
    #[must_use]
    pub const fn area(self) -> u32 {
        self.width() * self.height()
    }

    /// Convert from integer value.
    #[must_use]
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::Block4x4),
            1 => Some(Self::Block4x8),
            2 => Some(Self::Block8x4),
            3 => Some(Self::Block8x8),
            4 => Some(Self::Block8x16),
            5 => Some(Self::Block16x8),
            6 => Some(Self::Block16x16),
            7 => Some(Self::Block16x32),
            8 => Some(Self::Block32x16),
            9 => Some(Self::Block32x32),
            10 => Some(Self::Block32x64),
            11 => Some(Self::Block64x32),
            12 => Some(Self::Block64x64),
            13 => Some(Self::Block64x128),
            14 => Some(Self::Block128x64),
            15 => Some(Self::Block128x128),
            16 => Some(Self::Block4x16),
            17 => Some(Self::Block16x4),
            18 => Some(Self::Block8x32),
            19 => Some(Self::Block32x8),
            20 => Some(Self::Block16x64),
            21 => Some(Self::Block64x16),
            _ => None,
        }
    }

    /// Get block size from dimensions.
    #[must_use]
    pub const fn from_dimensions(width: u32, height: u32) -> Option<Self> {
        match (width, height) {
            (4, 4) => Some(Self::Block4x4),
            (4, 8) => Some(Self::Block4x8),
            (8, 4) => Some(Self::Block8x4),
            (8, 8) => Some(Self::Block8x8),
            (8, 16) => Some(Self::Block8x16),
            (16, 8) => Some(Self::Block16x8),
            (16, 16) => Some(Self::Block16x16),
            (16, 32) => Some(Self::Block16x32),
            (32, 16) => Some(Self::Block32x16),
            (32, 32) => Some(Self::Block32x32),
            (32, 64) => Some(Self::Block32x64),
            (64, 32) => Some(Self::Block64x32),
            (64, 64) => Some(Self::Block64x64),
            (64, 128) => Some(Self::Block64x128),
            (128, 64) => Some(Self::Block128x64),
            (128, 128) => Some(Self::Block128x128),
            (4, 16) => Some(Self::Block4x16),
            (16, 4) => Some(Self::Block16x4),
            (8, 32) => Some(Self::Block8x32),
            (32, 8) => Some(Self::Block32x8),
            (16, 64) => Some(Self::Block16x64),
            (64, 16) => Some(Self::Block64x16),
            _ => None,
        }
    }

    /// Get maximum transform size for this block.
    #[must_use]
    pub const fn max_tx_size(self) -> TxSize {
        match const_min_u32(self.width(), self.height()) {
            4 => TxSize::Tx4x4,
            8 => TxSize::Tx8x8,
            16 => TxSize::Tx16x16,
            32 => TxSize::Tx32x32,
            _ => TxSize::Tx64x64,
        }
    }

    /// Get subsampled block size for chroma planes.
    #[must_use]
    pub const fn subsampled(self, subx: bool, suby: bool) -> Option<Self> {
        let w = if subx { self.width() / 2 } else { self.width() };
        let h = if suby {
            self.height() / 2
        } else {
            self.height()
        };
        Self::from_dimensions(w, h)
    }
}

// =============================================================================
// Partition Type
// =============================================================================

/// Block partition type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PartitionType {
    /// No partition (leaf block).
    #[default]
    None = 0,
    /// Horizontal split into 2.
    Horz = 1,
    /// Vertical split into 2.
    Vert = 2,
    /// Split into 4 sub-blocks.
    Split = 3,
    /// Horizontal split, top A is smaller.
    HorzA = 4,
    /// Horizontal split, bottom B is smaller.
    HorzB = 5,
    /// Vertical split, left A is smaller.
    VertA = 6,
    /// Vertical split, right B is smaller.
    VertB = 7,
    /// Horizontal 4-way split.
    Horz4 = 8,
    /// Vertical 4-way split.
    Vert4 = 9,
}

impl PartitionType {
    /// Convert from integer value.
    #[must_use]
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::None),
            1 => Some(Self::Horz),
            2 => Some(Self::Vert),
            3 => Some(Self::Split),
            4 => Some(Self::HorzA),
            5 => Some(Self::HorzB),
            6 => Some(Self::VertA),
            7 => Some(Self::VertB),
            8 => Some(Self::Horz4),
            9 => Some(Self::Vert4),
            _ => None,
        }
    }

    /// Get number of sub-blocks for this partition.
    #[must_use]
    pub const fn num_sub_blocks(self) -> u8 {
        match self {
            Self::None => 1,
            Self::Horz | Self::Vert => 2,
            Self::Split => 4,
            Self::HorzA | Self::HorzB | Self::VertA | Self::VertB => 3,
            Self::Horz4 | Self::Vert4 => 4,
        }
    }

    /// Check if this partition is a split.
    #[must_use]
    pub const fn is_split(self) -> bool {
        matches!(self, Self::Split)
    }

    /// Check if this is a leaf partition (no further splits).
    #[must_use]
    pub const fn is_leaf(self) -> bool {
        matches!(self, Self::None)
    }
}

// =============================================================================
// Intra Mode
// =============================================================================

/// Intra prediction mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IntraMode {
    /// DC prediction.
    #[default]
    DcPred = 0,
    /// Vertical prediction.
    VPred = 1,
    /// Horizontal prediction.
    HPred = 2,
    /// Diagonal down-left prediction.
    D45Pred = 3,
    /// Diagonal down-right prediction.
    D135Pred = 4,
    /// Diagonal 113 degrees prediction.
    D113Pred = 5,
    /// Diagonal 157 degrees prediction.
    D157Pred = 6,
    /// Diagonal 203 degrees prediction.
    D203Pred = 7,
    /// Diagonal 67 degrees prediction.
    D67Pred = 8,
    /// Smooth prediction.
    SmoothPred = 9,
    /// Smooth vertical prediction.
    SmoothVPred = 10,
    /// Smooth horizontal prediction.
    SmoothHPred = 11,
    /// Paeth prediction.
    PaethPred = 12,
}

impl IntraMode {
    /// Convert from integer value.
    #[must_use]
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::DcPred),
            1 => Some(Self::VPred),
            2 => Some(Self::HPred),
            3 => Some(Self::D45Pred),
            4 => Some(Self::D135Pred),
            5 => Some(Self::D113Pred),
            6 => Some(Self::D157Pred),
            7 => Some(Self::D203Pred),
            8 => Some(Self::D67Pred),
            9 => Some(Self::SmoothPred),
            10 => Some(Self::SmoothVPred),
            11 => Some(Self::SmoothHPred),
            12 => Some(Self::PaethPred),
            _ => None,
        }
    }

    /// Check if this is a directional mode.
    #[must_use]
    pub const fn is_directional(self) -> bool {
        matches!(
            self,
            Self::VPred
                | Self::HPred
                | Self::D45Pred
                | Self::D135Pred
                | Self::D113Pred
                | Self::D157Pred
                | Self::D203Pred
                | Self::D67Pred
        )
    }

    /// Get angle delta allowed for this mode.
    #[must_use]
    pub const fn angle_delta_allowed(self) -> bool {
        self.is_directional()
    }
}

// =============================================================================
// Inter Mode
// =============================================================================

/// Inter prediction mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum InterMode {
    /// Nearest MV mode.
    #[default]
    NearestMv = 0,
    /// Near MV mode.
    NearMv = 1,
    /// Global MV mode.
    GlobalMv = 2,
    /// New MV mode.
    NewMv = 3,
}

impl InterMode {
    /// Convert from integer value.
    #[must_use]
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::NearestMv),
            1 => Some(Self::NearMv),
            2 => Some(Self::GlobalMv),
            3 => Some(Self::NewMv),
            _ => None,
        }
    }

    /// Check if this mode uses a new motion vector.
    #[must_use]
    pub const fn has_newmv(self) -> bool {
        matches!(self, Self::NewMv)
    }

    /// Check if this mode uses global motion.
    #[must_use]
    pub const fn is_global(self) -> bool {
        matches!(self, Self::GlobalMv)
    }
}

// =============================================================================
// Block Mode Info
// =============================================================================

/// Block mode information.
#[derive(Clone, Debug, Default)]
pub struct BlockModeInfo {
    /// Block size.
    pub block_size: BlockSize,
    /// Segment ID.
    pub segment_id: u8,
    /// Skip residual flag.
    pub skip: bool,
    /// Skip mode (compound prediction skip).
    pub skip_mode: bool,
    /// Is inter block.
    pub is_inter: bool,
    /// Intra mode (for intra blocks).
    pub intra_mode: IntraMode,
    /// UV intra mode.
    pub uv_mode: IntraMode,
    /// Intra angle delta.
    pub angle_delta: [i8; 2],
    /// Inter mode (for inter blocks).
    pub inter_mode: InterMode,
    /// Reference frames (up to 2 for compound).
    pub ref_frames: [i8; 2],
    /// Motion vectors (up to 2 for compound).
    pub mv: [[i16; 2]; 2],
    /// Transform size.
    pub tx_size: TxSize,
    /// Use palette mode.
    pub use_palette: bool,
    /// Filter intra mode.
    pub filter_intra_mode: u8,
    /// Compound type.
    pub compound_type: u8,
    /// Interpolation filter.
    pub interp_filter: [u8; 2],
    /// Motion mode (simple, obmc, warp).
    pub motion_mode: u8,
}

impl BlockModeInfo {
    /// Create a new block mode info.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            block_size: BlockSize::Block4x4,
            segment_id: 0,
            skip: false,
            skip_mode: false,
            is_inter: false,
            intra_mode: IntraMode::DcPred,
            uv_mode: IntraMode::DcPred,
            angle_delta: [0, 0],
            inter_mode: InterMode::NearestMv,
            ref_frames: [-1, -1],
            mv: [[0, 0], [0, 0]],
            tx_size: TxSize::Tx4x4,
            use_palette: false,
            filter_intra_mode: 0,
            compound_type: 0,
            interp_filter: [0, 0],
            motion_mode: 0,
        }
    }

    /// Check if this is an intra block.
    #[must_use]
    pub const fn is_intra(&self) -> bool {
        !self.is_inter
    }

    /// Check if this block uses compound prediction.
    #[must_use]
    pub const fn is_compound(&self) -> bool {
        self.ref_frames[1] >= 0
    }

    /// Get the number of reference frames used.
    #[must_use]
    pub const fn num_refs(&self) -> u8 {
        if self.ref_frames[1] >= 0 {
            2
        } else if self.ref_frames[0] >= 0 {
            1
        } else {
            0
        }
    }
}

// =============================================================================
// Plane Block Context
// =============================================================================

/// Per-plane block context.
#[derive(Clone, Debug, Default)]
pub struct PlaneBlockContext {
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// X position in samples.
    pub x: u32,
    /// Y position in samples.
    pub y: u32,
    /// Transform size.
    pub tx_size: TxSize,
    /// Number of transform blocks in width.
    pub tx_width: u32,
    /// Number of transform blocks in height.
    pub tx_height: u32,
    /// Subsampling X.
    pub subx: bool,
    /// Subsampling Y.
    pub suby: bool,
}

impl PlaneBlockContext {
    /// Create a new plane block context.
    #[must_use]
    pub const fn new(plane: usize, subx: bool, suby: bool) -> Self {
        Self {
            width: 4,
            height: 4,
            x: 0,
            y: 0,
            tx_size: TxSize::Tx4x4,
            tx_width: 1,
            tx_height: 1,
            subx: plane > 0 && subx,
            suby: plane > 0 && suby,
        }
    }

    /// Set position from block.
    pub fn set_from_block(&mut self, bx: u32, by: u32, bsize: BlockSize) {
        let scale_x = if self.subx { 2 } else { 1 };
        let scale_y = if self.suby { 2 } else { 1 };

        self.width = bsize.width() / scale_x;
        self.height = bsize.height() / scale_y;
        self.x = bx / scale_x;
        self.y = by / scale_y;

        // Compute transform blocks
        self.tx_width = self.width / self.tx_size.width();
        self.tx_height = self.height / self.tx_size.height();
    }

    /// Get number of transform blocks.
    #[must_use]
    pub const fn num_tx_blocks(&self) -> u32 {
        self.tx_width * self.tx_height
    }
}

// =============================================================================
// Block Context Manager
// =============================================================================

/// Manager for block-level context.
#[derive(Clone, Debug)]
pub struct BlockContextManager {
    /// Plane contexts.
    pub planes: [PlaneBlockContext; MAX_PLANES],
    /// Current mode info.
    pub mode_info: BlockModeInfo,
    /// Row in 4x4 units.
    pub mi_row: u32,
    /// Column in 4x4 units.
    pub mi_col: u32,
    /// Above mode info references.
    pub above_ctx: Vec<u8>,
    /// Left mode info references.
    pub left_ctx: Vec<u8>,
}

impl BlockContextManager {
    /// Create a new block context manager.
    #[must_use]
    pub fn new(width_mi: u32, subx: bool, suby: bool) -> Self {
        Self {
            planes: [
                PlaneBlockContext::new(0, subx, suby),
                PlaneBlockContext::new(1, subx, suby),
                PlaneBlockContext::new(2, subx, suby),
            ],
            mode_info: BlockModeInfo::new(),
            mi_row: 0,
            mi_col: 0,
            above_ctx: vec![0; width_mi as usize],
            left_ctx: vec![0; MAX_SB_SQUARE],
        }
    }

    /// Set current block position.
    pub fn set_position(&mut self, mi_row: u32, mi_col: u32, bsize: BlockSize) {
        self.mi_row = mi_row;
        self.mi_col = mi_col;
        self.mode_info.block_size = bsize;

        let bx = mi_col * 4;
        let by = mi_row * 4;

        for plane in &mut self.planes {
            plane.set_from_block(bx, by, bsize);
        }
    }

    /// Get context for partition.
    #[must_use]
    pub fn get_partition_context(&self, bsize: BlockSize) -> u8 {
        let bs = bsize.width_log2();
        let above = self.get_above_ctx(0);
        let left = self.get_left_ctx(0);

        // Simple context based on neighbors
        let ctx = u8::from(left < bs) + u8::from(above < bs);
        ctx.min(3)
    }

    /// Get above context value.
    #[must_use]
    pub fn get_above_ctx(&self, offset: u32) -> u8 {
        let col = self.mi_col as usize + offset as usize;
        if col < self.above_ctx.len() {
            self.above_ctx[col]
        } else {
            0
        }
    }

    /// Get left context value.
    #[must_use]
    pub fn get_left_ctx(&self, offset: u32) -> u8 {
        let row = (self.mi_row as usize + offset as usize) % MAX_SB_SQUARE;
        if row < self.left_ctx.len() {
            self.left_ctx[row]
        } else {
            0
        }
    }

    /// Update context after decoding a block.
    pub fn update_context(&mut self, bsize: BlockSize) {
        let w = bsize.width_mi() as usize;
        let h = bsize.height_mi() as usize;
        let ctx_val = if self.mode_info.is_inter { 1 } else { 0 };

        // Update above context
        let col = self.mi_col as usize;
        for i in 0..w {
            if col + i < self.above_ctx.len() {
                self.above_ctx[col + i] = ctx_val;
            }
        }

        // Update left context
        let row = self.mi_row as usize;
        for i in 0..h {
            let r = (row + i) % MAX_SB_SQUARE;
            if r < self.left_ctx.len() {
                self.left_ctx[r] = ctx_val;
            }
        }
    }

    /// Reset left context for new superblock row.
    pub fn reset_left_context(&mut self) {
        self.left_ctx.fill(0);
    }
}

impl Default for BlockContextManager {
    fn default() -> Self {
        Self::new(1920 / 4, true, true)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_size_dimensions() {
        assert_eq!(BlockSize::Block4x4.width(), 4);
        assert_eq!(BlockSize::Block4x4.height(), 4);
        assert_eq!(BlockSize::Block128x128.width(), 128);
        assert_eq!(BlockSize::Block128x128.height(), 128);
        assert_eq!(BlockSize::Block4x8.width(), 4);
        assert_eq!(BlockSize::Block4x8.height(), 8);
    }

    #[test]
    fn test_block_size_log2() {
        assert_eq!(BlockSize::Block4x4.width_log2(), 2);
        assert_eq!(BlockSize::Block8x8.width_log2(), 3);
        assert_eq!(BlockSize::Block16x16.width_log2(), 4);
        assert_eq!(BlockSize::Block128x128.width_log2(), 7);
    }

    #[test]
    fn test_block_size_mi() {
        assert_eq!(BlockSize::Block4x4.width_mi(), 1);
        assert_eq!(BlockSize::Block8x8.width_mi(), 2);
        assert_eq!(BlockSize::Block128x128.width_mi(), 32);
    }

    #[test]
    fn test_block_size_is_square() {
        assert!(BlockSize::Block4x4.is_square());
        assert!(BlockSize::Block8x8.is_square());
        assert!(!BlockSize::Block4x8.is_square());
        assert!(!BlockSize::Block8x4.is_square());
    }

    #[test]
    fn test_block_size_from_u8() {
        assert_eq!(BlockSize::from_u8(0), Some(BlockSize::Block4x4));
        assert_eq!(BlockSize::from_u8(15), Some(BlockSize::Block128x128));
        assert_eq!(BlockSize::from_u8(100), None);
    }

    #[test]
    fn test_block_size_from_dimensions() {
        assert_eq!(BlockSize::from_dimensions(4, 4), Some(BlockSize::Block4x4));
        assert_eq!(
            BlockSize::from_dimensions(128, 128),
            Some(BlockSize::Block128x128)
        );
        assert_eq!(BlockSize::from_dimensions(3, 3), None);
    }

    #[test]
    fn test_partition_type() {
        assert_eq!(PartitionType::None.num_sub_blocks(), 1);
        assert_eq!(PartitionType::Horz.num_sub_blocks(), 2);
        assert_eq!(PartitionType::Split.num_sub_blocks(), 4);
        assert!(PartitionType::Split.is_split());
        assert!(PartitionType::None.is_leaf());
    }

    #[test]
    fn test_intra_mode() {
        assert!(IntraMode::VPred.is_directional());
        assert!(!IntraMode::DcPred.is_directional());
        assert!(IntraMode::D45Pred.angle_delta_allowed());
    }

    #[test]
    fn test_inter_mode() {
        assert!(InterMode::NewMv.has_newmv());
        assert!(InterMode::GlobalMv.is_global());
        assert!(!InterMode::NearestMv.has_newmv());
    }

    #[test]
    fn test_block_mode_info() {
        let info = BlockModeInfo::new();
        assert!(!info.is_inter);
        assert!(info.is_intra());
        assert!(!info.is_compound());
        assert_eq!(info.num_refs(), 0);
    }

    #[test]
    fn test_plane_block_context() {
        let mut ctx = PlaneBlockContext::new(1, true, true);
        ctx.set_from_block(16, 16, BlockSize::Block8x8);

        assert_eq!(ctx.width, 4); // 8 / 2 for chroma
        assert_eq!(ctx.height, 4);
        assert_eq!(ctx.x, 8); // 16 / 2
    }

    #[test]
    fn test_block_context_manager() {
        let mut mgr = BlockContextManager::new(480, true, true);
        mgr.set_position(4, 8, BlockSize::Block16x16);

        assert_eq!(mgr.mi_row, 4);
        assert_eq!(mgr.mi_col, 8);
    }

    #[test]
    fn test_block_size_subsampled() {
        let size = BlockSize::Block16x16;
        let subsampled = size.subsampled(true, true);

        assert_eq!(subsampled, Some(BlockSize::Block8x8));
    }

    #[test]
    fn test_block_size_max_tx_size() {
        assert_eq!(BlockSize::Block4x4.max_tx_size(), TxSize::Tx4x4);
        assert_eq!(BlockSize::Block8x8.max_tx_size(), TxSize::Tx8x8);
        assert_eq!(BlockSize::Block16x16.max_tx_size(), TxSize::Tx16x16);
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_PLANES, 3);
        assert_eq!(MAX_SEGMENTS, 8);
        assert_eq!(MAX_SB_SIZE, 128);
        assert_eq!(BLOCK_SIZES, 22);
        assert_eq!(PARTITION_TYPES, 10);
        assert_eq!(INTRA_MODES, 13);
    }

    #[test]
    fn test_block_context_manager_update() {
        let mut mgr = BlockContextManager::new(480, true, true);
        mgr.set_position(0, 0, BlockSize::Block8x8);
        mgr.mode_info.is_inter = true;
        mgr.update_context(BlockSize::Block8x8);

        assert_eq!(mgr.above_ctx[0], 1);
        assert_eq!(mgr.above_ctx[1], 1);
    }
}
