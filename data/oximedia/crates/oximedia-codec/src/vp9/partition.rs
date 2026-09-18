//! VP9 Partition types and block sizes.
//!
//! This module defines the partition structure used in VP9 for recursive
//! block splitting and the various block sizes used during encoding/decoding.

#![forbid(unsafe_code)]
#![allow(dead_code)]

/// Number of partition types.
pub const PARTITION_TYPES: usize = 4;

/// Number of block sizes.
pub const BLOCK_SIZES: usize = 13;

/// Number of transform sizes.
pub const TX_SIZES: usize = 4;

/// Maximum superblock size in pixels.
pub const SB_SIZE: usize = 64;

/// Maximum superblock size log2.
pub const SB_SIZE_LOG2: usize = 6;

/// Minimum block size in pixels.
pub const MIN_BLOCK_SIZE: usize = 4;

/// Minimum block size log2.
pub const MIN_BLOCK_SIZE_LOG2: usize = 2;

/// Partition type for recursive block splitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum Partition {
    /// No partition - use the full block.
    #[default]
    None = 0,
    /// Horizontal partition - split into top and bottom.
    Horz = 1,
    /// Vertical partition - split into left and right.
    Vert = 2,
    /// Split partition - divide into 4 sub-blocks.
    Split = 3,
}

impl Partition {
    /// Converts from u8 value to `Partition`.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::None),
            1 => Some(Self::Horz),
            2 => Some(Self::Vert),
            3 => Some(Self::Split),
            _ => None,
        }
    }

    /// Returns true if this partition splits the block.
    #[must_use]
    pub const fn is_split(&self) -> bool {
        matches!(self, Self::Split)
    }

    /// Returns true if this partition has horizontal sub-blocks.
    #[must_use]
    pub const fn has_horizontal_split(&self) -> bool {
        matches!(self, Self::Horz | Self::Split)
    }

    /// Returns true if this partition has vertical sub-blocks.
    #[must_use]
    pub const fn has_vertical_split(&self) -> bool {
        matches!(self, Self::Vert | Self::Split)
    }

    /// Returns the number of sub-blocks for this partition.
    #[must_use]
    pub const fn num_sub_blocks(&self) -> usize {
        match self {
            Self::None => 1,
            Self::Horz | Self::Vert => 2,
            Self::Split => 4,
        }
    }
}

impl From<Partition> for u8 {
    fn from(value: Partition) -> Self {
        value as u8
    }
}

/// Block size enumeration.
///
/// VP9 supports block sizes from 4x4 to 64x64 pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum BlockSize {
    /// 4x4 block.
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
    /// 64x64 block (superblock).
    #[default]
    Block64x64 = 12,
}

impl BlockSize {
    /// All block sizes in order.
    pub const ALL: [BlockSize; BLOCK_SIZES] = [
        BlockSize::Block4x4,
        BlockSize::Block4x8,
        BlockSize::Block8x4,
        BlockSize::Block8x8,
        BlockSize::Block8x16,
        BlockSize::Block16x8,
        BlockSize::Block16x16,
        BlockSize::Block16x32,
        BlockSize::Block32x16,
        BlockSize::Block32x32,
        BlockSize::Block32x64,
        BlockSize::Block64x32,
        BlockSize::Block64x64,
    ];

    /// Block widths in pixels.
    const WIDTHS: [usize; BLOCK_SIZES] = [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64];

    /// Block heights in pixels.
    const HEIGHTS: [usize; BLOCK_SIZES] = [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64];

    /// Block width log2 values.
    const WIDTH_LOG2: [usize; BLOCK_SIZES] = [2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6];

    /// Block height log2 values.
    const HEIGHT_LOG2: [usize; BLOCK_SIZES] = [2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6];

    /// Converts from u8 value to `BlockSize`.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
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
            _ => None,
        }
    }

    /// Returns the width in pixels.
    #[must_use]
    pub const fn width(&self) -> usize {
        Self::WIDTHS[*self as usize]
    }

    /// Returns the height in pixels.
    #[must_use]
    pub const fn height(&self) -> usize {
        Self::HEIGHTS[*self as usize]
    }

    /// Returns log2 of width.
    #[must_use]
    pub const fn width_log2(&self) -> usize {
        Self::WIDTH_LOG2[*self as usize]
    }

    /// Returns log2 of height.
    #[must_use]
    pub const fn height_log2(&self) -> usize {
        Self::HEIGHT_LOG2[*self as usize]
    }

    /// Returns the number of 4x4 blocks in width.
    #[must_use]
    pub const fn width_mi(&self) -> usize {
        self.width() >> 2
    }

    /// Returns the number of 4x4 blocks in height.
    #[must_use]
    pub const fn height_mi(&self) -> usize {
        self.height() >> 2
    }

    /// Returns the area in pixels.
    #[must_use]
    pub const fn area(&self) -> usize {
        self.width() * self.height()
    }

    /// Returns the number of 4x4 blocks.
    #[must_use]
    pub const fn num_4x4_blocks(&self) -> usize {
        self.width_mi() * self.height_mi()
    }

    /// Returns true if this is a square block.
    #[must_use]
    pub const fn is_square(&self) -> bool {
        self.width() == self.height()
    }

    /// Returns true if this is the superblock size (64x64).
    #[must_use]
    pub const fn is_superblock(&self) -> bool {
        matches!(self, Self::Block64x64)
    }

    /// Returns true if this is the minimum block size (4x4).
    #[must_use]
    pub const fn is_minimum(&self) -> bool {
        matches!(self, Self::Block4x4)
    }

    /// Returns the subsize after applying a partition.
    ///
    /// VP9 only supports specific partition combinations. Returns `None`
    /// for invalid partition combinations.
    #[must_use]
    #[allow(clippy::match_same_arms)]
    pub const fn subsize(&self, partition: Partition) -> Option<BlockSize> {
        match (self, partition) {
            // 64x64 superblock partitions
            (Self::Block64x64, Partition::None) => Some(Self::Block64x64),
            (Self::Block64x64, Partition::Horz) => Some(Self::Block64x32),
            (Self::Block64x64, Partition::Vert) => Some(Self::Block32x64),
            (Self::Block64x64, Partition::Split) => Some(Self::Block32x32),

            // 32x32 block partitions
            (Self::Block32x32, Partition::None) => Some(Self::Block32x32),
            (Self::Block32x32, Partition::Horz) => Some(Self::Block32x16),
            (Self::Block32x32, Partition::Vert) => Some(Self::Block16x32),
            (Self::Block32x32, Partition::Split) => Some(Self::Block16x16),

            // 16x16 block partitions
            (Self::Block16x16, Partition::None) => Some(Self::Block16x16),
            (Self::Block16x16, Partition::Horz) => Some(Self::Block16x8),
            (Self::Block16x16, Partition::Vert) => Some(Self::Block8x16),
            (Self::Block16x16, Partition::Split) => Some(Self::Block8x8),

            // 8x8 block partitions
            (Self::Block8x8, Partition::None) => Some(Self::Block8x8),
            (Self::Block8x8, Partition::Horz) => Some(Self::Block8x4),
            (Self::Block8x8, Partition::Vert) => Some(Self::Block4x8),
            (Self::Block8x8, Partition::Split) => Some(Self::Block4x4),

            // Rectangular blocks only support NONE partition
            (Self::Block64x32, Partition::None) => Some(Self::Block64x32),
            (Self::Block32x64, Partition::None) => Some(Self::Block32x64),
            (Self::Block32x16, Partition::None) => Some(Self::Block32x16),
            (Self::Block16x32, Partition::None) => Some(Self::Block16x32),
            (Self::Block16x8, Partition::None) => Some(Self::Block16x8),
            (Self::Block8x16, Partition::None) => Some(Self::Block8x16),
            (Self::Block8x4, Partition::None) => Some(Self::Block8x4),
            (Self::Block4x8, Partition::None) => Some(Self::Block4x8),
            (Self::Block4x4, Partition::None) => Some(Self::Block4x4),

            // All other combinations are invalid
            _ => None,
        }
    }

    /// Returns the maximum transform size for this block size.
    #[must_use]
    pub const fn max_tx_size(&self) -> TxSize {
        match self {
            Self::Block64x64 | Self::Block64x32 | Self::Block32x64 | Self::Block32x32 => {
                TxSize::Tx32x32
            }
            Self::Block32x16 | Self::Block16x32 | Self::Block16x16 => TxSize::Tx16x16,
            Self::Block16x8 | Self::Block8x16 | Self::Block8x8 => TxSize::Tx8x8,
            Self::Block8x4 | Self::Block4x8 | Self::Block4x4 => TxSize::Tx4x4,
        }
    }
}

impl From<BlockSize> for u8 {
    fn from(value: BlockSize) -> Self {
        value as u8
    }
}

/// Transform size enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum TxSize {
    /// 4x4 transform.
    #[default]
    Tx4x4 = 0,
    /// 8x8 transform.
    Tx8x8 = 1,
    /// 16x16 transform.
    Tx16x16 = 2,
    /// 32x32 transform.
    Tx32x32 = 3,
}

impl TxSize {
    /// All transform sizes.
    pub const ALL: [TxSize; TX_SIZES] = [
        TxSize::Tx4x4,
        TxSize::Tx8x8,
        TxSize::Tx16x16,
        TxSize::Tx32x32,
    ];

    /// Transform widths in pixels.
    const SIZES: [usize; TX_SIZES] = [4, 8, 16, 32];

    /// Converts from u8 value to `TxSize`.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Tx4x4),
            1 => Some(Self::Tx8x8),
            2 => Some(Self::Tx16x16),
            3 => Some(Self::Tx32x32),
            _ => None,
        }
    }

    /// Returns the size in pixels.
    #[must_use]
    pub const fn size(&self) -> usize {
        Self::SIZES[*self as usize]
    }

    /// Returns log2 of size.
    #[must_use]
    pub const fn size_log2(&self) -> usize {
        *self as usize + 2
    }

    /// Returns the area in pixels.
    #[must_use]
    pub const fn area(&self) -> usize {
        self.size() * self.size()
    }

    /// Returns the number of coefficients.
    #[must_use]
    pub const fn num_coeffs(&self) -> usize {
        self.area()
    }

    /// Returns the smaller transform size, if any.
    #[must_use]
    pub const fn smaller(&self) -> Option<Self> {
        match self {
            Self::Tx4x4 => None,
            Self::Tx8x8 => Some(Self::Tx4x4),
            Self::Tx16x16 => Some(Self::Tx8x8),
            Self::Tx32x32 => Some(Self::Tx16x16),
        }
    }

    /// Returns the larger transform size, if any.
    #[must_use]
    pub const fn larger(&self) -> Option<Self> {
        match self {
            Self::Tx4x4 => Some(Self::Tx8x8),
            Self::Tx8x8 => Some(Self::Tx16x16),
            Self::Tx16x16 => Some(Self::Tx32x32),
            Self::Tx32x32 => None,
        }
    }
}

impl From<TxSize> for u8 {
    fn from(value: TxSize) -> Self {
        value as u8
    }
}

/// Transform mode for selecting transform sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum TxMode {
    /// Only 4x4 transforms.
    #[default]
    Only4x4 = 0,
    /// Allow 8x8 and smaller.
    Allow8x8 = 1,
    /// Allow 16x16 and smaller.
    Allow16x16 = 2,
    /// Allow 32x32 and smaller.
    Allow32x32 = 3,
    /// Transform size selected per block.
    Select = 4,
}

impl TxMode {
    /// Converts from u8 value to `TxMode`.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Only4x4),
            1 => Some(Self::Allow8x8),
            2 => Some(Self::Allow16x16),
            3 => Some(Self::Allow32x32),
            4 => Some(Self::Select),
            _ => None,
        }
    }

    /// Returns the maximum allowed transform size for this mode.
    #[must_use]
    pub const fn max_tx_size(&self) -> TxSize {
        match self {
            Self::Only4x4 => TxSize::Tx4x4,
            Self::Allow8x8 => TxSize::Tx8x8,
            Self::Allow16x16 => TxSize::Tx16x16,
            Self::Allow32x32 | Self::Select => TxSize::Tx32x32,
        }
    }

    /// Returns true if transform size selection is allowed.
    #[must_use]
    pub const fn is_select(&self) -> bool {
        matches!(self, Self::Select)
    }
}

impl From<TxMode> for u8 {
    fn from(value: TxMode) -> Self {
        value as u8
    }
}

/// Superblock structure representing a 64x64 region.
#[derive(Clone, Debug, Default)]
pub struct Superblock {
    /// Row index of the superblock (in superblock units).
    pub row: usize,
    /// Column index of the superblock (in superblock units).
    pub col: usize,
    /// Partition tree for this superblock.
    pub partition: Partition,
    /// Sub-partitions for recursive splitting.
    pub sub_partitions: [Option<Box<Superblock>>; 4],
}

impl Superblock {
    /// Creates a new superblock at the given position.
    #[must_use]
    pub const fn new(row: usize, col: usize) -> Self {
        Self {
            row,
            col,
            partition: Partition::None,
            sub_partitions: [None, None, None, None],
        }
    }

    /// Returns the pixel x coordinate.
    #[must_use]
    pub const fn pixel_x(&self) -> usize {
        self.col * SB_SIZE
    }

    /// Returns the pixel y coordinate.
    #[must_use]
    pub const fn pixel_y(&self) -> usize {
        self.row * SB_SIZE
    }

    /// Returns the 4x4 block x coordinate.
    #[must_use]
    pub const fn mi_col(&self) -> usize {
        self.col * (SB_SIZE / 4)
    }

    /// Returns the 4x4 block y coordinate.
    #[must_use]
    pub const fn mi_row(&self) -> usize {
        self.row * (SB_SIZE / 4)
    }
}

/// Block position in frame coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockPosition {
    /// Row in 4x4 block units.
    pub mi_row: usize,
    /// Column in 4x4 block units.
    pub mi_col: usize,
    /// Block size.
    pub size: BlockSize,
}

impl BlockPosition {
    /// Creates a new block position.
    #[must_use]
    pub const fn new(mi_row: usize, mi_col: usize, size: BlockSize) -> Self {
        Self {
            mi_row,
            mi_col,
            size,
        }
    }

    /// Returns the pixel x coordinate.
    #[must_use]
    pub const fn pixel_x(&self) -> usize {
        self.mi_col * 4
    }

    /// Returns the pixel y coordinate.
    #[must_use]
    pub const fn pixel_y(&self) -> usize {
        self.mi_row * 4
    }

    /// Returns the width in pixels.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.size.width()
    }

    /// Returns the height in pixels.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.size.height()
    }

    /// Returns true if this block is within the given frame dimensions.
    #[must_use]
    pub const fn is_within(&self, mi_rows: usize, mi_cols: usize) -> bool {
        self.mi_row < mi_rows && self.mi_col < mi_cols
    }
}

/// Partition context for entropy coding.
#[derive(Clone, Copy, Debug, Default)]
pub struct PartitionContext {
    /// Above partition context.
    pub above: u8,
    /// Left partition context.
    pub left: u8,
}

impl PartitionContext {
    /// Creates a new partition context.
    #[must_use]
    pub const fn new(above: u8, left: u8) -> Self {
        Self { above, left }
    }

    /// Returns the combined context index.
    #[must_use]
    pub const fn context_index(&self) -> usize {
        let above = if self.above > 0 { 1 } else { 0 };
        let left = if self.left > 0 { 1 } else { 0 };
        left * 2 + above
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_types() {
        assert!(!Partition::None.is_split());
        assert!(Partition::Split.is_split());
        assert_eq!(Partition::None.num_sub_blocks(), 1);
        assert_eq!(Partition::Horz.num_sub_blocks(), 2);
        assert_eq!(Partition::Vert.num_sub_blocks(), 2);
        assert_eq!(Partition::Split.num_sub_blocks(), 4);
    }

    #[test]
    fn test_partition_from_u8() {
        assert_eq!(Partition::from_u8(0), Some(Partition::None));
        assert_eq!(Partition::from_u8(1), Some(Partition::Horz));
        assert_eq!(Partition::from_u8(2), Some(Partition::Vert));
        assert_eq!(Partition::from_u8(3), Some(Partition::Split));
        assert_eq!(Partition::from_u8(4), None);
    }

    #[test]
    fn test_block_size_dimensions() {
        assert_eq!(BlockSize::Block4x4.width(), 4);
        assert_eq!(BlockSize::Block4x4.height(), 4);
        assert_eq!(BlockSize::Block64x64.width(), 64);
        assert_eq!(BlockSize::Block64x64.height(), 64);
        assert_eq!(BlockSize::Block8x16.width(), 8);
        assert_eq!(BlockSize::Block8x16.height(), 16);
    }

    #[test]
    fn test_block_size_square() {
        assert!(BlockSize::Block4x4.is_square());
        assert!(BlockSize::Block8x8.is_square());
        assert!(BlockSize::Block16x16.is_square());
        assert!(BlockSize::Block32x32.is_square());
        assert!(BlockSize::Block64x64.is_square());
        assert!(!BlockSize::Block4x8.is_square());
        assert!(!BlockSize::Block16x32.is_square());
    }

    #[test]
    fn test_block_size_superblock() {
        assert!(BlockSize::Block64x64.is_superblock());
        assert!(!BlockSize::Block32x32.is_superblock());
    }

    #[test]
    fn test_block_size_mi() {
        assert_eq!(BlockSize::Block4x4.width_mi(), 1);
        assert_eq!(BlockSize::Block4x4.height_mi(), 1);
        assert_eq!(BlockSize::Block64x64.width_mi(), 16);
        assert_eq!(BlockSize::Block64x64.height_mi(), 16);
    }

    #[test]
    fn test_block_size_area() {
        assert_eq!(BlockSize::Block4x4.area(), 16);
        assert_eq!(BlockSize::Block8x8.area(), 64);
        assert_eq!(BlockSize::Block64x64.area(), 4096);
    }

    #[test]
    fn test_block_size_subsize() {
        assert_eq!(
            BlockSize::Block64x64.subsize(Partition::Split),
            Some(BlockSize::Block32x32)
        );
        assert_eq!(
            BlockSize::Block32x32.subsize(Partition::Horz),
            Some(BlockSize::Block32x16)
        );
        assert_eq!(
            BlockSize::Block32x32.subsize(Partition::Vert),
            Some(BlockSize::Block16x32)
        );
        assert_eq!(
            BlockSize::Block8x8.subsize(Partition::Split),
            Some(BlockSize::Block4x4)
        );
    }

    #[test]
    fn test_tx_size_dimensions() {
        assert_eq!(TxSize::Tx4x4.size(), 4);
        assert_eq!(TxSize::Tx8x8.size(), 8);
        assert_eq!(TxSize::Tx16x16.size(), 16);
        assert_eq!(TxSize::Tx32x32.size(), 32);
    }

    #[test]
    fn test_tx_size_area() {
        assert_eq!(TxSize::Tx4x4.area(), 16);
        assert_eq!(TxSize::Tx8x8.area(), 64);
        assert_eq!(TxSize::Tx16x16.area(), 256);
        assert_eq!(TxSize::Tx32x32.area(), 1024);
    }

    #[test]
    fn test_tx_size_smaller_larger() {
        assert_eq!(TxSize::Tx4x4.smaller(), None);
        assert_eq!(TxSize::Tx8x8.smaller(), Some(TxSize::Tx4x4));
        assert_eq!(TxSize::Tx32x32.larger(), None);
        assert_eq!(TxSize::Tx16x16.larger(), Some(TxSize::Tx32x32));
    }

    #[test]
    fn test_tx_mode() {
        assert_eq!(TxMode::Only4x4.max_tx_size(), TxSize::Tx4x4);
        assert_eq!(TxMode::Allow8x8.max_tx_size(), TxSize::Tx8x8);
        assert_eq!(TxMode::Allow32x32.max_tx_size(), TxSize::Tx32x32);
        assert!(TxMode::Select.is_select());
        assert!(!TxMode::Allow32x32.is_select());
    }

    #[test]
    fn test_superblock() {
        let sb = Superblock::new(1, 2);
        assert_eq!(sb.row, 1);
        assert_eq!(sb.col, 2);
        assert_eq!(sb.pixel_x(), 128);
        assert_eq!(sb.pixel_y(), 64);
        assert_eq!(sb.mi_col(), 32);
        assert_eq!(sb.mi_row(), 16);
    }

    #[test]
    fn test_block_position() {
        let pos = BlockPosition::new(4, 8, BlockSize::Block16x16);
        assert_eq!(pos.pixel_x(), 32);
        assert_eq!(pos.pixel_y(), 16);
        assert_eq!(pos.width(), 16);
        assert_eq!(pos.height(), 16);
        assert!(pos.is_within(10, 20));
        assert!(!pos.is_within(3, 20));
    }

    #[test]
    fn test_partition_context() {
        let ctx = PartitionContext::new(1, 0);
        assert_eq!(ctx.context_index(), 1);
        let ctx2 = PartitionContext::new(0, 1);
        assert_eq!(ctx2.context_index(), 2);
        let ctx3 = PartitionContext::new(1, 1);
        assert_eq!(ctx3.context_index(), 3);
    }

    #[test]
    fn test_block_size_max_tx() {
        assert_eq!(BlockSize::Block64x64.max_tx_size(), TxSize::Tx32x32);
        assert_eq!(BlockSize::Block32x32.max_tx_size(), TxSize::Tx32x32);
        assert_eq!(BlockSize::Block16x16.max_tx_size(), TxSize::Tx16x16);
        assert_eq!(BlockSize::Block8x8.max_tx_size(), TxSize::Tx8x8);
        assert_eq!(BlockSize::Block4x4.max_tx_size(), TxSize::Tx4x4);
    }
}
