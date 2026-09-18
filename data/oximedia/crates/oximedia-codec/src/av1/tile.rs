//! AV1 tile processing.
//!
//! AV1 divides frames into rectangular tiles for parallel processing.
//! Each tile can be decoded independently, enabling efficient
//! multi-threaded decoding.
//!
//! # Tile Structure
//!
//! - Tiles are arranged in a grid pattern
//! - Minimum tile size is 256x256 luma samples (for level 5.0+: 128x128)
//! - Maximum of 64 tile columns and 64 tile rows
//! - Each tile contains superblocks (64x64 or 128x128)
//!
//! # Tile Groups
//!
//! Tile groups allow tiles to be packaged into separate OBUs
//! for flexible delivery and error resilience.
//!
//! # Reference
//!
//! See AV1 Specification Section 5.9.15 for tile info syntax and
//! Section 6.7.10 for tile info semantics.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::missing_errors_doc)]

use super::frame_header::FrameSize;
use super::sequence::SequenceHeader;
use crate::error::{CodecError, CodecResult};
use oximedia_io::BitReader;

// =============================================================================
// Constants
// =============================================================================

/// Maximum number of tile columns.
pub const MAX_TILE_COLS: usize = 64;

/// Maximum number of tile rows.
pub const MAX_TILE_ROWS: usize = 64;

/// Maximum number of tiles.
pub const MAX_TILE_COUNT: usize = MAX_TILE_COLS * MAX_TILE_ROWS;

/// Maximum tile area in superblocks.
pub const MAX_TILE_AREA_SB: usize = 4096;

/// Minimum tile width in superblocks (for uniform tiles).
pub const MIN_TILE_WIDTH_SB: usize = 1;

/// Maximum tile width in superblocks.
pub const MAX_TILE_WIDTH_SB: usize = 64;

/// Minimum tile height in superblocks.
pub const MIN_TILE_HEIGHT_SB: usize = 1;

/// Maximum tile height in superblocks.
pub const MAX_TILE_HEIGHT_SB: usize = 64;

/// Tile size bytes field length (for non-last tiles).
pub const TILE_SIZE_BYTES_MINUS_1_BITS: u8 = 2;

// =============================================================================
// Structures
// =============================================================================

/// Tile information from frame header.
#[derive(Clone, Debug, Default)]
pub struct TileInfo {
    /// Number of tile columns.
    pub tile_cols: u32,
    /// Number of tile rows.
    pub tile_rows: u32,
    /// Tile column start positions (in superblocks).
    pub tile_col_starts: Vec<u32>,
    /// Tile row start positions (in superblocks).
    pub tile_row_starts: Vec<u32>,
    /// Context update tile ID.
    pub context_update_tile_id: u32,
    /// Number of bytes to read for tile size.
    pub tile_size_bytes: u8,
    /// Uniform tile spacing flag.
    pub uniform_tile_spacing: bool,
    /// Tile columns log2.
    pub tile_cols_log2: u8,
    /// Tile rows log2.
    pub tile_rows_log2: u8,
    /// Minimum tile columns log2.
    pub min_tile_cols_log2: u8,
    /// Maximum tile columns log2.
    pub max_tile_cols_log2: u8,
    /// Minimum tile rows log2.
    pub min_tile_rows_log2: u8,
    /// Maximum tile rows log2.
    pub max_tile_rows_log2: u8,
    /// Superblock columns.
    pub sb_cols: u32,
    /// Superblock rows.
    pub sb_rows: u32,
    /// Superblock size (64 or 128).
    pub sb_size: u32,
}

impl TileInfo {
    /// Create a new tile info with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse tile info from the bitstream.
    ///
    /// # Errors
    ///
    /// Returns error if the bitstream is malformed.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    pub fn parse(
        reader: &mut BitReader<'_>,
        _seq: &SequenceHeader,
        frame_size: &FrameSize,
    ) -> CodecResult<Self> {
        let mut tile_info = Self::new();

        // Determine superblock size
        tile_info.sb_size = 64; // Simplified: would come from sequence header

        // Calculate superblock dimensions
        tile_info.sb_cols = frame_size.sb_cols(tile_info.sb_size);
        tile_info.sb_rows = frame_size.sb_rows(tile_info.sb_size);

        // Calculate tile limits
        let max_tile_width_sb = MAX_TILE_WIDTH_SB.min(tile_info.sb_cols as usize);
        let max_tile_area_sb = MAX_TILE_AREA_SB;

        tile_info.min_tile_cols_log2 = Self::tile_log2(max_tile_width_sb as u32, tile_info.sb_cols);
        tile_info.max_tile_cols_log2 =
            Self::tile_log2(1, tile_info.sb_cols.min(MAX_TILE_COLS as u32));
        tile_info.max_tile_rows_log2 =
            Self::tile_log2(1, tile_info.sb_rows.min(MAX_TILE_ROWS as u32));

        // Calculate min_log2_tile_rows
        let min_log2_tiles = Self::tile_log2(
            max_tile_area_sb as u32,
            tile_info.sb_cols * tile_info.sb_rows,
        );
        tile_info.min_tile_rows_log2 = min_log2_tiles.saturating_sub(tile_info.max_tile_cols_log2);

        // Parse uniform tile spacing flag
        tile_info.uniform_tile_spacing = reader.read_bit().map_err(CodecError::Core)? != 0;

        if tile_info.uniform_tile_spacing {
            // Parse tile columns log2
            tile_info.tile_cols_log2 = tile_info.min_tile_cols_log2;
            while tile_info.tile_cols_log2 < tile_info.max_tile_cols_log2 {
                let increment = reader.read_bit().map_err(CodecError::Core)?;
                if increment != 0 {
                    tile_info.tile_cols_log2 += 1;
                } else {
                    break;
                }
            }

            // Calculate tile column starts
            let tile_width_sb = (tile_info.sb_cols + (1 << tile_info.tile_cols_log2) - 1)
                >> tile_info.tile_cols_log2;
            let mut start_sb = 0u32;
            tile_info.tile_col_starts.clear();
            while start_sb < tile_info.sb_cols {
                tile_info.tile_col_starts.push(start_sb);
                start_sb += tile_width_sb;
            }
            tile_info.tile_col_starts.push(tile_info.sb_cols);
            tile_info.tile_cols = (tile_info.tile_col_starts.len() - 1) as u32;

            // Parse tile rows log2
            tile_info.tile_rows_log2 = tile_info.min_tile_rows_log2;
            while tile_info.tile_rows_log2 < tile_info.max_tile_rows_log2 {
                let increment = reader.read_bit().map_err(CodecError::Core)?;
                if increment != 0 {
                    tile_info.tile_rows_log2 += 1;
                } else {
                    break;
                }
            }

            // Calculate tile row starts
            let tile_height_sb = (tile_info.sb_rows + (1 << tile_info.tile_rows_log2) - 1)
                >> tile_info.tile_rows_log2;
            let mut start_sb = 0u32;
            tile_info.tile_row_starts.clear();
            while start_sb < tile_info.sb_rows {
                tile_info.tile_row_starts.push(start_sb);
                start_sb += tile_height_sb;
            }
            tile_info.tile_row_starts.push(tile_info.sb_rows);
            tile_info.tile_rows = (tile_info.tile_row_starts.len() - 1) as u32;
        } else {
            // Non-uniform tile spacing
            let mut widest_tile_sb = 0u32;
            let mut start_sb = 0u32;
            tile_info.tile_col_starts.clear();

            while start_sb < tile_info.sb_cols {
                tile_info.tile_col_starts.push(start_sb);
                let max_width = tile_info.sb_cols - start_sb;
                let width_in_sbs_minus_1 = Self::ns(reader, max_width)?;
                let size_sb = width_in_sbs_minus_1 + 1;
                widest_tile_sb = widest_tile_sb.max(size_sb);
                start_sb += size_sb;
            }
            tile_info.tile_col_starts.push(tile_info.sb_cols);
            tile_info.tile_cols = (tile_info.tile_col_starts.len() - 1) as u32;
            tile_info.tile_cols_log2 = Self::tile_log2(1, tile_info.tile_cols);

            // Tile row starts (non-uniform)
            let mut start_sb = 0u32;
            tile_info.tile_row_starts.clear();

            while start_sb < tile_info.sb_rows {
                tile_info.tile_row_starts.push(start_sb);
                let max_height = tile_info.sb_rows - start_sb;
                let height_in_sbs_minus_1 = Self::ns(reader, max_height)?;
                let size_sb = height_in_sbs_minus_1 + 1;
                start_sb += size_sb;
            }
            tile_info.tile_row_starts.push(tile_info.sb_rows);
            tile_info.tile_rows = (tile_info.tile_row_starts.len() - 1) as u32;
            tile_info.tile_rows_log2 = Self::tile_log2(1, tile_info.tile_rows);
        }

        // Context update tile ID
        if tile_info.tile_cols_log2 > 0 || tile_info.tile_rows_log2 > 0 {
            let tile_bits = tile_info.tile_cols_log2 + tile_info.tile_rows_log2;
            tile_info.context_update_tile_id =
                reader.read_bits(tile_bits).map_err(CodecError::Core)? as u32;
            tile_info.tile_size_bytes = reader
                .read_bits(TILE_SIZE_BYTES_MINUS_1_BITS)
                .map_err(CodecError::Core)? as u8
                + 1;
        } else {
            tile_info.context_update_tile_id = 0;
            tile_info.tile_size_bytes = 1;
        }

        Ok(tile_info)
    }

    /// Calculate tile_log2 value.
    #[must_use]
    fn tile_log2(blk_size: u32, target: u32) -> u8 {
        let mut k = 0u8;
        while (blk_size << k) < target {
            k += 1;
        }
        k
    }

    /// Read an ns() (non-symmetric) value from the bitstream.
    #[allow(clippy::cast_possible_truncation)]
    fn ns(reader: &mut BitReader<'_>, n: u32) -> CodecResult<u32> {
        if n <= 1 {
            return Ok(0);
        }

        let w = 32 - (n - 1).leading_zeros();
        let m = (1u32 << w) - n;

        let v = reader.read_bits(w as u8 - 1).map_err(CodecError::Core)? as u32;
        if v < m {
            Ok(v)
        } else {
            let extra_bit = u32::from(reader.read_bit().map_err(CodecError::Core)?);
            Ok((v << 1) - m + extra_bit)
        }
    }

    /// Get total number of tiles.
    #[must_use]
    pub const fn tile_count(&self) -> u32 {
        self.tile_cols * self.tile_rows
    }

    /// Get tile dimensions in superblocks for a specific tile.
    #[must_use]
    pub fn tile_size_sb(&self, tile_col: u32, tile_row: u32) -> (u32, u32) {
        let col_start = self
            .tile_col_starts
            .get(tile_col as usize)
            .copied()
            .unwrap_or(0);
        let col_end = self
            .tile_col_starts
            .get(tile_col as usize + 1)
            .copied()
            .unwrap_or(col_start);
        let row_start = self
            .tile_row_starts
            .get(tile_row as usize)
            .copied()
            .unwrap_or(0);
        let row_end = self
            .tile_row_starts
            .get(tile_row as usize + 1)
            .copied()
            .unwrap_or(row_start);

        (col_end - col_start, row_end - row_start)
    }

    /// Get tile dimensions in pixels for a specific tile.
    #[must_use]
    pub fn tile_size_pixels(&self, tile_col: u32, tile_row: u32) -> (u32, u32) {
        let (width_sb, height_sb) = self.tile_size_sb(tile_col, tile_row);
        (width_sb * self.sb_size, height_sb * self.sb_size)
    }

    /// Get the tile index for a given (col, row) position.
    #[must_use]
    pub const fn tile_index(&self, tile_col: u32, tile_row: u32) -> u32 {
        tile_row * self.tile_cols + tile_col
    }

    /// Get the (col, row) position for a tile index.
    #[must_use]
    pub const fn tile_position(&self, tile_idx: u32) -> (u32, u32) {
        (tile_idx % self.tile_cols, tile_idx / self.tile_cols)
    }

    /// Get the starting superblock column for a tile column.
    #[must_use]
    pub fn tile_col_start_sb(&self, tile_col: u32) -> u32 {
        self.tile_col_starts
            .get(tile_col as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Get the starting superblock row for a tile row.
    #[must_use]
    pub fn tile_row_start_sb(&self, tile_row: u32) -> u32 {
        self.tile_row_starts
            .get(tile_row as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Get the pixel position for the start of a tile.
    #[must_use]
    pub fn tile_start_pixels(&self, tile_col: u32, tile_row: u32) -> (u32, u32) {
        (
            self.tile_col_start_sb(tile_col) * self.sb_size,
            self.tile_row_start_sb(tile_row) * self.sb_size,
        )
    }

    /// Check if this is a single-tile frame.
    #[must_use]
    pub const fn is_single_tile(&self) -> bool {
        self.tile_cols == 1 && self.tile_rows == 1
    }

    /// Check if a tile is at the left edge.
    #[must_use]
    pub const fn is_left_edge(&self, tile_col: u32) -> bool {
        tile_col == 0
    }

    /// Check if a tile is at the right edge.
    #[must_use]
    pub fn is_right_edge(&self, tile_col: u32) -> bool {
        tile_col == self.tile_cols - 1
    }

    /// Check if a tile is at the top edge.
    #[must_use]
    pub const fn is_top_edge(&self, tile_row: u32) -> bool {
        tile_row == 0
    }

    /// Check if a tile is at the bottom edge.
    #[must_use]
    pub fn is_bottom_edge(&self, tile_row: u32) -> bool {
        tile_row == self.tile_rows - 1
    }
}

/// Tile configuration for backward compatibility.
pub type TileConfig = TileInfo;

/// Tile data reference for decoding.
#[derive(Clone, Debug)]
pub struct TileData {
    /// Tile column index.
    pub tile_col: u32,
    /// Tile row index.
    pub tile_row: u32,
    /// Tile data offset in bitstream.
    pub offset: usize,
    /// Tile data size in bytes.
    pub size: usize,
    /// Tile index in scan order.
    pub tile_idx: u32,
}

impl TileData {
    /// Create a new tile data reference.
    #[must_use]
    pub fn new(tile_col: u32, tile_row: u32, offset: usize, size: usize, tile_cols: u32) -> Self {
        Self {
            tile_col,
            tile_row,
            offset,
            size,
            tile_idx: tile_row * tile_cols + tile_col,
        }
    }

    /// Check if this tile data is valid.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.size > 0
    }
}

/// Tile group header and data.
#[derive(Clone, Debug)]
pub struct TileGroup {
    /// Starting tile index (inclusive).
    pub tile_start: u32,
    /// Ending tile index (inclusive).
    pub tile_end: u32,
    /// Tile data entries.
    pub tiles: Vec<TileData>,
    /// Number of tiles signaled.
    pub num_tiles: u32,
}

impl TileGroup {
    /// Create a new tile group.
    #[must_use]
    pub fn new(tile_start: u32, tile_end: u32) -> Self {
        Self {
            tile_start,
            tile_end,
            tiles: Vec::new(),
            num_tiles: tile_end - tile_start + 1,
        }
    }

    /// Get number of tiles in this group.
    #[must_use]
    pub fn tile_count(&self) -> u32 {
        self.tile_end - self.tile_start + 1
    }

    /// Add a tile to this group.
    pub fn add_tile(&mut self, tile: TileData) {
        self.tiles.push(tile);
    }

    /// Get a tile by its index within the group.
    #[must_use]
    pub fn get_tile(&self, idx: usize) -> Option<&TileData> {
        self.tiles.get(idx)
    }

    /// Check if this group contains a specific tile index.
    #[must_use]
    pub const fn contains_tile(&self, tile_idx: u32) -> bool {
        tile_idx >= self.tile_start && tile_idx <= self.tile_end
    }

    /// Check if this is a single-tile group.
    #[must_use]
    pub const fn is_single_tile(&self) -> bool {
        self.tile_start == self.tile_end
    }
}

/// Tile group OBU parser.
#[derive(Clone, Debug)]
pub struct TileGroupObu {
    /// Tile info from frame header.
    pub tile_info: TileInfo,
    /// Tile groups in this OBU.
    pub groups: Vec<TileGroup>,
}

impl TileGroupObu {
    /// Create a new tile group OBU parser.
    #[must_use]
    pub fn new(tile_info: TileInfo) -> Self {
        Self {
            tile_info,
            groups: Vec::new(),
        }
    }

    /// Parse tile group data from OBU payload.
    ///
    /// # Errors
    ///
    /// Returns error if the bitstream is malformed.
    #[allow(clippy::cast_possible_truncation)]
    pub fn parse(&mut self, data: &[u8]) -> CodecResult<()> {
        let mut reader = BitReader::new(data);
        let num_tiles = self.tile_info.tile_count();

        // Parse tile group header
        let (tile_start, tile_end) = if num_tiles > 1 {
            let tile_bits = self.tile_info.tile_cols_log2 + self.tile_info.tile_rows_log2;
            let tile_start = reader.read_bits(tile_bits).map_err(CodecError::Core)? as u32;
            let tile_end = reader.read_bits(tile_bits).map_err(CodecError::Core)? as u32;
            (tile_start, tile_end)
        } else {
            (0, 0)
        };

        let mut group = TileGroup::new(tile_start, tile_end);

        // Byte align
        reader.byte_align();

        // Parse tile data
        let header_bytes = reader.bits_read().div_ceil(8);
        let mut offset = header_bytes;

        for tile_idx in tile_start..=tile_end {
            let tile_size = if tile_idx < tile_end {
                // Read tile size (little-endian, tile_size_bytes bytes)
                let size_bytes = self.tile_info.tile_size_bytes as usize;
                let mut size = 0u32;
                for i in 0..size_bytes {
                    if offset + i >= data.len() {
                        return Err(CodecError::InvalidBitstream(
                            "Tile data truncated".to_string(),
                        ));
                    }
                    size |= u32::from(data[offset + i]) << (8 * i);
                }
                offset += size_bytes;
                (size + 1) as usize
            } else {
                // Last tile uses remaining data
                data.len() - offset
            };

            let (tile_col, tile_row) = self.tile_info.tile_position(tile_idx);
            let tile_data = TileData::new(
                tile_col,
                tile_row,
                offset,
                tile_size,
                self.tile_info.tile_cols,
            );

            group.add_tile(tile_data);
            offset += tile_size;
        }

        self.groups.push(group);
        Ok(())
    }

    /// Get total number of tiles across all groups.
    #[must_use]
    pub fn total_tiles(&self) -> usize {
        self.groups.iter().map(|g| g.tiles.len()).sum()
    }

    /// Get a specific tile by global index.
    #[must_use]
    pub fn get_tile(&self, tile_idx: u32) -> Option<&TileData> {
        for group in &self.groups {
            if group.contains_tile(tile_idx) {
                let local_idx = (tile_idx - group.tile_start) as usize;
                return group.get_tile(local_idx);
            }
        }
        None
    }

    /// Check if all tiles have been received.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        let total_expected = self.tile_info.tile_count() as usize;
        self.total_tiles() == total_expected
    }
}

/// Tile decoder state for a single tile.
#[derive(Clone, Debug, Default)]
pub struct TileDecoderState {
    /// Tile column.
    pub tile_col: u32,
    /// Tile row.
    pub tile_row: u32,
    /// Current superblock column within tile.
    pub sb_col: u32,
    /// Current superblock row within tile.
    pub sb_row: u32,
    /// Tile width in superblocks.
    pub tile_width_sb: u32,
    /// Tile height in superblocks.
    pub tile_height_sb: u32,
    /// Decoding completed for this tile.
    pub completed: bool,
}

impl TileDecoderState {
    /// Create a new tile decoder state.
    #[must_use]
    pub fn new(tile_info: &TileInfo, tile_col: u32, tile_row: u32) -> Self {
        let (width, height) = tile_info.tile_size_sb(tile_col, tile_row);
        Self {
            tile_col,
            tile_row,
            sb_col: 0,
            sb_row: 0,
            tile_width_sb: width,
            tile_height_sb: height,
            completed: false,
        }
    }

    /// Advance to the next superblock.
    pub fn advance(&mut self) {
        self.sb_col += 1;
        if self.sb_col >= self.tile_width_sb {
            self.sb_col = 0;
            self.sb_row += 1;
            if self.sb_row >= self.tile_height_sb {
                self.completed = true;
            }
        }
    }

    /// Check if decoding is complete.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.completed
    }

    /// Get the current position as (`sb_col`, `sb_row`).
    #[must_use]
    pub const fn position(&self) -> (u32, u32) {
        (self.sb_col, self.sb_row)
    }

    /// Get total superblocks in this tile.
    #[must_use]
    pub const fn total_superblocks(&self) -> u32 {
        self.tile_width_sb * self.tile_height_sb
    }

    /// Get number of decoded superblocks.
    #[must_use]
    pub fn decoded_superblocks(&self) -> u32 {
        self.sb_row * self.tile_width_sb + self.sb_col
    }
}

/// Multi-tile decoder coordinator.
#[derive(Clone, Debug)]
pub struct TileDecoder {
    /// Tile info.
    pub tile_info: TileInfo,
    /// Tile states.
    pub tile_states: Vec<TileDecoderState>,
}

impl TileDecoder {
    /// Create a new tile decoder.
    #[must_use]
    pub fn new(tile_info: TileInfo) -> Self {
        let mut tile_states = Vec::with_capacity(tile_info.tile_count() as usize);
        for row in 0..tile_info.tile_rows {
            for col in 0..tile_info.tile_cols {
                tile_states.push(TileDecoderState::new(&tile_info, col, row));
            }
        }
        Self {
            tile_info,
            tile_states,
        }
    }

    /// Get a tile state by index.
    #[must_use]
    pub fn get_state(&self, idx: usize) -> Option<&TileDecoderState> {
        self.tile_states.get(idx)
    }

    /// Get a mutable tile state by index.
    pub fn get_state_mut(&mut self, idx: usize) -> Option<&mut TileDecoderState> {
        self.tile_states.get_mut(idx)
    }

    /// Check if all tiles are complete.
    #[must_use]
    pub fn all_complete(&self) -> bool {
        self.tile_states.iter().all(TileDecoderState::is_complete)
    }

    /// Get number of complete tiles.
    #[must_use]
    pub fn complete_count(&self) -> usize {
        self.tile_states.iter().filter(|s| s.is_complete()).count()
    }

    /// Reset all tile states.
    pub fn reset(&mut self) {
        for state in &mut self.tile_states {
            state.sb_col = 0;
            state.sb_row = 0;
            state.completed = false;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tile_info() -> TileInfo {
        TileInfo {
            tile_cols: 2,
            tile_rows: 2,
            tile_col_starts: vec![0, 15, 30],
            tile_row_starts: vec![0, 8, 17],
            context_update_tile_id: 0,
            tile_size_bytes: 4,
            uniform_tile_spacing: true,
            tile_cols_log2: 1,
            tile_rows_log2: 1,
            min_tile_cols_log2: 0,
            max_tile_cols_log2: 2,
            min_tile_rows_log2: 0,
            max_tile_rows_log2: 2,
            sb_cols: 30,
            sb_rows: 17,
            sb_size: 64,
        }
    }

    #[test]
    fn test_tile_info_default() {
        let tile_info = TileInfo::default();
        assert_eq!(tile_info.tile_cols, 0);
        assert_eq!(tile_info.tile_rows, 0);
    }

    #[test]
    fn test_tile_count() {
        let tile_info = create_test_tile_info();
        assert_eq!(tile_info.tile_count(), 4);
    }

    #[test]
    fn test_tile_size_sb() {
        let tile_info = create_test_tile_info();
        assert_eq!(tile_info.tile_size_sb(0, 0), (15, 8));
        assert_eq!(tile_info.tile_size_sb(1, 0), (15, 8));
        assert_eq!(tile_info.tile_size_sb(0, 1), (15, 9));
        assert_eq!(tile_info.tile_size_sb(1, 1), (15, 9));
    }

    #[test]
    fn test_tile_size_pixels() {
        let tile_info = create_test_tile_info();
        let (width, height) = tile_info.tile_size_pixels(0, 0);
        assert_eq!(width, 15 * 64);
        assert_eq!(height, 8 * 64);
    }

    #[test]
    fn test_tile_index() {
        let tile_info = create_test_tile_info();
        assert_eq!(tile_info.tile_index(0, 0), 0);
        assert_eq!(tile_info.tile_index(1, 0), 1);
        assert_eq!(tile_info.tile_index(0, 1), 2);
        assert_eq!(tile_info.tile_index(1, 1), 3);
    }

    #[test]
    fn test_tile_position() {
        let tile_info = create_test_tile_info();
        assert_eq!(tile_info.tile_position(0), (0, 0));
        assert_eq!(tile_info.tile_position(1), (1, 0));
        assert_eq!(tile_info.tile_position(2), (0, 1));
        assert_eq!(tile_info.tile_position(3), (1, 1));
    }

    #[test]
    fn test_tile_start_sb() {
        let tile_info = create_test_tile_info();
        assert_eq!(tile_info.tile_col_start_sb(0), 0);
        assert_eq!(tile_info.tile_col_start_sb(1), 15);
        assert_eq!(tile_info.tile_row_start_sb(0), 0);
        assert_eq!(tile_info.tile_row_start_sb(1), 8);
    }

    #[test]
    fn test_tile_start_pixels() {
        let tile_info = create_test_tile_info();
        assert_eq!(tile_info.tile_start_pixels(0, 0), (0, 0));
        assert_eq!(tile_info.tile_start_pixels(1, 0), (15 * 64, 0));
        assert_eq!(tile_info.tile_start_pixels(0, 1), (0, 8 * 64));
    }

    #[test]
    fn test_is_single_tile() {
        let mut tile_info = TileInfo::default();
        tile_info.tile_cols = 1;
        tile_info.tile_rows = 1;
        assert!(tile_info.is_single_tile());

        tile_info.tile_cols = 2;
        assert!(!tile_info.is_single_tile());
    }

    #[test]
    fn test_tile_edges() {
        let tile_info = create_test_tile_info();

        assert!(tile_info.is_left_edge(0));
        assert!(!tile_info.is_left_edge(1));
        assert!(!tile_info.is_right_edge(0));
        assert!(tile_info.is_right_edge(1));
        assert!(tile_info.is_top_edge(0));
        assert!(!tile_info.is_top_edge(1));
        assert!(!tile_info.is_bottom_edge(0));
        assert!(tile_info.is_bottom_edge(1));
    }

    #[test]
    fn test_tile_log2() {
        assert_eq!(TileInfo::tile_log2(1, 1), 0);
        assert_eq!(TileInfo::tile_log2(1, 2), 1);
        assert_eq!(TileInfo::tile_log2(1, 4), 2);
        assert_eq!(TileInfo::tile_log2(2, 4), 1);
    }

    #[test]
    fn test_tile_data() {
        let tile_data = TileData::new(1, 2, 100, 500, 4);
        assert_eq!(tile_data.tile_col, 1);
        assert_eq!(tile_data.tile_row, 2);
        assert_eq!(tile_data.offset, 100);
        assert_eq!(tile_data.size, 500);
        assert_eq!(tile_data.tile_idx, 2 * 4 + 1);
        assert!(tile_data.is_valid());

        let empty_tile = TileData::new(0, 0, 0, 0, 1);
        assert!(!empty_tile.is_valid());
    }

    #[test]
    fn test_tile_group() {
        let mut group = TileGroup::new(0, 3);
        assert_eq!(group.tile_count(), 4);
        assert!(!group.is_single_tile());

        assert!(group.contains_tile(0));
        assert!(group.contains_tile(3));
        assert!(!group.contains_tile(4));

        group.add_tile(TileData::new(0, 0, 0, 100, 2));
        assert_eq!(group.tiles.len(), 1);
        assert!(group.get_tile(0).is_some());
    }

    #[test]
    fn test_tile_group_single() {
        let group = TileGroup::new(5, 5);
        assert_eq!(group.tile_count(), 1);
        assert!(group.is_single_tile());
    }

    #[test]
    fn test_tile_group_obu() {
        let tile_info = create_test_tile_info();
        let obu = TileGroupObu::new(tile_info);

        assert_eq!(obu.total_tiles(), 0);
        assert!(!obu.is_complete());
    }

    #[test]
    fn test_tile_decoder_state() {
        let tile_info = create_test_tile_info();
        let mut state = TileDecoderState::new(&tile_info, 0, 0);

        assert_eq!(state.tile_col, 0);
        assert_eq!(state.tile_row, 0);
        assert_eq!(state.position(), (0, 0));
        assert!(!state.is_complete());
        assert_eq!(state.tile_width_sb, 15);
        assert_eq!(state.tile_height_sb, 8);
        assert_eq!(state.total_superblocks(), 120);

        state.advance();
        assert_eq!(state.position(), (1, 0));
        assert_eq!(state.decoded_superblocks(), 1);
    }

    #[test]
    fn test_tile_decoder_state_wrap() {
        let tile_info = TileInfo {
            tile_cols: 1,
            tile_rows: 1,
            tile_col_starts: vec![0, 2],
            tile_row_starts: vec![0, 2],
            sb_cols: 2,
            sb_rows: 2,
            sb_size: 64,
            ..Default::default()
        };

        let mut state = TileDecoderState::new(&tile_info, 0, 0);
        assert_eq!(state.tile_width_sb, 2);
        assert_eq!(state.tile_height_sb, 2);

        state.advance();
        assert_eq!(state.position(), (1, 0));

        state.advance();
        assert_eq!(state.position(), (0, 1));

        state.advance();
        assert_eq!(state.position(), (1, 1));

        state.advance();
        assert!(state.is_complete());
    }

    #[test]
    fn test_tile_decoder() {
        let tile_info = create_test_tile_info();
        let decoder = TileDecoder::new(tile_info);

        assert_eq!(decoder.tile_states.len(), 4);
        assert!(!decoder.all_complete());
        assert_eq!(decoder.complete_count(), 0);

        assert!(decoder.get_state(0).is_some());
        assert!(decoder.get_state(3).is_some());
        assert!(decoder.get_state(4).is_none());
    }

    #[test]
    fn test_tile_decoder_reset() {
        let tile_info = create_test_tile_info();
        let mut decoder = TileDecoder::new(tile_info);

        // Advance some tiles
        if let Some(state) = decoder.get_state_mut(0) {
            state.advance();
            state.advance();
        }

        decoder.reset();

        for state in &decoder.tile_states {
            assert_eq!(state.position(), (0, 0));
            assert!(!state.is_complete());
        }
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_TILE_COLS, 64);
        assert_eq!(MAX_TILE_ROWS, 64);
        assert_eq!(MAX_TILE_COUNT, 4096);
        assert_eq!(MAX_TILE_AREA_SB, 4096);
    }
}
