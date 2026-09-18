//! Parallel tile decoding for AV1 frames using rayon.
//!
//! AV1 divides frames into independent rectangular tiles that can be decoded
//! concurrently.  This module provides [`ParallelTileDecoder`] which splits
//! a raw frame buffer into [`TileJob`]s, decodes them in parallel via rayon,
//! and re-assembles the results into a planar YUV 4:2:0 output buffer.
//!
//! # Structural Implementation
//!
//! A full AV1 tile decode requires entropy decoding, prediction, inverse
//! transforms, and loop filtering — all of which depend on codec state that
//! lives in the outer decoder.  This module provides the *structural*
//! scaffolding: correct splitting, parallel dispatch, and frame assembly.
//! Each tile's "decode" step currently copies the raw tile bytes as a
//! stand-in for the real decode pass.
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::av1::{ParallelTileDecoder, TileJob};
//!
//! let decoder = ParallelTileDecoder::new(1920, 1080, 4, 2);
//! let frame_data = vec![0u8; 1920 * 1080]; // synthetic luma
//! let tiles = decoder.split_into_tiles(&frame_data);
//! assert_eq!(tiles.len(), 8);
//! let output = decoder.decode_tiles_parallel(tiles).expect("decode");
//! assert_eq!(output.len(), 1920 * 1080 * 3 / 2);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::missing_errors_doc)]

use rayon::prelude::*;

use crate::error::{CodecError, CodecResult};

// =============================================================================
// Public types
// =============================================================================

/// A single tile extracted from a frame, ready for independent decoding.
#[derive(Clone, Debug)]
pub struct TileJob {
    /// Tile row index (0-based, top to bottom).
    pub tile_row: u32,
    /// Tile column index (0-based, left to right).
    pub tile_col: u32,
    /// Raw tile data (luma bytes for this tile region).
    pub tile_data: Vec<u8>,
    /// Pixel offset of this tile's top-left corner in the frame: `(x, y)`.
    pub tile_offset: (u32, u32),
    /// Pixel dimensions of this tile: `(width, height)`.
    pub tile_size: (u32, u32),
}

/// Parallel tile decoder for AV1 frames.
///
/// Divides a frame into a grid of `tile_cols × tile_rows` tiles, decodes them
/// concurrently with rayon, and assembles the results into a planar YUV 4:2:0
/// buffer.
#[derive(Clone, Debug)]
pub struct ParallelTileDecoder {
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Number of tile columns.
    pub tile_cols: u32,
    /// Number of tile rows.
    pub tile_rows: u32,
}

// =============================================================================
// Implementation
// =============================================================================

impl ParallelTileDecoder {
    /// Create a new `ParallelTileDecoder`.
    ///
    /// # Panics
    ///
    /// Does not panic; use [`Self::decode_tiles_parallel`] error return for
    /// invalid configurations.
    pub fn new(frame_width: u32, frame_height: u32, tile_cols: u32, tile_rows: u32) -> Self {
        Self {
            frame_width,
            frame_height,
            tile_cols,
            tile_rows,
        }
    }

    /// Split `frame_data` (luma plane bytes, row-major) into [`TileJob`]s.
    ///
    /// The frame is divided into a `tile_cols × tile_rows` grid.  The last
    /// tile column and row absorb any remainder pixels.
    ///
    /// `frame_data` is interpreted as a contiguous luma plane of
    /// `frame_width × frame_height` bytes.  If `frame_data` is shorter than
    /// the expected luma plane size the available bytes are distributed
    /// proportionally across tiles.
    pub fn split_into_tiles(&self, frame_data: &[u8]) -> Vec<TileJob> {
        if self.tile_cols == 0 || self.tile_rows == 0 {
            return Vec::new();
        }

        let base_tile_w = self.frame_width / self.tile_cols;
        let base_tile_h = self.frame_height / self.tile_rows;
        let rem_w = self.frame_width % self.tile_cols;
        let rem_h = self.frame_height % self.tile_rows;

        let total_tiles = (self.tile_rows * self.tile_cols) as usize;
        let mut jobs = Vec::with_capacity(total_tiles);

        for row in 0..self.tile_rows {
            let tile_h = if row == self.tile_rows - 1 {
                base_tile_h + rem_h
            } else {
                base_tile_h
            };
            let y_offset = row * base_tile_h;

            for col in 0..self.tile_cols {
                let tile_w = if col == self.tile_cols - 1 {
                    base_tile_w + rem_w
                } else {
                    base_tile_w
                };
                let x_offset = col * base_tile_w;

                // Extract the luma bytes belonging to this tile region.
                let tile_bytes = Self::extract_tile_bytes(
                    frame_data,
                    self.frame_width,
                    x_offset,
                    y_offset,
                    tile_w,
                    tile_h,
                );

                jobs.push(TileJob {
                    tile_row: row,
                    tile_col: col,
                    tile_data: tile_bytes,
                    tile_offset: (x_offset, y_offset),
                    tile_size: (tile_w, tile_h),
                });
            }
        }

        jobs
    }

    /// Decode all tiles in parallel using rayon and assemble the output frame.
    ///
    /// Returns a planar YUV 4:2:0 buffer of length
    /// `frame_width × frame_height × 3 / 2`.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` when the tile grid dimensions
    /// are zero, or `CodecError::InvalidBitstream` when an individual tile
    /// fails to decode.
    pub fn decode_tiles_parallel(&self, tiles: Vec<TileJob>) -> CodecResult<Vec<u8>> {
        if self.tile_cols == 0 || self.tile_rows == 0 {
            return Err(CodecError::InvalidParameter(
                "tile_cols and tile_rows must be non-zero".to_string(),
            ));
        }

        // Decode tiles in parallel; collect (job, decoded_bytes) pairs.
        let results: Result<Vec<(TileJob, Vec<u8>)>, CodecError> = tiles
            .into_par_iter()
            .map(|job| {
                let decoded = Self::decode_single_tile(&job)?;
                Ok((job, decoded))
            })
            .collect();

        let tile_outputs = results?;
        Ok(self.assemble_frame(&tile_outputs))
    }

    /// Assemble decoded tile outputs into a full planar YUV 4:2:0 frame.
    ///
    /// The luma (`Y`) plane is filled from each tile's decoded bytes.
    /// The chroma (`Cb`, `Cr`) planes are zeroed (neutral grey), which is
    /// appropriate for a structural pass that does not yet decode chroma.
    ///
    /// Returns a buffer of `frame_width × frame_height × 3 / 2` bytes:
    /// - bytes `[0 .. W*H)` — luma
    /// - bytes `[W*H .. W*H + W*H/4)` — Cb (zeroed)
    /// - bytes `[W*H + W*H/4 .. W*H*3/2)` — Cr (zeroed)
    pub fn assemble_frame(&self, tile_outputs: &[(TileJob, Vec<u8>)]) -> Vec<u8> {
        let luma_size = (self.frame_width * self.frame_height) as usize;
        let chroma_size = luma_size / 4;
        let total_size = luma_size + 2 * chroma_size;

        let mut frame = vec![0u8; total_size];
        let luma_plane = &mut frame[..luma_size];

        for (job, decoded) in tile_outputs {
            let (x_off, y_off) = job.tile_offset;
            let (tile_w, tile_h) = job.tile_size;

            // Copy decoded luma bytes row by row into the correct frame region.
            for row in 0..tile_h {
                let src_row_start = (row * tile_w) as usize;
                let src_row_end = src_row_start + tile_w as usize;

                let dst_row_start = ((y_off + row) * self.frame_width + x_off) as usize;
                let dst_row_end = dst_row_start + tile_w as usize;

                // Guard against decoded buffer being shorter than expected
                // (e.g. structural stub returning fewer bytes than tile area).
                let src_available = decoded.len().saturating_sub(src_row_start);
                if src_available == 0 {
                    continue;
                }
                let copy_len = (src_row_end - src_row_start).min(src_available);

                if dst_row_end <= luma_plane.len() {
                    luma_plane[dst_row_start..dst_row_start + copy_len]
                        .copy_from_slice(&decoded[src_row_start..src_row_start + copy_len]);
                }
            }
        }

        // Chroma planes remain zero-initialised (neutral 4:2:0 chroma).
        frame
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Extract the luma bytes for a tile region from a row-major frame buffer.
    fn extract_tile_bytes(
        frame_data: &[u8],
        frame_width: u32,
        x_offset: u32,
        y_offset: u32,
        tile_w: u32,
        tile_h: u32,
    ) -> Vec<u8> {
        let mut bytes = Vec::with_capacity((tile_w * tile_h) as usize);

        for row in 0..tile_h {
            let src_start = ((y_offset + row) * frame_width + x_offset) as usize;
            let src_end = src_start + tile_w as usize;

            if src_start >= frame_data.len() {
                // Pad remaining rows with zeros when input is shorter.
                bytes.extend(std::iter::repeat_n(0u8, tile_w as usize));
            } else {
                let available_end = src_end.min(frame_data.len());
                bytes.extend_from_slice(&frame_data[src_start..available_end]);
                if available_end < src_end {
                    bytes.extend(std::iter::repeat_n(0u8, src_end - available_end));
                }
            }
        }

        bytes
    }

    /// Structural single-tile decode.
    ///
    /// For now this copies the input tile bytes as the "decoded" output.  A
    /// real implementation would invoke the AV1 entropy / prediction /
    /// transform pipeline here.
    fn decode_single_tile(job: &TileJob) -> CodecResult<Vec<u8>> {
        // Structural pass: validate minimum size and return a copy.
        let (tile_w, tile_h) = job.tile_size;
        if tile_w == 0 || tile_h == 0 {
            return Err(CodecError::InvalidBitstream(format!(
                "Tile ({}, {}) has zero dimension: {}×{}",
                job.tile_col, job.tile_row, tile_w, tile_h
            )));
        }
        // Return the raw tile bytes as decoded output.
        Ok(job.tile_data.clone())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_decoder_4x2() -> ParallelTileDecoder {
        ParallelTileDecoder::new(1920, 1080, 4, 2)
    }

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_stores_dimensions() {
        let dec = ParallelTileDecoder::new(3840, 2160, 8, 4);
        assert_eq!(dec.frame_width, 3840);
        assert_eq!(dec.frame_height, 2160);
        assert_eq!(dec.tile_cols, 8);
        assert_eq!(dec.tile_rows, 4);
    }

    // -----------------------------------------------------------------------
    // split_into_tiles
    // -----------------------------------------------------------------------

    #[test]
    fn test_split_tile_count() {
        let dec = make_decoder_4x2();
        let frame = vec![0u8; 1920 * 1080];
        let tiles = dec.split_into_tiles(&frame);
        assert_eq!(tiles.len(), 8, "4 cols × 2 rows = 8 tiles");
    }

    #[test]
    fn test_split_tile_offsets() {
        let dec = make_decoder_4x2();
        let frame = vec![0u8; 1920 * 1080];
        let tiles = dec.split_into_tiles(&frame);

        // First tile is always at (0, 0)
        let first = tiles
            .iter()
            .find(|t| t.tile_row == 0 && t.tile_col == 0)
            .expect("tile (0,0)");
        assert_eq!(first.tile_offset, (0, 0));

        // Second column of first row
        let t01 = tiles
            .iter()
            .find(|t| t.tile_row == 0 && t.tile_col == 1)
            .expect("tile (0,1)");
        assert_eq!(t01.tile_offset.1, 0, "y offset of row 0 should be 0");
        assert!(t01.tile_offset.0 > 0, "x offset of col 1 should be > 0");

        // Second row
        let t10 = tiles
            .iter()
            .find(|t| t.tile_row == 1 && t.tile_col == 0)
            .expect("tile (1,0)");
        assert!(t10.tile_offset.1 > 0, "y offset of row 1 should be > 0");
    }

    #[test]
    fn test_split_tile_sizes_sum_to_frame() {
        // Use a size where width and height are exact multiples.
        let dec = ParallelTileDecoder::new(800, 600, 4, 3);
        let frame = vec![0u8; 800 * 600];
        let tiles = dec.split_into_tiles(&frame);
        assert_eq!(tiles.len(), 12);

        // All tiles in col 0 should have the same width.
        let col0_widths: Vec<u32> = tiles
            .iter()
            .filter(|t| t.tile_col == 0)
            .map(|t| t.tile_size.0)
            .collect();
        assert!(col0_widths.iter().all(|&w| w == col0_widths[0]));

        // Widths of all tiles in row 0 should sum to frame_width.
        let row0_width_sum: u32 = tiles
            .iter()
            .filter(|t| t.tile_row == 0)
            .map(|t| t.tile_size.0)
            .sum();
        assert_eq!(row0_width_sum, 800);

        // Heights of all tiles in col 0 should sum to frame_height.
        let col0_height_sum: u32 = tiles
            .iter()
            .filter(|t| t.tile_col == 0)
            .map(|t| t.tile_size.1)
            .sum();
        assert_eq!(col0_height_sum, 600);
    }

    #[test]
    fn test_split_handles_non_divisible_dimensions() {
        // 1000 / 3 = 333 rem 1; 700 / 2 = 350 rem 0
        let dec = ParallelTileDecoder::new(1000, 700, 3, 2);
        let frame = vec![0u8; 1000 * 700];
        let tiles = dec.split_into_tiles(&frame);
        assert_eq!(tiles.len(), 6);

        let row0_width_sum: u32 = tiles
            .iter()
            .filter(|t| t.tile_row == 0)
            .map(|t| t.tile_size.0)
            .sum();
        assert_eq!(row0_width_sum, 1000, "widths must cover full frame width");

        let col0_height_sum: u32 = tiles
            .iter()
            .filter(|t| t.tile_col == 0)
            .map(|t| t.tile_size.1)
            .sum();
        assert_eq!(col0_height_sum, 700, "heights must cover full frame height");
    }

    #[test]
    fn test_split_zero_cols_returns_empty() {
        let dec = ParallelTileDecoder::new(1920, 1080, 0, 2);
        let frame = vec![0u8; 100];
        let tiles = dec.split_into_tiles(&frame);
        assert!(tiles.is_empty());
    }

    #[test]
    fn test_split_tile_data_length_matches_tile_area() {
        let dec = ParallelTileDecoder::new(400, 300, 2, 2);
        let frame = vec![0xAAu8; 400 * 300];
        let tiles = dec.split_into_tiles(&frame);
        for tile in &tiles {
            let expected_len = (tile.tile_size.0 * tile.tile_size.1) as usize;
            assert_eq!(
                tile.tile_data.len(),
                expected_len,
                "tile ({},{}) data length mismatch",
                tile.tile_row,
                tile.tile_col
            );
        }
    }

    #[test]
    fn test_split_preserves_pixel_values() {
        // Mark each pixel with a unique value to verify correct extraction.
        let width = 4u32;
        let height = 4u32;
        let frame: Vec<u8> = (0..(width * height) as u8).collect();

        let dec = ParallelTileDecoder::new(width, height, 2, 2);
        let tiles = dec.split_into_tiles(&frame);

        // Top-left tile (col=0, row=0) covers pixels [0,1] × [0,1]
        let tl = tiles
            .iter()
            .find(|t| t.tile_row == 0 && t.tile_col == 0)
            .expect("tl");
        assert_eq!(tl.tile_data[0], frame[0]); // (0,0)
        assert_eq!(tl.tile_data[1], frame[1]); // (0,1)
        assert_eq!(tl.tile_data[2], frame[width as usize]); // (1,0)
    }

    // -----------------------------------------------------------------------
    // decode_tiles_parallel
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_tiles_parallel_output_size() {
        let dec = make_decoder_4x2();
        let frame = vec![0u8; 1920 * 1080];
        let tiles = dec.split_into_tiles(&frame);
        let output = dec.decode_tiles_parallel(tiles).expect("decode");
        assert_eq!(output.len(), 1920 * 1080 * 3 / 2, "YUV 4:2:0 output size");
    }

    #[test]
    fn test_decode_tiles_parallel_single_tile() {
        let dec = ParallelTileDecoder::new(320, 240, 1, 1);
        let frame = vec![0x7Fu8; 320 * 240];
        let tiles = dec.split_into_tiles(&frame);
        let output = dec.decode_tiles_parallel(tiles).expect("decode");
        assert_eq!(output.len(), 320 * 240 * 3 / 2);
        // Luma plane should match the input (structural pass copies bytes).
        assert!(output[..320 * 240].iter().all(|&b| b == 0x7F));
    }

    #[test]
    fn test_decode_tiles_parallel_zero_cols_errors() {
        let dec = ParallelTileDecoder::new(640, 480, 0, 2);
        let result = dec.decode_tiles_parallel(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_tiles_parallel_preserves_content() {
        // Encode row index into luma bytes; verify round-trip through decode+assemble.
        let width = 64u32;
        let height = 32u32;
        let mut frame = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                frame[(y * width + x) as usize] = (y % 256) as u8;
            }
        }

        let dec = ParallelTileDecoder::new(width, height, 4, 2);
        let tiles = dec.split_into_tiles(&frame);
        let output = dec.decode_tiles_parallel(tiles).expect("decode");

        // Verify luma plane content matches original.
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                assert_eq!(output[idx], frame[idx], "luma mismatch at ({x},{y})");
            }
        }
    }

    // -----------------------------------------------------------------------
    // assemble_frame
    // -----------------------------------------------------------------------

    #[test]
    fn test_assemble_frame_output_size() {
        let dec = ParallelTileDecoder::new(640, 480, 2, 2);
        let tile_outputs: Vec<(TileJob, Vec<u8>)> = Vec::new();
        let frame = dec.assemble_frame(&tile_outputs);
        assert_eq!(frame.len(), 640 * 480 * 3 / 2);
    }

    #[test]
    fn test_assemble_frame_chroma_zeroed() {
        let dec = ParallelTileDecoder::new(64, 32, 1, 1);
        let luma_size = 64 * 32;
        let job = TileJob {
            tile_row: 0,
            tile_col: 0,
            tile_data: vec![0xFFu8; luma_size],
            tile_offset: (0, 0),
            tile_size: (64, 32),
        };
        let tile_outputs = vec![(job, vec![0xFFu8; luma_size])];
        let frame = dec.assemble_frame(&tile_outputs);

        // Luma should be all 0xFF.
        assert!(frame[..luma_size].iter().all(|&b| b == 0xFF));
        // Chroma should be all zeros.
        assert!(frame[luma_size..].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_assemble_frame_single_tile_full_coverage() {
        let dec = ParallelTileDecoder::new(8, 4, 1, 1);
        let tile_bytes: Vec<u8> = (0..32u8).collect();
        let job = TileJob {
            tile_row: 0,
            tile_col: 0,
            tile_data: tile_bytes.clone(),
            tile_offset: (0, 0),
            tile_size: (8, 4),
        };
        let tile_outputs = vec![(job, tile_bytes.clone())];
        let frame = dec.assemble_frame(&tile_outputs);

        for (i, &expected) in tile_bytes.iter().enumerate() {
            assert_eq!(frame[i], expected, "luma byte {i} mismatch");
        }
    }
}
