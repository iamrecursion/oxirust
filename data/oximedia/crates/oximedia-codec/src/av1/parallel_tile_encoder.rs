// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Parallel AV1 tile encoding over raw frame bytes.
//!
//! This module provides a **low-level**, frame-buffer-oriented API for
//! splitting a YUV420p luma plane into tile regions and encoding them in
//! parallel using rayon.  It is the structural companion to
//! `super::parallel_tile_decoder` on the encode side.
//!
//! For a higher-level API that works with [`crate::frame::VideoFrame`]
//! objects see [`super::tile_encoder::ParallelTileEncoder`].
//!
//! # Structural implementation note
//!
//! A full AV1 tile encode requires mode decision, transform coding,
//! quantisation, and entropy coding — all tightly coupled to codec state.
//! This module provides the *structural scaffolding*: correct tile splitting,
//! parallel dispatch via rayon, and a minimal binary encoding (tile header +
//! QP-XOR pixel data) suitable as a drop-in stand-in for the real pipeline.
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::av1::{RawTileEncoderConfig, encode_tiles_parallel};
//!
//! let config = RawTileEncoderConfig {
//!     tile_cols: 2,
//!     tile_rows: 2,
//!     threads: 0,
//!     base_qp: 32,
//! };
//! let frame = vec![128u8; 1920 * 1080]; // synthetic luma
//! let tiles = encode_tiles_parallel(&frame, 1920, 1080, &config).expect("encode");
//! assert_eq!(tiles.len(), 4);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::missing_errors_doc)]

use rayon::prelude::*;

use crate::error::{CodecError, CodecResult};

// ─────────────────────────────────────────────────────────────────────────────
// Magic bytes written into every tile header
// ─────────────────────────────────────────────────────────────────────────────

/// Four-byte magic that starts every encoded tile: `AV1T`.
const TILE_MAGIC: [u8; 4] = [0x41, 0x56, 0x31, 0x54]; // "AV1T"

/// Size of the tile header in bytes: magic(4) + width(4) + height(4) + qp(4).
const TILE_HEADER_SIZE: usize = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the low-level parallel tile encoder.
#[derive(Clone, Debug)]
pub struct TileEncoderConfig {
    /// Number of tile columns (must be ≥ 1).
    pub tile_cols: u32,
    /// Number of tile rows (must be ≥ 1).
    pub tile_rows: u32,
    /// Number of rayon threads to use (0 = auto-detect from rayon default pool).
    pub threads: usize,
    /// Base quantisation parameter (0 = highest quality / largest output,
    /// 255 = lowest quality / smallest output).
    pub base_qp: u32,
}

impl Default for TileEncoderConfig {
    fn default() -> Self {
        Self {
            tile_cols: 1,
            tile_rows: 1,
            threads: 0,
            base_qp: 32,
        }
    }
}

impl TileEncoderConfig {
    /// Validate that the configuration is internally consistent.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` if any field is out of range.
    pub fn validate(&self) -> CodecResult<()> {
        if self.tile_cols == 0 {
            return Err(CodecError::InvalidParameter(
                "tile_cols must be at least 1".to_string(),
            ));
        }
        if self.tile_rows == 0 {
            return Err(CodecError::InvalidParameter(
                "tile_rows must be at least 1".to_string(),
            ));
        }
        if self.base_qp > 255 {
            return Err(CodecError::InvalidParameter(
                "base_qp must be in range 0–255".to_string(),
            ));
        }
        Ok(())
    }

    /// Total number of tiles.
    #[must_use]
    pub const fn tile_count(&self) -> u32 {
        self.tile_cols * self.tile_rows
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tile region descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// Describes the location and dimensions of one tile within a frame.
#[derive(Clone, Debug)]
pub struct TileRegionInfo {
    /// Tile column index (0-based, left to right).
    pub col: u32,
    /// Tile row index (0-based, top to bottom).
    pub row: u32,
    /// Pixel X offset of this tile's top-left corner.
    pub x: u32,
    /// Pixel Y offset of this tile's top-left corner.
    pub y: u32,
    /// Tile width in pixels.
    pub width: u32,
    /// Tile height in pixels.
    pub height: u32,
}

impl TileRegionInfo {
    /// Raster-order index of this tile: `row * tile_cols + col`.
    #[must_use]
    pub fn raster_index(&self, tile_cols: u32) -> u32 {
        self.row * tile_cols + self.col
    }

    /// Area of this tile in pixels.
    #[must_use]
    pub const fn area(&self) -> u32 {
        self.width * self.height
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Encoded tile result
// ─────────────────────────────────────────────────────────────────────────────

/// The result of encoding a single tile.
#[derive(Clone, Debug)]
pub struct EncodedTile {
    /// Tile column index.
    pub tile_col: u32,
    /// Tile row index.
    pub tile_row: u32,
    /// Pixel offset of this tile's top-left corner: `(x, y)`.
    pub tile_offset: (u32, u32),
    /// Pixel dimensions of this tile: `(width, height)`.
    pub tile_size: (u32, u32),
    /// Encoded bitstream bytes for this tile.
    pub data: Vec<u8>,
    /// Quantisation parameter used.
    pub qp: u32,
}

impl EncodedTile {
    /// Raster-order index: `tile_row * tile_cols + tile_col`.
    #[must_use]
    pub fn raster_index(&self, tile_cols: u32) -> u32 {
        self.tile_row * tile_cols + self.tile_col
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free function (primary public API)
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a raw luma frame into parallel tile bitstreams.
///
/// `frame` is interpreted as a contiguous row-major luma plane of
/// `width × height` bytes.  The frame is split into a
/// `config.tile_cols × config.tile_rows` grid and each tile is encoded
/// independently and concurrently using rayon.
///
/// # Returns
///
/// `Vec<Vec<u8>>` — one inner `Vec<u8>` per tile, in raster order
/// (row-by-row, left to right within each row).
///
/// # Errors
///
/// Returns `CodecError::InvalidParameter` when the configuration is invalid
/// or `CodecError::InvalidBitstream` when an individual tile fails to encode.
pub fn encode_tiles_parallel(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &TileEncoderConfig,
) -> CodecResult<Vec<Vec<u8>>> {
    config.validate()?;

    if width == 0 || height == 0 {
        return Err(CodecError::InvalidParameter(
            "frame width and height must be non-zero".to_string(),
        ));
    }

    let encoder = ParallelTileEncoder::new(width, height, config.clone())?;
    let split_tiles = encoder.split_frame(frame);

    // Choose the rayon executor depending on threads setting.
    let encoded: CodecResult<Vec<(u32, Vec<u8>)>> = if config.threads > 0 {
        // Build a dedicated thread pool scoped to this call.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .build()
            .map_err(|e| CodecError::Internal(format!("thread pool error: {}", e)))?;

        pool.install(|| {
            split_tiles
                .par_iter()
                .map(|(region, tile_data)| {
                    let idx = region.raster_index(config.tile_cols);
                    let encoded = encode_single_tile(tile_data, region, config.base_qp)?;
                    Ok((idx, encoded))
                })
                .collect()
        })
    } else {
        split_tiles
            .par_iter()
            .map(|(region, tile_data)| {
                let idx = region.raster_index(config.tile_cols);
                let encoded = encode_single_tile(tile_data, region, config.base_qp)?;
                Ok((idx, encoded))
            })
            .collect()
    };

    let mut indexed = encoded?;
    // Sort by raster index to guarantee deterministic ordering.
    indexed.sort_by_key(|(idx, _)| *idx);
    Ok(indexed.into_iter().map(|(_, data)| data).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// ParallelTileEncoder struct
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel AV1 tile encoder operating on raw byte frames.
///
/// This struct provides the same functionality as [`encode_tiles_parallel`]
/// but as a reusable object that caches frame geometry.
#[derive(Clone, Debug)]
pub struct ParallelTileEncoder {
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Encoder configuration.
    pub config: TileEncoderConfig,
}

impl ParallelTileEncoder {
    /// Create a new `ParallelTileEncoder`.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` if the configuration is invalid
    /// or the frame dimensions are zero.
    pub fn new(
        frame_width: u32,
        frame_height: u32,
        config: TileEncoderConfig,
    ) -> CodecResult<Self> {
        config.validate()?;
        if frame_width == 0 || frame_height == 0 {
            return Err(CodecError::InvalidParameter(
                "frame width and height must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            frame_width,
            frame_height,
            config,
        })
    }

    /// Split `frame` (luma bytes, row-major) into `(TileRegionInfo, Vec<u8>)` pairs.
    ///
    /// Each pair contains the region descriptor and the extracted luma bytes
    /// for that tile, ready for encoding.  Tiles are returned in raster order.
    #[must_use]
    pub fn split_frame(&self, frame: &[u8]) -> Vec<(TileRegionInfo, Vec<u8>)> {
        let tile_cols = self.config.tile_cols;
        let tile_rows = self.config.tile_rows;

        if tile_cols == 0 || tile_rows == 0 {
            return Vec::new();
        }

        let base_w = self.frame_width / tile_cols;
        let rem_w = self.frame_width % tile_cols;
        let base_h = self.frame_height / tile_rows;
        let rem_h = self.frame_height % tile_rows;

        let mut result = Vec::with_capacity((tile_rows * tile_cols) as usize);

        for row in 0..tile_rows {
            let tile_h = if row == tile_rows - 1 {
                base_h + rem_h
            } else {
                base_h
            };
            let y_off = row * base_h;

            for col in 0..tile_cols {
                let tile_w = if col == tile_cols - 1 {
                    base_w + rem_w
                } else {
                    base_w
                };
                let x_off = col * base_w;

                let tile_bytes =
                    extract_luma_region(frame, self.frame_width, x_off, y_off, tile_w, tile_h);

                let region = TileRegionInfo {
                    col,
                    row,
                    x: x_off,
                    y: y_off,
                    width: tile_w,
                    height: tile_h,
                };

                result.push((region, tile_bytes));
            }
        }

        result
    }

    /// Encode all tiles in parallel and return [`EncodedTile`] results in
    /// raster order.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidBitstream` if any tile fails to encode.
    pub fn encode_frame_parallel(&self, frame: &[u8]) -> CodecResult<Vec<EncodedTile>> {
        let split = self.split_frame(frame);

        let results: CodecResult<Vec<EncodedTile>> = split
            .into_par_iter()
            .map(|(region, tile_data)| {
                let data = encode_single_tile(&tile_data, &region, self.config.base_qp)?;
                Ok(EncodedTile {
                    tile_col: region.col,
                    tile_row: region.row,
                    tile_offset: (region.x, region.y),
                    tile_size: (region.width, region.height),
                    data,
                    qp: self.config.base_qp,
                })
            })
            .collect();

        let mut tiles = results?;
        tiles.sort_by_key(|t| t.raster_index(self.config.tile_cols));
        Ok(tiles)
    }

    /// Assemble a slice of [`EncodedTile`]s into a single byte stream.
    ///
    /// Format: for each tile except the last, a 4-byte LE tile size prefix is
    /// written followed by the tile data.  The last tile has no size prefix
    /// (matching AV1 tile group conventions where the last tile size is
    /// implicit).
    #[must_use]
    pub fn assemble_encoded(&self, tiles: &[EncodedTile]) -> Vec<u8> {
        assemble_encoded_tiles(tiles)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the luma bytes of a rectangular tile region from a row-major buffer.
fn extract_luma_region(
    frame: &[u8],
    frame_width: u32,
    x_off: u32,
    y_off: u32,
    tile_w: u32,
    tile_h: u32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity((tile_w * tile_h) as usize);

    for row in 0..tile_h {
        let src_start = ((y_off + row) * frame_width + x_off) as usize;
        let src_end = src_start + tile_w as usize;

        if src_start >= frame.len() {
            // Pad with grey when input is exhausted.
            out.extend(std::iter::repeat_n(128u8, tile_w as usize));
        } else {
            let avail_end = src_end.min(frame.len());
            out.extend_from_slice(&frame[src_start..avail_end]);
            if avail_end < src_end {
                out.extend(std::iter::repeat_n(128u8, src_end - avail_end));
            }
        }
    }

    out
}

/// Structural single-tile encoder.
///
/// Produces:
/// - 16-byte header: `TILE_MAGIC` (4) + `width` u32-LE (4) + `height` u32-LE (4) + `qp` u32-LE (4)
/// - Payload: tile luma bytes XOR'd with `(qp & 0xFF) as u8`
///
/// # Errors
///
/// Returns `CodecError::InvalidBitstream` if the tile dimensions are zero.
fn encode_single_tile(tile_data: &[u8], region: &TileRegionInfo, qp: u32) -> CodecResult<Vec<u8>> {
    if region.width == 0 || region.height == 0 {
        return Err(CodecError::InvalidBitstream(format!(
            "tile ({},{}) has zero dimension: {}×{}",
            region.col, region.row, region.width, region.height
        )));
    }

    let payload_len = (region.width * region.height) as usize;
    let mut out = Vec::with_capacity(TILE_HEADER_SIZE + payload_len);

    // Write header.
    out.extend_from_slice(&TILE_MAGIC);
    out.extend_from_slice(&region.width.to_le_bytes());
    out.extend_from_slice(&region.height.to_le_bytes());
    out.extend_from_slice(&qp.to_le_bytes());

    // Write XOR-encoded payload (structural stand-in).
    let xor_mask = (qp & 0xFF) as u8;
    let copy_len = payload_len.min(tile_data.len());
    for &b in &tile_data[..copy_len] {
        out.push(b ^ xor_mask);
    }
    // Pad if tile_data was shorter than expected.
    for _ in copy_len..payload_len {
        out.push(128u8 ^ xor_mask);
    }

    Ok(out)
}

/// Assemble [`EncodedTile`] slices into a single stream.
///
/// Each tile except the last is prefixed with a 4-byte LE size field.
fn assemble_encoded_tiles(tiles: &[EncodedTile]) -> Vec<u8> {
    if tiles.is_empty() {
        return Vec::new();
    }

    let total: usize = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| {
            if i < tiles.len() - 1 {
                4 + t.data.len()
            } else {
                t.data.len()
            }
        })
        .sum();

    let mut out = Vec::with_capacity(total);

    for (i, tile) in tiles.iter().enumerate() {
        let is_last = i == tiles.len() - 1;
        if !is_last {
            let size = tile.data.len() as u32;
            out.extend_from_slice(&size.to_le_bytes());
        }
        out.extend_from_slice(&tile.data);
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_frame(width: u32, height: u32, fill: u8) -> Vec<u8> {
        vec![fill; (width * height) as usize]
    }

    fn default_config_2x2() -> TileEncoderConfig {
        TileEncoderConfig {
            tile_cols: 2,
            tile_rows: 2,
            threads: 0,
            base_qp: 32,
        }
    }

    // ── TileEncoderConfig ─────────────────────────────────────────────────────

    #[test]
    fn test_config_default_valid() {
        let cfg = TileEncoderConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_tile_count() {
        let cfg = TileEncoderConfig {
            tile_cols: 4,
            tile_rows: 2,
            ..Default::default()
        };
        assert_eq!(cfg.tile_count(), 8);
    }

    // ── ParallelTileEncoder::new ──────────────────────────────────────────────

    #[test]
    fn test_new_valid_config() {
        let cfg = default_config_2x2();
        let enc = ParallelTileEncoder::new(1920, 1080, cfg);
        assert!(enc.is_ok());
    }

    #[test]
    fn test_new_zero_cols_errors() {
        let cfg = TileEncoderConfig {
            tile_cols: 0,
            tile_rows: 2,
            threads: 0,
            base_qp: 32,
        };
        let result = ParallelTileEncoder::new(1920, 1080, cfg);
        assert!(result.is_err(), "zero tile_cols should fail");
    }

    #[test]
    fn test_new_zero_rows_errors() {
        let cfg = TileEncoderConfig {
            tile_cols: 2,
            tile_rows: 0,
            threads: 0,
            base_qp: 32,
        };
        let result = ParallelTileEncoder::new(1920, 1080, cfg);
        assert!(result.is_err(), "zero tile_rows should fail");
    }

    #[test]
    fn test_new_zero_width_errors() {
        let cfg = default_config_2x2();
        let result = ParallelTileEncoder::new(0, 1080, cfg);
        assert!(result.is_err(), "zero width should fail");
    }

    // ── split_frame ───────────────────────────────────────────────────────────

    #[test]
    fn test_split_frame_tile_count_2x2() {
        let cfg = default_config_2x2();
        let enc = ParallelTileEncoder::new(640, 480, cfg).expect("ok");
        let frame = make_frame(640, 480, 0);
        let tiles = enc.split_frame(&frame);
        assert_eq!(tiles.len(), 4, "2×2 grid must yield 4 tiles");
    }

    #[test]
    fn test_split_frame_tile_sizes_sum_to_frame() {
        let cfg = default_config_2x2();
        let enc = ParallelTileEncoder::new(800, 600, cfg).expect("ok");
        let frame = make_frame(800, 600, 0);
        let tiles = enc.split_frame(&frame);

        let row0_width: u32 = tiles
            .iter()
            .filter(|(r, _)| r.row == 0)
            .map(|(r, _)| r.width)
            .sum();
        let col0_height: u32 = tiles
            .iter()
            .filter(|(r, _)| r.col == 0)
            .map(|(r, _)| r.height)
            .sum();
        assert_eq!(
            row0_width, 800,
            "tile widths in row 0 must sum to frame width"
        );
        assert_eq!(
            col0_height, 600,
            "tile heights in col 0 must sum to frame height"
        );
    }

    #[test]
    fn test_split_frame_non_divisible() {
        // 1000 / 3 = 333 remainder 1; 700 / 2 = 350 exactly
        let cfg = TileEncoderConfig {
            tile_cols: 3,
            tile_rows: 2,
            threads: 0,
            base_qp: 16,
        };
        let enc = ParallelTileEncoder::new(1000, 700, cfg).expect("ok");
        let frame = make_frame(1000, 700, 0);
        let tiles = enc.split_frame(&frame);
        assert_eq!(tiles.len(), 6);

        let row0_width: u32 = tiles
            .iter()
            .filter(|(r, _)| r.row == 0)
            .map(|(r, _)| r.width)
            .sum();
        let col0_height: u32 = tiles
            .iter()
            .filter(|(r, _)| r.col == 0)
            .map(|(r, _)| r.height)
            .sum();
        assert_eq!(row0_width, 1000);
        assert_eq!(col0_height, 700);
    }

    #[test]
    fn test_split_frame_data_length_equals_area() {
        let cfg = default_config_2x2();
        let enc = ParallelTileEncoder::new(200, 100, cfg).expect("ok");
        let frame = make_frame(200, 100, 42);
        for (region, tile_data) in enc.split_frame(&frame) {
            let expected = (region.width * region.height) as usize;
            assert_eq!(
                tile_data.len(),
                expected,
                "tile ({},{}) data length mismatch",
                region.col,
                region.row
            );
        }
    }

    // ── encode_tiles_parallel (free function) ─────────────────────────────────

    #[test]
    fn test_encode_tiles_parallel_output_count() {
        let cfg = default_config_2x2();
        let frame = make_frame(640, 480, 128);
        let result = encode_tiles_parallel(&frame, 640, 480, &cfg).expect("ok");
        assert_eq!(result.len(), 4, "must return one Vec<u8> per tile");
    }

    #[test]
    fn test_encode_tiles_parallel_output_sizes() {
        let cfg = default_config_2x2();
        let frame = make_frame(640, 480, 0);
        let tiles = encode_tiles_parallel(&frame, 640, 480, &cfg).expect("ok");
        for tile in &tiles {
            assert!(
                tile.len() >= TILE_HEADER_SIZE,
                "each tile must be at least {} bytes",
                TILE_HEADER_SIZE
            );
        }
    }

    #[test]
    fn test_encode_tiles_parallel_single_tile() {
        let cfg = TileEncoderConfig {
            tile_cols: 1,
            tile_rows: 1,
            threads: 0,
            base_qp: 0,
        };
        let frame = make_frame(320, 240, 77);
        let tiles = encode_tiles_parallel(&frame, 320, 240, &cfg).expect("ok");
        assert_eq!(tiles.len(), 1);
        // With qp=0, XOR mask is 0, so payload must equal original pixels.
        let payload = &tiles[0][TILE_HEADER_SIZE..];
        assert!(
            payload.iter().all(|&b| b == 77),
            "with qp=0 payload must equal original pixels"
        );
    }

    #[test]
    fn test_encode_tiles_parallel_content_header_magic() {
        let cfg = default_config_2x2();
        let frame = make_frame(64, 32, 0);
        let tiles = encode_tiles_parallel(&frame, 64, 32, &cfg).expect("ok");
        for tile in &tiles {
            assert_eq!(
                &tile[0..4],
                &TILE_MAGIC,
                "tile header must start with TILE_MAGIC"
            );
        }
    }

    #[test]
    fn test_encode_tiles_parallel_header_width_height_encoded() {
        let cfg = TileEncoderConfig {
            tile_cols: 1,
            tile_rows: 1,
            threads: 0,
            base_qp: 8,
        };
        let frame = make_frame(128, 96, 0);
        let tiles = encode_tiles_parallel(&frame, 128, 96, &cfg).expect("ok");
        assert_eq!(tiles.len(), 1);
        // Bytes 4..8 = width LE, bytes 8..12 = height LE
        let w = u32::from_le_bytes(tiles[0][4..8].try_into().expect("slice"));
        let h = u32::from_le_bytes(tiles[0][8..12].try_into().expect("slice"));
        assert_eq!(w, 128);
        assert_eq!(h, 96);
    }

    #[test]
    fn test_encode_tiles_parallel_zero_cols_errors() {
        let cfg = TileEncoderConfig {
            tile_cols: 0,
            tile_rows: 2,
            threads: 0,
            base_qp: 32,
        };
        let frame = make_frame(640, 480, 0);
        let result = encode_tiles_parallel(&frame, 640, 480, &cfg);
        assert!(result.is_err());
    }

    // ── assemble_encoded ──────────────────────────────────────────────────────

    #[test]
    fn test_assemble_encoded_non_empty() {
        let cfg = default_config_2x2();
        let enc = ParallelTileEncoder::new(640, 480, cfg).expect("ok");
        let frame = make_frame(640, 480, 55);
        let encoded_tiles = enc.encode_frame_parallel(&frame).expect("ok");
        let assembled = enc.assemble_encoded(&encoded_tiles);
        assert!(!assembled.is_empty(), "assembled output must not be empty");
    }

    #[test]
    fn test_assemble_encoded_single_tile_no_size_prefix() {
        // A single tile must NOT have a size prefix (it is the last tile).
        let cfg = TileEncoderConfig {
            tile_cols: 1,
            tile_rows: 1,
            threads: 0,
            base_qp: 0,
        };
        let enc = ParallelTileEncoder::new(64, 32, cfg).expect("ok");
        let frame = make_frame(64, 32, 10);
        let encoded_tiles = enc.encode_frame_parallel(&frame).expect("ok");
        assert_eq!(encoded_tiles.len(), 1);

        let assembled = enc.assemble_encoded(&encoded_tiles);
        // Without prefix the assembled data equals the single tile data.
        assert_eq!(assembled.len(), encoded_tiles[0].data.len());
    }

    // ── TileRegionInfo ────────────────────────────────────────────────────────

    #[test]
    fn test_tile_region_info_fields() {
        let region = TileRegionInfo {
            col: 1,
            row: 2,
            x: 320,
            y: 240,
            width: 320,
            height: 240,
        };
        assert_eq!(region.raster_index(4), 2 * 4 + 1);
        assert_eq!(region.area(), 320 * 240);
    }
}
