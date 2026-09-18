//! Tile-based parallel frame encoding for OxiMedia codecs.
//!
//! This module provides pixel-level infrastructure for splitting raw video
//! frames into rectangular tiles, processing them concurrently, and
//! reassembling the result into a complete frame.
//!
//! Unlike [`crate::tile`] which works with [`crate::frame::VideoFrame`] and
//! codec-level bitstream output, this module operates on raw `&[u8]` / `Vec<u8>`
//! pixel buffers and is therefore codec-agnostic.
//!
//! # Architecture
//!
//! ```text
//! TileConfig  ─── tile grid parameters (cols, rows, frame size)
//!     │
//!     ▼
//! TileLayout  ─── pre-computed TileRegion grid (handles remainder pixels)
//!     │
//!     ▼
//! ParallelTileEncoder ─── split_frame → parallel encode_fn → merge_tiles
//! ```
//!
//! # Example
//!
//! ```
//! use oximedia_codec::tile_encoder::{TileConfig, ParallelTileEncoder};
//!
//! let config = TileConfig::new()
//!     .tile_cols(2)
//!     .tile_rows(2)
//!     .frame_width(64)
//!     .frame_height(64);
//!
//! let encoder = ParallelTileEncoder::new(config);
//!
//! // Create a simple 64×64 RGB frame (3 channels).
//! let frame: Vec<u8> = (0u8..=255).cycle().take(64 * 64 * 3).collect();
//!
//! let tiles = encoder.split_frame(&frame, 3);
//! assert_eq!(tiles.len(), 4);
//!
//! let merged = ParallelTileEncoder::merge_tiles(&tiles, 64, 64, 3);
//! assert_eq!(merged, frame);
//! ```

use rayon::prelude::*;
use std::ops::Range;

// =============================================================================
// TileConfig
// =============================================================================

/// Configuration for the tile grid and frame dimensions.
///
/// Use the builder-pattern methods to configure:
///
/// ```
/// use oximedia_codec::tile_encoder::TileConfig;
///
/// let cfg = TileConfig::new()
///     .tile_cols(4)
///     .tile_rows(4)
///     .num_threads(8)
///     .frame_width(1920)
///     .frame_height(1080);
///
/// assert_eq!(cfg.tile_cols, 4);
/// assert_eq!(cfg.num_threads, 8);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileConfig {
    /// Number of tile columns (1–64).
    pub tile_cols: u32,
    /// Number of tile rows (1–64).
    pub tile_rows: u32,
    /// Worker threads for parallel encoding (0 = use Rayon pool size).
    pub num_threads: usize,
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
}

impl TileConfig {
    /// Create a `TileConfig` with default values.
    ///
    /// Defaults: 1 column, 1 row, 0 threads (auto), 0×0 frame.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of tile columns (1–64).
    #[must_use]
    pub fn tile_cols(mut self, cols: u32) -> Self {
        self.tile_cols = cols.clamp(1, 64);
        self
    }

    /// Set the number of tile rows (1–64).
    #[must_use]
    pub fn tile_rows(mut self, rows: u32) -> Self {
        self.tile_rows = rows.clamp(1, 64);
        self
    }

    /// Set the worker thread count (0 = Rayon auto).
    #[must_use]
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Set the frame width in pixels.
    #[must_use]
    pub fn frame_width(mut self, width: u32) -> Self {
        self.frame_width = width;
        self
    }

    /// Set the frame height in pixels.
    #[must_use]
    pub fn frame_height(mut self, height: u32) -> Self {
        self.frame_height = height;
        self
    }

    /// Effective thread count (resolves 0 to the Rayon pool size).
    #[must_use]
    pub fn thread_count(&self) -> usize {
        if self.num_threads == 0 {
            rayon::current_num_threads()
        } else {
            self.num_threads
        }
    }
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_cols: 1,
            tile_rows: 1,
            num_threads: 0,
            frame_width: 0,
            frame_height: 0,
        }
    }
}

// =============================================================================
// TileRegion
// =============================================================================

/// Pixel coordinates and dimensions of a single tile within a frame.
///
/// ```
/// use oximedia_codec::tile_encoder::TileRegion;
///
/// let region = TileRegion::new(1, 0, 512, 0, 512, 288);
/// assert_eq!(region.area(), 512 * 288);
/// assert!(region.contains(600, 100));
/// assert!(!region.contains(200, 100)); // left of tile
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileRegion {
    /// Tile column index (0-based).
    pub col: u32,
    /// Tile row index (0-based).
    pub row: u32,
    /// X pixel offset from the left of the frame.
    pub x: u32,
    /// Y pixel offset from the top of the frame.
    pub y: u32,
    /// Tile width in pixels.
    pub width: u32,
    /// Tile height in pixels.
    pub height: u32,
}

impl TileRegion {
    /// Create a new `TileRegion`.
    #[must_use]
    pub const fn new(col: u32, row: u32, x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            col,
            row,
            x,
            y,
            width,
            height,
        }
    }

    /// Area of this tile in pixels.
    #[must_use]
    pub const fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Returns `true` if the pixel `(px, py)` falls within this tile.
    #[must_use]
    pub const fn contains(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    /// Pixel column range `x..(x + width)`.
    #[must_use]
    pub fn pixel_range_x(&self) -> Range<u32> {
        self.x..(self.x + self.width)
    }

    /// Pixel row range `y..(y + height)`.
    #[must_use]
    pub fn pixel_range_y(&self) -> Range<u32> {
        self.y..(self.y + self.height)
    }
}

// =============================================================================
// TileLayout
// =============================================================================

/// A grid of [`TileRegion`]s computed from a [`TileConfig`].
///
/// The last tile in each row/column absorbs any remainder pixels so the union
/// of all tiles exactly covers the full frame with no overlap.
///
/// ```
/// use oximedia_codec::tile_encoder::{TileConfig, TileLayout};
///
/// let cfg = TileConfig::new()
///     .tile_cols(2)
///     .tile_rows(2)
///     .frame_width(100)
///     .frame_height(100);
///
/// let layout = TileLayout::new(cfg);
/// assert_eq!(layout.tile_count(), 4);
///
/// // All tiles together cover 100×100 pixels.
/// let total: u64 = layout.tiles().iter().map(|t| t.area()).sum();
/// assert_eq!(total, 100 * 100);
/// ```
#[derive(Clone, Debug)]
pub struct TileLayout {
    /// The configuration used to build this layout.
    pub config: TileConfig,
    /// All tile regions in raster order (row-major).
    pub tiles: Vec<TileRegion>,
}

impl TileLayout {
    /// Compute a `TileLayout` from `config`.
    ///
    /// Tile boundaries are computed as `frame_width / tile_cols` (integer
    /// division); the last column and last row absorb the remainder pixels.
    #[must_use]
    pub fn new(config: TileConfig) -> Self {
        let cols = config.tile_cols.max(1);
        let rows = config.tile_rows.max(1);
        let fw = config.frame_width;
        let fh = config.frame_height;

        // Nominal tile sizes (last tile gets the remainder).
        let nominal_tw = fw / cols;
        let nominal_th = fh / rows;

        let mut tiles = Vec::with_capacity((cols * rows) as usize);

        for row in 0..rows {
            for col in 0..cols {
                let x = col * nominal_tw;
                let y = row * nominal_th;

                let width = if col == cols - 1 {
                    fw.saturating_sub(x)
                } else {
                    nominal_tw
                };
                let height = if row == rows - 1 {
                    fh.saturating_sub(y)
                } else {
                    nominal_th
                };

                tiles.push(TileRegion::new(col, row, x, y, width, height));
            }
        }

        Self { config, tiles }
    }

    /// Total number of tiles.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Return the tile at grid position `(col, row)`, or `None` if out of bounds.
    #[must_use]
    pub fn get_tile(&self, col: u32, row: u32) -> Option<&TileRegion> {
        let cols = self.config.tile_cols;
        let rows = self.config.tile_rows;
        if col >= cols || row >= rows {
            return None;
        }
        self.tiles.get((row * cols + col) as usize)
    }

    /// All tile regions in raster order.
    #[must_use]
    pub fn tiles(&self) -> &[TileRegion] {
        &self.tiles
    }

    /// Find which tile contains the pixel `(px, py)`.
    ///
    /// Returns `None` if the pixel is outside the frame.
    #[must_use]
    pub fn tile_for_pixel(&self, px: u32, py: u32) -> Option<&TileRegion> {
        self.tiles.iter().find(|t| t.contains(px, py))
    }
}

// =============================================================================
// TileBuffer
// =============================================================================

/// Raw pixel data extracted from (or destined for) a single tile.
///
/// ```
/// use oximedia_codec::tile_encoder::{TileRegion, TileBuffer};
///
/// let region = TileRegion::new(0, 0, 0, 0, 4, 4);
/// let buf = TileBuffer::new(region, 3); // 3 channels (RGB)
/// assert_eq!(buf.data.len(), 4 * 4 * 3);
/// assert_eq!(buf.stride, 4 * 3);
/// ```
#[derive(Clone, Debug)]
pub struct TileBuffer {
    /// Spatial position of this tile in the frame.
    pub region: TileRegion,
    /// Raw pixel bytes for this tile (row-major, tightly packed).
    pub data: Vec<u8>,
    /// Row stride in bytes (`width * channels`).
    pub stride: usize,
    /// Bytes per pixel.
    pub channels: u8,
}

impl TileBuffer {
    /// Allocate an all-zero `TileBuffer` for `region` with `channels` bytes per pixel.
    #[must_use]
    pub fn new(region: TileRegion, channels: u8) -> Self {
        let ch = channels as usize;
        let stride = region.width as usize * ch;
        let data = vec![0u8; region.height as usize * stride];
        Self {
            region,
            data,
            stride,
            channels,
        }
    }

    /// Copy the tile's pixels from `frame` (a packed, row-major buffer).
    ///
    /// `frame_stride` is the number of bytes per row in the full frame
    /// (i.e. `frame_width * channels`).
    pub fn extract_from_frame(&mut self, frame: &[u8], frame_stride: usize) {
        let ch = self.channels as usize;
        let x_byte = self.region.x as usize * ch;
        let w_bytes = self.region.width as usize * ch;

        for row in 0..self.region.height as usize {
            let frame_row_start = (self.region.y as usize + row) * frame_stride + x_byte;
            let tile_row_start = row * self.stride;

            let src_end = (frame_row_start + w_bytes).min(frame.len());
            let copy_len = src_end.saturating_sub(frame_row_start);

            self.data[tile_row_start..tile_row_start + copy_len]
                .copy_from_slice(&frame[frame_row_start..src_end]);
        }
    }

    /// Write this tile's pixels back into `frame`.
    ///
    /// `frame_stride` must match the full frame's row stride.
    pub fn write_to_frame(&self, frame: &mut [u8], frame_stride: usize) {
        let ch = self.channels as usize;
        let x_byte = self.region.x as usize * ch;
        let w_bytes = self.region.width as usize * ch;

        for row in 0..self.region.height as usize {
            let frame_row_start = (self.region.y as usize + row) * frame_stride + x_byte;
            let tile_row_start = row * self.stride;

            let dst_end = (frame_row_start + w_bytes).min(frame.len());
            let copy_len = dst_end.saturating_sub(frame_row_start);

            frame[frame_row_start..frame_row_start + copy_len]
                .copy_from_slice(&self.data[tile_row_start..tile_row_start + copy_len]);
        }
    }
}

// =============================================================================
// ParallelTileEncoder
// =============================================================================

/// Splits a raw pixel frame into tiles, processes them in parallel, and
/// reassembles the result.
///
/// # Example
///
/// ```
/// use oximedia_codec::tile_encoder::{TileConfig, ParallelTileEncoder};
///
/// let config = TileConfig::new()
///     .tile_cols(2)
///     .tile_rows(2)
///     .frame_width(64)
///     .frame_height(64);
///
/// let encoder = ParallelTileEncoder::new(config);
///
/// let frame: Vec<u8> = (0u8..=255).cycle().take(64 * 64 * 3).collect();
/// let tiles = encoder.split_frame(&frame, 3);
/// assert_eq!(tiles.len(), 4);
///
/// // Identity encode: return each tile unchanged.
/// let processed = encoder
///     .encode_tiles_parallel(tiles, |tile| Ok(tile))
///     ?;
///
/// let merged = ParallelTileEncoder::merge_tiles(&processed, 64, 64, 3);
/// assert_eq!(merged, frame);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ParallelTileEncoder {
    /// Pre-computed tile layout.
    pub layout: TileLayout,
}

impl ParallelTileEncoder {
    /// Create a `ParallelTileEncoder` from `config`.
    #[must_use]
    pub fn new(config: TileConfig) -> Self {
        Self {
            layout: TileLayout::new(config),
        }
    }

    /// Split `frame` into [`TileBuffer`]s, one per tile in the layout.
    ///
    /// `channels` is the number of bytes per pixel in `frame`.
    #[must_use]
    pub fn split_frame(&self, frame: &[u8], channels: u8) -> Vec<TileBuffer> {
        let fw = self.layout.config.frame_width;
        let frame_stride = fw as usize * channels as usize;

        self.layout
            .tiles
            .iter()
            .map(|region| {
                let mut buf = TileBuffer::new(region.clone(), channels);
                buf.extract_from_frame(frame, frame_stride);
                buf
            })
            .collect()
    }

    /// Merge a slice of [`TileBuffer`]s back into a complete frame.
    ///
    /// The returned `Vec<u8>` has `frame_width * frame_height * channels` bytes.
    #[must_use]
    pub fn merge_tiles(
        tiles: &[TileBuffer],
        frame_width: u32,
        frame_height: u32,
        channels: u8,
    ) -> Vec<u8> {
        let ch = channels as usize;
        let frame_stride = frame_width as usize * ch;
        let frame_size = frame_height as usize * frame_stride;
        let mut frame = vec![0u8; frame_size];

        for tile in tiles {
            tile.write_to_frame(&mut frame, frame_stride);
        }

        frame
    }

    /// Process `tiles` in parallel using `encode_fn`.
    ///
    /// Each tile is passed by value to `encode_fn`.  The closure must return
    /// either a (possibly modified) [`TileBuffer`] or an error string.
    ///
    /// Uses Rayon for parallel execution.  The output order matches the input
    /// order (raster order when produced by `split_frame`).
    ///
    /// # Errors
    ///
    /// Returns the first error string produced by any invocation of
    /// `encode_fn`.
    pub fn encode_tiles_parallel<F>(
        &self,
        tiles: Vec<TileBuffer>,
        encode_fn: F,
    ) -> Result<Vec<TileBuffer>, String>
    where
        F: Fn(TileBuffer) -> Result<TileBuffer, String> + Send + Sync,
    {
        let results: Vec<Result<TileBuffer, String>> =
            tiles.into_par_iter().map(|tile| encode_fn(tile)).collect();

        let mut out = Vec::with_capacity(results.len());
        for r in results {
            out.push(r?);
        }
        Ok(out)
    }
}

// =============================================================================
// Adaptive Tile Partitioning
// =============================================================================

/// Content complexity metric for a tile region.
#[derive(Clone, Debug, PartialEq)]
pub struct TileComplexity {
    /// Tile column index.
    pub col: u32,
    /// Tile row index.
    pub row: u32,
    /// Variance of pixel values (higher = more complex).
    pub variance: f64,
    /// Mean absolute difference between adjacent pixels (edge density).
    pub edge_density: f64,
    /// Normalised complexity score in [0.0, 1.0].
    pub score: f64,
}

/// Analyse content complexity for each tile in a frame.
///
/// `frame` is a packed row-major pixel buffer with `channels` bytes per pixel.
/// Returns a [`TileComplexity`] for every tile in `layout`.
pub fn analyse_tile_complexity(
    layout: &TileLayout,
    frame: &[u8],
    channels: u8,
) -> Vec<TileComplexity> {
    let fw = layout.config.frame_width;
    let frame_stride = fw as usize * channels as usize;

    let complexities: Vec<TileComplexity> = layout
        .tiles
        .iter()
        .map(|region| {
            let ch = channels as usize;
            let w = region.width as usize;
            let h = region.height as usize;
            let n = (w * h) as f64;

            if n < 1.0 {
                return TileComplexity {
                    col: region.col,
                    row: region.row,
                    variance: 0.0,
                    edge_density: 0.0,
                    score: 0.0,
                };
            }

            // Compute mean and variance of luma (average of channels).
            let mut sum: f64 = 0.0;
            let mut sum_sq: f64 = 0.0;
            let mut edge_sum: f64 = 0.0;
            let mut edge_count: u64 = 0;

            for row_idx in 0..h {
                let frame_y = region.y as usize + row_idx;
                for col_idx in 0..w {
                    let frame_x = region.x as usize + col_idx;
                    let base = frame_y * frame_stride + frame_x * ch;

                    // Average across channels for luma approximation.
                    let mut pixel_sum: u32 = 0;
                    for c in 0..ch.min(frame.len().saturating_sub(base)) {
                        pixel_sum += frame[base + c] as u32;
                    }
                    let luma = pixel_sum as f64 / ch.max(1) as f64;
                    sum += luma;
                    sum_sq += luma * luma;

                    // Horizontal edge detection.
                    if col_idx + 1 < w {
                        let next_base = base + ch;
                        let mut next_sum: u32 = 0;
                        for c in 0..ch.min(frame.len().saturating_sub(next_base)) {
                            next_sum += frame[next_base + c] as u32;
                        }
                        let next_luma = next_sum as f64 / ch.max(1) as f64;
                        edge_sum += (luma - next_luma).abs();
                        edge_count += 1;
                    }

                    // Vertical edge detection.
                    if row_idx + 1 < h {
                        let below_base = (frame_y + 1) * frame_stride + frame_x * ch;
                        let mut below_sum: u32 = 0;
                        for c in 0..ch.min(frame.len().saturating_sub(below_base)) {
                            below_sum += frame[below_base + c] as u32;
                        }
                        let below_luma = below_sum as f64 / ch.max(1) as f64;
                        edge_sum += (luma - below_luma).abs();
                        edge_count += 1;
                    }
                }
            }

            let mean = sum / n;
            let variance = (sum_sq / n) - (mean * mean);
            let edge_density = if edge_count > 0 {
                edge_sum / edge_count as f64
            } else {
                0.0
            };

            TileComplexity {
                col: region.col,
                row: region.row,
                variance: variance.max(0.0),
                edge_density,
                score: 0.0, // filled in below
            }
        })
        .collect();

    // Normalise scores to [0.0, 1.0].
    let max_var = complexities
        .iter()
        .map(|c| c.variance)
        .fold(0.0_f64, f64::max);
    let max_edge = complexities
        .iter()
        .map(|c| c.edge_density)
        .fold(0.0_f64, f64::max);

    complexities
        .into_iter()
        .map(|mut c| {
            let norm_var = if max_var > 0.0 {
                c.variance / max_var
            } else {
                0.0
            };
            let norm_edge = if max_edge > 0.0 {
                c.edge_density / max_edge
            } else {
                0.0
            };
            c.score = (0.6 * norm_var + 0.4 * norm_edge).clamp(0.0, 1.0);
            c
        })
        .collect()
}

/// Decides whether a tile should be split into sub-tiles based on complexity.
///
/// Returns a suggested partition: `(sub_cols, sub_rows)` for each tile.
/// Simple tiles get `(1,1)`, complex tiles get up to `(max_split, max_split)`.
pub fn adaptive_tile_partition(
    complexities: &[TileComplexity],
    threshold: f64,
    max_split: u32,
) -> Vec<(u32, u32)> {
    let max_split = max_split.max(1).min(8);
    complexities
        .iter()
        .map(|c| {
            if c.score > threshold {
                // Scale split factor by how far above threshold.
                let factor = ((c.score - threshold) / (1.0 - threshold.min(0.999))
                    * max_split as f64)
                    .ceil() as u32;
                let splits = factor.clamp(2, max_split);
                (splits, splits)
            } else {
                (1, 1)
            }
        })
        .collect()
}

// =============================================================================
// Tile-Level Rate Control
// =============================================================================

/// Bit budget allocation for a single tile.
#[derive(Clone, Debug, PartialEq)]
pub struct TileBitBudget {
    /// Tile column index.
    pub col: u32,
    /// Tile row index.
    pub row: u32,
    /// Allocated bits for this tile.
    pub bits: u64,
    /// Quality parameter (lower = higher quality, range depends on codec).
    pub qp: f64,
}

/// Allocate a total bit budget across tiles based on complexity.
///
/// More complex tiles receive proportionally more bits.  The `total_bits`
/// budget is distributed according to each tile's complexity score, with
/// a minimum floor of `min_bits_per_tile`.
pub fn allocate_tile_bits(
    complexities: &[TileComplexity],
    total_bits: u64,
    min_bits_per_tile: u64,
    base_qp: f64,
) -> Vec<TileBitBudget> {
    if complexities.is_empty() {
        return Vec::new();
    }

    // Ensure minimum allocation is possible.
    let min_total = min_bits_per_tile * complexities.len() as u64;
    let distributable = total_bits.saturating_sub(min_total);

    // Weight by complexity score (add small epsilon to avoid zero-weight).
    let weights: Vec<f64> = complexities.iter().map(|c| c.score + 0.01).collect();
    let total_weight: f64 = weights.iter().sum();

    complexities
        .iter()
        .zip(weights.iter())
        .map(|(c, &w)| {
            let share = if total_weight > 0.0 {
                (w / total_weight * distributable as f64) as u64
            } else {
                distributable / complexities.len() as u64
            };
            let bits = min_bits_per_tile + share;

            // QP adjustment: lower complexity → higher QP (save bits).
            // Higher complexity → lower QP (spend bits for quality).
            let qp_delta = (1.0 - c.score) * 6.0 - 3.0; // range [-3, +3]
            let qp = (base_qp + qp_delta).clamp(0.0, 51.0);

            TileBitBudget {
                col: c.col,
                row: c.row,
                bits,
                qp,
            }
        })
        .collect()
}

// =============================================================================
// Tile Dependency Tracking
// =============================================================================

/// The type of dependency one tile has on another.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileDependencyKind {
    /// Motion vector crosses into adjacent tile.
    MotionVector,
    /// In-loop filter requires border pixels from neighbour.
    LoopFilter,
    /// Entropy context is shared with the tile to the left.
    EntropyContext,
}

/// A dependency edge from one tile to another.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileDependency {
    /// Source tile (col, row).
    pub from: (u32, u32),
    /// Target tile (col, row) that `from` depends on.
    pub to: (u32, u32),
    /// Kind of dependency.
    pub kind: TileDependencyKind,
}

/// Dependency graph for a tile layout.
#[derive(Clone, Debug)]
pub struct TileDependencyGraph {
    /// All dependency edges.
    pub edges: Vec<TileDependency>,
    /// Number of tile columns.
    pub cols: u32,
    /// Number of tile rows.
    pub rows: u32,
}

impl TileDependencyGraph {
    /// Build a dependency graph for the given layout.
    ///
    /// By default, each tile depends on its left neighbour (entropy context)
    /// and its top neighbour (loop filter boundary).  The caller can add
    /// motion-vector dependencies afterwards.
    pub fn build(layout: &TileLayout) -> Self {
        let cols = layout.config.tile_cols;
        let rows = layout.config.tile_rows;
        let mut edges = Vec::new();

        for row in 0..rows {
            for col in 0..cols {
                // Left neighbour: entropy context dependency.
                if col > 0 {
                    edges.push(TileDependency {
                        from: (col, row),
                        to: (col - 1, row),
                        kind: TileDependencyKind::EntropyContext,
                    });
                }
                // Top neighbour: loop-filter dependency.
                if row > 0 {
                    edges.push(TileDependency {
                        from: (col, row),
                        to: (col, row - 1),
                        kind: TileDependencyKind::LoopFilter,
                    });
                }
            }
        }

        Self { edges, cols, rows }
    }

    /// Add a motion-vector dependency between two tiles.
    pub fn add_mv_dependency(&mut self, from: (u32, u32), to: (u32, u32)) {
        if from.0 < self.cols && from.1 < self.rows && to.0 < self.cols && to.1 < self.rows {
            self.edges.push(TileDependency {
                from,
                to,
                kind: TileDependencyKind::MotionVector,
            });
        }
    }

    /// Return all tiles that `(col, row)` depends on.
    pub fn dependencies_of(&self, col: u32, row: u32) -> Vec<&TileDependency> {
        self.edges.iter().filter(|e| e.from == (col, row)).collect()
    }

    /// Return tiles that can be encoded independently (no incoming dependencies
    /// from tiles that haven't been encoded yet).
    ///
    /// `encoded` is a set of already-encoded tile coordinates.
    pub fn ready_tiles(&self, encoded: &[(u32, u32)]) -> Vec<(u32, u32)> {
        let mut ready = Vec::new();
        for row in 0..self.rows {
            for col in 0..self.cols {
                let pos = (col, row);
                if encoded.contains(&pos) {
                    continue;
                }
                let deps = self.dependencies_of(col, row);
                let all_met = deps.iter().all(|d| encoded.contains(&d.to));
                if all_met {
                    ready.push(pos);
                }
            }
        }
        ready
    }
}

// =============================================================================
// Tile Work Queue (parallel encode with dependency awareness)
// =============================================================================

/// A work item for the tile encode queue.
#[derive(Clone, Debug)]
pub struct TileWorkItem {
    /// Tile coordinate.
    pub pos: (u32, u32),
    /// Tile buffer to encode.
    pub buffer: TileBuffer,
    /// Bit budget (if rate control is active).
    pub bit_budget: Option<TileBitBudget>,
}

/// Encodes tiles in dependency-aware waves using Rayon.
///
/// Tiles with satisfied dependencies are encoded in parallel waves.
/// Returns encoded tile buffers in the order they were submitted.
pub fn encode_tiles_wavefront<F>(
    graph: &TileDependencyGraph,
    mut work_items: Vec<TileWorkItem>,
    encode_fn: F,
) -> Result<Vec<TileBuffer>, String>
where
    F: Fn(TileWorkItem) -> Result<TileBuffer, String> + Send + Sync,
{
    let total = work_items.len();
    let mut encoded_positions: Vec<(u32, u32)> = Vec::with_capacity(total);
    let mut results: Vec<Option<TileBuffer>> = (0..total).map(|_| None).collect();

    // Build a position → index map.
    let pos_to_idx: std::collections::HashMap<(u32, u32), usize> = work_items
        .iter()
        .enumerate()
        .map(|(i, w)| (w.pos, i))
        .collect();

    while encoded_positions.len() < total {
        let ready = graph.ready_tiles(&encoded_positions);
        if ready.is_empty() && encoded_positions.len() < total {
            return Err("dependency deadlock: no tiles ready but not all encoded".to_string());
        }

        // Collect work items for this wave.
        let wave_items: Vec<(usize, TileWorkItem)> = ready
            .iter()
            .filter_map(|pos| {
                let idx = pos_to_idx.get(pos).copied();
                idx.map(|i| {
                    // Replace with a placeholder (empty buffer).
                    let item = std::mem::replace(
                        &mut work_items[i],
                        TileWorkItem {
                            pos: *pos,
                            buffer: TileBuffer::new(TileRegion::new(0, 0, 0, 0, 0, 0), 1),
                            bit_budget: None,
                        },
                    );
                    (i, item)
                })
            })
            .collect();

        let wave_results: Vec<(usize, Result<TileBuffer, String>)> = wave_items
            .into_par_iter()
            .map(|(i, item)| (i, encode_fn(item)))
            .collect();

        for (i, result) in wave_results {
            let buf = result?;
            let pos = work_items[i].pos;
            results[i] = Some(buf);
            encoded_positions.push(pos);
        }
    }

    // Collect results.
    results
        .into_iter()
        .enumerate()
        .map(|(i, r)| r.ok_or_else(|| format!("tile {} was not encoded", i)))
        .collect()
}

// =============================================================================
// Tile Quality Analysis
// =============================================================================

/// Per-tile quality metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct TileQualityMetrics {
    /// Tile column index.
    pub col: u32,
    /// Tile row index.
    pub row: u32,
    /// Estimated PSNR in dB (peak signal-to-noise ratio).
    pub psnr_db: f64,
    /// Estimated SSIM (structural similarity) in [0.0, 1.0].
    pub ssim: f64,
    /// Mean squared error.
    pub mse: f64,
}

/// Compute quality metrics between original and reconstructed tile buffers.
///
/// Both buffers must have the same dimensions and channel count.
pub fn compute_tile_quality(
    original: &TileBuffer,
    reconstructed: &TileBuffer,
) -> Result<TileQualityMetrics, String> {
    if original.data.len() != reconstructed.data.len() {
        return Err("tile buffer sizes do not match".to_string());
    }
    if original.data.is_empty() {
        return Ok(TileQualityMetrics {
            col: original.region.col,
            row: original.region.row,
            psnr_db: f64::INFINITY,
            ssim: 1.0,
            mse: 0.0,
        });
    }

    let n = original.data.len() as f64;

    // MSE
    let mse: f64 = original
        .data
        .iter()
        .zip(reconstructed.data.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>()
        / n;

    // PSNR
    let max_val = 255.0_f64;
    let psnr_db = if mse > 0.0 {
        10.0 * (max_val * max_val / mse).log10()
    } else {
        f64::INFINITY
    };

    // Simplified SSIM (block-level approximation)
    let ssim = compute_ssim_approx(&original.data, &reconstructed.data);

    Ok(TileQualityMetrics {
        col: original.region.col,
        row: original.region.row,
        psnr_db,
        ssim,
        mse,
    })
}

/// Simplified SSIM computation between two equal-length byte slices.
///
/// Uses the standard SSIM formula with C1 = (0.01*255)^2, C2 = (0.03*255)^2.
fn compute_ssim_approx(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len() as f64;
    if n < 1.0 {
        return 1.0;
    }

    let c1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    let c2: f64 = (0.03 * 255.0) * (0.03 * 255.0);

    let mut sum_a: f64 = 0.0;
    let mut sum_b: f64 = 0.0;
    let mut sum_a2: f64 = 0.0;
    let mut sum_b2: f64 = 0.0;
    let mut sum_ab: f64 = 0.0;

    for (&va, &vb) in a.iter().zip(b.iter()) {
        let fa = va as f64;
        let fb = vb as f64;
        sum_a += fa;
        sum_b += fb;
        sum_a2 += fa * fa;
        sum_b2 += fb * fb;
        sum_ab += fa * fb;
    }

    let mu_a = sum_a / n;
    let mu_b = sum_b / n;
    let sigma_a2 = (sum_a2 / n) - mu_a * mu_a;
    let sigma_b2 = (sum_b2 / n) - mu_b * mu_b;
    let sigma_ab = (sum_ab / n) - mu_a * mu_b;

    let numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2);
    let denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a2 + sigma_b2 + c2);

    if denominator > 0.0 {
        (numerator / denominator).clamp(-1.0, 1.0)
    } else {
        1.0
    }
}

/// Compute quality metrics for all tiles by comparing original and reconstructed frames.
pub fn analyse_frame_quality(
    layout: &TileLayout,
    original_frame: &[u8],
    reconstructed_frame: &[u8],
    channels: u8,
) -> Result<Vec<TileQualityMetrics>, String> {
    let fw = layout.config.frame_width;
    let frame_stride = fw as usize * channels as usize;

    layout
        .tiles
        .iter()
        .map(|region| {
            let mut orig_buf = TileBuffer::new(region.clone(), channels);
            let mut recon_buf = TileBuffer::new(region.clone(), channels);
            orig_buf.extract_from_frame(original_frame, frame_stride);
            recon_buf.extract_from_frame(reconstructed_frame, frame_stride);
            compute_tile_quality(&orig_buf, &recon_buf)
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // TileConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_config_default() {
        let cfg = TileConfig::default();
        assert_eq!(cfg.tile_cols, 1);
        assert_eq!(cfg.tile_rows, 1);
        assert_eq!(cfg.num_threads, 0);
        assert_eq!(cfg.frame_width, 0);
        assert_eq!(cfg.frame_height, 0);
    }

    #[test]
    fn test_tile_config_builder() {
        let cfg = TileConfig::new()
            .tile_cols(4)
            .tile_rows(3)
            .num_threads(8)
            .frame_width(1920)
            .frame_height(1080);

        assert_eq!(cfg.tile_cols, 4);
        assert_eq!(cfg.tile_rows, 3);
        assert_eq!(cfg.num_threads, 8);
        assert_eq!(cfg.frame_width, 1920);
        assert_eq!(cfg.frame_height, 1080);
    }

    #[test]
    fn test_tile_config_clamp_cols() {
        // Values > 64 are clamped.
        let cfg = TileConfig::new().tile_cols(100);
        assert_eq!(cfg.tile_cols, 64);
    }

    #[test]
    fn test_tile_config_thread_count_auto() {
        let cfg = TileConfig::new().num_threads(0);
        assert!(cfg.thread_count() >= 1);
    }

    #[test]
    fn test_tile_config_thread_count_explicit() {
        let cfg = TileConfig::new().num_threads(4);
        assert_eq!(cfg.thread_count(), 4);
    }

    // -----------------------------------------------------------------------
    // TileRegion
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_region_area() {
        let r = TileRegion::new(0, 0, 0, 0, 100, 50);
        assert_eq!(r.area(), 5000);
    }

    #[test]
    fn test_tile_region_contains() {
        let r = TileRegion::new(1, 0, 50, 0, 50, 50);
        assert!(r.contains(50, 0));
        assert!(r.contains(99, 49));
        assert!(!r.contains(49, 0)); // left of region
        assert!(!r.contains(100, 0)); // right boundary (exclusive)
        assert!(!r.contains(50, 50)); // bottom boundary (exclusive)
    }

    #[test]
    fn test_tile_region_pixel_ranges() {
        let r = TileRegion::new(0, 1, 0, 100, 200, 80);
        assert_eq!(r.pixel_range_x(), 0..200);
        assert_eq!(r.pixel_range_y(), 100..180);
    }

    // -----------------------------------------------------------------------
    // TileLayout – divisible dimensions
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_layout_2x2_divisible() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(100)
            .frame_height(100);

        let layout = TileLayout::new(cfg);
        assert_eq!(layout.tile_count(), 4);

        // All tiles should be 50×50 for an evenly-divisible frame.
        for tile in layout.tiles() {
            assert_eq!(tile.width, 50);
            assert_eq!(tile.height, 50);
        }

        // Total area must equal frame area.
        let total: u64 = layout.tiles().iter().map(|t| t.area()).sum();
        assert_eq!(total, 100 * 100);
    }

    #[test]
    fn test_tile_layout_get_tile() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(100)
            .frame_height(100);

        let layout = TileLayout::new(cfg);

        let tl = layout.get_tile(0, 0).expect("should succeed");
        assert_eq!((tl.x, tl.y), (0, 0));

        let tr = layout.get_tile(1, 0).expect("should succeed");
        assert_eq!(tr.x, 50);

        let bl = layout.get_tile(0, 1).expect("should succeed");
        assert_eq!(bl.y, 50);

        assert!(layout.get_tile(2, 0).is_none());
    }

    // -----------------------------------------------------------------------
    // TileLayout – non-divisible dimensions
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_layout_2x2_non_divisible() {
        // 101×101 with 2×2 tiles: nominal 50×50, last col/row gets remainder.
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(101)
            .frame_height(101);

        let layout = TileLayout::new(cfg);
        assert_eq!(layout.tile_count(), 4);

        // Top-left: 50×50
        let tl = layout.get_tile(0, 0).expect("should succeed");
        assert_eq!(tl.width, 50);
        assert_eq!(tl.height, 50);

        // Top-right: 51×50 (gets the 1-pixel remainder in x)
        let tr = layout.get_tile(1, 0).expect("should succeed");
        assert_eq!(tr.width, 51);
        assert_eq!(tr.height, 50);

        // Bottom-left: 50×51
        let bl = layout.get_tile(0, 1).expect("should succeed");
        assert_eq!(bl.width, 50);
        assert_eq!(bl.height, 51);

        // Bottom-right: 51×51
        let br = layout.get_tile(1, 1).expect("should succeed");
        assert_eq!(br.width, 51);
        assert_eq!(br.height, 51);

        // Total area == 101×101
        let total: u64 = layout.tiles().iter().map(|t| t.area()).sum();
        assert_eq!(total, 101 * 101);
    }

    #[test]
    fn test_tile_layout_non_divisible_coverage() {
        // Verify every pixel is covered exactly once.
        let fw = 97u32;
        let fh = 83u32;
        let cfg = TileConfig::new()
            .tile_cols(3)
            .tile_rows(3)
            .frame_width(fw)
            .frame_height(fh);

        let layout = TileLayout::new(cfg);
        let mut counts = vec![0u32; (fw * fh) as usize];

        for tile in layout.tiles() {
            for py in tile.pixel_range_y() {
                for px in tile.pixel_range_x() {
                    counts[(py * fw + px) as usize] += 1;
                }
            }
        }

        assert!(
            counts.iter().all(|&c| c == 1),
            "some pixels are covered 0 or 2+ times"
        );
    }

    // -----------------------------------------------------------------------
    // TileLayout – tile_for_pixel
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_for_pixel() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(100)
            .frame_height(100);

        let layout = TileLayout::new(cfg);

        let t = layout.tile_for_pixel(25, 25).expect("should succeed");
        assert_eq!((t.col, t.row), (0, 0));

        let t = layout.tile_for_pixel(75, 25).expect("should succeed");
        assert_eq!((t.col, t.row), (1, 0));

        let t = layout.tile_for_pixel(25, 75).expect("should succeed");
        assert_eq!((t.col, t.row), (0, 1));

        let t = layout.tile_for_pixel(75, 75).expect("should succeed");
        assert_eq!((t.col, t.row), (1, 1));

        // Out-of-frame pixel.
        assert!(layout.tile_for_pixel(200, 200).is_none());
    }

    // -----------------------------------------------------------------------
    // TileBuffer
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_buffer_new() {
        let region = TileRegion::new(0, 0, 0, 0, 8, 6);
        let buf = TileBuffer::new(region, 3);
        assert_eq!(buf.stride, 8 * 3);
        assert_eq!(buf.data.len(), 8 * 6 * 3);
        assert!(buf.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_tile_buffer_extract() {
        // 4×4 single-channel frame: pixels 0..16
        let frame: Vec<u8> = (0u8..16).collect();
        let region = TileRegion::new(0, 0, 1, 1, 2, 2); // 2×2 tile at offset (1,1)
        let mut buf = TileBuffer::new(region, 1);
        buf.extract_from_frame(&frame, 4); // frame stride = 4

        // Row 1, col 1 → index 5; row 1, col 2 → index 6
        // Row 2, col 1 → index 9; row 2, col 2 → index 10
        assert_eq!(buf.data, vec![5, 6, 9, 10]);
    }

    #[test]
    fn test_tile_buffer_write_back() {
        let region = TileRegion::new(0, 0, 1, 1, 2, 2);
        let mut buf = TileBuffer::new(region, 1);
        buf.data = vec![5, 6, 9, 10];

        let mut frame = vec![0u8; 16];
        buf.write_to_frame(&mut frame, 4);

        assert_eq!(frame[5], 5);
        assert_eq!(frame[6], 6);
        assert_eq!(frame[9], 9);
        assert_eq!(frame[10], 10);
        // Other pixels untouched.
        assert_eq!(frame[0], 0);
    }

    // -----------------------------------------------------------------------
    // ParallelTileEncoder – split and merge roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_split_merge_roundtrip_divisible() {
        let fw = 64u32;
        let fh = 64u32;
        let channels = 3u8;

        let config = TileConfig::new()
            .tile_cols(4)
            .tile_rows(4)
            .frame_width(fw)
            .frame_height(fh);

        let encoder = ParallelTileEncoder::new(config);

        // Create a unique frame.
        let frame: Vec<u8> = (0u8..=255)
            .cycle()
            .take((fw * fh * channels as u32) as usize)
            .collect();
        let tiles = encoder.split_frame(&frame, channels);
        assert_eq!(tiles.len(), 16);

        let merged = ParallelTileEncoder::merge_tiles(&tiles, fw, fh, channels);
        assert_eq!(merged, frame, "roundtrip failed for divisible dimensions");
    }

    #[test]
    fn test_split_merge_roundtrip_non_divisible() {
        let fw = 101u32;
        let fh = 99u32;
        let channels = 1u8;

        let config = TileConfig::new()
            .tile_cols(3)
            .tile_rows(3)
            .frame_width(fw)
            .frame_height(fh);

        let encoder = ParallelTileEncoder::new(config);

        let frame: Vec<u8> = (0u8..=255).cycle().take((fw * fh) as usize).collect();
        let tiles = encoder.split_frame(&frame, channels);

        let merged = ParallelTileEncoder::merge_tiles(&tiles, fw, fh, channels);
        assert_eq!(
            merged, frame,
            "roundtrip failed for non-divisible dimensions"
        );
    }

    // -----------------------------------------------------------------------
    // ParallelTileEncoder – encode_tiles_parallel
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_tiles_parallel_identity() {
        let fw = 64u32;
        let fh = 64u32;
        let channels = 3u8;

        let config = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(fw)
            .frame_height(fh);

        let encoder = ParallelTileEncoder::new(config);

        let frame: Vec<u8> = (0u8..=255)
            .cycle()
            .take((fw * fh * channels as u32) as usize)
            .collect();
        let tiles = encoder.split_frame(&frame, channels);

        // Identity encode: return each tile unchanged.
        let processed = encoder
            .encode_tiles_parallel(tiles, |tile| Ok(tile))
            .expect("should succeed");

        let merged = ParallelTileEncoder::merge_tiles(&processed, fw, fh, channels);
        assert_eq!(merged, frame, "parallel identity encode broke the frame");
    }

    #[test]
    fn test_encode_tiles_parallel_error_propagates() {
        let config = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(64)
            .frame_height(64);

        let encoder = ParallelTileEncoder::new(config);
        let frame = vec![0u8; 64 * 64 * 3];
        let tiles = encoder.split_frame(&frame, 3);

        let result = encoder.encode_tiles_parallel(tiles, |_| Err("deliberate error".to_string()));
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_tiles_parallel_transform() {
        // Invert all pixel values and check the result.
        let fw = 32u32;
        let fh = 32u32;
        let channels = 1u8;

        let config = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(fw)
            .frame_height(fh);

        let encoder = ParallelTileEncoder::new(config);

        let frame: Vec<u8> = (0u8..=255).cycle().take((fw * fh) as usize).collect();
        let tiles = encoder.split_frame(&frame, channels);

        let inverted = encoder
            .encode_tiles_parallel(tiles, |mut tile| {
                for b in &mut tile.data {
                    *b = 255 - *b;
                }
                Ok(tile)
            })
            .expect("should succeed");

        let merged = ParallelTileEncoder::merge_tiles(&inverted, fw, fh, channels);
        let expected: Vec<u8> = frame.iter().map(|&b| 255 - b).collect();
        assert_eq!(merged, expected, "inversion result mismatch");
    }

    // -----------------------------------------------------------------------
    // Tile Complexity Analysis
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyse_tile_complexity_uniform() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(8)
            .frame_height(8);
        let layout = TileLayout::new(cfg);
        // Uniform frame: all pixels = 128.
        let frame = vec![128u8; 8 * 8];
        let complexities = analyse_tile_complexity(&layout, &frame, 1);
        assert_eq!(complexities.len(), 4);
        for c in &complexities {
            assert!(
                c.variance < 1.0,
                "uniform frame should have near-zero variance"
            );
            assert!(
                c.edge_density < 1.0,
                "uniform frame should have near-zero edge density"
            );
        }
    }

    #[test]
    fn test_analyse_tile_complexity_gradient() {
        let cfg = TileConfig::new()
            .tile_cols(1)
            .tile_rows(1)
            .frame_width(16)
            .frame_height(16);
        let layout = TileLayout::new(cfg);
        // Gradient frame: increasing pixel values.
        let frame: Vec<u8> = (0..16 * 16).map(|i| (i % 256) as u8).collect();
        let complexities = analyse_tile_complexity(&layout, &frame, 1);
        assert_eq!(complexities.len(), 1);
        assert!(
            complexities[0].variance > 0.0,
            "gradient should have non-zero variance"
        );
        assert!(
            complexities[0].edge_density > 0.0,
            "gradient should have non-zero edge density"
        );
    }

    #[test]
    fn test_analyse_complexity_score_normalised() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(16)
            .frame_height(16);
        let layout = TileLayout::new(cfg);
        // Mixed frame: top-left noisy, rest uniform.
        let mut frame = vec![128u8; 16 * 16];
        for y in 0..8 {
            for x in 0..8 {
                frame[y * 16 + x] = ((x * 31 + y * 17) % 256) as u8;
            }
        }
        let complexities = analyse_tile_complexity(&layout, &frame, 1);
        for c in &complexities {
            assert!(
                c.score >= 0.0 && c.score <= 1.0,
                "score out of range: {}",
                c.score
            );
        }
    }

    // -----------------------------------------------------------------------
    // Adaptive Tile Partitioning
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_partition_below_threshold() {
        let complexities = vec![
            TileComplexity {
                col: 0,
                row: 0,
                variance: 10.0,
                edge_density: 5.0,
                score: 0.2,
            },
            TileComplexity {
                col: 1,
                row: 0,
                variance: 15.0,
                edge_density: 7.0,
                score: 0.3,
            },
        ];
        let partitions = adaptive_tile_partition(&complexities, 0.5, 4);
        assert_eq!(partitions, vec![(1, 1), (1, 1)]);
    }

    #[test]
    fn test_adaptive_partition_above_threshold() {
        let complexities = vec![TileComplexity {
            col: 0,
            row: 0,
            variance: 100.0,
            edge_density: 50.0,
            score: 0.9,
        }];
        let partitions = adaptive_tile_partition(&complexities, 0.5, 4);
        assert!(partitions[0].0 >= 2, "high-complexity tile should be split");
        assert!(partitions[0].1 >= 2, "high-complexity tile should be split");
    }

    #[test]
    fn test_adaptive_partition_max_split_clamped() {
        let complexities = vec![TileComplexity {
            col: 0,
            row: 0,
            variance: 1000.0,
            edge_density: 500.0,
            score: 1.0,
        }];
        let partitions = adaptive_tile_partition(&complexities, 0.0, 4);
        assert!(partitions[0].0 <= 4);
        assert!(partitions[0].1 <= 4);
    }

    // -----------------------------------------------------------------------
    // Tile-Level Rate Control
    // -----------------------------------------------------------------------

    #[test]
    fn test_allocate_tile_bits_proportional() {
        let complexities = vec![
            TileComplexity {
                col: 0,
                row: 0,
                variance: 100.0,
                edge_density: 50.0,
                score: 0.8,
            },
            TileComplexity {
                col: 1,
                row: 0,
                variance: 10.0,
                edge_density: 5.0,
                score: 0.2,
            },
        ];
        let budgets = allocate_tile_bits(&complexities, 10000, 100, 28.0);
        assert_eq!(budgets.len(), 2);
        // Higher complexity tile should get more bits.
        assert!(budgets[0].bits > budgets[1].bits);
        // Total should be close to the budget (rounding may cause +-1 per tile).
        let total: u64 = budgets.iter().map(|b| b.bits).sum();
        let diff = (total as i64 - 10000_i64).unsigned_abs();
        assert!(
            diff <= budgets.len() as u64,
            "total {} too far from 10000",
            total
        );
    }

    #[test]
    fn test_allocate_tile_bits_minimum_floor() {
        let complexities = vec![TileComplexity {
            col: 0,
            row: 0,
            variance: 0.0,
            edge_density: 0.0,
            score: 0.0,
        }];
        let budgets = allocate_tile_bits(&complexities, 1000, 500, 28.0);
        assert!(budgets[0].bits >= 500);
    }

    #[test]
    fn test_allocate_tile_bits_qp_range() {
        let complexities = vec![
            TileComplexity {
                col: 0,
                row: 0,
                variance: 0.0,
                edge_density: 0.0,
                score: 0.0,
            },
            TileComplexity {
                col: 1,
                row: 0,
                variance: 100.0,
                edge_density: 50.0,
                score: 1.0,
            },
        ];
        let budgets = allocate_tile_bits(&complexities, 10000, 100, 28.0);
        for b in &budgets {
            assert!(b.qp >= 0.0 && b.qp <= 51.0, "QP out of range: {}", b.qp);
        }
    }

    #[test]
    fn test_allocate_tile_bits_empty() {
        let budgets = allocate_tile_bits(&[], 10000, 100, 28.0);
        assert!(budgets.is_empty());
    }

    // -----------------------------------------------------------------------
    // Tile Dependency Tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_dependency_graph_build_2x2() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(64)
            .frame_height(64);
        let layout = TileLayout::new(cfg);
        let graph = TileDependencyGraph::build(&layout);

        // (0,0): no deps; (1,0): left; (0,1): top; (1,1): left + top
        assert_eq!(graph.dependencies_of(0, 0).len(), 0);
        assert_eq!(graph.dependencies_of(1, 0).len(), 1);
        assert_eq!(graph.dependencies_of(0, 1).len(), 1);
        assert_eq!(graph.dependencies_of(1, 1).len(), 2);
    }

    #[test]
    fn test_dependency_graph_ready_tiles() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(64)
            .frame_height(64);
        let layout = TileLayout::new(cfg);
        let graph = TileDependencyGraph::build(&layout);

        // Initially only (0,0) is ready (no dependencies).
        let ready = graph.ready_tiles(&[]);
        assert!(ready.contains(&(0, 0)));
        assert!(!ready.contains(&(1, 1)));

        // After encoding (0,0), (1,0) and (0,1) become ready.
        let ready = graph.ready_tiles(&[(0, 0)]);
        assert!(ready.contains(&(1, 0)));
        assert!(ready.contains(&(0, 1)));
    }

    #[test]
    fn test_dependency_graph_add_mv() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(64)
            .frame_height(64);
        let layout = TileLayout::new(cfg);
        let mut graph = TileDependencyGraph::build(&layout);

        let before = graph.dependencies_of(0, 0).len();
        graph.add_mv_dependency((0, 0), (1, 1));
        let after = graph.dependencies_of(0, 0).len();
        assert_eq!(after, before + 1);
    }

    // -----------------------------------------------------------------------
    // Tile Quality Analysis
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_quality_identical() {
        let region = TileRegion::new(0, 0, 0, 0, 4, 4);
        let mut buf = TileBuffer::new(region, 1);
        buf.data = vec![100; 16];
        let metrics = compute_tile_quality(&buf, &buf).expect("should succeed");
        assert_eq!(metrics.mse, 0.0);
        assert!(metrics.psnr_db.is_infinite());
        assert!((metrics.ssim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tile_quality_different() {
        let region = TileRegion::new(0, 0, 0, 0, 4, 4);
        let mut orig = TileBuffer::new(region.clone(), 1);
        let mut recon = TileBuffer::new(region, 1);
        orig.data = vec![100; 16];
        recon.data = vec![110; 16];

        let metrics = compute_tile_quality(&orig, &recon).expect("should succeed");
        assert!(metrics.mse > 0.0);
        assert!(metrics.psnr_db > 0.0 && metrics.psnr_db < 100.0);
        assert!(metrics.ssim < 1.0);
    }

    #[test]
    fn test_tile_quality_size_mismatch() {
        let r1 = TileRegion::new(0, 0, 0, 0, 4, 4);
        let r2 = TileRegion::new(0, 0, 0, 0, 4, 2);
        let buf1 = TileBuffer::new(r1, 1);
        let buf2 = TileBuffer::new(r2, 1);
        let result = compute_tile_quality(&buf1, &buf2);
        assert!(result.is_err());
    }

    #[test]
    fn test_analyse_frame_quality_full() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(8)
            .frame_height(8);
        let layout = TileLayout::new(cfg);
        let original: Vec<u8> = (0..64).collect();
        let reconstructed = original.clone();
        let metrics =
            analyse_frame_quality(&layout, &original, &reconstructed, 1).expect("should succeed");
        assert_eq!(metrics.len(), 4);
        for m in &metrics {
            assert_eq!(m.mse, 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // Wavefront Encoding
    // -----------------------------------------------------------------------

    #[test]
    fn test_wavefront_encode_2x2() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(8)
            .frame_height(8);
        let layout = TileLayout::new(cfg);
        let graph = TileDependencyGraph::build(&layout);

        let encoder = ParallelTileEncoder::new(
            TileConfig::new()
                .tile_cols(2)
                .tile_rows(2)
                .frame_width(8)
                .frame_height(8),
        );
        let frame: Vec<u8> = (0..64).collect();
        let tiles = encoder.split_frame(&frame, 1);

        let work_items: Vec<TileWorkItem> = tiles
            .into_iter()
            .map(|buf| TileWorkItem {
                pos: (buf.region.col, buf.region.row),
                buffer: buf,
                bit_budget: None,
            })
            .collect();

        let results = encode_tiles_wavefront(&graph, work_items, |item| Ok(item.buffer))
            .expect("should succeed");
        assert_eq!(results.len(), 4);
    }
}
