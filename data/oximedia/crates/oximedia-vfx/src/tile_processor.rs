//! Tile-based parallel processing helpers for [`VideoEffect`] chains.
//!
//! Large frames are subdivided into rectangular tiles that can be processed
//! independently and in parallel using [`rayon`].  The tile size is configurable;
//! tiles at the right/bottom edges are automatically clipped to the frame bounds.
//!
//! # Design
//!
//! `TileProcessor` splits a frame into `TileSpec`s (rectangles), applies a
//! caller-supplied closure to each tile in parallel, and writes the results into
//! an output buffer.  The closure receives a read-only view of the input data and
//! a mutable slice for its output tile, enabling safe Rust parallelism.
//!
//! Because the output tiles are non-overlapping the writes are safe to perform
//! in parallel without any locking.  This is achieved by pre-computing each
//! tile's byte range in the output buffer and using [`rayon`]'s parallel
//! iterators over disjoint mutable slices.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_vfx::tile_processor::{TileProcessor, TileConfig};
//! use oximedia_vfx::Frame;
//!
//! let config = TileConfig::new(128, 128);
//! let processor = TileProcessor::new(config);
//!
//! let input = Frame::new(1920, 1080).expect("frame");
//! let mut output = Frame::new(1920, 1080).expect("frame");
//!
//! processor.process(&input, &mut output, |_tile, src, dst| {
//!     // Invert every pixel
//!     for (i, (&s, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
//!         *d = if i % 4 == 3 { s } else { 255 - s };
//!     }
//! });
//! ```

use rayon::prelude::*;

/// A rectangular region of a frame (in pixel coordinates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileSpec {
    /// Leftmost pixel column.
    pub x: u32,
    /// Topmost pixel row.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl TileSpec {
    /// Byte offset of this tile's first pixel in a packed RGBA buffer.
    #[must_use]
    pub fn byte_start(&self, frame_width: u32) -> usize {
        ((self.y as usize) * (frame_width as usize) + (self.x as usize)) * 4
    }

    /// Total number of bytes in this tile's RGBA data (not necessarily contiguous
    /// in the frame buffer — individual rows must be read separately).
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Number of bytes in this tile's pixel data (4 channels per pixel).
    #[must_use]
    pub fn byte_count(&self) -> usize {
        self.pixel_count() * 4
    }
}

/// Configuration for the tile processor.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Tile width in pixels.
    pub tile_width: u32,
    /// Tile height in pixels.
    pub tile_height: u32,
}

impl TileConfig {
    /// Create a new tile configuration.
    ///
    /// Both dimensions are clamped to at least 1.
    #[must_use]
    pub fn new(tile_width: u32, tile_height: u32) -> Self {
        Self {
            tile_width: tile_width.max(1),
            tile_height: tile_height.max(1),
        }
    }

    /// Default 128×128 tiles (good for HD frames on modern CPUs).
    #[must_use]
    pub fn default_hd() -> Self {
        Self::new(128, 128)
    }
}

impl Default for TileConfig {
    fn default() -> Self {
        Self::default_hd()
    }
}

/// Parallel tile-based frame processor.
///
/// Splits a frame into non-overlapping rectangular tiles and applies a
/// per-tile closure in parallel using `rayon`.
pub struct TileProcessor {
    config: TileConfig,
}

impl TileProcessor {
    /// Create a new processor with the given configuration.
    #[must_use]
    pub fn new(config: TileConfig) -> Self {
        Self { config }
    }

    /// Enumerate all tiles for a frame of `width × height` pixels.
    #[must_use]
    pub fn tiles(&self, width: u32, height: u32) -> Vec<TileSpec> {
        if width == 0 || height == 0 {
            return Vec::new();
        }
        let tw = self.config.tile_width;
        let th = self.config.tile_height;
        let cols = width.div_ceil(tw);
        let rows = height.div_ceil(th);
        let mut specs = Vec::with_capacity((cols * rows) as usize);
        for row in 0..rows {
            for col in 0..cols {
                let x = col * tw;
                let y = row * th;
                let w = tw.min(width - x);
                let h = th.min(height - y);
                specs.push(TileSpec {
                    x,
                    y,
                    width: w,
                    height: h,
                });
            }
        }
        specs
    }

    /// Process `input` into `output` in parallel tiles.
    ///
    /// The closure `f` receives:
    /// - `tile`: the [`TileSpec`] describing which region is being processed,
    /// - `src`: a **compacted** RGBA byte slice for this tile (`tile.byte_count()` bytes,
    ///   pixels arranged row-major),
    /// - `dst`: a mutable slice of the same size for writing results.
    ///
    /// Pixels outside the frame bounds are never accessed.
    ///
    /// # Panics
    ///
    /// Panics if `input` and `output` have different dimensions.
    pub fn process<F>(&self, input: &crate::Frame, output: &mut crate::Frame, f: F)
    where
        F: Fn(&TileSpec, &[u8], &mut [u8]) + Send + Sync,
    {
        assert_eq!(
            (input.width, input.height),
            (output.width, output.height),
            "input and output must have the same dimensions"
        );

        let w = input.width;
        let h = input.height;
        let tiles = self.tiles(w, h);

        // Build compacted input tiles.
        // Each tile's pixels are copied row-by-row from the frame buffer into a
        // contiguous Vec so the closure works with simple slices.
        let tile_src: Vec<Vec<u8>> = tiles
            .iter()
            .map(|tile| {
                let mut buf = vec![0u8; tile.byte_count()];
                for row in 0..tile.height {
                    let fy = tile.y + row;
                    let fx = tile.x;
                    let src_offset = ((fy as usize) * (w as usize) + (fx as usize)) * 4;
                    let dst_offset = (row as usize) * (tile.width as usize) * 4;
                    let len = (tile.width as usize) * 4;
                    buf[dst_offset..dst_offset + len]
                        .copy_from_slice(&input.data[src_offset..src_offset + len]);
                }
                buf
            })
            .collect();

        // Allocate output tile buffers
        let mut tile_dst: Vec<Vec<u8>> = tiles
            .iter()
            .map(|tile| vec![0u8; tile.byte_count()])
            .collect();

        // Run tiles in parallel
        tile_src
            .par_iter()
            .zip(tile_dst.par_iter_mut())
            .zip(tiles.par_iter())
            .for_each(|((src, dst), tile)| {
                f(tile, src, dst);
            });

        // Write compacted output tiles back into the output frame
        for (tile, tile_out) in tiles.iter().zip(tile_dst.iter()) {
            for row in 0..tile.height {
                let fy = tile.y + row;
                let fx = tile.x;
                let dst_offset = ((fy as usize) * (w as usize) + (fx as usize)) * 4;
                let src_offset = (row as usize) * (tile.width as usize) * 4;
                let len = (tile.width as usize) * 4;
                output.data[dst_offset..dst_offset + len]
                    .copy_from_slice(&tile_out[src_offset..src_offset + len]);
            }
        }
    }

    /// Process a frame **in-place** via an internal copy.
    ///
    /// Creates a copy of `frame` as the source so the closure can read while
    /// writing back to the same logical buffer.
    pub fn process_inplace<F>(&self, frame: &mut crate::Frame, f: F)
    where
        F: Fn(&TileSpec, &[u8], &mut [u8]) + Send + Sync,
    {
        let input = frame.clone();
        self.process(&input, frame, f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Frame;

    fn solid_frame(w: u32, h: u32, rgba: [u8; 4]) -> Frame {
        let mut f = Frame::new(w, h).expect("frame");
        f.clear(rgba);
        f
    }

    #[test]
    fn test_tile_config_clamp_min_one() {
        let cfg = TileConfig::new(0, 0);
        assert_eq!(cfg.tile_width, 1);
        assert_eq!(cfg.tile_height, 1);
    }

    #[test]
    fn test_tiles_empty_on_zero_dim() {
        let tp = TileProcessor::new(TileConfig::new(64, 64));
        assert!(tp.tiles(0, 100).is_empty());
        assert!(tp.tiles(100, 0).is_empty());
    }

    #[test]
    fn test_tiles_count_exact_divisor() {
        let tp = TileProcessor::new(TileConfig::new(64, 64));
        let tiles = tp.tiles(128, 128);
        assert_eq!(tiles.len(), 4); // 2 × 2
    }

    #[test]
    fn test_tiles_count_non_divisor() {
        let tp = TileProcessor::new(TileConfig::new(64, 64));
        let tiles = tp.tiles(100, 100);
        // ceil(100/64) × ceil(100/64) = 2 × 2 = 4
        assert_eq!(tiles.len(), 4);
    }

    #[test]
    fn test_tiles_cover_full_frame() {
        let tp = TileProcessor::new(TileConfig::new(50, 50));
        let w = 200u32;
        let h = 150u32;
        let tiles = tp.tiles(w, h);
        // Every pixel should be covered by exactly one tile
        let mut coverage = vec![0u8; (w * h) as usize];
        for tile in &tiles {
            for ty in 0..tile.height {
                for tx in 0..tile.width {
                    let px = tile.x + tx;
                    let py = tile.y + ty;
                    coverage[(py * w + px) as usize] += 1;
                }
            }
        }
        assert!(
            coverage.iter().all(|&c| c == 1),
            "every pixel should be covered exactly once"
        );
    }

    #[test]
    fn test_tile_spec_byte_counts() {
        let tile = TileSpec {
            x: 10,
            y: 10,
            width: 32,
            height: 16,
        };
        assert_eq!(tile.pixel_count(), 32 * 16);
        assert_eq!(tile.byte_count(), 32 * 16 * 4);
    }

    #[test]
    fn test_process_identity_closure() {
        let tp = TileProcessor::new(TileConfig::new(32, 32));
        let input = solid_frame(64, 64, [200, 150, 100, 255]);
        let mut output = Frame::new(64, 64).expect("output");

        tp.process(&input, &mut output, |_tile, src, dst| {
            dst.copy_from_slice(src);
        });

        let p = output.get_pixel(32, 32).expect("center");
        assert_eq!(p, [200, 150, 100, 255]);
    }

    #[test]
    fn test_process_invert_closure() {
        let tp = TileProcessor::new(TileConfig::new(16, 16));
        let input = solid_frame(32, 32, [100, 50, 25, 255]);
        let mut output = Frame::new(32, 32).expect("output");

        tp.process(&input, &mut output, |_tile, src, dst| {
            for (chunk_in, chunk_out) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
                chunk_out[0] = 255 - chunk_in[0];
                chunk_out[1] = 255 - chunk_in[1];
                chunk_out[2] = 255 - chunk_in[2];
                chunk_out[3] = chunk_in[3]; // preserve alpha
            }
        });

        let p = output.get_pixel(16, 16).expect("center");
        assert_eq!(p[0], 155); // 255 - 100
        assert_eq!(p[1], 205); // 255 - 50
        assert_eq!(p[2], 230); // 255 - 25
        assert_eq!(p[3], 255);
    }

    #[test]
    fn test_process_inplace_darkens_frame() {
        let tp = TileProcessor::new(TileConfig::new(32, 32));
        let mut frame = solid_frame(64, 64, [200, 200, 200, 255]);

        tp.process_inplace(&mut frame, |_tile, src, dst| {
            for (i, (&s, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
                *d = if i % 4 == 3 { s } else { s / 2 };
            }
        });

        let p = frame.get_pixel(32, 32).expect("center");
        assert_eq!(p[0], 100);
        assert_eq!(p[3], 255);
    }

    #[test]
    fn test_process_single_pixel_frame() {
        let tp = TileProcessor::new(TileConfig::new(64, 64));
        let input = solid_frame(1, 1, [77, 88, 99, 255]);
        let mut output = Frame::new(1, 1).expect("output");

        tp.process(&input, &mut output, |_tile, src, dst| {
            dst.copy_from_slice(src);
        });

        let p = output.get_pixel(0, 0).expect("pixel");
        assert_eq!(p, [77, 88, 99, 255]);
    }
}
