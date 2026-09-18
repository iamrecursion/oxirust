//! Image tiling for spatial indexing and parallel processing.
//!
//! Beyond geometry types, this module provides [`scale_tiled`], a cache-blocked
//! execution strategy that partitions the destination image into a [`TileGrid`]
//! and scales each tile independently for improved L1/L2 cache locality on
//! large downscales.  Output is **bit-exact** versus a sequential full-image
//! bilinear scale because every destination pixel is computed solely from its
//! source coordinates with no inter-pixel state.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use rayon::prelude::*;

/// A rectangular tile within an image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tile {
    /// Left edge of the tile (inclusive)
    pub x: u32,
    /// Top edge of the tile (inclusive)
    pub y: u32,
    /// Width of the tile in pixels
    pub width: u32,
    /// Height of the tile in pixels
    pub height: u32,
}

impl Tile {
    /// Create a new tile.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Return the area (width * height) of this tile.
    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Return true if the pixel `(px, py)` is inside this tile.
    pub fn contains(&self, px: u32, py: u32) -> bool {
        px >= self.x
            && px < self.x.saturating_add(self.width)
            && py >= self.y
            && py < self.y.saturating_add(self.height)
    }

    /// Return true if this tile overlaps with `other`.
    pub fn overlaps(&self, other: &Tile) -> bool {
        let self_right = self.x.saturating_add(self.width);
        let self_bottom = self.y.saturating_add(self.height);
        let other_right = other.x.saturating_add(other.width);
        let other_bottom = other.y.saturating_add(other.height);

        self.x < other_right
            && self_right > other.x
            && self.y < other_bottom
            && self_bottom > other.y
    }
}

/// A regular grid of tiles covering an image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileGrid {
    /// Number of tile columns
    pub cols: u32,
    /// Number of tile rows
    pub rows: u32,
    /// Width of each tile in pixels
    pub tile_w: u32,
    /// Height of each tile in pixels
    pub tile_h: u32,
}

impl TileGrid {
    /// Create a `TileGrid` that covers an image of `img_w x img_h` pixels
    /// using tiles of size `tile_size x tile_size`.
    ///
    /// The number of columns and rows is rounded up so the grid covers
    /// the full image.
    pub fn new(img_w: u32, img_h: u32, tile_size: u32) -> Self {
        let tile_size = tile_size.max(1);
        let cols = img_w.div_ceil(tile_size);
        let rows = img_h.div_ceil(tile_size);
        Self {
            cols,
            rows,
            tile_w: tile_size,
            tile_h: tile_size,
        }
    }

    /// Return the total number of tiles in the grid.
    pub fn tile_count(&self) -> u32 {
        self.cols * self.rows
    }

    /// Return the `Tile` at grid position `(col, row)`.
    ///
    /// The tile's position is `(col * tile_w, row * tile_h)`.
    pub fn tile_at(&self, col: u32, row: u32) -> Tile {
        Tile {
            x: col * self.tile_w,
            y: row * self.tile_h,
            width: self.tile_w,
            height: self.tile_h,
        }
    }

    /// Return the tile that contains pixel `(px, py)`, or `None` if out of range.
    pub fn tile_for_pixel(&self, px: u32, py: u32) -> Option<Tile> {
        let col = px / self.tile_w;
        let row = py / self.tile_h;
        if col < self.cols && row < self.rows {
            Some(self.tile_at(col, row))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tiled scaling — cache-blocked execution strategy
// ---------------------------------------------------------------------------

/// Sample a single destination pixel from `src` using bilinear interpolation.
///
/// `src` is a packed interleaved image with `channels` bytes per pixel,
/// stored row-major with a stride of `src_w * channels` bytes.
///
/// The source coordinate is derived from the destination coordinates using
/// the standard half-pixel-centre mapping:
///   src_x = (dst_x + 0.5) * (src_w / dst_w) − 0.5
///
/// This function has **no inter-pixel state**, so it can be called for any
/// (dst_x, dst_y) in any order and always produces the same result.
#[inline]
fn bilinear_pixel(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_x: u32,
    dst_y: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u8,
) -> [u8; 4] {
    let ch = channels as usize;
    let sw = src_w as f64;
    let sh = src_h as f64;
    let dw = dst_w as f64;
    let dh = dst_h as f64;

    // Half-pixel-centre mapping for exact alignment at src corners.
    let sx = (dst_x as f64 + 0.5) * sw / dw - 0.5;
    let sy = (dst_y as f64 + 0.5) * sh / dh - 0.5;

    let sx = sx.max(0.0).min(sw - 1.0);
    let sy = sy.max(0.0).min(sh - 1.0);

    let x0 = sx.floor() as usize;
    let y0 = sy.floor() as usize;
    let x1 = (x0 + 1).min(src_w as usize - 1);
    let y1 = (y0 + 1).min(src_h as usize - 1);

    let fx = sx - sx.floor();
    let fy = sy - sy.floor();

    let wx0 = 1.0 - fx;
    let wx1 = fx;
    let wy0 = 1.0 - fy;
    let wy1 = fy;

    let sw_usize = src_w as usize;
    let mut out = [0u8; 4];
    for c in 0..ch {
        let v00 = src[(y0 * sw_usize + x0) * ch + c] as f64;
        let v10 = src[(y0 * sw_usize + x1) * ch + c] as f64;
        let v01 = src[(y1 * sw_usize + x0) * ch + c] as f64;
        let v11 = src[(y1 * sw_usize + x1) * ch + c] as f64;
        let v = wy0 * (wx0 * v00 + wx1 * v10) + wy1 * (wx0 * v01 + wx1 * v11);
        out[c] = v.round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Scale `src` (size `src_w × src_h`, `channels` bytes per pixel, row-major)
/// to `dst_w × dst_h` using a **tiled execution strategy** for cache locality.
///
/// The destination image is partitioned into a [`TileGrid`] of `tile_size ×
/// tile_size` tiles.  Each tile is scaled independently (using bilinear
/// interpolation) and written into the corresponding region of the output
/// buffer.  Tiles are processed in parallel via Rayon.
///
/// # Bit-exactness
///
/// Because each destination pixel is computed purely from its source
/// coordinates (no inter-pixel state), the output is **bit-exact** compared
/// to a sequential full-image bilinear scale with the same mapping function.
///
/// # Parameters
///
/// * `src` — source pixel buffer, `src_w * src_h * channels` bytes.
/// * `src_w`, `src_h` — source dimensions in pixels.
/// * `dst_w`, `dst_h` — destination dimensions in pixels.
/// * `channels` — bytes per pixel (1 = greyscale, 3 = RGB, 4 = RGBA).
/// * `tile_size` — tile dimension in pixels (e.g. 64).  Clamped to at least 1.
///
/// Returns an empty `Vec<u8>` if any dimension is zero.
pub fn scale_tiled(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u8,
    tile_size: u32,
) -> Vec<u8> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 || channels == 0 {
        return Vec::new();
    }
    let ch = channels as usize;
    let expected = src_w as usize * src_h as usize * ch;
    if src.len() < expected {
        return Vec::new();
    }

    let tile_size = tile_size.max(1);
    let grid = TileGrid::new(dst_w, dst_h, tile_size);
    let n_tiles = grid.tile_count() as usize;

    // Allocate destination buffer.
    let dst_stride = dst_w as usize * ch;
    let mut output: Vec<u8> = vec![0u8; dst_w as usize * dst_h as usize * ch];

    // Collect (tile_index -> (col, row)) pairs for parallel iteration.
    let tile_indices: Vec<(u32, u32)> = (0..n_tiles as u32)
        .map(|i| (i % grid.cols, i / grid.cols))
        .collect();

    // Process each tile in parallel.  Each tile writes to a disjoint region
    // of the output buffer — we use a temporary per-tile Vec and then copy
    // into the correct rows.  This avoids unsafe aliasing while still giving
    // parallelism.
    let tile_results: Vec<(u32, u32, u32, u32, Vec<u8>)> = tile_indices
        .par_iter()
        .map(|&(col, row)| {
            let tile = grid.tile_at(col, row);

            // Clip tile to actual dst dimensions (edge tiles may overhang).
            let tx_end = (tile.x + tile.width).min(dst_w);
            let ty_end = (tile.y + tile.height).min(dst_h);
            let tw = tx_end.saturating_sub(tile.x);
            let th = ty_end.saturating_sub(tile.y);

            if tw == 0 || th == 0 {
                return (tile.x, tile.y, 0, 0, Vec::new());
            }

            let mut tile_buf = vec![0u8; tw as usize * th as usize * ch];
            for ty in 0..th {
                for tx in 0..tw {
                    let dst_x = tile.x + tx;
                    let dst_y = tile.y + ty;
                    let pixel =
                        bilinear_pixel(src, src_w, src_h, dst_x, dst_y, dst_w, dst_h, channels);
                    let off = (ty as usize * tw as usize + tx as usize) * ch;
                    tile_buf[off..off + ch].copy_from_slice(&pixel[..ch]);
                }
            }
            (tile.x, tile.y, tw, th, tile_buf)
        })
        .collect();

    // Merge tile results into the output buffer (sequential, disjoint rows).
    for (tx, ty, tw, th, tile_buf) in tile_results {
        if tw == 0 || th == 0 {
            continue;
        }
        for row in 0..th as usize {
            let dst_row_off = (ty as usize + row) * dst_stride + tx as usize * ch;
            let src_row_off = row * tw as usize * ch;
            let len = tw as usize * ch;
            output[dst_row_off..dst_row_off + len]
                .copy_from_slice(&tile_buf[src_row_off..src_row_off + len]);
        }
    }

    output
}

/// Sequential full-image bilinear scale using the **same** per-pixel mapping
/// as [`scale_tiled`].
///
/// This function exists for testing the bit-exactness guarantee: tiled and
/// non-tiled paths must produce identical byte-for-byte output.
///
/// Returns an empty `Vec<u8>` if any dimension is zero.
pub fn scale_reference(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u8,
) -> Vec<u8> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 || channels == 0 {
        return Vec::new();
    }
    let ch = channels as usize;
    let expected = src_w as usize * src_h as usize * ch;
    if src.len() < expected {
        return Vec::new();
    }

    let mut output = vec![0u8; dst_w as usize * dst_h as usize * ch];
    let dst_stride = dst_w as usize * ch;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let pixel = bilinear_pixel(src, src_w, src_h, dx, dy, dst_w, dst_h, channels);
            let off = dy as usize * dst_stride + dx as usize * ch;
            output[off..off + ch].copy_from_slice(&pixel[..ch]);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_new() {
        let t = Tile::new(10, 20, 64, 64);
        assert_eq!(t.x, 10);
        assert_eq!(t.y, 20);
        assert_eq!(t.width, 64);
        assert_eq!(t.height, 64);
    }

    #[test]
    fn test_tile_area() {
        let t = Tile::new(0, 0, 100, 200);
        assert_eq!(t.area(), 20_000);
    }

    #[test]
    fn test_tile_area_large() {
        // Ensure no overflow with u32 dimensions
        let t = Tile::new(0, 0, 65535, 65535);
        assert_eq!(t.area(), 65535u64 * 65535u64);
    }

    #[test]
    fn test_tile_contains_inside() {
        let t = Tile::new(10, 10, 20, 20);
        assert!(t.contains(15, 15));
        assert!(t.contains(10, 10));
    }

    #[test]
    fn test_tile_contains_outside() {
        let t = Tile::new(10, 10, 20, 20);
        assert!(!t.contains(30, 15)); // right edge exclusive
        assert!(!t.contains(5, 15)); // left of tile
        assert!(!t.contains(15, 5)); // above tile
    }

    #[test]
    fn test_tile_contains_bottom_right() {
        let t = Tile::new(0, 0, 8, 8);
        assert!(t.contains(7, 7));
        assert!(!t.contains(8, 7));
        assert!(!t.contains(7, 8));
    }

    #[test]
    fn test_tile_overlaps_true() {
        let a = Tile::new(0, 0, 10, 10);
        let b = Tile::new(5, 5, 10, 10);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_tile_overlaps_adjacent_no_overlap() {
        let a = Tile::new(0, 0, 10, 10);
        let b = Tile::new(10, 0, 10, 10); // right next to a, no overlap
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_tile_overlaps_self() {
        let t = Tile::new(5, 5, 10, 10);
        assert!(t.overlaps(&t));
    }

    #[test]
    fn test_tile_overlaps_disjoint() {
        let a = Tile::new(0, 0, 5, 5);
        let b = Tile::new(100, 100, 5, 5);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_grid_new() {
        let g = TileGrid::new(100, 100, 32);
        assert_eq!(g.cols, 4); // ceil(100/32) = 4
        assert_eq!(g.rows, 4);
        assert_eq!(g.tile_w, 32);
        assert_eq!(g.tile_h, 32);
    }

    #[test]
    fn test_grid_tile_count() {
        let g = TileGrid::new(64, 64, 16);
        assert_eq!(g.tile_count(), 16); // 4*4
    }

    #[test]
    fn test_grid_tile_count_non_divisible() {
        let g = TileGrid::new(100, 100, 32);
        // ceil(100/32) = 4, so 4*4 = 16
        assert_eq!(g.tile_count(), 16);
    }

    #[test]
    fn test_grid_tile_at() {
        let g = TileGrid::new(256, 256, 64);
        let t = g.tile_at(1, 2);
        assert_eq!(t.x, 64);
        assert_eq!(t.y, 128);
        assert_eq!(t.width, 64);
        assert_eq!(t.height, 64);
    }

    #[test]
    fn test_grid_tile_for_pixel_found() {
        let g = TileGrid::new(256, 256, 64);
        let t = g.tile_for_pixel(70, 130);
        assert!(t.is_some());
        let t = t.expect("should succeed in test");
        // pixel (70,130) -> col=1, row=2 -> tile at (64, 128)
        assert_eq!(t.x, 64);
        assert_eq!(t.y, 128);
    }

    #[test]
    fn test_grid_tile_for_pixel_out_of_range() {
        let g = TileGrid::new(100, 100, 32);
        // Grid has 4 cols (0..=3) covering x 0..128, but image is 100 wide.
        // Pixel at (200, 50) is beyond cols
        let t = g.tile_for_pixel(200, 50);
        assert!(t.is_none());
    }

    #[test]
    fn test_grid_zero_size_image() {
        let g = TileGrid::new(0, 0, 32);
        assert_eq!(g.tile_count(), 0);
    }

    // ── scale_tiled tests ────────────────────────────────────────────────────

    /// Build a 256×256 RGB checkerboard (32-pixel squares alternating black/white).
    fn checkerboard_rgb(w: u32, h: u32, square: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(w as usize * h as usize * 3);
        for y in 0..h {
            for x in 0..w {
                let on = ((x / square) + (y / square)) % 2 == 0;
                let v = if on { 255u8 } else { 0u8 };
                buf.push(v);
                buf.push(v);
                buf.push(v);
            }
        }
        buf
    }

    #[test]
    fn test_scale_tiled_bitexact_downscale() {
        // 256×256 checkerboard scaled 4× down to 64×64.
        let src = checkerboard_rgb(256, 256, 32);
        let tiled = scale_tiled(&src, 256, 256, 64, 64, 3, 16);
        let reference = scale_reference(&src, 256, 256, 64, 64, 3);
        assert_eq!(tiled.len(), reference.len());
        assert_eq!(
            tiled, reference,
            "tiled downscale must be bit-exact vs reference"
        );
    }

    #[test]
    fn test_scale_tiled_bitexact_upscale() {
        // 64×64 checkerboard scaled 2× up to 128×128.
        let src = checkerboard_rgb(64, 64, 8);
        let tiled = scale_tiled(&src, 64, 64, 128, 128, 3, 32);
        let reference = scale_reference(&src, 64, 64, 128, 128, 3);
        assert_eq!(tiled.len(), reference.len());
        assert_eq!(
            tiled, reference,
            "tiled upscale must be bit-exact vs reference"
        );
    }

    #[test]
    fn test_scale_tiled_bitexact_rgba() {
        // Verify 4-channel (RGBA) path is also bit-exact.
        let src: Vec<u8> = (0u8..=255).cycle().take(64 * 64 * 4).collect();
        let tiled = scale_tiled(&src, 64, 64, 32, 32, 4, 16);
        let reference = scale_reference(&src, 64, 64, 32, 32, 4);
        assert_eq!(tiled, reference, "RGBA tiled must match reference");
    }

    #[test]
    fn test_scale_tiled_edge_tiles() {
        // 100×100 is not divisible by tile_size=32 → edge tiles are clipped.
        let src = checkerboard_rgb(100, 100, 10);
        let tiled = scale_tiled(&src, 100, 100, 100, 100, 3, 32);
        let reference = scale_reference(&src, 100, 100, 100, 100, 3);
        assert_eq!(
            tiled.len(),
            100 * 100 * 3,
            "output size must be dst_w * dst_h * channels"
        );
        assert_eq!(tiled, reference, "edge-tile result must match reference");
    }

    #[test]
    fn test_scale_tiled_zero_size() {
        // Zero-size src — must return empty vec without panic.
        let empty: &[u8] = &[];
        let r1 = scale_tiled(empty, 0, 0, 64, 64, 3, 32);
        assert!(r1.is_empty(), "zero-size src → empty result");

        // Zero-size dst.
        let src = checkerboard_rgb(64, 64, 8);
        let r2 = scale_tiled(&src, 64, 64, 0, 64, 3, 32);
        assert!(r2.is_empty(), "zero dst_w → empty result");

        let r3 = scale_tiled(&src, 64, 64, 64, 0, 3, 32);
        assert!(r3.is_empty(), "zero dst_h → empty result");
    }

    #[test]
    fn test_scale_tiled_single_pixel_tile() {
        // tile_size=1 forces every dst pixel to be its own tile.
        let src = checkerboard_rgb(32, 32, 4);
        let tiled = scale_tiled(&src, 32, 32, 16, 16, 3, 1);
        let reference = scale_reference(&src, 32, 32, 16, 16, 3);
        assert_eq!(tiled, reference, "tile_size=1 must match reference");
    }

    #[test]
    fn test_scale_tiled_output_size() {
        let src = vec![128u8; 64 * 64 * 3];
        let out = scale_tiled(&src, 64, 64, 128, 128, 3, 64);
        assert_eq!(out.len(), 128 * 128 * 3);
    }
}
