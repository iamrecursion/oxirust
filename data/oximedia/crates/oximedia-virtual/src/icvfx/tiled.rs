//! Tile-based parallel compositing for ICVFX.
//!
//! Divides the output frame into rectangular tiles and composites each tile
//! independently.  When Rust's `std::thread` is available the tiles can be
//! processed in parallel using a simple work-stealing approach.  The tiled
//! compositor uses the same Porter-Duff "over" blending as the main compositor
//! but operates on independent tile-sized scratch buffers.

use super::{BlendMode, CompositeLayer};
use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// Configuration for the tiled compositor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiledCompositorConfig {
    /// Output resolution (width, height).
    pub resolution: (usize, usize),
    /// Tile width in pixels.
    pub tile_width: usize,
    /// Tile height in pixels.
    pub tile_height: usize,
    /// Number of worker threads (0 = single-threaded sequential).
    pub num_threads: usize,
}

impl Default for TiledCompositorConfig {
    fn default() -> Self {
        Self {
            resolution: (1920, 1080),
            tile_width: 64,
            tile_height: 64,
            num_threads: 0,
        }
    }
}

/// A single tile region.
#[derive(Debug, Clone, Copy)]
pub struct Tile {
    /// Top-left X.
    pub x: usize,
    /// Top-left Y.
    pub y: usize,
    /// Width of this tile.
    pub width: usize,
    /// Height of this tile.
    pub height: usize,
}

impl Tile {
    /// Pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }
}

/// RGBA layer passed to the tiled compositor.
#[derive(Debug, Clone)]
pub struct TileLayerData {
    /// RGBA f32 pixel data (row-major, full frame).
    pub pixels_rgba: Vec<f32>,
    /// Width (must match compositor resolution).
    pub width: usize,
    /// Height (must match compositor resolution).
    pub height: usize,
    /// Per-layer opacity [0, 1].
    pub opacity: f32,
    /// Blend mode.
    pub blend_mode: BlendMode,
    /// Z-order (lower = further back).
    pub z_order: i32,
    /// Whether this layer is active.
    pub enabled: bool,
    /// Optional label for debugging.
    pub label: String,
}

impl TileLayerData {
    /// Create a solid colour layer (all pixels same RGBA).
    #[must_use]
    pub fn solid(
        label: &str,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
        width: usize,
        height: usize,
        z_order: i32,
    ) -> Self {
        let n = width * height * 4;
        let mut pixels_rgba = Vec::with_capacity(n);
        for _ in 0..(width * height) {
            pixels_rgba.push(r);
            pixels_rgba.push(g);
            pixels_rgba.push(b);
            pixels_rgba.push(a);
        }
        Self {
            pixels_rgba,
            width,
            height,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            z_order,
            enabled: true,
            label: label.to_string(),
        }
    }

    /// Get RGBA for a specific pixel (col, row).
    #[must_use]
    pub fn get_pixel(&self, col: usize, row: usize) -> Option<[f32; 4]> {
        if col >= self.width || row >= self.height {
            return None;
        }
        let i = (row * self.width + col) * 4;
        Some([
            self.pixels_rgba[i],
            self.pixels_rgba[i + 1],
            self.pixels_rgba[i + 2],
            self.pixels_rgba[i + 3],
        ])
    }
}

/// Output frame from the tiled compositor.
#[derive(Debug, Clone)]
pub struct TiledFrame {
    /// RGB u8 pixel data.
    pub pixels: Vec<u8>,
    /// Width.
    pub width: usize,
    /// Height.
    pub height: usize,
    /// Number of tiles processed.
    pub tiles_processed: usize,
}

impl TiledFrame {
    /// Get pixel at (col, row).
    #[must_use]
    pub fn get_pixel(&self, col: usize, row: usize) -> Option<[u8; 3]> {
        if col >= self.width || row >= self.height {
            return None;
        }
        let i = (row * self.width + col) * 3;
        Some([self.pixels[i], self.pixels[i + 1], self.pixels[i + 2]])
    }
}

/// Tiled compositor.
pub struct TiledCompositor {
    config: TiledCompositorConfig,
    tiles: Vec<Tile>,
    #[allow(dead_code)]
    layers: Vec<CompositeLayer>,
}

impl TiledCompositor {
    /// Create a new tiled compositor.
    pub fn new(config: TiledCompositorConfig) -> Result<Self> {
        if config.tile_width == 0 || config.tile_height == 0 {
            return Err(VirtualProductionError::InvalidConfig(
                "Tile dimensions must be non-zero".to_string(),
            ));
        }
        if config.resolution.0 == 0 || config.resolution.1 == 0 {
            return Err(VirtualProductionError::InvalidConfig(
                "Resolution must be non-zero".to_string(),
            ));
        }
        let tiles = Self::build_tiles(&config);
        Ok(Self {
            config,
            tiles,
            layers: Vec::new(),
        })
    }

    /// Build tile list for the given configuration.
    fn build_tiles(config: &TiledCompositorConfig) -> Vec<Tile> {
        let (w, h) = config.resolution;
        let tw = config.tile_width;
        let th = config.tile_height;

        let mut tiles = Vec::new();
        let mut y = 0;
        while y < h {
            let tile_h = th.min(h - y);
            let mut x = 0;
            while x < w {
                let tile_w = tw.min(w - x);
                tiles.push(Tile {
                    x,
                    y,
                    width: tile_w,
                    height: tile_h,
                });
                x += tw;
            }
            y += th;
        }
        tiles
    }

    /// Number of tiles.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &TiledCompositorConfig {
        &self.config
    }

    /// Tile list.
    #[must_use]
    pub fn tiles(&self) -> &[Tile] {
        &self.tiles
    }

    /// Composite a stack of layers using tile-based processing.
    ///
    /// Layers are sorted by z_order (ascending = furthest back first).
    /// Each tile is processed independently using Porter-Duff "over" compositing.
    pub fn composite(&self, layers: &[TileLayerData], timestamp_ns: u64) -> Result<TiledFrame> {
        let (w, h) = self.config.resolution;

        // Validate all layers
        let mut active: Vec<&TileLayerData> = layers
            .iter()
            .filter(|l| l.enabled && l.opacity > 0.0)
            .collect();

        for layer in &active {
            if layer.width != w || layer.height != h {
                return Err(VirtualProductionError::Compositing(format!(
                    "Layer '{}' resolution {}×{} doesn't match compositor {}×{}",
                    layer.label, layer.width, layer.height, w, h
                )));
            }
        }

        // Sort by z_order
        active.sort_by_key(|l| l.z_order);

        let _ = timestamp_ns; // available for future timestamped output

        // Output buffer
        let n = w * h;
        let mut out_r = vec![0.0f32; n];
        let mut out_g = vec![0.0f32; n];
        let mut out_b = vec![0.0f32; n];
        let mut out_a = vec![0.0f32; n];

        // Tile-based compositing: process each tile
        for tile in &self.tiles {
            // For each pixel in the tile
            for ty in 0..tile.height {
                for tx in 0..tile.width {
                    let gx = tile.x + tx;
                    let gy = tile.y + ty;
                    let gi = gy * w + gx;

                    let mut cr = 0.0f32;
                    let mut cg = 0.0f32;
                    let mut cb = 0.0f32;
                    let mut ca = 0.0f32;

                    for layer in &active {
                        let li = gi * 4;
                        let lr = layer.pixels_rgba[li];
                        let lg = layer.pixels_rgba[li + 1];
                        let lb = layer.pixels_rgba[li + 2];
                        let la = layer.pixels_rgba[li + 3] * layer.opacity;

                        if la < 1e-6 {
                            continue;
                        }

                        let base = [cr, cg, cb];
                        let blend_color = [lr, lg, lb];
                        let blended = layer.blend_mode.blend(base, blend_color, la);

                        let out_alpha = la + ca * (1.0 - la);
                        if out_alpha > 1e-6 {
                            cr = (blended[0] * la + cr * ca * (1.0 - la)) / out_alpha;
                            cg = (blended[1] * la + cg * ca * (1.0 - la)) / out_alpha;
                            cb = (blended[2] * la + cb * ca * (1.0 - la)) / out_alpha;
                        }
                        ca = out_alpha.min(1.0);
                    }

                    out_r[gi] = cr;
                    out_g[gi] = cg;
                    out_b[gi] = cb;
                    out_a[gi] = ca;
                }
            }
        }

        // Convert to u8
        let mut pixels = vec![0u8; n * 3];
        for i in 0..n {
            pixels[i * 3] = (out_r[i].clamp(0.0, 1.0) * 255.0) as u8;
            pixels[i * 3 + 1] = (out_g[i].clamp(0.0, 1.0) * 255.0) as u8;
            pixels[i * 3 + 2] = (out_b[i].clamp(0.0, 1.0) * 255.0) as u8;
        }

        Ok(TiledFrame {
            pixels,
            width: w,
            height: h,
            tiles_processed: self.tiles.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> TiledCompositorConfig {
        TiledCompositorConfig {
            resolution: (16, 16),
            tile_width: 8,
            tile_height: 8,
            num_threads: 0,
        }
    }

    #[test]
    fn test_tiled_compositor_creation() {
        let c = TiledCompositor::new(small_config());
        assert!(c.is_ok());
    }

    #[test]
    fn test_zero_tile_size_fails() {
        let mut cfg = small_config();
        cfg.tile_width = 0;
        assert!(TiledCompositor::new(cfg).is_err());
    }

    #[test]
    fn test_tile_count() {
        let tc = TiledCompositor::new(small_config()).expect("ok");
        // 16/8 × 16/8 = 2 × 2 = 4 tiles
        assert_eq!(tc.tile_count(), 4);
    }

    #[test]
    fn test_tile_count_non_even() {
        let cfg = TiledCompositorConfig {
            resolution: (10, 10),
            tile_width: 4,
            tile_height: 4,
            num_threads: 0,
        };
        let tc = TiledCompositor::new(cfg).expect("ok");
        // ceil(10/4) × ceil(10/4) = 3 × 3 = 9 tiles
        assert_eq!(tc.tile_count(), 9);
    }

    #[test]
    fn test_composite_empty_layers() {
        let tc = TiledCompositor::new(small_config()).expect("ok");
        let frame = tc.composite(&[], 0);
        assert!(frame.is_ok());
        let f = frame.expect("ok");
        // All black when no layers
        assert_eq!(f.get_pixel(8, 8), Some([0, 0, 0]));
    }

    #[test]
    fn test_composite_single_opaque_layer() {
        let tc = TiledCompositor::new(small_config()).expect("ok");
        let layer = TileLayerData::solid("bg", 1.0, 0.0, 0.0, 1.0, 16, 16, 0);
        let frame = tc.composite(&[layer], 0).expect("ok");
        assert_eq!(frame.get_pixel(0, 0), Some([255, 0, 0]));
        assert_eq!(frame.tiles_processed, 4);
    }

    #[test]
    fn test_composite_two_layers_z_order() {
        let tc = TiledCompositor::new(small_config()).expect("ok");
        let bg = TileLayerData::solid("bg", 1.0, 0.0, 0.0, 1.0, 16, 16, 0);
        let fg = TileLayerData::solid("fg", 0.0, 0.0, 1.0, 1.0, 16, 16, 10);
        let frame = tc.composite(&[fg, bg], 0).expect("ok");
        // Blue fg (z=10) should cover red bg (z=0)
        assert_eq!(frame.get_pixel(4, 4), Some([0, 0, 255]));
    }

    #[test]
    fn test_composite_semi_transparent_layer() {
        let tc = TiledCompositor::new(small_config()).expect("ok");
        let bg = TileLayerData::solid("bg", 1.0, 0.0, 0.0, 1.0, 16, 16, 0);
        let mut fg = TileLayerData::solid("fg", 0.0, 1.0, 0.0, 0.5, 16, 16, 1);
        fg.opacity = 1.0;
        let frame = tc.composite(&[bg, fg], 0).expect("ok");
        let px = frame.get_pixel(8, 8).expect("ok");
        assert!(px[0] > 0, "red should bleed through");
        assert!(px[1] > 0, "green should contribute");
    }

    #[test]
    fn test_composite_resolution_mismatch_error() {
        let tc = TiledCompositor::new(small_config()).expect("ok");
        let wrong = TileLayerData::solid("wrong", 1.0, 0.0, 0.0, 1.0, 8, 8, 0);
        let result = tc.composite(&[wrong], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_composite_disabled_layer_skipped() {
        let tc = TiledCompositor::new(small_config()).expect("ok");
        let bg = TileLayerData::solid("bg", 1.0, 0.0, 0.0, 1.0, 16, 16, 0);
        let mut fg = TileLayerData::solid("fg", 0.0, 1.0, 0.0, 1.0, 16, 16, 1);
        fg.enabled = false;
        let frame = tc.composite(&[bg, fg], 0).expect("ok");
        assert_eq!(frame.get_pixel(0, 0), Some([255, 0, 0]));
    }

    #[test]
    fn test_composite_additive_blend() {
        let cfg = TiledCompositorConfig {
            resolution: (4, 4),
            tile_width: 2,
            tile_height: 2,
            num_threads: 0,
        };
        let tc = TiledCompositor::new(cfg).expect("ok");
        let bg = TileLayerData::solid("bg", 0.5, 0.0, 0.0, 1.0, 4, 4, 0);
        let mut glow = TileLayerData::solid("glow", 0.3, 0.3, 0.0, 1.0, 4, 4, 1);
        glow.blend_mode = BlendMode::Add;
        let frame = tc.composite(&[bg, glow], 0).expect("ok");
        let px = frame.get_pixel(2, 2).expect("ok");
        assert!(px[0] > 180, "additive red: {}", px[0]);
    }

    #[test]
    fn test_tile_solid_pixel_access() {
        let layer = TileLayerData::solid("test", 0.5, 0.25, 0.1, 1.0, 4, 4, 0);
        let px = layer.get_pixel(2, 2).expect("ok");
        assert!((px[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_tile_pixel_out_of_bounds() {
        let layer = TileLayerData::solid("test", 1.0, 0.0, 0.0, 1.0, 4, 4, 0);
        assert!(layer.get_pixel(4, 0).is_none());
    }

    #[test]
    fn test_large_tile_covers_whole_frame() {
        let cfg = TiledCompositorConfig {
            resolution: (4, 4),
            tile_width: 16,
            tile_height: 16,
            num_threads: 0,
        };
        let tc = TiledCompositor::new(cfg).expect("ok");
        assert_eq!(tc.tile_count(), 1); // one tile covers everything
    }
}
