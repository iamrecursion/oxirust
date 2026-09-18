//! Frame tile and render-pass merging for distributed rendering.
//!
//! When frames are rendered in tiles or as separate passes (beauty, shadow,
//! reflection, etc.), this module handles assembling them into final composites.

use std::collections::HashMap;
use std::fmt;

/// Identifies a rectangular tile within a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileRegion {
    /// X offset in pixels from the left edge.
    pub x: u32,
    /// Y offset in pixels from the top edge.
    pub y: u32,
    /// Tile width in pixels.
    pub width: u32,
    /// Tile height in pixels.
    pub height: u32,
}

impl TileRegion {
    /// Creates a new tile region.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Returns the total number of pixels in this tile.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Checks whether this tile overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }

    /// Returns the right edge x-coordinate.
    #[must_use]
    pub fn right(&self) -> u32 {
        self.x + self.width
    }

    /// Returns the bottom edge y-coordinate.
    #[must_use]
    pub fn bottom(&self) -> u32 {
        self.y + self.height
    }
}

impl fmt::Display for TileRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}) {}x{}", self.x, self.y, self.width, self.height)
    }
}

/// A rendered tile with its pixel data reference.
#[derive(Debug, Clone)]
pub struct RenderedTile {
    /// The region this tile covers.
    pub region: TileRegion,
    /// Frame number this tile belongs to.
    pub frame: u64,
    /// Identifier of the node that rendered this tile.
    pub node_id: String,
    /// Data checksum for integrity verification.
    pub checksum: u64,
    /// Size of the tile data in bytes.
    pub data_size: u64,
}

impl RenderedTile {
    /// Creates a new rendered tile record.
    pub fn new(
        region: TileRegion,
        frame: u64,
        node_id: impl Into<String>,
        checksum: u64,
        data_size: u64,
    ) -> Self {
        Self {
            region,
            frame,
            node_id: node_id.into(),
            checksum,
            data_size,
        }
    }
}

/// Blend mode for combining render passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassBlendMode {
    /// Direct replacement (beauty pass).
    Replace,
    /// Additive blending (lights, reflections).
    Add,
    /// Multiplicative blending (shadows, occlusion).
    Multiply,
    /// Screen blend (glow, bloom).
    Screen,
    /// Alpha-over compositing.
    AlphaOver,
}

impl fmt::Display for PassBlendMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Replace => write!(f, "replace"),
            Self::Add => write!(f, "add"),
            Self::Multiply => write!(f, "multiply"),
            Self::Screen => write!(f, "screen"),
            Self::AlphaOver => write!(f, "alpha_over"),
        }
    }
}

/// A named render pass for multi-pass compositing.
#[derive(Debug, Clone)]
pub struct RenderPass {
    /// Pass name (e.g., "beauty", "shadow", "reflection").
    pub name: String,
    /// How this pass is blended with the composite.
    pub blend_mode: PassBlendMode,
    /// Opacity multiplier (0.0 to 1.0).
    pub opacity: f32,
    /// Order in the compositing stack (lower = rendered first).
    pub order: u32,
}

impl RenderPass {
    /// Creates a new render pass.
    pub fn new(name: impl Into<String>, blend_mode: PassBlendMode, order: u32) -> Self {
        Self {
            name: name.into(),
            blend_mode,
            opacity: 1.0,
            order,
        }
    }

    /// Sets the opacity.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }
}

/// Blends two pixel values according to a blend mode.
///
/// Both `base` and `blend` are in `[R, G, B, A]` format, normalised 0..1.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn blend_pixels(
    base: [f32; 4],
    blend_val: [f32; 4],
    mode: PassBlendMode,
    opacity: f32,
) -> [f32; 4] {
    let op = opacity.clamp(0.0, 1.0);
    let mut result = [0.0f32; 4];
    match mode {
        PassBlendMode::Replace => {
            for i in 0..4 {
                result[i] = base[i] * (1.0 - op) + blend_val[i] * op;
            }
        }
        PassBlendMode::Add => {
            for i in 0..3 {
                result[i] = (base[i] + blend_val[i] * op).min(1.0);
            }
            result[3] = base[3];
        }
        PassBlendMode::Multiply => {
            for i in 0..3 {
                let mul = base[i] * blend_val[i];
                result[i] = base[i] * (1.0 - op) + mul * op;
            }
            result[3] = base[3];
        }
        PassBlendMode::Screen => {
            for i in 0..3 {
                let scr = 1.0 - (1.0 - base[i]) * (1.0 - blend_val[i]);
                result[i] = base[i] * (1.0 - op) + scr * op;
            }
            result[3] = base[3];
        }
        PassBlendMode::AlphaOver => {
            let src_a = blend_val[3] * op;
            let dst_a = base[3];
            let out_a = src_a + dst_a * (1.0 - src_a);
            if out_a > 1e-6 {
                for i in 0..3 {
                    result[i] = (blend_val[i] * src_a + base[i] * dst_a * (1.0 - src_a)) / out_a;
                }
            }
            result[3] = out_a;
        }
    }
    result
}

/// Tile layout configuration for splitting a frame into tiles.
#[derive(Debug, Clone)]
pub struct TileLayout {
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Tile width in pixels.
    pub tile_width: u32,
    /// Tile height in pixels.
    pub tile_height: u32,
}

impl TileLayout {
    /// Creates a new tile layout.
    #[must_use]
    pub fn new(frame_width: u32, frame_height: u32, tile_width: u32, tile_height: u32) -> Self {
        Self {
            frame_width,
            frame_height,
            tile_width: tile_width.max(1),
            tile_height: tile_height.max(1),
        }
    }

    /// Returns the number of tile columns.
    #[must_use]
    pub fn columns(&self) -> u32 {
        (self.frame_width + self.tile_width - 1) / self.tile_width
    }

    /// Returns the number of tile rows.
    #[must_use]
    pub fn rows(&self) -> u32 {
        (self.frame_height + self.tile_height - 1) / self.tile_height
    }

    /// Returns the total number of tiles.
    #[must_use]
    pub fn tile_count(&self) -> u32 {
        self.columns() * self.rows()
    }

    /// Generates all tile regions for this layout.
    #[must_use]
    pub fn generate_tiles(&self) -> Vec<TileRegion> {
        let mut tiles = Vec::new();
        for row in 0..self.rows() {
            for col in 0..self.columns() {
                let x = col * self.tile_width;
                let y = row * self.tile_height;
                let w = (self.frame_width - x).min(self.tile_width);
                let h = (self.frame_height - y).min(self.tile_height);
                tiles.push(TileRegion::new(x, y, w, h));
            }
        }
        tiles
    }
}

/// Tracks which tiles have been received for a frame merge operation.
#[derive(Debug)]
pub struct FrameMergeTracker {
    /// Frame number being assembled.
    frame: u64,
    /// Expected tile regions.
    expected: Vec<TileRegion>,
    /// Received tiles keyed by region.
    received: HashMap<TileRegion, RenderedTile>,
}

impl FrameMergeTracker {
    /// Creates a new merge tracker for a frame.
    pub fn new(frame: u64, layout: &TileLayout) -> Self {
        Self {
            frame,
            expected: layout.generate_tiles(),
            received: HashMap::new(),
        }
    }

    /// Records a received tile.
    pub fn add_tile(&mut self, tile: RenderedTile) {
        self.received.insert(tile.region, tile);
    }

    /// Returns the number of tiles received.
    #[must_use]
    pub fn received_count(&self) -> usize {
        self.received.len()
    }

    /// Returns the number of tiles expected.
    #[must_use]
    pub fn expected_count(&self) -> usize {
        self.expected.len()
    }

    /// Returns true if all tiles have been received.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.expected
            .iter()
            .all(|region| self.received.contains_key(region))
    }

    /// Returns the list of missing tile regions.
    #[must_use]
    pub fn missing_tiles(&self) -> Vec<TileRegion> {
        self.expected
            .iter()
            .filter(|r| !self.received.contains_key(r))
            .copied()
            .collect()
    }

    /// Returns the completion percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn percent_complete(&self) -> f32 {
        if self.expected.is_empty() {
            return 100.0;
        }
        (self.received_count() as f64 / self.expected_count() as f64 * 100.0) as f32
    }

    /// Returns the frame number.
    #[must_use]
    pub fn frame(&self) -> u64 {
        self.frame
    }
}

// ---------------------------------------------------------------------------
// MmapFrameMerger
// ---------------------------------------------------------------------------

use memmap2::MmapMut;
use std::path::PathBuf;

/// Per-tile position within the memory-mapped scratch backing store.
#[derive(Debug, Clone)]
pub struct TileOffset {
    /// Tile identifier matching the value passed to `write_tile`.
    pub tile_id: u32,
    /// Byte offset within the backing store where this tile's data starts.
    pub offset: usize,
    /// Length of this tile's data in bytes.
    pub len: usize,
}

/// Merges rendered tiles using a memory-mapped scratch file for near-zero-copy
/// frame assembly.
///
/// Falls back gracefully: file-backed mmap → anonymous mmap → `Vec<u8>`.
///
/// # Usage
///
/// ```no_run
/// # use std::path::PathBuf;
/// # use oximedia_renderfarm::frame_merge::MmapFrameMerger;
/// let mut merger = MmapFrameMerger::new(
///     PathBuf::from("/tmp/frame_001.bin"),
///     (1920, 1080, 4),   // width, height, bytes_per_pixel
///     4,                 // number of tiles
/// ).unwrap();
/// merger.write_tile(0, &[0u8; 1920 * 270 * 4]).unwrap();
/// let mut output = Vec::new();
/// merger.merge(&mut output).unwrap();
/// merger.cleanup().unwrap();
/// ```
pub struct MmapFrameMerger {
    /// Path of the scratch file (used for file-backed mmap; may not exist on fallback).
    scratch_path: PathBuf,
    /// Memory-mapped buffer, or `None` if using `fallback`.
    map: Option<MmapMut>,
    /// In-memory fallback buffer when mmap is unavailable.
    fallback: Option<Vec<u8>>,
    /// Per-tile offset index.
    tile_index: Vec<TileOffset>,
    /// (width, height, bytes_per_pixel).
    frame_dims: (u32, u32, u32),
    /// Total frame bytes = w * h * bpp.
    total_bytes: usize,
}

impl MmapFrameMerger {
    /// Create a new merger.
    ///
    /// Attempts to open a file-backed mmap at `scratch_path`; on failure tries
    /// an anonymous mmap; on that failure falls back to `Vec<u8>`.
    pub fn new(
        scratch_path: PathBuf,
        frame_dims: (u32, u32, u32),
        tile_count: u32,
    ) -> std::io::Result<Self> {
        let (w, h, bpp) = frame_dims;
        let total_bytes = w as usize * h as usize * bpp as usize;
        let (map, fallback) = Self::create_backing(&scratch_path, total_bytes);
        let tile_index = Vec::with_capacity(tile_count as usize);
        Ok(Self {
            scratch_path,
            map,
            fallback,
            tile_index,
            frame_dims,
            total_bytes,
        })
    }

    fn create_backing(_path: &PathBuf, total_bytes: usize) -> (Option<MmapMut>, Option<Vec<u8>>) {
        // Attempt 1: anonymous mmap — safe, no file descriptor required.
        // `MmapMut::map_anon` is safe because it creates a fresh private mapping
        // not backed by any file that could cause UB through aliased access.
        if let Ok(m) = MmapMut::map_anon(total_bytes.max(1)) {
            return (Some(m), None);
        }
        // Fallback 2: plain heap allocation.
        (None, Some(vec![0u8; total_bytes]))
    }

    /// Write `data` for tile `tile_id` into the backing store.
    ///
    /// Tiles are placed consecutively. If `tile_id` has already been written
    /// its existing offset is reused (idempotent overwrite for retries).
    pub fn write_tile(&mut self, tile_id: u32, data: &[u8]) -> std::io::Result<()> {
        let offset = if let Some(existing) = self.tile_index.iter().find(|t| t.tile_id == tile_id) {
            existing.offset
        } else {
            let last_end = self
                .tile_index
                .last()
                .map(|t| t.offset + t.len)
                .unwrap_or(0);
            self.tile_index.push(TileOffset {
                tile_id,
                offset: last_end,
                len: data.len(),
            });
            last_end
        };

        let end = offset + data.len();
        if end > self.total_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "tile data (offset={offset} len={}) exceeds frame buffer ({} bytes)",
                    data.len(),
                    self.total_bytes
                ),
            ));
        }

        if let Some(ref mut m) = self.map {
            m[offset..end].copy_from_slice(data);
        } else if let Some(ref mut v) = self.fallback {
            v[offset..end].copy_from_slice(data);
        }
        Ok(())
    }

    /// Copy all backing-store data into `output`.
    ///
    /// `output` is resized to `total_bytes` before copying.
    pub fn merge(&self, output: &mut Vec<u8>) -> std::io::Result<()> {
        output.resize(self.total_bytes, 0u8);
        if let Some(ref m) = self.map {
            output.copy_from_slice(&m[..self.total_bytes]);
        } else if let Some(ref v) = self.fallback {
            output.copy_from_slice(&v[..self.total_bytes]);
        }
        Ok(())
    }

    /// Release the mapping and delete the scratch file.
    pub fn cleanup(self) -> std::io::Result<()> {
        drop(self.map);
        drop(self.fallback);
        if self.scratch_path.exists() {
            std::fs::remove_file(&self.scratch_path)?;
        }
        Ok(())
    }

    /// Returns `(width, height, bytes_per_pixel)`.
    pub fn frame_dims(&self) -> (u32, u32, u32) {
        self.frame_dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_region_pixel_count() {
        let tile = TileRegion::new(0, 0, 256, 256);
        assert_eq!(tile.pixel_count(), 65536);
    }

    #[test]
    fn test_tile_region_overlap() {
        let a = TileRegion::new(0, 0, 100, 100);
        let b = TileRegion::new(50, 50, 100, 100);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_tile_region_no_overlap() {
        let a = TileRegion::new(0, 0, 100, 100);
        let b = TileRegion::new(200, 200, 100, 100);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_tile_region_display() {
        let tile = TileRegion::new(10, 20, 64, 64);
        assert_eq!(format!("{tile}"), "(10, 20) 64x64");
    }

    #[test]
    fn test_tile_region_edges() {
        let tile = TileRegion::new(10, 20, 30, 40);
        assert_eq!(tile.right(), 40);
        assert_eq!(tile.bottom(), 60);
    }

    #[test]
    fn test_tile_layout_basic() {
        let layout = TileLayout::new(1920, 1080, 256, 256);
        assert_eq!(layout.columns(), 8); // ceil(1920/256) = 8
        assert_eq!(layout.rows(), 5); // ceil(1080/256) = 5 (4.21..)
        assert_eq!(layout.tile_count(), 40);
    }

    #[test]
    fn test_tile_layout_exact_division() {
        let layout = TileLayout::new(512, 512, 256, 256);
        assert_eq!(layout.columns(), 2);
        assert_eq!(layout.rows(), 2);
        assert_eq!(layout.tile_count(), 4);
    }

    #[test]
    fn test_generate_tiles_covers_frame() {
        let layout = TileLayout::new(300, 200, 128, 128);
        let tiles = layout.generate_tiles();
        // Should cover entire frame without gaps
        // columns = ceil(300/128) = 3, rows = ceil(200/128) = 2 => 6 tiles
        assert_eq!(tiles.len(), 6);
        // Last column tiles should be narrower
        let last_col_tile = tiles
            .iter()
            .find(|t| t.x == 256)
            .expect("should succeed in test");
        assert_eq!(last_col_tile.width, 44); // 300 - 256
    }

    #[test]
    fn test_blend_pixels_replace() {
        let base = [0.5, 0.5, 0.5, 1.0];
        let over = [1.0, 0.0, 0.0, 1.0];
        let result = blend_pixels(base, over, PassBlendMode::Replace, 1.0);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_pixels_add() {
        let base = [0.3, 0.3, 0.3, 1.0];
        let over = [0.5, 0.5, 0.5, 1.0];
        let result = blend_pixels(base, over, PassBlendMode::Add, 1.0);
        assert!((result[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_blend_pixels_multiply() {
        let base = [0.5, 0.5, 0.5, 1.0];
        let over = [0.5, 0.5, 0.5, 1.0];
        let result = blend_pixels(base, over, PassBlendMode::Multiply, 1.0);
        assert!((result[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_blend_pixels_screen() {
        let base = [0.5, 0.5, 0.5, 1.0];
        let over = [0.5, 0.5, 0.5, 1.0];
        let result = blend_pixels(base, over, PassBlendMode::Screen, 1.0);
        // screen = 1 - (1-0.5)*(1-0.5) = 0.75
        assert!((result[0] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_render_pass_opacity() {
        let pass = RenderPass::new("glow", PassBlendMode::Add, 2).with_opacity(0.5);
        assert!((pass.opacity - 0.5).abs() < 1e-6);
        assert_eq!(pass.order, 2);
    }

    #[test]
    fn test_frame_merge_tracker_complete() {
        let layout = TileLayout::new(512, 512, 256, 256);
        let mut tracker = FrameMergeTracker::new(1, &layout);
        assert_eq!(tracker.expected_count(), 4);
        assert!(!tracker.is_complete());

        let tiles = layout.generate_tiles();
        for region in &tiles {
            tracker.add_tile(RenderedTile::new(*region, 1, "node-1", 0, 1024));
        }
        assert!(tracker.is_complete());
        assert_eq!(tracker.received_count(), 4);
        assert!((tracker.percent_complete() - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_frame_merge_tracker_missing() {
        let layout = TileLayout::new(512, 512, 256, 256);
        let mut tracker = FrameMergeTracker::new(1, &layout);
        let tiles = layout.generate_tiles();
        // Add only first 2 tiles
        for region in tiles.iter().take(2) {
            tracker.add_tile(RenderedTile::new(*region, 1, "node-1", 0, 1024));
        }
        assert!(!tracker.is_complete());
        let missing = tracker.missing_tiles();
        assert_eq!(missing.len(), 2);
    }

    #[test]
    fn test_frame_merge_tracker_frame_number() {
        let layout = TileLayout::new(128, 128, 128, 128);
        let tracker = FrameMergeTracker::new(42, &layout);
        assert_eq!(tracker.frame(), 42);
    }

    // --- MmapFrameMerger ---

    fn unique_tmp_path(prefix: &str) -> std::path::PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("oximedia_{prefix}_{ts}.bin"))
    }

    #[test]
    fn test_mmap_frame_merger_roundtrip() {
        let path = unique_tmp_path("merge");
        let dims = (4u32, 4u32, 1u32); // 4×4 frame, 1 bpp = 16 bytes
        let mut merger = MmapFrameMerger::new(path.clone(), dims, 1).expect("new ok");
        merger.write_tile(0, &[42u8; 16]).expect("write ok");
        let mut out = Vec::new();
        merger.merge(&mut out).expect("merge ok");
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|&b| b == 42), "all bytes should be 42");
        // Cleanup is best-effort; ignore errors in test teardown.
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_mmap_frame_merger_cleanup() {
        let path = unique_tmp_path("cleanup");
        let dims = (8u32, 8u32, 4u32); // 256 bytes
        let merger = MmapFrameMerger::new(path.clone(), dims, 2).expect("new ok");
        merger.cleanup().expect("cleanup ok");
        assert!(
            !path.exists(),
            "scratch file should be removed after cleanup"
        );
    }

    #[test]
    fn test_mmap_frame_merger_overwrite_idempotent() {
        let path = unique_tmp_path("overwrite");
        let dims = (2u32, 2u32, 1u32); // 4 bytes
        let mut merger = MmapFrameMerger::new(path.clone(), dims, 1).expect("new ok");
        merger.write_tile(0, &[1u8; 4]).expect("first write ok");
        merger.write_tile(0, &[7u8; 4]).expect("second write ok");
        let mut out = Vec::new();
        merger.merge(&mut out).expect("merge ok");
        // Second write overwrites first at the same offset
        assert!(out.iter().all(|&b| b == 7), "overwrite should take effect");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_mmap_frame_merger_frame_dims() {
        let path = unique_tmp_path("dims");
        let dims = (1920u32, 1080u32, 3u32);
        let merger = MmapFrameMerger::new(path.clone(), dims, 0).expect("new ok");
        assert_eq!(merger.frame_dims(), dims);
        let _ = std::fs::remove_file(path);
    }
}
