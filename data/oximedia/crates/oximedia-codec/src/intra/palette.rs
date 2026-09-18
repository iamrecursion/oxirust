//! Palette mode implementation (AV1).
//!
//! Palette mode uses a small set of colors (2-8) to represent the block,
//! with each pixel being an index into the palette. This is particularly
//! effective for blocks with limited color variation, such as graphics
//! or screen content.
//!
//! # Structure
//!
//! - **PaletteInfo**: Contains palette colors and size
//! - **ColorCache**: Previously used colors for prediction
//! - **ColorIndexMap**: Per-pixel palette indices
//!
//! # Encoding
//!
//! 1. Palette colors are signaled in the bitstream
//! 2. Color indices are entropy coded using context
//! 3. The decoder reconstructs pixels by looking up palette colors

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]

use super::{BitDepth, BlockDimensions, IntraPredContext, IntraPredictor};

/// Maximum palette size (AV1 supports 2-8 colors).
pub const MAX_PALETTE_SIZE: usize = 8;

/// Minimum palette size.
pub const MIN_PALETTE_SIZE: usize = 2;

/// Maximum color cache size.
pub const MAX_COLOR_CACHE_SIZE: usize = 64;

/// Palette information for a block.
#[derive(Clone, Debug)]
pub struct PaletteInfo {
    /// Palette colors (Y/U/V or R/G/B values).
    colors: [u16; MAX_PALETTE_SIZE],
    /// Number of colors in the palette (2-8).
    size: usize,
    /// Bit depth of color values.
    bit_depth: BitDepth,
}

impl PaletteInfo {
    /// Create a new empty palette.
    #[must_use]
    pub const fn new(bit_depth: BitDepth) -> Self {
        Self {
            colors: [0; MAX_PALETTE_SIZE],
            size: 0,
            bit_depth,
        }
    }

    /// Create a palette with specified colors.
    #[must_use]
    pub fn with_colors(colors: &[u16], bit_depth: BitDepth) -> Self {
        let size = colors.len().min(MAX_PALETTE_SIZE);
        let mut palette_colors = [0u16; MAX_PALETTE_SIZE];
        palette_colors[..size].copy_from_slice(&colors[..size]);

        Self {
            colors: palette_colors,
            size,
            bit_depth,
        }
    }

    /// Get the palette size.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Set the palette size.
    pub fn set_size(&mut self, size: usize) {
        self.size = size.clamp(MIN_PALETTE_SIZE, MAX_PALETTE_SIZE);
    }

    /// Get a color at the specified index.
    #[must_use]
    pub fn get_color(&self, idx: usize) -> u16 {
        if idx < self.size {
            self.colors[idx]
        } else {
            0
        }
    }

    /// Set a color at the specified index.
    pub fn set_color(&mut self, idx: usize, color: u16) {
        if idx < MAX_PALETTE_SIZE {
            self.colors[idx] = color.min(self.bit_depth.max_value());
            if idx >= self.size {
                self.size = idx + 1;
            }
        }
    }

    /// Get all colors as a slice.
    #[must_use]
    pub fn colors(&self) -> &[u16] {
        &self.colors[..self.size]
    }

    /// Check if the palette is valid.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.size >= MIN_PALETTE_SIZE && self.size <= MAX_PALETTE_SIZE
    }

    /// Sort colors in ascending order.
    pub fn sort_colors(&mut self) {
        self.colors[..self.size].sort_unstable();
    }

    /// Find the nearest color index for a given value.
    #[must_use]
    pub fn find_nearest(&self, value: u16) -> usize {
        let mut best_idx = 0;
        let mut best_diff = u32::MAX;

        for (idx, &color) in self.colors[..self.size].iter().enumerate() {
            let diff = (i32::from(value) - i32::from(color)).unsigned_abs();
            if diff < best_diff {
                best_diff = diff;
                best_idx = idx;
            }
        }

        best_idx
    }
}

impl Default for PaletteInfo {
    fn default() -> Self {
        Self::new(BitDepth::Bits8)
    }
}

/// Color cache for palette mode.
///
/// Stores recently used colors to improve entropy coding efficiency.
#[derive(Clone, Debug)]
pub struct ColorCache {
    /// Cached colors.
    colors: Vec<u16>,
    /// Maximum cache size.
    max_size: usize,
    /// Bit depth.
    bit_depth: BitDepth,
}

impl ColorCache {
    /// Create a new color cache.
    #[must_use]
    pub fn new(max_size: usize, bit_depth: BitDepth) -> Self {
        Self {
            colors: Vec::with_capacity(max_size),
            max_size,
            bit_depth,
        }
    }

    /// Add a color to the cache.
    pub fn add(&mut self, color: u16) {
        // Don't add duplicates
        if self.colors.contains(&color) {
            return;
        }

        if self.colors.len() >= self.max_size {
            // Remove oldest color
            self.colors.remove(0);
        }

        self.colors.push(color);
    }

    /// Check if a color is in the cache.
    #[must_use]
    pub fn contains(&self, color: u16) -> bool {
        self.colors.contains(&color)
    }

    /// Find the index of a color in the cache.
    #[must_use]
    pub fn find(&self, color: u16) -> Option<usize> {
        self.colors.iter().position(|&c| c == color)
    }

    /// Get all cached colors.
    #[must_use]
    pub fn colors(&self) -> &[u16] {
        &self.colors
    }

    /// Get the cache size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// Check if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.colors.clear();
    }

    /// Build cache from neighbor samples.
    pub fn build_from_neighbors(&mut self, top: &[u16], left: &[u16]) {
        self.clear();

        // Add unique colors from top
        for &color in top {
            self.add(color);
        }

        // Add unique colors from left
        for &color in left {
            self.add(color);
        }
    }
}

impl Default for ColorCache {
    fn default() -> Self {
        Self::new(MAX_COLOR_CACHE_SIZE, BitDepth::Bits8)
    }
}

/// Color index map for a block.
#[derive(Clone, Debug)]
pub struct ColorIndexMap {
    /// Per-pixel palette indices.
    indices: Vec<u8>,
    /// Block width.
    width: usize,
    /// Block height.
    height: usize,
}

impl ColorIndexMap {
    /// Create a new color index map.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            indices: vec![0; width * height],
            width,
            height,
        }
    }

    /// Get the index at a position.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        if x < self.width && y < self.height {
            self.indices[y * self.width + x]
        } else {
            0
        }
    }

    /// Set the index at a position.
    pub fn set(&mut self, x: usize, y: usize, idx: u8) {
        if x < self.width && y < self.height {
            self.indices[y * self.width + x] = idx;
        }
    }

    /// Get all indices as a slice.
    #[must_use]
    pub fn indices(&self) -> &[u8] {
        &self.indices
    }

    /// Get indices as mutable slice.
    pub fn indices_mut(&mut self) -> &mut [u8] {
        &mut self.indices
    }
}

/// Palette predictor.
#[derive(Clone, Debug)]
pub struct PalettePredictor {
    /// Palette information.
    palette: PaletteInfo,
    /// Color index map.
    index_map: ColorIndexMap,
}

impl PalettePredictor {
    /// Create a new palette predictor.
    #[must_use]
    pub fn new(palette: PaletteInfo, width: usize, height: usize) -> Self {
        Self {
            palette,
            index_map: ColorIndexMap::new(width, height),
        }
    }

    /// Get the palette info.
    #[must_use]
    pub const fn palette(&self) -> &PaletteInfo {
        &self.palette
    }

    /// Get mutable palette info.
    pub fn palette_mut(&mut self) -> &mut PaletteInfo {
        &mut self.palette
    }

    /// Get the index map.
    #[must_use]
    pub const fn index_map(&self) -> &ColorIndexMap {
        &self.index_map
    }

    /// Get mutable index map.
    pub fn index_map_mut(&mut self) -> &mut ColorIndexMap {
        &mut self.index_map
    }

    /// Set a color index at a position.
    pub fn set_index(&mut self, x: usize, y: usize, idx: u8) {
        self.index_map.set(x, y, idx);
    }

    /// Reconstruct the block from palette indices.
    pub fn reconstruct(&self, output: &mut [u16], stride: usize, dims: BlockDimensions) {
        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 0..dims.width {
                let idx = self.index_map.get(x, y) as usize;
                output[row_start + x] = self.palette.get_color(idx);
            }
        }
    }
}

impl IntraPredictor for PalettePredictor {
    fn predict(
        &self,
        _ctx: &IntraPredContext,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
    ) {
        self.reconstruct(output, stride, dims);
    }
}

/// Decode color indices using run-length coding.
pub fn decode_color_indices_rle(
    data: &[u8],
    index_map: &mut ColorIndexMap,
    palette_size: usize,
) -> usize {
    let mut offset = 0;
    let mut x = 0;
    let mut y = 0;
    let width = index_map.width;
    let height = index_map.height;

    while y < height && offset < data.len() {
        let color_idx = data[offset] % (palette_size as u8);
        offset += 1;

        let mut run_length = 1;
        if offset < data.len() {
            run_length = data[offset] as usize + 1;
            offset += 1;
        }

        for _ in 0..run_length {
            if y >= height {
                break;
            }
            index_map.set(x, y, color_idx);
            x += 1;
            if x >= width {
                x = 0;
                y += 1;
            }
        }
    }

    offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palette_info_creation() {
        let palette = PaletteInfo::new(BitDepth::Bits8);
        assert_eq!(palette.size(), 0);
        assert!(!palette.is_valid());

        let palette = PaletteInfo::with_colors(&[100, 150, 200], BitDepth::Bits8);
        assert_eq!(palette.size(), 3);
        assert!(palette.is_valid());
        assert_eq!(palette.get_color(0), 100);
        assert_eq!(palette.get_color(1), 150);
        assert_eq!(palette.get_color(2), 200);
    }

    #[test]
    fn test_palette_set_color() {
        let mut palette = PaletteInfo::new(BitDepth::Bits8);
        palette.set_color(0, 100);
        palette.set_color(1, 200);

        assert_eq!(palette.size(), 2);
        assert!(palette.is_valid());
        assert_eq!(palette.get_color(0), 100);
        assert_eq!(palette.get_color(1), 200);
    }

    #[test]
    fn test_palette_find_nearest() {
        let palette = PaletteInfo::with_colors(&[0, 100, 200, 255], BitDepth::Bits8);

        assert_eq!(palette.find_nearest(0), 0);
        assert_eq!(palette.find_nearest(50), 0); // Closer to 0 than 100 (dist 50 vs 50, first wins)
        assert_eq!(palette.find_nearest(60), 1); // Closer to 100 (dist 40 vs 60)
        assert_eq!(palette.find_nearest(150), 1); // Equal distance to 100 and 200, first wins
        assert_eq!(palette.find_nearest(160), 2); // Closer to 200 (dist 40 vs 60)
        assert_eq!(palette.find_nearest(255), 3);
    }

    #[test]
    fn test_palette_sort() {
        let mut palette = PaletteInfo::with_colors(&[200, 50, 150, 100], BitDepth::Bits8);
        palette.sort_colors();

        assert_eq!(palette.colors(), &[50, 100, 150, 200]);
    }

    #[test]
    fn test_color_cache() {
        let mut cache = ColorCache::new(4, BitDepth::Bits8);

        cache.add(100);
        cache.add(150);
        cache.add(200);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains(100));
        assert!(cache.contains(150));
        assert!(!cache.contains(50));

        assert_eq!(cache.find(150), Some(1));
        assert_eq!(cache.find(50), None);
    }

    #[test]
    fn test_color_cache_overflow() {
        let mut cache = ColorCache::new(3, BitDepth::Bits8);

        cache.add(100);
        cache.add(150);
        cache.add(200);
        cache.add(250); // Should evict 100

        assert_eq!(cache.len(), 3);
        assert!(!cache.contains(100));
        assert!(cache.contains(150));
        assert!(cache.contains(250));
    }

    #[test]
    fn test_color_cache_no_duplicates() {
        let mut cache = ColorCache::new(4, BitDepth::Bits8);

        cache.add(100);
        cache.add(100);
        cache.add(100);

        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_color_index_map() {
        let mut map = ColorIndexMap::new(4, 4);

        map.set(0, 0, 1);
        map.set(1, 0, 2);
        map.set(0, 1, 3);

        assert_eq!(map.get(0, 0), 1);
        assert_eq!(map.get(1, 0), 2);
        assert_eq!(map.get(0, 1), 3);
        assert_eq!(map.get(2, 2), 0); // Default
    }

    #[test]
    fn test_palette_predictor() {
        let palette = PaletteInfo::with_colors(&[0, 128, 255], BitDepth::Bits8);
        let mut predictor = PalettePredictor::new(palette, 2, 2);

        predictor.set_index(0, 0, 0);
        predictor.set_index(1, 0, 1);
        predictor.set_index(0, 1, 2);
        predictor.set_index(1, 1, 1);

        let dims = BlockDimensions::new(2, 2);
        let mut output = vec![0u16; 4];

        predictor.reconstruct(&mut output, 2, dims);

        assert_eq!(output[0], 0);
        assert_eq!(output[1], 128);
        assert_eq!(output[2], 255);
        assert_eq!(output[3], 128);
    }

    #[test]
    fn test_decode_color_indices_rle() {
        let mut map = ColorIndexMap::new(4, 2);

        // Color 0 with run of 4, Color 1 with run of 4
        let data = [0, 3, 1, 3];

        let bytes_read = decode_color_indices_rle(&data, &mut map, 3);

        assert_eq!(bytes_read, 4);
        // First row: 0, 0, 0, 0
        assert_eq!(map.get(0, 0), 0);
        assert_eq!(map.get(3, 0), 0);
        // Second row: 1, 1, 1, 1
        assert_eq!(map.get(0, 1), 1);
        assert_eq!(map.get(3, 1), 1);
    }

    #[test]
    fn test_build_cache_from_neighbors() {
        let mut cache = ColorCache::new(16, BitDepth::Bits8);

        let top = [100, 100, 150, 200];
        let left = [100, 125, 175, 200];

        cache.build_from_neighbors(&top, &left);

        // Should have unique colors: 100, 150, 200, 125, 175
        assert!(cache.contains(100));
        assert!(cache.contains(150));
        assert!(cache.contains(200));
        assert!(cache.contains(125));
        assert!(cache.contains(175));
    }
}
