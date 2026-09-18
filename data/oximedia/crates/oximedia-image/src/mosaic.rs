#![allow(dead_code)]
//! Image mosaicing, tiling, and contact sheet generation.
//!
//! This module provides utilities for composing multiple images into grid layouts,
//! contact sheets, and tiled mosaics commonly used in VFX review, editorial
//! presentation, and thumbnail generation.

use std::fmt;

/// Strategy for fitting images into mosaic cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FitMode {
    /// Scale to fill the cell, cropping overflow.
    Fill,
    /// Scale to fit entirely within the cell, leaving letterbox/pillarbox.
    Contain,
    /// Stretch to exactly match cell dimensions.
    Stretch,
    /// No scaling; center the image in the cell.
    Center,
}

/// Pixel fill for empty areas in the mosaic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillColor {
    /// Red channel (0.0..1.0).
    pub r: f32,
    /// Green channel (0.0..1.0).
    pub g: f32,
    /// Blue channel (0.0..1.0).
    pub b: f32,
    /// Alpha channel (0.0..1.0).
    pub a: f32,
}

impl FillColor {
    /// Black fill color.
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// White fill color.
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };

    /// Transparent fill color.
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    /// Creates a new fill color from RGBA values.
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Creates a gray fill color.
    #[allow(clippy::cast_precision_loss)]
    pub fn gray(level: f32) -> Self {
        Self {
            r: level,
            g: level,
            b: level,
            a: 1.0,
        }
    }
}

impl Default for FillColor {
    fn default() -> Self {
        Self::BLACK
    }
}

/// Dimensions of a rectangular region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dimensions {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Dimensions {
    /// Creates new dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Returns the aspect ratio as width / height.
    #[allow(clippy::cast_precision_loss)]
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        f64::from(self.width) / f64::from(self.height)
    }

    /// Returns the total number of pixels.
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

impl fmt::Display for Dimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// Configuration for a mosaic / contact sheet layout.
#[derive(Clone, Debug)]
pub struct MosaicConfig {
    /// Number of columns in the grid.
    pub columns: u32,
    /// Number of rows in the grid.
    pub rows: u32,
    /// Cell dimensions (each tile).
    pub cell_size: Dimensions,
    /// Gap between cells in pixels.
    pub gap: u32,
    /// Outer margin in pixels.
    pub margin: u32,
    /// How to fit images into cells.
    pub fit_mode: FitMode,
    /// Background fill color.
    pub background: FillColor,
}

impl MosaicConfig {
    /// Creates a new mosaic config with the specified grid size and cell dimensions.
    pub fn new(columns: u32, rows: u32, cell_width: u32, cell_height: u32) -> Self {
        Self {
            columns,
            rows,
            cell_size: Dimensions::new(cell_width, cell_height),
            gap: 0,
            margin: 0,
            fit_mode: FitMode::Contain,
            background: FillColor::BLACK,
        }
    }

    /// Sets the gap between cells.
    pub fn with_gap(mut self, gap: u32) -> Self {
        self.gap = gap;
        self
    }

    /// Sets the outer margin.
    pub fn with_margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Sets the fit mode.
    pub fn with_fit_mode(mut self, mode: FitMode) -> Self {
        self.fit_mode = mode;
        self
    }

    /// Sets the background fill color.
    pub fn with_background(mut self, color: FillColor) -> Self {
        self.background = color;
        self
    }

    /// Returns the total output dimensions of the mosaic.
    pub fn total_dimensions(&self) -> Dimensions {
        let w = self.margin * 2
            + self.columns * self.cell_size.width
            + self.gap * self.columns.saturating_sub(1);
        let h = self.margin * 2
            + self.rows * self.cell_size.height
            + self.gap * self.rows.saturating_sub(1);
        Dimensions::new(w, h)
    }

    /// Returns the total number of cells in the grid.
    pub fn cell_count(&self) -> u32 {
        self.columns * self.rows
    }

    /// Returns the pixel offset (x, y) of the cell at the given grid position.
    pub fn cell_origin(&self, col: u32, row: u32) -> (u32, u32) {
        let x = self.margin + col * (self.cell_size.width + self.gap);
        let y = self.margin + row * (self.cell_size.height + self.gap);
        (x, y)
    }
}

/// Computes the scale factor to fit `src` into `dst` using the given mode.
#[allow(clippy::cast_precision_loss)]
pub fn compute_scale(src: Dimensions, dst: Dimensions, mode: FitMode) -> (f64, f64) {
    let sx = f64::from(dst.width) / f64::from(src.width).max(1.0);
    let sy = f64::from(dst.height) / f64::from(src.height).max(1.0);
    match mode {
        FitMode::Fill => {
            let s = sx.max(sy);
            (s, s)
        }
        FitMode::Contain => {
            let s = sx.min(sy);
            (s, s)
        }
        FitMode::Stretch => (sx, sy),
        FitMode::Center => (1.0, 1.0),
    }
}

/// Represents a positioned tile in a mosaic.
#[derive(Clone, Debug)]
pub struct MosaicTile {
    /// Zero-based index of this tile.
    pub index: usize,
    /// Column position in the grid.
    pub column: u32,
    /// Row position in the grid.
    pub row: u32,
    /// Pixel X offset of the tile origin.
    pub origin_x: u32,
    /// Pixel Y offset of the tile origin.
    pub origin_y: u32,
    /// Tile cell dimensions.
    pub cell_size: Dimensions,
    /// Optional label for this tile.
    pub label: Option<String>,
}

/// Generates the tile layout for a mosaic configuration.
pub fn generate_layout(config: &MosaicConfig) -> Vec<MosaicTile> {
    let mut tiles = Vec::with_capacity((config.columns * config.rows) as usize);
    let mut index = 0;
    for row in 0..config.rows {
        for col in 0..config.columns {
            let (ox, oy) = config.cell_origin(col, row);
            tiles.push(MosaicTile {
                index,
                column: col,
                row,
                origin_x: ox,
                origin_y: oy,
                cell_size: config.cell_size,
                label: None,
            });
            index += 1;
        }
    }
    tiles
}

/// Contact sheet configuration builder for common review workflows.
#[derive(Clone, Debug)]
pub struct ContactSheetBuilder {
    /// Number of thumbnails per row.
    pub thumbnails_per_row: u32,
    /// Thumbnail dimensions.
    pub thumb_size: Dimensions,
    /// Total number of frames to include.
    pub frame_count: u32,
    /// Gap between thumbnails.
    pub gap: u32,
    /// Margin around the sheet.
    pub margin: u32,
    /// Whether to add frame number labels.
    pub show_labels: bool,
}

impl ContactSheetBuilder {
    /// Creates a new contact sheet builder.
    pub fn new(frame_count: u32, per_row: u32) -> Self {
        Self {
            thumbnails_per_row: per_row,
            thumb_size: Dimensions::new(160, 90),
            frame_count,
            gap: 4,
            margin: 8,
            show_labels: true,
        }
    }

    /// Sets the thumbnail size.
    pub fn with_thumb_size(mut self, width: u32, height: u32) -> Self {
        self.thumb_size = Dimensions::new(width, height);
        self
    }

    /// Sets the gap.
    pub fn with_gap(mut self, gap: u32) -> Self {
        self.gap = gap;
        self
    }

    /// Builds a `MosaicConfig` from the contact sheet settings.
    pub fn build(&self) -> MosaicConfig {
        let rows = self.frame_count.div_ceil(self.thumbnails_per_row);
        MosaicConfig {
            columns: self.thumbnails_per_row,
            rows,
            cell_size: self.thumb_size,
            gap: self.gap,
            margin: self.margin,
            fit_mode: FitMode::Contain,
            background: FillColor::BLACK,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensions_new() {
        let d = Dimensions::new(1920, 1080);
        assert_eq!(d.width, 1920);
        assert_eq!(d.height, 1080);
    }

    #[test]
    fn test_dimensions_aspect_ratio() {
        let d = Dimensions::new(1920, 1080);
        let ar = d.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_dimensions_zero_height() {
        let d = Dimensions::new(100, 0);
        assert_eq!(d.aspect_ratio(), 0.0);
    }

    #[test]
    fn test_dimensions_pixel_count() {
        let d = Dimensions::new(1920, 1080);
        assert_eq!(d.pixel_count(), 2_073_600);
    }

    #[test]
    fn test_dimensions_display() {
        let d = Dimensions::new(640, 480);
        assert_eq!(format!("{d}"), "640x480");
    }

    #[test]
    fn test_mosaic_config_basic() {
        let config = MosaicConfig::new(4, 3, 100, 100);
        assert_eq!(config.cell_count(), 12);
    }

    #[test]
    fn test_mosaic_total_dimensions_no_gap() {
        let config = MosaicConfig::new(4, 3, 100, 80);
        let dims = config.total_dimensions();
        assert_eq!(dims.width, 400);
        assert_eq!(dims.height, 240);
    }

    #[test]
    fn test_mosaic_total_dimensions_with_gap_and_margin() {
        let config = MosaicConfig::new(2, 2, 100, 100)
            .with_gap(10)
            .with_margin(5);
        let dims = config.total_dimensions();
        // 5 + 100 + 10 + 100 + 5 = 220
        assert_eq!(dims.width, 220);
        assert_eq!(dims.height, 220);
    }

    #[test]
    fn test_mosaic_cell_origin() {
        let config = MosaicConfig::new(3, 2, 100, 80).with_gap(10).with_margin(5);
        let (x0, y0) = config.cell_origin(0, 0);
        assert_eq!(x0, 5);
        assert_eq!(y0, 5);

        let (x1, y1) = config.cell_origin(1, 1);
        assert_eq!(x1, 5 + 110);
        assert_eq!(y1, 5 + 90);
    }

    #[test]
    fn test_generate_layout_count() {
        let config = MosaicConfig::new(4, 3, 100, 100);
        let layout = generate_layout(&config);
        assert_eq!(layout.len(), 12);
    }

    #[test]
    fn test_generate_layout_indices() {
        let config = MosaicConfig::new(2, 2, 50, 50);
        let layout = generate_layout(&config);
        for (i, tile) in layout.iter().enumerate() {
            assert_eq!(tile.index, i);
        }
    }

    #[test]
    fn test_compute_scale_contain() {
        let src = Dimensions::new(200, 100);
        let dst = Dimensions::new(100, 100);
        let (sx, sy) = compute_scale(src, dst, FitMode::Contain);
        assert!((sx - 0.5).abs() < 1e-10);
        assert!((sy - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_scale_fill() {
        let src = Dimensions::new(200, 100);
        let dst = Dimensions::new(100, 100);
        let (sx, sy) = compute_scale(src, dst, FitMode::Fill);
        assert!((sx - 1.0).abs() < 1e-10);
        assert!((sy - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_scale_stretch() {
        let src = Dimensions::new(200, 100);
        let dst = Dimensions::new(100, 200);
        let (sx, sy) = compute_scale(src, dst, FitMode::Stretch);
        assert!((sx - 0.5).abs() < 1e-10);
        assert!((sy - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_fill_color_presets() {
        assert_eq!(FillColor::BLACK.r, 0.0);
        assert_eq!(FillColor::WHITE.r, 1.0);
        assert_eq!(FillColor::TRANSPARENT.a, 0.0);
    }

    #[test]
    fn test_contact_sheet_builder() {
        let builder = ContactSheetBuilder::new(24, 6)
            .with_thumb_size(160, 90)
            .with_gap(4);
        let config = builder.build();
        assert_eq!(config.columns, 6);
        assert_eq!(config.rows, 4); // 24 / 6
        assert_eq!(config.cell_size.width, 160);
    }

    #[test]
    fn test_contact_sheet_rounding() {
        // 10 frames, 3 per row => 4 rows (ceil)
        let builder = ContactSheetBuilder::new(10, 3);
        let config = builder.build();
        assert_eq!(config.rows, 4);
    }
}
