//! Thumbnail strip generation: multi-frame extraction, strip layout, and
//! contact sheet composition.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A single frame extracted for use in a strip.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExtractedFrame {
    /// Frame index (0-based) in the source
    pub frame_index: u64,
    /// Timestamp in seconds
    pub timestamp_secs: f64,
    /// Thumbnail width in pixels
    pub thumb_width: u32,
    /// Thumbnail height in pixels
    pub thumb_height: u32,
    /// RGBA pixel data (row-major, 4 bytes per pixel)
    pub pixels: Vec<u8>,
}

impl ExtractedFrame {
    /// Create a synthetic frame filled with a solid colour.
    #[must_use]
    pub fn solid(
        frame_index: u64,
        timestamp_secs: f64,
        width: u32,
        height: u32,
        colour: [u8; 4],
    ) -> Self {
        let pixel_count = (width * height) as usize;
        let mut pixels = Vec::with_capacity(pixel_count * 4);
        for _ in 0..pixel_count {
            pixels.extend_from_slice(&colour);
        }
        Self {
            frame_index,
            timestamp_secs,
            thumb_width: width,
            thumb_height: height,
            pixels,
        }
    }

    /// Number of pixels in this frame.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        (self.thumb_width * self.thumb_height) as usize
    }

    /// Byte size of the pixel buffer.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.pixels.len()
    }
}

/// Strategy for selecting frames to include in a strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSelectionStrategy {
    /// Evenly spaced across the video duration
    Uniform,
    /// Every N-th frame
    EveryNth(u64),
    /// Keyframes only (approximated by indices)
    Keyframe,
}

/// Configuration for thumbnail strip generation.
#[derive(Debug, Clone)]
pub struct StripConfig {
    /// Number of thumbnails in the strip
    pub thumb_count: u32,
    /// Width of each thumbnail
    pub thumb_width: u32,
    /// Height of each thumbnail
    pub thumb_height: u32,
    /// Number of columns in the contact sheet
    pub columns: u32,
    /// Padding between thumbnails in pixels
    pub padding: u32,
    /// Frame selection strategy
    pub strategy: FrameSelectionStrategy,
}

impl StripConfig {
    /// Create a default configuration with 12 thumbnails in a 4-column grid.
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            thumb_count: 12,
            thumb_width: 160,
            thumb_height: 90,
            columns: 4,
            padding: 4,
            strategy: FrameSelectionStrategy::Uniform,
        }
    }

    /// Number of rows in the contact sheet.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.thumb_count.div_ceil(self.columns)
    }

    /// Total sheet width in pixels.
    #[must_use]
    pub fn sheet_width(&self) -> u32 {
        self.columns * self.thumb_width + (self.columns + 1) * self.padding
    }

    /// Total sheet height in pixels.
    #[must_use]
    pub fn sheet_height(&self) -> u32 {
        self.rows() * self.thumb_height + (self.rows() + 1) * self.padding
    }
}

/// Holds the computed layout for each thumbnail in a contact sheet.
#[derive(Debug, Clone)]
pub struct ThumbnailLayout {
    /// Index of the thumbnail (0-based)
    pub index: u32,
    /// X offset on the sheet
    pub x: u32,
    /// Y offset on the sheet
    pub y: u32,
    /// Width of cell
    pub width: u32,
    /// Height of cell
    pub height: u32,
}

/// Computes layout positions for all thumbnails in a contact sheet.
#[derive(Debug)]
pub struct StripLayoutEngine {
    config: StripConfig,
}

impl StripLayoutEngine {
    /// Create from a config.
    #[must_use]
    pub fn new(config: StripConfig) -> Self {
        Self { config }
    }

    /// Generate layout for all thumbnail slots.
    #[must_use]
    pub fn layouts(&self) -> Vec<ThumbnailLayout> {
        let c = &self.config;
        (0..c.thumb_count)
            .map(|i| {
                let col = i % c.columns;
                let row = i / c.columns;
                let x = c.padding + col * (c.thumb_width + c.padding);
                let y = c.padding + row * (c.thumb_height + c.padding);
                ThumbnailLayout {
                    index: i,
                    x,
                    y,
                    width: c.thumb_width,
                    height: c.thumb_height,
                }
            })
            .collect()
    }

    /// Config reference.
    #[must_use]
    pub fn config(&self) -> &StripConfig {
        &self.config
    }
}

/// Selects frame timestamps for extraction based on video duration.
#[derive(Debug, Clone)]
pub struct FrameSelector {
    /// Total video duration in seconds
    pub duration_secs: f64,
    /// Total number of frames
    pub total_frames: u64,
}

impl FrameSelector {
    /// Create a new selector.
    #[must_use]
    pub fn new(duration_secs: f64, total_frames: u64) -> Self {
        Self {
            duration_secs,
            total_frames,
        }
    }

    /// Select N evenly-spaced timestamps.
    #[must_use]
    pub fn uniform_timestamps(&self, count: u32) -> Vec<f64> {
        if count == 0 || self.duration_secs <= 0.0 {
            return vec![];
        }
        let step = self.duration_secs / f64::from(count + 1);
        (1..=count).map(|i| step * f64::from(i)).collect()
    }

    /// Select every Nth frame index.
    #[must_use]
    pub fn every_nth_frames(&self, n: u64) -> Vec<u64> {
        if n == 0 {
            return vec![];
        }
        (0..self.total_frames).step_by(n as usize).collect()
    }

    /// Fraction of duration where a timestamp falls (0.0–1.0).
    #[must_use]
    pub fn timestamp_fraction(&self, ts: f64) -> f64 {
        if self.duration_secs <= 0.0 {
            return 0.0;
        }
        (ts / self.duration_secs).clamp(0.0, 1.0)
    }
}

/// Composites extracted frames into a contact sheet pixel buffer.
#[derive(Debug)]
pub struct ContactSheetCompositor {
    layout_engine: StripLayoutEngine,
}

impl ContactSheetCompositor {
    /// Create from a strip config.
    #[must_use]
    pub fn new(config: StripConfig) -> Self {
        Self {
            layout_engine: StripLayoutEngine::new(config),
        }
    }

    /// Sheet dimensions (width, height).
    #[must_use]
    pub fn sheet_dimensions(&self) -> (u32, u32) {
        let cfg = self.layout_engine.config();
        (cfg.sheet_width(), cfg.sheet_height())
    }

    /// Compose a minimal RGBA sheet (fills background, places frames).
    /// Returns RGBA pixel buffer.
    #[must_use]
    pub fn compose(&self, frames: &[ExtractedFrame]) -> Vec<u8> {
        let cfg = self.layout_engine.config();
        let (w, h) = (cfg.sheet_width(), cfg.sheet_height());
        let mut sheet = vec![0u8; (w * h * 4) as usize];

        let layouts = self.layout_engine.layouts();

        for (layout, frame) in layouts.iter().zip(frames.iter()) {
            // Copy frame pixels into the sheet (simplified: copies row by row)
            for row in 0..layout.height.min(frame.thumb_height) {
                let src_row_start = (row * frame.thumb_width * 4) as usize;
                let src_row_end = (src_row_start
                    + layout.width.min(frame.thumb_width) as usize * 4)
                    .min(frame.pixels.len());
                let dst_row_start = ((layout.y + row) * w * 4 + layout.x * 4) as usize;
                let copy_len = src_row_end - src_row_start;
                if dst_row_start + copy_len <= sheet.len() {
                    sheet[dst_row_start..dst_row_start + copy_len]
                        .copy_from_slice(&frame.pixels[src_row_start..src_row_end]);
                }
            }
        }

        sheet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extracted_frame_pixel_count() {
        let f = ExtractedFrame::solid(0, 0.0, 160, 90, [255, 0, 0, 255]);
        assert_eq!(f.pixel_count(), 160 * 90);
    }

    #[test]
    fn test_extracted_frame_byte_size() {
        let f = ExtractedFrame::solid(0, 0.0, 10, 10, [0, 0, 0, 255]);
        assert_eq!(f.byte_size(), 10 * 10 * 4);
    }

    #[test]
    fn test_extracted_frame_colour_filled() {
        let colour = [100u8, 150, 200, 255];
        let f = ExtractedFrame::solid(0, 0.0, 2, 2, colour);
        assert_eq!(&f.pixels[0..4], &colour);
    }

    #[test]
    fn test_strip_config_rows() {
        let cfg = StripConfig::default_config(); // 12 thumbs, 4 cols
        assert_eq!(cfg.rows(), 3);
    }

    #[test]
    fn test_strip_config_sheet_dimensions_consistent() {
        let cfg = StripConfig::default_config();
        let (w, h) = (cfg.sheet_width(), cfg.sheet_height());
        assert!(w > cfg.thumb_width);
        assert!(h > cfg.thumb_height);
    }

    #[test]
    fn test_strip_layout_engine_count() {
        let cfg = StripConfig::default_config();
        let engine = StripLayoutEngine::new(cfg);
        assert_eq!(engine.layouts().len(), 12);
    }

    #[test]
    fn test_strip_layout_engine_first_position() {
        let mut cfg = StripConfig::default_config();
        cfg.padding = 4;
        let engine = StripLayoutEngine::new(cfg);
        let layouts = engine.layouts();
        assert_eq!(layouts[0].x, 4);
        assert_eq!(layouts[0].y, 4);
    }

    #[test]
    fn test_frame_selector_uniform_count() {
        let sel = FrameSelector::new(60.0, 1800);
        let ts = sel.uniform_timestamps(6);
        assert_eq!(ts.len(), 6);
    }

    #[test]
    fn test_frame_selector_uniform_range() {
        let sel = FrameSelector::new(60.0, 1800);
        let ts = sel.uniform_timestamps(5);
        for t in &ts {
            assert!(*t > 0.0);
            assert!(*t < 60.0);
        }
    }

    #[test]
    fn test_frame_selector_zero_count() {
        let sel = FrameSelector::new(60.0, 1800);
        assert!(sel.uniform_timestamps(0).is_empty());
    }

    #[test]
    fn test_frame_selector_every_nth() {
        let sel = FrameSelector::new(10.0, 10);
        let frames = sel.every_nth_frames(3);
        assert_eq!(frames, vec![0, 3, 6, 9]);
    }

    #[test]
    fn test_frame_selector_timestamp_fraction() {
        let sel = FrameSelector::new(100.0, 3000);
        assert!((sel.timestamp_fraction(50.0) - 0.5).abs() < 1e-9);
        assert!((sel.timestamp_fraction(0.0)).abs() < 1e-9);
        assert!((sel.timestamp_fraction(100.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compositor_sheet_dimensions() {
        let cfg = StripConfig::default_config();
        let compositor = ContactSheetCompositor::new(cfg.clone());
        let (w, h) = compositor.sheet_dimensions();
        assert_eq!(w, cfg.sheet_width());
        assert_eq!(h, cfg.sheet_height());
    }

    #[test]
    fn test_compositor_compose_returns_correct_size() {
        let cfg = StripConfig::default_config();
        let (w, h) = (cfg.sheet_width(), cfg.sheet_height());
        let compositor = ContactSheetCompositor::new(cfg.clone());

        let frames: Vec<ExtractedFrame> = (0..cfg.thumb_count)
            .map(|i| ExtractedFrame::solid(u64::from(i), f64::from(i), 160, 90, [0, 0, 0, 255]))
            .collect();

        let sheet = compositor.compose(&frames);
        assert_eq!(sheet.len(), (w * h * 4) as usize);
    }
}
